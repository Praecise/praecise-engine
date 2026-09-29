//! A runtime host and its protocol.
//!
//! The application speaks to a host over the process's standard input and
//! output, one JSON object per line. The protocol is the same for every
//! engine kind:
//!
//! * every request carries an `"id"` the application chose; the first request
//!   is always `{"id": 0, "op": "load", "entry": {...}}`;
//! * for each request the host may write any number of events
//!   `{"id": n, "event": ...}` (streamed tokens, reasoning, tool calls,
//!   progress), which that request's caller receives in order;
//! * the host ends each request with exactly one reply `{"id": n, "ok": true,
//!   ...}` or `{"id": n, "ok": false, "error": "..."}`.
//!
//! Requests may be outstanding concurrently, so an engine that batches (a
//! continuous-batching inference server) sees them together; one that does not
//! simply answers them in turn. A line that breaks the protocol stops the
//! host. A host that exits fails every outstanding request with the last lines
//! of its error output. The host is expected to fetch nothing: the application
//! passes it the model's files read-only, and the confinement gives it no
//! route out.

use std::collections::{HashMap, VecDeque};
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, ExitStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::sandbox::{self, Confinement};
use crate::{Error, Result};

/// How to start one kind of engine.
#[derive(Debug, Clone)]
pub struct HostSpec {
    /// Engine kind, used for its admission queue and engine claim.
    pub kind: String,
    /// The program to run.
    pub program: PathBuf,
    /// Its arguments.
    pub args: Vec<OsString>,
    /// Its environment, beyond the settings every host gets.
    pub env: Vec<(String, String)>,
    /// Paths the engine runs from (interpreter, libraries), readable by it.
    pub read_only: Vec<PathBuf>,
    /// Bring up a loopback interface in the host's (otherwise empty) network
    /// namespace and allow internet-family sockets, for engines whose
    /// processes rendezvous over 127.0.0.1. The namespace still has no route
    /// out.
    pub loopback: bool,
}

/// Lines of the host's error output kept for reporting an exit.
const STDERR_TAIL: usize = 40;

/// What the reader delivers to one outstanding request.
enum Delivery {
    Event(Value),
    Reply(Value),
}

type Pending = Arc<Mutex<HashMap<u64, Sender<Delivery>>>>;

/// One running runtime host serving one model.
pub struct RuntimeHost {
    model_id: String,
    scratch: PathBuf,
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    next_id: AtomicU64,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    protocol_error: Arc<Mutex<Option<String>>>,
}

impl std::fmt::Debug for RuntimeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeHost").field("model_id", &self.model_id).finish_non_exhaustive()
    }
}

impl Drop for RuntimeHost {
    fn drop(&mut self) {
        let c = self.child.get_mut();
        let _ = c.kill();
        let _ = c.wait();
    }
}

impl RuntimeHost {
    /// Start a host from `spec` for `model_id` over the checkpoint in
    /// `model_dir`, and load `entry` (the catalog row, with its repositories
    /// replaced by local paths) into it. Blocking.
    ///
    /// # Errors
    /// When the host cannot be confined or started, or refuses the model (for
    /// example because it cannot reach the GPU on a GPU host).
    pub fn start(spec: &HostSpec, model_id: &str, model_dir: &Path, scratch: &Path, entry: Value) -> Result<Self> {
        std::fs::create_dir_all(scratch)?;
        let mut read_only = spec.read_only.clone();
        read_only.push(model_dir.to_path_buf());
        let confinement = Confinement {
            read_only,
            scratch: scratch.to_path_buf(),
            devices: sandbox::gpu_devices(),
            loopback: spec.loopback,
        };
        let s = scratch.display().to_string();
        let mut environment = vec![
            ("HOME".to_string(), s.clone()),
            ("TMPDIR".to_string(), s),
            ("PATH".to_string(), "/usr/bin:/bin".to_string()),
            // Set when the machine has GPU hardware: the engine must then
            // refuse to load rather than run on the CPU.
            ("PRAECISE_GPU_REQUIRED".to_string(), if confinement.devices.is_empty() { "0" } else { "1" }.to_string()),
        ];
        environment.extend(spec.env.iter().cloned());
        let mut child = sandbox::spawn(&confinement, &spec.program, &spec.args, &environment)?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        let stderr_tail = drain_stderr(model_id, child.stderr.take().expect("piped"));
        let pending: Pending = Arc::default();
        let protocol_error: Arc<Mutex<Option<String>>> = Arc::default();
        spawn_reader(stdout, pending.clone(), protocol_error.clone());
        let host = Self {
            model_id: model_id.to_string(),
            scratch: scratch.to_path_buf(),
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicU64::new(0),
            stderr_tail,
            protocol_error,
        };
        host.call(&json!({ "op": "load", "entry": entry }))?;
        Ok(host)
    }

    /// The directory the host writes outputs into.
    #[must_use]
    pub fn scratch(&self) -> &Path {
        &self.scratch
    }

    /// The host process id.
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.child.lock().id()
    }

    /// The host's exit status if it has exited, without waiting.
    #[must_use]
    pub fn try_exit(&self) -> Option<ExitStatus> {
        self.child.lock().try_wait().ok().flatten()
    }

    /// The last lines the host wrote to its error output.
    #[must_use]
    pub fn stderr_tail(&self) -> Vec<String> {
        self.stderr_tail.lock().iter().cloned().collect()
    }

    /// Send one request and wait for its reply, discarding events. Blocking.
    ///
    /// # Errors
    /// The host's refusal, or its exit (with what it last wrote to stderr).
    pub fn call(&self, request: &Value) -> Result<Value> {
        self.call_streaming(request, |_| Ok(()))
    }

    /// Send one request, passing each of its events to `on_event` in order,
    /// and return its reply. Blocking; other requests may be outstanding at
    /// the same time. An error from `on_event` (a caller that went away)
    /// abandons this request: its remaining output is discarded, and other
    /// requests are unaffected.
    ///
    /// # Errors
    /// The host's refusal, its exit or a protocol break, or `on_event`'s
    /// error.
    pub fn call_streaming(&self, request: &Value, mut on_event: impl FnMut(&Value) -> Result<()>) -> Result<Value> {
        let obj = request.as_object().ok_or_else(|| Error::Failed("a request must be a JSON object".into()))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut req = obj.clone();
        req.insert("id".into(), Value::from(id));
        let (tx, rx): (Sender<Delivery>, Receiver<Delivery>) = channel();
        self.pending.lock().insert(id, tx);
        let mut line = serde_json::to_vec(&Value::Object(req))?;
        line.push(b'\n');
        let written = {
            let mut stdin = self.stdin.lock();
            stdin.write_all(&line).and_then(|()| stdin.flush())
        };
        if written.is_err() {
            self.pending.lock().remove(&id);
            return Err(self.exited());
        }
        loop {
            match rx.recv() {
                Ok(Delivery::Event(e)) => {
                    if let Err(err) = on_event(&e) {
                        self.pending.lock().remove(&id);
                        return Err(err);
                    }
                }
                Ok(Delivery::Reply(v)) => {
                    return if v.get("ok").and_then(Value::as_bool) == Some(true) {
                        Ok(v)
                    } else {
                        let msg = v.get("error").and_then(Value::as_str).unwrap_or("no error given");
                        Err(Error::Refused(format!("{}: {msg}", self.model_id)))
                    };
                }
                Err(_) => return Err(self.exited()),
            }
        }
    }

    /// The error every request sees once the host has stopped answering.
    fn exited(&self) -> Error {
        let tail = self.stderr_tail.lock().iter().cloned().collect::<Vec<_>>().join("\n");
        let mut child = self.child.lock();
        let _ = child.kill();
        let status = child.wait().ok();
        let why = match self.protocol_error.lock().clone() {
            Some(p) => format!("broke the protocol ({p})"),
            None => format!("exited ({})", status.map_or("unknown status".to_string(), |s| s.to_string())),
        };
        Error::Failed(format!("{}: the runtime host {why}: {tail}", self.model_id))
    }
}

/// Parse one line of host output into its request id and delivery.
fn parse_line(line: &str) -> std::result::Result<(u64, Delivery), String> {
    let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let id = v.get("id").and_then(Value::as_u64).ok_or("a line without a request id")?;
    if let Some(event) = v.get("event") {
        return Ok((id, Delivery::Event(event.clone())));
    }
    if v.get("ok").and_then(Value::as_bool).is_some() {
        return Ok((id, Delivery::Reply(v)));
    }
    Err("neither an event nor a reply".into())
}

/// Read the host's output, delivering each line to the request it names.
/// Output for an abandoned request is dropped. A line that breaks the
/// protocol, or the end of the output, ends every outstanding request; the
/// application then stops the host.
fn spawn_reader(stdout: ChildStdout, pending: Pending, protocol_error: Arc<Mutex<Option<String>>>) {
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            match parse_line(&line) {
                Ok((id, delivery)) => {
                    let is_reply = matches!(delivery, Delivery::Reply(_));
                    let mut p = pending.lock();
                    if let Some(tx) = p.get(&id)
                        && tx.send(delivery).is_ok()
                        && !is_reply
                    {
                        continue;
                    }
                    p.remove(&id);
                }
                Err(e) => {
                    *protocol_error.lock() = Some(e);
                    break;
                }
            }
        }
        pending.lock().clear();
    });
}

/// Read the host's error output as it is written, so the pipe never fills,
/// logging it and keeping the last lines for an exit report.
fn drain_stderr(model_id: &str, stderr: std::process::ChildStderr) -> Arc<Mutex<VecDeque<String>>> {
    let tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL)));
    let keep = tail.clone();
    let id = model_id.to_string();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(std::result::Result::ok) {
            tracing::debug!(model_id = %id, "runtime host: {line}");
            let mut t = keep.lock();
            if t.len() == STDERR_TAIL {
                t.pop_front();
            }
            t.push_back(line);
        }
    });
    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_routed_by_request_id() {
        assert!(matches!(parse_line(r#"{"id":3,"event":{"token":"a"}}"#), Ok((3, Delivery::Event(_)))));
        assert!(matches!(parse_line(r#"{"id":4,"ok":true}"#), Ok((4, Delivery::Reply(_)))));
        assert!(matches!(parse_line(r#"{"id":4,"ok":false,"error":"x"}"#), Ok((4, Delivery::Reply(_)))));
    }

    #[test]
    fn a_line_that_breaks_the_protocol_is_never_taken_for_output() {
        for bad in ["hello", r#"{"event":{}}"#, r#"{"ok":true}"#, r#"{"id":1,"x":1}"#, "[1]", r#"{"id":"a","ok":true}"#] {
            assert!(parse_line(bad).is_err(), "{bad}");
        }
    }
}
