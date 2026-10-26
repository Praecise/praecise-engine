//! Job engines: fixed programs run once, to completion, in a confined host.
//!
//! A model engine stays loaded and answers requests; a job engine runs one
//! program with typed parameters over input files and leaves output files.
//! Both start from a pinned [`EngineEnv`] (the SHA-256 of the environment's
//! whole tree, checked before every start) and run under the same
//! confinement ([`crate::sandbox`]): no network, the environment and any
//! model files read-only, the inputs read-only, one scratch directory
//! writable, and the GPU device nodes.
//!
//! The programs are a fixed catalogue ([`programs`]), each with its
//! parameters declared with a type and a range. A request is checked against
//! the declaration before anything starts; nothing a caller sends runs as
//! code. The programs themselves live in one Python adapter,
//! [`JOB_ADAPTER`], compiled into the binary and placed read-only beside the
//! scratch directory for each run, so the program that runs is the one the
//! binary was built with.
//!
//! The protocol is the runtime host protocol ([`crate::host`]) with one
//! request: `{"id": 0, "op": "run", "program", "params", "inputs", "out",
//! "work"}`. The adapter streams `{"id": 0, "event": {"progress": f,
//! "message": m}}` lines and ends with one reply `{"id": 0, "ok": true,
//! "result": {...}}` or `{"id": 0, "ok": false, "error": "..."}`. A run is
//! supervised: it is stopped at its time limit, and a host that exits without
//! a reply fails the run with the last lines of its error output.
//!
//! On a machine with a GPU the host is told so (`PRAECISE_GPU_REQUIRED=1`)
//! and every program refuses rather than run on the CPU.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use crate::engine::EngineEnv;
use crate::sandbox::{self, Confinement};
use crate::{Error, Result};

/// File name of the job adapter in a run's adapter directory.
pub const JOB_ADAPTER_FILE: &str = "praecise_job.py";

/// The job programs, in Python.
pub const JOB_ADAPTER: &str = include_str!("../adapters/praecise_job.py");

/// A parameter's type.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    /// An integer in `min..=max`.
    Int(i64, i64),
    /// A number in `min..=max`.
    Float(f64, f64),
    /// Text of at most this many bytes.
    Str(usize),
    /// A name: letters, digits, `.`, `_` and `-`, starting with a letter or digit, at most 128.
    Name,
    /// One of these values.
    Enum(&'static [&'static str]),
    /// Any JSON value of at most this many bytes, checked by the program.
    Json(usize),
    /// `true` or `false`.
    Bool,
}

/// A declared parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Its name.
    pub name: &'static str,
    /// Its type.
    pub ty: ParamType,
    /// Whether a request must send it.
    pub required: bool,
}

/// A declared input file.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    /// Its name, or a prefix ending in `_` naming a family of inputs.
    pub name: &'static str,
    /// Whether a request must send it.
    pub required: bool,
}

/// A job program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Its name, `<system>.<program>`.
    pub name: &'static str,
    /// The engine kind whose environment runs it.
    pub engine: &'static str,
    /// What it does and what it writes.
    pub about: &'static str,
    /// Its parameters.
    pub params: Vec<Param>,
    /// Its input files.
    pub inputs: Vec<Input>,
}

const fn p(name: &'static str, ty: ParamType, required: bool) -> Param {
    Param { name, ty, required }
}

const SEED: (i64, i64) = (0, (1 << 31) - 1);

/// The catalogue of job programs.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn programs() -> Vec<Program> {
    use ParamType::{Enum, Float, Int, Json, Name, Str};
    let cudaq_targets: &'static [&'static str] = &["nvidia", "tensornet", "qpp-cpu"];
    vec![
        Program {
            name: "sionna.radio_map",
            engine: "sionna",
            about: "Sionna RT ray-traces a built-in or uploaded Mitsuba scene and maps path gain, received power and SINR \
                    per transmitter on a grid above the ground. Writes radio_map.npz, radio_map.png and result.json.",
            params: vec![
                p("scene", Str(64), false),
                p("transmitters", Json(16 * 1024), false),
                p("frequency_ghz", Float(0.1, 100.0), false),
                p("max_depth", Int(0, 10), false),
                p("samples", Int(10_000, 100_000_000), false),
                p("cell_m", Float(0.25, 50.0), false),
                p("height_m", Float(0.0, 500.0), false),
                p("seed", Int(SEED.0, SEED.1), false),
            ],
            inputs: vec![Input { name: "scene", required: false }],
        },
        Program {
            name: "warp.example",
            engine: "warp",
            about: "One of NVIDIA Warp's own examples by folder and name (core.sph, fem.navier_stokes), headless. \
                    Writes sim.usd when the example writes a stage.",
            params: vec![p("example", Name, true), p("frames", Int(1, 2000), false)],
            inputs: vec![],
        },
        Program {
            name: "newton.example",
            engine: "newton",
            about: "One of Newton's own examples by name, headless, written as a USD stage. Writes sim.usd.",
            params: vec![p("example", Name, true), p("frames", Int(1, 2000), false)],
            inputs: vec![],
        },
        Program {
            name: "cosmos.generate",
            engine: "cosmos",
            about: "NVIDIA Cosmos 3 (Edge or Nano) generates a video from a prompt and an optional first image, with the \
                    Cosmos guardrail always on: the prompt is checked first and faces are blurred in the output. \
                    Writes video.mp4 and result.json.",
            params: vec![
                p("prompt", Str(8192), true),
                p("model", Enum(&["edge", "nano"]), false),
                p("frames", Int(1, 241), false),
                p("steps", Int(1, 100), false),
                p("seed", Int(SEED.0, SEED.1), false),
                p("fps", Int(1, 60), false),
            ],
            inputs: vec![Input { name: "image", required: false }],
        },
        Program {
            name: "physicsnemo.darcy_fno",
            engine: "physicsnemo",
            about: "NVIDIA PhysicsNeMo trains a Fourier neural operator on Darcy flow fields generated on the GPU and \
                    reports its relative error on unseen fields. Writes fno_darcy.pt and result.json.",
            params: vec![
                p("steps", Int(1, 20_000), false),
                p("resolution", Int(32, 512), false),
                p("batch", Int(1, 256), false),
                p("seed", Int(SEED.0, SEED.1), false),
            ],
            inputs: vec![],
        },
        Program {
            name: "cudaq.ghz",
            engine: "cudaq",
            about: "NVIDIA CUDA-Q prepares an n-qubit GHZ state on the chosen simulator and samples it. Writes result.json.",
            params: vec![p("qubits", Int(2, 34), true), p("shots", Int(1, 100_000), false), p("target", Enum(cudaq_targets), false)],
            inputs: vec![],
        },
        Program {
            name: "cudaq.maxcut",
            engine: "cudaq",
            about: "NVIDIA CUDA-Q solves MaxCut on the input graph ({\"nodes\": n, \"edges\": [[i, j], ...]}) with QAOA \
                    and compares the best cut with the exact optimum on small graphs. Writes result.json.",
            params: vec![
                p("layers", Int(1, 12), false),
                p("shots", Int(1, 100_000), false),
                p("target", Enum(cudaq_targets), false),
                p("seed", Int(SEED.0, SEED.1), false),
            ],
            inputs: vec![Input { name: "graph", required: true }],
        },
        Program {
            name: "boltz.predict",
            engine: "boltz",
            about: "Boltz-2 predicts the structure of proteins, nucleic acids and small molecules together, and a \
                    ligand's binding affinity, from a Boltz YAML or a FASTA query, single-sequence or with the caller's \
                    own alignments (inputs msa_*), never a server. Writes each structure, its confidence and result.json.",
            params: vec![
                p("recycling_steps", Int(1, 10), false),
                p("sampling_steps", Int(20, 400), false),
                p("samples", Int(1, 10), false),
                p("seed", Int(SEED.0, SEED.1), false),
                p("format", Enum(&["mmcif", "pdb"]), false),
            ],
            inputs: vec![Input { name: "query", required: true }, Input { name: "msa_", required: false }],
        },
    ]
}

/// The program called `name`.
#[must_use]
pub fn program(name: &str) -> Option<Program> {
    programs().into_iter().find(|p| p.name == name)
}

fn is_name(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 128
        && b[0].is_ascii_alphanumeric()
        && b.iter().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

impl Program {
    /// Check `params` against the declaration: every one declared, of its
    /// type and in its range, and every required one present.
    ///
    /// # Errors
    /// [`Error::Refused`] naming the first parameter that fails.
    pub fn check_params(&self, params: &Map<String, Value>) -> Result<()> {
        for (k, v) in params {
            let Some(decl) = self.params.iter().find(|d| d.name == k) else {
                return Err(Error::Refused(format!("{}: no parameter {k}", self.name)));
            };
            let bad = |why: String| Error::Refused(format!("{}: {k} {why}", self.name));
            match &decl.ty {
                ParamType::Int(lo, hi) => {
                    let n = v.as_i64().filter(|_| v.is_i64() || v.is_u64()).ok_or_else(|| bad("is an integer".into()))?;
                    if n < *lo || n > *hi {
                        return Err(bad(format!("is between {lo} and {hi}")));
                    }
                }
                ParamType::Float(lo, hi) => {
                    let n = v.as_f64().ok_or_else(|| bad("is a number".into()))?;
                    if !(n >= *lo && n <= *hi) {
                        return Err(bad(format!("is between {lo} and {hi}")));
                    }
                }
                ParamType::Str(max) => {
                    let s = v.as_str().ok_or_else(|| bad("is text".into()))?;
                    if s.len() > *max {
                        return Err(bad(format!("is at most {max} bytes")));
                    }
                }
                ParamType::Name => {
                    if !v.as_str().is_some_and(is_name) {
                        return Err(bad("is a name (letters, digits, '.', '_' and '-')".into()));
                    }
                }
                ParamType::Enum(values) => {
                    if !v.as_str().is_some_and(|s| values.contains(&s)) {
                        return Err(bad(format!("is one of {}", values.join(", "))));
                    }
                }
                ParamType::Json(max) => {
                    if serde_json::to_vec(v)?.len() > *max {
                        return Err(bad(format!("is at most {max} bytes of JSON")));
                    }
                }
                ParamType::Bool => {
                    if !v.is_boolean() {
                        return Err(bad("is true or false".into()));
                    }
                }
            }
        }
        for d in self.params.iter().filter(|d| d.required) {
            if !params.contains_key(d.name) {
                return Err(Error::Refused(format!("{}: the parameter {} is required", self.name, d.name)));
            }
        }
        Ok(())
    }

    /// Parse `value` as the declared type of parameter `name`, for callers
    /// that pass parameters as text (`name=value`).
    ///
    /// # Errors
    /// When the program has no such parameter or the text is not of its type.
    pub fn parse_param(&self, name: &str, value: &str) -> Result<Value> {
        let decl = self
            .params
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| Error::Refused(format!("{}: no parameter {name}", self.name)))?;
        let bad = || Error::Refused(format!("{}: {name}={value} is not of its type", self.name));
        Ok(match decl.ty {
            ParamType::Int(..) => Value::from(value.parse::<i64>().map_err(|_| bad())?),
            ParamType::Float(..) => Value::from(value.parse::<f64>().map_err(|_| bad())?),
            ParamType::Bool => Value::from(value.parse::<bool>().map_err(|_| bad())?),
            ParamType::Json(_) => serde_json::from_str(value).map_err(|_| bad())?,
            ParamType::Str(_) | ParamType::Name | ParamType::Enum(_) => Value::from(value),
        })
    }

    /// Check the input names: each declared (by name or family prefix), every
    /// required one present.
    ///
    /// # Errors
    /// [`Error::Refused`] naming the first input that fails.
    pub fn check_inputs(&self, inputs: &BTreeMap<String, PathBuf>) -> Result<()> {
        for (name, path) in inputs {
            let declared = self.inputs.iter().any(|i| if i.name.ends_with('_') { name.starts_with(i.name) && is_name(name) } else { i.name == name });
            if !declared {
                return Err(Error::Refused(format!("{}: no input {name}", self.name)));
            }
            if !path.is_file() {
                return Err(Error::Refused(format!("{}: input {name} ({}) is not a file", self.name, path.display())));
            }
        }
        for i in self.inputs.iter().filter(|i| i.required) {
            if !inputs.keys().any(|k| if i.name.ends_with('_') { k.starts_with(i.name) } else { k == i.name }) {
                return Err(Error::Refused(format!("{}: the input {} is required", self.name, i.name)));
            }
        }
        Ok(())
    }

    /// The declaration as JSON, for callers that list the catalogue.
    #[must_use]
    pub fn describe(&self) -> Value {
        let params: Map<String, Value> = self
            .params
            .iter()
            .map(|d| {
                let ty = match &d.ty {
                    ParamType::Int(lo, hi) => json!({"type": "int", "min": lo, "max": hi}),
                    ParamType::Float(lo, hi) => json!({"type": "float", "min": lo, "max": hi}),
                    ParamType::Str(max) => json!({"type": "str", "max_bytes": max}),
                    ParamType::Name => json!({"type": "name"}),
                    ParamType::Enum(v) => json!({"type": "enum", "values": v}),
                    ParamType::Json(max) => json!({"type": "json", "max_bytes": max}),
                    ParamType::Bool => json!({"type": "bool"}),
                };
                let mut ty = ty;
                ty["required"] = json!(d.required);
                (d.name.to_string(), ty)
            })
            .collect();
        let inputs: Vec<Value> = self.inputs.iter().map(|i| json!({"name": i.name, "required": i.required})).collect();
        json!({"name": self.name, "engine": self.engine, "about": self.about, "params": params, "inputs": inputs})
    }
}

/// One job engine: an environment and what a run of it may read.
#[derive(Debug, Clone)]
pub struct JobEngine {
    /// The environment, pinned. Its program is the environment's Python.
    pub env: EngineEnv,
    /// Further paths readable by the job (model caches, data sets).
    pub read_only: Vec<PathBuf>,
    /// Environment variables set for the job.
    pub vars: Vec<(String, String)>,
}

/// How strictly a run is confined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confine {
    /// User, mount, network, IPC and PID namespaces, Landlock and the syscall
    /// filter. The default; refused on a kernel that forbids unprivileged
    /// user namespaces for this executable.
    Full,
    /// Landlock and the syscall filter only, chosen explicitly by an operator
    /// (see [`Confinement::landlock_only`]).
    LandlockOnly,
}

/// The engines a machine offers, read from its configuration:
///
/// ```json
/// {"engines": {"sionna": {"root": "/opt/envs/sionna", "digest": "<sha256 of the tree>",
///                          "read_only": ["/data/scenes"], "env": {"KEY": "value"}}}}
/// ```
///
/// # Errors
/// When the configuration is not of that form.
pub fn engines_from_json(v: &Value) -> Result<BTreeMap<String, JobEngine>> {
    let engines = v.get("engines").and_then(Value::as_object).ok_or_else(|| Error::Refused("the configuration has no \"engines\" object".into()))?;
    let mut out = BTreeMap::new();
    for (kind, e) in engines {
        let s = |key: &str| e.get(key).and_then(Value::as_str).map(str::to_string);
        let root = PathBuf::from(s("root").ok_or_else(|| Error::Refused(format!("engine {kind}: no root")))?);
        let digest = s("digest").ok_or_else(|| Error::Refused(format!("engine {kind}: no digest")))?;
        let python = s("python").unwrap_or_else(|| "bin/python3".to_string());
        let read_only = e
            .get("read_only")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(PathBuf::from).collect())
            .unwrap_or_default();
        let vars = e
            .get("env")
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string()))).collect())
            .unwrap_or_default();
        let env = EngineEnv { kind: kind.clone(), root, program: PathBuf::from(python), args: Vec::new(), digest: digest.to_ascii_lowercase(), loopback: false };
        out.insert(kind.clone(), JobEngine { env, read_only, vars });
    }
    Ok(out)
}

/// One run.
#[derive(Debug, Clone)]
pub struct JobRequest {
    /// The program.
    pub program: String,
    /// Its parameters.
    pub params: Map<String, Value>,
    /// Its input files by name.
    pub inputs: BTreeMap<String, PathBuf>,
    /// A new directory the run may create its adapter and scratch
    /// directories in; the outputs are left in `<run_dir>/scratch/out`.
    pub run_dir: PathBuf,
    /// Stop the run after this long.
    pub timeout: Duration,
    /// How strictly it is confined.
    pub confine: Confine,
}

/// A finished run.
#[derive(Debug, Clone)]
pub struct JobOutcome {
    /// What the program returned.
    pub result: Value,
    /// The directory holding its output files.
    pub out_dir: PathBuf,
    /// Wall time, environment check included.
    pub seconds: f64,
    /// Time spent checking the environment's digest.
    pub verify_seconds: f64,
}

/// Lines of error output kept for reporting a failure.
const STDERR_TAIL: usize = 60;

/// Run `req` on `engine`, passing each progress event to `on_event`.
/// Blocking.
///
/// # Errors
/// [`Error::Refused`] for a request the program's declaration refuses or an
/// environment whose digest differs from its pin; [`Error::Start`] when the
/// host cannot be confined; [`Error::Failed`] when the program fails, exits
/// without a reply or runs past its time limit.
#[allow(clippy::needless_pass_by_value)]
pub fn run_job(engine: &JobEngine, req: JobRequest, mut on_event: impl FnMut(&Value)) -> Result<JobOutcome> {
    let started = Instant::now();
    let prog = program(&req.program).ok_or_else(|| Error::Refused(format!("no job program {}", req.program)))?;
    if prog.engine != engine.env.kind {
        return Err(Error::Refused(format!("{} runs on the {} engine, not {}", prog.name, prog.engine, engine.env.kind)));
    }
    prog.check_params(&req.params)?;
    prog.check_inputs(&req.inputs)?;
    let spec = engine.env.verified_spec()?;
    let verify_seconds = started.elapsed().as_secs_f64();

    if req.run_dir.exists() && std::fs::read_dir(&req.run_dir)?.next().is_some() {
        return Err(Error::Refused(format!("the run directory {} is not empty", req.run_dir.display())));
    }
    let adapter_dir = req.run_dir.join("adapter");
    let scratch = req.run_dir.join("scratch");
    let (out, work) = (scratch.join("out"), scratch.join("work"));
    for d in [&adapter_dir, &out, &work] {
        std::fs::create_dir_all(d)?;
    }
    let adapter = adapter_dir.join(JOB_ADAPTER_FILE);
    std::fs::write(&adapter, JOB_ADAPTER)?;

    let mut read_only = spec.read_only.clone();
    read_only.extend(engine.read_only.iter().cloned());
    read_only.push(adapter_dir.clone());
    let mut inputs = Map::new();
    for (name, path) in &req.inputs {
        let abs = std::fs::canonicalize(path)?;
        read_only.push(abs.clone());
        inputs.insert(name.clone(), Value::from(abs.display().to_string()));
    }
    let devices = sandbox::gpu_devices();
    let confinement = Confinement {
        read_only,
        scratch: scratch.clone(),
        devices: devices.clone(),
        loopback: false,
        landlock_only: req.confine == Confine::LandlockOnly,
    };
    let w = work.display().to_string();
    let mut env = vec![
        ("HOME".to_string(), w.clone()),
        ("TMPDIR".to_string(), w),
        ("PATH".to_string(), format!("{}:/usr/bin:/bin", engine.env.root.join("bin").display())),
        ("PRAECISE_GPU_REQUIRED".to_string(), if devices.is_empty() { "0" } else { "1" }.to_string()),
        ("PRAECISE_ENGINE".to_string(), engine.env.kind.clone()),
        ("PYTHONUNBUFFERED".to_string(), "1".to_string()),
        ("HF_HUB_OFFLINE".to_string(), "1".to_string()),
    ];
    // the operator's choice of GPUs carries through
    for k in ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"] {
        if let Ok(v) = std::env::var(k) {
            env.push((k.to_string(), v));
        }
    }
    env.extend(engine.vars.iter().cloned());
    let args: Vec<OsString> = vec!["-I".into(), adapter.into_os_string()];
    let mut child = sandbox::spawn(&confinement, &spec.program, &args, &env)?;
    let mut stdin = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    let tail: Arc<Mutex<VecDeque<String>>> = Arc::default();
    let keep = tail.clone();
    let err_thread = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(std::result::Result::ok) {
            tracing::debug!("job: {line}");
            let mut t = keep.lock();
            if t.len() == STDERR_TAIL {
                t.pop_front();
            }
            t.push_back(line);
        }
    });
    let request = json!({"id": 0, "op": "run", "program": prog.name, "params": req.params, "inputs": inputs,
                         "out": out.display().to_string(), "work": work.display().to_string()});
    let mut line = serde_json::to_vec(&request)?;
    line.push(b'\n');
    // the adapter reads one request; closing stdin after it tells it no more follow
    let wrote = stdin.write_all(&line).and_then(|()| stdin.flush());
    drop(stdin);

    // the supervisor: a reader thread delivers lines, this thread enforces the time limit
    let (tx, rx) = std::sync::mpsc::channel::<std::result::Result<String, String>>();
    std::thread::spawn(move || {
        for l in BufReader::new(stdout).lines() {
            match l {
                Ok(l) => {
                    if tx.send(Ok(l)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e.to_string()));
                    return;
                }
            }
        }
    });
    let deadline = Instant::now() + req.timeout;
    let mut reply: Option<Value> = None;
    let mut broke: Option<String> = wrote.err().map(|e| format!("writing the request: {e}"));
    while broke.is_none() && reply.is_none() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            broke = Some(format!("ran past its time limit of {} s", req.timeout.as_secs()));
            break;
        }
        match rx.recv_timeout(left.min(Duration::from_secs(1))) {
            Ok(Ok(l)) => {
                let v: Value = match serde_json::from_str(&l) {
                    Ok(v) => v,
                    Err(_) => {
                        broke = Some(format!("broke the protocol: {}", l.chars().take(200).collect::<String>()));
                        break;
                    }
                };
                if let Some(e) = v.get("event") {
                    on_event(e);
                } else if v.get("ok").and_then(Value::as_bool).is_some() {
                    reply = Some(v);
                } else {
                    broke = Some("broke the protocol: a line that is neither an event nor a reply".into());
                }
            }
            Ok(Err(e)) => broke = Some(format!("reading its output: {e}")),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if broke.is_none() {
        // the program replied: give it a moment to exit on its own
        let until = Instant::now() + Duration::from_secs(10);
        while child.try_wait()?.is_none() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    // whatever the program left running goes with it
    kill_tree(child.id());
    let _ = child.kill();
    let status = child.wait()?;
    let _ = err_thread.join();
    let stderr_tail = tail.lock().iter().cloned().collect::<Vec<_>>().join("\n");
    if let Some(why) = broke {
        return Err(Error::Failed(format!("{}: the job {why}\n{stderr_tail}", prog.name)));
    }
    let Some(reply) = reply else {
        return Err(Error::Failed(format!("{}: the job exited ({status}) without a reply\n{stderr_tail}", prog.name)));
    };
    if reply.get("ok").and_then(Value::as_bool) != Some(true) {
        let msg = reply.get("error").and_then(Value::as_str).unwrap_or("no error given");
        return Err(Error::Failed(format!("{}: {msg}", prog.name)));
    }
    Ok(JobOutcome {
        result: reply.get("result").cloned().unwrap_or(Value::Null),
        out_dir: out,
        seconds: started.elapsed().as_secs_f64(),
        verify_seconds,
    })
}

/// The processes descended from `pid`, deepest first.
#[must_use]
pub fn descendants(pid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else { return out };
    for t in tasks.flatten() {
        let Ok(list) = std::fs::read_to_string(t.path().join("children")) else { continue };
        for c in list.split_whitespace().filter_map(|c| c.parse::<u32>().ok()) {
            out.extend(descendants(c));
            out.push(c);
        }
    }
    out
}

/// Kill every process descended from `pid` (not `pid` itself).
pub fn kill_tree(pid: u32) {
    for p in descendants(pid) {
        if let Ok(p) = i32::try_from(p) {
            // SAFETY: sends a signal; no memory is touched.
            unsafe { libc::kill(p, libc::SIGKILL) };
        }
    }
}

/// Copy the files of `from` (recursively) into `to`, creating it.
///
/// # Errors
/// On a filesystem error, or a link among the outputs (refused, never
/// followed).
pub fn copy_outputs(from: &Path, to: &Path) -> Result<Vec<String>> {
    fn walk(base: &Path, dir: &Path, to: &Path, names: &mut Vec<String>) -> Result<()> {
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let ft = e.file_type()?;
            let rel = e.path().strip_prefix(base).expect("under base").to_path_buf();
            if ft.is_symlink() {
                return Err(Error::Refused(format!("the output {} is a link", rel.display())));
            } else if ft.is_dir() {
                std::fs::create_dir_all(to.join(&rel))?;
                walk(base, &e.path(), to, names)?;
            } else if ft.is_file() {
                std::fs::copy(e.path(), to.join(&rel))?;
                names.push(rel.display().to_string());
            }
        }
        Ok(())
    }
    std::fs::create_dir_all(to)?;
    let mut names = Vec::new();
    walk(from, from, to, &mut names)?;
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn the_catalogue_is_consistent() {
        let all = programs();
        let mut names: Vec<&str> = all.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len());
        for p in &all {
            assert!(JOB_ADAPTER.contains(&format!("\"{}\"", p.name)), "the adapter has no program {}", p.name);
        }
    }

    #[test]
    fn parameters_are_held_to_their_declaration() {
        let ghz = program("cudaq.ghz").unwrap();
        ghz.check_params(&params(json!({"qubits": 10, "shots": 100, "target": "nvidia"}))).unwrap();
        for bad in [
            json!({"shots": 100}),
            json!({"qubits": 35}),
            json!({"qubits": 1.5}),
            json!({"qubits": "10"}),
            json!({"qubits": 4, "target": "gpu"}),
            json!({"qubits": 4, "script": "import os"}),
        ] {
            assert!(matches!(ghz.check_params(&params(bad.clone())), Err(Error::Refused(_))), "{bad}");
        }
        let warp = program("warp.example").unwrap();
        warp.check_params(&params(json!({"example": "core.sph"}))).unwrap();
        for bad in ["../x", "-m", "a b", ""] {
            assert!(warp.check_params(&params(json!({"example": bad}))).is_err(), "{bad}");
        }
        assert_eq!(ghz.parse_param("qubits", "12").unwrap(), json!(12));
        assert!(ghz.parse_param("qubits", "twelve").is_err());
        let radio = program("sionna.radio_map").unwrap();
        assert_eq!(radio.parse_param("transmitters", "[{\"position\": [0, 0, 10]}]").unwrap()[0]["position"][2], json!(10));
    }

    #[test]
    fn inputs_are_held_to_their_declaration() {
        let dir = std::env::temp_dir().join(format!("praecise-job-inputs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("q.yaml");
        std::fs::write(&f, "x").unwrap();
        let boltz = program("boltz.predict").unwrap();
        let ok: BTreeMap<String, PathBuf> = [("query".to_string(), f.clone()), ("msa_a".to_string(), f.clone())].into();
        boltz.check_inputs(&ok).unwrap();
        assert!(boltz.check_inputs(&BTreeMap::new()).is_err());
        let extra: BTreeMap<String, PathBuf> = [("query".to_string(), f.clone()), ("other".to_string(), f.clone())].into();
        assert!(boltz.check_inputs(&extra).is_err());
        let missing: BTreeMap<String, PathBuf> = [("query".to_string(), dir.join("none"))].into();
        assert!(boltz.check_inputs(&missing).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn engines_are_read_from_their_configuration() {
        let v = json!({"engines": {"cudaq": {"root": "/opt/q", "digest": "AB", "read_only": ["/data"], "env": {"K": "v"}}}});
        let e = engines_from_json(&v).unwrap();
        let q = &e["cudaq"];
        assert_eq!(q.env.digest, "ab");
        assert_eq!(q.env.program, PathBuf::from("bin/python3"));
        assert_eq!(q.read_only, vec![PathBuf::from("/data")]);
        assert_eq!(q.vars, vec![("K".to_string(), "v".to_string())]);
        assert!(engines_from_json(&json!({"engines": {"x": {"root": "/a"}}})).is_err());
    }

    #[test]
    fn a_program_on_the_wrong_engine_or_a_changed_environment_is_refused() {
        let root = std::env::temp_dir().join(format!("praecise-job-env-{}", std::process::id()));
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(root.join("bin/python3"), b"#!/bin/sh\n").unwrap();
        let engine = JobEngine {
            env: EngineEnv { kind: "cudaq".into(), root: root.clone(), program: "bin/python3".into(), args: Vec::new(), digest: "00".repeat(32), loopback: false },
            read_only: Vec::new(),
            vars: Vec::new(),
        };
        let req = |program: &str| JobRequest {
            program: program.into(),
            params: params(json!({"qubits": 4})),
            inputs: BTreeMap::new(),
            run_dir: root.join("run"),
            timeout: Duration::from_secs(5),
            confine: Confine::Full,
        };
        let wrong = run_job(&engine, req("warp.example"), |_| {}).unwrap_err();
        assert!(matches!(&wrong, Error::Refused(m) if m.contains("warp engine")), "{wrong}");
        let changed = run_job(&engine, req("cudaq.ghz"), |_| {}).unwrap_err();
        assert!(matches!(&changed, Error::Refused(m) if m.contains("pinned")), "{changed}");
        assert!(!root.join("run").exists(), "nothing is created before the environment is verified");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn outputs_are_copied_and_links_refused() {
        let base = std::env::temp_dir().join(format!("praecise-job-out-{}", std::process::id()));
        let from = base.join("from");
        std::fs::create_dir_all(from.join("sub")).unwrap();
        std::fs::write(from.join("a.json"), b"{}").unwrap();
        std::fs::write(from.join("sub/b.bin"), b"1").unwrap();
        let names = copy_outputs(&from, &base.join("to")).unwrap();
        assert_eq!(names, vec!["a.json".to_string(), "sub/b.bin".to_string()]);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/passwd", from.join("leak")).unwrap();
            assert!(copy_outputs(&from, &base.join("to2")).is_err());
        }
        std::fs::remove_dir_all(base).unwrap();
    }
}
