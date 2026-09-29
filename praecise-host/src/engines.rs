//! The engines serving an application's models.
//!
//! [`Engines`] assigns every model to exactly one engine. A load claims the
//! model id first; a claim held by any other engine refuses it with
//! [`Error::Conflict`], so no model is ever in memory twice. Each claim also
//! reserves the model's memory from one [`Budget`] shared by every engine,
//! and a model that does not fit is refused with [`Error::OverBudget`]. The
//! budget is the application's own ledger when it keeps one, or a
//! [`FixedBudget`].
//!
//! Models of a linked backend ([`Backend::LlamaCpp`]) live in the
//! application's process: the application loads them itself under a
//! [`Claim`]. Every other engine is started by
//! [`Engines::serve`] as a runtime host, one per model, from a pinned
//! [`EngineEnv`]. A host is never started twice for one model. When a host
//! exits it is restarted in the background from the same verified
//! environment while the other engines keep serving; after
//! [`MAX_CONSECUTIVE_FAILURES`] failures with no request served in between it
//! is left stopped, and its requests are refused with the last failure until
//! the model is released and served again.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use serde_json::Value;

use praecise_runtime::backend::{Backend, Integration};

use crate::engine::EngineEnv;
use crate::host::RuntimeHost;
use crate::{Error, Result};

/// Failures (exits and refused starts) a host may have in a row, with no
/// request served in between, before it is left stopped.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// The memory ledger every engine's models are admitted against.
pub trait Budget: Send + Sync + std::fmt::Debug {
    /// Reserve `bytes` for `model_id`, or refuse with [`Error::OverBudget`].
    ///
    /// # Errors
    /// When the bytes do not fit.
    fn reserve(&self, model_id: &str, bytes: u64) -> Result<()>;
    /// Give back what `model_id` reserved.
    fn give_back(&self, model_id: &str);
}

/// A [`Budget`] of a fixed number of bytes.
#[derive(Debug)]
pub struct FixedBudget {
    total: u64,
    reserved: Mutex<HashMap<String, u64>>,
}

impl FixedBudget {
    /// A budget of `total` bytes, none reserved.
    #[must_use]
    pub fn new(total: u64) -> Arc<Self> {
        Arc::new(Self { total, reserved: Mutex::new(HashMap::new()) })
    }

    /// Bytes not yet reserved.
    #[must_use]
    pub fn free(&self) -> u64 {
        self.total.saturating_sub(self.reserved.lock().values().sum())
    }
}

impl Budget for FixedBudget {
    fn reserve(&self, model_id: &str, bytes: u64) -> Result<()> {
        let mut r = self.reserved.lock();
        let free = self.total.saturating_sub(r.values().sum());
        if bytes > free {
            return Err(Error::OverBudget { model_id: model_id.to_string(), needed: bytes, free });
        }
        r.insert(model_id.to_string(), bytes);
        Ok(())
    }

    fn give_back(&self, model_id: &str) {
        self.reserved.lock().remove(model_id);
    }
}

/// Every engine of one application. See the module docs.
#[derive(Debug)]
pub struct Engines {
    budget: Arc<dyn Budget>,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    held: HashMap<String, Held>,
}

#[derive(Debug)]
struct Held {
    engine: String,
    gpu_bytes: u64,
    committed: bool,
    hosted: Option<Arc<HostedModel>>,
}

/// How one model's engine is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Health {
    /// Claimed, not yet loaded.
    Loading,
    /// Loaded in the application's process.
    InProcess,
    /// A runtime host is serving it; `restarts` counts the restarts so far.
    Running {
        /// Restarts since the model was served.
        restarts: u32,
    },
    /// The host exited and is being started again.
    Restarting {
        /// Restarts since the model was served, this one included.
        restarts: u32,
        /// Why the last host stopped.
        last_failure: String,
    },
    /// The host failed too often and is left stopped.
    Stopped {
        /// Why the last host stopped.
        last_failure: String,
    },
}

/// One model and the engine serving it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineStatus {
    /// The model.
    pub model_id: String,
    /// The engine kind serving it.
    pub engine: String,
    /// GPU memory reserved for it.
    pub gpu_bytes: u64,
    /// How its engine is doing.
    pub health: Health,
}

impl Engines {
    /// No models yet, admitted against `budget`.
    #[must_use]
    pub fn new(budget: Arc<dyn Budget>) -> Self {
        Self { budget, state: Mutex::new(State::default()) }
    }

    /// Claim `model_id` for the in-process `engine` ahead of loading it,
    /// reserving `gpu_bytes` of the budget. An application that admits its
    /// in-process loads against the same ledger itself passes 0.
    ///
    /// A claim `engine` already holds (a reload in place) succeeds and keeps
    /// its reservation. The returned [`Claim`] gives a new claim back on drop
    /// unless the load [`commit`](Claim::commit)s it.
    ///
    /// # Errors
    /// [`Error::Conflict`] when another engine holds the model,
    /// [`Error::OverBudget`] when it does not fit.
    pub fn claim(&self, model_id: &str, engine: &str, gpu_bytes: u64) -> Result<Claim<'_>> {
        self.take(model_id, engine, gpu_bytes, true)
    }

    fn take(&self, model_id: &str, engine: &str, gpu_bytes: u64, reuse: bool) -> Result<Claim<'_>> {
        let mut s = self.state.lock();
        if let Some(held) = s.held.get(model_id) {
            if reuse && held.engine == engine && held.hosted.is_none() {
                return Ok(Claim { engines: self, model_id: model_id.to_string(), engine: engine.to_string(), fresh: false });
            }
            return Err(Error::Conflict {
                model_id: model_id.to_string(),
                held_by: held.engine.clone(),
                requested: engine.to_string(),
            });
        }
        if gpu_bytes > 0 {
            self.budget.reserve(model_id, gpu_bytes)?;
        }
        s.held.insert(
            model_id.to_string(),
            Held { engine: engine.to_string(), gpu_bytes, committed: false, hosted: None },
        );
        Ok(Claim { engines: self, model_id: model_id.to_string(), engine: engine.to_string(), fresh: true })
    }

    /// Serve `model_id` from the checkpoint in `model_dir` with a runtime
    /// host started from `env`, loading `entry` into it (a JSON object: the
    /// catalog row, its repositories replaced by local paths) with
    /// `model_dir` and `gpu_bytes`, the engine's GPU memory cap, added.
    /// Blocking until the model is loaded.
    ///
    /// # Errors
    /// [`Error::Conflict`] when any engine, this one included, already serves
    /// the model; [`Error::OverBudget`]; an environment that fails its pin;
    /// and the host's own start errors.
    pub fn serve(
        &self,
        model_id: &str,
        env: &EngineEnv,
        model_dir: &Path,
        scratch: &Path,
        mut entry: Value,
        gpu_bytes: u64,
    ) -> Result<Arc<HostedModel>> {
        if let Ok(backend) = Backend::parse(&env.kind)
            && backend.supports().integration == Integration::Linked
        {
            return Err(Error::Refused(format!("{backend} runs in the process; claim the model instead")));
        }
        let claim = self.take(model_id, &env.kind, gpu_bytes, false)?;
        let Some(obj) = entry.as_object_mut() else {
            return Err(Error::Refused(format!("{model_id}: the load entry must be a JSON object")));
        };
        obj.insert("model_dir".into(), model_dir.display().to_string().into());
        obj.insert("gpu_bytes".into(), gpu_bytes.into());
        let hosted = Arc::new(HostedModel {
            model_id: model_id.to_string(),
            engine: env.kind.clone(),
            env: env.clone(),
            model_dir: model_dir.to_path_buf(),
            scratch: scratch.to_path_buf(),
            entry,
            run: Mutex::new(Run::default()),
            changed: Condvar::new(),
        });
        hosted.start()?;
        if let Some(held) = self.state.lock().held.get_mut(model_id) {
            held.hosted = Some(Arc::clone(&hosted));
        }
        claim.commit();
        Ok(hosted)
    }

    /// The runtime host serving `model_id`, when one does.
    #[must_use]
    pub fn hosted(&self, model_id: &str) -> Option<Arc<HostedModel>> {
        self.state.lock().held.get(model_id).and_then(|h| h.hosted.clone())
    }

    /// The engine holding `model_id`, when one does.
    #[must_use]
    pub fn holder(&self, model_id: &str) -> Option<String> {
        self.state.lock().held.get(model_id).map(|h| h.engine.clone())
    }

    /// Give back `engine`'s claim on `model_id`, returning its GPU memory to
    /// the budget and stopping its runtime host once its last request ends.
    /// A claim held by another engine is left alone. Returns whether a claim
    /// was released.
    pub fn release(&self, model_id: &str, engine: &str) -> bool {
        let released = {
            let mut s = self.state.lock();
            match s.held.get(model_id) {
                Some(h) if h.engine == engine => {}
                _ => return false,
            }
            s.held.remove(model_id).expect("present")
        };
        if released.gpu_bytes > 0 {
            self.budget.give_back(model_id);
        }
        if let Some(hosted) = released.hosted {
            hosted.stop();
        }
        true
    }

    /// Every claimed model and its engine's health, ordered by model id.
    #[must_use]
    pub fn status(&self) -> Vec<EngineStatus> {
        let s = self.state.lock();
        let mut out: Vec<EngineStatus> = s
            .held
            .iter()
            .map(|(id, h)| EngineStatus {
                model_id: id.clone(),
                engine: h.engine.clone(),
                gpu_bytes: h.gpu_bytes,
                health: match (&h.hosted, h.committed) {
                    (Some(hosted), _) => hosted.health(),
                    (None, true) => Health::InProcess,
                    (None, false) => Health::Loading,
                },
            })
            .collect();
        out.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        out
    }
}

/// A claim on a model id taken ahead of a load. See [`Engines::claim`].
#[derive(Debug)]
#[must_use = "a claim is released on drop unless committed"]
pub struct Claim<'a> {
    engines: &'a Engines,
    model_id: String,
    engine: String,
    fresh: bool,
}

impl Claim<'_> {
    /// The load succeeded: the claim now lasts until the model is released.
    pub fn commit(mut self) {
        if let Some(held) = self.engines.state.lock().held.get_mut(&self.model_id) {
            held.committed = true;
        }
        self.fresh = false;
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if self.fresh {
            self.engines.release(&self.model_id, &self.engine);
        }
    }
}

/// Tokens one request consumed and produced, as its engine counted them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u64,
    /// Generated tokens.
    pub completion_tokens: u64,
}

impl Usage {
    /// The `usage` object of a generation reply.
    ///
    /// # Errors
    /// When the reply carries no complete token counts: a generation that
    /// cannot be metered is refused, never counted as free.
    pub fn from_reply(reply: &Value) -> Result<Self> {
        let count = |k: &str| reply.get("usage").and_then(|u| u.get(k)).and_then(Value::as_u64);
        match (count("prompt_tokens"), count("completion_tokens")) {
            (Some(prompt_tokens), Some(completion_tokens)) => Ok(Self { prompt_tokens, completion_tokens }),
            _ => Err(Error::Failed("the engine's reply carries no token usage".into())),
        }
    }
}

/// A model served by a supervised runtime host. See the module docs.
#[derive(Debug)]
pub struct HostedModel {
    model_id: String,
    engine: String,
    env: EngineEnv,
    model_dir: PathBuf,
    scratch: PathBuf,
    entry: Value,
    run: Mutex<Run>,
    /// Signalled whenever a host starts, fails to start or is stopped.
    changed: Condvar,
}

#[derive(Debug, Default)]
struct Run {
    host: Option<Arc<RuntimeHost>>,
    restarts: u32,
    failures: u32,
    last_failure: Option<String>,
    starting: bool,
    stopped: bool,
}

impl HostedModel {
    /// The model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The engine kind serving it.
    #[must_use]
    pub fn engine(&self) -> &str {
        &self.engine
    }

    /// The directory the host writes its outputs into.
    #[must_use]
    pub fn scratch(&self) -> &Path {
        &self.scratch
    }

    /// How the host is doing.
    #[must_use]
    pub fn health(&self) -> Health {
        let r = self.run.lock();
        let last = || r.last_failure.clone().unwrap_or_default();
        if r.host.is_some() {
            Health::Running { restarts: r.restarts }
        } else if r.failures >= MAX_CONSECUTIVE_FAILURES || r.stopped {
            Health::Stopped { last_failure: last() }
        } else {
            Health::Restarting { restarts: r.restarts, last_failure: last() }
        }
    }

    /// Run one generation request, passing each streamed event to
    /// `on_event`, and return the engine's reply with its token usage.
    /// Blocking. A host that exits during the request fails the request and
    /// is restarted in the background.
    ///
    /// # Errors
    /// The engine's refusal; the host's exit; a stopped host; a reply with no
    /// usage.
    pub fn generate(
        self: &Arc<Self>,
        request: &Value,
        on_event: impl FnMut(&Value) -> Result<()>,
    ) -> Result<(Value, Usage)> {
        let reply = self.call(request, on_event)?;
        let usage = Usage::from_reply(&reply)?;
        Ok((reply, usage))
    }

    /// Run one request of any kind (an engine whose replies are not token
    /// generations, such as a rendered file in the scratch directory),
    /// passing each streamed event to `on_event`, and return the reply.
    /// Blocking. A host that exits during the request fails the request and
    /// is restarted in the background.
    ///
    /// # Errors
    /// The engine's refusal; the host's exit; a stopped host.
    pub fn call(self: &Arc<Self>, request: &Value, on_event: impl FnMut(&Value) -> Result<()>) -> Result<Value> {
        let host = self.running()?;
        match host.call_streaming(request, on_event) {
            Ok(reply) => {
                self.run.lock().failures = 0;
                Ok(reply)
            }
            Err(Error::Failed(msg)) => {
                self.exited(&host, &msg);
                Err(Error::Failed(msg))
            }
            Err(e) => Err(e),
        }
    }

    /// Wait for a host that is being started, and return the running one.
    fn running(&self) -> Result<Arc<RuntimeHost>> {
        let mut r = self.run.lock();
        loop {
            if let Some(h) = &r.host {
                return Ok(Arc::clone(h));
            }
            if r.stopped || r.failures >= MAX_CONSECUTIVE_FAILURES {
                return Err(Error::Failed(format!(
                    "{}: the {} engine is stopped: {}",
                    self.model_id,
                    self.engine,
                    r.last_failure.as_deref().unwrap_or("released")
                )));
            }
            self.changed.wait(&mut r);
        }
    }

    /// Start a host from the verified environment and load the model.
    fn start(&self) -> Result<()> {
        let started = self.env.verified_spec().and_then(|spec| {
            RuntimeHost::start(&spec, &self.model_id, &self.model_dir, &self.scratch, self.entry.clone())
        });
        let mut r = self.run.lock();
        r.starting = false;
        let result = match started {
            Ok(host) if !r.stopped => {
                r.host = Some(Arc::new(host));
                Ok(())
            }
            Ok(_) => Err(Error::Failed(format!("{}: released while starting", self.model_id))),
            Err(e) => {
                r.failures += 1;
                r.last_failure = Some(e.to_string());
                Err(e)
            }
        };
        self.changed.notify_all();
        result
    }

    /// `host` exited: drop it and start another in the background, unless
    /// the failures ran out or another request already did.
    fn exited(self: &Arc<Self>, host: &Arc<RuntimeHost>, why: &str) {
        let mut r = self.run.lock();
        if !r.host.as_ref().is_some_and(|h| Arc::ptr_eq(h, host)) {
            return;
        }
        tracing::warn!(model_id = %self.model_id, engine = %self.engine, "runtime host exited: {why}");
        r.host = None;
        r.failures += 1;
        r.last_failure = Some(why.to_string());
        self.restart_locked(&mut r);
    }

    fn restart_locked(self: &Arc<Self>, r: &mut Run) {
        if r.stopped || r.starting || r.failures >= MAX_CONSECUTIVE_FAILURES {
            return;
        }
        r.starting = true;
        r.restarts += 1;
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            if me.start().is_err() {
                let mut r = me.run.lock();
                me.restart_locked(&mut r);
            }
        });
    }

    /// Stop the host once its last request ends, and start no other.
    fn stop(&self) {
        let mut r = self.run.lock();
        r.stopped = true;
        r.host = None;
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GIB: u64 = 1 << 30;
    const LLAMA_CPP: &str = Backend::LlamaCpp.as_str();

    #[test]
    fn a_model_is_held_by_one_engine_only() {
        let budget = FixedBudget::new(10 * GIB);
        let engines = Engines::new(budget.clone());
        engines.claim("m", LLAMA_CPP, GIB).unwrap().commit();
        let refused = engines.claim("m", "vllm", GIB).unwrap_err();
        assert!(matches!(&refused, Error::Conflict { held_by, requested, .. } if held_by == LLAMA_CPP && requested == "vllm"));
        engines.claim("m", LLAMA_CPP, GIB).unwrap().commit();
        assert_eq!(budget.free(), 9 * GIB, "a reload keeps its reservation");
        assert_eq!(engines.status()[0].health, Health::InProcess);
        assert!(!engines.release("m", "vllm"));
        assert!(engines.release("m", LLAMA_CPP));
        assert_eq!(budget.free(), 10 * GIB);
    }

    #[test]
    fn claims_share_one_budget() {
        let budget = FixedBudget::new(10 * GIB);
        let engines = Engines::new(budget.clone());
        engines.claim("a", LLAMA_CPP, 6 * GIB).unwrap().commit();
        let refused = engines.claim("b", "sglang", 5 * GIB).unwrap_err();
        assert!(matches!(refused, Error::OverBudget { needed, free, .. } if needed == 5 * GIB && free == 4 * GIB));
        assert!(engines.holder("b").is_none());
        engines.claim("b", "sglang", 4 * GIB).unwrap().commit();
        assert_eq!(budget.free(), 0);
        engines.claim("c", LLAMA_CPP, 0).unwrap().commit();
    }

    #[test]
    fn an_uncommitted_claim_is_given_back() {
        let budget = FixedBudget::new(GIB);
        let engines = Engines::new(budget.clone());
        {
            let _claim = engines.claim("m", LLAMA_CPP, GIB).unwrap();
            assert_eq!(engines.status()[0].health, Health::Loading);
        }
        assert!(engines.holder("m").is_none());
        assert_eq!(budget.free(), GIB);
    }

    #[test]
    fn serving_llama_cpp_in_a_host_is_refused() {
        let engines = Engines::new(FixedBudget::new(GIB));
        let env = EngineEnv { kind: LLAMA_CPP.into(), root: "/x".into(), program: "p".into(), args: Vec::new(), digest: String::new(), loopback: false };
        assert!(engines.serve("m", &env, Path::new("/x"), Path::new("/x"), json!({}), 0).is_err());
        assert!(engines.holder("m").is_none());
    }

    #[test]
    fn a_reply_without_usage_is_not_metered_as_free() {
        let u = Usage::from_reply(&json!({"ok": true, "usage": {"prompt_tokens": 3, "completion_tokens": 5}})).unwrap();
        assert_eq!(u, Usage { prompt_tokens: 3, completion_tokens: 5 });
        assert!(Usage::from_reply(&json!({"ok": true})).is_err());
        assert!(Usage::from_reply(&json!({"ok": true, "usage": {"prompt_tokens": 3}})).is_err());
    }
}
