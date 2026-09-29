//! Several engines serving several models at once, through the real runtime
//! host path: confined processes started from pinned environments.
//!
//! The engine is this test binary itself, copied into an environment
//! directory and run with `fake-engine`: it speaks the host protocol, and its
//! prompts steer it (`pid` answers its process id in its own PID namespace and
//! the time it started, which tells hosts apart; `crash` exits mid-request,
//! `net` tries to open internet sockets, `nousage` replies without token
//! counts). A kernel that refuses the confinement skips the run with the
//! reason, as the confinement test does.
//!
//! With `PRAECISE_TEST_TRANSFORMERS_ENV` (a Python environment with torch and
//! transformers) and `PRAECISE_TEST_TINY_MODEL` (a small causal language model
//! checkpoint) set, the bundled adapter is also run against that real engine.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use praecise_host::engine::{self, EngineEnv};
use praecise_host::generation::{self, Prompt};
use praecise_runtime::GenerationConfig;
use praecise_runtime::backend::Backend;
use praecise_host::{Engines, Error, FixedBudget, Health, HostedModel, sandbox};
use serde_json::{Value, json};

const GIB: u64 = 1 << 30;
const LLAMA_CPP: &str = Backend::LlamaCpp.as_str();

fn main() {
    sandbox::enter_if_requested();
    if std::env::args().nth(1).as_deref() == Some("fake-engine") {
        fake_engine();
        return;
    }
    if !cfg!(target_os = "linux") {
        println!("engines: skipped (not Linux)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("praecise-engines-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let t = Fixture::new(&dir);
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    match engines.serve("probe", &t.env_a, &t.model_dir, &dir.join("scratch-probe"), json!({}), 0) {
        Ok(_) => {
            engines.release("probe", "fake-a");
        }
        Err(e) if e.to_string().contains("confinement refused") => {
            println!("engines: skipped, the kernel refuses the confinement: {e}");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        Err(e) => panic!("probe start failed: {e}"),
    }
    let tests: [NamedTest; 7] = [
        ("two_engines_serve_two_models_at_once", two_engines_serve_two_models_at_once),
        ("a_model_is_never_loaded_twice", a_model_is_never_loaded_twice),
        ("an_engine_crash_is_isolated_and_restarted", an_engine_crash_is_isolated_and_restarted),
        ("the_engine_has_no_network", the_engine_has_no_network),
        ("every_engine_is_metered_the_same_way", every_engine_is_metered_the_same_way),
        ("a_changed_environment_is_never_run", a_changed_environment_is_never_run),
        ("the_budget_covers_hosted_and_in_process_engines", the_budget_covers_hosted_and_in_process_engines),
    ];
    for (name, test) in tests {
        test(&t);
        println!("test {name} ... ok");
    }
    real_transformers(&dir);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A test and the name it reports under.
type NamedTest = (&'static str, fn(&Fixture));

struct Fixture {
    dir: PathBuf,
    model_dir: PathBuf,
    env_a: EngineEnv,
    env_b: EngineEnv,
}

impl Fixture {
    fn new(dir: &Path) -> Self {
        let model_dir = dir.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("weights"), b"weights").unwrap();
        Self {
            dir: dir.to_path_buf(),
            model_dir,
            env_a: fake_env(dir, "fake-a"),
            env_b: fake_env(dir, "fake-b"),
        }
    }

    fn scratch(&self, model: &str) -> PathBuf {
        self.dir.join(format!("scratch-{model}"))
    }

    fn serve(&self, engines: &Engines, model: &str, env: &EngineEnv) -> praecise_host::Result<Arc<HostedModel>> {
        engines.serve(model, env, &self.model_dir, &self.scratch(model), json!({"id": model}), GIB)
    }
}

/// An environment holding a copy of this binary as its engine, pinned.
fn fake_env(dir: &Path, kind: &str) -> EngineEnv {
    let root = dir.join(format!("env-{kind}"));
    std::fs::create_dir_all(root.join("bin")).unwrap();
    std::fs::copy(std::env::current_exe().unwrap(), root.join("bin/engine")).unwrap();
    std::fs::write(root.join("KIND"), kind).unwrap();
    EngineEnv {
        kind: kind.into(),
        root: root.clone(),
        program: "bin/engine".into(),
        args: vec!["fake-engine".into()],
        digest: engine::digest(&root).unwrap(),
        loopback: false,
    }
}

fn ask(model: &Arc<HostedModel>, prompt: &str) -> praecise_host::Result<(Value, praecise_host::Usage, Vec<String>)> {
    let mut events = Vec::new();
    let (reply, usage) = model.generate(&json!({"op": "generate", "prompt": prompt}), |e| {
        events.push(e["text"].as_str().unwrap_or_default().to_string());
        Ok(())
    })?;
    Ok((reply, usage, events))
}

fn pid(model: &Arc<HostedModel>) -> String {
    ask(model, "pid").unwrap().0["text"].as_str().unwrap().to_string()
}

fn wait_for(model: &Arc<HostedModel>, want: impl Fn(&Health) -> bool) -> Health {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let h = model.health();
        if want(&h) {
            return h;
        }
        assert!(Instant::now() < deadline, "health never matched: {h:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn two_engines_serve_two_models_at_once(t: &Fixture) {
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    let a = t.serve(&engines, "alpha", &t.env_a).unwrap();
    let b = t.serve(&engines, "beta", &t.env_b).unwrap();
    let handles: Vec<_> = [Arc::clone(&a), Arc::clone(&b)]
        .into_iter()
        .map(|m| std::thread::spawn(move || (0..20).map(|_| ask(&m, "one two three").unwrap().0["text"].clone()).collect::<Vec<_>>()))
        .collect();
    let answers: Vec<Vec<Value>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(answers[0].iter().all(|v| v == "fake-a:alpha:one two three"), "{:?}", answers[0]);
    assert!(answers[1].iter().all(|v| v == "fake-b:beta:one two three"), "{:?}", answers[1]);
    assert_ne!(pid(&a), pid(&b));
    let status = engines.status();
    assert_eq!(
        status.iter().map(|s| (s.model_id.as_str(), s.engine.as_str(), s.health.clone())).collect::<Vec<_>>(),
        vec![("alpha", "fake-a", Health::Running { restarts: 0 }), ("beta", "fake-b", Health::Running { restarts: 0 })]
    );
}

fn a_model_is_never_loaded_twice(t: &Fixture) {
    let budget = FixedBudget::new(8 * GIB);
    let engines = Engines::new(budget.clone());
    let a = t.serve(&engines, "alpha", &t.env_a).unwrap();
    let first = pid(&a);
    for env in [&t.env_a, &t.env_b] {
        let refused = t.serve(&engines, "alpha", env).unwrap_err();
        assert!(matches!(&refused, Error::Conflict { held_by, .. } if held_by == "fake-a"), "{refused}");
    }
    let refused = engines.claim("alpha", LLAMA_CPP, GIB).unwrap_err();
    assert!(matches!(refused, Error::Conflict { .. }));
    engines.claim("gamma", LLAMA_CPP, GIB).unwrap().commit();
    let refused = t.serve(&engines, "gamma", &t.env_a).unwrap_err();
    assert!(matches!(&refused, Error::Conflict { held_by, .. } if held_by == LLAMA_CPP), "{refused}");
    assert_eq!(pid(&a), first, "the refused loads started no second host");
    assert_eq!(engines.status().len(), 2);
    assert_eq!(budget.free(), 6 * GIB);
}

fn an_engine_crash_is_isolated_and_restarted(t: &Fixture) {
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    let a = t.serve(&engines, "alpha", &t.env_a).unwrap();
    let b = t.serve(&engines, "beta", &t.env_b).unwrap();
    let (a_pid, b_pid) = (pid(&a), pid(&b));
    let crashed = ask(&a, "crash").unwrap_err();
    assert!(matches!(&crashed, Error::Failed(m) if m.contains("exited")), "{crashed}");
    assert_eq!(pid(&b), b_pid, "the other engine kept serving");
    wait_for(&a, |h| matches!(h, Health::Running { restarts: 1 }));
    assert_ne!(pid(&a), a_pid, "a new host serves the model");
    for _ in 0..praecise_host::engines::MAX_CONSECUTIVE_FAILURES {
        assert!(ask(&a, "crash").is_err());
    }
    let h = wait_for(&a, |h| matches!(h, Health::Stopped { .. }));
    assert!(matches!(&h, Health::Stopped { last_failure } if last_failure.contains("exited")), "{h:?}");
    let refused = ask(&a, "pid").unwrap_err();
    assert!(refused.to_string().contains("stopped"), "{refused}");
    assert_eq!(pid(&b), b_pid, "the other engine is still the same process");
    assert!(engines.release("alpha", "fake-a"));
    let again = t.serve(&engines, "alpha", &t.env_a).unwrap();
    assert!(!pid(&again).is_empty(), "a released model can be served again");
}

fn the_engine_has_no_network(t: &Fixture) {
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    let a = t.serve(&engines, "alpha", &t.env_a).unwrap();
    let text = ask(&a, "net").unwrap().0["text"].as_str().unwrap().to_string();
    assert!(text.starts_with("blocked"), "the engine opened an internet socket: {text}");
    assert!(pid(&a).starts_with("1:"), "the engine runs in its own PID namespace");
}

fn every_engine_is_metered_the_same_way(t: &Fixture) {
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    let a = t.serve(&engines, "alpha", &t.env_a).unwrap();
    let b = t.serve(&engines, "beta", &t.env_b).unwrap();
    for m in [&a, &b] {
        let (reply, usage, events) = ask(m, "one two three four").unwrap();
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 3));
        assert_eq!(events.concat(), reply["text"].as_str().unwrap(), "events stream the reply's text");
    }
    let config = GenerationConfig { max_tokens: 8, ..Default::default() };
    for m in [&a, &b] {
        let mut streamed = String::new();
        let r = generation::generate(m, Prompt::Text("one two"), &config, |t| {
            streamed.push_str(t);
            true
        })
        .unwrap();
        assert_eq!((r.input_tokens, r.output_tokens), (2, 3));
        assert_eq!(streamed, r.text);
    }
    let refused = ask(&a, "nousage").unwrap_err();
    assert!(refused.to_string().contains("no token usage"), "{refused}");
}

fn a_changed_environment_is_never_run(t: &Fixture) {
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    let mut wrong = t.env_a.clone();
    wrong.digest = "0".repeat(64);
    let refused = t.serve(&engines, "alpha", &wrong).unwrap_err();
    assert!(refused.to_string().contains("pinned"), "{refused}");
    assert!(engines.holder("alpha").is_none(), "a refused start leaves no claim");

    let env = fake_env(&t.dir, "fake-c");
    let c = t.serve(&engines, "gamma", &env).unwrap();
    std::fs::OpenOptions::new().append(true).open(env.root.join("KIND")).unwrap().write_all(b"!").unwrap();
    assert!(ask(&c, "crash").is_err());
    let h = wait_for(&c, |h| matches!(h, Health::Stopped { .. }));
    assert!(matches!(&h, Health::Stopped { last_failure } if last_failure.contains("pinned")), "{h:?}");
}

fn the_budget_covers_hosted_and_in_process_engines(t: &Fixture) {
    let engines = Engines::new(FixedBudget::new(2 * GIB));
    engines.claim("local", LLAMA_CPP, GIB).unwrap().commit();
    let _a = t.serve(&engines, "alpha", &t.env_a).unwrap();
    let refused = t.serve(&engines, "beta", &t.env_b).unwrap_err();
    assert!(matches!(refused, Error::OverBudget { free: 0, .. }), "{refused}");
    assert!(engines.release("alpha", "fake-a"));
    let _b = t.serve(&engines, "beta", &t.env_b).unwrap();
}

/// The bundled adapter over a real engine, when one is configured.
fn real_transformers(dir: &Path) {
    let (Ok(venv), Ok(model)) = (std::env::var("PRAECISE_TEST_TRANSFORMERS_ENV"), std::env::var("PRAECISE_TEST_TINY_MODEL")) else {
        println!("test real_transformers_engine ... skipped (PRAECISE_TEST_TRANSFORMERS_ENV and PRAECISE_TEST_TINY_MODEL unset)");
        return;
    };
    let root = PathBuf::from(venv);
    engine::install_adapter(&root).unwrap();
    let env = EngineEnv::adapter(Backend::Transformers, &root, &engine::digest(&root).unwrap()).unwrap();
    let engines = Engines::new(FixedBudget::new(8 * GIB));
    let fake = fake_env(dir, "fake-r");
    let tiny = engines
        .serve("tiny", &env, Path::new(&model), &dir.join("scratch-tiny"), json!({"options": {"dtype": "float32"}}), 0)
        .unwrap();
    let other = engines.serve("alpha", &fake, &dir.join("model"), &dir.join("scratch-alpha"), json!({}), 0).unwrap();
    let mut events = Vec::new();
    let (reply, usage) = tiny
        .generate(&json!({"op": "generate", "prompt": "hello world", "max_tokens": 8, "temperature": 0}), |e| {
            events.push(e["text"].as_str().unwrap().to_string());
            Ok(())
        })
        .unwrap();
    assert!(usage.prompt_tokens > 0 && usage.completion_tokens > 0 && usage.completion_tokens <= 8, "{usage:?}");
    assert_eq!(events.concat(), reply["text"].as_str().unwrap());
    assert!(ask(&other, "pid").is_ok(), "the fake engine serves beside the real one");
    let config = GenerationConfig { max_tokens: 6, temperature: 0.0, ..Default::default() };
    let typed = generation::generate(&tiny, Prompt::Text("hello world"), &config, |_| true).unwrap();
    assert!(typed.input_tokens > 0 && typed.output_tokens > 0 && typed.output_tokens <= 6, "{typed:?}");
    let refused = engines
        .serve("tiny", &env, Path::new(&model), &dir.join("scratch-tiny"), json!({}), 0)
        .unwrap_err();
    assert!(matches!(refused, Error::Conflict { .. }));
    println!("test real_transformers_engine ... ok ({usage:?}, {} events)", events.len());
}

/// The fake engine: the host protocol on stdin and stdout.
fn fake_engine() {
    let kind = std::fs::read_to_string(std::env::current_exe().unwrap().parent().unwrap().join("../KIND")).unwrap();
    let started = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let mut model = String::new();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let req: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let id = req["id"].clone();
        let mut reply = match req["op"].as_str() {
            Some("load") => {
                model = req["entry"]["id"].as_str().unwrap_or("probe").to_string();
                json!({"ok": true})
            }
            Some("generate") => {
                let prompt = req["prompt"].as_str().unwrap_or_default();
                let text = match prompt {
                    "pid" => format!("{}:{started}", std::process::id()),
                    "crash" => std::process::exit(3),
                    "net" => net_probe(),
                    _ => format!("{kind}:{model}:{prompt}"),
                };
                for piece in [&text[..text.len() / 2], &text[text.len() / 2..]] {
                    writeln!(out, "{}", json!({"id": id, "event": {"text": piece}})).unwrap();
                }
                if prompt == "nousage" {
                    json!({"ok": true, "text": text})
                } else {
                    json!({"ok": true, "text": text, "usage": {"prompt_tokens": prompt.split_whitespace().count(), "completion_tokens": 3}})
                }
            }
            _ => json!({"ok": false, "error": "unknown op"}),
        };
        reply["id"] = id;
        writeln!(out, "{reply}").unwrap();
        out.flush().unwrap();
    }
}

/// Try every way out: internet sockets of both families, TCP and UDP.
fn net_probe() -> String {
    let attempts = [
        std::net::TcpStream::connect_timeout(&"192.0.2.1:80".parse().unwrap(), Duration::from_secs(2)).map(|_| "tcp4"),
        std::net::TcpStream::connect_timeout(&"[2001:db8::1]:80".parse().unwrap(), Duration::from_secs(2)).map(|_| "tcp6"),
        std::net::UdpSocket::bind("0.0.0.0:0").map(|_| "udp4"),
        std::net::TcpListener::bind("127.0.0.1:0").map(|_| "listen4"),
    ];
    let opened: Vec<&str> = attempts.iter().filter_map(|a| a.as_ref().ok().copied()).collect();
    if opened.is_empty() {
        let errors: Vec<String> = attempts.iter().filter_map(|a| a.as_ref().err().map(ToString::to_string)).collect();
        format!("blocked: {}", errors.join("; "))
    } else {
        format!("open: {}", opened.join(","))
    }
}
