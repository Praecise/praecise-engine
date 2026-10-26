//! praecise-job: run a job engine.
//!
//! ```text
//! praecise-job programs
//! praecise-job digest <environment directory>
//! praecise-job run --engines engines.json --program cudaq.ghz --param qubits=12 [--params JSON]
//!                  [--input name=path]... --out DIR [--timeout SECONDS] [--confinement full|landlock-only]
//! ```
//!
//! `programs` prints the catalogue with every program's parameters and
//! inputs. `digest` prints the SHA-256 an environment is pinned by. `run`
//! checks the request against the program's declaration, verifies the
//! engine's environment against its pin, runs the program confined and
//! copies its output files into `--out`. While it runs, each progress event
//! is printed to stdout as one JSON line `{"progress": f, "message": m}`; the
//! last line is `{"ok": true, "result": {...}, "outputs": [...]}` or
//! `{"ok": false, "error": "..."}`. Exit status: 0 done, 1 failed, 2 refused.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use praecise_host::job::{self, Confine, JobRequest};
use praecise_host::{Error, sandbox};
use serde_json::{Map, Value, json};

const USAGE: &str = "usage: praecise-job programs | digest DIR | run --engines FILE --program NAME [--param k=v]... \
[--params JSON] [--input name=path]... --out DIR [--run-dir DIR] [--timeout SECONDS] [--confinement full|landlock-only]";

fn emit(v: &Value) {
    println!("{v}");
}

fn run(args: &[String]) -> Result<Value, Error> {
    let refused = |m: String| Error::Refused(m);
    let (mut engines, mut program, mut out, mut run_dir) = (None, None, None, None);
    let mut params = Map::new();
    let mut text_params = Vec::new();
    let mut inputs = BTreeMap::new();
    let mut timeout = Duration::from_secs(4 * 3600);
    let mut confine = Confine::Full;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| refused(format!("{a} needs a value")));
        match a.as_str() {
            "--engines" => engines = Some(PathBuf::from(val()?)),
            "--program" => program = Some(val()?),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--run-dir" => run_dir = Some(PathBuf::from(val()?)),
            "--param" => text_params.push(val()?),
            "--params" => {
                let v: Value = serde_json::from_str(&val()?).map_err(|e| refused(format!("--params: {e}")))?;
                let Value::Object(m) = v else { return Err(refused("--params is a JSON object".into())) };
                params.extend(m);
            }
            "--input" => {
                let v = val()?;
                let (k, p) = v.split_once('=').ok_or_else(|| refused(format!("--input {v}: name=path")))?;
                inputs.insert(k.to_string(), PathBuf::from(p));
            }
            "--timeout" => timeout = Duration::from_secs(val()?.parse().map_err(|_| refused("--timeout takes seconds".into()))?),
            "--confinement" => {
                confine = match val()?.as_str() {
                    "full" => Confine::Full,
                    "landlock-only" => Confine::LandlockOnly,
                    other => return Err(refused(format!("unknown confinement {other}"))),
                }
            }
            other => return Err(refused(format!("unknown argument {other}\n{USAGE}"))),
        }
    }
    let program = program.ok_or_else(|| refused(USAGE.into()))?;
    let out = out.ok_or_else(|| refused(USAGE.into()))?;
    let engines = engines.ok_or_else(|| refused(USAGE.into()))?;
    let decl = job::program(&program).ok_or_else(|| refused(format!("no job program {program}")))?;
    for t in text_params {
        let (k, v) = t.split_once('=').ok_or_else(|| refused(format!("--param {t}: name=value")))?;
        params.insert(k.to_string(), decl.parse_param(k, v)?);
    }
    let config: Value = serde_json::from_slice(&std::fs::read(&engines)?)?;
    let all = job::engines_from_json(&config)?;
    let engine = all.get(decl.engine).ok_or_else(|| refused(format!("{} needs the {} engine, which {} does not configure", decl.name, decl.engine, engines.display())))?;
    let run_dir = match run_dir {
        Some(d) => d,
        None => {
            let base = std::env::temp_dir().join(format!("praecise-job-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&base);
            base
        }
    };
    let keep = run_dir.clone();
    let req = JobRequest { program, params, inputs, run_dir, timeout, confine };
    let outcome = job::run_job(engine, req, emit);
    let outcome = match outcome {
        Ok(o) => o,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&keep);
            return Err(e);
        }
    };
    let outputs = job::copy_outputs(&outcome.out_dir, &out)?;
    let _ = std::fs::remove_dir_all(&keep);
    Ok(json!({"ok": true, "result": outcome.result, "outputs": outputs,
              "seconds": (outcome.seconds * 10.0).round() / 10.0, "verify_seconds": (outcome.verify_seconds * 10.0).round() / 10.0}))
}

fn main() -> ExitCode {
    sandbox::enter_if_requested();
    // a job's processes that outlive it are reparented here, so none is left behind
    #[cfg(target_os = "linux")]
    // SAFETY: sets a flag on this process only.
    unsafe {
        libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0);
    }
    let code = dispatch();
    job::kill_tree(std::process::id());
    code
}

fn dispatch() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("programs") => {
            emit(&Value::Array(job::programs().iter().map(job::Program::describe).collect()));
            ExitCode::SUCCESS
        }
        Some("digest") if args.len() == 2 => match praecise_host::engine::digest(std::path::Path::new(&args[1])) {
            Ok(d) => {
                println!("{d}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("praecise-job: {e}");
                ExitCode::FAILURE
            }
        },
        Some("run") => match run(&args[1..]) {
            Ok(v) => {
                emit(&v);
                ExitCode::SUCCESS
            }
            Err(e) => {
                let code = if matches!(e, Error::Refused(_)) { 2 } else { 1 };
                emit(&json!({"ok": false, "error": e.to_string()}));
                ExitCode::from(code)
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
