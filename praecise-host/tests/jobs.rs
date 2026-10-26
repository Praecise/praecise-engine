//! A job runs confined and supervised, through the real `run_job` and
//! launcher, with this test binary as the confinement launcher (so it has its
//! own `main`). The environment is a pinned directory whose `bin/python3` is a
//! shell script standing in for the adapter.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use praecise_host::engine::{EngineEnv, digest};
use praecise_host::job::{Confine, JobEngine, JobRequest, run_job};
use praecise_host::{Error, sandbox};
use serde_json::{Value, json};

const SCRIPT: &str = r#"#!/bin/sh
read -r line
case "$MODE" in
ok)
  echo '{"id":0,"event":{"progress":0.5,"message":"half"}}'
  echo hello > "$HOME/../out/hello.txt"
  if echo x > "$OUTSIDE" 2>/dev/null; then w=escaped; else w=held; fi
  if cat "$SECRET" >/dev/null 2>&1; then r=escaped; else r=held; fi
  echo "{\"id\":0,\"ok\":true,\"result\":{\"write\":\"$w\",\"read\":\"$r\",\"gpu\":\"$PRAECISE_GPU_REQUIRED\"}}"
  ;;
fail) echo '{"id":0,"ok":false,"error":"the program said no"}' ;;
sleep) sleep 30 ;;
exit) echo "dying" >&2; exit 3 ;;
esac
"#;

fn env_dir(base: &Path) -> PathBuf {
    let root = base.join("env");
    std::fs::create_dir_all(root.join("bin")).unwrap();
    let py = root.join("bin/python3");
    std::fs::write(&py, SCRIPT).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    root
}

fn engine(root: &Path, mode: &str, outside: &Path, secret: &Path) -> JobEngine {
    JobEngine {
        env: EngineEnv { kind: "cudaq".into(), root: root.to_path_buf(), program: "bin/python3".into(), args: Vec::new(), digest: digest(root).unwrap(), loopback: false },
        read_only: Vec::new(),
        vars: vec![
            ("MODE".into(), mode.into()),
            ("OUTSIDE".into(), outside.display().to_string()),
            ("SECRET".into(), secret.display().to_string()),
        ],
    }
}

fn request(run_dir: PathBuf, confine: Confine, timeout: u64) -> JobRequest {
    JobRequest {
        program: "cudaq.ghz".into(),
        params: json!({"qubits": 4}).as_object().unwrap().clone(),
        inputs: BTreeMap::new(),
        run_dir,
        timeout: Duration::from_secs(timeout),
        confine,
    }
}

fn main() {
    sandbox::enter_if_requested();
    if !cfg!(target_os = "linux") {
        println!("jobs: skipped (not Linux)");
        return;
    }
    let base = std::env::temp_dir().join(format!("praecise-host-jobs-{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let root = env_dir(&base);
    let outside = base.join("outside.txt");
    let secret = base.join("secret.txt");
    std::fs::write(&secret, "secret").unwrap();

    // landlock-only works on any kernel with Landlock: the run is confined to its scratch directory
    let mut events = Vec::new();
    let done = run_job(&engine(&root, "ok", &outside, &secret), request(base.join("run-ok"), Confine::LandlockOnly, 30), |e| events.push(e.clone()))
        .expect("a confined job runs");
    assert_eq!(events, vec![json!({"progress": 0.5, "message": "half"})]);
    assert_eq!(done.result["write"], "held", "the job wrote outside its scratch directory");
    assert_eq!(done.result["read"], "held", "the job read a file it was not given");
    assert!(!outside.exists());
    assert_eq!(std::fs::read_to_string(done.out_dir.join("hello.txt")).unwrap().trim(), "hello");
    let gpu = if sandbox::gpu_devices().is_empty() { "0" } else { "1" };
    assert_eq!(done.result["gpu"], Value::from(gpu));
    println!("jobs: landlock-only run confined ({:.2} s, digest check {:.3} s)", done.seconds, done.verify_seconds);

    let failed = run_job(&engine(&root, "fail", &outside, &secret), request(base.join("run-fail"), Confine::LandlockOnly, 30), |_| {}).unwrap_err();
    assert!(matches!(&failed, Error::Failed(m) if m.contains("the program said no")), "{failed}");

    let exited = run_job(&engine(&root, "exit", &outside, &secret), request(base.join("run-exit"), Confine::LandlockOnly, 30), |_| {}).unwrap_err();
    assert!(matches!(&exited, Error::Failed(m) if m.contains("without a reply") && m.contains("dying")), "{exited}");

    let t0 = std::time::Instant::now();
    let slow = run_job(&engine(&root, "sleep", &outside, &secret), request(base.join("run-sleep"), Confine::LandlockOnly, 2), |_| {}).unwrap_err();
    assert!(matches!(&slow, Error::Failed(m) if m.contains("time limit")), "{slow}");
    assert!(t0.elapsed() < Duration::from_secs(15), "the time limit was not enforced");
    println!("jobs: failures, exits and time limits are reported");

    // full confinement: confined, or refused naming the setting; never run unconfined
    match run_job(&engine(&root, "ok", &outside, &secret), request(base.join("run-full"), Confine::Full, 30), |_| {}) {
        Ok(done) => {
            assert_eq!(done.result["write"], "held");
            assert_eq!(done.result["read"], "held");
            println!("jobs: full confinement ran");
        }
        Err(Error::Failed(m)) => {
            assert!(m.contains("confinement refused") && m.contains("apparmor_restrict_unprivileged_userns"), "{m}");
            println!("jobs: full confinement refused on this kernel, as expected");
        }
        Err(e) => panic!("unexpected: {e}"),
    }
    let _ = std::fs::remove_dir_all(base);
}
