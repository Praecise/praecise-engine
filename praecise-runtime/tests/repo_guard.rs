//! Runs the repository guard (.githooks/guard.sh) as part of `cargo test`.
use std::path::PathBuf;
use std::process::Command;

#[test]
fn repository_guard() {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = loop {
        let candidate = dir.join(".githooks/guard.sh");
        if candidate.exists() {
            break candidate;
        }
        assert!(dir.pop(), ".githooks/guard.sh not found above the crate");
    };
    let status = Command::new("sh").arg(&script).status().expect("run guard");
    assert!(status.success(), "repository guard failed, see the lines above");
}
