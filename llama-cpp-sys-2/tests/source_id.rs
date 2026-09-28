//! `LLAMA_SOURCE_ID` names the llama.cpp source the crate was compiled from.

use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn the_source_id_is_the_compiled_llama_cpp_commit() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("llama.cpp");
    let Some(toplevel) = git(&src, &["rev-parse", "--show-toplevel"]) else {
        assert_eq!(llama_cpp_sys_2::LLAMA_SOURCE_ID, "unknown");
        return;
    };
    let expected = if Path::new(&toplevel).canonicalize().ok() == src.canonicalize().ok() {
        git(&src, &["rev-parse", "--short=12", "HEAD"])
    } else {
        let rel = src
            .canonicalize()
            .unwrap()
            .strip_prefix(Path::new(&toplevel).canonicalize().unwrap())
            .unwrap()
            .to_string_lossy()
            .into_owned();
        git(
            Path::new(&toplevel),
            &["rev-parse", "--short=12", &format!("HEAD:{rel}")],
        )
    };
    assert_eq!(Some(llama_cpp_sys_2::LLAMA_SOURCE_ID.to_string()), expected);
    assert_ne!(llama_cpp_sys_2::LLAMA_SOURCE_ID, "unknown");
}
