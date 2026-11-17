//! What the `download-binaries` feature brings into the build. Without it a
//! build that provides ONNX Runtime through ORT_LIB_LOCATION compiles no
//! download path and no TLS stack, so no second crypto provider beside the
//! host's own. Reads the locked graph with `cargo tree`.

use std::process::Command;

fn packages(features: &[&str]) -> Vec<String> {
    let out = Command::new(env!("CARGO"))
        .args(["tree", "--locked", "-p", "praecise-onnx", "-e", "normal,build"])
        .args(["--prefix", "none", "--format", "{p}"])
        .args(features)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run cargo tree");
    assert!(out.status.success(), "cargo tree: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
        .collect()
}

#[test]
fn without_the_download_no_tls_stack_is_built() {
    let with = packages(&[]);
    assert!(with.iter().any(|p| p == "ureq"), "the default build fetches the runtime");
    let without = packages(&["--no-default-features"]);
    for name in ["ureq", "rustls", "ring", "aws-lc-rs", "native-tls"] {
        assert!(!without.iter().any(|p| p == name), "{name} is built without download-binaries");
    }
    assert!(without.iter().any(|p| p == "ort-sys"), "the runtime bindings are still built");
}
