//! A host is confined or refused, never run unconfined.
//!
//! On a kernel that allows unprivileged user namespaces the host runs with no
//! network interface but loopback and cannot read the user's home directory;
//! on one that does not, it is refused with a message naming the setting to
//! change. The check runs through the real `spawn`, with this test binary as
//! the confinement launcher, so it has its own `main`.

use std::path::Path;

use praecise_host::sandbox::{self, Confinement};

fn main() {
    sandbox::enter_if_requested();
    if !cfg!(target_os = "linux") {
        println!("confinement: skipped (not Linux)");
        return;
    }
    let scratch = std::env::temp_dir().join(format!("praecise-host-confinement-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let c = Confinement { read_only: Vec::new(), scratch: scratch.clone(), devices: Vec::new(), loopback: false };
    let probe = "tail -n +3 /proc/self/net/dev | cut -d: -f1 | tr -d ' '; ls \"$HOME\" >/dev/null 2>&1 && echo READ-HOME; echo DONE";
    let out = sandbox::spawn(&c, Path::new("/bin/sh"), &["-c".into(), probe.into()], &[("HOME".into(), home)])
        .expect("spawn")
        .wait_with_output()
        .expect("wait");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stdout.contains("DONE") {
        let nets: Vec<&str> = stdout.lines().filter(|l| !l.starts_with("READ-") && *l != "DONE").collect();
        assert_eq!(nets, vec!["lo"], "only loopback expected: {stdout}");
        assert!(!stdout.contains("READ-HOME"), "the home directory was readable: {stdout}");
        println!("confinement: confined (loopback only, home unreadable)");
    } else {
        assert!(stderr.contains("confinement refused"), "neither ran confined nor refused: {stderr}");
        assert!(
            stderr.contains("apparmor_restrict_unprivileged_userns") && stderr.contains("max_user_namespaces"),
            "a refusal must name the settings that allow confinement: {stderr}"
        );
        println!("confinement: refused as expected on this kernel: {}", stderr.trim());
    }
    let _ = std::fs::remove_dir_all(scratch);
}
