//! Engines and the environments they run from.
//!
//! Every model is served by exactly one engine. A linked backend
//! ([`Backend::LlamaCpp`]) runs inside the application's own process; every
//! other engine runs in a runtime host started from an [`EngineEnv`]: a
//! directory holding the engine's program, its packages and its protocol
//! adapter, pinned by the SHA-256 digest of its whole tree (see [`digest`]).
//! The digest is checked before every start, so an environment that changed
//! after it was pinned is refused, never run.
//!
//! The hosted backends ([`Integration::Hosted`]) are driven through their
//! offline Python engine APIs by one adapter, [`ADAPTER`], which speaks the
//! runtime host protocol on its standard streams. The engine library runs
//! inside the confined host: it opens no port and reaches no network.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use praecise_runtime::backend::{Backend, Integration};
use sha2::{Digest, Sha256};

use crate::host::HostSpec;
use crate::{Error, Result};

/// File name of the adapter inside an environment.
pub const ADAPTER_FILE: &str = "praecise_adapter.py";

/// The protocol adapter for the hosted backends.
pub const ADAPTER: &str = include_str!("../adapters/praecise_adapter.py");

/// An engine environment pinned by the digest of its tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineEnv {
    /// Engine kind, the name its claims and admission use.
    pub kind: String,
    /// The environment directory.
    pub root: PathBuf,
    /// The program to run, relative to `root`.
    pub program: PathBuf,
    /// Its arguments.
    pub args: Vec<String>,
    /// Lowercase hex SHA-256 of the tree, as [`digest`] computes it.
    pub digest: String,
    /// The engine's processes rendezvous over 127.0.0.1 (see
    /// [`HostSpec::loopback`]).
    pub loopback: bool,
}

impl EngineEnv {
    /// An environment for a hosted `backend`: a Python environment at `root`
    /// with the engine's packages installed and [`ADAPTER`] written by
    /// [`install_adapter`], run as `bin/python3 praecise_adapter.py <backend>`.
    /// Every backend but transformers gets loopback: their workers and
    /// collective-communication setup meet on 127.0.0.1 even on one GPU.
    ///
    /// # Errors
    /// When `backend` is linked, not hosted.
    pub fn adapter(backend: Backend, root: &Path, digest: &str) -> Result<Self> {
        if backend.supports().integration != Integration::Hosted {
            return Err(Error::Refused(format!("{backend} is linked into the process, not hosted")));
        }
        Ok(Self {
            kind: backend.as_str().to_string(),
            root: root.to_path_buf(),
            program: PathBuf::from("bin/python3"),
            args: vec![root.join(ADAPTER_FILE).display().to_string(), backend.as_str().to_string()],
            digest: digest.to_ascii_lowercase(),
            loopback: backend != Backend::Transformers,
        })
    }

    /// Check the tree against the pinned digest and describe how to start
    /// the engine from it.
    ///
    /// # Errors
    /// When the program is outside the environment, the tree cannot be read,
    /// or its digest differs from the pin.
    pub fn verified_spec(&self) -> Result<HostSpec> {
        if self.program.is_absolute() || self.program.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(Error::Refused(format!(
                "{}: the program {} is not inside the environment",
                self.kind,
                self.program.display()
            )));
        }
        let found = digest(&self.root)?;
        if found != self.digest {
            return Err(Error::Refused(format!(
                "{}: environment {} has digest {found}, pinned {}",
                self.kind,
                self.root.display(),
                self.digest
            )));
        }
        Ok(HostSpec {
            kind: self.kind.clone(),
            program: self.root.join(&self.program),
            args: self.args.iter().map(Into::into).collect(),
            env: vec![("PRAECISE_ENGINE".to_string(), self.kind.clone())],
            read_only: vec![self.root.clone()],
            loopback: self.loopback,
        })
    }
}

/// Write [`ADAPTER`] into the environment at `root`. Run once when the
/// environment is provisioned, before its digest is taken.
///
/// # Errors
/// When the file cannot be written.
pub fn install_adapter(root: &Path) -> Result<()> {
    std::fs::write(root.join(ADAPTER_FILE), ADAPTER)?;
    Ok(())
}

/// One entry of a tree, as it enters the digest.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Entry {
    Dir,
    File { executable: bool, sha256: [u8; 32] },
    Link(Vec<u8>),
}

/// The SHA-256 digest of the tree at `root`, as lowercase hex.
///
/// Every directory, file and symbolic link under `root` enters the digest in
/// path order: its path relative to `root`, its type, and for a file its
/// executable bit and the SHA-256 of its contents, for a link its target as
/// written. Links are not followed.
///
/// # Errors
/// When the tree cannot be read, or holds something other than directories,
/// regular files and links.
pub fn digest(root: &Path) -> Result<String> {
    let mut entries = Vec::new();
    walk(root, root, &mut entries)?;
    entries.sort();
    let mut h = Sha256::new();
    for (rel, entry) in &entries {
        h.update(rel);
        h.update([0]);
        match entry {
            Entry::Dir => h.update(b"d"),
            Entry::File { executable, sha256 } => {
                h.update(if *executable { b"x" } else { b"f" });
                h.update(sha256);
            }
            Entry::Link(target) => {
                h.update(b"l");
                h.update(target);
                h.update([0]);
            }
        }
    }
    Ok(hex(&h.finalize()))
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(Vec<u8>, Entry)>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let path = e.path();
        let rel = path.strip_prefix(root).expect("under root").as_os_str().as_encoded_bytes().to_vec();
        let meta = std::fs::symlink_metadata(&path)?;
        let ft = meta.file_type();
        if ft.is_symlink() {
            let target = std::fs::read_link(&path)?.into_os_string().into_encoded_bytes();
            out.push((rel, Entry::Link(target)));
        } else if ft.is_dir() {
            out.push((rel, Entry::Dir));
            walk(root, &path, out)?;
        } else if ft.is_file() {
            out.push((rel, Entry::File { executable: executable(&meta), sha256: file_sha256(&path)? }));
        } else {
            return Err(Error::Refused(format!("{} is not a file, directory or link", path.display())));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn executable(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(_: &std::fs::Metadata) -> bool {
    false
}

fn file_sha256(path: &Path) -> Result<[u8; 32]> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().into())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> PathBuf {
        let root = std::env::temp_dir().join(format!("praecise-engine-env-{}-{}", std::process::id(), rand_suffix()));
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(root.join("bin/engine"), b"#!/bin/sh\n").unwrap();
        std::fs::write(root.join("lib.txt"), b"library").unwrap();
        root
    }

    fn rand_suffix() -> u128 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    }

    #[test]
    fn the_digest_is_stable_and_sees_every_change() {
        let root = tree();
        let pinned = digest(&root).unwrap();
        assert_eq!(pinned.len(), 64);
        assert_eq!(digest(&root).unwrap(), pinned);
        std::fs::write(root.join("lib.txt"), b"librarz").unwrap();
        let edited = digest(&root).unwrap();
        assert_ne!(edited, pinned);
        std::fs::write(root.join("extra"), b"").unwrap();
        assert_ne!(digest(&root).unwrap(), edited);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_changed_environment_is_refused_before_it_starts() {
        let root = tree();
        let env = EngineEnv {
            kind: "test".into(),
            root: root.clone(),
            program: "bin/engine".into(),
            args: Vec::new(),
            digest: digest(&root).unwrap(),
            loopback: false,
        };
        let spec = env.verified_spec().unwrap();
        assert_eq!(spec.program, root.join("bin/engine"));
        assert_eq!(spec.read_only, vec![root.clone()]);
        std::fs::write(root.join("bin/engine"), b"#!/bin/sh\nexit 1\n").unwrap();
        let refused = env.verified_spec().unwrap_err();
        assert!(matches!(&refused, Error::Refused(m) if m.contains("pinned")), "{refused}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_program_outside_the_environment_is_refused() {
        let root = tree();
        let digest = digest(&root).unwrap();
        for program in ["/bin/sh", "../bin/sh"] {
            let env = EngineEnv { kind: "t".into(), root: root.clone(), program: program.into(), args: Vec::new(), digest: digest.clone(), loopback: false };
            assert!(env.verified_spec().is_err(), "{program}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_hosted_backends_get_the_adapter() {
        let root = Path::new("/nonexistent");
        let env = EngineEnv::adapter(Backend::Vllm, root, "AB").unwrap();
        assert!(env.loopback);
        assert!(!EngineEnv::adapter(Backend::Transformers, root, "ab").unwrap().loopback);
        assert_eq!(env.args, vec![root.join(ADAPTER_FILE).display().to_string(), "vllm".to_string()]);
        assert_eq!(env.digest, "ab");
        assert!(EngineEnv::adapter(Backend::LlamaCpp, root, "ab").is_err());
    }
}
