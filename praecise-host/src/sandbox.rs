//! The confinement a runtime host runs in.
//!
//! A runtime host is a process the application starts for an engine it does
//! not run in its own process. It is confined by the kernel rather than
//! trusted:
//!
//! * new user, network, mount, IPC and PID namespaces, with the application's
//!   own uid and gid mapped one-to-one and nothing else, so the host has no network
//!   interface at all, sees no other process, and holds no capability;
//! * a Landlock ruleset: read-only access to the paths it runs from and the
//!   model files it serves, read-write access to one scratch directory and to
//!   the GPU device nodes, and nothing else on the filesystem;
//! * a seccomp filter refusing the calls that could widen any of that
//!   (namespace, mount, module, tracing, keyring and BPF calls) and every
//!   internet socket;
//! * `no_new_privs` and resource limits.
//!
//! The host talks to the application only over its stdin and stdout.
//!
//! The application starts a host by running its own executable with
//! [`SANDBOX_ARG`] first, so the confinement is applied by a process that is
//! still single-threaded; [`enter_if_requested`] runs that path from `main`
//! before anything else. When the kernel refuses unprivileged user namespaces
//! the host is not started at all, and the error names the setting to change.

use std::ffi::{CString, OsString};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use crate::{Error, Result};

/// First argument that turns the application's executable into the
/// confinement launcher.
pub const SANDBOX_ARG: &str = "__praecise-sandbox";

/// Exit status the launcher uses when the kernel refuses a confinement step.
const EXIT_REFUSED: i32 = 70;

/// What a host may touch.
#[derive(Debug, Clone, Default)]
pub struct Confinement {
    /// Paths readable (and executable) by the host: its interpreter, its
    /// libraries, the model files.
    pub read_only: Vec<PathBuf>,
    /// The one directory the host may write.
    pub scratch: PathBuf,
    /// Device nodes the host may open read-write (the GPU).
    pub devices: Vec<PathBuf>,
    /// Bring up loopback in the host's network namespace and allow
    /// internet-family sockets on it; the namespace has no route out.
    pub loopback: bool,
}

impl Confinement {
    fn encode(&self) -> Vec<OsString> {
        let mut args = vec![OsString::from(SANDBOX_ARG), OsString::from("--scratch"), self.scratch.clone().into()];
        if self.loopback {
            args.push("--loopback".into());
        }
        for p in &self.read_only {
            args.push("--ro".into());
            args.push(p.clone().into());
        }
        for p in &self.devices {
            args.push("--dev".into());
            args.push(p.clone().into());
        }
        args.push("--".into());
        args
    }
}

/// The GPU device nodes present on this host.
#[must_use]
pub fn gpu_devices() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/dev") {
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("nvidia") || name == "kfd" {
                out.push(e.path());
            }
        }
    }
    if Path::new("/dev/dri").is_dir() {
        out.push(PathBuf::from("/dev/dri"));
    }
    out.sort();
    out
}

/// Start `program args...` confined by `c`, with piped stdin and stdout.
///
/// # Errors
/// When the application's executable cannot be found or the process cannot
/// be started. A refusal by the kernel surfaces when the host's stderr is
/// read: the launcher exits with a message naming the setting that blocks it.
pub fn spawn(c: &Confinement, program: &Path, args: &[OsString], env: &[(String, String)]) -> Result<Child> {
    if !cfg!(target_os = "linux") {
        return Err(Error::Start("a runtime host needs Linux namespaces and Landlock".into()));
    }
    let exe = std::env::current_exe().map_err(|e| Error::Start(format!("locating the executable: {e}")))?;
    let mut cmd = Command::new(exe);
    cmd.args(c.encode()).arg(program).args(args);
    cmd.env_clear();
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.spawn().map_err(|e| Error::Start(e.to_string()))
}

/// When this process was started as the confinement launcher, confine and
/// exec the requested program; never returns in that case. Otherwise returns
/// immediately. Call first thing in `main`, before any thread is started.
pub fn enter_if_requested() {
    let mut args = std::env::args_os();
    let _exe = args.next();
    if args.next().as_deref() != Some(std::ffi::OsStr::new(SANDBOX_ARG)) {
        return;
    }
    let rest: Vec<OsString> = args.collect();
    #[cfg(target_os = "linux")]
    {
        let code = match linux::launch(&rest) {
            Ok(code) => code,
            Err(msg) => {
                eprintln!("runtime host confinement refused: {msg}");
                EXIT_REFUSED
            }
        };
        std::process::exit(code);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = rest;
        eprintln!("runtime host confinement refused: needs Linux");
        std::process::exit(EXIT_REFUSED);
    }
}

fn cstring(p: &Path) -> std::result::Result<CString, String> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(p.as_os_str().as_bytes()).map_err(|_| format!("path {} contains NUL", p.display()))
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{Confinement, cstring};
    use std::ffi::{CString, OsString};
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;

    fn errno() -> std::io::Error {
        std::io::Error::last_os_error()
    }

    fn parse(args: &[OsString]) -> std::result::Result<(Confinement, Vec<OsString>), String> {
        let mut c = Confinement::default();
        let mut i = 0;
        while i < args.len() {
            let a = args[i].to_string_lossy().into_owned();
            match a.as_str() {
                "--scratch" | "--ro" | "--dev" => {
                    let v = PathBuf::from(args.get(i + 1).ok_or("missing path")?);
                    match a.as_str() {
                        "--scratch" => c.scratch = v,
                        "--ro" => c.read_only.push(v),
                        _ => c.devices.push(v),
                    }
                    i += 2;
                }
                "--loopback" => {
                    c.loopback = true;
                    i += 1;
                }
                "--" => return Ok((c, args[i + 1..].to_vec())),
                other => return Err(format!("unexpected launcher argument {other}")),
            }
        }
        Err("no program given".into())
    }

    /// Set the loopback interface of the current network namespace up.
    fn loopback_up() -> std::result::Result<(), String> {
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0) };
        if fd < 0 {
            return Err(format!("socket for loopback: {}", errno()));
        }
        let mut req: libc::ifreq = unsafe { std::mem::zeroed() };
        for (dst, src) in req.ifr_name.iter_mut().zip(b"lo\0") {
            *dst = *src as libc::c_char;
        }
        let r = unsafe { libc::ioctl(fd, libc::SIOCGIFFLAGS, &mut req) };
        let r = if r == 0 {
            unsafe {
                req.ifr_ifru.ifru_flags |= (libc::IFF_UP | libc::IFF_RUNNING) as libc::c_short;
                libc::ioctl(fd, libc::SIOCSIFFLAGS, &req)
            }
        } else {
            r
        };
        unsafe { libc::close(fd) };
        if r != 0 {
            return Err(format!("bringing loopback up: {}", errno()));
        }
        Ok(())
    }

    fn write_file(path: &str, contents: &str) -> std::result::Result<(), String> {
        std::fs::write(path, contents).map_err(|e| format!("writing {path}: {e}"))
    }

    /// Confine this process and exec the program as PID 1 of a new PID
    /// namespace; relays its exit status.
    pub(super) fn launch(args: &[OsString]) -> std::result::Result<i32, String> {
        let (c, program) = parse(args)?;
        if program.is_empty() {
            return Err("no program given".into());
        }
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        let flags = libc::CLONE_NEWUSER
            | libc::CLONE_NEWNS
            | libc::CLONE_NEWNET
            | libc::CLONE_NEWIPC
            | libc::CLONE_NEWPID;
        // Under AppArmor's unprivileged user namespace restriction the unshare
        // succeeds but the namespace has no capabilities, so the refusal shows
        // up at the first write to the id maps.
        const ALLOW: &str = "Allow unprivileged user namespaces for this executable: on Ubuntu install an \
             AppArmor profile granting `userns` to it (or set kernel.apparmor_restrict_unprivileged_userns=0); \
             elsewhere check user.max_user_namespaces > 0 and kernel.unprivileged_userns_clone = 1";
        if unsafe { libc::unshare(flags) } != 0 {
            return Err(format!("the kernel refused new user namespaces ({}). {ALLOW}", errno()));
        }
        let refused = |e: String| format!("{e}; the kernel refused the user namespace's id mapping. {ALLOW}");
        write_file("/proc/self/setgroups", "deny").map_err(refused)?;
        write_file("/proc/self/uid_map", &format!("{uid} {uid} 1")).map_err(refused)?;
        write_file("/proc/self/gid_map", &format!("{gid} {gid} 1")).map_err(refused)?;

        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(format!("fork: {}", errno()));
        }
        if pid > 0 {
            let mut status = 0;
            loop {
                let r = unsafe { libc::waitpid(pid, &mut status, 0) };
                if r == pid {
                    break;
                }
                if r < 0 && errno().kind() != std::io::ErrorKind::Interrupted {
                    return Err(format!("waitpid: {}", errno()));
                }
            }
            return Ok(if libc::WIFEXITED(status) { libc::WEXITSTATUS(status) } else { 128 + libc::WTERMSIG(status) });
        }

        // PID 1 of the new namespaces from here on.
        let child = || -> std::result::Result<(), String> {
            if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } != 0 {
                return Err(format!("PR_SET_PDEATHSIG: {}", errno()));
            }
            let none = CString::new("none").unwrap();
            let root = CString::new("/").unwrap();
            if unsafe { libc::mount(none.as_ptr(), root.as_ptr(), std::ptr::null(), libc::MS_REC | libc::MS_PRIVATE, std::ptr::null()) } != 0 {
                return Err(format!("making mounts private: {}", errno()));
            }
            let proc_ = CString::new("proc").unwrap();
            let proc_dir = CString::new("/proc").unwrap();
            if unsafe {
                libc::mount(proc_.as_ptr(), proc_dir.as_ptr(), proc_.as_ptr(), libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC, std::ptr::null())
            } != 0
            {
                return Err(format!("mounting /proc: {}", errno()));
            }
            limits()?;
            if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
                return Err(format!("no_new_privs: {}", errno()));
            }
            if c.loopback {
                loopback_up()?;
            }
            landlock(&c)?;
            seccomp(c.loopback)?;
            let prog: Vec<CString> = program
                .iter()
                .map(|a| CString::new(a.as_bytes()).map_err(|_| "argument contains NUL".to_string()))
                .collect::<std::result::Result<_, _>>()?;
            let mut argv: Vec<*const libc::c_char> = prog.iter().map(|a| a.as_ptr()).collect();
            argv.push(std::ptr::null());
            unsafe { libc::execvp(prog[0].as_ptr(), argv.as_ptr()) };
            Err(format!("exec {}: {}", program[0].to_string_lossy(), errno()))
        };
        match child() {
            Ok(()) => unreachable!("exec returned without an error"),
            Err(e) => {
                eprintln!("runtime host confinement refused: {e}");
                unsafe { libc::_exit(super::EXIT_REFUSED) };
            }
        }
    }

    fn limits() -> std::result::Result<(), String> {
        for (res, v) in [(libc::RLIMIT_CORE, 0u64), (libc::RLIMIT_NOFILE, 4096)] {
            let l = libc::rlimit { rlim_cur: v, rlim_max: v };
            if unsafe { libc::setrlimit(res, &l) } != 0 {
                return Err(format!("setrlimit: {}", errno()));
            }
        }
        Ok(())
    }

    // Landlock ABI v1 filesystem rights.
    const FS_EXECUTE: u64 = 1 << 0;
    const FS_WRITE_FILE: u64 = 1 << 1;
    const FS_READ_FILE: u64 = 1 << 2;
    const FS_READ_DIR: u64 = 1 << 3;
    const FS_REMOVE_DIR: u64 = 1 << 4;
    const FS_REMOVE_FILE: u64 = 1 << 5;
    const FS_MAKE_CHAR: u64 = 1 << 6;
    const FS_MAKE_DIR: u64 = 1 << 7;
    const FS_MAKE_REG: u64 = 1 << 8;
    const FS_MAKE_SOCK: u64 = 1 << 9;
    const FS_MAKE_FIFO: u64 = 1 << 10;
    const FS_MAKE_BLOCK: u64 = 1 << 11;
    const FS_MAKE_SYM: u64 = 1 << 12;
    const FS_ALL_V1: u64 = (1 << 13) - 1;
    const READ: u64 = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR;
    const WRITE: u64 = FS_WRITE_FILE
        | FS_REMOVE_DIR
        | FS_REMOVE_FILE
        | FS_MAKE_DIR
        | FS_MAKE_REG
        | FS_MAKE_SOCK
        | FS_MAKE_FIFO
        | FS_MAKE_SYM;
    const _: u64 = FS_MAKE_CHAR | FS_MAKE_BLOCK;
    const RULE_PATH_BENEATH: libc::c_int = 1;

    #[repr(C)]
    struct RulesetAttr {
        handled_access_fs: u64,
    }

    #[repr(C, packed)]
    struct PathBeneath {
        allowed_access: u64,
        parent_fd: i32,
    }

    /// Restrict the filesystem to the confinement's paths. Everything not
    /// named is refused; the system's executables and libraries are named
    /// read-only because the interpreter needs them.
    fn landlock(c: &Confinement) -> std::result::Result<(), String> {
        let attr = RulesetAttr { handled_access_fs: FS_ALL_V1 };
        let fd = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                &attr as *const RulesetAttr,
                std::mem::size_of::<RulesetAttr>(),
                0u32,
            )
        } as i32;
        if fd < 0 {
            return Err(format!("the kernel has no Landlock ({}); the runtime host needs it", errno()));
        }
        let mut rules: Vec<(PathBuf, u64)> = Vec::new();
        for sys in ["/usr", "/lib", "/lib64", "/bin", "/etc", "/sys", "/proc", "/dev/null", "/dev/zero", "/dev/urandom"] {
            rules.push((PathBuf::from(sys), READ));
        }
        rules.push((PathBuf::from("/dev/null"), READ | FS_WRITE_FILE));
        for p in &c.read_only {
            rules.push((p.clone(), READ));
        }
        for p in &c.devices {
            rules.push((p.clone(), READ | FS_WRITE_FILE));
        }
        rules.push((c.scratch.clone(), READ | WRITE));
        for (path, access) in rules {
            if !path.exists() {
                continue;
            }
            let cpath = cstring(&path)?;
            let pfd = unsafe { libc::open(cpath.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
            if pfd < 0 {
                return Err(format!("opening {}: {}", path.display(), errno()));
            }
            // A rule on a file may only carry file rights.
            let access = if path.is_dir() { access } else { access & (FS_EXECUTE | FS_READ_FILE | FS_WRITE_FILE) };
            let rule = PathBeneath { allowed_access: access, parent_fd: pfd };
            let r = unsafe { libc::syscall(libc::SYS_landlock_add_rule, fd, RULE_PATH_BENEATH, &rule as *const PathBeneath, 0u32) };
            unsafe { libc::close(pfd) };
            if r != 0 {
                return Err(format!("Landlock rule for {}: {}", path.display(), errno()));
            }
        }
        let r = unsafe { libc::syscall(libc::SYS_landlock_restrict_self, fd, 0u32) };
        unsafe { libc::close(fd) };
        if r != 0 {
            return Err(format!("Landlock restrict: {}", errno()));
        }
        Ok(())
    }

    /// Calls refused with EPERM.
    const DENIED: &[libc::c_long] = &[
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_kexec_load,
        libc::SYS_kexec_file_load,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_bpf,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_swapon,
        libc::SYS_swapoff,
        libc::SYS_reboot,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_userfaultfd,
        libc::SYS_open_by_handle_at,
        libc::SYS_name_to_handle_at,
        libc::SYS_acct,
    ];

    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH: u32 = 0xC000_003E;
    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH: u32 = 0xC000_00B7;

    fn stmt(code: u16, k: u32) -> libc::sock_filter {
        libc::sock_filter { code, jt: 0, jf: 0, k }
    }
    fn jeq(k: u32, jt: u8, jf: u8) -> libc::sock_filter {
        libc::sock_filter { code: (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16, jt, jf, k }
    }

    /// Install the syscall filter: kill on a foreign architecture, EPERM for
    /// [`DENIED`], for packet sockets, and (unless `loopback`) for internet
    /// sockets; allow the rest.
    fn seccomp(loopback: bool) -> std::result::Result<(), String> {
        const LD_W_ABS: u16 = (libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16;
        const RET: u16 = (libc::BPF_RET | libc::BPF_K) as u16;
        let eperm = libc::SECCOMP_RET_ERRNO | (libc::EPERM as u32);
        let mut f = vec![
            stmt(LD_W_ABS, 4), // arch
            jeq(AUDIT_ARCH, 1, 0),
            stmt(RET, libc::SECCOMP_RET_KILL_PROCESS),
            stmt(LD_W_ABS, 0), // nr
        ];
        for nr in DENIED {
            f.push(jeq(*nr as u32, 0, 1));
            f.push(stmt(RET, eperm));
        }
        // socket(domain, ...): args[0] at offset 16.
        if loopback {
            f.push(jeq(libc::SYS_socket as u32, 0, 3));
            f.push(stmt(LD_W_ABS, 16));
            f.push(jeq(libc::AF_PACKET as u32, 0, 1));
        } else {
            f.push(jeq(libc::SYS_socket as u32, 0, 5));
            f.push(stmt(LD_W_ABS, 16));
            f.push(jeq(libc::AF_INET as u32, 2, 0));
            f.push(jeq(libc::AF_INET6 as u32, 1, 0));
            f.push(jeq(libc::AF_PACKET as u32, 0, 1));
        }
        f.push(stmt(RET, eperm));
        f.push(stmt(RET, libc::SECCOMP_RET_ALLOW));
        let prog = libc::sock_fprog { len: f.len() as u16, filter: f.as_mut_ptr() };
        if unsafe { libc::prctl(libc::PR_SET_SECCOMP, libc::SECCOMP_MODE_FILTER, &prog as *const libc::sock_fprog) } != 0 {
            return Err(format!("seccomp: {}", errno()));
        }
        Ok(())
    }
}
