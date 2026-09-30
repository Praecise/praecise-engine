//! Standby weights: resident in host RAM, costing no engine or device memory.
//!
//! A cold model load is dominated by reading the weights from storage. A
//! [`StagedWeights`] maps the weights file read-only and faults every page into
//! host memory ahead of time, locking the pages where the process is permitted
//! to, so the kernel cannot evict them. Nothing is allocated in the backend:
//! no context, no KV cache, no device buffers.
//!
//! [`StagedWeights::activate`] then loads the model through the backend as
//! usual. The backend maps the same file, so its reads are served from the
//! resident pages and activation costs only graph and buffer setup (plus the
//! host-to-device copy when layers are offloaded) rather than storage I/O.
//!
//! Whether the pages are locked is reported by [`StagedWeights::pinned`], never
//! assumed: without the lock privilege (`RLIMIT_MEMLOCK`) the pages are resident
//! but evictable under memory pressure, and [`StagedWeights::resident_fraction`]
//! measures what is actually still in memory.
//!
//! Unix only: staging relies on `mmap`, `mlock` and `mincore`.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Identity of the file as staged, used to refuse activating a file that was
/// replaced on disk after staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileId {
    dev: u64,
    ino: u64,
    len: u64,
    mtime_ns: i128,
}

fn file_id(meta: &std::fs::Metadata) -> FileId {
    use std::os::unix::fs::MetadataExt;
    FileId {
        dev: meta.dev(),
        ino: meta.ino(),
        len: meta.len(),
        mtime_ns: i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec()),
    }
}

/// A weights file held resident in host memory, ready to activate.
pub struct StagedWeights {
    path: PathBuf,
    id: FileId,
    ptr: *mut libc::c_void,
    len: usize,
    pinned: bool,
}

// The mapping is read-only and owned; sharing the address across threads is sound.
unsafe impl Send for StagedWeights {}
unsafe impl Sync for StagedWeights {}

impl std::fmt::Debug for StagedWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedWeights")
            .field("path", &self.path)
            .field("len", &self.len)
            .field("pinned", &self.pinned)
            .finish_non_exhaustive()
    }
}

fn page_size() -> usize {
    usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096).max(1)
}

impl StagedWeights {
    /// Map `path` and bring every page into host memory, locking the pages if
    /// the process may. Blocks until the file is resident.
    ///
    /// # Errors
    /// I/O errors opening the file; [`Error::Staging`] if it cannot be mapped.
    pub fn stage(path: impl AsRef<Path>) -> Result<Self> {
        use std::os::fd::AsRawFd;
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let meta = file.metadata()?;
        let id = file_id(&meta);
        let len = usize::try_from(meta.len()).map_err(|_| Error::Staging("file too large to map".into()))?;
        if len == 0 {
            return Err(Error::Staging(format!("{} is empty", path.display())));
        }
        #[cfg(target_os = "linux")]
        let flags = libc::MAP_PRIVATE | libc::MAP_POPULATE;
        #[cfg(not(target_os = "linux"))]
        let flags = libc::MAP_PRIVATE;
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, flags, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Staging(format!("mmap failed: {}", std::io::Error::last_os_error())));
        }
        unsafe {
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }
        // Lock first (it faults pages in as it goes); if that is not permitted,
        // touch one byte per page so the file is resident either way.
        let pinned = unsafe { libc::mlock(ptr, len) } == 0;
        if !pinned {
            let step = page_size();
            let base = ptr.cast::<u8>();
            let mut acc = 0u8;
            let mut off = 0;
            while off < len {
                acc = acc.wrapping_add(unsafe { std::ptr::read_volatile(base.add(off)) });
                off += step;
            }
            std::hint::black_box(acc);
        }
        Ok(Self { path, id, ptr, len, pinned })
    }

    /// The staged file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Size of the staged weights in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Always false: an empty file is refused at staging.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Whether the pages are locked in memory. False means resident but
    /// evictable under memory pressure.
    #[must_use]
    pub const fn pinned(&self) -> bool {
        self.pinned
    }

    /// The staged bytes (for example to fingerprint them without another read).
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }

    /// Fraction of the staged pages resident in memory right now, `0.0..=1.0`.
    ///
    /// # Errors
    /// [`Error::Staging`] if residency cannot be queried.
    pub fn resident_fraction(&self) -> Result<f64> {
        resident_fraction_of(self.ptr, self.len)
    }

    /// Refuse if the file on disk is no longer the one that was staged.
    ///
    /// # Errors
    /// [`Error::Staging`] if it was replaced or modified; I/O errors reading
    /// its metadata.
    pub fn check_unchanged(&self) -> Result<()> {
        let now = file_id(&std::fs::metadata(&self.path)?);
        if now != self.id {
            return Err(Error::Staging(format!("{} changed on disk since it was staged", self.path.display())));
        }
        Ok(())
    }

    /// Load the staged weights into the backend and return a model ready to
    /// serve. The staged pages stay resident until `self` is dropped.
    ///
    /// # Errors
    /// [`Error::Staging`] if the file changed since staging or the backend
    /// refuses to load it.
    #[cfg(feature = "bundled-llama")]
    pub fn activate(
        &self,
        backend: &llama_cpp_2::llama_backend::LlamaBackend,
        params: &llama_cpp_2::model::params::LlamaModelParams,
    ) -> Result<llama_cpp_2::model::LlamaModel> {
        self.check_unchanged()?;
        llama_cpp_2::model::LlamaModel::load_from_file(backend, &self.path, params)
            .map_err(|e| Error::Staging(format!("backend load failed: {e}")))
    }
}

impl Drop for StagedWeights {
    fn drop(&mut self) {
        unsafe {
            if self.pinned {
                libc::munlock(self.ptr, self.len);
            }
            libc::munmap(self.ptr, self.len);
        }
    }
}

fn resident_fraction_of(ptr: *mut libc::c_void, len: usize) -> Result<f64> {
    let page = page_size();
    let pages = len.div_ceil(page);
    #[cfg(target_os = "linux")]
    let mut vec = vec![0u8; pages];
    #[cfg(not(target_os = "linux"))]
    let mut vec = vec![0 as libc::c_char; pages];
    if unsafe { libc::mincore(ptr, len, vec.as_mut_ptr()) } != 0 {
        return Err(Error::Staging(format!("mincore failed: {}", std::io::Error::last_os_error())));
    }
    let resident = vec.iter().filter(|&&b| b & 1 != 0).count();
    Ok(resident as f64 / pages as f64)
}

/// Fraction of a file's pages in the page cache, without faulting any in.
///
/// # Errors
/// I/O errors opening the file; [`Error::Staging`] if it cannot be mapped or
/// queried.
pub fn page_cache_fraction(path: impl AsRef<Path>) -> Result<f64> {
    use std::os::fd::AsRawFd;
    let file = std::fs::File::open(path)?;
    let len = usize::try_from(file.metadata()?.len()).map_err(|_| Error::Staging("file too large".into()))?;
    if len == 0 {
        return Ok(1.0);
    }
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0) };
    if ptr == libc::MAP_FAILED {
        return Err(Error::Staging(format!("mmap failed: {}", std::io::Error::last_os_error())));
    }
    let r = resident_fraction_of(ptr, len);
    unsafe {
        libc::munmap(ptr, len);
    }
    r
}

/// Ask the kernel to drop a file's clean pages from the page cache, demoting
/// it to cold. Pages another process has mapped or locked stay resident; check
/// with [`page_cache_fraction`].
///
/// # Errors
/// I/O errors opening the file or from the advice call.
#[cfg(target_os = "linux")]
pub fn evict_from_page_cache(path: impl AsRef<Path>) -> Result<()> {
    use std::os::fd::AsRawFd;
    let file = std::fs::File::open(path)?;
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        return Err(Error::Io(std::io::Error::from_raw_os_error(rc)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_a_file_resident() {
        let dir = std::env::temp_dir().join(format!("praecise-stage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.bin");
        let data: Vec<u8> = (0..(3 * 1024 * 1024u32)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let staged = StagedWeights::stage(&path).unwrap();
        assert_eq!(staged.len(), data.len());
        assert_eq!(staged.bytes(), &data[..]);
        assert!(staged.resident_fraction().unwrap() > 0.99);
        staged.check_unchanged().unwrap();
        drop(staged);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
