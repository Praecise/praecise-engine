//! Kernel classes, deterministic mode and device resolution.
//!
//! A kernel class names the backend, the device architecture and the version of
//! the deterministic kernel set. Within one class a step is bitwise
//! reproducible: identical inputs give an identical resulting state root.
//! Across classes, results are compared in tolerance mode instead.
//!
//! Deterministic mode refuses a run whose graph contains an op that has no
//! deterministic variant on the chosen backend, and device resolution refuses
//! an accelerator that is not present. Neither falls back silently.

use std::fmt;

use crate::Error;

/// Version of the deterministic kernel set. Bumped whenever a kernel used in
/// deterministic mode changes its reduction order or rounding.
pub const KERNEL_SET_VERSION: u32 = 1;

/// Compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Host CPU.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// Vulkan.
    Vulkan,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        })
    }
}

/// `(backend, device architecture, kernel set version)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelClass {
    /// Backend.
    pub backend: Backend,
    /// Device architecture, for example `x86_64-avx512`, `sm_121` or `apple-m4`.
    pub device_arch: String,
    /// Deterministic kernel set version.
    pub kernel_set: u32,
}

impl KernelClass {
    /// A class on the current kernel set.
    #[must_use]
    pub fn new(backend: Backend, device_arch: impl Into<String>) -> Self {
        Self {
            backend,
            device_arch: device_arch.into(),
            kernel_set: KERNEL_SET_VERSION,
        }
    }

    /// Stable identifier, `backend/arch/ksN`.
    #[must_use]
    pub fn id(&self) -> String {
        format!(
            "{}/{}/ks{}",
            self.backend, self.device_arch, self.kernel_set
        )
    }

    /// The class of the host CPU this binary runs on.
    #[must_use]
    pub fn host_cpu() -> Self {
        Self::new(Backend::Cpu, host_cpu_arch())
    }
}

/// Host CPU architecture with the widest vector extension the build targets.
#[must_use]
pub fn host_cpu_arch() -> String {
    let arch = std::env::consts::ARCH;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return format!("{arch}-avx512");
        }
        if std::is_x86_feature_detected!("avx2") {
            return format!("{arch}-avx2");
        }
    }
    arch.to_string()
}

/// A device the backend registry reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Backend of the device.
    pub backend: Backend,
    /// Registry name, for example `CUDA0`.
    pub name: String,
    /// Architecture string used in the kernel class.
    pub arch: String,
}

/// What a run asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    /// The host CPU.
    Cpu,
    /// The `n`-th CUDA device.
    Cuda(usize),
    /// The Metal device.
    Metal,
    /// The `n`-th Vulkan device.
    Vulkan(usize),
}

/// Resolves a request against the devices present.
///
/// # Errors
/// [`Error::Refused`] when the requested accelerator is absent; a run never
/// moves to the CPU on its own.
pub fn resolve_device(
    request: DeviceRequest,
    available: &[DeviceInfo],
) -> Result<DeviceInfo, Error> {
    let pick = |backend: Backend, n: usize| {
        available
            .iter()
            .filter(|d| d.backend == backend)
            .nth(n)
            .cloned()
    };
    let found = match request {
        DeviceRequest::Cpu => Some(DeviceInfo {
            backend: Backend::Cpu,
            name: "CPU".into(),
            arch: host_cpu_arch(),
        }),
        DeviceRequest::Cuda(n) => pick(Backend::Cuda, n),
        DeviceRequest::Metal => pick(Backend::Metal, 0),
        DeviceRequest::Vulkan(n) => pick(Backend::Vulkan, n),
    };
    found.ok_or_else(|| {
        let present: Vec<&str> = available.iter().map(|d| d.name.as_str()).collect();
        Error::Refused(format!(
            "{request:?} requested but this host has no such device (present: {present:?})"
        ))
    })
}

/// Ops that a training graph may contain in deterministic mode on `backend`.
///
/// CPU kernels split work by output rows, so every output element is reduced
/// by one thread in a fixed order and the thread count does not change the
/// result. `CROSS_ENTROPY_LOSS` sums per-thread partials, but only its value
/// depends on that order and the value never reaches the state: its backward
/// is per row. The step reports the loss from its own fixed-order reduction.
/// `ARGMAX` is per row with ties to the lowest index, and `COUNT_EQUAL` adds
/// integers, which is exact in any order.
///
/// GPU kernel sets are not yet audited; until they are, deterministic mode
/// refuses every op on them rather than claiming reproducibility it has not
/// measured.
#[must_use]
pub fn deterministic_ops(backend: Backend) -> &'static [&'static str] {
    match backend {
        Backend::Cpu => CPU_DETERMINISTIC,
        Backend::Cuda | Backend::Metal | Backend::Vulkan => &[],
    }
}

const CPU_DETERMINISTIC: &[&str] = &[
    "NONE",
    "DUP",
    "ADD",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "ARGMAX",
    "COUNT_EQUAL",
    "MEAN",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",
    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",
    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "SET_ROWS",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_2D",
    "IM2COL",
    "IM2COL_BACK",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "PAD",
    "ARANGE",
    "FLASH_ATTN_EXT",
    "SSM_CONV",
    "SSM_SCAN",
    "SSM_SCAN_BACK",
    "UNARY",
    "GLU",
    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",
    "SILU_BACK",
];

/// Checks every op of a graph against the deterministic set of `backend`.
///
/// # Errors
/// [`Error::Refused`] naming the first op without a deterministic variant.
pub fn check_deterministic<'a>(
    backend: Backend,
    ops: impl IntoIterator<Item = &'a str>,
) -> Result<(), Error> {
    let allowed = deterministic_ops(backend);
    for op in ops {
        if !allowed.contains(&op) {
            return Err(Error::Refused(format!(
                "deterministic mode: op {op} has no deterministic variant on {backend} (kernel set {KERNEL_SET_VERSION})"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_stable() {
        let k = KernelClass::new(Backend::Cuda, "sm_121");
        assert_eq!(k.id(), format!("cuda/sm_121/ks{KERNEL_SET_VERSION}"));
        assert!(KernelClass::host_cpu().id().starts_with("cpu/"));
    }

    #[test]
    fn missing_accelerator_is_refused() {
        let cpu_only: Vec<DeviceInfo> = vec![];
        let err = resolve_device(DeviceRequest::Cuda(0), &cpu_only).unwrap_err();
        assert!(matches!(err, Error::Refused(_)), "{err}");
        assert_eq!(
            resolve_device(DeviceRequest::Cpu, &cpu_only)
                .unwrap()
                .backend,
            Backend::Cpu
        );
        let one = vec![DeviceInfo {
            backend: Backend::Cuda,
            name: "CUDA0".into(),
            arch: "sm_121".into(),
        }];
        assert_eq!(
            resolve_device(DeviceRequest::Cuda(0), &one).unwrap().name,
            "CUDA0"
        );
        assert!(resolve_device(DeviceRequest::Cuda(1), &one).is_err());
    }

    #[test]
    fn unaudited_ops_and_backends_are_refused() {
        check_deterministic(Backend::Cpu, ["MUL_MAT", "OUT_PROD", "FLASH_ATTN_EXT"]).unwrap();
        assert!(check_deterministic(Backend::Cpu, ["MUL_MAT", "RWKV_WKV7"]).is_err());
        assert!(check_deterministic(Backend::Cuda, ["MUL_MAT"]).is_err());
    }
}
