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
#[non_exhaustive]
pub enum Backend {
    /// Host CPU.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// Vulkan.
    Vulkan,
    /// AMD ROCm through HIP.
    Hip,
    /// Intel oneAPI SYCL.
    Sycl,
    /// Intel OpenVINO.
    OpenVino,
    /// OpenCL (Qualcomm Adreno).
    OpenCl,
    /// Qualcomm Hexagon NPU.
    Hexagon,
    /// Huawei Ascend through CANN.
    Cann,
    /// Moore Threads MUSA.
    Musa,
    /// WebGPU.
    WebGpu,
    /// IBM Z on-chip accelerator through zDNN.
    Zdnn,
    /// Esperanto ET-SoC.
    Et,
}

impl Backend {
    /// Every backend a device can belong to.
    pub const ALL: &'static [Backend] = &[
        Self::Cpu,
        Self::Cuda,
        Self::Metal,
        Self::Vulkan,
        Self::Hip,
        Self::Sycl,
        Self::OpenVino,
        Self::OpenCl,
        Self::Hexagon,
        Self::Cann,
        Self::Musa,
        Self::WebGpu,
        Self::Zdnn,
        Self::Et,
    ];

    /// The prefix of the device names the backend registers with the
    /// scheduler (`CUDA0`, `ROCm0`, `MTL0`, `Vulkan0`, `SYCL0`, `CANN0`, …).
    #[must_use]
    pub const fn device_prefix(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::Metal => "MTL",
            Self::Vulkan => "Vulkan",
            Self::Hip => "ROCm",
            Self::Sycl => "SYCL",
            Self::OpenVino => "OPENVINO",
            Self::OpenCl => "GPUOpenCL",
            Self::Hexagon => "HTP",
            Self::Cann => "CANN",
            Self::Musa => "MUSA",
            Self::WebGpu => "WebGPU",
            Self::Zdnn => "zDNN",
            Self::Et => "ET",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Hip => "hip",
            Self::Sycl => "sycl",
            Self::OpenVino => "openvino",
            Self::OpenCl => "opencl",
            Self::Hexagon => "hexagon",
            Self::Cann => "cann",
            Self::Musa => "musa",
            Self::WebGpu => "webgpu",
            Self::Zdnn => "zdnn",
            Self::Et => "et",
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
    /// The `n`-th device of any other backend.
    Device(Backend, usize),
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
        DeviceRequest::Device(Backend::Cpu, _) => return resolve_device(DeviceRequest::Cpu, available),
        DeviceRequest::Device(backend, n) => pick(backend, n),
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
/// CUDA kernels reduce inside a block in a fixed order and write each output
/// element once; none of the listed ops accumulates floats with atomics (only
/// the multi-device all-reduce and `COUNT_EQUAL`, which adds integers, use
/// them). Matrix products go through cuBLAS, which is bitwise reproducible run
/// to run on one architecture with one SM count, or through the quantized
/// kernels, whose stream-k fixup adds the partial tiles of preceding blocks in
/// block order; both choose their split from the SM count, so the CUDA device
/// architecture of a kernel class names it. Flash attention is not in the set:
/// a training context turns it off.
///
/// No other backend's kernel set is audited. That includes HIP and MUSA,
/// which compile the CUDA kernel sources: their matrix products go through
/// hipBLAS and muBLAS, and their split follows a different device, so the
/// CUDA audit does not carry over. Until a backend is audited on its own
/// hardware, deterministic mode refuses every op on it rather than claiming
/// reproducibility it has not measured. A run there is still possible in
/// tolerance mode, and is recognised and kept on its device.
#[must_use]
pub fn deterministic_ops(backend: Backend) -> &'static [&'static str] {
    match backend {
        Backend::Cpu => CPU_DETERMINISTIC,
        Backend::Cuda => CUDA_DETERMINISTIC,
        _ => &[],
    }
}

const CUDA_DETERMINISTIC: &[&str] = &[
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
    "DIAG_MASK_INF",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "UNARY",
    "GLU",
    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",
    "SILU_BACK",
];

/// Ops that compute nothing: they describe a tensor over another one's memory.
const NO_COMPUTE: &[&str] = &["NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"];

/// The backend a scheduler device name (`CPU`, `CUDA0`, `ROCm0`, `MTL0`,
/// `Vulkan0`, `SYCL0`, `CANN0`, `HTP0`, …) belongs to. `Metal` is accepted
/// beside `MTL`, the name older builds registered. A name of no backend
/// here (an RPC endpoint, BLAS) is none.
#[must_use]
pub fn backend_of_device(name: &str) -> Option<Backend> {
    let lower = name.to_ascii_lowercase();
    if lower == "cpu" {
        return Some(Backend::Cpu);
    }
    if lower.starts_with("metal") {
        return Some(Backend::Metal);
    }
    // the longest prefix wins, so no name is claimed by a shorter one
    Backend::ALL
        .iter()
        .copied()
        .filter(|b| *b != Backend::Cpu && lower.starts_with(&b.device_prefix().to_ascii_lowercase()))
        .max_by_key(|b| b.device_prefix().len())
}

/// Checks where every op of a graph ran, as `(op, device)` pairs, against a run
/// on `backend`: every op runs on that backend, except the input embedding
/// lookup (`GET_ROWS`), which the engine keeps beside the token table on the
/// host, and ops that compute nothing. Each op must be in the deterministic
/// set of the backend it ran on.
///
/// # Errors
/// [`Error::Refused`] naming the first op that ran elsewhere or has no
/// deterministic variant where it ran.
pub fn check_placements<'a>(
    backend: Backend,
    placements: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<(), Error> {
    for (op, device) in placements {
        if NO_COMPUTE.contains(&op) {
            continue;
        }
        let ran_on = backend_of_device(device)
            .ok_or_else(|| Error::Refused(format!("op {op} ran on an unknown device {device}")))?;
        if ran_on != backend && !(ran_on == Backend::Cpu && op == "GET_ROWS") {
            return Err(Error::Refused(format!(
                "op {op} ran on {device}, not on {backend}: a run does not leave the device it asked for"
            )));
        }
        check_deterministic(ran_on, [op])?;
    }
    Ok(())
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
        check_deterministic(Backend::Cuda, ["MUL_MAT", "OUT_PROD", "ROPE_BACK"]).unwrap();
        assert!(check_deterministic(Backend::Cuda, ["FLASH_ATTN_EXT"]).is_err());
        assert!(check_deterministic(Backend::Metal, ["MUL_MAT"]).is_err());
    }

    #[test]
    fn every_backend_is_recognised_by_its_device_name() {
        for (name, want) in [
            ("CPU", Backend::Cpu),
            ("CUDA0", Backend::Cuda),
            ("ROCm0", Backend::Hip),
            ("MTL0", Backend::Metal),
            ("Metal", Backend::Metal),
            ("Vulkan1", Backend::Vulkan),
            ("SYCL0", Backend::Sycl),
            ("OPENVINO", Backend::OpenVino),
            ("GPUOpenCL", Backend::OpenCl),
            ("HTP0:1", Backend::Hexagon),
            ("CANN0", Backend::Cann),
            ("MUSA0", Backend::Musa),
            ("WebGPU", Backend::WebGpu),
            ("zDNN", Backend::Zdnn),
        ] {
            assert_eq!(backend_of_device(name), Some(want), "{name}");
        }
        assert_eq!(backend_of_device("RPC[10.0.0.2:50052]"), None);
        assert_eq!(backend_of_device("BLAS"), None);
        for b in Backend::ALL.iter().filter(|b| **b != Backend::Cpu) {
            assert_eq!(backend_of_device(&format!("{}0", b.device_prefix())), Some(*b));
        }
    }

    #[test]
    fn other_vendors_resolve_and_stay_on_their_device_but_are_not_called_deterministic() {
        let amd = vec![DeviceInfo {
            backend: Backend::Hip,
            name: "ROCm0".into(),
            arch: "gfx950".into(),
        }];
        assert_eq!(
            resolve_device(DeviceRequest::Device(Backend::Hip, 0), &amd).unwrap().name,
            "ROCm0"
        );
        assert!(resolve_device(DeviceRequest::Device(Backend::Hip, 1), &amd).is_err());
        assert!(resolve_device(DeviceRequest::Device(Backend::Cann, 0), &amd).is_err());
        assert!(resolve_device(DeviceRequest::Cuda(0), &amd).is_err());
        assert_eq!(KernelClass::new(Backend::Hip, "gfx950").id(), format!("hip/gfx950/ks{KERNEL_SET_VERSION}"));
        // the CUDA audit does not carry over to the HIP build of the same kernels
        for b in Backend::ALL {
            if !matches!(b, Backend::Cpu | Backend::Cuda) {
                assert!(check_deterministic(*b, ["MUL_MAT"]).is_err(), "{b}");
                assert!(deterministic_ops(*b).is_empty(), "{b}");
            }
        }
        // a tolerance-mode run on ROCm that strays to the CPU is still caught
        let err = check_placements(Backend::Hip, [("VIEW", "ROCm0"), ("MUL_MAT", "CPU")]).unwrap_err();
        assert!(err.to_string().contains("MUL_MAT ran on CPU"), "{err}");
    }

    #[test]
    fn a_run_stays_on_its_device() {
        check_placements(
            Backend::Cuda,
            [("GET_ROWS", "CPU"), ("VIEW", "CPU"), ("MUL_MAT", "CUDA0"), ("OUT_PROD", "CUDA0")],
        )
        .unwrap();
        let err = check_placements(Backend::Cuda, [("MUL_MAT", "CUDA0"), ("OUT_PROD", "CPU")]).unwrap_err();
        assert!(err.to_string().contains("OUT_PROD ran on CPU"), "{err}");
        assert!(check_placements(Backend::Cuda, [("FLASH_ATTN_EXT", "CUDA0")]).is_err());
        assert!(check_placements(Backend::Cuda, [("MUL_MAT", "RPC[x]")]).is_err());
        check_placements(Backend::Cpu, [("GET_ROWS", "CPU"), ("MUL_MAT", "CPU")]).unwrap();
        assert!(check_placements(Backend::Cpu, [("MUL_MAT", "CUDA0")]).is_err());
    }
}
