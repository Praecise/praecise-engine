//! Which software drives which accelerator.
//!
//! Three routes reach an accelerator from this engine: a ggml backend linked
//! into the process, a torch device inside a hosted runtime, and a hosted
//! runtime's own platform. Each is a table here, so a build, an environment
//! and a refusal message all name the same thing.

use crate::detect::{Accelerator, Kind, Machine};
use crate::vendor::Vendor;

/// A ggml compute backend of the linked llama.cpp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GgmlBackend {
    /// NVIDIA CUDA.
    Cuda,
    /// AMD `ROCm` through HIP.
    Hip,
    /// Apple Metal.
    Metal,
    /// Vulkan, on any vendor's GPU with a Vulkan driver.
    Vulkan,
    /// Intel oneAPI SYCL.
    Sycl,
    /// Intel `OpenVINO`: CPU, GPU and NPU.
    OpenVino,
    /// `OpenCL`, tuned for Qualcomm Adreno.
    OpenCl,
    /// Qualcomm Hexagon NPU.
    Hexagon,
    /// Huawei Ascend through CANN.
    Cann,
    /// Moore Threads MUSA.
    Musa,
    /// WebGPU through Dawn.
    WebGpu,
    /// IBM Z on-chip accelerator through zDNN.
    Zdnn,
    /// AMD `ZenDNN`, on Zen CPUs.
    ZenDnn,
    /// Esperanto ET-SoC.
    Et,
}

impl GgmlBackend {
    /// Every backend.
    pub const ALL: &'static [GgmlBackend] = &[
        GgmlBackend::Cuda,
        GgmlBackend::Hip,
        GgmlBackend::Metal,
        GgmlBackend::Vulkan,
        GgmlBackend::Sycl,
        GgmlBackend::OpenVino,
        GgmlBackend::OpenCl,
        GgmlBackend::Hexagon,
        GgmlBackend::Cann,
        GgmlBackend::Musa,
        GgmlBackend::WebGpu,
        GgmlBackend::Zdnn,
        GgmlBackend::ZenDnn,
        GgmlBackend::Et,
    ];

    /// The cargo feature that builds it, the same name on `llama-cpp-sys-2`,
    /// `llama-cpp-2`, `praecise-runtime` and `praecise-diffusion`.
    #[must_use]
    pub const fn feature(self) -> &'static str {
        match self {
            GgmlBackend::Cuda => "cuda",
            GgmlBackend::Hip => "rocm",
            GgmlBackend::Metal => "metal",
            GgmlBackend::Vulkan => "vulkan",
            GgmlBackend::Sycl => "sycl",
            GgmlBackend::OpenVino => "openvino",
            GgmlBackend::OpenCl => "opencl",
            GgmlBackend::Hexagon => "hexagon",
            GgmlBackend::Cann => "cann",
            GgmlBackend::Musa => "musa",
            GgmlBackend::WebGpu => "webgpu",
            GgmlBackend::Zdnn => "zdnn",
            GgmlBackend::ZenDnn => "zendnn",
            GgmlBackend::Et => "et",
        }
    }

    /// The `CMake` option the feature sets.
    #[must_use]
    pub const fn cmake_option(self) -> &'static str {
        match self {
            GgmlBackend::Cuda => "GGML_CUDA",
            GgmlBackend::Hip => "GGML_HIP",
            GgmlBackend::Metal => "GGML_METAL",
            GgmlBackend::Vulkan => "GGML_VULKAN",
            GgmlBackend::Sycl => "GGML_SYCL",
            GgmlBackend::OpenVino => "GGML_OPENVINO",
            GgmlBackend::OpenCl => "GGML_OPENCL",
            GgmlBackend::Hexagon => "GGML_HEXAGON",
            GgmlBackend::Cann => "GGML_CANN",
            GgmlBackend::Musa => "GGML_MUSA",
            GgmlBackend::WebGpu => "GGML_WEBGPU",
            GgmlBackend::Zdnn => "GGML_ZDNN",
            GgmlBackend::ZenDnn => "GGML_ZENDNN",
            GgmlBackend::Et => "GGML_ET",
        }
    }

    /// The prefix of the device names the backend registers (`CUDA0`,
    /// `ROCm0`, `MTL0`, `Vulkan0`, `SYCL0`, `CANN0`, `HTP0`, …).
    #[must_use]
    pub const fn device_prefix(self) -> &'static str {
        match self {
            GgmlBackend::Cuda => "CUDA",
            GgmlBackend::Hip => "ROCm",
            GgmlBackend::Metal => "MTL",
            GgmlBackend::Vulkan => "Vulkan",
            GgmlBackend::Sycl => "SYCL",
            GgmlBackend::OpenVino => "OPENVINO",
            GgmlBackend::OpenCl => "GPUOpenCL",
            GgmlBackend::Hexagon => "HTP",
            GgmlBackend::Cann => "CANN",
            GgmlBackend::Musa => "MUSA",
            GgmlBackend::WebGpu => "WebGPU",
            GgmlBackend::Zdnn => "zDNN",
            GgmlBackend::ZenDnn => "ZenDNN",
            GgmlBackend::Et => "ET",
        }
    }

    /// The backend a registered device name belongs to. `Metal` is accepted
    /// beside `MTL`, the name older builds registered.
    #[must_use]
    pub fn of_device(name: &str) -> Option<Self> {
        if name.starts_with("Metal") {
            return Some(GgmlBackend::Metal);
        }
        // longest prefix first, so `ZenDNN` is not read as `zDNN` or `ET`
        let mut all: Vec<GgmlBackend> = GgmlBackend::ALL.to_vec();
        all.sort_by_key(|b| std::cmp::Reverse(b.device_prefix().len()));
        all.into_iter()
            .find(|b| name.starts_with(b.device_prefix()))
    }
}

/// The ggml backends that drive `accelerator`, best first. Empty when no
/// linked backend reaches it and only a hosted runtime does.
#[must_use]
pub fn ggml_backends(accelerator: Accelerator) -> &'static [GgmlBackend] {
    use GgmlBackend as B;
    match (accelerator.vendor, accelerator.kind) {
        (Vendor::Nvidia, Kind::Gpu) => &[B::Cuda, B::Vulkan],
        (Vendor::Amd, Kind::Gpu) => &[B::Hip, B::Vulkan],
        (Vendor::Intel, Kind::Gpu) => &[B::Sycl, B::OpenVino, B::Vulkan],
        (Vendor::Intel, Kind::Npu) => &[B::OpenVino],
        (Vendor::Apple, Kind::Gpu) => &[B::Metal],
        (Vendor::Qualcomm, Kind::Gpu) => &[B::OpenCl, B::Vulkan],
        (Vendor::Qualcomm, Kind::Npu) => &[B::Hexagon],
        (Vendor::Huawei, Kind::Npu) => &[B::Cann],
        (Vendor::MooreThreads, Kind::Gpu) => &[B::Musa, B::Vulkan],
        (Vendor::Ibm, Kind::Accelerator) => &[B::Zdnn],
        (Vendor::Esperanto, Kind::Accelerator) => &[B::Et],
        // any other GPU is reached through its Vulkan driver
        (_, Kind::Gpu) => &[B::Vulkan],
        _ => &[],
    }
}

/// The cargo features that would give this build a backend for `machine`:
/// the best backend of each accelerator a linked backend can reach.
#[must_use]
pub fn features_for(machine: &Machine) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = machine
        .accelerators()
        .iter()
        .filter_map(|a| ggml_backends(*a).first())
        .map(|b| b.feature())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The torch device type a hosted runtime addresses `accelerator` by, when
/// torch (or the vendor's torch plugin) has one.
///
/// AMD `ROCm` presents as `cuda`. Ascend and Moore Threads register theirs
/// through a vendor package (`torch_npu`, `torch_musa`).
#[must_use]
pub fn torch_device(accelerator: Accelerator) -> Option<&'static str> {
    match (accelerator.vendor, accelerator.kind) {
        (Vendor::Nvidia | Vendor::Amd, Kind::Gpu) => Some("cuda"),
        (Vendor::Intel, Kind::Gpu) => Some("xpu"),
        (Vendor::Intel, Kind::Accelerator) => Some("hpu"),
        (Vendor::Apple, Kind::Gpu) => Some("mps"),
        (Vendor::Huawei, Kind::Npu) => Some("npu"),
        (Vendor::MooreThreads, Kind::Gpu) => Some("musa"),
        _ => None,
    }
}

/// How vLLM reaches an accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VllmPlatform {
    /// The package that carries the platform: `vllm` itself for a built-in
    /// one, otherwise the vendor's plugin or fork.
    pub package: &'static str,
    /// Whether the platform is built into upstream vLLM.
    pub builtin: bool,
}

/// The vLLM platform for `accelerator`, when one is published.
///
/// Vendor plugins pin the upstream vLLM they were built against, and the
/// pins differ, so each one needs an engine environment of its own.
#[must_use]
pub fn vllm_platform(accelerator: Accelerator) -> Option<VllmPlatform> {
    let builtin = |package| {
        Some(VllmPlatform {
            package,
            builtin: true,
        })
    };
    let plugin = |package| {
        Some(VllmPlatform {
            package,
            builtin: false,
        })
    };
    match (accelerator.vendor, accelerator.kind) {
        (Vendor::Nvidia | Vendor::Amd | Vendor::Intel, Kind::Gpu) => builtin("vllm"),
        (Vendor::Google, _) => plugin("tpu-inference"),
        (Vendor::Intel, Kind::Accelerator) => plugin("vllm-gaudi"),
        (Vendor::Apple, Kind::Gpu) => plugin("vllm-metal"),
        (Vendor::Huawei, Kind::Npu) => plugin("vllm-ascend"),
        (Vendor::Aws, _) => plugin("vllm-neuron"),
        (Vendor::Ibm, _) => plugin("vllm-spyre"),
        (Vendor::MooreThreads, Kind::Gpu) => plugin("vllm-musa"),
        (Vendor::Cambricon, _) => plugin("vllm-mlu"),
        (Vendor::MetaX, _) => plugin("vllm-metax"),
        (Vendor::Iluvatar, _) => plugin("vllm-iluvatar"),
        (Vendor::Enflame, _) => plugin("vllm-gcu"),
        (Vendor::Furiosa, _) => plugin("furiosa-llm"),
        (Vendor::Tenstorrent, _) => plugin("tt-inference-server"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc(vendor: Vendor, kind: Kind) -> Accelerator {
        Accelerator { vendor, kind }
    }

    #[test]
    fn every_backend_has_a_distinct_feature_option_and_prefix() {
        for field in [
            GgmlBackend::feature,
            GgmlBackend::cmake_option,
            GgmlBackend::device_prefix,
        ] {
            let mut seen: Vec<&str> = GgmlBackend::ALL.iter().map(|b| field(*b)).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), GgmlBackend::ALL.len());
        }
    }

    #[test]
    fn device_names_resolve_to_their_backend() {
        for (name, want) in [
            ("CUDA0", GgmlBackend::Cuda),
            ("ROCm1", GgmlBackend::Hip),
            ("MTL0", GgmlBackend::Metal),
            ("Metal", GgmlBackend::Metal),
            ("Vulkan0", GgmlBackend::Vulkan),
            ("SYCL0", GgmlBackend::Sycl),
            ("CANN0", GgmlBackend::Cann),
            ("HTP0:1", GgmlBackend::Hexagon),
            ("GPUOpenCL", GgmlBackend::OpenCl),
            ("zDNN", GgmlBackend::Zdnn),
            ("ZenDNN", GgmlBackend::ZenDnn),
            ("MUSA0", GgmlBackend::Musa),
        ] {
            assert_eq!(GgmlBackend::of_device(name), Some(want), "{name}");
        }
        assert_eq!(GgmlBackend::of_device("CPU"), None);
        assert_eq!(GgmlBackend::of_device("RPC[10.0.0.2:50052]"), None);
    }

    #[test]
    fn each_vendor_gpu_has_a_linked_route() {
        for v in Vendor::ALL {
            assert!(!ggml_backends(acc(*v, Kind::Gpu)).is_empty(), "{v}");
        }
        assert_eq!(
            ggml_backends(acc(Vendor::Amd, Kind::Gpu))[0],
            GgmlBackend::Hip
        );
        assert_eq!(
            ggml_backends(acc(Vendor::Intel, Kind::Npu)),
            &[GgmlBackend::OpenVino]
        );
        assert_eq!(
            ggml_backends(acc(Vendor::Huawei, Kind::Npu)),
            &[GgmlBackend::Cann]
        );
        // an AMD NPU and a Gaudi card are reached only by a hosted runtime
        assert_eq!(ggml_backends(acc(Vendor::Amd, Kind::Npu)), []);
        assert_eq!(ggml_backends(acc(Vendor::Intel, Kind::Accelerator)), []);
    }

    #[test]
    fn a_build_is_named_for_the_machine_it_runs_on() {
        let laptop = Machine::with([acc(Vendor::Intel, Kind::Gpu), acc(Vendor::Intel, Kind::Npu)]);
        assert_eq!(features_for(&laptop), ["openvino", "sycl"]);
        let mixed = Machine::with([acc(Vendor::Nvidia, Kind::Gpu), acc(Vendor::Amd, Kind::Gpu)]);
        assert_eq!(features_for(&mixed), ["cuda", "rocm"]);
        assert_eq!(features_for(&Machine::cpu_only()), [] as [&str; 0]);
    }

    #[test]
    fn hosted_routes_follow_the_vendor() {
        assert_eq!(torch_device(acc(Vendor::Amd, Kind::Gpu)), Some("cuda"));
        assert_eq!(torch_device(acc(Vendor::Intel, Kind::Gpu)), Some("xpu"));
        assert_eq!(torch_device(acc(Vendor::Huawei, Kind::Npu)), Some("npu"));
        assert_eq!(torch_device(acc(Vendor::Qualcomm, Kind::Npu)), None);
        assert!(vllm_platform(acc(Vendor::Amd, Kind::Gpu)).unwrap().builtin);
        let ascend = vllm_platform(acc(Vendor::Huawei, Kind::Npu)).unwrap();
        assert_eq!((ascend.package, ascend.builtin), ("vllm-ascend", false));
        assert_eq!(vllm_platform(acc(Vendor::Qualcomm, Kind::Gpu)), None);
    }
}
