//! Which execution providers a session registers.
//!
//! The decision is a pure function of three things: the accelerators of the
//! machine ([`Gpus`]), the features this binary was built with, and the
//! providers the linked ONNX Runtime lists ([`ProviderFacts`]). It touches
//! neither ONNX Runtime nor the machine, so every rule is tested on any
//! build host. [`crate::session`] gathers the facts and registers the plan.

use praecise_hardware::{Kind, Machine, Vendor};

/// The accelerators visible to this process, as the host detected them. The
/// execution providers follow from these; the engine does not guess them.
///
/// Build it from the machine: `Gpus::from(&Machine::detect())`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gpus {
    /// An NVIDIA GPU is visible.
    pub nvidia: bool,
    /// An AMD GPU is visible.
    pub amd: bool,
    /// An Intel GPU is visible.
    pub intel: bool,
    /// A Qualcomm Adreno GPU is visible.
    pub qualcomm: bool,
    /// A Huawei Ascend NPU is visible. Ascend is the machine's accelerator,
    /// not a companion to a GPU, so it is planned like one.
    pub huawei: bool,
    /// An AMD XDNA NPU is visible.
    pub amd_npu: bool,
    /// An Intel NPU is visible.
    pub intel_npu: bool,
    /// A Qualcomm Hexagon NPU is visible.
    pub qualcomm_npu: bool,
}

impl From<&Machine> for Gpus {
    fn from(machine: &Machine) -> Self {
        Self {
            nvidia: machine.has_kind(Vendor::Nvidia, Kind::Gpu),
            amd: machine.has_kind(Vendor::Amd, Kind::Gpu),
            intel: machine.has_kind(Vendor::Intel, Kind::Gpu),
            qualcomm: machine.has_kind(Vendor::Qualcomm, Kind::Gpu),
            huawei: machine.has_kind(Vendor::Huawei, Kind::Npu),
            amd_npu: machine.has_kind(Vendor::Amd, Kind::Npu),
            intel_npu: machine.has_kind(Vendor::Intel, Kind::Npu),
            qualcomm_npu: machine.has_kind(Vendor::Qualcomm, Kind::Npu),
        }
    }
}

/// A hardware execution provider an ONNX session registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accelerator {
    /// NVIDIA TensorRT.
    TensorRt,
    /// NVIDIA CUDA.
    Cuda,
    /// AMD MIGraphX.
    MiGraphX,
    /// Intel OpenVINO on the GPU.
    OpenVinoGpu,
    /// Intel OpenVINO on the NPU.
    OpenVinoNpu,
    /// Qualcomm QNN on the Adreno GPU.
    QnnGpu,
    /// Qualcomm QNN on the Hexagon NPU.
    QnnHtp,
    /// Huawei CANN.
    Cann,
    /// AMD Vitis AI, on the XDNA NPU.
    VitisAi,
    /// Apple CoreML.
    CoreMl,
    /// DirectML, on any DirectX 12 GPU on Windows.
    DirectMl,
    /// WebGPU, on any vendor's GPU.
    WebGpu,
}

impl Accelerator {
    /// The provider's name, as logs and errors give it.
    pub fn name(self) -> &'static str {
        match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
            Self::MiGraphX => "MIGraphX",
            Self::OpenVinoGpu => "OpenVINO (GPU)",
            Self::OpenVinoNpu => "OpenVINO (NPU)",
            Self::QnnGpu => "QNN (GPU)",
            Self::QnnHtp => "QNN (HTP)",
            Self::Cann => "CANN",
            Self::VitisAi => "Vitis AI",
            Self::CoreMl => "CoreML",
            Self::DirectMl => "DirectML",
            Self::WebGpu => "WebGPU",
        }
    }
}

/// What this process has to accelerate ONNX sessions with: the accelerators
/// of the machine, the features of the build, and the providers the linked
/// ONNX Runtime lists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderFacts {
    /// An NVIDIA GPU was detected on this machine.
    pub cuda_gpu: bool,
    /// This binary was built with the `cuda` feature.
    pub cuda_build: bool,
    /// The linked ONNX Runtime lists the CUDA execution provider.
    pub cuda_provider: bool,
    /// The linked ONNX Runtime lists the TensorRT execution provider.
    pub tensorrt_provider: bool,
    /// The TensorRT runtime library can be loaded.
    pub tensorrt_runtime: bool,
    /// This binary was built with the `vulkan` feature.
    pub vulkan_build: bool,
    /// An AMD GPU was detected on this machine.
    pub amd_gpu: bool,
    /// This binary was built with the `rocm` feature.
    pub rocm_build: bool,
    /// The linked ONNX Runtime lists the MIGraphX execution provider.
    pub migraphx_provider: bool,
    /// An Intel GPU was detected on this machine.
    pub intel_gpu: bool,
    /// An Intel NPU was detected on this machine.
    pub intel_npu: bool,
    /// This binary was built with the `openvino` feature.
    pub openvino_build: bool,
    /// The linked ONNX Runtime lists the OpenVINO execution provider.
    pub openvino_provider: bool,
    /// A Qualcomm GPU was detected on this machine.
    pub qualcomm_gpu: bool,
    /// A Qualcomm NPU was detected on this machine.
    pub qualcomm_npu: bool,
    /// This binary was built with the `qnn` feature.
    pub qnn_build: bool,
    /// The linked ONNX Runtime lists the QNN execution provider.
    pub qnn_provider: bool,
    /// A Huawei Ascend NPU was detected on this machine.
    pub huawei_npu: bool,
    /// This binary was built with the `cann` feature.
    pub cann_build: bool,
    /// The linked ONNX Runtime lists the CANN execution provider.
    pub cann_provider: bool,
    /// An AMD NPU was detected on this machine.
    pub amd_npu: bool,
    /// This binary was built with the `vitis` feature.
    pub vitis_build: bool,
    /// The linked ONNX Runtime lists the Vitis AI execution provider.
    pub vitis_provider: bool,
    /// This binary was built with the `directml` feature.
    pub directml_build: bool,
    /// The linked ONNX Runtime lists the DirectML execution provider.
    pub directml_provider: bool,
    /// This binary was built with the `webgpu` feature.
    pub webgpu_build: bool,
    /// The linked ONNX Runtime lists the WebGPU execution provider.
    pub webgpu_provider: bool,
    /// This binary was built with the `coreml` feature.
    pub coreml_build: bool,
    /// The linked ONNX Runtime lists the CoreML execution provider.
    pub coreml_provider: bool,
    /// The operator asked for the neural processor ([`OnnxConfig::npu`]).
    pub npu: bool,
}

/// The provider that runs on any vendor's GPU, when this build has one:
/// DirectML, then WebGPU.
fn portable(facts: &ProviderFacts) -> Option<Accelerator> {
    if facts.directml_build && facts.directml_provider {
        Some(Accelerator::DirectMl)
    } else if facts.webgpu_build && facts.webgpu_provider {
        Some(Accelerator::WebGpu)
    } else {
        None
    }
}

/// Why a GPU of `vendor` cannot be used, when neither its own provider
/// (`feature`, in the build called `build`) nor a portable one is built.
fn no_provider(facts: &ProviderFacts, vendor: &str, feature: &str, build: &str) -> String {
    if facts.vulkan_build {
        format!(
            "this is a Vulkan build: it runs GGUF models on this {vendor} GPU, but ONNX Runtime \
             has no Vulkan execution provider, so it does not serve ONNX models here; install \
             the {build} build to serve them, or one with the `webgpu` feature"
        )
    } else {
        format!(
            "this machine has {article} {vendor} GPU but this binary was built without the \
             `{feature}` feature, so its ONNX runtimes cannot use it; install the {build} build, \
             or one with the `directml` or `webgpu` feature",
            article = if vendor.starts_with(['A', 'I']) { "an" } else { "a" },
        )
    }
}

/// Why a vendor's own provider cannot be used although the build has its
/// feature: the linked ONNX Runtime does not carry it.
fn not_listed(feature: &str, vendor: &str, provider: &str) -> String {
    format!(
        "this binary was built with the `{feature}` feature and this machine has {vendor} \
         hardware, but the linked ONNX Runtime has no {provider} execution provider; rebuild \
         against an ONNX Runtime that has it (set ORT_LIB_LOCATION to its install prefix)"
    )
}

/// The provider for the machine's neural processor, which the operator asked
/// for.
fn plan_npu(facts: &ProviderFacts) -> std::result::Result<Accelerator, String> {
    if facts.intel_npu && facts.openvino_build {
        return if facts.openvino_provider {
            Ok(Accelerator::OpenVinoNpu)
        } else {
            Err(not_listed("openvino", "an Intel NPU", "OpenVINO"))
        };
    }
    if facts.amd_npu && facts.vitis_build {
        return if facts.vitis_provider {
            Ok(Accelerator::VitisAi)
        } else {
            Err(not_listed("vitis", "an AMD NPU", "Vitis AI"))
        };
    }
    if facts.qualcomm_npu && facts.qnn_build {
        return if facts.qnn_provider {
            Ok(Accelerator::QnnHtp)
        } else {
            Err(not_listed("qnn", "a Qualcomm NPU", "QNN"))
        };
    }
    Err(match (facts.intel_npu, facts.amd_npu, facts.qualcomm_npu) {
        (true, _, _) => "the neural processor was asked for and this machine has an Intel NPU, \
                         but this binary was built without the `openvino` feature",
        (_, true, _) => "the neural processor was asked for and this machine has an AMD NPU, \
                         but this binary was built without the `vitis` feature",
        (_, _, true) => "the neural processor was asked for and this machine has a Qualcomm NPU, \
                         but this binary was built without the `qnn` feature",
        _ => "the neural processor was asked for but this machine has none",
    }
    .to_string())
}

/// The providers for the machine's GPU, its vendor's own first.
fn plan_gpu(facts: &ProviderFacts) -> std::result::Result<Vec<Accelerator>, String> {
    if facts.cuda_gpu {
        if facts.cuda_build {
            if !facts.cuda_provider {
                return Err(
                    "this binary was built with the `cuda` feature and this machine has an \
                     NVIDIA GPU, but the linked ONNX Runtime has no CUDA execution provider; \
                     rebuild against a CUDA-enabled ONNX Runtime (set ORT_LIB_LOCATION to its \
                     install prefix)"
                        .to_string(),
                );
            }
            let mut plan = Vec::with_capacity(2);
            if facts.tensorrt_provider && facts.tensorrt_runtime {
                plan.push(Accelerator::TensorRt);
            }
            plan.push(Accelerator::Cuda);
            return Ok(plan);
        }
        return portable(facts)
            .map(|p| vec![p])
            .ok_or_else(|| no_provider(facts, "NVIDIA", "cuda", "CUDA"));
    }
    if facts.amd_gpu {
        if facts.rocm_build {
            return if facts.migraphx_provider {
                Ok(vec![Accelerator::MiGraphX])
            } else {
                Err(
                    "this binary was built with the `rocm` feature and this machine has an \
                     AMD GPU, but the linked ONNX Runtime has no MIGraphX execution provider \
                     (ONNX Runtime removed its ROCm provider in 1.23); rebuild against a \
                     MIGraphX-enabled ONNX Runtime (set ORT_LIB_LOCATION to its install prefix)"
                        .to_string(),
                )
            };
        }
        return portable(facts)
            .map(|p| vec![p])
            .ok_or_else(|| no_provider(facts, "AMD", "rocm", "ROCm"));
    }
    if facts.intel_gpu {
        if facts.openvino_build {
            return if facts.openvino_provider {
                Ok(vec![Accelerator::OpenVinoGpu])
            } else {
                Err(not_listed("openvino", "an Intel GPU", "OpenVINO"))
            };
        }
        return portable(facts)
            .map(|p| vec![p])
            .ok_or_else(|| no_provider(facts, "Intel", "openvino", "OpenVINO"));
    }
    if facts.qualcomm_gpu {
        if facts.qnn_build {
            return if facts.qnn_provider {
                Ok(vec![Accelerator::QnnGpu])
            } else {
                Err(not_listed("qnn", "a Qualcomm GPU", "QNN"))
            };
        }
        return portable(facts)
            .map(|p| vec![p])
            .ok_or_else(|| no_provider(facts, "Qualcomm", "qnn", "QNN"));
    }
    if facts.huawei_npu {
        if !facts.cann_build {
            return Err(
                "this machine has a Huawei Ascend NPU but this binary was built without the \
                 `cann` feature, so its ONNX runtimes cannot use it; install the CANN build"
                    .to_string(),
            );
        }
        return if facts.cann_provider {
            Ok(vec![Accelerator::Cann])
        } else {
            Err(not_listed("cann", "a Huawei Ascend NPU", "CANN"))
        };
    }
    if facts.coreml_build {
        if !facts.coreml_provider {
            return Err(
                "this binary was built with the `coreml` feature but the linked \
                 ONNX Runtime has no CoreML execution provider"
                    .to_string(),
            );
        }
        return Ok(vec![Accelerator::CoreMl]);
    }
    Ok(Vec::new())
}

/// Decide which execution providers every session registers, or why this
/// binary cannot serve ONNX models on this machine.
///
/// The neural processor leads when the operator asked for it; the GPU's
/// provider follows. Nothing is planned for a machine without accelerators,
/// whose sessions run on the CPU provider.
pub fn plan_providers(facts: &ProviderFacts) -> std::result::Result<Vec<Accelerator>, String> {
    let mut plan = Vec::new();
    if facts.npu {
        plan.push(plan_npu(facts)?);
    }
    plan.extend(plan_gpu(facts)?);
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_machine() -> ProviderFacts {
        ProviderFacts {
            cuda_gpu: true,
            cuda_build: true,
            cuda_provider: true,
            ..ProviderFacts::default()
        }
    }

    #[test]
    fn cuda_machine_runs_every_session_on_cuda() {
        assert_eq!(plan_providers(&cuda_machine()), Ok(vec![Accelerator::Cuda]));
    }

    #[test]
    fn tensorrt_leads_only_when_listed_and_installed() {
        let listed_not_installed = ProviderFacts {
            tensorrt_provider: true,
            ..cuda_machine()
        };
        assert_eq!(
            plan_providers(&listed_not_installed),
            Ok(vec![Accelerator::Cuda])
        );
        let installed = ProviderFacts {
            tensorrt_provider: true,
            tensorrt_runtime: true,
            ..cuda_machine()
        };
        assert_eq!(
            plan_providers(&installed),
            Ok(vec![Accelerator::TensorRt, Accelerator::Cuda])
        );
    }

    #[test]
    fn cuda_build_with_cpu_only_onnx_runtime_refuses() {
        let facts = ProviderFacts {
            cuda_provider: false,
            ..cuda_machine()
        };
        let cause = plan_providers(&facts).unwrap_err();
        assert!(cause.contains("no CUDA execution provider"), "{cause}");
    }

    #[test]
    fn cpu_build_on_a_gpu_machine_refuses() {
        let facts = ProviderFacts {
            cuda_gpu: true,
            ..ProviderFacts::default()
        };
        let cause = plan_providers(&facts).unwrap_err();
        assert!(cause.contains("without the `cuda` feature"), "{cause}");
    }

    #[test]
    fn vulkan_build_refuses_onnx_on_a_gpu_another_build_accelerates() {
        for (gpu, build) in [
            (
                ProviderFacts {
                    cuda_gpu: true,
                    ..ProviderFacts::default()
                },
                "CUDA build",
            ),
            (
                ProviderFacts {
                    amd_gpu: true,
                    ..ProviderFacts::default()
                },
                "ROCm build",
            ),
        ] {
            let facts = ProviderFacts {
                vulkan_build: true,
                ..gpu
            };
            let cause = plan_providers(&facts).unwrap_err();
            assert!(
                cause.contains("Vulkan build") && cause.contains(build),
                "{cause}"
            );
        }
        let intel = ProviderFacts {
            vulkan_build: true,
            ..ProviderFacts::default()
        };
        assert_eq!(plan_providers(&intel), Ok(Vec::new()));
    }

    #[test]
    fn amd_machine_refuses_the_cpu() {
        let cpu_build = ProviderFacts {
            amd_gpu: true,
            ..ProviderFacts::default()
        };
        let cause = plan_providers(&cpu_build).unwrap_err();
        assert!(cause.contains("without the `rocm` feature"), "{cause}");

        let no_provider = ProviderFacts {
            rocm_build: true,
            ..cpu_build
        };
        let cause = plan_providers(&no_provider).unwrap_err();
        assert!(cause.contains("no MIGraphX execution provider"), "{cause}");
        assert!(cause.contains("removed its ROCm provider"), "{cause}");

        let migraphx = ProviderFacts {
            migraphx_provider: true,
            ..no_provider
        };
        assert_eq!(plan_providers(&migraphx), Ok(vec![Accelerator::MiGraphX]));
    }

    #[test]
    fn intel_gpu_runs_on_openvino_or_refuses() {
        let machine = ProviderFacts {
            intel_gpu: true,
            ..ProviderFacts::default()
        };
        let cause = plan_providers(&machine).unwrap_err();
        assert!(cause.contains("an Intel GPU") && cause.contains("`openvino` feature"), "{cause}");
        let built = ProviderFacts {
            openvino_build: true,
            ..machine
        };
        assert!(plan_providers(&built).unwrap_err().contains("no OpenVINO execution provider"));
        let listed = ProviderFacts {
            openvino_provider: true,
            ..built
        };
        assert_eq!(plan_providers(&listed), Ok(vec![Accelerator::OpenVinoGpu]));
    }

    #[test]
    fn qualcomm_and_ascend_run_on_their_vendor_provider() {
        let adreno = ProviderFacts {
            qualcomm_gpu: true,
            qnn_build: true,
            qnn_provider: true,
            ..ProviderFacts::default()
        };
        assert_eq!(plan_providers(&adreno), Ok(vec![Accelerator::QnnGpu]));
        let ascend = ProviderFacts {
            huawei_npu: true,
            ..ProviderFacts::default()
        };
        assert!(plan_providers(&ascend).unwrap_err().contains("without the `cann` feature"));
        let ascend = ProviderFacts {
            cann_build: true,
            cann_provider: true,
            ..ascend
        };
        assert_eq!(plan_providers(&ascend), Ok(vec![Accelerator::Cann]));
    }

    #[test]
    fn a_portable_provider_serves_a_gpu_whose_own_feature_is_not_built() {
        for gpu in [
            ProviderFacts { cuda_gpu: true, ..ProviderFacts::default() },
            ProviderFacts { amd_gpu: true, ..ProviderFacts::default() },
            ProviderFacts { intel_gpu: true, ..ProviderFacts::default() },
            ProviderFacts { qualcomm_gpu: true, ..ProviderFacts::default() },
        ] {
            let webgpu = ProviderFacts { webgpu_build: true, webgpu_provider: true, ..gpu };
            assert_eq!(plan_providers(&webgpu), Ok(vec![Accelerator::WebGpu]));
            // DirectML leads WebGPU where both are built
            let both = ProviderFacts { directml_build: true, directml_provider: true, ..webgpu };
            assert_eq!(plan_providers(&both), Ok(vec![Accelerator::DirectMl]));
            // a portable feature the linked runtime does not carry is no provider
            let unlisted = ProviderFacts { webgpu_build: true, ..gpu };
            assert!(plan_providers(&unlisted).is_err());
        }
        // the vendor's own provider still leads when it is built
        let nvidia = ProviderFacts {
            cuda_gpu: true,
            cuda_build: true,
            cuda_provider: true,
            webgpu_build: true,
            webgpu_provider: true,
            ..ProviderFacts::default()
        };
        assert_eq!(plan_providers(&nvidia), Ok(vec![Accelerator::Cuda]));
        // without a GPU a portable build plans nothing
        let none = ProviderFacts { webgpu_build: true, webgpu_provider: true, ..ProviderFacts::default() };
        assert_eq!(plan_providers(&none), Ok(Vec::new()));
    }

    #[test]
    fn the_neural_processor_is_used_only_when_asked_for() {
        let laptop = ProviderFacts {
            intel_gpu: true,
            intel_npu: true,
            openvino_build: true,
            openvino_provider: true,
            ..ProviderFacts::default()
        };
        assert_eq!(plan_providers(&laptop), Ok(vec![Accelerator::OpenVinoGpu]));
        let asked = ProviderFacts { npu: true, ..laptop };
        assert_eq!(
            plan_providers(&asked),
            Ok(vec![Accelerator::OpenVinoNpu, Accelerator::OpenVinoGpu])
        );
        let ryzen = ProviderFacts {
            amd_gpu: true,
            amd_npu: true,
            rocm_build: true,
            migraphx_provider: true,
            vitis_build: true,
            vitis_provider: true,
            npu: true,
            ..ProviderFacts::default()
        };
        assert_eq!(
            plan_providers(&ryzen),
            Ok(vec![Accelerator::VitisAi, Accelerator::MiGraphX])
        );
        let snapdragon = ProviderFacts {
            qualcomm_gpu: true,
            qualcomm_npu: true,
            qnn_build: true,
            qnn_provider: true,
            npu: true,
            ..ProviderFacts::default()
        };
        assert_eq!(
            plan_providers(&snapdragon),
            Ok(vec![Accelerator::QnnHtp, Accelerator::QnnGpu])
        );
    }

    #[test]
    fn asking_for_a_neural_processor_that_cannot_be_used_refuses() {
        let none = ProviderFacts { npu: true, ..ProviderFacts::default() };
        assert!(plan_providers(&none).unwrap_err().contains("this machine has none"));
        let unbuilt = ProviderFacts { npu: true, amd_npu: true, ..ProviderFacts::default() };
        assert!(plan_providers(&unbuilt).unwrap_err().contains("without the `vitis` feature"));
        let unlisted = ProviderFacts { npu: true, intel_npu: true, openvino_build: true, ..ProviderFacts::default() };
        assert!(plan_providers(&unlisted).unwrap_err().contains("no OpenVINO execution provider"));
    }

    #[test]
    fn the_machine_names_its_accelerators() {
        use praecise_hardware::Accelerator as Device;
        let machine = Machine::with([
            Device { vendor: Vendor::Amd, kind: Kind::Gpu },
            Device { vendor: Vendor::Amd, kind: Kind::Npu },
            Device { vendor: Vendor::Huawei, kind: Kind::Npu },
        ]);
        let gpus = Gpus::from(&machine);
        assert_eq!(
            gpus,
            Gpus { amd: true, amd_npu: true, huawei: true, ..Gpus::default() }
        );
        assert_eq!(Gpus::from(&Machine::cpu_only()), Gpus::default());
    }

    #[test]
    fn machine_without_a_gpu_runs_on_the_cpu() {
        let cuda_build_no_gpu = ProviderFacts {
            cuda_gpu: false,
            ..cuda_machine()
        };
        assert_eq!(plan_providers(&cuda_build_no_gpu), Ok(Vec::new()));
        assert_eq!(plan_providers(&ProviderFacts::default()), Ok(Vec::new()));
    }

    #[test]
    fn coreml_build_needs_the_coreml_provider() {
        let missing = ProviderFacts {
            coreml_build: true,
            ..ProviderFacts::default()
        };
        assert!(plan_providers(&missing).is_err());
        let present = ProviderFacts {
            coreml_build: true,
            coreml_provider: true,
            ..ProviderFacts::default()
        };
        assert_eq!(plan_providers(&present), Ok(vec![Accelerator::CoreMl]));
    }
}
