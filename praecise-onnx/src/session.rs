//! Shared ONNX Runtime session construction.
//!
//! Every ONNX-backed model family (forecasting, vision, text embedding,
//! segmentation, text-prompted segmentation, detection and speech) builds its
//! [`Session`] through [`build_onnx_session`].
//!
//! Execution providers are chosen from the accelerators the host declares
//! through [`install`], never from per-model configuration. One rule holds
//! for every vendor: a machine with a GPU runs its sessions on that GPU, and
//! a binary that cannot do so refuses instead of running them on the CPU.
//! [`verify_accelerators`] fails at startup and every model load fails with
//! the cause, which names the build that would work.
//!
//! | GPU | Provider, by the feature the binary was built with |
//! |---|---|
//! | NVIDIA | `cuda`: TensorRT (when its library is installed), then CUDA |
//! | AMD | `rocm`: MIGraphX |
//! | Intel | `openvino`: OpenVINO on the GPU |
//! | Qualcomm | `qnn`: QNN on the Adreno GPU |
//! | Huawei Ascend | `cann`: CANN |
//! | Apple | `coreml`: CoreML |
//! | any of them | `directml` (Windows) or `webgpu`, when the vendor's own feature is not built |
//!
//! - ONNX Runtime removed its ROCm provider in 1.23; MIGraphX is the AMD
//!   provider, and the `rocm` feature builds it.
//! - ONNX Runtime has no Vulkan execution provider. A build with the `vulkan`
//!   feature and no ONNX provider for the machine's GPU refuses ONNX models,
//!   naming the build that accelerates them. `webgpu` is the portable
//!   provider that runs on any vendor's GPU.
//! - Neural processors take quantized, static-shape models, so one is used
//!   only when the operator asks for it ([`OnnxConfig::npu`]): OpenVINO on an
//!   Intel NPU, Vitis AI on an AMD NPU, QNN on a Qualcomm NPU. It is
//!   registered ahead of the GPU's provider, which takes what it leaves.
//! - A provider that is expected and fails to register is an error. The
//!   model does not load, rather than silently running on the CPU.
//! - Without an accelerator, sessions run on ONNX Runtime's CPU provider.
//!
//! Thread pools are bounded: intra-op and inter-op threads default to a
//! quarter of the detected cores (at least one), so one burst of requests
//! cannot take every core. [`OnnxConfig`] overrides this once for every
//! session in the process.

use std::path::Path;
use std::sync::OnceLock;

#[cfg(feature = "cann")]
use ort::ep::CANN;
#[cfg(feature = "coreml")]
use ort::ep::CoreML;
#[cfg(feature = "directml")]
use ort::ep::DirectML;
#[cfg(feature = "rocm")]
use ort::ep::MIGraphX;
#[cfg(feature = "openvino")]
use ort::ep::OpenVINO;
#[cfg(feature = "qnn")]
use ort::ep::QNN;
#[cfg(feature = "vitis")]
use ort::ep::Vitis;
#[cfg(feature = "webgpu")]
use ort::ep::WebGPU;
#[cfg(feature = "cuda")]
use ort::ep::{CUDA, TensorRT};
use ort::ep::{CPU, ExecutionProvider};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
pub use crate::plan::{Accelerator, Gpus, ProviderFacts, plan_providers};

/// Operator settings applied to every ONNX session in the process. Every
/// field is optional; unset fields are derived from the detected core count.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxConfig {
    /// Intra-op and inter-op thread count for every ONNX session.
    #[serde(default)]
    pub threads: Option<usize>,
    /// Requests each ONNX runtime runs at once.
    #[serde(default)]
    pub concurrency: Option<u32>,
    /// Requests each ONNX runtime holds waiting for a slot before it answers
    /// busy.
    #[serde(default)]
    pub queue_depth: Option<u32>,
    /// Run sessions on the machine's neural processor ahead of its GPU. Off
    /// unless set: an NPU takes quantized, static-shape models only.
    #[serde(default)]
    pub npu: Option<bool>,
}

/// Resolved ONNX serving limits for this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxSettings {
    /// Intra-op and inter-op thread count passed to every session builder.
    pub threads: usize,
    /// Concurrent requests per ONNX runtime.
    pub concurrency: u32,
    /// Waiting requests per ONNX runtime before new ones are refused as busy.
    pub queue_depth: u32,
    /// Whether sessions go to the machine's neural processor first.
    pub npu: bool,
}

impl OnnxSettings {
    /// Resolve `config` against a machine with `cores` cores.
    ///
    /// Threads default to a quarter of the cores (at least one). Concurrency
    /// defaults to as many requests as fit the cores at that thread count, so
    /// a runtime at full concurrency uses the machine without oversubscribing
    /// it. The queue holds four waiting requests per slot.
    pub fn resolve(config: &OnnxConfig, cores: usize) -> Self {
        let cores = cores.max(1);
        let threads = config.threads.unwrap_or(cores / 4).max(1);
        let concurrency = config
            .concurrency
            .unwrap_or((cores / threads).max(1) as u32)
            .max(1);
        let queue_depth = config
            .queue_depth
            .unwrap_or_else(|| concurrency.saturating_mul(4));
        Self {
            threads,
            concurrency,
            queue_depth,
            npu: config.npu.unwrap_or(false),
        }
    }

    fn detected() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::resolve(&OnnxConfig::default(), cores)
    }
}

static SETTINGS: OnceLock<OnnxSettings> = OnceLock::new();
static GPUS: OnceLock<Gpus> = OnceLock::new();
static ADMISSION: OnceLock<Admission> = OnceLock::new();

/// A host check every model file passes before ONNX Runtime opens it, such
/// as a parse in a confined process. It returns why the file is refused.
pub type Admission = fn(&Path) -> std::result::Result<(), String>;

/// Install the admission check for this process. Called once at startup,
/// before any model loads. Returns `false` when one was already installed,
/// in which case the earlier one stays in force.
pub fn set_admission(check: Admission) -> bool {
    ADMISSION.set(check).is_ok()
}

/// Install the operator's ONNX settings and the machine's GPUs for this
/// process. Called once at startup, before any model loads. Returns `false`
/// when they were already fixed, in which case the earlier ones stay in force.
pub fn install(settings: OnnxSettings, gpus: Gpus) -> bool {
    let fresh = GPUS.set(gpus).is_ok();
    SETTINGS.set(settings).is_ok() && fresh
}

/// The ONNX settings in force: the installed ones, or the core-derived
/// defaults when nothing was installed.
pub fn settings() -> OnnxSettings {
    *SETTINGS.get_or_init(OnnxSettings::detected)
}

/// Whether the linked ONNX Runtime lists the provider `P`.
#[allow(dead_code)]
fn provider_listed<P: ExecutionProvider>() -> std::result::Result<bool, String> {
    P::is_available()
        .map_err(|e| format!("could not list ONNX Runtime execution providers: {e}"))
}

/// Whether the TensorRT runtime library can be loaded by this process.
#[cfg(feature = "cuda")]
fn tensorrt_runtime_loadable() -> bool {
    const NAMES: &[&str] = if cfg!(windows) {
        &["nvinfer_10.dll"]
    } else {
        &["libnvinfer.so.10"]
    };
    // SAFETY: loading the TensorRT library runs only its own initialisers;
    // the handle is dropped immediately and no symbol is called through it.
    NAMES
        .iter()
        .any(|name| unsafe { libloading::Library::new(*name) }.is_ok())
}

#[cfg(not(feature = "cuda"))]
fn tensorrt_runtime_loadable() -> bool {
    false
}

/// Whether the linked ONNX Runtime lists a provider; `false` in a build
/// without the provider's feature, where its type does not exist.
macro_rules! listed {
    ($feature:literal, $provider:ty) => {{
        #[cfg(feature = $feature)]
        let listed = provider_listed::<$provider>()?;
        #[cfg(not(feature = $feature))]
        let listed = false;
        listed
    }};
}

fn gather_facts() -> std::result::Result<ProviderFacts, String> {
    let gpus = GPUS.get().copied().unwrap_or_default();
    let tensorrt_provider = listed!("cuda", TensorRT);
    Ok(ProviderFacts {
        cuda_gpu: gpus.nvidia,
        cuda_build: cfg!(feature = "cuda"),
        cuda_provider: listed!("cuda", CUDA),
        tensorrt_provider,
        tensorrt_runtime: tensorrt_provider && tensorrt_runtime_loadable(),
        vulkan_build: cfg!(feature = "vulkan"),
        amd_gpu: gpus.amd,
        rocm_build: cfg!(feature = "rocm"),
        migraphx_provider: listed!("rocm", MIGraphX),
        intel_gpu: gpus.intel,
        intel_npu: gpus.intel_npu,
        openvino_build: cfg!(feature = "openvino"),
        openvino_provider: listed!("openvino", OpenVINO),
        qualcomm_gpu: gpus.qualcomm,
        qualcomm_npu: gpus.qualcomm_npu,
        qnn_build: cfg!(feature = "qnn"),
        qnn_provider: listed!("qnn", QNN),
        huawei_npu: gpus.huawei,
        cann_build: cfg!(feature = "cann"),
        cann_provider: listed!("cann", CANN),
        amd_npu: gpus.amd_npu,
        vitis_build: cfg!(feature = "vitis"),
        vitis_provider: listed!("vitis", Vitis),
        directml_build: cfg!(feature = "directml"),
        directml_provider: listed!("directml", DirectML),
        webgpu_build: cfg!(feature = "webgpu"),
        webgpu_provider: listed!("webgpu", WebGPU),
        coreml_build: cfg!(feature = "coreml"),
        coreml_provider: listed!("coreml", CoreML),
        npu: settings().npu,
    })
}

static ORT_ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();

/// The process's ONNX Runtime environment, created on first use.
pub(crate) fn ort_environment() -> Result<&'static Environment> {
    ORT_ENV
        .get_or_init(|| {
            ort::init()
                .with_name("praecise")
                .build()
                .map_err(|e| format!("ONNX Runtime environment: {e}"))
        })
        .as_ref()
        .map_err(|e| Error::InvalidModel(e.clone()))
}

static PLAN: OnceLock<std::result::Result<Vec<Accelerator>, String>> = OnceLock::new();

/// The execution providers every ONNX session registers on this machine.
///
/// Planned once per process from the GPUs declared through [`install`];
/// before that every call fails, so a session never starts on a guessed
/// provider. Hosts make the first call through [`verify_accelerators`].
pub fn accelerators() -> Result<&'static [Accelerator]> {
    if GPUS.get().is_none() {
        return Err(Error::ProviderNotAvailable(
            "the machine's GPUs were not declared; call praecise_onnx::session::install before \
             loading ONNX models"
                .to_string(),
        ));
    }
    PLAN.get_or_init(|| gather_facts().and_then(|facts| plan_providers(&facts)))
        .as_deref()
        .map_err(|cause| Error::ProviderNotAvailable(cause.clone()))
}

/// Startup check: fails when this binary cannot run ONNX models on the
/// accelerator this machine has.
pub fn verify_accelerators() -> Result<()> {
    let plan = accelerators()?;
    let names: Vec<&str> = plan.iter().map(|a| a.name()).collect();
    let threads = settings().threads;
    if names.is_empty() {
        tracing::info!(threads, "ONNX runtimes run on the CPU");
    } else {
        tracing::info!(providers = ?names, threads, "ONNX runtimes run on the accelerator");
    }
    Ok(())
}

/// The file name of a QNN backend library on this platform.
#[cfg(feature = "qnn")]
fn qnn_backend(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.dll")
    } else {
        format!("lib{name}.so")
    }
}

#[allow(unused_variables)]
fn register(
    accelerator: Accelerator,
    builder: &mut SessionBuilder,
) -> std::result::Result<(), ort::Error> {
    match accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::TensorRt => TensorRT::default().register(builder),
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => CUDA::default().register(builder),
        #[cfg(feature = "rocm")]
        Accelerator::MiGraphX => MIGraphX::default().register(builder),
        #[cfg(feature = "openvino")]
        Accelerator::OpenVinoGpu => OpenVINO::default().with_device_type("GPU").register(builder),
        #[cfg(feature = "openvino")]
        Accelerator::OpenVinoNpu => OpenVINO::default().with_device_type("NPU").register(builder),
        #[cfg(feature = "qnn")]
        Accelerator::QnnGpu => QNN::default().with_backend_path(qnn_backend("QnnGpu")).register(builder),
        #[cfg(feature = "qnn")]
        Accelerator::QnnHtp => QNN::default().with_backend_path(qnn_backend("QnnHtp")).register(builder),
        #[cfg(feature = "cann")]
        Accelerator::Cann => CANN::default().register(builder),
        #[cfg(feature = "vitis")]
        Accelerator::VitisAi => Vitis::default().register(builder),
        #[cfg(feature = "coreml")]
        Accelerator::CoreMl => CoreML::default().register(builder),
        #[cfg(feature = "directml")]
        Accelerator::DirectMl => DirectML::default().register(builder),
        #[cfg(feature = "webgpu")]
        Accelerator::WebGpu => WebGPU::default().register(builder),
        // the plan only names providers of this build
        #[allow(unreachable_patterns)]
        other => Err(ort::Error::new(format!(
            "the {} execution provider is not built into this binary",
            other.name()
        ))),
    }
    .map_err(ort::Error::from)
}

/// A session builder with Level3 graph optimizations, the thread bounds in
/// `settings`, and host memory that a run gives back when it ends.
///
/// Host tensors are allocated and freed per run instead of from an arena: an
/// arena keeps the largest working set it ever reached for the life of the
/// session, so one long input (a video of several thousand tokens) would
/// hold its peak for as long as the model stays loaded.
///
/// DirectML supports neither memory-pattern planning nor parallel graph
/// execution, so a session that registers it (`directml`) turns both off.
fn session_builder(settings: &OnnxSettings, directml: bool) -> Result<SessionBuilder> {
    let builder_error = |e: &dyn std::fmt::Display| Error::InvalidModel(format!("ORT session builder: {e}"));
    let mut builder = Session::builder(ort_environment()?)
        .map_err(|e| builder_error(&e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| builder_error(&e))?
        .with_intra_threads(settings.threads)
        .map_err(|e| builder_error(&e))?
        .with_inter_threads(settings.threads)
        .map_err(|e| builder_error(&e))?;
    if directml {
        builder = builder
            .with_memory_pattern(false)
            .map_err(|e| builder_error(&e))?
            .with_parallel_execution(false)
            .map_err(|e| builder_error(&e))?;
    }
    CPU::default()
        .with_arena_allocator(false)
        .register(&mut builder)
        .map_err(|e| Error::InvalidModel(format!("ORT session builder: {e}")))?;
    Ok(builder)
}

/// Build an ONNX Runtime session from a model file on this machine's
/// accelerator, with the process's thread bounds.
///
/// `what` names the artifact for error messages (e.g. `"model"`,
/// `"encoder"`, `"decoder"`). Fails without loading when the machine's
/// accelerator is unusable, its provider does not register, or the
/// installed [`Admission`] check refuses the file.
pub fn build_onnx_session(path: impl AsRef<Path>, what: &str) -> Result<Session> {
    let plan = accelerators()?;
    let mut builder = session_builder(&settings(), plan.contains(&Accelerator::DirectMl))?;
    for accelerator in plan {
        register(*accelerator, &mut builder).map_err(|e| {
            Error::ProviderNotAvailable(format!(
                "{} execution provider failed to register for the {what}: {e}; refusing to load \
                 it on the CPU",
                accelerator.name()
            ))
        })?;
    }
    if let Some(admit) = ADMISSION.get() {
        admit(path.as_ref()).map_err(|e| Error::InvalidModel(format!("{what}: {e}")))?;
    }
    builder
        .commit_from_file(path.as_ref())
        .map_err(|e| Error::ProviderNotAvailable(format!("ORT load {}: {}", what, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refuse_marked(path: &Path) -> std::result::Result<(), String> {
        if path.ends_with("refused.onnx") {
            Err("refused by the host".to_string())
        } else {
            Ok(())
        }
    }

    #[test]
    fn the_admission_check_runs_before_the_load() {
        install(OnnxSettings::resolve(&OnnxConfig::default(), 4), Gpus::default());
        set_admission(refuse_marked);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refused.onnx");
        std::fs::write(&path, b"not an onnx model").unwrap();
        match build_onnx_session(&path, "encoder") {
            Err(Error::InvalidModel(msg)) => assert_eq!(msg, "encoder: refused by the host"),
            other => panic!("expected the admission refusal, got {other:?}"),
        }
        let admitted = dir.path().join("admitted.onnx");
        std::fs::write(&admitted, b"not an onnx model").unwrap();
        assert!(matches!(
            build_onnx_session(&admitted, "model"),
            Err(Error::ProviderNotAvailable(_))
        ));
    }

    #[test]
    fn threads_default_to_a_quarter_of_the_cores() {
        assert_eq!(
            OnnxSettings::resolve(&OnnxConfig::default(), 56).threads,
            14
        );
        assert_eq!(OnnxSettings::resolve(&OnnxConfig::default(), 3).threads, 1);
        assert_eq!(OnnxSettings::resolve(&OnnxConfig::default(), 0).threads, 1);
    }

    #[test]
    fn configured_limits_override_the_defaults() {
        let config = OnnxConfig {
            threads: Some(3),
            concurrency: Some(2),
            queue_depth: Some(5),
            npu: Some(true),
        };
        let s = OnnxSettings::resolve(&config, 56);
        assert_eq!(
            s,
            OnnxSettings {
                threads: 3,
                concurrency: 2,
                queue_depth: 5,
                npu: true,
            }
        );
    }

    #[test]
    fn default_concurrency_fits_the_cores() {
        let s = OnnxSettings::resolve(&OnnxConfig::default(), 56);
        assert!(s.threads as u32 * s.concurrency <= 56);
        assert_eq!(s.queue_depth, s.concurrency * 4);
    }

    #[test]
    fn configured_threads_reach_the_session_builder() {
        let s = OnnxSettings::resolve(
            &OnnxConfig {
                threads: Some(2),
                ..OnnxConfig::default()
            },
            8,
        );
        assert_eq!(s.threads, 2);
        session_builder(&s, false).expect("the builder accepts the configured thread bounds");
    }
}
