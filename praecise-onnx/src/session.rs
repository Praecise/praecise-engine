//! Shared ONNX Runtime session construction.
//!
//! Every ONNX-backed model family (forecasting, vision, text embedding,
//! segmentation, text-prompted segmentation, detection and speech) builds its
//! [`Session`] through [`build_onnx_session`].
//!
//! Execution providers are chosen from the GPUs the host declares through
//! [`install`], never from per-model configuration:
//! - On a machine with an NVIDIA GPU every session runs on CUDA, with
//!   TensorRT ahead of it when the linked ONNX Runtime carries the TensorRT
//!   provider and the TensorRT library is installed. A binary that cannot
//!   do that (built without the `cuda` feature, or linked against an ONNX
//!   Runtime without the CUDA provider) refuses: [`verify_accelerators`]
//!   fails at startup and every model load fails with the cause.
//! - On a machine with an AMD GPU every session runs on MIGraphX and/or
//!   ROCm, whichever the linked ONNX Runtime lists. A binary built without
//!   the `rocm` feature, or linked against an ONNX Runtime with neither
//!   provider, refuses the same way.
//! - ONNX Runtime has no Vulkan execution provider, so on a machine whose
//!   only GPU is another Vulkan-class device (Intel and others) sessions run
//!   on the CPU provider. A build with the `vulkan` feature (and not the
//!   matching `cuda` or `rocm` one) refuses ONNX models on an NVIDIA or AMD
//!   GPU, naming the build that accelerates them.
//! - A provider that is expected and fails to register is an error. The
//!   model does not load, rather than silently running on the CPU.
//! - On Apple builds with `coreml`, sessions run on CoreML.
//! - Without an accelerator, sessions run on ONNX Runtime's CPU provider.
//!
//! Thread pools are bounded: intra-op and inter-op threads default to a
//! quarter of the detected cores (at least one), so one burst of requests
//! cannot take every core. [`OnnxConfig`] overrides this once for every
//! session in the process.

use std::path::Path;
use std::sync::OnceLock;

#[cfg(feature = "coreml")]
use ort::ep::CoreML;
#[cfg(feature = "cuda")]
use ort::ep::{CUDA, TensorRT};
use ort::ep::{CPU, ExecutionProvider};
#[cfg(feature = "rocm")]
use ort::ep::{MIGraphX, ROCm};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

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

/// The GPUs visible to this process, as the host detected them. The
/// execution providers follow from these; the engine does not guess them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gpus {
    /// An NVIDIA GPU is visible.
    pub nvidia: bool,
    /// An AMD GPU is visible.
    pub amd: bool,
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

/// A hardware execution provider an ONNX session registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accelerator {
    TensorRt,
    Cuda,
    MiGraphX,
    Rocm,
    CoreMl,
}

impl Accelerator {
    pub fn name(self) -> &'static str {
        match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
            Self::MiGraphX => "MIGraphX",
            Self::Rocm => "ROCm",
            Self::CoreMl => "CoreML",
        }
    }
}

/// What this process has to accelerate ONNX sessions with.
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
    /// The linked ONNX Runtime lists the ROCm execution provider.
    pub rocm_provider: bool,
    /// The linked ONNX Runtime lists the MIGraphX execution provider.
    pub migraphx_provider: bool,
    /// This binary was built with the `coreml` feature.
    pub coreml_build: bool,
    /// The linked ONNX Runtime lists the CoreML execution provider.
    pub coreml_provider: bool,
}

/// Decide which execution providers every session registers, or why this
/// binary cannot serve ONNX models on this machine.
pub fn plan_providers(facts: &ProviderFacts) -> std::result::Result<Vec<Accelerator>, String> {
    if facts.cuda_gpu {
        if !facts.cuda_build && facts.vulkan_build {
            return Err("this is a Vulkan build: it runs GGUF models on \
                        this NVIDIA GPU, but ONNX Runtime has no Vulkan execution provider, so \
                        it does not serve ONNX models here; install the CUDA build to serve them"
                .to_string());
        }
        if !facts.cuda_build {
            return Err(
                "this machine has an NVIDIA GPU but this binary was built without the \
                        `cuda` feature, so its ONNX runtimes cannot use it; install the CUDA \
                        build"
                    .to_string(),
            );
        }
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
    if facts.amd_gpu {
        if !facts.rocm_build && facts.vulkan_build {
            return Err("this is a Vulkan build: it runs GGUF models on \
                        this AMD GPU, but ONNX Runtime has no Vulkan execution provider, so it \
                        does not serve ONNX models here; install the ROCm build to serve them"
                .to_string());
        }
        if !facts.rocm_build {
            return Err(
                "this machine has an AMD GPU but this binary was built without the \
                        `rocm` feature, so its ONNX runtimes cannot use it; install the ROCm \
                        build"
                    .to_string(),
            );
        }
        let mut plan = Vec::with_capacity(2);
        if facts.migraphx_provider {
            plan.push(Accelerator::MiGraphX);
        }
        if facts.rocm_provider {
            plan.push(Accelerator::Rocm);
        }
        if plan.is_empty() {
            return Err(
                "this binary was built with the `rocm` feature and this machine has an \
                        AMD GPU, but the linked ONNX Runtime has neither the MIGraphX nor the \
                        ROCm execution provider; rebuild against a ROCm-enabled ONNX Runtime \
                        (set ORT_LIB_LOCATION to its install prefix)"
                    .to_string(),
            );
        }
        return Ok(plan);
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

/// Whether the linked ONNX Runtime lists the provider `P`.
#[cfg_attr(not(any(feature = "cuda", feature = "rocm", feature = "coreml")), allow(dead_code))]
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

fn gather_facts() -> std::result::Result<ProviderFacts, String> {
    let gpus = GPUS.get().copied().unwrap_or_default();
    let cuda_gpu = gpus.nvidia;
    let amd_gpu = gpus.amd;
    let cuda_build = cfg!(feature = "cuda");
    let rocm_build = cfg!(feature = "rocm");
    let vulkan_build = cfg!(feature = "vulkan");
    let coreml_build = cfg!(feature = "coreml");
    // the provider types exist only in a build with their feature
    #[cfg(feature = "cuda")]
    let (cuda_provider, tensorrt_provider) = (provider_listed::<CUDA>()?, provider_listed::<TensorRT>()?);
    #[cfg(not(feature = "cuda"))]
    let (cuda_provider, tensorrt_provider) = (false, false);
    let tensorrt_runtime = tensorrt_provider && tensorrt_runtime_loadable();
    #[cfg(feature = "rocm")]
    let (rocm_provider, migraphx_provider) = (provider_listed::<ROCm>()?, provider_listed::<MIGraphX>()?);
    #[cfg(not(feature = "rocm"))]
    let (rocm_provider, migraphx_provider) = (false, false);
    #[cfg(feature = "coreml")]
    let coreml_provider = provider_listed::<CoreML>()?;
    #[cfg(not(feature = "coreml"))]
    let coreml_provider = false;
    Ok(ProviderFacts {
        cuda_gpu,
        cuda_build,
        cuda_provider,
        tensorrt_provider,
        tensorrt_runtime,
        vulkan_build,
        amd_gpu,
        rocm_build,
        rocm_provider,
        migraphx_provider,
        coreml_build,
        coreml_provider,
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

#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "coreml")),
    allow(unused_variables)
)]
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
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => ROCm::default().register(builder),
        #[cfg(feature = "coreml")]
        Accelerator::CoreMl => CoreML::default().register(builder),
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
fn session_builder(settings: &OnnxSettings) -> Result<SessionBuilder> {
    let builder_error = |e: &dyn std::fmt::Display| Error::InvalidModel(format!("ORT session builder: {e}"));
    let mut builder = Session::builder(ort_environment()?)
        .map_err(|e| builder_error(&e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| builder_error(&e))?
        .with_intra_threads(settings.threads)
        .map_err(|e| builder_error(&e))?
        .with_inter_threads(settings.threads)
        .map_err(|e| builder_error(&e))?;
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
    let mut builder = session_builder(&settings())?;
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
        assert!(
            cause.contains("neither the MIGraphX nor the ROCm"),
            "{cause}"
        );

        let rocm = ProviderFacts {
            rocm_provider: true,
            ..no_provider
        };
        assert_eq!(plan_providers(&rocm), Ok(vec![Accelerator::Rocm]));
        let both = ProviderFacts {
            migraphx_provider: true,
            ..rocm
        };
        assert_eq!(
            plan_providers(&both),
            Ok(vec![Accelerator::MiGraphX, Accelerator::Rocm])
        );
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
        };
        let s = OnnxSettings::resolve(&config, 56);
        assert_eq!(
            s,
            OnnxSettings {
                threads: 3,
                concurrency: 2,
                queue_depth: 5
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
        session_builder(&s).expect("the builder accepts the configured thread bounds");
    }
}
