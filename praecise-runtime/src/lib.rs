//! Praecise general inference-acceleration runtime.
//!
//! Backend-agnostic acceleration on top of an inference backend (llama.cpp
//! first). Provides the generation API — configuration, results, sampling and
//! speculative-decode orchestration (block/DFlash and multi-token-prediction).
//!
//! ## Backend loading is configurable — never loaded twice
//!
//! The llama.cpp backend is an **optional** dependency, enabled by the
//! `bundled-llama` feature:
//!
//! - **Standalone consumer** — enable `bundled-llama` (or an accelerator
//!   feature such as `cuda`, which implies it). Praecise pulls, builds and
//!   initialises the backend.
//! - **Host that already links llama-cpp-2** — depend on this crate WITHOUT
//!   `bundled-llama` and pass in the host's existing backend and model handles.
//!   The native `libllama` and its GPU context are then loaded exactly once
//!   instead of a second copy contending for the device.
//!
//! The backend-agnostic surface — [`config::GenerationConfig`],
//! [`result::InferenceResult`], [`result::StopReason`], [`error::Error`] —
//! compiles and is usable with or without a bundled backend.

/// Backend selection: which inference runtime executes a request, and what
/// each one is actually capable of. Backend-agnostic by construction.
pub mod backend;
/// Allocator simulation — **retired**. Its cost model was refuted by
/// measurement; kept as a record of a self-consistent simulation being wrong.
pub mod bench_alloc;
pub mod config;
/// Published speculative drafters, their licences, and the known ways a
/// deployment silently underperforms. Backend-agnostic catalogue data.
pub mod drafters;
/// Entropy-routed speculation: deciding which positions are worth drafting at
/// all, rather than how to draft them better. Backend-agnostic.
pub mod entropy;
pub mod error;
/// N-gram self-speculation. Backend-agnostic: it drafts from token history
/// alone, so it compiles and is useful with or without a bundled backend.
pub mod ngram;
/// PRAX — **retired**. Draft-block sizing, measured on hardware to have no
/// effect on throughput (~5% across a 16x range, non-monotonic). Kept for the
/// negative result; nothing calls it. See the module docs.
pub mod prax;
pub mod prompt;
/// Speculation policy: which drafting method and how long a block, given the
/// model, the hardware and the current load. Backend-agnostic — it is a
/// decision, not a mechanism.
pub mod spec_policy;
pub mod result;
/// Adapters for backends reached over HTTP (vLLM, SGLang). Builds and parses
/// requests; the caller supplies the HTTP client. Backend-agnostic.
pub mod served;
/// Request shaping: grouping and rewriting requests so a batch is cheap before
/// it reaches a backend. Backend-agnostic.
pub mod shaping;
/// Draft verification: acceptance relations, and multi-draft sampling without
/// replacement. Backend-agnostic.
pub mod verify;
pub mod stream;
pub mod toploc;

/// llama.cpp sampler-chain assembly. Compiled only with a bundled backend.
#[cfg(feature = "bundled-llama")]
pub mod sampling;

/// Continuous batching engine. Compiled only with a bundled backend.
#[cfg(feature = "bundled-llama")]
pub mod batching;

/// Loaded model + drafter handles. Compiled only with a bundled backend.
#[cfg(feature = "bundled-llama")]
pub mod loaded;

/// N-gram self-speculative decode. Compiled only with a bundled backend;
/// the cache itself (`ngram`) is backend-agnostic.
#[cfg(feature = "bundled-llama")]
pub mod ngram_decode;

/// Self-speculative decode (DFlash / MTP). Compiled only with a bundled backend.
#[cfg(feature = "bundled-llama")]
pub mod speculative;

pub use config::GenerationConfig;
pub use backend::{plan_for as plan_speculation_for, Backend, Capabilities, Integration};
pub use drafters::{preflight as preflight_drafter, DeploymentIntent, Drafter, Licence, Trap};
pub use entropy::{block_size_for, EntropyThresholds, Regime, RegimeTracker};
pub use error::{Error, Result};
pub use served::{Completion, Dialect, Endpoint, HttpRequest};
pub use shaping::{group_by_class, BatchClass, Rewrite};
pub use prax::{Allocation, Prax, Signals};
pub use verify::{acceptance_rate, expected_tokens, MultiDraftSampling};
pub use ngram::NgramCache;
pub use spec_policy::{plan as plan_speculation, LoadState, ModelProfile, SpecMethod, SpecPlan, SpecPolicy};
pub use prompt::render_chatml_prompt;
pub use result::{ChatMessage, InferenceResult, StopReason};
pub use stream::{matched_stop_len, StopStream};

#[cfg(feature = "bundled-llama")]
pub use batching::{max_slots, BatchEngine, BatchPrompt, BatchRequest};
#[cfg(feature = "bundled-llama")]
pub use loaded::{LoadedDrafter, LoadedModel};
#[cfg(feature = "bundled-llama")]
pub use sampling::{build_sampler_chain, build_sampler_chain_with_grammar};
#[cfg(feature = "bundled-llama")]
pub use ngram_decode::{generate_ngram_speculative, NgramStats};
#[cfg(feature = "bundled-llama")]
pub use speculative::generate_speculative;
