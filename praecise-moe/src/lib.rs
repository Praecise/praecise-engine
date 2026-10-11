//! Expert-parallel mixture-of-experts execution for the Praecise Engine.
//!
//! - [`exec`]: expert FFNs and gating networks loaded from safetensors
//!   blobs, the expert residency cache ([`exec::MoeExpertRuntime`]), the
//!   request and response types a router and an expert holder exchange, and
//!   the combine step.
//! - [`compute`]: the expert matmul backends, chosen from the GPUs the host
//!   declares ([`compute::Gpus`]): the CPU, CUDA (`cuda` feature) or any
//!   other GPU through wgpu (`wgpu` feature). A GPU that no compiled backend
//!   can drive is refused rather than served on the CPU.
//! - [`quant`]: GGUF block codecs (Q4_K, Q6_K, Q8_0) for expert weights and
//!   activations.

pub mod compute;
#[cfg(feature = "cuda")]
pub mod compute_cuda;
#[cfg(feature = "wgpu")]
pub mod compute_wgpu;
pub mod exec;
pub mod quant;

pub use compute::{BackendKind, ComputeBackend, CpuCompute, ExpertCompute, Gpus, Weight};
pub use exec::*;
pub use quant::{QuantError, QuantKind, QuantMatrix};
