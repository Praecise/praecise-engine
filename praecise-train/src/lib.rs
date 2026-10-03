//! Praecise training substrate.
//!
//! Training runs on the same ggml graphs, kernels and devices that serve the
//! model. This crate holds the parts that make a training step replayable and
//! its products verifiable:
//!
//! - [`steplog`]: the step contract `step(StepSpec) -> StepResult` and the
//!   hash-linked step log whose head commits to the whole history.
//! - [`checkpoint`]: canonical tensor payloads cut into 4 MiB chunks, the
//!   SHA-256 Merkle `state_root`, manifests with lineage, and chunk proofs.
//! - [`philox`]: counter-based random numbers keyed by seed, step, tensor and
//!   element, for sampling, dropout, stochastic rounding and noise.
//! - [`sampler`]: stateless batch selection by keyed permutation.
//! - [`kernel_class`]: kernel classes, deterministic-mode op checks and device
//!   resolution that refuses a missing accelerator.
//! - [`recipe`]: the canonical recipe and its hash.
//! - [`update`]: parameter-delta and optimizer-state primitives.
//! - [`objective`]: token objectives (SFT, DPO, GRPO, distillation) as per-token
//!   weights, and the Muon orthogonalization.
//! - `engine` (feature `engine`): `LoRA` training steps on the serving graph.
//!
//! The engine contains no network code and no multi-machine coordination:
//! every function here is a pure function of local inputs.

pub mod canonical;
pub mod checkpoint;
#[cfg(feature = "engine")]
pub mod engine;
pub mod hash;
pub mod kernel_class;
pub mod merkle;
pub mod objective;
pub mod philox;
pub mod recipe;
pub mod sampler;
pub mod steplog;
pub mod update;

/// Errors of the training substrate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Bytes that are not a valid encoding of the expected object.
    #[error("format: {0}")]
    Format(String),
    /// Two things that must agree (hashes, shapes, log links) do not.
    #[error("mismatch: {0}")]
    Mismatch(String),
    /// A run or operation was refused, with the reason.
    #[error("refused: {0}")]
    Refused(String),
    /// Filesystem failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
