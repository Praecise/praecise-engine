//! Errors for the diffusion family.

use thiserror::Error;

/// Result type for diffusion operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised while loading or running a diffusion pipeline.
#[derive(Debug, Error)]
pub enum Error {
    /// The host has GPU hardware this build cannot drive, or the GPU backend
    /// failed to initialise. Never answered by running on the CPU instead: a
    /// model that silently lands on the CPU is a hundred times slower and the
    /// caller would have no way to tell why.
    #[error("GPU required: {0}")]
    GpuRequired(String),

    /// The selected backend cannot execute an operation the graph needs.
    #[error("backend {backend} does not support operation {op} ({tensor})")]
    UnsupportedOp {
        /// Backend name.
        backend: String,
        /// ggml operation name.
        op: String,
        /// Name of the graph node.
        tensor: String,
    },

    /// A weight file is missing, truncated or not in the expected layout.
    #[error("weights: {0}")]
    Weights(String),

    /// A tensor the architecture requires is absent from the weight files.
    #[error("missing tensor {0}")]
    MissingTensor(String),

    /// A tensor is present with a shape the architecture does not accept.
    #[error("tensor {name} has shape {found:?}, expected {expected:?}")]
    TensorShape {
        /// Tensor name.
        name: String,
        /// Shape found in the file (outermost first).
        found: Vec<u64>,
        /// Shape the configuration implies.
        expected: Vec<u64>,
    },

    /// A model configuration file is missing a field or holds a value this
    /// engine does not implement.
    #[error("config: {0}")]
    Config(String),

    /// The request cannot be served as written.
    #[error("request: {0}")]
    Request(String),

    /// Tokenizer failure.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// The backend failed to allocate or compute.
    #[error("backend: {0}")]
    Backend(String),

    /// Filesystem error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Video encoding or decoding failed.
    #[error("video: {0}")]
    Video(#[from] praecise_codec::Error),
}
