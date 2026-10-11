//! Errors raised while loading or running an ONNX model.

use thiserror::Error as ThisError;

/// Result type for ONNX model operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised while loading or running an ONNX model.
#[derive(Debug, ThisError)]
pub enum Error {
    /// A model, file or tensor the call names does not exist.
    #[error("model not found: {0}")]
    ModelNotFound(String),
    /// This build or machine cannot run the model: a missing execution
    /// provider, an undeclared GPU, or a session that would not register.
    #[error("execution provider not available: {0}")]
    ProviderNotAvailable(String),
    /// Inference failed while running a loaded model.
    #[error("inference error: {0}")]
    InferenceError(String),
    /// The model files are not what the family expects.
    #[error("invalid model: {0}")]
    InvalidModel(String),
    /// The request is not valid for the loaded model.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
