//! A model parse with no accelerator, for a process confined after start-up.
//!
//! A host that parses untrusted model files in a locked-down child process
//! starts ONNX Runtime and gathers the external data before the lock-down,
//! with [`ParseCheck::new`] and [`ParseCheck::with_external_data`], then asks
//! ONNX Runtime for its verdict on the model bytes with [`ParseCheck::check`].
//! The check runs ONNX Runtime's protobuf reader over the whole model and
//! the graph checks a session load makes, on one thread, with graph
//! optimisation off and no execution provider beyond the CPU.

use std::sync::Arc;

use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

/// A prepared parse: ONNX Runtime started and the external data held.
pub struct ParseCheck {
    builder: SessionBuilder,
}

impl std::fmt::Debug for ParseCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParseCheck").finish_non_exhaustive()
    }
}

impl ParseCheck {
    /// Start ONNX Runtime and make a one-thread session builder with graph
    /// optimisation off.
    pub fn new() -> Result<Self, String> {
        let environment = crate::session::ort_environment()
            .map_err(|e| format!("ONNX Runtime did not start: {e}"))?;
        let builder = Session::builder(environment)
            .map_err(|e| e.to_string())
            .and_then(|b| {
                b.with_optimization_level(GraphOptimizationLevel::Disable)
                    .map_err(|e| e.to_string())
            })
            .and_then(|b| b.with_intra_threads(1).map_err(|e| e.to_string()))
            .and_then(|b| b.with_inter_threads(1).map_err(|e| e.to_string()))
            .map_err(|e| format!("ONNX Runtime session builder: {e}"))?;
        Ok(Self { builder })
    }

    /// Hold the bytes of the external-data file the model names `name`.
    pub fn with_external_data(self, name: &str, data: Arc<[u8]>) -> Result<Self, String> {
        let builder = self
            .builder
            .with_external_initializer_file_in_memory(name, data)
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(Self { builder })
    }

    /// ONNX Runtime's verdict on the model bytes.
    pub fn check(mut self, model: &[u8]) -> Result<(), String> {
        self.builder
            .commit_from_memory_directly(model)
            .map(drop)
            .map_err(|e| format!("ONNX Runtime refused the model: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_that_are_not_a_model_are_refused() {
        let err = ParseCheck::new()
            .expect("ONNX Runtime starts")
            .check(b"not an onnx model")
            .unwrap_err();
        assert!(err.starts_with("ONNX Runtime refused the model"), "{err}");
    }

    #[test]
    fn external_data_is_held_by_name() {
        let check = ParseCheck::new()
            .expect("ONNX Runtime starts")
            .with_external_data("weights.bin", Arc::from(vec![0u8; 16]))
            .expect("external data is accepted");
        assert!(check.check(&[]).is_err());
    }
}
