//! Loaded model + drafter handles for the bundled llama.cpp backend.
//!
//! Backend-gated. A host that provides its own backend constructs equivalents
//! from its own binding and calls the speculative/batching entry points.

use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// A model loaded into the bundled backend, ready to serve.
pub struct LoadedModel {
    /// The loaded weights.
    pub model: LlamaModel,
    /// Shared backend handle.
    pub backend: Arc<LlamaBackend>,
    /// Effective context length (host-capped).
    pub context_length: u32,
}

/// A speculative-decoding drafter paired with a target model.
pub struct LoadedDrafter {
    /// The drafter weights (an MTP/DFlash head, or a small draft model).
    pub model: LlamaModel,
    /// Shared backend handle (echo of the target's, so a draft context can be
    /// constructed without re-resolving the backend).
    pub backend: Arc<LlamaBackend>,
    /// Context length for the draft model's context (same host cap as target).
    pub context_length: u32,
    /// Speculative algorithm for this drafter: 0 = draft-mtp, 1 = draft-dflash.
    pub spec_type: i32,
}

/// A LoRA adapter served on top of a base model, pinned by the SHA-256 of its GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterSpec {
    /// The adapter GGUF.
    pub path: PathBuf,
    /// SHA-256 of the file's bytes.
    pub sha256: [u8; 32],
    /// Scale the adapter is applied at (1.0 serves it as trained).
    pub scale: f32,
}

fn file_sha256(path: &std::path::Path) -> Result<[u8; 32]> {
    let bytes = std::fs::read(path).map_err(|e| Error::Adapter(format!("{}: {e}", path.display())))?;
    Ok(Sha256::digest(&bytes).into())
}

fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Loads the adapter of `spec` onto `model`, which then serves it on every context it creates.
/// The file is hashed before and after the backend reads it, and either hash differing from
/// `spec.sha256` refuses the adapter.
///
/// # Errors
/// [`Error::Adapter`] for an unreadable file, a hash mismatch, a model already serving an
/// adapter, or an adapter the backend refuses (for example one made for another model).
pub fn attach_adapter(model: &mut LlamaModel, spec: &AdapterSpec) -> Result<()> {
    if model.has_served_lora() {
        return Err(Error::Adapter("the model already serves an adapter".into()));
    }
    let check = |when: &str| -> Result<()> {
        let found = file_sha256(&spec.path)?;
        if found == spec.sha256 {
            Ok(())
        } else {
            Err(Error::Adapter(format!(
                "{} {when}: sha256 {} does not match the pinned {}",
                spec.path.display(),
                hex(&found),
                hex(&spec.sha256)
            )))
        }
    };
    check("before loading")?;
    let adapter = model
        .lora_adapter_init(&spec.path)
        .map_err(|e| Error::Adapter(format!("{}: {e}", spec.path.display())))?;
    check("after loading")?;
    model.set_served_lora(adapter, spec.scale);
    Ok(())
}
