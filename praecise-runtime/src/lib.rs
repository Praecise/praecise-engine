//! Praecise general inference-acceleration runtime.
//!
//! Backend-agnostic acceleration on top of a pulled inference backend
//! (llama.cpp first, via `llama-cpp-2`). Provides the inference API and the
//! speculative-decode orchestration (block/DFlash and multi-token-prediction),
//! so any consumer gets accelerated decode by depending on this crate.

// Modules:
//   - config      generation configuration
//   - result      inference result + stop reasons
//   - error       inference errors
//   - sampling    sampler chain + stop-sequence streaming
//   - speculative speculative decode loop + prefill
