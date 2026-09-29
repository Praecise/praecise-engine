//! Praecise Engine diffusion family.
//!
//! Iterative-denoiser models run natively on the same ggml backend as the
//! language-model family: one CUDA build, one device, one memory budget. The
//! first family is FLUX.2 [klein], a rectified-flow transformer conditioned on
//! a Qwen3 prompt encoder, decoded by a KL autoencoder.
//!
//! Every weight is read from the checkpoint's own safetensors files and made
//! resident once; the optional 8-bit format is produced from those files at
//! load, deterministically, so the files' hashes identify what runs.
//!
//! On a host with GPU hardware the pipeline refuses to load unless a GPU
//! backend is built in and initialises: it never falls back to the CPU.

pub mod error;
pub mod flux2;
pub mod ggml;
pub mod pipeline;
pub mod qwen3;
pub mod safetensors;
pub mod schedule;
pub mod vae;

pub use error::{Error, Result};
pub use pipeline::{
    CheckpointFiles, Flux2Klein, Image, LoadOptions, MAX_REFERENCE_PIXELS, Precision, Request, RgbImage, Timings,
};
