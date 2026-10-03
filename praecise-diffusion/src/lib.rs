//! Praecise Engine diffusion family.
//!
//! Iterative-denoiser models run natively on the same ggml backend as the
//! language-model family: one CUDA build, one device, one memory budget. The
//! image family is FLUX.2 [klein], a rectified-flow transformer conditioned on
//! a Qwen3 prompt encoder, decoded by a KL autoencoder. The music family is
//! ACE-Step 1.5, a flow-matching transformer over 1D audio latents conditioned
//! on caption, lyrics and timbre, decoded to a waveform by an Oobleck
//! autoencoder.
//!
//! Every weight is read from the checkpoint's own safetensors files and made
//! resident once; the optional 8-bit format is produced from those files at
//! load, deterministically, so the files' hashes identify what runs.
//!
//! On a host with GPU hardware the pipeline refuses to load unless a GPU
//! backend is built in and initialises: it never falls back to the CPU.

pub mod acestep;
pub mod error;
pub mod flux2;
pub mod ggml;
pub mod music;
pub mod oobleck;
pub mod pipeline;
pub mod qwen3;
pub mod safetensors;
pub mod schedule;
pub mod vae;

pub use error::{Error, Result};
pub use music::{AceStep, Audio, MusicRequest};
pub use pipeline::{
    CheckpointFiles, Flux2Klein, Image, LoadOptions, MAX_REFERENCE_PIXELS, Precision, Request, RgbImage, Timings,
};
