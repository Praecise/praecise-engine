//! Praecise Engine diffusion family.
//!
//! Iterative-denoiser models run natively on the same ggml backend as the
//! language-model family: one CUDA build, one device, one memory budget. The
//! image family is FLUX.2 [klein], a rectified-flow transformer conditioned on
//! a Qwen3 prompt encoder, decoded by a KL autoencoder. The music family is
//! ACE-Step 1.5, a flow-matching transformer over 1D audio latents conditioned
//! on caption, lyrics and timbre, decoded to a waveform by an Oobleck
//! autoencoder. The video family is Cosmos3, a joint text-and-video
//! transformer over patchified 3D latents sampled with UniPC, decoded frame by
//! frame by a causal video autoencoder. The LTX-2.3 audio-video transformer
//! denoises video and audio latents together.
//!
//! Every weight is read from the checkpoint's own safetensors files and made
//! resident once; the optional 8-bit format is produced from those files at
//! load, deterministically, so the files' hashes identify what runs.
//!
//! On a host with GPU hardware the pipeline refuses to load unless a GPU
//! backend is built in and initialises: it never falls back to the CPU.

pub mod acestep;
pub mod cosmos3;
pub mod error;
pub mod flux2;
pub mod flux3;
pub mod gemma3;
pub mod ggml;
pub mod ltx2;
pub mod matrix_game;
pub mod minimax_h3;
pub mod music;
pub mod oobleck;
pub mod pipeline;
pub mod qwen3;
pub mod qwen3_vl;
pub mod qwen_image;
pub mod qwen_image21;
pub mod qwen_vl;
pub mod s3dit;
pub mod safetensors;
pub mod schedule;
pub mod umt5;
pub mod unipc;
pub mod vae;
pub mod video;
pub mod wan;
pub mod wan_dit;
pub mod wan_video;
pub mod world;
pub mod zimage;

pub use error::{Error, Result};
pub use ggml::Device;
pub use matrix_game::MatrixGame;
pub use music::{AceStep, Audio, MusicRequest};
pub use video::{action_resolution_tier, ActionMode, ActionOutput, ActionRequest, Cosmos3, Embodiment, Video, VideoRequest};
pub use world::{SessionConfig, WorldChunk, WorldModel, WorldSession};
pub use zimage::ZImage;
pub use qwen_image::pipeline::QwenImageEdit;
pub use qwen_image21::pipeline::QwenImage21;
pub use qwen_image21::QwenImage21Transformer;
pub use ltx2::pipeline::{Guidance as Ltx2Guidance, Ltx2Output, Ltx2Pipeline, Ltx2Request};
pub use minimax_h3::pipeline::{H3Condition, H3Output, H3Partition, H3Request, MiniMaxH3Pipeline};
pub use pipeline::{
    CheckpointFiles, Flux2Klein, Image, LoadOptions, MAX_REFERENCE_PIXELS, Precision, Request, RgbImage, Timings,
};
