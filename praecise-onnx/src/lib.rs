//! ONNX model families of the Praecise Engine.
//!
//! Each family module holds the request and result types, the model trait
//! and the ONNX Runtime implementations for that family:
//!
//! - [`audio`]: speech-to-text (Moonshine, Whisper, Parakeet TDT, Canary).
//! - [`detection`]: NMS-free DETR object detection (RF-DETR, D-FINE).
//! - [`segmentation`]: point and box promptable segmentation (SAM family).
//! - [`text_segmentation`]: text-prompted segmentation (SAM 3).
//! - [`vision`]: image encoders (CLIP, SigLIP, DINO).
//! - [`forecast`]: time-series forecasting (Chronos 2, TimesFM, T0 and
//!   other foundation forecasters).
//! - [`text_embedding`]: text encoders and the request types every embedder
//!   shares.
//!
//! Every session is built through [`session::build_onnx_session`], which
//! registers the execution provider the machine's GPU needs and refuses to
//! load a model it cannot accelerate rather than running it on the CPU.
//! The host declares the GPUs once with [`session::install`].
//! A host that parses model files in a confined process first installs that
//! check with [`session::set_admission`] and runs it there with
//! [`parse::ParseCheck`].

pub mod audio;
pub mod detection;
pub mod error;
pub mod forecast;
pub mod parse;
pub mod segmentation;
pub mod session;
pub mod text_embedding;
pub mod text_segmentation;
pub mod vision;

pub use error::{Error, Result};
