//! Perception on Praecise Engine: video in, per-frame object tracks out.
//!
//! [`pipeline::track_video`] decodes a video in process (praecise-codec),
//! runs an ONNX detector on every frame (praecise-onnx: RF-DETR or D-FINE,
//! on the GPU's execution provider), and follows each object across frames
//! with ByteTrack ([`tracker`]). Each frame's tracks carry a stable id, a
//! class, a box, a score and a velocity, and with a ground-plane homography
//! a position and velocity in metres. [`nvframe`] writes them as NVIDIA
//! Metropolis `nv.Frame` messages, the input of Metropolis analytics and of
//! Halos Safety Core.
//!
//! The `praecise-track` binary runs the pipeline on a file.

// numeric code: pixel and frame counts move between integer and float types
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless)]

pub mod assignment;
pub mod kalman;
pub mod nvframe;
pub mod pipeline;
pub mod tracker;

/// Errors from the pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The video could not be read or decoded.
    #[error("video: {0}")]
    Video(String),
    /// The detector failed.
    #[error("detector: {0}")]
    Detector(String),
    /// Writing output failed.
    #[error("output: {0}")]
    Output(String),
}

/// Result type for the pipeline.
pub type Result<T> = std::result::Result<T, Error>;
