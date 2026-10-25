//! Image encoders on ONNX Runtime.
//!
//! # Scope
//!
//! Foundation vision encoders in 2026 (CLIP family, SigLIP / SigLIP2,
//! DINOv2, TIPS) all share a similar interface: a preprocessed image tensor
//! `[batch=1, 3, H, W]` (NCHW, normalized) goes in, a fixed-dim embedding
//! vector comes out. CLIP/SigLIP also expose a paired text encoder for
//! image-text similarity.
//!
//! This runtime focuses on the **image side** — image embedding and dense
//! features. Text-side encoders for CLIP/SigLIP are handled by the LLM
//! tokenizer + model runtime since their text towers are transformer
//! decoders that load fine through llama-cpp / standard chat infrastructure.
//! For the rare case where a custom ONNX text tower is required, callers
//! can register a second `GenericImageEncoder` with text input shape and
//! call it manually — but the common path is image-only.
//!
//! # Preprocessing
//!
//! Standard ImageNet/CLIP normalization is applied:
//!   - Decode (PNG/JPEG/WebP via the `image` crate)
//!   - Resize to `(input_size, input_size)` with Lanczos3 filter
//!   - Convert to f32 in `[0, 1]`
//!   - Subtract per-channel mean, divide by per-channel std
//!   - Reshape NHWC → NCHW
//!
//! Default mean/std are CLIP-style (`[0.48145466, 0.4578275, 0.40821073]`,
//! `[0.26862954, 0.26130258, 0.27577711]`) but configurable for
//! ImageNet/SigLIP/DINOv2 variants at registration time.
//!
//! # Threading
//!
//! Models are `Send + Sync`. ORT sessions are not safe to call
//! concurrently, so each session lives behind a `parking_lot::Mutex`.
//! Inference is synchronous; async callers run it on a blocking thread.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Read an encoded image's pixel dimensions without decoding its pixels.
///
/// Billing an image call needs the geometry — a model's
/// image tokenization turns width and height into a token count — but not the pixels. `into_dimensions` stops
/// after the header, so a 4K JPEG costs a few hundred bytes of parsing
/// instead of a full RGB decode.
///
/// Returns `None` for bytes no supported decoder recognizes; the caller
/// bills such a call at its per-request floor rather than rejecting it,
/// since the encoder itself is what decides whether the bytes are usable.
pub fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Per-channel mean and std used to normalize pixel values before feeding
/// the encoder. Channels are RGB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageNormalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageNormalization {
    /// CLIP / OpenCLIP standard normalization.
    pub const CLIP: Self = Self {
        mean: [0.48145466, 0.4578275, 0.40821073],
        std: [0.26862954, 0.261_302_6, 0.275_777_1],
    };

    /// ImageNet standard normalization (used by DINOv2, many ResNets).
    pub const IMAGENET: Self = Self {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };

    /// SigLIP / SigLIP2 normalization (symmetric around 0.5).
    pub const SIGLIP: Self = Self {
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    };
}

impl Default for ImageNormalization {
    fn default() -> Self {
        Self::CLIP
    }
}

/// Configuration for an image embedding request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageEmbedConfig {
    /// L2-normalize the output embedding. Most retrieval / similarity
    /// pipelines want this on; raw dense features want it off.
    #[serde(default)]
    pub normalize: bool,
}

/// Result of an image embedding call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbedResult {
    /// Embedding vector. Length is the encoder's native embedding dim
    /// (e.g. 512 for CLIP ViT-B/32, 768 for SigLIP-So400M, 1024 for DINOv2-L).
    pub embedding: Vec<f32>,
    /// Embedding dimensionality, mirrors `embedding.len()` for convenience.
    pub dim: usize,
    /// Total inference wall time in milliseconds (decode + preprocess + ORT run).
    pub generation_time_ms: u64,
}

/// Trait for image encoders. Implementations adapt model-specific
/// preprocessing and tensor layout to a common signature.
pub trait ImageEncoder: Send + Sync {
    /// Encode a single image (raw bytes — PNG, JPEG, WebP).
    fn embed(&self, image_bytes: &[u8], config: &ImageEmbedConfig) -> Result<ImageEmbedResult>;

    /// Native input resolution (e.g. 224, 336, 384, 448).
    fn input_size(&self) -> u32;

    /// Native embedding dimension.
    fn embedding_dim(&self) -> usize;
}

mod onnx_backend {
    use super::*;
    use image::imageops::FilterType;
    use ndarray::Array4;
    use ort::session::Session;
    use ort::value::Tensor;
    use std::time::Instant;

    /// Generic ONNX image encoder — fits CLIP, SigLIP/SigLIP2, DINOv2,
    /// TIPS, and similar architectures with `[1, 3, H, W] -> [1, D]` shape.
    ///
    /// `Session::run` requires `&mut self` in ort 2.x, so the session is
    /// wrapped in a `Mutex` to expose a `&self` API. ONNX Runtime sessions
    /// are not safe to call concurrently from multiple threads regardless,
    /// so the mutex matches the underlying contract rather than restricting it.
    pub struct GenericImageEncoder {
        session: parking_lot::Mutex<Session>,
        input_size: u32,
        embedding_dim: usize,
        normalization: ImageNormalization,
        input_name: String,
        output_name: String,
    }

    impl GenericImageEncoder {
        /// Load an ONNX file and inspect its input/output names.
        pub fn from_onnx(
            path: impl AsRef<Path>,
            input_size: u32,
            embedding_dim: usize,
            normalization: ImageNormalization,
        ) -> Result<Self> {
            let session = crate::session::build_onnx_session(path.as_ref(), "model")?;

            let input_name = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .ok_or_else(|| Error::InvalidModel("ONNX model has no inputs".to_string()))?;
            let output_name = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .ok_or_else(|| Error::InvalidModel("ONNX model has no outputs".to_string()))?;

            Ok(Self {
                session: parking_lot::Mutex::new(session),
                input_size,
                embedding_dim,
                normalization,
                input_name,
                output_name,
            })
        }

        /// Decode → resize → normalize → NCHW. Returns a `[1, 3, H, W]`
        /// tensor ready for ORT.
        fn preprocess(&self, image_bytes: &[u8]) -> Result<Array4<f32>> {
            let img = image::load_from_memory(image_bytes)
                .map_err(|e| Error::InvalidModel(format!("image decode: {}", e)))?
                .to_rgb8();

            let size = self.input_size;
            let resized = image::imageops::resize(&img, size, size, FilterType::Lanczos3);

            let h = size as usize;
            let w = size as usize;
            let mut arr = Array4::<f32>::zeros((1, 3, h, w));
            let mean = self.normalization.mean;
            let std = self.normalization.std;

            for y in 0..h {
                for x in 0..w {
                    let px = resized.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let v = (px[c] as f32 / 255.0 - mean[c]) / std[c];
                        arr[[0, c, y, x]] = v;
                    }
                }
            }

            Ok(arr)
        }
    }

    impl ImageEncoder for GenericImageEncoder {
        fn embed(&self, image_bytes: &[u8], config: &ImageEmbedConfig) -> Result<ImageEmbedResult> {
            if image_bytes.is_empty() {
                return Err(Error::InvalidModel("image bytes are empty".to_string()));
            }

            let start = Instant::now();
            let arr = self.preprocess(image_bytes)?;

            let input_tensor = Tensor::from_array(arr)
                .map_err(|e| Error::InferenceError(format!("ORT tensor: {}", e)))?;

            // Lock the session for the duration of run() — ORT sessions are
            // not safe to call concurrently. Extract the result into owned
            // memory before releasing the lock.
            let (dims, mut raw): (Vec<i64>, Vec<f32>) = {
                let mut session = self.session.lock();
                let outputs = session
                    .run(ort::inputs![self.input_name.as_str() => input_tensor])
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;

                let out_value = outputs.get(self.output_name.as_str()).ok_or_else(|| {
                    Error::InferenceError(format!("missing output '{}'", self.output_name))
                })?;

                let (shape, data) = out_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract: {}", e)))?;
                (shape.iter().copied().collect(), data.to_vec())
            };

            // Expect [1, D] (pooled embedding) — accept [1, 1, D] too as
            // some exports add a singleton sequence axis.
            let embedding: Vec<f32> = match dims.as_slice() {
                [1, d] => {
                    let d = *d as usize;
                    raw.truncate(d);
                    raw
                }
                [1, 1, d] => {
                    let d = *d as usize;
                    raw.truncate(d);
                    raw
                }
                other => {
                    return Err(Error::InferenceError(format!(
                        "unexpected output shape {:?}, expected [1, D] or [1, 1, D]",
                        other
                    )));
                }
            };

            let mut embedding = embedding;
            if config.normalize {
                let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in embedding.iter_mut() {
                        *v /= norm;
                    }
                }
            }

            let dim = embedding.len();
            Ok(ImageEmbedResult {
                embedding,
                dim,
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }

        fn input_size(&self) -> u32 {
            self.input_size
        }

        fn embedding_dim(&self) -> usize {
            self.embedding_dim
        }
    }
}

pub use onnx_backend::GenericImageEncoder;

/// Cosine similarity between two equal-length embeddings. Returns 0.0 if
/// either vector is empty or has zero norm. Range is `[-1.0, 1.0]`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_config_default() {
        let c = ImageEmbedConfig::default();
        assert!(!c.normalize);
    }

    #[test]
    fn embed_result_serializes() {
        let r = ImageEmbedResult {
            embedding: vec![0.1, 0.2, 0.3],
            dim: 3,
            generation_time_ms: 7,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"embedding\""));
        assert!(json.contains("\"dim\":3"));
        let back: ImageEmbedResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn normalization_constants() {
        let clip = ImageNormalization::CLIP;
        assert!((clip.mean[0] - 0.48145466).abs() < 1e-6);
        let inet = ImageNormalization::IMAGENET;
        assert!((inet.std[0] - 0.229).abs() < 1e-6);
        let sig = ImageNormalization::SIGLIP;
        assert_eq!(sig.mean, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn cosine_similarity_basics() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        // mismatched lengths
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        // empty
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        // zero vector
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
