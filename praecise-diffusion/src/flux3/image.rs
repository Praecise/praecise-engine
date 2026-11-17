//! FLUX 3 Image Klein: text-to-image, and editing from up to
//! [`MAX_REFERENCES`] reference images, on the FLUX 3 transformer with one
//! image stream and one reference stream.
//!
//! A checkpoint is the release's own layout: the transformer as one
//! `model.safetensors` at the root (or `variants/<name>/model.safetensors`,
//! in float8 or NVFP4), the FLUX.2 autoencoder under `vae/`, and the
//! Qwen3-VL text encoder with its tokenizer under `text_encoder/`.
//!
//! The step-distilled model is a flow map (it has `ref_time_in`): each of
//! its steps, 16 in the release, also gives the transformer the time it
//! jumps to, and runs without guidance. The base model runs 50 steps under
//! classifier-free guidance against the empty prompt. Both take Euler steps
//! on a linear grid shifted by [`SCHEDULE_SHIFT`], place the latent grid at
//! half steps on the rotary `h` and `w` axes ([`IMAGE_ROPE_SCALE`]), and
//! give each reference its own negative `l` coordinate, as the reference
//! pipeline does. Prompt upsampling, which the release recommends, is the
//! caller's: the prompt is used as given.

use std::time::Instant;

use super::sampling::euler_times;
use super::text_encoder::{CONTEXT_LAYERS, TextEncoder};
use super::{Flux3Transformer, StreamInput, TextInput};
use crate::error::{Error, Result};
use crate::ggml::Backend;
use crate::pipeline::{Autoencoder, CheckpointFiles, Image, LoadOptions, Request, Timings, check_reference_size};
use crate::safetensors::SafeTensors;
use crate::schedule;

/// Shift of the Euler time grid.
pub const SCHEDULE_SHIFT: f32 = 6.93;
/// Scale of the latent grid's `h` and `w` rotary positions.
pub const IMAGE_ROPE_SCALE: f32 = 0.5;
/// Rotary position of every text padding token in the text encoder.
pub const TEXT_PADDING_POSITION: i32 = 1;
/// Most reference images per request.
pub const MAX_REFERENCES: usize = 10;
/// Largest reference image, in pixels.
pub const MAX_REFERENCE_PIXELS: u32 = 4 * 1024 * 1024;
/// The release's step count for the step-distilled model.
pub const DISTILLED_STEPS: u32 = 16;
/// The release's step count and guidance scale for the base model.
pub const BASE_STEPS: u32 = 50;
/// See [`BASE_STEPS`].
pub const BASE_GUIDANCE: f32 = 3.5;

/// A loaded FLUX 3 Image Klein pipeline.
pub struct Flux3Image {
    backend: Backend,
    dit: Flux3Transformer,
    te: TextEncoder,
    ae: Autoencoder,
}

impl std::fmt::Debug for Flux3Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flux3Image")
            .field("device", &self.dit.device())
            .field("distilled", &self.is_distilled())
            .field("resident_bytes", &self.resident_bytes())
            .finish_non_exhaustive()
    }
}

impl Flux3Image {
    /// Load a checkpoint, with the transformer of `variant`
    /// (`variants/<variant>/model.safetensors`) when given. Float8 and NVFP4
    /// transformers are widened to the load precision.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named; a checkpoint
    /// whose transformer is not an image model, or whose parts disagree.
    pub fn load(files: &CheckpointFiles, variant: Option<&str>, opts: LoadOptions) -> Result<Self> {
        let dit_path = match variant {
            Some(v) => files.root.join("variants").join(v).join("model.safetensors"),
            None => files.root.join("model.safetensors"),
        };
        let st = SafeTensors::open(&[dit_path])?.dequantize_nvfp4()?.dequantize_scaled_f8()?;
        let dit = Flux3Transformer::from_safetensors(&st, opts)?.with_spatial_rope_scale(IMAGE_ROPE_SCALE);
        drop(st);
        let cfg = dit.config();
        let names: Vec<&str> = cfg.streams.iter().map(|(m, _)| m.as_str()).collect();
        if names != ["image", "image_cond"] || cfg.vec_in_dim.is_some() || cfg.depth_late_blocks > 0 {
            return Err(Error::Config(format!("transformer streams {names:?} are not an image model's")));
        }
        if let Ok(package) = files.json("config.json") {
            let flow_map = package.pointer("/dit/flow_map").and_then(serde_json::Value::as_bool).unwrap_or(false);
            if flow_map != cfg.flow_map {
                return Err(Error::Config("config.json and the transformer disagree on the flow map".into()));
            }
        }
        let te = TextEncoder::load(&files.root.join("text_encoder"), &CONTEXT_LAYERS, opts)?.with_padding_position(TEXT_PADDING_POSITION);
        if te.context_width() as u64 != cfg.context_in_dim {
            return Err(Error::Config(format!(
                "transformer expects {}-wide text context, the encoder gives {}",
                cfg.context_in_dim,
                te.context_width()
            )));
        }
        let backend = opts.backend()?;
        let ae = Autoencoder::load(&backend, files, Autoencoder::config(files)?)?;
        let latent = u64::from(ae.cfg.latent_channels) * 4;
        if cfg.streams.iter().any(|(_, c)| *c != latent) {
            return Err(Error::Config("transformer and autoencoder latent widths disagree".into()));
        }
        Ok(Self { backend, dit, te, ae })
    }

    /// The step-distilled model (a flow map), which runs without guidance.
    #[must_use]
    pub fn is_distilled(&self) -> bool {
        self.dit.config().flow_map
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.dit.resident_bytes() + self.te.resident_bytes() + self.ae.w.bytes()
    }

    /// Generate one image.
    ///
    /// # Errors
    /// [`Error::Request`] for sizes, references or step counts the model
    /// cannot serve; backend failures otherwise.
    pub fn generate(&self, req: &Request) -> Result<Image> {
        let cell = self.ae.cell();
        let side = |w: u32, h: u32| w > 0 && h > 0 && w % cell == 0 && h % cell == 0;
        if !side(req.width, req.height) {
            return Err(Error::Request(format!("width and height must be positive multiples of {cell}")));
        }
        if req.steps == 0 {
            return Err(Error::Request("steps must be at least 1".into()));
        }
        if req.references.len() > MAX_REFERENCES {
            return Err(Error::Request(format!("at most {MAX_REFERENCES} reference images")));
        }
        for (i, r) in req.references.iter().enumerate() {
            if !side(r.width, r.height) {
                return Err(Error::Request(format!("reference {i}: sides must be positive multiples of {cell}")));
            }
            check_reference_size(i, r, MAX_REFERENCE_PIXELS)?;
        }
        let n = (req.width / cell) as usize * (req.height / cell) as usize * self.ae.cfg.latent_channels as usize * 4;
        self.run(req, schedule::gaussian(req.seed, n))
    }

    /// The denoising pipeline from a given starting latent, laid out
    /// `[tokens][channels]` with tokens in row-major order.
    fn run(&self, req: &Request, noise: Vec<f32>) -> Result<Image> {
        let cell = self.ae.cell();
        let distilled = self.is_distilled();
        let use_cfg = !distilled && req.guidance_scale > 1.0;
        let t0 = Instant::now();
        let cond = self.te.encode(&req.prompt, None)?;
        let uncond = if use_cfg { Some(self.te.encode("", None)?) } else { None };
        let (mut ref_tokens, mut ref_ids) = (Vec::new(), Vec::new());
        for (k, r) in req.references.iter().enumerate() {
            ref_tokens.extend(self.ae.encode(&self.backend, r)?);
            ref_ids.extend(grid_ids((r.height / cell) as usize, (r.width / cell) as usize, -(1 + k as i32)));
        }
        let encode_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let (gh, gw) = ((req.height / cell) as usize, (req.width / cell) as usize);
        let img_ids = grid_ids(gh, gw, 0);
        let ref_t = vec![0f32; ref_ids.len()];
        let mut x = noise;
        let mut evaluations = 0u32;
        let times = euler_times(req.steps as usize, SCHEDULE_SHIFT);
        for w in times.windows(2) {
            let (t, next) = (w[0], w[1]);
            let img_t = vec![t; img_ids.len()];
            let mut evaluate = |(ctx, n): &(Vec<f32>, usize)| -> Result<Vec<f32>> {
                let ids = text_ids(*n);
                let zeros = vec![0f32; *n];
                let mut streams = vec![StreamInput { name: "image", tokens: &x, ids: &img_ids, timesteps: &img_t }];
                if !ref_ids.is_empty() {
                    streams.push(StreamInput { name: "image_cond", tokens: &ref_tokens, ids: &ref_ids, timesteps: &ref_t });
                }
                let text = TextInput { tokens: ctx, ids: &ids, timesteps: &zeros };
                evaluations += 1;
                Ok(self.dit.forward_to(text, None, &streams, distilled.then_some(next))?.swap_remove(0))
            };
            let mut v = evaluate(&cond)?;
            if let Some(u) = &uncond {
                let vu = evaluate(u)?;
                for (a, b) in v.iter_mut().zip(vu) {
                    *a = b + req.guidance_scale * (*a - b);
                }
            }
            let dt = next - t;
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += dt * vi;
            }
        }
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let rgb = self.ae.decode(&self.backend, &x, gh, gw)?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        Ok(Image {
            width: req.width,
            height: req.height,
            rgb,
            seed: req.seed,
            evaluations,
            timings: Timings { encode_ms, denoise_ms, decode_ms },
        })
    }
}

/// Positions `(0, y, x, l)` of a `gh` by `gw` token grid in row-major order.
pub(crate) fn grid_ids(gh: usize, gw: usize, l: i32) -> Vec<[i32; 4]> {
    (0..gh).flat_map(|y| (0..gw).map(move |x| [0, y as i32, x as i32, l])).collect()
}

/// Positions `(0, 0, 0, i)` of `n` text tokens.
pub(crate) fn text_ids(n: usize) -> Vec<[i32; 4]> {
    (0..n as i32).map(|i| [0, 0, 0, i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_take_their_own_l_coordinate_and_text_runs_along_l() {
        assert_eq!(grid_ids(2, 3, -2), vec![[0, 0, 0, -2], [0, 0, 1, -2], [0, 0, 2, -2], [0, 1, 0, -2], [0, 1, 1, -2], [0, 1, 2, -2]]);
        assert_eq!(text_ids(3), vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]);
    }
}

#[cfg(test)]
mod parity;
