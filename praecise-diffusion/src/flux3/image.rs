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
//! on a linear grid shifted by the [`Recipe`]'s shift, place the latent grid at
//! half steps on the rotary `h` and `w` axes ([`IMAGE_ROPE_SCALE`]), and
//! give each reference its own negative `l` coordinate, as the reference
//! pipeline does. Prompt upsampling, which the release recommends, is the
//! caller's: the prompt is used as given. Reference images of any size are
//! prepared as the release does by [`Flux3Image::edit`] ([`prepare`]).
//!
//! The release's own sampling code (BFL's `flux3` repository) and the
//! diffusers modular pipeline disagree on two numbers, the time shift and the
//! base guidance; the [`Recipe`] holds both sets and defaults to BFL's.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "tensor and kernel math: sizes, indices and counts move between usize, i64, u32 and f32 within bounds the tensor shapes fix; a checked conversion at each site would change no value"
)]

use std::time::Instant;

use super::sampling::euler_times;
use super::text_encoder::{CONTEXT_LAYERS, TextEncoder};
use super::{Flux3Transformer, StreamInput, TextInput};
use crate::error::{Error, Result};
use crate::ggml::Backend;
use crate::pipeline::{Autoencoder, CheckpointFiles, Image, LoadOptions, Request, RgbImage, Timings, check_reference_size};
use crate::safetensors::SafeTensors;
use crate::schedule;

/// Shift of the Euler time grid in the diffusers modular pipeline.
pub const SCHEDULE_SHIFT: f32 = 6.93;
/// Shift (`alpha`) of the Euler time grid in the release's own reference code.
pub const REFERENCE_SHIFT: f32 = 16.0;
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
/// The release's step count for the base model.
pub const BASE_STEPS: u32 = 50;
/// The base model's guidance scale in the release's own reference code.
pub const BASE_GUIDANCE: f32 = 4.0;
/// The base model's guidance scale in the diffusers modular pipeline.
pub const DIFFUSERS_BASE_GUIDANCE: f32 = 3.5;
/// Longest prompt the release encodes, chat template included.
pub const MAX_TEXT_TOKENS: usize = 4096;

/// The release's quantised transformers, under `variants/`.
pub const QUANT_VARIANTS: [&str; 2] = ["fp8r", "nvfp4"];

pub mod prepare;

/// The sampling numbers two public implementations of the release give
/// differently. Only a run on the release's weights can say which the model
/// was trained with; the reference code's is the default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recipe {
    /// Shift of the Euler time grid.
    pub shift: f32,
    /// Guidance scale the release recommends for the base model; a hosting
    /// layer's default, `Request::guidance_scale` always decides.
    pub base_guidance: f32,
    /// Feed the transformer timesteps rounded to bfloat16, as the reference
    /// code's bfloat16 timestep tensors do (its sinusoid sees
    /// `bf16(1000 * bf16(t))`), and start from noise rounded to bfloat16.
    pub bf16_inputs: bool,
}

impl Recipe {
    /// BFL's `flux3` reference code.
    pub const REFERENCE: Self = Self { shift: REFERENCE_SHIFT, base_guidance: BASE_GUIDANCE, bf16_inputs: true };
    /// The diffusers modular pipeline.
    pub const DIFFUSERS: Self = Self { shift: SCHEDULE_SHIFT, base_guidance: DIFFUSERS_BASE_GUIDANCE, bf16_inputs: false };
}

impl Default for Recipe {
    fn default() -> Self {
        Self::REFERENCE
    }
}

/// Round to the nearest bfloat16 value.
#[must_use]
pub fn bf16_round(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

/// The value that, scaled by the transformer's 1000, gives the reference
/// code's bfloat16 `1000 * t` for a bfloat16 `t`.
#[must_use]
pub fn reference_time(t: f32) -> f32 {
    bf16_round(bf16_round(t) * 1000.0) / 1000.0
}

/// Where [`Flux3Image::edit`] takes the output size from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSize {
    /// The request's width and height.
    AsRequested,
    /// The first reference's prepared size (1024 x 1024 without one); a
    /// single padded reference is cropped back to its own extent.
    FromReference,
}

/// A loaded FLUX 3 Image Klein pipeline.
pub struct Flux3Image {
    backend: Backend,
    dit: Flux3Transformer,
    te: TextEncoder,
    ae: Autoencoder,
    recipe: Recipe,
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
        let te = TextEncoder::load(&files.root.join("text_encoder"), &CONTEXT_LAYERS, opts)?
            .with_padding_position(TEXT_PADDING_POSITION)
            .with_max_tokens(MAX_TEXT_TOKENS);
        if te.context_width() as u64 != cfg.context_in_dim {
            return Err(Error::Config(format!(
                "transformer expects {}-wide text context, the encoder gives {}",
                cfg.context_in_dim,
                te.context_width()
            )));
        }
        let backend = opts.backend()?;
        let ae = Autoencoder::load(&backend, files, Autoencoder::config(files)?)?;
        let latent = ae.cfg.latent_channels * 4;
        if cfg.streams.iter().any(|(_, c)| *c != latent) {
            return Err(Error::Config("transformer and autoencoder latent widths disagree".into()));
        }
        Ok(Self { backend, dit, te, ae, recipe: Recipe::default() })
    }

    /// Load whichever transformer the checkpoint directory holds: the
    /// bfloat16 `model.safetensors`, else the first of [`QUANT_VARIANTS`]
    /// present under `variants/` (a quantised entry's directory carries only
    /// its own transformer beside the shared text encoder and autoencoder).
    ///
    /// # Errors
    /// No transformer in the directory; otherwise as [`Flux3Image::load`].
    pub fn load_release(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        if files.root.join("model.safetensors").is_file() {
            return Self::load(files, None, opts);
        }
        match QUANT_VARIANTS.iter().find(|v| files.root.join("variants").join(v).join("model.safetensors").is_file()) {
            Some(v) => Self::load(files, Some(v), opts),
            None => Err(Error::Weights(format!(
                "{}: no model.safetensors, nor variants/{{{}}}/model.safetensors",
                files.root.display(),
                QUANT_VARIANTS.join(",")
            ))),
        }
    }

    /// Sample with `recipe` instead of the reference code's numbers.
    #[must_use]
    pub fn with_recipe(mut self, recipe: Recipe) -> Self {
        self.recipe = recipe;
        self
    }

    /// The sampling numbers in use.
    #[must_use]
    pub fn recipe(&self) -> Recipe {
        self.recipe
    }

    /// The step-distilled model (a flow map), which runs without guidance.
    #[must_use]
    pub fn is_distilled(&self) -> bool {
        self.dit.config().flow_map
    }

    /// Name of the backend device the pipeline runs on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
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
        let side = |w: u32, h: u32| w > 0 && h > 0 && w.is_multiple_of(cell) && h.is_multiple_of(cell);
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
        let mut noise = schedule::gaussian(req.seed, n);
        if self.recipe.bf16_inputs {
            for v in &mut noise {
                *v = bf16_round(*v);
            }
        }
        self.run(req, noise)
    }

    /// Generate from references of any size, prepared as the release does
    /// (see [`prepare`]): one reference is kept whole on its padded canvas,
    /// several are each fitted under one megapixel and cropped to the grid.
    /// `req.references` are read as given, not required to sit on the token
    /// grid or under the pixel limit; with [`OutputSize::FromReference`] the
    /// width and height of `req` are replaced by the first reference's, and a
    /// padded single reference is cropped back out of the result.
    ///
    /// # Errors
    /// As [`Flux3Image::generate`], and malformed or too small references.
    pub fn edit(&self, req: &Request, size: OutputSize) -> Result<Image> {
        let (references, fill) = match req.references.as_slice() {
            [] => (Vec::new(), None),
            [one] => {
                let (image, fill) = prepare::prepare_reference(one)?;
                (vec![image], fill)
            }
            many => (many.iter().map(prepare::prepare_multi_reference).collect::<Result<Vec<_>>>()?, None),
        };
        let mut prepared = req.clone();
        let mut crop_to = None;
        if size == OutputSize::FromReference {
            (prepared.width, prepared.height) = prepare::default_size(&references);
            crop_to = fill;
        }
        prepared.references = references;
        let mut image = self.generate(&prepared)?;
        if let Some([l, t, r, b]) = crop_to {
            let whole = RgbImage { width: image.width, height: image.height, rgb: std::mem::take(&mut image.rgb) };
            let c = prepare::crop(&whole, l, t, r - l, b - t);
            (image.width, image.height, image.rgb) = (c.width, c.height, c.rgb);
        }
        Ok(image)
    }

    /// The denoising pipeline from a given starting latent, laid out
    /// `[tokens][channels]` with tokens in row-major order.
    fn run(&self, req: &Request, noise: Vec<f32>) -> Result<Image> {
        let latents = self.denoise(req, noise)?;
        let (gh, gw) = ((req.height / self.ae.cell()) as usize, (req.width / self.ae.cell()) as usize);
        let t2 = Instant::now();
        let rgb = self.ae.decode(&self.backend, &latents.tokens, gh, gw)?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        Ok(Image {
            width: req.width,
            height: req.height,
            rgb,
            seed: req.seed,
            evaluations: latents.evaluations,
            timings: Timings { encode_ms: latents.encode_ms, denoise_ms: latents.denoise_ms, decode_ms },
        })
    }

    /// Prompt and reference encoding and the denoising loop, up to the clean
    /// latent tokens.
    fn denoise(&self, req: &Request, noise: Vec<f32>) -> Result<Denoised> {
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
        let times = euler_times(req.steps as usize, self.recipe.shift);
        let model_time = |t: f32| if self.recipe.bf16_inputs { reference_time(t) } else { t };
        for w in times.windows(2) {
            let (t, next) = (w[0], w[1]);
            let img_t = vec![model_time(t); img_ids.len()];
            let mut evaluate = |(ctx, n): &(Vec<f32>, usize)| -> Result<Vec<f32>> {
                let ids = text_ids(*n);
                let zeros = vec![0f32; *n];
                let mut streams = vec![StreamInput { name: "image", tokens: &x, ids: &img_ids, timesteps: &img_t }];
                if !ref_ids.is_empty() {
                    streams.push(StreamInput { name: "image_cond", tokens: &ref_tokens, ids: &ref_ids, timesteps: &ref_t });
                }
                let text = TextInput { tokens: ctx, ids: &ids, timesteps: &zeros };
                evaluations += 1;
                Ok(self.dit.forward_to(text, None, &streams, distilled.then_some(model_time(next)))?.swap_remove(0))
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
        Ok(Denoised { tokens: x, evaluations, encode_ms, denoise_ms })
    }
}

/// The clean latent tokens of a run and what it cost.
struct Denoised {
    tokens: Vec<f32>,
    evaluations: u32,
    encode_ms: u64,
    denoise_ms: u64,
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
    fn the_default_recipe_is_the_release_reference_code() {
        let r = Recipe::default();
        assert_eq!((r.shift, r.base_guidance, r.bf16_inputs), (16.0, 4.0, true));
        assert_eq!(Recipe::DIFFUSERS.shift, SCHEDULE_SHIFT);
        let grid = euler_times(16, r.shift);
        assert_eq!((grid[0], grid[16]), (1.0, 0.0));
        // alpha*t/(1+(alpha-1)t) at t = 0.5.
        assert!((grid[8] - 16.0 * 0.5 / (1.0 + 15.0 * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn reference_times_are_the_bfloat16_products() {
        assert_eq!(reference_time(1.0), 1.0);
        assert_eq!(reference_time(0.0), 0.0);
        // bf16(0.3) = 0.30078125; 1000x = 300.78 rounds to 300 (bf16 steps of 2 there).
        assert!((reference_time(0.3) - 0.3).abs() < 1e-6);
        // Every value is a bfloat16 number once scaled, within bfloat16's relative error of the input.
        for i in 1..=100 {
            let t = i as f32 / 100.0;
            let scaled = reference_time(t) * 1000.0;
            assert!((bf16_round(scaled) - scaled).abs() <= scaled * 1e-6, "t {t}");
            assert!((reference_time(t) - t).abs() <= t * 0.01, "t {t}");
        }
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
    }

    #[test]
    fn a_directory_without_a_transformer_is_refused_by_name() {
        let dir = std::env::temp_dir().join("praecise-flux3-image-empty");
        std::fs::create_dir_all(&dir).unwrap();
        let err = Flux3Image::load_release(&CheckpointFiles::new(&dir), LoadOptions::default()).unwrap_err();
        assert!(err.to_string().contains("variants/{fp8r,nvfp4}/model.safetensors"), "{err}");
    }

    #[test]
    fn references_take_their_own_l_coordinate_and_text_runs_along_l() {
        assert_eq!(grid_ids(2, 3, -2), vec![[0, 0, 0, -2], [0, 0, 1, -2], [0, 0, 2, -2], [0, 1, 0, -2], [0, 1, 1, -2], [0, 1, 2, -2]]);
        assert_eq!(text_ids(3), vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]);
    }
}

#[cfg(test)]
mod parity;
