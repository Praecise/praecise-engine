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

/// The release's tensor name of a diffusers `Flux3Transformer2DModel` tensor
/// (the inverse of diffusers' `convert_flux3_transformer_checkpoint_to_diffusers`).
#[must_use]
pub fn native_name(key: &str) -> String {
    const EXACT: [(&str, &str); 6] = [
        ("timestep_embedder.linear_1.weight", "time_in.in_layer.weight"),
        ("timestep_embedder.linear_2.weight", "time_in.out_layer.weight"),
        ("target_timestep_embedder.linear_1.weight", "ref_time_in.in_layer.weight"),
        ("target_timestep_embedder.linear_2.weight", "ref_time_in.out_layer.weight"),
        ("proj_out.weight", "final_layer.image.linear.weight"),
        ("norm_out.linear.weight", "final_layer.image.adaLN_modulation.1.weight"),
    ];
    const PREFIXES: [(&str, &str); 4] = [
        ("x_embedders.", "emb_in."),
        ("context_embedder.", "txt_in."),
        ("mode_blocks.", "content_mode_blocks."),
        ("single_transformer_blocks.", "single_blocks."),
    ];
    const SUFFIXES: [(&str, &str); 8] = [
        ("attn.to_q.weight", "q_proj.weight"),
        ("attn.to_k.weight", "k_proj.weight"),
        ("attn.to_v.weight", "v_proj.weight"),
        ("attn.to_out.0.weight", "attn_out.weight"),
        ("attn.norm_q.weight", "norm.query_norm.scale"),
        ("attn.norm_k.weight", "norm.key_norm.scale"),
        ("ff.linear_in.weight", "mlp_in.weight"),
        ("ff.linear_out.weight", "mlp_out.weight"),
    ];
    if let Some((_, native)) = EXACT.iter().find(|(d, _)| *d == key) {
        return (*native).to_string();
    }
    let mut name = key.to_string();
    if let Some((from, to)) = PREFIXES.iter().find(|(d, _)| name.starts_with(d)) {
        name = format!("{to}{}", &name[from.len()..]);
    }
    if name.contains("stream_modulations.") {
        name = name.replace(".linear.weight", ".lin.weight");
    }
    if let Some((from, to)) = SUFFIXES.iter().find(|(d, _)| name.ends_with(d)) {
        name = format!("{}{to}", &name[..name.len() - from.len()]);
    }
    name
}

/// The diffusers shards of the transformer under the release's tensor names.
/// The final layer's modulation is stored `[scale, shift]` there and
/// `[shift, scale]` in the release's own file: its halves swap.
fn native_names(st: SafeTensors) -> Result<SafeTensors> {
    let rows = st.get("norm_out.linear.weight").map(|v| v.shape[0]).ok_or_else(|| Error::MissingTensor("norm_out.linear.weight".into()))?;
    let st = st.renamed(|k| Some(native_name(k)))?;
    let name = native_name("norm_out.linear.weight");
    st.regroup_rows(&name.clone(), &[(name, vec![(rows / 2, rows), (0, rows / 2)])])
}

/// The FLUX.2 autoencoder's configuration (diffusers `AutoencoderKLFlux2`), which the release's
/// `ae.safetensors` matches tensor for tensor in shape: 4 levels of 128/256/512/512 channels, two
/// residual blocks each, 32 latent channels patched 2x2 to 128, and batch statistics.
pub const FLUX2_VAE_CONFIG: &str = r#"{"act_fn":"silu","batch_norm_eps":0.0001,"block_out_channels":[128,256,512,512],"in_channels":3,"latent_channels":32,"layers_per_block":2,"mid_block_add_attention":true,"norm_num_groups":32,"out_channels":3,"patch_size":[2,2],"use_post_quant_conv":true,"use_quant_conv":true}"#;

/// The diffusers name of a tensor of the release's `ae.safetensors` (BFL's own autoencoder naming,
/// `flux3/autoencoder.py`). Decoder levels run in the opposite order there: `decoder.up.N` is
/// diffusers' `up_blocks.(3 - N)`.
#[must_use]
pub fn ae_name(key: &str) -> String {
    let parts: Vec<&str> = key.split('.').collect();
    let tail = |from: usize| parts[from..].join(".");
    match parts.as_slice() {
        ["encoder", "quant_conv", ..] => format!("quant_conv.{}", tail(2)),
        ["decoder", "post_quant_conv", ..] => format!("post_quant_conv.{}", tail(2)),
        [side @ ("encoder" | "decoder"), "norm_out", ..] => format!("{side}.conv_norm_out.{}", tail(2)),
        [side @ ("encoder" | "decoder"), "mid", blk, rest @ ..] => {
            let rest = rest.join(".");
            match *blk {
                "block_1" => format!("{side}.mid_block.resnets.0.{rest}"),
                "block_2" => format!("{side}.mid_block.resnets.1.{rest}"),
                _ => format!("{side}.mid_block.attentions.0.{}", attn_tail(&rest)),
            }
        }
        ["encoder", "down", n, "downsample", rest @ ..] => format!("encoder.down_blocks.{n}.downsamplers.0.{}", rest.join(".")),
        ["encoder", "down", n, "block", m, rest @ ..] => format!("encoder.down_blocks.{n}.resnets.{m}.{}", resnet_tail(&rest.join("."))),
        ["decoder", "up", n, rest @ ..] if n.parse::<usize>().is_ok_and(|n| n < 4) => {
            let d = 3 - n.parse::<usize>().unwrap_or(0);
            match rest {
                ["upsample", r @ ..] => format!("decoder.up_blocks.{d}.upsamplers.0.{}", r.join(".")),
                ["block", m, r @ ..] => format!("decoder.up_blocks.{d}.resnets.{m}.{}", resnet_tail(&r.join("."))),
                _ => key.to_string(),
            }
        }
        _ => key.to_string(),
    }
}

fn resnet_tail(rest: &str) -> String {
    rest.replacen("nin_shortcut", "conv_shortcut", 1)
}

fn attn_tail(rest: &str) -> String {
    match rest.split_once('.') {
        Some(("norm", r)) => format!("group_norm.{r}"),
        Some(("q", r)) => format!("to_q.{r}"),
        Some(("k", r)) => format!("to_k.{r}"),
        Some(("v", r)) => format!("to_v.{r}"),
        Some(("proj_out", r)) => format!("to_out.0.{r}"),
        _ => rest.to_string(),
    }
}

/// The release's autoencoder tensors under the diffusers names, the attention projections (1x1
/// convolutions there) viewed as matrices.
fn diffusers_ae(st: SafeTensors) -> Result<SafeTensors> {
    let attn: Vec<(String, u64)> = st
        .names()
        .filter(|n| n.contains(".attn_1.") && n.ends_with(".weight") && !n.contains(".norm."))
        .filter_map(|n| st.get(n).map(|v| (ae_name(n), v.shape[0])))
        .collect();
    let mut st = st.renamed(|k| Some(ae_name(k)))?;
    for (name, c) in attn {
        st = st.reshaped(&name, &[c, c])?;
    }
    Ok(st)
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
        let st = match variant {
            Some(v) => SafeTensors::open(&[files.root.join("variants").join(v).join("model.safetensors")])?,
            None if files.root.join("model.safetensors").is_file() => SafeTensors::open(&[files.root.join("model.safetensors")])?,
            // The diffusers shards of the same weights, under diffusers' names.
            None => native_names(SafeTensors::open(&files.weights("transformer")?)?)?,
        };
        let st = st.dequantize_nvfp4()?.dequantize_scaled_f8()?;
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
        let ae = if files.root.join("vae").is_dir() {
            Autoencoder::load(&backend, files, Autoencoder::config(files)?)?
        } else {
            // The release's own layout: the autoencoder is `ae.safetensors` in BFL's tensor names,
            // the FLUX.2 architecture (every tensor is shape-checked against the config at load).
            let st = SafeTensors::open(&[files.root.join("ae.safetensors")])?;
            Autoencoder::load_from(&backend, diffusers_ae(st)?, Autoencoder::config_from_json(FLUX2_VAE_CONFIG)?)?
        };
        let latent = ae.cfg.latent_channels * 4;
        if cfg.streams.iter().any(|(_, c)| *c != latent) {
            return Err(Error::Config("transformer and autoencoder latent widths disagree".into()));
        }
        Ok(Self { backend, dit, te, ae, recipe: Recipe::default() })
    }

    /// Load whichever transformer the checkpoint directory holds: the
    /// bfloat16 `model.safetensors`, else the first of [`QUANT_VARIANTS`]
    /// present under `variants/` (a quantised entry's directory carries only
    /// its own transformer beside the shared text encoder and autoencoder),
    /// else the diffusers shards under `transformer/`, whose tensors are read
    /// under the release's names.
    ///
    /// # Errors
    /// No transformer in the directory; otherwise as [`Flux3Image::load`].
    pub fn load_release(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        if files.root.join("model.safetensors").is_file() {
            return Self::load(files, None, opts);
        }
        if let Some(v) = QUANT_VARIANTS.iter().find(|v| files.root.join("variants").join(v).join("model.safetensors").is_file()) {
            return Self::load(files, Some(v), opts);
        }
        if files.weights("transformer").is_ok() {
            return Self::load(files, None, opts);
        }
        Err(Error::Weights(format!(
            "{}: no model.safetensors, nor variants/{{{}}}/model.safetensors, nor transformer/*.safetensors",
            files.root.display(),
            QUANT_VARIANTS.join(",")
        )))
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

    /// diffusers' `convert_flux3_transformer_checkpoint_to_diffusers`, the other way.
    fn diffusers_name(native: &str) -> String {
        const EXACT: [(&str, &str); 6] = [
            ("time_in.in_layer.weight", "timestep_embedder.linear_1.weight"),
            ("time_in.out_layer.weight", "timestep_embedder.linear_2.weight"),
            ("ref_time_in.in_layer.weight", "target_timestep_embedder.linear_1.weight"),
            ("ref_time_in.out_layer.weight", "target_timestep_embedder.linear_2.weight"),
            ("final_layer.image.linear.weight", "proj_out.weight"),
            ("final_layer.image.adaLN_modulation.1.weight", "norm_out.linear.weight"),
        ];
        if let Some((_, d)) = EXACT.iter().find(|(n, _)| *n == native) {
            return (*d).to_string();
        }
        let mut k = native.to_string();
        for (from, to) in [("emb_in.", "x_embedders."), ("txt_in.", "context_embedder."), ("content_mode_blocks.", "mode_blocks."), ("single_blocks.", "single_transformer_blocks.")] {
            if k.starts_with(from) {
                k = format!("{to}{}", &k[from.len()..]);
                break;
            }
        }
        k = k.replace(".lin.weight", ".linear.weight");
        for (from, to) in [
            ("q_proj.weight", "attn.to_q.weight"),
            ("k_proj.weight", "attn.to_k.weight"),
            ("v_proj.weight", "attn.to_v.weight"),
            ("attn_out.weight", "attn.to_out.0.weight"),
            ("norm.query_norm.scale", "attn.norm_q.weight"),
            ("norm.key_norm.scale", "attn.norm_k.weight"),
            ("mlp_in.weight", "ff.linear_in.weight"),
            ("mlp_out.weight", "ff.linear_out.weight"),
        ] {
            if k.ends_with(from) {
                k = format!("{}{to}", &k[..k.len() - from.len()]);
                break;
            }
        }
        k
    }

    #[test]
    fn diffusers_tensor_names_map_back_to_the_releases() {
        let mut names = vec![
            "emb_in.image.weight".to_string(),
            "emb_in.image_cond.weight".into(),
            "txt_in.weight".into(),
            "time_in.in_layer.weight".into(),
            "time_in.out_layer.weight".into(),
            "ref_time_in.in_layer.weight".into(),
            "ref_time_in.out_layer.weight".into(),
            "final_layer.image.linear.weight".into(),
            "final_layer.image.adaLN_modulation.1.weight".into(),
        ];
        for kind in ["early", "single"] {
            for m in ["image", "image_cond", "txt"] {
                names.push(format!("{kind}_stream_modulations.{m}.lin.weight"));
            }
        }
        let block = |p: &str, names: &mut Vec<String>| {
            for t in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "attn_out.weight", "mlp_in.weight", "mlp_out.weight", "norm.query_norm.scale", "norm.key_norm.scale"] {
                names.push(format!("{p}.{t}"));
            }
        };
        for m in ["image", "image_cond"] {
            block(&format!("content_mode_blocks.{m}.3"), &mut names);
        }
        block("txt_mode_blocks.4", &mut names);
        block("single_blocks.27", &mut names);
        for n in &names {
            let d = diffusers_name(n);
            assert_ne!(&d, n, "{n} is renamed in diffusers");
            assert_eq!(&native_name(&d), n, "{d}");
        }
        assert_eq!(diffusers_name("txt_mode_blocks.4.q_proj.weight"), "txt_mode_blocks.4.attn.to_q.weight");
        assert_eq!(native_name("txt_mode_blocks.0.ff.linear_out.weight"), "txt_mode_blocks.0.mlp_out.weight");
    }

    #[test]
    fn the_releases_real_autoencoder_names_map_onto_the_diffusers_ones_exactly() {
        let list = include_str!("image/ae_keys.txt");
        let pick = |tag: &str| -> Vec<&str> { list.lines().filter_map(|l| l.strip_prefix(tag)).collect() };
        let (release, diffusers) = (pick("A "), pick("D "));
        assert_eq!((release.len(), diffusers.len()), (251, 251));
        let mut mapped: Vec<String> = release.iter().map(|k| ae_name(k)).collect();
        mapped.sort();
        let mut want: Vec<String> = diffusers.iter().map(|k| (*k).to_string()).collect();
        want.sort();
        assert_eq!(mapped, want);
    }

    #[test]
    fn the_flux2_autoencoder_config_parses() {
        let cfg = Autoencoder::config_from_json(FLUX2_VAE_CONFIG).unwrap();
        assert_eq!(cfg.block_out_channels, [128, 256, 512, 512]);
    }

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
        assert!(err.to_string().contains("variants/{fp8r,nvfp4}/model.safetensors, nor transformer/*.safetensors"), "{err}");
    }

    #[test]
    fn references_take_their_own_l_coordinate_and_text_runs_along_l() {
        assert_eq!(grid_ids(2, 3, -2), vec![[0, 0, 0, -2], [0, 0, 1, -2], [0, 0, 2, -2], [0, 1, 0, -2], [0, 1, 1, -2], [0, 1, 2, -2]]);
        assert_eq!(text_ids(3), vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]);
    }
}

#[cfg(test)]
mod parity;
