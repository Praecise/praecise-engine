//! Qwen-Image 2.1 text-to-image and multi-reference editing: the prompt and
//! the condition images go through the Qwen3-VL encoder, the condition
//! images are also encoded to latents that take the place of their image
//! tokens in the joint sequence, and the target is denoised with
//! flow-matching Euler steps (classifier-free guidance against an empty
//! prompt when the scale is above 1). The condition prefix is computed once
//! per prompt and reused by every step.

use std::fmt::Write as _;
use std::time::Instant;

use super::vae::QwenImage21Vae;
use super::{QwenImage21Transformer, Segment};
use crate::error::{Error, Result};
use crate::pipeline::{parse, CheckpointFiles, Image, LoadOptions, Request, RgbImage, Timings};
use crate::qwen3_vl::{Qwen3VlEncoder, Vl3Image};
use crate::qwen_image::pipeline::{dimensions, planar, SchedulerConfig};
use crate::schedule;

/// The system turn; its tokens are dropped from the conditioning.
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const USER: &str = "<|im_start|>user\n";
const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n";
const PICTURE: &str = "<|vision_start|><|image_pad|><|vision_end|>";
/// Negative prompt used under guidance (an empty prompt also becomes this).
const NEGATIVE: &str = " ";
/// Latent tokens per image token of the encoder.
const SLOT: usize = 4;
/// Pixel side the default output and the condition images are sized around.
pub const RESOLUTION: f64 = 1024.0;

/// Prompt states after the system turn and which of them are image tokens.
#[derive(Debug, Clone)]
pub(crate) struct Prompt {
    pub(crate) states: Vec<f32>,
    pub(crate) image: Vec<bool>,
}

/// The pipeline.
#[derive(Debug)]
pub struct QwenImage21 {
    tf: QwenImage21Transformer,
    vae: QwenImage21Vae,
    enc: Qwen3VlEncoder,
    tokenizer: tokenizers::Tokenizer,
    sched: SchedulerConfig,
    drop: usize,
    /// Side of the square whose area the condition images are resized to.
    pub resolution: f64,
}

impl QwenImage21 {
    /// Load a checkpoint in the diffusers layout (`transformer/`, `vae/`,
    /// `text_encoder/`, `processor/tokenizer.json`, `scheduler/`).
    ///
    /// # Errors
    /// A missing or malformed component, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let tf = QwenImage21Transformer::load(files, opts)?;
        let vae = QwenImage21Vae::load(files, &tf.backend, opts.precision, true)?;
        let enc = Qwen3VlEncoder::load(files, "text_encoder", opts)?;
        if enc.config().text_config.hidden_size != tf.config().context_in_dim {
            return Err(Error::Config("encoder width differs from the transformer's text width".into()));
        }
        if vae.config().z_dim != tf.config().in_channels || vae.config().z_dim != tf.config().out_dim() {
            return Err(Error::Config("transformer token width is not the latent width".into()));
        }
        if vae.config().scale() * 2 != enc.config().image_unit() {
            return Err(Error::Config("an encoder image token does not cover 2x2 latent tokens".into()));
        }
        let path = files.root.join("processor/tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        let drop = tokenizer.encode(SYSTEM, false).map_err(|e| Error::Tokenizer(e.to_string()))?.get_ids().len();
        let sched = parse(files.json("scheduler/scheduler_config.json")?, "scheduler config")?;
        Ok(Self { tf, vae, enc, tokenizer, sched, drop, resolution: RESOLUTION })
    }

    /// Name of the backend the pipeline runs on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.tf.device()
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.tf.resident_bytes() + self.vae.resident_bytes() + self.enc.resident_bytes()
    }

    /// Output size the reference pipeline picks: a [`RESOLUTION`]-sided
    /// square's area at the last condition image's aspect ratio (sides
    /// multiples of 32), or the square without one.
    #[must_use]
    pub fn default_size(reference: Option<&RgbImage>) -> (u32, u32) {
        reference.map_or((RESOLUTION as u32, RESOLUTION as u32), |r| {
            let (w, h) = dimensions(RESOLUTION * RESOLUTION, f64::from(r.width) / f64::from(r.height));
            (w as u32, h as u32)
        })
    }

    fn check(req: &Request) -> Result<()> {
        for (i, r) in req.references.iter().enumerate() {
            if r.width == 0 || r.height == 0 || r.rgb.len() != (r.width * r.height * 3) as usize {
                return Err(Error::Request(format!("reference {i}: pixel buffer does not match its size")));
            }
        }
        if !req.width.is_multiple_of(32) || !req.height.is_multiple_of(32) || req.width == 0 || req.height == 0 || req.steps == 0 {
            return Err(Error::Request("output sides must be non-zero multiples of 32 and steps positive".into()));
        }
        Ok(())
    }

    /// Prompt states for `prompt` with the condition images shown to the
    /// encoder.
    pub(crate) fn encode(&self, prompt: &str, images: &[Vl3Image]) -> Result<Prompt> {
        let prompt = if prompt.is_empty() { NEGATIVE } else { prompt };
        let mut pictures = String::new();
        for i in 1..=images.len() {
            let sep = if i > 1 { " " } else { "" };
            let _ = write!(pictures, "{sep}<image{i}>{PICTURE}");
        }
        let text = format!("{SYSTEM}{USER}{pictures}{prompt}{ASSISTANT}");
        let ids = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let grids: Vec<(usize, usize)> = images.iter().map(|r| r.grid).collect();
        let tokens = self.enc.expand_placeholders(ids.get_ids(), &grids);
        if tokens.len() <= self.drop {
            return Err(Error::Request("prompt shorter than the template".into()));
        }
        let h = self.enc.forward(&tokens, images)?;
        let d = self.enc.config().text_config.hidden_size as usize;
        let image_id = self.enc.config().image_token_id;
        Ok(Prompt { states: h[self.drop * d..].to_vec(), image: tokens[self.drop..].iter().map(|&t| t == image_id).collect() })
    }

    /// The joint layout and text states of `p`, the condition image blocks
    /// `(rows, cols)` taking the place of their image tokens.
    fn layout(&self, p: &Prompt, blocks: &[(usize, usize)], target: (usize, usize)) -> Result<(Vec<Segment>, Vec<f32>)> {
        let d = self.enc.config().text_config.hidden_size as usize;
        let mut layout = Vec::new();
        let mut text = Vec::new();
        let mut k = 0;
        let mut i = 0;
        while i < p.image.len() {
            let run = p.image[i..].iter().take_while(|&&b| b == p.image[i]).count();
            if p.image[i] {
                let &(rows, cols) = blocks.get(k).ok_or_else(|| Error::Request("more image runs than condition images".into()))?;
                if rows * cols != run * SLOT {
                    return Err(Error::Request(format!("condition image {k}: {run} encoder tokens for a {rows}x{cols} latent")));
                }
                layout.push(Segment::Image { rows, cols });
                k += 1;
            } else {
                layout.push(Segment::Text(run));
                text.extend_from_slice(&p.states[i * d..(i + run) * d]);
            }
            i += run;
        }
        if k != blocks.len() {
            return Err(Error::Request("fewer image runs than condition images".into()));
        }
        layout.push(Segment::Image { rows: target.0, cols: target.1 });
        Ok((layout, text))
    }

    /// Generate from the prompt, editing the references when there are any.
    ///
    /// # Errors
    /// An invalid request or a backend failure.
    pub fn generate(&mut self, req: &Request) -> Result<Image> {
        Self::check(req)?;
        let (z, s) = (self.vae.config().z_dim as usize, self.vae.config().scale());
        let noise = schedule::gaussian(req.seed, z * (req.height as usize / s) * (req.width as usize / s));
        self.generate_from(req, &noise)
    }

    /// [`Self::generate`] from given starting noise `[z][H/s][W/s]`, also
    /// returning the final latent tokens `[tokens][z]`.
    pub(crate) fn generate_with_latents(&mut self, req: &Request, noise: &[f32]) -> Result<(Image, Vec<f32>)> {
        Self::check(req)?;
        let s = self.vae.config().scale();
        let z = self.vae.config().z_dim as usize;
        let (lh, lw) = (req.height as usize / s, req.width as usize / s);
        if noise.len() != z * lh * lw {
            return Err(Error::Request("starting noise does not match the output size".into()));
        }
        let t0 = Instant::now();
        let channels = self.vae.config().in_channels as usize;
        let mut vl = Vec::new();
        let mut cond = Vec::new();
        let mut blocks = Vec::new();
        for r in &req.references {
            let (vw, vh) = dimensions(self.resolution * self.resolution, f64::from(r.width) / f64::from(r.height));
            let px = planar(r, vh, vw);
            vl.push(self.enc.image(&px, (vh, vw))?);
            let mut pixels: Vec<f32> = px.iter().map(|v| v * 2.0 - 1.0).collect();
            pixels.resize(channels * vh * vw, 1.0);
            let lat = self.vae.encode(&self.tf.backend, &pixels, (vh, vw))?;
            cond.extend(tokens(&lat, z, vh / s * (vw / s)));
            blocks.push((vh / s, vw / s));
        }
        let p = self.encode(&req.prompt, &vl)?;
        let (layout, text) = self.layout(&p, &blocks, (lh, lw))?;
        let prefix = self.tf.prefill(&text, &cond, &layout)?;
        let guided = req.guidance_scale > 1.0;
        let uncond = if guided {
            let n = self.encode(NEGATIVE, &vl)?;
            let (layout, text) = self.layout(&n, &blocks, (lh, lw))?;
            Some(self.tf.prefill(&text, &cond, &layout)?)
        } else {
            None
        };
        let encode_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let sigmas = self.sched.sigmas(req.steps as usize, lh * lw);
        let mut x = tokens(noise, z, lh * lw);
        let mut evaluations = 0;
        for i in 0..req.steps as usize {
            let mut v = self.tf.forward(&prefix, &x, sigmas[i])?;
            evaluations += 1;
            if let Some(u) = &uncond {
                let n = self.tf.forward(u, &x, sigmas[i])?;
                evaluations += 1;
                for (c, u) in v.iter_mut().zip(&n) {
                    *c = u + req.guidance_scale * (*c - u);
                }
            }
            let dt = sigmas[i + 1] - sigmas[i];
            for (a, b) in x.iter_mut().zip(&v) {
                *a += dt * b;
            }
        }
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let px = self.vae.decode(&self.tf.backend, &planes(&x, z, lh * lw), (lh, lw))?;
        let (h, w) = (req.height as usize, req.width as usize);
        let mut rgb = vec![0u8; h * w * 3];
        for c in 0..3 {
            for i in 0..h * w {
                rgb[i * 3 + c] = ((px[c * h * w + i] / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        let decode_ms = t2.elapsed().as_millis() as u64;
        let image = Image {
            width: req.width,
            height: req.height,
            rgb,
            seed: req.seed,
            evaluations,
            timings: Timings { encode_ms, denoise_ms, decode_ms },
        };
        Ok((image, x))
    }

    /// [`Self::generate`] from given starting noise `[z][H/s][W/s]`.
    ///
    /// # Errors
    /// An invalid request or a backend failure.
    pub fn generate_from(&mut self, req: &Request, noise: &[f32]) -> Result<Image> {
        self.generate_with_latents(req, noise).map(|r| r.0)
    }
}

/// Latent `[c][n]` to tokens `[n][c]`.
fn tokens(lat: &[f32], c: usize, n: usize) -> Vec<f32> {
    (0..n).flat_map(|i| (0..c).map(move |ch| lat[ch * n + i])).collect()
}

/// Inverse of [`tokens`].
fn planes(tok: &[f32], c: usize, n: usize) -> Vec<f32> {
    (0..c).flat_map(|ch| (0..n).map(move |i| tok[i * c + ch])).collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::pipeline::Precision;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn tokens_round_trip() {
        let lat: Vec<f32> = (0..4 * 6).map(|i| i as f32).collect();
        assert_eq!(planes(&tokens(&lat, 4, 6), 4, 6), lat);
        assert_eq!(tokens(&lat, 4, 6)[..4], [0.0, 6.0, 12.0, 18.0]);
    }

    #[test]
    fn the_default_size_keeps_the_last_reference_aspect() {
        assert_eq!(QwenImage21::default_size(None), (1024, 1024));
        let wide = RgbImage { width: 200, height: 100, rgb: vec![0; 200 * 100 * 3] };
        assert_eq!(QwenImage21::default_size(Some(&wide)), (1440, 736));
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64, max_px: f32) {
        let d = PathBuf::from(std::env::var("PRAECISE_QWEN_IMAGE21_PIPELINE").expect("PRAECISE_QWEN_IMAGE21_PIPELINE names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let mut p = QwenImage21::load(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap();
        let side = m["side"].as_u64().unwrap() as u32;
        p.resolution = f64::from(side);
        for case in m["cases"].as_array().unwrap() {
            let tag = case["tag"].as_str().unwrap();
            let references: Vec<RgbImage> = (0..case["images"].as_u64().unwrap())
                .map(|i| RgbImage { width: side, height: side, rgb: bin(&d, &format!("image_{i}")).iter().map(|&v| v as u8).collect() })
                .collect();
            let req = Request {
                prompt: m["prompt"].as_str().unwrap().into(),
                references,
                width: side,
                height: side,
                steps: m["steps"].as_u64().unwrap() as u32,
                guidance_scale: m["cfg"].as_f64().unwrap() as f32,
                seed: 0,
            };
            let vl: Vec<Vl3Image> =
                req.references.iter().map(|r| p.enc.image(&planar(r, side as usize, side as usize), (side as usize, side as usize)).unwrap()).collect();
            let prompt = p.encode(&req.prompt, &vl).unwrap();
            assert_close(&format!("{tag} prompt states"), &prompt.states, &bin(&d, &format!("{tag}_prompt_embeds")), min_cos, max_rel);
            let mask: Vec<bool> = bin(&d, &format!("{tag}_image_mask")).iter().map(|&v| v > 0.5).collect();
            assert_eq!(prompt.image, mask, "{tag}: image token positions");
            let (img, lat) = p.generate_with_latents(&req, &bin(&d, &format!("{tag}_noise"))).unwrap();
            assert_close(&format!("{tag} final latents"), &lat, &bin(&d, &format!("{tag}_latents")), min_cos, max_rel);
            let reference = bin(&d, &format!("{tag}_decoded"));
            let worst = img.rgb.iter().zip(&reference).map(|(&a, &b)| (f32::from(a) - b).abs()).fold(0f32, f32::max);
            eprintln!("{tag} decoded pixels: worst difference {worst:.2} of 255");
            assert!(worst <= max_px, "{tag}: decoded pixels differ by {worst}");
        }
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen_image21_pipeline_parity_f32() {
        run(Precision::F32, 0.999_999, 1e-4, 1.0);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen_image21_pipeline_parity_bf16() {
        run(Precision::Bf16, 0.9999, 2e-2, 8.0);
    }
}
