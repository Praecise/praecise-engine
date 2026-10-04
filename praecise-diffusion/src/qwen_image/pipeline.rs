//! Instruction-guided image editing: the prompt and the reference images go
//! through the vision-language encoder, the references are also encoded to
//! latents that ride along the target's tokens, and the target is denoised
//! with flow-matching Euler steps under norm-preserving guidance.

use std::fmt::Write as _;
use std::time::Instant;

use serde::Deserialize;

use super::vae::QwenImageVae;
use super::QwenImageTransformer;
use crate::error::{Error, Result};
use crate::pipeline::{parse, CheckpointFiles, Image, LoadOptions, Request, RgbImage, Timings};
use crate::qwen_vl::{QwenVlEncoder, VlImage};
use crate::schedule;

/// The editing prompt template; its first [`TEMPLATE_PREFIX_TOKENS`] tokens
/// (the system turn) are dropped from the conditioning.
const TEMPLATE: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
const TEMPLATE_PREFIX_TOKENS: usize = 64;
/// Negative prompt used under guidance.
const NEGATIVE: &str = " ";
/// Pixel area the encoder sees each reference at.
pub const CONDITION_AREA: f64 = 384.0 * 384.0;
/// Pixel area each reference is encoded to latents at.
pub const LATENT_AREA: f64 = 1024.0 * 1024.0;

/// The flow-matching schedule's settings.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SchedulerConfig {
    base_image_seq_len: f64,
    max_image_seq_len: f64,
    base_shift: f64,
    max_shift: f64,
    #[serde(default)]
    shift_terminal: Option<f64>,
}

impl SchedulerConfig {
    /// `steps` sigmas from 1 shifted for `tokens` image tokens, then 0.
    pub(crate) fn sigmas(&self, steps: usize, tokens: usize) -> Vec<f32> {
        let m = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len);
        let mu = tokens as f64 * m + (self.base_shift - m * self.base_image_seq_len);
        // The reference builds the linear ramp in float64, then shifts in float32.
        let lin: Vec<f32> = (0..steps)
            .map(|i| if steps == 1 { 1.0 } else { 1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64 } as f32)
            .collect();
        let e = mu.exp() as f32;
        let mut s: Vec<f32> = lin.iter().map(|&x| e / (e + (1.0 / x - 1.0))).collect();
        if let Some(term) = self.shift_terminal {
            let last = *s.last().unwrap_or(&0.0);
            let k = (1.0 - last) / (1.0 - term as f32);
            for v in &mut s {
                *v = 1.0 - (1.0 - *v) / k;
            }
        }
        s.push(0.0);
        s
    }
}

/// Width and height of `area` pixels at aspect `ratio`, multiples of 32.
pub(crate) fn dimensions(area: f64, ratio: f64) -> (usize, usize) {
    let w = (area * ratio).sqrt();
    let h = w / ratio;
    (((w / 32.0).round() * 32.0) as usize, ((h / 32.0).round() * 32.0) as usize)
}

/// Sides rounded to multiples of `unit` the way the encoder's image
/// processor does.
fn encoder_size(h: usize, w: usize, unit: usize) -> (usize, usize) {
    let r = |x: usize| ((x as f64 / unit as f64).round() as usize).max(1) * unit;
    (r(h), r(w))
}

/// Planar `[3][h][w]` pixels in `[0, 1]` from 8-bit RGB, resampled to `h x w`
/// with a separable bicubic filter (identity when the size is unchanged).
pub(crate) fn planar(img: &RgbImage, h: usize, w: usize) -> Vec<f32> {
    let (sh, sw) = (img.height as usize, img.width as usize);
    let src: Vec<f32> = (0..3).flat_map(|c| (0..sh * sw).map(move |i| (c, i))).map(|(c, i)| f32::from(img.rgb[i * 3 + c]) / 255.0).collect();
    if (sh, sw) == (h, w) {
        return src;
    }
    let cubic = |x: f64| {
        let (a, x) = (-0.5, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    // Per output index: source taps and weights, widened when shrinking.
    let taps = |n_in: usize, n_out: usize| -> Vec<Vec<(usize, f64)>> {
        let scale = n_in as f64 / n_out as f64;
        let support = 2.0 * scale.max(1.0);
        (0..n_out)
            .map(|o| {
                let centre = (o as f64 + 0.5) * scale;
                let lo = (centre - support).floor().max(0.0) as usize;
                let hi = ((centre + support).ceil() as usize).min(n_in);
                let mut t: Vec<(usize, f64)> =
                    (lo..hi).map(|i| (i, cubic((i as f64 + 0.5 - centre) / scale.max(1.0)))).collect();
                let sum: f64 = t.iter().map(|x| x.1).sum();
                for x in &mut t {
                    x.1 /= sum;
                }
                t
            })
            .collect()
    };
    let (th, tw) = (taps(sh, h), taps(sw, w));
    let mut out = vec![0f32; 3 * h * w];
    let mut row = vec![0f32; sh * w];
    for c in 0..3 {
        let plane = &src[c * sh * sw..(c + 1) * sh * sw];
        for y in 0..sh {
            for (x, t) in tw.iter().enumerate() {
                row[y * w + x] = t.iter().map(|&(i, k)| f64::from(plane[y * sw + i]) * k).sum::<f64>() as f32;
            }
        }
        for (y, t) in th.iter().enumerate() {
            for x in 0..w {
                let v = t.iter().map(|&(i, k)| f64::from(row[i * w + x]) * k).sum::<f64>();
                out[(c * h + y) * w + x] = (v as f32).clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// Latent `[c][h][w]` to tokens `[(h/2)(w/2)][c*4]` over 2x2 patches.
pub(crate) fn pack(lat: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(lat.len());
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            for ch in 0..c {
                for dy in 0..2 {
                    for dx in 0..2 {
                        out.push(lat[(ch * h + 2 * y + dy) * w + 2 * x + dx]);
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`pack`].
pub(crate) fn unpack(tok: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; tok.len()];
    let mut i = 0;
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            for ch in 0..c {
                for dy in 0..2 {
                    for dx in 0..2 {
                        out[(ch * h + 2 * y + dy) * w + 2 * x + dx] = tok[i];
                        i += 1;
                    }
                }
            }
        }
    }
    out
}

/// The editing pipeline.
#[derive(Debug)]
pub struct QwenImageEdit {
    tf: QwenImageTransformer,
    vae: QwenImageVae,
    enc: QwenVlEncoder,
    tokenizer: tokenizers::Tokenizer,
    sched: SchedulerConfig,
    /// Pixel area references are shown to the encoder at.
    pub condition_area: f64,
    /// Pixel area references are encoded to latents at.
    pub latent_area: f64,
}

impl QwenImageEdit {
    /// Load a checkpoint in the diffusers layout (`transformer/`, `vae/`,
    /// `text_encoder/`, `processor/tokenizer.json`, `scheduler/`).
    ///
    /// # Errors
    /// A missing or malformed component, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let tf = QwenImageTransformer::load(files, opts)?;
        let vae = QwenImageVae::load(files, &tf.backend, opts.precision, true)?;
        let enc = QwenVlEncoder::load(files, "text_encoder", opts)?;
        if enc.config().hidden_size != tf.config().joint_attention_dim {
            return Err(Error::Config("encoder width differs from the transformer's text width".into()));
        }
        if vae.config().z_dim * tf.config().patch_size * tf.config().patch_size != tf.config().in_channels {
            return Err(Error::Config("transformer input width is not a packed latent".into()));
        }
        let path = files.root.join("processor/tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        let sched = parse(files.json("scheduler/scheduler_config.json")?, "scheduler config")?;
        Ok(Self { tf, vae, enc, tokenizer, sched, condition_area: CONDITION_AREA, latent_area: LATENT_AREA })
    }

    /// Name of the backend the pipeline runs on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.tf.backend.name()
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.tf.resident_bytes() + self.vae.resident_bytes() + self.enc.resident_bytes()
    }

    fn check(req: &Request) -> Result<()> {
        if req.references.is_empty() {
            return Err(Error::Request("editing needs at least one reference image".into()));
        }
        for (i, r) in req.references.iter().enumerate() {
            if r.width == 0 || r.height == 0 || r.rgb.len() != (r.width * r.height * 3) as usize {
                return Err(Error::Request(format!("reference {i}: pixel buffer does not match its size")));
            }
        }
        if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) || req.width == 0 || req.height == 0 || req.steps == 0 {
            return Err(Error::Request("output sides must be non-zero multiples of 16 and steps positive".into()));
        }
        Ok(())
    }

    /// Conditioning `[tokens][width]` for `prompt` with the references shown
    /// to the encoder.
    fn encode(&self, prompt: &str, refs: &[VlImage]) -> Result<Vec<f32>> {
        let pictures = (1..=refs.len()).fold(String::new(), |mut s, i| {
            let _ = write!(s, "Picture {i}: <|vision_start|><|image_pad|><|vision_end|>");
            s
        });
        let text = TEMPLATE.replace("{}", &format!("{pictures}{prompt}"));
        let enc = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let grids: Vec<(usize, usize)> = refs.iter().map(|r| r.grid).collect();
        let tokens = self.enc.expand_placeholders(enc.get_ids(), &grids);
        if tokens.len() <= TEMPLATE_PREFIX_TOKENS {
            return Err(Error::Request("prompt shorter than the template".into()));
        }
        let h = self.enc.forward(&tokens, refs)?;
        let d = self.enc.config().hidden_size as usize;
        Ok(h[TEMPLATE_PREFIX_TOKENS * d..].to_vec())
    }

    /// Edit the references by the prompt. `guidance_scale` above 1 enables
    /// guidance against an empty prompt.
    ///
    /// # Errors
    /// An invalid request or a backend failure.
    pub fn generate(&mut self, req: &Request) -> Result<Image> {
        Self::check(req)?;
        let (z, s) = (self.vae.config().z_dim as usize, self.vae.config().scale());
        let noise = schedule::gaussian(req.seed, z * (req.height as usize / s) * (req.width as usize / s));
        self.generate_from(req, &noise)
    }

    /// [`Self::generate`] from given starting noise `[z][H/s][W/s]` (`s` the
    /// autoencoder's scale), also
    /// returning the final packed latent tokens.
    pub(crate) fn generate_with_latents(&mut self, req: &Request, noise: &[f32]) -> Result<(Image, Vec<f32>)> {
        Self::check(req)?;
        let s = self.vae.config().scale();
        let z = self.vae.config().z_dim as usize;
        let (lh, lw) = (req.height as usize / s, req.width as usize / s);
        if noise.len() != z * lh * lw {
            return Err(Error::Request("starting noise does not match the output size".into()));
        }
        let t0 = Instant::now();
        let unit = self.enc.config().image_unit();
        let mut vl = Vec::new();
        let mut ref_tokens = Vec::new();
        let mut grids = vec![(lh / 2, lw / 2)];
        for r in &req.references {
            let ratio = f64::from(r.width) / f64::from(r.height);
            let (cw, ch) = dimensions(self.condition_area, ratio);
            let (eh, ew) = encoder_size(ch, cw, unit);
            vl.push(self.enc.image(&planar(r, eh, ew), (eh, ew))?);
            let (vw, vh) = dimensions(self.latent_area, ratio);
            let px: Vec<f32> = planar(r, vh, vw).iter().map(|v| v * 2.0 - 1.0).collect();
            let lat = self.vae.encode(&self.tf.backend, &px, (vh, vw))?;
            ref_tokens.extend(pack(&lat, z, vh / s, vw / s));
            grids.push((vh / s / 2, vw / s / 2));
        }
        let cond = self.encode(&req.prompt, &vl)?;
        let guided = req.guidance_scale > 1.0;
        let uncond = if guided { self.encode(NEGATIVE, &vl)? } else { Vec::new() };
        let encode_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let target = (lh / 2) * (lw / 2);
        let width = z * 4;
        let sigmas = self.sched.sigmas(req.steps as usize, target);
        let mut x = pack(noise, z, lh, lw);
        let mut evaluations = 0;
        for i in 0..req.steps as usize {
            let mut input = x.clone();
            input.extend_from_slice(&ref_tokens);
            let mut v = self.tf.forward(&input, &cond, &grids, sigmas[i])?;
            v.truncate(target * width);
            evaluations += 1;
            if guided {
                let mut n = self.tf.forward(&input, &uncond, &grids, sigmas[i])?;
                n.truncate(target * width);
                evaluations += 1;
                for (vc, nc) in v.chunks_exact_mut(width).zip(n.chunks_exact(width)) {
                    let comb: Vec<f32> = vc.iter().zip(nc).map(|(c, u)| u + req.guidance_scale * (c - u)).collect();
                    let norm = |a: &[f32]| a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let k = norm(vc) / norm(&comb);
                    for (o, c) in vc.iter_mut().zip(&comb) {
                        *o = c * k;
                    }
                }
            }
            let dt = sigmas[i + 1] - sigmas[i];
            for (a, b) in x.iter_mut().zip(&v) {
                *a += dt * b;
            }
        }
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let px = self.vae.decode(&self.tf.backend, &unpack(&x, z, lh, lw), (lh, lw))?;
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

    /// Output size the reference pipeline picks for the last reference:
    /// a megapixel at its aspect ratio, sides multiples of 16.
    #[must_use]
    pub fn default_size(reference: &RgbImage) -> (u32, u32) {
        let (w, h) = dimensions(LATENT_AREA, f64::from(reference.width) / f64::from(reference.height));
        ((w / 16 * 16) as u32, (h / 16 * 16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::pipeline::Precision;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn sigmas_match_the_reference_schedule() {
        let s = SchedulerConfig { base_image_seq_len: 256.0, max_image_seq_len: 8192.0, base_shift: 0.5, max_shift: 0.9, shift_terminal: Some(0.02) };
        let v = s.sigmas(4, 4096);
        assert_eq!(v.len(), 5);
        assert!((v[0] - 1.0).abs() < 1e-6 && (v[3] - 0.02).abs() < 1e-6 && v[4] == 0.0, "{v:?}");
    }

    #[test]
    fn pack_round_trips() {
        let lat: Vec<f32> = (0..4 * 6 * 8).map(|i| i as f32).collect();
        assert_eq!(unpack(&pack(&lat, 4, 6, 8), 4, 6, 8), lat);
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64, max_px: f32) {
        let d = PathBuf::from(std::env::var("PRAECISE_QWEN_EDIT_PARITY").expect("PRAECISE_QWEN_EDIT_PARITY names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let mut p = QwenImageEdit::load(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap();
        let side = m["side"].as_u64().unwrap() as u32;
        p.condition_area = f64::from(side * side);
        p.latent_area = f64::from(side * side);
        let rgb: Vec<u8> = bin(&d, "image").iter().map(|&v| v as u8).collect();
        let req = Request {
            prompt: m["prompt"].as_str().unwrap().into(),
            references: vec![RgbImage { width: side, height: side, rgb }],
            width: side,
            height: side,
            steps: m["steps"].as_u64().unwrap() as u32,
            guidance_scale: m["cfg"].as_f64().unwrap() as f32,
            seed: 0,
        };
        let refs: Vec<VlImage> = req.references.iter().map(|r| p.enc.image(&planar(r, side as usize, side as usize), (side as usize, side as usize)).unwrap()).collect();
        assert_close("prompt states", &p.encode(&req.prompt, &refs).unwrap(), &bin(&d, "prompt_embeds"), min_cos, max_rel);
        let (img, lat) = p.generate_with_latents(&req, &bin(&d, "noise")).unwrap();
        assert_close("final latents", &lat, &bin(&d, "latents"), min_cos, max_rel);
        let reference = bin(&d, "decoded");
        let worst = img.rgb.iter().zip(&reference).map(|(&a, &b)| (f32::from(a) - b).abs()).fold(0f32, f32::max);
        eprintln!("decoded pixels: worst difference {worst:.2} of 255");
        assert!(worst <= max_px, "decoded pixels differ by {worst}");
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen_image_edit_parity_f32() {
        run(Precision::F32, 0.999_999, 1e-4, 1.0);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen_image_edit_parity_bf16() {
        run(Precision::Bf16, 0.999, 3e-2, 8.0);
    }
}
