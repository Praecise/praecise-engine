//! Z-Image text-to-image: a single-stream transformer conditioned on a Qwen3
//! prompt encoder's second-to-last hidden states, sampled with a shifted
//! flow-matching Euler schedule, decoded by a 16-channel KL autoencoder.
//!
//! The step-distilled checkpoint runs unguided; with a positive guidance
//! scale the prediction is pushed away from the empty prompt's,
//! `cond + scale * (cond - uncond)`.

use std::time::Instant;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, Image, LoadOptions, Precision, Request, Timings};
use crate::qwen3::{self, Qwen3Config};
use crate::s3dit::{self, S3DitConfig};
use crate::safetensors::SafeTensors;
use crate::schedule;
use crate::vae::{self, VaeConfig};

/// Longest prompt, in tokens (the reference truncates here).
pub const MAX_PROMPT_TOKENS: usize = 512;

/// `n` values from `start` to `end` in single precision, computed as the
/// reference's `linspace` computes them (from each end towards the middle).
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    let half = n / 2;
    (0..n).map(|i| if i < half { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 }).collect()
}

/// Noise levels for `steps` steps, `1` down to `1 / steps` shifted towards
/// the noisy end by `shift`, then a final zero; single precision throughout.
#[must_use]
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let mut s: Vec<f32> = linspace(1.0, 1.0 / steps as f32, steps)
        .into_iter()
        .map(|t| shift * t / (1.0 + (shift - 1.0) * t))
        .collect();
    s.push(0.0);
    s
}

/// A loaded Z-Image pipeline.
pub struct ZImage {
    backend: Backend,
    cfg: S3DitConfig,
    tf: Weights,
    te_cfg: Qwen3Config,
    te_theta: f32,
    te: Weights,
    vae_cfg: VaeConfig,
    vae: Weights,
    scale: f32,
    shift_factor: f32,
    shift: f32,
    tokenizer: tokenizers::Tokenizer,
    exact: bool,
}

impl std::fmt::Debug for ZImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZImage").field("backend", &self.backend).field("resident_bytes", &self.resident_bytes()).finish_non_exhaustive()
    }
}

impl ZImage {
    /// Load a checkpoint in the diffusers layout. Refuses to run on the CPU
    /// when the host has GPU hardware this build cannot drive.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        Self::load_parts(files, opts, true, true)
    }

    /// Load the text side (tokenizer and text encoder), the image side
    /// (transformer and autoencoder), or both; a part left out holds a
    /// placeholder and must not be run.
    pub(crate) fn load_parts(files: &CheckpointFiles, opts: LoadOptions, text: bool, image: bool) -> Result<Self> {
        let index = files.json("model_index.json")?;
        let class = index.get("_class_name").and_then(Value::as_str).unwrap_or_default();
        if class != "ZImagePipeline" {
            return Err(Error::Config(format!("pipeline class {class:?} is not ZImagePipeline")));
        }
        let sched = files.json("scheduler/scheduler_config.json")?;
        if sched.get("use_dynamic_shifting").and_then(Value::as_bool) == Some(true)
            || sched.get("invert_sigmas").and_then(Value::as_bool) == Some(true)
            || sched.get("shift_terminal").is_some_and(|v| !v.is_null())
            || sched.get("stochastic_sampling").and_then(Value::as_bool) == Some(true)
            || ["use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas"]
                .iter()
                .any(|k| sched.get(*k).and_then(Value::as_bool) == Some(true))
        {
            return Err(Error::Config("scheduler: only a fixed-shift flow-matching Euler schedule is implemented".into()));
        }
        let shift = sched.get("shift").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let cfg: S3DitConfig = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let te_cfg: Qwen3Config = parse(files.json("text_encoder/config.json")?, "text encoder config")?;
        let te_theta = te_cfg.theta()? as f32;
        let vae_cfg: VaeConfig = parse(files.json("vae/config.json")?, "vae config")?;
        vae_cfg.validate()?;
        let (Some(scale), Some(shift_factor)) = (vae_cfg.scaling_factor, vae_cfg.shift_factor) else {
            return Err(Error::Config("vae config gives no latent scale and shift".into()));
        };
        if te_cfg.hidden_size != cfg.cap_feat_dim || vae_cfg.latent_channels != cfg.in_channels || vae_cfg.use_post_quant_conv {
            return Err(Error::Config("text encoder, transformer and autoencoder widths disagree".into()));
        }

        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "image backend selected");
        let linear = opts.precision.wtype();
        let placeholder = || Weights::zeros(&backend, &[WeightSpec::new("placeholder", &[1], WType::F32)]);
        let part = |want: bool, dir: &str, specs: &[WeightSpec]| -> Result<Weights> {
            if !want {
                return placeholder();
            }
            let st = SafeTensors::open(&files.weights(dir)?)?;
            Weights::load(&backend, &st, specs)
        };
        let tf = part(image, "transformer", &cfg.weight_specs(linear))?;
        let te = part(text, "text_encoder", &te_cfg.weight_specs(qwen3::Layout::CAUSAL_LM, te_cfg.num_hidden_layers - 1, linear)?)?;
        let vae = part(image, "vae", &vae_cfg.weight_specs())?;
        let tok_path = files.root.join("tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        let exact = opts.precision == Precision::F32;
        Ok(Self { backend, cfg, tf, te_cfg, te_theta, te, vae_cfg, vae, scale: scale as f32, shift_factor: shift_factor as f32, shift, tokenizer, exact })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.tf.bytes() + self.te.bytes() + self.vae.bytes()
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Token ids of the chat-templated prompt.
    pub(crate) fn tokens(&self, prompt: &str) -> Result<Vec<i32>> {
        let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let enc = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().iter().take(MAX_PROMPT_TOKENS).map(|&i| i as i32).collect())
    }

    /// Caption features `[tokens][width]`: the hidden states after the
    /// second-to-last encoder layer.
    pub(crate) fn encode(&self, ids: &[i32]) -> Result<Vec<f32>> {
        let n = ids.len();
        let mut g = Graph::new(&self.backend)?;
        let layer = self.te_cfg.num_hidden_layers - 1;
        let io = qwen3::build(&mut g, &self.te_cfg, &self.te, qwen3::Layout::CAUSAL_LM, n as i64, &[layer], self.te_theta, self.exact);
        g.finish(&[io.out])?;
        g.set_i32(io.tokens, ids);
        let pos: Vec<i32> = (0..n as i32).collect();
        g.set_i32(io.positions, &pos);
        g.set_f16(io.mask, &qwen3::mask(n, n));
        g.compute()?;
        Ok(g.read_f32(io.out))
    }

    /// Patch tokens `[tokens][64]` of latents `[16][H][W]`, row-major.
    fn patches(&self, lat: &[f32], lh: usize, lw: usize) -> Vec<f32> {
        let c = self.cfg.in_channels as usize;
        let (gh, gw) = (lh / 2, lw / 2);
        let mut out = vec![0f32; gh * gw * 4 * c];
        for y in 0..gh {
            for x in 0..gw {
                let tok = y * gw + x;
                for ph in 0..2 {
                    for pw in 0..2 {
                        for ch in 0..c {
                            out[tok * 4 * c + (ph * 2 + pw) * c + ch] = lat[(ch * lh + 2 * y + ph) * lw + 2 * x + pw];
                        }
                    }
                }
            }
        }
        out
    }

    /// Inverse of [`Self::patches`].
    fn unpatch(&self, x: &[f32], lh: usize, lw: usize) -> Vec<f32> {
        let c = self.cfg.in_channels as usize;
        let (gh, gw) = (lh / 2, lw / 2);
        let mut out = vec![0f32; c * lh * lw];
        for y in 0..gh {
            for xx in 0..gw {
                let tok = y * gw + xx;
                for ph in 0..2 {
                    for pw in 0..2 {
                        for ch in 0..c {
                            out[(ch * lh + 2 * y + ph) * lw + 2 * xx + pw] = x[tok * 4 * c + (ph * 2 + pw) * c + ch];
                        }
                    }
                }
            }
        }
        out
    }

    /// Positions of the padded image tokens then the padded caption tokens:
    /// the caption counts from 1 along the first axis, the image sits one
    /// past the padded caption with its row and column, image padding at
    /// the origin.
    fn positions(&self, gh: usize, gw: usize, n_cap: usize) -> Vec<[u32; 3]> {
        let pc = s3dit::padded(n_cap);
        let n_img = gh * gw;
        let mut p = Vec::with_capacity(s3dit::padded(n_img) + pc);
        for y in 0..gh {
            for x in 0..gw {
                p.push([pc as u32 + 1, y as u32, x as u32]);
            }
        }
        p.resize(s3dit::padded(n_img), [0, 0, 0]);
        p.extend((0..pc).map(|i| [i as u32 + 1, 0, 0]));
        p
    }

    /// The sampling loop from `noise` `[16][H][W]` given caption features
    /// (and the empty prompt's, under guidance).
    pub(crate) fn denoise(&self, cond: &[f32], uncond: &[f32], noise: Vec<f32>, (lh, lw): (usize, usize), steps: usize, guidance: f32) -> Result<(Vec<f32>, u32)> {
        let (gh, gw) = (lh / 2, lw / 2);
        let width = self.cfg.cap_feat_dim as usize;
        let use_cfg = guidance > 0.0;
        let mut passes = Vec::new();
        for feats in if use_cfg { vec![cond, uncond] } else { vec![cond] } {
            let n_cap = feats.len() / width;
            let mut g = Graph::new(&self.backend)?;
            let io = s3dit::build(&mut g, &self.cfg, &self.tf, (gh * gw) as i64, n_cap as i64, self.exact);
            g.finish(&[io.out])?;
            let (cos, sin) = self.cfg.rotary_tables(&self.positions(gh, gw, n_cap));
            passes.push((g, io, feats, cos, sin));
        }
        let sig = sigmas(steps, self.shift);
        let mut x = noise;
        let mut evaluations = 0u32;
        for i in 0..steps {
            let t = sig[i] * 1000.0;
            let tt = (1000.0 - t) / 1000.0;
            let time = S3DitConfig::time_features(tt * self.cfg.t_scale as f32);
            let patches = self.patches(&x, lh, lw);
            let mut preds = Vec::with_capacity(passes.len());
            for (g, io, feats, cos, sin) in &passes {
                g.set_f32(io.patches, &patches);
                g.set_f32(io.caption, feats);
                g.set_f32(io.time, &time);
                g.set_f32(io.cos, cos);
                g.set_f32(io.sin, sin);
                g.compute()?;
                evaluations += 1;
                preds.push(self.unpatch(&g.read_f32(io.out), lh, lw));
            }
            let pred: Vec<f32> = if use_cfg {
                preds[0].iter().zip(&preds[1]).map(|(c, u)| c + guidance * (c - u)).collect()
            } else {
                preds.pop().expect("one pass")
            };
            let dt = sig[i + 1] - sig[i];
            for (xi, p) in x.iter_mut().zip(&pred) {
                *xi += dt * -p;
            }
        }
        Ok((x, evaluations))
    }

    /// Decode latents `[16][H][W]` to pixels `[3][H*8][W*8]` in `[-1, 1]`
    /// (unclamped).
    pub(crate) fn decode(&self, lat: &[f32], (lh, lw): (usize, usize)) -> Result<Vec<f32>> {
        let z: Vec<f32> = lat.iter().map(|v| v / self.scale + self.shift_factor).collect();
        let mut g = Graph::new(&self.backend)?;
        let io = vae::build_decoder(&mut g, &self.vae_cfg, &self.vae, lw as i64, lh as i64);
        g.finish(&[io.out])?;
        g.set_f32(io.latents, &z);
        g.compute()?;
        Ok(g.read_f32(io.out))
    }

    fn check(&self, req: &Request) -> Result<()> {
        let cell = 16;
        if req.width == 0 || req.height == 0 || req.width % cell != 0 || req.height % cell != 0 {
            return Err(Error::Request(format!("width and height must be positive multiples of {cell}")));
        }
        let (gh, gw) = ((req.height / cell) as u64, (req.width / cell) as u64);
        if gh > self.cfg.axes_lens[1] || gw > self.cfg.axes_lens[2] {
            return Err(Error::Request("image larger than the rotary tables cover".into()));
        }
        if req.steps == 0 {
            return Err(Error::Request("steps must be at least 1".into()));
        }
        if !req.references.is_empty() {
            return Err(Error::Request("reference images are not supported by this model".into()));
        }
        Ok(())
    }

    /// Generate one image.
    ///
    /// # Errors
    /// [`Error::Request`] for dimensions or step counts the model cannot
    /// serve; backend failures otherwise.
    pub fn generate(&mut self, req: &Request) -> Result<Image> {
        self.check(req)?;
        let (lh, lw) = ((req.height / 8) as usize, (req.width / 8) as usize);
        let noise = schedule::gaussian(req.seed, self.cfg.in_channels as usize * lh * lw);
        self.generate_from(req, noise)
    }

    /// [`Self::generate`] from given starting noise `[16][H/8][W/8]`.
    pub(crate) fn generate_from(&mut self, req: &Request, noise: Vec<f32>) -> Result<Image> {
        self.check(req)?;
        let shape = ((req.height / 8) as usize, (req.width / 8) as usize);
        let t0 = Instant::now();
        let cond = self.encode(&self.tokens(&req.prompt)?)?;
        let uncond = if req.guidance_scale > 0.0 { self.encode(&self.tokens("")?)? } else { Vec::new() };
        let encode_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let (lat, evaluations) = self.denoise(&cond, &uncond, noise, shape, req.steps as usize, req.guidance_scale)?;
        let denoise_ms = t1.elapsed().as_millis() as u64;
        let t2 = Instant::now();
        let px = self.decode(&lat, shape)?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        let (w, h) = (req.width as usize, req.height as usize);
        let mut rgb = vec![0u8; w * h * 3];
        for c in 0..3 {
            for i in 0..w * h {
                let v = px[c * w * h + i];
                rgb[i * 3 + c] = ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        Ok(Image { width: req.width, height: req.height, rgb, seed: req.seed, evaluations, timings: Timings { encode_ms, denoise_ms, decode_ms } })
    }
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_meets_in_the_middle_like_the_reference() {
        let s = linspace(1.0, 0.125, 8);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[7], 0.125);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn the_schedule_is_shifted_and_ends_at_zero() {
        let s = sigmas(4, 3.0);
        assert_eq!(s.len(), 5);
        assert_eq!(s[0], 1.0);
        assert!((s[3] - 0.5).abs() < 1e-6, "{}", s[3]);
        assert_eq!(s[4], 0.0);
    }
}
