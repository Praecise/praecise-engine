//! FLUX.2 [klein] text-to-image: prompt encoding, the denoising loop and the
//! decode, run on one backend with every weight resident once.

use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::flux2::{self, Flux2Config};
use crate::ggml::{Backend, Graph, WType, Weights};
use crate::qwen3::{self, Qwen3Config};
use crate::safetensors::SafeTensors;
use crate::schedule;
use crate::vae::{self, VaeConfig};

/// Maximum prompt length in tokens, template included.
pub const MAX_PROMPT_TOKENS: usize = 512;
/// Hidden-state layers (1-based) whose outputs form the prompt conditioning.
pub const PROMPT_LAYERS: [usize; 3] = [9, 18, 27];
/// Time-axis position step between reference images: the `k`-th reference
/// (1-based) sits at `t = 10 k`, the generated image at `t = 0`.
const REFERENCE_T_STEP: f32 = 10.0;
/// Padding token of the prompt encoder's vocabulary.
const PAD_TOKEN: u32 = 151_643;

/// Numeric format of the transformer and text-encoder linear weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// The checkpoints' own bfloat16.
    #[default]
    Bf16,
    /// 8-bit blocks, quantised deterministically at load.
    Q8_0,
    /// float32 weights and float32 matrix products: the exact reference the
    /// faster formats are measured against.
    F32,
}

impl Precision {
    pub(crate) fn wtype(self) -> WType {
        match self {
            Self::Bf16 => WType::Bf16,
            Self::Q8_0 => WType::Q8_0,
            Self::F32 => WType::F32,
        }
    }
}

/// Load-time options.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LoadOptions {
    /// Weight format for the transformer and the text encoder.
    pub precision: Precision,
    /// CPU threads, used only on a host without GPU hardware.
    pub cpu_threads: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { precision: Precision::Bf16, cpu_threads: std::thread::available_parallelism().map_or(4, usize::from) }
    }
}

/// An 8-bit RGB image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGB, three bytes per pixel.
    pub rgb: Vec<u8>,
}

/// Largest reference image, in pixels; larger ones must be resized first.
pub const MAX_REFERENCE_PIXELS: u32 = 1024 * 1024;

/// A generation request: a prompt, and optionally reference images the
/// output is conditioned on (editing and multi-reference generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Prompt.
    pub prompt: String,
    /// Reference images, each at most [`MAX_REFERENCE_PIXELS`] with sides a
    /// multiple of 16.
    pub references: Vec<RgbImage>,
    /// Output width in pixels, a multiple of 16.
    pub width: u32,
    /// Output height in pixels, a multiple of 16.
    pub height: u32,
    /// Denoising steps.
    pub steps: u32,
    /// Classifier-free guidance scale; ignored by step-distilled checkpoints.
    pub guidance_scale: f32,
    /// Seed for the starting noise.
    pub seed: u64,
}

/// Wall-clock time spent in each stage.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Timings {
    /// Prompt encoding.
    pub encode_ms: u64,
    /// The denoising loop.
    pub denoise_ms: u64,
    /// Latent decode.
    pub decode_ms: u64,
}

/// A generated image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major 8-bit RGB.
    pub rgb: Vec<u8>,
    /// Seed used.
    pub seed: u64,
    /// Transformer evaluations run (steps, doubled under guidance).
    pub evaluations: u32,
    /// Per-stage timings.
    pub timings: Timings,
}

/// Paths of one checkpoint in the diffusers directory layout.
#[derive(Debug, Clone)]
pub struct CheckpointFiles {
    pub(crate) root: PathBuf,
}

impl CheckpointFiles {
    /// A checkpoint rooted at `root` (holding `model_index.json`).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub(crate) fn json(&self, rel: &str) -> Result<Value> {
        let p = self.root.join(rel);
        let bytes = std::fs::read(&p).map_err(|e| Error::Config(format!("{}: {e}", p.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
    }

    pub(crate) fn weights(&self, dir: &str) -> Result<Vec<PathBuf>> {
        let d = self.root.join(dir);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&d)
            .map_err(|e| Error::Weights(format!("{}: {e}", d.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(Error::Weights(format!("no safetensors files in {}", d.display())));
        }
        Ok(files)
    }
}

/// A loaded FLUX.2 [klein] pipeline.
pub struct Flux2Klein {
    backend: Backend,
    dit_cfg: Flux2Config,
    dit: Weights,
    te_cfg: Qwen3Config,
    te_theta: f32,
    te: Weights,
    vae_cfg: VaeConfig,
    vae: Weights,
    bn_mean: Vec<f32>,
    bn_std: Vec<f32>,
    tokenizer: tokenizers::Tokenizer,
    distilled: bool,
}

impl std::fmt::Debug for Flux2Klein {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flux2Klein")
            .field("backend", &self.backend)
            .field("distilled", &self.distilled)
            .field("resident_bytes", &self.resident_bytes())
            .finish_non_exhaustive()
    }
}

pub(crate) fn parse<T: for<'de> Deserialize<'de>>(v: Value, what: &str) -> Result<T> {
    serde_json::from_value(v).map_err(|e| Error::Config(format!("{what}: {e}")))
}

impl Flux2Klein {
    /// Load a checkpoint. Refuses to run on the CPU when the host has GPU
    /// hardware this build cannot drive.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let index = files.json("model_index.json")?;
        let class = index.get("_class_name").and_then(Value::as_str).unwrap_or_default();
        if class != "Flux2KleinPipeline" {
            return Err(Error::Config(format!("pipeline class {class:?} is not Flux2KleinPipeline")));
        }
        let distilled = index.get("is_distilled").and_then(Value::as_bool).unwrap_or(false);
        let sched = files.json("scheduler/scheduler_config.json")?;
        if sched.get("time_shift_type").and_then(Value::as_str) != Some("exponential")
            || sched.get("use_dynamic_shifting").and_then(Value::as_bool) != Some(true)
        {
            return Err(Error::Config("expected an exponential, dynamically shifted flow-match schedule".into()));
        }

        let dit_cfg: Flux2Config = parse(files.json("transformer/config.json")?, "transformer config")?;
        dit_cfg.validate()?;
        let te_cfg: Qwen3Config = parse(files.json("text_encoder/config.json")?, "text encoder config")?;
        let te_theta = te_cfg.theta()? as f32;
        let vae_cfg: VaeConfig = parse(files.json("vae/config.json")?, "vae config")?;
        vae_cfg.validate()?;
        let want_ctx = te_cfg.hidden_size * PROMPT_LAYERS.len() as u64;
        if dit_cfg.joint_attention_dim != want_ctx {
            return Err(Error::Config(format!(
                "transformer expects {}-wide text conditioning, the encoder gives {want_ctx}",
                dit_cfg.joint_attention_dim
            )));
        }
        if dit_cfg.in_channels != vae_cfg.latent_channels * 4 {
            return Err(Error::Config("transformer and autoencoder latent widths disagree".into()));
        }

        let backend = Backend::select(opts.cpu_threads)?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "diffusion backend selected");
        let linear = opts.precision.wtype();

        let dit_files = SafeTensors::open(&files.weights("transformer")?)?;
        let dit = Weights::load(&backend, &dit_files, &dit_cfg.weight_specs(linear))?;
        drop(dit_files);

        let last = *PROMPT_LAYERS.iter().max().expect("non-empty");
        let te_files = SafeTensors::open(&files.weights("text_encoder")?)?;
        let te = Weights::load(&backend, &te_files, &te_cfg.weight_specs(qwen3::Layout::CAUSAL_LM, last, linear)?)?;
        drop(te_files);

        let vae_files = SafeTensors::open(&files.weights("vae")?)?;
        let mut vae_specs = vae_cfg.weight_specs();
        vae_specs.extend(vae_cfg.encoder_weight_specs());
        let vae = Weights::load(&backend, &vae_files, &vae_specs)?;
        let (bn_mean, bn_std) = vae::latent_stats(&vae_files, &vae_cfg)?;
        drop(vae_files);

        let tok_path = files.root.join("tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;

        Ok(Self { backend, dit_cfg, dit, te_cfg, te_theta, te, vae_cfg, vae, bn_mean, bn_std, tokenizer, distilled })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.dit.bytes() + self.te.bytes() + self.vae.bytes()
    }

    /// Backend device name.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Whether the checkpoint is step-distilled (guidance has no effect).
    #[must_use]
    pub fn is_distilled(&self) -> bool {
        self.distilled
    }

    fn tokens(&self, prompt: &str) -> Result<(Vec<i32>, usize)> {
        let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let enc = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        ids.truncate(MAX_PROMPT_TOKENS);
        let real = ids.len();
        ids.resize(MAX_PROMPT_TOKENS, PAD_TOKEN);
        Ok((ids.into_iter().map(|t| t as i32).collect(), real))
    }

    /// Prompt conditioning `[MAX_PROMPT_TOKENS][hidden * layers]`.
    fn encode(&self, prompt: &str) -> Result<Vec<f32>> {
        let (ids, real) = self.tokens(prompt)?;
        let n = ids.len();
        let mut g = Graph::new(&self.backend)?;
        let io = qwen3::build(&mut g, &self.te_cfg, &self.te, qwen3::Layout::CAUSAL_LM, n as i64, &PROMPT_LAYERS, self.te_theta, false);
        g.finish(&[io.out])?;
        g.set_i32(io.tokens, &ids);
        let pos: Vec<i32> = (0..n as i32).collect();
        g.set_i32(io.positions, &pos);
        g.set_f16(io.mask, &qwen3::mask(n, real));
        g.compute()?;
        Ok(g.read_f32(io.out))
    }

    /// Generate one image.
    ///
    /// # Errors
    /// [`Error::Request`] for dimensions or step counts the model cannot
    /// serve; backend failures otherwise.
    pub fn generate(&mut self, req: &Request) -> Result<Image> {
        let cell = self.vae_cfg.scale_factor() as u32 * 2;
        if req.width == 0 || req.height == 0 || req.width % cell != 0 || req.height % cell != 0 {
            return Err(Error::Request(format!("width and height must be positive multiples of {cell}")));
        }
        if req.steps == 0 {
            return Err(Error::Request("steps must be at least 1".into()));
        }
        for (i, r) in req.references.iter().enumerate() {
            if r.width == 0 || r.height == 0 || r.width % cell != 0 || r.height % cell != 0 {
                return Err(Error::Request(format!("reference {i}: sides must be positive multiples of {cell}")));
            }
            if r.width * r.height > MAX_REFERENCE_PIXELS {
                return Err(Error::Request(format!("reference {i}: larger than {MAX_REFERENCE_PIXELS} pixels")));
            }
            if r.rgb.len() != (r.width * r.height * 3) as usize {
                return Err(Error::Request(format!("reference {i}: pixel buffer does not match its size")));
            }
        }
        let n = (req.width / cell) as usize * (req.height / cell) as usize * self.dit_cfg.in_channels as usize;
        let noise = schedule::gaussian(req.seed, n);
        self.run(req, noise)
    }

    /// The denoising pipeline from a given starting latent, laid out
    /// `[tokens][channels]` with tokens in row-major order.
    fn run(&mut self, req: &Request, noise: Vec<f32>) -> Result<Image> {
        let cell = self.vae_cfg.scale_factor() as u32 * 2;
        let use_cfg = !self.distilled && req.guidance_scale > 1.0;

        let t0 = Instant::now();
        let cond = self.encode(&req.prompt)?;
        let uncond = if use_cfg { Some(self.encode("")?) } else { None };
        let mut grids = vec![(0.0f32, (req.height / cell) as usize, (req.width / cell) as usize)];
        let mut reference_tokens = Vec::new();
        for (k, r) in req.references.iter().enumerate() {
            reference_tokens.extend(self.encode_reference(r)?);
            grids.push((REFERENCE_T_STEP * (k + 1) as f32, (r.height / cell) as usize, (r.width / cell) as usize));
        }
        let encode_ms = t0.elapsed().as_millis() as u64;

        let (_, gh, gw) = grids[0];
        let n_img = gh * gw;
        let n_all = grids.iter().map(|g| g.1 * g.2).sum::<usize>();
        let ch = self.dit_cfg.in_channels as usize;
        let n_txt = MAX_PROMPT_TOKENS;

        let t1 = Instant::now();
        let mut g = Graph::new(&self.backend)?;
        let io = flux2::build(&mut g, &self.dit_cfg, &self.dit, n_txt as i64, n_all as i64);
        g.finish(&[io.out])?;
        let pos = flux2::rope_positions(&flux2::positions(n_txt, &grids));
        let freq_factors = flux2::rope_freq_factors(&self.dit_cfg);
        let sig = schedule::sigmas(req.steps as usize, n_img);
        debug_assert_eq!(noise.len(), n_img * ch);
        let mut x = noise;
        let mut tokens = Vec::with_capacity(n_all * ch);
        let mut evaluations = 0u32;
        // The allocator reuses an input's memory once the graph has consumed
        // it, so every input is written before every evaluation.
        let evaluate = |tokens: &[f32], feat: &[f32], txt: &[f32]| -> Result<Vec<f32>> {
            g.set_i32(io.pos, &pos);
            g.set_f32(io.freq_factors, &freq_factors);
            g.set_f32(io.t_feat, feat);
            g.set_f32(io.img, tokens);
            g.set_f32(io.txt, txt);
            g.compute()?;
            let mut v = g.read_f32(io.out);
            v.truncate(n_img * ch);
            Ok(v)
        };
        for i in 0..req.steps as usize {
            let feat = flux2::timestep_features(sig[i] * 1000.0, self.dit_cfg.timestep_guidance_channels as usize);
            tokens.clear();
            tokens.extend_from_slice(&x);
            tokens.extend_from_slice(&reference_tokens);
            let mut v = evaluate(&tokens, &feat, &cond)?;
            evaluations += 1;
            if let Some(u) = &uncond {
                let vu = evaluate(&tokens, &feat, u)?;
                evaluations += 1;
                for (a, b) in v.iter_mut().zip(vu) {
                    *a = b + req.guidance_scale * (*a - b);
                }
            }
            let dt = sig[i + 1] - sig[i];
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += dt * vi;
            }
        }
        drop(g);
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let lat = self.unpatch(&x, gh, gw);
        let (lw, lh) = (2 * gw, 2 * gh);
        let mut g = Graph::new(&self.backend)?;
        let vio = vae::build_decoder(&mut g, &self.vae_cfg, &self.vae, lw as i64, lh as i64);
        g.finish(&[vio.out])?;
        g.set_f32(vio.latents, &lat);
        g.compute()?;
        let px = g.read_f32(vio.out);
        drop(g);
        let rgb = to_rgb(&px, req.width as usize, req.height as usize);
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

    /// Encode a reference image to normalised, patched latent tokens
    /// `[tokens][channels]` in row-major order.
    fn encode_reference(&self, r: &RgbImage) -> Result<Vec<f32>> {
        let (w, h) = (r.width as usize, r.height as usize);
        let mut px = vec![0f32; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    px[c * h * w + y * w + x] = f32::from(r.rgb[(y * w + x) * 3 + c]) / 127.5 - 1.0;
                }
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let io = vae::build_encoder(&mut g, &self.vae_cfg, &self.vae, w as i64, h as i64);
        g.finish(&[io.mean])?;
        g.set_f32(io.pixels, &px);
        g.compute()?;
        let mean = g.read_f32(io.mean);
        let f = self.vae_cfg.scale_factor() as usize;
        Ok(self.patch(&mean, h / f / 2, w / f / 2))
    }

    /// The 2x2 patching and latent normalisation: a `[W, H, C]` latent image
    /// to tokens `[gh * gw][c * 4]`. Inverse of [`Self::unpatch`].
    fn patch(&self, lat: &[f32], gh: usize, gw: usize) -> Vec<f32> {
        let pc = self.dit_cfg.in_channels as usize;
        let (w, h) = (2 * gw, 2 * gh);
        let mut out = vec![0f32; gh * gw * pc];
        for ty in 0..gh {
            for tx in 0..gw {
                let tok = ty * gw + tx;
                for k in 0..pc {
                    let (cl, ph, pw) = (k / 4, (k / 2) % 2, k % 2);
                    let v = lat[cl * h * w + (2 * ty + ph) * w + 2 * tx + pw];
                    out[tok * pc + k] = (v - self.bn_mean[k]) / self.bn_std[k];
                }
            }
        }
        out
    }

    /// Undo the latent normalisation and the 2x2 patching: tokens
    /// `[gh * gw][c * 4]` to a `[W, H, C]` latent image.
    fn unpatch(&self, x: &[f32], gh: usize, gw: usize) -> Vec<f32> {
        let pc = self.dit_cfg.in_channels as usize;
        let c = pc / 4;
        let (w, h) = (2 * gw, 2 * gh);
        let mut out = vec![0f32; c * h * w];
        for ty in 0..gh {
            for tx in 0..gw {
                let tok = ty * gw + tx;
                for k in 0..pc {
                    let v = x[tok * pc + k] * self.bn_std[k] + self.bn_mean[k];
                    let (cl, ph, pw) = (k / 4, (k / 2) % 2, k % 2);
                    let (yy, xx) = (2 * ty + ph, 2 * tx + pw);
                    out[cl * h * w + yy * w + xx] = v;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod parity;

fn to_rgb(px: &[f32], w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = px[c * h * w + y * w + x];
                let u = ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                rgb[(y * w + x) * 3 + c] = u;
            }
        }
    }
    rgb
}
