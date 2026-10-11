//! Image generation on a language backbone: unified models that read and write images in the
//! same token sequence as text.
//!
//! Two families run here, each with its backbone loaded once in the llama.cpp context and its
//! image side (heads and decoders) on the same device through `praecise-diffusion`:
//!
//! - Discrete image tokens (Janus-Pro): the backbone predicts one codebook index at a time
//!   through the image-token head; classifier-free guidance pairs the prompt with an
//!   unconditioned copy in a second sequence and mixes the two logit vectors before sampling;
//!   the finished grid goes through the VQ decoder.
//! - In-sequence flow matching (HunyuanImage 3.0): the backbone denoises a latent that sits in
//!   the sequence as image tokens attending to each other in full and to the prompt causally.
//!   The prompt is decoded once per guidance sequence; each step re-decodes the timestep token
//!   (causal) and the image tokens (non-causal) over it, and the final layer's velocity, mixed
//!   by guidance, takes one Euler step of the shifted flow schedule.
//!
//! With a fixed seed (or fixed starting noise) the same prompt, settings and weights give the
//! same image.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::LlamaToken;
use praecise_diffusion::hunyuan_image3::{HunyuanImage3Head, HunyuanImage3Vae};
use praecise_diffusion::janus::{JanusImageHead, JanusVqDecoder};

use crate::error::{Error, Result};

fn err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Inference(format!("{what}: {e}"))
}

/// A small deterministic generator for sampling and starting noise.
#[derive(Debug, Clone)]
pub struct SplitMix(u64);

impl SplitMix {
    /// Seeded generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// How the next image token is chosen from guided logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSampling {
    /// Guidance scale: `uncond + scale * (cond - uncond)`; 1 disables the mix.
    pub guidance_scale: f32,
    /// 0 picks the most likely token.
    pub temperature: f32,
    /// Keep the `top_k` most likely tokens (0 keeps all).
    pub top_k: usize,
    /// Keep the smallest set reaching this probability (1 keeps all).
    pub top_p: f32,
    /// Seed of the sampler.
    pub seed: u64,
}

impl Default for TokenSampling {
    fn default() -> Self {
        Self { guidance_scale: 5.0, temperature: 1.0, top_k: 0, top_p: 1.0, seed: 0 }
    }
}

/// Mix conditioned and unconditioned logits.
#[must_use]
pub fn guide(cond: &[f32], uncond: &[f32], scale: f32) -> Vec<f32> {
    cond.iter().zip(uncond).map(|(c, u)| u + scale * (c - u)).collect()
}

/// Choose a token from logits.
pub fn sample(logits: &[f32], s: &TokenSampling, rng: &mut SplitMix) -> u32 {
    if s.temperature <= 0.0 {
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if s.top_k > 0 && s.top_k < idx.len() {
        idx.truncate(s.top_k);
    }
    let m = logits[idx[0]];
    let mut p: Vec<f64> = idx.iter().map(|&i| f64::from((logits[i] - m) / s.temperature).exp()).collect();
    let total: f64 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= total);
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        let mut keep = p.len();
        for (k, v) in p.iter().enumerate() {
            acc += v;
            if acc >= f64::from(s.top_p) {
                keep = k + 1;
                break;
            }
        }
        p.truncate(keep);
        idx.truncate(keep);
        let t: f64 = p.iter().sum();
        p.iter_mut().for_each(|v| *v /= t);
    }
    let r = rng.uniform();
    let mut acc = 0.0;
    for (k, v) in p.iter().enumerate() {
        acc += v;
        if r < acc {
            return idx[k] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

/// The unconditioned copy of a prompt: every token except those in `keep` becomes `pad`
/// (positions and length unchanged).
#[must_use]
pub fn unconditioned(tokens: &[LlamaToken], keep: &[LlamaToken], pad: LlamaToken) -> Vec<LlamaToken> {
    tokens.iter().map(|t| if keep.contains(t) { *t } else { pad }).collect()
}

fn clear(ctx: &mut LlamaContext) -> Result<()> {
    ctx.clear_kv_cache_seq(None, None, None).map_err(|e| err("clearing the cache", e))?;
    Ok(())
}

/// Decode the prompts of the guidance sequences (sequence `i` reads `prompts[i]` from
/// position 0) and return each sequence's last hidden state.
fn prefill(ctx: &mut LlamaContext, prompts: &[&[LlamaToken]]) -> Result<Vec<Vec<f32>>> {
    let total: usize = prompts.iter().map(|p| p.len()).sum();
    let mut batch = LlamaBatch::new(total, prompts.len() as i32);
    let mut last = Vec::new();
    for (s, p) in prompts.iter().enumerate() {
        if p.is_empty() {
            return Err(Error::Inference("an image prompt is empty".into()));
        }
        for (i, &t) in p.iter().enumerate() {
            let out = i + 1 == p.len();
            batch.add(t, i as i32, &[s as i32], out).map_err(|e| err("prompt batch", e))?;
        }
        last.push(batch.n_tokens() - 1);
    }
    ctx.decode(&mut batch).map_err(|e| err("prompt decode", e))?;
    last.iter().map(|&i| ctx.embeddings_ith(i).map(<[f32]>::to_vec).map_err(|e| err("prompt hidden state", e))).collect()
}

/// A Janus-Pro image request. `cond` and `uncond` are full prompts ending with the
/// begin-of-image token; `uncond` must be as long as `cond` when guided.
#[derive(Debug, Clone)]
pub struct JanusRequest<'a> {
    /// The prompt.
    pub cond: &'a [LlamaToken],
    /// Its unconditioned copy (see [`unconditioned`]); `None` disables guidance.
    pub uncond: Option<&'a [LlamaToken]>,
    /// Sampling.
    pub sampling: TokenSampling,
    /// Tokens to feed instead of the sampled ones (to check logits against a reference).
    pub forced: Option<&'a [u32]>,
    /// Keep the guided logits of every step.
    pub keep_logits: bool,
}

/// A generated Janus-Pro image.
#[derive(Debug, Clone)]
pub struct JanusImage {
    /// Image tokens, row-major.
    pub tokens: Vec<u32>,
    /// Planar RGB `[3, side, side]` in `[-1, 1]`.
    pub pixels: Vec<f32>,
    /// Side in pixels.
    pub side: usize,
    /// Guided logits per step when requested.
    pub logits: Vec<Vec<f32>>,
}

/// Generate one Janus-Pro image. The context must have embeddings enabled with no pooling and
/// room for two sequences of prompt plus image tokens.
///
/// # Errors
/// Backbone, head or decoder failures, or inconsistent prompts.
pub fn generate_janus(ctx: &mut LlamaContext, head: &mut JanusImageHead, decoder: &JanusVqDecoder, req: &JanusRequest<'_>) -> Result<JanusImage> {
    let n_tok = head.config().tokens_per_image();
    if let Some(f) = req.forced {
        if f.len() != n_tok {
            return Err(Error::Inference(format!("forced tokens: {} for a grid of {n_tok}", f.len())));
        }
    }
    let guided = req.uncond.is_some();
    let mut prompts: Vec<&[LlamaToken]> = vec![req.cond];
    if let Some(u) = req.uncond {
        if u.len() != req.cond.len() {
            return Err(Error::Inference("the unconditioned prompt differs in length".into()));
        }
        prompts.push(u);
    }
    let seqs = prompts.len();
    clear(ctx)?;
    let mut hidden = prefill(ctx, &prompts)?;
    let width = hidden[0].len();
    let mut rng = SplitMix::new(req.sampling.seed);
    let mut tokens = Vec::with_capacity(n_tok);
    let mut kept = Vec::new();
    let base = req.cond.len() as i32;
    for step in 0..n_tok {
        let stacked: Vec<f32> = hidden.iter().flatten().copied().collect();
        let logits = head.logits_n(&stacked, seqs).map_err(|e| err("image head", e))?;
        let v = head.config().num_embeddings as usize;
        let mixed = if guided { guide(&logits[..v], &logits[v..2 * v], req.sampling.guidance_scale) } else { logits[..v].to_vec() };
        let tok = match req.forced {
            Some(f) => f[step],
            None => sample(&mixed, &req.sampling, &mut rng),
        };
        if req.keep_logits {
            kept.push(mixed);
        }
        tokens.push(tok);
        if step + 1 == n_tok {
            break;
        }
        let emb = head.embed_n(&vec![tok; seqs]).map_err(|e| err("image embedder", e))?;
        let mut batch = LlamaBatch::new_hidden(seqs, width, seqs as i32);
        for s in 0..seqs {
            batch.add_hidden(&emb[s * width..(s + 1) * width], base + step as i32, &[s as i32], true).map_err(|e| err("image batch", e))?;
        }
        ctx.decode(&mut batch).map_err(|e| err("image decode", e))?;
        hidden = (0..seqs as i32).map(|i| ctx.embeddings_ith(i).map(<[f32]>::to_vec).map_err(|e| err("image hidden state", e))).collect::<Result<_>>()?;
    }
    let pixels = decoder.decode(&tokens).map_err(|e| err("VQ decoder", e))?;
    Ok(JanusImage { tokens, pixels, side: decoder.image_side(), logits: kept })
}

/// The shifted flow schedule: `steps + 1` noise levels from 1 to 0, each `s` mapped to
/// `shift * s / (1 + (shift - 1) * s)`.
#[must_use]
pub fn flow_sigmas(steps: usize, shift: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let s = 1.0 - i as f32 / steps as f32;
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect()
}

/// Row and column position of image token `(r, c)` of an `h x w` grid starting at sequence
/// position `start`: the grid is centred on the positions it occupies.
#[must_use]
pub fn grid_position(start: i32, (h, w): (usize, usize), (r, c): (usize, usize)) -> (i32, i32) {
    let n = (h * w) as i64;
    let s2 = 2 * i64::from(start);
    let y = (s2 + n - h as i64 + 2 * r as i64).div_euclid(2);
    let x = (s2 + n - w as i64 + 2 * c as i64).div_euclid(2);
    (y as i32, x as i32)
}

/// A HunyuanImage 3.0 request. `cond` and `uncond` run up to (not including) the timestep
/// token, which the engine places at position `cond.len()`; the image tokens follow it.
#[derive(Debug, Clone)]
pub struct HunyuanImage3Request<'a> {
    /// The prompt through the image-size tokens.
    pub cond: &'a [LlamaToken],
    /// Its unconditioned copy (prompt text replaced by the model's guidance token); `None`
    /// disables guidance.
    pub uncond: Option<&'a [LlamaToken]>,
    /// Latent grid `(rows, cols)`.
    pub latent: (usize, usize),
    /// Denoising steps.
    pub steps: usize,
    /// Guidance scale on the velocity.
    pub guidance_scale: f32,
    /// Flow schedule shift.
    pub shift: f32,
    /// Seed of the starting noise when `noise` is `None`.
    pub seed: u64,
    /// Starting noise, planar `[C, rows, cols]`.
    pub noise: Option<&'a [f32]>,
    /// Keep the guided velocity of every step.
    pub keep_velocity: bool,
}

/// A generated HunyuanImage 3.0 image.
#[derive(Debug, Clone)]
pub struct HunyuanImage3Image {
    /// The final latent, planar `[C, rows, cols]`.
    pub latent: Vec<f32>,
    /// Planar RGB `[3, H, W]` in `[-1, 1]`.
    pub pixels: Vec<f32>,
    /// Width and height in pixels.
    pub size: (usize, usize),
    /// Guided velocity per step when requested.
    pub velocity: Vec<Vec<f32>>,
}

/// One forward of the image tokens: the velocity of every guidance sequence for `latent` at
/// timestep `t`, over prompts already in the cache (the cache keeps the prompts afterwards).
///
/// # Errors
/// Backbone or head failures.
pub fn hunyuan_image3_velocity(ctx: &mut LlamaContext, head: &HunyuanImage3Head, prefix_len: usize, seqs: usize, latent: &[f32], (h, w): (usize, usize), t: f32) -> Result<Vec<f32>> {
    let lt = prefix_len as i32;
    for s in 0..seqs {
        ctx.clear_kv_cache_seq(Some(s as u32), Some(prefix_len as u32), None).map_err(|e| err("trimming the cache", e))?;
    }
    let tt = head.timestep_token(t).map_err(|e| err("timestep token", e))?;
    let width = tt.len();
    let mut batch = LlamaBatch::new_hidden_positions(seqs, width, 4, seqs as i32);
    for s in 0..seqs {
        batch.add_hidden_positions(&tt, &[lt; 4], &[s as i32], false).map_err(|e| err("timestep batch", e))?;
    }
    ctx.decode(&mut batch).map_err(|e| err("timestep decode", e))?;

    let stacked: Vec<f32> = (0..seqs).flat_map(|_| latent.iter().copied()).collect();
    let tokens = head.embed_latents(&stacked, seqs, (h, w), t).map_err(|e| err("patch embedding", e))?;
    let n = h * w;
    let mut batch = LlamaBatch::new_hidden_positions(seqs * n, width, 4, seqs as i32);
    for s in 0..seqs {
        for r in 0..h {
            for c in 0..w {
                let (y, x) = grid_position(lt + 1, (h, w), (r, c));
                let row = s * n + r * w + c;
                batch.add_hidden_positions(&tokens[row * width..(row + 1) * width], &[y, x, 0, 0], &[s as i32], true).map_err(|e| err("image batch", e))?;
            }
        }
    }
    ctx.set_causal_attn(false);
    let decoded = ctx.decode(&mut batch);
    ctx.set_causal_attn(true);
    decoded.map_err(|e| err("image decode", e))?;
    let mut hidden = Vec::with_capacity(seqs * n * width);
    for i in 0..(seqs * n) as i32 {
        hidden.extend_from_slice(ctx.embeddings_ith(i).map_err(|e| err("image hidden state", e))?);
    }
    head.predict(&hidden, seqs, (h, w), t).map_err(|e| err("final layer", e))
}

/// Decode the guidance prompts into the cache (for [`hunyuan_image3_velocity`]).
///
/// # Errors
/// Backbone failures or inconsistent prompts.
pub fn hunyuan_image3_prefill(ctx: &mut LlamaContext, cond: &[LlamaToken], uncond: Option<&[LlamaToken]>) -> Result<usize> {
    let mut prompts: Vec<&[LlamaToken]> = vec![cond];
    if let Some(u) = uncond {
        if u.len() != cond.len() {
            return Err(Error::Inference("the unconditioned prompt differs in length".into()));
        }
        prompts.push(u);
    }
    clear(ctx)?;
    prefill(ctx, &prompts)?;
    Ok(prompts.len())
}

/// Generate one HunyuanImage 3.0 image. The context must have embeddings enabled with no
/// pooling, room for the guidance sequences (prompt + timestep + image tokens each) and a
/// micro-batch at least as large as all their image tokens together.
///
/// # Errors
/// Backbone, head or decoder failures, or inconsistent inputs.
pub fn generate_hunyuan_image3(ctx: &mut LlamaContext, head: &HunyuanImage3Head, vae: &HunyuanImage3Vae, req: &HunyuanImage3Request<'_>) -> Result<HunyuanImage3Image> {
    let (h, w) = req.latent;
    let lc = head.config().latent_channels();
    if req.steps == 0 || h == 0 || w == 0 {
        return Err(Error::Inference("an image needs at least one step and one latent cell".into()));
    }
    let mut latent = match req.noise {
        Some(n) if n.len() == lc * h * w => n.to_vec(),
        Some(n) => return Err(Error::Inference(format!("starting noise: {} floats for {}", n.len(), lc * h * w))),
        None => {
            let mut rng = SplitMix::new(req.seed);
            (0..lc * h * w).map(|_| rng.normal()).collect()
        }
    };
    let seqs = hunyuan_image3_prefill(ctx, req.cond, req.uncond)?;
    let sigmas = flow_sigmas(req.steps, req.shift);
    let per = lc * h * w;
    let mut kept = Vec::new();
    for i in 0..req.steps {
        let t = sigmas[i] * 1000.0;
        let v = hunyuan_image3_velocity(ctx, head, req.cond.len(), seqs, &latent, (h, w), t)?;
        let pred = if seqs == 2 { guide(&v[..per], &v[per..2 * per], req.guidance_scale) } else { v[..per].to_vec() };
        let dt = sigmas[i + 1] - sigmas[i];
        latent.iter_mut().zip(&pred).for_each(|(x, d)| *x += d * dt);
        if req.keep_velocity {
            kept.push(pred);
        }
    }
    let pixels = vae.decode(&latent, (h, w)).map_err(|e| err("latent decoder", e))?;
    let f = head.config().scale_factor() as usize;
    Ok(HunyuanImage3Image { latent, pixels, size: (w * f, h * f), velocity: kept })
}

/// Planar `[3, H, W]` in `[-1, 1]` to packed RGB bytes.
#[must_use]
pub fn to_rgb8(planar: &[f32], width: usize, height: usize) -> Vec<u8> {
    let n = width * height;
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        for c in 0..3 {
            let v = ((planar[c * n + i] + 1.0) * 0.5).clamp(0.0, 1.0);
            out[i * 3 + c] = (v * 255.0).round() as u8;
        }
    }
    out
}
