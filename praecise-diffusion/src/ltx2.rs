//! LTX-2.3 audio-video diffusion transformer.
//!
//! Two token streams, video latents and audio latents, run side by side
//! through the same blocks. Each block gives every stream its own
//! self-attention, its own prompt cross-attention and its own feed-forward,
//! and in between lets each stream attend to the other (audio to video and
//! video to audio). Every attention multiplies each head's output by a learned
//! per-token gate, `2 * sigmoid(linear(x))`. Modulation comes from timestep
//! embeddings plus learned per-block tables; the prompt-side keys and values
//! are modulated too.
//!
//! Rotary embeddings rotate split halves of each head by angles computed on
//! the host from token coordinates in seconds and pixels (video: frame time,
//! row, column; audio: time), normalised by their maximum positions. Every
//! head has its own frequency band.
//!
//! The caption features fed here are the output of the prompt connectors (one
//! per stream), already at the stream widths: the released LTX-2.3 layout has
//! no caption projection inside the transformer.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use crate::s3dit::{S3DitConfig, TIME_FEATURES};
use llama_cpp_sys_2 as sys;

pub mod connectors;
pub mod single_file;

#[cfg(test)]
mod parity;

/// Attention query/key norm epsilon (fixed in the reference).
const QK_EPS: f32 = 1e-6;
/// Output norm epsilon (fixed in the reference).
const OUT_EPS: f32 = 1e-6;

fn yes() -> bool {
    true
}

/// Transformer configuration, read from `transformer/config.json`.
#[derive(Debug, Clone, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Ltx2Config {
    /// Video latent channels.
    pub in_channels: u64,
    /// Video output channels.
    pub out_channels: u64,
    /// Spatial patch size (one supported: 1).
    pub patch_size: u64,
    /// Temporal patch size (one supported: 1).
    pub patch_size_t: u64,
    /// Video attention heads.
    pub num_attention_heads: u64,
    /// Video head width.
    pub attention_head_dim: u64,
    /// Video prompt feature width.
    pub cross_attention_dim: u64,
    /// Autoencoder compression (frames, rows, columns).
    pub vae_scale_factors: Vec<u64>,
    /// Maximum temporal position, in seconds.
    pub pos_embed_max_pos: u64,
    /// Maximum row position, in pixels.
    pub base_height: u64,
    /// Maximum column position, in pixels.
    pub base_width: u64,
    /// Per-head output gates on video attentions.
    pub gated_attn: bool,
    /// Modulated video prompt cross-attention.
    pub cross_attn_mod: bool,
    /// Audio latent channels.
    pub audio_in_channels: u64,
    /// Audio output channels.
    pub audio_out_channels: u64,
    /// Audio patch size (one supported: 1).
    pub audio_patch_size: u64,
    /// Audio temporal patch size (one supported: 1).
    pub audio_patch_size_t: u64,
    /// Audio attention heads.
    pub audio_num_attention_heads: u64,
    /// Audio head width.
    pub audio_attention_head_dim: u64,
    /// Audio prompt feature width.
    pub audio_cross_attention_dim: u64,
    /// Mel frames per audio latent.
    pub audio_scale_factor: u64,
    /// Maximum audio position, in seconds.
    pub audio_pos_embed_max_pos: u64,
    /// Audio sample rate.
    pub audio_sampling_rate: u64,
    /// Samples per mel frame.
    pub audio_hop_length: u64,
    /// Per-head output gates on audio attentions.
    pub audio_gated_attn: bool,
    /// Modulated audio prompt cross-attention.
    pub audio_cross_attn_mod: bool,
    /// Blocks.
    pub num_layers: usize,
    /// Feed-forward activation (one supported: `gelu-approximate`).
    pub activation_fn: String,
    /// Query/key norm (one supported: `rms_norm_across_heads`).
    pub qk_norm: String,
    /// Learned block norm weights (unsupported).
    pub norm_elementwise_affine: bool,
    /// Block norm epsilon.
    pub norm_eps: f64,
    /// Rotary base.
    pub rope_theta: f64,
    /// Added to frame positions before clamping at zero.
    pub causal_offset: u64,
    /// Timestep scale of the denoising timestep.
    pub timestep_scale_multiplier: u64,
    /// Timestep scale of the audio-video cross-attention gates.
    pub cross_attn_timestep_scale_multiplier: u64,
    /// Rotary layout (one supported: `split`).
    pub rope_type: String,
    /// Caption projection inside the transformer (unsupported).
    #[serde(default = "yes")]
    pub use_prompt_embeddings: bool,
    /// Timestep-modulated prompt keys and values.
    #[serde(default = "yes")]
    pub use_prompt_adaln_single: bool,
    /// Attention projection biases.
    #[serde(default = "yes")]
    pub attention_bias: bool,
    /// Attention output biases.
    #[serde(default = "yes")]
    pub attention_out_bias: bool,
    /// Video feed-forward biases.
    #[serde(default = "yes")]
    pub ff_bias: bool,
    /// Audio feed-forward biases.
    #[serde(default = "yes")]
    pub audio_ff_bias: bool,
    /// Keyframe position embedding (unsupported).
    #[serde(default)]
    pub use_keyframes_abs_pos_embedding: bool,
}

impl Ltx2Config {
    /// Refuse variants this implementation has not been checked against: the
    /// LTX-2.3 layout only.
    ///
    /// # Errors
    /// When the configuration names a different layout.
    pub fn validate(&self) -> Result<()> {
        let checks = [
            (self.patch_size == 1 && self.patch_size_t == 1 && self.audio_patch_size == 1 && self.audio_patch_size_t == 1, "patch sizes other than 1"),
            (self.rope_type == "split", "rotary layouts other than split"),
            (self.qk_norm == "rms_norm_across_heads", "query/key norms other than rms_norm_across_heads"),
            (self.activation_fn == "gelu-approximate", "activations other than gelu-approximate"),
            (self.gated_attn && self.audio_gated_attn, "ungated attention"),
            (self.cross_attn_mod && self.audio_cross_attn_mod && self.use_prompt_adaln_single, "unmodulated prompt cross-attention"),
            (!self.use_prompt_embeddings, "a caption projection inside the transformer"),
            (!self.norm_elementwise_affine, "learned block norm weights"),
            (self.attention_bias && self.attention_out_bias && self.ff_bias && self.audio_ff_bias, "bias-free projections"),
            (!self.use_keyframes_abs_pos_embedding, "keyframe position embeddings"),
            (self.vae_scale_factors.len() == 3, "autoencoder scale factors other than three"),
            (self.inner() == self.cross_attention_dim, "a video prompt width other than the video width"),
            (
                self.audio_inner() == self.audio_cross_attention_dim && self.num_attention_heads == self.audio_num_attention_heads,
                "audio-video rotary tables wider or with other heads than the audio stream",
            ),
            (self.attention_head_dim % 2 == 0 && self.audio_attention_head_dim % 2 == 0, "odd head widths"),
        ];
        for (ok, what) in checks {
            if !ok {
                return Err(Error::Config(format!("transformer: {what} not implemented")));
            }
        }
        Ok(())
    }

    /// Video stream width.
    #[must_use]
    pub fn inner(&self) -> u64 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio stream width.
    #[must_use]
    pub fn audio_inner(&self) -> u64 {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    fn adaln_specs(v: &mut Vec<WeightSpec>, p: &str, d: u64, n: u64, linear: WType) {
        let e = format!("{p}.emb.timestep_embedder");
        v.push(WeightSpec::new(format!("{e}.linear_1.weight"), &[d, TIME_FEATURES as u64], WType::F32));
        v.push(WeightSpec::new(format!("{e}.linear_1.bias"), &[d], WType::F32));
        v.push(WeightSpec::new(format!("{e}.linear_2.weight"), &[d, d], linear));
        v.push(WeightSpec::new(format!("{e}.linear_2.bias"), &[d], WType::F32));
        v.push(WeightSpec::new(format!("{p}.linear.weight"), &[n * d, d], linear));
        v.push(WeightSpec::new(format!("{p}.linear.bias"), &[n * d], WType::F32));
    }

    fn attn_specs(v: &mut Vec<WeightSpec>, p: &str, (dq, dkv, inner, heads): (u64, u64, u64, u64), linear: WType) {
        for (n, cin) in [("to_q", dq), ("to_k", dkv), ("to_v", dkv)] {
            v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[inner, cin], linear));
            v.push(WeightSpec::new(format!("{p}.{n}.bias"), &[inner], WType::F32));
        }
        v.push(WeightSpec::new(format!("{p}.norm_q.weight"), &[inner], WType::F32));
        v.push(WeightSpec::new(format!("{p}.norm_k.weight"), &[inner], WType::F32));
        v.push(WeightSpec::new(format!("{p}.to_out.0.weight"), &[dq, inner], linear));
        v.push(WeightSpec::new(format!("{p}.to_out.0.bias"), &[dq], WType::F32));
        v.push(WeightSpec::new(format!("{p}.to_gate_logits.weight"), &[heads, dq], WType::F32));
        v.push(WeightSpec::new(format!("{p}.to_gate_logits.bias"), &[heads], WType::F32));
    }

    /// Every transformer weight.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, ad) = (self.inner(), self.audio_inner());
        let (h, ah) = (self.num_attention_heads, self.audio_num_attention_heads);
        let mut v = vec![
            WeightSpec::new("proj_in.weight", &[d, self.in_channels], WType::F32),
            WeightSpec::new("proj_in.bias", &[d], WType::F32),
            WeightSpec::new("audio_proj_in.weight", &[ad, self.audio_in_channels], WType::F32),
            WeightSpec::new("audio_proj_in.bias", &[ad], WType::F32),
            WeightSpec::new("proj_out.weight", &[self.out_channels, d], WType::F32),
            WeightSpec::new("proj_out.bias", &[self.out_channels], WType::F32),
            WeightSpec::new("audio_proj_out.weight", &[self.audio_out_channels, ad], WType::F32),
            WeightSpec::new("audio_proj_out.bias", &[self.audio_out_channels], WType::F32),
            WeightSpec::new("scale_shift_table", &[2, d], WType::F32),
            WeightSpec::new("audio_scale_shift_table", &[2, ad], WType::F32),
        ];
        for (p, w, n) in [
            ("time_embed", d, 9),
            ("audio_time_embed", ad, 9),
            ("prompt_adaln", d, 2),
            ("audio_prompt_adaln", ad, 2),
            ("av_cross_attn_video_scale_shift", d, 4),
            ("av_cross_attn_audio_scale_shift", ad, 4),
            ("av_cross_attn_video_a2v_gate", d, 1),
            ("av_cross_attn_audio_v2a_gate", ad, 1),
        ] {
            Self::adaln_specs(&mut v, p, w, n, linear);
        }
        for i in 0..self.num_layers {
            let b = format!("transformer_blocks.{i}");
            Self::attn_specs(&mut v, &format!("{b}.attn1"), (d, d, d, h), linear);
            Self::attn_specs(&mut v, &format!("{b}.attn2"), (d, d, d, h), linear);
            Self::attn_specs(&mut v, &format!("{b}.audio_attn1"), (ad, ad, ad, ah), linear);
            Self::attn_specs(&mut v, &format!("{b}.audio_attn2"), (ad, ad, ad, ah), linear);
            Self::attn_specs(&mut v, &format!("{b}.audio_to_video_attn"), (d, ad, ad, ah), linear);
            Self::attn_specs(&mut v, &format!("{b}.video_to_audio_attn"), (ad, d, ad, ah), linear);
            for (f, w) in [("ff", d), ("audio_ff", ad)] {
                v.push(WeightSpec::new(format!("{b}.{f}.net.0.proj.weight"), &[4 * w, w], linear));
                v.push(WeightSpec::new(format!("{b}.{f}.net.0.proj.bias"), &[4 * w], WType::F32));
                v.push(WeightSpec::new(format!("{b}.{f}.net.2.weight"), &[w, 4 * w], linear));
                v.push(WeightSpec::new(format!("{b}.{f}.net.2.bias"), &[w], WType::F32));
            }
            for (t, rows, w) in [
                ("scale_shift_table", 9, d),
                ("audio_scale_shift_table", 9, ad),
                ("prompt_scale_shift_table", 2, d),
                ("audio_prompt_scale_shift_table", 2, ad),
                ("video_a2v_cross_attn_scale_shift_table", 5, d),
                ("audio_a2v_cross_attn_scale_shift_table", 5, ad),
            ] {
                v.push(WeightSpec::new(format!("{b}.{t}"), &[rows, w], WType::F32));
            }
        }
        v
    }

    /// Video token coordinates `[axis][token]` (frame time in seconds, row and
    /// column in pixels, each the middle of the token's span) for a latent
    /// grid of `frames x height x width`, tokens frame-major.
    #[must_use]
    pub fn video_coords(&self, frames: usize, height: usize, width: usize, fps: f32) -> [Vec<f32>; 3] {
        let s: Vec<f32> = self.vae_scale_factors.iter().map(|&x| x as f32).collect();
        let off = self.causal_offset as f32;
        let time = |f: usize| ((f as f32 * s[0] + off - s[0]).max(0.0)) / fps;
        let n = frames * height * width;
        let mut c = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
        for f in 0..frames {
            let t = (time(f) + time(f + 1)) / 2.0;
            for y in 0..height {
                let r = (y as f32 * s[1] + (y + 1) as f32 * s[1]) / 2.0;
                for x in 0..width {
                    c[0].push(t);
                    c[1].push(r);
                    c[2].push((x as f32 * s[2] + (x + 1) as f32 * s[2]) / 2.0);
                }
            }
        }
        c
    }

    /// Audio token times in seconds (the middle of each latent's span).
    #[must_use]
    pub fn audio_coords(&self, frames: usize) -> Vec<f32> {
        let s = self.audio_scale_factor as f32;
        let off = self.causal_offset as f32;
        let secs = |i: usize| ((i as f32 * s + off - s).max(0.0)) * self.audio_hop_length as f32 / self.audio_sampling_rate as f32;
        (0..frames).map(|i| (secs(i) + secs(i + 1)) / 2.0).collect()
    }

    /// Video, audio and the two cross-stream rotary tables for one shape.
    #[must_use]
    pub fn rotary(&self, frames: usize, height: usize, width: usize, audio_frames: usize, fps: f32) -> Rotary {
        let vc = self.video_coords(frames, height, width, fps);
        let ac = self.audio_coords(audio_frames);
        let theta = self.rope_theta;
        let (vmax, amax) = (self.pos_embed_max_pos as f32, self.audio_pos_embed_max_pos as f32);
        let cmax = vmax.max(amax);
        let max = [vmax, self.base_height as f32, self.base_width as f32];
        let (d, ad, ch) = (self.inner() as usize, self.audio_inner() as usize, self.audio_cross_attention_dim as usize);
        let (h, ah) = (self.num_attention_heads as usize, self.audio_num_attention_heads as usize);
        Rotary {
            video: rope_tables(&[(&vc[0], max[0]), (&vc[1], max[1]), (&vc[2], max[2])], d, h, theta),
            audio: rope_tables(&[(&ac, amax)], ad, ah, theta),
            cross_video: rope_tables(&[(&vc[0], cmax)], ch, h, theta),
            cross_audio: rope_tables(&[(&ac, cmax)], ch, ah, theta),
        }
    }
}

/// `n` evenly spaced values from 0 to 1, computed from both ends as the
/// reference computes them.
fn linspace01(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![0.0];
    }
    let step = 1.0 / (n - 1) as f64;
    let half = n / 2;
    (0..n).map(|i| if i < half { i as f64 * step } else { 1.0 - (n - 1 - i) as f64 * step }).collect()
}

/// Cos and sin tables `[token][head][head width]` for coordinates per axis
/// (each with its maximum position) over a rotary width `dim` split across
/// `heads`. Frequencies interleave the axes; the unused leading pairs turn by
/// zero. Within a head, the two halves share each angle.
fn rope_tables(axes: &[(&Vec<f32>, f32)], dim: usize, heads: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let n_axes = axes.len();
    let per_axis = dim / (2 * n_axes);
    let freqs: Vec<f32> = linspace01(per_axis).iter().map(|&x| (theta.powf(x) * std::f64::consts::PI / 2.0) as f32).collect();
    let half = dim / 2;
    let pad = half - per_axis * n_axes;
    let (hd, r) = (dim / heads, dim / heads / 2);
    let n = axes[0].0.len();
    let mut cos = Vec::with_capacity(n * dim);
    let mut sin = Vec::with_capacity(n * dim);
    let mut ang = vec![0f32; half];
    for t in 0..n {
        let grid: Vec<f32> = axes.iter().map(|(c, m)| c[t] / m * 2.0 - 1.0).collect();
        for (f, &fr) in freqs.iter().enumerate() {
            for (a, g) in grid.iter().enumerate() {
                ang[pad + f * n_axes + a] = g * fr;
            }
        }
        for hh in 0..heads {
            for j in 0..hd {
                let x = ang[hh * r + j % r];
                let (s, c) = if hh * r + j % r < pad { (0.0, 1.0) } else { (x.sin(), x.cos()) };
                cos.push(c);
                sin.push(s);
            }
        }
    }
    (cos, sin)
}

/// Host rotary tables for one shape: `(cos, sin)` each.
#[derive(Debug, Clone)]
pub struct Rotary {
    /// Video self-attention.
    pub video: (Vec<f32>, Vec<f32>),
    /// Audio self-attention.
    pub audio: (Vec<f32>, Vec<f32>),
    /// Video side of the cross-stream attentions.
    pub cross_video: (Vec<f32>, Vec<f32>),
    /// Audio side of the cross-stream attentions.
    pub cross_audio: (Vec<f32>, Vec<f32>),
}

/// Graph inputs and outputs of one transformer evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Ltx2Io {
    /// Video latent tokens `[channels, tokens]`.
    pub video: Tn,
    /// Audio latent tokens `[channels, tokens]`.
    pub audio: Tn,
    /// Video prompt features `[width, tokens]`.
    pub text: Tn,
    /// Audio prompt features `[width, tokens]`.
    pub audio_text: Tn,
    /// Timestep features `[256, 4]`: video, audio, and each scaled for the
    /// cross-stream gates.
    pub time: Tn,
    /// Rotary tables, in [`Rotary`] order, cos then sin.
    pub rope: [Tn; 8],
    /// Predicted video velocity `[channels, tokens]`.
    pub out_video: Tn,
    /// Predicted audio velocity `[channels, tokens]`.
    pub out_audio: Tn,
}

#[derive(Clone, Copy)]
struct Ctx {
    eps: f32,
    exact: bool,
}

fn row(g: &mut Graph, t: Tn, d: i64, k: i64) -> Tn {
    g.view_1d(t, d, (k * d) as usize)
}

/// Table row `k` plus timestep embedding segment `k`.
fn param(g: &mut Graph, table: Tn, temb: Tn, d: i64, k: i64) -> Tn {
    let a = row(g, table, d, k);
    let b = row(g, temb, d, k);
    g.add(a, b)
}

/// `x * (1 + scale) + shift`.
fn modulate(g: &mut Graph, x: Tn, shift: Tn, scale: Tn) -> Tn {
    let s = g.scale_bias(scale, 1.0, 1.0);
    let y = g.mul(x, s);
    g.add(y, shift)
}

/// Timestep embedding module: `(linear(silu(emb)), emb)`.
fn adaln(g: &mut Graph, w: &Weights, p: &str, features: Tn) -> (Tn, Tn) {
    let e = format!("{p}.emb.timestep_embedder");
    let h = g.linear_b(w.get(&format!("{e}.linear_1.weight")), w.get(&format!("{e}.linear_1.bias")), features);
    let h = g.silu(h);
    let emb = g.linear_b(w.get(&format!("{e}.linear_2.weight")), w.get(&format!("{e}.linear_2.bias")), h);
    let s = g.silu(emb);
    (g.linear_b(w.get(&format!("{p}.linear.weight")), w.get(&format!("{p}.linear.bias")), s), emb)
}

fn gelu(g: &mut Graph, x: Tn, exact: bool) -> Tn {
    if !exact {
        return g.gelu_tanh(x);
    }
    g.gelu_tanh_exact(x)
}

/// Rotary tables of one attention side, `[head width, heads, tokens]`.
type Rope = (Tn, Tn);

#[allow(clippy::too_many_arguments)]
fn attention(g: &mut Graph, w: &Weights, a: &str, c: Ctx, (hd, heads): (i64, i64), xq: Tn, xkv: Tn, rope: Option<(Rope, Rope)>) -> Tn {
    let (n, m) = (xq.ne(1), xkv.ne(1));
    let wn = |s: &str| w.get(&format!("{a}.{s}"));
    let q = g.linear_b(wn("to_q.weight"), wn("to_q.bias"), xq);
    let k = g.linear_b(wn("to_k.weight"), wn("to_k.bias"), xkv);
    let v = g.linear_b(wn("to_v.weight"), wn("to_v.bias"), xkv);
    let q = g.rms_norm(q, QK_EPS);
    let q = g.mul(q, wn("norm_q.weight"));
    let k = g.rms_norm(k, QK_EPS);
    let k = g.mul(k, wn("norm_k.weight"));
    let mut q = g.reshape(q, &[hd, heads, n]);
    let mut k = g.reshape(k, &[hd, heads, m]);
    let v = g.reshape(v, &[hd, heads, m]);
    if let Some(((qc, qs), (kc, ks))) = rope {
        q = g.rotate_half_rope(q, qc, qs);
        k = g.rotate_half_rope(k, kc, ks);
    }
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    let scale = 1.0 / (hd as f32).sqrt();
    let o = if c.exact {
        let k = g.cont(k);
        let v = g.cont(v);
        g.attention_exact(q, k, v, None, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, None, scale, true)
    };
    let o = g.reshape(o, &[hd, heads, n]);
    let gate = g.linear_b(wn("to_gate_logits.weight"), wn("to_gate_logits.bias"), xq);
    let gate = g.sigmoid(gate);
    let gate = g.scale_bias(gate, 2.0, 0.0);
    let gate = g.reshape(gate, &[1, heads, n]);
    let o = g.mul(o, gate);
    let o = g.reshape(o, &[hd * heads, n]);
    g.linear_b(wn("to_out.0.weight"), wn("to_out.0.bias"), o)
}

fn feed_forward(g: &mut Graph, w: &Weights, p: &str, x: Tn, exact: bool) -> Tn {
    let h = g.linear_b(w.get(&format!("{p}.net.0.proj.weight")), w.get(&format!("{p}.net.0.proj.bias")), x);
    let h = gelu(g, h, exact);
    g.linear_b(w.get(&format!("{p}.net.2.weight")), w.get(&format!("{p}.net.2.bias")), h)
}

/// Per-evaluation modulation inputs shared by every block.
struct Conditioning {
    temb: Tn,
    temb_audio: Tn,
    prompt: Tn,
    prompt_audio: Tn,
    ca_video: Tn,
    ca_audio: Tn,
    gate_video: Tn,
    gate_audio: Tn,
}

/// Rotary tables per attention side.
struct Ropes {
    video: Rope,
    audio: Rope,
    cross_video: Rope,
    cross_audio: Rope,
}

#[allow(clippy::too_many_arguments)]
fn block(g: &mut Graph, w: &Weights, b: &str, cfg: &Ltx2Config, c: Ctx, (x, a): (Tn, Tn), (text, atext): (Tn, Tn), m: &Conditioning, r: &Ropes) -> (Tn, Tn) {
    let (d, ad) = (cfg.inner() as i64, cfg.audio_inner() as i64);
    let vh = (cfg.attention_head_dim as i64, cfg.num_attention_heads as i64);
    let ah = (cfg.audio_attention_head_dim as i64, cfg.audio_num_attention_heads as i64);
    let t = |s: &str| w.get(&format!("{b}.{s}"));
    let (vt, at) = (t("scale_shift_table"), t("audio_scale_shift_table"));
    let vp = |g: &mut Graph, k: i64| param(g, vt, m.temb, d, k);
    let ap = |g: &mut Graph, k: i64| param(g, at, m.temb_audio, ad, k);

    // Self-attention.
    let (sh, sc, gt) = (vp(g, 0), vp(g, 1), vp(g, 2));
    let h = g.rms_norm(x, c.eps);
    let h = modulate(g, h, sh, sc);
    let o = attention(g, w, &format!("{b}.attn1"), c, vh, h, h, Some((r.video, r.video)));
    let o = g.mul(o, gt);
    let x = g.add(x, o);
    let (sh, sc, gt) = (ap(g, 0), ap(g, 1), ap(g, 2));
    let h = g.rms_norm(a, c.eps);
    let h = modulate(g, h, sh, sc);
    let o = attention(g, w, &format!("{b}.audio_attn1"), c, ah, h, h, Some((r.audio, r.audio)));
    let o = g.mul(o, gt);
    let a = g.add(a, o);

    // Prompt cross-attention, queries and keys/values both modulated.
    let (pt, apt) = (t("prompt_scale_shift_table"), t("audio_prompt_scale_shift_table"));
    let (ksh, ksc) = (param(g, pt, m.prompt, d, 0), param(g, pt, m.prompt, d, 1));
    let (sh, sc, gt) = (vp(g, 6), vp(g, 7), vp(g, 8));
    let h = g.rms_norm(x, c.eps);
    let h = modulate(g, h, sh, sc);
    let kv = modulate(g, text, ksh, ksc);
    let o = attention(g, w, &format!("{b}.attn2"), c, vh, h, kv, None);
    let o = g.mul(o, gt);
    let x = g.add(x, o);
    let (ksh, ksc) = (param(g, apt, m.prompt_audio, ad, 0), param(g, apt, m.prompt_audio, ad, 1));
    let (sh, sc, gt) = (ap(g, 6), ap(g, 7), ap(g, 8));
    let h = g.rms_norm(a, c.eps);
    let h = modulate(g, h, sh, sc);
    let kv = modulate(g, atext, ksh, ksc);
    let o = attention(g, w, &format!("{b}.audio_attn2"), c, ah, h, kv, None);
    let o = g.mul(o, gt);
    let a = g.add(a, o);

    // Audio to video, then video to audio, both from the same normed inputs.
    let nx = g.rms_norm(x, c.eps);
    let na = g.rms_norm(a, c.eps);
    let (vct, act) = (t("video_a2v_cross_attn_scale_shift_table"), t("audio_a2v_cross_attn_scale_shift_table"));
    let vca: Vec<Tn> = (0..4).map(|k| param(g, vct, m.ca_video, d, k)).collect();
    let aca: Vec<Tn> = (0..4).map(|k| param(g, act, m.ca_audio, ad, k)).collect();
    // The gates take the fifth table row and a one-row embedding.
    let a2v_gate = {
        let r4 = row(g, vct, d, 4);
        g.add(r4, m.gate_video)
    };
    let v2a_gate = {
        let r4 = row(g, act, ad, 4);
        g.add(r4, m.gate_audio)
    };
    let q = modulate(g, nx, vca[1], vca[0]);
    let kv = modulate(g, na, aca[1], aca[0]);
    let o = attention(g, w, &format!("{b}.audio_to_video_attn"), c, ah, q, kv, Some((r.cross_video, r.cross_audio)));
    let o = g.mul(o, a2v_gate);
    let x = g.add(x, o);
    let q = modulate(g, na, aca[3], aca[2]);
    let kv = modulate(g, nx, vca[3], vca[2]);
    let o = attention(g, w, &format!("{b}.video_to_audio_attn"), c, ah, q, kv, Some((r.cross_audio, r.cross_video)));
    let o = g.mul(o, v2a_gate);
    let a = g.add(a, o);

    // Feed-forward.
    let (sh, sc, gt) = (vp(g, 3), vp(g, 4), vp(g, 5));
    let h = g.rms_norm(x, c.eps);
    let h = modulate(g, h, sh, sc);
    let o = feed_forward(g, w, &format!("{b}.ff"), h, c.exact);
    let o = g.mul(o, gt);
    let x = g.add(x, o);
    let (sh, sc, gt) = (ap(g, 3), ap(g, 4), ap(g, 5));
    let h = g.rms_norm(a, c.eps);
    let h = modulate(g, h, sh, sc);
    let o = feed_forward(g, w, &format!("{b}.audio_ff"), h, c.exact);
    let o = g.mul(o, gt);
    let a = g.add(a, o);
    (x, a)
}

/// Build one evaluation over `n_video` video tokens, `n_audio` audio tokens
/// and `n_text` prompt tokens per stream.
pub fn build(g: &mut Graph, cfg: &Ltx2Config, w: &Weights, (n_video, n_audio, n_text): (i64, i64, i64), exact: bool) -> Ltx2Io {
    let (d, ad) = (cfg.inner() as i64, cfg.audio_inner() as i64);
    let c = Ctx { eps: cfg.norm_eps as f32, exact };
    let video = g.input(sys::GGML_TYPE_F32, &[cfg.in_channels as i64, n_video]);
    let audio = g.input(sys::GGML_TYPE_F32, &[cfg.audio_in_channels as i64, n_audio]);
    let text = g.input(sys::GGML_TYPE_F32, &[d, n_text]);
    let audio_text = g.input(sys::GGML_TYPE_F32, &[ad, n_text]);
    let time = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64, 4]);
    let (vhd, vh) = (cfg.attention_head_dim as i64, cfg.num_attention_heads as i64);
    let (ahd, ah) = (cfg.audio_attention_head_dim as i64, cfg.audio_num_attention_heads as i64);
    let shapes = [(vhd, vh, n_video), (ahd, ah, n_audio), (ahd, vh, n_video), (ahd, ah, n_audio)];
    let mut rope = Vec::with_capacity(8);
    for (hd, h, n) in shapes {
        rope.push(g.input(sys::GGML_TYPE_F32, &[hd, h, n]));
        rope.push(g.input(sys::GGML_TYPE_F32, &[hd, h, n]));
    }
    let rope: [Tn; 8] = rope.try_into().unwrap_or_else(|_| unreachable!());
    let r = Ropes { video: (rope[0], rope[1]), audio: (rope[2], rope[3]), cross_video: (rope[4], rope[5]), cross_audio: (rope[6], rope[7]) };

    let tf = |g: &mut Graph, k: i64| g.view_1d(time, TIME_FEATURES as i64, (k * TIME_FEATURES as i64) as usize);
    let (f_v, f_a, f_ag, f_vg) = (tf(g, 0), tf(g, 1), tf(g, 2), tf(g, 3));
    let (temb, emb_v) = adaln(g, w, "time_embed", f_v);
    let (temb_audio, emb_a) = adaln(g, w, "audio_time_embed", f_a);
    let m = Conditioning {
        temb,
        temb_audio,
        prompt: adaln(g, w, "prompt_adaln", f_v).0,
        prompt_audio: adaln(g, w, "audio_prompt_adaln", f_a).0,
        // Each stream's cross-attention modulation follows the other stream's
        // timestep.
        ca_video: adaln(g, w, "av_cross_attn_video_scale_shift", f_a).0,
        gate_video: adaln(g, w, "av_cross_attn_video_a2v_gate", f_ag).0,
        ca_audio: adaln(g, w, "av_cross_attn_audio_scale_shift", f_v).0,
        gate_audio: adaln(g, w, "av_cross_attn_audio_v2a_gate", f_vg).0,
    };

    let mut x = g.linear_b(w.get("proj_in.weight"), w.get("proj_in.bias"), video);
    let mut a = g.linear_b(w.get("audio_proj_in.weight"), w.get("audio_proj_in.bias"), audio);
    for i in 0..cfg.num_layers {
        (x, a) = block(g, w, &format!("transformer_blocks.{i}"), cfg, c, (x, a), (text, audio_text), &m, &r);
    }
    let out = |g: &mut Graph, x: Tn, table: &str, emb: Tn, width: i64, proj: &str| {
        let t = w.get(table);
        let shift = param(g, t, emb, width, 0);
        let scale = {
            let r1 = row(g, t, width, 1);
            g.add(r1, emb)
        };
        let h = g.norm(x, OUT_EPS);
        let h = modulate(g, h, shift, scale);
        g.linear_b(w.get(&format!("{proj}.weight")), w.get(&format!("{proj}.bias")), h)
    };
    let out_video = out(g, x, "scale_shift_table", emb_v, d, "proj_out");
    let out_audio = out(g, a, "audio_scale_shift_table", emb_a, ad, "audio_proj_out");
    Ltx2Io { video, audio, text, audio_text, time, rope, out_video, out_audio }
}

/// Latent grid of one audio-video evaluation.
#[derive(Debug, Clone, Copy)]
pub struct AvShape {
    /// Latent frames.
    pub frames: usize,
    /// Latent rows.
    pub height: usize,
    /// Latent columns.
    pub width: usize,
    /// Audio latents.
    pub audio_frames: usize,
    /// Video frame rate.
    pub fps: f32,
}

/// A resident LTX-2.3 transformer.
pub struct Ltx2Transformer {
    backend: Backend,
    cfg: Ltx2Config,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for Ltx2Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Transformer").field("device", &self.backend.name()).field("layers", &self.cfg.num_layers).finish_non_exhaustive()
    }
}

impl Ltx2Transformer {
    /// Load `transformer/` of a checkpoint.
    ///
    /// # Errors
    /// On an unsupported configuration, missing weights or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: Ltx2Config = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let backend = Backend::select(opts.cpu_threads)?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "audio-video backend selected");
        let st = SafeTensors::open(&files.weights("transformer")?)?;
        let w = Weights::load(&backend, &st, &cfg.weight_specs(opts.precision.wtype()))?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32 })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &Ltx2Config {
        &self.cfg
    }

    /// One evaluation: velocity for video tokens `[tokens][channels]` and
    /// audio tokens `[tokens][channels]`, given connector outputs for each
    /// stream `[tokens][width]` and the two streams' timesteps (already
    /// multiplied by the timestep scale).
    ///
    /// # Errors
    /// On mismatched input lengths or a backend failure.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, video: &[f32], audio: &[f32], text: &[f32], audio_text: &[f32], s: AvShape, t_video: f32, t_audio: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.cfg;
        let n_video = s.frames * s.height * s.width;
        let n_text = text.len() / cfg.inner() as usize;
        if video.len() != n_video * cfg.in_channels as usize
            || audio.len() != s.audio_frames * cfg.audio_in_channels as usize
            || audio_text.len() != n_text * cfg.audio_inner() as usize
            || n_text == 0
        {
            return Err(Error::Request("audio-video transformer inputs disagree with the shape".into()));
        }
        let mut g = Graph::new(&self.backend)?;
        let io = build(&mut g, cfg, &self.w, (n_video as i64, s.audio_frames as i64, n_text as i64), self.exact);
        g.finish(&[io.out_video, io.out_audio])?;
        let gate = cfg.cross_attn_timestep_scale_multiplier as f32 / cfg.timestep_scale_multiplier as f32;
        let mut time = S3DitConfig::time_features(t_video);
        for t in [t_audio, t_audio * gate, t_video * gate] {
            time.extend(S3DitConfig::time_features(t));
        }
        let rot = cfg.rotary(s.frames, s.height, s.width, s.audio_frames, s.fps);
        g.set_f32(io.video, video);
        g.set_f32(io.audio, audio);
        g.set_f32(io.text, text);
        g.set_f32(io.audio_text, audio_text);
        g.set_f32(io.time, &time);
        for (i, (cos, sin)) in [&rot.video, &rot.audio, &rot.cross_video, &rot.cross_audio].into_iter().enumerate() {
            g.set_f32(io.rope[2 * i], cos);
            g.set_f32(io.rope[2 * i + 1], sin);
        }
        g.compute()?;
        Ok((g.read_f32(io.out_video), g.read_f32(io.out_audio)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_meets_at_both_ends() {
        let l = linspace01(5);
        assert_eq!(l[0], 0.0);
        assert_eq!(l[4], 1.0);
        assert!((l[2] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn audio_times_start_causally_at_zero() {
        let cfg: Ltx2Config = serde_json::from_value(serde_json::json!({
            "in_channels": 128, "out_channels": 128, "patch_size": 1, "patch_size_t": 1,
            "num_attention_heads": 32, "attention_head_dim": 128, "cross_attention_dim": 4096,
            "vae_scale_factors": [8, 32, 32], "pos_embed_max_pos": 20, "base_height": 2048, "base_width": 2048,
            "gated_attn": true, "cross_attn_mod": true, "audio_in_channels": 128, "audio_out_channels": 128,
            "audio_patch_size": 1, "audio_patch_size_t": 1, "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64, "audio_cross_attention_dim": 2048, "audio_scale_factor": 4,
            "audio_pos_embed_max_pos": 20, "audio_sampling_rate": 16000, "audio_hop_length": 160,
            "audio_gated_attn": true, "audio_cross_attn_mod": true, "num_layers": 48,
            "activation_fn": "gelu-approximate", "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": false, "norm_eps": 1e-6, "rope_theta": 10000.0, "causal_offset": 1,
            "timestep_scale_multiplier": 1000, "cross_attn_timestep_scale_multiplier": 1000,
            "rope_type": "split", "use_prompt_embeddings": false
        }))
        .unwrap();
        cfg.validate().unwrap();
        let a = cfg.audio_coords(2);
        // First latent spans mel frames [0, 1), the second [1, 5).
        assert!((a[0] - 0.005).abs() < 1e-7);
        assert!((a[1] - 0.03).abs() < 1e-7);
        let v = cfg.video_coords(2, 1, 1, 24.0);
        assert!((v[0][0] - 1.0 / 48.0).abs() < 1e-7);
        assert!((v[1][0] - 16.0).abs() < 1e-7);
        let (cos, _) = cfg.rotary(1, 1, 1, 1, 24.0).video;
        // The leading pairs of the first head are padding.
        assert_eq!(cos[0], 1.0);
        assert_eq!(cos[1], 1.0);
        assert_eq!(cos.len(), 4096);
    }
}
