//! Video diffusion transformer over the 48-channel causal video latent.
//!
//! The latent is cut into `1 x 2 x 2` patches (one token each), embedded by
//! a linear map, and run through blocks of modulated self-attention with a
//! three-axis interleaved rotary embedding (frame, row, column), an
//! unmodulated cross-attention over the projected text states, and a
//! modulated tanh-GELU feed-forward. Six modulation vectors per block (shift,
//! scale and gate for attention and feed-forward) are a learned table plus a
//! projection of the timestep embedding; the timestep may differ per token,
//! which keeps a clean conditioning frame at time zero while the rest is
//! denoised.
//!
//! Host layout: a patch token is `[channel][dy][dx]` on input and
//! `[dy][dx][channel]` on output, tokens ordered frame, row, column.

use serde::Deserialize;

use crate::error::{Error, Result};
use llama_cpp_sys_2 as sys;

use crate::ggml::{Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::safetensors::SafeTensors;

/// Sinusoidal timestep features.
pub const FREQ_DIM: usize = 256;

/// Transformer hyperparameters (the checkpoint's `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct WanDitConfig {
    /// `patch_size` from the checkpoint configuration.
    pub patch_size: [u64; 3],
    /// `num_attention_heads` from the checkpoint configuration.
    pub num_attention_heads: u64,
    /// `attention_head_dim` from the checkpoint configuration.
    pub attention_head_dim: u64,
    /// `in_channels` from the checkpoint configuration.
    pub in_channels: u64,
    /// `out_channels` from the checkpoint configuration.
    pub out_channels: u64,
    /// `text_dim` from the checkpoint configuration.
    pub text_dim: u64,
    /// `freq_dim` from the checkpoint configuration.
    #[serde(default = "freq_dim")]
    pub freq_dim: u64,
    /// `ffn_dim` from the checkpoint configuration.
    pub ffn_dim: u64,
    /// `num_layers` from the checkpoint configuration.
    pub num_layers: u64,
    /// `cross_attn_norm` from the checkpoint configuration.
    #[serde(default = "yes")]
    pub cross_attn_norm: bool,
    /// `qk_norm` from the checkpoint configuration.
    #[serde(default)]
    pub qk_norm: Option<String>,
    /// `eps` from the checkpoint configuration.
    #[serde(default = "eps")]
    pub eps: f64,
    /// `image_dim` from the checkpoint configuration.
    #[serde(default)]
    pub image_dim: Option<u64>,
    /// `added_kv_proj_dim` from the checkpoint configuration.
    #[serde(default)]
    pub added_kv_proj_dim: Option<u64>,
    /// `rope_max_seq_len` from the checkpoint configuration.
    #[serde(default = "rope_len")]
    pub rope_max_seq_len: u64,
}

fn freq_dim() -> u64 {
    FREQ_DIM as u64
}
fn yes() -> bool {
    true
}
fn eps() -> f64 {
    1e-6
}
fn rope_len() -> u64 {
    1024
}

impl WanDitConfig {
    /// Model width.
    #[must_use]
    pub fn dim(&self) -> u64 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Values per input patch token.
    #[must_use]
    pub fn patch_in(&self) -> u64 {
        self.in_channels * self.patch_size.iter().product::<u64>()
    }

    /// Values per output patch token.
    #[must_use]
    pub fn patch_out(&self) -> u64 {
        self.out_channels * self.patch_size.iter().product::<u64>()
    }

    /// Refuse layouts this implementation does not carry.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size[0] != 1 {
            return Err(Error::Config("temporal patching is not supported".into()));
        }
        if self.freq_dim as usize != FREQ_DIM {
            return Err(Error::Config(format!("freq_dim {} (expected {FREQ_DIM})", self.freq_dim)));
        }
        if self.qk_norm.as_deref() != Some("rms_norm_across_heads") {
            return Err(Error::Config(format!("qk_norm {:?} is not supported", self.qk_norm)));
        }
        if !self.cross_attn_norm {
            return Err(Error::Config("cross_attn_norm false is not supported".into()));
        }
        if self.image_dim.is_some() || self.added_kv_proj_dim.is_some() {
            return Err(Error::Config("image-embedding cross-attention is not supported".into()));
        }
        if self.attention_head_dim % 2 != 0 || self.attention_head_dim < 6 {
            return Err(Error::Config("attention_head_dim must be even".into()));
        }
        Ok(())
    }

    /// Rotary pairs per axis (frame, row, column).
    #[must_use]
    pub fn rope_axes(&self) -> [usize; 3] {
        let hd = self.attention_head_dim as usize;
        let hw = 2 * (hd / 6);
        [(hd - 2 * hw) / 2, hw / 2, hw / 2]
    }

    /// Device weights, except the patch embedding (see [`patch_weights`]).
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, ff, td) = (self.dim(), self.ffn_dim, self.text_dim);
        let f = WType::F32;
        let ce = "condition_embedder";
        let mut v = vec![
            WeightSpec::new("patch_embedding.bias", &[d], f),
            WeightSpec::new(format!("{ce}.time_embedder.linear_1.weight"), &[d, self.freq_dim], f),
            WeightSpec::new(format!("{ce}.time_embedder.linear_1.bias"), &[d], f),
            WeightSpec::new(format!("{ce}.time_embedder.linear_2.weight"), &[d, d], f),
            WeightSpec::new(format!("{ce}.time_embedder.linear_2.bias"), &[d], f),
            WeightSpec::new(format!("{ce}.time_proj.weight"), &[6 * d, d], f),
            WeightSpec::new(format!("{ce}.time_proj.bias"), &[6 * d], f),
            WeightSpec::new(format!("{ce}.text_embedder.linear_1.weight"), &[d, td], linear),
            WeightSpec::new(format!("{ce}.text_embedder.linear_1.bias"), &[d], f),
            WeightSpec::new(format!("{ce}.text_embedder.linear_2.weight"), &[d, d], linear),
            WeightSpec::new(format!("{ce}.text_embedder.linear_2.bias"), &[d], f),
            WeightSpec::new("scale_shift_table", &[1, 2, d], f),
            WeightSpec::new("proj_out.weight", &[self.patch_out(), d], f),
            WeightSpec::new("proj_out.bias", &[self.patch_out()], f),
        ];
        for i in 0..self.num_layers {
            let p = format!("blocks.{i}");
            for a in ["attn1", "attn2"] {
                for m in ["to_q", "to_k", "to_v", "to_out.0"] {
                    v.push(WeightSpec::new(format!("{p}.{a}.{m}.weight"), &[d, d], linear));
                    v.push(WeightSpec::new(format!("{p}.{a}.{m}.bias"), &[d], f));
                }
                v.push(WeightSpec::new(format!("{p}.{a}.norm_q.weight"), &[d], f));
                v.push(WeightSpec::new(format!("{p}.{a}.norm_k.weight"), &[d], f));
            }
            v.push(WeightSpec::new(format!("{p}.norm2.weight"), &[d], f));
            v.push(WeightSpec::new(format!("{p}.norm2.bias"), &[d], f));
            v.push(WeightSpec::new(format!("{p}.ffn.net.0.proj.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.ffn.net.0.proj.bias"), &[ff], f));
            v.push(WeightSpec::new(format!("{p}.ffn.net.2.weight"), &[d, ff], linear));
            v.push(WeightSpec::new(format!("{p}.ffn.net.2.bias"), &[d], f));
            v.push(WeightSpec::new(format!("{p}.scale_shift_table"), &[1, 6, d], f));
        }
        v
    }

    /// The patch embedding as a matrix `[dim, channels * dy * dx]` (the
    /// file holds it as a five-dimensional convolution kernel).
    pub fn patch_weights(&self, files: &SafeTensors) -> Result<Vec<HostTensor>> {
        let [pt, ph, pw] = self.patch_size;
        let w = files.require("patch_embedding.weight", &[self.dim(), self.in_channels, pt, ph, pw])?.to_f32();
        Ok(vec![HostTensor {
            name: "patch_embedding.weight".into(),
            shape: vec![self.dim(), self.patch_in()],
            ty: WType::F32,
            data: w,
        }])
    }

    /// Interleaved rotary tables `[pairs]` per token for a latent grid of
    /// `frames x rows x cols` patches, in token order.
    #[must_use]
    pub fn rotary_tables(&self, frames: usize, rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        self.rotary_tables_at(&(0..frames).collect::<Vec<_>>(), rows, cols)
    }

    /// [`Self::rotary_tables`] with an explicit time position per latent
    /// frame (frames kept from earlier in a stream sit at their own
    /// positions, not at their place in the sequence).
    #[must_use]
    pub fn rotary_tables_at(&self, positions: &[usize], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let axes = self.rope_axes();
        let inv: Vec<Vec<f64>> = axes
            .iter()
            .map(|&p| (0..p).map(|j| 1.0 / 10000f64.powf((2 * j) as f64 / (2 * p) as f64)).collect())
            .collect();
        let half = axes.iter().sum::<usize>();
        let n = positions.len() * rows * cols;
        let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
        for &f in positions {
            for r in 0..rows {
                for c in 0..cols {
                    for (axis, pos) in [f, r, c].into_iter().enumerate() {
                        for w in &inv[axis] {
                            let a = pos as f64 * w;
                            cos.push(a.cos() as f32);
                            sin.push(a.sin() as f32);
                        }
                    }
                }
            }
        }
        (cos, sin)
    }
}

/// Sinusoidal features of a (possibly fractional) timestep: `cos` then `sin`.
#[must_use]
pub fn time_features(t: f32) -> Vec<f32> {
    let half = FREQ_DIM / 2;
    let k = -(10000f64.ln());
    let freqs: Vec<f64> = (0..half).map(|i| (k * i as f64 / half as f64).exp()).collect();
    let t = f64::from(t);
    let mut out: Vec<f32> = freqs.iter().map(|f| (t * f).cos() as f32).collect();
    out.extend(freqs.iter().map(|f| (t * f).sin() as f32));
    out
}

/// Cut a latent `[channel][frame][row][col]` into patch tokens.
#[must_use]
pub fn patchify(cfg: &WanDitConfig, latent: &[f32], frames: usize, h: usize, w: usize) -> Vec<f32> {
    let (c, ph, pw) = (cfg.in_channels as usize, cfg.patch_size[1] as usize, cfg.patch_size[2] as usize);
    let (rows, cols) = (h / ph, w / pw);
    let mut out = Vec::with_capacity(latent.len());
    for f in 0..frames {
        for r in 0..rows {
            for col in 0..cols {
                for ch in 0..c {
                    for dy in 0..ph {
                        for dx in 0..pw {
                            out.push(latent[((ch * frames + f) * h + r * ph + dy) * w + col * pw + dx]);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Reassemble output patch tokens into a latent `[channel][frame][row][col]`.
#[must_use]
pub fn unpatchify(cfg: &WanDitConfig, tokens: &[f32], frames: usize, h: usize, w: usize) -> Vec<f32> {
    let (c, ph, pw) = (cfg.out_channels as usize, cfg.patch_size[1] as usize, cfg.patch_size[2] as usize);
    let (rows, cols) = (h / ph, w / pw);
    let mut out = vec![0f32; c * frames * h * w];
    let mut i = 0;
    for f in 0..frames {
        for r in 0..rows {
            for col in 0..cols {
                for dy in 0..ph {
                    for dx in 0..pw {
                        for ch in 0..c {
                            out[((ch * frames + f) * h + r * ph + dy) * w + col * pw + dx] = tokens[i];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inputs and output of one transformer evaluation.
#[derive(Debug, Clone, Copy)]
pub struct DitIo {
    /// Patch tokens `[patch_in, tokens]`.
    pub patches: Tn,
    /// Timestep features `[FREQ_DIM, 1]`, or `[FREQ_DIM, tokens]` per token.
    pub time: Tn,
    /// Text encoder states `[text_dim, text tokens]`.
    pub context: Tn,
    /// Rotary tables `[1, pairs, 1, tokens]`.
    pub cos: Tn,
    /// Sine table, like `cos`.
    pub sin: Tn,
    /// Predicted velocity `[patch_out, tokens]`.
    pub out: Tn,
}

fn lin(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{p}.weight")), w.get(&format!("{p}.bias")), x)
}

/// Row `k` of a `[d, 6, m]` modulation tensor, as `[d, m]`.
fn chunk(g: &mut Graph, m: Tn, k: usize) -> Tn {
    let v = g.view_4d(m, [m.ne(0), m.ne(2), 1, 1], m.nb(2), m.nb(3), m.nb(3), k * m.nb(1));
    g.cont(v)
}

/// `norm(x) * (1 + scale) + shift`.
fn modulate(g: &mut Graph, x: Tn, shift: Tn, scale: Tn, eps: f32) -> Tn {
    let h = g.norm(x, eps);
    let s = g.scale_bias(scale, 1.0, 1.0);
    let h = g.mul(h, s);
    g.add(h, shift)
}

/// Rotate adjacent pairs of `x` `[hd, heads, n]` by the tables `[1, hd/2, 1, n]`.
fn rope_pairs(g: &mut Graph, x: Tn, cos: Tn, sin: Tn) -> Tn {
    let (hd, heads, n) = (x.ne(0), x.ne(1), x.ne(2));
    let half = hd / 2;
    let x = g.reshape(x, &[2, half, heads, n]);
    let es = x.nb(0);
    let a = g.view_4d(x, [1, half, heads, n], x.nb(1), x.nb(2), x.nb(3), 0);
    let b = g.view_4d(x, [1, half, heads, n], x.nb(1), x.nb(2), x.nb(3), es);
    let a = g.cont(a);
    let b = g.cont(b);
    let ac = g.mul(a, cos);
    let bs = g.mul(b, sin);
    let even = g.sub(ac, bs);
    let as_ = g.mul(a, sin);
    let bc = g.mul(b, cos);
    let odd = g.add(as_, bc);
    let r = g.concat(even, odd, 0);
    g.reshape(r, &[hd, heads, n])
}

/// `q` `[hd, heads, n]`, `k`/`v` `[hd, heads, m]`; result `[hd * heads, n]`.
fn attend(g: &mut Graph, q: Tn, k: Tn, v: Tn, exact: bool) -> Tn {
    let (hd, heads, n) = (q.ne(0), q.ne(1), q.ne(2));
    let scale = 1.0 / (hd as f32).sqrt();
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let k = g.cont(k);
    let v = g.permute(v, [0, 2, 1, 3]);
    let v = g.cont(v);
    let o = if exact {
        g.attention_exact(q, k, v, None, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, None, scale, true)
    };
    g.reshape(o, &[hd * heads, n])
}

#[allow(clippy::too_many_arguments)]
fn attention(g: &mut Graph, cfg: &WanDitConfig, w: &Weights, p: &str, x: Tn, ctx: Tn, rope: Option<(Tn, Tn)>, exact: bool) -> Tn {
    let (hd, heads) = (cfg.attention_head_dim as i64, cfg.num_attention_heads as i64);
    let eps = cfg.eps as f32;
    let q = lin(g, w, &format!("{p}.to_q"), x);
    let k = lin(g, w, &format!("{p}.to_k"), ctx);
    let v = lin(g, w, &format!("{p}.to_v"), ctx);
    let q = g.rms_norm(q, eps);
    let q = g.mul(q, w.get(&format!("{p}.norm_q.weight")));
    let k = g.rms_norm(k, eps);
    let k = g.mul(k, w.get(&format!("{p}.norm_k.weight")));
    let (n, m) = (q.ne(1), k.ne(1));
    let mut q = g.reshape(q, &[hd, heads, n]);
    let mut k = g.reshape(k, &[hd, heads, m]);
    let v = g.reshape(v, &[hd, heads, m]);
    if let Some((cos, sin)) = rope {
        q = rope_pairs(g, q, cos, sin);
        k = rope_pairs(g, k, cos, sin);
    }
    let o = attend(g, q, k, v, exact);
    lin(g, w, &format!("{p}.to_out.0"), o)
}

/// Build one denoising evaluation over `tokens` patches, `text` context
/// tokens, and `time_tokens` timesteps (1, or `tokens` for per-token time).
/// `pe` holds [`WanDitConfig::patch_weights`].
#[allow(clippy::too_many_arguments)]
pub fn build(g: &mut Graph, cfg: &WanDitConfig, w: &Weights, pe: &Weights, tokens: i64, text: i64, time_tokens: i64, exact: bool) -> DitIo {
    let d = cfg.dim() as i64;
    let eps = cfg.eps as f32;
    let half = (cfg.attention_head_dim / 2) as i64;
    let patches = g.input(sys::GGML_TYPE_F32, &[cfg.patch_in() as i64, tokens]);
    let time = g.input(sys::GGML_TYPE_F32, &[FREQ_DIM as i64, time_tokens]);
    let context = g.input(sys::GGML_TYPE_F32, &[cfg.text_dim as i64, text]);
    let cos = g.input(sys::GGML_TYPE_F32, &[1, half, 1, tokens]);
    let sin = g.input(sys::GGML_TYPE_F32, &[1, half, 1, tokens]);

    let x = g.linear(pe.get("patch_embedding.weight"), patches);
    let mut x = g.add(x, w.get("patch_embedding.bias"));

    let ce = "condition_embedder";
    let t = lin(g, w, &format!("{ce}.time_embedder.linear_1"), time);
    let t = g.silu(t);
    let temb = lin(g, w, &format!("{ce}.time_embedder.linear_2"), t);
    let t = g.silu(temb);
    let proj = lin(g, w, &format!("{ce}.time_proj"), t);
    let proj = g.reshape(proj, &[d, 6, time_tokens]);

    let c = lin(g, w, &format!("{ce}.text_embedder.linear_1"), context);
    let c = if exact { g.gelu_tanh_exact(c) } else { g.gelu_tanh(c) };
    let ctx = lin(g, w, &format!("{ce}.text_embedder.linear_2"), c);

    for i in 0..cfg.num_layers {
        let p = format!("blocks.{i}");
        let table = g.reshape(w.get(&format!("{p}.scale_shift_table")), &[d, 6, 1]);
        let m = g.add(proj, table);
        let [shift, scale, gate, c_shift, c_scale, c_gate] = [0, 1, 2, 3, 4, 5].map(|k| chunk(g, m, k));

        let h = modulate(g, x, shift, scale, eps);
        let a = attention(g, cfg, w, &format!("{p}.attn1"), h, h, Some((cos, sin)), exact);
        let a = g.mul(a, gate);
        x = g.add(x, a);

        let h = g.norm(x, eps);
        let h = g.mul(h, w.get(&format!("{p}.norm2.weight")));
        let h = g.add(h, w.get(&format!("{p}.norm2.bias")));
        let a = attention(g, cfg, w, &format!("{p}.attn2"), h, ctx, None, exact);
        x = g.add(x, a);

        let h = modulate(g, x, c_shift, c_scale, eps);
        let f = lin(g, w, &format!("{p}.ffn.net.0.proj"), h);
        let f = if exact { g.gelu_tanh_exact(f) } else { g.gelu_tanh(f) };
        let f = lin(g, w, &format!("{p}.ffn.net.2"), f);
        let f = g.mul(f, c_gate);
        x = g.add(x, f);
    }

    let table = g.reshape(w.get("scale_shift_table"), &[d, 2, 1]);
    let temb = g.reshape(temb, &[d, 1, time_tokens]);
    let temb = g.repeat_to(temb, [d, 2, time_tokens, 1]);
    let m = g.add(temb, table);
    let v = g.view_4d(m, [d, time_tokens, 1, 1], m.nb(2), m.nb(3), m.nb(3), 0);
    let shift = g.cont(v);
    let v = g.view_4d(m, [d, time_tokens, 1, 1], m.nb(2), m.nb(3), m.nb(3), m.nb(1));
    let scale = g.cont(v);
    let h = modulate(g, x, shift, scale, eps);
    let out = lin(g, w, "proj_out", h);
    DitIo { patches, time, context, cos, sin, out }
}

#[cfg(test)]
mod parity;
