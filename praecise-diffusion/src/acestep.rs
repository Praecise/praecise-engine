//! ACE-Step 1.5 networks: the condition encoder (lyric and timbre encoders and
//! the text projection) and the flow-matching diffusion transformer over 1D
//! audio latents.
//!
//! Sequences are laid out `[width, positions]`, one position per column. The
//! transformer's self-attention alternates between a sliding window of
//! `sliding_window` positions each side and the full sequence; its
//! cross-attention reads the packed conditioning sequence.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// ggml's rotary mode for split-half (NeoX) rotation, `GGML_ROPE_TYPE_NEOX`.
const ROPE_NEOX: i32 = 2;
/// Width of the sinusoidal timestep features.
pub const TIME_FEATURES: usize = 256;
/// Timesteps in `[0, 1]` are scaled by this before the sinusoid.
const TIME_SCALE: f32 = 1000.0;

fn default_window() -> u64 {
    128
}

/// Per-layer attention span, from a config's `layer_types` or the default
/// alternation (sliding first).
fn sliding_layers(types: Option<&[String]>, n: usize) -> Result<Vec<bool>> {
    match types {
        None => Ok((0..n).map(|i| i % 2 == 0).collect()),
        Some(t) if t.len() == n => t
            .iter()
            .map(|s| match s.as_str() {
                "sliding_attention" => Ok(true),
                "full_attention" => Ok(false),
                other => Err(Error::Config(format!("layer type {other:?} is not implemented"))),
            })
            .collect(),
        Some(t) => Err(Error::Config(format!("{} layer types for {n} layers", t.len()))),
    }
}

/// Transformer configuration, read from `transformer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    /// Model width.
    pub hidden_size: u64,
    /// Feed-forward width.
    pub intermediate_size: u64,
    /// Blocks.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Key/value heads.
    pub num_key_value_heads: u64,
    /// Head width.
    pub head_dim: u64,
    /// Channels per latent frame entering the patch embedding: source
    /// latents, chunk mask and noisy latents.
    pub in_channels: u64,
    /// Channels of one audio latent frame.
    pub audio_acoustic_hidden_dim: u64,
    /// Latent frames per transformer position.
    pub patch_size: u64,
    /// Rotary base.
    pub rope_theta: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Half-width of the sliding attention window.
    #[serde(default = "default_window")]
    pub sliding_window: u64,
    /// Attention span of each block.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Width of the conditioning sequence.
    #[serde(default)]
    pub encoder_hidden_size: Option<u64>,
    /// Guidance-distilled (guidance has no effect).
    #[serde(default)]
    pub is_turbo: bool,
    /// Release name of the weights.
    #[serde(default)]
    pub model_version: Option<String>,
    /// Biases on the attention projections.
    #[serde(default)]
    pub attention_bias: bool,
}

impl DitConfig {
    /// Check the configuration describes a network this module builds.
    ///
    /// # Errors
    /// [`Error::Config`] naming the first unsupported setting.
    pub fn validate(&self) -> Result<()> {
        if self.attention_bias {
            return Err(Error::Config("attention biases are not implemented".into()));
        }
        if self.in_channels != 3 * self.audio_acoustic_hidden_dim {
            return Err(Error::Config("transformer input is not source, mask and noisy latents".into()));
        }
        if self.patch_size == 0 || self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::Config("transformer patch or head layout is not supported".into()));
        }
        self.sliding()?;
        Ok(())
    }

    /// Whether each block's self-attention is windowed.
    ///
    /// # Errors
    /// An unknown or miscounted layer type.
    pub fn sliding(&self) -> Result<Vec<bool>> {
        sliding_layers(self.layer_types.as_deref(), self.num_hidden_layers)
    }

    /// Guidance-distilled weights.
    #[must_use]
    pub fn turbo(&self) -> bool {
        self.is_turbo || self.model_version.as_deref() == Some("turbo")
    }

    /// Width of the conditioning sequence.
    #[must_use]
    pub fn context_width(&self) -> u64 {
        self.encoder_hidden_size.unwrap_or(self.hidden_size)
    }

    /// Weights read directly from the files.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size;
        let ff = self.intermediate_size;
        let mut v = Vec::new();
        for e in ["time_embed", "time_embed_r"] {
            v.push(WeightSpec::new(format!("{e}.linear_1.weight"), &[d, TIME_FEATURES as u64], WType::F32));
            v.push(WeightSpec::new(format!("{e}.linear_1.bias"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{e}.linear_2.weight"), &[d, d], linear));
            v.push(WeightSpec::new(format!("{e}.linear_2.bias"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{e}.time_proj.weight"), &[6 * d, d], linear));
            v.push(WeightSpec::new(format!("{e}.time_proj.bias"), &[6 * d], WType::F32));
        }
        v.push(WeightSpec::new("condition_embedder.weight", &[d, self.context_width()], linear));
        v.push(WeightSpec::new("condition_embedder.bias", &[d], WType::F32));
        v.push(WeightSpec::new("proj_in_conv.bias", &[d], WType::F32));
        v.push(WeightSpec::new("proj_out_conv.bias", &[self.audio_acoustic_hidden_dim], WType::F32));
        v.push(WeightSpec::new("norm_out.weight", &[d], WType::F32));
        v.push(WeightSpec::new("scale_shift_table", &[1, 2, d], WType::F32));
        for i in 0..self.num_hidden_layers {
            let p = format!("layers.{i}");
            v.push(WeightSpec::new(format!("{p}.scale_shift_table"), &[1, 6, d], WType::F32));
            for n in ["self_attn_norm", "cross_attn_norm", "mlp_norm"] {
                v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[d], WType::F32));
            }
            for a in ["self_attn", "cross_attn"] {
                attention_specs(&mut v, &format!("{p}.{a}"), d, self.num_attention_heads, self.num_key_value_heads, self.head_dim, linear);
            }
            mlp_specs(&mut v, &p, d, ff, linear);
        }
        v
    }

    /// The patch embedding and its inverse, re-laid out at load so each is
    /// one matrix product over a position's frames.
    ///
    /// The input convolution's kernel `[d, c, p]` becomes `[d, p * c]`
    /// (frame-major, matching `p` consecutive `c`-wide frames); the output
    /// transposed convolution's `[d, a, p]` becomes `[p * a, d]`.
    ///
    /// # Errors
    /// Missing or mis-shaped tensors.
    pub fn host_tensors(&self, files: &SafeTensors, linear: WType) -> Result<Vec<HostTensor>> {
        let d = self.hidden_size as usize;
        let c = self.in_channels as usize;
        let a = self.audio_acoustic_hidden_dim as usize;
        let p = self.patch_size as usize;
        let w_in = files.require("proj_in_conv.weight", &[d as u64, c as u64, p as u64])?.to_f32();
        let mut win = vec![0f32; d * p * c];
        for o in 0..d {
            for ch in 0..c {
                for k in 0..p {
                    win[o * p * c + k * c + ch] = w_in[(o * c + ch) * p + k];
                }
            }
        }
        let w_out = files.require("proj_out_conv.weight", &[d as u64, a as u64, p as u64])?.to_f32();
        let mut wout = vec![0f32; p * a * d];
        for i in 0..d {
            for ch in 0..a {
                for k in 0..p {
                    wout[(k * a + ch) * d + i] = w_out[(i * a + ch) * p + k];
                }
            }
        }
        // The patch matrices are small; keep them exact unless the caller
        // asked for 8-bit, whose rows must be multiples of 32.
        let ty = |cols: usize| if linear == WType::Q8_0 && cols % 32 != 0 { WType::F32 } else { linear };
        Ok(vec![
            HostTensor { name: "proj_in.weight".into(), shape: vec![d as u64, (p * c) as u64], ty: ty(p * c), data: win },
            HostTensor { name: "proj_out.weight".into(), shape: vec![(p * a) as u64, d as u64], ty: ty(d), data: wout },
        ])
    }
}

/// Condition-encoder configuration, read from `condition_encoder/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ConditionConfig {
    /// Encoder width (the conditioning sequence's width).
    pub hidden_size: u64,
    /// Feed-forward width.
    pub intermediate_size: u64,
    /// Width of the text encoder's hidden states.
    pub text_hidden_dim: u64,
    /// Channels of one reference-audio latent frame.
    pub timbre_hidden_dim: u64,
    /// Lyric encoder layers.
    pub num_lyric_encoder_hidden_layers: usize,
    /// Timbre encoder layers.
    pub num_timbre_encoder_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Key/value heads.
    pub num_key_value_heads: u64,
    /// Head width.
    pub head_dim: u64,
    /// Rotary base.
    pub rope_theta: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Half-width of the sliding attention window.
    #[serde(default = "default_window")]
    pub sliding_window: u64,
    /// Attention span of each lyric-encoder layer.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Biases on the attention projections.
    #[serde(default)]
    pub attention_bias: bool,
}

impl ConditionConfig {
    /// Check the configuration describes a network this module builds.
    ///
    /// # Errors
    /// [`Error::Config`] naming the first unsupported setting.
    pub fn validate(&self) -> Result<()> {
        if self.attention_bias {
            return Err(Error::Config("attention biases are not implemented".into()));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::Config("condition encoder head layout is not supported".into()));
        }
        self.lyric_sliding()?;
        Ok(())
    }

    fn lyric_sliding(&self) -> Result<Vec<bool>> {
        sliding_layers(self.layer_types.as_deref(), self.num_lyric_encoder_hidden_layers)
    }

    fn timbre_sliding(&self) -> Vec<bool> {
        (0..self.num_timbre_encoder_hidden_layers).map(|i| i % 2 == 0).collect()
    }

    /// Every weight the encoder reads (the timbre encoder's unused special
    /// token and the host-side buffers excepted).
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size;
        let mut v = vec![
            WeightSpec::new("text_projector.weight", &[d, self.text_hidden_dim], linear),
            WeightSpec::new("lyric_encoder.embed_tokens.weight", &[d, self.text_hidden_dim], linear),
            WeightSpec::new("lyric_encoder.embed_tokens.bias", &[d], WType::F32),
            WeightSpec::new("lyric_encoder.norm.weight", &[d], WType::F32),
            WeightSpec::new("timbre_encoder.embed_tokens.weight", &[d, self.timbre_hidden_dim], WType::F32),
            WeightSpec::new("timbre_encoder.embed_tokens.bias", &[d], WType::F32),
            WeightSpec::new("timbre_encoder.norm.weight", &[d], WType::F32),
        ];
        for (enc, n) in [("lyric_encoder", self.num_lyric_encoder_hidden_layers), ("timbre_encoder", self.num_timbre_encoder_hidden_layers)] {
            for i in 0..n {
                let p = format!("{enc}.layers.{i}");
                v.push(WeightSpec::new(format!("{p}.input_layernorm.weight"), &[d], WType::F32));
                v.push(WeightSpec::new(format!("{p}.post_attention_layernorm.weight"), &[d], WType::F32));
                attention_specs(&mut v, &format!("{p}.self_attn"), d, self.num_attention_heads, self.num_key_value_heads, self.head_dim, linear);
                mlp_specs(&mut v, &p, d, self.intermediate_size, linear);
            }
        }
        v
    }
}

fn attention_specs(v: &mut Vec<WeightSpec>, p: &str, d: u64, heads: u64, kv: u64, hd: u64, linear: WType) {
    v.push(WeightSpec::new(format!("{p}.to_q.weight"), &[heads * hd, d], linear));
    v.push(WeightSpec::new(format!("{p}.to_k.weight"), &[kv * hd, d], linear));
    v.push(WeightSpec::new(format!("{p}.to_v.weight"), &[kv * hd, d], linear));
    v.push(WeightSpec::new(format!("{p}.to_out.0.weight"), &[d, heads * hd], linear));
    v.push(WeightSpec::new(format!("{p}.norm_q.weight"), &[hd], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm_k.weight"), &[hd], WType::F32));
}

fn mlp_specs(v: &mut Vec<WeightSpec>, p: &str, d: u64, ff: u64, linear: WType) {
    v.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[ff, d], linear));
    v.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[ff, d], linear));
    v.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[d, ff], linear));
}

/// Head layout shared by every attention in one network.
#[derive(Debug, Clone, Copy)]
struct Heads {
    heads: i64,
    kv: i64,
    dim: i64,
    eps: f32,
    theta: f32,
    /// Float32 scores, probabilities and values instead of the fused kernel's
    /// half-precision keys and values.
    exact: bool,
}

/// Self-attention when `context` is `None` (rotary on queries and keys),
/// cross-attention over `context` otherwise.
fn attention(g: &mut Graph, w: &Weights, p: &str, h: Heads, x: Tn, context: Option<Tn>, positions: Tn, mask: Option<Tn>) -> Tn {
    let wn = |s: &str| w.get(&format!("{p}.{s}"));
    let n = x.ne(1);
    let src = context.unwrap_or(x);
    let m = src.ne(1);
    let q = g.linear(wn("to_q.weight"), x);
    let k = g.linear(wn("to_k.weight"), src);
    let v = g.linear(wn("to_v.weight"), src);
    let q = g.reshape(q, &[h.dim, h.heads, n]);
    let k = g.reshape(k, &[h.dim, h.kv, m]);
    let v = g.reshape(v, &[h.dim, h.kv, m]);
    let q = g.rms_norm(q, h.eps);
    let mut q = g.mul(q, wn("norm_q.weight"));
    let k = g.rms_norm(k, h.eps);
    let mut k = g.mul(k, wn("norm_k.weight"));
    if context.is_none() {
        q = g.rope(q, positions, h.dim as i32, ROPE_NEOX, h.theta);
        k = g.rope(k, positions, h.dim as i32, ROPE_NEOX, h.theta);
    }
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    let o = if h.exact {
        g.attention_exact(q, k, v, mask, 1.0 / (h.dim as f32).sqrt())
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, mask, 1.0 / (h.dim as f32).sqrt(), true)
    };
    let o = g.reshape(o, &[h.dim * h.heads, n]);
    g.linear(wn("to_out.0.weight"), o)
}

fn mlp(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let gate = g.linear(w.get(&format!("{p}.mlp.gate_proj.weight")), x);
    let up = g.linear(w.get(&format!("{p}.mlp.up_proj.weight")), x);
    let m = g.swiglu_split(gate, up);
    g.linear(w.get(&format!("{p}.mlp.down_proj.weight")), m)
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    g.mul(h, w.get(name))
}

/// The additive attention mask over `n` positions: `|i - j| <= window` when
/// `window` is given, everything otherwise. Row `i` is query `i`.
#[must_use]
pub fn window_mask(n: usize, window: Option<usize>) -> Vec<f32> {
    let mut m = vec![0f32; n * n];
    if let Some(wd) = window {
        for q in 0..n {
            for k in 0..n {
                if q.abs_diff(k) > wd {
                    m[q * n + k] = f32::NEG_INFINITY;
                }
            }
        }
    }
    m
}

/// A pre-norm encoder stack (lyric or timbre encoder) over `x`.
#[allow(clippy::too_many_arguments)]
fn encoder_stack(
    g: &mut Graph,
    w: &Weights,
    enc: &str,
    h: Heads,
    sliding: &[bool],
    x: Tn,
    positions: Tn,
    mask: Tn,
) -> Tn {
    let mut x = x;
    for (i, s) in sliding.iter().enumerate() {
        let p = format!("{enc}.layers.{i}");
        let n = rms(g, w, &format!("{p}.input_layernorm.weight"), x, h.eps);
        let a = attention(g, w, &format!("{p}.self_attn"), h, n, None, positions, s.then_some(mask));
        x = g.add(x, a);
        let n = rms(g, w, &format!("{p}.post_attention_layernorm.weight"), x, h.eps);
        let m = mlp(g, w, &p, n);
        x = g.add(x, m);
    }
    rms(g, w, &format!("{enc}.norm.weight"), x, h.eps)
}

/// Graph inputs and output of one condition encode.
#[derive(Debug, Clone, Copy)]
pub struct ConditionIo {
    /// Text-encoder hidden states `[text_hidden_dim, text tokens]`.
    pub text: Tn,
    /// Lyric token embeddings `[text_hidden_dim, lyric tokens]`.
    pub lyrics: Tn,
    /// Lyric positions `[lyric tokens]`.
    pub lyric_positions: Tn,
    /// Lyric sliding-window mask `[lyric tokens, lyric tokens]` (f16).
    pub lyric_mask: Tn,
    /// Reference-audio latents `[timbre_hidden_dim, frames]`.
    pub timbre: Tn,
    /// Timbre positions `[frames]`.
    pub timbre_positions: Tn,
    /// Timbre sliding-window mask `[frames, frames]` (f16).
    pub timbre_mask: Tn,
    /// The packed conditioning sequence `[hidden, lyric tokens + 1 + text
    /// tokens]`: lyrics, then the timbre embedding, then the text.
    pub out: Tn,
}

/// Build one condition encode for a single request (no padding, so every
/// token is real and packing is concatenation). `exact` computes attention
/// entirely in float32.
///
/// # Errors
/// A malformed layer-type list.
#[allow(clippy::too_many_arguments)]
pub fn build_condition(
    g: &mut Graph,
    cfg: &ConditionConfig,
    w: &Weights,
    text_tokens: i64,
    lyric_tokens: i64,
    frames: i64,
    exact: bool,
) -> Result<ConditionIo> {
    let h = Heads {
        heads: cfg.num_attention_heads as i64,
        kv: cfg.num_key_value_heads as i64,
        dim: cfg.head_dim as i64,
        eps: cfg.rms_norm_eps as f32,
        theta: cfg.rope_theta as f32,
        exact,
    };
    let text = g.input(sys::GGML_TYPE_F32, &[cfg.text_hidden_dim as i64, text_tokens]);
    let lyrics = g.input(sys::GGML_TYPE_F32, &[cfg.text_hidden_dim as i64, lyric_tokens]);
    let lyric_positions = g.input(sys::GGML_TYPE_I32, &[lyric_tokens]);
    let lyric_mask = g.input(sys::GGML_TYPE_F16, &[lyric_tokens, lyric_tokens]);
    let timbre = g.input(sys::GGML_TYPE_F32, &[cfg.timbre_hidden_dim as i64, frames]);
    let timbre_positions = g.input(sys::GGML_TYPE_I32, &[frames]);
    let timbre_mask = g.input(sys::GGML_TYPE_F16, &[frames, frames]);

    let t = g.linear(w.get("text_projector.weight"), text);
    let l = g.linear_b(w.get("lyric_encoder.embed_tokens.weight"), w.get("lyric_encoder.embed_tokens.bias"), lyrics);
    let l = encoder_stack(g, w, "lyric_encoder", h, &cfg.lyric_sliding()?, l, lyric_positions, lyric_mask);
    let b = g.linear_b(w.get("timbre_encoder.embed_tokens.weight"), w.get("timbre_encoder.embed_tokens.bias"), timbre);
    let b = encoder_stack(g, w, "timbre_encoder", h, &cfg.timbre_sliding(), b, timbre_positions, timbre_mask);
    let b = g.view_cols(b, 0, 1);
    let b = g.cont(b);
    let out = g.concat(l, b, 1);
    let out = g.concat(out, t, 1);
    Ok(ConditionIo { text, lyrics, lyric_positions, lyric_mask, timbre, timbre_positions, timbre_mask, out })
}

/// Graph inputs and output of one transformer evaluation.
#[derive(Debug, Clone, Copy)]
pub struct DitIo {
    /// Frames `[in_channels * patch_size, positions]`: per frame the source
    /// latents, the chunk mask and the noisy latents, frames zero-padded to
    /// a whole number of positions.
    pub frames: Tn,
    /// Sinusoidal features of `t` `[256]`.
    pub t_features: Tn,
    /// Sinusoidal features of `t - r` `[256]`.
    pub r_features: Tn,
    /// Conditioning sequence `[context width, tokens]`.
    pub context: Tn,
    /// Positions `[positions]`.
    pub positions: Tn,
    /// Sliding-window mask `[positions, positions]` (f16).
    pub mask: Tn,
    /// Predicted velocity `[acoustic, positions * patch_size]` (the padded
    /// frames included).
    pub out: Tn,
}

/// The sinusoidal timestep features: `cos` then `sin` of `t * 1000` at 128
/// log-spaced frequencies.
#[must_use]
pub fn time_features(t: f32) -> Vec<f32> {
    let half = TIME_FEATURES / 2;
    let arg = t * TIME_SCALE;
    let freqs: Vec<f32> =
        (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
    let mut out: Vec<f32> = freqs.iter().map(|f| (arg * f).cos()).collect();
    out.extend(freqs.iter().map(|f| (arg * f).sin()));
    out
}

fn time_embed(g: &mut Graph, w: &Weights, e: &str, features: Tn) -> (Tn, Tn) {
    let wn = |s: &str| w.get(&format!("{e}.{s}"));
    let h = g.linear_b(wn("linear_1.weight"), wn("linear_1.bias"), features);
    let h = g.silu(h);
    let temb = g.linear_b(wn("linear_2.weight"), wn("linear_2.bias"), h);
    let a = g.silu(temb);
    let proj = g.linear_b(wn("time_proj.weight"), wn("time_proj.bias"), a);
    (temb, proj)
}

/// `norm * (1 + scale) + shift`.
fn modulate(g: &mut Graph, x: Tn, shift: Tn, scale: Tn) -> Tn {
    let s = g.scale_bias(scale, 1.0, 1.0);
    let y = g.mul(x, s);
    g.add(y, shift)
}

/// Build one transformer evaluation over `positions` patch positions and
/// `tokens` conditioning tokens. `exact` computes attention entirely in
/// float32.
///
/// # Errors
/// A malformed layer-type list.
#[allow(clippy::too_many_arguments)]
pub fn build_dit(
    g: &mut Graph,
    cfg: &DitConfig,
    w: &Weights,
    conv: &Weights,
    positions: i64,
    tokens: i64,
    exact: bool,
) -> Result<DitIo> {
    let d = cfg.hidden_size as i64;
    let p = cfg.patch_size as i64;
    let a = cfg.audio_acoustic_hidden_dim as i64;
    let h = Heads {
        heads: cfg.num_attention_heads as i64,
        kv: cfg.num_key_value_heads as i64,
        dim: cfg.head_dim as i64,
        eps: cfg.rms_norm_eps as f32,
        theta: cfg.rope_theta as f32,
        exact,
    };
    let frames = g.input(sys::GGML_TYPE_F32, &[cfg.in_channels as i64 * p, positions]);
    let t_features = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
    let r_features = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
    let context = g.input(sys::GGML_TYPE_F32, &[cfg.context_width() as i64, tokens]);
    let pos = g.input(sys::GGML_TYPE_I32, &[positions]);
    let mask = g.input(sys::GGML_TYPE_F16, &[positions, positions]);

    let (temb_t, proj_t) = time_embed(g, w, "time_embed", t_features);
    let (temb_r, proj_r) = time_embed(g, w, "time_embed_r", r_features);
    let temb = g.add(temb_t, temb_r);
    let tproj = g.add(proj_t, proj_r);

    let mut x = g.linear_b(conv.get("proj_in.weight"), w.get("proj_in_conv.bias"), frames);
    let ctx = g.linear_b(w.get("condition_embedder.weight"), w.get("condition_embedder.bias"), context);

    for (i, sliding) in cfg.sliding()?.into_iter().enumerate() {
        let pre = format!("layers.{i}");
        let table = w.get(&format!("{pre}.scale_shift_table"));
        let mut m = [x; 6];
        for (j, slot) in m.iter_mut().enumerate() {
            let tv = g.view_1d(tproj, d, j * d as usize);
            let sv = g.view_1d(table, d, j * d as usize);
            *slot = g.add(tv, sv);
        }
        let [shift, scale, gate, c_shift, c_scale, c_gate] = m;
        let n = rms(g, w, &format!("{pre}.self_attn_norm.weight"), x, h.eps);
        let n = modulate(g, n, shift, scale);
        let o = attention(g, w, &format!("{pre}.self_attn"), h, n, None, pos, sliding.then_some(mask));
        let o = g.mul(o, gate);
        x = g.add(x, o);
        let n = rms(g, w, &format!("{pre}.cross_attn_norm.weight"), x, h.eps);
        let o = attention(g, w, &format!("{pre}.cross_attn"), h, n, Some(ctx), pos, None);
        x = g.add(x, o);
        let n = rms(g, w, &format!("{pre}.mlp_norm.weight"), x, h.eps);
        let n = modulate(g, n, c_shift, c_scale);
        let f = mlp(g, w, &pre, n);
        let f = g.mul(f, c_gate);
        x = g.add(x, f);
    }

    let table = w.get("scale_shift_table");
    let sv = g.view_1d(table, d, 0);
    let shift = g.add(sv, temb);
    let sv = g.view_1d(table, d, d as usize);
    let scale = g.add(sv, temb);
    let n = rms(g, w, "norm_out.weight", x, h.eps);
    let n = modulate(g, n, shift, scale);
    let y = g.linear(conv.get("proj_out.weight"), n);
    let y = g.reshape(y, &[a, positions * p]);
    let out = g.add(y, w.get("proj_out_conv.bias"));
    Ok(DitIo { frames, t_features, r_features, context, positions: pos, mask, out })
}
