//! Qwen3 decoder used as a prompt encoder.
//!
//! The prompt conditioning is the stack of hidden states after chosen decoder
//! layers, so only the layers up to the deepest one requested are built. The
//! attention mask is causal and hides padding keys, so a padding position sees
//! only the real prompt tokens before it.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use llama_cpp_sys_2 as sys;

/// ggml's rotary mode for split-half (NeoX) rotation, `GGML_ROPE_TYPE_NEOX`
/// in ggml.h (a preprocessor define, so not in the generated bindings).
const ROPE_NEOX: i32 = 2;

/// Text-encoder configuration, read from its `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    /// Model width.
    pub hidden_size: u64,
    /// MLP width.
    pub intermediate_size: u64,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Key/value heads.
    pub num_key_value_heads: u64,
    /// Head width.
    pub head_dim: u64,
    /// Decoder layers.
    pub num_hidden_layers: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Rotary base, where the file states it at the top level.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Rotary settings, where the file groups them.
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    /// Vocabulary size.
    pub vocab_size: u64,
}

/// Grouped rotary settings.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// Rotary base.
    pub rope_theta: f64,
    /// Scaling scheme; only unscaled rotation is implemented.
    #[serde(default)]
    pub rope_type: Option<String>,
}

impl Qwen3Config {
    /// The rotary base.
    ///
    /// # Errors
    /// [`Error::Config`] when the file gives none, or asks for a scaled
    /// rotation this encoder does not implement.
    pub fn theta(&self) -> Result<f64> {
        match (&self.rope_parameters, self.rope_theta) {
            (Some(p), _) => match p.rope_type.as_deref() {
                None | Some("default") => Ok(p.rope_theta),
                Some(other) => Err(Error::Config(format!("rope scaling {other:?} is not implemented"))),
            },
            (None, Some(t)) => Ok(t),
            (None, None) => Err(Error::Config("text encoder config gives no rotary base".into())),
        }
    }

    /// Every weight needed to produce hidden states after layer `last`
    /// (1-based, as in a hidden-state list whose entry 0 is the embeddings).
    ///
    /// # Errors
    /// When `last` exceeds the layer count.
    pub fn weight_specs(&self, last: usize, linear: WType) -> Result<Vec<WeightSpec>> {
        if last == 0 || last > self.num_hidden_layers {
            return Err(Error::Config(format!("hidden-state layer {last} outside 1..={}", self.num_hidden_layers)));
        }
        let d = self.hidden_size;
        let hd = self.head_dim;
        let q = self.num_attention_heads * hd;
        let kv = self.num_key_value_heads * hd;
        let ff = self.intermediate_size;
        let mut v = vec![WeightSpec::new("model.embed_tokens.weight", &[self.vocab_size, d], WType::F16)];
        for i in 0..last {
            let p = format!("model.layers.{i}");
            v.push(WeightSpec::new(format!("{p}.input_layernorm.weight"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{p}.post_attention_layernorm.weight"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{p}.self_attn.q_proj.weight"), &[q, d], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.k_proj.weight"), &[kv, d], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.v_proj.weight"), &[kv, d], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.o_proj.weight"), &[d, q], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.q_norm.weight"), &[hd], WType::F32));
            v.push(WeightSpec::new(format!("{p}.self_attn.k_norm.weight"), &[hd], WType::F32));
            v.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[d, ff], linear));
        }
        Ok(v)
    }
}

/// Graph inputs and output of one encode.
#[derive(Debug, Clone, Copy)]
pub struct Qwen3Io {
    /// Token ids `[n]`.
    pub tokens: Tn,
    /// Positions `[n]`.
    pub positions: Tn,
    /// Additive attention mask `[n, n]` (f16).
    pub mask: Tn,
    /// Concatenated hidden states `[hidden_size * layers.len(), n]`.
    pub out: Tn,
}

/// Build an encode of `n` tokens returning the hidden states after each layer
/// in `layers` (1-based), concatenated per token in the order given. `theta`
/// is the rotary base from [`Qwen3Config::theta`].
#[must_use]
pub fn build(g: &mut Graph, cfg: &Qwen3Config, w: &Weights, n: i64, layers: &[usize], theta: f32) -> Qwen3Io {
    let hd = cfg.head_dim as i64;
    let nh = cfg.num_attention_heads as i64;
    let nkv = cfg.num_key_value_heads as i64;
    let eps = cfg.rms_norm_eps as f32;
    let last = layers.iter().copied().max().unwrap_or(0);

    let tokens = g.input(sys::GGML_TYPE_I32, &[n]);
    let positions = g.input(sys::GGML_TYPE_I32, &[n]);
    let mask = g.input(sys::GGML_TYPE_F16, &[n, n]);

    let mut x = g.get_rows(w.get("model.embed_tokens.weight"), tokens);
    let mut captured: Vec<(usize, Tn)> = Vec::new();
    for i in 0..last {
        let p = format!("model.layers.{i}");
        let wn = |s: &str| w.get(&format!("{p}.{s}"));
        let h = g.rms_norm(x, eps);
        let h = g.mul(h, wn("input_layernorm.weight"));
        let q = g.linear(wn("self_attn.q_proj.weight"), h);
        let k = g.linear(wn("self_attn.k_proj.weight"), h);
        let v = g.linear(wn("self_attn.v_proj.weight"), h);
        let q = g.reshape(q, &[hd, nh, n]);
        let k = g.reshape(k, &[hd, nkv, n]);
        let v = g.reshape(v, &[hd, nkv, n]);
        let q = g.rms_norm(q, eps);
        let q = g.mul(q, wn("self_attn.q_norm.weight"));
        let k = g.rms_norm(k, eps);
        let k = g.mul(k, wn("self_attn.k_norm.weight"));
        let q = g.rope(q, positions, hd as i32, ROPE_NEOX, theta);
        let k = g.rope(k, positions, hd as i32, ROPE_NEOX, theta);
        let q = g.permute(q, [0, 2, 1, 3]);
        let k = g.permute(k, [0, 2, 1, 3]);
        let v = g.permute(v, [0, 2, 1, 3]);
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        let o = g.attention(q, k, v, Some(mask), 1.0 / (hd as f32).sqrt(), true);
        let o = g.reshape(o, &[hd * nh, n]);
        let o = g.linear(wn("self_attn.o_proj.weight"), o);
        x = g.add(x, o);
        let h = g.rms_norm(x, eps);
        let h = g.mul(h, wn("post_attention_layernorm.weight"));
        let gate = g.linear(wn("mlp.gate_proj.weight"), h);
        let up = g.linear(wn("mlp.up_proj.weight"), h);
        let m = g.swiglu_split(gate, up);
        let m = g.linear(wn("mlp.down_proj.weight"), m);
        x = g.add(x, m);
        if layers.contains(&(i + 1)) {
            captured.push((i + 1, x));
        }
    }
    let mut out: Option<Tn> = None;
    for l in layers {
        let t = captured.iter().find(|(k, _)| k == l).map(|(_, t)| *t).expect("every requested layer is built");
        out = Some(match out {
            None => t,
            Some(prev) => g.concat(prev, t, 0),
        });
    }
    Qwen3Io { tokens, positions, mask, out: out.expect("at least one layer requested") }
}

/// The additive mask for `n` positions of which the first `real` are prompt
/// tokens: causal, and padding keys hidden from every query.
#[must_use]
pub fn mask(n: usize, real: usize) -> Vec<f32> {
    let mut m = vec![0f32; n * n];
    for q in 0..n {
        for k in 0..n {
            if k > q || k >= real {
                m[q * n + k] = f32::NEG_INFINITY;
            }
        }
    }
    m
}
