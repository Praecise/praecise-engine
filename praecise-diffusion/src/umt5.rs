//! Multilingual T5 encoder (the per-layer relative-bias variant) as a text
//! encoder: token embedding, pre-norm blocks of unscaled self-attention with
//! a learned bucketed relative-position bias in every layer, and a gated
//! tanh-GELU feed-forward, then a final RMS norm.
//!
//! Padding is never encoded: a padded key is masked out of every query, so
//! the states of the real tokens equal those of the prompt alone.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};

use llama_cpp_sys_2 as sys;

/// Encoder hyperparameters (the checkpoint's `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Umt5Config {
    /// `vocab_size` from the checkpoint configuration.
    pub vocab_size: u64,
    /// `d_model` from the checkpoint configuration.
    pub d_model: u64,
    /// `d_kv` from the checkpoint configuration.
    pub d_kv: u64,
    /// `d_ff` from the checkpoint configuration.
    pub d_ff: u64,
    /// `num_heads` from the checkpoint configuration.
    pub num_heads: u64,
    /// `num_layers` from the checkpoint configuration.
    pub num_layers: u64,
    /// `relative_attention_num_buckets` from the checkpoint configuration.
    pub relative_attention_num_buckets: u64,
    /// `relative_attention_max_distance` from the checkpoint configuration.
    pub relative_attention_max_distance: u64,
    /// `layer_norm_epsilon` from the checkpoint configuration.
    #[serde(default = "eps")]
    pub layer_norm_epsilon: f64,
    /// `feed_forward_proj` from the checkpoint configuration.
    pub feed_forward_proj: String,
}

fn eps() -> f64 {
    1e-6
}

impl Umt5Config {
    /// Refuse layouts this implementation does not carry.
    pub fn validate(&self) -> Result<()> {
        if self.feed_forward_proj != "gated-gelu" {
            return Err(Error::Config(format!("feed_forward_proj {} is not supported", self.feed_forward_proj)));
        }
        if self.relative_attention_num_buckets < 4 || self.relative_attention_max_distance < 4 {
            return Err(Error::Config("relative attention buckets out of range".into()));
        }
        Ok(())
    }

    /// Device weights.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, inner, ff) = (self.d_model, self.num_heads * self.d_kv, self.d_ff);
        let f = WType::F32;
        let mut v = vec![
            WeightSpec::new("shared.weight", &[self.vocab_size, d], linear),
            WeightSpec::new("encoder.final_layer_norm.weight", &[d], f),
        ];
        for i in 0..self.num_layers {
            let a = format!("encoder.block.{i}.layer.0");
            for m in ["q", "k", "v"] {
                v.push(WeightSpec::new(format!("{a}.SelfAttention.{m}.weight"), &[inner, d], linear));
            }
            v.push(WeightSpec::new(format!("{a}.SelfAttention.o.weight"), &[d, inner], linear));
            v.push(WeightSpec::new(
                format!("{a}.SelfAttention.relative_attention_bias.weight"),
                &[self.relative_attention_num_buckets, self.num_heads],
                f,
            ));
            v.push(WeightSpec::new(format!("{a}.layer_norm.weight"), &[d], f));
            let p = format!("encoder.block.{i}.layer.1");
            v.push(WeightSpec::new(format!("{p}.DenseReluDense.wi_0.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.DenseReluDense.wi_1.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.DenseReluDense.wo.weight"), &[d, ff], linear));
            v.push(WeightSpec::new(format!("{p}.layer_norm.weight"), &[d], f));
        }
        v
    }

    /// Bidirectional relative-position bucket of every (query, key) pair,
    /// key-major within a query: `[q][k]`.
    #[must_use]
    pub fn buckets(&self, n: usize) -> Vec<i32> {
        let half = (self.relative_attention_num_buckets / 2) as i64;
        let exact = half / 2;
        let max_d = self.relative_attention_max_distance as f64;
        let mut out = Vec::with_capacity(n * n);
        for q in 0..n as i64 {
            for k in 0..n as i64 {
                let rel = k - q;
                let mut b = if rel > 0 { half } else { 0 };
                let r = rel.abs();
                b += if r < exact {
                    r
                } else {
                    let large = exact + ((r as f64 / exact as f64).ln() / (max_d / exact as f64).ln() * (half - exact) as f64) as i64;
                    large.min(half - 1)
                };
                out.push(b as i32);
            }
        }
        out
    }
}

/// Inputs and output of one encoding.
#[derive(Debug, Clone, Copy)]
pub struct Umt5Io {
    /// Token ids `[n]`.
    pub ids: Tn,
    /// [`Umt5Config::buckets`] `[n * n]`.
    pub buckets: Tn,
    /// Final states `[d_model, n]`.
    pub out: Tn,
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    g.mul(h, w.get(name))
}

/// `[a, b, c]` to `[a, c, b]`, contiguous.
fn swap12(g: &mut Graph, x: Tn) -> Tn {
    let p = g.permute(x, [0, 2, 1, 3]);
    g.cont(p)
}

/// Build the encoder over `n` tokens.
pub fn build(g: &mut Graph, cfg: &Umt5Config, w: &Weights, n: i64) -> Umt5Io {
    let (hd, heads) = (cfg.d_kv as i64, cfg.num_heads as i64);
    let eps = cfg.layer_norm_epsilon as f32;
    let ids = g.input(sys::GGML_TYPE_I32, &[n]);
    let buckets = g.input(sys::GGML_TYPE_I32, &[n * n]);
    let mut x = g.get_rows(w.get("shared.weight"), ids);
    for i in 0..cfg.num_layers {
        let a = format!("encoder.block.{i}.layer.0");
        let h = rms(g, w, &format!("{a}.layer_norm.weight"), x, eps);
        let proj = |g: &mut Graph, m: &str| {
            let y = g.linear(w.get(&format!("{a}.SelfAttention.{m}.weight")), h);
            let y = g.reshape(y, &[hd, heads, n]);
            swap12(g, y)
        };
        let q = proj(g, "q");
        let k = proj(g, "k");
        let v = proj(g, "v");
        // scores [k, q, head]: unscaled dot products plus the bucket bias
        let s = g.linear(k, q);
        let bias = g.get_rows(w.get(&format!("{a}.SelfAttention.relative_attention_bias.weight")), buckets);
        let bias = g.reshape(bias, &[heads, n, n]);
        let bias = g.permute(bias, [2, 0, 1, 3]);
        let bias = g.cont(bias);
        let s = g.add(s, bias);
        let p = g.soft_max(s, 1.0);
        let vt = g.permute(v, [1, 0, 2, 3]);
        let vt = g.cont(vt);
        let o = g.linear(vt, p);
        let o = swap12(g, o);
        let o = g.reshape(o, &[hd * heads, n]);
        let o = g.linear(w.get(&format!("{a}.SelfAttention.o.weight")), o);
        x = g.add(x, o);

        let p = format!("encoder.block.{i}.layer.1");
        let h = rms(g, w, &format!("{p}.layer_norm.weight"), x, eps);
        let gate = g.linear(w.get(&format!("{p}.DenseReluDense.wi_0.weight")), h);
        let gate = g.gelu_tanh_exact(gate);
        let up = g.linear(w.get(&format!("{p}.DenseReluDense.wi_1.weight")), h);
        let f = g.mul(gate, up);
        let f = g.linear(w.get(&format!("{p}.DenseReluDense.wo.weight")), f);
        x = g.add(x, f);
    }
    let out = rms(g, w, "encoder.final_layer_norm.weight", x, eps);
    Umt5Io { ids, buckets, out }
}

#[cfg(test)]
mod parity;
