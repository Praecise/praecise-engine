//! Gemma 3 decoder used as a prompt encoder.
//!
//! The prompt conditioning is every hidden state of the decoder: the scaled
//! token embeddings, the output of each layer, and in place of the last
//! layer's output its final-normalised form. Layers alternate between a
//! sliding-window and a full causal attention, each with its own rotary base
//! (the full layers optionally with linearly stretched positions). Every
//! norm scales by `1 + weight`; each layer normalises its attention and
//! feed-forward both before and after; queries and keys are RMS-normalised
//! per head; the feed-forward is a tanh-GELU gated MLP.
//!
//! Only the prompt tokens are encoded. Padding before them is hidden from
//! every query in the reference, so leaving it out changes nothing except
//! the absolute positions, which are passed in.

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Tensor-name prefixes of the decoder, by checkpoint flavour.
const PREFIXES: [&str; 4] = ["language_model.model.", "model.language_model.", "model.", ""];

/// Rotary settings of one attention kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotary {
    /// Rotary base.
    pub theta: f64,
    /// Linear position divisor (1 when unscaled).
    pub factor: f64,
}

/// Text configuration (`text_config` of a multimodal checkpoint, or the whole
/// file of a text-only one).
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct Gemma3Config {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: u64,
    pub query_pre_attn_scalar: f64,
    #[serde(default)]
    pub sliding_window: Option<u64>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_local_base_freq: Option<f64>,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub use_bidirectional_attention: Option<bool>,
}

fn scaled(p: &Value, default_theta: f64) -> Result<Rotary> {
    let theta = p.get("rope_theta").and_then(Value::as_f64).unwrap_or(default_theta);
    match p.get("rope_type").and_then(Value::as_str) {
        None | Some("default") => Ok(Rotary { theta, factor: 1.0 }),
        Some("linear") => {
            let factor = p.get("factor").and_then(Value::as_f64).ok_or_else(|| Error::Config("linear rope scaling without a factor".into()))?;
            Ok(Rotary { theta, factor })
        }
        Some(other) => Err(Error::Config(format!("rope scaling {other:?} is not implemented"))),
    }
}

impl Gemma3Config {
    /// Read from a checkpoint's `config.json`.
    ///
    /// # Errors
    /// [`Error::Config`] when it does not parse or asks for something this
    /// encoder does not implement.
    pub fn from_json(v: Value) -> Result<Self> {
        let v = match v.get("text_config") {
            Some(t) => t.clone(),
            None => v,
        };
        let c: Self = parse(v, "text encoder config")?;
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        if self.attn_logit_softcapping.is_some() {
            return Err(Error::Config("attention logit soft-capping is not implemented".into()));
        }
        if self.use_bidirectional_attention == Some(true) {
            return Err(Error::Config("bidirectional attention is not implemented".into()));
        }
        if let Some(a) = &self.hidden_activation {
            if a != "gelu_pytorch_tanh" {
                return Err(Error::Config(format!("activation {a:?} is not implemented")));
            }
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 || self.head_dim % 2 != 0 {
            return Err(Error::Config("text encoder head layout".into()));
        }
        let t = self.layer_types();
        if t.len() != self.num_hidden_layers {
            return Err(Error::Config("layer_types disagrees with num_hidden_layers".into()));
        }
        self.rotary(true)?;
        self.rotary(false)?;
        Ok(())
    }

    /// Whether each layer uses sliding-window attention.
    #[must_use]
    pub fn layer_types(&self) -> Vec<bool> {
        match &self.layer_types {
            Some(t) => t.iter().map(|s| s == "sliding_attention").collect(),
            None => {
                let p = self.sliding_window_pattern.unwrap_or(6);
                (0..self.num_hidden_layers).map(|i| (i + 1) % p != 0).collect()
            }
        }
    }

    /// Rotary settings of the sliding (`true`) or full layers.
    ///
    /// # Errors
    /// On an unimplemented scaling scheme.
    pub fn rotary(&self, sliding: bool) -> Result<Rotary> {
        if let Some(p) = &self.rope_parameters {
            let key = if sliding { "sliding_attention" } else { "full_attention" };
            if let Some(p) = p.get(key) {
                return scaled(p, if sliding { 10_000.0 } else { 1_000_000.0 });
            }
        }
        if sliding {
            return Ok(Rotary { theta: self.rope_local_base_freq.unwrap_or(10_000.0), factor: 1.0 });
        }
        let theta = self.rope_theta.unwrap_or(1_000_000.0);
        match &self.rope_scaling {
            Some(s) if !s.is_null() => scaled(s, theta).map(|r| Rotary { theta, ..r }),
            _ => Ok(Rotary { theta, factor: 1.0 }),
        }
    }

    /// Every weight of the decoder under `prefix`.
    #[must_use]
    pub fn weight_specs(&self, prefix: &str, embed: WType, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size;
        let hd = self.head_dim;
        let q = self.num_attention_heads * hd;
        let kv = self.num_key_value_heads * hd;
        let ff = self.intermediate_size;
        let mut v = vec![
            WeightSpec::new(format!("{prefix}embed_tokens.weight"), &[self.vocab_size, d], embed),
            WeightSpec::new(format!("{prefix}norm.weight"), &[d], WType::F32),
        ];
        for i in 0..self.num_hidden_layers {
            let p = format!("{prefix}layers.{i}");
            for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[d], WType::F32));
            }
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
        v
    }

    /// Rotary tables `(cos, sin)` over `[token][head width]` for positions
    /// `start..start + n`, computed in float32 as the reference does.
    #[must_use]
    pub fn rotary_tables(&self, r: Rotary, start: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim as usize;
        let half = hd / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0f32 / (r.theta as f32).powf((2 * i) as f32 / hd as f32) / r.factor as f32).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(n * hd), Vec::with_capacity(n * hd));
        for t in start..start + n {
            for j in 0..hd {
                let a = t as f32 * inv[j % half];
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        (cos, sin)
    }

    /// Additive causal mask `[query][key]` over `n` consecutive tokens,
    /// limited to the sliding window when `sliding`.
    #[must_use]
    pub fn mask(&self, n: usize, sliding: bool) -> Vec<f32> {
        let w = if sliding { self.sliding_window.map_or(usize::MAX, |w| w as usize) } else { usize::MAX };
        let mut m = vec![0f32; n * n];
        for q in 0..n {
            for k in 0..n {
                if k > q || q - k >= w {
                    m[q * n + k] = f32::NEG_INFINITY;
                }
            }
        }
        m
    }
}

/// `x * (1 + weight)` after an RMS norm.
fn norm(g: &mut Graph, x: Tn, w: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    let w = g.scale_bias(w, 1.0, 1.0);
    g.mul(h, w)
}

/// A loaded Gemma 3 prompt encoder.
pub struct Gemma3Encoder {
    backend: Backend,
    cfg: Gemma3Config,
    w: Weights,
    prefix: &'static str,
    embed_scale: f32,
    exact: bool,
}

impl std::fmt::Debug for Gemma3Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma3Encoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl Gemma3Encoder {
    /// Device bytes held.
    pub(crate) fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Load `dir/` (its `config.json` and safetensors files) of a checkpoint.
    ///
    /// # Errors
    /// On an unsupported configuration, missing weights or no usable backend.
    pub fn load(files: &CheckpointFiles, dir: &str, opts: LoadOptions) -> Result<Self> {
        let cfg = Gemma3Config::from_json(files.json(&format!("{dir}/config.json"))?)?;
        let st = SafeTensors::open(&files.weights(dir)?)?;
        let prefix = PREFIXES
            .into_iter()
            .find(|p| st.get(&format!("{p}embed_tokens.weight")).is_some())
            .ok_or_else(|| Error::MissingTensor("embed_tokens.weight".into()))?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let embed = if exact { WType::F32 } else { WType::F16 };
        let w = Weights::load(&backend, &st, &cfg.weight_specs(prefix, embed, opts.precision.wtype()))?;
        // The reference multiplies by sqrt(width) held in the weights' type.
        let s = (cfg.hidden_size as f32).sqrt();
        let embed_scale = if exact { s } else { half::bf16::from_f32(s).to_f32() };
        Ok(Self { backend, cfg, w, prefix, embed_scale, exact })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &Gemma3Config {
        &self.cfg
    }

    /// Every hidden state of `tokens` placed at positions
    /// `start..start + tokens.len()`, as `[token][width][state]` with
    /// `num_hidden_layers + 1` states (embeddings first, final-normalised
    /// last).
    ///
    /// # Errors
    /// When there are no tokens, a token is outside the vocabulary, or the
    /// backend fails.
    pub fn forward(&self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let n = tokens.len();
        if n == 0 || tokens.iter().any(|&t| u64::from(t) >= cfg.vocab_size) {
            return Err(Error::Request("prompt tokens outside the vocabulary".into()));
        }
        let (d, hd) = (cfg.hidden_size as i64, cfg.head_dim as i64);
        let (nh, nkv) = (cfg.num_attention_heads as i64, cfg.num_key_value_heads as i64);
        let eps = cfg.rms_norm_eps as f32;
        let ni = n as i64;
        let pre = self.prefix;
        let w = &self.w;
        let kinds = cfg.layer_types();

        let mut g = Graph::new(&self.backend)?;
        let ids = g.input(sys::GGML_TYPE_I32, &[ni]);
        // Index 0: sliding, 1: full.
        let tables: Vec<(Tn, Tn, Tn)> = (0..2)
            .map(|_| (g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]), g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]), g.input(sys::GGML_TYPE_F16, &[ni, ni])))
            .collect();
        let x = g.get_rows(w.get(&format!("{pre}embed_tokens.weight")), ids);
        let mut x = g.scale_bias(x, self.embed_scale, 0.0);
        let mut states = vec![x];
        let scale = (1.0 / cfg.query_pre_attn_scalar.sqrt()) as f32;
        for (i, &sliding) in kinds.iter().enumerate() {
            let (cos, sin, mask) = tables[usize::from(!sliding)];
            let p = format!("{pre}layers.{i}");
            let wn = |s: &str| w.get(&format!("{p}.{s}"));
            let h = norm(&mut g, x, wn("input_layernorm.weight"), eps);
            let q = g.linear(wn("self_attn.q_proj.weight"), h);
            let k = g.linear(wn("self_attn.k_proj.weight"), h);
            let v = g.linear(wn("self_attn.v_proj.weight"), h);
            let q = g.reshape(q, &[hd, nh, ni]);
            let k = g.reshape(k, &[hd, nkv, ni]);
            let v = g.reshape(v, &[hd, nkv, ni]);
            let q = norm(&mut g, q, wn("self_attn.q_norm.weight"), eps);
            let k = norm(&mut g, k, wn("self_attn.k_norm.weight"), eps);
            let q = g.rotate_half_rope(q, cos, sin);
            let k = g.rotate_half_rope(k, cos, sin);
            let q = g.permute(q, [0, 2, 1, 3]);
            let k = g.permute(k, [0, 2, 1, 3]);
            let v = g.permute(v, [0, 2, 1, 3]);
            let o = if self.exact {
                g.attention_exact(q, k, v, Some(mask), scale)
            } else {
                let k = g.cast(k, sys::GGML_TYPE_F16);
                let v = g.cast(v, sys::GGML_TYPE_F16);
                g.attention(q, k, v, Some(mask), scale, true)
            };
            let o = g.reshape(o, &[hd * nh, ni]);
            let o = g.linear(wn("self_attn.o_proj.weight"), o);
            let o = norm(&mut g, o, wn("post_attention_layernorm.weight"), eps);
            x = g.add(x, o);
            let h = norm(&mut g, x, wn("pre_feedforward_layernorm.weight"), eps);
            let gate = g.linear(wn("mlp.gate_proj.weight"), h);
            let gate = if self.exact { g.gelu_tanh_exact(gate) } else { g.gelu_tanh(gate) };
            let up = g.linear(wn("mlp.up_proj.weight"), h);
            let m = g.mul(gate, up);
            let m = g.linear(wn("mlp.down_proj.weight"), m);
            let m = norm(&mut g, m, wn("post_feedforward_layernorm.weight"), eps);
            x = g.add(x, m);
            states.push(x);
        }
        let last = states.len() - 1;
        states[last] = norm(&mut g, x, w.get(&format!("{pre}norm.weight")), eps);
        g.finish(&states)?;
        g.set_i32(ids, &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>());
        for (j, &(cos, sin, mask)) in tables.iter().enumerate() {
            let sliding = j == 0;
            let (c, s) = cfg.rotary_tables(cfg.rotary(sliding)?, start, n);
            g.set_f32(cos, &c);
            g.set_f32(sin, &s);
            g.set_f16(mask, &cfg.mask(n, sliding));
        }
        g.compute()?;
        let du = d as usize;
        let l = states.len();
        let mut out = vec![0f32; n * du * l];
        for (si, &t) in states.iter().enumerate() {
            let s = g.read_f32(t);
            for (tok, row) in s.chunks_exact(du).enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    out[(tok * du + c) * l + si] = v;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(extra: Value) -> Gemma3Config {
        let mut v = serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "num_hidden_layers": 6, "rms_norm_eps": 1e-6, "vocab_size": 10,
            "query_pre_attn_scalar": 4, "sliding_window": 2
        });
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        Gemma3Config::from_json(serde_json::json!({ "text_config": v })).unwrap()
    }

    #[test]
    fn legacy_rotary_and_pattern() {
        let c = cfg(serde_json::json!({"rope_theta": 1e6, "rope_local_base_freq": 1e4, "rope_scaling": {"rope_type": "linear", "factor": 8.0}}));
        assert_eq!(c.rotary(false).unwrap(), Rotary { theta: 1e6, factor: 8.0 });
        assert_eq!(c.rotary(true).unwrap(), Rotary { theta: 1e4, factor: 1.0 });
        assert_eq!(c.layer_types(), vec![true, true, true, true, true, false]);
    }

    #[test]
    fn grouped_rotary() {
        let c = cfg(serde_json::json!({"rope_parameters": {
            "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1e6},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}}}));
        assert_eq!(c.rotary(false).unwrap(), Rotary { theta: 1e6, factor: 8.0 });
        assert_eq!(c.rotary(true).unwrap(), Rotary { theta: 1e4, factor: 1.0 });
    }

    #[test]
    fn sliding_mask_limits_distance() {
        let c = cfg(serde_json::json!({}));
        let m = c.mask(3, true);
        assert!(m[2 * 3].is_infinite() && m[2 * 3 + 1] == 0.0 && m[2 * 3 + 2] == 0.0 && m[1].is_infinite());
        assert_eq!(c.mask(3, false)[2 * 3], 0.0);
    }
}

/// Agreement with the reference on fixtures produced by
/// `tests/parity/make_gemma3_fixtures.py`: a random checkpoint with the
/// released layout (both attention kinds, a biting sliding window, linearly
/// stretched full-layer positions, grouped keys) at tiny widths.
///
/// Run with `PRAECISE_GEMMA3_PARITY=<fixture dir> cargo test -p
/// praecise-diffusion -- --ignored gemma3_parity`.
#[cfg(test)]
mod parity {
    use std::path::PathBuf;

    use super::*;

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = PathBuf::from(std::env::var("PRAECISE_GEMMA3_PARITY").expect("PRAECISE_GEMMA3_PARITY names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
        let e = Gemma3Encoder::load(&CheckpointFiles::new(d.join("checkpoint")), "text_encoder", opts).unwrap();
        for case in m["cases"].as_array().unwrap() {
            let tag = case["tag"].as_str().unwrap();
            let tokens: Vec<u32> = case["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
            let ours = e.forward(&tokens, case["start"].as_u64().unwrap() as usize).unwrap();
            let reference: Vec<f32> = std::fs::read(d.join(format!("out_{tag}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            assert_eq!(ours.len(), reference.len(), "length");
            let (mut dot, mut na, mut nb, mut maxerr, mut maxref) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (a, b) in ours.iter().zip(&reference) {
                let (a, b) = (f64::from(*a), f64::from(*b));
                dot += a * b;
                na += a * a;
                nb += b * b;
                maxerr = maxerr.max((a - b).abs());
                maxref = maxref.max(b.abs());
            }
            let (cos, rel) = (dot / (na.sqrt() * nb.sqrt()), maxerr / maxref);
            eprintln!("{tag}: cosine {cos:.6}, max relative error {rel:.6}");
            assert!(cos >= min_cos && rel <= max_rel, "{tag}: cosine {cos}, max relative error {rel}");
        }
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn gemma3_parity_f32() {
        run(Precision::F32, 0.999_999, 1e-4);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn gemma3_parity_bf16() {
        run(Precision::Bf16, 0.9999, 2e-2);
    }
}
