//! The GR00T N1.7 action head: a diffusion transformer that turns the
//! backbone's hidden states, the robot state and a noisy action chunk into a
//! flow-matching velocity, integrated from noise at `t = 0` to actions at
//! `t = 1` by forward Euler.
//!
//! The backbone features pass a LayerNorm and a stack of self-attention
//! blocks. The state and the actions are embedded by per-embodiment
//! projections (one embodiment's weights are loaded); the action embedding
//! is fused with a sinusoidal embedding of the discretised time and gets a
//! learned position per step. The state token and the action tokens pass
//! blocks that alternate between cross-attention to the backbone tokens
//! (every other cross-attention block reads the text tokens, the rest the
//! image tokens) and self-attention; each block's first norm is modulated
//! by the time embedding, its feed-forward is a tanh-GELU MLP. A modulated
//! output norm and a per-embodiment decoder give the velocity of every
//! action step.

use std::path::Path;

use serde_json::Value;

use super::{attend, read_json};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const P: &str = "action_head.";

/// The head's settings, from the checkpoint's `config.json`.
#[derive(Debug, Clone)]
pub struct GrootHeadConfig {
    /// Backbone hidden width.
    pub backbone_dim: u64,
    /// Width of the state and action embeddings (the transformer width).
    pub embed_dim: u64,
    /// Hidden width of the per-embodiment projections.
    pub hidden: u64,
    /// Padded state width.
    pub max_state_dim: usize,
    /// Padded action width.
    pub max_action_dim: usize,
    /// Actions per chunk.
    pub horizon: usize,
    /// Learned positions available.
    pub max_seq_len: u64,
    /// Transformer blocks.
    pub layers: usize,
    /// Transformer heads.
    pub heads: u64,
    /// Transformer head width.
    pub head_dim: u64,
    /// Transformer output width.
    pub output_dim: u64,
    /// Cross-attention to text every this many cross-attention blocks.
    pub text_every: usize,
    /// Backbone self-attention blocks.
    pub vl_layers: usize,
    /// Backbone self-attention heads.
    pub vl_heads: u64,
    /// Backbone self-attention head width.
    pub vl_head_dim: u64,
    /// Euler steps.
    pub steps: usize,
    /// Time discretisation.
    pub buckets: usize,
    /// Embodiments the projections hold.
    pub embodiments: usize,
}

fn u(v: &Value, k: &str) -> Result<u64> {
    v.get(k).and_then(Value::as_u64).ok_or_else(|| Error::Config(format!("config gives no {k}")))
}

impl GrootHeadConfig {
    /// Read a checkpoint config.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field or a variant this engine does
    /// not implement.
    pub fn from_json(c: &Value) -> Result<Self> {
        let flag = |k: &str, want: bool| -> Result<()> {
            match c.get(k).and_then(Value::as_bool) {
                Some(v) if v != want => Err(Error::Config(format!("GR00T {k} = {v} is not implemented"))),
                _ => Ok(()),
            }
        };
        flag("use_alternate_vl_dit", true)?;
        flag("add_pos_embed", true)?;
        flag("use_vlln", true)?;
        if u(c, "state_history_length").unwrap_or(1) != 1 {
            return Err(Error::Config("GR00T state history is not implemented".into()));
        }
        let d = c.get("diffusion_model_cfg").ok_or_else(|| Error::Config("config gives no diffusion_model_cfg".into()))?;
        if d.get("norm_type").and_then(Value::as_str).unwrap_or("ada_norm") != "ada_norm"
            || !d.get("interleave_self_attention").and_then(Value::as_bool).unwrap_or(false)
            || d.get("positional_embeddings").is_some_and(|p| !p.is_null())
        {
            return Err(Error::Config("only the interleaved, time-modulated GR00T transformer is implemented".into()));
        }
        let vl = c.get("vl_self_attention_cfg").cloned().unwrap_or(Value::Null);
        if vl.get("positional_embeddings").is_some_and(|p| !p.is_null()) {
            return Err(Error::Config("positional embeddings in the backbone self-attention are not implemented".into()));
        }
        let cfg = Self {
            backbone_dim: u(c, "backbone_embedding_dim")?,
            embed_dim: u(c, "input_embedding_dim").unwrap_or(1536),
            hidden: u(c, "hidden_size")?,
            max_state_dim: u(c, "max_state_dim")? as usize,
            max_action_dim: u(c, "max_action_dim")? as usize,
            horizon: u(c, "action_horizon")? as usize,
            max_seq_len: u(c, "max_seq_len")?,
            layers: u(d, "num_layers")? as usize,
            heads: u(d, "num_attention_heads")?,
            head_dim: u(d, "attention_head_dim")?,
            output_dim: u(d, "output_dim")?,
            text_every: u(c, "attend_text_every_n_blocks").unwrap_or(2) as usize,
            vl_layers: vl.get("num_layers").and_then(Value::as_u64).unwrap_or(0) as usize,
            vl_heads: vl.get("num_attention_heads").and_then(Value::as_u64).unwrap_or(1),
            vl_head_dim: vl.get("attention_head_dim").and_then(Value::as_u64).unwrap_or(1),
            steps: u(c, "num_inference_timesteps")? as usize,
            buckets: u(c, "num_timestep_buckets")? as usize,
            embodiments: u(c, "max_num_embodiments")? as usize,
        };
        if cfg.heads * cfg.head_dim != cfg.embed_dim {
            return Err(Error::Config("transformer width differs from the embedding width".into()));
        }
        if cfg.vl_layers > 0 && cfg.vl_heads * cfg.vl_head_dim != cfg.backbone_dim {
            return Err(Error::Config("backbone self-attention width differs from the backbone width".into()));
        }
        Ok(cfg)
    }

    fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let f = WType::F32;
        let (e, b) = (self.embed_dim, self.backbone_dim);
        let mut s = vec![
            WeightSpec::new(format!("{P}vlln.weight"), &[b], f),
            WeightSpec::new(format!("{P}vlln.bias"), &[b], f),
            WeightSpec::new(format!("{P}position_embedding.weight"), &[self.max_seq_len, e], f),
        ];
        let lin = |s: &mut Vec<WeightSpec>, n: String, o: u64, i: u64| {
            s.push(WeightSpec::new(format!("{n}.weight"), &[o, i], linear));
            s.push(WeightSpec::new(format!("{n}.bias"), &[o], f));
        };
        for i in 0..self.vl_layers {
            let p = format!("{P}vl_self_attention.transformer_blocks.{i}");
            for n in ["norm1", "norm3"] {
                s.push(WeightSpec::new(format!("{p}.{n}.weight"), &[b], f));
                s.push(WeightSpec::new(format!("{p}.{n}.bias"), &[b], f));
            }
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                lin(&mut s, format!("{p}.attn1.{n}"), b, b);
            }
            lin(&mut s, format!("{p}.ff.net.0.proj"), 4 * b, b);
            lin(&mut s, format!("{p}.ff.net.2"), b, 4 * b);
        }
        let m = format!("{P}model");
        lin(&mut s, format!("{m}.timestep_encoder.timestep_embedder.linear_1"), e, 256);
        lin(&mut s, format!("{m}.timestep_encoder.timestep_embedder.linear_2"), e, e);
        for i in 0..self.layers {
            let p = format!("{m}.transformer_blocks.{i}");
            let kv_in = if i % 2 == 1 { e } else { b };
            lin(&mut s, format!("{p}.norm1.linear"), 2 * e, e);
            lin(&mut s, format!("{p}.attn1.to_q"), e, e);
            lin(&mut s, format!("{p}.attn1.to_k"), e, kv_in);
            lin(&mut s, format!("{p}.attn1.to_v"), e, kv_in);
            lin(&mut s, format!("{p}.attn1.to_out.0"), e, e);
            lin(&mut s, format!("{p}.ff.net.0.proj"), 4 * e, e);
            lin(&mut s, format!("{p}.ff.net.2"), e, 4 * e);
        }
        lin(&mut s, format!("{m}.proj_out_1"), 2 * e, e);
        lin(&mut s, format!("{m}.proj_out_2"), self.output_dim, e);
        s
    }

    /// The per-embodiment projections `(name, in, out)`.
    fn category_linears(&self) -> Vec<(String, u64, u64)> {
        let (e, h) = (self.embed_dim, self.hidden);
        vec![
            (format!("{P}state_encoder.layer1"), self.max_state_dim as u64, h),
            (format!("{P}state_encoder.layer2"), h, e),
            (format!("{P}action_encoder.W1"), self.max_action_dim as u64, e),
            (format!("{P}action_encoder.W2"), 2 * e, e),
            (format!("{P}action_encoder.W3"), e, e),
            (format!("{P}action_decoder.layer1"), self.output_dim, h),
            (format!("{P}action_decoder.layer2"), h, self.max_action_dim as u64),
        ]
    }
}

/// Sine-then-cosine embedding of an integer time over `dim` channels with
/// frequencies `exp(-k ln(10000) / (dim / 2))`.
#[must_use]
pub fn action_time_embedding(t: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = (10_000f32).ln() / half as f32;
    let mut out = vec![0f32; 2 * half];
    for k in 0..half {
        let x = t as f32 * (-(k as f32) * step).exp();
        out[k] = x.sin();
        out[half + k] = x.cos();
    }
    out
}

/// Cosine-then-sine projection of an integer time over 256 channels with
/// frequencies `exp(-k ln(10000) / 127)`, the transformer's time input.
#[must_use]
pub fn transformer_time_projection(t: usize) -> Vec<f32> {
    let half = 128;
    let mut out = vec![0f32; 2 * half];
    for k in 0..half {
        let x = t as f32 * (-(10_000f32).ln() * k as f32 / (half as f32 - 1.0)).exp();
        out[k] = x.cos();
        out[half + k] = x.sin();
    }
    out
}

/// A loaded GR00T N1.7 action head for one embodiment.
pub struct GrootHead {
    cfg: GrootHeadConfig,
    embodiment: usize,
    backend: Backend,
    weights: Weights,
    categories: Weights,
    exact: bool,
}

impl std::fmt::Debug for GrootHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrootHead").field("device", &self.backend.name()).field("embodiment", &self.embodiment).finish_non_exhaustive()
    }
}

fn lin_b(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{name}.weight")), w.get(&format!("{name}.bias")), x)
}

fn layer_norm(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.norm(x, eps);
    let h = g.mul(h, w.get(&format!("{name}.weight")));
    g.add(h, w.get(&format!("{name}.bias")))
}

/// Rows `[from, from + n)` of a column vector, made contiguous.
fn rows(g: &mut Graph, x: Tn, from: i64, n: i64) -> Tn {
    let v = g.view_rows(x, from, n);
    g.cont(v)
}

impl GrootHead {
    /// Load the head from a checkpoint directory (`config.json` and the
    /// safetensors files named by `model.safetensors.index.json`, or one
    /// `model.safetensors`), keeping embodiment `embodiment`'s projections.
    ///
    /// # Errors
    /// On a missing or malformed file, an unknown embodiment, or a backend
    /// failure.
    pub fn load(dir: &Path, embodiment: usize, opts: LoadOptions) -> Result<Self> {
        let cfg = GrootHeadConfig::from_json(&read_json(&dir.join("config.json"))?)?;
        if embodiment >= cfg.embodiments {
            return Err(Error::Config(format!("embodiment {embodiment} outside the checkpoint's {}", cfg.embodiments)));
        }
        let files = match read_json(&dir.join("model.safetensors.index.json")) {
            Ok(index) => {
                let mut f: Vec<String> = index
                    .get("weight_map")
                    .and_then(Value::as_object)
                    .map(|m| m.values().filter_map(Value::as_str).map(str::to_owned).collect())
                    .unwrap_or_default();
                f.sort();
                f.dedup();
                f.into_iter().map(|n| dir.join(n)).collect()
            }
            Err(_) => vec![dir.join("model.safetensors")],
        };
        let st = SafeTensors::open(&files)?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let linear = match opts.precision {
            Precision::F32 => WType::F32,
            Precision::Bf16 => WType::Bf16,
            Precision::Q8_0 => WType::Q8_0,
        };
        let weights = Weights::load(&backend, &st, &cfg.weight_specs(linear))?;
        let mut host = Vec::new();
        for (name, i, o) in cfg.category_linears() {
            let w = st.require(&format!("{name}.W"), &[cfg.embodiments as u64, i, o])?.to_f32();
            let b = st.require(&format!("{name}.b"), &[cfg.embodiments as u64, o])?.to_f32();
            let (i, o) = (i as usize, o as usize);
            let slab = &w[embodiment * i * o..(embodiment + 1) * i * o];
            // Stored `[in][out]`; the products read `[out][in]`.
            let mut t = vec![0f32; i * o];
            for r in 0..i {
                for c in 0..o {
                    t[c * i + r] = slab[r * o + c];
                }
            }
            let ty = if linear == WType::Q8_0 && i % 32 != 0 { WType::Bf16 } else { linear };
            host.push(HostTensor { name: format!("{name}.weight"), shape: vec![o as u64, i as u64], ty, data: t });
            host.push(HostTensor { name: format!("{name}.bias"), shape: vec![o as u64], ty: WType::F32, data: b[embodiment * o..(embodiment + 1) * o].to_vec() });
        }
        let categories = Weights::from_host(&backend, &host)?;
        Ok(Self { cfg, embodiment, backend, weights, categories, exact })
    }

    /// The head's settings.
    #[must_use]
    pub fn config(&self) -> &GrootHeadConfig {
        &self.cfg
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Run F32 matrix products in full FP32 on a backend that would round
    /// them (TF32 on CUDA).
    pub fn set_fp32_gemm(&mut self, fp32: bool) {
        self.backend.set_fp32_gemm(fp32);
    }

    /// Bytes of weights resident on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.weights.bytes() + self.categories.bytes()
    }

    /// One chunk `[horizon][max_action_dim]` in the policy's normalised
    /// units. `features` are the backbone's hidden states `[tokens][width]`;
    /// `image` marks the image tokens and `valid` the tokens to attend to;
    /// `state` is the normalised, padded state; `noise` the initial chunk.
    ///
    /// # Errors
    /// On inputs of the wrong size, a token kind with no token to attend to,
    /// or a backend failure.
    #[allow(clippy::too_many_lines)]
    pub fn sample(&self, features: &[f32], image: &[bool], valid: &[bool], state: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let bd = c.backbone_dim as usize;
        let n_vl = image.len();
        if features.len() != n_vl * bd || valid.len() != n_vl || state.len() != c.max_state_dim || noise.len() != c.horizon * c.max_action_dim {
            return Err(Error::Request("backbone features, masks, state or noise of the wrong size".into()));
        }
        let text_keys: Vec<bool> = image.iter().zip(valid).map(|(&i, &v)| !i && v).collect();
        let image_keys: Vec<bool> = image.iter().zip(valid).map(|(&i, &v)| i && v).collect();
        if !text_keys.iter().any(|&k| k) || !image_keys.iter().any(|&k| k) {
            return Err(Error::Request("the backbone features need both text and image tokens".into()));
        }
        let (w, cw) = (&self.weights, &self.categories);
        let e = c.embed_dim as i64;
        let (nh, hd) = (c.heads as i64, c.head_dim as i64);
        let n = (c.horizon + 1) as i64;
        let h = c.horizon as i64;
        let exact = self.exact;
        let mut g = Graph::new(&self.backend)?;

        let vl = g.input(sys::GGML_TYPE_F32, &[bd as i64, n_vl as i64]);
        let text_mask = g.input(sys::GGML_TYPE_F16, &[n_vl as i64, n]);
        let image_mask = g.input(sys::GGML_TYPE_F16, &[n_vl as i64, n]);
        let state_in = g.input(sys::GGML_TYPE_F32, &[c.max_state_dim as i64, 1]);
        let noise_in = g.input(sys::GGML_TYPE_F32, &[c.max_action_dim as i64, h]);
        let act_time = g.input(sys::GGML_TYPE_F32, &[e, c.steps as i64]);
        let dit_time = g.input(sys::GGML_TYPE_F32, &[256, c.steps as i64]);

        // Backbone features: LayerNorm, then self-attention blocks.
        let mut v = layer_norm(&mut g, w, &format!("{P}vlln"), vl, 1e-5);
        let (vh, vhd) = (c.vl_heads as i64, c.vl_head_dim as i64);
        for i in 0..c.vl_layers {
            let p = format!("{P}vl_self_attention.transformer_blocks.{i}");
            let x = layer_norm(&mut g, w, &format!("{p}.norm1"), v, 1e-5);
            let q = lin_b(&mut g, w, &format!("{p}.attn1.to_q"), x);
            let k = lin_b(&mut g, w, &format!("{p}.attn1.to_k"), x);
            let vv = lin_b(&mut g, w, &format!("{p}.attn1.to_v"), x);
            let q = g.reshape(q, &[vhd, vh, n_vl as i64]);
            let k = g.reshape(k, &[vhd, vh, n_vl as i64]);
            let vv = g.reshape(vv, &[vhd, vh, n_vl as i64]);
            let o = attend(&mut g, q, k, vv, None, 1.0 / (vhd as f32).sqrt(), exact);
            let o = g.reshape(o, &[vhd * vh, n_vl as i64]);
            let o = lin_b(&mut g, w, &format!("{p}.attn1.to_out.0"), o);
            v = g.add(v, o);
            let x = layer_norm(&mut g, w, &format!("{p}.norm3"), v, 1e-5);
            let x = lin_b(&mut g, w, &format!("{p}.ff.net.0.proj"), x);
            let x = if exact { g.gelu_tanh_exact(x) } else { g.gelu_tanh(x) };
            let x = lin_b(&mut g, w, &format!("{p}.ff.net.2"), x);
            v = g.add(v, x);
        }

        // State token.
        let s = lin_b(&mut g, cw, &format!("{P}state_encoder.layer1"), state_in);
        let s = g.relu(s);
        let state_tok = lin_b(&mut g, cw, &format!("{P}state_encoder.layer2"), s);
        let pos = w.get(&format!("{P}position_embedding.weight"));
        let pos = g.view_cols(pos, 0, h);

        let dt = 1.0 / c.steps as f32;
        let m = format!("{P}model");
        let mut xt = noise_in;
        for step in 0..c.steps {
            let te = g.view_cols(act_time, step as i64, 1);
            let te = g.repeat_to(te, [e, h, 1, 1]);
            let a = lin_b(&mut g, cw, &format!("{P}action_encoder.W1"), xt);
            let a = g.concat(a, te, 0);
            let a = lin_b(&mut g, cw, &format!("{P}action_encoder.W2"), a);
            let a = g.silu(a);
            let a = lin_b(&mut g, cw, &format!("{P}action_encoder.W3"), a);
            let a = g.add(a, pos);
            let mut x = g.concat(state_tok, a, 1);

            let tp = g.view_cols(dit_time, step as i64, 1);
            let temb = lin_b(&mut g, w, &format!("{m}.timestep_encoder.timestep_embedder.linear_1"), tp);
            let temb = g.silu(temb);
            let temb = lin_b(&mut g, w, &format!("{m}.timestep_encoder.timestep_embedder.linear_2"), temb);
            let st = g.silu(temb);
            for i in 0..c.layers {
                let p = format!("{m}.transformer_blocks.{i}");
                let md = lin_b(&mut g, w, &format!("{p}.norm1.linear"), st);
                let scale = rows(&mut g, md, 0, e);
                let shift = rows(&mut g, md, e, e);
                let hn = g.norm(x, 1e-5);
                let sc = g.scale_bias(scale, 1.0, 1.0);
                let hn = g.mul(hn, sc);
                let hn = g.add(hn, shift);
                let q = lin_b(&mut g, w, &format!("{p}.attn1.to_q"), hn);
                let q = g.reshape(q, &[hd, nh, n]);
                let (kv_src, kv_n, mask) = if i % 2 == 1 {
                    (hn, n, None)
                } else if i % (2 * c.text_every) == 0 {
                    (v, n_vl as i64, Some(text_mask))
                } else {
                    (v, n_vl as i64, Some(image_mask))
                };
                let k = lin_b(&mut g, w, &format!("{p}.attn1.to_k"), kv_src);
                let vv = lin_b(&mut g, w, &format!("{p}.attn1.to_v"), kv_src);
                let k = g.reshape(k, &[hd, nh, kv_n]);
                let vv = g.reshape(vv, &[hd, nh, kv_n]);
                let o = attend(&mut g, q, k, vv, mask, 1.0 / (hd as f32).sqrt(), exact);
                let o = g.reshape(o, &[e, n]);
                let o = lin_b(&mut g, w, &format!("{p}.attn1.to_out.0"), o);
                x = g.add(x, o);
                let hn = g.norm(x, 1e-5);
                let f = lin_b(&mut g, w, &format!("{p}.ff.net.0.proj"), hn);
                let f = if exact { g.gelu_tanh_exact(f) } else { g.gelu_tanh(f) };
                let f = lin_b(&mut g, w, &format!("{p}.ff.net.2"), f);
                x = g.add(x, f);
            }
            let md = lin_b(&mut g, w, &format!("{m}.proj_out_1"), st);
            let shift = rows(&mut g, md, 0, e);
            let scale = rows(&mut g, md, e, e);
            let hn = g.norm(x, 1e-6);
            let sc = g.scale_bias(scale, 1.0, 1.0);
            let hn = g.mul(hn, sc);
            let hn = g.add(hn, shift);
            let out = lin_b(&mut g, w, &format!("{m}.proj_out_2"), hn);
            let d = lin_b(&mut g, cw, &format!("{P}action_decoder.layer1"), out);
            let d = g.relu(d);
            let d = lin_b(&mut g, cw, &format!("{P}action_decoder.layer2"), d);
            let d = g.view_cols(d, 1, h);
            let d = g.scale_bias(d, dt, 0.0);
            xt = g.add(xt, d);
        }
        g.finish(&[xt])?;

        g.set_f32(vl, features);
        let mask_of = |keys: &[bool]| -> Vec<f32> {
            let row: Vec<f32> = keys.iter().map(|&k| if k { 0.0 } else { f32::NEG_INFINITY }).collect();
            row.iter().copied().cycle().take(row.len() * n as usize).collect()
        };
        g.set_f16(text_mask, &mask_of(&text_keys));
        g.set_f16(image_mask, &mask_of(&image_keys));
        g.set_f32(state_in, state);
        g.set_f32(noise_in, noise);
        let times: Vec<usize> = (0..c.steps).map(|s| (s as f32 / c.steps as f32 * c.buckets as f32) as usize).collect();
        let at: Vec<f32> = times.iter().flat_map(|&t| action_time_embedding(t, e as usize)).collect();
        g.set_f32(act_time, &at);
        let tt: Vec<f32> = times.iter().flat_map(|&t| transformer_time_projection(t)).collect();
        g.set_f32(dit_time, &tt);
        g.compute()?;
        Ok(g.read_f32(xt))
    }
}

/// A GR00T N1.7 policy: the Qwen3-VL backbone truncated after the release's
/// `select_layer` decoder layers, whose last hidden states (before any
/// norm) feed the action head.
pub struct GrootPolicy {
    encoder: crate::qwen3_vl::Qwen3VlEncoder,
    head: GrootHead,
}

impl std::fmt::Debug for GrootPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrootPolicy").field("device", &self.head.device()).finish_non_exhaustive()
    }
}

impl GrootPolicy {
    /// Load a release directory (`config.json` and its safetensors, the
    /// backbone stored under `backbone.model.`) with the backbone's Qwen3-VL
    /// `config.json` given separately, keeping embodiment `embodiment`.
    ///
    /// # Errors
    /// On a missing or malformed file, an unknown embodiment, or a backend
    /// failure.
    pub fn load(dir: &Path, backbone_config: &Path, embodiment: usize, opts: LoadOptions) -> Result<Self> {
        let pc = read_json(&dir.join("config.json"))?;
        let layers = pc.get("select_layer").and_then(Value::as_u64).ok_or_else(|| Error::Config("config gives no select_layer".into()))? as usize;
        let cfg = crate::qwen3_vl::Qwen3VlConfig::from_json(read_json(backbone_config)?)?;
        let head = GrootHead::load(dir, embodiment, opts)?;
        if cfg.text_config.hidden_size != head.config().backbone_dim {
            return Err(Error::Config("backbone width differs from the head's backbone_embedding_dim".into()));
        }
        let files = match read_json(&dir.join("model.safetensors.index.json")) {
            Ok(index) => {
                let mut f: Vec<String> = index
                    .get("weight_map")
                    .and_then(Value::as_object)
                    .map(|m| m.values().filter_map(Value::as_str).map(str::to_owned).collect())
                    .unwrap_or_default();
                f.sort();
                f.dedup();
                f.into_iter().map(|n| dir.join(n)).collect()
            }
            Err(_) => vec![dir.join("model.safetensors")],
        };
        let st = SafeTensors::open(&files)?.renamed(|n| n.strip_prefix("backbone.model.").map(str::to_owned))?;
        let encoder = crate::qwen3_vl::Qwen3VlEncoder::from_safetensors(cfg, st, opts, Some(layers))?;
        Ok(Self { encoder, head })
    }

    /// The backbone.
    #[must_use]
    pub fn encoder(&self) -> &crate::qwen3_vl::Qwen3VlEncoder {
        &self.encoder
    }

    /// The action head.
    #[must_use]
    pub fn head(&self) -> &GrootHead {
        &self.head
    }

    /// Run F32 matrix products in full FP32 in the backbone and the head on
    /// a backend that would round them (TF32 on CUDA).
    pub fn set_fp32_gemm(&mut self, fp32: bool) {
        self.encoder.set_fp32_gemm(fp32);
        self.head.set_fp32_gemm(fp32);
    }

    /// The backbone hidden states `[tokens][width]` and the image-token mask
    /// for a prompt with one image placeholder per image (`[3][h][w]` in
    /// `[0, 1]`, sides multiples of the merged patch).
    ///
    /// # Errors
    /// As [`crate::qwen3_vl::Qwen3VlEncoder::forward`].
    pub fn features(&self, tokens: &[u32], images: &[(&[f32], (usize, usize))]) -> Result<(Vec<f32>, Vec<bool>)> {
        let imgs = images.iter().map(|(px, hw)| self.encoder.image(px, *hw)).collect::<Result<Vec<_>>>()?;
        self.features_of(tokens, &imgs)
    }

    /// As [`Self::features`] for images already laid out for the tower.
    ///
    /// # Errors
    /// As [`crate::qwen3_vl::Qwen3VlEncoder::forward`].
    pub fn features_of(&self, tokens: &[u32], images: &[crate::qwen3_vl::Vl3Image]) -> Result<(Vec<f32>, Vec<bool>)> {
        let grids: Vec<(usize, usize)> = images.iter().map(|i| i.grid).collect();
        let ids = self.encoder.expand_placeholders(tokens, &grids);
        let image_id = self.encoder.config().image_token_id;
        let feats = self.encoder.forward(&ids, images)?;
        Ok((feats, ids.iter().map(|&t| t == image_id).collect()))
    }

    /// One action chunk `[horizon][max_action_dim]` in normalised units.
    ///
    /// # Errors
    /// As [`Self::features`] and [`GrootHead::sample`].
    pub fn predict(&self, tokens: &[u32], images: &[(&[f32], (usize, usize))], state: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let imgs = images.iter().map(|(px, hw)| self.encoder.image(px, *hw)).collect::<Result<Vec<_>>>()?;
        self.predict_images(tokens, &imgs, state, noise)
    }

    /// As [`Self::predict`] for images already laid out for the tower.
    ///
    /// # Errors
    /// As [`Self::features_of`] and [`GrootHead::sample`].
    pub fn predict_images(&self, tokens: &[u32], images: &[crate::qwen3_vl::Vl3Image], state: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let (feats, image) = self.features_of(tokens, images)?;
        let valid = vec![true; image.len()];
        self.head.sample(&feats, &image, &valid, state, noise)
    }
}
