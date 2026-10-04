//! Cosmos3 Omni transformer: one decoder stack carrying two token streams.
//!
//! The text stream runs causal self-attention over the prompt. The generation
//! stream (patchified video latents) attends to all of its own tokens and to
//! the text stream's keys and values. Each stream has its own projections,
//! norms and feed-forward. The text stream never reads the generation stream,
//! so its keys and values are computed once per prompt ([`build_text`]) and
//! reused at every denoising step ([`build_gen`]).
//!
//! The generation stream can also carry action tokens after the video
//! tokens: a chunk of robot actions projected in and out by per-embodiment
//! weights. Video and action tokens attend to each other fully, so one
//! evaluation predicts both (policy), or predicts video under given actions
//! (forward dynamics).
//!
//! Sequences are laid out `[width, tokens]`. Rotary embeddings take three
//! positions per token (time, height, width) whose frequencies interleave;
//! they are applied from cos/sin tables computed on the host, since the
//! temporal positions are fractional when the frame rate differs from the
//! training rate.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use llama_cpp_sys_2 as sys;

/// Width of the sinusoidal timestep features.
pub const TIME_FEATURES: usize = 256;

/// Rotary section sizes under their older key.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Rotary pairs per axis.
    pub mrope_section: Vec<u64>,
}

/// Transformer configuration, read from `transformer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Cosmos3Config {
    /// Model width.
    pub hidden_size: u64,
    /// Feed-forward width.
    pub intermediate_size: u64,
    /// Decoder layers.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Key/value heads.
    pub num_key_value_heads: u64,
    /// Head width.
    pub head_dim: u64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Rotary base.
    pub rope_theta: f64,
    /// Rotary pairs per axis (time, height, width).
    #[serde(default)]
    pub rope_axes_dim: Option<Vec<u64>>,
    /// Older spelling of `rope_axes_dim`.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// Latent channels.
    pub latent_channel: u64,
    /// Spatial patch side.
    pub latent_patch_size: u64,
    /// Width of one patch token.
    pub patch_latent_dim: u64,
    /// Timesteps are multiplied by this before the sinusoid.
    pub timestep_scale: f64,
    /// Feed-forward activation.
    pub hidden_act: String,
    /// Per-head norms on the text stream's queries and keys.
    #[serde(default = "yes")]
    pub qk_norm_for_text: bool,
    /// A per-head norm on the text keys the generation stream reads.
    #[serde(default)]
    pub use_und_k_norm_for_gen: bool,
    /// Biases on the attention projections.
    #[serde(default)]
    pub attention_bias: bool,
    /// Text vocabulary.
    pub vocab_size: u64,
    /// Frame rate the temporal positions are measured in.
    pub base_fps: f64,
    /// Temporal positions follow the request's frame rate.
    pub enable_fps_modulation: bool,
    /// Gap between the last text position and the first video position.
    pub unified_3d_mrope_temporal_modality_margin: u64,
    /// Video height and width positions start at zero rather than after the
    /// text.
    pub unified_3d_mrope_reset_spatial_ids: bool,
    /// Action projections are present.
    #[serde(default)]
    pub action_gen: bool,
    /// Width of one (padded) action vector.
    #[serde(default)]
    pub action_dim: Option<u64>,
    /// Embodiments with their own action projections.
    #[serde(default = "domains")]
    pub num_embodiment_domains: u64,
}

fn yes() -> bool {
    true
}

fn domains() -> u64 {
    32
}

impl Cosmos3Config {
    /// Refuse variants this implementation has not been checked against.
    ///
    /// # Errors
    /// [`Error::Config`] naming the first unsupported setting.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Config(m));
        if self.hidden_act != "relu2" && self.hidden_act != "silu" {
            return bad(format!("feed-forward activation {:?} is not implemented", self.hidden_act));
        }
        if self.attention_bias {
            return bad("attention biases are not implemented".into());
        }
        if self.action_gen && self.action_dim.is_none_or(|d| d == 0) {
            return bad("action projections without an action width".into());
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 || self.head_dim % 2 != 0 {
            return bad("query heads must be a multiple of key/value heads and the head width even".into());
        }
        if self.patch_latent_dim != self.latent_channel * self.latent_patch_size * self.latent_patch_size {
            return bad("patch width is not channels times the patch area".into());
        }
        let axes = self.axes()?;
        let half = self.head_dim / 2;
        if axes.iter().sum::<u64>() != half || 3 * axes[1] > half || 3 * axes[2] > half {
            return bad(format!("rotary sections {axes:?} do not interleave over {half} pairs"));
        }
        Ok(())
    }

    /// Rotary pairs per axis.
    ///
    /// # Errors
    /// When neither spelling gives three axes.
    pub fn axes(&self) -> Result<[u64; 3]> {
        let v = self
            .rope_axes_dim
            .clone()
            .or_else(|| self.rope_scaling.as_ref().map(|r| r.mrope_section.clone()))
            .unwrap_or_else(|| vec![24, 20, 20]);
        <[u64; 3]>::try_from(v.as_slice()).map_err(|_| Error::Config(format!("{} rotary sections, expected 3", v.len())))
    }

    /// Weights the two graphs read. The text stream's output projection and
    /// feed-forward in the last layer, its final norm and the language-model
    /// head never contribute to a video and are not loaded.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, ff, hd) = (self.hidden_size, self.intermediate_size, self.head_dim);
        let (q, kv) = (self.num_attention_heads * hd, self.num_key_value_heads * hd);
        let mut v = vec![
            WeightSpec::new("embed_tokens.weight", &[self.vocab_size, d], linear),
            WeightSpec::new("proj_in.weight", &[d, self.patch_latent_dim], WType::F32),
            WeightSpec::new("proj_in.bias", &[d], WType::F32),
            WeightSpec::new("proj_out.weight", &[self.patch_latent_dim, d], WType::F32),
            WeightSpec::new("proj_out.bias", &[self.patch_latent_dim], WType::F32),
            WeightSpec::new("time_embedder.linear_1.weight", &[d, TIME_FEATURES as u64], WType::F32),
            WeightSpec::new("time_embedder.linear_1.bias", &[d], WType::F32),
            WeightSpec::new("time_embedder.linear_2.weight", &[d, d], WType::F32),
            WeightSpec::new("time_embedder.linear_2.bias", &[d], WType::F32),
            WeightSpec::new("norm_moe_gen.weight", &[d], WType::F32),
        ];
        if let (true, Some(ad)) = (self.action_gen, self.action_dim) {
            let nd = self.num_embodiment_domains;
            v.push(WeightSpec::new("action_proj_in.fc.weight", &[nd, ad * d], WType::F32));
            v.push(WeightSpec::new("action_proj_in.bias.weight", &[nd, d], WType::F32));
            v.push(WeightSpec::new("action_proj_out.fc.weight", &[nd, d * ad], WType::F32));
            v.push(WeightSpec::new("action_proj_out.bias.weight", &[nd, ad], WType::F32));
            v.push(WeightSpec::new("action_modality_embed", &[d], WType::F32));
        }
        let gated = self.gated();
        let last = self.num_hidden_layers - 1;
        for i in 0..self.num_hidden_layers {
            let p = format!("layers.{i}");
            let a = format!("{p}.self_attn");
            for n in ["input_layernorm", "input_layernorm_moe_gen", "post_attention_layernorm_moe_gen"] {
                v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[d], WType::F32));
            }
            let mut head_norms = vec!["norm_added_q", "norm_added_k"];
            if self.qk_norm_for_text {
                head_norms.push("norm_k");
                if i != last {
                    head_norms.push("norm_q");
                }
            } else if self.use_und_k_norm_for_gen {
                head_norms.push("k_norm_und_for_gen");
            }
            for n in head_norms {
                v.push(WeightSpec::new(format!("{a}.{n}.weight"), &[hd], WType::F32));
            }
            for (n, rows) in [("to_q", q), ("to_k", kv), ("to_v", kv), ("add_q_proj", q), ("add_k_proj", kv), ("add_v_proj", kv)] {
                v.push(WeightSpec::new(format!("{a}.{n}.weight"), &[rows, d], linear));
            }
            v.push(WeightSpec::new(format!("{a}.to_add_out.weight"), &[d, q], linear));
            v.push(WeightSpec::new(format!("{p}.mlp_moe_gen.up_proj.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.mlp_moe_gen.down_proj.weight"), &[d, ff], linear));
            if gated {
                v.push(WeightSpec::new(format!("{p}.mlp_moe_gen.gate_proj.weight"), &[ff, d], linear));
            }
            if i != last {
                if gated {
                    v.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[ff, d], linear));
                }
                v.push(WeightSpec::new(format!("{a}.to_out.weight"), &[d, q], linear));
                v.push(WeightSpec::new(format!("{p}.post_attention_layernorm.weight"), &[d], WType::F32));
                v.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[ff, d], linear));
                v.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[d, ff], linear));
            }
        }
        v
    }

    /// Specs of the per-prompt key/value store for `tokens` text tokens:
    /// `k{layer}` and `v{layer}`, each `[kv heads, tokens, head width]`.
    #[must_use]
    pub fn text_cache_specs(&self, tokens: usize) -> Vec<WeightSpec> {
        let shape = [self.num_key_value_heads, tokens as u64, self.head_dim];
        (0..self.num_hidden_layers)
            .flat_map(|i| [WeightSpec::new(format!("k{i}"), &shape, WType::F32), WeightSpec::new(format!("v{i}"), &shape, WType::F32)])
            .collect()
    }

    /// Gated (SiLU) feed-forward rather than squared ReLU.
    fn gated(&self) -> bool {
        self.hidden_act == "silu"
    }

    fn eps(&self) -> f32 {
        self.rms_norm_eps as f32
    }

    /// Rotary cos and sin tables, each `[tokens][head width]`, for positions
    /// `[tokens][time, height, width]`. Frequencies interleave the axes: pair
    /// `i` follows height when `i % 3 == 1`, width when `i % 3 == 2` (within
    /// three times that axis's pair count), time otherwise. Computed in single
    /// precision as the reference computes them.
    ///
    /// # Panics
    /// Never for a validated configuration.
    #[must_use]
    pub fn rotary_tables(&self, positions: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim as usize;
        let half = hd / 2;
        let axes = self.axes().expect("validated");
        let theta = self.rope_theta as f32;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / hd as f32)).collect();
        let axis = |i: usize| {
            if i % 3 == 1 && i < 3 * axes[1] as usize {
                1
            } else if i % 3 == 2 && i < 3 * axes[2] as usize {
                2
            } else {
                0
            }
        };
        let mut cos = Vec::with_capacity(positions.len() * hd);
        let mut sin = Vec::with_capacity(positions.len() * hd);
        for p in positions {
            let f: Vec<f32> = (0..half).map(|i| p[axis(i)] * inv[i]).collect();
            for j in 0..hd {
                cos.push(f[j % half].cos());
                sin.push(f[j % half].sin());
            }
        }
        (cos, sin)
    }

    /// The sinusoidal features of an integer scheduler timestep: `cos` then
    /// `sin` of `t * timestep_scale` at 128 log-spaced frequencies.
    #[must_use]
    pub fn time_features(&self, t: i64) -> Vec<f32> {
        let half = TIME_FEATURES / 2;
        let arg = t as f32 * self.timestep_scale as f32;
        let k = -(10000f64.ln() as f32);
        let freqs: Vec<f32> = (0..half).map(|i| (k * i as f32 / half as f32).exp()).collect();
        let mut out: Vec<f32> = freqs.iter().map(|f| (arg * f).cos()).collect();
        out.extend(freqs.iter().map(|f| (arg * f).sin()));
        out
    }
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    g.mul(h, w.get(name))
}

fn mlp(g: &mut Graph, cfg: &Cosmos3Config, w: &Weights, p: &str, x: Tn) -> Tn {
    let up = g.linear(w.get(&format!("{p}.up_proj.weight")), x);
    let h = if cfg.gated() {
        let gate = g.linear(w.get(&format!("{p}.gate_proj.weight")), x);
        let gate = g.silu(gate);
        g.mul(gate, up)
    } else {
        let h = g.relu(up);
        g.sqr(h)
    };
    g.linear(w.get(&format!("{p}.down_proj.weight")), h)
}

/// Per-embodiment projection: row `domain` of `{name}.fc` is an
/// `[in][out]` matrix (`x @ m + b`), row `domain` of `{name}.bias` the bias.
fn domain_linear(g: &mut Graph, w: &Weights, name: &str, domain: Tn, inp: i64, out: i64, x: Tn) -> Tn {
    let row = g.get_rows(w.get(&format!("{name}.fc.weight")), domain);
    let m = g.reshape(row, &[out, inp]);
    let m = g.permute(m, [1, 0, 2, 3]);
    let m = g.cont(m);
    let b = g.get_rows(w.get(&format!("{name}.bias.weight")), domain);
    let y = g.linear(m, x);
    g.add(y, b)
}

/// `q` `[hd, heads, n]`, `k`/`v` `[hd, m, kv heads]`; result `[hd * heads, n]`.
fn attend(g: &mut Graph, q: Tn, k: Tn, v: Tn, mask: Option<Tn>, exact: bool) -> Tn {
    let (hd, heads, n) = (q.ne(0), q.ne(1), q.ne(2));
    let scale = 1.0 / (hd as f32).sqrt();
    let q = g.permute(q, [0, 2, 1, 3]);
    let o = if exact {
        g.attention_exact(q, k, v, mask, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, mask, scale, true)
    };
    g.reshape(o, &[hd * heads, n])
}

/// `[hd, heads, n]` to `[hd, n, heads]`, contiguous.
fn by_head(g: &mut Graph, x: Tn) -> Tn {
    let p = g.permute(x, [0, 2, 1, 3]);
    g.cont(p)
}

/// Inputs and outputs of the text-stream graph.
#[derive(Debug, Clone)]
pub struct TextIo {
    /// Token ids `[tokens]` (i32).
    pub ids: Tn,
    /// Rotary cosine table `[head width, tokens]`.
    pub cos: Tn,
    /// Sine table, same layout.
    pub sin: Tn,
    /// Causal additive mask `[tokens, tokens]` (f16).
    pub mask: Tn,
    /// Per layer, the keys the generation stream reads (normed and rotated)
    /// and the values, each `[head width, tokens, kv heads]`.
    pub keys: Vec<Tn>,
    /// Per layer, the values, same layout.
    pub values: Vec<Tn>,
}

/// The text stream over `n` tokens, producing every layer's keys and values
/// for the generation stream.
#[must_use]
pub fn build_text(g: &mut Graph, cfg: &Cosmos3Config, w: &Weights, n: i64, exact: bool) -> TextIo {
    let (hd, heads, kvh) = (cfg.head_dim as i64, cfg.num_attention_heads as i64, cfg.num_key_value_heads as i64);
    let eps = cfg.eps();
    let ids = g.input(sys::GGML_TYPE_I32, &[n]);
    let cos = g.input(sys::GGML_TYPE_F32, &[hd, n]);
    let sin = g.input(sys::GGML_TYPE_F32, &[hd, n]);
    let mask = g.input(sys::GGML_TYPE_F16, &[n, n]);
    let cos3 = g.reshape(cos, &[hd, 1, n]);
    let sin3 = g.reshape(sin, &[hd, 1, n]);
    let mut x = g.get_rows(w.get("embed_tokens.weight"), ids);
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        let a = format!("{p}.self_attn");
        let wn = |s: &str| w.get(&format!("{a}.{s}.weight"));
        let h = rms(g, w, &format!("{p}.input_layernorm.weight"), x, eps);
        let q = g.linear(wn("to_q"), h);
        let k = g.linear(wn("to_k"), h);
        let v = g.linear(wn("to_v"), h);
        let k = g.reshape(k, &[hd, kvh, n]);
        let v = g.reshape(v, &[hd, kvh, n]);
        // With query/key norms the generation stream reads the normed keys;
        // without them, optionally keys under a norm of their own.
        let k = if cfg.qk_norm_for_text { rms(g, w, &format!("{a}.norm_k.weight"), k, eps) } else { k };
        let separate = !cfg.qk_norm_for_text && cfg.use_und_k_norm_for_gen;
        let kg = if separate { rms(g, w, &format!("{a}.k_norm_und_for_gen.weight"), k, eps) } else { k };
        let kg = g.rotate_half_rope(kg, cos3, sin3);
        let kg = by_head(g, kg);
        keys.push(kg);
        let v = by_head(g, v);
        values.push(v);
        if i + 1 == cfg.num_hidden_layers {
            break;
        }
        let q = g.reshape(q, &[hd, heads, n]);
        let q = if cfg.qk_norm_for_text { rms(g, w, &format!("{a}.norm_q.weight"), q, eps) } else { q };
        let q = g.rotate_half_rope(q, cos3, sin3);
        let k = if separate {
            let k = g.rotate_half_rope(k, cos3, sin3);
            by_head(g, k)
        } else {
            kg
        };
        let o = attend(g, q, k, v, Some(mask), exact);
        let o = g.linear(wn("to_out"), o);
        x = g.add(x, o);
        let h = rms(g, w, &format!("{p}.post_attention_layernorm.weight"), x, eps);
        let m = mlp(g, cfg, w, &format!("{p}.mlp"), h);
        x = g.add(x, m);
    }
    TextIo { ids, cos, sin, mask, keys, values }
}

/// The causal additive mask over `n` text tokens; row `i` is query `i`.
#[must_use]
pub fn causal_mask(n: usize) -> Vec<f32> {
    let mut m = vec![0f32; n * n];
    for q in 0..n {
        for k in q + 1..n {
            m[q * n + k] = f32::NEG_INFINITY;
        }
    }
    m
}

/// Action tokens appended to the generation stream.
#[derive(Debug, Clone, Copy)]
pub struct ActionSpan {
    /// Action tokens (one per action step).
    pub tokens: i64,
    /// Leading tokens that are given rather than predicted.
    pub cond: i64,
}

/// Inputs and outputs of the action part of the generation stream.
#[derive(Debug, Clone)]
pub struct ActionIo {
    /// Action vectors `[action width, tokens]`.
    pub values: Tn,
    /// Timestep features of the action tokens `[256]`.
    pub time: Tn,
    /// Embodiment index `[1]` (i32).
    pub domain: Tn,
    /// Velocity of the predicted actions `[action width, tokens - cond]`.
    pub out: Tn,
}

/// Inputs and outputs of the generation-stream graph.
#[derive(Debug, Clone)]
pub struct GenIo {
    /// Patch tokens `[patch width, tokens]`, conditioning frames first.
    pub patches: Tn,
    /// Timestep features `[256]`.
    pub time: Tn,
    /// Rotary cosine table `[head width, tokens]`, video tokens then action
    /// tokens.
    pub cos: Tn,
    /// Sine table, same layout.
    pub sin: Tn,
    /// Velocity of the noisy tokens `[patch width, tokens - cond]`.
    pub out: Tn,
    /// The action tokens, when the graph carries them.
    pub actions: Option<ActionIo>,
}

/// Add the timestep embedding of `time` to every column of `x` after the
/// first `cond`.
fn add_time(g: &mut Graph, w: &Weights, time: Tn, x: Tn, n: i64, cond: i64) -> Tn {
    let t = g.linear_b(w.get("time_embedder.linear_1.weight"), w.get("time_embedder.linear_1.bias"), time);
    let t = g.silu(t);
    let temb = g.linear_b(w.get("time_embedder.linear_2.weight"), w.get("time_embedder.linear_2.bias"), t);
    if cond == n {
        return x;
    }
    if cond > 0 {
        let c = g.view_cols(x, 0, cond);
        let rest = g.view_cols(x, cond, n - cond);
        let rest = g.add(rest, temb);
        let c = g.cont(c);
        g.concat(c, rest, 1)
    } else {
        g.add(x, temb)
    }
}

/// One evaluation of the generation stream over `n` patch tokens, the first
/// `cond` of which are clean conditioning tokens (no timestep embedding, no
/// prediction), followed by the action tokens of `actions` if given,
/// attending to the text keys and values in `cache`. At least one video
/// token must be predicted.
///
/// # Panics
/// When `actions` is given and the configuration has no action width.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_gen(
    g: &mut Graph,
    cfg: &Cosmos3Config,
    w: &Weights,
    cache: &Weights,
    n: i64,
    cond: i64,
    actions: Option<ActionSpan>,
    exact: bool,
) -> GenIo {
    let (hd, heads, kvh) = (cfg.head_dim as i64, cfg.num_attention_heads as i64, cfg.num_key_value_heads as i64);
    let (d, eps) = (cfg.hidden_size as i64, cfg.eps());
    let na = actions.map_or(0, |a| a.tokens);
    let total = n + na;
    let patches = g.input(sys::GGML_TYPE_F32, &[cfg.patch_latent_dim as i64, n]);
    let time = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
    let cos = g.input(sys::GGML_TYPE_F32, &[hd, total]);
    let sin = g.input(sys::GGML_TYPE_F32, &[hd, total]);
    let cos3 = g.reshape(cos, &[hd, 1, total]);
    let sin3 = g.reshape(sin, &[hd, 1, total]);

    let x = g.linear_b(w.get("proj_in.weight"), w.get("proj_in.bias"), patches);
    let mut x = add_time(g, w, time, x, n, cond);
    let mut action = None;
    if let Some(span) = actions {
        let ad = cfg.action_dim.expect("action width") as i64;
        let values = g.input(sys::GGML_TYPE_F32, &[ad, span.tokens]);
        let atime = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
        let domain = g.input(sys::GGML_TYPE_I32, &[1]);
        let xa = domain_linear(g, w, "action_proj_in", domain, ad, d, values);
        let xa = g.add(xa, w.get("action_modality_embed"));
        let xa = add_time(g, w, atime, xa, span.tokens, span.cond);
        x = g.concat(x, xa, 1);
        action = Some((span, values, atime, domain));
    }
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        let a = format!("{p}.self_attn");
        let wn = |s: &str| w.get(&format!("{a}.{s}.weight"));
        let h = rms(g, w, &format!("{p}.input_layernorm_moe_gen.weight"), x, eps);
        let q = g.linear(wn("add_q_proj"), h);
        let k = g.linear(wn("add_k_proj"), h);
        let v = g.linear(wn("add_v_proj"), h);
        let q = g.reshape(q, &[hd, heads, total]);
        let k = g.reshape(k, &[hd, kvh, total]);
        let v = g.reshape(v, &[hd, kvh, total]);
        let q = rms(g, w, &format!("{a}.norm_added_q.weight"), q, eps);
        let k = rms(g, w, &format!("{a}.norm_added_k.weight"), k, eps);
        let q = g.rotate_half_rope(q, cos3, sin3);
        let k = g.rotate_half_rope(k, cos3, sin3);
        let k = by_head(g, k);
        let v = by_head(g, v);
        let k = g.concat(cache.get(&format!("k{i}")), k, 1);
        let v = g.concat(cache.get(&format!("v{i}")), v, 1);
        let o = attend(g, q, k, v, None, exact);
        let o = g.linear(wn("to_add_out"), o);
        x = g.add(x, o);
        let h = rms(g, w, &format!("{p}.post_attention_layernorm_moe_gen.weight"), x, eps);
        let m = mlp(g, cfg, w, &format!("{p}.mlp_moe_gen"), h);
        x = g.add(x, m);
    }
    let h = rms(g, w, "norm_moe_gen.weight", x, eps);
    let hv = g.view_cols(h, cond, n - cond);
    let hv = g.cont(hv);
    let out = g.linear_b(w.get("proj_out.weight"), w.get("proj_out.bias"), hv);
    let actions = action.map(|(span, values, time, domain)| {
        let ad = cfg.action_dim.expect("action width") as i64;
        let ha = g.view_cols(h, n + span.cond, span.tokens - span.cond);
        let ha = g.cont(ha);
        let out = domain_linear(g, w, "action_proj_out", domain, d, ad, ha);
        ActionIo { values, time, domain, out }
    });
    GenIo { patches, time, cos, sin, out, actions }
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cosmos3Config {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128,
            "rms_norm_eps": 1e-5, "rope_theta": 1e8, "rope_scaling": {"mrope_section": [24, 20, 20]},
            "latent_channel": 48, "latent_patch_size": 2, "patch_latent_dim": 192,
            "timestep_scale": 0.001, "hidden_act": "relu2", "qk_norm_for_text": false,
            "use_und_k_norm_for_gen": true, "vocab_size": 10, "base_fps": 24,
            "enable_fps_modulation": true, "unified_3d_mrope_temporal_modality_margin": 15000,
            "unified_3d_mrope_reset_spatial_ids": true
        }))
        .unwrap()
    }

    #[test]
    fn the_released_layout_validates() {
        cfg().validate().unwrap();
    }

    #[test]
    fn rotary_pairs_interleave_time_height_and_width() {
        let c = cfg();
        // Only the height axis moves: pairs 1, 4, ..., 58 rotate, the rest
        // stay at angle zero.
        let (cos, sin) = c.rotary_tables(&[[0.0, 1.0, 0.0]]);
        for i in 0..64 {
            let moved = sin[i] != 0.0;
            assert_eq!(moved, i % 3 == 1 && i < 60, "pair {i}");
            assert_eq!(cos[i], cos[i + 64]);
            assert_eq!(sin[i], sin[i + 64]);
        }
    }

    #[test]
    fn the_causal_mask_hides_later_tokens() {
        let m = causal_mask(3);
        assert_eq!(m[0], 0.0);
        assert!(m[1].is_infinite() && m[2].is_infinite() && m[5].is_infinite());
        assert_eq!(m[3], 0.0);
        assert_eq!(m[8], 0.0);
    }
}
