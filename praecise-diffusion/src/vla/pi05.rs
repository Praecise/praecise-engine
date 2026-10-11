//! pi0.5: a PaliGemma backbone with a Gemma action expert.
//!
//! Each camera frame is resized with centred padding to the vision input,
//! encoded by a SigLIP tower and linearly projected to the decoder width.
//! The measured state is normalised, discretised into 256 bins over
//! `[-1, 1]` and written into the prompt (`Task: ..., State: ...;\nAction: `),
//! whose token embeddings are scaled by the square root of the width. The
//! whole prefix attends bidirectionally; one pass fills a key/value cache
//! per layer.
//!
//! The expert is a narrower Gemma whose norms are adaptive: a linear map of
//! the flow-matching time condition gives a scale, a shift and a residual
//! gate for each norm. Its tokens are the projected noisy action chunk; they
//! attend to the whole prefix and to each other, at positions after the
//! prefix. Gemma norms scale by `1 + weight`, the feed-forward is a
//! tanh-GELU gated MLP, rotary positions are split-half with base 10000.
//! The chunk is integrated from `t = 1` to `t = 0` by forward Euler; the
//! prefix pass and every step run as one graph.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use super::siglip::{self, SiglipConfig};
use super::{attend, euler_times, read_json, resize_with_pad, sinusoidal_time, ChunkPolicy, Normalization, PadAt, VlaObservation};
use crate::error::{Error, Result};
use crate::flux3::packing::Frame;
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const ROPE_NEOX: i32 = 2;
const ROPE_BASE: f32 = 10_000.0;
const RMS_EPS: f32 = 1e-6;

const VLM: &str = "model.paligemma_with_expert.paligemma.model.";
const EXPERT: &str = "model.paligemma_with_expert.gemma_expert.model.";
const HEAD: &str = "model.";

/// Sizes of one Gemma decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDims {
    /// Width.
    pub width: u64,
    /// Layers.
    pub depth: usize,
    /// MLP width.
    pub mlp: u64,
    /// Query heads.
    pub heads: u64,
    /// Key/value heads.
    pub kv_heads: u64,
    /// Head width.
    pub head_dim: u64,
}

impl GemmaDims {
    /// The sizes of a named variant.
    #[must_use]
    pub fn variant(name: &str) -> Option<Self> {
        match name {
            "gemma_300m" => Some(Self { width: 1024, depth: 18, mlp: 4096, heads: 8, kv_heads: 1, head_dim: 256 }),
            "gemma_2b" => Some(Self { width: 2048, depth: 18, mlp: 16_384, heads: 8, kv_heads: 1, head_dim: 256 }),
            _ => None,
        }
    }
}

/// Sizes of the whole network.
#[derive(Debug, Clone)]
pub struct Pi05Dims {
    /// Backbone decoder.
    pub vlm: GemmaDims,
    /// Action expert.
    pub expert: GemmaDims,
    /// Vision tower.
    pub vision: SiglipConfig,
    /// Vocabulary.
    pub vocab: u64,
}

impl Pi05Dims {
    /// The released network: the named decoder variants and a SigLIP
    /// so400m tower at `image_size`.
    ///
    /// # Errors
    /// For a variant name this engine does not know.
    pub fn released(vlm: &str, expert: &str, image_size: usize) -> Result<Self> {
        let get = |n: &str| GemmaDims::variant(n).ok_or_else(|| Error::Config(format!("gemma variant {n:?} is not implemented")));
        Ok(Self {
            vlm: get(vlm)?,
            expert: get(expert)?,
            vision: SiglipConfig {
                hidden_size: 1152,
                num_hidden_layers: 27,
                num_attention_heads: 16,
                intermediate_size: 4304,
                patch_size: 14,
                image_size,
                layer_norm_eps: 1e-6,
            },
            vocab: 257_152,
        })
    }
}

/// A pi0.5 release's settings.
#[derive(Debug, Clone)]
pub struct Pi05Config {
    /// Camera feature names, in the order frames are given.
    pub camera_keys: Vec<String>,
    /// Measured state width.
    pub state_dim: usize,
    /// Action width.
    pub action_dim: usize,
    /// Actions per chunk.
    pub chunk_size: usize,
    /// Actions executed per chunk.
    pub n_action_steps: usize,
    /// Action width the policy integrates.
    pub max_action_dim: usize,
    /// Prompt tokens kept.
    pub tokenizer_max_length: usize,
    /// Euler steps.
    pub num_steps: usize,
    /// Time embedding periods.
    pub min_period: f64,
    /// Time embedding periods.
    pub max_period: f64,
    /// Network sizes.
    pub dims: Pi05Dims,
}

fn get_usize(v: &Value, k: &str) -> Result<usize> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error::Config(format!("config gives no {k}")))
}

fn get_f64(v: &Value, k: &str) -> Result<f64> {
    v.get(k).and_then(Value::as_f64).ok_or_else(|| Error::Config(format!("config gives no {k}")))
}

impl Pi05Config {
    /// Read a policy config. `dims` overrides the released network sizes.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field or a variant this engine does
    /// not implement.
    pub fn from_json(policy: &Value, dims: Option<Pi05Dims>) -> Result<Self> {
        if policy.get("type").and_then(Value::as_str) != Some("pi05") {
            return Err(Error::Config("not a pi05 policy config".into()));
        }
        for (k, off) in [("use_visual_memory", false), ("use_proprioceptive_memory", false), ("use_relative_actions", false)] {
            if policy.get(k).and_then(Value::as_bool).unwrap_or(off) != off {
                return Err(Error::Config(format!("pi05 {k} is not implemented")));
            }
        }
        if policy.get("empty_cameras").and_then(Value::as_u64).unwrap_or(0) != 0 {
            return Err(Error::Config("pi05 empty_cameras is not implemented".into()));
        }
        if policy.get("rtc_config").is_some_and(|v| !v.is_null()) {
            return Err(Error::Config("pi05 real-time chunking is not implemented".into()));
        }
        let features = policy.get("input_features").and_then(Value::as_object).ok_or_else(|| Error::Config("config gives no input_features".into()))?;
        let mut camera_keys = Vec::new();
        let mut state_dim = None;
        for (k, f) in features {
            match f.get("type").and_then(Value::as_str) {
                Some("VISUAL") => camera_keys.push(k.clone()),
                Some("STATE") => state_dim = f.pointer("/shape/0").and_then(Value::as_u64).map(|x| x as usize),
                _ => {}
            }
        }
        let action_dim = policy
            .pointer("/output_features/action/shape/0")
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .ok_or_else(|| Error::Config("config gives no action shape".into()))?;
        let res = policy.get("image_resolution").and_then(Value::as_array).cloned().unwrap_or_default();
        let (h, w) = (res.first().and_then(Value::as_u64).unwrap_or(224) as usize, res.get(1).and_then(Value::as_u64).unwrap_or(224) as usize);
        if h != w {
            return Err(Error::Config("pi05 needs a square image resolution".into()));
        }
        let dims = match dims {
            Some(d) => d,
            None => Pi05Dims::released(
                policy.get("paligemma_variant").and_then(Value::as_str).unwrap_or("gemma_2b"),
                policy.get("action_expert_variant").and_then(Value::as_str).unwrap_or("gemma_300m"),
                h,
            )?,
        };
        if dims.vlm.depth != dims.expert.depth || dims.vlm.head_dim != dims.expert.head_dim || dims.vlm.kv_heads != dims.expert.kv_heads || dims.vlm.heads != dims.expert.heads {
            return Err(Error::Config("backbone and expert attention shapes differ".into()));
        }
        let max_action_dim = get_usize(policy, "max_action_dim")?;
        if action_dim > max_action_dim {
            return Err(Error::Config("action wider than the policy integrates".into()));
        }
        Ok(Self {
            camera_keys,
            state_dim: state_dim.ok_or_else(|| Error::Config("config gives no state feature".into()))?,
            action_dim,
            chunk_size: get_usize(policy, "chunk_size")?,
            n_action_steps: get_usize(policy, "n_action_steps")?,
            max_action_dim,
            tokenizer_max_length: get_usize(policy, "tokenizer_max_length")?,
            num_steps: get_usize(policy, "num_inference_steps")?,
            min_period: get_f64(policy, "min_period")?,
            max_period: get_f64(policy, "max_period")?,
            dims,
        })
    }

    /// Every weight the policy reads.
    #[must_use]
    pub fn weight_specs(&self, linear: WType, embed: WType) -> Vec<WeightSpec> {
        let f = WType::F32;
        let (v, e) = (self.dims.vlm, self.dims.expert);
        let mut s = self.dims.vision.weight_specs(&format!("{VLM}vision_tower.vision_model."), linear);
        s.push(WeightSpec::new(format!("{VLM}multi_modal_projector.linear.weight"), &[v.width, self.dims.vision.hidden_size], linear));
        s.push(WeightSpec::new(format!("{VLM}multi_modal_projector.linear.bias"), &[v.width], f));
        s.push(WeightSpec::new(format!("{VLM}language_model.embed_tokens.weight"), &[self.dims.vocab, v.width], embed));
        let layer = |s: &mut Vec<WeightSpec>, p: String, g: GemmaDims, adaptive: bool| {
            let (q, kv) = (g.heads * g.head_dim, g.kv_heads * g.head_dim);
            for n in ["input_layernorm", "post_attention_layernorm"] {
                if adaptive {
                    s.push(WeightSpec::new(format!("{p}.{n}.dense.weight"), &[3 * g.width, g.width], f));
                    s.push(WeightSpec::new(format!("{p}.{n}.dense.bias"), &[3 * g.width], f));
                } else {
                    s.push(WeightSpec::new(format!("{p}.{n}.weight"), &[g.width], f));
                }
            }
            s.push(WeightSpec::new(format!("{p}.self_attn.q_proj.weight"), &[q, g.width], linear));
            s.push(WeightSpec::new(format!("{p}.self_attn.k_proj.weight"), &[kv, g.width], linear));
            s.push(WeightSpec::new(format!("{p}.self_attn.v_proj.weight"), &[kv, g.width], linear));
            s.push(WeightSpec::new(format!("{p}.self_attn.o_proj.weight"), &[g.width, q], linear));
            s.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[g.mlp, g.width], linear));
            s.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[g.mlp, g.width], linear));
            s.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[g.width, g.mlp], linear));
        };
        for i in 0..v.depth {
            layer(&mut s, format!("{VLM}language_model.layers.{i}"), v, false);
            layer(&mut s, format!("{EXPERT}layers.{i}"), e, true);
        }
        s.push(WeightSpec::new(format!("{EXPERT}norm.dense.weight"), &[3 * e.width, e.width], f));
        s.push(WeightSpec::new(format!("{EXPERT}norm.dense.bias"), &[3 * e.width], f));
        let ma = self.max_action_dim as u64;
        for (n, o, i) in [("action_in_proj", e.width, ma), ("action_out_proj", ma, e.width), ("time_mlp_in", e.width, e.width), ("time_mlp_out", e.width, e.width)] {
            s.push(WeightSpec::new(format!("{HEAD}{n}.weight"), &[o, i], f));
            s.push(WeightSpec::new(format!("{HEAD}{n}.bias"), &[o], f));
        }
        s
    }
}

/// The 256-bin index of each normalised state value: the number of bin
/// edges `-1 + 2k/256` (k = 0..255) at or below it, minus one.
#[must_use]
pub fn discretize_state(state: &[f32]) -> Vec<i32> {
    state
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            (0..256).filter(|&k| -1.0 + f64::from(k) * (2.0 / 256.0) <= x).count() as i32 - 1
        })
        .collect()
}

/// The prompt for an instruction and a discretised state.
#[must_use]
pub fn prompt(instruction: &str, bins: &[i32]) -> String {
    let task = instruction.trim().replace('_', " ").replace('\n', " ");
    let state: Vec<String> = bins.iter().map(i32::to_string).collect();
    format!("Task: {task}, State: {};\nAction: ", state.join(" "))
}

struct ChunkIo {
    patches: Vec<Tn>,
    tokens: Tn,
    prefix_pos: Tn,
    noise: Tn,
    times: Tn,
    suffix_pos: Tn,
    out: Tn,
}

/// A loaded pi0.5 policy.
pub struct Pi05 {
    cfg: Pi05Config,
    norm: Normalization,
    tokenizer: Tokenizer,
    backend: Backend,
    weights: Weights,
    exact: bool,
}

impl std::fmt::Debug for Pi05 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pi05").field("device", &self.backend.name()).field("exact", &self.exact).finish_non_exhaustive()
    }
}

fn lin_b(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{name}.weight")), w.get(&format!("{name}.bias")), x)
}

/// Gemma RMS norm: scale by `1 + weight`.
fn gemma_norm(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    let h = g.rms_norm(x, RMS_EPS);
    let s = g.scale_bias(w.get(name), 1.0, 1.0);
    g.mul(h, s)
}

/// Adaptive RMS norm: the condition's scale and shift modulate the
/// normalised input; the gate is returned for the residual.
fn adaptive_norm(g: &mut Graph, w: &Weights, name: &str, x: Tn, cond: Tn, width: i64) -> (Tn, Tn) {
    let m = lin_b(g, w, &format!("{name}.dense"), cond);
    let scale = g.view_rows(m, 0, width);
    let scale = g.cont(scale);
    let shift = g.view_rows(m, width, width);
    let shift = g.cont(shift);
    let gate = g.view_rows(m, 2 * width, width);
    let gate = g.cont(gate);
    let h = g.rms_norm(x, RMS_EPS);
    let s = g.scale_bias(scale, 1.0, 1.0);
    let h = g.mul(h, s);
    (g.add(h, shift), gate)
}

impl Pi05 {
    /// Load a release: `policy` holds `config.json`, `model.safetensors` and
    /// the processor files; `tokenizer` the backbone's `tokenizer.json`.
    /// `dims` overrides the released network sizes.
    ///
    /// # Errors
    /// On a missing or malformed file, an unimplemented variant, or a
    /// backend failure.
    pub fn load(policy: &Path, tokenizer: &Path, opts: LoadOptions, dims: Option<Pi05Dims>) -> Result<Self> {
        let pc = read_json(&policy.join("config.json"))?;
        let cfg = Pi05Config::from_json(&pc, dims)?;
        let norm = Normalization::from_release(policy, &pc)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let (linear, embed) = match opts.precision {
            Precision::F32 => (WType::F32, WType::F32),
            Precision::Bf16 => (WType::Bf16, WType::F16),
            Precision::Q8_0 => (WType::Q8_0, WType::F16),
        };
        let st = SafeTensors::open(&[policy.join("model.safetensors")])?;
        // Releases store the network either under the policy's `model.`
        // or at the top level.
        let st = if st.get("action_in_proj.weight").is_some() { st.renamed(|n| Some(format!("{HEAD}{n}")))? } else { st };
        // Newer savers drop the `vision_model.` level under the vision tower.
        let flat = format!("{VLM}vision_tower.");
        let st = if st.get(&format!("{flat}embeddings.patch_embedding.weight")).is_some() {
            st.renamed(|n| Some(n.strip_prefix(flat.as_str()).map_or_else(|| n.to_string(), |rest| format!("{flat}vision_model.{rest}"))))?
        } else {
            st
        };
        let weights = Weights::load(&backend, &st, &cfg.weight_specs(linear, embed))?;
        Ok(Self { cfg, norm, tokenizer, backend, weights, exact })
    }

    /// Execute `n` actions of each chunk instead of the release's count.
    pub fn set_n_action_steps(&mut self, n: usize) {
        self.cfg.n_action_steps = n.clamp(1, self.cfg.chunk_size);
    }

    /// The release's settings.
    #[must_use]
    pub fn config(&self) -> &Pi05Config {
        &self.cfg
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Bytes of weights resident on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.weights.bytes()
    }

    /// Prompt token ids for an instruction and a normalised state.
    ///
    /// # Errors
    /// On a tokenizer failure.
    pub fn tokens(&self, instruction: &str, normalized_state: &[f32]) -> Result<Vec<i32>> {
        let text = prompt(instruction, &discretize_state(normalized_state));
        let enc = self.tokenizer.encode(text, true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().iter().take(self.cfg.tokenizer_max_length).map(|&i| i as i32).collect())
    }

    /// One frame as the vision tower reads it.
    #[must_use]
    pub fn prepare_frame(&self, f: &Frame) -> Frame {
        let s = self.cfg.dims.vision.image_size;
        let mut r = resize_with_pad(f, s, s, PadAt::Centre, 0.0);
        for v in &mut r.data {
            *v = v.clamp(0.0, 1.0) * 2.0 - 1.0;
        }
        r
    }

    fn build(&self, g: &mut Graph, cameras: usize, n_tok: usize) -> ChunkIo {
        let c = &self.cfg;
        let w = &self.weights;
        let (v, e) = (c.dims.vlm, c.dims.expert);
        let (hd, nh, nkv) = (v.head_dim as i64, v.heads as i64, v.kv_heads as i64);
        let ew = e.width as i64;
        let chunk = c.chunk_size as i64;
        let n_img = c.dims.vision.patches();
        let n_pre = (cameras * n_img + n_tok) as i64;
        let scale = 1.0 / (hd as f32).sqrt();
        let exact = self.exact;
        let mlp = |g: &mut Graph, p: &str, h: Tn| -> Tn {
            let gate = g.linear(w.get(&format!("{p}.mlp.gate_proj.weight")), h);
            let up = g.linear(w.get(&format!("{p}.mlp.up_proj.weight")), h);
            let gate = if exact { g.gelu_tanh_exact(gate) } else { g.gelu_tanh(gate) };
            let m = g.mul(gate, up);
            g.linear(w.get(&format!("{p}.mlp.down_proj.weight")), m)
        };

        let mut patches = Vec::with_capacity(cameras);
        let mut x: Option<Tn> = None;
        for _ in 0..cameras {
            let p = g.input(sys::GGML_TYPE_F32, &[c.dims.vision.patch_len() as i64, n_img as i64]);
            patches.push(p);
            let y = siglip::build(g, &c.dims.vision, w, &format!("{VLM}vision_tower.vision_model."), p, exact);
            let y = lin_b(g, w, &format!("{VLM}multi_modal_projector.linear"), y);
            x = Some(match x {
                None => y,
                Some(prev) => g.concat(prev, y, 1),
            });
        }
        let tokens = g.input(sys::GGML_TYPE_I32, &[n_tok as i64]);
        let lang = g.get_rows(w.get(&format!("{VLM}language_model.embed_tokens.weight")), tokens);
        let lang = g.scale_bias(lang, (v.width as f32).sqrt(), 0.0);
        let mut x = g.concat(x.expect("at least one camera"), lang, 1);

        let prefix_pos = g.input(sys::GGML_TYPE_I32, &[n_pre]);
        let mut cache: Vec<(Tn, Tn)> = Vec::with_capacity(v.depth);
        for i in 0..v.depth {
            let p = format!("{VLM}language_model.layers.{i}");
            let h = gemma_norm(g, w, &format!("{p}.input_layernorm.weight"), x);
            let q = g.linear(w.get(&format!("{p}.self_attn.q_proj.weight")), h);
            let k = g.linear(w.get(&format!("{p}.self_attn.k_proj.weight")), h);
            let vv = g.linear(w.get(&format!("{p}.self_attn.v_proj.weight")), h);
            let q = g.reshape(q, &[hd, nh, n_pre]);
            let k = g.reshape(k, &[hd, nkv, n_pre]);
            let vv = g.reshape(vv, &[hd, nkv, n_pre]);
            let q = g.rope(q, prefix_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
            let k = g.rope(k, prefix_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
            cache.push((k, vv));
            if i + 1 == v.depth {
                break;
            }
            let o = attend(g, q, k, vv, None, scale, exact);
            let o = g.reshape(o, &[hd * nh, n_pre]);
            let o = g.linear(w.get(&format!("{p}.self_attn.o_proj.weight")), o);
            x = g.add(x, o);
            let h = gemma_norm(g, w, &format!("{p}.post_attention_layernorm.weight"), x);
            let m = mlp(g, &p, h);
            x = g.add(x, m);
        }

        let noise = g.input(sys::GGML_TYPE_F32, &[c.max_action_dim as i64, chunk]);
        let times = g.input(sys::GGML_TYPE_F32, &[ew, c.num_steps as i64]);
        let suffix_pos = g.input(sys::GGML_TYPE_I32, &[chunk]);
        let (_, dt) = euler_times(c.num_steps);
        let mut xt = noise;
        for step in 0..c.num_steps {
            let te = g.view_cols(times, step as i64, 1);
            let cond = lin_b(g, w, &format!("{HEAD}time_mlp_in"), te);
            let cond = g.silu(cond);
            let cond = lin_b(g, w, &format!("{HEAD}time_mlp_out"), cond);
            let cond = g.silu(cond);
            let mut h = lin_b(g, w, &format!("{HEAD}action_in_proj"), xt);
            for (i, &(pk, pv)) in cache.iter().enumerate() {
                let p = format!("{EXPERT}layers.{i}");
                let (hn, gate) = adaptive_norm(g, w, &format!("{p}.input_layernorm"), h, cond, ew);
                let q = g.linear(w.get(&format!("{p}.self_attn.q_proj.weight")), hn);
                let k = g.linear(w.get(&format!("{p}.self_attn.k_proj.weight")), hn);
                let vv = g.linear(w.get(&format!("{p}.self_attn.v_proj.weight")), hn);
                let q = g.reshape(q, &[hd, nh, chunk]);
                let k = g.reshape(k, &[hd, nkv, chunk]);
                let vv = g.reshape(vv, &[hd, nkv, chunk]);
                let q = g.rope(q, suffix_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
                let k = g.rope(k, suffix_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
                let k = g.concat(pk, k, 2);
                let vv = g.concat(pv, vv, 2);
                let o = attend(g, q, k, vv, None, scale, exact);
                let o = g.reshape(o, &[hd * nh, chunk]);
                let o = g.linear(w.get(&format!("{p}.self_attn.o_proj.weight")), o);
                let o = g.mul(o, gate);
                h = g.add(h, o);
                let (hn, gate) = adaptive_norm(g, w, &format!("{p}.post_attention_layernorm"), h, cond, ew);
                let m = mlp(g, &p, hn);
                let m = g.mul(m, gate);
                h = g.add(h, m);
            }
            let (h, _) = adaptive_norm(g, w, &format!("{EXPERT}norm"), h, cond, ew);
            let vel = lin_b(g, w, &format!("{HEAD}action_out_proj"), h);
            let vel = g.scale_bias(vel, dt, 0.0);
            xt = g.add(xt, vel);
        }
        ChunkIo { patches, tokens, prefix_pos, noise, times, suffix_pos, out: xt }
    }

    /// One chunk `[chunk_size][max_action_dim]` in normalised units, from
    /// prepared frames ([`Pi05::prepare_frame`]) and prompt token ids.
    ///
    /// # Errors
    /// On a malformed input or a backend failure.
    pub fn sample(&self, frames: &[Frame], tokens: &[i32], noise: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if frames.is_empty() || frames.len() != c.camera_keys.len() {
            return Err(Error::Request(format!("the policy reads {} cameras", c.camera_keys.len())));
        }
        if noise.len() != c.chunk_size * c.max_action_dim || tokens.is_empty() {
            return Err(Error::Request("empty prompt or noise of the wrong length".into()));
        }
        let n_pre = frames.len() * c.dims.vision.patches() + tokens.len();
        let mut g = Graph::new(&self.backend)?;
        let io = self.build(&mut g, frames.len(), tokens.len());
        g.finish(&[io.out])?;
        for (p, f) in io.patches.iter().zip(frames) {
            g.set_f32(*p, &c.dims.vision.patchify(f));
        }
        g.set_i32(io.tokens, tokens);
        let pos: Vec<i32> = (0..n_pre as i32).collect();
        g.set_i32(io.prefix_pos, &pos);
        g.set_f32(io.noise, noise);
        let (ts, _) = euler_times(c.num_steps);
        let ew = c.dims.expert.width as usize;
        let times: Vec<f32> = ts.iter().flat_map(|&t| sinusoidal_time(t, ew, c.min_period, c.max_period)).collect();
        g.set_f32(io.times, &times);
        let sp: Vec<i32> = (0..c.chunk_size as i32).map(|j| n_pre as i32 + j).collect();
        g.set_i32(io.suffix_pos, &sp);
        g.compute()?;
        Ok(g.read_f32(io.out))
    }
}

impl ChunkPolicy for Pi05 {
    fn chunk_size(&self) -> usize {
        self.cfg.chunk_size
    }
    fn action_dim(&self) -> usize {
        self.cfg.action_dim
    }
    fn n_action_steps(&self) -> usize {
        self.cfg.n_action_steps
    }
    fn noise_len(&self) -> usize {
        self.cfg.chunk_size * self.cfg.max_action_dim
    }
    fn predict(&self, obs: &VlaObservation<'_>, noise: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if obs.state.len() != c.state_dim {
            return Err(Error::Request(format!("the policy reads a state of {} values", c.state_dim)));
        }
        let frames: Vec<Frame> = obs.cameras.iter().map(|f| self.prepare_frame(f)).collect();
        let state = self.norm.state.normalize(obs.state)?;
        let tokens = self.tokens(obs.instruction, &state)?;
        let raw = self.sample(&frames, &tokens, noise)?;
        let cut: Vec<f32> = raw.chunks(c.max_action_dim).flat_map(|r| r[..c.action_dim].to_vec()).collect();
        self.norm.action.unnormalize(&cut)
    }
}
