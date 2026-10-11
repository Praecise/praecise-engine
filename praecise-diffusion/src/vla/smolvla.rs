//! SmolVLA: a compact vision-language-action policy.
//!
//! The backbone is a small vision-language model cut to its first
//! `num_vlm_layers` decoder layers. Each camera frame is resized with
//! padding to the vision tower's input, encoded by a SigLIP tower,
//! pixel-shuffled (`scale_factor` squared patches per token) and projected
//! to the decoder width. The prefix is the image tokens, the instruction
//! tokens (both scaled by the square root of the width) and one projected
//! state token; images and instruction attend to each other and the state
//! token attends to everything before it. One pass over the prefix fills a
//! key/value cache per layer.
//!
//! The action expert is a narrower decoder with one layer per backbone
//! layer. Its tokens are the noisy action chunk fused with a sinusoidal
//! time embedding. Every `self_attn_every_n_layers`-th layer the expert
//! attends to the cached prefix keys and values plus its own, causally
//! among the action tokens; the other layers cross-attend to the cached
//! prefix only, through the expert's own key and value projections of the
//! cached (already rotated) prefix keys and values. Rotary positions are
//! split-half with base 10000; the action tokens follow the prefix in
//! self-attention layers and start at zero in cross-attention layers.
//!
//! The chunk is integrated from noise at `t = 1` to `t = 0` by forward
//! Euler in `num_steps` steps; the whole prefix pass and every step run as
//! one graph.

use std::path::{Path, PathBuf};

use serde_json::Value;
use tokenizers::Tokenizer;

use super::siglip::{self, SiglipConfig};
use super::{euler_times, pad_to, read_json, resize_with_pad, sinusoidal_time, ChunkPolicy, Normalization, PadAt, VlaObservation};
use crate::error::{Error, Result};
use crate::flux3::packing::Frame;
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// ggml's split-half rotary mode, `GGML_ROPE_TYPE_NEOX`.
const ROPE_NEOX: i32 = 2;
/// The rotary base every layer uses.
const ROPE_BASE: f32 = 10_000.0;

const VLM: &str = "model.vlm_with_expert.vlm.model.";
const EXPERT: &str = "model.vlm_with_expert.lm_expert.";
const HEAD: &str = "model.";

/// Decoder settings of the backbone (`text_config`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    /// Width.
    pub hidden_size: u64,
    /// MLP width.
    pub intermediate_size: u64,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Key/value heads.
    pub num_key_value_heads: u64,
    /// Head width.
    pub head_dim: u64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Vocabulary.
    pub vocab_size: u64,
}

/// A SmolVLA release's settings: the policy's `config.json` and the
/// backbone's.
#[derive(Debug, Clone)]
pub struct SmolVlaConfig {
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
    /// State width the policy pads to.
    pub max_state_dim: usize,
    /// Action width the policy integrates.
    pub max_action_dim: usize,
    /// Frames are resized with padding to this `(height, width)`.
    pub resize: (usize, usize),
    /// Instruction tokens kept.
    pub tokenizer_max_length: usize,
    /// Euler steps.
    pub num_steps: usize,
    /// Backbone decoder layers used.
    pub num_vlm_layers: usize,
    /// Expert layers with self-attention: every this many.
    pub self_attn_every_n_layers: usize,
    /// Expert width.
    pub expert_hidden: u64,
    /// Expert MLP width.
    pub expert_intermediate: u64,
    /// Time embedding periods.
    pub min_period: f64,
    /// Time embedding periods.
    pub max_period: f64,
    /// Backbone decoder.
    pub text: TextConfig,
    /// Vision tower.
    pub vision: SiglipConfig,
    /// Patches merged per image token, per side.
    pub scale_factor: usize,
}

fn get_usize(v: &Value, k: &str) -> Result<usize> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error::Config(format!("config gives no {k}")))
}

fn get_f64(v: &Value, k: &str) -> Result<f64> {
    v.get(k).and_then(Value::as_f64).ok_or_else(|| Error::Config(format!("config gives no {k}")))
}

/// The expert's MLP width for an expert width `d`: two thirds of four
/// times the width, rounded up to a multiple of 256.
#[must_use]
pub fn expert_intermediate(d: u64) -> u64 {
    let h = 4 * (2 * d / 3);
    h.div_ceil(256) * 256
}

impl SmolVlaConfig {
    /// Read a policy config and its backbone's.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field or a variant this engine does
    /// not implement (it is refused, never approximated).
    pub fn from_json(policy: &Value, backbone: &Value) -> Result<Self> {
        if policy.get("type").and_then(Value::as_str) != Some("smolvla") {
            return Err(Error::Config("not a smolvla policy config".into()));
        }
        let refuse = |k: &str, ok: &dyn Fn(&Value) -> bool| -> Result<()> {
            match policy.get(k) {
                Some(v) if !ok(v) => Err(Error::Config(format!("smolvla {k} = {v} is not implemented"))),
                _ => Ok(()),
            }
        };
        refuse("attention_mode", &|v| v.as_str() == Some("cross_attn"))?;
        refuse("add_image_special_tokens", &|v| v.as_bool() == Some(false))?;
        refuse("adapt_to_pi_aloha", &|v| v.as_bool() == Some(false))?;
        refuse("empty_cameras", &|v| v.as_u64() == Some(0))?;
        refuse("n_obs_steps", &|v| v.as_u64() == Some(1))?;
        let features = policy.get("input_features").and_then(Value::as_object).ok_or_else(|| Error::Config("config gives no input_features".into()))?;
        let mut camera_keys = Vec::new();
        let mut state_dim = None;
        for (k, f) in features {
            let shape = f.get("shape").and_then(Value::as_array).cloned().unwrap_or_default();
            match f.get("type").and_then(Value::as_str) {
                Some("VISUAL") => camera_keys.push(k.clone()),
                Some("STATE") => state_dim = shape.first().and_then(Value::as_u64).map(|x| x as usize),
                _ => {}
            }
        }
        let action_dim = policy
            .pointer("/output_features/action/shape/0")
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .ok_or_else(|| Error::Config("config gives no action shape".into()))?;
        let state_dim = state_dim.ok_or_else(|| Error::Config("config gives no state feature".into()))?;
        let resize = policy
            .get("resize_imgs_with_padding")
            .and_then(Value::as_array)
            .and_then(|a| Some((a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize)))
            // stored as (width, height)
            .map(|(w, h)| (h, w))
            .ok_or_else(|| Error::Config("config gives no resize_imgs_with_padding".into()))?;
        let text: TextConfig = serde_json::from_value(backbone.get("text_config").cloned().unwrap_or(Value::Null))
            .map_err(|e| Error::Config(format!("backbone text_config: {e}")))?;
        let vision: SiglipConfig = serde_json::from_value(backbone.get("vision_config").cloned().unwrap_or(Value::Null))
            .map_err(|e| Error::Config(format!("backbone vision_config: {e}")))?;
        let scale_factor = get_usize(backbone, "scale_factor")?;
        let num_vlm_layers = get_usize(policy, "num_vlm_layers")?;
        let num_expert_layers = policy.get("num_expert_layers").and_then(Value::as_i64).unwrap_or(-1);
        if num_expert_layers > 0 && num_expert_layers as usize != num_vlm_layers {
            return Err(Error::Config("an expert with fewer layers than the backbone is not implemented".into()));
        }
        let mult = get_f64(policy, "expert_width_multiplier")?;
        let expert_hidden = (text.hidden_size as f64 * mult) as u64;
        if resize != (vision.image_size, vision.image_size) {
            return Err(Error::Config(format!("frames resized to {resize:?} do not match the {} pixel vision input", vision.image_size)));
        }
        if vision.grid() % scale_factor != 0 {
            return Err(Error::Config("the patch grid does not divide by the pixel-shuffle factor".into()));
        }
        Ok(Self {
            camera_keys,
            state_dim,
            action_dim,
            chunk_size: get_usize(policy, "chunk_size")?,
            n_action_steps: get_usize(policy, "n_action_steps")?,
            max_state_dim: get_usize(policy, "max_state_dim")?,
            max_action_dim: get_usize(policy, "max_action_dim")?,
            resize,
            tokenizer_max_length: get_usize(policy, "tokenizer_max_length")?,
            num_steps: get_usize(policy, "num_steps")?,
            num_vlm_layers,
            self_attn_every_n_layers: get_usize(policy, "self_attn_every_n_layers")?.max(1),
            expert_hidden,
            expert_intermediate: expert_intermediate(expert_hidden),
            min_period: get_f64(policy, "min_period")?,
            max_period: get_f64(policy, "max_period")?,
            text,
            vision,
            scale_factor,
        })
    }

    /// Tokens per image after the pixel shuffle.
    #[must_use]
    pub fn image_tokens(&self) -> usize {
        self.vision.patches() / (self.scale_factor * self.scale_factor)
    }

    fn self_attn(&self, layer: usize) -> bool {
        layer % self.self_attn_every_n_layers == 0
    }

    /// Every weight the policy reads.
    #[must_use]
    pub fn weight_specs(&self, linear: WType, embed: WType) -> Vec<WeightSpec> {
        let f = WType::F32;
        let t = &self.text;
        let (d, hd) = (t.hidden_size, t.head_dim);
        let (q, kv) = (t.num_attention_heads * hd, t.num_key_value_heads * hd);
        let e = self.expert_hidden;
        let mut v = self.vision.weight_specs(&format!("{VLM}vision_model."), linear);
        let shuffled = self.vision.hidden_size * (self.scale_factor * self.scale_factor) as u64;
        v.push(WeightSpec::new(format!("{VLM}connector.modality_projection.proj.weight"), &[d, shuffled], linear));
        v.push(WeightSpec::new(format!("{VLM}text_model.embed_tokens.weight"), &[t.vocab_size, d], embed));
        let layer = |v: &mut Vec<WeightSpec>, p: String, width: u64, ff: u64, kv_in: u64| {
            v.push(WeightSpec::new(format!("{p}.input_layernorm.weight"), &[width], f));
            v.push(WeightSpec::new(format!("{p}.post_attention_layernorm.weight"), &[width], f));
            v.push(WeightSpec::new(format!("{p}.self_attn.q_proj.weight"), &[q, width], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.k_proj.weight"), &[kv, kv_in], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.v_proj.weight"), &[kv, kv_in], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.o_proj.weight"), &[width, q], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[ff, width], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[ff, width], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[width, ff], linear));
        };
        for i in 0..self.num_vlm_layers {
            layer(&mut v, format!("{VLM}text_model.layers.{i}"), d, t.intermediate_size, d);
            let kv_in = if self.self_attn(i) { e } else { kv };
            layer(&mut v, format!("{EXPERT}layers.{i}"), e, self.expert_intermediate, kv_in);
        }
        v.push(WeightSpec::new(format!("{EXPERT}norm.weight"), &[e], f));
        let ms = self.max_state_dim as u64;
        let ma = self.max_action_dim as u64;
        for (n, o, i) in [
            ("state_proj", d, ms),
            ("action_in_proj", e, ma),
            ("action_out_proj", ma, e),
            ("action_time_mlp_in", e, 2 * e),
            ("action_time_mlp_out", e, e),
        ] {
            v.push(WeightSpec::new(format!("{HEAD}{n}.weight"), &[o, i], f));
            v.push(WeightSpec::new(format!("{HEAD}{n}.bias"), &[o], f));
        }
        v
    }
}

/// Graph inputs of one chunk.
struct ChunkIo {
    patches: Vec<Tn>,
    tokens: Tn,
    state: Tn,
    prefix_pos: Tn,
    prefix_mask: Tn,
    noise: Tn,
    times: Tn,
    self_pos: Tn,
    cross_pos: Tn,
    suffix_mask: Tn,
    out: Tn,
}

/// A loaded SmolVLA policy.
pub struct SmolVla {
    cfg: SmolVlaConfig,
    norm: Normalization,
    tokenizer: Tokenizer,
    backend: Backend,
    weights: Weights,
    exact: bool,
}

impl std::fmt::Debug for SmolVla {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmolVla").field("device", &self.backend.name()).field("exact", &self.exact).finish_non_exhaustive()
    }
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    g.mul(h, w.get(name))
}

fn lin_b(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{name}.weight")), w.get(&format!("{name}.bias")), x)
}

impl SmolVla {
    /// Load a release: `policy` holds `config.json`, `model.safetensors`
    /// and the processor files; `backbone` the backbone's `config.json` and
    /// `tokenizer.json`.
    ///
    /// # Errors
    /// On a missing or malformed file, an unimplemented variant, or a
    /// backend failure.
    pub fn load(policy: &Path, backbone: &Path, opts: LoadOptions) -> Result<Self> {
        let pc = read_json(&policy.join("config.json"))?;
        let bc = read_json(&backbone.join("config.json"))?;
        let cfg = SmolVlaConfig::from_json(&pc, &bc)?;
        let norm = Normalization::from_release(policy, &pc)?;
        if cfg.state_dim > cfg.max_state_dim || cfg.action_dim > cfg.max_action_dim {
            return Err(Error::Config("state or action wider than the policy pads to".into()));
        }
        let tokenizer = Tokenizer::from_file(backbone.join("tokenizer.json")).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let (linear, embed) = match opts.precision {
            Precision::F32 => (WType::F32, WType::F32),
            Precision::Bf16 => (WType::Bf16, WType::F16),
            Precision::Q8_0 => (WType::Q8_0, WType::F16),
        };
        let files: Vec<PathBuf> = vec![policy.join("model.safetensors")];
        let st = SafeTensors::open(&files)?;
        let weights = Weights::load(&backend, &st, &cfg.weight_specs(linear, embed))?;
        Ok(Self { cfg, norm, tokenizer, backend, weights, exact })
    }

    /// Execute `n` actions of each chunk instead of the release's count.
    pub fn set_n_action_steps(&mut self, n: usize) {
        self.cfg.n_action_steps = n.clamp(1, self.cfg.chunk_size);
    }

    /// The release's settings.
    #[must_use]
    pub fn config(&self) -> &SmolVlaConfig {
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

    /// Instruction token ids: the instruction ends with a newline, special
    /// tokens as the tokenizer adds them, at most `tokenizer_max_length`.
    ///
    /// # Errors
    /// On a tokenizer failure or an empty instruction.
    pub fn tokens(&self, instruction: &str) -> Result<Vec<i32>> {
        let text = if instruction.ends_with('\n') { instruction.to_owned() } else { format!("{instruction}\n") };
        let enc = self.tokenizer.encode(text, true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids: Vec<i32> = enc.get_ids().iter().take(self.cfg.tokenizer_max_length).map(|&i| i as i32).collect();
        if ids.is_empty() {
            return Err(Error::Request("the instruction has no tokens".into()));
        }
        Ok(ids)
    }

    /// One frame as the vision tower reads it: resized with padding at the
    /// top and left, values mapped from `[0, 1]` to `[-1, 1]`.
    #[must_use]
    pub fn prepare_frame(&self, f: &Frame) -> Frame {
        let (h, w) = self.cfg.resize;
        let mut r = resize_with_pad(f, h, w, PadAt::Start, 0.0);
        for v in &mut r.data {
            *v = *v * 2.0 - 1.0;
        }
        r
    }

    fn pixel_shuffle(&self, g: &mut Graph, x: Tn) -> Tn {
        let s = self.cfg.scale_factor as i64;
        let d = self.cfg.vision.hidden_size as i64;
        let side = self.cfg.vision.grid() as i64;
        let x = g.reshape(x, &[d * s, side / s, side]);
        let x = g.permute(x, [0, 2, 1, 3]);
        let x = g.cont(x);
        let x = g.reshape(x, &[d * s * s, side / s, side / s]);
        let x = g.permute(x, [0, 2, 1, 3]);
        let x = g.cont(x);
        g.reshape(x, &[d * s * s, (side / s) * (side / s)])
    }

    #[allow(clippy::too_many_lines)]
    fn build(&self, g: &mut Graph, cameras: usize, n_tok: usize) -> ChunkIo {
        let c = &self.cfg;
        let t = &c.text;
        let w = &self.weights;
        let (d, hd) = (t.hidden_size as i64, t.head_dim as i64);
        let (nh, nkv) = (t.num_attention_heads as i64, t.num_key_value_heads as i64);
        let eps = t.rms_norm_eps as f32;
        let e = c.expert_hidden as i64;
        let chunk = c.chunk_size as i64;
        let n_pre = (cameras * c.image_tokens() + n_tok + 1) as i64;
        let scale = 1.0 / (hd as f32).sqrt();
        let exact = self.exact;
        let attend = |g: &mut Graph, q: Tn, k: Tn, v: Tn, mask: Option<Tn>| super::attend(g, q, k, v, mask, scale, exact);
        let mlp = |g: &mut Graph, p: &str, x: Tn| -> Tn {
            let h = rms(g, w, &format!("{p}.post_attention_layernorm.weight"), x, eps);
            let gate = g.linear(w.get(&format!("{p}.mlp.gate_proj.weight")), h);
            let up = g.linear(w.get(&format!("{p}.mlp.up_proj.weight")), h);
            let m = g.swiglu_split(gate, up);
            let m = g.linear(w.get(&format!("{p}.mlp.down_proj.weight")), m);
            g.add(x, m)
        };

        // Prefix embeddings.
        let sqrt_d = (d as f32).sqrt();
        let mut patches = Vec::with_capacity(cameras);
        let mut prefix: Option<Tn> = None;
        for _ in 0..cameras {
            let p = g.input(sys::GGML_TYPE_F32, &[c.vision.patch_len() as i64, c.vision.patches() as i64]);
            patches.push(p);
            let x = siglip::build(g, &c.vision, w, &format!("{VLM}vision_model."), p, exact);
            let x = self.pixel_shuffle(g, x);
            let x = g.linear(w.get(&format!("{VLM}connector.modality_projection.proj.weight")), x);
            let x = g.scale_bias(x, sqrt_d, 0.0);
            prefix = Some(match prefix {
                None => x,
                Some(p) => g.concat(p, x, 1),
            });
        }
        let tokens = g.input(sys::GGML_TYPE_I32, &[n_tok as i64]);
        let lang = g.get_rows(w.get(&format!("{VLM}text_model.embed_tokens.weight")), tokens);
        let lang = g.scale_bias(lang, sqrt_d, 0.0);
        let state = g.input(sys::GGML_TYPE_F32, &[c.max_state_dim as i64, 1]);
        let state_emb = lin_b(g, w, &format!("{HEAD}state_proj"), state);
        let x = prefix.expect("at least one camera");
        let x = g.concat(x, lang, 1);
        let mut x = g.concat(x, state_emb, 1);

        // Prefix pass: fill the per-layer cache.
        let prefix_pos = g.input(sys::GGML_TYPE_I32, &[n_pre]);
        let prefix_mask = g.input(sys::GGML_TYPE_F16, &[n_pre, n_pre]);
        let mut cache: Vec<(Tn, Tn)> = Vec::with_capacity(c.num_vlm_layers);
        for i in 0..c.num_vlm_layers {
            let p = format!("{VLM}text_model.layers.{i}");
            let h = rms(g, w, &format!("{p}.input_layernorm.weight"), x, eps);
            let q = g.linear(w.get(&format!("{p}.self_attn.q_proj.weight")), h);
            let k = g.linear(w.get(&format!("{p}.self_attn.k_proj.weight")), h);
            let v = g.linear(w.get(&format!("{p}.self_attn.v_proj.weight")), h);
            let q = g.reshape(q, &[hd, nh, n_pre]);
            let k = g.reshape(k, &[hd, nkv, n_pre]);
            let v = g.reshape(v, &[hd, nkv, n_pre]);
            let q = g.rope(q, prefix_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
            let k = g.rope(k, prefix_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
            cache.push((k, v));
            if i + 1 == c.num_vlm_layers {
                // The last layer's output is never read.
                break;
            }
            let o = attend(g, q, k, v, Some(prefix_mask));
            let o = g.reshape(o, &[hd * nh, n_pre]);
            let o = g.linear(w.get(&format!("{p}.self_attn.o_proj.weight")), o);
            let r = g.add(x, o);
            x = mlp(g, &p, r);
        }

        // Denoising steps.
        let noise = g.input(sys::GGML_TYPE_F32, &[c.max_action_dim as i64, chunk]);
        let times = g.input(sys::GGML_TYPE_F32, &[e, c.num_steps as i64]);
        let self_pos = g.input(sys::GGML_TYPE_I32, &[chunk]);
        let cross_pos = g.input(sys::GGML_TYPE_I32, &[chunk]);
        let suffix_mask = g.input(sys::GGML_TYPE_F16, &[n_pre + chunk, chunk]);
        let (_, dt) = euler_times(c.num_steps);
        let mut xt = noise;
        for step in 0..c.num_steps {
            let a = lin_b(g, w, &format!("{HEAD}action_in_proj"), xt);
            let te = g.view_cols(times, step as i64, 1);
            let te = g.repeat_to(te, [e, chunk, 1, 1]);
            let at = g.concat(a, te, 0);
            let at = lin_b(g, w, &format!("{HEAD}action_time_mlp_in"), at);
            let at = g.silu(at);
            let mut h = lin_b(g, w, &format!("{HEAD}action_time_mlp_out"), at);
            for (i, &(pk, pv)) in cache.iter().enumerate() {
                let p = format!("{EXPERT}layers.{i}");
                let hn = rms(g, w, &format!("{p}.input_layernorm.weight"), h, eps);
                let q = g.linear(w.get(&format!("{p}.self_attn.q_proj.weight")), hn);
                let q = g.reshape(q, &[hd, nh, chunk]);
                let o = if c.self_attn(i) {
                    let q = g.rope(q, self_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
                    let k = g.linear(w.get(&format!("{p}.self_attn.k_proj.weight")), hn);
                    let v = g.linear(w.get(&format!("{p}.self_attn.v_proj.weight")), hn);
                    let k = g.reshape(k, &[hd, nkv, chunk]);
                    let v = g.reshape(v, &[hd, nkv, chunk]);
                    let k = g.rope(k, self_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
                    let k = g.concat(pk, k, 2);
                    let v = g.concat(pv, v, 2);
                    attend(g, q, k, v, Some(suffix_mask))
                } else {
                    let q = g.rope(q, cross_pos, hd as i32, ROPE_NEOX, ROPE_BASE);
                    let kin = g.reshape(pk, &[hd * nkv, n_pre]);
                    let vin = g.reshape(pv, &[hd * nkv, n_pre]);
                    let k = g.linear(w.get(&format!("{p}.self_attn.k_proj.weight")), kin);
                    let v = g.linear(w.get(&format!("{p}.self_attn.v_proj.weight")), vin);
                    let k = g.reshape(k, &[hd, nkv, n_pre]);
                    let v = g.reshape(v, &[hd, nkv, n_pre]);
                    attend(g, q, k, v, None)
                };
                let o = g.reshape(o, &[hd * nh, chunk]);
                let o = g.linear(w.get(&format!("{p}.self_attn.o_proj.weight")), o);
                let r = g.add(h, o);
                h = mlp(g, &p, r);
            }
            let h = rms(g, w, &format!("{EXPERT}norm.weight"), h, eps);
            let v = lin_b(g, w, &format!("{HEAD}action_out_proj"), h);
            let v = g.scale_bias(v, dt, 0.0);
            xt = g.add(xt, v);
        }
        ChunkIo { patches, tokens, state, prefix_pos, prefix_mask, noise, times, self_pos, cross_pos, suffix_mask, out: xt }
    }

    /// One chunk `[chunk_size][max_action_dim]` in the policy's normalised
    /// units, from frames already prepared ([`SmolVla::prepare_frame`]),
    /// instruction token ids and the normalised, padded state.
    ///
    /// # Errors
    /// On a malformed input or a backend failure.
    pub fn sample(&self, frames: &[Frame], tokens: &[i32], state: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if frames.is_empty() || frames.len() != c.camera_keys.len() {
            return Err(Error::Request(format!("the policy reads {} cameras", c.camera_keys.len())));
        }
        if state.len() != c.max_state_dim || noise.len() != c.chunk_size * c.max_action_dim {
            return Err(Error::Request("state or noise of the wrong length".into()));
        }
        let n_img = frames.len() * c.image_tokens();
        let n_pre = n_img + tokens.len() + 1;
        let chunk = c.chunk_size;
        let mut g = Graph::new(&self.backend)?;
        let io = self.build(&mut g, frames.len(), tokens.len());
        g.finish(&[io.out])?;
        for (p, f) in io.patches.iter().zip(frames) {
            g.set_f32(*p, &c.vision.patchify(f));
        }
        g.set_i32(io.tokens, tokens);
        g.set_f32(io.state, state);
        let pos: Vec<i32> = (0..n_pre as i32).collect();
        g.set_i32(io.prefix_pos, &pos);
        // Images and instruction attend to each other; the state token
        // attends to everything, and nothing else attends to it.
        let mut m = vec![0f32; n_pre * n_pre];
        for q in 0..n_pre - 1 {
            m[q * n_pre + n_pre - 1] = f32::NEG_INFINITY;
        }
        g.set_f16(io.prefix_mask, &m);
        g.set_f32(io.noise, noise);
        let (ts, _) = euler_times(c.num_steps);
        let times: Vec<f32> = ts.iter().flat_map(|&t| sinusoidal_time(t, c.expert_hidden as usize, c.min_period, c.max_period)).collect();
        g.set_f32(io.times, &times);
        let sp: Vec<i32> = (0..chunk as i32).map(|j| n_pre as i32 + j).collect();
        g.set_i32(io.self_pos, &sp);
        let cp: Vec<i32> = (0..chunk as i32).collect();
        g.set_i32(io.cross_pos, &cp);
        let kv = n_pre + chunk;
        let mut sm = vec![0f32; kv * chunk];
        for q in 0..chunk {
            for k in n_pre + q + 1..kv {
                sm[q * kv + k] = f32::NEG_INFINITY;
            }
        }
        g.set_f16(io.suffix_mask, &sm);
        g.compute()?;
        Ok(g.read_f32(io.out))
    }
}

impl ChunkPolicy for SmolVla {
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
        let tokens = self.tokens(obs.instruction)?;
        let state = pad_to(&self.norm.state.normalize(obs.state)?, c.max_state_dim)?;
        let raw = self.sample(&frames, &tokens, &state, noise)?;
        let cut: Vec<f32> = raw.chunks(c.max_action_dim).flat_map(|r| r[..c.action_dim].to_vec()).collect();
        self.norm.action.unnormalize(&cut)
    }
}
