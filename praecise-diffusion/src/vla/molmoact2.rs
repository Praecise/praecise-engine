//! The MolmoAct2 vision backbone: a ViT over image crops (patches flattened
//! and embedded by one linear map, a learned position per patch, pre-norm
//! blocks of LayerNorm, biased multi-head attention and a tanh-GELU MLP),
//! whose hidden states after two chosen blocks are concatenated per patch.
//! Each image token pools a few of those patch features: their masked mean
//! queries an attention over them, and a SwiGLU projector maps the result
//! to the language model's width.
//!
//! The language model reads the prompt with those features added to the
//! embeddings of its image-patch tokens: pre-norm decoder blocks of RMS
//! norm, one fused query/key/value projection, per-head query and key RMS
//! norms, split-half rotary positions, grouped-query attention and a gated
//! SiLU MLP. Attention is causal except that image tokens see each other
//! both ways. The keys and values of every block are kept for the action
//! expert.
//!
//! The action expert turns noise into an action chunk by flow matching:
//! forward-Euler steps from `t = 0` (noise) to `t = 1` (actions), each a
//! velocity from a stack of blocks, one per language-model block. A block
//! modulates (shift, scale, gate from the time embedding) a self-attention
//! over the chunk with split-half rotary positions, a cross-attention to
//! that language-model block's keys and values (projected to the expert's
//! width), and a gated SiLU MLP.

use std::path::Path;

use serde_json::Value;

use super::{attend, read_json};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const P: &str = "model.vision_backbone.";
const T: &str = "model.transformer.";
const E: &str = "model.action_expert.";

/// The backbone's settings, from the checkpoint's `vit_config` and
/// `adapter_config`.
#[derive(Debug, Clone)]
pub struct MolmoAct2VisionConfig {
    /// ViT width.
    pub hidden: u64,
    /// ViT MLP width.
    pub intermediate: u64,
    /// ViT heads.
    pub heads: u64,
    /// ViT head width.
    pub head_dim: u64,
    /// LayerNorm epsilon.
    pub eps: f64,
    /// Patch side in pixels.
    pub patch_size: usize,
    /// Crop height and width in pixels.
    pub crop: (usize, usize),
    /// Blocks whose outputs are concatenated, in order (0-based).
    pub layers: Vec<usize>,
    /// Pooling-attention width.
    pub pool_hidden: u64,
    /// Pooling-attention heads.
    pub pool_heads: u64,
    /// Pooling-attention head width.
    pub pool_head_dim: u64,
    /// Whether pooling ignores the padding slots of a token.
    pub pool_mask: bool,
    /// Projector hidden width.
    pub projector: u64,
    /// Output (language model) width.
    pub out: u64,
}

fn u(v: &Value, k: &str) -> Result<u64> {
    v.get(k).and_then(Value::as_u64).ok_or_else(|| Error::Config(format!("config gives no {k}")))
}

impl MolmoAct2VisionConfig {
    /// Read the backbone settings from a checkpoint config.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field or a variant this engine does
    /// not implement.
    pub fn from_json(c: &Value) -> Result<Self> {
        let v = c.get("vit_config").ok_or_else(|| Error::Config("config gives no vit_config".into()))?;
        let a = c.get("adapter_config").ok_or_else(|| Error::Config("config gives no adapter_config".into()))?;
        for (part, act) in [(v, "gelu_pytorch_tanh"), (a, "silu")] {
            if part.get("hidden_act").and_then(Value::as_str).unwrap_or(act) != act {
                return Err(Error::Config(format!("MolmoAct2 activations other than {act} are not implemented")));
            }
            if part.get("num_key_value_heads").and_then(Value::as_u64) != part.get("num_attention_heads").and_then(Value::as_u64) {
                return Err(Error::Config("grouped-query attention in the MolmoAct2 vision backbone is not implemented".into()));
            }
        }
        let total = u(v, "num_hidden_layers")? as i64;
        let layers = a
            .get("vit_layers")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config("config gives no vit_layers".into()))?
            .iter()
            .map(|l| {
                let l = l.as_i64().ok_or_else(|| Error::Config("vit_layers holds a non-integer".into()))?;
                let l = if l < 0 { l + total } else { l };
                if (0..total).contains(&l) { Ok(l as usize) } else { Err(Error::Config(format!("vit layer {l} outside the ViT"))) }
            })
            .collect::<Result<Vec<_>>>()?;
        let size = v.get("image_default_input_size").and_then(Value::as_array).ok_or_else(|| Error::Config("config gives no image_default_input_size".into()))?;
        let side = |i: usize| size.get(i).and_then(Value::as_u64).map(|s| s as usize).ok_or_else(|| Error::Config("malformed image_default_input_size".into()));
        let cfg = Self {
            hidden: u(v, "hidden_size")?,
            intermediate: u(v, "intermediate_size")?,
            heads: u(v, "num_attention_heads")?,
            head_dim: u(v, "head_dim")?,
            eps: v.get("layer_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            patch_size: u(v, "image_patch_size")? as usize,
            crop: (side(0)?, side(1)?),
            layers,
            pool_hidden: u(a, "hidden_size")?,
            pool_heads: u(a, "num_attention_heads")?,
            pool_head_dim: u(a, "head_dim")?,
            pool_mask: a.get("pooling_attention_mask").and_then(Value::as_bool).unwrap_or(false),
            projector: u(a, "intermediate_size")?,
            out: u(a, "text_hidden_size")?,
        };
        let (gh, gw) = cfg.grid();
        if gh * cfg.patch_size != cfg.crop.0 || gw * cfg.patch_size != cfg.crop.1 {
            return Err(Error::Config("crop size is not a whole number of patches".into()));
        }
        // The learned positions form a square grid; the reference resizes
        // them only when the crop grid differs, which no release does.
        if u(v, "image_num_pos")? as usize != gh * gw || gh != gw {
            return Err(Error::Config("MolmoAct2 position grids other than the crop grid are not implemented".into()));
        }
        Ok(cfg)
    }

    /// Patches per crop, down and across.
    #[must_use]
    pub fn grid(&self) -> (usize, usize) {
        (self.crop.0 / self.patch_size, self.crop.1 / self.patch_size)
    }

    /// Patches per crop.
    #[must_use]
    pub fn patches(&self) -> usize {
        let (h, w) = self.grid();
        h * w
    }

    /// Values per patch.
    #[must_use]
    pub fn patch_len(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// ViT blocks run: up to the last one read.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.layers.iter().max().map_or(0, |l| l + 1)
    }

    fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let f = WType::F32;
        let (d, ff) = (self.hidden, self.intermediate);
        let vd = self.heads * self.head_dim;
        let pd = self.pool_heads * self.pool_head_dim;
        let cat = d * self.layers.len() as u64;
        let pl = self.patch_len() as u64;
        // Patch rows are 3 * p * p values, not always a multiple of the 8-bit block.
        let kernel = if linear == WType::Q8_0 && pl % 32 != 0 { WType::Bf16 } else { linear };
        let mut s = vec![
            WeightSpec::new(format!("{P}image_vit.patch_embedding.weight"), &[d, pl], kernel),
            WeightSpec::new(format!("{P}image_vit.patch_embedding.bias"), &[d], f),
            WeightSpec::new(format!("{P}image_vit.positional_embedding"), &[self.patches() as u64, d], f),
        ];
        for i in 0..self.blocks() {
            let l = format!("{P}image_vit.transformer.resblocks.{i}");
            for n in ["attention_norm", "ffn_norm"] {
                s.push(WeightSpec::new(format!("{l}.{n}.weight"), &[d], f));
                s.push(WeightSpec::new(format!("{l}.{n}.bias"), &[d], f));
            }
            for n in ["wq", "wk", "wv"] {
                s.push(WeightSpec::new(format!("{l}.attention.{n}.weight"), &[vd, d], linear));
                s.push(WeightSpec::new(format!("{l}.attention.{n}.bias"), &[vd], f));
            }
            s.push(WeightSpec::new(format!("{l}.attention.wo.weight"), &[d, vd], linear));
            s.push(WeightSpec::new(format!("{l}.attention.wo.bias"), &[d], f));
            s.push(WeightSpec::new(format!("{l}.feed_forward.w1.weight"), &[ff, d], linear));
            s.push(WeightSpec::new(format!("{l}.feed_forward.w1.bias"), &[ff], f));
            s.push(WeightSpec::new(format!("{l}.feed_forward.w2.weight"), &[d, ff], linear));
            s.push(WeightSpec::new(format!("{l}.feed_forward.w2.bias"), &[d], f));
        }
        let pp = format!("{P}image_pooling_2d");
        for n in ["wq", "wk", "wv"] {
            s.push(WeightSpec::new(format!("{pp}.{n}.weight"), &[pd, cat], linear));
            s.push(WeightSpec::new(format!("{pp}.{n}.bias"), &[pd], f));
        }
        s.push(WeightSpec::new(format!("{pp}.wo.weight"), &[self.pool_hidden, pd], linear));
        s.push(WeightSpec::new(format!("{pp}.wo.bias"), &[self.pool_hidden], f));
        let pr = format!("{P}image_projector");
        s.push(WeightSpec::new(format!("{pr}.w1.weight"), &[self.projector, self.pool_hidden], linear));
        s.push(WeightSpec::new(format!("{pr}.w3.weight"), &[self.projector, self.pool_hidden], linear));
        s.push(WeightSpec::new(format!("{pr}.w2.weight"), &[self.out, self.projector], linear));
        s
    }
}

/// The safetensors files of a checkpoint directory: those named by
/// `model.safetensors.index.json`, or one `model.safetensors`.
pub(crate) fn checkpoint_files(dir: &Path) -> Vec<std::path::PathBuf> {
    match read_json(&dir.join("model.safetensors.index.json")) {
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
    }
}

/// Snap processor pixels in `[-1, 1]` to the 8-bit grid the reference
/// normalises on the device (`round` ties to even, as torch's).
#[must_use]
pub fn snap_pixels(px: &[f32]) -> Vec<f32> {
    px.iter()
        .map(|&x| {
            let v = (((x + 1.0) * 0.5) * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0;
            v * 2.0 - 1.0
        })
        .collect()
}

/// The MolmoAct2 vision backbone on one device.
pub struct MolmoAct2Vision {
    cfg: MolmoAct2VisionConfig,
    backend: Backend,
    weights: Weights,
    exact: bool,
}

impl std::fmt::Debug for MolmoAct2Vision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MolmoAct2Vision").field("device", &self.backend.name()).finish_non_exhaustive()
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

impl MolmoAct2Vision {
    /// Load the backbone from a checkpoint directory (`config.json` and its
    /// safetensors files); ViT blocks after the last one read are skipped.
    ///
    /// # Errors
    /// On a missing or malformed file, an unsupported variant, or a backend
    /// failure.
    pub fn load(dir: &Path, opts: LoadOptions) -> Result<Self> {
        let cfg = MolmoAct2VisionConfig::from_json(&read_json(&dir.join("config.json"))?)?;
        let st = SafeTensors::open(&checkpoint_files(dir))?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let linear = match opts.precision {
            Precision::F32 => WType::F32,
            Precision::Bf16 => WType::Bf16,
            Precision::Q8_0 => WType::Q8_0,
        };
        let weights = Weights::load(&backend, &st, &cfg.weight_specs(linear))?;
        Ok(Self { cfg, backend, weights, exact })
    }

    /// The backbone's settings.
    #[must_use]
    pub fn config(&self) -> &MolmoAct2VisionConfig {
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
        self.weights.bytes()
    }

    /// Image-token features `[tokens][out]` of one observation. `pixels`
    /// are the processor's crops `[crops][patches][patch_len]` in `[-1, 1]`;
    /// `pooling` gives, for each token, `pool` patch indices into the crops'
    /// patches in order (`-1` pads). Tokens whose indices are all padding
    /// are dropped, as the reference does.
    ///
    /// # Errors
    /// On inputs of the wrong size, an index outside the crops, or a backend
    /// failure.
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self, pixels: &[f32], crops: usize, pooling: &[i32], pool: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (n, pl) = (c.patches(), c.patch_len());
        if crops == 0 || pool == 0 || pixels.len() != crops * n * pl || pooling.len() % pool != 0 {
            return Err(Error::Request("crops or pooling indices of the wrong size".into()));
        }
        let total = (crops * n) as i32;
        if pooling.iter().any(|&i| i < -1 || i >= total) {
            return Err(Error::Request("a pooling index lies outside the crops".into()));
        }
        let tokens = pooling.len() / pool;
        let valid: Vec<bool> = pooling.iter().map(|&i| i >= 0).collect();
        let keep: Vec<bool> = valid.chunks(pool).map(|t| t.iter().any(|&v| v)).collect();

        let w = &self.weights;
        let exact = self.exact;
        let eps = c.eps as f32;
        let d = c.hidden as i64;
        let (nh, hd) = (c.heads as i64, c.head_dim as i64);
        let (n, crops_i, pool_i, tok_i) = (n as i64, crops as i64, pool as i64, tokens as i64);
        let mut g = Graph::new(&self.backend)?;
        let px = g.input(sys::GGML_TYPE_F32, &[pl as i64, n, crops_i]);
        let ids = g.input(sys::GGML_TYPE_I32, &[pool_i * tok_i]);
        let slot = g.input(sys::GGML_TYPE_F32, &[1, pool_i * tok_i]);
        let weigh = g.input(sys::GGML_TYPE_F32, &[pool_i, 1, tok_i]);
        let mask = if c.pool_mask { Some(g.input(sys::GGML_TYPE_F32, &[pool_i, 1, 1, tok_i])) } else { None };

        // ViT over every crop, keeping the chosen blocks' outputs.
        let vit = format!("{P}image_vit");
        let x = lin_b(&mut g, w, &format!("{vit}.patch_embedding"), px);
        let mut x = g.add(x, w.get(&format!("{vit}.positional_embedding")));
        let mut picked: Vec<Option<Tn>> = vec![None; c.layers.len()];
        for i in 0..c.blocks() {
            let l = format!("{vit}.transformer.resblocks.{i}");
            let h = layer_norm(&mut g, w, &format!("{l}.attention_norm"), x, eps);
            let q = lin_b(&mut g, w, &format!("{l}.attention.wq"), h);
            let k = lin_b(&mut g, w, &format!("{l}.attention.wk"), h);
            let v = lin_b(&mut g, w, &format!("{l}.attention.wv"), h);
            let q = g.reshape(q, &[hd, nh, n, crops_i]);
            let k = g.reshape(k, &[hd, nh, n, crops_i]);
            let v = g.reshape(v, &[hd, nh, n, crops_i]);
            let o = attend(&mut g, q, k, v, None, 1.0 / (hd as f32).sqrt(), exact);
            let o = g.reshape(o, &[hd * nh, n, crops_i]);
            let o = lin_b(&mut g, w, &format!("{l}.attention.wo"), o);
            x = g.add(x, o);
            let h = layer_norm(&mut g, w, &format!("{l}.ffn_norm"), x, eps);
            let h = lin_b(&mut g, w, &format!("{l}.feed_forward.w1"), h);
            let h = if exact { g.gelu_tanh_exact(h) } else { g.gelu_tanh(h) };
            let h = lin_b(&mut g, w, &format!("{l}.feed_forward.w2"), h);
            x = g.add(x, h);
            for (j, &layer) in c.layers.iter().enumerate() {
                if layer == i {
                    picked[j] = Some(x);
                }
            }
        }
        let mut feats = picked[0].expect("every chosen block runs");
        for p in &picked[1..] {
            feats = g.concat(feats, p.expect("every chosen block runs"), 0);
        }
        let cat = d * c.layers.len() as i64;
        let feats = g.reshape(feats, &[cat, n * crops_i]);

        // Gather each token's patches (padding reads patch 0, zeroed).
        let to_pool = g.get_rows(feats, ids);
        let to_pool = g.mul(to_pool, slot);
        let to_pool = g.reshape(to_pool, &[cat, pool_i, tok_i]);
        // Masked mean (or plain mean) as the query.
        let tp = g.permute(to_pool, [1, 0, 2, 3]);
        let tp = g.cont(tp);
        let query = g.linear(tp, weigh);

        let (ph, phd) = (c.pool_heads as i64, c.pool_head_dim as i64);
        let pp = format!("{P}image_pooling_2d");
        let q = lin_b(&mut g, w, &format!("{pp}.wq"), query);
        let k = lin_b(&mut g, w, &format!("{pp}.wk"), to_pool);
        let v = lin_b(&mut g, w, &format!("{pp}.wv"), to_pool);
        let q = g.reshape(q, &[phd, ph, 1, tok_i]);
        let k = g.reshape(k, &[phd, ph, pool_i, tok_i]);
        let v = g.reshape(v, &[phd, ph, pool_i, tok_i]);
        let q = g.permute(q, [0, 2, 1, 3]);
        let k = g.permute(k, [0, 2, 1, 3]);
        let v = g.permute(v, [0, 2, 1, 3]);
        // Unfused float32 attention: each token's own key mask is added to
        // its scores before the softmax.
        let kq = g.linear(k, q);
        let kq = match mask {
            Some(m) => g.add(kq, m),
            None => kq,
        };
        let kq = g.soft_max(kq, 1.0 / (phd as f32).sqrt());
        let vt = g.permute(v, [1, 0, 2, 3]);
        let vt = g.cont(vt);
        let o = g.linear(vt, kq);
        let o = g.permute(o, [0, 2, 1, 3]);
        let o = g.cont(o);
        let o = g.reshape(o, &[phd * ph, tok_i]);
        let o = lin_b(&mut g, w, &format!("{pp}.wo"), o);

        let pr = format!("{P}image_projector");
        let a = g.linear(w.get(&format!("{pr}.w1.weight")), o);
        let b = g.linear(w.get(&format!("{pr}.w3.weight")), o);
        let h = g.swiglu_split(a, b);
        let out = g.linear(w.get(&format!("{pr}.w2.weight")), h);
        g.finish(&[out])?;

        g.set_f32(px, &snap_pixels(pixels));
        let rows: Vec<i32> = pooling.iter().map(|&i| i.max(0)).collect();
        g.set_i32(ids, &rows);
        let slots: Vec<f32> = valid.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        g.set_f32(slot, &slots);
        let mut shares = Vec::with_capacity(pooling.len());
        for t in valid.chunks(pool) {
            if c.pool_mask {
                let count = t.iter().filter(|&&v| v).count().max(1) as f32;
                shares.extend(t.iter().map(|&v| if v { 1.0 / count } else { 0.0 }));
            } else {
                shares.extend(std::iter::repeat(1.0 / pool as f32).take(pool));
            }
        }
        g.set_f32(weigh, &shares);
        if let Some(m) = mask {
            let bias: Vec<f32> = valid.iter().map(|&v| if v { 0.0 } else { f32::NEG_INFINITY }).collect();
            g.set_f32(m, &bias);
        }
        g.compute()?;
        let got = g.read_f32(out);
        let o = c.out as usize;
        Ok(got.chunks(o).zip(&keep).filter(|&(_, &k)| k).flat_map(|(r, _)| r.iter().copied()).collect())
    }
}

/// The language model's settings, from the checkpoint's `text_config`.
#[derive(Debug, Clone)]
pub struct MolmoAct2TextConfig {
    /// Model width.
    pub hidden: u64,
    /// MLP width.
    pub intermediate: u64,
    /// Query heads.
    pub heads: u64,
    /// Key/value heads.
    pub kv_heads: u64,
    /// Head width.
    pub head_dim: u64,
    /// Decoder blocks.
    pub layers: usize,
    /// RMS norm epsilon.
    pub eps: f64,
    /// Rotary base.
    pub theta: f64,
    /// Tokens in the base embedding table.
    pub vocab: u64,
    /// Tokens in the added embedding table, numbered after the base ones.
    pub new_vocab: u64,
}

impl MolmoAct2TextConfig {
    /// Read the language-model settings from a checkpoint config.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field or a variant this engine does
    /// not implement.
    pub fn from_json(c: &Value) -> Result<Self> {
        let t = c.get("text_config").ok_or_else(|| Error::Config("config gives no text_config".into()))?;
        let unsupported = |what: &str| Err(Error::Config(format!("MolmoAct2 language models with {what} are not implemented")));
        if t.get("hidden_act").and_then(Value::as_str).unwrap_or("silu") != "silu" {
            return unsupported("activations other than silu");
        }
        if t.get("norm_after").and_then(Value::as_bool).unwrap_or(false) {
            return unsupported("post-norm blocks");
        }
        if t.get("qkv_bias").and_then(Value::as_bool).unwrap_or(false) {
            return unsupported("attention biases");
        }
        if !t.get("use_qk_norm").and_then(Value::as_bool).unwrap_or(false) || t.get("qk_norm_type").and_then(Value::as_str) != Some("qwen3") {
            return unsupported("query/key norms other than per-head ones");
        }
        if t.get("rope_scaling_layers").is_some_and(|v| !v.is_null()) {
            return unsupported("per-layer rotary scaling");
        }
        let rope = t.get("rope_parameters");
        if rope.and_then(|r| r.get("rope_type")).and_then(Value::as_str).is_some_and(|r| r != "default") {
            return unsupported("scaled rotary positions");
        }
        let theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| t.get("rope_theta"))
            .and_then(Value::as_f64)
            .ok_or_else(|| Error::Config("text_config gives no rotary base".into()))?;
        let cfg = Self {
            hidden: u(t, "hidden_size")?,
            intermediate: u(t, "intermediate_size")?,
            heads: u(t, "num_attention_heads")?,
            kv_heads: u(t, "num_key_value_heads")?,
            head_dim: u(t, "head_dim")?,
            layers: u(t, "num_hidden_layers")? as usize,
            eps: t.get("layer_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            theta,
            vocab: u(t, "vocab_size")?,
            new_vocab: t.get("additional_vocab_size").and_then(Value::as_u64).unwrap_or(0),
        };
        if cfg.kv_heads == 0 || cfg.heads % cfg.kv_heads != 0 || cfg.head_dim % 2 != 0 {
            return Err(Error::Config("malformed MolmoAct2 attention heads".into()));
        }
        Ok(cfg)
    }

    fn weight_specs(&self, linear: WType, table: WType) -> Vec<WeightSpec> {
        let f = WType::F32;
        let (d, ff, hd) = (self.hidden, self.intermediate, self.head_dim);
        let (q, kv) = (self.heads * hd, self.kv_heads * hd);
        let mut s = vec![
            WeightSpec::new(format!("{T}wte.embedding"), &[self.vocab, d], table),
            WeightSpec::new(format!("{T}ln_f.weight"), &[d], f),
        ];
        if self.new_vocab > 0 {
            s.push(WeightSpec::new(format!("{T}wte.new_embedding"), &[self.new_vocab, d], table));
        }
        // The MLP projection holds [x | gate]; the gate half goes first on
        // the device so one fused SwiGLU computes silu(gate) * x.
        let ff = ff as usize;
        let gate_first: Vec<usize> = (ff..2 * ff).chain(0..ff).collect();
        for i in 0..self.layers {
            let l = format!("{T}blocks.{i}");
            s.push(WeightSpec::new(format!("{l}.attn_norm.weight"), &[d], f));
            s.push(WeightSpec::new(format!("{l}.ff_norm.weight"), &[d], f));
            s.push(WeightSpec::new(format!("{l}.self_attn.att_proj.weight"), &[q + 2 * kv, d], linear));
            s.push(WeightSpec::new(format!("{l}.self_attn.q_norm.weight"), &[hd], f));
            s.push(WeightSpec::new(format!("{l}.self_attn.k_norm.weight"), &[hd], f));
            s.push(WeightSpec::new(format!("{l}.self_attn.attn_out.weight"), &[d, q], linear));
            s.push(WeightSpec::new(format!("{l}.mlp.ff_proj.weight"), &[2 * ff as u64, d], linear).with_rows(gate_first.clone()));
            s.push(WeightSpec::new(format!("{l}.mlp.ff_out.weight"), &[d, ff as u64], linear));
        }
        s
    }

    /// Rotary tables `[tokens][head_dim]` (cosines, sines) for positions
    /// `0..n`, in `float32` as the reference builds them: inverse
    /// frequencies `1 / theta^(2i / head_dim)`, angles `position * inverse
    /// frequency`, each half of a head sharing the same angles.
    #[must_use]
    pub fn rotary_tables(&self, n: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim as usize;
        let theta = self.theta as f32;
        let inv: Vec<f32> = (0..hd / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / hd as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(n * hd), Vec::with_capacity(n * hd));
        for p in 0..n {
            for _ in 0..2 {
                for &f in &inv {
                    let a = f64::from(p as f32 * f);
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        (cos, sin)
    }
}

/// The token ids the language model treats specially, from the checkpoint
/// config.
#[derive(Debug, Clone)]
pub struct MolmoAct2ImageTokens {
    /// The token whose embedding receives an image feature.
    pub patch: i32,
    /// Every image-layout token (patches, row ends, image and frame
    /// starts and ends): these attend to each other both ways.
    pub all: Vec<i32>,
}

impl MolmoAct2ImageTokens {
    /// Read the image token ids from a checkpoint config.
    ///
    /// # Errors
    /// [`Error::Config`] when the config gives no image-patch token.
    pub fn from_json(c: &Value) -> Result<Self> {
        let id = |k: &str| c.get(k).and_then(Value::as_i64).map(|v| v as i32);
        let patch = id("image_patch_id").ok_or_else(|| Error::Config("config gives no image_patch_id".into()))?;
        let mut all: Vec<i32> = [
            "image_patch_id",
            "image_col_id",
            "image_start_token_id",
            "low_res_image_start_token_id",
            "frame_start_token_id",
            "image_end_token_id",
            "frame_end_token_id",
            "image_low_res_id",
        ]
        .iter()
        .filter_map(|k| id(k))
        .collect();
        all.sort_unstable();
        all.dedup();
        Ok(Self { patch, all })
    }
}

/// The language model's pass over a prompt.
#[derive(Debug, Clone)]
pub struct MolmoAct2Prefix {
    /// Prompt tokens.
    pub tokens: usize,
    /// Final normalised hidden states `[tokens][hidden]`.
    pub hidden: Vec<f32>,
    /// Per block, the keys after the rotary embedding `[tokens][kv_heads *
    /// head_dim]`.
    pub keys: Vec<Vec<f32>>,
    /// Per block, the values `[tokens][kv_heads * head_dim]`.
    pub values: Vec<Vec<f32>>,
}

/// The MolmoAct2 language model on one device.
pub struct MolmoAct2Lm {
    cfg: MolmoAct2TextConfig,
    image: MolmoAct2ImageTokens,
    backend: Backend,
    weights: Weights,
    exact: bool,
}

impl std::fmt::Debug for MolmoAct2Lm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MolmoAct2Lm").field("device", &self.backend.name()).finish_non_exhaustive()
    }
}

impl MolmoAct2Lm {
    /// Load the language model from a checkpoint directory (`config.json`
    /// and its safetensors files).
    ///
    /// # Errors
    /// On a missing or malformed file, an unsupported variant, or a backend
    /// failure.
    pub fn load(dir: &Path, opts: LoadOptions) -> Result<Self> {
        let c = read_json(&dir.join("config.json"))?;
        let cfg = MolmoAct2TextConfig::from_json(&c)?;
        let image = MolmoAct2ImageTokens::from_json(&c)?;
        let st = SafeTensors::open(&checkpoint_files(dir))?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let (linear, table) = match opts.precision {
            Precision::F32 => (WType::F32, WType::F32),
            Precision::Bf16 => (WType::Bf16, WType::F16),
            Precision::Q8_0 => (WType::Q8_0, WType::F16),
        };
        let weights = Weights::load(&backend, &st, &cfg.weight_specs(linear, table))?;
        Ok(Self { cfg, image, backend, weights, exact })
    }

    /// The language model's settings.
    #[must_use]
    pub fn config(&self) -> &MolmoAct2TextConfig {
        &self.cfg
    }

    /// The image token ids.
    #[must_use]
    pub fn image_tokens(&self) -> &MolmoAct2ImageTokens {
        &self.image
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
        self.weights.bytes()
    }

    /// The attention mask over `ids`: causal, and image tokens attend to
    /// each other both ways.
    #[must_use]
    pub fn mask(&self, ids: &[i32]) -> Vec<f32> {
        let n = ids.len();
        let img: Vec<bool> = ids.iter().map(|i| self.image.all.binary_search(i).is_ok()).collect();
        let mut m = vec![0f32; n * n];
        for q in 0..n {
            for k in 0..n {
                if k > q && !(img[q] && img[k]) {
                    m[q * n + k] = f32::NEG_INFINITY;
                }
            }
        }
        m
    }

    /// Run the prompt `ids` (no padding; positions `0..n`) with `features`
    /// `[image tokens][hidden]` (as [`MolmoAct2Vision::encode`] gives them)
    /// added to the embeddings of the image-patch tokens, in order.
    ///
    /// # Errors
    /// On a token outside the vocabulary, a feature count that does not
    /// match the image-patch tokens, or a backend failure.
    #[allow(clippy::too_many_lines)]
    pub fn prefill(&self, ids: &[i32], features: &[f32]) -> Result<MolmoAct2Prefix> {
        let c = &self.cfg;
        let d = c.hidden as usize;
        let n = ids.len();
        let total = (c.vocab + c.new_vocab) as i32;
        if n == 0 || ids.iter().any(|&i| i < 0 || i >= total) {
            return Err(Error::Request("an empty prompt or a token outside the vocabulary".into()));
        }
        let patches = ids.iter().filter(|&&i| i == self.image.patch).count();
        if features.len() != patches * d {
            return Err(Error::Request(format!("{patches} image-patch tokens but {} features", features.len() / d.max(1))));
        }

        let w = &self.weights;
        let exact = self.exact;
        let eps = c.eps as f32;
        let (hd, nh, nkv) = (c.head_dim as i64, c.heads as i64, c.kv_heads as i64);
        let (n_i, d_i) = (n as i64, d as i64);
        let mut g = Graph::new(&self.backend)?;
        let base = g.input(sys::GGML_TYPE_I32, &[n_i]);
        let pick_base = g.input(sys::GGML_TYPE_F32, &[1, n_i]);
        let added = if c.new_vocab > 0 { Some((g.input(sys::GGML_TYPE_I32, &[n_i]), g.input(sys::GGML_TYPE_F32, &[1, n_i]))) } else { None };
        let img = g.input(sys::GGML_TYPE_F32, &[d_i, n_i]);
        let mask = g.input(sys::GGML_TYPE_F16, &[n_i, n_i]);
        let cos = g.input(sys::GGML_TYPE_F32, &[hd, 1, n_i]);
        let sin = g.input(sys::GGML_TYPE_F32, &[hd, 1, n_i]);

        // Each token reads one of the two tables (the other is weighted 0).
        let x = g.get_rows(w.get(&format!("{T}wte.embedding")), base);
        let mut x = g.mul(x, pick_base);
        if let Some((ids_new, pick_new)) = added {
            let y = g.get_rows(w.get(&format!("{T}wte.new_embedding")), ids_new);
            let y = g.mul(y, pick_new);
            x = g.add(x, y);
        }
        let mut x = g.add(x, img);
        let (qw, kvw) = (nh * hd, nkv * hd);
        let mut kv_out = Vec::with_capacity(2 * c.layers);
        for i in 0..c.layers {
            let l = format!("{T}blocks.{i}");
            let wn = |s: &str| w.get(&format!("{l}.{s}"));
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("attn_norm.weight"));
            let qkv = g.linear(wn("self_attn.att_proj.weight"), h);
            let q = g.view_heads(qkv, 0, hd, nh);
            let q = g.cont(q);
            let k = g.view_heads(qkv, qw, hd, nkv);
            let k = g.cont(k);
            let v = g.view_heads(qkv, qw + kvw, hd, nkv);
            let v = g.cont(v);
            let q = g.rms_norm(q, eps);
            let q = g.mul(q, wn("self_attn.q_norm.weight"));
            let k = g.rms_norm(k, eps);
            let k = g.mul(k, wn("self_attn.k_norm.weight"));
            let q = g.rotate_half_rope(q, cos, sin);
            let k = g.rotate_half_rope(k, cos, sin);
            kv_out.push(k);
            kv_out.push(v);
            let o = attend(&mut g, q, k, v, Some(mask), 1.0 / (hd as f32).sqrt(), exact);
            let o = g.reshape(o, &[qw, n_i]);
            let o = g.linear(wn("self_attn.attn_out.weight"), o);
            x = g.add(x, o);
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("ff_norm.weight"));
            let h = g.linear(wn("mlp.ff_proj.weight"), h);
            let h = g.swiglu(h);
            let h = g.linear(wn("mlp.ff_out.weight"), h);
            x = g.add(x, h);
        }
        let h = g.rms_norm(x, eps);
        let out = g.mul(h, w.get(&format!("{T}ln_f.weight")));
        let mut outputs = vec![out];
        outputs.extend_from_slice(&kv_out);
        g.finish(&outputs)?;

        let vocab = c.vocab as i32;
        g.set_i32(base, &ids.iter().map(|&i| i.min(vocab - 1)).collect::<Vec<_>>());
        g.set_f32(pick_base, &ids.iter().map(|&i| if i < vocab { 1.0 } else { 0.0 }).collect::<Vec<_>>());
        if let Some((ids_new, pick_new)) = added {
            g.set_i32(ids_new, &ids.iter().map(|&i| (i - vocab).max(0)).collect::<Vec<_>>());
            g.set_f32(pick_new, &ids.iter().map(|&i| if i >= vocab { 1.0 } else { 0.0 }).collect::<Vec<_>>());
        }
        let mut add = vec![0f32; n * d];
        for (row, feat) in ids.iter().enumerate().filter(|&(_, &i)| i == self.image.patch).map(|(r, _)| r).zip(features.chunks(d)) {
            add[row * d..(row + 1) * d].copy_from_slice(feat);
        }
        g.set_f32(img, &add);
        g.set_f16(mask, &self.mask(ids));
        let (tc, ts) = c.rotary_tables(n);
        g.set_f32(cos, &tc);
        g.set_f32(sin, &ts);
        g.compute()?;
        let hidden = g.read_f32(out);
        let (mut keys, mut values) = (Vec::with_capacity(c.layers), Vec::with_capacity(c.layers));
        for pair in kv_out.chunks(2) {
            keys.push(g.read_f32(pair[0]));
            values.push(g.read_f32(pair[1]));
        }
        Ok(MolmoAct2Prefix { tokens: n, hidden, keys, values })
    }
}

/// The action expert's settings, from the checkpoint's
/// `action_expert_config` and the top-level action fields.
#[derive(Debug, Clone)]
pub struct MolmoAct2ActionConfig {
    /// Expert width.
    pub hidden: u64,
    /// Expert heads.
    pub heads: u64,
    /// Expert head width.
    pub head_dim: u64,
    /// Blocks, one per language-model block.
    pub layers: usize,
    /// MLP width.
    pub intermediate: u64,
    /// Sinusoidal time-embedding width.
    pub time_dim: usize,
    /// Width of one language-model block's keys or values.
    pub kv_dim: u64,
    /// Whether the projected context keys and values are RMS-normalised.
    pub context_norm: bool,
    /// Whether queries and keys are RMS-normalised per head.
    pub qk_norm: bool,
    /// Query/key norm epsilon.
    pub qk_eps: f64,
    /// Whether self-attention uses rotary positions.
    pub rope: bool,
    /// Action width (padded).
    pub max_action_dim: usize,
    /// Longest chunk.
    pub max_horizon: usize,
    /// Euler steps.
    pub steps: usize,
    /// Whether padded action dimensions are held at zero.
    pub mask_padding: bool,
    /// Whether the checkpoint was trained with discrete and continuous
    /// actions, in which case the context hides end-of-turn tokens and
    /// discrete action spans.
    pub both: bool,
    /// End-of-turn token.
    pub eos: Option<i32>,
    /// Discrete action span start and end tokens.
    pub action_span: Option<(i32, i32)>,
}

impl MolmoAct2ActionConfig {
    /// Read the action-expert settings from a checkpoint config.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field, a checkpoint without an action
    /// expert, or a variant this engine does not implement.
    pub fn from_json(c: &Value) -> Result<Self> {
        if !c.get("add_action_expert").and_then(Value::as_bool).unwrap_or(false) {
            return Err(Error::Config("this MolmoAct2 checkpoint has no action expert".into()));
        }
        let unsupported = |what: &str| Err(Error::Config(format!("MolmoAct2 action experts with {what} are not implemented")));
        if c.get("action_expert_depth_gate").and_then(Value::as_bool).unwrap_or(false) {
            return unsupported("depth gating");
        }
        // The reference cannot run a causal expert at inference either (it
        // gives attention both an explicit mask and the causal flag).
        if c.get("action_expert_config").and_then(|a| a.get("causal_attn")).and_then(Value::as_bool).unwrap_or(false) {
            return unsupported("causal self-attention");
        }
        if c.get("state_format").and_then(Value::as_str).unwrap_or("discrete") != "discrete" {
            return unsupported("continuous state embeddings");
        }
        let a = c.get("action_expert_config").ok_or_else(|| Error::Config("config gives no action_expert_config".into()))?;
        let t = c.get("text_config").ok_or_else(|| Error::Config("config gives no text_config".into()))?;
        let flag = |k: &str, d: bool| a.get(k).and_then(Value::as_bool).unwrap_or(d);
        let num = |v: &Value, k: &str, d: u64| v.get(k).and_then(Value::as_u64).unwrap_or(d);
        let hidden = num(a, "hidden_size", 1024);
        let heads = num(a, "num_heads", 16);
        let layers = num(a, "num_layers", 32) as usize;
        let ratio = a.get("mlp_ratio").and_then(Value::as_f64).unwrap_or(8.0 / 3.0);
        let multiple = a.get("ffn_multiple_of").and_then(Value::as_i64).unwrap_or(256);
        let inner = (hidden as f64 * ratio) as u64;
        let intermediate = if multiple > 0 { inner.div_ceil(multiple as u64) * multiple as u64 } else { inner };
        let id = |k: &str| c.get(k).and_then(Value::as_i64).map(|v| v as i32);
        let cfg = Self {
            hidden,
            heads,
            head_dim: if heads > 0 { hidden / heads } else { 0 },
            layers,
            intermediate,
            time_dim: num(a, "timestep_embed_dim", 256) as usize,
            kv_dim: u(t, "num_key_value_heads")? * u(t, "head_dim")?,
            context_norm: flag("context_layer_norm", true),
            qk_norm: flag("qk_norm", true),
            qk_eps: a.get("qk_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            rope: flag("rope", true),
            max_action_dim: num(c, "max_action_dim", 32) as usize,
            max_horizon: num(c, "max_action_horizon", 30) as usize,
            steps: num(c, "flow_matching_num_steps", 10) as usize,
            mask_padding: c.get("mask_action_dim_padding").and_then(Value::as_bool).unwrap_or(true),
            both: c.get("action_mode").and_then(Value::as_str) == Some("both"),
            eos: id("eos_token_id"),
            action_span: id("action_start_token_id").zip(id("action_end_token_id")),
        };
        if heads == 0 || hidden % heads != 0 || (cfg.rope && cfg.head_dim % 2 != 0) {
            return Err(Error::Config("malformed MolmoAct2 action-expert heads".into()));
        }
        if cfg.layers != u(t, "num_hidden_layers")? as usize {
            return unsupported("a block count other than the language model's");
        }
        if cfg.time_dim % 2 != 0 || cfg.time_dim < 4 {
            return unsupported("an odd or tiny time embedding");
        }
        if cfg.steps == 0 || cfg.max_horizon == 0 || cfg.max_action_dim == 0 {
            return Err(Error::Config("malformed MolmoAct2 action settings".into()));
        }
        Ok(cfg)
    }

    fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let f = WType::F32;
        let (d, ff, a) = (self.hidden, self.intermediate, self.max_action_dim as u64);
        // Rows of the 8-bit type are whole blocks of 32 values.
        let fit = |cols: u64| if linear == WType::Q8_0 && cols % 32 != 0 { WType::Bf16 } else { linear };
        let mut s = Vec::new();
        let mut lin = |name: String, rows: u64, cols: u64, bias: bool| {
            s.push(WeightSpec::new(format!("{name}.weight"), &[rows, cols], fit(cols)));
            if bias {
                s.push(WeightSpec::new(format!("{name}.bias"), &[rows], f));
            }
        };
        lin(format!("{E}time_embed.1"), d, self.time_dim as u64, true);
        lin(format!("{E}time_embed.3"), d, d, true);
        lin(format!("{E}action_embed"), d, a, true);
        lin(format!("{E}context_k_proj"), d, self.kv_dim, false);
        lin(format!("{E}context_v_proj"), d, self.kv_dim, false);
        for i in 0..self.layers {
            let b = format!("{E}blocks.{i}");
            lin(format!("{b}.self_attn.qkv"), 3 * d, d, true);
            lin(format!("{b}.self_attn.out_proj"), d, d, true);
            lin(format!("{b}.cross_attn.q_proj"), d, d, true);
            lin(format!("{b}.cross_attn.out_proj"), d, d, true);
            lin(format!("{b}.mlp.gate_proj"), ff, d, true);
            lin(format!("{b}.mlp.up_proj"), ff, d, true);
            lin(format!("{b}.mlp.down_proj"), d, ff, true);
            lin(format!("{b}.modulation.linear"), 9 * d, d, true);
        }
        lin(format!("{E}final_layer.modulation.linear"), 2 * d, d, true);
        lin(format!("{E}final_layer.linear"), a, d, true);
        s
    }

    /// The flow times `i / steps` and the Euler step `1 / steps`, in
    /// `float32` as the reference has them.
    #[must_use]
    pub fn flow_times(&self) -> (Vec<f32>, f32) {
        let n = self.steps;
        ((0..n).map(|i| (i as f64 / n as f64) as f32).collect(), (1.0 / n as f64) as f32)
    }

    /// The sinusoidal embedding of a flow time (sines then cosines),
    /// following the reference's `float32` arithmetic: frequencies
    /// `exp(-i ln(10000) / (half - 1))`.
    #[must_use]
    pub fn time_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.time_dim / 2;
        let step = (-(10000f64.ln()) / (half - 1).max(1) as f64) as f32;
        let mut out = vec![0f32; 2 * half];
        for i in 0..half {
            let freq = f64::from(i as f32 * step).exp() as f32;
            let x = f64::from(t * freq);
            out[i] = x.sin() as f32;
            out[half + i] = x.cos() as f32;
        }
        out
    }

    /// Rotary tables `[horizon][head_dim]` (cosines, sines) of the chunk
    /// positions, base 10000, both halves of a head sharing the angles
    /// (`float32` as the reference builds them).
    #[must_use]
    pub fn rotary_tables(&self, horizon: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_dim as usize / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / 10000f64.powf(f64::from(i as f32 / half as f32)) as f32).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(horizon * 2 * half), Vec::with_capacity(horizon * 2 * half));
        for p in 0..horizon {
            for _ in 0..2 {
                for &f in &inv {
                    let a = f64::from(p as f32 * f);
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        (cos, sin)
    }

    /// Which prompt tokens the expert reads: all of them, except, for a
    /// checkpoint trained with both action kinds, end-of-turn tokens and
    /// every discrete action span (start to end inclusive; an unclosed
    /// span runs to the end). A fine-tuned checkpoint trained on
    /// continuous actions alone reads every token instead.
    #[must_use]
    pub fn context_mask(&self, ids: &[i32]) -> Vec<bool> {
        let mut m: Vec<bool> = ids.iter().map(|&i| !(self.both && Some(i) == self.eos)).collect();
        if let (true, Some((start, end))) = (self.both, self.action_span) {
            let ends: Vec<usize> = ids.iter().enumerate().filter(|&(_, &i)| i == end).map(|(p, _)| p).collect();
            let mut e = 0;
            for s in ids.iter().enumerate().filter(|&(_, &i)| i == start).map(|(p, _)| p) {
                while e < ends.len() && ends[e] < s {
                    e += 1;
                }
                if e == ends.len() {
                    m[s..].fill(false);
                    break;
                }
                m[s..=ends[e]].fill(false);
                e += 1;
            }
        }
        m
    }
}

/// The MolmoAct2 action expert on one device.
pub struct MolmoAct2ActionExpert {
    cfg: MolmoAct2ActionConfig,
    backend: Backend,
    weights: Weights,
    exact: bool,
}

impl std::fmt::Debug for MolmoAct2ActionExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MolmoAct2ActionExpert").field("device", &self.backend.name()).finish_non_exhaustive()
    }
}

/// `norm(x) * (1 + scale) + shift`, the modulated RMS norm.
fn modulate(g: &mut Graph, x: Tn, shift: Tn, scale: Tn) -> Tn {
    let h = g.rms_norm(x, 1e-6);
    let s = g.scale_bias(scale, 1.0, 1.0);
    let h = g.mul(h, s);
    g.add(h, shift)
}

impl MolmoAct2ActionExpert {
    /// Load the action expert from a checkpoint directory (`config.json`
    /// and its safetensors files).
    ///
    /// # Errors
    /// On a missing or malformed file, a checkpoint without an action
    /// expert, an unsupported variant, or a backend failure.
    pub fn load(dir: &Path, opts: LoadOptions) -> Result<Self> {
        let cfg = MolmoAct2ActionConfig::from_json(&read_json(&dir.join("config.json"))?)?;
        let st = SafeTensors::open(&checkpoint_files(dir))?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let linear = match opts.precision {
            Precision::F32 => WType::F32,
            Precision::Bf16 => WType::Bf16,
            Precision::Q8_0 => WType::Q8_0,
        };
        let weights = Weights::load(&backend, &st, &cfg.weight_specs(linear))?;
        Ok(Self { cfg, backend, weights, exact })
    }

    /// The expert's settings.
    #[must_use]
    pub fn config(&self) -> &MolmoAct2ActionConfig {
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
        self.weights.bytes()
    }

    /// The per-block modulations of every flow step `[steps][layers][9 *
    /// hidden]`, the final layer's `[steps][2 * hidden]`, and every block's
    /// context keys `[tokens][heads][head_dim]` (normalised) and values
    /// `[tokens][hidden]`.
    #[allow(clippy::type_complexity)]
    fn context(&self, keys: &[Vec<f32>], values: &[Vec<f32>], n: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let c = &self.cfg;
        let w = &self.weights;
        let (hd, nh) = (c.head_dim as i64, c.heads as i64);
        let (steps, n_i) = (c.steps as i64, n as i64);
        let mut g = Graph::new(&self.backend)?;
        let temb = g.input(sys::GGML_TYPE_F32, &[c.time_dim as i64, steps]);
        let cond = lin_b(&mut g, w, &format!("{E}time_embed.1"), temb);
        let cond = g.silu(cond);
        let cond = lin_b(&mut g, w, &format!("{E}time_embed.3"), cond);
        let act = g.silu(cond);
        let mut mods = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let m = lin_b(&mut g, w, &format!("{E}blocks.{i}.modulation.linear"), act);
            mods.push(m);
        }
        let fin = lin_b(&mut g, w, &format!("{E}final_layer.modulation.linear"), act);
        let mut ins = Vec::with_capacity(2 * c.layers);
        let mut outs = Vec::with_capacity(2 * c.layers);
        for _ in 0..c.layers {
            let kin = g.input(sys::GGML_TYPE_F32, &[c.kv_dim as i64, n_i]);
            let vin = g.input(sys::GGML_TYPE_F32, &[c.kv_dim as i64, n_i]);
            let mut k = g.linear(w.get(&format!("{E}context_k_proj.weight")), kin);
            let mut v = g.linear(w.get(&format!("{E}context_v_proj.weight")), vin);
            if c.context_norm {
                k = g.rms_norm(k, 1e-6);
                v = g.rms_norm(v, 1e-6);
            }
            if c.qk_norm {
                let kh = g.reshape(k, &[hd, nh, n_i]);
                k = g.rms_norm(kh, c.qk_eps as f32);
            }
            ins.push((kin, vin));
            outs.push((k, v));
        }
        let mut all = mods.clone();
        all.push(fin);
        all.extend(outs.iter().flat_map(|&(k, v)| [k, v]));
        g.finish(&all)?;
        let temb_host: Vec<f32> = c.flow_times().0.iter().flat_map(|&t| c.time_embedding(t)).collect();
        g.set_f32(temb, &temb_host);
        for (l, &(kin, vin)) in ins.iter().enumerate() {
            g.set_f32(kin, &keys[l]);
            g.set_f32(vin, &values[l]);
        }
        g.compute()?;
        // Regroup the modulations step-major: [steps][layers][9 * hidden].
        let per = 9 * c.hidden as usize;
        let read: Vec<Vec<f32>> = mods.iter().map(|&m| g.read_f32(m)).collect();
        let mut block = Vec::with_capacity(c.steps * c.layers * per);
        for s in 0..c.steps {
            for r in &read {
                block.extend_from_slice(&r[s * per..(s + 1) * per]);
            }
        }
        let fin = g.read_f32(fin);
        let ks = outs.iter().map(|&(k, _)| g.read_f32(k)).collect();
        let vs = outs.iter().map(|&(_, v)| g.read_f32(v)).collect();
        Ok((block, fin, ks, vs))
    }

    /// One action chunk `[horizon][max_action_dim]` by flow matching from
    /// `noise` (same shape), reading per language-model block the `keys`
    /// (after the rotary embedding) and `values` `[tokens][kv_dim]` (as
    /// [`MolmoAct2Lm::prefill`] gives them) at the tokens `mask` keeps
    /// ([`MolmoAct2ActionConfig::context_mask`]). `padding` marks action
    /// dimensions held at zero when the checkpoint masks them.
    ///
    /// # Errors
    /// On inputs of the wrong size, a horizon outside the checkpoint's, an
    /// empty context, or a backend failure.
    #[allow(clippy::too_many_lines)]
    pub fn sample(&self, keys: &[Vec<f32>], values: &[Vec<f32>], mask: &[bool], noise: &[f32], horizon: usize, padding: Option<&[bool]>) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = mask.len();
        let kv = c.kv_dim as usize;
        let a = c.max_action_dim;
        if keys.len() != c.layers || values.len() != c.layers || keys.iter().chain(values).any(|x| x.len() != n * kv) {
            return Err(Error::Request(format!("the expert reads {} blocks of {n} tokens x {kv} keys and values", c.layers)));
        }
        if !mask.iter().any(|&m| m) {
            return Err(Error::Request("the expert's context hides every token".into()));
        }
        if horizon == 0 || horizon > c.max_horizon || noise.len() != horizon * a {
            return Err(Error::Request(format!("noise must be [1..={}][{a}]", c.max_horizon)));
        }
        if padding.is_some_and(|p| p.len() != a) {
            return Err(Error::Request(format!("the padding mask covers {a} action dimensions")));
        }
        let keep: Option<Vec<f32>> = match padding {
            Some(p) if c.mask_padding => Some(p.iter().map(|&pad| if pad { 0.0 } else { 1.0 }).collect()),
            _ => None,
        };
        let (mods, fin, ck, cv) = self.context(keys, values, n)?;

        let w = &self.weights;
        let exact = self.exact;
        let (d, hd, nh) = (c.hidden as i64, c.head_dim as i64, c.heads as i64);
        let (h_i, n_i, a_i) = (horizon as i64, n as i64, a as i64);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut g = Graph::new(&self.backend)?;
        let traj = g.input(sys::GGML_TYPE_F32, &[a_i, h_i]);
        let step_mods = g.input(sys::GGML_TYPE_F32, &[9 * d, c.layers as i64]);
        let step_fin = g.input(sys::GGML_TYPE_F32, &[2 * d]);
        let keep_t = keep.as_ref().map(|_| g.input(sys::GGML_TYPE_F32, &[a_i]));
        let cross_mask = g.input(sys::GGML_TYPE_F16, &[n_i, h_i]);
        let rope = if c.rope { Some((g.input(sys::GGML_TYPE_F32, &[hd, 1, h_i]), g.input(sys::GGML_TYPE_F32, &[hd, 1, h_i]))) } else { None };
        let ctx: Vec<(Tn, Tn)> = (0..c.layers).map(|_| (g.input(sys::GGML_TYPE_F32, &[hd, nh, n_i]), g.input(sys::GGML_TYPE_F32, &[hd, nh, n_i]))).collect();

        let chunk = |g: &mut Graph, l: usize, j: usize| g.view_1d(step_mods, d, (l * 9 + j) * d as usize);
        let mut x = lin_b(&mut g, w, &format!("{E}action_embed"), traj);
        for (l, &(k_ctx, v_ctx)) in ctx.iter().enumerate() {
            let b = format!("{E}blocks.{l}");
            let m: Vec<Tn> = (0..9).map(|j| chunk(&mut g, l, j)).collect();
            // Self-attention over the chunk.
            let h = modulate(&mut g, x, m[0], m[1]);
            let qkv = lin_b(&mut g, w, &format!("{b}.self_attn.qkv"), h);
            let q = g.view_heads(qkv, 0, hd, nh);
            let mut q = g.cont(q);
            let k = g.view_heads(qkv, d, hd, nh);
            let mut k = g.cont(k);
            let v = g.view_heads(qkv, 2 * d, hd, nh);
            let v = g.cont(v);
            if c.qk_norm {
                q = g.rms_norm(q, c.qk_eps as f32);
                k = g.rms_norm(k, c.qk_eps as f32);
            }
            if let Some((cos, sin)) = rope {
                q = g.rotate_half_rope(q, cos, sin);
                k = g.rotate_half_rope(k, cos, sin);
            }
            let o = attend(&mut g, q, k, v, None, scale, exact);
            let o = g.reshape(o, &[d, h_i]);
            let o = lin_b(&mut g, w, &format!("{b}.self_attn.out_proj"), o);
            let o = g.mul(o, m[2]);
            x = g.add(x, o);
            // Cross-attention to the language model's block.
            let h = modulate(&mut g, x, m[3], m[4]);
            let q = lin_b(&mut g, w, &format!("{b}.cross_attn.q_proj"), h);
            let mut q = g.reshape(q, &[hd, nh, h_i]);
            if c.qk_norm {
                q = g.rms_norm(q, c.qk_eps as f32);
            }
            let o = attend(&mut g, q, k_ctx, v_ctx, Some(cross_mask), scale, exact);
            let o = g.reshape(o, &[d, h_i]);
            let o = lin_b(&mut g, w, &format!("{b}.cross_attn.out_proj"), o);
            let o = g.mul(o, m[5]);
            x = g.add(x, o);
            // Gated MLP.
            let h = modulate(&mut g, x, m[6], m[7]);
            let gate = lin_b(&mut g, w, &format!("{b}.mlp.gate_proj"), h);
            let up = lin_b(&mut g, w, &format!("{b}.mlp.up_proj"), h);
            let h = g.swiglu_split(gate, up);
            let h = lin_b(&mut g, w, &format!("{b}.mlp.down_proj"), h);
            let h = g.mul(h, m[8]);
            x = g.add(x, h);
        }
        let shift = g.view_1d(step_fin, d, 0);
        let fscale = g.view_1d(step_fin, d, d as usize);
        let h = modulate(&mut g, x, shift, fscale);
        let mut vel = lin_b(&mut g, w, &format!("{E}final_layer.linear"), h);
        if let Some(kt) = keep_t {
            vel = g.mul(vel, kt);
        }
        let (_, dt) = c.flow_times();
        let vel = g.scale_bias(vel, dt, 0.0);
        let mut next = g.add(traj, vel);
        if let Some(kt) = keep_t {
            next = g.mul(next, kt);
        }
        // The graph runs once per step. Inputs that stay the same are marked
        // as outputs too, so the allocator never reuses their memory.
        let mut outs = vec![next, cross_mask];
        outs.extend(keep_t);
        outs.extend(rope.iter().flat_map(|&(cs, sn)| [cs, sn]));
        outs.extend(ctx.iter().flat_map(|&(k, v)| [k, v]));
        g.finish(&outs)?;

        for (l, &(k, v)) in ctx.iter().enumerate() {
            g.set_f32(k, &ck[l]);
            g.set_f32(v, &cv[l]);
        }
        g.set_f16(cross_mask, &(0..horizon).flat_map(|_| mask.iter().map(|&m| if m { 0.0 } else { f32::NEG_INFINITY })).collect::<Vec<_>>());
        if let Some((cs, sn)) = rope {
            let (tc, ts) = c.rotary_tables(horizon);
            g.set_f32(cs, &tc);
            g.set_f32(sn, &ts);
        }
        let mut x: Vec<f32> = noise.to_vec();
        if let (Some(kt), Some(kv)) = (keep_t, keep.as_ref()) {
            g.set_f32(kt, kv);
            for row in x.chunks_mut(a) {
                for (v, k) in row.iter_mut().zip(kv) {
                    *v *= k;
                }
            }
        }
        let per = 9 * c.hidden as usize * c.layers;
        let fd = 2 * c.hidden as usize;
        for s in 0..c.steps {
            g.set_f32(traj, &x);
            g.set_f32(step_mods, &mods[s * per..(s + 1) * per]);
            g.set_f32(step_fin, &fin[s * fd..(s + 1) * fd]);
            g.compute()?;
            x = g.read_f32(next);
        }
        Ok(x)
    }
}
