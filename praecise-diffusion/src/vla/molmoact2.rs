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
