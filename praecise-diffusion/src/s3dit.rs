//! Single-stream diffusion transformer (the Z-Image layout).
//!
//! Image patches and caption tokens are refined separately (two modulated
//! blocks for the image, two plain blocks for the caption), then run as one
//! sequence, image first, through the main modulated blocks. Each stream is
//! padded to a multiple of 32 tokens with a learned pad token; nothing is
//! masked, so the padding takes part in attention exactly as it does in the
//! reference.
//!
//! Rotary embeddings use three integer positions per token (caption index,
//! row, column), each axis with its own frequency band. The checkpoint rotates
//! adjacent pairs; the query and key projection rows (and their norms) are
//! reordered at load so each head's pairs sit half a head apart, and the
//! rotation is applied from host tables.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use llama_cpp_sys_2 as sys;

/// Width of the sinusoidal timestep features.
pub const TIME_FEATURES: usize = 256;
/// Width of the timestep embedding the blocks are modulated by.
const ADALN_WIDTH: u64 = 256;
/// Hidden width of the timestep MLP.
const TIME_HIDDEN: u64 = 1024;
/// Streams are padded to a multiple of this many tokens.
pub const SEQ_MULTIPLE: usize = 32;

/// Transformer configuration, read from `transformer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct S3DitConfig {
    /// Spatial patch sizes (one supported: 2).
    pub all_patch_size: Vec<u64>,
    /// Temporal patch sizes (one supported: 1).
    pub all_f_patch_size: Vec<u64>,
    /// Latent channels.
    pub in_channels: u64,
    /// Model width.
    pub dim: u64,
    /// Main blocks.
    pub n_layers: usize,
    /// Refiner blocks per stream.
    pub n_refiner_layers: usize,
    /// Attention heads.
    pub n_heads: u64,
    /// Key/value heads (equal to the heads).
    pub n_kv_heads: u64,
    /// Norm epsilon.
    pub norm_eps: f64,
    /// Per-head query/key norms.
    pub qk_norm: bool,
    /// Caption feature width.
    pub cap_feat_dim: u64,
    /// Rotary base.
    pub rope_theta: f64,
    /// Timesteps in `[0, 1]` are multiplied by this.
    pub t_scale: f64,
    /// Head dimensions per rotary axis.
    pub axes_dims: Vec<u64>,
    /// Positions per rotary axis.
    pub axes_lens: Vec<u64>,
    /// Image-reference features (unsupported).
    #[serde(default)]
    pub siglip_feat_dim: Option<u64>,
}

impl S3DitConfig {
    /// Refuse variants this implementation has not been checked against.
    ///
    /// # Errors
    /// [`Error::Config`] naming the first unsupported setting.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("transformer: {m}")));
        if self.all_patch_size != [2] || self.all_f_patch_size != [1] {
            return bad("only 2x2 spatial patches are implemented");
        }
        if self.n_kv_heads != self.n_heads || !self.qk_norm || self.siglip_feat_dim.is_some() {
            return bad("only full-head attention with query/key norms and no image features is implemented");
        }
        if self.dim % self.n_heads != 0 || self.axes_dims.len() != 3 || self.axes_lens.len() != 3 {
            return bad("expected three rotary axes over whole heads");
        }
        if self.axes_dims.iter().sum::<u64>() != self.head_dim() || self.axes_dims.iter().any(|d| d % 2 != 0) {
            return bad("rotary axes do not split the head");
        }
        if self.dim < ADALN_WIDTH {
            return bad("width below the modulation width");
        }
        Ok(())
    }

    /// Head width.
    #[must_use]
    pub fn head_dim(&self) -> u64 {
        self.dim / self.n_heads
    }

    /// Width of one patch token.
    #[must_use]
    pub fn patch_dim(&self) -> u64 {
        4 * self.in_channels
    }

    fn ff(&self) -> u64 {
        (self.dim as f64 / 3.0 * 8.0) as u64
    }

    /// Row order that puts each head's rotated pairs `(2j, 2j + 1)` at
    /// `(j, j + head / 2)`.
    fn pair_rows(&self, heads: u64) -> Vec<usize> {
        let hd = self.head_dim() as usize;
        let half = hd / 2;
        (0..heads as usize * hd)
            .map(|r| {
                let (h, j) = (r / hd, r % hd);
                h * hd + if j < half { 2 * j } else { 2 * (j - half) + 1 }
            })
            .collect()
    }

    fn block_specs(&self, v: &mut Vec<WeightSpec>, p: &str, modulated: bool, linear: WType) {
        let (d, hd, ff) = (self.dim, self.head_dim(), self.ff());
        let a = format!("{p}.attention");
        v.push(WeightSpec::new(format!("{a}.to_q.weight"), &[d, d], linear).with_rows(self.pair_rows(self.n_heads)));
        v.push(WeightSpec::new(format!("{a}.to_k.weight"), &[d, d], linear).with_rows(self.pair_rows(self.n_heads)));
        v.push(WeightSpec::new(format!("{a}.to_v.weight"), &[d, d], linear));
        v.push(WeightSpec::new(format!("{a}.to_out.0.weight"), &[d, d], linear));
        v.push(WeightSpec::new(format!("{a}.norm_q.weight"), &[hd], WType::F32).with_rows(self.pair_rows(1)));
        v.push(WeightSpec::new(format!("{a}.norm_k.weight"), &[hd], WType::F32).with_rows(self.pair_rows(1)));
        for n in ["attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"] {
            v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[d], WType::F32));
        }
        v.push(WeightSpec::new(format!("{p}.feed_forward.w1.weight"), &[ff, d], linear));
        v.push(WeightSpec::new(format!("{p}.feed_forward.w3.weight"), &[ff, d], linear));
        v.push(WeightSpec::new(format!("{p}.feed_forward.w2.weight"), &[d, ff], linear));
        if modulated {
            v.push(WeightSpec::new(format!("{p}.adaLN_modulation.0.weight"), &[4 * d, ADALN_WIDTH], linear));
            v.push(WeightSpec::new(format!("{p}.adaLN_modulation.0.bias"), &[4 * d], WType::F32));
        }
    }

    /// Every transformer weight.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, pd, c) = (self.dim, self.patch_dim(), self.cap_feat_dim);
        let mut v = vec![
            WeightSpec::new("all_x_embedder.2-1.weight", &[d, pd], WType::F32),
            WeightSpec::new("all_x_embedder.2-1.bias", &[d], WType::F32),
            WeightSpec::new("x_pad_token", &[1, d], WType::F32),
            WeightSpec::new("cap_pad_token", &[1, d], WType::F32),
            WeightSpec::new("cap_embedder.0.weight", &[c], WType::F32),
            WeightSpec::new("cap_embedder.1.weight", &[d, c], linear),
            WeightSpec::new("cap_embedder.1.bias", &[d], WType::F32),
            WeightSpec::new("t_embedder.mlp.0.weight", &[TIME_HIDDEN, TIME_FEATURES as u64], WType::F32),
            WeightSpec::new("t_embedder.mlp.0.bias", &[TIME_HIDDEN], WType::F32),
            WeightSpec::new("t_embedder.mlp.2.weight", &[ADALN_WIDTH, TIME_HIDDEN], WType::F32),
            WeightSpec::new("t_embedder.mlp.2.bias", &[ADALN_WIDTH], WType::F32),
            WeightSpec::new("all_final_layer.2-1.adaLN_modulation.1.weight", &[d, ADALN_WIDTH], WType::F32),
            WeightSpec::new("all_final_layer.2-1.adaLN_modulation.1.bias", &[d], WType::F32),
            WeightSpec::new("all_final_layer.2-1.linear.weight", &[pd, d], WType::F32),
            WeightSpec::new("all_final_layer.2-1.linear.bias", &[pd], WType::F32),
        ];
        for i in 0..self.n_refiner_layers {
            self.block_specs(&mut v, &format!("noise_refiner.{i}"), true, linear);
            self.block_specs(&mut v, &format!("context_refiner.{i}"), false, linear);
        }
        for i in 0..self.n_layers {
            self.block_specs(&mut v, &format!("layers.{i}"), true, linear);
        }
        v
    }

    /// Rotary cos and sin tables `[tokens][head width]` for integer positions
    /// `[tokens][3]`: pair `j` of axis `a` turns by `pos[a] / theta^(2j / d_a)`,
    /// computed in double precision and rounded as the reference rounds it.
    #[must_use]
    pub fn rotary_tables(&self, positions: &[[u32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim() as usize;
        let half = hd / 2;
        let mut freqs = Vec::with_capacity(half);
        for (a, &d) in self.axes_dims.iter().enumerate() {
            for j in 0..d / 2 {
                freqs.push((a, 1.0 / self.rope_theta.powf((2 * j) as f64 / d as f64)));
            }
        }
        let mut cos = Vec::with_capacity(positions.len() * hd);
        let mut sin = Vec::with_capacity(positions.len() * hd);
        for p in positions {
            let ang: Vec<f32> = freqs.iter().map(|&(a, f)| (f64::from(p[a]) * f) as f32).collect();
            for j in 0..hd {
                cos.push(ang[j % half].cos());
                sin.push(ang[j % half].sin());
            }
        }
        (cos, sin)
    }

    /// Sinusoidal features of a timestep already scaled by `t_scale`: `cos`
    /// then `sin` at 128 log-spaced frequencies.
    #[must_use]
    pub fn time_features(t: f32) -> Vec<f32> {
        let half = TIME_FEATURES / 2;
        let k = -(10000f64.ln() as f32);
        let freqs: Vec<f32> = (0..half).map(|i| (k * i as f32 / half as f32).exp()).collect();
        let mut out: Vec<f32> = freqs.iter().map(|f| (t * f).cos()).collect();
        out.extend(freqs.iter().map(|f| (t * f).sin()));
        out
    }
}

/// The number of tokens a stream of `n` occupies after padding.
#[must_use]
pub fn padded(n: usize) -> usize {
    n.div_ceil(SEQ_MULTIPLE) * SEQ_MULTIPLE
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    g.mul(h, w.get(name))
}

/// Per-block modulation: `(1 + scale_msa, tanh gate_msa, 1 + scale_mlp, tanh gate_mlp)`.
struct Modulation {
    scale_msa: Tn,
    gate_msa: Tn,
    scale_mlp: Tn,
    gate_mlp: Tn,
}

fn modulation(g: &mut Graph, w: &Weights, p: &str, temb: Tn, d: i64) -> Modulation {
    let m = g.linear_b(w.get(&format!("{p}.adaLN_modulation.0.weight")), w.get(&format!("{p}.adaLN_modulation.0.bias")), temb);
    let mut part = |i: i64| g.view_1d(m, d, (i * d) as usize);
    let (a, b, c, e) = (part(0), part(1), part(2), part(3));
    let scale_msa = g.scale_bias(a, 1.0, 1.0);
    let gate_msa = g.tanh(b);
    let scale_mlp = g.scale_bias(c, 1.0, 1.0);
    let gate_mlp = g.tanh(e);
    Modulation { scale_msa, gate_msa, scale_mlp, gate_mlp }
}

#[derive(Clone, Copy)]
struct Ctx {
    hd: i64,
    heads: i64,
    eps: f32,
    exact: bool,
}

fn attention(g: &mut Graph, w: &Weights, a: &str, c: Ctx, x: Tn, cos: Tn, sin: Tn) -> Tn {
    let n = x.ne(1);
    let wn = |s: &str| w.get(&format!("{a}.{s}"));
    let q = g.linear(wn("to_q.weight"), x);
    let k = g.linear(wn("to_k.weight"), x);
    let v = g.linear(wn("to_v.weight"), x);
    let q = g.reshape(q, &[c.hd, c.heads, n]);
    let k = g.reshape(k, &[c.hd, c.heads, n]);
    let v = g.reshape(v, &[c.hd, c.heads, n]);
    let q = rms(g, w, &format!("{a}.norm_q.weight"), q, 1e-5);
    let k = rms(g, w, &format!("{a}.norm_k.weight"), k, 1e-5);
    let q = g.rotate_half_rope(q, cos, sin);
    let k = g.rotate_half_rope(k, cos, sin);
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    let scale = 1.0 / (c.hd as f32).sqrt();
    let o = if c.exact {
        let k = g.cont(k);
        let v = g.cont(v);
        g.attention_exact(q, k, v, None, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, None, scale, true)
    };
    let o = g.reshape(o, &[c.hd * c.heads, n]);
    g.linear(wn("to_out.0.weight"), o)
}

fn block(g: &mut Graph, w: &Weights, p: &str, c: Ctx, x: Tn, m: Option<&Modulation>, cos: Tn, sin: Tn) -> Tn {
    let h = rms(g, w, &format!("{p}.attention_norm1.weight"), x, c.eps);
    let h = match m {
        Some(m) => g.mul(h, m.scale_msa),
        None => h,
    };
    let o = attention(g, w, &format!("{p}.attention"), c, h, cos, sin);
    let o = rms(g, w, &format!("{p}.attention_norm2.weight"), o, c.eps);
    let o = match m {
        Some(m) => g.mul(o, m.gate_msa),
        None => o,
    };
    let x = g.add(x, o);
    let h = rms(g, w, &format!("{p}.ffn_norm1.weight"), x, c.eps);
    let h = match m {
        Some(m) => g.mul(h, m.scale_mlp),
        None => h,
    };
    let a = g.linear(w.get(&format!("{p}.feed_forward.w1.weight")), h);
    let b = g.linear(w.get(&format!("{p}.feed_forward.w3.weight")), h);
    let f = g.swiglu_split(a, b);
    let f = g.linear(w.get(&format!("{p}.feed_forward.w2.weight")), f);
    let f = rms(g, w, &format!("{p}.ffn_norm2.weight"), f, c.eps);
    let f = match m {
        Some(m) => g.mul(f, m.gate_mlp),
        None => f,
    };
    g.add(x, f)
}

/// Inputs and output of one transformer evaluation.
#[derive(Debug, Clone, Copy)]
pub struct S3DitIo {
    /// Image patch tokens `[64, image tokens]`.
    pub patches: Tn,
    /// Caption features `[cap width, caption tokens]`.
    pub caption: Tn,
    /// Timestep features `[256]`.
    pub time: Tn,
    /// Rotary tables over the padded image then the padded caption
    /// `[head width, tokens]`.
    pub cos: Tn,
    /// Sine table, same layout.
    pub sin: Tn,
    /// Prediction for the image tokens `[64, image tokens]`.
    pub out: Tn,
}

/// One evaluation over `n_img` image tokens and `n_cap` caption tokens.
#[must_use]
pub fn build(g: &mut Graph, cfg: &S3DitConfig, w: &Weights, n_img: i64, n_cap: i64, exact: bool) -> S3DitIo {
    let d = cfg.dim as i64;
    let c = Ctx { hd: cfg.head_dim() as i64, heads: cfg.n_heads as i64, eps: cfg.norm_eps as f32, exact };
    let (pi, pc) = (padded(n_img as usize) as i64, padded(n_cap as usize) as i64);
    let patches = g.input(sys::GGML_TYPE_F32, &[cfg.patch_dim() as i64, n_img]);
    let caption = g.input(sys::GGML_TYPE_F32, &[cfg.cap_feat_dim as i64, n_cap]);
    let time = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
    let cos = g.input(sys::GGML_TYPE_F32, &[c.hd, pi + pc]);
    let sin = g.input(sys::GGML_TYPE_F32, &[c.hd, pi + pc]);
    let cos3 = g.reshape(cos, &[c.hd, 1, pi + pc]);
    let sin3 = g.reshape(sin, &[c.hd, 1, pi + pc]);
    let cos_x = g.view_4d(cos3, [c.hd, 1, pi, 1], cos3.nb(1), cos3.nb(2), cos3.nb(3), 0);
    let sin_x = g.view_4d(sin3, [c.hd, 1, pi, 1], sin3.nb(1), sin3.nb(2), sin3.nb(3), 0);
    let cos_c = g.view_4d(cos3, [c.hd, 1, pc, 1], cos3.nb(1), cos3.nb(2), cos3.nb(3), pi as usize * cos3.nb(2));
    let sin_c = g.view_4d(sin3, [c.hd, 1, pc, 1], sin3.nb(1), sin3.nb(2), sin3.nb(3), pi as usize * sin3.nb(2));

    let t = g.linear_b(w.get("t_embedder.mlp.0.weight"), w.get("t_embedder.mlp.0.bias"), time);
    let t = g.silu(t);
    let temb = g.linear_b(w.get("t_embedder.mlp.2.weight"), w.get("t_embedder.mlp.2.bias"), t);

    let pad_with = |g: &mut Graph, x: Tn, token: &str, total: i64| {
        let n = x.ne(1);
        if n == total {
            return x;
        }
        let pad = g.repeat_to(w.get(token), [d, total - n, 1, 1]);
        g.concat(x, pad, 1)
    };
    let x = g.linear_b(w.get("all_x_embedder.2-1.weight"), w.get("all_x_embedder.2-1.bias"), patches);
    let mut x = pad_with(g, x, "x_pad_token", pi);
    for i in 0..cfg.n_refiner_layers {
        let p = format!("noise_refiner.{i}");
        let m = modulation(g, w, &p, temb, d);
        x = block(g, w, &p, c, x, Some(&m), cos_x, sin_x);
    }
    let cap = rms(g, w, "cap_embedder.0.weight", caption, c.eps);
    let cap = g.linear_b(w.get("cap_embedder.1.weight"), w.get("cap_embedder.1.bias"), cap);
    let mut cap = pad_with(g, cap, "cap_pad_token", pc);
    for i in 0..cfg.n_refiner_layers {
        cap = block(g, w, &format!("context_refiner.{i}"), c, cap, None, cos_c, sin_c);
    }
    let mut u = g.concat(x, cap, 1);
    for i in 0..cfg.n_layers {
        let p = format!("layers.{i}");
        let m = modulation(g, w, &p, temb, d);
        u = block(g, w, &p, c, u, Some(&m), cos3, sin3);
    }
    let img = g.view_cols(u, 0, n_img);
    let img = g.cont(img);
    let h = g.norm(img, 1e-6);
    let s = g.silu(temb);
    let s = g.linear_b(w.get("all_final_layer.2-1.adaLN_modulation.1.weight"), w.get("all_final_layer.2-1.adaLN_modulation.1.bias"), s);
    let s = g.scale_bias(s, 1.0, 1.0);
    let h = g.mul(h, s);
    let out = g.linear_b(w.get("all_final_layer.2-1.linear.weight"), w.get("all_final_layer.2-1.linear.bias"), h);
    S3DitIo { patches, caption, time, cos, sin, out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> S3DitConfig {
        serde_json::from_value(serde_json::json!({
            "all_f_patch_size": [1], "all_patch_size": [2], "axes_dims": [32, 48, 48],
            "axes_lens": [1536, 512, 512], "cap_feat_dim": 2560, "dim": 3840, "in_channels": 16,
            "n_heads": 30, "n_kv_heads": 30, "n_layers": 30, "n_refiner_layers": 2,
            "norm_eps": 1e-5, "qk_norm": true, "rope_theta": 256.0, "t_scale": 1000.0
        }))
        .unwrap()
    }

    #[test]
    fn the_released_layout_validates() {
        let c = cfg();
        c.validate().unwrap();
        assert_eq!(c.ff(), 10240);
        assert_eq!(c.head_dim(), 128);
    }

    #[test]
    fn pair_rows_move_adjacent_pairs_half_a_head_apart() {
        let r = cfg().pair_rows(2);
        assert_eq!(&r[..3], &[0, 2, 4]);
        assert_eq!(r[64], 1);
        assert_eq!(r[127], 127);
        assert_eq!(r[128 + 64], 129);
    }

    #[test]
    fn rotary_axes_take_their_own_bands() {
        let c = cfg();
        // Only the column axis moves: pairs 40.. of each head turn.
        let (_, sin) = c.rotary_tables(&[[0, 0, 3]]);
        for j in 0..64 {
            assert_eq!(sin[j] != 0.0, j >= 40, "pair {j}");
        }
        assert_eq!(padded(1), 32);
        assert_eq!(padded(64), 64);
    }
}
