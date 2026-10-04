//! Qwen-Image: a dual-stream diffusion transformer over packed 2x2 latent
//! patches, conditioned on Qwen2.5-VL hidden states.
//!
//! Every block keeps separate image and text weights and joins the two
//! streams in one attention: text tokens first, then the image tokens of the
//! target followed by those of any reference images. Positions are
//! three-axis rotary (frame, row, column) with rows and columns centred on
//! the image and text tokens placed on the diagonal past the largest image
//! half-extent. Editing checkpoints (`zero_cond_t`) modulate reference tokens
//! with the embedding of timestep zero while the target and the text use the
//! current timestep.

#[cfg(test)]
pub(crate) mod parity;
pub mod pipeline;
pub mod vae;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::s3dit::{S3DitConfig, TIME_FEATURES};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

/// `transformer/config.json` of a Qwen-Image checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenImageConfig {
    pub patch_size: u64,
    pub in_channels: u64,
    #[serde(default)]
    pub out_channels: Option<u64>,
    pub num_layers: usize,
    pub attention_head_dim: u64,
    pub num_attention_heads: u64,
    pub joint_attention_dim: u64,
    #[serde(default)]
    pub guidance_embeds: bool,
    pub axes_dims_rope: [u64; 3],
    #[serde(default)]
    pub zero_cond_t: bool,
    #[serde(default)]
    pub use_additional_t_cond: bool,
    #[serde(default)]
    pub use_layer3d_rope: bool,
}

impl QwenImageConfig {
    /// Refuse layouts this implementation does not compute.
    ///
    /// # Errors
    /// [`Error::Config`] naming the unsupported field.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("image transformer: {m}")));
        if self.guidance_embeds || self.use_additional_t_cond || self.use_layer3d_rope {
            return bad("guidance embeddings, additional timestep conditions and layered rotary positions are not supported");
        }
        if self.axes_dims_rope.iter().sum::<u64>() != self.attention_head_dim || self.axes_dims_rope.iter().any(|d| d % 2 != 0) {
            return bad("rotary axes must be even and fill the head");
        }
        if self.num_layers == 0 || self.num_attention_heads == 0 || self.patch_size == 0 {
            return bad("empty layout");
        }
        Ok(())
    }

    /// Model width.
    #[must_use]
    pub fn inner(&self) -> u64 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Channels of one output token.
    #[must_use]
    pub fn out_dim(&self) -> u64 {
        self.patch_size * self.patch_size * self.out_channels.unwrap_or(self.in_channels)
    }

    /// Row order that puts each head's rotated pairs `(2j, 2j + 1)` at
    /// `(j, j + head / 2)`.
    fn pair_rows(&self, heads: u64) -> Vec<usize> {
        let hd = self.attention_head_dim as usize;
        let half = hd / 2;
        (0..heads as usize * hd)
            .map(|r| {
                let (h, j) = (r / hd, r % hd);
                h * hd + if j < half { 2 * j } else { 2 * (j - half) + 1 }
            })
            .collect()
    }

    fn linear(v: &mut Vec<WeightSpec>, name: &str, out: u64, inp: u64, ty: WType, rows: Option<Vec<usize>>) {
        let w = WeightSpec::new(format!("{name}.weight"), &[out, inp], ty);
        let b = WeightSpec::new(format!("{name}.bias"), &[out], WType::F32);
        match rows {
            Some(r) => {
                v.push(w.with_rows(r.clone()));
                v.push(b.with_rows(r));
            }
            None => {
                v.push(w);
                v.push(b);
            }
        }
    }

    /// Every transformer weight.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, hd, jd) = (self.inner(), self.attention_head_dim, self.joint_attention_dim);
        let mut v = vec![WeightSpec::new("txt_norm.weight", &[jd], WType::F32)];
        Self::linear(&mut v, "img_in", d, self.in_channels, WType::F32, None);
        Self::linear(&mut v, "txt_in", d, jd, linear, None);
        Self::linear(&mut v, "time_text_embed.timestep_embedder.linear_1", d, TIME_FEATURES as u64, WType::F32, None);
        Self::linear(&mut v, "time_text_embed.timestep_embedder.linear_2", d, d, WType::F32, None);
        Self::linear(&mut v, "norm_out.linear", 2 * d, d, WType::F32, None);
        Self::linear(&mut v, "proj_out", self.out_dim(), d, WType::F32, None);
        let heads = self.num_attention_heads;
        for i in 0..self.num_layers {
            let p = format!("transformer_blocks.{i}");
            for s in ["img", "txt"] {
                Self::linear(&mut v, &format!("{p}.{s}_mod.1"), 6 * d, d, linear, None);
                Self::linear(&mut v, &format!("{p}.{s}_mlp.net.0.proj"), 4 * d, d, linear, None);
                Self::linear(&mut v, &format!("{p}.{s}_mlp.net.2"), d, 4 * d, linear, None);
            }
            let a = format!("{p}.attn");
            for q in ["to_q", "to_k", "add_q_proj", "add_k_proj"] {
                Self::linear(&mut v, &format!("{a}.{q}"), d, d, linear, Some(self.pair_rows(heads)));
            }
            for o in ["to_v", "add_v_proj", "to_out.0", "to_add_out"] {
                Self::linear(&mut v, &format!("{a}.{o}"), d, d, linear, None);
            }
            for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
                v.push(WeightSpec::new(format!("{a}.{n}.weight"), &[hd], WType::F32).with_rows(self.pair_rows(1)));
            }
        }
        v
    }

    /// Rotary cos and sin tables `[tokens][head width]` for the joint
    /// sequence: text first, then every image of `images` (`(rows, cols)` in
    /// patches, target first). Angles are computed in single precision as the
    /// reference computes them.
    #[must_use]
    pub fn rotary_tables(&self, n_text: usize, images: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
        let mut freqs = Vec::new();
        for (a, &d) in self.axes_dims_rope.iter().enumerate() {
            for j in 0..d / 2 {
                let e = (2 * j) as f32 / d as f32;
                freqs.push((a, 1.0f32 / ROPE_THETA.powf(e)));
            }
        }
        let origin = images.iter().map(|&(h, w)| (h / 2).max(w / 2)).max().unwrap_or(0) as i64;
        let mut pos: Vec<[i64; 3]> = (0..n_text as i64).map(|j| [origin + j; 3]).collect();
        for (idx, &(h, w)) in images.iter().enumerate() {
            let (h0, w0) = ((h - h / 2) as i64, (w - w / 2) as i64);
            for r in 0..h as i64 {
                for c in 0..w as i64 {
                    pos.push([idx as i64, r - h0, c - w0]);
                }
            }
        }
        let hd = self.attention_head_dim as usize;
        let half = hd / 2;
        let mut cos = Vec::with_capacity(pos.len() * hd);
        let mut sin = Vec::with_capacity(pos.len() * hd);
        for p in &pos {
            let ang: Vec<f32> = freqs.iter().map(|&(a, f)| p[a] as f32 * f).collect();
            for j in 0..hd {
                cos.push(ang[j % half].cos());
                sin.push(ang[j % half].sin());
            }
        }
        (cos, sin)
    }
}

#[derive(Clone, Copy)]
struct Ctx {
    d: i64,
    hd: i64,
    heads: i64,
    exact: bool,
}

fn lin(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{name}.weight")), w.get(&format!("{name}.bias")), x)
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    let h = g.rms_norm(x, EPS);
    g.mul(h, w.get(name))
}

/// `(shift, scale, gate)` of one half of a block's modulation.
#[derive(Clone, Copy)]
struct Mod {
    shift: Tn,
    scale: Tn,
    gate: Tn,
}

/// The two modulations (attention, feed-forward) of `{p}.{s}_mod` for the
/// embedding `temb` (already through SiLU).
fn mods(g: &mut Graph, w: &Weights, name: &str, temb: Tn, d: i64) -> [Mod; 2] {
    let m = lin(g, w, name, temb);
    let mut part = |i: i64| g.view_1d(m, d, (i * d) as usize);
    let v: Vec<Tn> = (0..6).map(&mut part).collect();
    let one = |g: &mut Graph, k: usize| {
        let scale = g.scale_bias(v[k + 1], 1.0, 1.0);
        Mod { shift: v[k], scale, gate: v[k + 2] }
    };
    [one(g, 0), one(g, 3)]
}

/// `norm(x) * (1 + scale) + shift`, with the tokens past `split` (reference
/// tokens) taking the second modulation when there is one.
fn modulate(g: &mut Graph, x: Tn, m: Mod, alt: Option<(Mod, i64)>) -> Tn {
    let h = g.norm(x, EPS);
    let apply = |g: &mut Graph, h: Tn, m: Mod| {
        let h = g.mul(h, m.scale);
        g.add(h, m.shift)
    };
    match alt {
        Some((r, split)) if split < x.ne(1) => {
            let a = g.view_cols(h, 0, split);
            let b = g.view_cols(h, split, x.ne(1) - split);
            let a = apply(g, a, m);
            let b = apply(g, b, r);
            g.concat(a, b, 1)
        }
        _ => apply(g, h, m),
    }
}

/// `x + gate * y`, with the same per-token choice of gate as [`modulate`].
fn gated(g: &mut Graph, x: Tn, y: Tn, m: Mod, alt: Option<(Mod, i64)>) -> Tn {
    let y = match alt {
        Some((r, split)) if split < y.ne(1) => {
            let a = g.view_cols(y, 0, split);
            let b = g.view_cols(y, split, y.ne(1) - split);
            let a = g.mul(a, m.gate);
            let b = g.mul(b, r.gate);
            g.concat(a, b, 1)
        }
        _ => g.mul(y, m.gate),
    };
    g.add(x, y)
}

fn heads(g: &mut Graph, w: &Weights, c: Ctx, a: &str, (q, k, v): (&str, &str, &str), (nq, nk): (&str, &str), x: Tn) -> (Tn, Tn, Tn) {
    let n = x.ne(1);
    let q = lin(g, w, &format!("{a}.{q}"), x);
    let k = lin(g, w, &format!("{a}.{k}"), x);
    let v = lin(g, w, &format!("{a}.{v}"), x);
    let q = g.reshape(q, &[c.hd, c.heads, n]);
    let k = g.reshape(k, &[c.hd, c.heads, n]);
    let v = g.reshape(v, &[c.hd, c.heads, n]);
    let q = rms(g, w, &format!("{a}.{nq}.weight"), q);
    let k = rms(g, w, &format!("{a}.{nk}.weight"), k);
    (q, k, v)
}

fn ff(g: &mut Graph, w: &Weights, p: &str, x: Tn, exact: bool) -> Tn {
    let h = lin(g, w, &format!("{p}.net.0.proj"), x);
    let h = if exact { g.gelu_tanh_exact(h) } else { g.gelu_tanh(h) };
    lin(g, w, &format!("{p}.net.2"), h)
}

#[allow(clippy::too_many_arguments)]
fn block(g: &mut Graph, w: &Weights, p: &str, c: Ctx, (img, txt): (Tn, Tn), (temb, temb_ref): (Tn, Option<(Tn, i64)>), (cos, sin): (Tn, Tn)) -> (Tn, Tn) {
    let [im1, im2] = mods(g, w, &format!("{p}.img_mod.1"), temb, c.d);
    let alt = temb_ref.map(|(t, split)| (mods(g, w, &format!("{p}.img_mod.1"), t, c.d), split));
    let (alt1, alt2) = (alt.map(|(m, s)| (m[0], s)), alt.map(|(m, s)| (m[1], s)));
    let [tm1, tm2] = mods(g, w, &format!("{p}.txt_mod.1"), temb, c.d);
    let (n_img, n_txt) = (img.ne(1), txt.ne(1));

    let hi = modulate(g, img, im1, alt1);
    let ht = modulate(g, txt, tm1, None);
    let a = format!("{p}.attn");
    let (qi, ki, vi) = heads(g, w, c, &a, ("to_q", "to_k", "to_v"), ("norm_q", "norm_k"), hi);
    let (qt, kt, vt) = heads(g, w, c, &a, ("add_q_proj", "add_k_proj", "add_v_proj"), ("norm_added_q", "norm_added_k"), ht);
    let q = g.concat(qt, qi, 2);
    let k = g.concat(kt, ki, 2);
    let v = g.concat(vt, vi, 2);
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
    let o = g.reshape(o, &[c.d, n_txt + n_img]);
    let ot = g.view_cols(o, 0, n_txt);
    let oi = g.view_cols(o, n_txt, n_img);
    let ot = g.cont(ot);
    let oi = g.cont(oi);
    let oi = lin(g, w, &format!("{a}.to_out.0"), oi);
    let ot = lin(g, w, &format!("{a}.to_add_out"), ot);
    let img = gated(g, img, oi, im1, alt1);
    let txt = gated(g, txt, ot, tm1, None);

    let hi = modulate(g, img, im2, alt2);
    let fi = ff(g, w, &format!("{p}.img_mlp"), hi, c.exact);
    let img = gated(g, img, fi, im2, alt2);
    let ht = modulate(g, txt, tm2, None);
    let ft = ff(g, w, &format!("{p}.txt_mlp"), ht, c.exact);
    let txt = gated(g, txt, ft, tm2, None);
    (img, txt)
}

struct Io {
    img: Tn,
    txt: Tn,
    time: Tn,
    time_ref: Option<Tn>,
    cos: Tn,
    sin: Tn,
    out: Tn,
}

fn temb(g: &mut Graph, w: &Weights, time: Tn) -> Tn {
    let t = lin(g, w, "time_text_embed.timestep_embedder.linear_1", time);
    let t = g.silu(t);
    lin(g, w, "time_text_embed.timestep_embedder.linear_2", t)
}

fn build(g: &mut Graph, cfg: &QwenImageConfig, w: &Weights, (n_img, n_target, n_txt): (i64, i64, i64), exact: bool) -> Io {
    let c = Ctx { d: cfg.inner() as i64, hd: cfg.attention_head_dim as i64, heads: cfg.num_attention_heads as i64, exact };
    let img_in = g.input(sys::GGML_TYPE_F32, &[cfg.in_channels as i64, n_img]);
    let txt_in = g.input(sys::GGML_TYPE_F32, &[cfg.joint_attention_dim as i64, n_txt]);
    let time = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
    let n = n_img + n_txt;
    let cos = g.input(sys::GGML_TYPE_F32, &[c.hd, 1, n]);
    let sin = g.input(sys::GGML_TYPE_F32, &[c.hd, 1, n]);
    let te = temb(g, w, time);
    let te_act = g.silu(te);
    let refs = cfg.zero_cond_t && n_target < n_img;
    let time_ref = refs.then(|| g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]));
    let te_ref = time_ref.map(|t| {
        let e = temb(g, w, t);
        (g.silu(e), n_target)
    });

    let mut img = lin(g, w, "img_in", img_in);
    let t = rms(g, w, "txt_norm.weight", txt_in);
    let mut txt = lin(g, w, "txt_in", t);
    for i in 0..cfg.num_layers {
        (img, txt) = block(g, w, &format!("transformer_blocks.{i}"), c, (img, txt), (te_act, te_ref), (cos, sin));
    }
    let m = lin(g, w, "norm_out.linear", te_act);
    let scale = g.view_1d(m, c.d, 0);
    let shift = g.view_1d(m, c.d, c.d as usize);
    let scale = g.scale_bias(scale, 1.0, 1.0);
    let h = g.norm(img, EPS);
    let h = g.mul(h, scale);
    let h = g.add(h, shift);
    let out = lin(g, w, "proj_out", h);
    Io { img: img_in, txt: txt_in, time, time_ref, cos, sin, out }
}

/// A loaded Qwen-Image transformer.
pub struct QwenImageTransformer {
    backend: Backend,
    cfg: QwenImageConfig,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for QwenImageTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImageTransformer").field("device", &self.backend.name()).field("layers", &self.cfg.num_layers).finish_non_exhaustive()
    }
}

impl QwenImageTransformer {
    /// Load `transformer/` of a checkpoint.
    ///
    /// # Errors
    /// A missing or malformed config or weight, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: QwenImageConfig = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "image transformer backend selected");
        let st = SafeTensors::open(&files.weights("transformer")?)?;
        let w = Weights::load(&backend, &st, &cfg.weight_specs(opts.precision.wtype()))?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32 })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &QwenImageConfig {
        &self.cfg
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Velocity for packed image tokens `[tokens][in_channels]` (the target
    /// image, then any reference images; `images` gives each one's
    /// `(rows, cols)` in patches) given text states `[tokens][joint width]`
    /// at timestep `t` in `[0, 1]`. Returns `[target tokens][out width]`
    /// followed by the outputs of the reference tokens.
    ///
    /// # Errors
    /// Inputs that disagree with the shapes, or a backend failure.
    pub fn forward(&self, img: &[f32], text: &[f32], images: &[(usize, usize)], t: f32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let n_img: usize = images.iter().map(|(h, w)| h * w).sum();
        let n_target = images.first().map_or(0, |(h, w)| h * w);
        let n_txt = text.len() / cfg.joint_attention_dim as usize;
        if n_img == 0 || n_txt == 0 || img.len() != n_img * cfg.in_channels as usize || text.len() != n_txt * cfg.joint_attention_dim as usize {
            return Err(Error::Request("image transformer inputs disagree with the shapes".into()));
        }
        let mut g = Graph::new(&self.backend)?;
        let io = build(&mut g, cfg, &self.w, (n_img as i64, n_target as i64, n_txt as i64), self.exact);
        g.finish(&[io.out])?;
        let (cos, sin) = cfg.rotary_tables(n_txt, images);
        g.set_f32(io.img, img);
        g.set_f32(io.txt, text);
        g.set_f32(io.time, &S3DitConfig::time_features(t * 1000.0));
        if let Some(r) = io.time_ref {
            g.set_f32(r, &S3DitConfig::time_features(0.0));
        }
        g.set_f32(io.cos, &cos);
        g.set_f32(io.sin, &sin);
        g.compute()?;
        Ok(g.read_f32(io.out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QwenImageConfig {
        serde_json::from_value(serde_json::json!({
            "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56], "guidance_embeds": false,
            "in_channels": 64, "joint_attention_dim": 3584, "num_attention_heads": 24,
            "num_layers": 60, "out_channels": 16, "patch_size": 2, "zero_cond_t": true
        }))
        .unwrap()
    }

    #[test]
    fn the_released_edit_layout_validates() {
        let c = cfg();
        c.validate().unwrap();
        assert_eq!(c.inner(), 3072);
        assert_eq!(c.out_dim(), 64);
    }

    #[test]
    fn text_sits_on_the_diagonal_past_the_largest_half_extent() {
        let c = cfg();
        let (cos, sin) = c.rotary_tables(2, &[(4, 6)]);
        assert_eq!(cos.len(), (2 + 24) * 128);
        // Text token 0 sits at 3 on every axis: the first frame pair turns by 3.
        assert!((sin[0] - 3f32.sin()).abs() < 1e-6);
        // The first image token sits at (0, -2, -3): no frame turn, row turns by -2.
        let first = 2 * 128;
        assert_eq!(sin[first], 0.0);
        assert!((sin[first + 8] - (-2f32).sin()).abs() < 1e-6);
        assert!((sin[first + 8 + 28] - (-3f32).sin()).abs() < 1e-6);
    }
}
