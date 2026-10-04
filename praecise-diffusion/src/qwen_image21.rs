//! Qwen-Image 2.1 transformer: single-stream blocks over one joint sequence
//! of prompt tokens and image latents, with block-causal attention.
//!
//! The joint sequence follows the prompt: text runs and condition image
//! blocks in prompt order, then the target image. Text attends causally,
//! every image block also sees all of its own tokens. Text and condition
//! tokens are modulated with timestep zero, and nothing before the target
//! attends to it, so their keys and values do not change across denoising
//! steps: [`QwenImage21Transformer::prefill`] computes them once and every
//! step runs only the target tokens against them.
//!
//! Modulation is one shared projection of the timestep embedding (scale and
//! tanh gate for attention and feed-forward, no shift); queries and keys
//! carry per-head RMS norms and three-axis rotary positions (a frame axis
//! advanced by text tokens, centred row and column axes inside images).

#[cfg(test)]
mod parity;
pub mod pipeline;
pub mod vae;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::s3dit::{S3DitConfig, TIME_FEATURES};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const ROPE_THETA: f32 = 10000.0;

/// `transformer/config.json` of a Qwen-Image 2.1 checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenImage21Config {
    pub patch_size: u64,
    pub in_channels: u64,
    #[serde(default)]
    pub out_channels: Option<u64>,
    pub num_layers: usize,
    pub attention_head_dim: u64,
    pub num_attention_heads: u64,
    pub context_in_dim: u64,
    pub mlp_ratio: u64,
    pub axes_dims_rope: [u64; 3],
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_true")]
    pub causal_condition: bool,
}

fn default_eps() -> f64 {
    1e-6
}

fn default_true() -> bool {
    true
}

/// One run of the joint sequence: prompt tokens, or an image block of
/// `rows x cols` latent tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    Text(usize),
    Image { rows: usize, cols: usize },
}

impl Segment {
    fn len(self) -> usize {
        match self {
            Self::Text(n) => n,
            Self::Image { rows, cols } => rows * cols,
        }
    }
}

impl QwenImage21Config {
    /// Refuse layouts this implementation does not compute.
    ///
    /// # Errors
    /// [`Error::Config`] naming the unsupported field.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("image transformer: {m}")));
        if self.patch_size != 1 {
            return bad("only single-token patches are supported");
        }
        if !self.causal_condition {
            return bad("only timestep-zero condition tokens are supported");
        }
        if self.axes_dims_rope.iter().sum::<u64>() != self.attention_head_dim || self.axes_dims_rope.iter().any(|d| d % 2 != 0) {
            return bad("rotary axes must be even and fill the head");
        }
        if self.num_layers == 0 || self.num_attention_heads == 0 || self.mlp_ratio == 0 {
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
        self.out_channels.unwrap_or(self.in_channels)
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

    /// Every transformer weight.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, hd, c) = (self.inner(), self.attention_head_dim, self.context_in_dim);
        let m = d * self.mlp_ratio;
        let w = |name: &str, out: u64, inp: u64, ty: WType| WeightSpec::new(format!("{name}.weight"), &[out, inp], ty);
        let mut v = vec![
            w("img_in", d, self.in_channels, WType::F32),
            WeightSpec::new("txt_in.text_norm.weight", &[c], WType::F32),
            w("txt_in.in_layer", d, c, linear),
            w("txt_in.out_layer", d, d, linear),
            w("time_text_embed.timestep_embedder.linear_1", d, TIME_FEATURES as u64, WType::F32),
            w("time_text_embed.timestep_embedder.linear_2", d, d, WType::F32),
            w("modulation.1", 4 * d, d, WType::F32),
            w("norm_out.linear", d, d, WType::F32),
            w("proj_out", self.out_dim(), d, WType::F32),
        ];
        let heads = self.num_attention_heads;
        for i in 0..self.num_layers {
            let p = format!("transformer_blocks.{i}");
            for q in ["to_q", "to_k"] {
                v.push(w(&format!("{p}.attn.{q}"), d, d, linear).with_rows(self.pair_rows(heads)));
            }
            v.push(w(&format!("{p}.attn.to_v"), d, d, linear));
            v.push(w(&format!("{p}.attn.to_out.0"), d, d, linear));
            for n in ["norm_q", "norm_k"] {
                v.push(WeightSpec::new(format!("{p}.attn.{n}.weight"), &[hd], WType::F32).with_rows(self.pair_rows(1)));
            }
            v.push(w(&format!("{p}.img_mlp.gate_layer"), m, d, linear));
            v.push(w(&format!("{p}.img_mlp.proj"), m, d, linear));
            v.push(w(&format!("{p}.img_mlp.out"), d, m, linear));
        }
        v
    }

    /// Positions `(frame, row, column)` of every token of `layout`: text
    /// advances the frame axis on all three axes, an image block sits at the
    /// current frame on a grid centred on zero and then advances the frame
    /// axis by its larger side.
    #[must_use]
    pub fn positions(layout: &[Segment]) -> Vec<[i64; 3]> {
        let mut out = Vec::new();
        let mut pos = 0i64;
        for &s in layout {
            match s {
                Segment::Text(n) => {
                    out.extend((0..n as i64).map(|j| [pos + j; 3]));
                    pos += n as i64;
                }
                Segment::Image { rows, cols } => {
                    let (h0, w0) = ((rows - rows / 2) as i64, (cols - cols / 2) as i64);
                    for r in 0..rows as i64 {
                        for c in 0..cols as i64 {
                            out.push([pos, r - h0, c - w0]);
                        }
                    }
                    pos += rows.max(cols) as i64;
                }
            }
        }
        out
    }

    /// Rotary cos and sin tables `[tokens][head width]` for `pos`, in the
    /// split-half order the permuted query and key rows use.
    #[must_use]
    pub fn rotary_tables(&self, pos: &[[i64; 3]]) -> (Vec<f32>, Vec<f32>) {
        let mut freqs = Vec::new();
        for (a, &d) in self.axes_dims_rope.iter().enumerate() {
            for j in 0..d / 2 {
                let e = (2 * j) as f32 / d as f32;
                freqs.push((a, 1.0f32 / ROPE_THETA.powf(e)));
            }
        }
        let hd = self.attention_head_dim as usize;
        let half = hd / 2;
        let mut cos = Vec::with_capacity(pos.len() * hd);
        let mut sin = Vec::with_capacity(pos.len() * hd);
        for p in pos {
            let ang: Vec<f32> = freqs.iter().map(|&(a, f)| p[a] as f32 * f).collect();
            for j in 0..hd {
                cos.push(ang[j % half].cos());
                sin.push(ang[j % half].sin());
            }
        }
        (cos, sin)
    }

    /// Additive block-causal mask `[query][key]` over `layout`: a token sees
    /// every token before it and every token of its own image block.
    #[must_use]
    pub fn block_causal_mask(layout: &[Segment]) -> Vec<f32> {
        let mut block = Vec::new();
        for (i, &s) in layout.iter().enumerate() {
            let id = matches!(s, Segment::Image { .. }).then_some(i);
            block.extend(std::iter::repeat_n(id, s.len()));
        }
        let n = block.len();
        let mut m = vec![0f32; n * n];
        for q in 0..n {
            for k in q + 1..n {
                if block[q].is_none() || block[q] != block[k] {
                    m[q * n + k] = f32::NEG_INFINITY;
                }
            }
        }
        m
    }
}

#[derive(Clone, Copy)]
struct Ctx {
    d: i64,
    hd: i64,
    heads: i64,
    eps: f32,
    exact: bool,
}

/// Shared modulation: attention scale and gate, feed-forward scale and gate.
#[derive(Clone, Copy)]
struct Mod {
    scale1: Tn,
    gate1: Tn,
    scale2: Tn,
    gate2: Tn,
}

fn lin(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear(w.get(&format!("{name}.weight")), x)
}

/// Timestep embedding for features `time`, already through SiLU.
fn temb(g: &mut Graph, w: &Weights, time: Tn) -> Tn {
    let t = lin(g, w, "time_text_embed.timestep_embedder.linear_1", time);
    let t = g.silu(t);
    let t = lin(g, w, "time_text_embed.timestep_embedder.linear_2", t);
    g.silu(t)
}

fn modulation(g: &mut Graph, w: &Weights, te: Tn, d: i64) -> Mod {
    let m = lin(g, w, "modulation.1", te);
    let mut part = |i: i64| g.view_1d(m, d, (i * d) as usize);
    let v: Vec<Tn> = (0..4).map(&mut part).collect();
    Mod {
        scale1: g.scale_bias(v[0], 1.0, 1.0),
        gate1: g.tanh(v[1]),
        scale2: g.scale_bias(v[2], 1.0, 1.0),
        gate2: g.tanh(v[3]),
    }
}

/// Prompt states `[context width, n]` to the model width.
fn txt_in(g: &mut Graph, w: &Weights, x: Tn, c: Ctx) -> Tn {
    let h = g.rms_norm(x, c.eps);
    let s = g.mul(h, w.get("txt_in.text_norm.weight"));
    let h = g.add(s, h);
    let h = lin(g, w, "txt_in.in_layer", h);
    let h = if c.exact { g.gelu_tanh_exact(h) } else { g.gelu_tanh(h) };
    lin(g, w, "txt_in.out_layer", h)
}

/// Normed, rotated queries and keys and values `[head width, heads, n]`.
fn qkv(g: &mut Graph, w: &Weights, a: &str, c: Ctx, x: Tn, (cos, sin): (Tn, Tn)) -> (Tn, Tn, Tn) {
    let n = x.ne(1);
    let mut out = [x; 3];
    for (o, name) in out.iter_mut().zip(["to_q", "to_k", "to_v"]) {
        let t = lin(g, w, &format!("{a}.{name}"), x);
        *o = g.reshape(t, &[c.hd, c.heads, n]);
    }
    let [q, k, v] = out;
    let q = g.rms_norm(q, c.eps);
    let q = g.mul(q, w.get(&format!("{a}.norm_q.weight")));
    let k = g.rms_norm(k, c.eps);
    let k = g.mul(k, w.get(&format!("{a}.norm_k.weight")));
    let q = g.rotate_half_rope(q, cos, sin);
    let k = g.rotate_half_rope(k, cos, sin);
    (q, k, v)
}

fn attend(g: &mut Graph, c: Ctx, (q, k, v): (Tn, Tn, Tn), mask: Option<Tn>) -> Tn {
    let n = q.ne(2);
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    let scale = 1.0 / (c.hd as f32).sqrt();
    let o = if c.exact {
        let k = g.cont(k);
        let v = g.cont(v);
        g.attention_exact(q, k, v, mask, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, mask, scale, true)
    };
    g.reshape(o, &[c.d, n])
}

/// The rest of a block once attention has produced `o`.
fn finish_block(g: &mut Graph, w: &Weights, p: &str, c: Ctx, x: Tn, o: Tn, m: Mod) -> Tn {
    let o = lin(g, w, &format!("{p}.attn.to_out.0"), o);
    let o = g.mul(o, m.gate1);
    let x = g.add(x, o);
    let h = g.norm(x, c.eps);
    let h = g.mul(h, m.scale2);
    let gate = lin(g, w, &format!("{p}.img_mlp.gate_layer"), h);
    let up = lin(g, w, &format!("{p}.img_mlp.proj"), h);
    let f = g.swiglu_split(gate, up);
    let f = lin(g, w, &format!("{p}.img_mlp.out"), f);
    let f = g.mul(f, m.gate2);
    g.add(x, f)
}

fn modulated(g: &mut Graph, c: Ctx, x: Tn, m: Mod) -> Tn {
    let h = g.norm(x, c.eps);
    g.mul(h, m.scale1)
}

/// Resident keys and values of every layer for the tokens before the target.
pub struct QwenImage21Prefix {
    cache: Weights,
    target: (usize, usize),
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl std::fmt::Debug for QwenImage21Prefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImage21Prefix").field("target", &self.target).finish_non_exhaustive()
    }
}

impl QwenImage21Prefix {
    /// `(rows, cols)` of the target image.
    #[must_use]
    pub fn target(&self) -> (usize, usize) {
        self.target
    }

    /// Bytes held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.cache.bytes()
    }
}

/// A loaded Qwen-Image 2.1 transformer.
pub struct QwenImage21Transformer {
    backend: Backend,
    cfg: QwenImage21Config,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for QwenImage21Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImage21Transformer").field("device", &self.backend.name()).field("layers", &self.cfg.num_layers).finish_non_exhaustive()
    }
}

impl QwenImage21Transformer {
    /// Load `transformer/` of a checkpoint.
    ///
    /// # Errors
    /// A missing or malformed config or weight, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: QwenImage21Config = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "image transformer backend selected");
        let st = SafeTensors::open(&files.weights("transformer")?)?;
        let w = Weights::load(&backend, &st, &cfg.weight_specs(opts.precision.wtype()))?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32 })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &QwenImage21Config {
        &self.cfg
    }

    /// The backend the weights live on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes()
    }

    fn ctx(&self) -> Ctx {
        let c = &self.cfg;
        Ctx { d: c.inner() as i64, hd: c.attention_head_dim as i64, heads: c.num_attention_heads as i64, eps: c.eps as f32, exact: self.exact }
    }

    /// Keys and values of the tokens before the target. `layout` is the
    /// whole joint sequence and ends with the target image; `text` holds the
    /// prompt states `[text tokens][context width]` of every text run in
    /// order, `condition` the latents `[tokens][in channels]` of every image
    /// block before the target in order.
    ///
    /// # Errors
    /// Inputs that disagree with the layout, or a backend failure.
    pub fn prefill(&self, text: &[f32], condition: &[f32], layout: &[Segment]) -> Result<QwenImage21Prefix> {
        let cfg = &self.cfg;
        let Some((&Segment::Image { rows, cols }, prefix)) = layout.split_last() else {
            return Err(Error::Request("the joint sequence must end with the target image".into()));
        };
        let n_txt: usize = prefix.iter().filter(|s| matches!(s, Segment::Text(_))).map(|s| s.len()).sum();
        let n_cond: usize = prefix.iter().filter(|s| matches!(s, Segment::Image { .. })).map(|s| s.len()).sum();
        let n = n_txt + n_cond;
        let (cw, ic) = (cfg.context_in_dim as usize, cfg.in_channels as usize);
        if n_txt == 0 || rows * cols == 0 || text.len() != n_txt * cw || condition.len() != n_cond * ic {
            return Err(Error::Request("image transformer inputs disagree with the layout".into()));
        }
        let c = self.ctx();
        let shape = [n as u64, c.heads as u64, c.hd as u64];
        let ty = if self.exact { WType::F32 } else { WType::F16 };
        let specs: Vec<WeightSpec> = (0..cfg.num_layers)
            .flat_map(|i| [WeightSpec::new(format!("k.{i}"), &shape, ty), WeightSpec::new(format!("v.{i}"), &shape, ty)])
            .collect();
        let cache = Weights::zeros(&self.backend, &specs)?;

        let mut g = Graph::new(&self.backend)?;
        let txt = g.input(sys::GGML_TYPE_F32, &[cw as i64, n_txt as i64]);
        let cond = (n_cond > 0).then(|| g.input(sys::GGML_TYPE_F32, &[ic as i64, n_cond as i64]));
        let time = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
        let cos = g.input(sys::GGML_TYPE_F32, &[c.hd, 1, n as i64]);
        let sin = g.input(sys::GGML_TYPE_F32, &[c.hd, 1, n as i64]);
        let mask = g.input(sys::GGML_TYPE_F16, &[n as i64, n as i64]);
        let te = temb(&mut g, &self.w, time);
        let m = modulation(&mut g, &self.w, te, c.d);
        let xt = txt_in(&mut g, &self.w, txt, c);
        let xi = cond.map(|x| lin(&mut g, &self.w, "img_in", x));
        let (mut at_t, mut at_i, mut x) = (0i64, 0i64, None);
        for &s in prefix {
            let len = s.len() as i64;
            let part = match s {
                Segment::Text(_) => {
                    at_t += len;
                    g.view_cols(xt, at_t - len, len)
                }
                Segment::Image { .. } => {
                    at_i += len;
                    g.view_cols(xi.expect("condition tokens present"), at_i - len, len)
                }
            };
            x = Some(match x {
                None => g.cont(part),
                Some(x) => g.concat(x, part, 1),
            });
        }
        let mut x = x.expect("text present");
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            let h = modulated(&mut g, c, x, m);
            let (q, k, v) = qkv(&mut g, &self.w, &format!("{p}.attn"), c, h, (cos, sin));
            g.copy_into(k, cache.get(&format!("k.{i}")));
            g.copy_into(v, cache.get(&format!("v.{i}")));
            if i + 1 < cfg.num_layers {
                let o = attend(&mut g, c, (q, k, v), Some(mask));
                x = finish_block(&mut g, &self.w, &p, c, x, o, m);
            }
        }
        g.finish(&[])?;
        let pos = QwenImage21Config::positions(layout);
        let (cos_t, sin_t) = cfg.rotary_tables(&pos);
        let hd = c.hd as usize;
        g.set_f32(txt, text);
        if let Some(t) = cond {
            g.set_f32(t, condition);
        }
        g.set_f32(time, &S3DitConfig::time_features(0.0));
        g.set_f32(cos, &cos_t[..n * hd]);
        g.set_f32(sin, &sin_t[..n * hd]);
        g.set_f16(mask, &QwenImage21Config::block_causal_mask(prefix));
        g.compute()?;
        Ok(QwenImage21Prefix { cache, target: (rows, cols), cos: cos_t[n * hd..].to_vec(), sin: sin_t[n * hd..].to_vec() })
    }

    /// Velocity `[target tokens][out channels]` for the target latents
    /// `[target tokens][in channels]` at timestep `t` in `[0, 1]`.
    ///
    /// # Errors
    /// Latents that disagree with the prefix's target, or a backend failure.
    pub fn forward(&self, prefix: &QwenImage21Prefix, target: &[f32], t: f32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let nt = prefix.target.0 * prefix.target.1;
        if target.len() != nt * cfg.in_channels as usize {
            return Err(Error::Request("target latents disagree with the prefix".into()));
        }
        let c = self.ctx();
        let mut g = Graph::new(&self.backend)?;
        let img = g.input(sys::GGML_TYPE_F32, &[cfg.in_channels as i64, nt as i64]);
        let time = g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]);
        let cos = g.input(sys::GGML_TYPE_F32, &[c.hd, 1, nt as i64]);
        let sin = g.input(sys::GGML_TYPE_F32, &[c.hd, 1, nt as i64]);
        let te = temb(&mut g, &self.w, time);
        let m = modulation(&mut g, &self.w, te, c.d);
        let mut x = lin(&mut g, &self.w, "img_in", img);
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            let h = modulated(&mut g, c, x, m);
            let (q, k, v) = qkv(&mut g, &self.w, &format!("{p}.attn"), c, h, (cos, sin));
            let (ck, cv) = (prefix.cache.get(&format!("k.{i}")), prefix.cache.get(&format!("v.{i}")));
            let (k, v) = if self.exact {
                (g.concat(ck, k, 2), g.concat(cv, v, 2))
            } else {
                let k = g.cast(k, sys::GGML_TYPE_F16);
                let v = g.cast(v, sys::GGML_TYPE_F16);
                (g.concat(ck, k, 2), g.concat(cv, v, 2))
            };
            let o = attend(&mut g, c, (q, k, v), None);
            x = finish_block(&mut g, &self.w, &p, c, x, o, m);
        }
        let s = lin(&mut g, &self.w, "norm_out.linear", te);
        let s = g.scale_bias(s, 1.0, 1.0);
        let h = g.norm(x, c.eps);
        let h = g.mul(h, s);
        let out = lin(&mut g, &self.w, "proj_out", h);
        g.finish(&[out])?;
        g.set_f32(img, target);
        g.set_f32(time, &S3DitConfig::time_features(t * 1000.0));
        g.set_f32(cos, &prefix.cos);
        g.set_f32(sin, &prefix.sin);
        g.compute()?;
        Ok(g.read_f32(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_blocks_sit_on_a_centred_grid_at_the_text_frame() {
        let pos = QwenImage21Config::positions(&[Segment::Text(2), Segment::Image { rows: 2, cols: 3 }, Segment::Text(1)]);
        assert_eq!(pos[..2], [[0, 0, 0], [1, 1, 1]]);
        assert_eq!(pos[2], [2, -1, -2]);
        assert_eq!(pos[7], [2, 0, 0]);
        assert_eq!(pos[8], [5, 5, 5], "text resumes past the larger side");
    }

    #[test]
    fn text_is_causal_and_image_blocks_see_themselves() {
        let layout = [Segment::Text(2), Segment::Image { rows: 1, cols: 2 }, Segment::Text(1)];
        let m = QwenImage21Config::block_causal_mask(&layout);
        let allowed = |q: usize, k: usize| m[q * 5 + k] == 0.0;
        assert!(allowed(1, 0) && !allowed(0, 1));
        assert!(allowed(2, 3) && allowed(3, 2), "an image block is bidirectional");
        assert!(!allowed(2, 4) && allowed(4, 3));
    }
}
