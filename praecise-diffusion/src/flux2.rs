//! The FLUX.2 rectified-flow transformer.
//!
//! Double-stream blocks keep text and image tokens in separate weights and
//! attend jointly; single-stream blocks run one fused projection over the
//! concatenated sequence. Modulation (shift, scale, gate) is computed once per
//! step from the timestep and shared by every block of a kind.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use llama_cpp_sys_2 as sys;

/// Transformer configuration, read from the model's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Flux2Config {
    /// Latent channels after 2x2 patching.
    pub in_channels: u64,
    /// Double-stream block count.
    pub num_layers: usize,
    /// Single-stream block count.
    pub num_single_layers: usize,
    /// Attention head width.
    pub attention_head_dim: u64,
    /// Attention head count.
    pub num_attention_heads: u64,
    /// Width of the text conditioning.
    pub joint_attention_dim: u64,
    /// Width of the sinusoidal timestep projection.
    pub timestep_guidance_channels: u64,
    /// MLP expansion ratio.
    pub mlp_ratio: f64,
    /// Rotary widths per position axis.
    pub axes_dims_rope: Vec<u64>,
    /// Rotary base.
    pub rope_theta: f64,
    /// Norm epsilon.
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// Whether the model embeds a guidance scale.
    #[serde(default)]
    pub guidance_embeds: bool,
    /// Patch size inside the transformer.
    #[serde(default = "one")]
    pub patch_size: u64,
}

fn default_eps() -> f64 {
    1e-6
}
fn one() -> u64 {
    1
}

impl Flux2Config {
    /// Validate the parts this implementation depends on.
    ///
    /// # Errors
    /// [`Error::Config`] for a configuration this engine does not implement.
    pub fn validate(&self) -> Result<()> {
        if self.guidance_embeds {
            return Err(Error::Config("guidance-embedding checkpoints are not implemented".into()));
        }
        if self.patch_size != 1 {
            return Err(Error::Config(format!("transformer patch size {} is not implemented", self.patch_size)));
        }
        if self.axes_dims_rope.iter().sum::<u64>() != self.attention_head_dim {
            return Err(Error::Config("rotary axes do not sum to the head width".into()));
        }
        if self.axes_dims_rope.len() != 4 || self.axes_dims_rope.iter().any(|d| d % 2 != 0) {
            return Err(Error::Config("expected four even rotary axes".into()));
        }
        Ok(())
    }

    /// Model width.
    #[must_use]
    pub fn dim(&self) -> u64 {
        self.attention_head_dim * self.num_attention_heads
    }

    /// MLP hidden width.
    #[must_use]
    pub fn mlp_hidden(&self) -> u64 {
        (self.dim() as f64 * self.mlp_ratio) as u64
    }

    /// Every weight the transformer needs, with the storage type for `linear`
    /// weights (norm vectors stay f32).
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let d = self.dim();
        let h = self.mlp_hidden();
        let hd = self.attention_head_dim;
        let mut v = vec![
            WeightSpec::new("x_embedder.weight", &[d, self.in_channels], linear),
            WeightSpec::new("context_embedder.weight", &[d, self.joint_attention_dim], linear),
            WeightSpec::new(
                "time_guidance_embed.timestep_embedder.linear_1.weight",
                &[d, self.timestep_guidance_channels],
                WType::F32,
            ),
            WeightSpec::new("time_guidance_embed.timestep_embedder.linear_2.weight", &[d, d], WType::F32),
            WeightSpec::new("double_stream_modulation_img.linear.weight", &[6 * d, d], WType::F32),
            WeightSpec::new("double_stream_modulation_txt.linear.weight", &[6 * d, d], WType::F32),
            WeightSpec::new("single_stream_modulation.linear.weight", &[3 * d, d], WType::F32),
            WeightSpec::new("norm_out.linear.weight", &[2 * d, d], WType::F32),
            WeightSpec::new("proj_out.weight", &[self.in_channels, d], linear),
        ];
        let head = rope_head_order(hd as usize);
        let heads = self.num_attention_heads as usize;
        let qk = |offset: usize, total: usize| -> Vec<usize> {
            let mut rows: Vec<usize> = (0..total).collect();
            for h in 0..heads {
                for (new, old) in head.iter().enumerate() {
                    rows[offset + h * head.len() + new] = offset + h * head.len() + old;
                }
            }
            rows
        };
        for i in 0..self.num_layers {
            let p = format!("transformer_blocks.{i}");
            // Query, key and value projections of one stream stacked into one
            // matrix; query and key heads in rotary order.
            let du = d as usize;
            let mut rows = qk(0, 3 * du);
            rows[du..2 * du].copy_from_slice(&qk(du, 3 * du)[du..2 * du]);
            for (fused, parts) in [("qkv", ["to_q", "to_k", "to_v"]), ("add_qkv", ["add_q_proj", "add_k_proj", "add_v_proj"])] {
                let parts: Vec<String> = parts.iter().map(|n| format!("{p}.attn.{n}.weight")).collect();
                v.push(WeightSpec::stacked(format!("{p}.attn.{fused}"), &parts, &[d, d], linear).with_rows(rows.clone()));
            }
            for n in ["to_add_out", "to_out.0"] {
                v.push(WeightSpec::new(format!("{p}.attn.{n}.weight"), &[d, d], linear));
            }
            for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
                v.push(WeightSpec::new(format!("{p}.attn.{n}.weight"), &[hd], WType::F32).with_rows(head.clone()));
            }
            for ff in ["ff", "ff_context"] {
                v.push(WeightSpec::new(format!("{p}.{ff}.linear_in.weight"), &[2 * h, d], linear));
                v.push(WeightSpec::new(format!("{p}.{ff}.linear_out.weight"), &[d, h], linear));
            }
        }
        for i in 0..self.num_single_layers {
            let p = format!("single_transformer_blocks.{i}");
            let total = (3 * d + 2 * h) as usize;
            let mut rows = qk(0, total);
            rows[d as usize..2 * d as usize].copy_from_slice(&qk(d as usize, total)[d as usize..2 * d as usize]);
            v.push(WeightSpec::new(format!("{p}.attn.to_qkv_mlp_proj.weight"), &[3 * d + 2 * h, d], linear).with_rows(rows));
            v.push(WeightSpec::new(format!("{p}.attn.to_out.weight"), &[d, d + h], linear));
            v.push(WeightSpec::new(format!("{p}.attn.norm_q.weight"), &[hd], WType::F32).with_rows(head.clone()));
            v.push(WeightSpec::new(format!("{p}.attn.norm_k.weight"), &[hd], WType::F32).with_rows(head.clone()));
        }
        v
    }
}

/// Sinusoidal timestep features, cosine half first, max period 10000.
#[must_use]
pub fn timestep_features(t: f32, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let mut out = vec![0f32; channels];
    for j in 0..half {
        let freq = (-(10000f64.ln()) * j as f64 / half as f64).exp();
        let arg = f64::from(t) * freq;
        out[j] = arg.cos() as f32;
        out[j + half] = arg.sin() as f32;
    }
    out
}

/// Per-pair divisors that turn ggml's single rotary frequency ladder
/// (`base^(-2i / head_dim)` over all pairs) into one ladder per position
/// axis, each restarting at the axis's first pair.
#[must_use]
pub fn rope_freq_factors(cfg: &Flux2Config) -> Vec<f32> {
    axis_rope_freq_factors(&cfg.axes_dims_rope, cfg.attention_head_dim, cfg.rope_theta)
}

/// [`rope_freq_factors`] for any per-axis split `axes` of a `head_dim`-wide
/// head with base `theta`.
#[must_use]
pub fn axis_rope_freq_factors(axes: &[u64], head_dim: u64, theta: f64) -> Vec<f32> {
    let hd = head_dim as f64;
    let mut out = Vec::with_capacity((head_dim / 2) as usize);
    let mut i = 0usize;
    for &width in axes {
        for j in 0..(width / 2) as usize {
            let ladder = theta.powf(-2.0 * i as f64 / hd);
            let wanted = theta.powf(-2.0 * j as f64 / width as f64);
            out.push((ladder / wanted) as f32);
            i += 1;
        }
    }
    out
}

/// The head-dimension order that makes each interleaved rotary pair
/// `(2i, 2i + 1)` a split-half pair `(i, i + head_dim / 2)`: `order[new] =
/// old`. Applied to the query and key projections at load, it lets ggml's
/// multi-axis rotary kernel rotate the pairs the model was trained with.
/// Attention scores are unchanged, since queries and keys are reordered
/// alike.
#[must_use]
pub fn rope_head_order(head_dim: usize) -> Vec<usize> {
    let half = head_dim / 2;
    (0..head_dim).map(|new| if new < half { 2 * new } else { 2 * (new - half) + 1 }).collect()
}

/// Rotary positions for ggml's multi-axis rotary: all tokens' first axis,
/// then all tokens' second axis, and so on.
#[must_use]
pub fn rope_positions(positions: &[[f32; 4]]) -> Vec<i32> {
    let n = positions.len();
    let mut out = vec![0i32; 4 * n];
    for (tok, p) in positions.iter().enumerate() {
        for axis in 0..4 {
            out[axis * n + tok] = p[axis] as i32;
        }
    }
    out
}

/// Positions for text tokens `(0, 0, 0, l)` followed by each image grid's
/// tokens `(t, h, w, 0)` in row-major order. `grids` holds `(t, h, w)`: the
/// generated image at `t = 0`, reference images at their own `t`.
#[must_use]
pub fn positions(text_tokens: usize, grids: &[(f32, usize, usize)]) -> Vec<[f32; 4]> {
    let mut p = Vec::with_capacity(text_tokens + grids.iter().map(|g| g.1 * g.2).sum::<usize>());
    for l in 0..text_tokens {
        p.push([0.0, 0.0, 0.0, l as f32]);
    }
    for &(t, grid_h, grid_w) in grids {
        for h in 0..grid_h {
            for w in 0..grid_w {
                p.push([t, h as f32, w as f32, 0.0]);
            }
        }
    }
    p
}

/// Graph inputs and output of one transformer evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Flux2Io {
    /// Image tokens `[in_channels, n_img]`.
    pub img: Tn,
    /// Text conditioning `[joint_attention_dim, n_txt]`.
    pub txt: Tn,
    /// Timestep features `[timestep_guidance_channels]`.
    pub t_feat: Tn,
    /// Rotary positions `[4 * (n_txt + n_img)]` (see [`rope_positions`]).
    pub pos: Tn,
    /// Rotary frequency divisors `[head_dim / 2]` (see [`rope_freq_factors`]).
    pub freq_factors: Tn,
    /// Velocity prediction `[in_channels, n_img]`.
    pub out: Tn,
}

struct Mod {
    shift: Tn,
    scale1: Tn,
    gate: Tn,
}

fn modulation(g: &mut Graph, vec: Tn, set: usize, d: i64) -> Mod {
    let base = set * 3 * d as usize;
    let shift = g.view_1d(vec, d, base);
    let scale = g.view_1d(vec, d, base + d as usize);
    let gate = g.view_1d(vec, d, base + 2 * d as usize);
    let scale1 = g.scale_bias(scale, 1.0, 1.0);
    // Placed now, so each use reads norm, scale, shift as adjacent nodes the
    // backend can fuse.
    g.expand(scale1);
    g.expand(shift);
    g.expand(gate);
    Mod { shift, scale1, gate }
}

fn modulate(g: &mut Graph, x: Tn, m: &Mod, eps: f32) -> Tn {
    let n = g.norm(x, eps);
    let s = g.mul(n, m.scale1);
    g.add(s, m.shift)
}

/// Apply the four-axis rotary embedding to `x` `[hd, heads, n]`, whose head
/// dimensions are in [`rope_head_order`].
fn apply_rope(g: &mut Graph, x: Tn, r: &Rope) -> Tn {
    let hd = x.ne(0) as i32;
    g.rope_multi(x, r.pos, r.freq_factors, hd, r.sections, r.base)
}

struct Rope {
    pos: Tn,
    freq_factors: Tn,
    sections: [i32; 4],
    base: f32,
}

fn head_norm(g: &mut Graph, x: Tn, w: Tn, eps: f32) -> Tn {
    let n = g.rms_norm(x, eps);
    g.mul(n, w)
}

fn attend(g: &mut Graph, q: Tn, k: Tn, v: Tn, hd: i64) -> Tn {
    // [hd, heads, n] -> [hd, n, heads]
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    let k = g.cast(k, sys::GGML_TYPE_F16);
    let v = g.cast(v, sys::GGML_TYPE_F16);
    // Queries and keys are RMS-normalised per head, which bounds the scores.
    let o = g.attention(q, k, v, None, 1.0 / (hd as f32).sqrt(), false);
    // [hd, heads, n] -> [hd * heads, n]
    let heads = o.ne(1);
    let n = o.ne(2);
    g.reshape(o, &[hd * heads, n])
}

/// Build one transformer evaluation into `g`.
#[must_use]
pub fn build(g: &mut Graph, cfg: &Flux2Config, w: &Weights, n_txt: i64, n_img: i64) -> Flux2Io {
    let d = cfg.dim() as i64;
    let hd = cfg.attention_head_dim as i64;
    let heads = cfg.num_attention_heads as i64;
    let h = cfg.mlp_hidden() as i64;
    let eps = cfg.eps as f32;
    let n_all = n_txt + n_img;

    let img = g.input(sys::GGML_TYPE_F32, &[cfg.in_channels as i64, n_img]);
    let txt = g.input(sys::GGML_TYPE_F32, &[cfg.joint_attention_dim as i64, n_txt]);
    let t_feat = g.input(sys::GGML_TYPE_F32, &[cfg.timestep_guidance_channels as i64]);
    let pos = g.input(sys::GGML_TYPE_I32, &[4 * n_all]);
    let freq_factors = g.input(sys::GGML_TYPE_F32, &[hd / 2]);
    let mut sections = [0i32; 4];
    for (s, w) in sections.iter_mut().zip(&cfg.axes_dims_rope) {
        *s = (*w / 2) as i32;
    }
    let rope = Rope { pos, freq_factors, sections, base: cfg.rope_theta as f32 };

    let t1 = g.linear(w.get("time_guidance_embed.timestep_embedder.linear_1.weight"), t_feat);
    let t1 = g.silu(t1);
    let temb = g.linear(w.get("time_guidance_embed.timestep_embedder.linear_2.weight"), t1);
    let temb_act = g.silu(temb);
    let mod_img = g.linear(w.get("double_stream_modulation_img.linear.weight"), temb_act);
    let mod_txt = g.linear(w.get("double_stream_modulation_txt.linear.weight"), temb_act);
    let mod_single = g.linear(w.get("single_stream_modulation.linear.weight"), temb_act);
    let (img_msa, img_mlp) = (modulation(g, mod_img, 0, d), modulation(g, mod_img, 1, d));
    let (txt_msa, txt_mlp) = (modulation(g, mod_txt, 0, d), modulation(g, mod_txt, 1, d));
    let single = modulation(g, mod_single, 0, d);

    let mut x = g.linear(w.get("x_embedder.weight"), img);
    let mut c = g.linear(w.get("context_embedder.weight"), txt);

    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        let wn = |n: &str| w.get(&format!("{p}.{n}"));
        let xn = modulate(g, x, &img_msa, eps);
        let cn = modulate(g, c, &txt_msa, eps);

        let qkv = g.linear(wn("attn.qkv"), xn);
        let cqkv = g.linear(wn("attn.add_qkv"), cn);
        let q = g.view_heads(qkv, 0, hd, heads);
        let k = g.view_heads(qkv, d, hd, heads);
        let v = g.view_heads(qkv, 2 * d, hd, heads);
        let cq = g.view_heads(cqkv, 0, hd, heads);
        let ck = g.view_heads(cqkv, d, hd, heads);
        let cv = g.view_heads(cqkv, 2 * d, hd, heads);
        let q = head_norm(g, q, wn("attn.norm_q.weight"), eps);
        let k = head_norm(g, k, wn("attn.norm_k.weight"), eps);
        let cq = head_norm(g, cq, wn("attn.norm_added_q.weight"), eps);
        let ck = head_norm(g, ck, wn("attn.norm_added_k.weight"), eps);
        let q = g.concat(cq, q, 2);
        let k = g.concat(ck, k, 2);
        let v = g.concat(cv, v, 2);
        let q = apply_rope(g, q, &rope);
        let k = apply_rope(g, k, &rope);
        let attn = attend(g, q, k, v, hd);
        let attn_c = g.view_cols(attn, 0, n_txt);
        let attn_x = g.view_cols(attn, n_txt, n_img);

        let ox = g.linear(wn("attn.to_out.0.weight"), attn_x);
        let ox = g.mul(ox, img_msa.gate);
        x = g.add(x, ox);
        let oc = g.linear(wn("attn.to_add_out.weight"), attn_c);
        let oc = g.mul(oc, txt_msa.gate);
        c = g.add(c, oc);

        let xn = modulate(g, x, &img_mlp, eps);
        let f = g.linear(wn("ff.linear_in.weight"), xn);
        let f = g.swiglu(f);
        let f = g.linear(wn("ff.linear_out.weight"), f);
        let f = g.mul(f, img_mlp.gate);
        x = g.add(x, f);

        let cn = modulate(g, c, &txt_mlp, eps);
        let f = g.linear(wn("ff_context.linear_in.weight"), cn);
        let f = g.swiglu(f);
        let f = g.linear(wn("ff_context.linear_out.weight"), f);
        let f = g.mul(f, txt_mlp.gate);
        c = g.add(c, f);
    }

    let mut s = g.concat(c, x, 1);
    for i in 0..cfg.num_single_layers {
        let p = format!("single_transformer_blocks.{i}");
        let wn = |n: &str| w.get(&format!("{p}.{n}"));
        let sn = modulate(g, s, &single, eps);
        let proj = g.linear(wn("attn.to_qkv_mlp_proj.weight"), sn);
        let q = g.view_heads(proj, 0, hd, heads);
        let k = g.view_heads(proj, d, hd, heads);
        let v = g.view_heads(proj, 2 * d, hd, heads);
        let mlp = g.view_rows(proj, 3 * d, 2 * h);
        let q = head_norm(g, q, wn("attn.norm_q.weight"), eps);
        let k = head_norm(g, k, wn("attn.norm_k.weight"), eps);
        let q = apply_rope(g, q, &rope);
        let k = apply_rope(g, k, &rope);
        let attn = attend(g, q, k, v, hd);
        let mlp = g.cont(mlp);
        let mlp = g.swiglu(mlp);
        let cat = g.concat(attn, mlp, 0);
        let o = g.linear(wn("attn.to_out.weight"), cat);
        let o = g.mul(o, single.gate);
        s = g.add(s, o);
    }

    let x = g.view_cols(s, n_txt, n_img);
    let x = g.cont(x);
    let emb = g.linear(w.get("norm_out.linear.weight"), temb_act);
    let scale = g.view_1d(emb, d, 0);
    let shift = g.view_1d(emb, d, d as usize);
    let scale1 = g.scale_bias(scale, 1.0, 1.0);
    g.expand(scale1);
    g.expand(shift);
    let xn = g.norm(x, eps);
    let xn = g.mul(xn, scale1);
    let xn = g.add(xn, shift);
    let out = g.linear(w.get("proj_out.weight"), xn);
    Flux2Io { img, txt, t_feat, pos, freq_factors, out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Flux2Config {
        serde_json::from_value(serde_json::json!({
            "in_channels": 128, "num_layers": 1, "num_single_layers": 1, "attention_head_dim": 128,
            "num_attention_heads": 1, "joint_attention_dim": 64, "timestep_guidance_channels": 256,
            "mlp_ratio": 3.0, "axes_dims_rope": [32, 32, 32, 32], "rope_theta": 2000.0
        }))
        .unwrap()
    }

    #[test]
    fn the_head_order_puts_each_interleaved_pair_half_a_head_apart() {
        let o = rope_head_order(8);
        assert_eq!(o, vec![0, 2, 4, 6, 1, 3, 5, 7]);
        // pair i of the model (2i, 2i+1) sits at (i, i + 4)
        for i in 0..4 {
            assert_eq!((o[i], o[i + 4]), (2 * i, 2 * i + 1));
        }
    }

    #[test]
    fn frequency_divisors_restart_the_ladder_on_every_axis() {
        let c = cfg();
        let ff = rope_freq_factors(&c);
        assert_eq!(ff.len(), 64);
        for (i, f) in ff.iter().enumerate() {
            let (axis_pair, width) = (i % 16, 32.0);
            let ladder = 2000f64.powf(-2.0 * i as f64 / 128.0);
            let wanted = 2000f64.powf(-2.0 * axis_pair as f64 / width);
            assert!(((ladder / f64::from(*f)) / wanted - 1.0).abs() < 1e-5, "pair {i}");
        }
    }

    #[test]
    fn positions_are_laid_out_axis_by_axis() {
        let p = rope_positions(&positions(2, &[(10.0, 1, 2)]));
        // tokens: text 0, text 1, image (0,0), image (0,1)
        assert_eq!(p, vec![0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]);
    }
}
