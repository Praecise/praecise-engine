//! Action-conditioned world model on the Wan2.2 transformer.
//!
//! The backbone is the Wan video transformer (see [`crate::wan_dit`]) with
//! four changes: every head has its own rotary base (spread linearly around
//! 10000), the cross-attention norm output replaces the residual stream
//! before cross-attention, an optional camera embedding (Plucker rays,
//! patchified like the latent) scales and shifts the stream after
//! self-attention, and the first blocks carry an action module after
//! cross-attention.
//!
//! The action module reads two streams, one row per pixel frame: a mouse
//! vector and a keyboard vector. Each latent frame sees a window of the
//! `windows_size * vae_time_compression_ratio` rows ending at its own pixel
//! frames (the first row repeated before the start). Mouse windows are
//! concatenated with each spatial token and attended over time per spatial
//! position; keyboard windows are embedded and attended from every token.
//! Both use a rotary embedding over the latent frame index.
//!
//! Host layouts: a window block is `[window][dims]` per latent frame, frames
//! in order; the camera input is `[channel][dy][dx]` per token like the
//! latent patches.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use crate::wan_dit::{attend_batched, rope_pairs, WanDitConfig};
use llama_cpp_sys_2 as sys;

pub mod camera;
pub mod model;

pub use model::MatrixGame;

/// Action module settings, as in the checkpoint configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ActionConfig {
    /// Blocks that carry the module.
    pub blocks: Vec<u64>,
    /// Keyboard values per row.
    pub keyboard_dim_in: u64,
    /// Mouse values per row.
    pub mouse_dim_in: u64,
    /// Keyboard embedding width.
    pub hidden_size: u64,
    /// Keyboard attention width.
    pub keyboard_hidden_dim: u64,
    /// Mouse attention width.
    pub mouse_hidden_dim: u64,
    /// Attention heads of both streams.
    pub heads_num: u64,
    /// Latent frames per window.
    pub windows_size: u64,
    /// Pixel frames per latent frame.
    pub vae_time_compression_ratio: u64,
    /// Rotary base over the frame index.
    pub rope_theta: f64,
    /// Rotary dims per axis (frame, row, column) of one head.
    pub rope_dim_list: [u64; 3],
    #[serde(default = "yes")]
    /// Mouse stream present.
    pub enable_mouse: bool,
    #[serde(default = "yes")]
    /// Keyboard stream present.
    pub enable_keyboard: bool,
}

fn yes() -> bool {
    true
}

impl ActionConfig {
    /// Pixel-frame rows in one latent frame's window.
    #[must_use]
    pub fn window(&self) -> usize {
        (self.windows_size * self.vae_time_compression_ratio) as usize
    }

    /// Values per action row (keyboard then mouse).
    #[must_use]
    pub fn action_dims(&self) -> usize {
        (self.keyboard_dim_in + self.mouse_dim_in) as usize
    }
}

/// The world-model additions to a [`WanDitConfig`].
#[derive(Debug, Clone)]
pub struct WorldDit {
    /// Action module settings.
    pub action: ActionConfig,
    /// Rotary base spread: head `h` uses `10000 * (1 + spread * e_h)`,
    /// `e_h` evenly spaced over `[-1, 1]`.
    pub theta_spread: f64,
    /// Channels per latent cell of the camera input (before patching).
    pub camera_channels: u64,
}

/// The original-layout checkpoint configuration.
#[derive(Debug, Clone, Deserialize)]
struct OriginalConfig {
    dim: u64,
    ffn_dim: u64,
    freq_dim: u64,
    #[serde(default = "text_dim")]
    text_dim: u64,
    in_dim: u64,
    out_dim: u64,
    num_heads: u64,
    num_layers: u64,
    eps: f64,
    #[serde(default)]
    sigma_theta: f64,
    action_config: ActionConfig,
}

fn text_dim() -> u64 {
    4096
}

/// Camera channels per latent cell: six Plucker coordinates over a 16 x 16
/// pixel cell.
pub const CAMERA_CHANNELS: u64 = 6 * 256;

/// Read the original-layout `config.json` into a transformer configuration.
pub fn config(json: &[u8]) -> Result<WanDitConfig> {
    let c: OriginalConfig = serde_json::from_slice(json).map_err(|e| Error::Config(format!("world model config: {e}")))?;
    if c.dim % c.num_heads != 0 {
        return Err(Error::Config("dim is not a multiple of num_heads".into()));
    }
    let a = &c.action_config;
    if a.mouse_hidden_dim % a.heads_num != 0 || a.keyboard_hidden_dim % a.heads_num != 0 {
        return Err(Error::Config("action hidden dims must divide by heads_num".into()));
    }
    if a.rope_dim_list.iter().sum::<u64>() != a.mouse_hidden_dim / a.heads_num || a.keyboard_hidden_dim != a.mouse_hidden_dim {
        return Err(Error::Config("action rotary dims must cover one head".into()));
    }
    if !(a.enable_mouse && a.enable_keyboard) {
        return Err(Error::Config("both action streams are required".into()));
    }
    let cfg = WanDitConfig {
        patch_size: [1, 2, 2],
        num_attention_heads: c.num_heads,
        attention_head_dim: c.dim / c.num_heads,
        in_channels: c.in_dim,
        out_channels: c.out_dim,
        text_dim: c.text_dim,
        freq_dim: c.freq_dim,
        ffn_dim: c.ffn_dim,
        num_layers: c.num_layers,
        cross_attn_norm: true,
        qk_norm: Some("rms_norm_across_heads".into()),
        eps: c.eps,
        image_dim: None,
        added_kv_proj_dim: None,
        rope_max_seq_len: 1024,
        world: Some(WorldDit { action: c.action_config, theta_spread: c.sigma_theta, camera_channels: CAMERA_CHANNELS }),
    };
    cfg.validate()?;
    Ok(cfg)
}

/// Map an original-layout tensor name onto the names [`WanDitConfig`] reads.
#[must_use]
pub fn rename(name: &str) -> Option<String> {
    const TOP: [(&str, &str); 7] = [
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("head.head.", "proj_out."),
        ("head.modulation", "scale_shift_table"),
    ];
    for (from, to) in TOP {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    let Some(rest) = name.strip_prefix("blocks.") else {
        return Some(name.to_owned());
    };
    let (i, tail) = rest.split_once('.')?;
    const BLOCK: [(&str, &str); 14] = [
        ("self_attn.q.", "attn1.to_q."),
        ("self_attn.k.", "attn1.to_k."),
        ("self_attn.v.", "attn1.to_v."),
        ("self_attn.o.", "attn1.to_out.0."),
        ("self_attn.norm_", "attn1.norm_"),
        ("cross_attn.q.", "attn2.to_q."),
        ("cross_attn.k.", "attn2.to_k."),
        ("cross_attn.v.", "attn2.to_v."),
        ("cross_attn.o.", "attn2.to_out.0."),
        ("cross_attn.norm_", "attn2.norm_"),
        ("norm3.", "norm2."),
        ("ffn.0.", "ffn.net.0.proj."),
        ("ffn.2.", "ffn.net.2."),
        ("modulation", "scale_shift_table"),
    ];
    for (from, to) in BLOCK {
        if let Some(r) = tail.strip_prefix(from) {
            return Some(format!("blocks.{i}.{to}{r}"));
        }
    }
    Some(name.to_owned())
}

/// Weights of the world-model additions.
#[must_use]
pub fn weight_specs(cfg: &WanDitConfig, wd: &WorldDit, linear: WType) -> Vec<WeightSpec> {
    let d = cfg.dim();
    let f = WType::F32;
    let a = &wd.action;
    let cam_in = wd.camera_channels * cfg.patch_size.iter().product::<u64>();
    let mut v = vec![
        WeightSpec::new("patch_embedding_wancamctrl.weight", &[d, cam_in], linear),
        WeightSpec::new("patch_embedding_wancamctrl.bias", &[d], f),
    ];
    for l in ["c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2"] {
        v.push(WeightSpec::new(format!("{l}.weight"), &[d, d], linear));
        v.push(WeightSpec::new(format!("{l}.bias"), &[d], f));
    }
    let (h, c, kd) = (a.hidden_size, a.mouse_hidden_dim, a.keyboard_hidden_dim);
    let win = a.window() as u64;
    for i in 0..cfg.num_layers {
        let p = format!("blocks.{i}");
        for l in ["cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer"] {
            v.push(WeightSpec::new(format!("{p}.{l}.weight"), &[d, d], linear));
            v.push(WeightSpec::new(format!("{p}.{l}.bias"), &[d], f));
        }
        if !a.blocks.contains(&i) {
            continue;
        }
        let m = format!("{p}.action_model");
        v.extend([
            WeightSpec::new(format!("{m}.keyboard_embed.0.weight"), &[h, a.keyboard_dim_in], f),
            WeightSpec::new(format!("{m}.keyboard_embed.0.bias"), &[h], f),
            WeightSpec::new(format!("{m}.keyboard_embed.2.weight"), &[h, h], f),
            WeightSpec::new(format!("{m}.keyboard_embed.2.bias"), &[h], f),
            WeightSpec::new(format!("{m}.mouse_mlp.0.weight"), &[c, a.mouse_dim_in * win + d], linear),
            WeightSpec::new(format!("{m}.mouse_mlp.0.bias"), &[c], f),
            WeightSpec::new(format!("{m}.mouse_mlp.2.weight"), &[c, c], linear),
            WeightSpec::new(format!("{m}.mouse_mlp.2.bias"), &[c], f),
            WeightSpec::new(format!("{m}.mouse_mlp.3.weight"), &[c], f),
            WeightSpec::new(format!("{m}.mouse_mlp.3.bias"), &[c], f),
            WeightSpec::new(format!("{m}.t_qkv.weight"), &[3 * c, c], linear),
            WeightSpec::new(format!("{m}.proj_mouse.weight"), &[d, c], linear),
            WeightSpec::new(format!("{m}.mouse_attn_q.weight"), &[kd, d], linear),
            WeightSpec::new(format!("{m}.keyboard_attn_kv.weight"), &[2 * kd, h * win], linear),
            WeightSpec::new(format!("{m}.proj_keyboard.weight"), &[d, kd], linear),
        ]);
    }
    // 8-bit blocks hold 32 columns; narrower rows (the released mouse input
    // width) stay float32.
    for w in &mut v {
        if w.ty == WType::Q8_0 && !w.shape.last().copied().unwrap_or(0).is_multiple_of(32) {
            w.ty = f;
        }
    }
    v
}

/// Per-head rotary tables `[heads][pairs]` per token for latent frames at
/// `positions`, in token order (token-major: `[token][head][pair]`).
#[must_use]
pub fn rotary_tables(cfg: &WanDitConfig, wd: &WorldDit, positions: &[usize], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let heads = cfg.num_attention_heads as usize;
    let axes = cfg.rope_axes();
    let inv: Vec<[Vec<f64>; 3]> = (0..heads)
        .map(|h| {
            let e = if heads == 1 { -1.0 } else { -1.0 + 2.0 * h as f64 / (heads - 1) as f64 };
            let theta = 10000.0 * (1.0 + wd.theta_spread * e);
            axes.map(|p| (0..p).map(|j| 1.0 / theta.powf(j as f64 / p as f64)).collect())
        })
        .collect();
    let half: usize = axes.iter().sum();
    let n = positions.len() * rows * cols * heads * half;
    let (mut cos, mut sin) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for &f in positions {
        for r in 0..rows {
            for c in 0..cols {
                for hv in &inv {
                    for (axis, pos) in [f, r, c].into_iter().enumerate() {
                        for w in &hv[axis] {
                            let a = pos as f64 * w;
                            cos.push(a.cos() as f32);
                            sin.push(a.sin() as f32);
                        }
                    }
                }
            }
        }
    }
    (cos, sin)
}

/// Action-module rotary tables `[pairs]` per latent frame: the first
/// `rope_dim_list[0] / 2` pairs turn with the frame position, the rest
/// stay fixed (their axes have extent one).
#[must_use]
pub fn action_rotary(a: &ActionConfig, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let half = (a.mouse_hidden_dim / a.heads_num / 2) as usize;
    let t = (a.rope_dim_list[0] / 2) as usize;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &p in positions {
        for j in 0..half {
            let ang = if j < t { p as f64 / a.rope_theta.powf((2 * j) as f64 / (2 * t) as f64) } else { 0.0 };
            cos.push(ang.cos() as f32);
            sin.push(ang.sin() as f32);
        }
    }
    (cos, sin)
}

/// Window blocks for the latent frames of per-pixel-frame rows (`1 + 4 k`
/// rows give `k + 1` frames, `4 k` rows of a continuation give `k`), each
/// `dims` wide: `[frame][window][dims]`. Frame `i` reads padded rows
/// `4 i .. 4 i + window`, the padding being the first row repeated. `memory`
/// rows (one per memory latent frame) come first, each repeated across its
/// window.
pub fn windows(a: &ActionConfig, rows: &[f32], dims: usize, memory: &[f32]) -> Result<Vec<f32>> {
    let r = a.vae_time_compression_ratio as usize;
    let win = a.window();
    let n = rows.len() / dims;
    if rows.len() % dims != 0 || n == 0 {
        return Err(Error::Request("action rows are not whole".into()));
    }
    let (frames, pad) = if (n - 1) % r == 0 { ((n - 1) / r + 1, win) } else if n % r == 0 { (n / r, win - r) } else {
        return Err(Error::Request(format!("{n} action rows do not span whole latent frames")));
    };
    let padded: Vec<&[f32]> = std::iter::repeat_n(&rows[..dims], pad).chain(rows.chunks_exact(dims)).collect();
    let mut out = Vec::with_capacity((memory.len() / dims.max(1) + frames) * win * dims);
    for m in memory.chunks_exact(dims) {
        for _ in 0..win {
            out.extend_from_slice(m);
        }
    }
    for i in 0..frames {
        for row in &padded[r * i..r * i + win] {
            out.extend_from_slice(row);
        }
    }
    Ok(out)
}

/// Graph inputs of the action module.
#[derive(Debug, Clone, Copy)]
pub struct ActionIo {
    /// Keyboard windows `[keyboard_dim_in, window, frames]`.
    pub keyboard: Tn,
    /// Mouse windows `[mouse_dim_in * window, frames]`.
    pub mouse: Tn,
    /// Rotary tables `[1, pairs, 1, frames]`.
    pub cos: Tn,
    /// Sine table, like `cos`.
    pub sin: Tn,
}

/// Declare the action inputs for `frames` latent frames.
pub fn action_inputs(g: &mut Graph, a: &ActionConfig, frames: i64) -> ActionIo {
    let win = a.window() as i64;
    let half = (a.mouse_hidden_dim / a.heads_num / 2) as i64;
    ActionIo {
        keyboard: g.input(sys::GGML_TYPE_F32, &[a.keyboard_dim_in as i64, win, frames]),
        mouse: g.input(sys::GGML_TYPE_F32, &[a.mouse_dim_in as i64 * win, frames]),
        cos: g.input(sys::GGML_TYPE_F32, &[1, half, 1, frames]),
        sin: g.input(sys::GGML_TYPE_F32, &[1, half, 1, frames]),
    }
}

fn lin(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{p}.weight")), w.get(&format!("{p}.bias")), x)
}

/// `[c, S, T]` to `[c, T, S]` (or back), contiguous.
fn swap12(g: &mut Graph, x: Tn) -> Tn {
    let p = g.permute(x, [0, 2, 1, 3]);
    g.cont(p)
}

/// Rotate `x` `[hd, heads, T, S]` by per-frame tables `[1, hd/2, 1, T]`.
fn rope_frames(g: &mut Graph, x: Tn, cos: Tn, sin: Tn) -> Tn {
    let (hd, heads, t, s) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
    let x = g.reshape(x, &[hd, heads, t * s]);
    let x = rope_pairs(g, x, cos, sin);
    g.reshape(x, &[hd, heads, t, s])
}

/// The action module over `x` `[dim, frames * spatial]` (token order frame,
/// position).
pub fn build_action(g: &mut Graph, a: &ActionConfig, w: &Weights, p: &str, x: Tn, io: &ActionIo, frames: i64, exact: bool) -> Tn {
    let d = x.ne(0);
    let s = x.ne(1) / frames;
    let heads = a.heads_num as i64;
    let c = a.mouse_hidden_dim as i64;
    let hd = c / heads;
    let kd = a.keyboard_hidden_dim as i64;

    // Mouse: per spatial position, attention over frames.
    let xs = g.reshape(x, &[d, s, frames]);
    let m = g.reshape(io.mouse, &[io.mouse.ne(0), 1, frames]);
    let m = g.repeat_to(m, [io.mouse.ne(0), s, frames, 1]);
    let h = g.concat(xs, m, 0);
    let h = swap12(g, h);
    let h = lin(g, w, &format!("{p}.mouse_mlp.0"), h);
    let h = if exact { g.gelu_tanh_exact(h) } else { g.gelu_tanh(h) };
    let h = lin(g, w, &format!("{p}.mouse_mlp.2"), h);
    let h = g.norm(h, 1e-5);
    let h = g.mul(h, w.get(&format!("{p}.mouse_mlp.3.weight")));
    let h = g.add(h, w.get(&format!("{p}.mouse_mlp.3.bias")));
    let qkv = g.linear(w.get(&format!("{p}.t_qkv.weight")), h);
    let part = |g: &mut Graph, k: usize| {
        let v = g.view_4d(qkv, [c, frames, s, 1], qkv.nb(1), qkv.nb(2), qkv.nb(3), k * c as usize * qkv.nb(0));
        let v = g.cont(v);
        g.reshape(v, &[hd, heads, frames, s])
    };
    let q = part(g, 0);
    let k = part(g, 1);
    let v = part(g, 2);
    let q = g.rms_norm(q, 1e-6);
    let k = g.rms_norm(k, 1e-6);
    let q = rope_frames(g, q, io.cos, io.sin);
    let k = rope_frames(g, k, io.cos, io.sin);
    let o = attend_batched(g, q, k, v, exact);
    let o = g.reshape(o, &[c, frames, s]);
    let o = swap12(g, o);
    let o = g.reshape(o, &[c, frames * s]);
    let o = g.linear(w.get(&format!("{p}.proj_mouse.weight")), o);
    let x = g.add(x, o);

    // Keyboard: every token attends to the windowed keyboard embeddings.
    let kb = g.reshape(io.keyboard, &[a.keyboard_dim_in as i64, io.keyboard.ne(1) * frames]);
    let kb = lin(g, w, &format!("{p}.keyboard_embed.0"), kb);
    let kb = g.silu(kb);
    let kb = lin(g, w, &format!("{p}.keyboard_embed.2"), kb);
    let kb = g.reshape(kb, &[kb.ne(0) * io.keyboard.ne(1), frames]);
    let kv = g.linear(w.get(&format!("{p}.keyboard_attn_kv.weight")), kb);
    let kpart = |g: &mut Graph, k: usize| {
        let v = g.view_4d(kv, [kd, frames, 1, 1], kv.nb(1), kv.nb(2), kv.nb(3), k * kd as usize * kv.nb(0));
        let v = g.cont(v);
        g.reshape(v, &[hd, heads, frames, 1])
    };
    let kk = kpart(g, 0);
    let vv = kpart(g, 1);
    let kk = g.rms_norm(kk, 1e-6);
    let kk = rope_frames(g, kk, io.cos, io.sin);
    let kk = g.repeat_to(kk, [hd, heads, frames, s]);
    let vv = g.repeat_to(vv, [hd, heads, frames, s]);
    let q = g.linear(w.get(&format!("{p}.mouse_attn_q.weight")), x);
    let q = g.reshape(q, &[kd, s, frames]);
    let q = swap12(g, q);
    let q = g.reshape(q, &[hd, heads, frames, s]);
    let q = g.rms_norm(q, 1e-6);
    let q = rope_frames(g, q, io.cos, io.sin);
    let o = attend_batched(g, q, kk, vv, exact);
    let o = g.reshape(o, &[kd, frames, s]);
    let o = swap12(g, o);
    let o = g.reshape(o, &[kd, frames * s]);
    let o = g.linear(w.get(&format!("{p}.proj_keyboard.weight")), o);
    g.add(x, o)
}

/// The camera embedding shared by all blocks, from patchified camera rays
/// `[camera_channels * 4, tokens]`.
pub fn build_camera(g: &mut Graph, w: &Weights, rays: Tn) -> Tn {
    let e = lin(g, w, "patch_embedding_wancamctrl", rays);
    let h = lin(g, w, "c2ws_hidden_states_layer1", e);
    let h = g.silu(h);
    let h = lin(g, w, "c2ws_hidden_states_layer2", h);
    g.add(e, h)
}

/// Scale and shift the stream by the camera embedding in block `p`.
pub fn inject_camera(g: &mut Graph, w: &Weights, p: &str, x: Tn, cam: Tn) -> Tn {
    let h = lin(g, w, &format!("{p}.cam_injector_layer1"), cam);
    let h = g.silu(h);
    let h = lin(g, w, &format!("{p}.cam_injector_layer2"), h);
    let h = g.add(h, cam);
    let scale = lin(g, w, &format!("{p}.cam_scale_layer"), h);
    let shift = lin(g, w, &format!("{p}.cam_shift_layer"), h);
    let s = g.mul(x, scale);
    let x = g.add(x, s);
    g.add(x, shift)
}

/// Patchify camera rays `[channels][frames][h][w]` (latent grid) into
/// tokens `[channels * 4]` each, like the latent.
#[must_use]
pub fn patchify_rays(rays: &[f32], channels: usize, frames: usize, h: usize, w: usize) -> Vec<f32> {
    let (rows, cols) = (h / 2, w / 2);
    let mut out = Vec::with_capacity(rays.len());
    for f in 0..frames {
        for r in 0..rows {
            for c in 0..cols {
                for ch in 0..channels {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            out.push(rays[((ch * frames + f) * h + 2 * r + dy) * w + 2 * c + dx]);
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ActionConfig {
        serde_json::from_value(serde_json::json!({
            "blocks": [0], "keyboard_dim_in": 1, "mouse_dim_in": 1, "hidden_size": 4,
            "keyboard_hidden_dim": 8, "mouse_hidden_dim": 8, "heads_num": 2, "windows_size": 3,
            "vae_time_compression_ratio": 4, "rope_theta": 256.0, "rope_dim_list": [2, 1, 1]
        }))
        .unwrap()
    }

    #[test]
    fn windows_end_at_each_latent_frame() {
        let a = cfg();
        let rows: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let w = windows(&a, &rows, 1, &[]).unwrap();
        assert_eq!(w.len(), 3 * 12);
        assert_eq!(&w[..12], &[0.0; 12]);
        assert_eq!(&w[12..24], &[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3.]);
        assert_eq!(&w[24..], &[0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7.]);
        let cont: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let w = windows(&a, &cont, 1, &[9.0]).unwrap();
        assert_eq!(w.len(), 3 * 12);
        assert_eq!(&w[..12], &[9.0; 12]);
        assert_eq!(&w[12..24], &[1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 3., 4.]);
        assert_eq!(&w[24..], &[1., 1., 1., 1., 1., 2., 3., 4., 5., 6., 7., 8.]);
        assert!(windows(&a, &rows[..3], 1, &[]).is_err());
    }

    #[test]
    fn original_names_map_onto_the_transformer() {
        assert_eq!(rename("blocks.3.self_attn.o.weight").unwrap(), "blocks.3.attn1.to_out.0.weight");
        assert_eq!(rename("blocks.3.norm3.bias").unwrap(), "blocks.3.norm2.bias");
        assert_eq!(rename("blocks.0.modulation").unwrap(), "blocks.0.scale_shift_table");
        assert_eq!(rename("head.modulation").unwrap(), "scale_shift_table");
        assert_eq!(rename("time_projection.1.weight").unwrap(), "condition_embedder.time_proj.weight");
        assert_eq!(rename("blocks.0.action_model.t_qkv.weight").unwrap(), "blocks.0.action_model.t_qkv.weight");
    }
}
