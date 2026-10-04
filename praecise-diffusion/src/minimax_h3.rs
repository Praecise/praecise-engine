//! MiniMax-H3 transformer: one single-stream sequence of prompt, video and
//! audio tokens.
//!
//! Prompt states are projected and refined by a few plain blocks first. Every
//! main block is modulated per token: the timestep embeddings (one per
//! distinct timestep in the sequence) are projected to one modulation per
//! (timestep, modality) pair, and each token reads the row of its own pair.
//! Queries and keys carry per-head RMS norms and a partial rotary embedding
//! over three position axes (time, row, column) that leaves the tail of each
//! head unrotated.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::s3dit::S3DitConfig;
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

pub mod audio_vae;
pub mod pipeline;
pub mod vae;

/// Modulation rows per timestep: one per modality tag.
pub const MODALITIES: usize = 3;

/// `transformer/config.json` of a MiniMax-H3 checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniMaxH3Config {
    pub num_attention_heads: u64,
    pub attention_head_dim: u64,
    pub hidden_size: u64,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
    pub ffn_dim: u64,
    pub in_channels: u64,
    pub audio_in_channels: u64,
    pub patch_size: [u64; 3],
    pub text_dim: u64,
    pub freq_dim: u64,
    pub time_embed_hidden_dim: u64,
    pub time_embed_dim: u64,
    pub rope_freq_dim: u64,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub qk_norm_eps: f64,
    pub final_norm_eps: f64,
}

impl MiniMaxH3Config {
    /// Refuse layouts this implementation does not compute.
    ///
    /// # Errors
    /// [`Error::Config`] naming the problem.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("MiniMax-H3 transformer: {m}")));
        if self.num_attention_heads * self.attention_head_dim != self.hidden_size {
            return bad("heads do not fill the width");
        }
        if 6 * self.rope_freq_dim > self.attention_head_dim {
            return bad("rotary part wider than a head");
        }
        if self.freq_dim % 2 != 0 || self.num_layers == 0 {
            return bad("empty or odd layout");
        }
        Ok(())
    }

    /// Channels of one video token.
    #[must_use]
    pub fn patch_dim(&self) -> u64 {
        self.in_channels * self.patch_size.iter().product::<u64>()
    }

    fn rotary_dim(&self) -> u64 {
        6 * self.rope_freq_dim
    }

    /// Every weight.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let (d, hd, ff, te) = (self.hidden_size, self.attention_head_dim, self.ffn_dim, self.time_embed_dim);
        let f = WType::F32;
        let mut v = Vec::new();
        let mut lin = |name: &str, out: u64, inp: u64, ty: WType, bias: bool| {
            v.push(WeightSpec::new(format!("{name}.weight"), &[out, inp], ty));
            if bias {
                v.push(WeightSpec::new(format!("{name}.bias"), &[out], f));
            }
        };
        lin("proj_in", d, self.patch_dim(), f, true);
        lin("audio_proj_in", d, self.audio_in_channels, f, true);
        lin("context_embedder", d, self.text_dim, linear, true);
        lin("time_embedder.linear_1", self.time_embed_hidden_dim, self.freq_dim, f, true);
        lin("time_embedder.linear_2", te, self.time_embed_hidden_dim, f, true);
        lin("norm_out.linear", 2 * d, te, f, true);
        lin("proj_out", self.patch_dim(), d, f, true);
        lin("audio_proj_out", self.audio_in_channels, d, f, true);
        let block = |v: &mut Vec<WeightSpec>, p: &str| {
            for n in ["norm1", "norm2"] {
                v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[d], f));
            }
            for n in ["to_q", "to_k", "to_v"] {
                v.push(WeightSpec::new(format!("{p}.attn.{n}.weight"), &[d, d], linear));
            }
            v.push(WeightSpec::new(format!("{p}.attn.to_out.0.weight"), &[d, d], linear));
            for n in ["norm_q", "norm_k"] {
                v.push(WeightSpec::new(format!("{p}.attn.{n}.weight"), &[hd], f));
            }
            v.push(WeightSpec::new(format!("{p}.ff.net.0.proj.weight"), &[2 * ff, d], linear));
            v.push(WeightSpec::new(format!("{p}.ff.net.2.weight"), &[d, ff], linear));
        };
        for i in 0..self.num_refiner_layers {
            block(&mut v, &format!("token_refiner.refiner_blocks.{i}"));
        }
        v.push(WeightSpec::new("token_refiner.final_norm.weight", &[d], f));
        for i in 0..self.num_layers {
            let p = format!("transformer_blocks.{i}");
            block(&mut v, &p);
            v.push(WeightSpec::new(format!("{p}.adaln_proj.linear.weight"), &[6 * d * MODALITIES as u64, te], linear));
            v.push(WeightSpec::new(format!("{p}.adaln_proj.linear.bias"), &[6 * d * MODALITIES as u64], f));
        }
        v.push(WeightSpec::new("norm_out.norm.weight", &[d], f));
        v
    }

    /// Rotary cos and sin tables `[tokens][rotary width]` for `(t, h, w)`
    /// positions: the time, row and column angles side by side, repeated.
    #[must_use]
    pub fn rotary_tables(&self, pos: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let n = self.rope_freq_dim as usize;
        let inv: Vec<f32> = (0..n).map(|j| 1.0 / (self.rope_theta as f32).powf((2 * j) as f32 / (2 * n) as f32)).collect();
        let mut cos = Vec::with_capacity(pos.len() * 6 * n);
        let mut sin = Vec::with_capacity(pos.len() * 6 * n);
        for p in pos {
            let ang: Vec<f32> = (0..3).flat_map(|a| inv.iter().map(move |f| p[a] * f)).collect();
            for _ in 0..2 {
                cos.extend(ang.iter().map(|a| a.cos()));
                sin.extend(ang.iter().map(|a| a.sin()));
            }
        }
        (cos, sin)
    }
}

/// Where a token of the joint sequence comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Source {
    Text(usize),
    Video(usize),
    Audio(usize),
}

/// One token of the joint sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H3Token {
    pub source: H3Source,
    /// Modality tag selecting the modulation row, below [`MODALITIES`].
    pub tag: usize,
    /// Index into the timesteps.
    pub timestep: usize,
    /// `(t, h, w)` rotary position.
    pub pos: [f32; 3],
}

/// Inputs of one forward pass.
#[derive(Debug, Clone, Copy)]
pub struct H3Input<'a> {
    /// Video patches `[tokens][patch channels]`.
    pub video: &'a [f32],
    /// Audio latents `[tokens][audio channels]`.
    pub audio: &'a [f32],
    /// Prompt states `[tokens][text width]`.
    pub text: &'a [f32],
    /// The distinct timesteps tokens refer to (the model's own scale).
    pub timesteps: &'a [f32],
    /// The joint sequence in order.
    pub tokens: &'a [H3Token],
}

#[derive(Clone, Copy)]
struct Ctx {
    d: i64,
    hd: i64,
    heads: i64,
    rot: i64,
    eps: f32,
    qk_eps: f32,
    exact: bool,
}

fn lin(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear(w.get(&format!("{name}.weight")), x)
}

fn lin_b(g: &mut Graph, w: &Weights, name: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{name}.weight")), w.get(&format!("{name}.bias")), x)
}

fn rms(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.rms_norm(x, eps);
    g.mul(h, w.get(name))
}

/// Rotate the leading `c.rot` channels of every head of `x` `[hd, heads, n]`.
fn rope(g: &mut Graph, c: Ctx, x: Tn, (cos, sin): (Tn, Tn)) -> Tn {
    if c.rot == c.hd {
        return g.rotate_half_rope(x, cos, sin);
    }
    let (heads, n) = (x.ne(1), x.ne(2));
    let es = x.nb(0);
    let head = g.view_4d(x, [c.rot, heads, n, 1], x.nb(1), x.nb(2), x.nb(3), 0);
    let head = g.cont(head);
    let tail = g.view_4d(x, [c.hd - c.rot, heads, n, 1], x.nb(1), x.nb(2), x.nb(3), c.rot as usize * es);
    let tail = g.cont(tail);
    let head = g.rotate_half_rope(head, cos, sin);
    g.concat(head, tail, 0)
}

fn attention(g: &mut Graph, w: &Weights, p: &str, c: Ctx, x: Tn, rot: Option<(Tn, Tn)>) -> Tn {
    let n = x.ne(1);
    let mut qkv = [x; 3];
    for (o, name) in qkv.iter_mut().zip(["to_q", "to_k", "to_v"]) {
        let t = lin(g, w, &format!("{p}.attn.{name}"), x);
        *o = g.reshape(t, &[c.hd, c.heads, n]);
    }
    let [q, k, v] = qkv;
    let q = rms(g, w, &format!("{p}.attn.norm_q.weight"), q, c.qk_eps);
    let k = rms(g, w, &format!("{p}.attn.norm_k.weight"), k, c.qk_eps);
    let (q, k) = match rot {
        Some(t) => (rope(g, c, q, t), rope(g, c, k, t)),
        None => (q, k),
    };
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
    let o = g.reshape(o, &[c.d, n]);
    lin(g, w, &format!("{p}.attn.to_out.0"), o)
}

fn feed_forward(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let h = lin(g, w, &format!("{p}.ff.net.0.proj"), x);
    let half = h.ne(0) / 2;
    let up = g.view_rows(h, 0, half);
    let gate = g.view_rows(h, half, half);
    let up = g.cont(up);
    let gate = g.cont(gate);
    let f = g.swiglu_split(gate, up);
    lin(g, w, &format!("{p}.ff.net.2"), f)
}

/// A loaded MiniMax-H3 transformer.
pub struct MiniMaxH3Transformer {
    backend: Backend,
    cfg: MiniMaxH3Config,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for MiniMaxH3Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniMaxH3Transformer").field("device", &self.backend.name()).field("layers", &self.cfg.num_layers).finish_non_exhaustive()
    }
}

impl MiniMaxH3Transformer {
    /// Load `transformer/` of a checkpoint.
    ///
    /// # Errors
    /// A missing or malformed config or weight, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: MiniMaxH3Config = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "video transformer backend selected");
        let st = SafeTensors::open(&files.weights("transformer")?)?;
        let w = Weights::load(&backend, &st, &cfg.weight_specs(opts.precision.wtype()))?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32 })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &MiniMaxH3Config {
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

    /// Velocities of the video tokens `[tokens][patch channels]` and of the
    /// audio tokens `[tokens][audio channels]`, in their input order.
    ///
    /// # Errors
    /// Inputs that disagree with each other, or a backend failure.
    pub fn forward(&self, inp: &H3Input<'_>) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.cfg;
        let (pd, ad, td) = (cfg.patch_dim() as usize, cfg.audio_in_channels as usize, cfg.text_dim as usize);
        let (nv, na, nt) = (inp.video.len() / pd, inp.audio.len() / ad, inp.text.len() / td);
        let k = inp.timesteps.len();
        let n = inp.tokens.len();
        let bad = |m: &str| Err(Error::Request(format!("MiniMax-H3 inputs: {m}")));
        if nv * pd != inp.video.len() || na * ad != inp.audio.len() || nt * td != inp.text.len() || nt == 0 || k == 0 {
            return bad("token buffers are not whole tokens");
        }
        if n != nt + nv + na {
            return bad("the sequence must hold every text, video and audio token once");
        }
        let mut seen = vec![false; n];
        let mut perm = Vec::with_capacity(n);
        let (mut vid_rows, mut aud_rows) = (vec![0i32; nv], vec![0i32; na]);
        for (j, t) in inp.tokens.iter().enumerate() {
            let r = match t.source {
                H3Source::Text(i) if i < nt => i,
                H3Source::Video(i) if i < nv => {
                    vid_rows[i] = j as i32;
                    nt + i
                }
                H3Source::Audio(i) if i < na => {
                    aud_rows[i] = j as i32;
                    nt + nv + i
                }
                _ => return bad("a token refers past its buffer"),
            };
            if seen[r] || t.tag >= MODALITIES || t.timestep >= k {
                return bad("a source is used twice, or a tag or timestep is out of range");
            }
            seen[r] = true;
            perm.push(r as i32);
        }
        let c = Ctx {
            d: cfg.hidden_size as i64,
            hd: cfg.attention_head_dim as i64,
            heads: cfg.num_attention_heads as i64,
            rot: cfg.rotary_dim() as i64,
            eps: cfg.norm_eps as f32,
            qk_eps: cfg.qk_norm_eps as f32,
            exact: self.exact,
        };
        let w = &self.w;
        let mut g = Graph::new(&self.backend)?;
        let text = g.input(sys::GGML_TYPE_F32, &[td as i64, nt as i64]);
        let video = (nv > 0).then(|| g.input(sys::GGML_TYPE_F32, &[pd as i64, nv as i64]));
        let audio = (na > 0).then(|| g.input(sys::GGML_TYPE_F32, &[ad as i64, na as i64]));
        let time = g.input(sys::GGML_TYPE_F32, &[cfg.freq_dim as i64, k as i64]);
        let perm_t = g.input(sys::GGML_TYPE_I32, &[n as i64]);
        let mod_idx = g.input(sys::GGML_TYPE_I32, &[n as i64]);
        let out_idx = g.input(sys::GGML_TYPE_I32, &[n as i64]);
        let cos = g.input(sys::GGML_TYPE_F32, &[c.rot, 1, n as i64]);
        let sin = g.input(sys::GGML_TYPE_F32, &[c.rot, 1, n as i64]);

        let mut t = lin_b(&mut g, w, "context_embedder", text);
        for i in 0..cfg.num_refiner_layers {
            let p = format!("token_refiner.refiner_blocks.{i}");
            let h = rms(&mut g, w, &format!("{p}.norm1.weight"), t, c.eps);
            let a = attention(&mut g, w, &p, c, h, None);
            t = g.add(t, a);
            let h = rms(&mut g, w, &format!("{p}.norm2.weight"), t, c.eps);
            let f = feed_forward(&mut g, w, &p, h);
            t = g.add(t, f);
        }
        let mut all = rms(&mut g, w, "token_refiner.final_norm.weight", t, cfg.final_norm_eps as f32);
        if let Some(v) = video {
            let v = lin_b(&mut g, w, "proj_in", v);
            all = g.concat(all, v, 1);
        }
        if let Some(a) = audio {
            let a = lin_b(&mut g, w, "audio_proj_in", a);
            all = g.concat(all, a, 1);
        }
        let mut x = g.get_rows(all, perm_t);

        let te = lin_b(&mut g, w, "time_embedder.linear_1", time);
        let te = g.silu(te);
        let te = lin_b(&mut g, w, "time_embedder.linear_2", te);
        let te = g.silu(te);
        let d = c.d;
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            // [6d * modalities, k] -> one row of 6d per (timestep, modality).
            let m = lin_b(&mut g, w, &format!("{p}.adaln_proj.linear"), te);
            let m = g.reshape(m, &[6 * d, (k * MODALITIES) as i64]);
            let m = g.get_rows(m, mod_idx);
            let part = |g: &mut Graph, j: i64| {
                let v = g.view_rows(m, j * d, d);
                g.cont(v)
            };
            let (shift1, scale1, gate1) = (part(&mut g, 0), part(&mut g, 1), part(&mut g, 2));
            let (shift2, scale2, gate2) = (part(&mut g, 3), part(&mut g, 4), part(&mut g, 5));
            let h = rms(&mut g, w, &format!("{p}.norm1.weight"), x, c.eps);
            let s = g.scale_bias(scale1, 1.0, 1.0);
            let h = g.mul(h, s);
            let h = g.add(h, shift1);
            let a = attention(&mut g, w, &p, c, h, Some((cos, sin)));
            let a = g.mul(a, gate1);
            x = g.add(x, a);
            let h = rms(&mut g, w, &format!("{p}.norm2.weight"), x, c.eps);
            let s = g.scale_bias(scale2, 1.0, 1.0);
            let h = g.mul(h, s);
            let h = g.add(h, shift2);
            let f = feed_forward(&mut g, w, &p, h);
            let f = g.mul(f, gate2);
            x = g.add(x, f);
        }
        let m = lin_b(&mut g, w, "norm_out.linear", te);
        let m = g.get_rows(m, out_idx);
        let shift = g.view_rows(m, 0, d);
        let shift = g.cont(shift);
        let scale = g.view_rows(m, d, d);
        let scale = g.cont(scale);
        let scale = g.scale_bias(scale, 1.0, 1.0);
        let h = rms(&mut g, w, "norm_out.norm.weight", x, cfg.final_norm_eps as f32);
        let h = g.mul(h, scale);
        let h = g.add(h, shift);
        let mut outs = Vec::new();
        let vid = (nv > 0).then(|| {
            let idx = g.input(sys::GGML_TYPE_I32, &[nv as i64]);
            let r = g.get_rows(h, idx);
            (idx, lin_b(&mut g, w, "proj_out", r))
        });
        let aud = (na > 0).then(|| {
            let idx = g.input(sys::GGML_TYPE_I32, &[na as i64]);
            let r = g.get_rows(h, idx);
            (idx, lin_b(&mut g, w, "audio_proj_out", r))
        });
        outs.extend(vid.map(|v| v.1));
        outs.extend(aud.map(|a| a.1));
        g.finish(&outs)?;

        g.set_f32(text, inp.text);
        if let Some(v) = video {
            g.set_f32(v, inp.video);
        }
        if let Some(a) = audio {
            g.set_f32(a, inp.audio);
        }
        let feats: Vec<f32> = inp.timesteps.iter().flat_map(|&s| S3DitConfig::time_features(s)).collect();
        if feats.len() != k * cfg.freq_dim as usize {
            return Err(Error::Config("timestep features disagree with the configured width".into()));
        }
        g.set_f32(time, &feats);
        g.set_i32(perm_t, &perm);
        g.set_i32(mod_idx, &inp.tokens.iter().map(|t| (t.timestep * MODALITIES + t.tag) as i32).collect::<Vec<_>>());
        g.set_i32(out_idx, &inp.tokens.iter().map(|t| t.timestep as i32).collect::<Vec<_>>());
        let pos: Vec<[f32; 3]> = inp.tokens.iter().map(|t| t.pos).collect();
        let (cs, sn) = cfg.rotary_tables(&pos);
        g.set_f32(cos, &cs);
        g.set_f32(sin, &sn);
        if let Some((idx, _)) = vid {
            g.set_i32(idx, &vid_rows);
        }
        if let Some((idx, _)) = aud {
            g.set_i32(idx, &aud_rows);
        }
        g.compute()?;
        Ok((vid.map_or_else(Vec::new, |v| g.read_f32(v.1)), aud.map_or_else(Vec::new, |a| g.read_f32(a.1))))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn rotary_angles_run_time_then_row_then_column() {
        let cfg: MiniMaxH3Config = serde_json::from_value(serde_json::json!({
            "num_attention_heads": 2, "attention_head_dim": 24, "hidden_size": 48, "num_layers": 1,
            "num_refiner_layers": 1, "ffn_dim": 8, "in_channels": 4, "audio_in_channels": 4, "patch_size": [1, 2, 2],
            "text_dim": 8, "freq_dim": 256, "time_embed_hidden_dim": 8, "time_embed_dim": 8, "rope_freq_dim": 2,
            "rope_theta": 10000.0, "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5
        }))
        .unwrap();
        cfg.validate().unwrap();
        let (_, sin) = cfg.rotary_tables(&[[1.0, 2.0, 3.0]]);
        assert_eq!(sin.len(), 12);
        assert!((sin[0] - 1f32.sin()).abs() < 1e-6 && (sin[2] - 2f32.sin()).abs() < 1e-6 && (sin[4] - 3f32.sin()).abs() < 1e-6);
        assert_eq!(sin[..6], sin[6..]);
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = PathBuf::from(std::env::var("PRAECISE_MINIMAX_H3_PARITY").expect("PRAECISE_MINIMAX_H3_PARITY names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let tf = MiniMaxH3Transformer::load(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap();
        let ints = |k: &str| m[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect::<Vec<_>>();
        let (tags, steps, text_at, video_at, audio_at) = (ints("tags"), ints("timestep_indices"), ints("text_indices"), ints("video_indices"), ints("audio_indices"));
        let pos = bin(&d, "positions");
        let mut source = vec![H3Source::Text(0); tags.len()];
        for (i, &j) in text_at.iter().enumerate() {
            source[j] = H3Source::Text(i);
        }
        for (i, &j) in video_at.iter().enumerate() {
            source[j] = H3Source::Video(i);
        }
        for (i, &j) in audio_at.iter().enumerate() {
            source[j] = H3Source::Audio(i);
        }
        let tokens: Vec<H3Token> =
            (0..tags.len()).map(|j| H3Token { source: source[j], tag: tags[j], timestep: steps[j], pos: [pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]] }).collect();
        let (video, audio, text, timesteps) = (bin(&d, "video"), bin(&d, "audio"), bin(&d, "text"), bin(&d, "timesteps"));
        let (v, a) = tf.forward(&H3Input { video: &video, audio: &audio, text: &text, timesteps: &timesteps, tokens: &tokens }).unwrap();
        assert_close("video", &v, &bin(&d, "out_video"), min_cos, max_rel);
        assert_close("audio", &a, &bin(&d, "out_audio"), min_cos, max_rel);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_transformer_f32() {
        run(Precision::F32, 0.999_999, 1e-4);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_transformer_bf16() {
        run(Precision::Bf16, 0.9999, 2e-2);
    }
}
