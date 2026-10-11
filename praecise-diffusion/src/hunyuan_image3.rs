//! HunyuanImage 3.0 image side: the parts around the language backbone that turn latents into
//! backbone inputs and backbone outputs into a velocity, and the latent decoder.
//!
//! The backbone denoises the image inside its own token sequence: every step, the noisy latent
//! goes through the patch embedding (a convolution and a timestep-modulated residual block) to
//! become the image tokens, a timestep token is embedded beside it, and the backbone's last
//! hidden states at the image positions go through the final layer (a modulated residual block,
//! group norm, convolution) to give the flow velocity.
//!
//! The decoder is a 3D-convolution autoencoder decoder run on a single latent frame: residual
//! blocks with group norm over channels, frames and space, single-head attention in the middle,
//! and sub-pixel upsampling stages (a convolution to `factor` times the channels rearranged into
//! space and time, plus the input repeated as a shortcut). The last decoded frame is the image.
//!
//! Weights and the configuration come from the image-side safetensors file written next to the
//! backbone GGUF by the converter (`<model>.image.safetensors`, original tensor names, the
//! configuration in the `config` metadata entry).

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const HEAD_GN_EPS: f32 = 1e-5;
const VAE_GN_EPS: f32 = 1e-6;
const GN_GROUPS: i32 = 32;
const FREQ_DIM: usize = 256;
const ATTN_QUERY_BLOCK: i64 = 4096;

/// Latent autoencoder configuration (the checkpoint's `vae` entry).
#[derive(Debug, Clone, Deserialize)]
pub struct Vae3dConfig {
    /// Encoder widths per level, shallowest first; the decoder runs them reversed.
    pub block_out_channels: Vec<u64>,
    /// Residual blocks per encoder level; the decoder has one more.
    pub layers_per_block: usize,
    /// Latent channels.
    pub latent_channels: u64,
    /// Image channels out.
    pub out_channels: u64,
    /// Spatial downsampling factor.
    pub ffactor_spatial: u64,
    /// Temporal downsampling factor.
    pub ffactor_temporal: u64,
    /// Latents are divided by this before decoding.
    pub scaling_factor: f64,
    /// Then shifted by this, when present.
    #[serde(default)]
    pub shift_factor: Option<f64>,
    /// Whether upsampling stages change width to the next level's.
    #[serde(default = "yes")]
    pub upsample_match_channel: bool,
}

fn yes() -> bool {
    true
}

/// The image-side configuration stored with the weights.
#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanImage3Config {
    /// Backbone width.
    pub hidden_size: u64,
    /// Width of the patch embedding and final layer.
    pub patch_embed_hidden_dim: u64,
    /// Latent cells per image token side; only 1 is implemented.
    pub patch_size: u64,
    /// Latent autoencoder.
    pub vae: Vae3dConfig,
}

impl HunyuanImage3Config {
    fn validate(&self) -> Result<()> {
        if self.patch_size != 1 {
            return Err(Error::Config(format!("HunyuanImage 3.0 patch size {} is not implemented", self.patch_size)));
        }
        let v = &self.vae;
        if !v.ffactor_spatial.is_power_of_two() || !v.ffactor_temporal.is_power_of_two() || v.block_out_channels.is_empty() {
            return Err(Error::Config("unsupported latent decoder factors".into()));
        }
        if !v.upsample_match_channel {
            return Err(Error::Config("latent decoder without channel-matching upsampling is not implemented".into()));
        }
        let ups = v.ffactor_spatial.trailing_zeros() as usize;
        if ups >= v.block_out_channels.len() || v.ffactor_temporal.trailing_zeros() as usize > ups {
            return Err(Error::Config("latent decoder factors exceed its levels".into()));
        }
        Ok(())
    }

    /// Pixels per latent cell side.
    #[must_use]
    pub fn scale_factor(&self) -> u64 {
        self.vae.ffactor_spatial
    }

    /// Latent channels.
    #[must_use]
    pub fn latent_channels(&self) -> usize {
        self.vae.latent_channels as usize
    }
}

/// Sinusoidal timestep features `[cos, sin]` of width 256 (period 10000).
#[must_use]
pub fn timestep_features(t: f32) -> Vec<f32> {
    let half = FREQ_DIM / 2;
    let mut out = vec![0f32; FREQ_DIM];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let a = t * freq;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

/// Read the image-side file.
///
/// # Errors
/// A missing file, a missing or malformed `config` entry, or an unsupported layout.
pub fn open(path: &Path) -> Result<(SafeTensors, HunyuanImage3Config)> {
    let files = SafeTensors::open(&[path.to_path_buf()])?;
    let raw = files.metadata("config").ok_or_else(|| Error::Config(format!("{} has no config entry", path.display())))?;
    let cfg: HunyuanImage3Config = serde_json::from_str(raw).map_err(|e| Error::Config(format!("image-side config: {e}")))?;
    cfg.validate()?;
    Ok((files, cfg))
}

/// Host tensors for a set of weights: 3D kernels `[o, i, kt, kh, kw]` split into one 2D kernel
/// per temporal tap (`<name>.t<k>`), everything else as is.
struct Hosts<'a> {
    files: &'a SafeTensors,
    kt: WType,
    v: Vec<HostTensor>,
}

impl Hosts<'_> {
    fn vec(&mut self, name: &str, n: u64) -> Result<()> {
        let d = self.files.require(name, &[n])?.to_f32();
        self.v.push(HostTensor { name: name.into(), shape: vec![n], ty: WType::F32, data: d });
        Ok(())
    }

    fn mat(&mut self, name: &str, o: u64, i: u64) -> Result<()> {
        let d = self.files.require(name, &[o, i])?.to_f32();
        self.v.push(HostTensor { name: name.into(), shape: vec![o, i], ty: WType::F32, data: d });
        Ok(())
    }

    fn lin(&mut self, p: &str, o: u64, i: u64) -> Result<()> {
        self.mat(&format!("{p}.weight"), o, i)?;
        self.vec(&format!("{p}.bias"), o)
    }

    fn conv2d(&mut self, p: &str, o: u64, i: u64, k: u64) -> Result<()> {
        let name = format!("{p}.weight");
        let d = self.files.require(&name, &[o, i, k, k])?.to_f32();
        self.v.push(HostTensor { name, shape: vec![o, i, k, k], ty: self.kt, data: d });
        self.vec(&format!("{p}.bias"), o)
    }

    fn conv3d(&mut self, p: &str, o: u64, i: u64, k: u64) -> Result<()> {
        let name = format!("{p}.weight");
        let d = self.files.require(&name, &[o, i, k, k, k])?.to_f32();
        let kk = (k * k) as usize;
        for t in 0..k as usize {
            let mut tap = Vec::with_capacity((o * i) as usize * kk);
            for oi in 0..(o * i) as usize {
                let base = oi * k as usize * kk + t * kk;
                tap.extend_from_slice(&d[base..base + kk]);
            }
            self.v.push(HostTensor { name: format!("{name}.t{t}"), shape: vec![o, i, k, k], ty: self.kt, data: tap });
        }
        self.vec(&format!("{p}.bias"), o)
    }

    fn norm(&mut self, p: &str, c: u64) -> Result<()> {
        self.vec(&format!("{p}.weight"), c)?;
        self.vec(&format!("{p}.bias"), c)
    }
}

fn bias4(g: &mut Graph, b: Tn) -> Tn {
    let c = b.ne(0);
    g.reshape(b, &[1, 1, c, 1])
}

fn conv(g: &mut Graph, w: &Weights, p: &str, x: Tn, pad: i32) -> Tn {
    let y = g.conv2d(w.get(&format!("{p}.weight")), x, pad);
    let b = bias4(g, w.get(&format!("{p}.bias")));
    g.add(y, b)
}

/// Group norm over channels and space of `[W, H, C, N]`, per batch entry.
fn gn2d(g: &mut Graph, w: &Weights, p: &str, x: Tn, eps: f32) -> Tn {
    let s = bias4(g, w.get(&format!("{p}.weight")));
    let b = bias4(g, w.get(&format!("{p}.bias")));
    let y = g.group_norm(x, GN_GROUPS, eps);
    let y = g.mul(y, s);
    g.add(y, b)
}

// ---- patch embedding and final layer --------------------------------------------------------

/// A residual block whose second norm is scaled and shifted by the timestep embedding.
fn res_block(g: &mut Graph, w: &Weights, p: &str, x: Tn, emb: Tn) -> Tn {
    let h = gn2d(g, w, &format!("{p}.in_layers.0"), x, HEAD_GN_EPS);
    let h = g.silu(h);
    let h = conv(g, w, &format!("{p}.in_layers.2"), h, 1);
    let cout = h.ne(2);
    let e = g.silu(emb);
    let e = g.linear_b(w.get(&format!("{p}.emb_layers.1.weight")), w.get(&format!("{p}.emb_layers.1.bias")), e);
    let scale = g.view_1d(e, cout, 0);
    let shift = g.view_1d(e, cout, cout as usize);
    let scale = g.cont(scale);
    let shift = g.cont(shift);
    let scale = g.reshape(scale, &[1, 1, cout, 1]);
    let shift = g.reshape(shift, &[1, 1, cout, 1]);
    let h = gn2d(g, w, &format!("{p}.out_layers.0"), h, HEAD_GN_EPS);
    let hs = g.mul(h, scale);
    let h = g.add(h, hs);
    let h = g.add(h, shift);
    let h = g.silu(h);
    let h = conv(g, w, &format!("{p}.out_layers.3"), h, 1);
    let skip = if x.ne(2) == cout { x } else { conv(g, w, &format!("{p}.skip_connection"), x, 0) };
    g.add(skip, h)
}

fn timestep_mlp(g: &mut Graph, w: &Weights, p: &str, feats: Tn) -> Tn {
    let x = g.linear_b(w.get(&format!("{p}.mlp.0.weight")), w.get(&format!("{p}.mlp.0.bias")), feats);
    let x = g.gelu_erf(x);
    g.linear_b(w.get(&format!("{p}.mlp.2.weight")), w.get(&format!("{p}.mlp.2.bias")), x)
}

/// Patch embedding, timestep token embedding and final layer, resident on one backend.
pub struct HunyuanImage3Head {
    cfg: HunyuanImage3Config,
    weights: Weights,
    backend: Backend,
}

impl std::fmt::Debug for HunyuanImage3Head {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HunyuanImage3Head").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl HunyuanImage3Head {
    /// Load from the image-side file. `exact` keeps convolution kernels in f32.
    ///
    /// # Errors
    /// A missing file or weight, or an unsupported configuration.
    pub fn load(path: &Path, backend: Backend, exact: bool) -> Result<Self> {
        let (files, cfg) = open(path)?;
        let (hd, pd, lc) = (cfg.hidden_size, cfg.patch_embed_hidden_dim, cfg.vae.latent_channels);
        let mut h = Hosts { files: &files, kt: if exact { WType::F32 } else { WType::F16 }, v: Vec::new() };
        for p in ["timestep_emb", "time_embed", "time_embed_2"] {
            h.lin(&format!("{p}.mlp.0"), hd, FREQ_DIM as u64)?;
            h.lin(&format!("{p}.mlp.2"), hd, hd)?;
        }
        h.conv2d("patch_embed.model.0", pd, lc, 3)?;
        let rb = |h: &mut Hosts, p: &str, cin: u64, cout: u64| -> Result<()> {
            h.norm(&format!("{p}.in_layers.0"), cin)?;
            h.conv2d(&format!("{p}.in_layers.2"), cout, cin, 3)?;
            h.lin(&format!("{p}.emb_layers.1"), 2 * cout, hd)?;
            h.norm(&format!("{p}.out_layers.0"), cout)?;
            h.conv2d(&format!("{p}.out_layers.3"), cout, cout, 3)?;
            if cin != cout {
                h.conv2d(&format!("{p}.skip_connection"), cout, cin, 1)?;
            }
            Ok(())
        };
        rb(&mut h, "patch_embed.model.1", pd, hd)?;
        rb(&mut h, "final_layer.model.0", hd, pd)?;
        h.norm("final_layer.model.1.0", pd)?;
        h.conv2d("final_layer.model.1.2", lc, pd, 3)?;
        let weights = Weights::from_host(&backend, &h.v)?;
        Ok(Self { cfg, weights, backend })
    }

    /// The configuration read with the weights.
    #[must_use]
    pub fn config(&self) -> &HunyuanImage3Config {
        &self.cfg
    }

    fn features(g: &mut Graph, t: f32) -> (Tn, Vec<f32>) {
        let f = g.input(sys::GGML_TYPE_F32, &[FREQ_DIM as i64, 1]);
        (f, timestep_features(t))
    }

    /// The timestep token's backbone input at timestep `t`.
    ///
    /// # Errors
    /// A failed compute.
    pub fn timestep_token(&self, t: f32) -> Result<Vec<f32>> {
        let mut g = Graph::new(&self.backend)?;
        let (f, fv) = Self::features(&mut g, t);
        let out = timestep_mlp(&mut g, &self.weights, "timestep_emb", f);
        g.finish(&[out])?;
        g.set_f32(f, &fv);
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// Image tokens for `n` latents (each planar `[C, h, w]`) at timestep `t`:
    /// `n * h * w` rows of the backbone width, row-major cells.
    ///
    /// # Errors
    /// A wrong input size or a failed compute.
    pub fn embed_latents(&self, latents: &[f32], n: usize, (h, w): (usize, usize), t: f32) -> Result<Vec<f32>> {
        let lc = self.cfg.latent_channels();
        if latents.len() != n * lc * h * w {
            return Err(Error::Config(format!("patch embedding takes {} floats, got {}", n * lc * h * w, latents.len())));
        }
        let mut g = Graph::new(&self.backend)?;
        let (f, fv) = Self::features(&mut g, t);
        let emb = timestep_mlp(&mut g, &self.weights, "time_embed", f);
        let x = g.input(sys::GGML_TYPE_F32, &[w as i64, h as i64, lc as i64, n as i64]);
        let y = conv(&mut g, &self.weights, "patch_embed.model.0", x, 1);
        let y = res_block(&mut g, &self.weights, "patch_embed.model.1", y, emb);
        // [W, H, D, N] -> rows [D, W, H, N]
        let y = g.permute(y, [1, 2, 0, 3]);
        let out = g.cont(y);
        g.finish(&[out])?;
        g.set_f32(f, &fv);
        g.set_f32(x, latents);
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// Velocity for `n` images from the backbone's hidden states at their image positions
    /// (`n * h * w` rows of the backbone width), at timestep `t`; planar `[C, h, w]` each.
    ///
    /// # Errors
    /// A wrong input size or a failed compute.
    pub fn predict(&self, hidden: &[f32], n: usize, (h, w): (usize, usize), t: f32) -> Result<Vec<f32>> {
        let hd = self.cfg.hidden_size as usize;
        if hidden.len() != n * h * w * hd {
            return Err(Error::Config(format!("final layer takes {} floats, got {}", n * h * w * hd, hidden.len())));
        }
        let mut g = Graph::new(&self.backend)?;
        let (f, fv) = Self::features(&mut g, t);
        let emb = timestep_mlp(&mut g, &self.weights, "time_embed_2", f);
        let x = g.input(sys::GGML_TYPE_F32, &[hd as i64, w as i64, h as i64, n as i64]);
        // rows [D, W, H, N] -> [W, H, D, N]
        let y = g.permute(x, [2, 0, 1, 3]);
        let y = g.cont(y);
        let y = res_block(&mut g, &self.weights, "final_layer.model.0", y, emb);
        let y = gn2d(&mut g, &self.weights, "final_layer.model.1.0", y, HEAD_GN_EPS);
        let y = g.silu(y);
        let out = conv(&mut g, &self.weights, "final_layer.model.1.2", y, 1);
        g.finish(&[out])?;
        g.set_f32(f, &fv);
        g.set_f32(x, hidden);
        g.compute()?;
        Ok(g.read_f32(out))
    }
}

// ---- latent decoder ----------------------------------------------------------------------------

/// Group norm over channels, frames and space of `[W, H, C, T]`.
fn gn3d(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let (wd, ht, c, t) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
    if t == 1 {
        return gn2d(g, w, p, x, VAE_GN_EPS);
    }
    let a = g.reshape(x, &[wd * ht, c, t, 1]);
    let a = g.permute(a, [0, 2, 1, 3]);
    let a = g.cont(a);
    let a = g.group_norm(a, GN_GROUPS, VAE_GN_EPS);
    let a = g.permute(a, [0, 2, 1, 3]);
    let a = g.cont(a);
    let a = g.reshape(a, &[wd, ht, c, t]);
    let s = bias4(g, w.get(&format!("{p}.weight")));
    let b = bias4(g, w.get(&format!("{p}.bias")));
    let a = g.mul(a, s);
    g.add(a, b)
}

/// A 3x3x3 convolution with zero padding of one in every dimension, as one 2D convolution per
/// temporal tap; taps that only see padding are skipped.
fn conv3(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let (wd, ht, c, t) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
    let b = bias4(g, w.get(&format!("{p}.bias")));
    if t == 1 {
        let y = g.conv2d(w.get(&format!("{p}.weight.t1")), x, 1);
        return g.add(y, b);
    }
    let xp = g.pad_ext(x, [0, 0, 0, 1], [0, 0, 0, 1]);
    let xp = g.cont(xp);
    let mut acc: Option<Tn> = None;
    for k in 0..3 {
        let v = g.view_4d(xp, [wd, ht, c, t], xp.nb(1), xp.nb(2), xp.nb(3), k * xp.nb(3));
        let v = g.cont(v);
        let y = g.conv2d(w.get(&format!("{p}.weight.t{k}")), v, 1);
        acc = Some(match acc {
            None => y,
            Some(a) => g.add(a, y),
        });
    }
    let y = acc.expect("three taps");
    g.add(y, b)
}

fn conv1(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let y = g.conv2d(w.get(&format!("{p}.weight.t0")), x, 0);
    let b = bias4(g, w.get(&format!("{p}.bias")));
    g.add(y, b)
}

fn vae_resnet(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let h = gn3d(g, w, &format!("{p}.norm1"), x);
    let h = g.silu(h);
    let h = conv3(g, w, &format!("{p}.conv1"), h);
    let h = gn3d(g, w, &format!("{p}.norm2"), h);
    let h = g.silu(h);
    let h = conv3(g, w, &format!("{p}.conv2"), h);
    let skip = if x.ne(2) == h.ne(2) { x } else { conv1(g, w, &format!("{p}.nin_shortcut"), x) };
    g.add(skip, h)
}

fn vae_attn(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let (wd, ht, c, t) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
    let h = gn3d(g, w, &format!("{p}.norm"), x);
    // [W, H, C, T] -> tokens [C, W*H*T] in (t, h, w) order
    let h = g.permute(h, [0, 1, 3, 2]);
    let h = g.cont(h);
    let h = g.reshape(h, &[wd * ht * t, c]);
    let h = g.permute(h, [1, 0, 2, 3]);
    let h = g.cont(h);
    let proj = |g: &mut Graph, n: &str, a: Tn| {
        let wt = w.get(&format!("{p}.{n}.weight.t0"));
        let wt = g.reshape(wt, &[c, c]);
        g.linear_b(wt, w.get(&format!("{p}.{n}.bias")), a)
    };
    let q = proj(g, "q", h);
    let k = proj(g, "k", h);
    let v = proj(g, "v", h);
    let n = wd * ht * t;
    let vt = g.permute(v, [1, 0, 2, 3]);
    let vt = g.cont(vt);
    let scale = 1.0 / (c as f32).sqrt();
    let mut o: Option<Tn> = None;
    let mut from = 0;
    while from < n {
        let m = ATTN_QUERY_BLOCK.min(n - from);
        let qb = g.view_cols(q, from, m);
        let s = g.linear(k, qb);
        let pr = g.soft_max(s, scale);
        let ob = g.linear(vt, pr);
        o = Some(match o {
            None => ob,
            Some(prev) => g.concat(prev, ob, 1),
        });
        from += m;
    }
    let o = o.expect("at least one cell");
    let o = proj(g, "proj_out", o);
    let o = g.permute(o, [1, 0, 2, 3]);
    let o = g.cont(o);
    let o = g.reshape(o, &[wd, ht, t, c]);
    let o = g.permute(o, [0, 1, 3, 2]);
    let o = g.cont(o);
    g.add(x, o)
}

/// `[W, H, C * 4, T]` with channel `c * 4 + ih * 2 + iw` to `[2W, 2H, C, T]`.
fn pixel_shuffle(g: &mut Graph, z: Tn) -> Tn {
    let (w, h, c4, t) = (z.ne(0), z.ne(1), z.ne(2), z.ne(3));
    let r = c4 / 2 * t;
    let a = g.reshape(z, &[w, h, 2, r]);
    let a = g.permute(a, [1, 2, 0, 3]);
    let a = g.cont(a);
    let a = g.reshape(a, &[2 * w, h, 2, r / 2]);
    let a = g.permute(a, [0, 2, 1, 3]);
    let a = g.cont(a);
    g.reshape(a, &[2 * w, 2 * h, c4 / 4, t])
}

/// Sub-pixel upsampling: a convolution to `factor * cout` channels laid out
/// `(r_t, r_h, r_w, c)` plus the input repeated `factor * cout / cin` times per channel, both
/// rearranged into time (`r_t`, when `temporal`) and space.
fn upsample(g: &mut Graph, w: &Weights, p: &str, x: Tn, cout: i64, temporal: bool) -> Result<Tn> {
    let (wd, ht, cin, t) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
    let rt = if temporal { 2 } else { 1 };
    let f = 4 * rt;
    if (f * cout) % cin != 0 {
        return Err(Error::Config(format!("{p}: {f} x {cout} channels do not repeat {cin} evenly")));
    }
    let h = conv3(g, w, &format!("{p}.conv"), x);
    let rep = f * cout / cin;
    let s = g.reshape(x, &[wd * ht, 1, cin, t]);
    let s = g.repeat_to(s, [wd * ht, rep, cin, t]);
    let s = g.reshape(s, &[wd, ht, cin * rep, t]);
    let y = g.add(h, s);
    // channel s * cout + c  ->  c * f + s
    let y = g.reshape(y, &[wd * ht, cout, f, t]);
    let y = g.permute(y, [0, 2, 1, 3]);
    let y = g.cont(y);
    let y = g.reshape(y, &[wd, ht, cout * f, t]);
    // channel (c * rt + r_t) * 4 + r_h * 2 + r_w  ->  space
    let y = pixel_shuffle(g, y);
    if !temporal {
        return Ok(y);
    }
    // channel c * 2 + r_t  ->  frame t * 2 + r_t
    let (w2, h2) = (y.ne(0), y.ne(1));
    let y = g.reshape(y, &[w2 * h2, 2, cout, t]);
    let y = g.permute(y, [0, 2, 1, 3]);
    let y = g.cont(y);
    Ok(g.reshape(y, &[w2, h2, cout, 2 * t]))
}

/// The latent decoder, resident on one backend.
pub struct HunyuanImage3Vae {
    cfg: HunyuanImage3Config,
    weights: Weights,
    backend: Backend,
}

impl std::fmt::Debug for HunyuanImage3Vae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HunyuanImage3Vae").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl HunyuanImage3Vae {
    fn channels(&self) -> Vec<u64> {
        self.cfg.vae.block_out_channels.iter().rev().copied().collect()
    }

    /// Load from the image-side file. `exact` keeps convolution kernels in f32.
    ///
    /// # Errors
    /// A missing file or weight, or an unsupported configuration.
    pub fn load(path: &Path, backend: Backend, exact: bool) -> Result<Self> {
        let (files, cfg) = open(path)?;
        let v = &cfg.vae;
        let ch: Vec<u64> = v.block_out_channels.iter().rev().copied().collect();
        if ch[0] % v.latent_channels != 0 {
            return Err(Error::Config("decoder input width is not a multiple of the latent channels".into()));
        }
        let mut h = Hosts { files: &files, kt: if exact { WType::F32 } else { WType::F16 }, v: Vec::new() };
        let d = "vae.decoder";
        let resnet = |h: &mut Hosts, p: &str, cin: u64, cout: u64| -> Result<()> {
            h.norm(&format!("{p}.norm1"), cin)?;
            h.conv3d(&format!("{p}.conv1"), cout, cin, 3)?;
            h.norm(&format!("{p}.norm2"), cout)?;
            h.conv3d(&format!("{p}.conv2"), cout, cout, 3)?;
            if cin != cout {
                h.conv3d(&format!("{p}.nin_shortcut"), cout, cin, 1)?;
            }
            Ok(())
        };
        let top = ch[0];
        h.conv3d(&format!("{d}.conv_in"), top, v.latent_channels, 3)?;
        resnet(&mut h, &format!("{d}.mid.block_1"), top, top)?;
        h.norm(&format!("{d}.mid.attn_1.norm"), top)?;
        for n in ["q", "k", "v", "proj_out"] {
            h.conv3d(&format!("{d}.mid.attn_1.{n}"), top, top, 1)?;
        }
        resnet(&mut h, &format!("{d}.mid.block_2"), top, top)?;
        let ups = v.ffactor_spatial.trailing_zeros() as usize;
        let tups = v.ffactor_temporal.trailing_zeros() as usize;
        let mut cin = top;
        for (i, &c) in ch.iter().enumerate() {
            for b in 0..=v.layers_per_block {
                resnet(&mut h, &format!("{d}.up.{i}.block.{b}"), cin, c)?;
                cin = c;
            }
            if i < ups {
                let next = ch[i + 1];
                let f = if i < tups { 8 } else { 4 };
                h.conv3d(&format!("{d}.up.{i}.upsample.conv"), next * f, cin, 3)?;
                cin = next;
            }
        }
        h.norm(&format!("{d}.norm_out"), cin)?;
        h.conv3d(&format!("{d}.conv_out"), v.out_channels, cin, 3)?;
        let weights = Weights::from_host(&backend, &h.v)?;
        Ok(Self { cfg, weights, backend })
    }

    /// Decode one latent (planar `[C, h, w]`, as the sampler produced it: scaled) to planar RGB
    /// `[3, H, W]` in `[-1, 1]`.
    ///
    /// # Errors
    /// A wrong input size or a failed compute.
    pub fn decode(&self, latent: &[f32], (lh, lw): (usize, usize)) -> Result<Vec<f32>> {
        let v = &self.cfg.vae;
        let lc = v.latent_channels as usize;
        if latent.len() != lc * lh * lw {
            return Err(Error::Config(format!("decoder takes {} floats, got {}", lc * lh * lw, latent.len())));
        }
        let shift = v.shift_factor.unwrap_or(0.0) as f32;
        let inv = 1.0 / v.scaling_factor as f32;
        let z: Vec<f32> = latent.iter().map(|x| x * inv + shift).collect();
        let ch = self.channels();
        let w = &self.weights;
        let d = "vae.decoder";
        let mut g = Graph::new(&self.backend)?;
        let zin = g.input(sys::GGML_TYPE_F32, &[lw as i64, lh as i64, lc as i64, 1]);
        let rep = ch[0] as i64 / lc as i64;
        let x = conv3(&mut g, w, &format!("{d}.conv_in"), zin);
        let s = g.reshape(zin, &[(lw * lh) as i64, 1, lc as i64, 1]);
        let s = g.repeat_to(s, [(lw * lh) as i64, rep, lc as i64, 1]);
        let s = g.reshape(s, &[lw as i64, lh as i64, ch[0] as i64, 1]);
        let mut x = g.add(x, s);
        x = vae_resnet(&mut g, w, &format!("{d}.mid.block_1"), x);
        x = vae_attn(&mut g, w, &format!("{d}.mid.attn_1"), x);
        x = vae_resnet(&mut g, w, &format!("{d}.mid.block_2"), x);
        let ups = v.ffactor_spatial.trailing_zeros() as usize;
        let tups = v.ffactor_temporal.trailing_zeros() as usize;
        for i in 0..ch.len() {
            for b in 0..=v.layers_per_block {
                x = vae_resnet(&mut g, w, &format!("{d}.up.{i}.block.{b}"), x);
            }
            if i < ups {
                x = upsample(&mut g, w, &format!("{d}.up.{i}.upsample"), x, ch[i + 1] as i64, i < tups)?;
            }
        }
        x = gn3d(&mut g, w, &format!("{d}.norm_out"), x);
        x = g.silu(x);
        x = conv3(&mut g, w, &format!("{d}.conv_out"), x);
        // The last frame.
        let t = x.ne(3);
        let last = g.view_4d(x, [x.ne(0), x.ne(1), x.ne(2), 1], x.nb(1), x.nb(2), x.nb(3), (t - 1) as usize * x.nb(3));
        let out = g.cont(last);
        g.finish(&[out])?;
        g.set_f32(zin, &z);
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// The configuration read with the weights.
    #[must_use]
    pub fn config(&self) -> &HunyuanImage3Config {
        &self.cfg
    }
}
