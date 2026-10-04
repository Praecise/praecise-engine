//! Causal 3D video autoencoder with residual down- and up-sampling blocks
//! and a 2x2 pixel patch (the 48-channel, 16x spatial, 4x temporal layout).
//!
//! Activations are laid out `[W, H, C, T]`: every causal convolution with
//! three temporal taps is three 2D convolutions over shifted frames, summed.
//! The decoder streams one latent frame at a time; each such convolution
//! carries its last two input frames to the next frame in a resident cache,
//! which is exactly the zero-padded causal convolution over the whole
//! sequence. The first latent frame decodes to one video frame, every later
//! one to four.
//!
//! The encoder serves single frames (a conditioning image): with two zero
//! frames of causal padding only the last temporal tap of each convolution
//! sees data, and the temporal downsampling convolutions do not run on a
//! first frame.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::pipeline::RgbImage;
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Norm epsilon: the channel norm divides by the L2 norm (clamped at 1e-12),
/// which the RMS norm matches up to this term.
const NORM_EPS: f32 = 1e-24;
/// Spatial patch side folded into channels before the encoder.
const PATCH: usize = 2;

/// Autoencoder configuration, read from `vae/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct WanVaeConfig {
    /// Encoder base width.
    pub base_dim: u64,
    /// Decoder base width.
    #[serde(default)]
    pub decoder_base_dim: Option<u64>,
    /// Latent channels.
    pub z_dim: u64,
    /// Width multiples per resolution.
    pub dim_mult: Vec<u64>,
    /// Residual blocks per resolution (the decoder has one more).
    pub num_res_blocks: u64,
    /// Resolutions with attention (none supported).
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    /// Which downsampling stages also halve time.
    pub temperal_downsample: Vec<bool>,
    /// Residual (averaging/duplicating) shortcuts around each stage.
    #[serde(default)]
    pub is_residual: bool,
    /// Input channels (RGB times the patch area).
    pub in_channels: u64,
    /// Output channels.
    pub out_channels: u64,
    /// Pixel patch side.
    #[serde(default)]
    pub patch_size: Option<u64>,
    /// Pixels per latent, spatially.
    pub scale_factor_spatial: u64,
    /// Frames per latent frame (after the first).
    pub scale_factor_temporal: u64,
    /// Per-channel latent mean.
    pub latents_mean: Vec<f32>,
    /// Per-channel latent standard deviation.
    pub latents_std: Vec<f32>,
}

impl WanVaeConfig {
    /// Refuse variants this implementation has not been checked against.
    ///
    /// # Errors
    /// [`Error::Config`] naming the first unsupported setting.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("autoencoder: {m}")));
        if !self.is_residual || self.patch_size != Some(PATCH as u64) || !self.attn_scales.is_empty() {
            return bad("only the residual layout with a 2x2 patch and no attention stages is implemented");
        }
        if self.dim_mult.len() != 4 || self.temperal_downsample.len() != 3 || self.num_res_blocks == 0 {
            return bad("expected four resolutions and three downsampling stages");
        }
        if self.in_channels != 3 * (PATCH * PATCH) as u64 || self.out_channels != self.in_channels {
            return bad("expected RGB in and out, patched");
        }
        if self.latents_mean.len() != self.z_dim as usize || self.latents_std.len() != self.z_dim as usize {
            return bad("latent statistics do not match the latent channels");
        }
        let temporal = self.temperal_downsample.iter().filter(|t| **t).count() as u32;
        if self.scale_factor_spatial != 8 * PATCH as u64 || self.scale_factor_temporal != 1 << temporal {
            return bad("scale factors disagree with the stages");
        }
        Ok(())
    }

    fn enc_dims(&self) -> Vec<u64> {
        std::iter::once(1).chain(self.dim_mult.iter().copied()).map(|m| self.base_dim * m).collect()
    }

    fn dec_dims(&self) -> Vec<u64> {
        let b = self.decoder_base_dim.unwrap_or(self.base_dim);
        let last = *self.dim_mult.last().expect("validated");
        std::iter::once(last).chain(self.dim_mult.iter().rev().copied()).map(|m| b * m).collect()
    }

    /// `(in, out, temporal, spatial)` per encoder stage.
    fn down_stages(&self) -> Vec<(u64, u64, bool, bool)> {
        let d = self.enc_dims();
        let n = self.dim_mult.len();
        (0..n).map(|i| (d[i], d[i + 1], i != n - 1 && self.temperal_downsample[i], i != n - 1)).collect()
    }

    /// `(in, out, temporal, spatial)` per decoder stage.
    fn up_stages(&self) -> Vec<(u64, u64, bool, bool)> {
        let d = self.dec_dims();
        let n = self.dim_mult.len();
        let up: Vec<bool> = self.temperal_downsample.iter().rev().copied().collect();
        (0..n).map(|i| (d[i], d[i + 1], i != n - 1 && up[i], i != n - 1)).collect()
    }

    /// Every weight both directions read, re-laid out: one `[out, in, kh,
    /// kw]` kernel per temporal tap (`.t0` to `.t2`), 1x1 attention projections
    /// as matrices, and the fixed averaging kernels of the encoder's
    /// shortcuts (`avg_shortcut` for the later frame of a pair, `.prev` for
    /// the earlier one where the stage halves time).
    ///
    /// # Errors
    /// Missing or mis-shaped tensors.
    pub fn host_tensors(&self, files: &SafeTensors, exact: bool) -> Result<Vec<HostTensor>> {
        let kt = if exact { WType::F32 } else { WType::F16 };
        let mut h = Hosts { files, v: Vec::new(), kt };
        // Encoder.
        let z2 = 2 * self.z_dim;
        let ed = self.enc_dims();
        h.causal_taps("encoder.conv_in", self.in_channels, ed[0])?;
        for (i, (cin, cout, temporal, spatial)) in self.down_stages().into_iter().enumerate() {
            let p = format!("encoder.down_blocks.{i}");
            let mut c = cin;
            for r in 0..self.num_res_blocks {
                h.resnet(&format!("{p}.resnets.{r}"), c, cout)?;
                c = cout;
            }
            if spatial {
                h.conv2d(&format!("{p}.downsampler.resample.1"), cout, cout)?;
                if temporal {
                    h.time_conv(&format!("{p}.downsampler.time_conv"), cout, cout)?;
                }
            }
            let phases: &[(usize, &str)] = if temporal { &[(1, ""), (0, ".prev")] } else { &[(0, "")] };
            for &(phase, suffix) in phases {
                if let Some(k) = avg_down_kernel(cin, cout, temporal, spatial, phase) {
                    h.v.push(HostTensor { name: format!("{p}.avg_shortcut{suffix}"), shape: vec![cout, cin, 2, 2], ty: self.kernel_type(exact), data: k });
                }
            }
        }
        let top = *ed.last().expect("five widths");
        h.mid("encoder.mid_block", top)?;
        h.gamma("encoder.norm_out", top)?;
        h.causal_taps("encoder.conv_out", top, z2)?;
        h.pointwise("quant_conv", z2, z2)?;
        // Decoder.
        let dd = self.dec_dims();
        h.pointwise("post_quant_conv", self.z_dim, self.z_dim)?;
        h.causal_taps("decoder.conv_in", self.z_dim, dd[0])?;
        h.mid("decoder.mid_block", dd[0])?;
        for (i, (cin, cout, temporal, spatial)) in self.up_stages().into_iter().enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            let mut c = cin;
            for r in 0..=self.num_res_blocks {
                h.resnet(&format!("{p}.resnets.{r}"), c, cout)?;
                c = cout;
            }
            if spatial {
                if temporal {
                    h.time_conv(&format!("{p}.upsampler.time_conv"), cout, 2 * cout)?;
                }
                h.conv2d(&format!("{p}.upsampler.resample.1"), cout, cout)?;
            }
        }
        let last = *dd.last().expect("five widths");
        h.gamma("decoder.norm_out", last)?;
        h.causal_taps("decoder.conv_out", last, self.out_channels)?;
        Ok(h.v)
    }

    fn kernel_type(&self, exact: bool) -> WType {
        if exact { WType::F32 } else { WType::F16 }
    }

    /// The decoder's per-convolution caches for a latent of `lw` x `lh`:
    /// `[2, C, H, W]` each, named by the convolution.
    #[must_use]
    pub fn decoder_cache_specs(&self, lw: i64, lh: i64) -> Vec<WeightSpec> {
        let mut v = Vec::new();
        let mut add = |name: String, c: u64, w: i64, h: i64| v.push(WeightSpec::new(name, &[2, c, h as u64, w as u64], WType::F32));
        let dd = self.dec_dims();
        let (mut w, mut h) = (lw, lh);
        add("decoder.conv_in".into(), self.z_dim, w, h);
        for r in 0..2 {
            add(format!("decoder.mid_block.resnets.{r}.conv1"), dd[0], w, h);
            add(format!("decoder.mid_block.resnets.{r}.conv2"), dd[0], w, h);
        }
        for (i, (cin, cout, temporal, spatial)) in self.up_stages().into_iter().enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            let mut c = cin;
            for r in 0..=self.num_res_blocks {
                add(format!("{p}.resnets.{r}.conv1"), c, w, h);
                add(format!("{p}.resnets.{r}.conv2"), cout, w, h);
                c = cout;
            }
            if spatial {
                if temporal {
                    add(format!("{p}.upsampler.time_conv"), cout, w, h);
                }
                w *= 2;
                h *= 2;
            }
        }
        add("decoder.conv_out".into(), *dd.last().expect("five widths"), w, h);
        v
    }

    /// The encoder's resident frame caches for a `width` x `height` video:
    /// the last two inputs of every causal convolution and the last frame
    /// before every temporal downsampling, zeroed before the first chunk.
    #[must_use]
    pub fn encoder_cache_specs(&self, width: i64, height: i64) -> Vec<WeightSpec> {
        let mut v = Vec::new();
        let mut add = |name: String, t: u64, c: u64, w: i64, h: i64| v.push(WeightSpec::new(name, &[t, c, h as u64, w as u64], WType::F32));
        let ed = self.enc_dims();
        let p = PATCH as i64;
        let (mut w, mut h) = (width / p, height / p);
        add("encoder.conv_in".into(), 2, self.in_channels, w, h);
        for (i, (cin, cout, temporal, spatial)) in self.down_stages().into_iter().enumerate() {
            let pre = format!("encoder.down_blocks.{i}");
            let mut c = cin;
            for r in 0..self.num_res_blocks {
                add(format!("{pre}.resnets.{r}.conv1"), 2, c, w, h);
                add(format!("{pre}.resnets.{r}.conv2"), 2, cout, w, h);
                c = cout;
            }
            if spatial {
                w /= 2;
                h /= 2;
                if temporal {
                    add(format!("{pre}.downsampler.time_conv"), 1, cout, w, h);
                }
            }
        }
        let top = *ed.last().expect("five widths");
        for r in 0..2 {
            add(format!("encoder.mid_block.resnets.{r}.conv1"), 2, top, w, h);
            add(format!("encoder.mid_block.resnets.{r}.conv2"), 2, top, w, h);
        }
        add("encoder.conv_out".into(), 2, top, w, h);
        v
    }
}

/// One temporal phase of the encoder's averaging shortcut as a 2x2, stride-2
/// convolution kernel `[out, in, 2, 2]`: output channel `o` averages the
/// `group` space-to-depth channels `o * group ..`, and this kernel keeps the
/// ones reading frame `phase` of each pair (the only phase when the stage
/// keeps time; on a lone first frame phase 0 is the zero padding). `None`
/// when the shortcut is the identity.
fn avg_down_kernel(cin: u64, cout: u64, temporal: bool, spatial: bool, phase: usize) -> Option<Vec<f32>> {
    let (ft, fs) = (if temporal { 2 } else { 1 }, if spatial { 2 } else { 1 });
    if ft == 1 && fs == 1 && cin == cout {
        return None;
    }
    assert_eq!(fs, 2, "a channel-changing shortcut always downsamples space");
    let (cin, cout) = (cin as usize, cout as usize);
    let factor = ft * fs * fs;
    let group = cin * factor / cout;
    let mut k = vec![0f32; cout * cin * 4];
    for o in 0..cout {
        for gi in 0..group {
            let j = o * group + gi;
            let (c, sub) = (j / factor, j % factor);
            let (it, rem) = (sub / (fs * fs), sub % (fs * fs));
            if it != phase {
                continue;
            }
            let (ih, iw) = (rem / fs, rem % fs);
            k[((o * cin + c) * 2 + ih) * 2 + iw] += 1.0 / group as f32;
        }
    }
    Some(k)
}

struct Hosts<'a> {
    files: &'a SafeTensors,
    v: Vec<HostTensor>,
    kt: WType,
}

impl Hosts<'_> {
    fn bias(&mut self, p: &str, c: u64) -> Result<()> {
        let b = self.files.require(&format!("{p}.bias"), &[c])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.bias"), shape: vec![c], ty: WType::F32, data: b });
        Ok(())
    }

    fn taps(&mut self, p: &str, cin: u64, cout: u64, kh: u64, which: &[(usize, String)]) -> Result<()> {
        let w = self.files.require(&format!("{p}.weight"), &[cout, cin, 3, kh, kh])?.to_f32();
        let (ci, co, k2) = (cin as usize, cout as usize, (kh * kh) as usize);
        for (t, name) in which {
            let mut d = Vec::with_capacity(co * ci * k2);
            for o in 0..co {
                for i in 0..ci {
                    let base = ((o * ci + i) * 3 + t) * k2;
                    d.extend_from_slice(&w[base..base + k2]);
                }
            }
            self.v.push(HostTensor { name: name.clone(), shape: vec![cout, cin, kh, kh], ty: self.kt, data: d });
        }
        self.bias(p, cout)
    }

    fn causal_taps(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        self.taps(p, cin, cout, 3, &(0..3).map(|t| (t, format!("{p}.weight.t{t}"))).collect::<Vec<_>>())
    }

    fn time_conv(&mut self, p: &str, c: u64, cout: u64) -> Result<()> {
        let w = self.files.require(&format!("{p}.weight"), &[cout, c, 3, 1, 1])?.to_f32();
        let (ci, co) = (c as usize, cout as usize);
        for t in 0..3 {
            let d: Vec<f32> = (0..co * ci).map(|j| w[j * 3 + t]).collect();
            self.v.push(HostTensor { name: format!("{p}.weight.t{t}"), shape: vec![cout, c, 1, 1], ty: self.kt, data: d });
        }
        self.bias(p, cout)
    }

    fn pointwise(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.files.require(&format!("{p}.weight"), &[cout, cin, 1, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, 1, 1], ty: self.kt, data: w });
        self.bias(p, cout)
    }

    fn conv2d(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.files.require(&format!("{p}.weight"), &[cout, cin, 3, 3])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, 3, 3], ty: self.kt, data: w });
        self.bias(p, cout)
    }

    fn gamma(&mut self, p: &str, c: u64) -> Result<()> {
        let g = self.files.require(&format!("{p}.gamma"), &[c, 1, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.gamma"), shape: vec![c], ty: WType::F32, data: g });
        Ok(())
    }

    fn resnet(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        self.gamma(&format!("{p}.norm1"), cin)?;
        self.gamma(&format!("{p}.norm2"), cout)?;
        for (n, i) in [("conv1", cin), ("conv2", cout)] {
            self.causal_taps(&format!("{p}.{n}"), i, cout)?;
        }
        if cin != cout {
            self.pointwise(&format!("{p}.conv_shortcut"), cin, cout)?;
        }
        Ok(())
    }

    fn mid(&mut self, p: &str, c: u64) -> Result<()> {
        self.resnet(&format!("{p}.resnets.0"), c, c)?;
        let a = format!("{p}.attentions.0");
        let g = self.files.require(&format!("{a}.norm.gamma"), &[c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.norm.gamma"), shape: vec![c], ty: WType::F32, data: g });
        let qkv = self.files.require(&format!("{a}.to_qkv.weight"), &[3 * c, c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.to_qkv.weight"), shape: vec![3 * c, c], ty: WType::F32, data: qkv });
        self.bias(&format!("{a}.to_qkv"), 3 * c)?;
        let proj = self.files.require(&format!("{a}.proj.weight"), &[c, c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.proj.weight"), shape: vec![c, c], ty: WType::F32, data: proj });
        self.bias(&format!("{a}.proj"), c)?;
        self.resnet(&format!("{p}.resnets.1"), c, c)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
    /// Resident frame caches, read and updated in place by every run.
    cache: &'a Weights,
    /// Gather indices to set before each run.
    feeds: Vec<(Tn, Vec<i32>)>,
}

fn bias4(g: &mut Graph, b: Tn) -> Tn {
    g.reshape(b, &[1, 1, b.ne(0), 1])
}

impl Net<'_, '_> {
    fn wt(&self, n: &str) -> Tn {
        self.w.get(n)
    }

    fn add_bias(&mut self, y: Tn, p: &str) -> Tn {
        let b = self.wt(&format!("{p}.bias"));
        let b = bias4(self.g, b);
        self.g.add(y, b)
    }

    /// A causal 3x3x3 (or 3x1x1 with `pad` 0) convolution over `[W, H, C, T]`
    /// after the two cached frames.
    fn causal(&mut self, p: &str, x: Tn, pad: i32) -> Tn {
        let cache = self.cache.get(p);
        let full = self.g.concat(cache, x, 3);
        let t = x.ne(3);
        let shape = [x.ne(0), x.ne(1), x.ne(2), t];
        let mut acc: Option<Tn> = None;
        for k in 0..3 {
            let v = self.g.view_4d(full, shape, full.nb(1), full.nb(2), full.nb(3), k as usize * full.nb(3));
            let kern = self.wt(&format!("{p}.weight.t{k}"));
            let y = self.g.conv2d(kern, v, pad);
            acc = Some(match acc {
                None => y,
                Some(a) => self.g.add(a, y),
            });
        }
        let tail = self.g.view_4d(full, [x.ne(0), x.ne(1), x.ne(2), 2], full.nb(1), full.nb(2), full.nb(3), t as usize * full.nb(3));
        self.g.copy_into(tail, cache);
        let y = acc.expect("three taps");
        self.add_bias(y, p)
    }

    /// Every second frame of `x` from frame `start`: `count` frames.
    fn every_other(&mut self, x: Tn, start: i64, count: i64) -> Tn {
        let v = self.g.view_4d(x, [x.ne(0), x.ne(1), x.ne(2), count], x.nb(1), x.nb(2), 2 * x.nb(3), start as usize * x.nb(3));
        self.g.cont(v)
    }

    /// The stride-2 temporal 3x1x1 convolution of a downsampling stage over
    /// the cached last frame then `x` (an even number of frames); halves time
    /// and keeps the last frame of `x` for the next chunk.
    fn time_down(&mut self, p: &str, x: Tn) -> Tn {
        let cache = self.cache.get(p);
        let full = self.g.concat(cache, x, 3);
        let t = x.ne(3);
        let mut acc: Option<Tn> = None;
        for k in 0..3 {
            let v = self.every_other(full, k, t / 2);
            let kern = self.wt(&format!("{p}.weight.t{k}"));
            let y = self.g.conv2d(kern, v, 0);
            acc = Some(match acc {
                None => y,
                Some(a) => self.g.add(a, y),
            });
        }
        // Copy from `full`, not `x`: the copy then depends on the concat
        // that reads the cache, so it cannot overwrite the cache first.
        self.keep_last(p, full);
        let y = acc.expect("three taps");
        self.add_bias(y, p)
    }

    /// Store the last frame of `x` in the one-frame cache `p`.
    fn keep_last(&mut self, p: &str, x: Tn) {
        let cache = self.cache.get(p);
        let last = self.g.view_4d(x, [x.ne(0), x.ne(1), x.ne(2), 1], x.nb(1), x.nb(2), x.nb(3), (x.ne(3) - 1) as usize * x.nb(3));
        self.g.copy_into(last, cache);
    }

    fn pointwise(&mut self, p: &str, x: Tn) -> Tn {
        let k = self.wt(&format!("{p}.weight"));
        let y = self.g.conv2d(k, x, 0);
        self.add_bias(y, p)
    }

    /// Channel RMS norm with gain, then SiLU, on `[W, H, C, T]`.
    fn norm_silu(&mut self, p: &str, x: Tn) -> Tn {
        let c = self.g.permute(x, [1, 2, 0, 3]);
        let c = self.g.cont(c);
        let n = self.g.rms_norm(c, NORM_EPS);
        let n = self.g.mul(n, self.wt(&format!("{p}.gamma")));
        let n = self.g.silu(n);
        let back = self.g.permute(n, [2, 0, 1, 3]);
        self.g.cont(back)
    }

    fn resnet(&mut self, p: &str, x: Tn, cin: u64, cout: u64) -> Tn {
        let h = if cin == cout { x } else { self.pointwise(&format!("{p}.conv_shortcut"), x) };
        let y = self.norm_silu(&format!("{p}.norm1"), x);
        let y = self.causal(&format!("{p}.conv1"), y, 1);
        let y = self.norm_silu(&format!("{p}.norm2"), y);
        let y = self.causal(&format!("{p}.conv2"), y, 1);
        self.g.add(y, h)
    }

    /// Single-head self-attention within each frame.
    fn attention(&mut self, p: &str, x: Tn) -> Tn {
        let (w, h, c, t) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
        let n = w * h;
        let xc = self.g.permute(x, [1, 2, 0, 3]);
        let xc = self.g.cont(xc);
        let xn = self.g.rms_norm(xc, NORM_EPS);
        let xn = self.g.mul(xn, self.wt(&format!("{p}.norm.gamma")));
        let xn = self.g.reshape(xn, &[c, n, t]);
        let qkv = self.g.linear_b(self.wt(&format!("{p}.to_qkv.weight")), self.wt(&format!("{p}.to_qkv.bias")), xn);
        let es = qkv.nb(0);
        let part = |g: &mut Graph, i: i64| {
            let v = g.view_4d(qkv, [c, n, t, 1], qkv.nb(1), qkv.nb(2), qkv.nb(3), (i * c) as usize * es);
            g.cont(v)
        };
        let q = part(self.g, 0);
        let k = part(self.g, 1);
        let v = part(self.g, 2);
        let o = self.g.attention_exact(q, k, v, None, 1.0 / (c as f32).sqrt());
        let o = self.g.permute(o, [0, 2, 1, 3]);
        let o = self.g.cont(o);
        let o = self.g.linear_b(self.wt(&format!("{p}.proj.weight")), self.wt(&format!("{p}.proj.bias")), o);
        let o = self.g.reshape(o, &[c, w, h, t]);
        let o = self.g.permute(o, [2, 0, 1, 3]);
        let o = self.g.cont(o);
        self.g.add(o, x)
    }

    fn mid(&mut self, p: &str, x: Tn, c: u64) -> Tn {
        let x = self.resnet(&format!("{p}.resnets.0"), x, c, c);
        let x = self.attention(&format!("{p}.attentions.0"), x);
        self.resnet(&format!("{p}.resnets.1"), x, c, c)
    }

    /// The decoder's duplicating shortcut: output channel `o` at sub-position
    /// `(it, ih, iw)` repeats input channel `(o * factor + it * 4 + ih * 2 +
    /// iw) / repeats`. Frames interleave as `t * ft + it`; on the first chunk
    /// only the last temporal phase is kept.
    fn dup_up(&mut self, x: Tn, cout: u64, temporal: bool, first: bool) -> Tn {
        let (w, h, cin, t) = (x.ne(0), x.ne(1), x.ne(2) as usize, x.ne(3));
        let cout = cout as usize;
        let ft = if temporal { 2 } else { 1 };
        let factor = ft * 4;
        let repeats = cout * factor / cin;
        let flat = self.g.reshape(x, &[w * h, cin as i64, t]);
        let phases: Vec<usize> = if first { vec![ft - 1] } else { (0..ft).collect() };
        let mut out: Option<Tn> = None;
        for it in phases {
            let ids: Vec<i32> = (0..cout * 4).map(|s| ((s / 4 * factor + it * 4 + s % 4) / repeats) as i32).collect();
            let idt = self.g.input(sys::GGML_TYPE_I32, &[(cout * 4) as i64, t]);
            self.feeds.push((idt, ids.iter().copied().cycle().take(cout * 4 * t as usize).collect()));
            let z = self.g.get_rows(flat, idt);
            let z = self.g.reshape(z, &[w, h, (cout * 4) as i64, t]);
            let y = pixel_shuffle(self.g, z);
            let y = self.g.reshape(y, &[4 * w * h * cout as i64, 1, t]);
            out = Some(match out {
                None => y,
                Some(a) => self.g.concat(a, y, 1),
            });
        }
        let n = out.expect("at least one phase");
        let frames = n.ne(1) * t;
        self.g.reshape(n, &[2 * w, 2 * h, cout as i64, frames])
    }
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

/// Inputs and outputs of one streamed encoder chunk.
#[derive(Debug, Clone)]
pub struct EncodeIo {
    /// Patched frames `[W / 2, H / 2, 12, T]` in `[-1, 1]`: one on the first
    /// chunk, four after.
    pub pixels: Tn,
    /// One latent frame's mean `[W / 16, H / 16, z]` (unnormalised).
    pub mean: Tn,
}

/// One chunk of a `width` x `height` video through the encoder, reading and
/// updating `cache` ([`WanVaeConfig::encoder_cache_specs`]). `first` takes
/// the lone first frame (time padded with zeros), otherwise the next four
/// frames; either way the chunk yields one latent frame.
#[must_use]
pub fn build_encoder(g: &mut Graph, cfg: &WanVaeConfig, w: &Weights, cache: &Weights, width: i64, height: i64, first: bool) -> EncodeIo {
    let p = PATCH as i64;
    let frames = if first { 1 } else { 4 };
    let pixels = g.input(sys::GGML_TYPE_F32, &[width / p, height / p, cfg.in_channels as i64, frames]);
    let mut n = Net { g, w, cache, feeds: Vec::new() };
    let mut x = n.causal("encoder.conv_in", pixels, 1);
    for (i, (cin, cout, temporal, spatial)) in cfg.down_stages().into_iter().enumerate() {
        let pre = format!("encoder.down_blocks.{i}");
        let skip = x;
        let mut c = cin;
        for r in 0..cfg.num_res_blocks {
            x = n.resnet(&format!("{pre}.resnets.{r}"), x, c, cout);
            c = cout;
        }
        if spatial {
            let padded = n.g.pad_end(x, 1, 1);
            let k = n.wt(&format!("{pre}.downsampler.resample.1.weight"));
            let y = n.g.conv2d_stride2(k, padded);
            x = n.add_bias(y, &format!("{pre}.downsampler.resample.1"));
            if temporal {
                let tc = format!("{pre}.downsampler.time_conv");
                if first {
                    n.keep_last(&tc, x);
                } else {
                    x = n.time_down(&tc, x);
                }
            }
        }
        let s = if avg_down_kernel(cin, cout, temporal, spatial, 0).is_none() {
            skip
        } else if temporal && !first {
            let half = skip.ne(3) / 2;
            let even = n.every_other(skip, 0, half);
            let odd = n.every_other(skip, 1, half);
            let k0 = n.wt(&format!("{pre}.avg_shortcut.prev"));
            let k1 = n.wt(&format!("{pre}.avg_shortcut"));
            let a = n.g.conv2d_strided(k0, even, 2);
            let b = n.g.conv2d_strided(k1, odd, 2);
            n.g.add(a, b)
        } else {
            let k = n.wt(&format!("{pre}.avg_shortcut"));
            n.g.conv2d_strided(k, skip, 2)
        };
        x = n.g.add(x, s);
    }
    let top = *cfg.enc_dims().last().expect("five widths");
    x = n.mid("encoder.mid_block", x, top);
    x = n.norm_silu("encoder.norm_out", x);
    x = n.causal("encoder.conv_out", x, 1);
    x = n.pointwise("quant_conv", x);
    let z = cfg.z_dim as i64;
    let mean = n.g.view_4d(x, [x.ne(0), x.ne(1), z, 1], x.nb(1), x.nb(2), x.nb(3), 0);
    let mean = n.g.cont(mean);
    EncodeIo { pixels, mean }
}

/// Encode `1 + 4k` frames of one size to normalised latents
/// `[z][1 + k][H/16][W/16]`: the first frame alone, then four at a time
/// through the encoder's frame caches.
///
/// # Errors
/// An empty clip, a frame count not `1 + 4k`, or backend failures.
pub fn encode_frames(backend: &Backend, cfg: &WanVaeConfig, vae: &Weights, frames: &[RgbImage]) -> Result<Vec<f32>> {
    let Some(head) = frames.first() else {
        return Err(Error::Request("no frames to encode".into()));
    };
    if (frames.len() - 1) % 4 != 0 {
        return Err(Error::Request(format!("{} frames to encode; the encoder takes 1 + 4k", frames.len())));
    }
    let (w, h) = (head.width as usize, head.height as usize);
    let patched = |img: &RgbImage| {
        let mut px = vec![0f32; 3 * w * h];
        for (i, p) in img.rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                px[c * w * h + i] = f32::from(p[c]) / 127.5 - 1.0;
            }
        }
        patchify(&px, w, h)
    };
    let cache = Weights::zeros(&backend, &cfg.encoder_cache_specs(w as i64, h as i64))?;
    let lt = 1 + (frames.len() - 1) / 4;
    let mut graphs: Vec<(Graph, EncodeIo)> = Vec::new();
    for first in [true, false].into_iter().take(lt.min(2)) {
        let mut g = Graph::new(&backend)?;
        let io = build_encoder(&mut g, cfg, vae, &cache, w as i64, h as i64, first);
        g.finish(&[io.mean])?;
        graphs.push((g, io));
    }
    let mut per_frame: Vec<Vec<f32>> = Vec::with_capacity(lt);
    for t in 0..lt {
        let chunk: &[RgbImage] = if t == 0 { &frames[..1] } else { &frames[1 + 4 * (t - 1)..1 + 4 * t] };
        let mut input = Vec::new();
        for f in chunk {
            input.extend(patched(f));
        }
        let (g, io) = &graphs[usize::from(t > 0)];
        g.set_f32(io.pixels, &input);
        g.compute()?;
        per_frame.push(g.read_f32(io.mean));
    }
    let z = cfg.z_dim as usize;
    let plane = per_frame[0].len() / z;
    let mut out = vec![0f32; z * lt * plane];
    for (t, mean) in per_frame.iter().enumerate() {
        for c in 0..z {
            let (m, inv) = (cfg.latents_mean[c], 1.0 / cfg.latents_std[c]);
            for i in 0..plane {
                out[(c * lt + t) * plane + i] = (mean[c * plane + i] - m) * inv;
            }
        }
    }
    Ok(out)
}

/// Decode normalised latents `[z][T][H][W]` to frames `[F][3][H*16][W*16]`
/// in `[-1, 1]`, one latent frame at a time through the decoder's caches.
///
/// # Errors
/// Backend failures.
pub fn decode(backend: &Backend, cfg: &WanVaeConfig, vae: &Weights, latents: &[f32], (lt, lh, lw): (usize, usize, usize)) -> Result<Vec<f32>> {
    let z = cfg.z_dim as usize;
    let plane = lh * lw;
    let cache = Weights::zeros(&backend, &cfg.decoder_cache_specs(lw as i64, lh as i64))?;
    let mut out = Vec::new();
    let mut graphs: Vec<(Graph, DecodeIo)> = Vec::new();
    for first in [true, false].into_iter().take(lt.min(2)) {
        let mut g = Graph::new(&backend)?;
        let io = build_decoder(&mut g, cfg, vae, &cache, lw as i64, lh as i64, first);
        g.finish(&[io.out])?;
        graphs.push((g, io));
    }
    for t in 0..lt {
        let mut frame = vec![0f32; z * plane];
        for c in 0..z {
            let (m, inv) = (cfg.latents_mean[c], 1.0 / cfg.latents_std[c]);
            for i in 0..plane {
                frame[c * plane + i] = latents[(c * lt + t) * plane + i] / inv + m;
            }
        }
        let (g, io) = &graphs[usize::from(t > 0)];
        g.set_f32(io.latent, &frame);
        for (tn, ids) in &io.feeds {
            g.set_i32(*tn, ids);
        }
        g.compute()?;
        let x = g.read_f32(io.out);
        out.extend(unpatchify(&x, io.out.ne(0) as usize, io.out.ne(1) as usize, io.out.ne(3) as usize));
    }
    Ok(out)
}

/// Inputs and outputs of one streamed decoder chunk.
#[derive(Debug)]
pub struct DecodeIo {
    /// One latent frame `[W / 16, H / 16, z]` (unnormalised).
    pub latent: Tn,
    /// Patched frames `[W / 2, H / 2, 12, T]`: one on the first chunk, four
    /// after.
    pub out: Tn,
    /// Gather indices to set before each run.
    pub feeds: Vec<(Tn, Vec<i32>)>,
}

/// One latent frame through the decoder, reading and updating `cache`
/// ([`WanVaeConfig::decoder_cache_specs`]). `first` selects the first-frame
/// variant: no temporal upsampling.
#[must_use]
pub fn build_decoder(g: &mut Graph, cfg: &WanVaeConfig, w: &Weights, cache: &Weights, lw: i64, lh: i64, first: bool) -> DecodeIo {
    let latent = g.input(sys::GGML_TYPE_F32, &[lw, lh, cfg.z_dim as i64, 1]);
    let mut n = Net { g, w, cache, feeds: Vec::new() };
    let x = n.pointwise("post_quant_conv", latent);
    let dd = cfg.dec_dims();
    let mut x = n.causal("decoder.conv_in", x, 1);
    x = n.mid("decoder.mid_block", x, dd[0]);
    for (i, (cin, cout, temporal, spatial)) in cfg.up_stages().into_iter().enumerate() {
        let pre = format!("decoder.up_blocks.{i}");
        let skip = x;
        let mut c = cin;
        for r in 0..=cfg.num_res_blocks {
            x = n.resnet(&format!("{pre}.resnets.{r}"), x, c, cout);
            c = cout;
        }
        if spatial {
            if temporal && !first {
                let y = n.causal(&format!("{pre}.upsampler.time_conv"), x, 0);
                x = n.g.reshape(y, &[y.ne(0), y.ne(1), cout as i64, 2 * y.ne(3)]);
            }
            let up = n.g.upscale_nearest(x, 2);
            let k = n.wt(&format!("{pre}.upsampler.resample.1.weight"));
            let y = n.g.conv2d(k, up, 1);
            x = n.add_bias(y, &format!("{pre}.upsampler.resample.1"));
            let s = n.dup_up(skip, cout, temporal, first);
            x = n.g.add(x, s);
        }
    }
    x = n.norm_silu("decoder.norm_out", x);
    let out = n.causal("decoder.conv_out", x, 1);
    DecodeIo { latent, out, feeds: n.feeds }
}

/// Fold 2x2 pixel patches into channels: `[3][H][W]` to `[12][H/2][W/2]`,
/// channel `c * 4 + pw * 2 + ph`.
#[must_use]
pub fn patchify(px: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (w2, h2) = (w / 2, h / 2);
    let mut out = vec![0f32; 12 * w2 * h2];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let ch = c * 4 + (x % 2) * 2 + y % 2;
                out[(ch * h2 + y / 2) * w2 + x / 2] = px[(c * h + y) * w + x];
            }
        }
    }
    out
}

/// Inverse of [`patchify`] for `frames` frames laid out `[T][12][H/2][W/2]`
/// (the decoder output's memory order), giving `[T][3][H][W]` clamped to
/// `[-1, 1]`.
#[must_use]
pub fn unpatchify(x: &[f32], w2: usize, h2: usize, frames: usize) -> Vec<f32> {
    let (w, h) = (2 * w2, 2 * h2);
    let mut out = vec![0f32; frames * 3 * w * h];
    for f in 0..frames {
        for c in 0..3 {
            for y in 0..h {
                for xx in 0..w {
                    let ch = c * 4 + (xx % 2) * 2 + y % 2;
                    let v = x[((f * 12 + ch) * h2 + y / 2) * w2 + xx / 2];
                    out[((f * 3 + c) * h + y) * w + xx] = v.clamp(-1.0, 1.0);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patching_round_trips() {
        let (w, h) = (6, 4);
        let px: Vec<f32> = (0..3 * w * h).map(|i| (i as f32 / 100.0) - 0.3).collect();
        let p = patchify(&px, w, h);
        assert_eq!(unpatchify(&p, w / 2, h / 2, 1), px);
    }

    #[test]
    fn the_averaging_shortcut_reads_only_the_real_frame() {
        // 2 -> 4 channels, halving time and space: groups of 4, the first
        // half of each channel's 8 space-to-depth slots is the zero frame.
        let k = avg_down_kernel(2, 4, true, true, 1).unwrap();
        let sum = |o: usize| k[o * 8..(o + 1) * 8].iter().sum::<f32>();
        assert_eq!(sum(0), 0.0);
        assert!((sum(1) - 1.0).abs() < 1e-6);
        assert_eq!(sum(2), 0.0);
        assert!((sum(3) - 1.0).abs() < 1e-6);
        assert!(avg_down_kernel(4, 4, false, false, 0).is_none());
        // The earlier phase is the complement: the two sum to a plain mean.
        let prev = avg_down_kernel(2, 4, true, true, 0).unwrap();
        for (a, b) in k.iter().zip(&prev) {
            assert!(a * b == 0.0);
        }
        let total: f32 = k.iter().chain(&prev).sum();
        assert!((total - 4.0).abs() < 1e-5);
    }
}
