//! The Oobleck audio autoencoder's decoder: latent frames to waveform.
//!
//! Every convolution is weight-normalised in the files; the normalisation is
//! folded into one kernel at load. A convolution runs as one matrix product per
//! kernel tap over a shifted view of the zero-padded input, and a transposed
//! convolution as two products whose halves overlap by one stride, so the
//! decoder needs only matrix products, element-wise operations, padding and
//! rotation on every backend.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, HostTensor, Tn, WType};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Decoder configuration, read from `vae/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct OobleckConfig {
    /// Output audio channels.
    pub audio_channels: u64,
    /// Channel multiples per resolution, finest first.
    pub channel_multiples: Vec<u64>,
    /// Base decoder width.
    pub decoder_channels: u64,
    /// Latent channels.
    pub decoder_input_channels: u64,
    /// Encoder strides, finest first; the decoder runs them in reverse.
    pub downsampling_ratios: Vec<u64>,
    /// Output sample rate.
    pub sampling_rate: u64,
}

/// Dilations of the three residual units in each decoder block.
const DILATIONS: [i64; 3] = [1, 3, 9];
/// Residual-unit kernel width.
const RES_KERNEL: i64 = 7;

impl OobleckConfig {
    /// Check the configuration describes a decoder this module builds.
    ///
    /// # Errors
    /// [`Error::Config`] when the strides and multiples disagree.
    pub fn validate(&self) -> Result<()> {
        if self.channel_multiples.len() != self.downsampling_ratios.len() || self.downsampling_ratios.is_empty() {
            return Err(Error::Config("autoencoder strides and channel multiples disagree".into()));
        }
        if self.downsampling_ratios.contains(&0) {
            return Err(Error::Config("autoencoder stride of zero".into()));
        }
        Ok(())
    }

    /// Audio samples per latent frame.
    #[must_use]
    pub fn hop(&self) -> u64 {
        self.downsampling_ratios.iter().product()
    }

    /// `(input width, output width, stride)` of each decoder block in order.
    fn blocks(&self) -> Vec<(u64, u64, u64)> {
        let mut m = vec![1u64];
        m.extend(&self.channel_multiples);
        let strides: Vec<u64> = self.downsampling_ratios.iter().rev().copied().collect();
        let n = strides.len();
        let c = self.decoder_channels;
        strides.iter().enumerate().map(|(i, s)| (c * m[n - i], c * m[n - i - 1], *s)).collect()
    }

    /// Samples produced from `frames` latent frames.
    #[must_use]
    pub fn samples(&self, frames: u64) -> u64 {
        let mut t = frames;
        for (_, _, s) in self.blocks() {
            t = (t + 1) * s - 2 * s.div_ceil(2);
        }
        t
    }

    /// Every decoder weight, normalisation folded in and kernels laid out per
    /// tap, ready to upload.
    ///
    /// # Errors
    /// Missing or mis-shaped tensors.
    pub fn host_tensors(&self, files: &SafeTensors) -> Result<Vec<HostTensor>> {
        let mut v = Vec::new();
        let top = self.decoder_channels * self.channel_multiples.last().copied().unwrap_or(1);
        conv(&mut v, files, "decoder.conv1", self.decoder_input_channels, top, 7, true)?;
        for (i, (cin, cout, s)) in self.blocks().into_iter().enumerate() {
            let p = format!("decoder.block.{i}");
            snake(&mut v, files, &format!("{p}.snake1"), cin)?;
            conv_transpose(&mut v, files, &format!("{p}.conv_t1"), cin, cout, s)?;
            for r in 1..=3 {
                let u = format!("{p}.res_unit{r}");
                snake(&mut v, files, &format!("{u}.snake1"), cout)?;
                conv(&mut v, files, &format!("{u}.conv1"), cout, cout, RES_KERNEL as u64, true)?;
                snake(&mut v, files, &format!("{u}.snake2"), cout)?;
                conv(&mut v, files, &format!("{u}.conv2"), cout, cout, 1, true)?;
            }
        }
        snake(&mut v, files, "decoder.snake1", self.decoder_channels)?;
        conv(&mut v, files, "decoder.conv2", self.decoder_channels, self.audio_channels, 7, false)?;
        Ok(v)
    }
}

/// `g * v / |v|`, the norm taken over everything but dimension 0.
fn weight_norm(files: &SafeTensors, p: &str, shape: &[u64]) -> Result<Vec<f32>> {
    let v = files.require(&format!("{p}.weight_v"), shape)?.to_f32();
    let g = files.require(&format!("{p}.weight_g"), &[shape[0], 1, 1])?.to_f32();
    let per = v.len() / shape[0] as usize;
    let mut out = Vec::with_capacity(v.len());
    for (row, gain) in v.chunks_exact(per).zip(&g) {
        let norm = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        let k = f64::from(*gain) / norm;
        out.extend(row.iter().map(|x| (f64::from(*x) * k) as f32));
    }
    Ok(out)
}

fn host(name: String, shape: Vec<u64>, data: Vec<f32>) -> HostTensor {
    HostTensor { name, shape, ty: WType::F32, data }
}

/// A convolution's kernel as `[k, cout, cin]` (one matrix per tap) plus its
/// bias.
fn conv(v: &mut Vec<HostTensor>, files: &SafeTensors, p: &str, cin: u64, cout: u64, k: u64, bias: bool) -> Result<()> {
    let w = weight_norm(files, p, &[cout, cin, k])?;
    let (ci, co, kk) = (cin as usize, cout as usize, k as usize);
    let mut taps = vec![0f32; w.len()];
    for o in 0..co {
        for i in 0..ci {
            for j in 0..kk {
                taps[(j * co + o) * ci + i] = w[(o * ci + i) * kk + j];
            }
        }
    }
    v.push(host(format!("{p}.taps"), vec![k, cout, cin], taps));
    if bias {
        v.push(host(format!("{p}.bias"), vec![cout], files.require(&format!("{p}.bias"), &[cout])?.to_f32()));
    }
    Ok(())
}

/// A transposed convolution of stride `s` and kernel `2 s` as one matrix
/// `[2 s * cout, cin]` whose row `j * cout + o` is tap `j` of output `o`.
fn conv_transpose(v: &mut Vec<HostTensor>, files: &SafeTensors, p: &str, cin: u64, cout: u64, s: u64) -> Result<()> {
    let k = 2 * s;
    let w = weight_norm(files, p, &[cin, cout, k])?;
    let (ci, co, kk) = (cin as usize, cout as usize, k as usize);
    let mut m = vec![0f32; w.len()];
    for i in 0..ci {
        for o in 0..co {
            for j in 0..kk {
                m[(j * co + o) * ci + i] = w[(i * co + o) * kk + j];
            }
        }
    }
    v.push(host(format!("{p}.matrix"), vec![k * cout, cin], m));
    v.push(host(format!("{p}.bias"), vec![cout], files.require(&format!("{p}.bias"), &[cout])?.to_f32()));
    Ok(())
}

/// Snake parameters as the per-channel `exp(alpha)` and
/// `1 / (exp(beta) + 1e-9)`.
fn snake(v: &mut Vec<HostTensor>, files: &SafeTensors, p: &str, c: u64) -> Result<()> {
    let a = files.require(&format!("{p}.alpha"), &[1, c, 1])?.to_f32();
    let b = files.require(&format!("{p}.beta"), &[1, c, 1])?.to_f32();
    v.push(host(format!("{p}.freq"), vec![c], a.iter().map(|x| x.exp()).collect()));
    v.push(host(format!("{p}.gain"), vec![c], b.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect()));
    Ok(())
}

/// `x + gain * sin(freq * x)^2` per channel.
fn apply_snake(g: &mut Graph, w: &crate::ggml::Weights, p: &str, x: Tn) -> Tn {
    let s = g.mul(x, w.get(&format!("{p}.freq")));
    let s = g.sin(s);
    let s = g.sqr(s);
    let s = g.mul(s, w.get(&format!("{p}.gain")));
    g.add(x, s)
}

/// `x` `[c, t]` with `pad` zero columns on each side.
fn pad_both(g: &mut Graph, x: Tn, pad: i64) -> Tn {
    if pad == 0 {
        return x;
    }
    let y = g.pad_end(x, 0, 2 * pad as i32);
    g.roll(y, 0, pad as i32)
}

/// A same-length convolution of kernel `k` and dilation `dil`.
fn apply_conv(g: &mut Graph, w: &crate::ggml::Weights, p: &str, x: Tn, k: i64, dil: i64, bias: bool) -> Tn {
    let taps = w.get(&format!("{p}.taps"));
    let (cin, cout) = (taps.ne(0), taps.ne(1));
    let t = x.ne(1);
    let xp = pad_both(g, x, dil * (k - 1) / 2);
    let mut y: Option<Tn> = None;
    for j in 0..k {
        let wj = g.view_4d(taps, [cin, cout, 1, 1], taps.nb(1), taps.nb(2), taps.nb(3), j as usize * taps.nb(2));
        let xj = g.view_cols(xp, j * dil, t);
        let yj = g.linear(wj, xj);
        y = Some(match y {
            None => yj,
            Some(acc) => g.add(acc, yj),
        });
    }
    let y = y.expect("kernel has at least one tap");
    if bias { g.add(y, w.get(&format!("{p}.bias"))) } else { y }
}

/// A transposed convolution of stride `s`, kernel `2 s` and padding
/// `ceil(s / 2)`: output length `(t + 1) s - 2 ceil(s / 2)`.
fn apply_conv_transpose(g: &mut Graph, w: &crate::ggml::Weights, p: &str, x: Tn, s: i64) -> Tn {
    let m = w.get(&format!("{p}.matrix"));
    let cout = m.ne(1) / (2 * s);
    let t = x.ne(1);
    let first = g.view_cols(m, 0, s * cout);
    let second = g.view_cols(m, s * cout, s * cout);
    // Column `t` of `a` is output block `t`; of `b`, block `t + 1`.
    let a = g.linear(first, x);
    let b = g.linear(second, x);
    let a = g.pad_end(a, 0, 1);
    let b = g.pad_end(b, 0, 1);
    let b = g.roll(b, 0, 1);
    let full = g.add(a, b);
    let full = g.reshape(full, &[cout, s * (t + 1)]);
    let crop = (s + 1) / 2;
    let y = g.view_cols(full, crop, s * (t + 1) - 2 * crop);
    let y = g.cont(y);
    g.add(y, w.get(&format!("{p}.bias")))
}

/// Graph input and output of one decode.
#[derive(Debug, Clone, Copy)]
pub struct DecodeIo {
    /// Latent frames `[latent channels, frames]`.
    pub latents: Tn,
    /// Audio `[audio channels, samples]` (channel-major after a transpose:
    /// `ne0` is the sample index).
    pub out: Tn,
}

/// Build a decode of `frames` latent frames.
#[must_use]
pub fn build_decoder(g: &mut Graph, cfg: &OobleckConfig, w: &crate::ggml::Weights, frames: i64) -> DecodeIo {
    let latents = g.input(sys::GGML_TYPE_F32, &[cfg.decoder_input_channels as i64, frames]);
    let mut x = apply_conv(g, w, "decoder.conv1", latents, 7, 1, true);
    for (i, (_, _, s)) in cfg.blocks().into_iter().enumerate() {
        let p = format!("decoder.block.{i}");
        x = apply_snake(g, w, &format!("{p}.snake1"), x);
        x = apply_conv_transpose(g, w, &format!("{p}.conv_t1"), x, s as i64);
        for (r, dil) in DILATIONS.iter().enumerate() {
            let u = format!("{p}.res_unit{}", r + 1);
            let h = apply_snake(g, w, &format!("{u}.snake1"), x);
            let h = apply_conv(g, w, &format!("{u}.conv1"), h, RES_KERNEL, *dil, true);
            let h = apply_snake(g, w, &format!("{u}.snake2"), h);
            let h = apply_conv(g, w, &format!("{u}.conv2"), h, 1, 1, true);
            x = g.add(x, h);
        }
    }
    x = apply_snake(g, w, "decoder.snake1", x);
    let y = apply_conv(g, w, "decoder.conv2", x, 7, 1, false);
    let y = g.permute(y, [1, 0, 2, 3]);
    let out = g.cont(y);
    DecodeIo { latents, out }
}
