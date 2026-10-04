//! LTX-2 video autoencoder: the decoder, latents to pixels.
//!
//! Activations are laid out `[W, H, C, T]`. A 3x3x3 convolution is three 2D
//! convolutions over shifted frames, summed. The frame axis is padded by
//! repeating edge frames (one at each end for the non-causal decoder, the
//! first frame twice for a causal one); the spatial axes are zero padded.
//!
//! The decoder is a list of stages read from the checkpoint header, in weight
//! order from the latent side: groups of residual blocks (pixel norm, SiLU,
//! convolution, twice, plus the input) and upsampling convolutions that widen
//! the channels and fold them into time and space (depth to space). A
//! temporal fold drops its first output frame, so with three of them `T`
//! latent frames decode to `8 (T - 1) + 1` video frames. The last convolution
//! emits `3 x p x p` channels that unfold into `p x p` pixel patches.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::single_file::{header_config, open_part, Part};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Pixel norm epsilon (fixed in the reference).
const NORM_EPS: f32 = 1e-8;

/// One decoder stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// `layers` residual blocks at `width` channels.
    Res {
        /// Residual blocks.
        layers: usize,
        /// Channels.
        width: u64,
    },
    /// A convolution from `cin` to `cin * t * s * s / multiplier` channels,
    /// folded into a `t x s x s` upsampling with `cin / multiplier` channels.
    Up {
        /// Input channels.
        cin: u64,
        /// Temporal factor.
        t: u64,
        /// Spatial factor.
        s: u64,
        /// Channel reduction.
        multiplier: u64,
    },
}

impl Stage {
    fn out_width(self) -> u64 {
        match self {
            Self::Res { width, .. } => width,
            Self::Up { cin, multiplier, .. } => cin / multiplier,
        }
    }
}

/// Decoder configuration, from the `vae` section of a single-file header.
#[derive(Debug, Clone)]
pub struct VideoVaeConfig {
    /// Latent channels.
    pub latent_channels: u64,
    /// Pixel channels.
    pub out_channels: u64,
    /// Spatial pixel patch side.
    pub patch: u64,
    /// Causal frame padding in the decoder.
    pub causal: bool,
    /// Width after the input convolution.
    pub top: u64,
    /// Stages from the latent side, in weight order.
    pub stages: Vec<Stage>,
}

#[derive(Deserialize)]
struct Header {
    #[serde(default = "three")]
    dims: u64,
    latent_channels: u64,
    #[serde(default = "three")]
    out_channels: u64,
    patch_size: u64,
    decoder_blocks: Vec<(String, Value)>,
    decoder_base_channels: u64,
    #[serde(default)]
    causal_decoder: bool,
    #[serde(default)]
    timestep_conditioning: bool,
    #[serde(default = "zeros")]
    spatial_padding_mode: String,
    #[serde(default = "pixel_norm")]
    norm_layer: String,
}

fn three() -> u64 {
    3
}

fn zeros() -> String {
    "zeros".into()
}

fn pixel_norm() -> String {
    "pixel_norm".into()
}

fn unsupported(what: String) -> Error {
    Error::Config(format!("video autoencoder: {what} is not supported"))
}

impl VideoVaeConfig {
    /// Parse the `vae` section of a single-file header.
    ///
    /// # Errors
    /// A layout the native decoder does not implement.
    pub fn from_single_file(vae: &Value) -> Result<Self> {
        let h: Header = serde_json::from_value(vae.clone()).map_err(|e| Error::Config(format!("video autoencoder config: {e}")))?;
        if h.dims != 3 {
            return Err(unsupported(format!("dims {}", h.dims)));
        }
        if h.timestep_conditioning {
            return Err(unsupported("timestep conditioning".into()));
        }
        if h.spatial_padding_mode != "zeros" {
            return Err(unsupported(format!("spatial padding {}", h.spatial_padding_mode)));
        }
        if h.norm_layer != "pixel_norm" {
            return Err(unsupported(format!("norm {}", h.norm_layer)));
        }
        let mut blocks = Vec::new();
        for (name, p) in h.decoder_blocks.iter().rev() {
            let multiplier = p.get("multiplier").and_then(Value::as_u64).unwrap_or(1);
            if p.get("residual").and_then(Value::as_bool).unwrap_or(false) || p.get("inject_noise").and_then(Value::as_bool).unwrap_or(false) {
                return Err(unsupported(format!("{name} with residual or noise")));
            }
            let kind = match name.as_str() {
                "res_x" => None,
                "compress_all" => Some((2, 2)),
                "compress_time" => Some((2, 1)),
                "compress_space" => Some((1, 2)),
                other => return Err(unsupported(format!("decoder block {other}"))),
            };
            let layers = p.get("num_layers").and_then(Value::as_u64).unwrap_or(1) as usize;
            blocks.push((kind, layers, multiplier));
        }
        let top = h.decoder_base_channels * blocks.iter().filter(|b| b.0.is_some()).map(|b| b.2).product::<u64>();
        let mut width = top;
        let mut stages = Vec::new();
        for (kind, layers, multiplier) in blocks {
            let stage = match kind {
                None => Stage::Res { layers, width },
                Some((t, s)) => {
                    if multiplier == 0 || !width.is_multiple_of(multiplier) {
                        return Err(unsupported(format!("multiplier {multiplier} at width {width}")));
                    }
                    Stage::Up { cin: width, t, s, multiplier }
                }
            };
            width = stage.out_width();
            stages.push(stage);
        }
        if width != h.decoder_base_channels {
            return Err(Error::Config("video autoencoder: decoder widths disagree with the base width".into()));
        }
        Ok(Self { latent_channels: h.latent_channels, out_channels: h.out_channels, patch: h.patch_size, causal: h.causal_decoder, top, stages })
    }

    /// Spatial and temporal upsampling of the stages (pixel patch included).
    #[must_use]
    pub fn factors(&self) -> (u64, u64) {
        self.stages.iter().fold((self.patch, 1), |(s, t), st| match st {
            Stage::Up { s: a, t: b, .. } => (s * a, t * b),
            Stage::Res { .. } => (s, t),
        })
    }

    /// Video frames decoded from `frames` latent frames.
    #[must_use]
    pub fn video_frames(&self, frames: usize) -> usize {
        self.stages.iter().fold(frames, |f, st| match st {
            Stage::Up { t, .. } => f * *t as usize - (*t as usize - 1),
            Stage::Res { .. } => f,
        })
    }

    fn host_tensors(&self, st: &SafeTensors, kt: WType) -> Result<Vec<HostTensor>> {
        let mut v = Vec::new();
        let mut conv = |p: &str, cin: u64, cout: u64| -> Result<()> {
            let w = st.require(&format!("{p}.weight"), &[cout, cin, 3, 3, 3])?.to_f32();
            for k in 0..3 {
                let d: Vec<f32> = w.chunks_exact(27).flat_map(|c| (0..9).map(move |j| c[k * 9 + j])).collect();
                v.push(HostTensor { name: format!("{p}.t{k}"), shape: vec![cout, cin, 3, 3], ty: kt, data: d });
            }
            v.push(HostTensor { name: format!("{p}.bias"), shape: vec![cout], ty: WType::F32, data: st.require(&format!("{p}.bias"), &[cout])?.to_f32() });
            Ok(())
        };
        conv("decoder.conv_in.conv", self.latent_channels, self.top)?;
        for (i, stage) in self.stages.iter().enumerate() {
            match *stage {
                Stage::Res { layers, width } => {
                    for r in 0..layers {
                        for c in ["conv1", "conv2"] {
                            conv(&format!("decoder.up_blocks.{i}.res_blocks.{r}.{c}.conv"), width, width)?;
                        }
                    }
                }
                Stage::Up { cin, t, s, multiplier } => conv(&format!("decoder.up_blocks.{i}.conv.conv"), cin, cin * t * s * s / multiplier)?,
            }
        }
        let last = self.stages.last().map_or(self.top, |s| s.out_width());
        conv("decoder.conv_out.conv", last, self.out_channels * self.patch * self.patch)?;
        Ok(v)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
    causal: bool,
}

impl Net<'_, '_> {
    fn frame(&mut self, x: Tn, i: i64) -> Tn {
        let v = self.g.view_4d(x, [x.ne(0), x.ne(1), x.ne(2), 1], x.nb(1), x.nb(2), x.nb(3), i as usize * x.nb(3));
        self.g.cont(v)
    }

    /// A 3x3x3 convolution with frame padding by edge repetition.
    fn conv(&mut self, p: &str, x: Tn) -> Tn {
        let t = x.ne(3);
        let first = self.frame(x, 0);
        let padded = if self.causal {
            let two = self.g.concat(first, first, 3);
            self.g.concat(two, x, 3)
        } else {
            let last = self.frame(x, t - 1);
            let a = self.g.concat(first, x, 3);
            self.g.concat(a, last, 3)
        };
        let shape = [x.ne(0), x.ne(1), x.ne(2), t];
        let mut acc: Option<Tn> = None;
        for k in 0..3 {
            let v = self.g.view_4d(padded, shape, padded.nb(1), padded.nb(2), padded.nb(3), k * padded.nb(3));
            let y = self.g.conv2d(self.w.get(&format!("{p}.t{k}")), v, 1);
            acc = Some(match acc {
                None => y,
                Some(a) => self.g.add(a, y),
            });
        }
        let b = self.w.get(&format!("{p}.bias"));
        let b = self.g.reshape(b, &[1, 1, b.ne(0), 1]);
        self.g.add(acc.expect("three taps"), b)
    }

    /// Pixel norm over channels, then SiLU.
    fn norm_silu(&mut self, x: Tn) -> Tn {
        let c = self.g.permute(x, [1, 2, 0, 3]);
        let c = self.g.cont(c);
        let n = self.g.rms_norm(c, NORM_EPS);
        let n = self.g.silu(n);
        let back = self.g.permute(n, [2, 0, 1, 3]);
        self.g.cont(back)
    }

    fn resnet(&mut self, p: &str, x: Tn) -> Tn {
        let h = self.norm_silu(x);
        let h = self.conv(&format!("{p}.conv1.conv"), h);
        let h = self.norm_silu(h);
        let h = self.conv(&format!("{p}.conv2.conv"), h);
        self.g.add(h, x)
    }

    /// `[W, H, C t s s, T]` to `[W s, H s, C, T t - (t - 1)]`; the channel
    /// index is `((c t + it) s + ih) s + iw`.
    fn depth_to_space(&mut self, x: Tn, t: i64, s: i64) -> Tn {
        let (w, h, c, f) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
        let mut x = x;
        if s > 1 {
            let r = c / s * f;
            let a = self.g.reshape(x, &[w, h, s, r]);
            let a = self.g.permute(a, [1, 2, 0, 3]);
            let a = self.g.cont(a);
            let a = self.g.reshape(a, &[s * w, h, s, r / s]);
            let a = self.g.permute(a, [0, 2, 1, 3]);
            let a = self.g.cont(a);
            x = self.g.reshape(a, &[s * w, s * h, c / (s * s), f]);
        }
        if t > 1 {
            let (w, h, c) = (x.ne(0), x.ne(1), x.ne(2));
            let a = self.g.reshape(x, &[w * h, t, c / t, f]);
            let a = self.g.permute(a, [0, 2, 1, 3]);
            let a = self.g.cont(a);
            let a = self.g.reshape(a, &[w, h, c / t, t * f]);
            let v = self.g.view_4d(a, [w, h, c / t, t * f - (t - 1)], a.nb(1), a.nb(2), a.nb(3), (t - 1) as usize * a.nb(3));
            x = self.g.cont(v);
        }
        x
    }
}

fn build(g: &mut Graph, cfg: &VideoVaeConfig, w: &Weights, latent: Tn) -> Tn {
    let mut n = Net { g, w, causal: cfg.causal };
    let mut x = n.conv("decoder.conv_in.conv", latent);
    for (i, stage) in cfg.stages.iter().enumerate() {
        match *stage {
            Stage::Res { layers, .. } => {
                for r in 0..layers {
                    x = n.resnet(&format!("decoder.up_blocks.{i}.res_blocks.{r}"), x);
                }
            }
            Stage::Up { t, s, .. } => {
                let y = n.conv(&format!("decoder.up_blocks.{i}.conv.conv"), x);
                x = n.depth_to_space(y, t as i64, s as i64);
            }
        }
    }
    let x = n.norm_silu(x);
    n.conv("decoder.conv_out.conv", x)
}

/// The video decoder with its resident weights.
pub struct Ltx2VideoDecoder {
    backend: Backend,
    cfg: VideoVaeConfig,
    w: Weights,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl std::fmt::Debug for Ltx2VideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2VideoDecoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl Ltx2VideoDecoder {
    /// Load the video decoder of a single-file checkpoint.
    ///
    /// # Errors
    /// On an unsupported layout, missing weights or no usable backend.
    pub fn load_single_file(path: &Path, opts: LoadOptions) -> Result<Self> {
        let st = SafeTensors::open(&[path.to_path_buf()])?;
        let header = header_config(&st)?;
        let cfg = VideoVaeConfig::from_single_file(&header["vae"])?;
        let st = open_part(st, Part::VideoVae)?;
        let backend = Backend::select(opts.cpu_threads)?;
        let kt = if opts.precision == Precision::F32 { WType::F32 } else { WType::F16 };
        let w = Weights::from_host(&backend, &cfg.host_tensors(&st, kt)?)?;
        let c = cfg.latent_channels;
        let mean = st.require("latents_mean", &[c])?.to_f32();
        let std = st.require("latents_std", &[c])?.to_f32();
        Ok(Self { backend, cfg, w, mean, std })
    }

    /// The decoder configuration.
    #[must_use]
    pub fn config(&self) -> &VideoVaeConfig {
        &self.cfg
    }

    /// Decode normalised latents `[C][T][H][W]` (as the transformer produces
    /// them) to pixels `[3][T'][H s][W s]` in about `[-1, 1]`.
    ///
    /// # Errors
    /// A latent size that disagrees with the shape, or a backend failure.
    pub fn decode(&self, latent: &[f32], frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        let c = self.cfg.latent_channels as usize;
        let hw = height * width;
        if frames == 0 || hw == 0 || latent.len() != c * frames * hw {
            return Err(Error::Request("video latent size disagrees with its shape".into()));
        }
        // [C][T][H][W] denormalised to the graph's [T][C][H][W].
        let mut z = vec![0f32; latent.len()];
        for ch in 0..c {
            for t in 0..frames {
                let src = &latent[(ch * frames + t) * hw..][..hw];
                let dst = &mut z[(t * c + ch) * hw..][..hw];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = s * self.std[ch] + self.mean[ch];
                }
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[width as i64, height as i64, c as i64, frames as i64]);
        let out = build(&mut g, &self.cfg, &self.w, input);
        g.finish(&[out])?;
        g.set_f32(input, &z);
        g.compute()?;
        let y = g.read_f32(out);
        let (ow, oh, oc, of) = (out.ne(0) as usize, out.ne(1) as usize, out.ne(2) as usize, out.ne(3) as usize);
        let p = self.cfg.patch as usize;
        let rgb = oc / (p * p);
        let (pw, ph) = (ow * p, oh * p);
        let mut px = vec![0f32; rgb * of * ph * pw];
        // Channel `c p p + pw p + ph` holds the pixel at (`h p + ph`, `w p + pw`).
        for t in 0..of {
            for ch in 0..oc {
                let (k, a, b) = (ch / (p * p), (ch / p) % p, ch % p);
                let plane = &y[(t * oc + ch) * oh * ow..][..oh * ow];
                for hh in 0..oh {
                    for ww in 0..ow {
                        px[((k * of + t) * ph + hh * p + b) * pw + ww * p + a] = plane[hh * ow + ww];
                    }
                }
            }
        }
        Ok(px)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_layout() {
        let v: Value = serde_json::from_str(
            r#"{"dims":3,"latent_channels":128,"patch_size":4,"decoder_base_channels":128,"causal_decoder":false,
            "decoder_blocks":[["res_x",{"num_layers":4}],["compress_space",{"multiplier":2}],["res_x",{"num_layers":6}],
            ["compress_time",{"multiplier":2}],["res_x",{"num_layers":4}],["compress_all",{"multiplier":1}],
            ["res_x",{"num_layers":2}],["compress_all",{"multiplier":2}],["res_x",{"num_layers":2}]]}"#,
        )
        .unwrap();
        let c = VideoVaeConfig::from_single_file(&v).unwrap();
        assert_eq!(c.top, 1024);
        assert_eq!(c.stages[0], Stage::Res { layers: 2, width: 1024 });
        assert_eq!(c.stages[1], Stage::Up { cin: 1024, t: 2, s: 2, multiplier: 2 });
        assert_eq!(c.stages[5], Stage::Up { cin: 512, t: 2, s: 1, multiplier: 2 });
        assert_eq!(c.stages[7], Stage::Up { cin: 256, t: 1, s: 2, multiplier: 2 });
        assert_eq!(c.factors(), (32, 8));
        assert_eq!(c.video_frames(16), 121);
    }
}
