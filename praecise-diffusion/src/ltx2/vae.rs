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
        let backend = opts.backend()?;
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
    /// them) to pixels `[3][T'][H s][W s]` in about `[-1, 1]`, as one graph.
    ///
    /// # Errors
    /// A latent size that disagrees with the shape, or a backend failure.
    pub fn decode(&self, latent: &[f32], frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        let z = self.denormalise(latent, frames, height, width)?;
        Ok(self.run(&z)?.d)
    }

    /// Decode as [`Self::decode`], in overlapping tiles blended linearly over
    /// their overlap, so memory follows the tile size rather than the clip.
    /// Tiles cover `min` samples every `stride` samples on each tiled axis;
    /// an axis is tiled only when the latent is longer than one tile.
    ///
    /// # Errors
    /// A latent size that disagrees with the shape, a tiling that does not
    /// divide by the decoder's upsampling, or a backend failure.
    pub fn decode_tiled(&self, latent: &[f32], frames: usize, height: usize, width: usize, tiling: Tiling) -> Result<Vec<f32>> {
        let (s, t) = self.cfg.factors();
        tiling.check(s as usize, t as usize)?;
        let z = self.denormalise(latent, frames, height, width)?;
        let out = match tiling.temporal {
            Some(tt) if frames > tt.min / t as usize => self.temporal_tiled(&z, tiling, tt)?,
            _ => self.spatial_or_whole(&z, tiling)?,
        };
        Ok(out.d)
    }

    fn denormalise(&self, latent: &[f32], frames: usize, height: usize, width: usize) -> Result<Vol> {
        let c = self.cfg.latent_channels as usize;
        let hw = height * width;
        if frames == 0 || hw == 0 || latent.len() != c * frames * hw {
            return Err(Error::Request("video latent size disagrees with its shape".into()));
        }
        let mut d = latent.to_vec();
        for (ch, plane) in d.chunks_exact_mut(frames * hw).enumerate() {
            for v in plane {
                *v = *v * self.std[ch] + self.mean[ch];
            }
        }
        Ok(Vol { c, t: frames, h: height, w: width, d })
    }

    fn spatial_or_whole(&self, z: &Vol, tiling: Tiling) -> Result<Vol> {
        let s = self.cfg.factors().0 as usize;
        match tiling.spatial {
            Some(ts) if z.w > ts.min / s || z.h > ts.min / s => self.spatial_tiled(z, ts),
            _ => self.run(z),
        }
    }

    fn spatial_tiled(&self, z: &Vol, ts: Tile) -> Result<Vol> {
        let s = self.cfg.factors().0 as usize;
        let (lmin, lstride, fade) = (ts.min / s, ts.stride / s, ts.min - ts.stride);
        let mut rows = Vec::new();
        for i in (0..z.h).step_by(lstride) {
            let mut row = Vec::new();
            for j in (0..z.w).step_by(lstride) {
                row.push(self.run(&z.crop((0, z.t), (i, i + lmin), (j, j + lmin)))?);
            }
            rows.push(row);
        }
        // Each tile blends with its already blended upper and left neighbours.
        let mut bands = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            let (done, rest) = rows.split_at_mut(i);
            let row = &mut rest[0];
            for j in 0..row.len() {
                if i > 0 {
                    blend(&done[i - 1][j], &mut row[j], 2, fade);
                }
                if j > 0 {
                    let (left, cur) = row.split_at_mut(j);
                    blend(&left[j - 1], &mut cur[0], 3, fade);
                }
            }
            let kept: Vec<Vol> = row.iter().map(|tile| tile.crop((0, tile.t), (0, ts.stride), (0, ts.stride))).collect();
            bands.push(Vol::cat(&kept, 3));
        }
        let out = Vol::cat(&bands, 2);
        Ok(out.crop((0, out.t), (0, z.h * s), (0, z.w * s)))
    }

    fn temporal_tiled(&self, z: &Vol, tiling: Tiling, tt: Tile) -> Result<Vol> {
        let t = self.cfg.factors().1 as usize;
        let (lmin, lstride, blend_frames) = (tt.min / t, tt.stride / t, tt.min - tt.stride);
        let mut tiles = Vec::new();
        for i in (0..z.t).step_by(lstride) {
            let mut d = self.spatial_or_whole(&z.crop((i, i + lmin + 1), (0, z.h), (0, z.w)), tiling)?;
            if i > 0 {
                d = d.crop((0, d.t - 1), (0, d.h), (0, d.w));
            }
            tiles.push(d);
        }
        let mut kept = Vec::with_capacity(tiles.len());
        for i in 0..tiles.len() {
            let (done, rest) = tiles.split_at_mut(i);
            let tile = &mut rest[0];
            if i > 0 {
                blend(&done[i - 1], tile, 1, blend_frames);
                kept.push(tile.crop((0, tt.stride), (0, tile.h), (0, tile.w)));
            } else {
                kept.push(tile.crop((0, tt.stride + 1), (0, tile.h), (0, tile.w)));
            }
        }
        let out = Vol::cat(&kept, 1);
        Ok(out.crop((0, (z.t - 1) * t + 1), (0, out.h), (0, out.w)))
    }

    /// One decoder graph over denormalised latents.
    fn run(&self, z: &Vol) -> Result<Vol> {
        let (c, frames, height, width) = (z.c, z.t, z.h, z.w);
        let hw = height * width;
        // [C][T][H][W] to the graph's [T][C][H][W].
        let mut zt = vec![0f32; z.d.len()];
        for ch in 0..c {
            for t in 0..frames {
                zt[(t * c + ch) * hw..][..hw].copy_from_slice(&z.d[(ch * frames + t) * hw..][..hw]);
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[width as i64, height as i64, c as i64, frames as i64]);
        let out = build(&mut g, &self.cfg, &self.w, input);
        g.finish(&[out])?;
        g.set_f32(input, &zt);
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
        Ok(Vol { c: rgb, t: of, h: ph, w: pw, d: px })
    }
}

/// Tile extent and step along one axis, in video samples (frames or pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Samples one tile covers.
    pub min: usize,
    /// Samples between tile starts; `min - stride` samples are blended.
    pub stride: usize,
}

/// How [`Ltx2VideoDecoder::decode_tiled`] splits a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    /// Square tiles over height and width.
    pub spatial: Option<Tile>,
    /// Tiles over frames.
    pub temporal: Option<Tile>,
}

impl Default for Tiling {
    /// 512 pixel tiles every 448 pixels, 16 frame tiles every 8 frames.
    fn default() -> Self {
        Self { spatial: Some(Tile { min: 512, stride: 448 }), temporal: Some(Tile { min: 16, stride: 8 }) }
    }
}

impl Tiling {
    fn check(self, s: usize, t: usize) -> Result<()> {
        for (tile, f, axis) in [(self.spatial, s, "spatial"), (self.temporal, t, "temporal")] {
            if let Some(Tile { min, stride }) = tile {
                if stride == 0 || stride > min || min % f != 0 || stride % f != 0 {
                    return Err(Error::Request(format!("{axis} tiles must be multiples of {f} with 0 < stride <= min")));
                }
            }
        }
        Ok(())
    }
}

/// A `[C][T][H][W]` volume.
#[derive(Debug, Clone)]
struct Vol {
    c: usize,
    t: usize,
    h: usize,
    w: usize,
    d: Vec<f32>,
}

impl Vol {
    fn dims(&self) -> [usize; 4] {
        [self.c, self.t, self.h, self.w]
    }

    /// The sub-volume over half-open ranges, each clamped to the volume.
    fn crop(&self, (t0, t1): (usize, usize), (h0, h1): (usize, usize), (w0, w1): (usize, usize)) -> Vol {
        let (t1, h1, w1) = (t1.min(self.t), h1.min(self.h), w1.min(self.w));
        let (t, h, w) = (t1 - t0, h1 - h0, w1 - w0);
        let mut d = Vec::with_capacity(self.c * t * h * w);
        for c in 0..self.c {
            for tt in t0..t1 {
                for hh in h0..h1 {
                    let at = ((c * self.t + tt) * self.h + hh) * self.w;
                    d.extend_from_slice(&self.d[at + w0..at + w1]);
                }
            }
        }
        Vol { c: self.c, t, h, w, d }
    }

    /// Concatenation along `axis` (1 = frames, 2 = rows, 3 = columns).
    fn cat(parts: &[Vol], axis: usize) -> Vol {
        let mut dims = parts[0].dims();
        dims[axis] = parts.iter().map(|p| p.dims()[axis]).sum();
        // Every part splits into `outer` runs of `inner` contiguous values.
        let outer: usize = dims[..axis].iter().product();
        let mut d = Vec::with_capacity(dims.iter().product());
        for o in 0..outer {
            for p in parts {
                let inner = p.d.len() / outer;
                d.extend_from_slice(&p.d[o * inner..][..inner]);
            }
        }
        Vol { c: dims[0], t: dims[1], h: dims[2], w: dims[3], d }
    }
}

/// Fade the first `extent` samples of `b` along `axis` in from the last ones
/// of `a`: `b[k] = a[n - extent + k] (1 - k / extent) + b[k] k / extent`.
fn blend(a: &Vol, b: &mut Vol, axis: usize, extent: usize) {
    let (ad, bd) = (a.dims(), b.dims());
    let extent = extent.min(ad[axis]).min(bd[axis]);
    for c in 0..bd[0] {
        for t in 0..bd[1] {
            for h in 0..bd[2] {
                for w in 0..bd[3] {
                    let mut at = [c, t, h, w];
                    let k = at[axis];
                    if k >= extent {
                        continue;
                    }
                    at[axis] = ad[axis] - extent + k;
                    let src = a.d[((at[0] * ad[1] + at[1]) * ad[2] + at[2]) * ad[3] + at[3]];
                    let f = k as f32 / extent as f32;
                    let dst = &mut b.d[((c * bd[1] + t) * bd[2] + h) * bd[3] + w];
                    *dst = src * (1.0 - f) + *dst * f;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(c: usize, t: usize, h: usize, w: usize) -> Vol {
        Vol { c, t, h, w, d: (0..c * t * h * w).map(|i| i as f32).collect() }
    }

    #[test]
    fn crops_concatenate_back() {
        let v = ramp(2, 3, 4, 5);
        for axis in 1..4 {
            let n = v.dims()[axis];
            let r = |a: usize, k: usize| if a == axis { (k, k + 2) } else { (0, usize::MAX) };
            let parts: Vec<Vol> = (0..n).step_by(2).map(|k| v.crop(r(1, k), r(2, k), r(3, k))).collect();
            assert_eq!(Vol::cat(&parts, axis).d, v.d, "axis {axis}");
        }
    }

    #[test]
    fn blend_fades_linearly() {
        let a = Vol { c: 1, t: 1, h: 1, w: 4, d: vec![0.0, 0.0, 8.0, 8.0] };
        let mut b = Vol { c: 1, t: 1, h: 1, w: 3, d: vec![0.0, 4.0, 4.0] };
        blend(&a, &mut b, 3, 2);
        assert_eq!(b.d, vec![8.0, 6.0, 4.0]);
    }

    #[test]
    fn tiling_must_divide_by_the_upsampling() {
        assert!(Tiling::default().check(32, 8).is_ok());
        let bad = Tiling { spatial: Some(Tile { min: 500, stride: 448 }), temporal: None };
        assert!(bad.check(32, 8).is_err());
        let bad = Tiling { spatial: None, temporal: Some(Tile { min: 8, stride: 16 }) };
        assert!(bad.check(32, 8).is_err());
    }

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
