//! MiniMax-H3 video autoencoder: a causal 3D CNN encoder and a ViT decoder.
//!
//! Encoder activations are laid out `[W, H, C, T]`. A 3x3x3 convolution is
//! three 2D convolutions over shifted frames, summed; the frame axis gets two
//! leading zero frames (causal), the spatial axes are reflect padded by one
//! (downsampling convolutions instead reflect pad only the far edge by one
//! and stride by two). Group norms see one frame at a time.
//!
//! The decoder turns every latent voxel into one token, appends learned
//! register tokens and one zero token (all at position zero), runs plain
//! pre-norm blocks with per-channel residual scales, RMS-normed queries and
//! keys and a partial rotary embedding over `[-1, 1)` normalised `(t, h, w)`
//! coordinates, and projects each voxel token to a `t x s x s` pixel block.
//!
//! Videos are encoded in clips of `clip_length` frames (the last frame
//! repeated to fill the last clip) and the last `token_drop` latent frames
//! are dropped; decoding mirrors that with overlapping clips cross-faded in
//! time. Both directions tile space with linearly blended overlaps by
//! default, as the checkpoint was released. Pixels are in the autoencoder's
//! own space (ImageNet mean / std normalised RGB); latents are normalised
//! with the checkpoint's per-channel `latents_mean` / `latents_std`.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision, RgbImage};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// ImageNet channel means of the pixel space.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet channel spreads of the pixel space.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `vae/config.json` of a MiniMax-H3 checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct H3VideoVaeConfig {
    pub in_channels: u64,
    pub out_channels: u64,
    pub latent_channels: u64,
    pub block_out_channels: Vec<u64>,
    pub layers_per_block: usize,
    pub spatial_downsample_factors: Vec<u64>,
    pub temporal_downsample_factors: Vec<u64>,
    pub norm_num_groups: u64,
    pub norm_eps: f64,
    pub spatial_padding_mode: String,
    pub decoder_num_layers: usize,
    pub decoder_num_attention_heads: u64,
    pub decoder_attention_head_dim: u64,
    pub decoder_num_register_tokens: u64,
    pub decoder_ffn_mult: u64,
    pub decoder_rope_theta: f64,
    pub decoder_rope_dim_ratio: f64,
    pub decoder_norm_eps: f64,
    pub clip_length: usize,
    pub token_drop: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl H3VideoVaeConfig {
    /// Refuse layouts this implementation does not compute.
    ///
    /// # Errors
    /// The first unsupported or inconsistent field.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("MiniMax-H3 video autoencoder: {m}")));
        let n = self.block_out_channels.len();
        if n == 0 || self.spatial_downsample_factors.len() != n || self.temporal_downsample_factors.len() != n {
            return bad("per-level lists disagree in length");
        }
        for (&s, &t) in self.spatial_downsample_factors.iter().zip(&self.temporal_downsample_factors) {
            if !(s == 1 && t == 1 || s == 2 && (t == 1 || t == 2)) {
                return bad("each level must halve space (and optionally time) or keep both");
            }
        }
        if self.spatial_padding_mode != "reflect" {
            return bad("only reflect spatial padding is implemented");
        }
        let g = self.norm_num_groups;
        if g == 0 || self.block_out_channels.iter().any(|c| c % g != 0) {
            return bad("norm groups must divide every level width");
        }
        let rot = self.rope_dim();
        if rot == 0 || !rot.is_multiple_of(6) || rot > self.decoder_attention_head_dim {
            return bad("rotary width must be a positive multiple of 6 within a head");
        }
        let l = self.latent_channels as usize;
        if self.latents_mean.len() != l || self.latents_std.len() != l {
            return bad("latent statistics disagree with the latent width");
        }
        if self.clip_length == 0 || self.token_drop >= self.tokens_chunk() {
            return bad("clip geometry");
        }
        if self.in_channels == 0 || self.out_channels == 0 || l == 0 {
            return bad("empty channel counts");
        }
        Ok(())
    }

    /// Spatial compression ratio.
    #[must_use]
    pub fn spatial_ratio(&self) -> usize {
        self.spatial_downsample_factors.iter().product::<u64>() as usize
    }

    /// Temporal compression ratio.
    #[must_use]
    pub fn temporal_ratio(&self) -> usize {
        self.temporal_downsample_factors.iter().product::<u64>() as usize
    }

    fn rope_dim(&self) -> u64 {
        (self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio) as u64
    }

    fn width(&self) -> u64 {
        self.decoder_num_attention_heads * self.decoder_attention_head_dim
    }

    /// Latent frames per encoded clip.
    fn tokens_chunk(&self) -> usize {
        self.clip_length.div_ceil(self.temporal_ratio())
    }

    fn frame_pre_padding(&self) -> usize {
        let t = self.temporal_ratio();
        (t - self.clip_length % t) % t
    }

    fn token_overlap(&self) -> usize {
        let c = self.tokens_chunk();
        (c - self.token_drop % c) % c
    }

    fn frame_overlap(&self) -> usize {
        (self.token_overlap() * self.temporal_ratio()).saturating_sub(self.frame_pre_padding())
    }

    /// Latent frames of a video of `frames` pixel frames.
    #[must_use]
    pub fn latent_frames(&self, frames: usize) -> usize {
        if frames <= 1 {
            return frames;
        }
        frames.div_ceil(self.clip_length) * self.tokens_chunk() - self.token_drop
    }

    fn host_tensors(&self, st: &SafeTensors, lt: WType) -> Result<Vec<HostTensor>> {
        let mut h = Host { st, v: Vec::new() };
        let c = &self.block_out_channels;
        h.conv3("encoder.conv_in", self.in_channels, c[0])?;
        let mut cin = c[0];
        for (i, &cout) in c.iter().enumerate() {
            for r in 0..self.layers_per_block {
                let p = format!("encoder.down_blocks.{i}.resnets.{r}");
                let rin = if r == 0 { cin } else { cout };
                h.vector(&format!("{p}.norm1.weight"), rin)?;
                h.vector(&format!("{p}.norm1.bias"), rin)?;
                h.conv3(&format!("{p}.conv1"), rin, cout)?;
                h.vector(&format!("{p}.norm2.weight"), cout)?;
                h.vector(&format!("{p}.norm2.bias"), cout)?;
                h.conv3(&format!("{p}.conv2"), cout, cout)?;
                if rin != cout {
                    h.conv1(&format!("{p}.conv_shortcut"), rin, cout)?;
                }
            }
            if self.spatial_downsample_factors[i] * self.temporal_downsample_factors[i] > 1 {
                h.conv3(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"), cout, cout)?;
            }
            cin = cout;
        }
        let top = *c.last().expect("validated");
        h.vector("encoder.norm_out.weight", top)?;
        h.vector("encoder.norm_out.bias", top)?;
        let l = self.latent_channels;
        h.conv3("encoder.conv_out", top, 2 * l)?;
        h.conv1("quant_conv", 2 * l, 2 * l)?;

        let d = self.width();
        h.linear("post_quant_conv", l, l, WType::F32)?;
        h.linear("decoder.proj_in", l, d, lt)?;
        let r = self.decoder_num_register_tokens;
        let regs = h.st.require("decoder.register_tokens", &[1, r, d])?.to_f32();
        h.v.push(HostTensor { name: "decoder.register_tokens".into(), shape: vec![r, d], ty: WType::F32, data: regs });
        let inner = d * self.decoder_ffn_mult;
        for i in 0..self.decoder_num_layers {
            let p = format!("decoder.transformer_blocks.{i}");
            for n in ["norm1.weight", "scale1", "norm2.weight", "scale2"] {
                h.vector(&format!("{p}.{n}"), d)?;
            }
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                h.linear(&format!("{p}.attn.{n}"), d, d, lt)?;
            }
            h.linear(&format!("{p}.ff.net.0.proj"), d, 2 * inner, lt)?;
            h.linear(&format!("{p}.ff.net.2"), inner, d, lt)?;
        }
        h.vector("decoder.norm_out.weight", d)?;
        h.vector("decoder.norm_out.bias", d)?;
        let (s, t) = (self.spatial_ratio() as u64, self.temporal_ratio() as u64);
        h.linear("decoder.proj_out", d, self.out_channels * t * s * s, lt)?;
        Ok(h.v)
    }
}

struct Host<'a> {
    st: &'a SafeTensors,
    v: Vec<HostTensor>,
}

impl Host<'_> {
    fn vector(&mut self, name: &str, n: u64) -> Result<()> {
        let data = self.st.require(name, &[n])?.to_f32();
        self.v.push(HostTensor { name: name.into(), shape: vec![n], ty: WType::F32, data });
        Ok(())
    }

    /// A 3x3x3 kernel as three 3x3 frame taps `{p}.t0..t2`.
    fn conv3(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.st.require(&format!("{p}.weight"), &[cout, cin, 3, 3, 3])?.to_f32();
        for k in 0..3 {
            let d: Vec<f32> = w.chunks_exact(27).flat_map(|c| (0..9).map(move |j| c[k * 9 + j])).collect();
            self.v.push(HostTensor { name: format!("{p}.t{k}"), shape: vec![cout, cin, 3, 3], ty: WType::F32, data: d });
        }
        self.vector(&format!("{p}.bias"), cout)
    }

    /// A 1x1x1 kernel as one 1x1 tap `{p}.t0`.
    fn conv1(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.st.require(&format!("{p}.weight"), &[cout, cin, 1, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.t0"), shape: vec![cout, cin, 1, 1], ty: WType::F32, data: w });
        self.vector(&format!("{p}.bias"), cout)
    }

    /// A linear map (a 1x1x1 convolution counts as one).
    fn linear(&mut self, p: &str, din: u64, dout: u64, ty: WType) -> Result<()> {
        let name = format!("{p}.weight");
        let t = self.st.get(&name).ok_or_else(|| Error::Weights(format!("missing {name}")))?;
        let shape: Vec<u64> = t.shape.to_vec();
        if shape.len() < 2 || shape[0] != dout || shape[1] != din || shape[2..].iter().any(|&x| x != 1) {
            return Err(Error::Weights(format!("{name}: shape {shape:?}, expected [{dout}, {din}]")));
        }
        self.v.push(HostTensor { name, shape: vec![dout, din], ty, data: t.to_f32() });
        self.vector(&format!("{p}.bias"), dout)
    }
}

fn bias4(g: &mut Graph, w: &Weights, name: &str) -> Tn {
    let b = w.get(name);
    g.reshape(b, &[1, 1, b.ne(0), 1])
}

struct Enc<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
    groups: i32,
    eps: f32,
}

impl Enc<'_, '_> {
    fn col(&mut self, x: Tn, i: i64) -> Tn {
        let v = self.g.view_4d(x, [1, x.ne(1), x.ne(2), x.ne(3)], x.nb(1), x.nb(2), x.nb(3), i as usize * x.nb(0));
        self.g.cont(v)
    }

    fn row(&mut self, x: Tn, i: i64) -> Tn {
        let v = self.g.view_4d(x, [x.ne(0), 1, x.ne(2), x.ne(3)], x.nb(1), x.nb(2), x.nb(3), i as usize * x.nb(1));
        self.g.cont(v)
    }

    /// Reflect pad width and height by one: on both sides, or only after.
    fn reflect(&mut self, x: Tn, both: bool) -> Tn {
        let right = self.col(x, x.ne(0) - 2);
        let mut y = self.g.concat(x, right, 0);
        if both {
            let left = self.col(x, 1);
            y = self.g.concat(left, y, 0);
        }
        let bottom = self.row(y, y.ne(1) - 2);
        let mut z = self.g.concat(y, bottom, 1);
        if both {
            let top = self.row(y, 1);
            z = self.g.concat(top, z, 1);
        }
        z
    }

    /// Causal 3x3x3 convolution of an already spatially padded input.
    fn conv3(&mut self, p: &str, x: Tn, t_stride: i64, s_stride: i32) -> Tn {
        let x = self.g.pad_ext(x, [0, 0, 0, 2], [0; 4]);
        let frames = (x.ne(3) - 3) / t_stride + 1;
        let mut acc: Option<Tn> = None;
        for k in 0..3 {
            let v = self.g.view_4d(x, [x.ne(0), x.ne(1), x.ne(2), frames], x.nb(1), x.nb(2), x.nb(3) * t_stride as usize, k * x.nb(3));
            let v = self.g.cont(v);
            let kern = self.w.get(&format!("{p}.t{k}"));
            let y = if s_stride == 1 { self.g.conv2d(kern, v, 0) } else { self.g.conv2d_strided(kern, v, s_stride) };
            acc = Some(match acc {
                None => y,
                Some(a) => self.g.add(a, y),
            });
        }
        let b = bias4(self.g, self.w, &format!("{p}.bias"));
        self.g.add(acc.expect("three taps"), b)
    }

    fn conv1(&mut self, p: &str, x: Tn) -> Tn {
        let y = self.g.conv2d(self.w.get(&format!("{p}.t0")), x, 0);
        let b = bias4(self.g, self.w, &format!("{p}.bias"));
        self.g.add(y, b)
    }

    fn norm_silu(&mut self, p: &str, x: Tn) -> Tn {
        let n = self.g.group_norm(x, self.groups, self.eps);
        let s = bias4(self.g, self.w, &format!("{p}.weight"));
        let n = self.g.mul(n, s);
        let b = bias4(self.g, self.w, &format!("{p}.bias"));
        let n = self.g.add(n, b);
        self.g.silu(n)
    }

    fn padded_conv(&mut self, p: &str, x: Tn) -> Tn {
        let x = self.reflect(x, true);
        self.conv3(p, x, 1, 1)
    }

    fn resnet(&mut self, p: &str, x: Tn, shortcut: bool) -> Tn {
        let h = self.norm_silu(&format!("{p}.norm1"), x);
        let h = self.padded_conv(&format!("{p}.conv1"), h);
        let h = self.norm_silu(&format!("{p}.norm2"), h);
        let h = self.padded_conv(&format!("{p}.conv2"), h);
        let r = if shortcut { self.conv1(&format!("{p}.conv_shortcut"), x) } else { x };
        self.g.add(r, h)
    }
}

/// Spatial tiling of encoding and decoding, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    pub height: usize,
    pub width: usize,
    pub overlap_height: usize,
    pub overlap_width: usize,
}

impl Default for Tiling {
    /// The released geometry: 256 pixel tiles overlapping by at least 64.
    fn default() -> Self {
        Self { height: 256, width: 256, overlap_height: 64, overlap_width: 64 }
    }
}

/// A host volume `[c][t][h][w]`.
#[derive(Debug, Clone)]
struct Vol {
    dims: [usize; 4],
    d: Vec<f32>,
}

impl Vol {
    fn new(dims: [usize; 4], d: Vec<f32>) -> Self {
        debug_assert_eq!(d.len(), dims.iter().product::<usize>());
        Self { dims, d }
    }

    /// `n` entries of `axis` from `from`, clamped to the extent.
    fn take(&self, axis: usize, from: usize, n: usize) -> Self {
        let len = self.dims[axis];
        let from = from.min(len);
        let n = n.min(len - from);
        let mut dims = self.dims;
        dims[axis] = n;
        let inner: usize = self.dims[axis + 1..].iter().product();
        let outer: usize = self.dims[..axis].iter().product();
        let mut d = Vec::with_capacity(outer * n * inner);
        for o in 0..outer {
            let base = (o * len + from) * inner;
            d.extend_from_slice(&self.d[base..base + n * inner]);
        }
        Self { dims, d }
    }

    fn cat(parts: &[Self], axis: usize) -> Self {
        let mut dims = parts[0].dims;
        dims[axis] = parts.iter().map(|p| p.dims[axis]).sum();
        let inner: usize = dims[axis + 1..].iter().product();
        let outer: usize = dims[..axis].iter().product();
        let mut d = Vec::with_capacity(dims.iter().product());
        for o in 0..outer {
            for p in parts {
                let n = p.dims[axis] * inner;
                d.extend_from_slice(&p.d[o * n..(o + 1) * n]);
            }
        }
        Self { dims, d }
    }

    fn len(&self, axis: usize) -> usize {
        self.dims[axis]
    }

    /// Cross-fade the last `extent` entries of `a` into the first of `b`.
    fn blend(a: &Self, b: &Self, extent: usize, axis: usize) -> Self {
        let e = extent.min(a.len(axis)).min(b.len(axis));
        if e == 0 {
            return b.clone();
        }
        let ta = a.take(axis, a.len(axis) - e, e);
        let mut mixed = b.take(axis, 0, e);
        let inner: usize = mixed.dims[axis + 1..].iter().product();
        for (i, (m, x)) in mixed.d.iter_mut().zip(&ta.d).enumerate() {
            let p = ((i / inner) % e) as f32 / e as f32;
            *m = *x * (1.0 - p) + *m * p;
        }
        if e == b.len(axis) {
            return mixed;
        }
        Self::cat(&[mixed, b.take(axis, e, b.len(axis) - e)], axis)
    }
}

/// Tile starts, lengths and overlaps over `length` pixels (the reference's
/// `_split_tiles`): the fewest tiles keeping every overlap at least
/// `min_overlap`, the slack spread over the overlaps in `ratio` steps.
fn split_tiles(length: usize, tile: usize, min_overlap: usize, ratio: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if tile >= length {
        return (vec![0], vec![length], Vec::new());
    }
    let mut n = length.div_ceil(tile);
    while tile * n < min_overlap * (n - 1) + length {
        n += 1;
    }
    let mut overlaps = vec![min_overlap; n - 1];
    let remaining = tile * n - min_overlap * (n - 1) - length;
    for i in 0..remaining / ratio {
        overlaps[i % (n - 1)] += ratio;
    }
    let mut starts = vec![0];
    for o in &overlaps {
        starts.push(starts.last().expect("nonempty") + tile - o);
    }
    (starts, vec![tile; n], overlaps)
}

/// Blend a grid of tiles over their overlaps (the reference's `_stitch_tiles`).
fn stitch(tiles: &[Vec<Vol>], h_overlaps: &[usize], w_overlaps: &[usize]) -> Vol {
    let mut rows = Vec::with_capacity(tiles.len());
    for (i, row) in tiles.iter().enumerate() {
        let mut out = Vec::with_capacity(row.len());
        for (j, tile) in row.iter().enumerate() {
            let mut t = tile.clone();
            if i > 0 {
                t = Vol::blend(&tiles[i - 1][j], &t, h_overlaps[i - 1], 2);
            }
            if j > 0 {
                t = Vol::blend(&row[j - 1], &t, w_overlaps[j - 1], 3);
            }
            if i + 1 < tiles.len() {
                t = t.take(2, 0, t.len(2).saturating_sub(h_overlaps[i]));
            }
            if j + 1 < row.len() {
                t = t.take(3, 0, t.len(3).saturating_sub(w_overlaps[j]));
            }
            out.push(t);
        }
        rows.push(Vol::cat(&out, 3));
    }
    Vol::cat(&rows, 2)
}

/// A loaded MiniMax-H3 video autoencoder.
pub struct H3VideoVae {
    backend: Backend,
    cfg: H3VideoVaeConfig,
    w: Weights,
    exact: bool,
    /// Spatial tiling; `None` processes whole frames.
    pub tiling: Option<Tiling>,
}

impl std::fmt::Debug for H3VideoVae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H3VideoVae").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl H3VideoVae {
    /// Load `vae/` of a checkpoint. The encoder always runs in f32 (as the
    /// reference keeps it); decoder matrices follow the precision.
    ///
    /// # Errors
    /// A missing or malformed config or weight, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: H3VideoVaeConfig = parse(files.json("vae/config.json")?, "video autoencoder config")?;
        cfg.validate()?;
        let backend = opts.backend()?;
        let st = SafeTensors::open(&files.weights("vae")?)?;
        let w = Weights::from_host(&backend, &cfg.host_tensors(&st, opts.precision.wtype())?)?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32, tiling: Some(Tiling::default()) })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &H3VideoVaeConfig {
        &self.cfg
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Encode pixels `[3][frames][height][width]` to normalised latents
    /// `[C][T][H][W]` (the posterior mean); returns them with `(T, H, W)`.
    ///
    /// # Errors
    /// A pixel buffer that disagrees with its shape, or a backend failure.
    pub fn encode(&self, pixels: &[f32], frames: usize, height: usize, width: usize) -> Result<(Vec<f32>, [usize; 3])> {
        let z = self.moments(pixels, frames, height, width)?;
        let l = self.cfg.latent_channels as usize;
        let mut z = z.take(0, 0, l);
        let plane = z.len(1) * z.len(2) * z.len(3);
        for (ch, p) in z.d.chunks_exact_mut(plane).enumerate() {
            let (m, s) = (self.cfg.latents_mean[ch], self.cfg.latents_std[ch]);
            for v in p {
                *v = (*v - m) / s;
            }
        }
        let dims = [z.len(1), z.len(2), z.len(3)];
        Ok((z.d, dims))
    }

    /// Latent `[C][1][H][W]` of one 8-bit image, as a conditioning anchor:
    /// ImageNet-normalised pixels, the posterior sampled with `eps` (standard
    /// normal noise of the latent's size), rounded to float16, normalised.
    ///
    /// # Errors
    /// Noise of the wrong size, or a backend failure.
    pub fn encode_condition(&self, image: &RgbImage, eps: &[f32]) -> Result<(Vec<f32>, [usize; 3])> {
        let (h, w) = (image.height as usize, image.width as usize);
        let plane = h * w;
        let mut px = vec![0f32; 3 * plane];
        for (i, rgb) in image.rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                px[c * plane + i] = f32::from(rgb[c]) / 255.0;
            }
        }
        from_unit_rgb(&mut px);
        let z = self.moments(&px, 1, h, w)?;
        let l = self.cfg.latent_channels as usize;
        let n = l * z.len(1) * z.len(2) * z.len(3);
        if eps.len() != n {
            return Err(Error::Request(format!("posterior noise has {} values, the latent {n}", eps.len())));
        }
        let lat = z.len(1) * z.len(2) * z.len(3);
        let mut out = Vec::with_capacity(n);
        for (i, &e) in eps.iter().enumerate() {
            let ch = i / lat;
            let (mean, logvar) = (z.d[i], z.d[n + i].clamp(-30.0, 20.0));
            let v = half::f16::from_f32(mean + (0.5 * logvar).exp() * e).to_f32();
            out.push((v - self.cfg.latents_mean[ch]) / self.cfg.latents_std[ch]);
        }
        Ok((out, [z.len(1), z.len(2), z.len(3)]))
    }

    /// Posterior moments `[2 C][T][H][W]` (mean channels, then log-variance)
    /// of pixels `[3][frames][height][width]`.
    fn moments(&self, pixels: &[f32], frames: usize, height: usize, width: usize) -> Result<Vol> {
        let c = self.cfg.in_channels as usize;
        if frames == 0 || height < 2 || width < 2 || pixels.len() != c * frames * height * width {
            return Err(Error::Request("video pixels disagree with their shape".into()));
        }
        let x = Vol::new([c, frames, height, width], pixels.to_vec());
        let z = if frames == 1 {
            self.encode_clip(&x)?
        } else {
            let clip = self.cfg.clip_length;
            let pad = (clip - frames % clip) % clip;
            let x = if pad > 0 {
                let last = x.take(1, frames - 1, 1);
                let mut parts = vec![x];
                parts.extend(std::iter::repeat_n(last, pad));
                Vol::cat(&parts, 1)
            } else {
                x
            };
            let clips = (0..x.len(1) / clip).map(|i| self.encode_clip(&x.take(1, i * clip, clip))).collect::<Result<Vec<_>>>()?;
            let z = Vol::cat(&clips, 1);
            z.take(1, 0, z.len(1).saturating_sub(self.cfg.token_drop))
        };
        Ok(z)
    }

    /// Decode normalised latents `[C][frames][height][width]` to pixels
    /// `[3][T][H s][W s]` in the autoencoder's pixel space; returns them with
    /// `(T, H s, W s)`. See [`to_unit_rgb`] for plain RGB.
    ///
    /// # Errors
    /// A latent buffer that disagrees with its shape, or a backend failure.
    pub fn decode(&self, latent: &[f32], frames: usize, height: usize, width: usize) -> Result<(Vec<f32>, [usize; 3])> {
        let c = self.cfg.latent_channels as usize;
        if frames == 0 || height == 0 || width == 0 || latent.len() != c * frames * height * width {
            return Err(Error::Request("video latent size disagrees with its shape".into()));
        }
        let mut d = latent.to_vec();
        for (ch, p) in d.chunks_exact_mut(frames * height * width).enumerate() {
            let (m, s) = (self.cfg.latents_mean[ch], self.cfg.latents_std[ch]);
            for v in p {
                *v = *v * s + m;
            }
        }
        let z = Vol::new([c, frames, height, width], d);
        let cfg = &self.cfg;
        let (chunk, drop, tr) = (cfg.tokens_chunk(), cfg.token_drop, cfg.temporal_ratio());
        let chunk_frames = chunk * tr;
        let num_tokens = frames + drop;
        let pad = (chunk - num_tokens % chunk) % chunk;
        let dropping = usize::from(drop > 0);
        let chunks = (num_tokens + pad) / chunk - dropping;
        if chunks == 0 {
            return Err(Error::Request(format!("a video latent needs at least {} frames", chunk + 1 - drop)));
        }
        let z = if pad > 0 {
            let last = z.take(1, frames - 1, 1);
            let mut parts = vec![z];
            parts.extend(std::iter::repeat_n(last, pad));
            Vol::cat(&parts, 1)
        } else {
            z
        };
        let (pre, fo) = (cfg.frame_pre_padding(), cfg.frame_overlap());
        let mut out = Vec::new();
        let mut overlap: Option<Vol> = None;
        for i in 0..chunks {
            let clip = self.decode_clip(&z.take(1, i * chunk, chunk + cfg.token_overlap()))?;
            for j in 0..=dropping {
                let part = clip.take(1, j * chunk_frames, chunk_frames);
                let part = part.take(1, pre, part.len(1).saturating_sub(pre));
                if j == 0 {
                    out.push(match &overlap {
                        Some(o) => Vol::blend(o, &part, fo, 1),
                        None => part,
                    });
                } else {
                    overlap = Some(part);
                }
            }
        }
        out.extend(overlap);
        let mut dec = Vol::cat(&out, 1);
        if pad > 0 {
            let tail = cfg.clip_length % tr;
            let before = z.len(1) - pad;
            let cut: usize = (0..pad).map(|k| if tail != 0 && (before + k).is_multiple_of(chunk) { tail } else { tr }).sum();
            dec = dec.take(1, 0, dec.len(1).saturating_sub(cut));
        }
        let dims = [dec.len(1), dec.len(2), dec.len(3)];
        Ok((dec.d, dims))
    }

    fn encode_clip(&self, x: &Vol) -> Result<Vol> {
        let Some(t) = self.tiling else {
            return self.encode_tile(x);
        };
        let r = self.cfg.spatial_ratio();
        let (ys, yl, yo) = split_tiles(x.len(2), t.height, t.overlap_height, r);
        let (xs, xl, xo) = split_tiles(x.len(3), t.width, t.overlap_width, r);
        let mut rows = Vec::with_capacity(ys.len());
        for (&y, &h) in ys.iter().zip(&yl) {
            let band = x.take(2, y, h);
            rows.push(xs.iter().zip(&xl).map(|(&x0, &w)| self.encode_tile(&band.take(3, x0, w))).collect::<Result<Vec<_>>>()?);
        }
        let yo: Vec<usize> = yo.iter().map(|o| o / r).collect();
        let xo: Vec<usize> = xo.iter().map(|o| o / r).collect();
        Ok(stitch(&rows, &yo, &xo))
    }

    fn decode_clip(&self, z: &Vol) -> Result<Vol> {
        let Some(t) = self.tiling else {
            return self.decode_tile(z);
        };
        let r = self.cfg.spatial_ratio();
        let (ys, yl, yo) = split_tiles(z.len(2) * r, t.height, t.overlap_height, r);
        let (xs, xl, xo) = split_tiles(z.len(3) * r, t.width, t.overlap_width, r);
        let mut rows = Vec::with_capacity(ys.len());
        for (&y, &h) in ys.iter().zip(&yl) {
            let band = z.take(2, y / r, h / r);
            rows.push(xs.iter().zip(&xl).map(|(&x0, &w)| self.decode_tile(&band.take(3, x0 / r, w / r))).collect::<Result<Vec<_>>>()?);
        }
        Ok(stitch(&rows, &yo, &xo))
    }

    /// The encoder on one clip tile: the posterior mean, not normalised.
    fn encode_tile(&self, x: &Vol) -> Result<Vol> {
        let cfg = &self.cfg;
        let [c, t, h, w] = x.dims;
        if h < 2 || w < 2 {
            return Err(Error::Request("video tile is too small to reflect pad".into()));
        }
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[w as i64, h as i64, c as i64, t as i64]);
        let mut e = Enc { g: &mut g, w: &self.w, groups: cfg.norm_num_groups as i32, eps: cfg.norm_eps as f32 };
        let mut y = e.padded_conv("encoder.conv_in", input);
        let mut cin = cfg.block_out_channels[0];
        for (i, &cout) in cfg.block_out_channels.iter().enumerate() {
            for r in 0..cfg.layers_per_block {
                let rin = if r == 0 { cin } else { cout };
                y = e.resnet(&format!("encoder.down_blocks.{i}.resnets.{r}"), y, rin != cout);
            }
            let (s, ts) = (cfg.spatial_downsample_factors[i], cfg.temporal_downsample_factors[i]);
            if s * ts > 1 {
                if y.ne(0) < 2 || y.ne(1) < 2 {
                    return Err(Error::Request("video tile is too small to downsample".into()));
                }
                let p = e.reflect(y, false);
                y = e.conv3(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"), p, ts as i64, s as i32);
            }
            cin = cout;
        }
        let y = e.norm_silu("encoder.norm_out", y);
        let y = e.padded_conv("encoder.conv_out", y);
        let y = e.conv1("quant_conv", y);
        g.finish(&[y])?;
        // [C][T][H][W] -> [T][C][H][W].
        let plane = h * w;
        let mut d = Vec::with_capacity(x.d.len());
        for f in 0..t {
            for ch in 0..c {
                let o = (ch * t + f) * plane;
                d.extend_from_slice(&x.d[o..o + plane]);
            }
        }
        g.set_f32(input, &d);
        g.compute()?;
        let out = g.read_f32(y);
        let (ow, oh, oc, ot) = (y.ne(0) as usize, y.ne(1) as usize, y.ne(2) as usize, y.ne(3) as usize);
        // Mean and log-variance channels both: the moments.
        let p = oh * ow;
        let mut m = Vec::with_capacity(oc * ot * p);
        for ch in 0..oc {
            for f in 0..ot {
                let o = (f * oc + ch) * p;
                m.extend_from_slice(&out[o..o + p]);
            }
        }
        Ok(Vol::new([oc, ot, oh, ow], m))
    }

    /// The decoder on one clip tile of raw latents.
    fn decode_tile(&self, z: &Vol) -> Result<Vol> {
        let cfg = &self.cfg;
        let [l, t, h, w] = z.dims;
        let n = t * h * w;
        let regs = cfg.decoder_num_register_tokens as usize;
        let total = n + regs + 1;
        let (d, hd, heads) = (cfg.width() as i64, cfg.decoder_attention_head_dim as i64, cfg.decoder_num_attention_heads as i64);
        let rot = cfg.rope_dim() as i64;
        let eps = cfg.decoder_norm_eps as f32;
        let wt = &self.w;
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[l as i64, n as i64]);
        let cos = g.input(sys::GGML_TYPE_F32, &[rot, 1, total as i64]);
        let sin = g.input(sys::GGML_TYPE_F32, &[rot, 1, total as i64]);
        let lin = |g: &mut Graph, p: &str, x: Tn| g.linear_b(wt.get(&format!("{p}.weight")), wt.get(&format!("{p}.bias")), x);
        let x = lin(&mut g, "post_quant_conv", input);
        let x = lin(&mut g, "decoder.proj_in", x);
        let x = g.concat(x, wt.get("decoder.register_tokens"), 1);
        let mut x = g.pad_ext(x, [0; 4], [0, 1, 0, 0]);
        let tot = total as i64;
        for i in 0..cfg.decoder_num_layers {
            let p = format!("decoder.transformer_blocks.{i}");
            let hn = g.rms_norm(x, eps);
            let hn = g.mul(hn, wt.get(&format!("{p}.norm1.weight")));
            let mut qkv = [hn; 3];
            for (o, name) in qkv.iter_mut().zip(["to_q", "to_k", "to_v"]) {
                let y = lin(&mut g, &format!("{p}.attn.{name}"), hn);
                *o = g.reshape(y, &[hd, heads, tot]);
            }
            let [q, k, v] = qkv;
            let q = g.rms_norm(q, eps);
            let k = g.rms_norm(k, eps);
            let q = partial_rope(&mut g, q, rot, (cos, sin));
            let k = partial_rope(&mut g, k, rot, (cos, sin));
            let q = g.permute(q, [0, 2, 1, 3]);
            let k = g.permute(k, [0, 2, 1, 3]);
            let v = g.permute(v, [0, 2, 1, 3]);
            let scale = 1.0 / (hd as f32).sqrt();
            let o = if self.exact {
                let k = g.cont(k);
                let v = g.cont(v);
                g.attention_exact(q, k, v, None, scale)
            } else {
                let k = g.cast(k, sys::GGML_TYPE_F16);
                let v = g.cast(v, sys::GGML_TYPE_F16);
                g.attention(q, k, v, None, scale, true)
            };
            let o = g.reshape(o, &[d, tot]);
            let a = lin(&mut g, &format!("{p}.attn.to_out.0"), o);
            let a = g.mul(a, wt.get(&format!("{p}.scale1")));
            x = g.add(x, a);
            let hn = g.rms_norm(x, eps);
            let hn = g.mul(hn, wt.get(&format!("{p}.norm2.weight")));
            let f = lin(&mut g, &format!("{p}.ff.net.0.proj"), hn);
            let half = f.ne(0) / 2;
            let up = g.view_rows(f, 0, half);
            let up = g.cont(up);
            let gate = g.view_rows(f, half, half);
            let gate = g.cont(gate);
            let f = g.swiglu_split(gate, up);
            let f = lin(&mut g, &format!("{p}.ff.net.2"), f);
            let f = g.mul(f, wt.get(&format!("{p}.scale2")));
            x = g.add(x, f);
        }
        let x = g.norm(x, eps);
        let x = g.mul(x, wt.get("decoder.norm_out.weight"));
        let x = g.add(x, wt.get("decoder.norm_out.bias"));
        let x = g.view_cols(x, 0, n as i64);
        let x = g.cont(x);
        let y = lin(&mut g, "decoder.proj_out", x);
        g.finish(&[y])?;

        let mut tok = vec![0f32; l * n];
        let plane = n;
        for ch in 0..l {
            for (j, v) in z.d[ch * plane..(ch + 1) * plane].iter().enumerate() {
                tok[j * l + ch] = *v;
            }
        }
        g.set_f32(input, &tok);
        let (cs, sn) = decoder_rotary(cfg, rot as usize, [t, h, w], total);
        g.set_f32(cos, &cs);
        g.set_f32(sin, &sn);
        g.compute()?;
        let out = g.read_f32(y);
        let (s, tr, oc) = (cfg.spatial_ratio(), cfg.temporal_ratio(), cfg.out_channels as usize);
        let width_tok = oc * tr * s * s;
        let (ot, oh, ow) = (t * tr, h * s, w * s);
        let mut px = vec![0f32; oc * ot * oh * ow];
        for (j, row) in out.chunks_exact(width_tok).enumerate() {
            let (f, yy, xx) = (j / (h * w), (j / w) % h, j % w);
            for (k, v) in row.iter().enumerate() {
                let (c, it, ih, iw) = (k / (tr * s * s), (k / (s * s)) % tr, (k / s) % s, k % s);
                px[((c * ot + f * tr + it) * oh + yy * s + ih) * ow + xx * s + iw] = *v;
            }
        }
        Ok(Vol::new([oc, ot, oh, ow], px))
    }
}

/// Rotate the leading `rot` channels of every head of `x` `[hd, heads, n]`.
fn partial_rope(g: &mut Graph, x: Tn, rot: i64, (cos, sin): (Tn, Tn)) -> Tn {
    let hd = x.ne(0);
    if rot == hd {
        return g.rotate_half_rope(x, cos, sin);
    }
    let (heads, n) = (x.ne(1), x.ne(2));
    let head = g.view_4d(x, [rot, heads, n, 1], x.nb(1), x.nb(2), x.nb(3), 0);
    let head = g.cont(head);
    let tail = g.view_4d(x, [hd - rot, heads, n, 1], x.nb(1), x.nb(2), x.nb(3), rot as usize * x.nb(0));
    let tail = g.cont(tail);
    let head = g.rotate_half_rope(head, cos, sin);
    g.concat(head, tail, 0)
}

/// Decoder rotary cos and sin `[tokens][rot]`: voxel tokens at their
/// normalised `(t, h, w)` centres, then the register and zero tokens at 0.
fn decoder_rotary(cfg: &H3VideoVaeConfig, rot: usize, [t, h, w]: [usize; 3], total: usize) -> (Vec<f32>, Vec<f32>) {
    let f = rot / 6;
    let theta = cfg.decoder_rope_theta as f32;
    let step = 6.0 / rot as f32;
    let inv: Vec<f32> = (0..f).map(|j| 1.0 / theta.powf(j as f32 * step)).collect();
    let grid = |size: usize, i: usize| 2.0 * ((i as f32 + 0.5) / size as f32) - 1.0;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut cos = Vec::with_capacity(total * rot);
    let mut sin = Vec::with_capacity(total * rot);
    for j in 0..total {
        let pos = if j < t * h * w { [grid(t, j / (h * w)), grid(h, (j / w) % h), grid(w, j % w)] } else { [0.0; 3] };
        let ang: Vec<f32> = (0..3).flat_map(|a| inv.iter().map(move |q| two_pi * pos[a] * q)).collect();
        for _ in 0..2 {
            cos.extend(ang.iter().map(|a| a.cos()));
            sin.extend(ang.iter().map(|a| a.sin()));
        }
    }
    (cos, sin)
}

/// Pixels in the autoencoder's space to RGB in `[0, 1]`, in place.
pub fn to_unit_rgb(pixels: &mut [f32]) {
    let plane = pixels.len() / 3;
    for (c, p) in pixels.chunks_exact_mut(plane.max(1)).take(3).enumerate() {
        for v in p {
            *v = (*v * PIXEL_STD[c] + PIXEL_MEAN[c]).clamp(0.0, 1.0);
        }
    }
}

/// RGB in `[0, 1]` to the autoencoder's pixel space, in place.
pub fn from_unit_rgb(pixels: &mut [f32]) {
    let plane = pixels.len() / 3;
    for (c, p) in pixels.chunks_exact_mut(plane.max(1)).take(3).enumerate() {
        for v in p {
            *v = (*v - PIXEL_MEAN[c]) / PIXEL_STD[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn tiles_follow_the_reference_split() {
        assert_eq!(split_tiles(40, 32, 8, 4), (vec![0, 8], vec![32, 32], vec![24]));
        assert_eq!(split_tiles(60, 32, 8, 4), (vec![0, 12, 28], vec![32, 32, 32], vec![20, 16]));
        assert_eq!(split_tiles(20, 32, 8, 4), (vec![0], vec![20], vec![]));
    }

    #[test]
    fn released_clip_geometry() {
        let mut cfg: H3VideoVaeConfig = serde_json::from_value(serde_json::json!({
            "in_channels": 3, "out_channels": 3, "latent_channels": 1, "block_out_channels": [32, 32, 32],
            "layers_per_block": 1, "spatial_downsample_factors": [2, 2, 1], "temporal_downsample_factors": [2, 2, 1],
            "norm_num_groups": 32, "norm_eps": 1e-6, "spatial_padding_mode": "reflect", "decoder_num_layers": 1,
            "decoder_num_attention_heads": 1, "decoder_attention_head_dim": 64, "decoder_num_register_tokens": 4,
            "decoder_ffn_mult": 4, "decoder_rope_theta": 100.0, "decoder_rope_dim_ratio": 0.75, "decoder_norm_eps": 1e-5,
            "clip_length": 17, "token_drop": 3, "latents_mean": [0.0], "latents_std": [1.0]
        }))
        .unwrap();
        cfg.validate().unwrap();
        assert_eq!((cfg.tokens_chunk(), cfg.frame_pre_padding(), cfg.token_overlap(), cfg.frame_overlap()), (5, 3, 2, 5));
        assert_eq!(cfg.latent_frames(17 * 3 + 5), 17);
        assert_eq!(cfg.latent_frames(1), 1);
        cfg.decoder_rope_dim_ratio = 0.7;
        assert!(cfg.validate().is_err());
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = PathBuf::from(std::env::var("PRAECISE_MINIMAX_H3_VAE_PARITY").expect("PRAECISE_MINIMAX_H3_VAE_PARITY names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let mut vae = H3VideoVae::load(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap();
        let (tile, overlap) = (m["tile"].as_u64().unwrap() as usize, m["overlap"].as_u64().unwrap() as usize);
        let dims = |v: &Value| -> [usize; 3] { [0, 1, 2].map(|i| v[i].as_u64().unwrap() as usize) };
        for (name, case) in m["cases"].as_object().unwrap() {
            let [t, h, w] = dims(&case["input"]);
            vae.tiling = if case.get("tiled") == Some(&Value::Bool(false)) {
                None
            } else {
                Some(Tiling { height: tile, width: tile, overlap_height: overlap, overlap_width: overlap })
            };
            let input = bin(&d, &format!("{name}_in"));
            let (out, shape) = if name.starts_with("enc") { vae.encode(&input, t, h, w).unwrap() } else { vae.decode(&input, t, h, w).unwrap() };
            assert_eq!(shape, dims(&case["output"]), "{name}: shape");
            // The encoder is f32 at every precision.
            let (c, r) = if name.starts_with("enc") { (0.999_999, 1e-4) } else { (min_cos, max_rel) };
            assert_close(name, &out, &bin(&d, &format!("{name}_out")), c, r);
        }
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_video_vae_f32() {
        run(Precision::F32, 0.999_999, 1e-4);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_video_vae_bf16() {
        run(Precision::Bf16, 0.9999, 2e-2);
    }
}
