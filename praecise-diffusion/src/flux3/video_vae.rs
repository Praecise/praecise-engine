//! FLUX 3 video autoencoder.
//!
//! A hierarchical 3D transformer autoencoder. The encoder embeds 1x4x4
//! patches, then runs four stages of transformer blocks whose attention is a
//! 3D neighborhood (every token attends to a fixed window around itself,
//! shifted inward at the borders; causal along time in the encoder), with
//! rotary positions over time, height and width. Stages halve the spatial
//! grid by merging 2x2 patches, and the last two also halve time (an odd
//! frame count first repeats the first frame), so a clip of `4k + 1` frames
//! becomes `k + 1` latent frames at 1/32 of the resolution. The decoder
//! mirrors it with expansions. Latents are normalised per channel with the
//! running statistics stored in the checkpoint.
//!
//! Tokens live on the host between layers as `[tokens][channels]` with the
//! token order time-major, then height, then width; every layer's dense math
//! runs on the selected backend.

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::{Dtype, SafeTensors};
use llama_cpp_sys_2 as sys;

const LN_EPS: f32 = 1e-5;
const ROPE_BASE: f64 = 256.0;
/// Frames per encoder chunk.
pub const CHUNK_FRAMES: usize = 45;
/// Frames shared by consecutive encoder chunks.
pub const CHUNK_OVERLAP: usize = 1;

/// Most query chunks in one block graph.
const MAX_ATTENTION_CHUNKS: usize = 256;
/// Frames per latent frame (after the first).
pub const TEMPORAL_DOWNSAMPLE: usize = 4;
/// Pixels per latent cell along height and width.
pub const SPATIAL_DOWNSAMPLE: usize = 32;
/// Default byte budget for the gathered keys of one attention chunk.
const ATTENTION_CHUNK_BYTES: usize = 256 << 20;

/// Architecture of the autoencoder. The defaults are the released model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaeConfig {
    /// Latent channels.
    pub z_dim: usize,
    /// Width of the first stage; each later stage doubles it.
    pub embed_dim: usize,
    /// Patch size (time, height, width).
    pub patch: [usize; 3],
    /// Attention window (time, height, width).
    pub window: [usize; 3],
    /// Window for odd-indexed blocks of a stage, when set.
    pub alternate_window: Option<[usize; 3]>,
    /// Encoder blocks per stage.
    pub enc_depths: Vec<usize>,
    /// Decoder blocks per stage.
    pub dec_depths: Vec<usize>,
    /// Attention heads per stage.
    pub num_heads: Vec<usize>,
    /// Stages that also halve (encoder) or double (decoder) time.
    pub temporal: Vec<bool>,
    /// Causal attention along time in the encoder.
    pub enc_causal: bool,
    /// Causal attention along time in the decoder.
    pub dec_causal: bool,
    /// RMS-normalise queries and keys per head.
    pub qk_norm: bool,
    /// Layer norm after the patch embedding.
    pub patch_norm: bool,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            z_dim: 96,
            embed_dim: 256,
            patch: [1, 4, 4],
            window: [5, 5, 5],
            alternate_window: None,
            enc_depths: vec![1, 4, 8, 8],
            dec_depths: vec![1, 4, 8, 8],
            num_heads: vec![4, 8, 16, 32],
            temporal: vec![false, false, true, true],
            enc_causal: true,
            dec_causal: false,
            qk_norm: true,
            patch_norm: false,
        }
    }
}

impl VaeConfig {
    fn check(&self) -> Result<()> {
        let n = self.enc_depths.len();
        if n == 0 || [self.dec_depths.len(), self.num_heads.len(), self.temporal.len()].iter().any(|&l| l != n) {
            return Err(Error::Config("stage lists must all have one entry per stage".into()));
        }
        for (i, &h) in self.num_heads.iter().enumerate() {
            let dim = self.embed_dim << i;
            if h == 0 || dim % h != 0 || (dim / h) % 8 != 0 {
                return Err(Error::Config(format!("stage {i}: width {dim} over {h} heads needs a head width divisible by 8")));
            }
        }
        if self.window.iter().chain(self.alternate_window.iter().flatten()).any(|&k| k == 0 || k % 2 == 0) {
            return Err(Error::Config("attention windows must be odd".into()));
        }
        Ok(())
    }

    fn window_for(&self, i: usize) -> [usize; 3] {
        match self.alternate_window {
            Some(a) if i % 2 == 1 => a,
            _ => self.window,
        }
    }

    /// Latent grid `(frames, height, width)` for a clip.
    #[must_use]
    pub fn latent_grid(&self, frames: usize, h: usize, w: usize) -> (usize, usize, usize) {
        let down = 1usize << (self.enc_depths.len() - 1);
        let t_down = 1usize << self.temporal.iter().filter(|&&t| t).count();
        let mut t = frames.div_ceil(self.patch[0]);
        for _ in 0..t_down.trailing_zeros() {
            t = t.div_ceil(2);
        }
        (t, h.div_ceil(self.patch[1]).div_ceil(down), w.div_ceil(self.patch[2]).div_ceil(down))
    }
}

fn glu_hidden(dim: usize) -> usize {
    (dim * 8 / 3).div_ceil(64) * 64
}

#[derive(Debug, Clone)]
enum Layer {
    Block { p: String, dim: usize, heads: usize, window: [usize; 3], causal: bool },
    PatchMerge { p: String, dim: usize, out: usize },
    TemporalMerge { p: String, dim: usize },
    PatchExpand { p: String, dim: usize, out: usize },
    TemporalExpand { p: String, dim: usize },
}

fn encoder_layers(c: &VaeConfig) -> Vec<Layer> {
    let n = c.enc_depths.len();
    let mut out = Vec::new();
    let mut idx = 0;
    for stage in 0..n {
        let dim = c.embed_dim << stage;
        for j in 0..c.enc_depths[stage] {
            out.push(Layer::Block { p: format!("encoder.features.{idx}.{j}"), dim, heads: c.num_heads[stage], window: c.window_for(j), causal: c.enc_causal });
        }
        idx += 1;
        let down = stage + 1 < n;
        if down {
            out.push(Layer::PatchMerge { p: format!("encoder.features.{idx}"), dim, out: 2 * dim });
            idx += 1;
        }
        if c.temporal[stage] {
            let (s, d) = if down { (stage + 1, 2 * dim) } else { (stage, dim) };
            out.push(Layer::Block { p: format!("encoder.features.{idx}"), dim: d, heads: c.num_heads[s], window: c.window_for(c.enc_depths[stage]), causal: c.enc_causal });
            out.push(Layer::TemporalMerge { p: format!("encoder.features.{}", idx + 1), dim: d });
            idx += 2;
        }
    }
    out
}

fn decoder_layers(c: &VaeConfig) -> Vec<Layer> {
    let mut out = Vec::new();
    let mut idx = 0;
    for stage in (0..c.dec_depths.len()).rev() {
        let dim = c.embed_dim << stage;
        let heads = c.num_heads[stage];
        for j in 0..c.dec_depths[stage] {
            out.push(Layer::Block { p: format!("decoder.features.{idx}.{j}"), dim, heads, window: c.window_for(j), causal: c.dec_causal });
        }
        idx += 1;
        if c.temporal[stage] {
            out.push(Layer::TemporalExpand { p: format!("decoder.features.{idx}"), dim });
            out.push(Layer::Block { p: format!("decoder.features.{}", idx + 1), dim, heads, window: c.window_for(c.dec_depths[stage]), causal: c.dec_causal });
            idx += 2;
        }
        if stage > 0 {
            out.push(Layer::PatchExpand { p: format!("decoder.features.{idx}"), dim, out: dim / 2 });
            idx += 1;
        }
    }
    out
}

fn norm_specs(v: &mut Vec<WeightSpec>, p: &str, d: usize) {
    v.push(WeightSpec::new(format!("{p}.weight"), &[d as u64], WType::F32));
    v.push(WeightSpec::new(format!("{p}.bias"), &[d as u64], WType::F32));
}

fn layer_specs(v: &mut Vec<WeightSpec>, l: &Layer, lin: WType) {
    let s = |n: String, shape: &[usize], ty| WeightSpec::new(n, &shape.iter().map(|&x| x as u64).collect::<Vec<_>>(), ty);
    match l {
        Layer::Block { p, dim, .. } => {
            let (d, h) = (*dim, glu_hidden(*dim));
            norm_specs(v, &format!("{p}.norm1"), d);
            norm_specs(v, &format!("{p}.norm2"), d);
            v.push(s(format!("{p}.attn.qkv.weight"), &[3 * d, d], lin));
            v.push(s(format!("{p}.attn.qkv.bias"), &[3 * d], WType::F32));
            v.push(s(format!("{p}.attn.proj.weight"), &[d, d], lin));
            v.push(s(format!("{p}.attn.proj.bias"), &[d], WType::F32));
            v.push(s(format!("{p}.mlp.gate_up_proj.weight"), &[2 * h, d], lin));
            v.push(s(format!("{p}.mlp.down_proj.weight"), &[d, h], lin));
        }
        Layer::PatchMerge { p, dim, out } => {
            norm_specs(v, &format!("{p}.norm"), 4 * dim);
            v.push(s(format!("{p}.reduction.weight"), &[*out, 4 * dim], lin));
        }
        Layer::TemporalMerge { p, dim } => {
            norm_specs(v, &format!("{p}.norm"), 2 * dim);
            v.push(s(format!("{p}.reduction.weight"), &[*dim, 2 * dim], lin));
        }
        Layer::PatchExpand { p, dim, out } => {
            norm_specs(v, &format!("{p}.norm"), *dim);
            v.push(s(format!("{p}.expansion.weight"), &[4 * out, *dim], lin));
        }
        Layer::TemporalExpand { p, dim } => {
            norm_specs(v, &format!("{p}.norm"), *dim);
            v.push(s(format!("{p}.expansion.weight"), &[2 * dim, *dim], lin));
        }
    }
}

/// Tokens on a `(t, h, w)` grid, `[t][h][w][c]` row-major.
#[derive(Debug, Clone)]
struct Grid {
    t: usize,
    h: usize,
    w: usize,
    c: usize,
    x: Vec<f32>,
}

impl Grid {
    fn n(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// Key indices (and an additive mask when the time window is causal) of a
/// 3D neighborhood attention over a `(t, h, w)` grid: `[tokens][window]`.
fn neighborhood(dims: [usize; 3], window: [usize; 3], causal: bool) -> Result<(Vec<i32>, Option<Vec<f32>>, usize)> {
    // A single frame attends over the spatial window only.
    let (window, causal) = if dims[0] == 1 { ([1, window[1], window[2]], false) } else { (window, causal) };
    for a in 0..3 {
        if window[a] > dims[a] {
            return Err(Error::Config(format!("attention window {window:?} exceeds the token grid {dims:?}")));
        }
    }
    let [t, h, w] = dims;
    let [kt, kh, kw] = window;
    let k = kt * kh * kw;
    let start = |i: usize, kk: usize, l: usize| i.saturating_sub(kk / 2).min(l - kk);
    let mut idx = Vec::with_capacity(t * h * w * k);
    let mut mask = causal.then(|| Vec::with_capacity(t * h * w * k));
    for it in 0..t {
        for ih in 0..h {
            for iw in 0..w {
                let (sh, sw) = (start(ih, kh, h), start(iw, kw, w));
                let me = ((it * h + ih) * w + iw) as i32;
                for a in 0..kt {
                    let tt = if causal { (it + a + 1).checked_sub(kt) } else { Some(start(it, kt, t) + a) };
                    for b in 0..kh {
                        for c in 0..kw {
                            match tt {
                                Some(tt) => idx.push(((tt * h + sh + b) * w + sw + c) as i32),
                                None => idx.push(me),
                            }
                            if let Some(m) = mask.as_mut() {
                                m.push(if tt.is_some() { 0.0 } else { f32::NEG_INFINITY });
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((idx, mask, k))
}

/// Rotary tables `[tokens][head_dim]` over time, height and width; the last
/// quarter of each half is unrotated.
fn rope_tables(dims: [usize; 3], hd: usize) -> (Vec<f32>, Vec<f32>) {
    let chunk = hd / 4;
    let freqs: Vec<f64> = (0..chunk / 2).map(|j| 1.0 / ROPE_BASE.powf((2 * j) as f64 / chunk as f64)).collect();
    let n = dims.iter().product::<usize>();
    let (mut cos, mut sin) = (Vec::with_capacity(n * hd), Vec::with_capacity(n * hd));
    let mut half = vec![0f64; hd / 2];
    for t in 0..dims[0] {
        for y in 0..dims[1] {
            for x in 0..dims[2] {
                for (axis, pos) in [t, y, x].into_iter().enumerate() {
                    for (j, f) in freqs.iter().enumerate() {
                        half[axis * chunk / 2 + j] = (pos as f32 * *f as f32) as f64;
                    }
                }
                for _ in 0..2 {
                    for a in &half {
                        cos.push(a.cos() as f32);
                        sin.push(a.sin() as f32);
                    }
                }
            }
        }
    }
    (cos, sin)
}

/// The loaded autoencoder.
pub struct VideoVae {
    backend: Backend,
    cfg: VaeConfig,
    w: Weights,
    host: Weights,
    mean: Vec<f32>,
    std: Vec<f32>,
    rms_eps: f32,
    chunk_bytes: usize,
}

impl std::fmt::Debug for VideoVae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoVae").field("device", &self.backend.name()).field("z_dim", &self.cfg.z_dim).finish_non_exhaustive()
    }
}

impl VideoVae {
    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Load from safetensors files whose tensor names carry `prefix`
    /// (stripped; tensors without it are ignored).
    ///
    /// # Errors
    /// On a configuration the checkpoint does not match, missing weights or
    /// no usable backend.
    pub fn load(files: &[std::path::PathBuf], prefix: &str, cfg: VaeConfig, opts: LoadOptions) -> Result<Self> {
        cfg.check()?;
        let st = SafeTensors::open(files)?.renamed(|n| n.strip_prefix(prefix).map(str::to_owned))?;
        let lin = match opts.precision {
            Precision::F32 => WType::F32,
            Precision::Bf16 => WType::Bf16,
            Precision::Q8_0 => return Err(Error::Config("the video autoencoder runs in bf16 or f32".into())),
        };
        let mut specs = Vec::new();
        for l in encoder_layers(&cfg).iter().chain(decoder_layers(&cfg).iter()) {
            layer_specs(&mut specs, l, lin);
        }
        let top = (cfg.embed_dim << (cfg.enc_depths.len() - 1)) as u64;
        let (z, e) = (cfg.z_dim as u64, cfg.embed_dim as u64);
        let pv = cfg.patch.iter().product::<usize>() as u64;
        specs.push(WeightSpec::new("encoder.patch_embed.proj.bias", &[e], WType::F32));
        if cfg.patch_norm {
            norm_specs(&mut specs, "encoder.patch_embed.norm", cfg.embed_dim);
        }
        specs.push(WeightSpec::new("encoder.proj.weight", &[2 * z, top], lin));
        specs.push(WeightSpec::new("encoder.proj.bias", &[2 * z], WType::F32));
        specs.push(WeightSpec::new("decoder.proj_in.weight", &[top, z], lin));
        specs.push(WeightSpec::new("decoder.proj_in.bias", &[top], WType::F32));
        specs.push(WeightSpec::new("decoder.proj_out.weight", &[pv * 3, e], lin));
        specs.push(WeightSpec::new("decoder.proj_out.bias", &[pv * 3], WType::F32));
        let backend = Backend::select(opts.cpu_threads)?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "video autoencoder backend selected");
        let w = Weights::load(&backend, &st, &specs)?;
        let conv = st.require("encoder.patch_embed.proj.weight", &[e, 3, cfg.patch[0] as u64, cfg.patch[1] as u64, cfg.patch[2] as u64])?;
        let host = Weights::from_host(&backend, &[HostTensor { name: "patch".into(), shape: vec![e, 3 * pv], ty: lin, data: conv.to_f32() }])?;
        let mean = st.require("z_normalizer.running_mean", &[z])?.to_f32();
        let std = st.require("z_normalizer.running_var", &[z])?.to_f32().iter().map(|v| v.sqrt()).collect();
        // Per-head query/key RMS norm takes the epsilon of the dtype the
        // checkpoint runs in.
        let rms_eps = match st.get(&format!("{}.attn.qkv.weight", block_prefix(&cfg))).map(|v| v.dtype) {
            Some(Dtype::Bf16) => 0.007_812_5,
            Some(Dtype::F16) => 0.000_976_562_5,
            _ => f32::EPSILON,
        };
        Ok(Self { backend, cfg, w, host, mean, std, rms_eps, chunk_bytes: ATTENTION_CHUNK_BYTES })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// Cap the gathered-key bytes of one attention chunk (smaller trades
    /// speed for memory).
    pub fn set_attention_chunk_bytes(&mut self, bytes: usize) {
        self.chunk_bytes = bytes.max(1);
    }

    /// Encode one clip `[3][frames][h][w]` in `[-1, 1]` to normalised latents
    /// `[z][t][h/32][w/32]`, in one pass.
    ///
    /// # Errors
    /// On a size mismatch or a backend failure.
    pub fn encode(&self, video: &[f32], frames: usize, h: usize, w: usize) -> Result<(Vec<f32>, [usize; 3])> {
        if video.len() != 3 * frames * h * w || frames == 0 {
            return Err(Error::Config(format!("video of {} values is not 3x{frames}x{h}x{w}", video.len())));
        }
        let mut g = self.patch_embed(video, frames, h, w)?;
        for l in encoder_layers(&self.cfg) {
            g = self.layer(&l, g)?;
        }
        let z = self.cfg.z_dim;
        let y = self.affine(&g, "encoder.proj", None, None)?;
        let n = g.n();
        let mut out = vec![0f32; z * n];
        for i in 0..n {
            for c in 0..z {
                out[c * n + i] = (y[i * 2 * z + c] - self.mean[c]) / self.std[c];
            }
        }
        Ok((out, [g.t, g.h, g.w]))
    }

    /// Encode a clip of `4k + 1` frames the way the policies were trained:
    /// chunks of [`CHUNK_FRAMES`] overlapping by one frame, the clip first
    /// padded by repeating its last frame, the first latent of every later
    /// chunk dropped, and the result trimmed to `k + 1` latent frames.
    ///
    /// # Errors
    /// On a frame count off the `4k + 1` grid, a size mismatch or a backend
    /// failure.
    pub fn encode_chunked(&self, video: &[f32], frames: usize, h: usize, w: usize) -> Result<(Vec<f32>, [usize; 3])> {
        if frames == 0 || (frames - 1) % TEMPORAL_DOWNSAMPLE != 0 {
            return Err(Error::Config(format!("the autoencoder takes 4k + 1 frames, got {frames}")));
        }
        if video.len() != 3 * frames * h * w {
            return Err(Error::Config(format!("video of {} values is not 3x{frames}x{h}x{w}", video.len())));
        }
        let stride = CHUNK_FRAMES - CHUNK_OVERLAP;
        let total = CHUNK_FRAMES + frames.saturating_sub(CHUNK_FRAMES).div_ceil(stride) * stride;
        let plane = h * w;
        let frame = |c: usize, f: usize| &video[(c * frames + f.min(frames - 1)) * plane..][..plane];
        let skip = 1 + (CHUNK_OVERLAP - 1) / TEMPORAL_DOWNSAMPLE;
        let target = 1 + (frames - 1) / TEMPORAL_DOWNSAMPLE;
        let mut pieces: Vec<(Vec<f32>, [usize; 3], usize)> = Vec::new();
        let mut start = 0;
        while start + CHUNK_FRAMES <= total {
            let mut clip = Vec::with_capacity(3 * CHUNK_FRAMES * plane);
            for c in 0..3 {
                for f in start..start + CHUNK_FRAMES {
                    clip.extend_from_slice(frame(c, f));
                }
            }
            let (z, dims) = self.encode(&clip, CHUNK_FRAMES, h, w)?;
            pieces.push((z, dims, if start == 0 { 0 } else { skip }));
            start += stride;
        }
        let [_, lh, lw] = pieces[0].1;
        let zc = self.cfg.z_dim;
        let lp = lh * lw;
        let mut out = vec![0f32; zc * target * lp];
        let mut t_out = 0;
        for (z, dims, skip) in &pieces {
            for t in *skip..dims[0] {
                if t_out == target {
                    break;
                }
                for c in 0..zc {
                    out[(c * target + t_out) * lp..][..lp].copy_from_slice(&z[(c * dims[0] + t) * lp..][..lp]);
                }
                t_out += 1;
            }
        }
        if t_out < target {
            return Err(Error::Config(format!("chunked encode produced {t_out} latent frames, need {target}")));
        }
        Ok((out, [target, lh, lw]))
    }

    /// Decode normalised latents `[z][t][h][w]` to a clip `[3][frames][H][W]`.
    ///
    /// # Errors
    /// On a size mismatch or a backend failure.
    pub fn decode(&self, latents: &[f32], dims: [usize; 3]) -> Result<(Vec<f32>, [usize; 3])> {
        let z = self.cfg.z_dim;
        let n = dims.iter().product::<usize>();
        if latents.len() != z * n {
            return Err(Error::Config(format!("latents of {} values are not {z}x{dims:?}", latents.len())));
        }
        let mut x = vec![0f32; n * z];
        for c in 0..z {
            for i in 0..n {
                x[i * z + c] = latents[c * n + i] * self.std[c] + self.mean[c];
            }
        }
        let g = Grid { t: dims[0], h: dims[1], w: dims[2], c: z, x };
        let top = self.cfg.embed_dim << (self.cfg.dec_depths.len() - 1);
        let y = self.affine(&g, "decoder.proj_in", None, None)?;
        let mut g = Grid { c: top, x: y, ..g };
        for l in decoder_layers(&self.cfg) {
            g = self.layer(&l, g)?;
        }
        let y = self.affine(&g, "decoder.proj_out", None, None)?;
        let [pt, ph, pw] = self.cfg.patch;
        let (ft, fh, fw) = (g.t * pt, g.h * ph, g.w * pw);
        let mut out = vec![0f32; 3 * ft * fh * fw];
        let pc = pt * ph * pw * 3;
        for t in 0..g.t {
            for yy in 0..g.h {
                for xx in 0..g.w {
                    let v = &y[((t * g.h + yy) * g.w + xx) * pc..][..pc];
                    for kt in 0..pt {
                        for kh in 0..ph {
                            for kw in 0..pw {
                                for c in 0..3 {
                                    let o = ((c * ft + t * pt + kt) * fh + yy * ph + kh) * fw + xx * pw + kw;
                                    out[o] = v[((kt * ph + kh) * pw + kw) * 3 + c];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((out, [ft, fh, fw]))
    }

    fn patch_embed(&self, video: &[f32], frames: usize, h: usize, w: usize) -> Result<Grid> {
        let [pt, ph, pw] = self.cfg.patch;
        let (t, gh, gw) = (frames.div_ceil(pt), h.div_ceil(ph), w.div_ceil(pw));
        let pv = 3 * pt * ph * pw;
        let mut x = vec![0f32; t * gh * gw * pv];
        for i in 0..t {
            for y in 0..gh {
                for xx in 0..gw {
                    let v = &mut x[((i * gh + y) * gw + xx) * pv..][..pv];
                    for c in 0..3 {
                        for kt in 0..pt {
                            for kh in 0..ph {
                                for kw in 0..pw {
                                    let (f, r, col) = (i * pt + kt, y * ph + kh, xx * pw + kw);
                                    if f < frames && r < h && col < w {
                                        v[((c * pt + kt) * ph + kh) * pw + kw] = video[((c * frames + f) * h + r) * w + col];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let g = Grid { t, h: gh, w: gw, c: pv, x };
        let norm = self.cfg.patch_norm.then_some("encoder.patch_embed.norm");
        let y = self.run_linear(&g, None, (self.host.get("patch"), Some(self.w.get("encoder.patch_embed.proj.bias"))), norm, None)?;
        Ok(Grid { c: self.cfg.embed_dim, x: y, ..g })
    }

    /// `linear(norm?(x)) + skip?` with `norm_pre` before and `norm_post`
    /// after the linear.
    fn run_linear(&self, g: &Grid, norm_pre: Option<&str>, lin: (Tn, Option<Tn>), norm_post: Option<&str>, skip: Option<&[f32]>) -> Result<Vec<f32>> {
        let mut gr = Graph::new(&self.backend)?;
        let n = g.n() as i64;
        let x = gr.input(sys::GGML_TYPE_F32, &[g.c as i64, n]);
        let mut h = x;
        if let Some(p) = norm_pre {
            h = self.layer_norm(&mut gr, h, p);
        }
        h = match lin.1 {
            Some(b) => gr.linear_b(lin.0, b, h),
            None => gr.linear(lin.0, h),
        };
        if let Some(p) = norm_post {
            h = self.layer_norm(&mut gr, h, p);
        }
        let s = skip.map(|_| gr.input(sys::GGML_TYPE_F32, &[h.ne(0), n]));
        if let Some(s) = s {
            h = gr.add(h, s);
        }
        gr.finish(&[h])?;
        gr.set_f32(x, &g.x);
        if let (Some(s), Some(d)) = (s, skip) {
            gr.set_f32(s, d);
        }
        gr.compute()?;
        Ok(gr.read_f32(h))
    }

    fn affine(&self, g: &Grid, p: &str, norm: Option<&str>, skip: Option<&[f32]>) -> Result<Vec<f32>> {
        let b = self.w.get(&format!("{p}.bias"));
        self.run_linear(g, norm, (self.w.get(&format!("{p}.weight")), Some(b)), None, skip)
    }

    fn layer_norm(&self, gr: &mut Graph, x: Tn, p: &str) -> Tn {
        let h = gr.norm(x, LN_EPS);
        let h = gr.mul(h, self.w.get(&format!("{p}.weight")));
        gr.add(h, self.w.get(&format!("{p}.bias")))
    }

    fn layer(&self, l: &Layer, g: Grid) -> Result<Grid> {
        match l {
            Layer::Block { p, dim, heads, window, causal } => {
                debug_assert_eq!(g.c, *dim);
                self.block(p, g, *heads, *window, *causal)
            }
            Layer::PatchMerge { p, dim, out } => {
                let (h2, w2) = (g.h.div_ceil(2), g.w.div_ceil(2));
                let c = *dim;
                let mut x = vec![0f32; g.t * h2 * w2 * 4 * c];
                for t in 0..g.t {
                    for y in 0..h2 {
                        for xx in 0..w2 {
                            let o = &mut x[((t * h2 + y) * w2 + xx) * 4 * c..][..4 * c];
                            for hs in 0..2 {
                                for ws in 0..2 {
                                    let (r, col) = (2 * y + hs, 2 * xx + ws);
                                    if r < g.h && col < g.w {
                                        o[(hs * 2 + ws) * c..][..c].copy_from_slice(&g.x[((t * g.h + r) * g.w + col) * c..][..c]);
                                    }
                                }
                            }
                        }
                    }
                }
                let m = Grid { t: g.t, h: h2, w: w2, c: 4 * c, x };
                let y = self.run_linear(&m, Some(&format!("{p}.norm")), (self.w.get(&format!("{p}.reduction.weight")), None), None, None)?;
                Ok(Grid { c: *out, x: y, ..m })
            }
            Layer::TemporalMerge { p, dim } => {
                let c = *dim;
                let fp = g.h * g.w * c;
                let frame = |i: usize| &g.x[i * fp..][..fp];
                // An odd frame count repeats the first frame in front.
                let order: Vec<usize> = if g.t % 2 == 1 { std::iter::once(0).chain(0..g.t).collect() } else { (0..g.t).collect() };
                let t2 = order.len() / 2;
                let mut x = vec![0f32; t2 * fp * 2];
                let mut skip = vec![0f32; t2 * fp];
                for d in 0..t2 {
                    let (a, b) = (frame(order[2 * d]), frame(order[2 * d + 1]));
                    for s in 0..g.h * g.w {
                        let o = &mut x[(d * g.h * g.w + s) * 2 * c..][..2 * c];
                        o[..c].copy_from_slice(&a[s * c..][..c]);
                        o[c..].copy_from_slice(&b[s * c..][..c]);
                        for k in 0..c {
                            skip[(d * g.h * g.w + s) * c + k] = (a[s * c + k] + b[s * c + k]) * 0.5;
                        }
                    }
                }
                let m = Grid { t: t2, h: g.h, w: g.w, c: 2 * c, x };
                let y = self.run_linear(&m, Some(&format!("{p}.norm")), (self.w.get(&format!("{p}.reduction.weight")), None), None, Some(&skip))?;
                Ok(Grid { c, x: y, ..m })
            }
            Layer::PatchExpand { p, dim, out } => {
                let y = self.run_linear(&g, Some(&format!("{p}.norm")), (self.w.get(&format!("{p}.expansion.weight")), None), None, None)?;
                let (oc, h2, w2) = (*out, 2 * g.h, 2 * g.w);
                debug_assert_eq!(g.c, *dim);
                let mut x = vec![0f32; g.t * h2 * w2 * oc];
                for t in 0..g.t {
                    for r in 0..g.h {
                        for col in 0..g.w {
                            let v = &y[((t * g.h + r) * g.w + col) * 4 * oc..][..4 * oc];
                            for hs in 0..2 {
                                for ws in 0..2 {
                                    x[((t * h2 + 2 * r + hs) * w2 + 2 * col + ws) * oc..][..oc].copy_from_slice(&v[(hs * 2 + ws) * oc..][..oc]);
                                }
                            }
                        }
                    }
                }
                Ok(Grid { t: g.t, h: h2, w: w2, c: oc, x })
            }
            Layer::TemporalExpand { p, dim } => {
                let c = *dim;
                let skip: Vec<f32> = g.x.chunks(c).flat_map(|v| v.iter().chain(v.iter()).copied()).collect();
                let y = self.run_linear(&g, Some(&format!("{p}.norm")), (self.w.get(&format!("{p}.expansion.weight")), None), None, Some(&skip))?;
                let s = g.h * g.w;
                let t2 = 2 * g.t - 1;
                let mut x = vec![0f32; t2 * s * c];
                for d in 0..g.t {
                    for i in 0..s {
                        let v = &y[(d * s + i) * 2 * c..][..2 * c];
                        for half in 0..2 {
                            // The first expanded frame is dropped.
                            if let Some(f) = (2 * d + half).checked_sub(1) {
                                x[(f * s + i) * c..][..c].copy_from_slice(&v[half * c..][..c]);
                            }
                        }
                    }
                }
                Ok(Grid { t: t2, h: g.h, w: g.w, c, x })
            }
        }
    }

    /// One transformer block: neighborhood attention then a gated MLP, both
    /// residual. Queries run in chunks so the gathered keys stay within the
    /// byte budget; every chunk carries its tokens through the whole block.
    fn block(&self, p: &str, g: Grid, heads: usize, window: [usize; 3], causal: bool) -> Result<Grid> {
        let dims = [g.t, g.h, g.w];
        let (c, n) = (g.c, g.n());
        let hd = c / heads;
        let (idx, mask, k) = neighborhood(dims, window, causal)?;
        let (cos, sin) = rope_tables(dims, hd);
        // Each chunk adds a fixed set of graph nodes; cap the chunk count so a
        // small byte budget on a large grid still fits one graph.
        let per = (self.chunk_bytes / (k * c * 4)).clamp(1, n).max(n.div_ceil(MAX_ATTENTION_CHUNKS));
        let chunks: Vec<(usize, usize)> = (0..n).step_by(per).map(|s| (s, per.min(n - s))).collect();

        let mut gr = Graph::new(&self.backend)?;
        let w = |s: &str| self.w.get(&format!("{p}.{s}"));
        let x = gr.input(sys::GGML_TYPE_F32, &[c as i64, n as i64]);
        let tc = gr.input(sys::GGML_TYPE_F32, &[hd as i64, 1, n as i64]);
        let ts = gr.input(sys::GGML_TYPE_F32, &[hd as i64, 1, n as i64]);
        let h = self.layer_norm(&mut gr, x, &format!("{p}.norm1"));
        let qkv = gr.linear_b(w("attn.qkv.weight"), w("attn.qkv.bias"), h);
        let mut parts = [0usize, 1, 2].map(|i| {
            let v = gr.view_heads(qkv, (i * c) as i64, hd as i64, heads as i64);
            gr.cont(v)
        });
        for t in parts.iter_mut().take(2) {
            if self.cfg.qk_norm {
                *t = gr.rms_norm(*t, self.rms_eps);
            }
            *t = gr.rotate_half_rope(*t, tc, ts);
        }
        let [q, kk, v] = parts;
        let kf = gr.reshape(kk, &[c as i64, n as i64]);
        let vf = gr.reshape(v, &[c as i64, n as i64]);
        let scale = 1.0 / (hd as f32).sqrt();
        let (hd_i, heads_i, k_i, c_i) = (hd as i64, heads as i64, k as i64, c as i64);
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        for &(s, m) in &chunks {
            let mi = m as i64;
            let ids = gr.input(sys::GGML_TYPE_I32, &[(m * k) as i64]);
            let qc = gr.view_4d(q, [hd_i, 1, heads_i, mi], q.nb(1), q.nb(1), q.nb(2), s * q.nb(2));
            let gather = |gr: &mut Graph, t: Tn| {
                let r = gr.get_rows(t, ids);
                gr.reshape(r, &[hd_i, heads_i, k_i, mi])
            };
            let kn = gather(&mut gr, kf);
            let kn = gr.permute(kn, [0, 2, 1, 3]);
            let kn = gr.cont(kn);
            let mut sc = gr.linear(kn, qc);
            let mk = mask.as_ref().map(|_| gr.input(sys::GGML_TYPE_F32, &[k_i, 1, 1, mi]));
            if let Some(mk) = mk {
                sc = gr.add(sc, mk);
            }
            let pr = gr.soft_max(sc, scale);
            let vn = gather(&mut gr, vf);
            let vn = gr.permute(vn, [1, 2, 0, 3]);
            let vn = gr.cont(vn);
            let o = gr.linear(vn, pr);
            let o = gr.reshape(o, &[c_i, mi]);
            let o = gr.linear_b(w("attn.proj.weight"), w("attn.proj.bias"), o);
            let xc = gr.view_cols(x, s as i64, mi);
            let y = gr.add(xc, o);
            let hm = self.layer_norm(&mut gr, y, &format!("{p}.norm2"));
            let gu = gr.linear(w("mlp.gate_up_proj.weight"), hm);
            let a = gr.swiglu(gu);
            let a = gr.linear(w("mlp.down_proj.weight"), a);
            let y = gr.add(y, a);
            ins.push((ids, mk, s, m));
            outs.push(y);
        }
        gr.finish(&outs)?;
        gr.set_f32(x, &g.x);
        gr.set_f32(tc, &cos);
        gr.set_f32(ts, &sin);
        for &(ids, mk, s, m) in &ins {
            gr.set_i32(ids, &idx[s * k..(s + m) * k]);
            if let (Some(mk), Some(mask)) = (mk, mask.as_ref()) {
                gr.set_f32(mk, &mask[s * k..(s + m) * k]);
            }
        }
        gr.compute()?;
        let mut out = Vec::with_capacity(n * c);
        for y in outs {
            out.extend(gr.read_f32(y));
        }
        Ok(Grid { x: out, ..g })
    }
}

fn block_prefix(cfg: &VaeConfig) -> String {
    match encoder_layers(cfg).into_iter().next() {
        Some(Layer::Block { p, .. }) => p,
        _ => "encoder.features.0.0".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_layout_downsamples_4x_in_time_and_32x_in_space() {
        let c = VaeConfig::default();
        assert_eq!(c.latent_grid(45, 256, 512), (12, 8, 16));
        assert_eq!(c.latent_grid(1, 544, 736), (1, 17, 23));
        let enc = encoder_layers(&c);
        let merges = enc.iter().filter(|l| matches!(l, Layer::TemporalMerge { .. })).count();
        assert_eq!(merges, 2);
        assert_eq!(enc.iter().filter(|l| matches!(l, Layer::Block { .. })).count(), 21 + 2);
        assert_eq!(glu_hidden(256), 704);
    }

    #[test]
    fn border_windows_shift_inward_and_causal_windows_shrink() {
        let (idx, mask, k) = neighborhood([3, 1, 4], [3, 1, 3], false).unwrap();
        assert_eq!(k, 9);
        // Token (t=0, w=0): window starts at the origin.
        assert_eq!(&idx[..9], &[0, 1, 2, 4, 5, 6, 8, 9, 10]);
        assert!(mask.is_none());
        let (idx, mask, _) = neighborhood([3, 1, 1], [3, 1, 1], true).unwrap();
        let mask = mask.unwrap();
        assert_eq!(&idx[..3], &[0, 0, 0]);
        assert_eq!(&mask[..3], &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]);
        assert_eq!(&idx[6..9], &[0, 1, 2]);
        assert!(neighborhood([2, 4, 4], [3, 3, 3], false).is_err());
    }
}

#[cfg(test)]
mod parity {
    //! Against fixtures from `tests/parity/make_flux3_vae_fixtures.py`.
    use super::*;
    use std::path::PathBuf;

    fn f32s(p: &std::path::Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn agreement(ours: &[f32], reference: &[f32]) -> (f64, f64) {
        assert_eq!(ours.len(), reference.len());
        let (mut dot, mut a2, mut b2, mut d2) = (0f64, 0f64, 0f64, 0f64);
        for (&a, &b) in ours.iter().zip(reference) {
            let (a, b) = (f64::from(a), f64::from(b));
            dot += a * b;
            a2 += a * a;
            b2 += b * b;
            d2 += (a - b) * (a - b);
        }
        (dot / (a2.sqrt() * b2.sqrt()), (d2 / b2).sqrt())
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let root = PathBuf::from(std::env::var("PRAECISE_FLUX3_VAE_PARITY").expect("PRAECISE_FLUX3_VAE_PARITY names the fixture dir"));
        for name in ["a", "b"] {
            let d = root.join(name);
            let j: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("config.json")).unwrap()).unwrap();
            let arr3 = |k: &str| -> [usize; 3] { let v: Vec<usize> = serde_json::from_value(j[k].clone()).unwrap(); [v[0], v[1], v[2]] };
            let list = |k: &str| -> Vec<usize> { serde_json::from_value(j[k].clone()).unwrap() };
            let cfg = VaeConfig {
                z_dim: j["z_dim"].as_u64().unwrap() as usize,
                embed_dim: j["embed_dim"].as_u64().unwrap() as usize,
                patch: arr3("patch_size"),
                window: arr3("window_size"),
                alternate_window: (!j["alternate_window_size"].is_null()).then(|| arr3("alternate_window_size")),
                enc_depths: list("enc_depths"),
                dec_depths: list("dec_depths"),
                num_heads: list("num_heads"),
                temporal: serde_json::from_value(j["temporal"].clone()).unwrap(),
                enc_causal: j["enc_causal"].as_bool().unwrap(),
                dec_causal: j["dec_causal"].as_bool().unwrap(),
                qk_norm: j["qk_norm"].as_bool().unwrap(),
                patch_norm: j["patch_norm"].as_bool().unwrap(),
            };
            let mut vae = VideoVae::load(&[d.join("vae.safetensors")], "model.", cfg, LoadOptions { precision, cpu_threads: 8, device: None }).unwrap();
            for case in 0..2 {
                // The second case also runs attention over many small chunks.
                vae.set_attention_chunk_bytes(if case == 1 { 4096 } else { ATTENTION_CHUNK_BYTES });
                let s: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join(format!("shape{case}.json"))).unwrap()).unwrap();
                let v: Vec<usize> = serde_json::from_value(s["video"].clone()).unwrap();
                let lat: Vec<usize> = serde_json::from_value(s["latent"].clone()).unwrap();
                let video = f32s(&d.join(format!("video{case}.f32")));
                let (z, dims) = vae.encode(&video, v[2], v[3], v[4]).unwrap();
                assert_eq!(dims.to_vec(), lat[2..].to_vec());
                let (cz, rz) = agreement(&z, &f32s(&d.join(format!("latent{case}.f32"))));
                // Decode the reference latent so the decoder is measured alone.
                let (y, _) = vae.decode(&f32s(&d.join(format!("latent{case}.f32"))), dims).unwrap();
                let (cy, ry) = agreement(&y, &f32s(&d.join(format!("decoded{case}.f32"))));
                println!("{name}/{case} {precision:?}: encode cosine {cz:.6} rel {rz:.2e}; decode cosine {cy:.6} rel {ry:.2e}");
                assert!(cz >= min_cos && rz <= max_rel && cy >= min_cos && ry <= max_rel);
            }
        }
    }

    #[test]
    #[ignore = "needs PRAECISE_FLUX3_VAE_PARITY fixtures"]
    fn flux3_vae_parity_f32() {
        run(Precision::F32, 0.999_99, 1e-3);
    }

    #[test]
    #[ignore = "needs PRAECISE_FLUX3_VAE_PARITY fixtures"]
    fn flux3_vae_parity_bf16() {
        run(Precision::Bf16, 0.999, 5e-2);
    }
}
