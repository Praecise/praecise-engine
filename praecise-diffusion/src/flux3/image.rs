//! FLUX 3 image generation and editing.
//!
//! The image models of the FLUX 3 family run the multi-stream transformer
//! ([`Flux3Transformer`]) on two content streams beside a Qwen3-VL text
//! context ([`TextEncoder`]): `image`, the latent being generated, and
//! `image_cond`, the clean latents of up to [`MAX_REFERENCES`] reference
//! images (single-image editing and multi-reference composition). Latents
//! come from the FLUX.2 autoencoder: 32 channels at an eighth of the
//! resolution, patched 2x2 into 128-channel tokens and normalised by
//! per-channel statistics.
//!
//! Token positions are `(t, h, w, l)`: generated tokens `(0, row, col, 0)`,
//! the `k`-th reference (0-based) `(0, row, col, -(k + 1))`, text tokens
//! `(0, 0, 0, i)`; the row and column axes are scaled by one half
//! ([`POSITION_SCALE`]). Reference and text tokens sit at time 0.
//!
//! Two kinds of checkpoint load. A flow map (step-distilled) takes, at every
//! step, the time the step jumps to as well as the current one, and samples
//! without guidance. An undistilled model samples with classifier-free
//! guidance against the empty prompt. Both step with Euler over the same
//! rationally shifted time grid, and both see their times as bfloat16, the
//! precision they were trained with.
//!
//! Weights load from either layout the checkpoints ship in:
//!
//! - the package layout: `config.json` and `model.safetensors` at the root
//!   or under `variants/<name>/` (bfloat16, or a float8 or NVFP4 transformer),
//!   `ae.safetensors` and `text_encoder/`;
//! - the diffusers layout: `transformer/`, `vae/` and `text_encoder/`.
//!
//! The two can be mixed (a quantised package transformer beside a diffusers
//! `vae/`), and every transformer format runs on the backend the load
//! options select; a host with GPU hardware this build cannot drive refuses
//! to load rather than run on the CPU.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;

use super::text_encoder::{CONTEXT_LAYERS, TextEncoder};
use super::{Flux3Transformer, StreamInput, TextInput};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, WType, Weights};
use crate::pipeline::{Image, LoadOptions, Precision, Request, RgbImage, Timings};
use crate::safetensors::SafeTensors;
use crate::schedule;
use crate::vae::{self, VaeConfig};

/// Most reference images one request may carry.
pub const MAX_REFERENCES: usize = 10;
/// Scale of each position axis `(t, h, w, l)` before the rotary embedding.
pub const POSITION_SCALE: [f32; 4] = [1.0, 0.5, 0.5, 1.0];
/// Time shift of the sampling grid when the checkpoint names none.
pub const DEFAULT_SHIFT: f32 = 6.93;
/// Rotary position of every padding token of the prompt.
pub const PAD_POSITION: i32 = 1;
/// Largest single reference, in pixels (its 16-pixel canvas included).
pub const SINGLE_REFERENCE_MAX_PIXELS: u32 = 4 * 1024 * 1024;
/// Largest reference of a multi-reference request, in pixels.
pub const MULTI_REFERENCE_MAX_PIXELS: u32 = 1024 * 1024;
/// Shortest and longest side of a single reference's canvas.
pub const REFERENCE_SIDES: (u32, u32) = (64, 4096);
/// Pixels per latent token along each side.
pub const CELL: u32 = 16;

const IMAGE: &str = "image";
const COND: &str = "image_cond";

/// Where the transformer weights are read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformerSource {
    /// A `transformer/` directory in the diffusers layout.
    Diffusers(PathBuf),
    /// One package file (`model.safetensors`), named as the released
    /// packages name their tensors.
    Package(PathBuf),
}

impl TransformerSource {
    /// The transformer under `root`: `variants/<variant>/model.safetensors`
    /// when a variant is named; otherwise `transformer/`, then a root
    /// `model.safetensors`, then the only variant present.
    ///
    /// # Errors
    /// When none is present, or several variants and none is named.
    pub fn find(root: &Path, variant: Option<&str>) -> Result<Self> {
        if let Some(v) = variant {
            let file = root.join("variants").join(v).join("model.safetensors");
            return if file.is_file() { Ok(Self::Package(file)) } else { Err(Error::Config(format!("no transformer variant {}", file.display()))) };
        }
        if root.join("transformer").is_dir() {
            return Ok(Self::Diffusers(root.join("transformer")));
        }
        if root.join("model.safetensors").is_file() {
            return Ok(Self::Package(root.join("model.safetensors")));
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(root.join("variants"))
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path().join("model.safetensors"))).filter(|p| p.is_file()).collect())
            .unwrap_or_default();
        found.sort();
        match found.len() {
            1 => Ok(Self::Package(found.remove(0))),
            0 => Err(Error::Config(format!("{} holds no transformer (transformer/, model.safetensors or variants/<name>/)", root.display()))),
            _ => Err(Error::Config(format!("{} holds several transformer variants; name one", root.display()))),
        }
    }

    /// The package `config.json` beside the transformer, when there is one.
    fn package_config(&self, root: &Path) -> Option<Value> {
        let near = match self {
            Self::Package(f) => f.parent().map(|p| p.join("config.json")),
            Self::Diffusers(_) => None,
        };
        [near, Some(root.join("config.json"))].into_iter().flatten().find_map(|p| {
            let v: Value = serde_json::from_slice(&std::fs::read(p).ok()?).ok()?;
            v.get("dit").is_some().then_some(v)
        })
    }

    fn open(&self) -> Result<SafeTensors> {
        match self {
            Self::Package(f) => SafeTensors::open(std::slice::from_ref(f)),
            Self::Diffusers(dir) => {
                let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
                    .map_err(|e| Error::Weights(format!("{}: {e}", dir.display())))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                    .collect();
                shards.sort();
                if shards.is_empty() {
                    return Err(Error::Weights(format!("no safetensors files in {}", dir.display())));
                }
                let st = SafeTensors::open(&shards)?.renamed(package_name)?;
                // The diffusers output norm produces (scale, shift); the
                // package's produces (shift, scale).
                let rows = st.get(NORM_OUT).map(|v| v.shape[0]).ok_or_else(|| Error::MissingTensor(NORM_OUT.into()))?;
                st.regroup_rows(NORM_OUT, &[(FINAL_MODULATION.to_string(), vec![(rows / 2, rows), (0, rows / 2)])])
            }
        }
    }
}

const NORM_OUT: &str = "norm_out.linear.weight";
const FINAL_MODULATION: &str = "final_layer.image.adaLN_modulation.1.weight";

/// The package name of a diffusers transformer tensor (the output norm keeps
/// its name until its halves are swapped).
fn package_name(name: &str) -> Option<String> {
    const EXACT: [(&str, &str); 6] = [
        ("timestep_embedder.linear_1.weight", "time_in.in_layer.weight"),
        ("timestep_embedder.linear_2.weight", "time_in.out_layer.weight"),
        ("target_timestep_embedder.linear_1.weight", "ref_time_in.in_layer.weight"),
        ("target_timestep_embedder.linear_2.weight", "ref_time_in.out_layer.weight"),
        ("proj_out.weight", "final_layer.image.linear.weight"),
        (NORM_OUT, NORM_OUT),
    ];
    const PREFIXES: [(&str, &str); 4] =
        [("x_embedders.", "emb_in."), ("context_embedder.", "txt_in."), ("mode_blocks.", "content_mode_blocks."), ("single_transformer_blocks.", "single_blocks.")];
    const SUFFIXES: [(&str, &str); 8] = [
        ("attn.to_q.weight", "q_proj.weight"),
        ("attn.to_k.weight", "k_proj.weight"),
        ("attn.to_v.weight", "v_proj.weight"),
        ("attn.to_out.0.weight", "attn_out.weight"),
        ("attn.norm_q.weight", "norm.query_norm.scale"),
        ("attn.norm_k.weight", "norm.key_norm.scale"),
        ("ff.linear_in.weight", "mlp_in.weight"),
        ("ff.linear_out.weight", "mlp_out.weight"),
    ];
    if let Some((_, p)) = EXACT.iter().find(|(d, _)| *d == name) {
        return Some((*p).to_string());
    }
    let mut n = name.to_string();
    if let Some((d, p)) = PREFIXES.iter().find(|(d, _)| name.starts_with(d)) {
        n = format!("{p}{}", &name[d.len()..]);
    }
    if let Some(stem) = n.strip_suffix(".linear.weight") {
        n = format!("{stem}.lin.weight");
    }
    if let Some((d, p)) = SUFFIXES.iter().find(|(d, _)| n.ends_with(d)) {
        n = format!("{}{p}", &n[..n.len() - d.len()]);
    }
    Some(n)
}

/// The diffusers name of a package autoencoder tensor (`ae.safetensors`),
/// `None` for tensors the pipeline does not use.
fn autoencoder_name(name: &str, levels: usize) -> Option<String> {
    let resnet = |s: &str| s.replace("nin_shortcut", "conv_shortcut");
    let attn = |s: &str| match s {
        "norm" => "group_norm".to_string(),
        "q" | "k" | "v" => format!("to_{s}"),
        "proj_out" => "to_out.0".to_string(),
        _ => s.to_string(),
    };
    let (side, rest) = name.split_once('.')?;
    if !matches!(side, "encoder" | "decoder" | "bn") {
        return None;
    }
    if side == "bn" {
        return matches!(rest, "running_mean" | "running_var").then(|| name.to_string());
    }
    let parts: Vec<&str> = rest.split('.').collect();
    Some(match parts.as_slice() {
        ["quant_conv", p] => format!("quant_conv.{p}"),
        ["post_quant_conv", p] => format!("post_quant_conv.{p}"),
        ["norm_out", p] => format!("{side}.conv_norm_out.{p}"),
        ["conv_in" | "conv_out", _] => name.to_string(),
        ["mid", block @ ("block_1" | "block_2"), tail @ ..] => {
            let r = if *block == "block_1" { 0 } else { 1 };
            format!("{side}.mid_block.resnets.{r}.{}", resnet(&tail.join(".")))
        }
        ["mid", "attn_1", layer, p] => format!("{side}.mid_block.attentions.0.{}.{p}", attn(layer)),
        ["down", i, "block", j, tail @ ..] => format!("{side}.down_blocks.{i}.resnets.{j}.{}", resnet(&tail.join("."))),
        ["down", i, "downsample", "conv", p] => format!("{side}.down_blocks.{i}.downsamplers.0.conv.{p}"),
        ["up", i, "block", j, tail @ ..] => {
            let i: usize = i.parse().ok()?;
            format!("{side}.up_blocks.{}.resnets.{j}.{}", levels - 1 - i, resnet(&tail.join(".")))
        }
        ["up", i, "upsample", "conv", p] => {
            let i: usize = i.parse().ok()?;
            format!("{side}.up_blocks.{}.upsamplers.0.conv.{p}", levels - 1 - i)
        }
        _ => return None,
    })
}

/// The autoencoder: its configuration, the decoder and encoder weights and
/// the latent statistics.
struct Autoencoder {
    cfg: VaeConfig,
    w: Weights,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Autoencoder {
    fn load(root: &Path, backend: &Backend) -> Result<Self> {
        let (cfg, st) = if root.join("vae").is_dir() {
            let dir = root.join("vae");
            let cfg: VaeConfig = crate::pipeline::parse(read_json(&dir.join("config.json"))?, "vae config")?;
            let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
                .map_err(|e| Error::Weights(format!("{}: {e}", dir.display())))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            files.sort();
            (cfg, SafeTensors::open(&files)?)
        } else if root.join("ae.safetensors").is_file() {
            let cfg: VaeConfig = crate::pipeline::parse(
                serde_json::json!({
                    "in_channels": 3, "out_channels": 3, "block_out_channels": [128, 256, 512, 512], "latent_channels": 32,
                    "layers_per_block": 2, "norm_num_groups": 32, "patch_size": [2, 2], "batch_norm_eps": 1e-4,
                    "use_post_quant_conv": true, "mid_block_add_attention": true
                }),
                "package autoencoder",
            )?;
            let levels = cfg.block_out_channels.len();
            let mut st = SafeTensors::open(&[root.join("ae.safetensors")])?.renamed(|n| autoencoder_name(n, levels))?;
            // The middle attention's projections are 1x1 convolutions there.
            for side in ["encoder", "decoder"] {
                for p in ["to_q", "to_k", "to_v", "to_out.0"] {
                    let name = format!("{side}.mid_block.attentions.0.{p}.weight");
                    let c = st.get(&name).map(|v| v.shape[0]).ok_or_else(|| Error::MissingTensor(name.clone()))?;
                    st = st.reshaped(&name, &[c, c])?;
                }
            }
            (cfg, st)
        } else {
            return Err(Error::Config(format!("{} holds no autoencoder (vae/ or ae.safetensors)", root.display())));
        };
        cfg.validate()?;
        if cfg.patch_size != [2, 2] || !cfg.use_post_quant_conv {
            return Err(Error::Config("expected patched latent statistics and a post-quantisation convolution".into()));
        }
        let mut specs = cfg.weight_specs();
        specs.extend(cfg.encoder_weight_specs());
        let w = Weights::load(backend, &st, &specs)?;
        let (mean, std) = vae::latent_stats(&st, &cfg)?;
        Ok(Self { cfg, w, mean, std })
    }

    /// Token width: latent channels times the 2x2 patch.
    fn channels(&self) -> usize {
        self.cfg.latent_channels as usize * 4
    }
}

fn read_json(p: &Path) -> Result<Value> {
    let bytes = std::fs::read(p).map_err(|e| Error::Config(format!("{}: {e}", p.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
}

/// `t` as the model sees it: rounded to bfloat16, scaled by the time factor
/// and rounded again, then divided back.
#[must_use]
pub fn model_time(t: f32) -> f32 {
    let bf = |v: f32| half::bf16::from_f32(v).to_f32();
    bf(bf(t) * super::TIME_FACTOR) / super::TIME_FACTOR
}

/// The `steps + 1` times from 1 to 0: an even grid `s` shifted to
/// `shift * s / (1 + (shift - 1) * s)`.
#[must_use]
pub fn sampling_times(steps: usize, shift: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let s = 1.0 - i as f32 / steps as f32;
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect()
}

/// The part of a padded reference that holds the image, in pixels of the
/// generated image: `(left, top, right, bottom)`, right and bottom exclusive.
pub type Crop = (u32, u32, u32, u32);

/// Prepare reference images as the released models expect them. One
/// reference is resized (Lanczos, aspect kept) to fit [`SINGLE_REFERENCE_MAX_PIXELS`]
/// and [`REFERENCE_SIDES`] once rounded up to [`CELL`], then reflect-padded,
/// centred, to that canvas; the crop that recovers it from an output of the
/// canvas size is returned. Each of several references is resized under
/// [`MULTI_REFERENCE_MAX_PIXELS`] and centre-cropped to the cell grid.
///
/// # Errors
/// More than [`MAX_REFERENCES`] images, or an empty one.
pub fn prepare_references(images: &[RgbImage]) -> Result<(Vec<RgbImage>, Option<Crop>)> {
    if images.len() > MAX_REFERENCES {
        return Err(Error::Request(format!("at most {MAX_REFERENCES} reference images")));
    }
    if images.iter().any(|i| i.width == 0 || i.height == 0 || i.rgb.len() != 3 * i.width as usize * i.height as usize) {
        return Err(Error::Request("a reference image is empty or its pixels disagree with its size".into()));
    }
    if let [one] = images {
        let ((fw, fh), (cw, ch)) = reference_canvas(one.width, one.height);
        let fitted = if (fw, fh) == (one.width, one.height) { one.clone() } else { one.lanczos(fw, fh) };
        if (cw, ch) == (fw, fh) {
            return Ok((vec![fitted], None));
        }
        let (left, top) = ((cw - fw) / 2, (ch - fh) / 2);
        // A reflection needs a side longer than the cell, else the edge repeats.
        let symmetric = fw.min(fh) <= CELL;
        let mut rgb = vec![0u8; 3 * cw as usize * ch as usize];
        for y in 0..ch {
            let sy = reflect(i64::from(y) - i64::from(top), fh as usize, symmetric);
            for x in 0..cw {
                let sx = reflect(i64::from(x) - i64::from(left), fw as usize, symmetric);
                let (d, s) = (3 * (y as usize * cw as usize + x as usize), 3 * (sy * fw as usize + sx));
                rgb[d..d + 3].copy_from_slice(&fitted.rgb[s..s + 3]);
            }
        }
        return Ok((vec![RgbImage { width: cw, height: ch, rgb }], Some((left, top, left + fw, top + fh))));
    }
    let prepared = images
        .iter()
        .map(|image| {
            let (mut w, mut h) = (image.width, image.height);
            let mut out = image.clone();
            if u64::from(w) * u64::from(h) > u64::from(MULTI_REFERENCE_MAX_PIXELS) {
                let s = (f64::from(MULTI_REFERENCE_MAX_PIXELS) / (f64::from(w) * f64::from(h))).sqrt();
                (w, h) = ((f64::from(w) * s) as u32, (f64::from(h) * s) as u32);
                out = image.lanczos(w.max(1), h.max(1));
            }
            let (cw, ch) = (w / CELL * CELL, h / CELL * CELL);
            if cw == 0 || ch == 0 {
                return Err(Error::Request(format!("a {w}x{h} reference is smaller than one {CELL}-pixel cell")));
            }
            let (left, top) = ((w - cw) / 2, (h - ch) / 2);
            let mut rgb = Vec::with_capacity(3 * cw as usize * ch as usize);
            for y in top..top + ch {
                let row = 3 * (y as usize * w as usize + left as usize);
                rgb.extend_from_slice(&out.rgb[row..row + 3 * cw as usize]);
            }
            Ok(RgbImage { width: cw, height: ch, rgb })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((prepared, None))
}

/// The output size a request with these prepared references defaults to:
/// the first reference's own size, else 1024x1024.
#[must_use]
pub fn default_size(references: &[RgbImage]) -> (u32, u32) {
    references.first().map_or((1024, 1024), |r| (r.width, r.height))
}

/// Round half to even, as Python's `round` does.
fn round_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
}

/// `(fitted size, canvas)` of a single reference.
fn reference_canvas(width: u32, height: u32) -> ((u32, u32), (u32, u32)) {
    let (w, h) = (f64::from(width), f64::from(height));
    let (lo, hi) = REFERENCE_SIDES;
    let max_px = f64::from(SINGLE_REFERENCE_MAX_PIXELS);
    let mut scale = 1f64.min((max_px / (w * h)).sqrt()).min(f64::from(hi) / w.max(h));
    loop {
        let fit = ((round_even(w * scale) as u32).max(1), (round_even(h * scale) as u32).max(1));
        let canvas = |v: u32| (v.div_ceil(CELL) * CELL).clamp(lo, hi);
        let c = (canvas(fit.0), canvas(fit.1));
        if u64::from(c.0) * u64::from(c.1) <= u64::from(SINGLE_REFERENCE_MAX_PIXELS) {
            return (fit, c);
        }
        scale *= 0.995;
    }
}

/// Source index of padded index `i` over `n` samples: mirrored about the
/// edge samples (`reflect`), or about the edges themselves (`symmetric`).
fn reflect(i: i64, n: usize, symmetric: bool) -> usize {
    let n = n as i64;
    if n == 1 {
        return 0;
    }
    let (period, i) = if symmetric { (2 * n, i) } else { (2 * n - 2, i) };
    let m = i.rem_euclid(period);
    let idx = if symmetric {
        if m < n { m } else { period - 1 - m }
    } else if m < n {
        m
    } else {
        period - m
    };
    idx as usize
}

/// Tokens `[n][4 * c]` from a `[W, H, C]` latent image (2x2 patches,
/// channel-major within a patch), normalised by the latent statistics.
fn patch(lat: &[f32], gh: usize, gw: usize, mean: &[f32], std: &[f32]) -> Vec<f32> {
    let pc = mean.len();
    let (w, h) = (2 * gw, 2 * gh);
    let mut out = vec![0f32; gh * gw * pc];
    for ty in 0..gh {
        for tx in 0..gw {
            for k in 0..pc {
                let (cl, ph, pw) = (k / 4, (k / 2) % 2, k % 2);
                out[(ty * gw + tx) * pc + k] = (lat[cl * h * w + (2 * ty + ph) * w + 2 * tx + pw] - mean[k]) / std[k];
            }
        }
    }
    out
}

/// The inverse of [`patch`].
fn unpatch(x: &[f32], gh: usize, gw: usize, mean: &[f32], std: &[f32]) -> Vec<f32> {
    let pc = mean.len();
    let (w, h) = (2 * gw, 2 * gh);
    let mut out = vec![0f32; pc / 4 * h * w];
    for ty in 0..gh {
        for tx in 0..gw {
            for k in 0..pc {
                let (cl, ph, pw) = (k / 4, (k / 2) % 2, k % 2);
                out[cl * h * w + (2 * ty + ph) * w + 2 * tx + pw] = x[(ty * gw + tx) * pc + k] * std[k] + mean[k];
            }
        }
    }
    out
}

/// Positions of a `gh x gw` token grid on reference axis `l`.
fn grid_ids(gh: usize, gw: usize, l: i32) -> Vec<[i32; 4]> {
    (0..gh).flat_map(|r| (0..gw).map(move |c| [0, r as i32, c as i32, l])).collect()
}

/// One prompt's context and its positions and times.
struct Context {
    tokens: Vec<f32>,
    ids: Vec<[i32; 4]>,
    times: Vec<f32>,
}

impl Context {
    fn input(&self) -> TextInput<'_> {
        TextInput { tokens: &self.tokens, ids: &self.ids, timesteps: &self.times }
    }
}

/// A loaded FLUX 3 image pipeline.
pub struct Flux3Image {
    backend: Backend,
    dit: Flux3Transformer,
    te: TextEncoder,
    ae: Autoencoder,
    shift: f32,
    source: TransformerSource,
}

impl std::fmt::Debug for Flux3Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flux3Image")
            .field("device", &self.backend.name())
            .field("transformer", &self.source)
            .field("flow_map", &self.is_flow_map())
            .field("resident_bytes", &self.resident_bytes())
            .finish_non_exhaustive()
    }
}

impl Flux3Image {
    /// Load the checkpoint under `root` with the transformer
    /// [`TransformerSource::find`] picks (`variant` names one of
    /// `variants/`). Projection matrices are stored at the load precision,
    /// except that a float8 transformer keeps 8-bit storage (8-bit blocks)
    /// unless the precision is float32, and an NVFP4 one keeps its 4-bit
    /// blocks at every precision.
    ///
    /// # Errors
    /// Missing or malformed files, a transformer that is not a FLUX 3 image
    /// model, or no usable backend.
    pub fn load(root: &Path, variant: Option<&str>, opts: LoadOptions) -> Result<Self> {
        let source = TransformerSource::find(root, variant)?;
        let package = source.package_config(root);
        let shift = package.as_ref().and_then(|c| c.pointer("/sampler/alpha")).and_then(Value::as_f64).map_or(DEFAULT_SHIFT, |a| a as f32);
        let layers: Vec<usize> = package
            .as_ref()
            .and_then(|c| c.pointer("/text_encoder/layers"))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_else(|| CONTEXT_LAYERS.to_vec());

        let st = source.open()?;
        let float8 = st.names().any(|n| n.ends_with(".weight_scale") && !st.names().any(|m| m.ends_with(".weight_scale_2")));
        let linear = match opts.precision {
            Precision::F32 => WType::F32,
            Precision::Bf16 if float8 => WType::Q8_0,
            p => p.wtype(),
        };
        let dit = Flux3Transformer::from_safetensors(st, linear, opts)?.with_position_scale(POSITION_SCALE);
        let cfg = dit.config();
        if cfg.channels(IMAGE).is_none() || cfg.channels(COND).is_none() || cfg.outputs != [IMAGE] || cfg.vec_in_dim.is_some() || cfg.depth_late_blocks > 0 {
            return Err(Error::Config(format!("{source:?} is not a FLUX 3 image transformer (image and image_cond streams, no global vector)")));
        }
        let te = TextEncoder::load(&root.join("text_encoder"), &layers, opts)?.with_pad_position(PAD_POSITION);
        if te.context_width() as u64 != cfg.context_in_dim {
            return Err(Error::Config(format!("the transformer takes {}-wide text, the encoder gives {}", cfg.context_in_dim, te.context_width())));
        }
        let backend = opts.backend()?;
        let ae = Autoencoder::load(root, &backend)?;
        if cfg.channels(IMAGE) != Some(ae.channels() as u64) || cfg.channels(COND) != Some(ae.channels() as u64) {
            return Err(Error::Config("transformer and autoencoder latent widths disagree".into()));
        }
        tracing::info!(device = backend.name(), flow_map = cfg.flow_map, shift, ?source, "FLUX 3 image pipeline loaded");
        Ok(Self { backend, dit, te, ae, shift, source })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.dit.resident_bytes() + self.te.resident_bytes() + self.ae.w.bytes()
    }

    /// Backend device name.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Whether the checkpoint is a flow map (step-distilled: no guidance).
    #[must_use]
    pub fn is_flow_map(&self) -> bool {
        self.dit.config().flow_map
    }

    /// Time shift of the sampling grid.
    #[must_use]
    pub fn shift(&self) -> f32 {
        self.shift
    }

    /// Sample on a grid shifted by `shift` instead of the checkpoint's own.
    ///
    /// # Errors
    /// A shift that is not positive and finite.
    pub fn set_shift(&mut self, shift: f32) -> Result<()> {
        if !(shift.is_finite() && shift > 0.0) {
            return Err(Error::Request(format!("time shift {shift} is not positive")));
        }
        self.shift = shift;
        Ok(())
    }

    /// Where the transformer was read from.
    #[must_use]
    pub fn transformer_source(&self) -> &TransformerSource {
        &self.source
    }

    /// Generate one image. References must already be prepared (see
    /// [`prepare_references`]): sides multiples of [`CELL`], one reference
    /// within [`SINGLE_REFERENCE_MAX_PIXELS`], several each within
    /// [`MULTI_REFERENCE_MAX_PIXELS`]. The guidance scale is ignored by a
    /// flow map; an undistilled model applies it above 1.
    ///
    /// # Errors
    /// [`Error::Request`] for sizes, step counts or references the model
    /// cannot serve; backend failures otherwise.
    pub fn generate(&mut self, req: &Request) -> Result<Image> {
        validate(req)?;
        let n = (req.width / CELL) as usize * (req.height / CELL) as usize * self.ae.channels();
        let noise = schedule::gaussian(req.seed, n).into_iter().map(|v| half::bf16::from_f32(v).to_f32()).collect();
        self.run(req, noise, &mut |_, _| {})
    }
}

/// Sizes, step count and references a request may carry.
fn validate(req: &Request) -> Result<()> {
    if req.width == 0 || req.height == 0 || !req.width.is_multiple_of(CELL) || !req.height.is_multiple_of(CELL) {
        return Err(Error::Request(format!("width and height must be positive multiples of {CELL}")));
    }
    if req.steps == 0 {
        return Err(Error::Request("steps must be at least 1".into()));
    }
    if req.references.len() > MAX_REFERENCES {
        return Err(Error::Request(format!("at most {MAX_REFERENCES} reference images")));
    }
    let limit = if req.references.len() == 1 { SINGLE_REFERENCE_MAX_PIXELS } else { MULTI_REFERENCE_MAX_PIXELS };
    for (i, r) in req.references.iter().enumerate() {
        if r.width == 0 || r.height == 0 || !r.width.is_multiple_of(CELL) || !r.height.is_multiple_of(CELL) {
            return Err(Error::Request(format!("reference {i}: sides must be positive multiples of {CELL}")));
        }
        if u64::from(r.width) * u64::from(r.height) > u64::from(limit) {
            return Err(Error::Request(format!("reference {i}: larger than {limit} pixels")));
        }
        if r.rgb.len() != 3 * r.width as usize * r.height as usize {
            return Err(Error::Request(format!("reference {i}: pixel buffer does not match its size")));
        }
    }
    Ok(())
}

impl Flux3Image {
    fn context(&self, prompt: &str) -> Result<Context> {
        let (tokens, n) = self.te.encode(prompt, None)?;
        Ok(Context { tokens, ids: (0..n as i32).map(|i| [0, 0, 0, i]).collect(), times: vec![0.0; n] })
    }

    /// Reference latents `[tokens][channels]` and their positions, all
    /// references in order.
    fn references(&self, refs: &[RgbImage]) -> Result<(Vec<f32>, Vec<[i32; 4]>)> {
        let (mut tokens, mut ids) = (Vec::new(), Vec::new());
        for (k, r) in refs.iter().enumerate() {
            let (w, h) = (r.width as usize, r.height as usize);
            let mut px = vec![0f32; 3 * w * h];
            for (i, p) in r.rgb.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    px[c * h * w + i] = f32::from(p[c]) / 127.5 - 1.0;
                }
            }
            let mut g = Graph::new(&self.backend)?;
            let io = vae::build_encoder(&mut g, &self.ae.cfg, &self.ae.w, w as i64, h as i64);
            g.finish(&[io.mean])?;
            g.set_f32(io.pixels, &px);
            g.compute()?;
            let mean = g.read_f32(io.mean);
            let (gh, gw) = (h / CELL as usize, w / CELL as usize);
            tokens.extend(patch(&mean, gh, gw, &self.ae.mean, &self.ae.std));
            ids.extend(grid_ids(gh, gw, -(k as i32 + 1)));
        }
        Ok((tokens, ids))
    }

    /// The pipeline from a given starting latent `[tokens][channels]`
    /// (tokens row-major); `on_step` sees each step's index and the latent
    /// after it.
    pub(crate) fn run(&mut self, req: &Request, noise: Vec<f32>, on_step: &mut dyn FnMut(usize, &[f32])) -> Result<Image> {
        let use_cfg = !self.is_flow_map() && req.guidance_scale > 1.0;
        let (gh, gw) = ((req.height / CELL) as usize, (req.width / CELL) as usize);
        let n_img = gh * gw;
        if noise.len() != n_img * self.ae.channels() {
            return Err(Error::Request("starting latent disagrees with the image size".into()));
        }

        let t0 = Instant::now();
        let cond = self.context(&req.prompt)?;
        let uncond = if use_cfg { Some(self.context("")?) } else { None };
        let (ref_tokens, ref_ids) = self.references(&req.references)?;
        let encode_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let times = sampling_times(req.steps as usize, self.shift);
        let refs = Conditioning { tokens: &ref_tokens, ids: &ref_ids };
        let (x, evaluations) = sample(&self.dit, (&cond, uncond.as_ref()), refs, noise, (gh, gw), &times, req.guidance_scale, on_step)?;
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let lat = unpatch(&x, gh, gw, &self.ae.mean, &self.ae.std);
        let mut g = Graph::new(&self.backend)?;
        let io = vae::build_decoder(&mut g, &self.ae.cfg, &self.ae.w, 2 * gw as i64, 2 * gh as i64);
        g.finish(&[io.out])?;
        g.set_f32(io.latents, &lat);
        g.compute()?;
        let px = g.read_f32(io.out);
        drop(g);
        let rgb = to_rgb(&px, req.width as usize, req.height as usize);
        let decode_ms = t2.elapsed().as_millis() as u64;
        Ok(Image { width: req.width, height: req.height, rgb, seed: req.seed, evaluations, timings: Timings { encode_ms, denoise_ms, decode_ms } })
    }
}

/// Clean reference latents `[tokens][channels]` and their positions.
#[derive(Debug, Clone, Copy)]
struct Conditioning<'a> {
    tokens: &'a [f32],
    ids: &'a [[i32; 4]],
}

/// Euler over `times` from the latent `x` on a `gh x gw` grid: one
/// evaluation a step for a flow map (which also takes the next time), two
/// under guidance (an undistilled model given an unconditional context).
/// Returns the final latent and the evaluations run.
#[allow(clippy::too_many_arguments)]
fn sample(
    dit: &Flux3Transformer,
    (cond, uncond): (&Context, Option<&Context>),
    refs: Conditioning<'_>,
    mut x: Vec<f32>,
    (gh, gw): (usize, usize),
    times: &[f32],
    guidance: f32,
    on_step: &mut dyn FnMut(usize, &[f32]),
) -> Result<(Vec<f32>, u32)> {
    let flow_map = dit.config().flow_map;
    let img_ids = grid_ids(gh, gw, 0);
    let ref_times = vec![0f32; refs.ids.len()];
    let mut evaluations = 0u32;
    for i in 0..times.len() - 1 {
        let t = vec![model_time(times[i]); img_ids.len()];
        let target = flow_map.then(|| model_time(times[i + 1]));
        let mut streams = vec![StreamInput { name: IMAGE, tokens: &x, ids: &img_ids, timesteps: &t }];
        if !refs.ids.is_empty() {
            streams.push(StreamInput { name: COND, tokens: refs.tokens, ids: refs.ids, timesteps: &ref_times });
        }
        let mut v = dit.forward_to(cond.input(), None, &streams, target)?.swap_remove(0);
        evaluations += 1;
        if let Some(u) = uncond.filter(|_| !flow_map) {
            let vu = dit.forward_to(u.input(), None, &streams, target)?.swap_remove(0);
            evaluations += 1;
            for (a, b) in v.iter_mut().zip(vu) {
                *a = b + guidance * (*a - b);
            }
        }
        let dt = times[i + 1] - times[i];
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * vi;
        }
        on_step(i, &x);
    }
    Ok((x, evaluations))
}

/// Pixels `[C][H][W]` in `[-1, 1]` to 8-bit RGB, truncating as the reference
/// conversion does.
fn to_rgb(px: &[f32], w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        for c in 0..3 {
            rgb[i * 3 + c] = ((px[c * h * w + i].clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_names_map_to_package_names() {
        let cases = [
            ("x_embedders.image_cond.weight", "emb_in.image_cond.weight"),
            ("context_embedder.weight", "txt_in.weight"),
            ("timestep_embedder.linear_2.weight", "time_in.out_layer.weight"),
            ("target_timestep_embedder.linear_1.weight", "ref_time_in.in_layer.weight"),
            ("early_stream_modulations.txt.linear.weight", "early_stream_modulations.txt.lin.weight"),
            ("single_stream_modulations.image_cond.linear.weight", "single_stream_modulations.image_cond.lin.weight"),
            ("mode_blocks.image.3.attn.to_q.weight", "content_mode_blocks.image.3.q_proj.weight"),
            ("mode_blocks.image_cond.0.attn.norm_k.weight", "content_mode_blocks.image_cond.0.norm.key_norm.scale"),
            ("txt_mode_blocks.4.attn.to_out.0.weight", "txt_mode_blocks.4.attn_out.weight"),
            ("single_transformer_blocks.27.ff.linear_in.weight", "single_blocks.27.mlp_in.weight"),
            ("single_transformer_blocks.0.ff.linear_out.weight", "single_blocks.0.mlp_out.weight"),
            ("proj_out.weight", "final_layer.image.linear.weight"),
            ("norm_out.linear.weight", "norm_out.linear.weight"),
        ];
        for (d, p) in cases {
            assert_eq!(package_name(d).as_deref(), Some(p), "{d}");
        }
    }

    #[test]
    fn package_autoencoder_names_map_to_diffusers_names() {
        let cases = [
            ("encoder.conv_in.weight", Some("encoder.conv_in.weight")),
            ("encoder.quant_conv.bias", Some("quant_conv.bias")),
            ("decoder.post_quant_conv.weight", Some("post_quant_conv.weight")),
            ("encoder.down.1.block.0.nin_shortcut.weight", Some("encoder.down_blocks.1.resnets.0.conv_shortcut.weight")),
            ("encoder.down.2.downsample.conv.bias", Some("encoder.down_blocks.2.downsamplers.0.conv.bias")),
            ("encoder.mid.attn_1.proj_out.weight", Some("encoder.mid_block.attentions.0.to_out.0.weight")),
            ("decoder.mid.block_2.norm1.bias", Some("decoder.mid_block.resnets.1.norm1.bias")),
            ("decoder.up.0.block.2.conv2.weight", Some("decoder.up_blocks.3.resnets.2.conv2.weight")),
            ("decoder.up.3.upsample.conv.weight", Some("decoder.up_blocks.0.upsamplers.0.conv.weight")),
            ("decoder.norm_out.weight", Some("decoder.conv_norm_out.weight")),
            ("bn.running_var", Some("bn.running_var")),
            ("bn.num_batches_tracked", None),
        ];
        for (b, d) in cases {
            assert_eq!(autoencoder_name(b, 4).as_deref(), d, "{b}");
        }
    }

    #[test]
    fn times_shift_the_even_grid_and_round_as_bfloat16() {
        let t = sampling_times(4, 6.93);
        assert_eq!(t.len(), 5);
        assert!((t[0] - 1.0).abs() < 1e-7 && t[4] == 0.0);
        let s = 0.5f32;
        assert!((t[2] - 6.93 * s / (1.0 + 5.93 * s)).abs() < 1e-6);
        assert!(t.windows(2).all(|w| w[1] < w[0]));
        // 0.9 is 0.8984375 in bfloat16; 898.4375 rounds to 900 (spacing 4).
        assert!((model_time(0.9) - 0.9).abs() < 1e-6);
        // 0.9993 rounds to 1 in bfloat16.
        assert_eq!(model_time(0.9993), 1.0);
        assert_eq!(model_time(0.0), 0.0);
        assert_eq!(model_time(1.0), 1.0);
    }

    #[test]
    fn reflection_matches_numpy_padding() {
        // np.pad([0, 1, 2, 3], 3, mode="reflect") -> 3 2 1 | 0 1 2 3 | 2 1 0
        let r: Vec<usize> = (-3..7).map(|i| reflect(i, 4, false)).collect();
        assert_eq!(r, [3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
        // mode="symmetric" -> 2 1 0 | 0 1 2 3 | 3 2 1
        let s: Vec<usize> = (-3..7).map(|i| reflect(i, 4, true)).collect();
        assert_eq!(s, [2, 1, 0, 0, 1, 2, 3, 3, 2, 1]);
    }

    #[test]
    fn references_are_fitted_padded_or_cropped() {
        let img = |w: u32, h: u32| RgbImage { width: w, height: h, rgb: (0..3 * w * h).map(|i| (i % 251) as u8).collect() };
        let (one, crop) = prepare_references(&[img(100, 70)]).unwrap();
        assert_eq!((one[0].width, one[0].height), (112, 80));
        assert_eq!(crop, Some((6, 5, 106, 75)));
        let (one, crop) = prepare_references(&[img(3000, 2000)]).unwrap();
        let (w, h) = (one[0].width, one[0].height);
        assert!(w % CELL == 0 && h % CELL == 0 && w * h <= SINGLE_REFERENCE_MAX_PIXELS);
        assert!(crop.is_none() || crop.unwrap().2 <= w);
        let (many, crop) = prepare_references(&[img(1500, 1000), img(40, 40)]).unwrap();
        assert!(crop.is_none());
        assert!(many[0].width * many[0].height <= MULTI_REFERENCE_MAX_PIXELS && many[0].width % CELL == 0);
        assert_eq!((many[1].width, many[1].height), (32, 32));
        assert!(prepare_references(&vec![img(32, 32); MAX_REFERENCES + 1]).is_err());
    }

    #[test]
    fn patching_round_trips() {
        let (gh, gw, c) = (2, 3, 2);
        let lat: Vec<f32> = (0..c * 4 * gh * gw).map(|i| i as f32 * 0.25 - 3.0).collect();
        let mean: Vec<f32> = (0..4 * c).map(|i| i as f32 * 0.1).collect();
        let std: Vec<f32> = (0..4 * c).map(|i| 1.0 + i as f32 * 0.05).collect();
        let back = unpatch(&patch(&lat, gh, gw, &mean, &std), gh, gw, &mean, &std);
        for (a, b) in lat.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod parity;
