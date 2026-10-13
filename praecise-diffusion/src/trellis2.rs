//! TRELLIS.2 image-to-3D: an image encoder conditions three flow
//! transformers in sequence. The first samples an `8 x 16^3` structure
//! latent that an occupancy decoder turns into the occupied cells of a `32^3`
//! grid; the second samples a 32-channel shape latent on those cells; the
//! third samples a 32-channel material latent conditioned on the shape. A
//! sparse convolutional decoder grows the shape latent into a `512^3`
//! dual-grid surface, a second one into per-voxel base colour, metallic,
//! roughness and alpha on the same voxels, and the surface is simplified,
//! unwrapped, textured from those voxels and written as a binary glTF.
//!
//! Every random draw comes from the request's seed, so a seed names one
//! asset on every device up to floating-point agreement.

pub mod decoder;
pub mod flow;
pub mod image;
pub mod occupancy;
#[cfg(test)]
mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use serde_json::Value;

use crate::dinov3::{Dinov3, Dinov3Config};
use crate::error::{Error, Result};
use crate::ggml::{Backend, WType, Weights};
use crate::glb::{write_glb, TexturedMesh, Texture};
use crate::mesh3d::{self, AttrVolume, GridFrame, Mesh, MeshStats};
use crate::pipeline::{parse, LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use crate::schedule::gaussian;
use crate::sparse3d::Voxels;

use decoder::{DecoderConfig, SparseDecoder};
use flow::{FlowConfig, FlowModel};
use occupancy::{OccupancyConfig, OccupancyDecoder};

pub use image::RgbaImage;

/// Side of the image the encoder sees.
pub const ENCODER_SIDE: usize = 512;

/// Euler sampler settings with classifier-free guidance limited to an
/// interval of flow time.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SamplerParams {
    /// Steps.
    pub steps: usize,
    /// Guidance strength (1 = unguided).
    pub guidance_strength: f32,
    /// Fraction of the guided prediction's spread restored to the
    /// unguided one's.
    #[serde(default)]
    pub guidance_rescale: f32,
    /// Flow times `[lo, hi]` that are guided. Held in double precision as
    /// the reference compares them with its double time grid; a float32
    /// bound such as 0.6 would drop a step that lands exactly on it.
    pub guidance_interval: [f64; 2],
    /// Time warp towards the noisy end.
    pub rescale_t: f64,
}

/// Per-channel latent normalisation.
#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    /// Means.
    pub mean: Vec<f32>,
    /// Deviations.
    pub std: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct SamplerEntry {
    args: SamplerArgs,
    params: SamplerParams,
}

#[derive(Debug, Clone, Deserialize)]
struct SamplerArgs {
    sigma_min: f32,
}

/// One generation request.
#[derive(Debug, Clone)]
pub struct Request3d {
    /// The object on a transparent background.
    pub image: RgbaImage,
    /// Seed for every random draw.
    pub seed: u64,
    /// Face budget of the exported mesh.
    pub max_faces: usize,
    /// Texture atlas side in texels.
    pub texture_size: u32,
    /// Step count override for all three samplers.
    pub steps: Option<usize>,
}

impl Request3d {
    /// Defaults: 200k faces, 2048 texels, the checkpoint's steps.
    #[must_use]
    pub fn new(image: RgbaImage, seed: u64) -> Self {
        Self { image, seed, max_faces: 200_000, texture_size: 2048, steps: None }
    }
}

/// Wall time of each stage, in seconds.
#[derive(Debug, Clone, Default)]
pub struct Timings3d {
    /// Image encoding.
    pub encode: f64,
    /// Structure sampling and occupancy decoding.
    pub structure: f64,
    /// Shape latent sampling.
    pub shape: f64,
    /// Material latent sampling.
    pub material: f64,
    /// Both decoders.
    pub decode: f64,
    /// Surface extraction, simplification, unwrapping, baking, export.
    pub export: f64,
}

/// A generated asset.
#[derive(Debug, Clone)]
pub struct Asset3d {
    /// Binary glTF 2.0.
    pub glb: Vec<u8>,
    /// Occupied cells of the structure grid.
    pub structure_cells: usize,
    /// Voxels of the finest grid.
    pub surface_voxels: usize,
    /// Mesh before simplification.
    pub raw: MeshStats,
    /// Exported mesh.
    pub mesh: MeshStats,
    /// Atlas charts.
    pub charts: usize,
    /// Stage times.
    pub timings: Timings3d,
}

/// Locations of a TRELLIS.2 checkpoint: `root` holds `pipeline.json` and the
/// models it names relative to itself; models named by another repository
/// (`owner/name/path`) and the image encoder (`owner/name`) are looked up
/// under `root` at that path.
#[derive(Debug, Clone)]
pub struct Trellis2Files {
    /// Checkpoint root.
    pub root: PathBuf,
}

impl Trellis2Files {
    fn json(&self, rel: &Path) -> Result<Value> {
        let p = self.root.join(rel);
        let b = std::fs::read(&p).map_err(|e| Error::Config(format!("{}: {e}", p.display())))?;
        serde_json::from_slice(&b).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
    }

    /// `(config, weights)` paths of a model named in `pipeline.json`.
    fn model(&self, name: &str) -> (PathBuf, PathBuf) {
        (PathBuf::from(format!("{name}.json")), self.root.join(format!("{name}.safetensors")))
    }
}

/// Which resolution path to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trellis2Path {
    /// `32^3` structure, `512^3` surface.
    R512,
}

/// A loaded TRELLIS.2 pipeline.
pub struct Trellis2 {
    backend: Backend,
    encoder: Dinov3,
    ss_flow: FlowModel,
    occupancy: OccupancyDecoder,
    shape_flow: FlowModel,
    tex_flow: FlowModel,
    shape_dec: SparseDecoder,
    tex_dec: SparseDecoder,
    ss_sampler: (SamplerParams, f32),
    shape_sampler: (SamplerParams, f32),
    tex_sampler: (SamplerParams, f32),
    shape_norm: Normalization,
    tex_norm: Normalization,
}

impl std::fmt::Debug for Trellis2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trellis2").field("backend", &self.backend).field("resident_bytes", &self.resident_bytes()).finish_non_exhaustive()
    }
}

fn load_flow(backend: &Backend, files: &Trellis2Files, name: &str, linear: WType, exact: bool) -> Result<FlowModel> {
    let (cfg_path, w_path) = files.model(name);
    let j = files.json(&cfg_path)?;
    let class = j.get("name").and_then(Value::as_str).unwrap_or_default();
    if class != "SparseStructureFlowModel" && class != "SLatFlowModel" {
        return Err(Error::Config(format!("{name}: model class {class:?} is not a flow transformer")));
    }
    let cfg: FlowConfig = parse(j["args"].clone(), "flow transformer config")?;
    cfg.validate()?;
    let st = SafeTensors::open(&[w_path])?;
    let w = Weights::load(backend, &st, &cfg.weight_specs(linear))?;
    Ok(FlowModel::new(cfg, w, exact))
}

fn load_decoder(backend: &Backend, files: &Trellis2Files, name: &str, expect: &str) -> Result<SparseDecoder> {
    let (cfg_path, w_path) = files.model(name);
    let j = files.json(&cfg_path)?;
    let class = j.get("name").and_then(Value::as_str).unwrap_or_default();
    if class != expect {
        return Err(Error::Config(format!("{name}: model class {class:?} is not {expect}")));
    }
    let cfg: DecoderConfig = parse(j["args"].clone(), "decoder config")?;
    cfg.validate()?;
    let mut st = SafeTensors::open(&[w_path])?;
    for (n, shape) in cfg.kernel_shapes() {
        st = st.reshaped(&n, &shape)?;
    }
    let w = Weights::load(backend, &st, &cfg.weight_specs())?;
    Ok(SparseDecoder::new(cfg, w))
}

fn sampler(p: &Value, key: &str) -> Result<(SamplerParams, f32)> {
    let e: SamplerEntry = parse(p[key].clone(), key)?;
    let name = p[key]["name"].as_str().unwrap_or_default();
    if name != "FlowEulerGuidanceIntervalSampler" {
        return Err(Error::Config(format!("{key}: sampler {name:?} is not implemented")));
    }
    Ok((e.params, e.args.sigma_min))
}

impl Trellis2 {
    /// Load every model of the 512 path.
    ///
    /// # Errors
    /// Unsupported configuration, missing files or tensors, or a backend
    /// that cannot run here.
    pub fn load(files: &Trellis2Files, opts: LoadOptions, path: Trellis2Path) -> Result<Self> {
        let p = files.json(Path::new("pipeline.json"))?;
        if p.get("name").and_then(Value::as_str) != Some("Trellis2ImageTo3DPipeline") {
            return Err(Error::Config("pipeline.json does not describe Trellis2ImageTo3DPipeline".into()));
        }
        let a = &p["args"];
        let m = &a["models"];
        let name = |k: &str| -> Result<String> {
            m.get(k).and_then(Value::as_str).map(str::to_owned).ok_or_else(|| Error::Config(format!("pipeline.json names no {k}")))
        };
        let Trellis2Path::R512 = path;
        let enc_name = a["image_cond_model"]["args"]["model_name"].as_str().ok_or_else(|| Error::Config("pipeline.json names no image encoder".into()))?;
        if a["image_cond_model"]["name"].as_str() != Some("DinoV3FeatureExtractor") {
            return Err(Error::Config("only the DINOv3 image encoder is implemented".into()));
        }
        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "3D backend selected");
        let linear = opts.precision.wtype();
        let exact = opts.precision == Precision::F32;

        let enc_dir = files.root.join(enc_name);
        let enc_cfg: Dinov3Config = parse(files.json(&enc_dir.join("config.json"))?, "image encoder config")?;
        enc_cfg.validate()?;
        let enc_files: Vec<PathBuf> = std::fs::read_dir(&enc_dir)
            .map_err(|e| Error::Weights(format!("{}: {e}", enc_dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        if enc_files.is_empty() {
            return Err(Error::Weights(format!("no safetensors in {}", enc_dir.display())));
        }
        let mut enc_files = enc_files;
        enc_files.sort();
        let enc_st = SafeTensors::open(&enc_files)?;
        let encoder = Dinov3::new(enc_cfg.clone(), Weights::load(&backend, &enc_st, &enc_cfg.weight_specs(linear))?, exact);

        let ss_flow = load_flow(&backend, files, &name("sparse_structure_flow_model")?, linear, exact)?;
        let shape_flow = load_flow(&backend, files, &name("shape_slat_flow_model_512")?, linear, exact)?;
        let tex_flow = load_flow(&backend, files, &name("tex_slat_flow_model_512")?, linear, exact)?;
        for f in [&ss_flow, &shape_flow, &tex_flow] {
            if f.cfg.cond_channels != enc_cfg.hidden_size {
                return Err(Error::Config("flow condition width differs from the image encoder's".into()));
            }
        }
        let occ_name = name("sparse_structure_decoder")?;
        let (occ_cfg_path, occ_w) = files.model(&occ_name);
        let occ_j = files.json(&occ_cfg_path)?;
        if occ_j.get("name").and_then(Value::as_str) != Some("SparseStructureDecoder") {
            return Err(Error::Config("the structure decoder is not a SparseStructureDecoder".into()));
        }
        let occ_cfg: OccupancyConfig = parse(occ_j["args"].clone(), "structure decoder config")?;
        let occupancy = OccupancyDecoder::load(&backend, occ_cfg, &SafeTensors::open(&[occ_w])?)?;
        let shape_dec = load_decoder(&backend, files, &name("shape_slat_decoder")?, "FlexiDualGridVaeDecoder")?;
        let tex_dec = load_decoder(&backend, files, &name("tex_slat_decoder")?, "SparseUnetVaeDecoder")?;
        if shape_dec.cfg.pred_subdiv == tex_dec.cfg.pred_subdiv || tex_dec.cfg.outputs() != 6 {
            return Err(Error::Config("material decoder must follow the shape decoder's subdivisions and emit 6 channels".into()));
        }
        let shape_norm: Normalization = parse(a["shape_slat_normalization"].clone(), "shape normalisation")?;
        let tex_norm: Normalization = parse(a["tex_slat_normalization"].clone(), "material normalisation")?;
        if shape_norm.mean.len() != shape_flow.cfg.out_channels || tex_norm.mean.len() != tex_flow.cfg.out_channels {
            return Err(Error::Config("latent normalisation widths differ from the flow outputs".into()));
        }
        Ok(Self {
            backend,
            encoder,
            ss_flow,
            occupancy,
            shape_flow,
            tex_flow,
            shape_dec,
            tex_dec,
            ss_sampler: sampler(a, "sparse_structure_sampler")?,
            shape_sampler: sampler(a, "shape_slat_sampler")?,
            tex_sampler: sampler(a, "tex_slat_sampler")?,
            shape_norm,
            tex_norm,
        })
    }

    /// Device bytes of all weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.ss_flow.bytes() + self.shape_flow.bytes() + self.tex_flow.bytes() + self.shape_dec.bytes() + self.tex_dec.bytes()
    }

    /// Image features for the encoder input of a prepared image.
    ///
    /// # Errors
    /// Backend failures.
    pub fn encode(&self, img: &image::RgbaImage) -> Result<Vec<f32>> {
        let rgb = image::crop_to_object(img)?;
        let px = image::encoder_input(&rgb, ENCODER_SIDE);
        self.encoder.encode(&self.backend, &px, ENCODER_SIDE, ENCODER_SIDE)
    }

    /// Generate an asset.
    ///
    /// # Errors
    /// Request validation, an empty structure, or backend failures.
    pub fn generate(&self, req: &Request3d) -> Result<Asset3d> {
        if req.texture_size < 64 || req.texture_size > 8192 || req.max_faces < 1000 {
            return Err(Error::Request("texture side must be 64..=8192 and the face budget at least 1000".into()));
        }
        let mut tm = Timings3d::default();
        let t0 = Instant::now();
        let cond = self.encode(&req.image)?;
        let neg = vec![0f32; cond.len()];
        tm.encode = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let cells = self.structure(&cond, &neg, req.seed, req.steps)?;
        tm.structure = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let shape = self.shape_latent(&cond, &neg, &cells, req.seed, req.steps)?;
        tm.shape = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let tex = self.material_latent(&cond, &neg, &cells, &shape, req.seed, req.steps)?;
        tm.material = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let (surface, materials) = self.decode(&cells, &shape, &tex)?;
        tm.decode = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let (glb, raw, mesh, charts) = export(&surface, &materials, req.max_faces, req.texture_size)?;
        tm.export = t0.elapsed().as_secs_f64();
        tracing::info!(?tm, faces = mesh.faces, "3D asset generated");
        Ok(Asset3d { glb, structure_cells: cells.len(), surface_voxels: surface.voxels.len(), raw, mesh, charts, timings: tm })
    }

    /// Sample the structure latent and decode the occupied cells of the
    /// `32^3` grid.
    ///
    /// # Errors
    /// An empty structure or backend failures.
    pub fn structure(&self, cond: &[f32], neg: &[f32], seed: u64, steps: Option<usize>) -> Result<Voxels> {
        let cfg = &self.ss_flow.cfg;
        let r = cfg.resolution;
        let grid = Voxels::dense(r as i32);
        let n = grid.len();
        let c = cfg.in_channels;
        // Channel-major noise, as a `[C, r, r, r]` volume, then token-major.
        let noise = gaussian(stream(seed, 0), c * n);
        let mut x = vec![0f32; c * n];
        for ch in 0..c {
            for t in 0..n {
                x[t * c + ch] = noise[ch * n + t];
            }
        }
        let (p, sigma) = override_steps(self.ss_sampler, steps);
        let x0 = sample(&p, sigma, true, &mut |x, t, cnd| self.ss_flow.forward(&self.backend, x, &grid.coords, t, cnd), x, cond, neg)?;
        let mut vol = vec![0f32; c * n];
        for ch in 0..c {
            for t in 0..n {
                vol[ch * n + t] = x0[t * c + ch];
            }
        }
        let logits = self.occupancy.decode(&self.backend, &vol, r)?;
        let fine = r * (1 << (self.occupancy.cfg.channels.len() - 1));
        let cells = occupancy::occupied_cells(&logits, fine, fine / 32);
        if cells.is_empty() {
            return Err(Error::Request("the structure stage produced no occupied cells".into()));
        }
        Ok(Voxels::new(cells, 32))
    }

    /// Sample the shape latent on `cells`, denormalised.
    ///
    /// # Errors
    /// Backend failures.
    pub fn shape_latent(&self, cond: &[f32], neg: &[f32], cells: &Voxels, seed: u64, steps: Option<usize>) -> Result<Vec<f32>> {
        let cfg = &self.shape_flow.cfg;
        let x = gaussian(stream(seed, 1), cells.len() * cfg.in_channels);
        let (p, sigma) = override_steps(self.shape_sampler, steps);
        let mut s = sample(&p, sigma, false, &mut |x, t, cnd| self.shape_flow.forward(&self.backend, x, &cells.coords, t, cnd), x, cond, neg)?;
        denormalize(&mut s, &self.shape_norm);
        Ok(s)
    }

    /// Sample the material latent on `cells` given the (denormalised) shape
    /// latent, denormalised.
    ///
    /// # Errors
    /// Backend failures or mismatched widths.
    pub fn material_latent(&self, cond: &[f32], neg: &[f32], cells: &Voxels, shape: &[f32], seed: u64, steps: Option<usize>) -> Result<Vec<f32>> {
        let cfg = &self.tex_flow.cfg;
        let sc = self.shape_norm.mean.len();
        if cfg.in_channels <= sc {
            return Err(Error::Config("material flow input is not wider than the shape latent".into()));
        }
        let nc = cfg.in_channels - sc;
        let n = cells.len();
        let mut norm_shape = shape.to_vec();
        for t in 0..n {
            for c in 0..sc {
                let v = &mut norm_shape[t * sc + c];
                *v = (*v - self.shape_norm.mean[c]) / self.shape_norm.std[c];
            }
        }
        let x = gaussian(stream(seed, 2), n * nc);
        let (p, sigma) = override_steps(self.tex_sampler, steps);
        let mut s = sample(
            &p,
            sigma,
            false,
            &mut |x, t, cnd| {
                let mut inp = vec![0f32; n * cfg.in_channels];
                for tk in 0..n {
                    inp[tk * cfg.in_channels..tk * cfg.in_channels + nc].copy_from_slice(&x[tk * nc..(tk + 1) * nc]);
                    inp[tk * cfg.in_channels + nc..(tk + 1) * cfg.in_channels].copy_from_slice(&norm_shape[tk * sc..(tk + 1) * sc]);
                }
                self.tex_flow.forward(&self.backend, &inp, &cells.coords, t, cnd)
            },
            x,
            cond,
            neg,
        )?;
        denormalize(&mut s, &self.tex_norm);
        Ok(s)
    }

    /// Decode both latents: the surface voxels' raw shape outputs and the
    /// material outputs mapped to `[0, 1]`.
    ///
    /// # Errors
    /// Backend failures or voxel disagreement between the decoders.
    pub fn decode(&self, cells: &Voxels, shape: &[f32], tex: &[f32]) -> Result<(decoder::Decoded, Vec<f32>)> {
        let surface = self.shape_dec.decode(&self.backend, cells, shape, None)?;
        let mats = self.tex_dec.decode(&self.backend, cells, tex, Some(&surface.subdivisions))?;
        if mats.voxels != surface.voxels {
            return Err(Error::Backend("material and shape decoders reached different voxels".into()));
        }
        let m = mats.feats.iter().map(|v| v * 0.5 + 0.5).collect();
        Ok((surface, m))
    }
}

/// Independent noise streams per stage from one request seed.
fn stream(seed: u64, k: u64) -> u64 {
    seed ^ (k.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn override_steps(s: (SamplerParams, f32), steps: Option<usize>) -> (SamplerParams, f32) {
    let (mut p, sigma) = s;
    if let Some(n) = steps {
        p.steps = n.max(1);
    }
    (p, sigma)
}

fn denormalize(x: &mut [f32], n: &Normalization) {
    let c = n.mean.len();
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * n.std[i % c] + n.mean[i % c];
    }
}

/// Spread of a latent: the sample deviation (`unbiased`, as the dense
/// stage's reference computes it) or the population deviation (as the
/// sparse stages' reference does).
fn spread(x: &[f32], unbiased: bool) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let ss = x.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>();
    (ss / if unbiased { n - 1.0 } else { n }).sqrt()
}

/// The time grid: `steps + 1` values from 1 to 0 warped by `rescale_t`.
#[must_use]
pub fn time_grid(steps: usize, rescale_t: f64) -> Vec<f64> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps as f64;
            rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
        })
        .collect()
}

/// Euler flow sampling from `x` with guidance inside the interval and
/// spread-restoring rescale.
///
/// # Errors
/// Model failures.
#[allow(clippy::too_many_arguments)]
pub fn sample(
    p: &SamplerParams,
    sigma_min: f32,
    unbiased: bool,
    model: &mut dyn FnMut(&[f32], f32, &[f32]) -> Result<Vec<f32>>,
    mut x: Vec<f32>,
    cond: &[f32],
    neg: &[f32],
) -> Result<Vec<f32>> {
    let ts = time_grid(p.steps, p.rescale_t);
    let sm = f64::from(sigma_min);
    for w in ts.windows(2) {
        let (t, tp) = (w[0], w[1]);
        let tm = (1000.0 * t) as f32;
        let guided = p.guidance_interval[0] <= t && t <= p.guidance_interval[1];
        let s = if guided { p.guidance_strength } else { 1.0 };
        let pred = if s == 1.0 {
            model(&x, tm, cond)?
        } else if s == 0.0 {
            model(&x, tm, neg)?
        } else {
            let pos = model(&x, tm, cond)?;
            let ng = model(&x, tm, neg)?;
            let mut pred: Vec<f32> = pos.iter().zip(&ng).map(|(a, b)| s * a + (1.0 - s) * b).collect();
            if p.guidance_rescale > 0.0 {
                let k = sm + (1.0 - sm) * t;
                let x0 = |v: &[f32]| -> Vec<f32> { x.iter().zip(v).map(|(xi, vi)| ((1.0 - sm) * f64::from(*xi) - k * f64::from(*vi)) as f32).collect() };
                let x0p = x0(&pos);
                let x0c = x0(&pred);
                let ratio = spread(&x0p, unbiased) / spread(&x0c, unbiased);
                let r = f64::from(p.guidance_rescale);
                pred = x
                    .iter()
                    .zip(&x0c)
                    .map(|(xi, c)| {
                        let x0r = r * f64::from(*c) * ratio + (1.0 - r) * f64::from(*c);
                        (((1.0 - sm) * f64::from(*xi) - x0r) / k) as f32
                    })
                    .collect();
            }
            pred
        };
        let dt = (t - tp) as f32;
        for (xi, v) in x.iter_mut().zip(&pred) {
            *xi -= dt * v;
        }
    }
    Ok(x)
}

/// Surface from the shape decoder's outputs, then simplify, unwrap, bake the
/// materials and write the binary glTF. Returns the file, the raw and
/// exported mesh statistics and the chart count.
///
/// # Errors
/// An empty surface or an export failure.
pub fn export(surface: &decoder::Decoded, materials: &[f32], max_faces: usize, texture_size: u32) -> Result<(Vec<u8>, MeshStats, MeshStats, usize)> {
    let n = surface.voxels.len();
    let f = &surface.feats;
    if f.len() != n * 7 || materials.len() != n * 6 {
        return Err(Error::Request("decoder outputs do not match the surface voxels".into()));
    }
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
    let offsets: Vec<[f32; 3]> = (0..n).map(|i| [2.0 * sig(f[i * 7]) - 0.5, 2.0 * sig(f[i * 7 + 1]) - 0.5, 2.0 * sig(f[i * 7 + 2]) - 0.5]).collect();
    let crossed: Vec<[bool; 3]> = (0..n).map(|i| [f[i * 7 + 3] > 0.0, f[i * 7 + 4] > 0.0, f[i * 7 + 5] > 0.0]).collect();
    let split: Vec<f32> = (0..n).map(|i| softplus(f[i * 7 + 6])).collect();
    let frame = GridFrame { lo: [-0.5; 3], size: 1.0, grid: surface.voxels.resolution };
    let mut mesh = mesh3d::dual_grid_mesh(&surface.voxels.coords, &offsets, &crossed, &split, frame);
    let raw = mesh.stats();
    if mesh.faces.is_empty() {
        return Err(Error::Request("the decoded surface has no faces".into()));
    }
    // The reference's order: repair before closing holes (hole tracing needs
    // consistent winding), simplify to three times the budget, repair and
    // close again, simplify to the budget, repair.
    mesh3d::repair(&mut mesh, 1e-5);
    mesh3d::fill_small_holes(&mut mesh, 3e-2);
    mesh3d::simplify(&mut mesh, max_faces.saturating_mul(3));
    mesh3d::repair(&mut mesh, 1e-5);
    mesh3d::fill_small_holes(&mut mesh, 3e-2);
    mesh3d::simplify(&mut mesh, max_faces);
    mesh3d::repair(&mut mesh, 1e-5);
    let normals = mesh.vertex_normals();
    let u = mesh3d::unwrap(&mesh, 60.0, texture_size, 2);
    let mats: Vec<f32> = materials.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let vol = AttrVolume { coords: &surface.voxels.coords, values: &mats, channels: 6, frame };
    let atlas = mesh3d::bake(&u, &vol, texture_size, 16);
    let px = (texture_size * texture_size) as usize;
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
    let mut base = vec![0u8; px * 4];
    let mut mr = vec![0u8; px * 3];
    for k in 0..px {
        let a = &atlas.values[k * 6..k * 6 + 6];
        base[k * 4] = q(a[0]);
        base[k * 4 + 1] = q(a[1]);
        base[k * 4 + 2] = q(a[2]);
        base[k * 4 + 3] = q(a[5]);
        mr[k * 3 + 1] = q(a[4]);
        mr[k * 3 + 2] = q(a[3]);
    }
    // Z-up generator frame to glTF's Y-up: (x, y, z) -> (x, z, -y).
    let yup = |v: [f32; 3]| [v[0], v[2], -v[1]];
    let out = TexturedMesh {
        positions: u.vertices.iter().map(|&v| yup(v)).collect(),
        normals: u.source.iter().map(|&s| yup(normals[s as usize])).collect(),
        uvs: u.uvs.clone(),
        indices: u.faces.clone(),
        base_color: Some(Texture { width: texture_size, height: texture_size, channels: 4, data: base }),
        metallic_roughness: Some(Texture { width: texture_size, height: texture_size, channels: 3, data: mr }),
        double_sided: true,
    };
    let exported = Mesh { vertices: out.positions.clone(), faces: out.indices.clone() }.stats();
    Ok((write_glb(&out)?, raw, exported, u.charts))
}
