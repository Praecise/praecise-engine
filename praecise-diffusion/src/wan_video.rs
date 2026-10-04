//! Wan2.2 video generation: text (and optionally a first frame) to video.
//!
//! The prompt is whitespace-normalised, tokenized (with the end token) and
//! encoded by the multilingual T5 encoder over its real tokens; the states are
//! zero-padded to [`TEXT_TOKENS`] and cross-attended without a mask, so the
//! padding rows take part through the transformer's text projection. The
//! video is generated in a 48-channel latent at 1/16 of the resolution and
//! 1/4 of the frame rate (plus the first frame), sampled with flow-shifted
//! UniPC under classifier-free guidance, and decoded frame by frame by the
//! causal autoencoder.
//!
//! A conditioning image becomes the first latent frame: it is encoded, its
//! tokens carry timestep 0, and it stays clean through sampling (zero
//! velocity keeps a UniPC sample fixed).

use std::time::Instant;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision, Timings};
use crate::safetensors::SafeTensors;
use crate::schedule;
use crate::umt5::{self, Umt5Config};
use crate::unipc::{flow_sigmas_schedule, UniPc, UniPcConfig};
use crate::video::{to_rgb8, Cosmos3, Video, VideoRequest};
use crate::wan::{self, WanVaeConfig};
use crate::wan_dit::{self, WanDitConfig};

/// Text states the transformer attends to: the prompt's, then zeros.
pub const TEXT_TOKENS: usize = 512;

/// A loaded Wan2.2 single-transformer checkpoint (the 5B text/image-to-video
/// layout).
pub struct Wan22 {
    backend: Backend,
    cfg: WanDitConfig,
    tf: Weights,
    pe: Weights,
    te_cfg: Umt5Config,
    te: Weights,
    vae_cfg: WanVaeConfig,
    vae: Weights,
    sched: UniPcConfig,
    tokenizer: tokenizers::Tokenizer,
    exact: bool,
}

impl std::fmt::Debug for Wan22 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wan22").field("backend", &self.backend.name()).field("layers", &self.cfg.num_layers).finish_non_exhaustive()
    }
}

/// Collapse whitespace runs to one space and trim, as the reference prompt
/// cleaning does for plain text.
#[must_use]
pub fn clean_prompt(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Wan22 {
    /// Load a checkpoint in the diffusers layout. Two-expert checkpoints
    /// (a second transformer switched at a boundary timestep) are refused.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let index = files.json("model_index.json")?;
        let class = index.get("_class_name").and_then(Value::as_str).unwrap_or_default();
        if class != "WanPipeline" {
            return Err(Error::Config(format!("pipeline class {class:?} is not WanPipeline")));
        }
        let two_expert = index.get("transformer_2").and_then(Value::as_array).is_some_and(|a| a.iter().any(|v| !v.is_null()))
            || index.get("boundary_ratio").is_some_and(|v| !v.is_null());
        if two_expert {
            return Err(Error::Config("two-expert checkpoints are not supported".into()));
        }
        if index.get("expand_timesteps").and_then(Value::as_bool) != Some(true) {
            return Err(Error::Config("only per-token timestep checkpoints (expand_timesteps) are supported".into()));
        }
        let cfg: WanDitConfig = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let te_cfg: Umt5Config = parse(files.json("text_encoder/config.json")?, "text encoder config")?;
        te_cfg.validate()?;
        let vae_cfg: WanVaeConfig = parse(files.json("vae/config.json")?, "vae config")?;
        vae_cfg.validate()?;
        if vae_cfg.z_dim != cfg.in_channels || cfg.in_channels != cfg.out_channels || te_cfg.d_model != cfg.text_dim {
            return Err(Error::Config("transformer, text encoder and autoencoder widths disagree".into()));
        }
        let sched: UniPcConfig = parse(files.json("scheduler/scheduler_config.json")?, "scheduler config")?;
        sched.validate()?;
        if !sched.use_flow_sigmas || sched.use_karras_sigmas || sched.use_dynamic_shifting || sched.shift_terminal.is_some() || sched.final_sigmas_type != "zero" {
            return Err(Error::Config("only shifted flow sigmas ending at zero are supported".into()));
        }

        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "video backend selected");
        let exact = opts.precision == Precision::F32;
        let wt = opts.precision.wtype();
        let tf_files = SafeTensors::open(&files.weights("transformer")?)?;
        let tf = Weights::load(&backend, &tf_files, &cfg.weight_specs(wt))?;
        let pe = Weights::from_host(&backend, &cfg.patch_weights(&tf_files)?)?;
        drop(tf_files);
        let te_files = SafeTensors::open(&files.weights("text_encoder")?)?;
        let te = Weights::load(&backend, &te_files, &te_cfg.weight_specs(wt))?;
        drop(te_files);
        let vae_files = SafeTensors::open(&files.weights("vae")?)?;
        let vae = Weights::from_host(&backend, &vae_cfg.host_tensors(&vae_files, exact)?)?;
        drop(vae_files);
        let tok_path = files.root.join("tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        Ok(Self { backend, cfg, tf, pe, te_cfg, te, vae_cfg, vae, sched, tokenizer, exact })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.tf.bytes() + self.pe.bytes() + self.te.bytes() + self.vae.bytes()
    }

    /// The backend device name.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Token ids of a cleaned prompt with the end token, at most
    /// [`TEXT_TOKENS`].
    pub(crate) fn tokens(&self, prompt: &str) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(clean_prompt(prompt), true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        ids.truncate(TEXT_TOKENS);
        Ok(ids)
    }

    /// Text states `[text_dim][TEXT_TOKENS]` (token-major): the prompt's,
    /// then zeros.
    fn context(&self, ids: &[i32]) -> Result<Vec<f32>> {
        let n = ids.len();
        let mut g = Graph::new(&self.backend)?;
        let io = umt5::build(&mut g, &self.te_cfg, &self.te, n as i64);
        g.finish(&[io.out])?;
        g.set_i32(io.ids, ids);
        g.set_i32(io.buckets, &self.te_cfg.buckets(n));
        g.compute()?;
        let mut states = g.read_f32(io.out);
        states.resize(self.te_cfg.d_model as usize * TEXT_TOKENS, 0.0);
        Ok(states)
    }

    /// Generate one video.
    ///
    /// # Errors
    /// [`Error::Request`] for a size, length or step count the model cannot
    /// serve; backend failures otherwise.
    pub fn generate(&mut self, req: &VideoRequest) -> Result<Video> {
        Cosmos3::check(req)?;
        let (lt, lh, lw) = grid(req);
        let noise = schedule::gaussian(req.seed, self.cfg.in_channels as usize * lt * lh * lw);
        self.generate_from(req, noise)
    }

    /// [`Self::generate`] from given starting noise `[z][T][H][W]`.
    pub(crate) fn generate_from(&mut self, req: &VideoRequest, noise: Vec<f32>) -> Result<Video> {
        let t0 = Instant::now();
        let (latents, evaluations, encode_ms, denoise_ms) = self.sample(req, noise, t0)?;
        let t2 = Instant::now();
        let px = wan::decode(&self.backend, &self.vae_cfg, &self.vae, &latents, grid(req))?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        let (rgb, frames) = to_rgb8(&px, req.width as usize, req.height as usize);
        Ok(Video {
            width: req.width,
            height: req.height,
            frames: frames as u32,
            fps: req.fps,
            rgb,
            seed: req.seed,
            evaluations,
            timings: Timings { encode_ms, denoise_ms, decode_ms },
        })
    }

    /// Final normalised latents `[z][T][H][W]` from `latents` (starting
    /// noise), the transformer evaluations run, and encode/denoise times.
    pub(crate) fn sample(&self, req: &VideoRequest, latents: Vec<f32>, t0: Instant) -> Result<(Vec<f32>, u32, u64, u64)> {
        self.sample_traced(req, latents, t0, &mut Vec::new())
    }

    /// [`Self::sample`], also keeping the guided velocity and the latents
    /// after every step in `trace`.
    pub(crate) fn sample_traced(&self, req: &VideoRequest, mut latents: Vec<f32>, t0: Instant, trace: &mut Vec<Vec<f32>>) -> Result<(Vec<f32>, u32, u64, u64)> {
        Cosmos3::check(req)?;
        let shape = grid(req);
        let (lt, lh, lw) = shape;
        if latents.len() != self.cfg.in_channels as usize * lt * lh * lw {
            return Err(Error::Request("starting noise does not match the latent grid".into()));
        }
        let use_cfg = req.guidance_scale > 1.0;
        let negative = req.negative_prompt.clone().unwrap_or_default();
        let mut contexts = vec![self.context(&self.tokens(&req.prompt)?)?];
        if use_cfg {
            contexts.push(self.context(&self.tokens(&negative)?)?);
        }
        let conditioned = match &req.image {
            Some(img) => {
                let first = wan::encode_frames(&self.backend, &self.vae_cfg, &self.vae, std::slice::from_ref(img))?;
                place_first_frame(&mut latents, &first, lt, lh * lw);
                true
            }
            None => false,
        };
        let encode_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let (rows, cols) = (lh / 2, lw / 2);
        let n = lt * rows * cols;
        let time_tokens = if conditioned { n } else { 1 };
        let (cos, sin) = self.cfg.rotary_tables(lt, rows, cols);
        let mut passes = Vec::with_capacity(contexts.len());
        for ctx in contexts {
            let mut g = Graph::new(&self.backend)?;
            let io = wan_dit::build(&mut g, &self.cfg, &self.tf, &self.pe, n as i64, TEXT_TOKENS as i64, time_tokens as i64, self.exact);
            g.finish(&[io.out])?;
            passes.push((g, io, ctx));
        }
        let (sigmas, timesteps) = flow_sigmas_schedule(req.steps as usize, self.sched.flow_shift, self.sched.num_train_timesteps);
        let mut sampler = UniPc::new(sigmas);
        let mut evaluations = 0u32;
        let first = rows * cols;
        for &t in &timesteps {
            let t = t as f32;
            let time: Vec<f32> = if conditioned {
                (0..n).flat_map(|i| wan_dit::time_features(if i < first { 0.0 } else { t })).collect()
            } else {
                wan_dit::time_features(t)
            };
            let patches = wan_dit::patchify(&self.cfg, &latents, lt, lh, lw);
            let mut preds = Vec::with_capacity(passes.len());
            for (g, io, ctx) in &passes {
                g.set_f32(io.patches, &patches);
                g.set_f32(io.time, &time);
                g.set_f32(io.context, ctx);
                g.set_f32(io.cos, &cos);
                g.set_f32(io.sin, &sin);
                g.compute()?;
                evaluations += 1;
                preds.push(wan_dit::unpatchify(&self.cfg, &g.read_f32(io.out), lt, lh, lw));
            }
            let mut v: Vec<f32> = if use_cfg {
                preds[1].iter().zip(&preds[0]).map(|(u, c)| u + req.guidance_scale * (c - u)).collect()
            } else {
                preds.pop().expect("one pass")
            };
            if conditioned {
                zero_first_frame(&mut v, lt, lh * lw);
            }
            latents = sampler.step(&v, &latents)?;
            trace.push(v);
            trace.push(latents.clone());
        }
        Ok((latents, evaluations, encode_ms, t1.elapsed().as_millis() as u64))
    }
}

/// Latent grid `(frames, rows, cols)` of a request.
fn grid(req: &VideoRequest) -> (usize, usize, usize) {
    (1 + (req.num_frames as usize - 1) / 4, req.height as usize / 16, req.width as usize / 16)
}

/// Write one encoded frame `[z][plane]` over frame 0 of `[z][T][plane]`.
fn place_first_frame(latents: &mut [f32], first: &[f32], lt: usize, plane: usize) {
    for (c, src) in first.chunks_exact(plane).enumerate() {
        latents[c * lt * plane..c * lt * plane + plane].copy_from_slice(src);
    }
}

/// Zero frame 0 of `[z][T][plane]`.
fn zero_first_frame(v: &mut [f32], lt: usize, plane: usize) {
    let z = v.len() / (lt * plane);
    for c in 0..z {
        v[c * lt * plane..c * lt * plane + plane].fill(0.0);
    }
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_whitespace_normalised() {
        assert_eq!(clean_prompt("  a  red\tboat \n"), "a red boat");
    }

    #[test]
    fn the_first_frame_is_placed_and_held() {
        let (lt, plane, z) = (3, 4, 2);
        let mut lat = vec![1.0; z * lt * plane];
        place_first_frame(&mut lat, &[5.0; 8], lt, plane);
        assert_eq!(&lat[0..4], &[5.0; 4]);
        assert_eq!(&lat[12..16], &[5.0; 4]);
        assert_eq!(lat[4], 1.0);
        zero_first_frame(&mut lat, lt, plane);
        assert_eq!(&lat[12..16], &[0.0; 4]);
        assert_eq!(lat[16], 1.0);
    }
}
