//! Cosmos3 video generation: text (and optionally a first frame) to video.
//!
//! The prompt is wrapped in the model's chat template with duration and
//! resolution sentences appended, and fed as raw token ids to the joint
//! transformer (there is no separate text encoder). The video is generated in
//! a 48-channel latent at 1/16 of the resolution and 1/4 of the frame rate
//! (plus the first frame), sampled with UniPC under classifier-free guidance,
//! and decoded frame by frame by the causal autoencoder.
//!
//! A conditioning image becomes the first latent frame: it is encoded, kept
//! clean through sampling (no timestep embedding, zero velocity), and every
//! other frame is generated around it.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cosmos3::{self, Cosmos3Config};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, WType, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision, RgbImage, Timings};
use crate::safetensors::SafeTensors;
use crate::schedule;
use crate::unipc::{UniPc, UniPcConfig};
use crate::wan::{self, WanVaeConfig};

/// Longest templated prompt, in tokens.
pub const MAX_PROMPT_TOKENS: usize = 4096;
const SYSTEM_PROMPT_VIDEO: &str = "You are a helpful assistant who will generate videos from a give prompt.";
const SYSTEM_PROMPT_IMAGE: &str = "You are a helpful assistant who will generate images from a give prompt.";

/// A video generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoRequest {
    /// What to show.
    pub prompt: String,
    /// What to avoid; empty when absent.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// First frame, exactly `width` x `height`; text-to-video when absent.
    #[serde(default)]
    pub image: Option<RgbImage>,
    /// Width in pixels, a multiple of 32.
    pub width: u32,
    /// Height in pixels, a multiple of 32.
    pub height: u32,
    /// Frames: one (an image) or `4k + 1`.
    pub num_frames: u32,
    /// Frame rate the video is generated for.
    pub fps: f32,
    /// Denoising steps.
    pub steps: u32,
    /// Classifier-free guidance scale; 1 disables guidance.
    pub guidance_scale: f32,
    /// Seed for the starting noise.
    pub seed: u64,
}

/// A generated video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame count.
    pub frames: u32,
    /// Frame rate.
    pub fps: f32,
    /// Frames one after another, each row-major 8-bit RGB.
    pub rgb: Vec<u8>,
    /// Seed used.
    pub seed: u64,
    /// Transformer evaluations run (steps, doubled under guidance).
    pub evaluations: u32,
    /// Per-stage timings.
    pub timings: Timings,
}

impl Video {
    /// The video as an uncompressed YUV4MPEG2 stream (full-resolution
    /// chroma, BT.601 full range).
    #[must_use]
    pub fn y4m(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (num, den) = rational(self.fps);
        let mut out = format!("YUV4MPEG2 W{w} H{h} F{num}:{den} Ip A1:1 C444 XCOLORRANGE=FULL\n").into_bytes();
        for f in self.rgb.chunks_exact(w * h * 3) {
            out.extend_from_slice(b"FRAME\n");
            let mut planes = vec![0u8; 3 * w * h];
            for (i, px) in f.chunks_exact(3).enumerate() {
                let (r, g, b) = (f32::from(px[0]), f32::from(px[1]), f32::from(px[2]));
                let y = 0.299 * r + 0.587 * g + 0.114 * b;
                planes[i] = y.round().clamp(0.0, 255.0) as u8;
                planes[w * h + i] = (128.0 + 0.564 * (b - y)).round().clamp(0.0, 255.0) as u8;
                planes[2 * w * h + i] = (128.0 + 0.713 * (r - y)).round().clamp(0.0, 255.0) as u8;
            }
            out.extend_from_slice(&planes);
        }
        out
    }
}

fn rational(fps: f32) -> (u32, u32) {
    let den = 1000u32;
    ((f64::from(fps) * f64::from(den)).round() as u32, den)
}

/// A loaded Cosmos3 pipeline.
pub struct Cosmos3 {
    backend: Backend,
    cfg: Cosmos3Config,
    tf: Weights,
    vae_cfg: WanVaeConfig,
    vae: Weights,
    sched: UniPcConfig,
    tokenizer: tokenizers::Tokenizer,
    eos: i32,
    vision_start: i32,
    system_prompt: bool,
    /// Float32 attention throughout, for the full-precision format.
    exact: bool,
}

impl std::fmt::Debug for Cosmos3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cosmos3").field("backend", &self.backend).field("resident_bytes", &self.resident_bytes()).finish_non_exhaustive()
    }
}

/// `base. addition`, or `addition` alone when `base` is empty; trailing
/// periods of `base` are dropped first.
fn append(base: &str, addition: &str) -> String {
    let b = base.trim_end_matches('.');
    if b.is_empty() { addition.to_string() } else { format!("{b}. {addition}") }
}

/// Latent grid `(frames, height, width)` of a request.
fn grid(req: &VideoRequest) -> (usize, usize, usize) {
    (((req.num_frames - 1) / 4 + 1) as usize, (req.height / 16) as usize, (req.width / 16) as usize)
}

impl Cosmos3 {
    /// Load a checkpoint in the diffusers layout. Refuses to run on the CPU
    /// when the host has GPU hardware this build cannot drive.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let index = files.json("model_index.json")?;
        let class = index.get("_class_name").and_then(Value::as_str).unwrap_or_default();
        if class != "Cosmos3OmniPipeline" {
            return Err(Error::Config(format!("pipeline class {class:?} is not Cosmos3OmniPipeline")));
        }
        let system_prompt = index.get("default_use_system_prompt").and_then(Value::as_bool).unwrap_or(true);
        let cfg: Cosmos3Config = parse(files.json("transformer/config.json")?, "transformer config")?;
        cfg.validate()?;
        let vae_cfg: WanVaeConfig = parse(files.json("vae/config.json")?, "vae config")?;
        vae_cfg.validate()?;
        if vae_cfg.z_dim != cfg.latent_channel || cfg.latent_patch_size != 2 {
            return Err(Error::Config("transformer and autoencoder latents disagree".into()));
        }
        let sched: UniPcConfig = parse(files.json("scheduler/scheduler_config.json")?, "scheduler config")?;
        sched.validate()?;

        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "video backend selected");
        let exact = opts.precision == Precision::F32;
        let tf_files = SafeTensors::open(&files.weights("transformer")?)?;
        let tf = Weights::load(&backend, &tf_files, &cfg.weight_specs(opts.precision.wtype()))?;
        drop(tf_files);
        let vae_files = SafeTensors::open(&files.weights("vae")?)?;
        let vae = Weights::from_host(&backend, &vae_cfg.host_tensors(&vae_files, exact)?)?;
        drop(vae_files);

        let tok_path = files.root.join("text_tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        let id = |t: &str| tokenizer.token_to_id(t).map(|i| i as i32).ok_or_else(|| Error::Tokenizer(format!("no {t} token")));
        let eos = id("<|im_end|>")?;
        let vision_start = id("<|vision_start|>")?;
        Ok(Self { backend, cfg, tf, vae_cfg, vae, sched, tokenizer, eos, vision_start, system_prompt, exact })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.tf.bytes() + self.vae.bytes()
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// The conditional and unconditional texts: the prompt and the negative
    /// prompt with the duration (videos only) and resolution sentences, and
    /// their negations, appended.
    #[must_use]
    pub fn prompts(req: &VideoRequest) -> (String, String) {
        let image = req.num_frames == 1;
        let fps = f64::from(req.fps);
        let duration = f64::from(req.num_frames) / fps;
        let kind = if image { "image" } else { "video" };
        let (h, w) = (req.height, req.width);
        let apply = |text: &str, negative: bool| {
            let not = if negative { "not " } else { "" };
            let mut t = text.to_string();
            if !image {
                t = append(&t, &format!("The video is {not}{duration:.1} seconds long and is {not}of {fps:.0} FPS."));
            }
            append(&t, &format!("This {kind} is {not}of {h}x{w} resolution."))
        };
        (apply(&req.prompt, false), apply(req.negative_prompt.as_deref().unwrap_or(""), true))
    }

    /// Token ids of a templated text, ending with the end-of-turn and
    /// start-of-generation tokens.
    pub(crate) fn tokens(&self, text: &str, image: bool) -> Result<Vec<i32>> {
        let system = if !self.system_prompt {
            ""
        } else if image {
            SYSTEM_PROMPT_IMAGE
        } else {
            SYSTEM_PROMPT_VIDEO
        };
        let chat = format!("\n<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n");
        let enc = self.tokenizer.encode(chat, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        ids.push(self.eos);
        ids.push(self.vision_start);
        if ids.len() > MAX_PROMPT_TOKENS {
            return Err(Error::Request(format!("prompt is {} tokens, at most {MAX_PROMPT_TOKENS}", ids.len())));
        }
        Ok(ids)
    }

    /// Positions `[time, height, width]` of the latent patch tokens after a
    /// text of `text` tokens, in frame, row, column order.
    fn video_positions(&self, text: usize, frames: usize, gh: usize, gw: usize, fps: f32) -> Vec<[f32; 3]> {
        let offset = (text as u64 + self.cfg.unified_3d_mrope_temporal_modality_margin) as f32;
        let spatial = if self.cfg.unified_3d_mrope_reset_spatial_ids { 0.0 } else { offset };
        let temporal = self.vae_cfg.scale_factor_temporal as f64;
        let modulate = self.cfg.enable_fps_modulation && frames > 1;
        let tps = (f64::from(fps) / temporal) as f32;
        let base_tps = (self.cfg.base_fps / temporal) as f32;
        let mut out = Vec::with_capacity(frames * gh * gw);
        for f in 0..frames {
            let t = if modulate { f as f32 / tps * base_tps + offset } else { f as f32 + offset };
            for y in 0..gh {
                for x in 0..gw {
                    out.push([t, y as f32 + spatial, x as f32 + spatial]);
                }
            }
        }
        out
    }

    /// Every layer's text keys and values for the generation stream.
    pub(crate) fn text_cache(&self, ids: &[i32]) -> Result<Weights> {
        let n = ids.len();
        let mut g = Graph::new(&self.backend)?;
        let io = cosmos3::build_text(&mut g, &self.cfg, &self.tf, n as i64, self.exact);
        let outputs: Vec<_> = io.keys.iter().chain(&io.values).copied().collect();
        g.finish(&outputs)?;
        let pos: Vec<[f32; 3]> = (0..n).map(|i| [i as f32; 3]).collect();
        let (cos, sin) = self.cfg.rotary_tables(&pos);
        g.set_i32(io.ids, ids);
        g.set_f32(io.cos, &cos);
        g.set_f32(io.sin, &sin);
        g.set_f16(io.mask, &cosmos3::causal_mask(n));
        g.compute()?;
        let shape = vec![self.cfg.num_key_value_heads, n as u64, self.cfg.head_dim];
        let mut host = Vec::with_capacity(2 * self.cfg.num_hidden_layers);
        for (i, (k, v)) in io.keys.iter().zip(&io.values).enumerate() {
            host.push(HostTensor { name: format!("k{i}"), shape: shape.clone(), ty: WType::F32, data: g.read_f32(*k) });
            host.push(HostTensor { name: format!("v{i}"), shape: shape.clone(), ty: WType::F32, data: g.read_f32(*v) });
        }
        Weights::from_host(&self.backend, &host)
    }

    /// Encode a first frame to its normalised latent `[z][1][H/16][W/16]`.
    pub(crate) fn encode_image(&self, img: &RgbImage) -> Result<Vec<f32>> {
        let (w, h) = (img.width as usize, img.height as usize);
        let mut px = vec![0f32; 3 * w * h];
        for (i, p) in img.rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                px[c * w * h + i] = f32::from(p[c]) / 127.5 - 1.0;
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let io = wan::build_encoder(&mut g, &self.vae_cfg, &self.vae, w as i64, h as i64);
        g.finish(&[io.mean])?;
        g.set_f32(io.pixels, &wan::patchify(&px, w, h));
        g.compute()?;
        let mut mean = g.read_f32(io.mean);
        let per = mean.len() / self.vae_cfg.z_dim as usize;
        for (c, chunk) in mean.chunks_exact_mut(per).enumerate() {
            let (m, inv) = (self.vae_cfg.latents_mean[c], 1.0 / self.vae_cfg.latents_std[c]);
            for v in chunk {
                *v = (*v - m) * inv;
            }
        }
        Ok(mean)
    }

    /// Patch tokens `[tokens][192]` of latents `[z][T][H][W]`, frame-major.
    fn patches(&self, lat: &[f32], (lt, lh, lw): (usize, usize, usize)) -> Vec<f32> {
        let z = self.cfg.latent_channel as usize;
        let (gh, gw) = (lh / 2, lw / 2);
        let pd = 4 * z;
        let mut out = vec![0f32; lt * gh * gw * pd];
        for t in 0..lt {
            for y in 0..gh {
                for x in 0..gw {
                    let tok = (t * gh + y) * gw + x;
                    for ph in 0..2 {
                        for pw in 0..2 {
                            for c in 0..z {
                                out[tok * pd + (ph * 2 + pw) * z + c] = lat[((c * lt + t) * lh + 2 * y + ph) * lw + 2 * x + pw];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Velocities `[z][T][H][W]` from the noisy tokens' predictions, zero on
    /// the first `cond` frames.
    fn velocity(&self, pred: &[f32], cond: usize, (lt, lh, lw): (usize, usize, usize)) -> Vec<f32> {
        let z = self.cfg.latent_channel as usize;
        let (gh, gw) = (lh / 2, lw / 2);
        let pd = 4 * z;
        let mut out = vec![0f32; z * lt * lh * lw];
        for t in cond..lt {
            for y in 0..gh {
                for x in 0..gw {
                    let tok = ((t - cond) * gh + y) * gw + x;
                    for ph in 0..2 {
                        for pw in 0..2 {
                            for c in 0..z {
                                out[((c * lt + t) * lh + 2 * y + ph) * lw + 2 * x + pw] = pred[tok * pd + (ph * 2 + pw) * z + c];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The sampling loop from `latents` `[z][T][H][W]` (conditioning frames
    /// already in place), returning the final latents and the transformer
    /// evaluations run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denoise(
        &self,
        cond_ids: &[i32],
        uncond_ids: &[i32],
        mut latents: Vec<f32>,
        cond_frames: usize,
        shape: (usize, usize, usize),
        fps: f32,
        steps: usize,
        guidance: f32,
    ) -> Result<(Vec<f32>, u32)> {
        let (lt, lh, lw) = shape;
        let (gh, gw) = (lh / 2, lw / 2);
        let n = (lt * gh * gw) as i64;
        let cond = (cond_frames * gh * gw) as i64;
        let use_cfg = guidance != 1.0;
        let mut passes = Vec::new();
        for ids in if use_cfg { vec![cond_ids, uncond_ids] } else { vec![cond_ids] } {
            let cache = self.text_cache(ids)?;
            let mut g = Graph::new(&self.backend)?;
            let io = cosmos3::build_gen(&mut g, &self.cfg, &self.tf, &cache, n, cond, None, self.exact);
            g.finish(&[io.out])?;
            let (cos, sin) = self.cfg.rotary_tables(&self.video_positions(ids.len(), lt, gh, gw, fps));
            passes.push((g, io, cache, cos, sin));
        }
        let (sigmas, timesteps) = self.sched.schedule(steps);
        let mut sampler = UniPc::new(sigmas);
        let mut evaluations = 0u32;
        for &t in &timesteps {
            let patches = self.patches(&latents, shape);
            let time = self.cfg.time_features(t);
            let mut preds = Vec::with_capacity(passes.len());
            for (g, io, _, cos, sin) in &passes {
                g.set_f32(io.patches, &patches);
                g.set_f32(io.time, &time);
                g.set_f32(io.cos, cos);
                g.set_f32(io.sin, sin);
                g.compute()?;
                evaluations += 1;
                preds.push(self.velocity(&g.read_f32(io.out), cond_frames, shape));
            }
            let v = if use_cfg {
                preds[1].iter().zip(&preds[0]).map(|(u, c)| u + guidance * (c - u)).collect()
            } else {
                preds.pop().expect("one pass")
            };
            latents = sampler.step(&v, &latents)?;
        }
        Ok((latents, evaluations))
    }

    /// Decode normalised latents `[z][T][H][W]` to frames `[F][3][H*16][W*16]`
    /// in `[-1, 1]`.
    pub(crate) fn decode(&self, latents: &[f32], (lt, lh, lw): (usize, usize, usize)) -> Result<Vec<f32>> {
        let z = self.vae_cfg.z_dim as usize;
        let plane = lh * lw;
        let cache = Weights::zeros(&self.backend, &self.vae_cfg.decoder_cache_specs(lw as i64, lh as i64))?;
        let mut out = Vec::new();
        let mut graphs: Vec<(Graph, wan::DecodeIo)> = Vec::new();
        for first in [true, false].into_iter().take(lt.min(2)) {
            let mut g = Graph::new(&self.backend)?;
            let io = wan::build_decoder(&mut g, &self.vae_cfg, &self.vae, &cache, lw as i64, lh as i64, first);
            g.finish(&[io.out])?;
            graphs.push((g, io));
        }
        for t in 0..lt {
            let mut frame = vec![0f32; z * plane];
            for c in 0..z {
                let (m, inv) = (self.vae_cfg.latents_mean[c], 1.0 / self.vae_cfg.latents_std[c]);
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
            out.extend(wan::unpatchify(&x, io.out.ne(0) as usize, io.out.ne(1) as usize, io.out.ne(3) as usize));
        }
        Ok(out)
    }

    fn check(req: &VideoRequest) -> Result<()> {
        let bad = |m: String| Err(Error::Request(m));
        if req.width == 0 || req.height == 0 || req.width % 32 != 0 || req.height % 32 != 0 {
            return bad("width and height must be positive multiples of 32".into());
        }
        if req.num_frames == 0 || (req.num_frames - 1) % 4 != 0 {
            return bad("frames must be 1 or 4k + 1".into());
        }
        if !(req.fps > 0.0 && req.fps.is_finite()) {
            return bad("fps must be positive".into());
        }
        if req.steps == 0 {
            return bad("steps must be at least 1".into());
        }
        if let Some(img) = &req.image {
            if img.width != req.width || img.height != req.height {
                return bad(format!("the first frame is {}x{}, the video {}x{}", img.width, img.height, req.width, req.height));
            }
            if img.rgb.len() != (img.width * img.height * 3) as usize {
                return bad("the first frame's pixel buffer does not match its size".into());
            }
        }
        Ok(())
    }

    /// Generate one video.
    ///
    /// # Errors
    /// [`Error::Request`] for a size, length or step count the model cannot
    /// serve; backend failures otherwise.
    pub fn generate(&mut self, req: &VideoRequest) -> Result<Video> {
        Self::check(req)?;
        let (lt, lh, lw) = grid(req);
        let noise = schedule::gaussian(req.seed, self.cfg.latent_channel as usize * lt * lh * lw);
        self.generate_from(req, noise)
    }

    /// [`Self::generate`] from given starting noise `[z][T][H][W]`.
    pub(crate) fn generate_from(&mut self, req: &VideoRequest, mut latents: Vec<f32>) -> Result<Video> {
        Self::check(req)?;
        let shape = grid(req);
        let (lt, lh, lw) = shape;
        let t0 = Instant::now();
        let (cond_text, uncond_text) = Self::prompts(req);
        let image = req.num_frames == 1;
        let cond_ids = self.tokens(&cond_text, image)?;
        let uncond_ids = self.tokens(&uncond_text, image)?;
        let cond_frames = match &req.image {
            Some(img) => {
                let first = self.encode_image(img)?;
                let plane = lh * lw;
                for (c, chunk) in first.chunks_exact(plane).enumerate() {
                    latents[c * lt * plane..c * lt * plane + plane].copy_from_slice(chunk);
                }
                1
            }
            None => 0,
        };
        let encode_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let (latents, evaluations) =
            self.denoise(&cond_ids, &uncond_ids, latents, cond_frames, shape, req.fps, req.steps as usize, req.guidance_scale)?;
        let denoise_ms = t1.elapsed().as_millis() as u64;
        let t2 = Instant::now();
        let px = self.decode(&latents, shape)?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        let (w, h) = (req.width as usize, req.height as usize);
        let frames = px.len() / (3 * w * h);
        let mut rgb = vec![0u8; frames * w * h * 3];
        for f in 0..frames {
            for c in 0..3 {
                for i in 0..w * h {
                    let v = px[(f * 3 + c) * w * h + i];
                    rgb[(f * w * h + i) * 3 + c] = ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
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
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    fn req(frames: u32) -> VideoRequest {
        VideoRequest {
            prompt: "A red kite over a beach.".into(),
            negative_prompt: None,
            image: None,
            width: 832,
            height: 480,
            num_frames: frames,
            fps: 24.0,
            steps: 35,
            guidance_scale: 6.0,
            seed: 0,
        }
    }

    #[test]
    fn prompts_carry_duration_and_resolution_and_their_negations() {
        let (c, u) = Cosmos3::prompts(&req(121));
        assert_eq!(c, "A red kite over a beach. The video is 5.0 seconds long and is of 24 FPS. This video is of 480x832 resolution.");
        assert_eq!(u, "The video is not 5.0 seconds long and is not of 24 FPS. This video is not of 480x832 resolution.");
        let (c, _) = Cosmos3::prompts(&req(1));
        assert_eq!(c, "A red kite over a beach. This image is of 480x832 resolution.");
    }

    #[test]
    fn the_latent_grid_counts_the_first_frame_apart() {
        assert_eq!(grid(&req(121)), (31, 30, 52));
        assert_eq!(grid(&req(1)), (1, 30, 52));
    }

    #[test]
    fn y4m_has_a_header_and_one_plane_set_per_frame() {
        let v = Video { width: 2, height: 2, frames: 2, fps: 24.0, rgb: vec![255; 24], seed: 0, evaluations: 0, timings: Timings::default() };
        let y = v.y4m();
        let header = b"YUV4MPEG2 W2 H2 F24000:1000 Ip A1:1 C444 XCOLORRANGE=FULL\n";
        assert!(y.starts_with(header));
        assert_eq!(y.len(), header.len() + 2 * (6 + 12));
    }
}
