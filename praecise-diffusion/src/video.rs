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

/// Pipeline classes in `model_index.json`: the current layout, then the older
/// one of the Nano and Super checkpoints.
pub const PIPELINE_CLASSES: [&str; 2] = ["Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline"];

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
    /// The video as an MP4 file (H.264, BT.709), with `sound` as its
    /// soundtrack (Opus) when given.
    ///
    /// # Errors
    /// An odd width or height, or an encoder failure.
    pub fn mp4(&self, sound: Option<&crate::music::Audio>) -> Result<Vec<u8>> {
        let sound = sound.map(|a| praecise_codec::Sound { sample_rate: a.sample_rate, channels: a.channels, samples: &a.samples });
        Ok(praecise_codec::encode(praecise_codec::EncodeOptions::default(), self.width, self.height, self.fps, &self.rgb, sound)?)
    }
}

/// The frames of a video file (MP4 or Matroska/WebM; H.264, H.265 or AV1)
/// in display order, and its frame rate.
///
/// # Errors
/// A file that is not one of those formats, or fails to decode.
pub fn frames_from_video(bytes: &[u8]) -> Result<(Vec<RgbImage>, f32)> {
    let mut frames = Vec::new();
    let info = praecise_codec::decode_each(bytes, |_, rgb| {
        frames.push(rgb.to_vec());
        true
    })?;
    let images = frames.into_iter().map(|rgb| RgbImage { width: info.width, height: info.height, rgb }).collect();
    Ok((images, info.fps))
}

/// Frames `[F][3][H][W]` in `[-1, 1]` to interleaved 8-bit RGB, and the
/// frame count.
pub(crate) fn to_rgb8(px: &[f32], w: usize, h: usize) -> (Vec<u8>, usize) {
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
    (rgb, frames)
}

/// How the tokenizer's chat template frames a prompt; the template ships
/// with the tokenizer and differs between checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Edge (Nemotron): a leading newline, the system turn always present,
    /// and an opened thinking block after the assistant header.
    Nemotron,
    /// Nano and Super (Qwen3-VL): the system turn only when used, and a bare
    /// assistant header.
    Qwen,
}

impl ChatTemplate {
    /// Recognise a chat template from its Jinja source.
    ///
    /// # Errors
    /// [`Error::Config`] for a template that is neither form.
    pub fn recognise(source: &str) -> Result<Self> {
        if source.contains("enable_thinking") && source.contains("<|im_start|>assistant\n<think>") {
            Ok(Self::Nemotron)
        } else if source.contains("{{- '<|im_start|>assistant\\n' }}") && !source.contains("<think>") {
            Ok(Self::Qwen)
        } else {
            Err(Error::Config("the tokenizer's chat template is not one this pipeline renders".into()))
        }
    }

    /// The prompt text before tokenization; `system` is empty when the
    /// system turn is not used.
    #[must_use]
    pub fn render(self, system: &str, text: &str) -> String {
        match self {
            Self::Nemotron => format!("\n<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
            Self::Qwen if system.is_empty() => format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"),
            Self::Qwen => format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"),
        }
    }
}

/// The chat template source of a tokenizer directory: `chat_template.jinja`,
/// else the `chat_template` entry of `tokenizer_config.json`.
fn chat_template(dir: &std::path::Path) -> Result<String> {
    if let Ok(s) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
        return Ok(s);
    }
    let path = dir.join("tokenizer_config.json");
    let cfg: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    cfg.get("chat_template")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::Config(format!("{} has no chat template", dir.display())))
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
    chat: ChatTemplate,
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
    grid_of(req.num_frames, req.height, req.width)
}

/// Latent grid `(frames, height, width)` of `num_frames` frames of
/// `height` x `width`.
fn grid_of(num_frames: u32, height: u32, width: u32) -> (usize, usize, usize) {
    (((num_frames - 1) / 4 + 1) as usize, (height / 16) as usize, (width / 16) as usize)
}

impl Cosmos3 {
    /// Load a checkpoint in the diffusers layout: the current one (Edge) or
    /// the older one the Nano and Super checkpoints ship in (another pipeline
    /// class name, no pipeline flags, extra vision-encoder and sound
    /// components, which video generation does not read). Refuses to run on
    /// the CPU when the host has GPU hardware this build cannot drive.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let index = files.json("model_index.json")?;
        let class = index.get("_class_name").and_then(Value::as_str).unwrap_or_default();
        if !PIPELINE_CLASSES.contains(&class) {
            return Err(Error::Config(format!("pipeline class {class:?} is not one of {PIPELINE_CLASSES:?}")));
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

        let chat = ChatTemplate::recognise(&chat_template(&files.root.join("text_tokenizer"))?)?;
        let tok_path = files.root.join("text_tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        let id = |t: &str| tokenizer.token_to_id(t).map(|i| i as i32).ok_or_else(|| Error::Tokenizer(format!("no {t} token")));
        let eos = id("<|im_end|>")?;
        let vision_start = id("<|vision_start|>")?;
        Ok(Self { backend, cfg, tf, vae_cfg, vae, sched, tokenizer, eos, vision_start, system_prompt, chat, exact })
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
        let chat = self.chat.render(system, text);
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
        self.encode_frames(std::slice::from_ref(img))
    }

    /// Encode `1 + 4k` frames of one size to normalised latents
    /// `[z][1 + k][H/16][W/16]`: the first frame alone, then four at a time
    /// through the encoder's frame caches.
    pub(crate) fn encode_frames(&self, frames: &[RgbImage]) -> Result<Vec<f32>> {
        wan::encode_frames(&self.backend, &self.vae_cfg, &self.vae, frames)
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
            g.finish(&[io.out.expect("a noisy video token")])?;
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
                preds.push(self.velocity(&g.read_f32(io.out.expect("a noisy video token")), cond_frames, shape));
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
        wan::decode(&self.backend, &self.vae_cfg, &self.vae, latents, (lt, lh, lw))
    }

    pub(crate) fn check(req: &VideoRequest) -> Result<()> {
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
        let (rgb, frames) = to_rgb8(&px, w, h);
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

mod action;
pub use action::{action_caption, action_resolution_tier, ActionMode, ActionOutput, ActionRequest, Embodiment};

#[cfg(test)]
mod parity;

#[cfg(test)]
mod action_parity;

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
    fn mp4_round_trips_every_frame() {
        let (w, h, n) = (32u32, 32u32, 3usize);
        let rgb: Vec<u8> = (0..n).flat_map(|f| (0..w * h).flat_map(move |i| [(i % 32 * 8) as u8, (f * 80) as u8, 128])).collect();
        let v = Video { width: w, height: h, frames: n as u32, fps: 24.0, rgb, seed: 0, evaluations: 0, timings: Timings::default() };
        let file = v.mp4(None).unwrap();
        let (frames, fps) = frames_from_video(&file).unwrap();
        assert_eq!(frames.len(), n);
        assert!((fps - 24.0).abs() < 1e-3);
        for (f, img) in frames.iter().enumerate() {
            assert_eq!((img.width, img.height), (w, h));
            // Frame order: the green channel steps by frame.
            let g = img.rgb.iter().skip(1).step_by(3).map(|&x| f64::from(x)).sum::<f64>() / f64::from(w * h);
            assert!((g - f as f64 * 80.0).abs() < 4.0, "frame {f}: mean green {g}");
        }
    }
}

#[cfg(test)]
mod chat_tests {
    use super::*;

    #[test]
    fn chat_templates_are_recognised_and_rendered() {
        let nemotron = "{%- if add_generation_prompt %}{%- if enable_thinking %}{{- '<|im_start|>assistant\n<think>\n' }}{%- endif %}{%- endif %}";
        let qwen = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}";
        assert_eq!(ChatTemplate::recognise(nemotron).unwrap(), ChatTemplate::Nemotron);
        assert_eq!(ChatTemplate::recognise(qwen).unwrap(), ChatTemplate::Qwen);
        assert!(ChatTemplate::recognise("{{ messages }}").is_err());
        assert_eq!(ChatTemplate::Qwen.render("", "hi"), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(
            ChatTemplate::Qwen.render("S", "hi"),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(ChatTemplate::Nemotron.render("", "hi").starts_with("\n<|im_start|>system\n<|im_end|>"));
    }
}
