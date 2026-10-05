//! LTX-2.3 text-to-audio-video generation: prompt encoding, the guided
//! flow-matching loop over both streams, and decoding to frames and a
//! waveform.
//!
//! Every component comes from the release's single checkpoint file except the
//! text encoder (Gemma 3) and its tokenizer, which live in a separate
//! directory as `text_encoder/` and `tokenizer/tokenizer.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use super::audio_vae::Ltx2AudioDecoder;
use super::connectors::Ltx2Connectors;
use super::upsampler::Ltx2LatentUpsampler;
use super::vae::{Ltx2VideoDecoder, Tiling};
use super::vocoder::Ltx2Vocoder;
use super::{AvShape, Ltx2Transformer, Pass};
use crate::error::{Error, Result};
use crate::gemma3::Gemma3Encoder;
use crate::music::Audio;
use crate::pipeline::{CheckpointFiles, LoadOptions, Precision, Timings};
use crate::schedule::gaussian;
use crate::video::Video;

/// Guidance strengths of one stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Guidance {
    /// Classifier-free guidance against the negative prompt (1 = off).
    pub scale: f32,
    /// Spatio-temporal guidance against the perturbed blocks (0 = off).
    pub stg: f32,
    /// Guidance against the run with the streams isolated (1 = off).
    pub modality: f32,
    /// Share of the guided prediction rescaled to the conditional one's
    /// spread (0 = off).
    pub rescale: f32,
}

impl Guidance {
    /// The release's video defaults.
    pub const VIDEO: Self = Self { scale: 3.0, stg: 1.0, modality: 3.0, rescale: 0.7 };
    /// The release's audio defaults.
    pub const AUDIO: Self = Self { scale: 7.0, stg: 1.0, modality: 3.0, rescale: 0.7 };
    /// No guidance: one conditional pass per step (distilled checkpoints).
    pub const OFF: Self = Self { scale: 1.0, stg: 0.0, modality: 1.0, rescale: 0.0 };
}

/// Noise levels of the denoising loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Schedule {
    /// `steps` levels shifted by the clip's token count (full checkpoints).
    #[default]
    Shifted,
    /// The fixed eight levels distilled checkpoints are trained on; `steps`
    /// is ignored.
    Distilled,
}

/// Noise levels of a distilled checkpoint's first stage.
pub const DISTILLED_SIGMAS: [f32; 8] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875];
/// Noise levels of a distilled checkpoint's refinement stage after latent
/// upsampling; the upsampled latents are re-noised to the first level.
pub const DISTILLED_REFINE_SIGMAS: [f32; 3] = [0.909375, 0.725, 0.421875];

/// One audio-video generation.
#[derive(Debug, Clone)]
pub struct Ltx2Request {
    /// What to generate.
    pub prompt: String,
    /// What to steer away from.
    pub negative_prompt: String,
    /// Width in pixels (a multiple of the spatial compression).
    pub width: usize,
    /// Height in pixels (a multiple of the spatial compression).
    pub height: usize,
    /// Frames (one more than a multiple of the temporal compression).
    pub num_frames: usize,
    /// Frame rate.
    pub fps: f32,
    /// Denoising steps.
    pub steps: usize,
    /// Starting-noise seed.
    pub seed: u64,
    /// Video guidance.
    pub video: Guidance,
    /// Audio guidance.
    pub audio: Guidance,
    /// Blocks perturbed by spatio-temporal guidance.
    pub stg_blocks: Vec<usize>,
    /// Padded prompt length.
    pub max_sequence_length: usize,
    /// Noise levels of the loop.
    pub schedule: Schedule,
    /// Two stages: generate at half the width and height, upsample the
    /// video latents, re-noise and refine at full size (needs a latent
    /// upsampler and the distilled schedule).
    pub upsample: bool,
}

impl Ltx2Request {
    /// A request for a distilled checkpoint: the distilled schedule, no
    /// guidance on either stream, and the two-stage recipe when `upsample`.
    #[must_use]
    pub fn distilled(prompt: impl Into<String>, upsample: bool) -> Self {
        Self { prompt: prompt.into(), schedule: Schedule::Distilled, upsample, video: Guidance::OFF, audio: Guidance::OFF, ..Self::default() }
    }
}

impl Default for Ltx2Request {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 768,
            height: 512,
            num_frames: 121,
            fps: 24.0,
            steps: 30,
            seed: 0,
            video: Guidance::VIDEO,
            audio: Guidance::AUDIO,
            stg_blocks: vec![28],
            max_sequence_length: 1024,
            schedule: Schedule::Shifted,
            upsample: false,
        }
    }
}

/// A generated clip with its soundtrack.
#[derive(Debug, Clone)]
pub struct Ltx2Output {
    /// The frames.
    pub video: Video,
    /// The soundtrack.
    pub audio: Audio,
}

impl Ltx2Output {
    /// The clip and its soundtrack as one MP4 file.
    ///
    /// # Errors
    /// As [`Video::mp4`].
    pub fn mp4(&self) -> crate::Result<Vec<u8>> {
        self.video.mp4(Some(&self.audio))
    }
}

/// Flow-matching noise levels for `steps` steps over `video_tokens` latent
/// tokens, ending in 0: a linear ramp shifted exponentially by a
/// token-count-dependent amount, then stretched to end at 0.1.
#[must_use]
pub fn sigmas(steps: usize, video_tokens: usize) -> Vec<f32> {
    let (base_len, max_len, base_shift, max_shift, terminal) = (1024.0, 4096.0, 0.95, 2.05, 0.1);
    let m = (max_shift - base_shift) / (max_len - base_len);
    let mu: f64 = video_tokens as f64 * m + (base_shift - m * base_len);
    let e = mu.exp();
    let n = steps.max(1);
    let mut s: Vec<f64> = (0..n)
        .map(|i| {
            let lin = if n == 1 { 1.0 } else { 1.0 + (1.0 / n as f64 - 1.0) * i as f64 / (n - 1) as f64 };
            let lin = f64::from(lin as f32);
            e / (e + (1.0 / lin - 1.0))
        })
        .collect();
    let scale = (1.0 - s[n - 1]) / (1.0 - terminal);
    for v in &mut s {
        *v = 1.0 - (1.0 - *v) / scale;
    }
    let mut out: Vec<f32> = s.into_iter().map(|v| v as f32).collect();
    out.push(0.0);
    out
}

/// Audio latents for a clip of `frames` frames at `fps`.
#[must_use]
pub fn audio_frames(frames: usize, fps: f32, sampling_rate: u64, hop_length: u64, compression: u64) -> usize {
    let per_second = sampling_rate as f64 / hop_length as f64 / compression as f64;
    (frames as f64 / f64::from(fps) * per_second).round() as usize
}

/// Unbiased standard deviation, as the reference's rescale takes it.
fn std(v: &[f32]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / n;
    (v.iter().map(|x| (f64::from(*x) - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Velocity predictions of the guidance passes of one stream, combined into
/// the guided velocity, all on denoised estimates.
fn guide(latent: &[f32], sigma: f32, g: Guidance, cond: &[f32], uncond: Option<&[f32]>, stg: Option<&[f32]>, iso: Option<&[f32]>) -> Vec<f32> {
    let x0 = |v: &[f32]| -> Vec<f32> { latent.iter().zip(v).map(|(x, v)| x - v * sigma).collect() };
    let cond = x0(cond);
    let (unc, pert, isol) = (uncond.map(x0), stg.map(x0), iso.map(x0));
    let mut out = cond.clone();
    for (k, o) in out.iter_mut().enumerate() {
        if let Some(other) = &unc {
            *o += (g.scale - 1.0) * (cond[k] - other[k]);
        }
        if let Some(other) = &pert {
            *o += g.stg * (cond[k] - other[k]);
        }
        if let Some(other) = &isol {
            *o += (g.modality - 1.0) * (cond[k] - other[k]);
        }
    }
    if g.rescale > 0.0 {
        let ratio = (std(&cond) / std(&out)) as f32;
        for o in &mut out {
            *o = g.rescale * (*o * ratio) + (1.0 - g.rescale) * *o;
        }
    }
    latent.iter().zip(&out).map(|(x, g)| (x - g) / sigma).collect()
}

/// Prompt features of one prompt for both streams.
#[derive(Debug, Clone)]
pub(crate) struct Prompt {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// The loop's result: packed latents of both streams.
#[derive(Debug, Clone)]
pub(crate) struct Latents {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    pub evaluations: u32,
}

/// Denoise packed starting noise `video` `[tokens][channels]` and `audio`
/// `[frames][channels]` over `sigmas`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise(tf: &Ltx2Transformer, mut video: Vec<f32>, mut audio: Vec<f32>, pos: &Prompt, neg: &Prompt, s: AvShape, sigmas: &[f32], req: &Ltx2Request) -> Result<Latents> {
    let (gv, ga) = (req.video, req.audio);
    let cfg = gv.scale > 1.0 || ga.scale > 1.0;
    let stg = gv.stg > 0.0 || ga.stg > 0.0;
    let iso = gv.modality > 1.0 || ga.modality > 1.0;
    let plain = Pass::default();
    let stg_pass = Pass { perturbed_blocks: req.stg_blocks.clone(), isolate_modalities: false };
    let iso_pass = Pass { perturbed_blocks: Vec::new(), isolate_modalities: true };
    let mut evaluations = 0;
    for i in 0..sigmas.len() - 1 {
        let (sigma, next) = (sigmas[i], sigmas[i + 1]);
        let t = sigma * 1000.0;
        let mut run = |p: &Prompt, pass: &Pass| {
            evaluations += 1;
            tf.forward(&video, &audio, &p.video, &p.audio, s, (t, t), pass)
        };
        let (vc, ac) = run(pos, &plain)?;
        let u = if cfg { Some(run(neg, &plain)?) } else { None };
        let st = if stg { Some(run(pos, &stg_pass)?) } else { None };
        let is = if iso { Some(run(pos, &iso_pass)?) } else { None };
        let vel_v = guide(&video, sigma, gv, &vc, u.as_ref().map(|x| x.0.as_slice()), st.as_ref().map(|x| x.0.as_slice()), is.as_ref().map(|x| x.0.as_slice()));
        let vel_a = guide(&audio, sigma, ga, &ac, u.as_ref().map(|x| x.1.as_slice()), st.as_ref().map(|x| x.1.as_slice()), is.as_ref().map(|x| x.1.as_slice()));
        let dt = next - sigma;
        for (x, v) in video.iter_mut().zip(&vel_v) {
            *x += dt * v;
        }
        for (x, v) in audio.iter_mut().zip(&vel_a) {
            *x += dt * v;
        }
    }
    Ok(Latents { video, audio, evaluations })
}

/// `[C][N]` to `[N][C]`.
pub(crate) fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// A resident LTX-2.3 pipeline.
pub struct Ltx2Pipeline {
    tokenizer: Option<tokenizers::Tokenizer>,
    text: Gemma3Encoder,
    connectors: Ltx2Connectors,
    transformer: Ltx2Transformer,
    video_vae: Ltx2VideoDecoder,
    audio_vae: Ltx2AudioDecoder,
    vocoder: Ltx2Vocoder,
    upsampler: Option<Ltx2LatentUpsampler>,
}

impl std::fmt::Debug for Ltx2Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Pipeline").field("transformer", &self.transformer).finish_non_exhaustive()
    }
}

impl Ltx2Pipeline {
    /// Load the checkpoint `files` and the text encoder (`text_encoder/`,
    /// and `tokenizer/tokenizer.json` when present) under `text`. The
    /// checkpoint is the single file or its sections split over several
    /// files (see [`open_checkpoint`](super::single_file::open_checkpoint)); a latent upsampler file among them is
    /// loaded as the upsampler.
    ///
    /// # Errors
    /// On an unsupported configuration, missing weights, an unreadable
    /// tokenizer or no usable backend.
    pub fn load(files: &[PathBuf], text: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let tok_path = text.root.join("tokenizer/tokenizer.json");
        let tokenizer = if tok_path.exists() {
            Some(tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?)
        } else {
            None
        };
        let mut checkpoint = Vec::new();
        let mut upsampler = None;
        for f in files {
            let st = crate::safetensors::SafeTensors::open(std::slice::from_ref(f))?;
            if Ltx2LatentUpsampler::recognises(&st) {
                if upsampler.is_some() {
                    return Err(Error::Weights("more than one latent upsampler file".into()));
                }
                upsampler = Some(Ltx2LatentUpsampler::from_files(&st, opts)?);
            } else {
                checkpoint.push(f.clone());
            }
        }
        // The audio path runs in f32 at every precision: the vocoder's
        // second stage re-analyses its own output through a log-mel, which
        // amplifies reduced-precision mel error into the waveform.
        let audio_opts = LoadOptions { precision: Precision::F32, ..opts };
        let c = &checkpoint;
        Ok(Self {
            tokenizer,
            text: Gemma3Encoder::load(text, "text_encoder", opts)?,
            connectors: Ltx2Connectors::load_single_file(c, opts)?,
            transformer: Ltx2Transformer::load_single_file(c, opts)?,
            video_vae: Ltx2VideoDecoder::load_single_file(c, opts)?,
            audio_vae: Ltx2AudioDecoder::load_single_file(c, audio_opts)?,
            vocoder: Ltx2Vocoder::load_single_file(c, audio_opts)?,
            upsampler,
        })
    }

    /// Load from directories: every `.safetensors` and `.gguf` file under
    /// `checkpoint_roots` (searched recursively, skipping `text_encoder/`
    /// and `tokenizer/` directories) is a checkpoint file; `text_root` holds
    /// `text_encoder/` and `tokenizer/tokenizer.json` and may be one of the
    /// checkpoint roots.
    ///
    /// # Errors
    /// As [`Self::load`], or when no checkpoint file is found.
    pub fn load_dir(checkpoint_roots: &[PathBuf], text_root: &Path, opts: LoadOptions) -> Result<Self> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
            for e in std::fs::read_dir(dir).map_err(|e| Error::Weights(format!("{}: {e}", dir.display())))? {
                let p = e?.path();
                if p.is_dir() {
                    if !p.file_name().is_some_and(|n| n == "text_encoder" || n == "tokenizer") {
                        walk(&p, out)?;
                    }
                } else if p.extension().is_some_and(|x| x == "safetensors" || x == "gguf") {
                    out.push(p);
                }
            }
            Ok(())
        }
        let mut found = Vec::new();
        for r in checkpoint_roots {
            walk(r, &mut found)?;
        }
        found.sort();
        if found.is_empty() {
            return Err(Error::Weights("no checkpoint file under the checkpoint directories".into()));
        }
        Self::load(&found, &CheckpointFiles::new(text_root), opts)
    }

    /// Device bytes held by every component.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.text.bytes() + self.connectors.bytes() + self.transformer.bytes() + self.video_vae.bytes() + self.audio_vae.bytes() + self.vocoder.bytes() + self.upsampler.as_ref().map_or(0, Ltx2LatentUpsampler::bytes)
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.transformer.device()
    }

    /// The prompt's tokens as the reference tokenizes them: special tokens
    /// added, surrounding whitespace dropped, cut at `max_len`.
    fn tokens(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| Error::Tokenizer("no tokenizer/tokenizer.json beside the text encoder".into()))?;
        let enc = tok.encode(text.trim(), true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(max_len);
        Ok(ids)
    }

    /// Both streams' prompt features for the prompt's tokens, left-padded to
    /// `max_len` (padding is masked out by the reference, so it is never
    /// encoded).
    fn encode(&self, tokens: &[u32], max_len: usize) -> Result<Prompt> {
        if tokens.is_empty() || tokens.len() > max_len {
            return Err(Error::Request(format!("a prompt must have 1..={max_len} tokens")));
        }
        let hidden = self.text.forward(tokens, max_len - tokens.len())?;
        let (video, audio) = self.connectors.forward(&hidden, max_len)?;
        Ok(Prompt { video, audio })
    }

    fn check(&self, req: &Ltx2Request) -> Result<(AvShape, usize)> {
        let cfg = self.transformer.config();
        let (t, sp) = (cfg.vae_scale_factors[0] as usize, cfg.vae_scale_factors[1] as usize);
        if req.width == 0 || req.height == 0 || req.width % sp != 0 || req.height % sp != 0 {
            return Err(Error::Request(format!("width and height must be positive multiples of {sp}")));
        }
        if req.num_frames == 0 || (req.num_frames - 1) % t != 0 {
            return Err(Error::Request(format!("frames must be one more than a multiple of {t}")));
        }
        if (req.schedule == Schedule::Shifted && req.steps < 2) || !(req.fps > 0.0) {
            return Err(Error::Request("at least two steps and a positive frame rate are required".into()));
        }
        if req.upsample {
            if req.schedule != Schedule::Distilled || self.upsampler.is_none() {
                return Err(Error::Request("two-stage generation needs the distilled schedule and a latent upsampler".into()));
            }
            if req.width % (2 * sp) != 0 || req.height % (2 * sp) != 0 {
                return Err(Error::Request(format!("two-stage width and height must be multiples of {}", 2 * sp)));
            }
        }
        let audio = audio_frames(req.num_frames, req.fps, cfg.audio_sampling_rate, cfg.audio_hop_length, cfg.audio_scale_factor);
        if audio == 0 {
            return Err(Error::Request("the clip is too short for one audio latent".into()));
        }
        let s = AvShape { frames: (req.num_frames - 1) / t + 1, height: req.height / sp, width: req.width / sp, audio_frames: audio, fps: req.fps };
        Ok((s, (cfg.in_channels as usize) * s.frames * s.height * s.width))
    }

    /// The first stage's request: half the width and height for two-stage
    /// generation, else the request itself.
    fn first_stage(req: &Ltx2Request) -> Ltx2Request {
        if req.upsample {
            Ltx2Request { width: req.width / 2, height: req.height / 2, upsample: false, ..req.clone() }
        } else {
            req.clone()
        }
    }

    /// Packed noise of both streams for shape `s` from `seed`.
    fn noise(&self, seed: u64, s: AvShape, n_video: usize) -> (Vec<f32>, Vec<f32>) {
        let cfg = self.transformer.config();
        let n_audio = s.audio_frames * cfg.audio_in_channels as usize;
        let noise = gaussian(seed, n_video + n_audio);
        let video = transpose(&noise[..n_video], cfg.in_channels as usize, n_video / cfg.in_channels as usize);
        // Audio noise is drawn `[channels][frames][mel]`; packing puts every
        // frame's channels and mel bins together.
        let ac = self.audio_vae.config().z_channels as usize;
        let mel = n_audio / ac / s.audio_frames;
        let mut audio = vec![0.0; n_audio];
        for c in 0..ac {
            for f in 0..s.audio_frames {
                for m in 0..mel {
                    audio[(f * ac + c) * mel + m] = noise[n_video + (c * s.audio_frames + f) * mel + m];
                }
            }
        }
        (video, audio)
    }

    /// Generate a clip and its soundtrack.
    ///
    /// # Errors
    /// On a request the model cannot serve, a missing tokenizer, or a
    /// backend failure.
    pub fn generate(&self, req: &Ltx2Request) -> Result<Ltx2Output> {
        let started = Instant::now();
        let pos = self.tokens(&req.prompt, req.max_sequence_length)?;
        let neg = self.tokens(&req.negative_prompt, req.max_sequence_length)?;
        let (full, n_full) = self.check(req)?;
        let (s, n_video) = self.check(&Self::first_stage(req))?;
        let (video, audio) = self.noise(req.seed, s, n_video);
        let refine = req.upsample.then(|| self.noise(req.seed.wrapping_add(1), full, n_full));
        let mut out = self.generate_from(&pos, &neg, video, audio, refine, req)?;
        out.video.seed = req.seed;
        out.audio.seed = req.seed;
        tracing::debug!(ms = started.elapsed().as_millis() as u64, "audio-video generation done");
        Ok(out)
    }

    /// The final packed latents of both streams for given prompt tokens,
    /// packed starting noise of the first stage and, for two-stage
    /// generation, the packed noise the upsampled latents are mixed with.
    pub(crate) fn latents_from(&self, pos: &[u32], neg: &[u32], video: Vec<f32>, audio: Vec<f32>, refine: Option<(Vec<f32>, Vec<f32>)>, req: &Ltx2Request) -> Result<(Latents, AvShape, Timings)> {
        let first = Self::first_stage(req);
        let (s1, n1) = self.check(&first)?;
        let (s2, n2) = self.check(req)?;
        let cfg = self.transformer.config();
        let c = cfg.in_channels as usize;
        let n_audio = s1.audio_frames * cfg.audio_in_channels as usize;
        if video.len() != n1 || audio.len() != n_audio {
            return Err(Error::Request("starting noise disagrees with the request".into()));
        }
        let mut timings = Timings::default();
        let t0 = Instant::now();
        let p = self.encode(pos, req.max_sequence_length)?;
        // Without classifier-free guidance the negative prompt is never read.
        let n = if req.video.scale > 1.0 || req.audio.scale > 1.0 { self.encode(neg, req.max_sequence_length)? } else { p.clone() };
        timings.encode_ms = t0.elapsed().as_millis() as u64;
        let t0 = Instant::now();
        let sig = match req.schedule {
            Schedule::Shifted => sigmas(req.steps, s1.frames * s1.height * s1.width),
            Schedule::Distilled => DISTILLED_SIGMAS.iter().copied().chain([0.0]).collect(),
        };
        let mut lat = denoise(&self.transformer, video, audio, &p, &n, s1, &sig, &first)?;
        if !req.upsample {
            timings.denoise_ms = t0.elapsed().as_millis() as u64;
            return Ok((lat, s1, timings));
        }
        let (nv, na) = refine.ok_or_else(|| Error::Request("two-stage generation needs refinement noise".into()))?;
        if nv.len() != n2 || na.len() != n_audio {
            return Err(Error::Request("refinement noise disagrees with the request".into()));
        }
        let up = self.upsampler.as_ref().ok_or_else(|| Error::Request("no latent upsampler loaded".into()))?;
        // The upsampler works on decoder-space latents.
        let (mean, std) = self.video_vae.latent_stats();
        let tokens1 = s1.frames * s1.height * s1.width;
        let mut z = transpose(&lat.video, tokens1, c);
        for (ch, plane) in z.chunks_exact_mut(tokens1).enumerate() {
            for v in plane {
                *v = *v * std[ch] + mean[ch];
            }
        }
        let mut z = up.upsample(&z, s1.frames, s1.height, s1.width)?;
        let tokens2 = n2 / c;
        for (ch, plane) in z.chunks_exact_mut(tokens2).enumerate() {
            for v in plane {
                *v = (*v - mean[ch]) / std[ch];
            }
        }
        let video = transpose(&z, c, tokens2);
        let level = DISTILLED_REFINE_SIGMAS[0];
        let mix = |x: &[f32], noise: &[f32]| -> Vec<f32> { x.iter().zip(noise).map(|(x, e)| level * e + (1.0 - level) * x).collect() };
        let (video, audio) = (mix(&video, &nv), mix(&lat.audio, &na));
        let sig: Vec<f32> = DISTILLED_REFINE_SIGMAS.iter().copied().chain([0.0]).collect();
        let first_evaluations = lat.evaluations;
        lat = denoise(&self.transformer, video, audio, &p, &n, s2, &sig, req)?;
        lat.evaluations += first_evaluations;
        timings.denoise_ms = t0.elapsed().as_millis() as u64;
        Ok((lat, s2, timings))
    }

    /// Generate from prompt tokens and packed starting noise.
    pub(crate) fn generate_from(&self, pos: &[u32], neg: &[u32], video: Vec<f32>, audio: Vec<f32>, refine: Option<(Vec<f32>, Vec<f32>)>, req: &Ltx2Request) -> Result<Ltx2Output> {
        let (lat, s, mut timings) = self.latents_from(pos, neg, video, audio, refine, req)?;
        let t0 = Instant::now();
        let (frames, wave) = self.decode(&lat, s)?;
        timings.decode_ms = t0.elapsed().as_millis() as u64;
        let sp = self.video_vae.config().factors().0 as usize;
        let (h, w) = (s.height * sp, s.width * sp);
        let nf = frames.len() / (3 * h * w);
        let mut rgb = vec![0u8; frames.len()];
        for c in 0..3 {
            for i in 0..nf * h * w {
                rgb[i * 3 + c] = ((frames[c * nf * h * w + i] / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        let channels = self.vocoder.channels();
        Ok(Ltx2Output {
            video: Video { width: w as u32, height: h as u32, frames: nf as u32, fps: req.fps, rgb, seed: 0, evaluations: lat.evaluations, timings },
            audio: Audio { sample_rate: self.vocoder.sample_rate() as u32, channels: channels as u32, samples: wave, seed: 0, evaluations: lat.evaluations, timings },
        })
    }

    /// Pixels `[3][T][H][W]` in about `[-1, 1]` and the waveform
    /// `[channels][samples]` of final packed latents.
    pub(crate) fn decode(&self, lat: &Latents, s: AvShape) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.transformer.config().in_channels as usize;
        let z = transpose(&lat.video, s.frames * s.height * s.width, c);
        let frames = self.video_vae.decode_tiled(&z, s.frames, s.height, s.width, Tiling::default())?;
        let mel = self.audio_vae.decode(&lat.audio, s.audio_frames)?;
        let wave = self.vocoder.synthesize(&mel, self.audio_vae.config().spectrogram_frames(s.audio_frames))?;
        Ok((frames, wave))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_reference_values() {
        // Reference: FlowMatchEulerDiscreteScheduler with the release's
        // scheduler configuration, 4 steps, 12 video tokens.
        let s = sigmas(4, 12);
        assert_eq!(s.len(), 5);
        assert_eq!(s[4], 0.0);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[3] - 0.1).abs() < 1e-6, "ends at the terminal value");
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn audio_latents_follow_the_clip_length() {
        assert_eq!(audio_frames(121, 24.0, 16_000, 160, 4), 126);
        assert_eq!(audio_frames(9, 24.0, 16_000, 160, 4), 9);
    }

    #[test]
    fn guidance_without_extra_passes_is_the_conditional_velocity() {
        let lat = [0.5f32, -1.0, 2.0];
        let v = [0.1f32, 0.2, -0.3];
        let g = Guidance { scale: 1.0, stg: 0.0, modality: 1.0, rescale: 0.0 };
        let out = guide(&lat, 0.7, g, &v, None, None, None);
        for (a, b) in out.iter().zip(v) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rescale_uses_the_unbiased_spread() {
        assert!((std(&[1.0, 3.0]) - std::f64::consts::SQRT_2).abs() < 1e-12);
    }
}
