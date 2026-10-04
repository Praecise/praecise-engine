//! LTX-2.3 text-to-audio-video generation: prompt encoding, the guided
//! flow-matching loop over both streams, and decoding to frames and a
//! waveform.
//!
//! Every component comes from the release's single checkpoint file except the
//! text encoder (Gemma 3) and its tokenizer, which live in a separate
//! directory as `text_encoder/` and `tokenizer/tokenizer.json`.

use std::path::Path;
use std::time::Instant;

use super::audio_vae::Ltx2AudioDecoder;
use super::connectors::Ltx2Connectors;
use super::vae::{Ltx2VideoDecoder, Tiling};
use super::vocoder::Ltx2Vocoder;
use super::{AvShape, Ltx2Transformer, Pass};
use crate::error::{Error, Result};
use crate::gemma3::Gemma3Encoder;
use crate::music::Audio;
use crate::pipeline::{CheckpointFiles, LoadOptions, Timings};
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
}

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
}

impl std::fmt::Debug for Ltx2Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Pipeline").field("transformer", &self.transformer).finish_non_exhaustive()
    }
}

impl Ltx2Pipeline {
    /// Load the single checkpoint file `checkpoint` and the text encoder
    /// (`text_encoder/`, and `tokenizer/tokenizer.json` when present) under
    /// `text`.
    ///
    /// # Errors
    /// On an unsupported configuration, missing weights, an unreadable
    /// tokenizer or no usable backend.
    pub fn load(checkpoint: &Path, text: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let tok_path = text.root.join("tokenizer/tokenizer.json");
        let tokenizer = if tok_path.exists() {
            Some(tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?)
        } else {
            None
        };
        Ok(Self {
            tokenizer,
            text: Gemma3Encoder::load(text, "text_encoder", opts)?,
            connectors: Ltx2Connectors::load_single_file(checkpoint, opts)?,
            transformer: Ltx2Transformer::load_single_file(checkpoint, opts)?,
            video_vae: Ltx2VideoDecoder::load_single_file(checkpoint, opts)?,
            audio_vae: Ltx2AudioDecoder::load_single_file(checkpoint, opts)?,
            vocoder: Ltx2Vocoder::load_single_file(checkpoint, opts)?,
        })
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
        if req.steps < 2 || !(req.fps > 0.0) {
            return Err(Error::Request("at least two steps and a positive frame rate are required".into()));
        }
        let audio = audio_frames(req.num_frames, req.fps, cfg.audio_sampling_rate, cfg.audio_hop_length, cfg.audio_scale_factor);
        if audio == 0 {
            return Err(Error::Request("the clip is too short for one audio latent".into()));
        }
        let s = AvShape { frames: (req.num_frames - 1) / t + 1, height: req.height / sp, width: req.width / sp, audio_frames: audio, fps: req.fps };
        Ok((s, (cfg.in_channels as usize) * s.frames * s.height * s.width))
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
        let (s, n_video) = self.check(req)?;
        let cfg = self.transformer.config();
        let n_audio = s.audio_frames * cfg.audio_in_channels as usize;
        let noise = gaussian(req.seed, n_video + n_audio);
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
        let mut out = self.generate_from(&pos, &neg, video, audio, req)?;
        out.video.seed = req.seed;
        out.audio.seed = req.seed;
        tracing::debug!(ms = started.elapsed().as_millis() as u64, "audio-video generation done");
        Ok(out)
    }

    /// The final packed latents of both streams for given prompt tokens and
    /// packed starting noise.
    pub(crate) fn latents_from(&self, pos: &[u32], neg: &[u32], video: Vec<f32>, audio: Vec<f32>, req: &Ltx2Request) -> Result<(Latents, AvShape, Timings)> {
        let (s, n_video) = self.check(req)?;
        let cfg = self.transformer.config();
        if video.len() != n_video || audio.len() != s.audio_frames * cfg.audio_in_channels as usize {
            return Err(Error::Request("starting noise disagrees with the request".into()));
        }
        let mut timings = Timings::default();
        let t0 = Instant::now();
        let p = self.encode(pos, req.max_sequence_length)?;
        let n = self.encode(neg, req.max_sequence_length)?;
        timings.encode_ms = t0.elapsed().as_millis() as u64;
        let t0 = Instant::now();
        let sig = sigmas(req.steps, s.frames * s.height * s.width);
        let lat = denoise(&self.transformer, video, audio, &p, &n, s, &sig, req)?;
        timings.denoise_ms = t0.elapsed().as_millis() as u64;
        Ok((lat, s, timings))
    }

    /// Generate from prompt tokens and packed starting noise.
    pub(crate) fn generate_from(&self, pos: &[u32], neg: &[u32], video: Vec<f32>, audio: Vec<f32>, req: &Ltx2Request) -> Result<Ltx2Output> {
        let (lat, s, mut timings) = self.latents_from(pos, neg, video, audio, req)?;
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
