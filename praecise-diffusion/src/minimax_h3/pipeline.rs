//! MiniMax-H3 text-to-video-and-audio generation.
//!
//! The prompt is tokenized verbatim (no chat template, no special tokens) and
//! encoded by Qwen3-VL up to layer 50. One packed sequence `[prompt | stereo
//! audio latents | video patches]` carries a `(t, h, w)` rotary grid: prompt
//! tokens count along time; audio latents continue from the prompt length,
//! one per latent, the left channel at the first patch column and the right
//! at the last; video frames start at the prompt length and advance by
//! `5/3 x (1, 4, 4, 4, 4, ...)` latent-frame spans, their patches on a grid
//! centred on the frame and scaled to 32 over the side of a square of the
//! same area. The checkpoint is guidance-distilled: one forward per step
//! predicts every row's velocity, video and audio each step down their own
//! shifted schedule (`s b / (1 + (s - 1) b)` over a linear ramp, shift 12
//! for video and 3 for audio), and prompt rows ride the video timestep.

use std::time::Instant;

use super::audio_vae::H3AudioVae;
use super::vae::{to_unit_rgb, H3VideoVae};
use super::{H3Input, H3Source, H3Token, MiniMaxH3Transformer};
use crate::error::{Error, Result};
use crate::music::Audio;
use crate::pipeline::{CheckpointFiles, LoadOptions, Timings};
use crate::qwen3_vl::Qwen3VlEncoder;
use crate::schedule::gaussian;
use crate::video::Video;

/// Modality tags of the packed sequence.
pub const VIDEO_TAG: usize = 0;
pub const TEXT_TAG: usize = 1;
pub const AUDIO_TAG: usize = 2;
/// Frame rate of every clip.
pub const FPS: f32 = 24.0;
/// Audio latents per second.
pub const AUDIO_LATENTS_PER_SECOND: f64 = 40.0;
/// Stereo channels, each its own run of audio rows.
pub const AUDIO_CHANNELS: usize = 2;
/// The prompt encoder layer whose output conditions the transformer.
pub const TEXT_ENCODER_LAYER: usize = 50;
/// Clip length bounds in seconds.
pub const MIN_SECONDS: f32 = 5.0;
pub const MAX_SECONDS: f32 = 15.0;
/// Default canvas: short edge and pixel budget.
pub const CANVAS_SHORT_EDGE: f64 = 768.0;
pub const CANVAS_MAX_PIXELS: f64 = 768.0 * 1344.0;

const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// One generation request.
#[derive(Debug, Clone)]
pub struct H3Request {
    pub prompt: String,
    /// Frames; rounded up to the next `17 n + 5`.
    pub num_frames: usize,
    /// Canvas; `None` picks 16:9 at the default short edge.
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Inference steps as the reference counts them (the schedule has this
    /// many noise levels, so one fewer transformer evaluation).
    pub steps: usize,
    pub seed: u64,
}

impl Default for H3Request {
    fn default() -> Self {
        Self { prompt: String::new(), num_frames: 240, width: None, height: None, steps: 50, seed: 0 }
    }
}

/// A generated clip with its stereo soundtrack.
#[derive(Debug, Clone)]
pub struct H3Output {
    pub video: Video,
    pub audio: Audio,
}

impl H3Output {
    /// The clip and its soundtrack as one AVI file.
    #[must_use]
    pub fn avi(&self) -> Vec<u8> {
        self.video.avi(&self.audio)
    }
}

/// Resolved geometry of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H3Shape {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub audio_latents: usize,
}

/// Float outputs before 8-bit conversion: frames `[3][T][H][W]` in `[0, 1]`
/// and the waveform `[channels][samples]`.
pub(crate) struct Decoded {
    pub(crate) frames: Vec<f32>,
    pub(crate) wave: Vec<f32>,
    pub(crate) shape: H3Shape,
    pub(crate) timings: Timings,
    pub(crate) evaluations: u32,
}

/// The canvas `(height, width)` for an aspect ratio, a short edge and a pixel
/// budget, both multiples of `multiple` (the reference's `resolve_canvas_size`).
///
/// # Errors
/// An aspect ratio outside 1:4 to 4:1.
pub fn canvas(aspect_width: f64, aspect_height: f64, multiple: usize, short_edge: f64, max_pixels: f64) -> Result<(usize, usize)> {
    let ratio = aspect_width / aspect_height;
    if !(aspect_width > 0.0 && aspect_height > 0.0 && (0.25..=4.0).contains(&ratio)) {
        return Err(Error::Request(format!("MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}")));
    }
    let (mut w, mut h) = if ratio >= 1.0 { (short_edge * ratio, short_edge) } else { (short_edge, short_edge / ratio) };
    if w * h > max_pixels {
        let s = (max_pixels / (w * h)).sqrt();
        w *= s;
        h *= s;
    }
    let m = multiple as f64;
    let snap = |v: f64| multiple.max(((v / m).round_ties_even() * m) as usize);
    Ok((snap(h), snap(w)))
}

/// Flow-matching noise levels of the reference scheduler: `steps` levels of
/// a linear ramp from 1 to 0 shifted by `shift`, repeats dropped.
#[must_use]
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps.max(2);
    let step = -1.0f32 / (n - 1) as f32;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        // torch.linspace fills each half from its own end.
        let b = if i < n / 2 { 1.0 + step * i as f32 } else { 0.0 - step * (n - 1 - i) as f32 };
        let s = shift * b / (1.0 + (shift - 1.0) * b);
        if out.last() != Some(&s) {
            out.push(s);
        }
    }
    out
}

fn spatial_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let n = dim / patch;
    let step = ratio / n as f64;
    (0..n).map(|k| (k as f64 * step + left) * ROPE_SPATIAL_SCALE).collect()
}

/// Rotary positions and tags of `[prompt | audio | video]`, in that order.
pub(crate) fn layout(text: usize, s: &H3Shape, patch: [usize; 3]) -> (Vec<[f32; 3]>, Vec<usize>) {
    let [pt, ph, pw] = patch;
    let sqrt_area = ((s.latent_height * s.latent_width) as f64).sqrt();
    let hg = spatial_grid(s.latent_height, ph, sqrt_area);
    let wg = spatial_grid(s.latent_width, pw, sqrt_area);
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut tags = Vec::new();
    for i in 0..text {
        pos.push([i as f32, 0.0, 0.0]);
        tags.push(TEXT_TAG);
    }
    for ch in 0..AUDIO_CHANNELS {
        let col = if ch == 0 { wg[0] } else { wg[wg.len() - 1] };
        for j in 0..s.audio_latents {
            pos.push([(text as f64 + j as f64) as f32, 0.0, col as f32]);
            tags.push(AUDIO_TAG);
        }
    }
    let mut t = text as f64;
    for f in 0..s.latent_frames / pt {
        for h in &hg {
            for w in &wg {
                pos.push([t as f32, *h as f32, *w as f32]);
                tags.push(VIDEO_TAG);
            }
        }
        t += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[f % ROPE_FRAMES_PER_LATENT.len()];
    }
    (pos, tags)
}

/// Video latents `[C][T][H][W]` to patch rows `[(T', H', W')][(C, pt, ph, pw)]`.
pub(crate) fn patchify(z: &[f32], c: usize, [t, h, w]: [usize; 3], [pt, ph, pw]: [usize; 3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(z.len());
    for ft in 0..t / pt {
        for fh in 0..h / ph {
            for fw in 0..w / pw {
                for ch in 0..c {
                    for it in 0..pt {
                        for ih in 0..ph {
                            for iw in 0..pw {
                                out.push(z[((ch * t + ft * pt + it) * h + fh * ph + ih) * w + fw * pw + iw]);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`patchify`].
pub(crate) fn unpatchify(rows: &[f32], c: usize, [t, h, w]: [usize; 3], [pt, ph, pw]: [usize; 3]) -> Vec<f32> {
    let mut z = vec![0f32; rows.len()];
    let mut k = 0;
    for ft in 0..t / pt {
        for fh in 0..h / ph {
            for fw in 0..w / pw {
                for ch in 0..c {
                    for it in 0..pt {
                        for ih in 0..ph {
                            for iw in 0..pw {
                                z[((ch * t + ft * pt + it) * h + fh * ph + ih) * w + fw * pw + iw] = rows[k];
                                k += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    z
}

/// One step of the reference scheduler on `x` given velocity `v` at
/// timestep `t` (`1 - sigma`).
fn step(x: &mut [f32], v: &[f32], t: f32, sigma: f32, sigma_next: f32) {
    let s = 1.0 - t;
    let ratio = sigma_next / sigma;
    for (xi, vi) in x.iter_mut().zip(v) {
        let denoised = *xi + s * vi;
        *xi = ratio * *xi + (1.0 - ratio) * denoised;
    }
}

/// A loaded MiniMax-H3 pipeline.
pub struct MiniMaxH3Pipeline {
    transformer: MiniMaxH3Transformer,
    vae: H3VideoVae,
    audio_vae: H3AudioVae,
    text: Option<(Qwen3VlEncoder, tokenizers::Tokenizer)>,
    video_shift: f32,
    audio_shift: f32,
}

impl std::fmt::Debug for MiniMaxH3Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniMaxH3Pipeline").field("device", &self.transformer.device()).finish_non_exhaustive()
    }
}

fn shift(files: &CheckpointFiles, dir: &str) -> Result<f32> {
    let v = files.json(&format!("{dir}/scheduler_config.json"))?;
    let s = v["shift"].as_f64().ok_or_else(|| Error::Config(format!("{dir}: no shift")))?;
    if s <= 0.0 {
        return Err(Error::Config(format!("{dir}: shift must be positive")));
    }
    Ok(s as f32)
}

impl MiniMaxH3Pipeline {
    /// Load a diffusers-layout checkpoint: `transformer/`, `vae/`,
    /// `audio_vae/`, `text_encoder/` (only its first 50 layers),
    /// `tokenizer/tokenizer.json`, `scheduler/` and `audio_scheduler/`.
    ///
    /// # Errors
    /// A missing or malformed part, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let mut p = Self::load_denoiser(files, opts)?;
        let enc = Qwen3VlEncoder::load_layers(files, "text_encoder", opts, Some(TEXT_ENCODER_LAYER))?;
        let tok_path = files.root.join("tokenizer/tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        if enc.config().text_config.hidden_size != p.transformer.config().text_dim {
            return Err(Error::Config("prompt encoder width differs from the transformer's text width".into()));
        }
        p.text = Some((enc, tok));
        Ok(p)
    }

    /// Everything but the prompt encoder.
    pub(crate) fn load_denoiser(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let transformer = MiniMaxH3Transformer::load(files, opts)?;
        let vae = H3VideoVae::load(files, opts)?;
        let audio_vae = H3AudioVae::load(files, opts)?;
        let tc = transformer.config();
        if vae.config().latent_channels != tc.in_channels || audio_vae.config().latent_channels != tc.audio_in_channels {
            return Err(Error::Config("autoencoder latent widths differ from the transformer's".into()));
        }
        Ok(Self { transformer, vae, audio_vae, text: None, video_shift: shift(files, "scheduler")?, audio_shift: shift(files, "audio_scheduler")? })
    }

    /// The device the transformer runs on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.transformer.device()
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let text = self.text.as_ref().map_or(0, |(e, _)| e.resident_bytes());
        self.transformer.resident_bytes() + self.vae.resident_bytes() + self.audio_vae.resident_bytes() + text
    }

    /// Resolve a request's geometry as the reference does.
    ///
    /// # Errors
    /// A canvas off the patch grid, or a clip outside 5 to 15 seconds.
    pub fn shape(&self, req: &H3Request) -> Result<H3Shape> {
        let v = self.vae.config();
        let patch = self.transformer.config().patch_size.map(|p| p as usize);
        let ratio = v.spatial_ratio();
        let multiple = ratio * patch[2];
        let (height, width) = match (req.height, req.width) {
            (Some(h), Some(w)) => (h, w),
            (None, None) => canvas(16.0, 9.0, multiple, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS)?,
            _ => return Err(Error::Request("height and width go together".into())),
        };
        if !height.is_multiple_of(multiple) || !width.is_multiple_of(multiple) || height == 0 || width == 0 {
            return Err(Error::Request(format!("height and width must be multiples of {multiple}, got {height}x{width}")));
        }
        let (clip, chunk) = (v.clip_length, v.clip_length.div_ceil(v.temporal_ratio()));
        let mut frames = req.num_frames.max(1);
        while frames % clip != chunk {
            frames += 1;
        }
        let seconds = frames as f32 / FPS;
        if !(MIN_SECONDS..=MAX_SECONDS).contains(&seconds) {
            return Err(Error::Request(format!("MiniMax-H3 makes {MIN_SECONDS} to {MAX_SECONDS} s clips; {} frames round up to {frames}", req.num_frames)));
        }
        let shape = H3Shape {
            frames,
            height,
            width,
            latent_frames: v.latent_frames(frames),
            latent_height: height / ratio,
            latent_width: width / ratio,
            audio_latents: (frames as f64 / f64::from(FPS) * AUDIO_LATENTS_PER_SECOND).round() as usize,
        };
        if !shape.latent_frames.is_multiple_of(patch[0]) || !shape.latent_height.is_multiple_of(patch[1]) {
            return Err(Error::Request("latent grid is not whole patches".into()));
        }
        Ok(shape)
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let (enc, tok) = self.text.as_ref().ok_or_else(|| Error::Config("no prompt encoder loaded".into()))?;
        let ids = tok.encode(prompt, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids: Vec<u32> = ids.get_ids().to_vec();
        if ids.is_empty() {
            return Err(Error::Request("empty prompt".into()));
        }
        enc.forward(&ids, &[])
    }

    /// Generate a clip and its stereo soundtrack.
    ///
    /// # Errors
    /// A request the model cannot serve, or a backend failure.
    pub fn generate(&self, req: &H3Request) -> Result<H3Output> {
        let t0 = Instant::now();
        let text = self.encode_prompt(&req.prompt)?;
        let encode_ms = t0.elapsed().as_millis() as u64;
        let shape = self.shape(req)?;
        let c = self.vae.config().latent_channels as usize;
        let ac = self.audio_vae.config().latent_channels as usize;
        let nv = c * shape.latent_frames * shape.latent_height * shape.latent_width;
        let na = AUDIO_CHANNELS * ac * shape.audio_latents;
        let noise = gaussian(req.seed, nv + na);
        let mut d = self.generate_from(&text, req, &noise[..nv], &noise[nv..])?;
        d.timings.encode_ms = encode_ms;
        let s = d.shape;
        let mut rgb = Vec::with_capacity(d.frames.len());
        let plane = s.height * s.width;
        let nf = d.frames.len() / (3 * plane);
        for f in 0..nf {
            for i in 0..plane {
                for ch in 0..3 {
                    rgb.push((d.frames[(ch * nf + f) * plane + i] * 255.0).round() as u8);
                }
            }
        }
        let video = Video {
            width: s.width as u32,
            height: s.height as u32,
            frames: nf as u32,
            fps: FPS,
            rgb,
            seed: req.seed,
            evaluations: d.evaluations,
            timings: d.timings,
        };
        let audio = Audio {
            sample_rate: self.audio_vae.config().sampling_rate as u32,
            channels: AUDIO_CHANNELS as u32,
            samples: d.wave,
            seed: req.seed,
            evaluations: d.evaluations,
            timings: d.timings,
        };
        Ok(H3Output { video, audio })
    }

    /// The loop and both decoders from prompt states `[tokens][width]`,
    /// video noise `[C][T][H][W]` and audio noise `[channels][C][T]`.
    pub(crate) fn generate_from(&self, text: &[f32], req: &H3Request, video_noise: &[f32], audio_noise: &[f32]) -> Result<Decoded> {
        let s = self.shape(req)?;
        let tc = self.transformer.config();
        let patch = tc.patch_size.map(|p| p as usize);
        let (c, ac) = (tc.in_channels as usize, tc.audio_in_channels as usize);
        let nt = text.len() / tc.text_dim as usize;
        let lat = [s.latent_frames, s.latent_height, s.latent_width];
        if nt == 0 || nt * tc.text_dim as usize != text.len() || video_noise.len() != c * lat.iter().product::<usize>() || audio_noise.len() != AUDIO_CHANNELS * ac * s.audio_latents {
            return Err(Error::Request("prompt states or starting noise disagree with the request".into()));
        }
        let mut video = patchify(video_noise, c, lat, patch);
        let n_audio_rows = AUDIO_CHANNELS * s.audio_latents;
        let mut audio = vec![0f32; n_audio_rows * ac];
        for ch in 0..AUDIO_CHANNELS {
            for j in 0..s.audio_latents {
                for k in 0..ac {
                    audio[(ch * s.audio_latents + j) * ac + k] = audio_noise[(ch * ac + k) * s.audio_latents + j];
                }
            }
        }
        let n_video_rows = video.len() / tc.patch_dim() as usize;
        let (pos, tags) = layout(nt, &s, patch);
        let sv = sigmas(req.steps, self.video_shift);
        let sa = sigmas(req.steps, self.audio_shift);
        if sv.len() != sa.len() {
            return Err(Error::Request("the two schedules disagree in length".into()));
        }
        let t_loop = Instant::now();
        for i in 0..sv.len() - 1 {
            let (tv, ta) = (1.0 - sv[i], 1.0 - sa[i]);
            let mut ts = vec![tv, ta];
            ts.sort_by(f32::total_cmp);
            ts.dedup();
            let idx = |t: f32| ts.iter().position(|x| x.total_cmp(&t).is_eq()).expect("present");
            let (iv, ia) = (idx(tv), idx(ta));
            let tokens: Vec<H3Token> = pos
                .iter()
                .zip(&tags)
                .enumerate()
                .map(|(r, (&p, &tag))| {
                    let (source, timestep) = if r < nt {
                        (H3Source::Text(r), iv)
                    } else if r < nt + n_audio_rows {
                        (H3Source::Audio(r - nt), ia)
                    } else {
                        (H3Source::Video(r - nt - n_audio_rows), iv)
                    };
                    H3Token { source, tag, timestep, pos: p }
                })
                .collect();
            debug_assert_eq!(tokens.len(), nt + n_audio_rows + n_video_rows);
            let (vv, va) = self.transformer.forward(&H3Input { video: &video, audio: &audio, text, timesteps: &ts, tokens: &tokens })?;
            step(&mut video, &vv, tv, sv[i], sv[i + 1]);
            step(&mut audio, &va, ta, sa[i], sa[i + 1]);
        }
        let denoise_ms = t_loop.elapsed().as_millis() as u64;
        let t_dec = Instant::now();
        let z = unpatchify(&video, c, lat, patch);
        let (mut frames, _) = self.vae.decode(&z, s.latent_frames, s.latent_height, s.latent_width)?;
        to_unit_rgb(&mut frames);
        let mut wave = Vec::new();
        for ch in 0..AUDIO_CHANNELS {
            let mut zc = vec![0f32; ac * s.audio_latents];
            for j in 0..s.audio_latents {
                for k in 0..ac {
                    zc[k * s.audio_latents + j] = audio[(ch * s.audio_latents + j) * ac + k];
                }
            }
            wave.extend(self.audio_vae.decode(&zc, s.audio_latents)?);
        }
        let timings = Timings { encode_ms: 0, denoise_ms, decode_ms: t_dec.elapsed().as_millis() as u64 };
        Ok(Decoded { frames, wave, shape: s, timings, evaluations: (sv.len() - 1) as u32 })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::pipeline::Precision;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn released_geometry() {
        assert_eq!(canvas(16.0, 9.0, 32, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS).unwrap(), (768, 1344));
        let s = sigmas(3, 12.0);
        assert_eq!(s.len(), 3);
        assert!((s[1] - 12.0 * 0.5 / 6.5).abs() < 1e-7 && s[2] == 0.0 && s[0] == 1.0);
        let z: Vec<f32> = (0..2 * 2 * 4 * 6).map(|v| v as f32).collect();
        assert_eq!(unpatchify(&patchify(&z, 2, [2, 4, 6], [1, 2, 2]), 2, [2, 4, 6], [1, 2, 2]), z);
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = PathBuf::from(std::env::var("PRAECISE_MINIMAX_H3_PIPELINE").expect("PRAECISE_MINIMAX_H3_PIPELINE names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let p = MiniMaxH3Pipeline::load_denoiser(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap();
        let u = |k: &str| m[k].as_u64().unwrap() as usize;
        let req = H3Request { num_frames: u("frames"), height: Some(u("height")), width: Some(u("width")), steps: u("steps"), ..H3Request::default() };
        let out = p.generate_from(&bin(&d, "text"), &req, &bin(&d, "video_noise"), &bin(&d, "audio_noise")).unwrap();
        assert_eq!((out.shape.latent_frames, out.shape.audio_latents), (u("latent_frames"), u("audio_latents")));
        // Reference frames are [T][3][H][W].
        let (h, w) = (out.shape.height, out.shape.width);
        let nf = out.frames.len() / (3 * h * w);
        let mut ours = vec![0f32; out.frames.len()];
        for c in 0..3 {
            for f in 0..nf {
                ours[(f * 3 + c) * h * w..(f * 3 + c + 1) * h * w].copy_from_slice(&out.frames[(c * nf + f) * h * w..(c * nf + f + 1) * h * w]);
            }
        }
        assert_close("frames", &ours, &bin(&d, "frames"), min_cos, max_rel);
        assert_close("waveform", &out.wave, &bin(&d, "waveform"), min_cos, max_rel);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_pipeline_f32() {
        run(Precision::F32, 0.999_99, 1e-3);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_pipeline_bf16() {
        run(Precision::Bf16, 0.999, 5e-2);
    }
}
