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
//!
//! Two conditioned tasks share that machinery. First-and-last-frame
//! (`fl2va`) puts each keyframe's latent, noised to `t = 0.999`, in front of
//! the generated video rows at the time of the first or the last latent
//! frame; reference-to-video (`ref2va`, its own transformer weights) puts
//! reference images there, each on its own grid one time step after the
//! previous, and starts the generated audio and video after them. In both,
//! the prompt presentation gives each image a `"<Picture i>: "` label and a
//! vision block (tagged video), conditioning rows stay fixed through the
//! loop at `max(t, 0.999)`, and only the generated rows are stepped and
//! decoded. Conditioning latents are posterior samples under their own seed,
//! rounded to float16.

use std::time::Instant;

use super::audio_vae::H3AudioVae;
use super::vae::{to_unit_rgb, H3VideoVae};
use super::{H3Input, H3Source, H3Token, MiniMaxH3Transformer};
use crate::error::{Error, Result};
use crate::music::Audio;
use crate::pipeline::{CheckpointFiles, LoadOptions, Precision, RgbImage, Timings};
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

/// Seed every conditioning posterior is sampled under, whatever the request.
pub const KEYFRAME_ENCODE_SEED: u64 = 42;
/// The `t` conditioning rows are noised to and held at.
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;
/// Short edge a reference image is resampled to.
pub const REFERENCE_SHORT_EDGE: f64 = 2048.0;
/// Most reference images one request may carry.
pub const MAX_REFERENCE_IMAGES: usize = 9;
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
    /// What the clip is conditioned on besides the prompt.
    pub condition: H3Condition,
}

impl Default for H3Request {
    fn default() -> Self {
        Self { prompt: String::new(), num_frames: 240, width: None, height: None, steps: 50, seed: 0, condition: H3Condition::Text }
    }
}

/// The images a request is conditioned on.
#[derive(Debug, Clone, Default)]
pub enum H3Condition {
    /// The prompt alone (`t2va`).
    #[default]
    Text,
    /// Keyframes the clip starts and/or ends on (`fl2va`). Without a canvas
    /// size the first given keyframe's aspect ratio sets it; that keyframe is
    /// stretched onto the canvas and a second one cover-cropped.
    Keyframes { first: Option<RgbImage>, last: Option<RgbImage> },
    /// Reference images (`ref2va`), each resampled to a 2048 short edge.
    References(Vec<RgbImage>),
}

/// Which transformer weights a pipeline loads: `transformer/` serves the
/// prompt-only and keyframe tasks, `transformer_ref/` the reference task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum H3Partition {
    #[default]
    Keyframe,
    Reference,
}

impl H3Partition {
    /// The checkpoint directory of these weights.
    #[must_use]
    pub fn dir(self) -> &'static str {
        match self {
            Self::Keyframe => "transformer",
            Self::Reference => "transformer_ref",
        }
    }
}

/// Where one conditioning image sits in the packed sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Anchor {
    First,
    Last,
    Reference,
}

/// Conditioning images on their final sizes, in packed order, and the canvas.
#[derive(Debug, Clone)]
pub(crate) struct Prepared {
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) images: Vec<(Anchor, RgbImage)>,
}

/// A generated clip with its stereo soundtrack.
#[derive(Debug, Clone)]
pub struct H3Output {
    pub video: Video,
    pub audio: Audio,
}

impl H3Output {
    /// The clip and its soundtrack as one MP4 file.
    ///
    /// # Errors
    /// As [`crate::Video::mp4`].
    pub fn mp4(&self) -> crate::Result<Vec<u8>> {
        self.video.mp4(Some(&self.audio))
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

/// Rotary positions and tags of `[prompt | conditions | audio | video]`.
/// `conds` gives each conditioning block's time and latent `(h, w)`;
/// generated audio and video start at time `origin`.
pub(crate) fn layout(text_tags: &[usize], s: &H3Shape, patch: [usize; 3], conds: &[(f64, usize, usize)], origin: f64) -> (Vec<[f32; 3]>, Vec<usize>) {
    let [pt, ph, pw] = patch;
    let grid = |h: usize, w: usize| {
        let a = ((h * w) as f64).sqrt();
        (spatial_grid(h, ph, a), spatial_grid(w, pw, a))
    };
    let (hg, wg) = grid(s.latent_height, s.latent_width);
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut tags = Vec::new();
    for (i, &tag) in text_tags.iter().enumerate() {
        pos.push([i as f32, 0.0, 0.0]);
        tags.push(tag);
    }
    for &(t, h, w) in conds {
        let (ch, cw) = grid(h, w);
        for y in &ch {
            for x in &cw {
                pos.push([t as f32, *y as f32, *x as f32]);
                tags.push(VIDEO_TAG);
            }
        }
    }
    for ch in 0..AUDIO_CHANNELS {
        let col = if ch == 0 { wg[0] } else { wg[wg.len() - 1] };
        for j in 0..s.audio_latents {
            pos.push([(origin + j as f64) as f32, 0.0, col as f32]);
            tags.push(AUDIO_TAG);
        }
    }
    let mut t = origin;
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

/// Rotary time of a keyframe anchored at the last latent frame.
fn last_anchor_time(text: usize, latent_frames: usize) -> f64 {
    let spans: f64 = (0..latent_frames).map(|f| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[f % ROPE_FRAMES_PER_LATENT.len()]).sum();
    text as f64 + spans - ROPE_FRAME_RESCALE
}

/// Token ids and per-token modality tags of a prompt presentation: per image
/// a `"<Picture i>: "` label (text) and a vision block of `image_tokens[i]`
/// pads between start and end markers (video), then the prompt verbatim.
///
/// # Errors
/// A tokenizer without the vision markers.
pub(crate) fn presentation(tok: &tokenizers::Tokenizer, prompt: &str, image_tokens: &[usize]) -> Result<(Vec<u32>, Vec<usize>)> {
    let encode = |text: &str| -> Result<Vec<u32>> { Ok(tok.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?.get_ids().to_vec()) };
    let id = |t: &str| tok.token_to_id(t).ok_or_else(|| Error::Tokenizer(format!("no {t} token")));
    let (start, pad, end) = (id("<|vision_start|>")?, id("<|image_pad|>")?, id("<|vision_end|>")?);
    let (mut ids, mut tags) = (Vec::new(), Vec::new());
    for (i, &n) in image_tokens.iter().enumerate() {
        let label = encode(&format!("<Picture {}>: ", i + 1))?;
        tags.extend(std::iter::repeat_n(TEXT_TAG, label.len()));
        ids.extend(label);
        ids.push(start);
        ids.extend(std::iter::repeat_n(pad, n));
        ids.push(end);
        tags.extend(std::iter::repeat_n(VIDEO_TAG, n + 2));
    }
    let p = encode(prompt)?;
    tags.extend(std::iter::repeat_n(TEXT_TAG, p.len()));
    ids.extend(p);
    Ok((ids, tags))
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
    partition: H3Partition,
    vae: H3VideoVae,
    audio_vae: H3AudioVae,
    text: Option<(Qwen3VlEncoder, tokenizers::Tokenizer)>,
    /// Pixel counts the prompt encoder's image processor keeps unchanged.
    vision_pixels: (usize, usize),
    video_shift: f32,
    audio_shift: f32,
}

impl std::fmt::Debug for MiniMaxH3Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniMaxH3Pipeline").field("device", &self.transformer.device()).field("partition", &self.partition).finish_non_exhaustive()
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

/// The image processor's pixel bounds (`processor/preprocessor_config.json`,
/// else the released ones).
fn vision_pixels(files: &CheckpointFiles) -> Result<(usize, usize)> {
    if !files.root.join("processor/preprocessor_config.json").exists() {
        return Ok((65_536, 16_777_216));
    }
    let v = files.json("processor/preprocessor_config.json")?;
    let get = |k: &str| v["size"][k].as_u64().map(|x| x as usize).ok_or_else(|| Error::Config(format!("processor: no size.{k}")));
    Ok((get("shortest_edge")?, get("longest_edge")?))
}

/// Python's `round` (half to even) of a non-negative value.
fn round_even(v: f64) -> usize {
    v.round_ties_even() as usize
}

impl MiniMaxH3Pipeline {
    /// Load a diffusers-layout checkpoint: the partition's transformer
    /// (`transformer/` or `transformer_ref/`), `vae/`, `audio_vae/`,
    /// `text_encoder/` (only its first 50 layers), `tokenizer/tokenizer.json`,
    /// `scheduler/`, `audio_scheduler/` and, when present,
    /// `processor/preprocessor_config.json`.
    ///
    /// # Errors
    /// A missing or malformed part, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions, partition: H3Partition) -> Result<Self> {
        let mut p = Self::load_denoiser(files, opts, partition)?;
        let enc = Qwen3VlEncoder::load_layers(files, "text_encoder", opts, Some(TEXT_ENCODER_LAYER))?;
        let tok_path = files.root.join("tokenizer/tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        if enc.config().text_config.hidden_size != p.transformer.config().text_dim {
            return Err(Error::Config("prompt encoder width differs from the transformer's text width".into()));
        }
        if tok.token_to_id("<|image_pad|>") != Some(enc.config().image_token_id) {
            return Err(Error::Config("tokenizer and prompt encoder disagree on the image token".into()));
        }
        p.vision_pixels = vision_pixels(files)?;
        p.text = Some((enc, tok));
        Ok(p)
    }

    /// Everything but the prompt encoder.
    pub(crate) fn load_denoiser(files: &CheckpointFiles, opts: LoadOptions, partition: H3Partition) -> Result<Self> {
        let transformer = MiniMaxH3Transformer::load_dir(files, partition.dir(), opts)?;
        // 8-bit weights are for the transformer and the prompt encoder; the
        // autoencoders keep bfloat16.
        let ae = if opts.precision == Precision::Q8_0 { LoadOptions { precision: Precision::Bf16, ..opts } } else { opts };
        let vae = H3VideoVae::load(files, ae)?;
        let audio_vae = H3AudioVae::load(files, ae)?;
        let tc = transformer.config();
        if vae.config().latent_channels != tc.in_channels || audio_vae.config().latent_channels != tc.audio_in_channels {
            return Err(Error::Config("autoencoder latent widths differ from the transformer's".into()));
        }
        Ok(Self {
            transformer,
            partition,
            vae,
            audio_vae,
            text: None,
            vision_pixels: (65_536, 16_777_216),
            video_shift: shift(files, "scheduler")?,
            audio_shift: shift(files, "audio_scheduler")?,
        })
    }

    /// The device the transformer runs on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.transformer.device()
    }

    /// The transformer weights this pipeline serves.
    #[must_use]
    pub fn partition(&self) -> H3Partition {
        self.partition
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let text = self.text.as_ref().map_or(0, |(e, _)| e.resident_bytes());
        self.transformer.resident_bytes() + self.vae.resident_bytes() + self.audio_vae.resident_bytes() + text
    }

    /// Canvas sides are multiples of this.
    fn canvas_multiple(&self) -> usize {
        self.vae.config().spatial_ratio() * self.transformer.config().patch_size[2] as usize
    }

    /// Resolve a request's geometry as the reference does.
    ///
    /// # Errors
    /// A canvas off the patch grid, or a clip outside 5 to 15 seconds.
    pub fn shape(&self, req: &H3Request) -> Result<H3Shape> {
        let (h, w) = self.canvas(req)?;
        self.shape_at(req, h, w)
    }

    /// The canvas `(height, width)`: the request's, else the first keyframe's
    /// aspect ratio, else 16:9.
    fn canvas(&self, req: &H3Request) -> Result<(usize, usize)> {
        let multiple = self.canvas_multiple();
        match (req.height, req.width, &req.condition) {
            (Some(h), Some(w), _) => Ok((h, w)),
            (None, None, H3Condition::Keyframes { first, last }) => {
                let k = first.as_ref().or(last.as_ref()).ok_or_else(|| Error::Request("keyframe conditioning needs a keyframe".into()))?;
                canvas(f64::from(k.width), f64::from(k.height), multiple, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS)
            }
            (None, None, _) => canvas(16.0, 9.0, multiple, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS),
            _ => Err(Error::Request("height and width go together".into())),
        }
    }

    fn shape_at(&self, req: &H3Request, height: usize, width: usize) -> Result<H3Shape> {
        let v = self.vae.config();
        let patch = self.transformer.config().patch_size.map(|p| p as usize);
        let ratio = v.spatial_ratio();
        let multiple = self.canvas_multiple();
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

    /// The canvas and the conditioning images on their final sizes, after
    /// checking the request against the loaded partition.
    pub(crate) fn prepare(&self, req: &H3Request) -> Result<Prepared> {
        let (height, width) = self.canvas(req)?;
        let (w32, h32) = (width as u32, height as u32);
        let images = match (&req.condition, self.partition) {
            (H3Condition::Text, H3Partition::Keyframe) => Vec::new(),
            (H3Condition::Keyframes { first, last }, H3Partition::Keyframe) => {
                let given: Vec<(Anchor, &RgbImage)> = [(Anchor::First, first), (Anchor::Last, last)].into_iter().filter_map(|(a, k)| k.as_ref().map(|k| (a, k))).collect();
                if given.is_empty() {
                    return Err(Error::Request("keyframe conditioning needs a keyframe".into()));
                }
                given
                    .into_iter()
                    .enumerate()
                    .map(|(i, (a, k))| {
                        let k = if (k.width, k.height) == (w32, h32) {
                            k.clone()
                        } else if i == 0 {
                            k.lanczos(w32, h32)
                        } else {
                            // Cover the canvas, then crop the centre.
                            let s = (width as f64 / f64::from(k.width)).max(height as f64 / f64::from(k.height));
                            let rw = width.max(round_even(f64::from(k.width) * s));
                            let rh = height.max(round_even(f64::from(k.height) * s));
                            let r = k.lanczos(rw as u32, rh as u32);
                            let (left, top) = ((rw - width) / 2, (rh - height) / 2);
                            let mut rgb = Vec::with_capacity(3 * width * height);
                            for y in top..top + height {
                                rgb.extend_from_slice(&r.rgb[(y * rw + left) * 3..(y * rw + left + width) * 3]);
                            }
                            RgbImage { width: w32, height: h32, rgb }
                        };
                        (a, k)
                    })
                    .collect()
            }
            (H3Condition::References(refs), H3Partition::Reference) => {
                if refs.is_empty() || refs.len() > MAX_REFERENCE_IMAGES {
                    return Err(Error::Request(format!("reference conditioning takes 1 to {MAX_REFERENCE_IMAGES} images, got {}", refs.len())));
                }
                let m = self.canvas_multiple();
                refs.iter()
                    .map(|r| {
                        let (w, h) = (f64::from(r.width), f64::from(r.height));
                        if r.width == 0 || r.height == 0 || w > 4.0 * h || h > 4.0 * w {
                            return Err(Error::Request(format!("a reference image must be within 1:4 and 4:1, got {}x{}", r.width, r.height)));
                        }
                        let s = REFERENCE_SHORT_EDGE / w.min(h);
                        let th = m.max(round_even(h * s / m as f64) * m) as u32;
                        let tw = m.max(round_even(w * s / m as f64) * m) as u32;
                        let r = if (r.width, r.height) == (tw, th) { r.clone() } else { r.lanczos(tw, th) };
                        Ok((Anchor::Reference, r))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            (H3Condition::References(_), H3Partition::Keyframe) => return Err(Error::Request("reference images need the reference-conditioned weights".into())),
            (_, H3Partition::Reference) => return Err(Error::Request("the reference-conditioned weights need at least one reference image".into())),
        };
        Ok(Prepared { height, width, images })
    }

    /// Prompt states `[tokens][width]` and their modality tags for a prompt
    /// presented after its conditioning images.
    fn encode_prompt(&self, prompt: &str, images: &[(Anchor, RgbImage)]) -> Result<(Vec<f32>, Vec<usize>)> {
        let (enc, tok) = self.text.as_ref().ok_or_else(|| Error::Config("no prompt encoder loaded".into()))?;
        let unit = enc.config().image_unit();
        let mut vl = Vec::with_capacity(images.len());
        let mut counts = Vec::with_capacity(images.len());
        for (_, im) in images {
            let (h, w) = (im.height as usize, im.width as usize);
            let (lo, hi) = self.vision_pixels;
            if !(lo..=hi).contains(&(h * w)) {
                return Err(Error::Request(format!("a {w}x{h} image is outside the prompt encoder's {lo} to {hi} pixels")));
            }
            let plane = h * w;
            let mut px = vec![0f32; 3 * plane];
            for (i, rgb) in im.rgb.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    px[c * plane + i] = f32::from(rgb[c]) / 255.0;
                }
            }
            vl.push(enc.image(&px, (h, w))?);
            counts.push((h / unit) * (w / unit));
        }
        if prompt.trim().is_empty() {
            return Err(Error::Request("empty prompt".into()));
        }
        let (ids, tags) = presentation(tok, prompt, &counts)?;
        Ok((enc.forward(&ids, &vl)?, tags))
    }

    /// Generate a clip and its stereo soundtrack.
    ///
    /// # Errors
    /// A request the model cannot serve, or a backend failure.
    pub fn generate(&self, req: &H3Request) -> Result<H3Output> {
        let t0 = Instant::now();
        let prep = self.prepare(req)?;
        let (text, tags) = self.encode_prompt(&req.prompt, &prep.images)?;
        let encode_ms = t0.elapsed().as_millis() as u64;
        let shape = self.shape_at(req, prep.height, prep.width)?;
        let c = self.vae.config().latent_channels as usize;
        let ac = self.audio_vae.config().latent_channels as usize;
        let ratio = self.vae.config().spatial_ratio();
        let sizes: Vec<usize> = prep.images.iter().map(|(_, im)| c * (im.height as usize / ratio) * (im.width as usize / ratio)).collect();
        let nc: usize = sizes.iter().sum();
        let nv = c * shape.latent_frames * shape.latent_height * shape.latent_width;
        let na = AUDIO_CHANNELS * ac * shape.audio_latents;
        // Conditioning noise first, then video, then audio.
        let noise = gaussian(req.seed, nc + nv + na);
        let mut at = 0;
        let cond_noise: Vec<Vec<f32>> = sizes
            .iter()
            .map(|&n| {
                at += n;
                noise[at - n..at].to_vec()
            })
            .collect();
        let eps: Vec<Vec<f32>> = sizes.iter().map(|&n| gaussian(KEYFRAME_ENCODE_SEED, n)).collect();
        let mut d = self.generate_from(&text, &tags, &prep, &eps, &cond_noise, req, &noise[nc..nc + nv], &noise[nc + nv..])?;
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

    /// The loop and both decoders from prompt states `[tokens][width]` and
    /// their tags, the prepared conditioning images with their posterior
    /// noise (`eps`) and noise-augmentation noise (`cond_noise`), video noise
    /// `[C][T][H][W]` and audio noise `[channels][C][T]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_from(
        &self,
        text: &[f32],
        text_tags: &[usize],
        prep: &Prepared,
        eps: &[Vec<f32>],
        cond_noise: &[Vec<f32>],
        req: &H3Request,
        video_noise: &[f32],
        audio_noise: &[f32],
    ) -> Result<Decoded> {
        let s = self.shape_at(req, prep.height, prep.width)?;
        let tc = self.transformer.config();
        let patch = tc.patch_size.map(|p| p as usize);
        let (c, ac) = (tc.in_channels as usize, tc.audio_in_channels as usize);
        let pd = tc.patch_dim() as usize;
        let nt = text.len() / tc.text_dim as usize;
        let lat = [s.latent_frames, s.latent_height, s.latent_width];
        if nt == 0
            || nt * tc.text_dim as usize != text.len()
            || text_tags.len() != nt
            || eps.len() != prep.images.len()
            || cond_noise.len() != prep.images.len()
            || video_noise.len() != c * lat.iter().product::<usize>()
            || audio_noise.len() != AUDIO_CHANNELS * ac * s.audio_latents
        {
            return Err(Error::Request("prompt states, conditioning or starting noise disagree with the request".into()));
        }
        // Conditioning rows: encoded, noised to the augmentation level, packed.
        let t_aug = KEYFRAME_NOISE_AUG;
        let mut video = Vec::new();
        let mut conds = Vec::with_capacity(prep.images.len());
        let n_refs = prep.images.iter().filter(|(a, _)| *a == Anchor::Reference).count();
        let mut ref_index = 0;
        for (((anchor, im), e), n) in prep.images.iter().zip(eps).zip(cond_noise) {
            let (z, dims) = self.vae.encode_condition(im, e)?;
            if n.len() != z.len() {
                return Err(Error::Request("conditioning noise disagrees with its latent".into()));
            }
            let z: Vec<f32> = z.iter().zip(n).map(|(x, n)| t_aug * x + (1.0 - t_aug) * n).collect();
            video.extend(patchify(&z, c, dims, patch));
            let t = match anchor {
                Anchor::First => nt as f64,
                Anchor::Last => last_anchor_time(nt, s.latent_frames),
                Anchor::Reference => {
                    ref_index += 1;
                    (nt + ref_index - 1) as f64
                }
            };
            conds.push((t, dims[1], dims[2]));
        }
        let n_cond_rows = video.len() / pd;
        video.extend(patchify(video_noise, c, lat, patch));
        let n_audio_rows = AUDIO_CHANNELS * s.audio_latents;
        let mut audio = vec![0f32; n_audio_rows * ac];
        for ch in 0..AUDIO_CHANNELS {
            for j in 0..s.audio_latents {
                for k in 0..ac {
                    audio[(ch * s.audio_latents + j) * ac + k] = audio_noise[(ch * ac + k) * s.audio_latents + j];
                }
            }
        }
        let n_video_rows = video.len() / pd;
        let (pos, tags) = layout(text_tags, &s, patch, &conds, (nt + n_refs) as f64);
        let sv = sigmas(req.steps, self.video_shift);
        let sa = sigmas(req.steps, self.audio_shift);
        if sv.len() != sa.len() {
            return Err(Error::Request("the two schedules disagree in length".into()));
        }
        let t_loop = Instant::now();
        for i in 0..sv.len() - 1 {
            let (tv, ta) = (1.0 - sv[i], 1.0 - sa[i]);
            let tcond = tv.max(t_aug);
            let mut ts = vec![tv, ta];
            if n_cond_rows > 0 {
                ts.push(tcond);
            }
            ts.sort_by(f32::total_cmp);
            ts.dedup();
            let idx = |t: f32| ts.iter().position(|x| x.total_cmp(&t).is_eq()).expect("present");
            let (iv, ia) = (idx(tv), idx(ta));
            let ic = if n_cond_rows > 0 { idx(tcond) } else { iv };
            let tokens: Vec<H3Token> = pos
                .iter()
                .zip(&tags)
                .enumerate()
                .map(|(r, (&p, &tag))| {
                    let (source, timestep) = if r < nt {
                        (H3Source::Text(r), iv)
                    } else if r < nt + n_cond_rows {
                        (H3Source::Video(r - nt), ic)
                    } else if r < nt + n_cond_rows + n_audio_rows {
                        (H3Source::Audio(r - nt - n_cond_rows), ia)
                    } else {
                        (H3Source::Video(r - nt - n_audio_rows), iv)
                    };
                    H3Token { source, tag, timestep, pos: p }
                })
                .collect();
            debug_assert_eq!(tokens.len(), nt + n_audio_rows + n_video_rows);
            let (vv, va) = self.transformer.forward(&H3Input { video: &video, audio: &audio, text, timesteps: &ts, tokens: &tokens })?;
            let k = n_cond_rows * pd;
            step(&mut video[k..], &vv[k..], tv, sv[i], sv[i + 1]);
            step(&mut audio, &va, ta, sa[i], sa[i + 1]);
        }
        let denoise_ms = t_loop.elapsed().as_millis() as u64;
        let t_dec = Instant::now();
        let z = unpatchify(&video[n_cond_rows * pd..], c, lat, patch);
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

    fn opts(precision: Precision) -> LoadOptions {
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        LoadOptions { precision, cpu_threads: threads, device: None }
    }

    /// Reference frames are `[T][3][H][W]`; ours `[3][T][H][W]`.
    fn check_outputs(out: &Decoded, d: &std::path::Path, min_cos: f64, max_rel: f64) {
        let (h, w) = (out.shape.height, out.shape.width);
        let nf = out.frames.len() / (3 * h * w);
        let mut ours = vec![0f32; out.frames.len()];
        for c in 0..3 {
            for f in 0..nf {
                ours[(f * 3 + c) * h * w..(f * 3 + c + 1) * h * w].copy_from_slice(&out.frames[(c * nf + f) * h * w..(c * nf + f + 1) * h * w]);
            }
        }
        assert_close("frames", &ours, &bin(d, "frames"), min_cos, max_rel);
        assert_close("waveform", &out.wave, &bin(d, "waveform"), min_cos, max_rel);
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = PathBuf::from(std::env::var("PRAECISE_MINIMAX_H3_PIPELINE").expect("PRAECISE_MINIMAX_H3_PIPELINE names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let p = MiniMaxH3Pipeline::load_denoiser(&CheckpointFiles::new(d.join("checkpoint")), opts(precision), H3Partition::Keyframe).unwrap();
        let u = |k: &str| m[k].as_u64().unwrap() as usize;
        let req = H3Request { num_frames: u("frames"), height: Some(u("height")), width: Some(u("width")), steps: u("steps"), ..H3Request::default() };
        let text = bin(&d, "text");
        let tags = vec![TEXT_TAG; text.len() / p.transformer.config().text_dim as usize];
        let prep = Prepared { height: u("height"), width: u("width"), images: Vec::new() };
        let out = p.generate_from(&text, &tags, &prep, &[], &[], &req, &bin(&d, "video_noise"), &bin(&d, "audio_noise")).unwrap();
        assert_eq!((out.shape.latent_frames, out.shape.audio_latents), (u("latent_frames"), u("audio_latents")));
        check_outputs(&out, &d, min_cos, max_rel);
    }

    fn condition_dir() -> PathBuf {
        PathBuf::from(std::env::var("PRAECISE_MINIMAX_H3_CONDITION").expect("PRAECISE_MINIMAX_H3_CONDITION names the fixture dir"))
    }

    fn rgb(d: &std::path::Path, name: &str, size: &Value) -> RgbImage {
        let (w, h) = (size[0].as_u64().unwrap() as u32, size[1].as_u64().unwrap() as u32);
        let rgb = std::fs::read(d.join(format!("{name}.rgb"))).unwrap();
        assert_eq!(rgb.len(), 3 * (w * h) as usize);
        RgbImage { width: w, height: h, rgb }
    }

    fn run_condition(task: &str, partition: H3Partition, precision: Precision, min_cos: f64, max_rel: f64) {
        let root = condition_dir();
        let d = root.join(task);
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let p = MiniMaxH3Pipeline::load_denoiser(&CheckpointFiles::new(root.join("checkpoint")), opts(precision), partition).unwrap();
        let u = |k: &str| m[k].as_u64().unwrap() as usize;
        let n = m["images"].as_array().unwrap().len();
        let images: Vec<(Anchor, RgbImage)> = (0..n)
            .map(|i| {
                let a = match m["anchors"][i].as_str().unwrap() {
                    "first" => Anchor::First,
                    "last" => Anchor::Last,
                    _ => Anchor::Reference,
                };
                (a, rgb(&d, &format!("image_{i}"), &m["images"][i]))
            })
            .collect();
        let eps: Vec<Vec<f32>> = (0..n).map(|i| bin(&d, &format!("eps_{i}"))).collect();
        let noise: Vec<Vec<f32>> = (0..n).map(|i| bin(&d, &format!("cond_noise_{i}"))).collect();
        // The conditioning latents alone first.
        let mut lat = Vec::new();
        for ((_, im), e) in images.iter().zip(&eps) {
            lat.extend(p.vae.encode_condition(im, e).unwrap().0);
        }
        assert_close("conditions", &lat, &bin(&d, "conditions"), min_cos, max_rel);
        let tags: Vec<usize> = m["tags"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as usize).collect();
        let req = H3Request { num_frames: u("frames"), height: Some(u("height")), width: Some(u("width")), steps: u("steps"), ..H3Request::default() };
        let prep = Prepared { height: u("height"), width: u("width"), images };
        let out = p.generate_from(&bin(&d, "text"), &tags, &prep, &eps, &noise, &req, &bin(&d, "video_noise"), &bin(&d, "audio_noise")).unwrap();
        check_outputs(&out, &d, min_cos, max_rel);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_fl2va_f32() {
        run_condition("fl2va", H3Partition::Keyframe, Precision::F32, 0.999_99, 1e-3);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_fl2va_bf16() {
        run_condition("fl2va", H3Partition::Keyframe, Precision::Bf16, 0.999, 5e-2);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_ref2va_f32() {
        run_condition("ref2va", H3Partition::Reference, Precision::F32, 0.999_99, 1e-3);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_ref2va_bf16() {
        run_condition("ref2va", H3Partition::Reference, Precision::Bf16, 0.999, 5e-2);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_presentation() {
        let d = condition_dir().join("present");
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let tok = tokenizers::Tokenizer::from_file(d.join("tokenizer.json")).unwrap();
        // Released image processor: 16 px patches merged 2x2.
        let counts: Vec<usize> = m["sizes"].as_array().unwrap().iter().map(|s| (s[0].as_u64().unwrap() as usize / 32) * (s[1].as_u64().unwrap() as usize / 32)).collect();
        let (ids, tags) = presentation(&tok, m["prompt"].as_str().unwrap(), &counts).unwrap();
        for task in ["fl2va", "ref2va"] {
            let want: Vec<u32> = m[task]["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let want_tags: Vec<usize> = m[task]["tags"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(ids, want, "{task} ids");
            assert_eq!(tags, want_tags, "{task} tags");
        }
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_lanczos() {
        let d = condition_dir().join("lanczos");
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let src = rgb(&d, "src", &m["src"]);
        for (i, size) in m["outs"].as_array().unwrap().iter().enumerate() {
            let want = rgb(&d, &format!("out_{i}"), size);
            let got = src.lanczos(want.width, want.height);
            let off = got.rgb.iter().zip(&want.rgb).filter(|(a, b)| a != b).count();
            assert_eq!(off, 0, "{}x{}: {off} bytes differ", want.width, want.height);
        }
    }

    #[test]
    fn conditioning_layout() {
        // A last keyframe sits one span-unit before the end of the frame times.
        assert!((last_anchor_time(5, 2) - (5.0 + 5.0 / 3.0 * 5.0 - 5.0 / 3.0)).abs() < 1e-12);
        let s = H3Shape { frames: 124, height: 16, width: 24, latent_frames: 2, latent_height: 4, latent_width: 6, audio_latents: 3 };
        let (pos, tags) = layout(&[1, 0, 1], &s, [1, 2, 2], &[(3.0, 4, 4)], 4.0);
        assert_eq!(pos.len(), 3 + 4 + 2 * 3 + 2 * 6);
        assert_eq!(&tags[..3], &[1, 0, 1]);
        assert!(tags[3..7].iter().all(|&t| t == VIDEO_TAG));
        assert_eq!(pos[7][0], 4.0);
        assert_eq!(pos[3 + 4 + 6][0], 4.0);
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

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_pipeline_q8_0() {
        run(Precision::Q8_0, 0.999, 5e-2);
    }
}
