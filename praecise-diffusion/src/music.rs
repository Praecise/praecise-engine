//! ACE-Step 1.5 text-to-music: prompt and lyric encoding, the flow-matching
//! loop over audio latents and the waveform decode, on one backend with every
//! weight resident once.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::acestep::{self, ConditionConfig, DitConfig};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Weights};
use crate::oobleck::{self, OobleckConfig};
use crate::pipeline::{CheckpointFiles, LoadOptions, Precision, Timings, parse};
use crate::qwen3::{self, Qwen3Config};
use crate::safetensors::SafeTensors;
use crate::schedule;
use llama_cpp_sys_2 as sys;

/// Maximum caption length in tokens, template included.
pub const MAX_TEXT_TOKENS: usize = 256;
/// Maximum lyric length in tokens, template included.
pub const MAX_LYRIC_TOKENS: usize = 2048;
/// Longest generation served, in seconds.
pub const MAX_DURATION_SECS: f32 = 600.0;
/// Seconds of reference audio the timbre encoder reads.
const TIMBRE_SECS: f64 = 30.0;
/// Instruction for plain text-to-music generation.
const INSTRUCTION: &str = "Fill the audio semantic mask based on the given conditions:";
/// Momentum of the guidance-difference running average.
const GUIDANCE_MOMENTUM: f32 = -0.75;
/// Per-channel norm the guidance difference is clipped to.
const GUIDANCE_NORM_LIMIT: f32 = 2.5;
/// Peak level of the output, -1 dBFS.
const OUTPUT_PEAK_DB: f32 = -1.0;
/// Latent frames decoded at once, and the context decoded on each side of a
/// tile and discarded.
const DECODE_TILE: usize = 512;
const DECODE_OVERLAP: usize = 64;

/// One text-to-music request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicRequest {
    /// Description of the music: genre, instruments, mood.
    pub prompt: String,
    /// Lyrics, with optional `[verse]`-style section tags; empty for
    /// instrumental music.
    #[serde(default)]
    pub lyrics: String,
    /// Language code of the lyrics.
    #[serde(default = "default_language")]
    pub language: String,
    /// Length in seconds.
    pub duration_secs: f32,
    /// Denoising steps; the checkpoint's default when absent.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Classifier-free guidance scale; ignored by guidance-distilled
    /// checkpoints.
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    /// Timestep shift of the schedule.
    #[serde(default)]
    pub shift: Option<f32>,
    /// Seed of the starting noise.
    #[serde(default)]
    pub seed: u64,
    /// Tempo in beats per minute.
    #[serde(default)]
    pub bpm: Option<u32>,
    /// Key and scale, such as `"A minor"`.
    #[serde(default)]
    pub keyscale: Option<String>,
    /// Beats per bar, such as `"4"`.
    #[serde(default)]
    pub timesignature: Option<String>,
}

fn default_language() -> String {
    "en".into()
}

/// Generated audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audio {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channels.
    pub channels: u32,
    /// Samples, channel-major: all of channel 0, then channel 1.
    pub samples: Vec<f32>,
    /// Seed used.
    pub seed: u64,
    /// Transformer evaluations run (steps, doubled under guidance).
    pub evaluations: u32,
    /// Per-stage timings.
    pub timings: Timings,
}

impl Audio {
    /// Samples per channel.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    /// The audio as a 16-bit PCM WAV file.
    #[must_use]
    pub fn wav_pcm16(&self) -> Vec<u8> {
        let ch = self.channels as usize;
        let n = self.frames();
        let data = (n * ch * 2) as u32;
        let mut out = Vec::with_capacity(44 + data as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(ch as u16).to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&(self.sample_rate * ch as u32 * 2).to_le_bytes());
        out.extend_from_slice(&((ch * 2) as u16).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data.to_le_bytes());
        for i in 0..n {
            for c in 0..ch {
                let v = (self.samples[c * n + i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

/// A loaded ACE-Step 1.5 pipeline.
pub struct AceStep {
    backend: Backend,
    dit_cfg: DitConfig,
    dit: Weights,
    dit_conv: Weights,
    cond_cfg: ConditionConfig,
    cond: Weights,
    te_cfg: Qwen3Config,
    te_theta: f32,
    te: Weights,
    vae_cfg: OobleckConfig,
    vae: Weights,
    /// Encoded silence `[frames][acoustic]`: the source latents of plain
    /// generation and the timbre reference when none is given.
    silence: Vec<f32>,
    /// The learned null conditioning token `[width]`.
    null_token: Vec<f32>,
    tokenizer: tokenizers::Tokenizer,
    /// Float32 attention throughout, for the full-precision format: the fused
    /// kernel's half-precision keys and values cost about 5% of the largest
    /// velocity component on the released weights.
    exact: bool,
}

impl std::fmt::Debug for AceStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AceStep")
            .field("backend", &self.backend)
            .field("turbo", &self.dit_cfg.turbo())
            .field("resident_bytes", &self.resident_bytes())
            .finish_non_exhaustive()
    }
}

/// The schedule of noise levels, `1` down to just above `0`, shifted towards
/// the noisy end: `shift t / (1 + (shift - 1) t)`, in single precision as the
/// reference computes it.
#[must_use]
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps + 1;
    let step = -1.0f32 / steps as f32;
    let half = n / 2;
    (0..steps)
        .map(|i| {
            let t = if i < half { 1.0 + step * i as f32 } else { -step * (n - 1 - i) as f32 };
            if shift == 1.0 { t } else { shift * t / (1.0 + (shift - 1.0) * t) }
        })
        .collect()
}

/// Momentum-averaged, norm-clipped guidance with the component parallel to
/// the conditional prediction removed. Tensors are `[frames][channels]`; norms
/// run over frames, per channel.
fn guided(cond: &[f32], uncond: &[f32], momentum: &mut Vec<f32>, scale: f32, channels: usize) -> Vec<f32> {
    let frames = cond.len() / channels;
    if momentum.is_empty() {
        momentum.resize(cond.len(), 0.0);
    }
    for ((m, c), u) in momentum.iter_mut().zip(cond).zip(uncond) {
        *m = (c - u) + GUIDANCE_MOMENTUM * *m;
    }
    let mut diff = momentum.clone();
    for ch in 0..channels {
        let norm = (0..frames).map(|t| diff[t * channels + ch] * diff[t * channels + ch]).sum::<f32>().sqrt();
        let k = (GUIDANCE_NORM_LIMIT / norm).min(1.0);
        for t in 0..frames {
            diff[t * channels + ch] *= k;
        }
    }
    let mut out = cond.to_vec();
    for ch in 0..channels {
        let idx = |t: usize| t * channels + ch;
        let cn = (0..frames).map(|t| f64::from(cond[idx(t)]).powi(2)).sum::<f64>().sqrt().max(1e-12);
        let dot = (0..frames).map(|t| f64::from(diff[idx(t)]) * f64::from(cond[idx(t)]) / cn).sum::<f64>();
        for t in 0..frames {
            let par = dot * f64::from(cond[idx(t)]) / cn;
            let orth = (f64::from(diff[idx(t)]) - par) as f32;
            out[idx(t)] += scale * orth;
        }
    }
    out
}

impl AceStep {
    /// Load a checkpoint in the diffusers layout. Refuses to run on the CPU
    /// when the host has GPU hardware this build cannot drive.
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let index = files.json("model_index.json")?;
        let class = index.get("_class_name").and_then(Value::as_str).unwrap_or_default();
        if class != "AceStepPipeline" {
            return Err(Error::Config(format!("pipeline class {class:?} is not AceStepPipeline")));
        }
        let sched = files.json("scheduler/scheduler_config.json")?;
        let shift = sched.get("shift").and_then(Value::as_f64).unwrap_or(1.0);
        if shift != 1.0 || sched.get("use_dynamic_shifting").and_then(Value::as_bool) == Some(true) {
            return Err(Error::Config("expected an unshifted flow-match schedule (the pipeline shifts)".into()));
        }
        let dit_cfg: DitConfig = parse(files.json("transformer/config.json")?, "transformer config")?;
        dit_cfg.validate()?;
        let cond_cfg: ConditionConfig = parse(files.json("condition_encoder/config.json")?, "condition encoder config")?;
        cond_cfg.validate()?;
        let te_cfg: Qwen3Config = parse(files.json("text_encoder/config.json")?, "text encoder config")?;
        let te_theta = te_cfg.theta()? as f32;
        let vae_cfg: OobleckConfig = parse(files.json("vae/config.json")?, "vae config")?;
        vae_cfg.validate()?;
        if cond_cfg.text_hidden_dim != te_cfg.hidden_size
            || dit_cfg.context_width() != cond_cfg.hidden_size
            || cond_cfg.timbre_hidden_dim != dit_cfg.audio_acoustic_hidden_dim
            || vae_cfg.decoder_input_channels != dit_cfg.audio_acoustic_hidden_dim
        {
            return Err(Error::Config("text encoder, condition encoder, transformer and autoencoder widths disagree".into()));
        }

        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "music backend selected");
        let linear = opts.precision.wtype();

        let dit_files = SafeTensors::open(&files.weights("transformer")?)?;
        let dit = Weights::load(&backend, &dit_files, &dit_cfg.weight_specs(linear))?;
        let dit_conv = Weights::from_host(&backend, &dit_cfg.host_tensors(&dit_files, linear)?)?;
        drop(dit_files);

        let cond_files = SafeTensors::open(&files.weights("condition_encoder")?)?;
        let cond = Weights::load(&backend, &cond_files, &cond_cfg.weight_specs(linear))?;
        let a = dit_cfg.audio_acoustic_hidden_dim;
        let silence_view = cond_files
            .get("silence_latent")
            .ok_or_else(|| Error::MissingTensor("silence_latent".into()))?;
        if silence_view.shape.len() != 3 || silence_view.shape[0] != 1 || silence_view.shape[2] != a || silence_view.shape[1] == 0 {
            return Err(Error::TensorShape {
                name: "silence_latent".into(),
                found: silence_view.shape.to_vec(),
                expected: vec![1, 0, a],
            });
        }
        let silence = silence_view.to_f32();
        let null_token = cond_files.require("null_condition_emb", &[1, 1, cond_cfg.hidden_size])?.to_f32();
        drop(cond_files);

        let te_files = SafeTensors::open(&files.weights("text_encoder")?)?;
        let te = Weights::load(&backend, &te_files, &te_cfg.weight_specs(qwen3::Layout::LAST_HIDDEN, te_cfg.num_hidden_layers, linear)?)?;
        drop(te_files);

        let vae_files = SafeTensors::open(&files.weights("vae")?)?;
        let vae = Weights::from_host(&backend, &vae_cfg.host_tensors(&vae_files)?)?;
        drop(vae_files);

        let tok_path = files.root.join("tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;

        let exact = opts.precision == Precision::F32;
        Ok(Self { backend, dit_cfg, dit, dit_conv, cond_cfg, cond, te_cfg, te_theta, te, vae_cfg, vae, silence, null_token, tokenizer, exact })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.dit.bytes() + self.dit_conv.bytes() + self.cond.bytes() + self.te.bytes() + self.vae.bytes()
    }

    /// Backend device name.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Whether the checkpoint is guidance-distilled (guidance has no effect).
    #[must_use]
    pub fn is_turbo(&self) -> bool {
        self.dit_cfg.turbo()
    }

    /// Output sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.vae_cfg.sampling_rate as u32
    }

    fn latents_per_second(&self) -> f64 {
        self.vae_cfg.sampling_rate as f64 / self.vae_cfg.hop() as f64
    }

    fn acoustic(&self) -> usize {
        self.dit_cfg.audio_acoustic_hidden_dim as usize
    }

    /// Latent frames for a request.
    fn frames(&self, duration: f32) -> usize {
        (f64::from(duration) * self.latents_per_second()).ceil() as usize
    }

    /// The caption and lyric texts the encoders read.
    #[must_use]
    pub fn format(req: &MusicRequest) -> (String, String) {
        let bpm = req.bpm.filter(|b| *b > 0).map_or_else(|| "N/A".to_string(), |b| b.to_string());
        let pick = |s: &Option<String>| s.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("N/A").to_string();
        let ts = pick(&req.timesignature);
        let ks = pick(&req.keyscale);
        let dur = if req.duration_secs > 0.0 { format!("{} seconds", req.duration_secs as i64) } else { "30 seconds".into() };
        let metas = format!("- bpm: {bpm}\n- timesignature: {ts}\n- keyscale: {ks}\n- duration: {dur}\n");
        let text = format!("# Instruction\n{INSTRUCTION}\n\n# Caption\n{}\n\n# Metas\n{metas}<|endoftext|>\n", req.prompt);
        let lyrics = format!("# Languages\n{}\n\n# Lyric\n{}<|endoftext|>", req.language, req.lyrics);
        (text, lyrics)
    }

    fn tokens(&self, text: &str, max: usize) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(text, true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|t| *t as i32).collect();
        ids.truncate(max);
        Ok(ids)
    }

    /// Text-encoder hidden states `[text tokens][hidden]` and lyric token
    /// embeddings `[lyric tokens][hidden]`.
    pub(crate) fn encode_text(&self, text: &[i32], lyrics: &[i32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut g = Graph::new(&self.backend)?;
        let n = text.len() as i64;
        let layers = [self.te_cfg.num_hidden_layers];
        let io = qwen3::build(&mut g, &self.te_cfg, &self.te, qwen3::Layout::LAST_HIDDEN, n, &layers, self.te_theta, self.exact);
        let lyric_ids = g.input(sys::GGML_TYPE_I32, &[lyrics.len() as i64]);
        let lyric = g.get_rows(self.te.get("embed_tokens.weight"), lyric_ids);
        let lyric = g.cast(lyric, sys::GGML_TYPE_F32);
        g.finish(&[io.out, lyric])?;
        g.set_i32(io.tokens, text);
        let pos: Vec<i32> = (0..n as i32).collect();
        g.set_i32(io.positions, &pos);
        g.set_f16(io.mask, &qwen3::mask(text.len(), text.len()));
        g.set_i32(lyric_ids, lyrics);
        g.compute()?;
        Ok((g.read_f32(io.out), g.read_f32(lyric)))
    }

    /// The packed conditioning sequence `[tokens][width]` for one request.
    pub(crate) fn condition(&self, text: &[f32], lyrics: &[f32]) -> Result<Vec<f32>> {
        let hd = self.cond_cfg.text_hidden_dim as usize;
        let a = self.acoustic();
        let (nt, nl) = (text.len() / hd, lyrics.len() / hd);
        let nf = (TIMBRE_SECS * self.latents_per_second()).ceil() as usize;
        let nf = nf.min(self.silence.len() / a);
        let window = self.cond_cfg.sliding_window as usize;
        let mut g = Graph::new(&self.backend)?;
        let io = acestep::build_condition(&mut g, &self.cond_cfg, &self.cond, nt as i64, nl as i64, nf as i64, self.exact)?;
        g.finish(&[io.out])?;
        g.set_f32(io.text, text);
        g.set_f32(io.lyrics, lyrics);
        g.set_i32(io.lyric_positions, &(0..nl as i32).collect::<Vec<_>>());
        g.set_f16(io.lyric_mask, &acestep::window_mask(nl, Some(window)));
        g.set_f32(io.timbre, &self.silence[..nf * a]);
        g.set_i32(io.timbre_positions, &(0..nf as i32).collect::<Vec<_>>());
        g.set_f16(io.timbre_mask, &acestep::window_mask(nf, Some(window)));
        g.compute()?;
        Ok(g.read_f32(io.out))
    }

    /// Source latents of plain generation: encoded silence, repeated as
    /// needed, `[frames][acoustic]`.
    fn source(&self, frames: usize) -> Vec<f32> {
        let a = self.acoustic();
        self.silence.iter().copied().cycle().take(frames * a).collect()
    }

    /// Run the denoising loop from `noise` `[frames][acoustic]`, returning the
    /// final latents and the number of transformer evaluations.
    pub(crate) fn denoise(&self, context: &[f32], noise: &[f32], steps: usize, shift: f32, guidance: f32) -> Result<(Vec<f32>, u32)> {
        self.denoise_over(context, noise, &sigmas(steps, shift), guidance)
    }

    /// One unguided step from noise level `t` to zero.
    #[cfg(test)]
    pub(crate) fn denoise_at(&self, context: &[f32], noise: &[f32], t: f32) -> Result<(Vec<f32>, u32)> {
        self.denoise_over(context, noise, &[t], 1.0)
    }

    fn denoise_over(&self, context: &[f32], noise: &[f32], schedule: &[f32], guidance: f32) -> Result<(Vec<f32>, u32)> {
        let a = self.acoustic();
        let frames = noise.len() / a;
        let p = self.dit_cfg.patch_size as usize;
        let positions = frames.div_ceil(p);
        let width = self.cond_cfg.hidden_size as usize;
        let tokens = context.len() / width;
        let c = self.dit_cfg.in_channels as usize;
        let mut g = Graph::new(&self.backend)?;
        let io = acestep::build_dit(&mut g, &self.dit_cfg, &self.dit, &self.dit_conv, positions as i64, tokens as i64, self.exact)?;
        g.finish(&[io.out])?;
        let pos: Vec<i32> = (0..positions as i32).collect();
        let mask = acestep::window_mask(positions, Some(self.dit_cfg.sliding_window as usize));
        let r_features = acestep::time_features(0.0);
        let src = self.source(frames);
        let mut input = vec![0f32; positions * p * c];
        for t in 0..frames {
            input[t * c..t * c + a].copy_from_slice(&src[t * a..(t + 1) * a]);
            input[t * c + a..t * c + 2 * a].fill(1.0);
        }
        let null: Vec<f32> = self.null_token.iter().copied().cycle().take(context.len()).collect();
        let cfg = guidance > 1.0;
        let mut momentum = Vec::new();
        let mut x = noise.to_vec();
        let mut evaluations = 0u32;
        for (i, &sigma) in schedule.iter().enumerate() {
            for t in 0..frames {
                input[t * c + 2 * a..t * c + 3 * a].copy_from_slice(&x[t * a..(t + 1) * a]);
            }
            let t_features = acestep::time_features(sigma);
            // Every input is set before every evaluation: the graph allocator
            // may reuse an input's memory once its last reader has run.
            let mut eval = |ctx: &[f32]| -> Result<Vec<f32>> {
                g.set_f32(io.frames, &input);
                g.set_f32(io.t_features, &t_features);
                g.set_f32(io.r_features, &r_features);
                g.set_i32(io.positions, &pos);
                g.set_f16(io.mask, &mask);
                g.set_f32(io.context, ctx);
                g.compute()?;
                evaluations += 1;
                let mut v = g.read_f32(io.out);
                v.truncate(frames * a);
                Ok(v)
            };
            let mut v = eval(context)?;
            if cfg {
                let u = eval(&null)?;
                v = guided(&v, &u, &mut momentum, guidance - 1.0, a);
            }
            let next = schedule.get(i + 1).copied().unwrap_or(0.0);
            let dt = next - sigma;
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += dt * vi;
            }
        }
        Ok((x, evaluations))
    }

    /// Decode latents `[frames][acoustic]` to audio `[channels][samples]`,
    /// in overlapping tiles for long inputs.
    pub(crate) fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        let a = self.acoustic();
        let frames = latents.len() / a;
        let ch = self.vae_cfg.audio_channels as usize;
        if frames <= DECODE_TILE {
            return self.decode_once(latents, frames);
        }
        // Each tile is decoded with up to `DECODE_OVERLAP` frames of context
        // on each side, which is trimmed off in samples, rounded as the
        // reference rounds it.
        let stride = DECODE_TILE - 2 * DECODE_OVERLAP;
        let mut parts: Vec<Vec<f32>> = vec![Vec::new(); ch];
        let mut core = 0usize;
        while core < frames {
            let core_end = (core + stride).min(frames);
            let win = core.saturating_sub(DECODE_OVERLAP);
            let win_end = (core_end + DECODE_OVERLAP).min(frames);
            let tile = self.decode_once(&latents[win * a..win_end * a], win_end - win)?;
            let n = tile.len() / ch;
            let up = n as f64 / (win_end - win) as f64;
            let keep_l = ((core - win) as f64 * up).round() as usize;
            let keep_r = ((win_end - core_end) as f64 * up).round() as usize;
            for (c, part) in parts.iter_mut().enumerate() {
                part.extend_from_slice(&tile[c * n + keep_l..c * n + n - keep_r]);
            }
            core = core_end;
        }
        let out = parts.concat();
        Ok(out)
    }

    fn decode_once(&self, latents: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut g = Graph::new(&self.backend)?;
        let io = oobleck::build_decoder(&mut g, &self.vae_cfg, &self.vae, frames as i64);
        g.finish(&[io.out])?;
        g.set_f32(io.latents, latents);
        g.compute()?;
        Ok(g.read_f32(io.out))
    }

    /// Generate one piece of music.
    ///
    /// # Errors
    /// [`Error::Request`] for a duration or step count the model cannot
    /// serve; backend failures otherwise.
    pub fn generate(&mut self, req: &MusicRequest) -> Result<Audio> {
        if !(req.duration_secs > 0.0 && req.duration_secs <= MAX_DURATION_SECS) {
            return Err(Error::Request(format!("duration must be in (0, {MAX_DURATION_SECS}] seconds")));
        }
        let frames = self.frames(req.duration_secs);
        let noise = schedule::gaussian(req.seed, frames * self.acoustic());
        self.generate_from(req, &noise)
    }

    /// Generate from given starting noise `[frames][acoustic]`.
    pub(crate) fn generate_from(&mut self, req: &MusicRequest, noise: &[f32]) -> Result<Audio> {
        let turbo = self.is_turbo();
        let steps = req.steps.unwrap_or(if turbo { 8 } else { 50 }) as usize;
        if steps == 0 || steps > 200 {
            return Err(Error::Request("steps must be in 1..=200".into()));
        }
        let guidance = if turbo { 1.0 } else { req.guidance_scale.unwrap_or(7.0) };
        let shift = req.shift.unwrap_or(3.0);
        if !(shift > 0.0 && guidance.is_finite()) {
            return Err(Error::Request("shift must be positive and guidance finite".into()));
        }

        let t0 = Instant::now();
        let (text, lyrics) = Self::format(req);
        let text_ids = self.tokens(&text, MAX_TEXT_TOKENS)?;
        let lyric_ids = self.tokens(&lyrics, MAX_LYRIC_TOKENS)?;
        let (th, lh) = self.encode_text(&text_ids, &lyric_ids)?;
        let context = self.condition(&th, &lh)?;
        let encode_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let (latents, evaluations) = self.denoise(&context, noise, steps, shift, guidance)?;
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let mut samples = self.decode(&latents)?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        normalise_peak(&mut samples);

        Ok(Audio {
            sample_rate: self.sample_rate(),
            channels: self.vae_cfg.audio_channels as u32,
            samples,
            seed: req.seed,
            evaluations,
            timings: Timings { encode_ms, denoise_ms, decode_ms },
        })
    }
}

/// Scale to a fixed peak level, as the reference does for every output.
fn normalise_peak(samples: &mut [f32]) {
    let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 1.0 {
        samples.iter_mut().for_each(|v| *v /= peak);
    }
    let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let k = 10f32.powf(OUTPUT_PEAK_DB / 20.0) / peak;
    samples.iter_mut().for_each(|v| *v *= k);
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_starts_at_full_noise_and_shifts_toward_it() {
        let s = sigmas(8, 3.0);
        assert_eq!(s.len(), 8);
        assert_eq!(s[0], 1.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        let unshifted = sigmas(8, 1.0);
        assert!((unshifted[4] - 0.5).abs() < 1e-6);
        assert!(s[4] > unshifted[4]);
    }

    #[test]
    fn the_prompt_template_names_every_field() {
        let req = MusicRequest {
            prompt: "jazz".into(),
            lyrics: "la".into(),
            language: "en".into(),
            duration_secs: 12.7,
            steps: None,
            guidance_scale: None,
            shift: None,
            seed: 0,
            bpm: Some(90),
            keyscale: None,
            timesignature: Some(" 3 ".into()),
        };
        let (text, lyrics) = AceStep::format(&req);
        assert!(text.contains("# Caption\njazz\n"));
        assert!(text.contains("- bpm: 90\n- timesignature:  3 \n- keyscale: N/A\n- duration: 12 seconds\n<|endoftext|>\n"));
        assert_eq!(lyrics, "# Languages\nen\n\n# Lyric\nla<|endoftext|>");
    }

    #[test]
    fn guidance_without_a_difference_is_the_conditional_prediction() {
        let cond: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut m = Vec::new();
        let out = guided(&cond, &cond, &mut m, 6.0, 4);
        assert_eq!(out, cond);
    }

    #[test]
    fn the_output_peaks_at_minus_one_dbfs() {
        let mut s = vec![0.1, -3.0, 0.5, 2.0];
        normalise_peak(&mut s);
        let peak = s.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 10f32.powf(-0.05)).abs() < 1e-6);
    }
}
