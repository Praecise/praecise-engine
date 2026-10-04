//! The FLUX 3 action policy: camera frames, robot state and an instruction in,
//! one action chunk out.
//!
//! The transformer jointly denoises the future video latents and the action
//! chunk, conditioned on the encoded current observation (or a set of history
//! snapshots), the robot state and the instruction context. Two observation
//! packers exist: `frame` (one current frame, state at `t = 0`) and `history`
//! (snapshots of the last `n_obs` frames, the state history and optionally the
//! past actions on conditioning layer `-1`).

use std::path::PathBuf;

use serde_json::Value;

use super::packing::{self, CameraLayout, Frame, LATENT_CHANNELS};
use super::sampling::{self, SamplerKind, SamplerSettings};
use super::text_encoder::{TextEncoder, CONTEXT_LAYERS};
use super::video_vae::{VaeConfig, VideoVae};
use super::{Flux3Transformer, StreamInput, TextInput};
use crate::error::{Error, Result};
use crate::pipeline::LoadOptions;
use crate::schedule;

/// How an observation conditions the prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conditioning {
    /// The current frame and state only.
    Frame,
    /// Snapshots of the observation history and the state history.
    History,
}

/// Inference settings of a released policy (its `config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyConfig {
    /// Observation packer.
    pub conditioning: Conditioning,
    /// History packer: past actions concatenated to the state history.
    pub condition_on_past_actions: bool,
    /// Observation steps (history length).
    pub n_obs_steps: usize,
    /// History frames encoded as conditioning snapshots.
    pub history_snapshots: usize,
    /// Actions per predicted chunk.
    pub chunk_size: usize,
    /// Actions executed from each chunk.
    pub n_action_steps: usize,
    /// Action (and state) channels.
    pub action_dim: usize,
    /// Control rate.
    pub fps: f32,
    /// Clock of the video positions, when it differs from the control rate.
    pub video_position_fps: Option<f32>,
    /// Canvas `(h, w)` the cameras are composed onto.
    pub canvas_hw: (usize, usize),
    /// Canvas area carrying image content (the canvas minus layout padding).
    pub content_hw: (usize, usize),
    /// Camera layout.
    pub camera_layout: CameraLayout,
    /// Action stream name; its conditioning stream is `{name}_cond`.
    pub action_modality: String,
    /// Scale of actions (and frame-packer states) in model units.
    pub action_scale: f32,
    /// Channels stored as `1 - x` (grippers).
    pub gripper_flip_dims: Vec<usize>,
    /// Solver, steps and shift.
    pub sampler: SamplerSettings,
    /// Guidance on the video stream.
    pub guidance_scale: f32,
    /// Guidance on the action stream (defaults to `guidance_scale`).
    pub guidance_scale_action: f32,
    /// Fixed text length, or bucketed padding when `None`.
    pub text_fixed_length: Option<usize>,
}

fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

fn num(v: &Value, key: &str, default: f64) -> Result<f64> {
    match field(v, key) {
        None => Ok(default),
        Some(x) => x.as_f64().ok_or_else(|| Error::Config(format!("policy config: {key} is not a number"))),
    }
}

fn count(v: &Value, key: &str, default: usize) -> Result<usize> {
    match field(v, key) {
        None => Ok(default),
        Some(x) => x.as_u64().map(|n| n as usize).ok_or_else(|| Error::Config(format!("policy config: {key} is not a count"))),
    }
}

fn pair(v: &Value, key: &str) -> Result<Option<(usize, usize)>> {
    let Some(x) = field(v, key) else { return Ok(None) };
    let a = x.as_array().filter(|a| a.len() == 2).ok_or_else(|| Error::Config(format!("policy config: {key} is not a pair")))?;
    let get = |i: usize| a[i].as_u64().map(|n| n as usize).ok_or_else(|| Error::Config(format!("policy config: {key} is not a pair of counts")));
    Ok(Some((get(0)?, get(1)?)))
}

/// Channel indices (negative ones count from the end) resolved against `d`.
fn channel_list(v: &Value, key: &str, d: usize) -> Result<Vec<usize>> {
    let Some(x) = field(v, key) else { return Ok(Vec::new()) };
    let a = x.as_array().ok_or_else(|| Error::Config(format!("policy config: {key} is not a list")))?;
    a.iter()
        .map(|n| {
            let i = n.as_i64().ok_or_else(|| Error::Config(format!("policy config: {key} holds a non-integer")))?;
            let r = if i < 0 { i + d as i64 } else { i };
            if r < 0 || r >= d as i64 {
                return Err(Error::Config(format!("policy config: {key} index {i} is outside {d} channels")));
            }
            Ok(r as usize)
        })
        .collect()
}

impl PolicyConfig {
    /// Settings from a policy `config.json`.
    ///
    /// # Errors
    /// On a missing action feature, an unknown packer, layout or sampler, or
    /// malformed values.
    pub fn from_json(v: &Value) -> Result<Self> {
        let conditioning = match field(v, "packer").or_else(|| field(v, "conditioning")).and_then(Value::as_str).unwrap_or("history") {
            "frame" => Conditioning::Frame,
            "history" => Conditioning::History,
            other => return Err(Error::Config(format!("policy config: unknown packer {other}"))),
        };
        let action_dim = v
            .pointer("/output_features/action/shape/0")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Config("policy config: no action feature shape".into()))? as usize;
        let camera_layout = CameraLayout::parse(field(v, "camera_layout").and_then(Value::as_str).unwrap_or("side_by_side"))?;
        let canvas_hw = pair(v, "canvas_hw")?.unwrap_or((256, 512));
        let content_hw = if camera_layout == CameraLayout::WristOverPair {
            let feats = field(v, "input_features").and_then(Value::as_object);
            let first = field(v, "camera_keys")
                .and_then(|k| k.get(0))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| feats.and_then(|f| f.keys().find(|k| k.starts_with("observation.images")).cloned()))
                .ok_or_else(|| Error::Config("policy config: the droid layout needs a camera".into()))?;
            let shape = feats
                .and_then(|f| f.get(&first))
                .and_then(|f| f.get("shape"))
                .and_then(Value::as_array)
                .filter(|s| s.len() >= 2)
                .ok_or_else(|| Error::Config(format!("policy config: no shape for {first}")))?;
            let dim = |i: usize| shape[shape.len() - 2 + i].as_u64().unwrap_or(0) as usize;
            (dim(0) + dim(0) / 2, dim(1))
        } else {
            canvas_hw
        };
        let kind = match field(v, "sampler").and_then(Value::as_str).unwrap_or("cosmos_unipc") {
            "cosmos_unipc" => SamplerKind::UniPc,
            "euler" => SamplerKind::Euler,
            other => return Err(Error::Config(format!("policy config: unknown sampler {other}"))),
        };
        let guidance_scale = num(v, "guidance_scale", 4.0)? as f32;
        let chunk_size = count(v, "chunk_size", 32)?;
        Ok(Self {
            conditioning,
            condition_on_past_actions: field(v, "condition_on_past_actions").and_then(Value::as_bool).unwrap_or(false),
            n_obs_steps: count(v, "n_obs_steps", 1)?,
            history_snapshots: count(v, "history_snapshots", 1)?,
            chunk_size,
            n_action_steps: count(v, "n_action_steps", chunk_size)?,
            action_dim,
            fps: num(v, "fps", 30.0)? as f32,
            video_position_fps: match field(v, "video_position_fps") {
                None => None,
                Some(_) => Some(num(v, "video_position_fps", 0.0)? as f32),
            },
            canvas_hw,
            content_hw,
            camera_layout,
            action_modality: field(v, "action_modality").and_then(Value::as_str).unwrap_or("action").to_owned(),
            action_scale: num(v, "action_scale", 2.0)? as f32,
            gripper_flip_dims: channel_list(v, "gripper_flip_dims", action_dim)?,
            sampler: SamplerSettings { kind, steps: count(v, "num_inference_steps", 4)?, shift: num(v, "sampler_shift", 5.0)? },
            guidance_scale,
            guidance_scale_action: match field(v, "guidance_scale_action") {
                None => guidance_scale,
                Some(_) => num(v, "guidance_scale_action", 1.0)? as f32,
            },
            text_fixed_length: match field(v, "text_fixed_length") {
                None => None,
                Some(_) => Some(count(v, "text_fixed_length", 0)?),
            },
        })
    }

    fn video_fps(&self) -> f32 {
        self.video_position_fps.unwrap_or(self.fps)
    }

    /// Latent grid `(h, w)` kept from the encoded canvas.
    #[must_use]
    pub fn latent_hw(&self) -> (usize, usize) {
        packing::latent_hw(self.content_hw.0, self.content_hw.1)
    }

    /// Predicted (future) latent frames.
    #[must_use]
    pub fn predicted_latent_frames(&self) -> usize {
        match self.conditioning {
            Conditioning::Frame => packing::latent_frames(self.chunk_size + 1) - 1,
            Conditioning::History => packing::latent_frames(self.chunk_size),
        }
    }

    /// Time ids of the predicted latent frames.
    #[must_use]
    pub fn predicted_video_time_ids(&self) -> Vec<i32> {
        let first = match self.conditioning {
            Conditioning::Frame => 1,
            Conditioning::History => packing::latent_frames(self.n_obs_steps),
        };
        (first..first + self.predicted_latent_frames()).map(|i| packing::time_id(packing::latent_time(i, self.video_fps()))).collect()
    }

    /// Seconds of each predicted action relative to the current frame.
    #[must_use]
    pub fn action_times(&self) -> Vec<f32> {
        let off = match self.conditioning {
            Conditioning::Frame => 1.0,
            Conditioning::History => 0.0,
        };
        (0..self.chunk_size).map(|k| (k as f32 + off) / self.fps).collect()
    }

    /// Channels of the action conditioning stream.
    #[must_use]
    pub fn conditioning_channels(&self) -> usize {
        match self.conditioning {
            Conditioning::History if self.condition_on_past_actions => 2 * self.action_dim,
            _ => self.action_dim,
        }
    }

    fn guidance(&self) -> [f32; 2] {
        [self.guidance_scale, self.guidance_scale_action]
    }
}

/// Packed conditioning of one observation.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditions {
    /// Video conditioning tokens `[n][96]`.
    pub video: Vec<f32>,
    /// Their positions.
    pub video_ids: Vec<[i32; 4]>,
    /// Action conditioning tokens `[n][conditioning_channels]`.
    pub action: Vec<f32>,
    /// Their positions.
    pub action_ids: Vec<[i32; 4]>,
}

/// Initial noise of one chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct Noise {
    /// Video latents `[96][t][h][w]`.
    pub video: Vec<f32>,
    /// Actions `[action_dim][chunk]`.
    pub action: Vec<f32>,
}

impl Noise {
    /// Standard normal noise from a seed.
    #[must_use]
    pub fn from_seed(cfg: &PolicyConfig, seed: u64) -> Self {
        let (h, w) = cfg.latent_hw();
        let nv = LATENT_CHANNELS * cfg.predicted_latent_frames() * h * w;
        let all = schedule::gaussian(seed, nv + cfg.action_dim * cfg.chunk_size);
        Self { video: all[..nv].to_vec(), action: all[nv..].to_vec() }
    }
}

/// Pack the action conditioning: the current state (`frame`, scaled, flipped)
/// or the state history with optional past actions (`history`, unscaled).
/// `states` is `[n][action_dim]` (one row for `frame`), `past_actions` the
/// same shape.
///
/// # Errors
/// On a shape that does not match the packer.
pub fn pack_actions(cfg: &PolicyConfig, states: &[f32], past_actions: Option<&[f32]>) -> Result<(Vec<f32>, Vec<[i32; 4]>)> {
    let d = cfg.action_dim;
    match cfg.conditioning {
        Conditioning::Frame => {
            if states.len() < d {
                return Err(Error::Request("the frame policy needs the current state".into()));
            }
            let mut s = states[states.len() - d..].to_vec();
            packing::flip_channels(&mut s, d, &cfg.gripper_flip_dims);
            Ok((s.iter().map(|x| x * cfg.action_scale).collect(), packing::sequence_ids(&[0.0], 0)))
        }
        Conditioning::History => {
            let n = cfg.n_obs_steps;
            if states.len() != n * d {
                return Err(Error::Request(format!("the history policy needs {n} states of {d} channels")));
            }
            let tokens = match (cfg.condition_on_past_actions, past_actions) {
                (false, _) => states.to_vec(),
                (true, Some(p)) if p.len() == states.len() => {
                    p.chunks(d).zip(states.chunks(d)).flat_map(|(a, s)| a.iter().chain(s).copied()).collect()
                }
                (true, _) => return Err(Error::Request("the policy needs past actions shaped like the states".into())),
            };
            let times: Vec<f32> = (0..n).map(|k| (k as f32 - (n as f32 - 1.0)) / cfg.fps).collect();
            Ok((tokens, packing::sequence_ids(&times, -1)))
        }
    }
}

/// Video conditioning tokens of encoded latents `[96][t][lh][lw]` (one entry
/// per conditioning frame, cropped to the content grid) placed at `seconds`.
fn pack_video(cfg: &PolicyConfig, latents: &[(Vec<f32>, [usize; 3])], seconds: &[f32]) -> (Vec<f32>, Vec<[i32; 4]>) {
    let (lh, lw) = cfg.latent_hw();
    let mut tokens = Vec::new();
    let mut ids = Vec::new();
    for ((lat, [t, h, w]), &s) in latents.iter().zip(seconds) {
        let mut crop = Vec::with_capacity(LATENT_CHANNELS * lh * lw);
        for c in 0..LATENT_CHANNELS {
            for y in 0..lh {
                let row = ((c * t) * h + y) * w;
                crop.extend_from_slice(&lat[row..row + lw]);
            }
        }
        tokens.extend(packing::video_tokens(&crop, LATENT_CHANNELS, 1, lh, lw));
        ids.extend(packing::video_ids(&[packing::time_id(s)], lh, lw, 0));
    }
    (tokens, ids)
}

/// Denoise one chunk. Returns the actions `[chunk][action_dim]` in model
/// units divided by the action scale (the normalised action space, with the
/// gripper channels still flipped).
///
/// # Errors
/// On mismatched shapes or a backend failure.
pub fn sample_chunk(
    dit: &Flux3Transformer,
    cfg: &PolicyConfig,
    cond: &Conditions,
    ctx: &[f32],
    ctx_uncond: Option<&[f32]>,
    noise: &Noise,
) -> Result<Vec<f32>> {
    let width = dit.config().context_in_dim as usize;
    let (lh, lw) = cfg.latent_hw();
    let nt = cfg.predicted_latent_frames();
    let (d, k) = (cfg.action_dim, cfg.chunk_size);
    if noise.video.len() != LATENT_CHANNELS * nt * lh * lw || noise.action.len() != d * k {
        return Err(Error::Request("noise does not match the policy shapes".into()));
    }
    if ctx.len() % width != 0 || ctx_uncond.is_some_and(|u| u.len() % width != 0) {
        return Err(Error::Request("text context width does not match the transformer".into()));
    }
    let action_name = cfg.action_modality.as_str();
    let action_cond_name = format!("{action_name}_cond");
    let video = packing::video_tokens(&noise.video, LATENT_CHANNELS, nt, lh, lw);
    let video_ids = packing::video_ids(&cfg.predicted_video_time_ids(), lh, lw, 0);
    let mut action = vec![0.0f32; k * d];
    for c in 0..d {
        for t in 0..k {
            action[t * d + c] = noise.action[c * k + t];
        }
    }
    let action_ids = packing::sequence_ids(&cfg.action_times(), 0);
    let video_cond_t = vec![0.0f32; cond.video_ids.len()];
    let action_cond_t = vec![0.0f32; cond.action_ids.len()];
    let vector = vec![0.0f32; dit.config().vec_in_dim.unwrap_or(0) as usize];
    let guidance = cfg.guidance();
    let guided = guidance.iter().any(|&g| g != 1.0);
    if guided && ctx_uncond.is_none() {
        return Err(Error::Request("guidance needs the empty-instruction context".into()));
    }
    let run = |samples: &[Vec<f32>], t: f32, text: &[f32]| -> Result<Vec<Vec<f32>>> {
        let n_txt = text.len() / width;
        let txt_ids = packing::text_ids(n_txt);
        let txt_t = vec![0.0f32; n_txt];
        let vt = vec![t; video_ids.len()];
        let at = vec![t; k];
        let streams = [
            StreamInput { name: "video", tokens: &samples[0], ids: &video_ids, timesteps: &vt },
            StreamInput { name: "video_cond", tokens: &cond.video, ids: &cond.video_ids, timesteps: &video_cond_t },
            StreamInput { name: action_name, tokens: &samples[1], ids: &action_ids, timesteps: &at },
            StreamInput { name: &action_cond_name, tokens: &cond.action, ids: &cond.action_ids, timesteps: &action_cond_t },
        ];
        let text = TextInput { tokens: text, ids: &txt_ids, timesteps: &txt_t };
        let mut out = dit.forward(text, (!vector.is_empty()).then_some(vector.as_slice()), &streams)?;
        Ok(vec![std::mem::take(&mut out[0]), std::mem::take(&mut out[2])])
    };
    let out = sampling::sample(vec![video, action], cfg.sampler, |samples, t| {
        let cond_pred = run(samples, t, ctx)?;
        match ctx_uncond.filter(|_| guided) {
            None => Ok(cond_pred),
            Some(u) => sampling::guide(&run(samples, t, u)?, &cond_pred, &guidance),
        }
    })?;
    Ok(out[1].iter().map(|x| x / cfg.action_scale).collect())
}

/// One observation: per-camera frames (`n_obs` each for `history`, the last
/// frame is the current one), states `[n][action_dim]` and optional past
/// actions shaped like the states.
#[derive(Debug, Clone)]
pub struct Observation<'a> {
    /// `cameras[camera][time]`, in the configured camera order.
    pub cameras: &'a [Vec<Frame>],
    /// Robot state rows.
    pub states: &'a [f32],
    /// Past commands (history packer with past-action conditioning).
    pub past_actions: Option<&'a [f32]>,
    /// The instruction.
    pub instruction: &'a str,
}

/// A loaded policy: transformer, video autoencoder and text encoder.
pub struct ActionPolicy {
    cfg: PolicyConfig,
    dit: Flux3Transformer,
    vae: VideoVae,
    text: TextEncoder,
}

impl std::fmt::Debug for ActionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActionPolicy").field("cfg", &self.cfg).finish_non_exhaustive()
    }
}

/// Check that a transformer carries the streams the settings name.
///
/// # Errors
/// When a stream is missing or has the wrong width.
pub fn check_transformer(cfg: &PolicyConfig, dit: &Flux3Transformer) -> Result<()> {
    let dc = dit.config();
    let want = [
        ("video", LATENT_CHANNELS),
        ("video_cond", LATENT_CHANNELS),
        (cfg.action_modality.as_str(), cfg.action_dim),
        (&format!("{}_cond", cfg.action_modality), cfg.conditioning_channels()),
    ];
    for (name, ch) in want {
        if dc.channels(name) != Some(ch as u64) {
            return Err(Error::Config(format!("transformer stream {name} does not carry {ch} channels")));
        }
    }
    Ok(())
}

/// Pack an observation into conditioning tokens with the video autoencoder.
///
/// # Errors
/// On a camera/layout mismatch, wrong history length or a backend failure.
pub fn encode_conditions(cfg: &PolicyConfig, vae: &VideoVae, obs: &Observation<'_>) -> Result<Conditions> {
    let (ch, cw) = cfg.canvas_hw;
    let frames = obs.cameras.first().map_or(0, Vec::len);
    let pick: Vec<usize> = match cfg.conditioning {
        Conditioning::Frame => vec![frames.checked_sub(1).ok_or_else(|| Error::Request("no camera frames".into()))?],
        Conditioning::History => {
            if frames != cfg.n_obs_steps {
                return Err(Error::Request(format!("the history policy needs {} frames per camera", cfg.n_obs_steps)));
            }
            packing::snapshot_indices(cfg.n_obs_steps, cfg.history_snapshots)
        }
    };
    let mut latents = Vec::with_capacity(pick.len());
    let mut seconds = Vec::with_capacity(pick.len());
    for &i in &pick {
        let cams: Vec<Vec<Frame>> = obs.cameras.iter().map(|c| vec![c[i].clone()]).collect();
        let canvas = packing::compose_canvas(&cams, cfg.camera_layout, ch, cw)?;
        // The frame packer encodes the current frame as a padded chunk;
        // history snapshots are encoded on their own.
        let (lat, dims) = match cfg.conditioning {
            Conditioning::Frame => vae.encode_chunked(&canvas, 1, ch, cw)?,
            Conditioning::History => vae.encode(&canvas, 1, ch, cw)?,
        };
        latents.push((lat, dims));
        seconds.push(match cfg.conditioning {
            Conditioning::Frame => 0.0,
            Conditioning::History => i as f32 / cfg.video_fps(),
        });
    }
    let (video, video_ids) = pack_video(cfg, &latents, &seconds);
    let (action, action_ids) = pack_actions(cfg, obs.states, obs.past_actions)?;
    Ok(Conditions { video, video_ids, action, action_ids })
}

/// Instruction context and, when guidance is on, the empty-prompt context.
///
/// # Errors
/// On a tokenizer or backend failure.
pub fn encode_instruction(cfg: &PolicyConfig, text: &TextEncoder, instruction: &str) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
let (ctx, _) = text.encode(instruction, cfg.text_fixed_length)?;
let uncond = if cfg.guidance().iter().any(|&g| g != 1.0) { Some(text.encode("", cfg.text_fixed_length)?.0) } else { None };
Ok((ctx, uncond))
}

/// Sample one action chunk `[chunk][action_dim]` in the normalised action
/// space (grippers unflipped) from packed conditions and text context.
///
/// # Errors
/// On mismatched inputs or a backend failure.
pub fn predict_chunk(
    dit: &Flux3Transformer,
    cfg: &PolicyConfig,
    cond: &Conditions,
    ctx: &[f32],
    ctx_uncond: Option<&[f32]>,
    noise: &Noise,
) -> Result<Vec<f32>> {
    let mut chunk = sample_chunk(dit, cfg, cond, ctx, ctx_uncond, noise)?;
    packing::flip_channels(&mut chunk, cfg.action_dim, &cfg.gripper_flip_dims);
    Ok(chunk)
}

impl ActionPolicy {
    /// Load a policy from its settings and component files: the transformer
    /// shards (keys `dit.*`), the autoencoder shards (keys `model.*`) and the
    /// text encoder directory.
    ///
    /// # Errors
    /// On missing or malformed files or a stream set that does not match the
    /// settings.
    pub fn load(
        cfg: PolicyConfig,
        dit_files: &[PathBuf],
        vae_files: &[PathBuf],
        text_dir: &std::path::Path,
        opts: LoadOptions,
    ) -> Result<Self> {
        let dit = Flux3Transformer::load(dit_files, "dit.", opts)?;
        check_transformer(&cfg, &dit)?;
        let vae = VideoVae::load(vae_files, "model.", VaeConfig::default(), opts)?;
        let text = TextEncoder::load(text_dir, &CONTEXT_LAYERS, opts)?;
        if text.context_width() as u64 != dit.config().context_in_dim {
            return Err(Error::Config("text encoder width does not match the transformer".into()));
        }
        Ok(Self { cfg, dit, vae, text })
    }

    /// The settings.
    #[must_use]
    pub fn config(&self) -> &PolicyConfig {
        &self.cfg
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.dit.resident_bytes() + self.vae.resident_bytes() + self.text.resident_bytes()
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.dit.device()
    }

    /// Pack an observation into conditioning tokens.
    ///
    /// # Errors
    /// On a camera/layout mismatch, wrong history length or a backend
    /// failure.
    pub fn conditions(&self, obs: &Observation<'_>) -> Result<Conditions> {
        encode_conditions(&self.cfg, &self.vae, obs)
    }

    /// Predict one action chunk `[chunk][action_dim]` in the normalised
    /// action space of the policy (grippers unflipped).
    ///
    /// # Errors
    /// On a malformed observation or a backend failure.
    pub fn predict(&self, obs: &Observation<'_>, noise: &Noise) -> Result<Vec<f32>> {
        let cond = self.conditions(obs)?;
        let (ctx, uncond) = encode_instruction(&self.cfg, &self.text, obs.instruction)?;
        predict_chunk(&self.dit, &self.cfg, &cond, &ctx, uncond.as_deref(), noise)
    }
}

#[cfg(test)]
mod parity;
