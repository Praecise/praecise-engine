//! Released FLUX 3 action policy layout.
//!
//! A released policy is two directories: the policy itself (`config.json`,
//! `model.safetensors` with the full transformer under `dit.*`, and, for
//! policies trained on normalised commands, the processor files carrying the
//! quantile statistics) and the shared base holding the video autoencoder
//! and the text encoder, at the paths the policy config names.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::packing::{self, ActionRepresentation, Frame, Quantiles};
use super::policy::{self, ActionPolicy, Conditioning, Noise, Observation, PolicyConfig};
use super::text_encoder::{TextEncoder, CONTEXT_LAYERS};
use super::video_vae::{VaeConfig, VideoVae};
use super::Flux3Transformer;
use crate::error::{Error, Result};
use crate::pipeline::LoadOptions;
use crate::safetensors::SafeTensors;

/// Processor step holding the state and action statistics.
const HISTORY_NORMALIZER: &str = "flux3_observation_history_normalizer";

/// Command normalisation of a policy trained on quantile-scaled commands.
#[derive(Debug, Clone, PartialEq)]
pub struct Normalization {
    /// How commands are encoded in the action stream.
    pub representation: ActionRepresentation,
    /// State statistics.
    pub state: Quantiles,
    /// Action statistics.
    pub action: Quantiles,
    /// Clip applied after scaling.
    pub clip: f32,
}

impl Normalization {
    /// Scaled state rows `[n][d]`.
    #[must_use]
    pub fn states(&self, states: &[f32]) -> Vec<f32> {
        self.state.normalize(states, self.clip)
    }

    /// Past-action conditioning from an absolute command history `[n][d]`.
    #[must_use]
    pub fn past_actions(&self, commands: &[f32]) -> Vec<f32> {
        packing::past_actions(commands, self.action.q01.len(), &self.representation, &self.action, self.clip)
    }

    /// Absolute commands from a predicted chunk, anchored at the last command.
    #[must_use]
    pub fn commands(&self, chunk: &[f32], last_command: &[f32]) -> Vec<f32> {
        packing::integrate_actions(chunk, self.action.q01.len(), &self.representation, &self.action, last_command)
    }
}

/// The files and settings of a released policy.
#[derive(Debug, Clone)]
pub struct PolicyRelease {
    /// Inference settings.
    pub config: PolicyConfig,
    /// Observation keys of the cameras, in canvas order.
    pub camera_keys: Vec<String>,
    /// Command normalisation, when the policy was trained on scaled commands.
    pub normalization: Option<Normalization>,
    /// Transformer shards (keys `dit.*`).
    pub dit_files: Vec<PathBuf>,
    /// Video autoencoder shards (keys `model.*`).
    pub vae_files: Vec<PathBuf>,
    /// Text encoder directory.
    pub text_dir: PathBuf,
    /// Repository of the shared base the policy names.
    pub base_repo: String,
    /// Revision of that base.
    pub base_revision: String,
}

/// A file of the shared base, named in the policy config as
/// `owner/repo:path@revision`.
struct BaseFile {
    repo: String,
    path: String,
    revision: String,
}

impl BaseFile {
    fn parse(raw: &Value, key: &str) -> Result<Self> {
        let id = raw.get(key).and_then(Value::as_str).ok_or_else(|| Error::Config(format!("policy config: no {key}")))?;
        let bad = || Error::Config(format!("policy config: {key} {id} is not repo:path@revision"));
        let (repo, rest) = id.split_once(':').ok_or_else(bad)?;
        let (path, revision) = rest.rsplit_once('@').ok_or_else(bad)?;
        let p = Path::new(path);
        if repo.is_empty() || revision.is_empty() || path.is_empty() || p.is_absolute() || p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
            return Err(bad());
        }
        Ok(Self { repo: repo.to_owned(), path: path.to_owned(), revision: revision.to_owned() })
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
}

fn existing(path: PathBuf) -> Result<PathBuf> {
    if path.exists() {
        Ok(path)
    } else {
        Err(Error::Config(format!("{} is missing", path.display())))
    }
}

fn quantiles(st: &SafeTensors, stream: &str, d: usize) -> Result<Quantiles> {
    let get = |q: &str| -> Result<Vec<f32>> { Ok(st.require(&format!("{stream}.{q}"), &[d as u64])?.to_f32()) };
    let q = Quantiles { q01: get("q01")?, q99: get("q99")? };
    if q.q01.iter().zip(&q.q99).any(|(lo, hi)| hi < lo || !lo.is_finite() || !hi.is_finite()) {
        return Err(Error::Config(format!("{stream} quantiles are not ordered")));
    }
    Ok(q)
}

impl PolicyRelease {
    /// Read a released policy directory and the directory of its base.
    ///
    /// # Errors
    /// On missing files, malformed settings, or statistics that do not match
    /// the action width.
    pub fn open(policy_dir: &Path, base_dir: &Path) -> Result<Self> {
        let raw = read_json(&policy_dir.join("config.json"))?;
        if raw.get("type").and_then(Value::as_str) != Some("flux3") {
            return Err(Error::Config("policy config: not a flux3 policy".into()));
        }
        if raw.get("trunk_weights").is_some_and(|v| !v.is_null()) {
            return Err(Error::Config("policy config: split trunk weights are not supported".into()));
        }
        let config = PolicyConfig::from_json(&raw)?;
        let camera_keys: Vec<String> = raw
            .get("camera_keys")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_owned).collect())
            .unwrap_or_default();
        if camera_keys.is_empty() {
            return Err(Error::Config("policy config: no camera keys".into()));
        }

        let vae = BaseFile::parse(&raw, "video_vae_id")?;
        let text = BaseFile::parse(&raw, "text_encoder_id")?;
        if vae.repo != text.repo || vae.revision != text.revision {
            return Err(Error::Config("policy config: autoencoder and text encoder come from different bases".into()));
        }
        let vae_files = vec![existing(base_dir.join(&vae.path))?];
        let text_dir = existing(base_dir.join(&text.path))?;
        let dit_files = vec![existing(policy_dir.join("model.safetensors"))?];

        let normalization = Self::normalization(policy_dir, &raw, config.action_dim)?;
        if config.conditioning == Conditioning::History && normalization.is_none() {
            return Err(Error::Config("history policy without command statistics".into()));
        }
        Ok(Self { config, camera_keys, normalization, dit_files, vae_files, text_dir, base_repo: vae.repo, base_revision: vae.revision })
    }

    fn normalization(policy_dir: &Path, raw: &Value, d: usize) -> Result<Option<Normalization>> {
        let pre = policy_dir.join("policy_preprocessor.json");
        if !pre.exists() {
            return Ok(None);
        }
        let steps = read_json(&pre)?;
        let Some(step) = steps
            .get("steps")
            .and_then(Value::as_array)
            .and_then(|s| s.iter().find(|s| s.get("registry_name").and_then(Value::as_str) == Some(HISTORY_NORMALIZER)))
        else {
            return Ok(None);
        };
        let file = step
            .get("state_file")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Config(format!("{HISTORY_NORMALIZER}: no state file")))?;
        let cfg = step.get("config").unwrap_or(&Value::Null);
        let representation = match cfg.get("action_representation").and_then(Value::as_str).unwrap_or("absolute") {
            "absolute" => ActionRepresentation::Absolute,
            "delta" => ActionRepresentation::Delta {
                absolute: cfg.get("absolute_dims").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default(),
            },
            other => return Err(Error::Config(format!("{HISTORY_NORMALIZER}: unknown action representation {other}"))),
        };
        let clip = cfg
            .get("normalization_clip")
            .or_else(|| raw.get("normalization_clip"))
            .and_then(Value::as_f64)
            .unwrap_or(6.0) as f32;
        let st = SafeTensors::open(&[existing(policy_dir.join(file))?])?;
        Ok(Some(Normalization { representation, state: quantiles(&st, "state", d)?, action: quantiles(&st, "action", d)?, clip }))
    }

    /// Load all three components.
    ///
    /// # Errors
    /// On malformed weights or mismatched components.
    pub fn load(&self, opts: LoadOptions) -> Result<ActionPolicy> {
        ActionPolicy::load(self.config.clone(), &self.dit_files, &self.vae_files, &self.text_dir, opts)
    }

    /// Load the transformer alone (staged inference).
    ///
    /// # Errors
    /// On malformed weights or a stream set that does not match the settings.
    pub fn load_transformer(&self, opts: LoadOptions) -> Result<Flux3Transformer> {
        let dit = Flux3Transformer::load(&self.dit_files, "dit.", opts)?;
        policy::check_transformer(&self.config, &dit)?;
        Ok(dit)
    }

    /// Load the video autoencoder alone (staged inference).
    ///
    /// # Errors
    /// On malformed weights.
    pub fn load_autoencoder(&self, opts: LoadOptions) -> Result<VideoVae> {
        VideoVae::load(&self.vae_files, "model.", VaeConfig::default(), opts)
    }

    /// Load the text encoder alone (staged inference).
    ///
    /// # Errors
    /// On missing or malformed files.
    pub fn load_text_encoder(&self, opts: LoadOptions) -> Result<TextEncoder> {
        TextEncoder::load(&self.text_dir, &CONTEXT_LAYERS, opts)
    }

    /// Load all three components with the release's statistics.
    ///
    /// # Errors
    /// On malformed weights or mismatched components.
    pub fn load_policy(self, opts: LoadOptions) -> Result<ReleasedPolicy> {
        let policy = self.load(opts)?;
        Ok(ReleasedPolicy { policy, release: self })
    }

    /// Frames per camera an observation carries.
    #[must_use]
    pub fn observation_frames(&self) -> usize {
        match self.config.conditioning {
            Conditioning::Frame => 1,
            Conditioning::History => self.config.n_obs_steps,
        }
    }
}

/// An observation in the robot's own units.
#[derive(Debug, Clone, Copy)]
pub struct RawObservation<'a> {
    /// `cameras[camera][time]`, in the release's camera order.
    pub cameras: &'a [Vec<Frame>],
    /// Measured state rows `[frames][action_dim]`.
    pub states: &'a [f32],
    /// Absolute command history `[frames][action_dim]`, for policies
    /// conditioned on past commands.
    pub commands: Option<&'a [f32]>,
    /// The instruction.
    pub instruction: &'a str,
}

/// A loaded released policy that maps raw observations to raw commands.
pub struct ReleasedPolicy {
    policy: ActionPolicy,
    release: PolicyRelease,
}

impl std::fmt::Debug for ReleasedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReleasedPolicy").field("config", &self.release.config).finish_non_exhaustive()
    }
}

impl ReleasedPolicy {
    /// The release.
    #[must_use]
    pub fn release(&self) -> &PolicyRelease {
        &self.release
    }

    /// The loaded policy.
    #[must_use]
    pub fn policy(&self) -> &ActionPolicy {
        &self.policy
    }

    /// One chunk of absolute commands `[chunk][action_dim]` for an
    /// observation, from the noise drawn with `seed`.
    ///
    /// # Errors
    /// On a malformed observation or a backend failure.
    pub fn act(&self, obs: &RawObservation<'_>, seed: u64) -> Result<Vec<f32>> {
        let cfg = &self.release.config;
        let d = cfg.action_dim;
        let frames = self.release.observation_frames();
        if obs.cameras.len() != self.release.camera_keys.len() {
            return Err(Error::Request(format!("the policy reads {} cameras", self.release.camera_keys.len())));
        }
        if obs.cameras.iter().any(|c| c.len() != frames) {
            return Err(Error::Request(format!("the policy reads {frames} frames per camera")));
        }
        if obs.states.len() != frames * d {
            return Err(Error::Request(format!("the policy reads {frames} state rows of {d} values")));
        }
        let wants_commands = cfg.conditioning == Conditioning::History && cfg.condition_on_past_actions;
        let commands = match (wants_commands, obs.commands) {
            (true, Some(c)) if c.len() == frames * d => Some(c),
            (true, _) => return Err(Error::Request(format!("the policy reads {frames} past command rows of {d} values"))),
            (false, _) => None,
        };
        let (states, past) = match &self.release.normalization {
            Some(n) => (n.states(obs.states), commands.map(|c| n.past_actions(c))),
            None => (obs.states.to_vec(), None),
        };
        let o = Observation { cameras: obs.cameras, states: &states, past_actions: past.as_deref(), instruction: obs.instruction };
        let chunk = self.policy.predict(&o, &Noise::from_seed(cfg, seed))?;
        Ok(match &self.release.normalization {
            Some(n) => {
                let anchor = commands.unwrap_or(obs.states);
                n.commands(&chunk, &anchor[anchor.len() - d..])
            }
            None => chunk,
        })
    }
}

/// Split an RGB grid image (rows = time steps oldest first, columns =
/// cameras, equal cells) into `cameras[camera][time]` frames.
///
/// # Errors
/// When the image does not divide into the grid.
pub fn frames_from_grid(width: usize, height: usize, rgb: &[u8], cameras: usize, frames: usize) -> Result<Vec<Vec<Frame>>> {
    if cameras == 0 || frames == 0 || rgb.len() != width * height * 3 || width % cameras != 0 || height % frames != 0 {
        return Err(Error::Request(format!("a {width}x{height} image is not a grid of {frames} rows by {cameras} cameras")));
    }
    let (cw, ch) = (width / cameras, height / frames);
    (0..cameras)
        .map(|c| {
            (0..frames)
                .map(|t| {
                    let mut px = Vec::with_capacity(cw * ch * 3);
                    for y in t * ch..(t + 1) * ch {
                        let row = (y * width + c * cw) * 3;
                        px.extend_from_slice(&rgb[row..row + cw * 3]);
                    }
                    Frame::from_u8(ch, cw, &px)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests;
