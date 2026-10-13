//! The GR00T N1.7 processor and control loop: what turns a robot's
//! observation into the policy's inputs and its prediction back into robot
//! actions, as the release's processor files describe it.
//!
//! A release carries `processor_config.json` (image geometry, percentile
//! or range normalisation, relative actions and, per embodiment, the camera,
//! state and action layout), `statistics.json` (per embodiment and group)
//! and `embodiment_id.json`. For each observation:
//!
//! - every camera frame is scaled to 8 bits, optionally padded to a square,
//!   resized so its shorter side is the release's edge, centre-cropped by
//!   the crop fraction and resized back (OpenCV area interpolation for
//!   releases trained with it, antialiased bicubic otherwise), then resized
//!   to multiples of the merged patch as the backbone's image processor
//!   does. The frames go to the prompt once per step of the camera history
//!   the embodiment reads (an observation carries the current frame, which
//!   stands in for the earlier ones, as the reference pads);
//! - the instruction is lower-cased and stripped to letters, digits, spaces
//!   and underscores, and goes after the images in one user turn of the
//!   backbone's chat format;
//! - the state is mapped to `[-1, 1]` by its 1st/99th percentiles (or
//!   range), clipped and zero-padded.
//!
//! The predicted chunk is cut to the embodiment's action steps and width;
//! each action group is mapped back from `[-1, 1]` by its statistics (per
//! step where the release keeps them per step). Relative groups are added to
//! the measured state they refer to (joints), or composed with it as poses
//! (end effector, position and 6-D rotation). The control loop executes the
//! first actions of each decoded chunk ([`super::rollout::ChunkRollout`]).

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use super::groot::GrootPolicy;
use super::resample::{resize_area, resize_bicubic_aa, Frame8};
use super::{read_json, ChunkPolicy, VlaObservation};
use crate::error::{Error, Result};
use crate::flux3::packing::Frame;
use crate::pipeline::LoadOptions;
use crate::qwen3_vl::Vl3Image;

/// The image geometry of a release.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageGeometry {
    /// Final size `(h, w)` (`None`: frames pass unchanged).
    pub target: Option<(usize, usize)>,
    /// Crop size, giving the crop fraction when none is set.
    pub crop: Option<(usize, usize)>,
    /// Shorter side frames are resized to (default: the target height).
    pub shortest_edge: Option<usize>,
    /// Fraction of each side the centre crop keeps.
    pub crop_fraction: Option<f64>,
    /// OpenCV area interpolation keeping the aspect ratio (otherwise
    /// antialiased bicubic to a square).
    pub area: bool,
    /// Pad frames to a square first.
    pub letter_box: bool,
}

/// Python's `round` (half to even) of a non-negative value.
fn round_even(v: f64) -> usize {
    v.round_ties_even() as usize
}

impl ImageGeometry {
    fn crop_fraction(&self, target_h: usize) -> Option<f64> {
        self.crop_fraction.or_else(|| self.crop.map(|(h, _)| h as f64 / target_h as f64)).filter(|f| *f > 0.0 && *f < 1.0)
    }

    /// A frame as the release's image transform leaves it.
    #[must_use]
    pub fn apply(&self, f: &Frame8) -> Frame8 {
        let Some((th, tw)) = self.target else { return f.clone() };
        let mut img = f.clone();
        if self.letter_box && img.h != img.w {
            let edge = img.h.max(img.w);
            img = img.pad((edge - img.h) / 2, (edge - img.w) / 2, edge, edge);
        }
        let edge = self.shortest_edge.unwrap_or(th);
        let fraction = self.crop_fraction(th);
        if self.area {
            let shortest = |i: &Frame8| {
                let s = i.h.min(i.w);
                if s == edge {
                    return i.clone();
                }
                let scale = edge as f64 / s as f64;
                resize_area(i, round_even(i.h as f64 * scale).max(1), round_even(i.w as f64 * scale).max(1))
            };
            img = shortest(&img);
            if let Some(cf) = fraction {
                let (ch, cw) = (((img.h as f64 * cf) as usize).max(1), ((img.w as f64 * cf) as usize).max(1));
                img = img.crop((img.h - ch) / 2, (img.w - cw) / 2, ch, cw);
            }
            return shortest(&img);
        }
        img = resize_bicubic_aa(&img, edge, edge);
        if let Some(cf) = fraction {
            let (ch, cw) = (round_even(img.h as f64 * cf).max(1), round_even(img.w as f64 * cf).max(1));
            img = img.crop((img.h - ch) / 2, (img.w - cw) / 2, ch, cw);
        }
        if (img.h, img.w) != (th, tw) {
            img = resize_bicubic_aa(&img, th, tw);
        }
        img
    }
}

/// The size the backbone's image processor resizes an `h x w` image to:
/// multiples of `factor`, the pixel count within `[min, max]`, the aspect
/// ratio kept as closely as possible.
///
/// # Errors
/// For an aspect ratio beyond 200.
pub fn smart_resize(h: usize, w: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    let (hf, wf, ff) = (h as f64, w as f64, factor as f64);
    if hf.max(wf) / hf.min(wf) > 200.0 {
        return Err(Error::Request(format!("image {w}x{h} is too elongated")));
    }
    let (mut hb, mut wb) = (round_even(hf / ff) * factor, round_even(wf / ff) * factor);
    if hb * wb > max_pixels {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = factor.max((hf / beta / ff).floor() as usize * factor);
        wb = factor.max((wf / beta / ff).floor() as usize * factor);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / ff).ceil() as usize * factor;
        wb = (wf * beta / ff).ceil() as usize * factor;
    }
    Ok((hb, wb))
}

/// The instruction as the release formats it: an empty one becomes the
/// default task; with `formalize`, lower case keeping only letters, digits,
/// whitespace and underscores.
#[must_use]
pub fn format_instruction(instruction: &str, formalize: bool) -> String {
    let text = if instruction.is_empty() { "Perform the task." } else { instruction };
    if !formalize {
        return text.to_owned();
    }
    text.to_lowercase().chars().filter(|&c| c.is_alphanumeric() || c.is_whitespace() || ('\x1c'..='\x1f').contains(&c) || c == '_').collect()
}

/// How an action group maps back to robot units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionRep {
    /// The values themselves.
    Absolute,
    /// Offsets added to the state values at `state` (`dim` of them).
    RelativeJoint {
        /// Offset of the referenced state group.
        state: usize,
    },
    /// A pose (position and 6-D rotation) relative to the pose at `state`.
    RelativePose {
        /// Offset of the referenced state group.
        state: usize,
    },
}

/// One action group of an embodiment.
#[derive(Debug, Clone)]
pub struct ActionGroup {
    /// Group name.
    pub key: String,
    /// Values per action.
    pub dim: usize,
    /// Lower statistic, `[rows][dim]`.
    pub lo: Vec<f32>,
    /// Upper statistic, `[rows][dim]`.
    pub hi: Vec<f32>,
    /// Rows of statistics: one, or one per chunk step.
    pub rows: usize,
    /// Representation.
    pub rep: ActionRep,
}

fn stat_values(entry: &Value, name: &str, what: &str) -> Result<(Vec<f32>, usize)> {
    let v = entry.get(name).and_then(Value::as_array).ok_or_else(|| Error::Config(format!("statistics give no {name} for {what}")))?;
    let num = |x: &Value| x.as_f64().map(|f| f as f32).ok_or_else(|| Error::Config(format!("{what}.{name} is not numeric")));
    if v.first().is_some_and(Value::is_array) {
        let mut out = Vec::new();
        for row in v {
            for x in row.as_array().ok_or_else(|| Error::Config(format!("{what}.{name} is ragged")))? {
                out.push(num(x)?);
            }
        }
        return Ok((out, v.len()));
    }
    Ok((v.iter().map(num).collect::<Result<_>>()?, 1))
}

fn stat_dim(entry: &Value) -> usize {
    ["mean", "q01", "min", "max", "std"]
        .iter()
        .find_map(|k| entry.get(*k).and_then(Value::as_array))
        .map_or(0, |v| v.first().and_then(Value::as_array).map_or(v.len(), Vec::len))
}

fn keys(v: &Value, what: &str) -> Result<Vec<String>> {
    v.get("modality_keys")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config(format!("the {what} layout names no modality_keys")))?
        .iter()
        .map(|k| k.as_str().map(str::to_owned).ok_or_else(|| Error::Config(format!("a {what} key is not a string"))))
        .collect()
}

fn lower(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_lowercase()
}

fn pair(v: Option<&Value>) -> Option<(usize, usize)> {
    let a = v?.as_array()?;
    match a.as_slice() {
        [h, w] => Some((h.as_u64()? as usize, w.as_u64()? as usize)),
        _ => None,
    }
}

/// The processor of one embodiment of a release.
pub struct GrootProcessor {
    tokenizer: Tokenizer,
    geometry: ImageGeometry,
    unit: usize,
    min_pixels: usize,
    max_pixels: usize,
    cameras: Vec<String>,
    history: usize,
    formalize: bool,
    clip: bool,
    state_lo: Vec<f32>,
    state_hi: Vec<f32>,
    max_state_dim: usize,
    groups: Vec<ActionGroup>,
    action_steps: usize,
    execution_steps: usize,
    embodiment: usize,
}

impl std::fmt::Debug for GrootProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrootProcessor").field("cameras", &self.cameras).field("embodiment", &self.embodiment).finish_non_exhaustive()
    }
}

impl GrootProcessor {
    /// Read the processor files of `release` for embodiment `tag`, with the
    /// backbone's `tokenizer.json` and `preprocessor_config.json` from
    /// `backbone`, for images in multiples of `unit` pixels and states padded
    /// to `max_state_dim`.
    ///
    /// # Errors
    /// On a missing or malformed file, an embodiment the release does not
    /// describe, or an action representation this engine does not implement.
    pub fn load(release: &Path, backbone: &Path, tag: &str, unit: usize, max_state_dim: usize) -> Result<Self> {
        let pc = read_json(&release.join("processor_config.json"))?;
        let kw = pc.get("processor_kwargs").cloned().unwrap_or(Value::Null);
        let flag = |k: &str, d: bool| kw.get(k).and_then(Value::as_bool).unwrap_or(d);
        let modality = kw
            .get("modality_configs")
            .and_then(|m| m.get(tag))
            .ok_or_else(|| Error::Config(format!("the release describes no embodiment {tag:?}")))?;
        let all = read_json(&release.join("statistics.json"))?;
        let stats = all.get(tag).ok_or_else(|| Error::Config(format!("statistics.json has nothing for {tag:?}")))?;
        let ids = read_json(&release.join("embodiment_id.json"))?;
        let embodiment = ids.get(tag).and_then(Value::as_u64).ok_or_else(|| Error::Config(format!("embodiment_id.json has no {tag:?}")))? as usize;

        let geometry = match pair(kw.get("image_target_size")) {
            Some(target) => ImageGeometry {
                target: Some(target),
                crop: pair(kw.get("image_crop_size")),
                shortest_edge: kw.get("shortest_image_edge").and_then(Value::as_u64).map(|v| v as usize),
                crop_fraction: kw.get("crop_fraction").and_then(Value::as_f64),
                area: flag("use_albumentations", false),
                letter_box: flag("letter_box_transform", false),
            },
            None => ImageGeometry {
                target: Some((256, 256)),
                crop: Some((230, 230)),
                shortest_edge: None,
                crop_fraction: None,
                area: flag("use_albumentations", false),
                letter_box: flag("letter_box_transform", false),
            },
        };
        let percentiles = flag("use_percentiles", false);
        let relative = flag("use_relative_action", false);
        let (lo_name, hi_name) = if percentiles { ("q01", "q99") } else { ("min", "max") };

        let video = modality.get("video").ok_or_else(|| Error::Config("the embodiment has no video layout".into()))?;
        let cameras = keys(video, "video")?;
        let history = video.get("delta_indices").and_then(Value::as_array).map_or(1, |d| d.len().max(1));

        let mut state_lo = Vec::new();
        let mut state_hi = Vec::new();
        let mut state_at = std::collections::HashMap::new();
        for k in keys(modality.get("state").unwrap_or(&Value::Null), "state")? {
            let e = stats.get("state").and_then(|s| s.get(&k)).ok_or_else(|| Error::Config(format!("no statistics for state {k}")))?;
            state_at.insert(k.clone(), state_lo.len());
            state_lo.extend(stat_values(e, lo_name, &k)?.0);
            state_hi.extend(stat_values(e, hi_name, &k)?.0);
        }
        if state_lo.len() > max_state_dim {
            return Err(Error::Config(format!("the state has {} values, the policy reads {max_state_dim}", state_lo.len())));
        }

        let action = modality.get("action").ok_or_else(|| Error::Config("the embodiment has no action layout".into()))?;
        let configs = action.get("action_configs").and_then(Value::as_array).cloned().unwrap_or_default();
        let action_steps = action.get("delta_indices").and_then(Value::as_array).map_or(0, Vec::len);
        let mut groups = Vec::new();
        for (i, k) in keys(action, "action")?.into_iter().enumerate() {
            let e = stats.get("action").and_then(|s| s.get(&k)).ok_or_else(|| Error::Config(format!("no statistics for action {k}")))?;
            let dim = stat_dim(e);
            let cfg = configs.get(i).cloned().unwrap_or(Value::Null);
            let is_rel = relative && lower(&cfg["rep"]) == "relative";
            let (lo, hi, rows) = if is_rel {
                let r = stats.get("relative_action").and_then(|s| s.get(&k)).ok_or_else(|| Error::Config(format!("no relative statistics for {k}")))?;
                let (lo, rows) = stat_values(r, "min", &k)?;
                (lo, stat_values(r, "max", &k)?.0, rows)
            } else {
                let (lo, rows) = stat_values(e, lo_name, &k)?;
                (lo, stat_values(e, hi_name, &k)?.0, rows)
            };
            if lo.len() != rows * dim || hi.len() != lo.len() {
                return Err(Error::Config(format!("statistics of {k} do not match its width {dim}")));
            }
            let rep = if is_rel {
                let sk = cfg.get("state_key").and_then(Value::as_str).unwrap_or(&k);
                let state = *state_at.get(sk).ok_or_else(|| Error::Config(format!("{k} refers to an unknown state {sk}")))?;
                match (lower(&cfg["type"]).as_str(), lower(&cfg["format"]).replace('_', "+").as_str()) {
                    ("non_eef", _) => ActionRep::RelativeJoint { state },
                    ("eef", "xyz+rot6d") => ActionRep::RelativePose { state },
                    other => return Err(Error::Config(format!("relative action {k} as {other:?} is not implemented"))),
                }
            } else {
                ActionRep::Absolute
            };
            groups.push(ActionGroup { key: k, dim, lo, hi, rows, rep });
        }
        // The reference control loop for the LIBERO embodiment replans after
        // eight steps.
        let execution_steps = if tag == "libero_sim" { action_steps.min(8) } else { action_steps };

        let tok_path = backbone.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        let ip = read_json(&backbone.join("preprocessor_config.json"))?;
        let size = |k: &str, alt: &str| ip.get("size").and_then(|s| s.get(k)).or_else(|| ip.get(alt)).and_then(Value::as_u64).map(|v| v as usize);
        let min_pixels = size("shortest_edge", "min_pixels").ok_or_else(|| Error::Config("the image processor gives no minimum size".into()))?;
        let max_pixels = size("longest_edge", "max_pixels").ok_or_else(|| Error::Config("the image processor gives no maximum size".into()))?;
        Ok(Self {
            tokenizer,
            geometry,
            unit,
            min_pixels,
            max_pixels,
            cameras,
            history,
            formalize: flag("formalize_language", true),
            clip: flag("clip_outliers", true),
            state_lo,
            state_hi,
            max_state_dim,
            groups,
            action_steps,
            execution_steps,
            embodiment,
        })
    }

    /// The embodiment's index among the policy's per-embodiment weights.
    #[must_use]
    pub fn embodiment(&self) -> usize {
        self.embodiment
    }

    /// Camera names, in the order observations give the frames.
    #[must_use]
    pub fn cameras(&self) -> &[String] {
        &self.cameras
    }

    /// Steps of camera history each prompt carries.
    #[must_use]
    pub fn history(&self) -> usize {
        self.history
    }

    /// The image transform.
    #[must_use]
    pub fn geometry(&self) -> &ImageGeometry {
        &self.geometry
    }

    /// Action groups, in action order.
    #[must_use]
    pub fn groups(&self) -> &[ActionGroup] {
        &self.groups
    }

    /// Measured state values.
    #[must_use]
    pub fn state_dim(&self) -> usize {
        self.state_lo.len()
    }

    /// Values per decoded action.
    #[must_use]
    pub fn action_dim(&self) -> usize {
        self.groups.iter().map(|g| g.dim).sum()
    }

    /// Action steps the embodiment predicts.
    #[must_use]
    pub fn action_steps(&self) -> usize {
        self.action_steps
    }

    /// Action steps the reference control loop executes per chunk.
    #[must_use]
    pub fn execution_steps(&self) -> usize {
        self.execution_steps
    }

    /// A camera frame as the backbone reads it: 8 bits, transformed, at
    /// multiples of the merged patch.
    ///
    /// # Errors
    /// For a degenerate frame.
    pub fn camera(&self, f: &Frame) -> Result<Frame8> {
        let img = self.geometry.apply(&Frame8::from_unit(f));
        let (h, w) = smart_resize(img.h, img.w, self.unit, self.min_pixels, self.max_pixels)?;
        Ok(resize_bicubic_aa(&img, h, w))
    }

    /// Prompt tokens for `images` images and an instruction, each image as
    /// one placeholder.
    ///
    /// # Errors
    /// When the tokenizer fails.
    pub fn tokens(&self, instruction: &str, images: usize) -> Result<Vec<u32>> {
        let text = format_instruction(instruction, self.formalize);
        let prompt = format!("<|im_start|>user\n{}{text}<|im_end|>\n", "<|vision_start|><|image_pad|><|vision_end|>".repeat(images));
        let enc = self.tokenizer.encode(prompt, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// The state mapped to `[-1, 1]` by its statistics (a constant value
    /// maps to 0), clipped, zero-padded to the policy's width.
    ///
    /// # Errors
    /// For a state of the wrong width.
    pub fn state(&self, state: &[f32]) -> Result<Vec<f32>> {
        if state.len() != self.state_lo.len() {
            return Err(Error::Request(format!("the embodiment's state has {} values, got {}", self.state_lo.len(), state.len())));
        }
        let mut out: Vec<f32> = state
            .iter()
            .zip(self.state_lo.iter().zip(&self.state_hi))
            .map(|(&x, (&lo, &hi))| {
                let den = hi - lo;
                let v = if den == 0.0 { 0.0 } else { 2.0 * (x - lo) / den - 1.0 };
                if self.clip {
                    v.clamp(-1.0, 1.0)
                } else {
                    v
                }
            })
            .collect();
        out.resize(self.max_state_dim, 0.0);
        Ok(out)
    }

    /// The first `steps` actions of a predicted chunk (`[horizon][width]`
    /// in the policy's units, `width` at least the embodiment's) in robot
    /// units, relative groups resolved against the measured `state`.
    ///
    /// # Errors
    /// When the chunk is too short or narrow, or per-step statistics run
    /// out.
    pub fn decode(&self, chunk: &[f32], width: usize, steps: usize, state: &[f32]) -> Result<Vec<f32>> {
        let d = self.action_dim();
        if width < d || chunk.len() < steps * width || state.len() != self.state_lo.len() {
            return Err(Error::Request(format!("a chunk of {} values does not hold {steps} actions of {d}", chunk.len())));
        }
        let mut out = vec![0f32; steps * d];
        let mut at = 0;
        for g in &self.groups {
            if g.rows > 1 && steps > g.rows {
                return Err(Error::Config(format!("{} has statistics for {} steps, not {steps}", g.key, g.rows)));
            }
            for t in 0..steps {
                let row = if g.rows > 1 { t } else { 0 };
                let (lo, hi) = (&g.lo[row * g.dim..(row + 1) * g.dim], &g.hi[row * g.dim..(row + 1) * g.dim]);
                let v: Vec<f32> = (0..g.dim).map(|j| (chunk[t * width + at + j].clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi[j] - lo[j]) + lo[j]).collect();
                let dst = &mut out[t * d + at..t * d + at + g.dim];
                match g.rep {
                    ActionRep::Absolute => dst.copy_from_slice(&v),
                    ActionRep::RelativeJoint { state: s } => {
                        for (j, o) in dst.iter_mut().enumerate() {
                            *o = v[j] + state[s + j];
                        }
                    }
                    ActionRep::RelativePose { state: s } => {
                        if g.dim != 9 {
                            return Err(Error::Config(format!("{} is not a 9-value pose", g.key)));
                        }
                        dst.copy_from_slice(&compose_pose(&state[s..s + 9], &v));
                    }
                }
            }
            at += g.dim;
        }
        Ok(out)
    }
}

fn rot6d_matrix(r: &[f64]) -> [[f64; 3]; 3] {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let a = [r[0], r[1], r[2]];
    let n = norm(a);
    let r1 = [a[0] / n, a[1] / n, a[2] / n];
    let b = [r[3], r[4], r[5]];
    let dot = r1[0] * b[0] + r1[1] * b[1] + r1[2] * b[2];
    let r2 = [b[0] - dot * r1[0], b[1] - dot * r1[1], b[2] - dot * r1[2]];
    let n2 = norm(r2);
    let r2 = [r2[0] / n2, r2[1] / n2, r2[2] / n2];
    let r3 = [r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]];
    [r1, r2, r3]
}

/// The pose `relative` (position and 6-D rotation, the first two rows of
/// the rotation) applied after `reference`, in float64.
pub(crate) fn compose_pose(reference: &[f32], relative: &[f32]) -> Vec<f32> {
    let homogeneous = |p: &[f32]| {
        let p: Vec<f64> = p.iter().map(|&v| f64::from(v)).collect();
        let r = rot6d_matrix(&p[3..9]);
        let mut m = [[0f64; 4]; 4];
        for i in 0..3 {
            m[i][..3].copy_from_slice(&r[i]);
            m[i][3] = p[i];
        }
        m[3][3] = 1.0;
        m
    };
    let (a, b) = (homogeneous(reference), homogeneous(relative));
    let mut m = [[0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    [m[0][3], m[1][3], m[2][3], m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]].iter().map(|&v| v as f32).collect()
}

/// The model inputs of one observation.
#[derive(Debug, Clone)]
pub struct GrootInputs {
    /// Prompt tokens, one placeholder per image.
    pub tokens: Vec<u32>,
    /// Images in prompt order (history steps, then cameras).
    pub images: Vec<Vl3Image>,
    /// Normalised, padded state.
    pub state: Vec<f32>,
}

/// A GR00T N1.7 release with its processor, as a [`ChunkPolicy`] in robot
/// units.
pub struct GrootChunkPolicy {
    policy: GrootPolicy,
    processor: GrootProcessor,
    n_action_steps: usize,
}

impl std::fmt::Debug for GrootChunkPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrootChunkPolicy").field("processor", &self.processor).field("n_action_steps", &self.n_action_steps).finish_non_exhaustive()
    }
}

impl GrootChunkPolicy {
    /// Load `release` (weights, `config.json` and the processor files) for
    /// embodiment `tag`, with the backbone's `config.json`, tokenizer and
    /// image processor settings from `backbone`.
    ///
    /// # Errors
    /// As [`GrootProcessor::load`] and [`GrootPolicy::load`].
    pub fn load(release: &Path, backbone: &Path, tag: &str, opts: LoadOptions) -> Result<Self> {
        let pc = read_json(&release.join("config.json"))?;
        let max_state_dim = pc.get("max_state_dim").and_then(Value::as_u64).ok_or_else(|| Error::Config("config gives no max_state_dim".into()))? as usize;
        let bc = crate::qwen3_vl::Qwen3VlConfig::from_json(read_json(&backbone.join("config.json"))?)?;
        let processor = GrootProcessor::load(release, backbone, tag, bc.image_unit(), max_state_dim)?;
        let policy = GrootPolicy::load(release, &backbone.join("config.json"), processor.embodiment(), opts)?;
        let n_action_steps = processor.execution_steps();
        Ok(Self { policy, processor, n_action_steps })
    }

    /// The model.
    #[must_use]
    pub fn policy(&self) -> &GrootPolicy {
        &self.policy
    }

    /// The processor.
    #[must_use]
    pub fn processor(&self) -> &GrootProcessor {
        &self.processor
    }

    /// As [`GrootPolicy::set_fp32_gemm`].
    pub fn set_fp32_gemm(&mut self, fp32: bool) {
        self.policy.set_fp32_gemm(fp32);
    }

    /// Execute `n` actions of each chunk (at most the embodiment's).
    pub fn set_n_action_steps(&mut self, n: usize) {
        self.n_action_steps = n.clamp(1, self.processor.execution_steps().max(1));
    }

    /// The model inputs of an observation.
    ///
    /// # Errors
    /// For the wrong number of cameras or state values, or a tokenizer
    /// failure.
    pub fn inputs(&self, obs: &VlaObservation<'_>) -> Result<GrootInputs> {
        let p = &self.processor;
        if obs.cameras.len() != p.cameras().len() {
            return Err(Error::Request(format!("the embodiment reads {} cameras, got {}", p.cameras().len(), obs.cameras.len())));
        }
        let frames = obs
            .cameras
            .iter()
            .map(|f| {
                let c = p.camera(f)?;
                self.policy.encoder().image_bytes(&c.data, (c.h, c.w))
            })
            .collect::<Result<Vec<_>>>()?;
        let images: Vec<Vl3Image> = (0..p.history()).flat_map(|_| frames.iter().cloned()).collect();
        Ok(GrootInputs { tokens: p.tokens(obs.instruction, images.len())?, images, state: p.state(obs.state)? })
    }
}

impl ChunkPolicy for GrootChunkPolicy {
    fn chunk_size(&self) -> usize {
        self.policy.head().config().horizon.min(self.processor.action_steps()).min(self.n_action_steps)
    }
    fn action_dim(&self) -> usize {
        self.processor.action_dim()
    }
    fn n_action_steps(&self) -> usize {
        self.n_action_steps
    }
    fn noise_len(&self) -> usize {
        let c = self.policy.head().config();
        c.horizon * c.max_action_dim
    }
    fn predict(&self, obs: &VlaObservation<'_>, noise: &[f32]) -> Result<Vec<f32>> {
        let inputs = self.inputs(obs)?;
        let raw = self.policy.predict_images(&inputs.tokens, &inputs.images, &inputs.state, noise)?;
        self.processor.decode(&raw, self.policy.head().config().max_action_dim, self.chunk_size(), obs.state)
    }
}
