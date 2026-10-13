//! Vision-language-action policies: a vision-language backbone reads the
//! camera frames, the instruction and the robot state, and an action expert
//! integrates a flow-matching velocity field from noise to a chunk of future
//! actions. The chunk is executed open loop for a number of control ticks
//! before the next one is predicted (see [`rollout::ChunkRollout`]).
//!
//! The pieces every such policy shares live here: the observation and
//! policy interface, mean/deviation and range normalisation as the policy
//! releases store it, the sinusoidal flow-matching time embedding, the
//! forward-Euler sampler schedule and the aspect-preserving image resize.

use std::path::Path;

use crate::error::{Error, Result};
use crate::flux3::packing::Frame;
use crate::safetensors::SafeTensors;

pub mod groot;
pub mod groot_processor;
pub mod molmoact2;
pub mod pi05;
pub mod resample;
pub mod rollout;
pub mod siglip;
pub mod smolvla;

#[cfg(test)]
mod tests;

/// One observation as the robot reports it: a frame per camera (in the
/// policy's camera order), the measured state and the task instruction.
#[derive(Debug, Clone, Copy)]
pub struct VlaObservation<'a> {
    /// One frame per camera, pixels in `[0, 1]`.
    pub cameras: &'a [Frame],
    /// Measured state, `state_dim` values, in robot units.
    pub state: &'a [f32],
    /// Task instruction.
    pub instruction: &'a str,
}

/// A policy that predicts whole action chunks.
pub trait ChunkPolicy {
    /// Actions per predicted chunk.
    fn chunk_size(&self) -> usize;
    /// Values per action, in robot units.
    fn action_dim(&self) -> usize;
    /// Actions executed from each chunk before the next is predicted.
    fn n_action_steps(&self) -> usize;
    /// Values of initial noise one chunk consumes.
    fn noise_len(&self) -> usize;
    /// One chunk `[chunk_size][action_dim]` in robot units, integrated from
    /// `noise` (`noise_len` standard-normal values).
    ///
    /// # Errors
    /// On a malformed observation or a backend failure.
    fn predict(&self, obs: &VlaObservation<'_>, noise: &[f32]) -> Result<Vec<f32>>;
}

/// How a feature is normalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    /// Passed through.
    Identity,
    /// `(x - mean) / (std + eps)`.
    MeanStd,
    /// `2 (x - min) / (max - min + eps) - 1`.
    MinMax,
    /// `2 (x - q01) / (q99 - q01 + eps) - 1`, the 1st and 99th percentiles.
    Quantiles,
    /// As [`NormMode::Quantiles`] with the 10th and 90th percentiles.
    Quantile10,
}

impl NormMode {
    /// The mode named as in a policy config.
    ///
    /// # Errors
    /// [`Error::Config`] for a name this engine does not implement.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "IDENTITY" => Ok(Self::Identity),
            "MEAN_STD" => Ok(Self::MeanStd),
            "MIN_MAX" => Ok(Self::MinMax),
            "QUANTILES" => Ok(Self::Quantiles),
            "QUANTILE10" => Ok(Self::Quantile10),
            other => Err(Error::Config(format!("normalisation mode {other:?} is not implemented"))),
        }
    }
}

/// Statistics of one feature and how they apply.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureNorm {
    /// Mode.
    pub mode: NormMode,
    /// Offset per value (mean, min or 1st percentile).
    pub lo: Vec<f32>,
    /// Scale per value (deviation, max or 99th percentile).
    pub hi: Vec<f32>,
    /// Added to the denominator.
    pub eps: f32,
}

impl FeatureNorm {
    /// Read the statistics of `feature` from a statistics file. A feature
    /// the file holds no statistics for at all passes through unchanged, as
    /// the reference processor treats it (some releases keep their
    /// statistics under other names, e.g. per robot).
    ///
    /// # Errors
    /// When the feature has statistics but not the ones the mode needs.
    pub fn read(stats: &SafeTensors, feature: &str, mode: NormMode, eps: f32) -> Result<Self> {
        let prefix = format!("{feature}.");
        if !stats.names().any(|n| n.starts_with(&prefix)) {
            return Ok(Self { mode: NormMode::Identity, lo: Vec::new(), hi: Vec::new(), eps });
        }
        let (lo, hi) = match mode {
            NormMode::Identity => return Ok(Self { mode, lo: Vec::new(), hi: Vec::new(), eps }),
            NormMode::MeanStd => ("mean", "std"),
            NormMode::MinMax => ("min", "max"),
            NormMode::Quantiles => ("q01", "q99"),
            NormMode::Quantile10 => ("q10", "q90"),
        };
        let get = |s: &str| {
            let name = format!("{feature}.{s}");
            stats.get(&name).map(|v| v.to_f32()).ok_or(Error::MissingTensor(name))
        };
        let (lo, hi) = (get(lo)?, get(hi)?);
        if lo.len() != hi.len() {
            return Err(Error::Weights(format!("{feature}: statistics of different lengths")));
        }
        Ok(Self { mode, lo, hi, eps })
    }

    fn check(&self, x: &[f32]) -> Result<()> {
        if self.mode != NormMode::Identity && x.len() % self.lo.len().max(1) != 0 {
            return Err(Error::Request(format!("{} values do not divide into rows of {}", x.len(), self.lo.len())));
        }
        Ok(())
    }

    /// Robot units to the policy's normalised units (rows of the feature's
    /// width).
    ///
    /// # Errors
    /// When the values do not divide into rows.
    pub fn normalize(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.check(x)?;
        let d = self.lo.len().max(1);
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let (lo, hi) = (self.lo.get(i % d).copied().unwrap_or(0.0), self.hi.get(i % d).copied().unwrap_or(1.0));
                match self.mode {
                    NormMode::Identity => v,
                    NormMode::MeanStd => (v - lo) / (hi + self.eps),
                    NormMode::MinMax | NormMode::Quantiles | NormMode::Quantile10 => {
                        let den = hi - lo;
                        let den = if den == 0.0 { self.eps } else { den };
                        2.0 * (v - lo) / den - 1.0
                    }
                }
            })
            .collect())
    }

    /// Normalised units back to robot units.
    ///
    /// # Errors
    /// When the values do not divide into rows.
    pub fn unnormalize(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.check(x)?;
        let d = self.lo.len().max(1);
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let (lo, hi) = (self.lo.get(i % d).copied().unwrap_or(0.0), self.hi.get(i % d).copied().unwrap_or(1.0));
                match self.mode {
                    NormMode::Identity => v,
                    NormMode::MeanStd => v * hi + lo,
                    NormMode::MinMax | NormMode::Quantiles | NormMode::Quantile10 => {
                        let den = hi - lo;
                        let den = if den == 0.0 { self.eps } else { den };
                        (v + 1.0) / 2.0 * den + lo
                    }
                }
            })
            .collect())
    }
}

/// State and action normalisation of a policy release.
#[derive(Debug, Clone)]
pub struct Normalization {
    /// Applied to the measured state.
    pub state: FeatureNorm,
    /// Inverted on the predicted actions.
    pub action: FeatureNorm,
}

impl Normalization {
    /// Read from a release directory: the mode of each feature type from
    /// `config.json` (`normalization_mapping`), the statistics from the
    /// preprocessor's normalizer step state file.
    ///
    /// # Errors
    /// When the release names no statistics file or misses a statistic.
    pub fn from_release(dir: &Path, config: &serde_json::Value) -> Result<Self> {
        let mapping = config.get("normalization_mapping").ok_or_else(|| Error::Config("config gives no normalization_mapping".into()))?;
        let mode = |k: &str| -> Result<NormMode> {
            mapping.get(k).and_then(serde_json::Value::as_str).map_or(Ok(NormMode::Identity), NormMode::parse)
        };
        let pre: serde_json::Value = read_json(&dir.join("policy_preprocessor.json"))?;
        let step = pre
            .get("steps")
            .and_then(serde_json::Value::as_array)
            .and_then(|s| s.iter().find(|s| s.get("registry_name").and_then(serde_json::Value::as_str) == Some("normalizer_processor")))
            .ok_or_else(|| Error::Config("the preprocessor has no normalizer step".into()))?;
        let file = step
            .get("state_file")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::Config("the normalizer step names no state file".into()))?;
        let eps = step.get("config").and_then(|c| c.get("eps")).and_then(serde_json::Value::as_f64).unwrap_or(1e-8) as f32;
        let stats = SafeTensors::open(&[dir.join(file)])?;
        Ok(Self {
            state: FeatureNorm::read(&stats, "observation.state", mode("STATE")?, eps)?,
            action: FeatureNorm::read(&stats, "action", mode("ACTION")?, eps)?,
        })
    }
}

pub(crate) fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
}

/// Sine-cosine embedding of a flow-matching time `t` over `dim` channels
/// (sines then cosines), periods spaced geometrically from `min_period` to
/// `max_period`, computed in f64 and rounded once.
#[must_use]
pub fn sinusoidal_time(t: f32, dim: usize, min_period: f64, max_period: f64) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; 2 * half];
    for i in 0..half {
        let fraction = if half > 1 { i as f64 / (half - 1) as f64 } else { 0.0 };
        let period = min_period * (max_period / min_period).powf(fraction);
        let x = f64::from(t) * (1.0 / period * 2.0 * std::f64::consts::PI);
        out[i] = x.sin() as f32;
        out[half + i] = x.cos() as f32;
    }
    out
}

/// Forward-Euler times from `t = 1` (noise) to `t = 0` (actions) in
/// `steps` equal steps: the time at each step and the (negative) step.
#[must_use]
pub fn euler_times(steps: usize) -> (Vec<f32>, f32) {
    let dt = -1.0 / steps as f64;
    ((0..steps).map(|s| (1.0 + s as f64 * dt) as f32).collect(), dt as f32)
}

/// Bilinear resize of a channel-major frame with half-pixel centres and
/// edge clamping (no antialiasing), as `interpolate(.., "bilinear",
/// align_corners=False)` computes it.
#[must_use]
pub fn resize_bilinear(f: &Frame, h: usize, w: usize) -> Frame {
    if f.h == h && f.w == w {
        return f.clone();
    }
    let axis = |n_in: usize, n_out: usize| -> Vec<(usize, usize, f32)> {
        let scale = n_in as f32 / n_out as f32;
        (0..n_out)
            .map(|o| {
                let src = ((o as f32 + 0.5) * scale - 0.5).max(0.0);
                let i0 = (src as usize).min(n_in - 1);
                let i1 = (i0 + 1).min(n_in - 1);
                (i0, i1, src - i0 as f32)
            })
            .collect()
    };
    let (ys, xs) = (axis(f.h, h), axis(f.w, w));
    let mut data = vec![0f32; 3 * h * w];
    for c in 0..3 {
        let src = &f.data[c * f.h * f.w..(c + 1) * f.h * f.w];
        for (y, &(y0, y1, ly)) in ys.iter().enumerate() {
            for (x, &(x0, x1, lx)) in xs.iter().enumerate() {
                let top = src[y0 * f.w + x0] * (1.0 - lx) + src[y0 * f.w + x1] * lx;
                let bot = src[y1 * f.w + x0] * (1.0 - lx) + src[y1 * f.w + x1] * lx;
                data[(c * h + y) * w + x] = top * (1.0 - ly) + bot * ly;
            }
        }
    }
    Frame { h, w, data }
}

/// Where [`resize_with_pad`] puts the padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadAt {
    /// All of it before the image (left and top).
    Start,
    /// Split evenly, the odd pixel after the image.
    Centre,
}

/// Resize keeping the aspect ratio so the frame fits `h x w`, then pad to
/// exactly `h x w` with `value`.
#[must_use]
pub fn resize_with_pad(f: &Frame, h: usize, w: usize, at: PadAt, value: f32) -> Frame {
    if f.h == h && f.w == w {
        return f.clone();
    }
    let ratio = (f.w as f64 / w as f64).max(f.h as f64 / h as f64);
    let (rh, rw) = ((f.h as f64 / ratio) as usize, (f.w as f64 / ratio) as usize);
    let r = resize_bilinear(f, rh, rw);
    let (ph, pw) = (h - rh, w - rw);
    let (top, left) = match at {
        PadAt::Start => (ph, pw),
        PadAt::Centre => (ph / 2, pw / 2),
    };
    let mut data = vec![value; 3 * h * w];
    for c in 0..3 {
        for y in 0..rh {
            let dst = (c * h + top + y) * w + left;
            data[dst..dst + rw].copy_from_slice(&r.data[(c * rh + y) * rw..(c * rh + y + 1) * rw]);
        }
    }
    Frame { h, w, data }
}

/// Pad a vector with zeros to `n` values (longer vectors are refused).
///
/// # Errors
/// When `x` is longer than `n`.
pub fn pad_to(x: &[f32], n: usize) -> Result<Vec<f32>> {
    if x.len() > n {
        return Err(Error::Request(format!("{} values exceed the policy's {n}", x.len())));
    }
    let mut v = x.to_vec();
    v.resize(n, 0.0);
    Ok(v)
}

/// Multi-head attention over `[head_dim, heads, tokens]` queries and
/// `[head_dim, kv_heads, kv_tokens]` keys and values (grouped heads share a
/// key/value head); result `[head_dim, heads, tokens]`. `exact` computes it
/// unfused in float32, otherwise fused with half-precision keys and values
/// and float32 scores.
pub(crate) fn attend(g: &mut crate::ggml::Graph, q: crate::ggml::Tn, k: crate::ggml::Tn, v: crate::ggml::Tn, mask: Option<crate::ggml::Tn>, scale: f32, exact: bool) -> crate::ggml::Tn {
    use llama_cpp_sys_2 as sys;
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    if exact {
        g.attention_exact(q, k, v, mask, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, mask, scale, true)
    }
}
