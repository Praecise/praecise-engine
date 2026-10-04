//! Joint denoising of several FLUX 3 streams (video latents and an action
//! chunk) that share one noise schedule.
//!
//! Two samplers: rectified-flow Euler on a rationally shifted linear schedule,
//! and second-order bh2 UniPC in clean-sample space on the shifted linear grid
//! with integer model ticks (one model evaluation per step; the corrector reuses
//! it). [`guide`] combines an unconditional and a conditional prediction per
//! stream so each stream can carry its own guidance scale.

use crate::error::{Error, Result};
use crate::unipc::{shifted_linear_schedule, UniPc};

/// Training timesteps of the UniPC grid.
pub const NUM_TRAIN_TIMESTEPS: u64 = 1000;

/// Which solver drives the denoising.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    /// First-order rectified-flow Euler.
    Euler,
    /// Second-order bh2 UniPC, predicting the clean sample.
    UniPc,
}

/// Solver, step count and schedule shift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerSettings {
    /// The solver.
    pub kind: SamplerKind,
    /// Denoising steps (model evaluations per guidance branch).
    pub steps: usize,
    /// Rational schedule shift.
    pub shift: f64,
}

/// `shift * t / (1 + (shift - 1) * t)` in single precision.
#[must_use]
pub fn rational_time_shift(t: f32, shift: f32) -> f32 {
    shift * t / (1.0 + (shift - 1.0) * t)
}

/// The Euler grid: `steps + 1` shifted times from 1 down to 0.
#[must_use]
pub fn euler_times(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps as f32;
    (0..=steps)
        .map(|i| {
            let t = if i == steps { 0.0 } else { 1.0 - i as f32 / n };
            rational_time_shift(t, shift)
        })
        .collect()
}

/// Denoise `samples` (one flat vector per stream) from pure noise to clean.
/// `predict(samples, t)` returns the velocity of every stream at model time
/// `t` in `[0, 1]`, in the same order and shapes.
///
/// # Errors
/// On zero steps, a prediction of the wrong shape, or a failed prediction.
pub fn sample<F>(mut samples: Vec<Vec<f32>>, settings: SamplerSettings, mut predict: F) -> Result<Vec<Vec<f32>>>
where
    F: FnMut(&[Vec<f32>], f32) -> Result<Vec<Vec<f32>>>,
{
    if settings.steps == 0 {
        return Err(Error::Request("sampler needs at least one step".into()));
    }
    let mut checked = |s: &[Vec<f32>], t: f32| -> Result<Vec<Vec<f32>>> {
        let v = predict(s, t)?;
        if v.len() != s.len() || v.iter().zip(s).any(|(a, b)| a.len() != b.len()) {
            return Err(Error::Request("prediction does not match the sampled streams".into()));
        }
        Ok(v)
    };
    match settings.kind {
        SamplerKind::Euler => {
            let times = euler_times(settings.steps, settings.shift as f32);
            for w in times.windows(2) {
                let (t_curr, t_prev) = (w[0], w[1]);
                let v = checked(&samples, t_curr)?;
                let dt = (f64::from(t_prev) - f64::from(t_curr)) as f32;
                for (x, v) in samples.iter_mut().zip(&v) {
                    for (x, v) in x.iter_mut().zip(v) {
                        *x += dt * v;
                    }
                }
            }
            Ok(samples)
        }
        SamplerKind::UniPc => {
            let (sigmas, ticks) = shifted_linear_schedule(settings.steps, settings.shift, NUM_TRAIN_TIMESTEPS);
            let mut solvers: Vec<UniPc> = samples.iter().map(|_| UniPc::new(sigmas.clone())).collect();
            for &tick in &ticks {
                let v = checked(&samples, tick as f32 / NUM_TRAIN_TIMESTEPS as f32)?;
                for ((solver, x), v) in solvers.iter_mut().zip(samples.iter_mut()).zip(&v) {
                    *x = solver.step(v, x)?;
                }
            }
            Ok(samples)
        }
    }
}

/// `uncond + scale * (cond - uncond)` per stream, with one scale per stream.
///
/// # Errors
/// When the stream counts or lengths differ.
pub fn guide(uncond: &[Vec<f32>], cond: &[Vec<f32>], scales: &[f32]) -> Result<Vec<Vec<f32>>> {
    if uncond.len() != cond.len() || cond.len() != scales.len() || uncond.iter().zip(cond).any(|(a, b)| a.len() != b.len()) {
        return Err(Error::Request("guidance branches do not match".into()));
    }
    Ok(uncond
        .iter()
        .zip(cond)
        .zip(scales)
        .map(|((u, c), &g)| u.iter().zip(c).map(|(u, c)| u + g * (c - u)).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_times_run_from_one_to_zero() {
        let t = euler_times(4, 6.93);
        assert_eq!(t.len(), 5);
        assert!((t[0] - 1.0).abs() < 1e-6);
        assert_eq!(t[4], 0.0);
        assert!((t[2] - rational_time_shift(0.5, 6.93)).abs() < 1e-7);
    }

    #[test]
    fn a_constant_velocity_reaches_its_endpoint_with_either_sampler() {
        // x_t = (1 - t) x0 + t eps has velocity eps - x0 everywhere.
        let x0 = [0.3f32, -1.2, 2.0];
        let eps = [1.0f32, 0.5, -0.7];
        let v: Vec<f32> = eps.iter().zip(&x0).map(|(e, x)| e - x).collect();
        for kind in [SamplerKind::Euler, SamplerKind::UniPc] {
            let start = vec![eps.to_vec(), eps.to_vec()];
            let out = sample(start, SamplerSettings { kind, steps: 4, shift: 5.0 }, |s, _| Ok(s.iter().map(|_| v.clone()).collect()))
                .unwrap();
            for stream in &out {
                for (o, x) in stream.iter().zip(&x0) {
                    // UniPC starts one tick below pure noise.
                    assert!((o - x).abs() < 2e-3 * (1.0 + x.abs()), "{kind:?}: {o} vs {x}");
                }
            }
        }
    }

    #[test]
    fn both_samplers_match_the_reference_on_a_state_dependent_velocity() {
        // Reference values from the original sampler on v = 0.8 x - 0.2 t x + 0.1.
        let cases = [
            (SamplerKind::Euler, 6.93, [0.136_663_3f32, -0.528_986_1, 0.802_312_6, 0.580_429_6]),
            (SamplerKind::UniPc, 5.0, [0.144_89, -0.536_044_1, 0.825_824_1, 0.598_846]),
        ];
        for (kind, shift, want) in cases {
            let start = vec![vec![0.5f32, -1.0, 2.0], vec![1.5]];
            let out = sample(start, SamplerSettings { kind, steps: 4, shift }, |s, t| {
                Ok(s.iter().map(|x| x.iter().map(|x| 0.8 * x - 0.2 * t * x + 0.1).collect()).collect())
            })
            .unwrap();
            let got: Vec<f32> = out.concat();
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 2e-6, "{kind:?}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn guidance_is_per_stream() {
        let g = guide(&[vec![1.0], vec![1.0]], &[vec![2.0], vec![2.0]], &[3.0, 1.0]).unwrap();
        assert_eq!(g, vec![vec![4.0], vec![2.0]]);
    }
}
