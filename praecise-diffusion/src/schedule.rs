//! Flow-matching noise schedule and seeded initial noise.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Resolution- and step-dependent shift for the FLUX.2 schedule, fitted by
/// the model authors on their training resolutions.
#[must_use]
pub fn empirical_mu(image_seq_len: usize, steps: usize) -> f64 {
    let (a1, b1) = (8.738_095_24e-05, 1.898_333_33);
    let (a2, b2) = (0.000_169_27, 0.456_666_66);
    let len = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * len + b2;
    }
    let m_200 = a2 * len + b2;
    let m_10 = a1 * len + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * steps as f64 + b
}

/// Noise levels for `steps` Euler steps, ending with the terminal 0:
/// `steps + 1` values, decreasing from about 1.
#[must_use]
pub fn sigmas(steps: usize, image_seq_len: usize) -> Vec<f32> {
    let mu = empirical_mu(image_seq_len, steps);
    let emu = mu.exp();
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            let s = if steps == 1 { 1.0 } else { 1.0 - (1.0 - 1.0 / steps as f64) * i as f64 / (steps - 1) as f64 };
            (emu / (emu + (1.0 / s - 1.0))) as f32
        })
        .collect();
    out.push(0.0);
    out
}

/// `n` standard-normal samples from `seed`. The generator is platform
/// independent, so a seed names the same starting latent on every device.
#[must_use]
pub fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        out.push((r * th.cos()) as f32);
        out.push((r * th.sin()) as f32);
    }
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_step_schedule_at_one_megapixel_matches_the_reference_values() {
        // Reference: diffusers FlowMatchEulerDiscreteScheduler with the
        // empirical mu for 4096 image tokens and 4 steps.
        let s = sigmas(4, 4096);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert_eq!(s[4], 0.0);
        assert!((empirical_mu(4096, 4) - 2.291_179_894).abs() < 1e-8);
        for (i, want) in [1.0, 0.967_383_99, 0.908_143_92, 0.767_199_96].iter().enumerate() {
            assert!((s[i] - want).abs() < 1e-6, "step {i}: {} vs {want}", s[i]);
        }
        assert!(s.windows(2).all(|w| w[0] > w[1]), "strictly decreasing");
    }

    #[test]
    fn noise_is_reproducible_per_seed_and_roughly_standard() {
        let a = gaussian(7, 10_001);
        assert_eq!(a, gaussian(7, 10_001));
        assert_ne!(a, gaussian(8, 10_001));
        let mean: f64 = a.iter().map(|v| f64::from(*v)).sum::<f64>() / a.len() as f64;
        let var: f64 = a.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.05, "mean {mean} var {var}");
    }
}
