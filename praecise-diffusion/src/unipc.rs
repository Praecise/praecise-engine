//! UniPC multistep sampler (B(h) = e^h - 1, second order, data prediction)
//! over flow-matching velocities, with Karras-spaced noise levels mapped to
//! the flow parameterisation.
//!
//! Each step converts the velocity to a clean-sample prediction, corrects the
//! previous step's result with it (from the second step on), then predicts
//! the next sample. The first step and the last run at first order.

use serde::Deserialize;

use crate::error::{Error, Result};

/// Karras schedule exponent.
const RHO: f64 = 7.0;

/// Scheduler configuration, read from `scheduler/scheduler_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct UniPcConfig {
    /// Training timesteps (noise levels map to `level * this`).
    pub num_train_timesteps: u64,
    /// Multistep order.
    pub solver_order: u64,
    /// Order-condition variant.
    pub solver_type: String,
    /// What the model predicts.
    pub prediction_type: String,
    /// Solve in clean-sample space.
    pub predict_x0: bool,
    /// Dynamic thresholding of predictions.
    #[serde(default)]
    pub thresholding: bool,
    /// Drop to first order on the last step.
    pub lower_order_final: bool,
    /// Steps whose correction is skipped.
    #[serde(default)]
    pub disable_corrector: Vec<u64>,
    /// Karras spacing of the noise levels.
    pub use_karras_sigmas: bool,
    /// Flow-matching parameterisation of the noise levels.
    pub use_flow_sigmas: bool,
    /// Noise level after the last step.
    pub final_sigmas_type: String,
    /// Smallest Karras noise level (Karras spacing only).
    pub sigma_min: Option<f64>,
    /// Largest Karras noise level (Karras spacing only).
    pub sigma_max: Option<f64>,
    /// Rational shift of linearly spaced flow noise levels (without Karras
    /// spacing).
    #[serde(default = "unit_shift")]
    pub flow_shift: f64,
    /// Terminal stretch of the noise levels; only its absence is implemented.
    #[serde(default)]
    pub shift_terminal: Option<f64>,
    /// Resolution-dependent shifting.
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    /// Exponential spacing.
    #[serde(default)]
    pub use_exponential_sigmas: bool,
    /// Beta spacing.
    #[serde(default)]
    pub use_beta_sigmas: bool,
}

fn unit_shift() -> f64 {
    1.0
}

impl UniPcConfig {
    /// Refuse settings other than the ones implemented.
    ///
    /// # Errors
    /// [`Error::Config`] naming the setting.
    pub fn validate(&self) -> Result<()> {
        let ok = self.solver_order == 2
            && self.solver_type == "bh2"
            && self.prediction_type == "flow_prediction"
            && self.predict_x0
            && !self.thresholding
            && self.lower_order_final
            && self.disable_corrector.is_empty()
            && self.use_flow_sigmas
            && self.final_sigmas_type == "zero"
            && !self.use_dynamic_shifting
            && !self.use_exponential_sigmas
            && !self.use_beta_sigmas
            && self.shift_terminal.is_none()
            && match (self.use_karras_sigmas, self.sigma_min, self.sigma_max) {
                (true, Some(lo), Some(hi)) => lo > 0.0 && hi > lo,
                (true, _, _) => false,
                (false, _, _) => self.flow_shift > 0.0,
            };
        if ok {
            Ok(())
        } else {
            Err(Error::Config("scheduler: only second-order bh2 UniPC on Karras or shifted linear flow sigmas ending at zero is implemented".into()))
        }
    }

    /// Noise levels for `steps` steps, plus the final zero, in single
    /// precision; and the integer timesteps the model is told.
    #[must_use]
    pub fn schedule(&self, steps: usize) -> (Vec<f32>, Vec<i64>) {
        let (Some(lo), Some(hi), true) = (self.sigma_min, self.sigma_max, self.use_karras_sigmas) else {
            return shifted_linear_schedule(steps, self.flow_shift, self.num_train_timesteps);
        };
        let (lo, hi) = (lo.powf(1.0 / RHO), hi.powf(1.0 / RHO));
        let mut sigmas = Vec::with_capacity(steps + 1);
        let mut timesteps = Vec::with_capacity(steps);
        for i in 0..steps {
            let ramp = if steps == 1 { 0.0 } else { i as f64 / (steps - 1) as f64 };
            let s = (hi + ramp * (lo - hi)).powf(RHO);
            let flow = s / (s + 1.0);
            timesteps.push((flow * self.num_train_timesteps as f64) as i64);
            sigmas.push(flow as f32);
        }
        sigmas.push(0.0);
        (sigmas, timesteps)
    }
}

#[cfg(test)]
mod flow_tests {
    use super::*;

    #[test]
    fn default_flow_sigmas_match_the_scheduler() {
        let (s, t) = flow_sigmas_schedule(3, 5.0, 1000);
        assert_eq!(t, vec![999, 909, 714]);
        assert!((s[0] - 0.999_999).abs() < 1e-6 && (s[1] - 0.909_2).abs() < 1e-4 && (s[2] - 0.714_9).abs() < 1e-4);
        assert_eq!(s[3], 0.0);
    }
}

/// The scheduler's own flow noise levels: `steps` levels linearly spaced
/// from 1 towards `1 / num_train_timesteps` (endpoint excluded), each mapped
/// through the rational shift, the first nudged just below 1 so its log-SNR
/// stays finite; plus the final zero in single precision, and the integer
/// timesteps the model is told (`level * num_train_timesteps`, truncated).
#[must_use]
pub fn flow_sigmas_schedule(steps: usize, shift: f64, num_train_timesteps: u64) -> (Vec<f32>, Vec<i64>) {
    let n = num_train_timesteps as f64;
    let end = 1.0 / n;
    let mut sigmas = Vec::with_capacity(steps + 1);
    let mut timesteps = Vec::with_capacity(steps);
    for i in 0..steps {
        let s = 1.0 + i as f64 * ((end - 1.0) / steps as f64);
        let mut s = shift * s / (1.0 + (shift - 1.0) * s);
        if i == 0 && (s - 1.0).abs() < 1e-6 {
            s -= 1e-6;
        }
        timesteps.push((s * n) as i64);
        sigmas.push(s as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// Linearly spaced flow noise levels from `1 - 1/num_train_timesteps` down
/// to zero, each mapped through the rational shift `shift * s / (1 + (shift - 1) * s)`:
/// the `steps` noise levels plus the final zero in single precision, and the
/// integer timesteps the model is told (`level * num_train_timesteps`, truncated).
#[must_use]
pub fn shifted_linear_schedule(steps: usize, shift: f64, num_train_timesteps: u64) -> (Vec<f32>, Vec<i64>) {
    let top = f64::from(1.0f32 - 1.0 / num_train_timesteps as f32);
    let mut sigmas = Vec::with_capacity(steps + 1);
    let mut timesteps = Vec::with_capacity(steps);
    for i in 0..steps {
        let s = top + i as f64 * ((0.0 - top) / steps as f64);
        let s = shift * s / (1.0 + (shift - 1.0) * s);
        timesteps.push((s * num_train_timesteps as f64) as i64);
        sigmas.push(s as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// `log(alpha) - log(sigma)` for flow noise level `s`.
fn lambda(s: f32) -> f64 {
    let s = f64::from(s);
    (1.0 - s).ln() - s.ln()
}

/// Sampler state across one denoising run.
#[derive(Debug, Clone)]
pub struct UniPc {
    sigmas: Vec<f32>,
    /// Clean-sample predictions of the last two steps, newest last.
    outputs: Vec<Vec<f32>>,
    last_sample: Option<Vec<f32>>,
    lower_order_nums: usize,
    this_order: usize,
    step: usize,
}

impl UniPc {
    /// A run over `sigmas` (from [`UniPcConfig::schedule`]).
    #[must_use]
    pub fn new(sigmas: Vec<f32>) -> Self {
        Self { sigmas, outputs: Vec::new(), last_sample: None, lower_order_nums: 0, this_order: 1, step: 0 }
    }

    /// Advance `sample` by one step given the model's velocity.
    ///
    /// # Errors
    /// When called more times than there are steps.
    pub fn step(&mut self, velocity: &[f32], sample: &[f32]) -> Result<Vec<f32>> {
        let n = self.sigmas.len() - 1;
        let i = self.step;
        if i >= n {
            return Err(Error::Request("sampler ran past its last step".into()));
        }
        let sigma = self.sigmas[i];
        let m: Vec<f32> = sample.iter().zip(velocity).map(|(x, v)| x - sigma * v).collect();
        let sample = match (&self.last_sample, i > 0) {
            (Some(last), true) => self.correct(&m, last),
            _ => sample.to_vec(),
        };
        if self.outputs.len() == 2 {
            self.outputs.remove(0);
        }
        self.outputs.push(m);
        let order = 2usize.min(n - i).min(self.lower_order_nums + 1);
        self.this_order = order;
        let next = self.predict(&sample, order);
        self.last_sample = Some(sample);
        self.lower_order_nums = (self.lower_order_nums + 1).min(2);
        self.step += 1;
        Ok(next)
    }

    /// `(h, alpha_t, sigma_t / sigma_s0, h_phi_1, B_h)` for a step from noise
    /// level index `s` to `t`.
    fn coefficients(&self, s: usize, t: usize) -> (f64, f64, f64, f64, f64) {
        let (st, ss) = (f64::from(self.sigmas[t]), f64::from(self.sigmas[s]));
        let h = lambda(self.sigmas[t]) - lambda(self.sigmas[s]);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        (h, 1.0 - st, st / ss, h_phi_1, hh.exp_m1())
    }

    /// Right-hand side `b` of the order conditions.
    fn rhs(h: f64, order: usize, b_h: f64) -> Vec<f64> {
        let hh = -h;
        let mut h_phi_k = hh.exp_m1() / hh - 1.0;
        let mut factorial = 1.0;
        let mut b = Vec::with_capacity(order);
        for i in 1..=order {
            b.push(h_phi_k * factorial / b_h);
            factorial *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial;
        }
        b
    }

    fn predict(&self, x: &[f32], order: usize) -> Vec<f32> {
        let i = self.step;
        let (h, alpha_t, ratio, h_phi_1, b_h) = self.coefficients(i, i + 1);
        let m0 = &self.outputs[self.outputs.len() - 1];
        let a = ratio as f32;
        let b = (alpha_t * h_phi_1) as f32;
        let mut out: Vec<f32> = x.iter().zip(m0).map(|(x, m)| a * x - b * m).collect();
        if order == 2 {
            let m1 = &self.outputs[0];
            let rk = ((lambda(self.sigmas[i - 1]) - lambda(self.sigmas[i])) / h) as f32;
            let c = (alpha_t * b_h) as f32;
            for ((o, p), q) in out.iter_mut().zip(m1).zip(m0) {
                let d1 = (p - q) / rk;
                *o -= c * (0.5 * d1);
            }
        }
        out
    }

    fn correct(&self, model_t: &[f32], last: &[f32]) -> Vec<f32> {
        let i = self.step;
        let order = self.this_order;
        let (h, alpha_t, ratio, h_phi_1, b_h) = self.coefficients(i - 1, i);
        let m0 = &self.outputs[self.outputs.len() - 1];
        let a = ratio as f32;
        let b = (alpha_t * h_phi_1) as f32;
        let c = (alpha_t * b_h) as f32;
        if order == 1 {
            return last.iter().zip(m0).zip(model_t).map(|((x, m), mt)| a * x - b * m - c * (0.5 * (mt - m))).collect();
        }
        let rk = (lambda(self.sigmas[i - 2]) - lambda(self.sigmas[i - 1])) / h;
        let rhs = Self::rhs(h, 2, b_h);
        // [[1, 1], [rk, 1]] rho = b.
        let det = 1.0 - rk;
        let rho0 = ((rhs[0] - rhs[1]) / det) as f32;
        let rho1 = ((rhs[1] - rk * rhs[0]) / det) as f32;
        let rk = rk as f32;
        let m1 = &self.outputs[0];
        last.iter()
            .zip(m0)
            .zip(m1)
            .zip(model_t)
            .map(|(((x, m), p), mt)| {
                let d1 = (p - m) / rk;
                a * x - b * m - c * (rho0 * d1 + rho1 * (mt - m))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> UniPcConfig {
        serde_json::from_value(json()).unwrap()
    }

    fn json() -> serde_json::Value {
        serde_json::json!({
            "num_train_timesteps": 1000, "solver_order": 2, "solver_type": "bh2",
            "prediction_type": "flow_prediction", "predict_x0": true, "thresholding": false,
            "lower_order_final": true, "disable_corrector": [], "use_karras_sigmas": true,
            "use_flow_sigmas": true, "final_sigmas_type": "zero", "sigma_min": 0.147, "sigma_max": 200.0
        })
    }

    #[test]
    fn the_schedule_runs_from_near_one_to_zero() {
        let c = cfg();
        c.validate().unwrap();
        let (s, t) = c.schedule(4);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 200.0 / 201.0).abs() < 1e-6);
        assert!((s[3] - 0.147 / 1.147).abs() < 1e-6);
        assert_eq!(s[4], 0.0);
        assert_eq!(t[0], 995);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn without_karras_spacing_the_levels_are_shifted_linear() {
        let mut v = json();
        v["use_karras_sigmas"] = false.into();
        v["sigma_min"] = serde_json::Value::Null;
        v["sigma_max"] = serde_json::Value::Null;
        v["flow_shift"] = 5.0.into();
        let c: UniPcConfig = serde_json::from_value(v).unwrap();
        c.validate().unwrap();
        assert_eq!(c.schedule(4), shifted_linear_schedule(4, 5.0, 1000));
        assert_eq!(c.schedule(4).1, vec![999, 937, 833, 624]);
        let mut v = json();
        v["sigma_max"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<UniPcConfig>(v).unwrap().validate().is_err());
    }

    #[test]
    fn the_shifted_schedule_starts_one_tick_below_one() {
        let (s, t) = shifted_linear_schedule(4, 5.0, 1000);
        assert_eq!(s.len(), 5);
        assert_eq!(s[4], 0.0);
        assert_eq!(t[0], 999);
        let mid = 0.4995f64;
        assert!((f64::from(s[2]) - 5.0 * mid / (1.0 + 4.0 * mid)).abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn the_last_step_lands_on_the_clean_prediction() {
        let (s, _) = cfg().schedule(3);
        let mut u = UniPc::new(s.clone());
        let mut x = vec![1.0f32, -0.5];
        let v = [0.3f32, 0.2];
        for _ in 0..2 {
            x = u.step(&v, &x).unwrap();
        }
        // The final step returns the clean prediction made from the sample
        // it was given (the correction only feeds the predictor's input).
        let out = u.step(&v, &x).unwrap();
        for ((o, x), v) in out.iter().zip(&x).zip(&v) {
            assert!((o - (x - s[2] * v)).abs() < 1e-6);
        }
        assert!(u.step(&v, &x).is_err());
    }
}
