//! Rewards for policy optimization.
//!
//! A [`Reward`] scores a completion of a prompt. The built-ins are pure functions of the tokens;
//! [`HostReward`] lets the host plug in any scorer (a verifier, a test runner in its own sandbox)
//! and reports its failures instead of guessing a score. Rewards consumed by a step are committed
//! to by [`reward_digest`].

use crate::Error;
use crate::hash::{Digest, domain_hash};

/// Scores a completion of a prompt.
pub trait Reward {
    /// The reward of `completion` after `prompt`.
    ///
    /// # Errors
    /// A scorer that cannot produce a value refuses rather than returning a default.
    fn score(&self, prompt: &[i32], completion: &[i32]) -> Result<f64, Error>;
}

/// 1 when the completion equals the reference exactly, else 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactMatch {
    /// The reference completion.
    pub reference: Vec<i32>,
}

impl Reward for ExactMatch {
    fn score(&self, _prompt: &[i32], completion: &[i32]) -> Result<f64, Error> {
        Ok(if completion == self.reference.as_slice() { 1.0 } else { 0.0 })
    }
}

/// The fraction of completion tokens that are in `tokens`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenFraction {
    /// Rewarded token ids.
    pub tokens: Vec<i32>,
}

impl Reward for TokenFraction {
    fn score(&self, _prompt: &[i32], completion: &[i32]) -> Result<f64, Error> {
        if completion.is_empty() {
            return Ok(0.0);
        }
        let hits = completion.iter().filter(|t| self.tokens.contains(t)).count();
        #[allow(clippy::cast_precision_loss)]
        Ok(hits as f64 / completion.len() as f64)
    }
}

/// `base` minus `per_token` for every completion token beyond `free_tokens`.
#[derive(Debug)]
pub struct LengthPenalty<R> {
    /// The penalized reward.
    pub base: R,
    /// Completion tokens without penalty.
    pub free_tokens: usize,
    /// Penalty per further token.
    pub per_token: f64,
}

impl<R: Reward> Reward for LengthPenalty<R> {
    fn score(&self, prompt: &[i32], completion: &[i32]) -> Result<f64, Error> {
        #[allow(clippy::cast_precision_loss)]
        let over = completion.len().saturating_sub(self.free_tokens) as f64;
        Ok(self.base.score(prompt, completion)? - self.per_token * over)
    }
}

/// A scorer supplied by the host, for example a verifier that runs in the host's own sandbox.
/// Its errors refuse the rollout's step.
pub struct HostReward<F> {
    name: String,
    scorer: F,
}

impl<F> std::fmt::Debug for HostReward<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostReward").field("name", &self.name).finish_non_exhaustive()
    }
}

impl<F: Fn(&[i32], &[i32]) -> Result<f64, String>> HostReward<F> {
    /// A named host scorer.
    pub fn new(name: impl Into<String>, scorer: F) -> Self {
        Self { name: name.into(), scorer }
    }
}

impl<F: Fn(&[i32], &[i32]) -> Result<f64, String>> Reward for HostReward<F> {
    fn score(&self, prompt: &[i32], completion: &[i32]) -> Result<f64, Error> {
        let v = (self.scorer)(prompt, completion).map_err(|e| Error::Refused(format!("reward {}: {e}", self.name)))?;
        if v.is_finite() { Ok(v) } else { Err(Error::Refused(format!("reward {}: not finite", self.name))) }
    }
}

/// Commitment to the rewards a step consumed, in order.
#[must_use]
pub fn reward_digest(rewards: &[f64]) -> Digest {
    let bytes: Vec<u8> = rewards.iter().flat_map(|r| r.to_le_bytes()).collect();
    domain_hash("praecise.rewards.v1", &[&bytes])
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn builtins_and_host_rewards() {
        assert_eq!(ExactMatch { reference: vec![1, 2] }.score(&[0], &[1, 2]).unwrap(), 1.0);
        assert_eq!(ExactMatch { reference: vec![1, 2] }.score(&[0], &[1, 3]).unwrap(), 0.0);
        assert_eq!(TokenFraction { tokens: vec![7] }.score(&[], &[7, 1, 7, 2]).unwrap(), 0.5);
        let p = LengthPenalty { base: TokenFraction { tokens: vec![7] }, free_tokens: 2, per_token: 0.1 };
        assert!((p.score(&[], &[7, 7, 7, 7]).unwrap() - 0.8).abs() < 1e-12);
        let host = HostReward::new("parity", |_: &[i32], c: &[i32]| if c.is_empty() { Err("empty".into()) } else { Ok(f64::from(c[0] % 2)) });
        assert_eq!(host.score(&[], &[3]).unwrap(), 1.0);
        assert!(matches!(host.score(&[], &[]), Err(Error::Refused(m)) if m.contains("parity")));
        let nan = HostReward::new("nan", |_: &[i32], _: &[i32]| Ok(f64::NAN));
        assert!(nan.score(&[], &[1]).is_err());
        assert_ne!(reward_digest(&[1.0, 0.0]), reward_digest(&[0.0, 1.0]));
    }
}
