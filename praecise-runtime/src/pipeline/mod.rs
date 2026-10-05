//! Layer-pipeline serving: one model's blocks split across machines on a LAN.
//!
//! Each machine runs one stage, a contiguous range of blocks loaded with
//! `LlamaModelParams::with_layer_stage`. The first stage, the driver, holds the token
//! embedding and the caller. Every later stage receives the residual stream of the
//! stage before it, runs its blocks and passes its output on. The last stage holds the
//! output head; its logits travel back up the chain to the driver.
//!
//! Stages are identified by public keys the host application supplies. Every link is
//! authenticated by both ends signing the handshake transcript through a
//! [`StageAuthenticator`], so the engine never holds a signing key; frames after the
//! handshake are encrypted and authenticated with keys from an ephemeral X25519
//! exchange bound to that transcript.
//!
//! The driver splits a prompt into micro-batches and sends each one on as soon as its
//! own blocks are done, so every stage works on a different micro-batch at the same
//! time; several sequences can be in flight the same way. Hidden states cross the wire
//! as exact `f32`, so the output equals single-machine output up to the arithmetic of
//! the batch shapes. A stage that fails, drops its link or stops answering fails every
//! request in flight with [`PipelineError::StageFailed`] naming that stage; nothing is
//! retried elsewhere.

pub mod transport;
pub mod wire;

#[cfg(feature = "bundled-llama")]
pub mod driver;
#[cfg(feature = "bundled-llama")]
pub mod stage;

use sha2::{Digest, Sha256};

/// A contiguous range of blocks, `begin..end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    /// First block of the range.
    pub begin: u32,
    /// One past the last block of the range.
    pub end: u32,
}

impl LayerRange {
    /// Number of blocks in the range.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.begin)
    }

    /// True for an empty range.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One stage of a plan: who runs it, where it listens, which blocks it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSpec {
    /// Public key that identifies the stage, in the host application's encoding.
    pub identity: Vec<u8>,
    /// `host:port` the stage listens on. Unused for the first stage (the driver).
    pub address: String,
    /// Blocks the stage holds.
    pub layers: LayerRange,
}

/// The whole pipeline: the model, its shape and the stages in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelinePlan {
    /// Digest that names the model's weights; every stage loads the same model.
    pub model_digest: [u8; 32],
    /// Blocks in the model.
    pub n_layer: u32,
    /// Width of the residual stream.
    pub n_embd: u32,
    /// Stages in pipeline order; the first is the driver.
    pub stages: Vec<StageSpec>,
}

impl PipelinePlan {
    /// Check that the stages cover every block once, in order, with distinct identities.
    ///
    /// # Errors
    ///
    /// [`PipelineError::Plan`] naming the first problem found.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.stages.len() < 2 {
            return Err(PipelineError::Plan("a pipeline needs at least two stages".into()));
        }
        if self.n_embd == 0 {
            return Err(PipelineError::Plan("n_embd is 0".into()));
        }
        let mut next = 0u32;
        for (i, s) in self.stages.iter().enumerate() {
            if s.identity.is_empty() || s.identity.len() > usize::from(u16::MAX) {
                return Err(PipelineError::Plan(format!("stage {i} has no usable identity")));
            }
            if s.layers.begin != next || s.layers.is_empty() {
                return Err(PipelineError::Plan(format!(
                    "stage {i} holds blocks {}..{}, expected a non-empty range starting at {next}",
                    s.layers.begin, s.layers.end
                )));
            }
            if i > 0 && s.address.is_empty() {
                return Err(PipelineError::Plan(format!("stage {i} has no address")));
            }
            if self.stages[..i].iter().any(|o| o.identity == s.identity) {
                return Err(PipelineError::Plan(format!("stage {i} repeats the identity of an earlier stage")));
            }
            next = s.layers.end;
        }
        if next != self.n_layer {
            return Err(PipelineError::Plan(format!(
                "stages cover blocks 0..{next} of a model with {} blocks",
                self.n_layer
            )));
        }
        Ok(())
    }

    /// Digest both ends of every link agree on during the handshake: the model, its
    /// shape, and every stage's identity and blocks. Addresses are left out, so a stage
    /// that moves to another address keeps its place in the plan.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"praecise-layer-pipeline-plan/1");
        h.update(self.model_digest);
        h.update(self.n_layer.to_le_bytes());
        h.update(self.n_embd.to_le_bytes());
        h.update((self.stages.len() as u64).to_le_bytes());
        for s in &self.stages {
            h.update((s.identity.len() as u64).to_le_bytes());
            h.update(&s.identity);
            h.update(s.layers.begin.to_le_bytes());
            h.update(s.layers.end.to_le_bytes());
        }
        h.finalize().into()
    }
}

/// Split `n_layer` blocks into contiguous ranges proportional to `weights` (for example
/// each machine's free memory). Every range gets at least one block.
///
/// # Errors
///
/// [`PipelineError::Plan`] if there are no weights, a zero weight, or more machines
/// than blocks.
pub fn split_layers(n_layer: u32, weights: &[u64]) -> Result<Vec<LayerRange>, PipelineError> {
    let n = weights.len();
    if n == 0 || weights.contains(&0) {
        return Err(PipelineError::Plan("every stage needs a non-zero weight".into()));
    }
    if n as u64 > u64::from(n_layer) {
        return Err(PipelineError::Plan(format!("{n} stages for {n_layer} blocks")));
    }
    let total: u128 = weights.iter().map(|w| u128::from(*w)).sum();
    let mut out = Vec::with_capacity(n);
    let mut begin = 0u32;
    let mut acc: u128 = 0;
    for (i, w) in weights.iter().enumerate() {
        acc += u128::from(*w);
        let remaining_stages = (n - i - 1) as u32;
        let ideal = u32::try_from(acc * u128::from(n_layer) / total).unwrap_or(n_layer);
        let end = if i + 1 == n {
            n_layer
        } else {
            ideal.clamp(begin + 1, n_layer - remaining_stages)
        };
        out.push(LayerRange { begin, end });
        begin = end;
    }
    Ok(out)
}

/// Signs and verifies handshake transcripts for one stage. Supplied by the host
/// application, which decides what an identity is and where its key lives.
pub trait StageAuthenticator: Send + Sync {
    /// The public key that identifies this stage in a [`PipelinePlan`].
    fn identity(&self) -> Vec<u8>;

    /// Sign `message` with this stage's key.
    ///
    /// # Errors
    ///
    /// A description of why the key could not sign.
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String>;

    /// True when `signature` over `message` verifies under the public key `identity`.
    fn verify(&self, identity: &[u8], message: &[u8], signature: &[u8]) -> bool;
}

/// Errors of layer-pipeline serving.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The plan is inconsistent.
    #[error("invalid pipeline plan: {0}")]
    Plan(String),
    /// A stage could not be reached, failed a request, or stopped answering.
    #[error("pipeline stage {stage} failed: {reason}")]
    StageFailed {
        /// Index of the stage in the plan.
        stage: usize,
        /// What happened.
        reason: String,
    },
    /// The peer on a link did not prove the identity the plan names.
    #[error("authentication with pipeline stage {stage} failed: {reason}")]
    Authentication {
        /// Index of the peer stage in the plan.
        stage: usize,
        /// What did not match.
        reason: String,
    },
    /// The pipeline did not answer in time; some stage is stalled.
    #[error("pipeline stalled: {0}")]
    Timeout(String),
    /// A malformed or unexpected message.
    #[error("pipeline protocol error: {0}")]
    Protocol(String),
    /// The local engine refused a request.
    #[error("pipeline engine error: {0}")]
    Engine(String),
    /// Socket error.
    #[error("pipeline io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(ranges: &[(u32, u32)]) -> PipelinePlan {
        PipelinePlan {
            model_digest: [7; 32],
            n_layer: ranges.last().map_or(0, |r| r.1),
            n_embd: 8,
            stages: ranges
                .iter()
                .enumerate()
                .map(|(i, r)| StageSpec {
                    identity: vec![i as u8 + 1; 4],
                    address: format!("127.0.0.1:{}", 9000 + i),
                    layers: LayerRange { begin: r.0, end: r.1 },
                })
                .collect(),
        }
    }

    #[test]
    fn split_is_contiguous_proportional_and_never_empty() {
        let r = split_layers(28, &[1, 1, 1]).unwrap();
        assert_eq!(r, vec![LayerRange { begin: 0, end: 9 }, LayerRange { begin: 9, end: 18 }, LayerRange { begin: 18, end: 28 }]);
        let r = split_layers(10, &[1000, 1, 1]).unwrap();
        assert_eq!(r.iter().map(LayerRange::len).collect::<Vec<_>>(), vec![8, 1, 1]);
        assert!(split_layers(2, &[1, 1, 1]).is_err());
        assert!(split_layers(8, &[1, 0]).is_err());
    }

    #[test]
    fn plan_validation_rejects_gaps_overlaps_and_repeated_identities() {
        plan(&[(0, 3), (3, 6)]).validate().unwrap();
        assert!(plan(&[(0, 3)]).validate().is_err());
        assert!(plan(&[(0, 3), (4, 6)]).validate().is_err());
        assert!(plan(&[(0, 3), (2, 6)]).validate().is_err());
        let mut p = plan(&[(0, 3), (3, 6)]);
        p.stages[1].identity = p.stages[0].identity.clone();
        assert!(p.validate().is_err());
        let mut p = plan(&[(0, 3), (3, 6)]);
        p.n_layer = 7;
        assert!(p.validate().is_err());
    }

    #[test]
    fn plan_digest_binds_identities_and_ranges_but_not_addresses() {
        let a = plan(&[(0, 3), (3, 6)]);
        let mut b = a.clone();
        b.stages[1].address = "10.0.0.9:1".into();
        assert_eq!(a.digest(), b.digest());
        b.stages[1].identity = vec![9; 4];
        assert_ne!(a.digest(), b.digest());
        let c = plan(&[(0, 2), (2, 6)]);
        assert_ne!(a.digest(), c.digest());
    }
}
