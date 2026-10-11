//! A round: consecutive supervised steps of one run over one data shard.
//!
//! A round starts from a trainable state (fresh adapter weights or a saved state), draws each
//! step's batch from the shard with the stateless sampler, and records every step in a step log.
//! Its commitment binds the base weights, the shard, the recipe, the seed, the kernel class, the
//! steps it covers, the starting and ending state roots and the log head, so anyone holding the
//! same inputs on the same kernel class replays the round to the same commitment.

use crate::Error;
use crate::canonical::Encoder;
use crate::engine::{LoraTrainer, SftExample, StepBatch};
use crate::hash::{Digest, domain_hash};
use crate::merkle;
use crate::sampler::batch_indices;
use crate::steplog::{StepLog, StepSpec};

/// Canonical bytes of one supervised record: its length, then each token with its target flag.
#[must_use]
pub fn sft_record_bytes(record: &SftExample) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u32(u32::try_from(record.tokens.len()).unwrap_or(u32::MAX));
    for (t, m) in record.tokens.iter().zip(&record.target_mask) {
        e.u32(u32::from_le_bytes(t.to_le_bytes())).bool(*m);
    }
    e.finish()
}

/// Root of a supervised shard: the Merkle root over its records in order.
#[must_use]
pub fn sft_data_root(records: &[SftExample]) -> Digest {
    let leaves: Vec<Digest> = records.iter().map(|r| merkle::leaf_hash(&sft_record_bytes(r))).collect();
    merkle::root(&leaves)
}

/// What a supervised round runs.
#[derive(Debug, Clone)]
pub struct SftRound<'a> {
    /// Content root of the frozen base weights.
    pub base_root: Digest,
    /// Seed of batch selection.
    pub sample_seed: u64,
    /// Kernel class the steps run on.
    pub kernel_class: String,
    /// Steps to take.
    pub steps: u64,
    /// The shard, in order.
    pub records: &'a [SftExample],
    /// Probe tokens whose outputs every step commits to.
    pub probe: &'a [i32],
}

/// What a round produced.
#[derive(Debug, Clone)]
pub struct RoundOutcome {
    /// Content root of the base weights.
    pub base_root: Digest,
    /// Root of the shard.
    pub data_root: Digest,
    /// Hash of the recipe.
    pub recipe_hash: Digest,
    /// Seed of batch selection.
    pub sample_seed: u64,
    /// Kernel class the steps ran on.
    pub kernel_class: String,
    /// Index of the first step.
    pub first_step: u64,
    /// Root of the state the round started from.
    pub start_root: Digest,
    /// Root of the state the round ended at.
    pub end_root: Digest,
    /// The steps, hash-linked.
    pub log: StepLog,
}

impl RoundOutcome {
    /// Steps the round took.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.log.records().len() as u64
    }

    /// Objective value of each step, in order.
    #[must_use]
    pub fn losses(&self) -> Vec<f64> {
        self.log.records().iter().map(|r| r.result.loss).collect()
    }

    /// The commitment two replays of the round agree on.
    #[must_use]
    pub fn commitment(&self) -> Digest {
        let mut e = Encoder::new();
        e.digest(&self.base_root)
            .digest(&self.data_root)
            .digest(&self.recipe_hash)
            .u64(self.sample_seed)
            .str(&self.kernel_class)
            .u64(self.first_step)
            .u64(self.steps())
            .digest(&self.start_root)
            .digest(&self.end_root)
            .digest(&self.log.head());
        domain_hash("praecise.round.v1", &[e.as_bytes()])
    }
}

/// Runs `round` on `trainer` from the state it holds now.
///
/// # Errors
/// [`Error::Refused`] for a recipe that is not supervised, an empty shard or no steps; step and
/// engine refusals. A refused step ends the round with the trainer at the last good step.
pub fn run_sft_round(trainer: &mut LoraTrainer<'_>, round: &SftRound<'_>) -> Result<RoundOutcome, Error> {
    if round.records.is_empty() {
        return Err(Error::Refused("a round needs at least one record".into()));
    }
    if round.steps == 0 {
        return Err(Error::Refused("a round takes at least one step".into()));
    }
    let recipe = trainer.recipe().clone();
    if recipe.objective != crate::recipe::Objective::Sft {
        return Err(Error::Refused("a supervised round runs a supervised recipe".into()));
    }
    let batch = usize::try_from(recipe.batch).map_err(|_| Error::Refused("batch size".into()))?;
    if batch == 0 {
        return Err(Error::Refused("the recipe's batch is empty".into()));
    }
    let n = round.records.len() as u64;
    let data_root = sft_data_root(round.records);
    let recipe_hash = recipe.hash();
    let first_step = trainer.steps_taken();
    let start_root = trainer.state()?.state_root();
    let mut log = StepLog::new();
    let mut state_root = start_root;
    for i in 0..round.steps {
        let step_index = first_step + i;
        let picked = batch_indices(round.sample_seed, step_index, 0, n, batch)
            .into_iter()
            .map(|k| round.records[usize::try_from(k).unwrap_or(usize::MAX).min(round.records.len() - 1)].clone())
            .collect();
        let spec = StepSpec {
            base_root: round.base_root,
            state_root,
            data_root,
            sample_seed: round.sample_seed,
            step_index,
            recipe_hash,
            kernel_class: round.kernel_class.clone(),
        };
        let result = trainer.run_step(&spec, &StepBatch::Sft(picked), round.probe)?;
        state_root = result.state_root;
        log.append(spec, result)?;
    }
    Ok(RoundOutcome {
        base_root: round.base_root,
        data_root,
        recipe_hash,
        sample_seed: round.sample_seed,
        kernel_class: round.kernel_class.clone(),
        first_step,
        start_root,
        end_root: state_root,
        log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shard_root_binds_tokens_targets_and_order() {
        let a = SftExample::all_targets(vec![1, 2, 3]);
        let b = SftExample::all_targets(vec![1, 2, 4]);
        let mut masked = a.clone();
        masked.target_mask[2] = false;
        let root = sft_data_root(&[a.clone(), b.clone()]);
        assert_eq!(root, sft_data_root(&[a.clone(), b.clone()]));
        assert_ne!(root, sft_data_root(&[b.clone(), a.clone()]));
        assert_ne!(root, sft_data_root(&[masked, b.clone()]));
        assert_ne!(sft_record_bytes(&a), sft_record_bytes(&b));
    }
}
