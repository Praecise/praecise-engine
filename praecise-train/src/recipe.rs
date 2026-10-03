//! Training recipes and their canonical hash.
//!
//! The recipe is everything about a run that is not data, state or seed: the
//! objective, the optimizer and its hyperparameters, the adapter layout, the
//! precision policy and the batch shape. Every low-precision cast is part of
//! the recipe, so it is part of `recipe_hash`, so a replay uses the same casts.

use crate::canonical::Encoder;
use crate::hash::{Digest, domain_hash};

/// Training objective.
#[derive(Debug, Clone, PartialEq)]
pub enum Objective {
    /// Token cross-entropy on unmasked positions.
    Sft,
    /// Direct preference optimization with temperature `beta`.
    Dpo {
        /// Inverse temperature of the implicit reward.
        beta: f64,
    },
    /// Group-relative policy optimization.
    Grpo {
        /// Clip range of the policy probability ratio.
        clip: f64,
        /// Weight of the KL penalty to the reference policy.
        kl_weight: f64,
    },
    /// On-policy distillation: per-token reverse KL to a teacher.
    Distill,
    /// Contrastive embeddings with in-batch negatives.
    InfoNce {
        /// Softmax temperature.
        temperature: f64,
    },
    /// Pointwise reranking with binary cross-entropy.
    RerankBce,
    /// Listwise reranking with a softmax over candidates.
    RerankListwise,
    /// Quantile regression with the pinball loss.
    Pinball {
        /// Quantile levels in `(0, 1)`.
        quantiles: Vec<f64>,
    },
}

/// Optimizer and hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerSpec {
    /// `AdamW`: Adam with decoupled weight decay.
    AdamW {
        /// Learning rate.
        lr: f32,
        /// First-moment decay.
        beta1: f32,
        /// Second-moment decay.
        beta2: f32,
        /// Denominator epsilon.
        eps: f32,
        /// Decoupled weight decay.
        weight_decay: f32,
    },
    /// Plain SGD with weight decay.
    Sgd {
        /// Learning rate.
        lr: f32,
        /// Weight decay.
        weight_decay: f32,
    },
    /// Momentum with Newton-Schulz orthogonalized updates for matrix parameters.
    Muon {
        /// Learning rate.
        lr: f32,
        /// Momentum.
        momentum: f32,
        /// Newton-Schulz iterations.
        ns_steps: u32,
        /// Decoupled weight decay.
        weight_decay: f32,
    },
}

/// Low-rank adapter layout.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterSpec {
    /// Rank of `B A`.
    pub rank: u32,
    /// Scale numerator: the update is `(alpha / rank) * B A x`.
    pub alpha: f32,
    /// Base tensor name suffixes that receive an adapter, for example `attn_q.weight`.
    pub targets: Vec<String>,
}

/// Numeric formats of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// FP32 master weights and compute.
    F32,
    /// FP32 master weights, BF16 compute, FP32 accumulation.
    Bf16Compute,
}

/// Recipe of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    /// Objective.
    pub objective: Objective,
    /// Optimizer.
    pub optimizer: OptimizerSpec,
    /// Adapter layout; `None` trains the selected base tensors directly.
    pub adapter: Option<AdapterSpec>,
    /// Precision policy.
    pub precision: Precision,
    /// Tokens per sequence.
    pub seq_len: u32,
    /// Sequences per step.
    pub batch: u32,
    /// Global gradient-norm clip, 0 for none.
    pub grad_clip: f32,
}

impl Recipe {
    /// Canonical encoding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.str("praecise.recipe.v1");
        match &self.objective {
            Objective::Sft => {
                e.u8(0);
            }
            Objective::Dpo { beta } => {
                e.u8(1).f64(*beta);
            }
            Objective::Grpo { clip, kl_weight } => {
                e.u8(2).f64(*clip).f64(*kl_weight);
            }
            Objective::Distill => {
                e.u8(3);
            }
            Objective::InfoNce { temperature } => {
                e.u8(4).f64(*temperature);
            }
            Objective::RerankBce => {
                e.u8(5);
            }
            Objective::RerankListwise => {
                e.u8(6);
            }
            Objective::Pinball { quantiles } => {
                e.u8(7).u64(quantiles.len() as u64);
                for q in quantiles {
                    e.f64(*q);
                }
            }
        }
        match &self.optimizer {
            OptimizerSpec::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                e.u8(0)
                    .f32(*lr)
                    .f32(*beta1)
                    .f32(*beta2)
                    .f32(*eps)
                    .f32(*weight_decay);
            }
            OptimizerSpec::Sgd { lr, weight_decay } => {
                e.u8(1).f32(*lr).f32(*weight_decay);
            }
            OptimizerSpec::Muon {
                lr,
                momentum,
                ns_steps,
                weight_decay,
            } => {
                e.u8(2)
                    .f32(*lr)
                    .f32(*momentum)
                    .u32(*ns_steps)
                    .f32(*weight_decay);
            }
        }
        match &self.adapter {
            None => {
                e.u8(0);
            }
            Some(a) => {
                let mut targets = a.targets.clone();
                targets.sort();
                targets.dedup();
                e.u8(1).u32(a.rank).f32(a.alpha).u64(targets.len() as u64);
                for t in &targets {
                    e.str(t);
                }
            }
        }
        e.u8(match self.precision {
            Precision::F32 => 0,
            Precision::Bf16Compute => 1,
        });
        e.u32(self.seq_len).u32(self.batch).f32(self.grad_clip);
        e.finish()
    }

    /// Hash of the canonical encoding.
    #[must_use]
    pub fn hash(&self) -> Digest {
        domain_hash("praecise.recipe", &[&self.encode()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> Recipe {
        Recipe {
            objective: Objective::Sft,
            optimizer: OptimizerSpec::AdamW {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            adapter: Some(AdapterSpec {
                rank: 8,
                alpha: 16.0,
                targets: vec!["attn_v.weight".into(), "attn_q.weight".into()],
            }),
            precision: Precision::F32,
            seq_len: 256,
            batch: 4,
            grad_clip: 1.0,
        }
    }

    #[test]
    fn hash_is_canonical_and_sensitive() {
        let a = recipe();
        let mut b = recipe();
        b.adapter.as_mut().unwrap().targets.reverse();
        assert_eq!(a.hash(), b.hash());
        let mut c = recipe();
        c.optimizer = OptimizerSpec::AdamW {
            lr: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        assert_ne!(a.hash(), c.hash());
        let mut d = recipe();
        d.precision = Precision::Bf16Compute;
        assert_ne!(a.hash(), d.hash());
    }
}
