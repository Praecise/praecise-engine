//! When to speculate, how far, and when not to bother.
//!
//! Speculative decoding is usually presented as a switch. It is not: it trades
//! compute for latency, and whether that trade pays depends on the model, the
//! hardware and — most of all — how loaded the server is. The published
//! speedups are almost all measured at concurrency 1, which is the one
//! operating point where spare compute is free.
//!
//! This module is the decision, not the mechanism. It takes what is known about
//! a request and returns a [`SpecPlan`]: which drafting method to use and how
//! many tokens to propose, or [`SpecMethod::None`] when the honest answer is
//! "decode normally". Keeping that judgement in one place means it can be
//! tested without a GPU, and audited without reading the decode loop.
//!
//! ## The three findings this encodes
//!
//! **Speculation decays under load, and no published method escapes it.** The
//! decay is not an artefact of a weak drafter: block-diffusion drafting, which
//! decouples *draft* cost from block size, still measures 3.43x at concurrency
//! 1, 2.84x at 8 and 1.45x at 32 on a reasoning benchmark — and 1.01x at 32 on
//! a chat benchmark, i.e. nothing at all. A tree-drafting method on the same
//! comparison falls to 0.6x, meaning it makes the server *slower*.
//!
//! The reason is that decoupling draft cost does not decouple *verification*
//! cost. Verifying gamma+1 tokens per sequence multiplies the batch, and at high
//! concurrency the target model is already compute-saturated, so those extra
//! positions come out of throughput that requests were queuing for. Better
//! drafters raise the ceiling and the floor; they do not change the shape of the
//! curve. That is why this gate is on utilization rather than on method.
//!
//! **MoE inverts the economics.** Verifying a block in parallel activates far
//! more experts than decoding one token at a time, because different positions
//! route to different experts. Measured: 3.28x on a dense 8B model against
//! 1.08x on a 120B MoE, same method.
//!
//! **Acceptance decays along a block.** Position *k* is only reached if every
//! earlier position matched, so the marginal value of a longer draft falls off
//! quickly while its cost stays linear. Past a point, a longer block buys
//! nothing and still pays.
//!
//! None of these thresholds are physical constants. They are starting points
//! drawn from published measurements, and [`SpecPolicy`] exposes them so a
//! deployment can replace them with its own numbers — which is the entire
//! argument for having a policy layer rather than a hardcoded `draft_n`.

use crate::config::GenerationConfig;

/// How a draft block is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecMethod {
    /// Do not speculate; decode one token at a time.
    None,
    /// Draft from token history with no model. Costs no GPU memory and no
    /// second set of weights, so it is the method that stays viable under load.
    Ngram,
    /// Draft with the target's own multi-token-prediction heads.
    ///
    /// No second model, but it constrains the serving shape: MTP paths
    /// typically force single-sequence batching and slow prefill, because
    /// hidden states move between device and host. A latency win bought with
    /// throughput.
    Mtp,
    /// Draft with a separate smaller model. The highest acceptance and the
    /// highest cost — a second set of weights resident and a second context.
    DraftModel,
}

impl SpecMethod {
    /// Wire/log spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ngram => "ngram",
            Self::Mtp => "mtp",
            Self::DraftModel => "draft_model",
        }
    }
}

/// What the engine knows about the model it is about to run.
#[derive(Clone, Copy, Debug)]
pub struct ModelProfile {
    /// Mixture-of-experts. Speculation pays far less here: parallel
    /// verification activates experts that single-token decode would not.
    pub is_moe: bool,
    /// The model carries multi-token-prediction heads.
    pub has_mtp: bool,
    /// A separate draft model is loaded and usable.
    pub has_draft_model: bool,
}

impl ModelProfile {
    /// A dense model with no speculative machinery of its own — the case where
    /// n-gram drafting is the only option available.
    #[must_use]
    pub fn dense() -> Self {
        Self { is_moe: false, has_mtp: false, has_draft_model: false }
    }
}

/// Live conditions at the moment of the decision.
#[derive(Clone, Copy, Debug)]
pub struct LoadState {
    /// Sequences currently decoding, including this one.
    pub active_sequences: u32,
    /// Slots the engine can run concurrently.
    pub max_sequences: u32,
    /// Acceptance rate observed so far, if any has been measured.
    ///
    /// This is the strongest signal available and it beats every heuristic
    /// below: a method that is not being accepted should be abandoned no matter
    /// what its profile predicts.
    pub observed_acceptance: Option<f64>,
}

impl LoadState {
    /// A single request on an engine with capacity to spare.
    ///
    /// `max_sequences` is deliberately not 1. A one-slot engine serving one
    /// request is at *full* utilization, not idle, and would be refused
    /// speculation by [`plan`] — which is correct behaviour for that case and
    /// the wrong meaning for this constructor. The number below just has to be
    /// large enough to leave headroom; callers with real numbers should build
    /// the struct directly rather than reach for this.
    #[must_use]
    pub fn idle() -> Self {
        Self { active_sequences: 1, max_sequences: 8, observed_acceptance: None }
    }

    /// Occupancy in `0.0..=1.0`. An engine reporting zero slots is treated as
    /// full rather than empty — the conservative reading, since a zero here
    /// means the caller does not know its own capacity.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_sequences == 0 {
            return 1.0;
        }
        f64::from(self.active_sequences) / f64::from(self.max_sequences)
    }
}

/// Thresholds governing the decision. Every one is a measured starting point,
/// not a constant of nature; override them with numbers from your own workload.
#[derive(Clone, Copy, Debug)]
pub struct SpecPolicy {
    /// Utilization past which speculation is abandoned entirely.
    ///
    /// Default 0.75. Above roughly three-quarters occupancy the verify slots a
    /// draft consumes are slots a queued request wanted, and measured speedups
    /// have collapsed toward 1.0x well before full.
    pub abandon_above_utilization: f64,
    /// Utilization past which the block is shortened rather than dropped.
    /// Default 0.35 — degrade before abandoning.
    pub shorten_above_utilization: f64,
    /// Acceptance below which speculation is abandoned, once measured.
    ///
    /// Default 0.30. Below this a block of typical size costs more in wasted
    /// verification than it saves in skipped decodes. For reference, a good
    /// n-gram implementation reports ~0.70.
    pub abandon_below_acceptance: f64,
    /// Longest block proposed for a dense model on an idle engine.
    pub max_draft: u8,
    /// Cap applied to MoE models, whose verification cost is much higher.
    /// Default 2: speculate, but briefly.
    pub moe_max_draft: u8,
}

impl Default for SpecPolicy {
    fn default() -> Self {
        Self {
            abandon_above_utilization: 0.75,
            shorten_above_utilization: 0.35,
            abandon_below_acceptance: 0.30,
            max_draft: 4,
            moe_max_draft: 2,
        }
    }
}

/// The decision: what to run, how far, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecPlan {
    pub method: SpecMethod,
    /// Tokens to propose per step. Always 0 when `method` is
    /// [`SpecMethod::None`], so a caller cannot act on a draft length that was
    /// never meant to be used.
    pub draft_n: u8,
    /// Why this plan was chosen, for logs and for anyone auditing a decision
    /// after the fact. A policy that cannot explain itself gets switched off.
    pub reason: &'static str,
}

impl SpecPlan {
    /// Decode normally, with the reason recorded.
    #[must_use]
    pub fn none(reason: &'static str) -> Self {
        Self { method: SpecMethod::None, draft_n: 0, reason }
    }

    /// Whether this plan actually speculates.
    #[must_use]
    pub fn speculates(&self) -> bool {
        self.method != SpecMethod::None && self.draft_n > 0
    }
}

/// Choose a speculation plan.
///
/// Order matters: hard refusals first, then method selection, then sizing. Each
/// gate is checked before the ones that would be wasted if it fires.
#[must_use]
pub fn plan(
    model: &ModelProfile,
    load: &LoadState,
    config: &GenerationConfig,
    policy: &SpecPolicy,
) -> SpecPlan {
    // A single token cannot be sped up by predicting ahead of it: the draft
    // would be discarded whatever it said.
    if config.max_tokens <= 1 {
        return SpecPlan::none("request is one token or fewer");
    }

    // Measured acceptance overrides every heuristic below. If drafts are not
    // being taken, the reason does not matter — stop paying for them.
    if let Some(acc) = load.observed_acceptance
        && acc < policy.abandon_below_acceptance
    {
        return SpecPlan::none("observed acceptance below the floor");
    }

    let util = load.utilization();
    if util >= policy.abandon_above_utilization {
        return SpecPlan::none("engine too busy for speculation to pay");
    }

    // Prefer the method with the best acceptance the model can actually
    // support, then fall back. n-gram is last by acceptance but first by cost:
    // it needs no weights and no GPU memory, so it remains available when
    // nothing else is.
    let method = if model.has_draft_model {
        SpecMethod::DraftModel
    } else if model.has_mtp {
        SpecMethod::Mtp
    } else {
        SpecMethod::Ngram
    };

    // MTP typically forces single-sequence batching, so it is only viable while
    // the engine is genuinely serving one request. Choosing it under
    // concurrency would trade a large throughput loss for a small latency gain.
    if method == SpecMethod::Mtp && load.active_sequences > 1 {
        return SpecPlan::none("MTP needs a single sequence; engine is batching");
    }

    let mut draft_n = policy.max_draft;

    // MoE first, because it is the tighter cap and must not be undone by a
    // later widening.
    if model.is_moe {
        draft_n = draft_n.min(policy.moe_max_draft);
    }

    // Degrade before abandoning: halve the block as the engine fills.
    if util >= policy.shorten_above_utilization {
        draft_n = (draft_n / 2).max(1);
    }

    // Never propose more tokens than the request can still use.
    let remaining = u8::try_from(config.max_tokens.saturating_sub(1)).unwrap_or(u8::MAX);
    draft_n = draft_n.min(remaining);

    if draft_n == 0 {
        return SpecPlan::none("no room left in the token budget");
    }

    let reason = match (model.is_moe, util >= policy.shorten_above_utilization) {
        (true, true) => "MoE under load: speculating briefly",
        (true, false) => "MoE: short block, verification activates extra experts",
        (false, true) => "engine filling: block shortened",
        (false, false) => "dense model, engine idle: full block",
    };

    SpecPlan { method, draft_n, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: u32) -> GenerationConfig {
        GenerationConfig { max_tokens, ..Default::default() }
    }

    #[test]
    fn an_idle_dense_model_gets_a_full_block() {
        let p = plan(&ModelProfile::dense(), &LoadState::idle(), &cfg(256), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::Ngram);
        assert_eq!(p.draft_n, 4);
        assert!(p.speculates());
    }

    #[test]
    fn a_busy_engine_does_not_speculate() {
        let load = LoadState { active_sequences: 8, max_sequences: 8, observed_acceptance: None };
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::None);
        assert!(!p.speculates());
    }

    #[test]
    fn a_filling_engine_shortens_before_abandoning() {
        // Half full: past the shorten threshold, short of the abandon one.
        let load = LoadState { active_sequences: 4, max_sequences: 8, observed_acceptance: None };
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert!(p.speculates(), "should still speculate at half load");
        assert_eq!(p.draft_n, 2, "block should be halved, not dropped");
    }

    #[test]
    fn moe_gets_a_shorter_block_than_dense() {
        let moe = ModelProfile { is_moe: true, ..ModelProfile::dense() };
        let dense = plan(&ModelProfile::dense(), &LoadState::idle(), &cfg(256), &SpecPolicy::default());
        let m = plan(&moe, &LoadState::idle(), &cfg(256), &SpecPolicy::default());
        assert!(m.speculates(), "MoE should still speculate when idle");
        assert!(m.draft_n < dense.draft_n, "MoE {} vs dense {}", m.draft_n, dense.draft_n);
    }

    #[test]
    fn measured_acceptance_overrides_everything() {
        // Idle engine, dense model — every heuristic says speculate. Measured
        // acceptance says it is not working, and that must win.
        let load = LoadState { active_sequences: 1, max_sequences: 8, observed_acceptance: Some(0.05) };
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::None);
    }

    #[test]
    fn good_acceptance_does_not_block_speculation() {
        let load = LoadState { active_sequences: 1, max_sequences: 8, observed_acceptance: Some(0.70) };
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert!(p.speculates());
    }

    #[test]
    fn a_draft_model_is_preferred_over_mtp_and_ngram() {
        let m = ModelProfile { is_moe: false, has_mtp: true, has_draft_model: true };
        let p = plan(&m, &LoadState::idle(), &cfg(256), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::DraftModel);
    }

    #[test]
    fn mtp_is_used_only_when_serving_one_sequence() {
        let m = ModelProfile { is_moe: false, has_mtp: true, has_draft_model: false };
        let alone = plan(&m, &LoadState::idle(), &cfg(256), &SpecPolicy::default());
        assert_eq!(alone.method, SpecMethod::Mtp);

        let batching = LoadState { active_sequences: 2, max_sequences: 8, observed_acceptance: None };
        let together = plan(&m, &batching, &cfg(256), &SpecPolicy::default());
        assert_eq!(together.method, SpecMethod::None, "MTP must not run while batching");
    }

    #[test]
    fn a_one_token_request_is_never_speculated() {
        let p = plan(&ModelProfile::dense(), &LoadState::idle(), &cfg(1), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::None);
    }

    #[test]
    fn the_block_never_exceeds_the_remaining_budget() {
        // Two tokens left: at most one can be usefully drafted.
        let p = plan(&ModelProfile::dense(), &LoadState::idle(), &cfg(2), &SpecPolicy::default());
        assert!(p.draft_n <= 1, "drafted {} with 2 tokens budgeted", p.draft_n);
    }

    #[test]
    fn a_single_slot_engine_serving_one_request_is_full_not_idle() {
        // One request on a one-slot engine is 100% utilization. It looks like
        // "just one request" and is in fact no headroom at all — the case that
        // made LoadState::idle() wrong when it used max_sequences: 1.
        let load = LoadState { active_sequences: 1, max_sequences: 1, observed_acceptance: None };
        assert!((load.utilization() - 1.0).abs() < f64::EPSILON);
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::None);
    }

    #[test]
    fn an_unknown_capacity_is_treated_as_full() {
        // max_sequences = 0 means the caller does not know its own capacity.
        // Assume the worst rather than speculate into an unknown.
        let load = LoadState { active_sequences: 1, max_sequences: 0, observed_acceptance: None };
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert_eq!(p.method, SpecMethod::None);
    }

    #[test]
    fn a_none_plan_never_carries_a_draft_length() {
        // A caller that checks only draft_n must not be handed a live-looking
        // block on a plan that decided against speculating.
        let load = LoadState { active_sequences: 8, max_sequences: 8, observed_acceptance: None };
        let p = plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default());
        assert_eq!(p.draft_n, 0);
    }

    #[test]
    fn policy_thresholds_are_honoured() {
        // A deployment that has measured its own numbers must be able to use
        // them; the defaults are a starting point, not a constant.
        let permissive = SpecPolicy { abandon_above_utilization: 0.99, ..SpecPolicy::default() };
        let load = LoadState { active_sequences: 7, max_sequences: 8, observed_acceptance: None };
        assert_eq!(plan(&ModelProfile::dense(), &load, &cfg(256), &SpecPolicy::default()).method, SpecMethod::None);
        assert!(plan(&ModelProfile::dense(), &load, &cfg(256), &permissive).speculates());
    }

    #[test]
    fn every_plan_explains_itself() {
        let cases = [
            (ModelProfile::dense(), LoadState::idle()),
            (ModelProfile { is_moe: true, ..ModelProfile::dense() }, LoadState::idle()),
            (ModelProfile::dense(), LoadState { active_sequences: 8, max_sequences: 8, observed_acceptance: None }),
        ];
        for (m, l) in cases {
            assert!(!plan(&m, &l, &cfg(256), &SpecPolicy::default()).reason.is_empty());
        }
    }
}
