//! Entropy-routed speculation: spending the draft budget where it pays.
//!
//! Published drafters all answer "how do we propose better tokens?". This
//! answers a different question — *which positions are worth proposing at all* —
//! and the theory says that is where the remaining headroom is.
//!
//! ## Why route on entropy
//!
//! The ceiling on any speculative method is `E[tokens per step] ~= log(P)/mu`,
//! for parallel verify capacity `P` and target entropy `mu` in nats. Two things
//! follow. Widening the draft tree buys only `log(P)` — a dead end. And speedup
//! is inversely proportional to entropy, which on measured data varies by
//! roughly 4x across tasks on the same model.
//!
//! Meanwhile **about half of all generated tokens carry entropy below 1e-2
//! nats** — closing brackets, formatting, the tail of an identifier already
//! begun. The rest concentrate at branch points: connectives like "however",
//! "wait", "thus", where the continuation genuinely forks.
//!
//! Every deployed engine picks one block size per *request*. The payoff varies
//! per *token*, by orders of magnitude. That mismatch is the opening.
//!
//! ## Why this can beat the published bound
//!
//! The bound assumes acceptance is **i.i.d.** across positions. It is not:
//! entropy is bursty and autocorrelated, arriving in runs. A method that
//! conditions on *which regime it is in* is not violating the bound — it is
//! operating outside the hypothesis the bound was proved under. That is what
//! [`RegimeTracker`] is for, and it is the part of this module with a real claim
//! to novelty rather than to careful engineering.
//!
//! ## What this module is not
//!
//! It does not estimate entropy from logits — that needs a backend, and lives
//! behind one. This is the decision layer: given an entropy estimate from
//! wherever, decide how far to draft. Keeping it separate means it is testable
//! without a GPU, which is the whole argument for the acceleration layer being
//! a layer.

/// Entropy regimes, in nats.
///
/// The thresholds are read off measured distributions rather than chosen for
/// tidiness: about half of all tokens fall below `deterministic_below`, and the
/// upper region is where connectives and genuine branch points concentrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// Near-deterministic. The next token is all but decided, so drafting it is
    /// nearly free and verification is nearly certain to accept.
    Deterministic,
    /// Ordinary text: worth drafting, worth sizing.
    Predictable,
    /// A branch point. The continuation genuinely forks, so drafting past it
    /// spends verify slots on a coin flip.
    Branch,
}

/// Thresholds separating the regimes, in nats.
#[derive(Clone, Copy, Debug)]
pub struct EntropyThresholds {
    /// Below this, treat the position as decided. Default 1e-2 — the measured
    /// mode of the low-entropy half of the distribution.
    pub deterministic_below: f64,
    /// Above this, treat the position as a branch point. Default 0.672, the
    /// measured 80th percentile: the top fifth of positions carry most of the
    /// uncertainty.
    pub branch_above: f64,
}

impl Default for EntropyThresholds {
    fn default() -> Self {
        Self { deterministic_below: 1e-2, branch_above: 0.672 }
    }
}

impl EntropyThresholds {
    /// Classify an entropy estimate.
    ///
    /// A negative or non-finite estimate is treated as [`Regime::Branch`]: an
    /// estimator that has failed should make the engine cautious, not
    /// confident, and the cost of an unnecessary caution is one un-drafted
    /// token against a wasted verify block.
    #[must_use]
    pub fn classify(&self, nats: f64) -> Regime {
        if !nats.is_finite() || nats < 0.0 {
            return Regime::Branch;
        }
        if nats < self.deterministic_below {
            Regime::Deterministic
        } else if nats > self.branch_above {
            Regime::Branch
        } else {
            Regime::Predictable
        }
    }
}

/// The block size the bound itself prescribes, applied locally.
///
/// `log(P)/mu` is the ceiling on expected accepted tokens for capacity `P` and
/// entropy `mu`. Using it as a *local* target — recomputed per position from
/// local entropy rather than once per request from a global average — is the
/// core of the routing idea.
///
/// Returns 0 when drafting should not happen at all. `verify_capacity` is how
/// many positions the engine can verify in one batch.
#[must_use]
pub fn block_size_for(nats: f64, verify_capacity: u8, thresholds: &EntropyThresholds) -> u8 {
    if verify_capacity <= 1 {
        return 0;
    }
    match thresholds.classify(nats) {
        // A fork: drafting past it is a coin flip, and the verify slots are
        // better left to another sequence.
        Regime::Branch => 0,
        // Effectively decided. Take the full capacity — this is the half of the
        // distribution that makes speculation pay at all.
        Regime::Deterministic => verify_capacity,
        Regime::Predictable => {
            // log(P)/mu, clamped into the usable range. mu is at least the
            // deterministic floor here, so the division is safe.
            let mu = nats.max(thresholds.deterministic_below);
            let target = f64::from(verify_capacity).ln() / mu;
            // Round down: overshooting spends verify slots that will be
            // rejected, and rejection is the expensive direction.
            let n = target.floor().clamp(1.0, f64::from(verify_capacity));
            // Precision-safe: n is already clamped to [1, verify_capacity].
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = n as u8;
            n
        }
    }
}

/// Tracks which entropy regime generation is currently in, and how long runs
/// last.
///
/// The published bounds assume acceptance is i.i.d. across positions. Measured
/// entropy is not — it is bursty, arriving in runs of near-deterministic tokens
/// punctuated by branch points. Knowing *how long the current run has lasted*
/// predicts whether it will continue, which is exactly what a block-size
/// decision needs and exactly what an i.i.d. model cannot supply.
///
/// Deliberately tiny: two counters and a state. Anything heavier would cost
/// more on the decode path than the prediction is worth.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegimeTracker {
    current: Option<Regime>,
    /// Positions observed in the current regime.
    run_length: u32,
    /// Exponential moving average of completed deterministic-run lengths.
    ///
    /// An EMA rather than a full histogram because the useful signal is "are
    /// runs currently long or short", which drifts within a single generation —
    /// code and prose in one response have different characters.
    mean_deterministic_run: f64,
    /// Completed deterministic runs seen, so a caller can tell "no data" from
    /// "genuinely short runs".
    runs_seen: u32,
    /// Total positions observed. Burstiness cannot be judged before enough
    /// have gone by for runs to start and finish.
    observations: u32,
}

/// Weight given to the newest observation in the run-length EMA. 0.25 keeps
/// roughly the last handful of runs in view — long enough to be stable, short
/// enough to follow a shift from prose into code.
const RUN_EMA_ALPHA: f64 = 0.25;

impl RegimeTracker {
    /// Record one observed position.
    pub fn observe(&mut self, regime: Regime) {
        self.observations = self.observations.saturating_add(1);
        match self.current {
            Some(prev) if prev == regime => {
                self.run_length = self.run_length.saturating_add(1);
            }
            Some(prev) => {
                // A run just ended. Only deterministic runs are worth
                // remembering: they are the ones whose continuation the block
                // size is betting on.
                if prev == Regime::Deterministic {
                    let len = f64::from(self.run_length);
                    self.mean_deterministic_run = if self.runs_seen == 0 {
                        len
                    } else {
                        RUN_EMA_ALPHA.mul_add(len, (1.0 - RUN_EMA_ALPHA) * self.mean_deterministic_run)
                    };
                    self.runs_seen = self.runs_seen.saturating_add(1);
                }
                self.current = Some(regime);
                self.run_length = 1;
            }
            None => {
                self.current = Some(regime);
                self.run_length = 1;
            }
        }
    }

    /// Positions observed so far.
    #[must_use]
    pub fn observations(&self) -> u32 {
        self.observations
    }

    /// The regime of the most recent position, if any has been observed.
    #[must_use]
    pub fn current(&self) -> Option<Regime> {
        self.current
    }

    /// How many consecutive positions have shared the current regime.
    #[must_use]
    pub fn run_length(&self) -> u32 {
        self.run_length
    }

    /// Average length of completed deterministic runs, once any have completed.
    #[must_use]
    pub fn mean_deterministic_run(&self) -> Option<f64> {
        (self.runs_seen > 0).then_some(self.mean_deterministic_run)
    }

    /// Whether the text seen so far is bursty enough for regime awareness to
    /// be worth its cost.
    ///
    /// PRAX's machinery — local sizing, run discounting — is insurance against
    /// entropy arriving in runs. On text with no such structure the insurance
    /// has no claim to pay and is pure overhead: measured on a uniform trace,
    /// PRAX lost to a fixed block at every concurrency, by up to 69%, because
    /// it kept drafting conservatively where every position would have been
    /// accepted. Notably a fixed block on such text already scores within a
    /// fraction of a perfect oracle, so there is nothing to win and real cost
    /// to pay.
    ///
    /// Burstiness shows up as *variety* in run length. Uniform text produces
    /// runs of one regime that never end, or none at all; bursty text produces
    /// deterministic runs that repeatedly start and finish. So: has this stream
    /// actually completed deterministic runs of non-trivial length?
    #[must_use]
    pub fn is_bursty(&self) -> bool {
        // Two completed runs is the minimum evidence of a *pattern* rather than
        // an accident, and a mean above 1 rules out single-position flickers.
        self.runs_seen >= 2 && self.mean_deterministic_run > 1.0
    }

    /// How likely the current run is to continue, in `(0, 1]`.
    ///
    /// This is the quantity an i.i.d. model cannot supply, and the reason
    /// regime tracking earns its place. A block spans the positions *after* the
    /// current one, so its value depends not on this token's acceptance but on
    /// whether the run it belongs to persists. Pricing a block at the current
    /// token's acceptance alone silently assumes the run never ends — which is
    /// how an allocator overdrafts into a loaded engine.
    ///
    /// A run already longer than typical is nearer its end, so the estimate
    /// decays as `run_length` passes `mean_deterministic_run`. Outside a
    /// deterministic run there is nothing to continue, so the answer is 1 and
    /// the caller's own acceptance estimate stands unmodified.
    #[must_use]
    pub fn continuation_odds(&self) -> f64 {
        if self.current != Some(Regime::Deterministic) {
            return 1.0;
        }
        let Some(mean) = self.mean_deterministic_run() else {
            // No completed runs yet: no evidence either way, so do not adjust.
            return 1.0;
        };
        if mean <= 0.0 {
            return 1.0;
        }
        // Geometric survival: a run of mean length `m` continues with
        // probability about `1 - 1/m` per position, and confidence falls the
        // further past `m` we already are. Floored so the estimate degrades
        // rather than collapsing to zero on an unusually long run.
        let progress = f64::from(self.run_length) / mean;
        (1.0 - 1.0 / mean.max(1.0)).powf(progress.max(1.0)).max(0.25)
    }

    /// Adjust a block size using regime history.
    ///
    /// The i.i.d. bound has no way to express this: inside a deterministic run
    /// that is already longer than typical, continuation is *more* likely than
    /// the position's own entropy suggests, because runs are autocorrelated.
    /// Conversely a run that has just started deserves no extra confidence.
    ///
    /// Deliberately conservative — it extends only inside a deterministic run,
    /// and never past `verify_capacity`. Speculating wrongly costs a whole
    /// verify block; speculating short costs one token.
    #[must_use]
    pub fn adjust(&self, base: u8, verify_capacity: u8) -> u8 {
        if base == 0 || self.current != Some(Regime::Deterministic) {
            return base;
        }
        match self.mean_deterministic_run() {
            // Established runs are long and this one is already past average:
            // the next positions are very likely still deterministic.
            Some(mean) if mean >= 2.0 && f64::from(self.run_length) >= mean => {
                base.saturating_add(1).min(verify_capacity)
            }
            _ => base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn th() -> EntropyThresholds {
        EntropyThresholds::default()
    }

    #[test]
    fn regimes_split_at_the_measured_thresholds() {
        let t = th();
        assert_eq!(t.classify(0.0), Regime::Deterministic);
        assert_eq!(t.classify(0.001), Regime::Deterministic);
        assert_eq!(t.classify(0.1), Regime::Predictable);
        assert_eq!(t.classify(1.5), Regime::Branch);
    }

    #[test]
    fn a_broken_estimate_is_treated_as_a_branch() {
        // An estimator that has failed must make us cautious, not confident.
        let t = th();
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            assert_eq!(t.classify(bad), Regime::Branch, "{bad} should be cautious");
        }
    }

    #[test]
    fn a_branch_point_is_not_drafted() {
        // The core claim: spending verify slots on a coin flip is the waste
        // this module exists to avoid.
        assert_eq!(block_size_for(2.0, 8, &th()), 0);
    }

    #[test]
    fn a_decided_token_takes_the_full_capacity() {
        assert_eq!(block_size_for(0.0001, 8, &th()), 8);
    }

    #[test]
    fn a_predictable_token_gets_a_middling_block() {
        let n = block_size_for(0.3, 8, &th());
        assert!((1..8).contains(&n), "expected a middling block, got {n}");
    }

    #[test]
    fn block_size_falls_as_entropy_rises() {
        // The monotonicity is the whole point: higher entropy, less drafting.
        let t = th();
        let a = block_size_for(0.05, 16, &t);
        let b = block_size_for(0.30, 16, &t);
        let c = block_size_for(0.60, 16, &t);
        assert!(a >= b && b >= c, "not monotone: {a}, {b}, {c}");
    }

    #[test]
    fn no_verify_capacity_means_no_drafting() {
        // Capacity 1 can verify only the token being generated, so a draft
        // could never be checked.
        assert_eq!(block_size_for(0.0, 1, &th()), 0);
        assert_eq!(block_size_for(0.0, 0, &th()), 0);
    }

    #[test]
    fn a_block_never_exceeds_verify_capacity() {
        let t = th();
        for nats in [0.0, 0.001, 0.05, 0.2, 0.5, 0.67] {
            for cap in [2u8, 4, 8, 16] {
                assert!(
                    block_size_for(nats, cap, &t) <= cap,
                    "nats={nats} cap={cap} produced an oversized block"
                );
            }
        }
    }

    #[test]
    fn a_tracker_starts_empty() {
        let t = RegimeTracker::default();
        assert_eq!(t.current(), None);
        assert_eq!(t.mean_deterministic_run(), None);
    }

    #[test]
    fn runs_are_counted() {
        let mut t = RegimeTracker::default();
        for _ in 0..5 {
            t.observe(Regime::Deterministic);
        }
        assert_eq!(t.current(), Some(Regime::Deterministic));
        assert_eq!(t.run_length(), 5);
    }

    #[test]
    fn a_completed_deterministic_run_is_remembered() {
        let mut t = RegimeTracker::default();
        for _ in 0..4 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch); // ends the run
        assert_eq!(t.mean_deterministic_run(), Some(4.0));
        assert_eq!(t.run_length(), 1, "the new run restarts the counter");
    }

    #[test]
    fn only_deterministic_runs_are_remembered() {
        // Branch runs say nothing about whether drafting will pay.
        let mut t = RegimeTracker::default();
        for _ in 0..6 {
            t.observe(Regime::Branch);
        }
        t.observe(Regime::Predictable);
        assert_eq!(t.mean_deterministic_run(), None);
    }

    #[test]
    fn a_fresh_tracker_is_not_bursty() {
        // With no evidence, assume no structure — the conservative default,
        // since claiming burstiness costs performance on uniform text.
        assert!(!RegimeTracker::default().is_bursty());
    }

    #[test]
    fn repeated_runs_read_as_bursty() {
        let mut t = RegimeTracker::default();
        for _ in 0..3 {
            for _ in 0..5 {
                t.observe(Regime::Deterministic);
            }
            t.observe(Regime::Branch);
        }
        assert!(t.is_bursty(), "three completed runs of five is clearly bursty");
    }

    #[test]
    fn uniform_text_never_reads_as_bursty() {
        // The regression this exists to prevent: uniform text has no
        // deterministic runs at all, so nothing should claim structure.
        let mut t = RegimeTracker::default();
        for _ in 0..200 {
            t.observe(Regime::Predictable);
        }
        assert!(!t.is_bursty());
    }

    #[test]
    fn one_run_is_not_yet_a_pattern() {
        let mut t = RegimeTracker::default();
        for _ in 0..8 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        assert!(!t.is_bursty(), "a single run is an accident, not a pattern");
    }

    #[test]
    fn continuation_odds_are_neutral_without_evidence() {
        // No completed runs means no basis to discount; the caller's own
        // estimate must pass through unchanged.
        let t = RegimeTracker::default();
        assert!((t.continuation_odds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn continuation_odds_are_neutral_outside_a_run() {
        let mut t = RegimeTracker::default();
        for _ in 0..5 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        assert!((t.continuation_odds() - 1.0).abs() < 1e-9, "a fork has no run to continue");
    }

    #[test]
    fn a_run_past_its_typical_length_is_discounted() {
        // The correction that stops an allocator assuming a run lasts forever.
        let mut t = RegimeTracker::default();
        for _ in 0..4 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch); // mean run = 4

        let mut fresh = t;
        fresh.observe(Regime::Deterministic);
        let early = fresh.continuation_odds();

        let mut stretched = t;
        for _ in 0..12 {
            stretched.observe(Regime::Deterministic);
        }
        let late = stretched.continuation_odds();

        assert!(late < early, "a long run should be discounted more: {late} vs {early}");
        assert!(late >= 0.25, "the estimate should degrade, not collapse");
    }

    #[test]
    fn a_long_established_run_extends_the_block() {
        // The claim the i.i.d. bound cannot express: inside a run already
        // longer than typical, continuation is likelier than this position's
        // own entropy implies.
        let mut t = RegimeTracker::default();
        for _ in 0..6 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch); // completes a 6-long run
        for _ in 0..6 {
            t.observe(Regime::Deterministic);
        }
        assert_eq!(t.adjust(4, 8), 5, "an established long run should extend");
    }

    #[test]
    fn a_fresh_run_gets_no_extension() {
        let mut t = RegimeTracker::default();
        for _ in 0..6 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        t.observe(Regime::Deterministic); // run of 1, well under the mean
        assert_eq!(t.adjust(4, 8), 4);
    }

    #[test]
    fn adjustment_never_exceeds_capacity_or_revives_a_refusal() {
        let mut t = RegimeTracker::default();
        for _ in 0..8 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        for _ in 0..8 {
            t.observe(Regime::Deterministic);
        }
        assert_eq!(t.adjust(8, 8), 8, "must not exceed capacity");
        assert_eq!(t.adjust(0, 8), 0, "a refusal to draft must stay a refusal");
    }

    #[test]
    fn a_branch_regime_is_never_extended() {
        let mut t = RegimeTracker::default();
        for _ in 0..9 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        assert_eq!(t.adjust(2, 8), 2, "extending at a fork is what we must not do");
    }

    #[test]
    fn the_run_average_follows_a_shift_in_character() {
        // Prose and code in one response have different run characters; a
        // fixed average would lag the transition.
        let mut t = RegimeTracker::default();
        for _ in 0..2 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        let short = t.mean_deterministic_run().unwrap();
        for _ in 0..20 {
            t.observe(Regime::Deterministic);
        }
        t.observe(Regime::Branch);
        assert!(t.mean_deterministic_run().unwrap() > short, "the average should move");
    }
}
