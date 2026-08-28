//! Allocator simulation. **RETIRED: its central assumption was refuted by
//! measurement.** Kept as a record of how a self-consistent simulation can be
//! confidently wrong.
//!
//! This harness modelled the cost of a speculative draft block as
//! `drafted x slot_cost`, i.e. every drafted position pays verification whether
//! or not it is accepted. Under that model an oversized block is expensive, an
//! adaptive allocator beats a fixed one by +152%, and [`crate::prax`] looks
//! valuable.
//!
//! **Hardware disagrees.** llama.cpp verifies a whole block in one batched
//! pass, so a larger block that is mostly rejected costs barely more than a
//! small one. Measured across a 16x range in block size, throughput moved ~5%
//! non-monotonically and acceptance did not move at all. The cost this harness
//! charges for is largely not charged in reality.
//!
//! What makes this worth keeping rather than deleting: the harness was
//! *rigorous*. It had an oracle upper bound, averaged over seeds, controlled
//! for regime memory against an otherwise-identical allocator, and caught
//! several real bugs including two metrics of its own that were wrong. None of
//! that could detect a mistaken premise. A simulation can only ever be as right
//! as its cost model, and the only test of a cost model is measurement.
//!
//! Comparing allocators on synthetic token streams.
//!
//! [`crate::prax`] claims that allocating the draft budget by *entropy regime*
//! beats allocating it by a fixed block size. That claim needs evidence, and
//! this module is built to produce it — including evidence against.
//!
//! ## What this measures, and what it cannot
//!
//! The figure of merit is **net time saved**, following the published model
//!
//! ```text
//!     dT(b) = D(b) * [ (1 - rho(b)) * t_A  -  t_V(b) ]
//! ```
//!
//! where `D(b)` is tokens drafted, `rho(b)` the rejection rate, `t_A` the time
//! saved per accepted token and `t_V` the verification cost paid **per drafted
//! token, accepted or not**. Every drafted position costs `t_V`; only accepted
//! ones return `t_A`. Drafting further therefore pays exactly while
//! `(1 - rho) * t_A > t_V`.
//!
//! ## Two metrics this module got wrong before, and why they mattered
//!
//! **Scoring with the closed form was wrong.** `(1 - a^(g+1)) / (1 - a)`
//! predicts yield from a block size and an average acceptance, which credits an
//! allocator for *asking* for a large block rather than for landing one. Under
//! it a perfect oracle scored *below* a fixed maximum, because the oracle
//! honestly declines to draft at a branch point. An impossible result is the
//! useful kind: it says the metric is wrong, not the allocator. The closed form
//! is right for *predicting* a block's value — [`crate::verify::expected_tokens`]
//! uses it inside PRAX for exactly that — and wrong for *scoring* what happened.
//!
//! **Counting emitted tokens was also wrong**, more subtly. With no cost for
//! drafting, asking for the maximum block is free, so a fixed maximum is optimal
//! *by construction* and ties the oracle exactly. No adaptive policy can win a
//! game where the resource it economises is free. That is not evidence against
//! adaptive allocation; it is a harness that cannot test it.
//!
//! Both are recorded because each produced a confident, wrong answer, and the
//! second looked entirely reasonable.
//!
//! What it cannot measure: real drafter acceptance, kernel behaviour, batching
//! effects, or anything about a specific model. The acceptance here is
//! *generated* from an entropy trace, using the standard relation that
//! acceptance falls as entropy rises. That makes this a test of the
//! **allocation policy**, holding the drafter constant — which is exactly the
//! comparison PRAX's claim is about, and no more.
//!
//! A result here is evidence about allocation. It is not a benchmark, and a
//! number from this module should never be reported as a speedup.
//!
//! ## Why synthetic traces
//!
//! The claim is conditional: PRAX should win where entropy is **bursty** and
//! merely tie where it is **uniform**. Testing that needs traces whose
//! burstiness is known and controllable, which real logs do not provide. The
//! generators below are deliberately simple and deterministic — a seeded LCG,
//! no dependencies — so a result is reproducible and a disagreement is about
//! the policy rather than about sampling noise.

use crate::entropy::EntropyThresholds;
use crate::prax::{Prax, Signals};

/// A deterministic stream of per-position entropies, in nats.
pub type Trace = Vec<f64>;

/// Tiny seeded generator, so traces are reproducible without a dependency.
///
/// Quality is irrelevant here: the traces need to be varied and repeatable, not
/// statistically pristine.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    /// Next value in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 53 bits, the usual trick for a uniform double.
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A **bursty** trace: long near-deterministic runs punctuated by branch points.
///
/// This is the structure measured in real generation — roughly half of tokens
/// near-deterministic, with high-entropy positions clustering at connectives —
/// and the structure the published i.i.d. bounds assume away. It is where PRAX
/// should win if it wins anywhere.
#[must_use]
pub fn bursty_trace(len: usize, seed: u64) -> Trace {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        // A run of near-deterministic positions, length 3..=12.
        let run = 3 + (rng.next_f64() * 10.0) as usize;
        for _ in 0..run.min(len - out.len()) {
            out.push(rng.next_f64() * 5e-3); // well under the 1e-2 floor
        }
        if out.len() < len {
            // A branch point, or a short cluster of them.
            let branch = 1 + (rng.next_f64() * 2.0) as usize;
            for _ in 0..branch.min(len - out.len()) {
                out.push(0.8 + rng.next_f64() * 1.5);
            }
        }
    }
    out.truncate(len);
    out
}

/// A **uniform** trace: every position drawn from the same middling band.
///
/// The adversarial case for PRAX. With no regime structure to exploit there is
/// nothing for regime awareness to find, so PRAX should tie a well-chosen fixed
/// block here — and if it *loses*, the extra machinery is not paying for itself.
#[must_use]
pub fn uniform_trace(len: usize, seed: u64) -> Trace {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| 0.15 + rng.next_f64() * 0.3).collect()
}

/// A **high-entropy** trace: mostly branch points.
///
/// Speculation should largely be declined here. An allocator that keeps
/// drafting into it is burning verify capacity, which a tokens-per-step figure
/// alone will not reveal — hence [`Outcome::wasted_draft_slots`].
#[must_use]
pub fn chaotic_trace(len: usize, seed: u64) -> Trace {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| 0.7 + rng.next_f64() * 2.0).collect()
}

/// Acceptance implied by an entropy, under the standard relation that
/// acceptance falls as the target distribution spreads.
///
/// `exp(-nats)` is the natural choice: it is 1 at zero entropy, decays
/// smoothly, and is bounded in `(0, 1]`. It is a *model*, not a measurement,
/// and it is applied identically to every allocator — so it cannot favour one.
#[must_use]
pub fn acceptance_from_entropy(nats: f64) -> f64 {
    (-nats.max(0.0)).exp()
}

/// Whether the position at `index` is accepted, deterministically.
///
/// Acceptance is a Bernoulli draw with probability `a`, and the harness needs
/// it to be reproducible — so the draw is replaced by a hash of the position,
/// which is deterministic per trace but uncorrelated with `a`.
///
/// An earlier version simply tested `a > 0.5`, which made acceptance *certain*
/// wherever entropy was middling. That is not a milder model, it is a different
/// one: real acceptance compounds along a block, so position `k` is reached
/// with probability `a^k`, while a threshold rule reaches it with probability 1.
/// Under the threshold rule the whole uniform-entropy band accepted perfectly
/// and the optimal block was always the maximum — which made any allocator that
/// correctly priced compounding risk look like it was under-drafting. PRAX was
/// right and the harness was wrong.
#[must_use]
fn accepts(nats: f64, index: usize) -> bool {
    let a = acceptance_from_entropy(nats);
    // Deterministic uniform in [0,1) from the position alone.
    let h = (index as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(31)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let u = ((h >> 11) as f64) / ((1u64 << 53) as f64);
    u < a
}

/// Relative cost of drafting and verifying one token.
///
/// Both are expressed against one target decode step, so `t_a = 1.0` means an
/// accepted token saves exactly one full decode.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    /// Time saved per accepted token. One full target step, by definition.
    pub t_a: f64,
    /// Verification cost per **drafted** token, paid whether or not it is
    /// accepted. This is the term that makes a longer block genuinely
    /// expensive, and its absence is what made an earlier version of this
    /// harness unable to distinguish any allocator from a fixed maximum.
    pub t_v: f64,
    /// Extra verification cost per token per concurrent sequence.
    ///
    /// Verify capacity is shared, so a drafted position under load displaces a
    /// queued request. This is why published speedups collapse from ~3x at
    /// concurrency 1 to ~1x at 32, and why an allocator that economises slots
    /// can win under load while merely tying when idle.
    pub t_v_per_concurrent: f64,
}

impl Default for CostModel {
    /// A single request on an idle engine: verification is cheap relative to a
    /// target step, which is the regime published speedups are measured in.
    fn default() -> Self {
        Self { t_a: 1.0, t_v: 0.15, t_v_per_concurrent: 0.0 }
    }
}

impl CostModel {
    /// Cost model at a given concurrency.
    #[must_use]
    pub fn at_concurrency(concurrent: u32) -> Self {
        Self {
            t_a: 1.0,
            t_v: 0.15,
            // Shared capacity: each extra sequence makes a drafted slot dearer.
            t_v_per_concurrent: 0.02 * f64::from(concurrent.saturating_sub(1)),
        }
    }

    /// Verification cost of one drafted token.
    #[must_use]
    pub fn draft_cost(&self) -> f64 {
        self.t_v + self.t_v_per_concurrent
    }
}

/// What one allocator achieved on one trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    /// Tokens emitted per verification step. Higher is better; 1.0 is plain
    /// decoding. Kept because it is the number papers quote, but it is *not*
    /// the figure of merit — see [`Outcome::net_saving`].
    pub tokens_per_step: f64,
    /// Net time saved over plain decoding, in target-step units. **This is the
    /// figure of merit**: it charges for drafting, so an allocator cannot win
    /// by asking for more than it can land.
    pub net_saving: f64,
    /// Verification steps taken to cover the trace.
    pub steps: u32,
    /// Draft slots spent on positions that were not accepted.
    pub wasted_draft_slots: u32,
}

/// An allocation strategy under test.
pub trait Allocator {
    /// Block size for the position at `index`, given the trace.
    ///
    /// The whole trace is passed because a *fixed* allocator ignores it while
    /// PRAX uses only the current position — passing it makes clear that no
    /// allocator here is allowed to look ahead.
    fn block_for(&mut self, trace: &Trace, index: usize, verify_capacity: u8) -> u8;

    /// Observe what happened, for allocators that adapt.
    fn record(&mut self, _accepted: u8, _block: u8) {}

    fn name(&self) -> &'static str;
}

/// The universal baseline: one block size, always. Every deployed engine.
pub struct Fixed(pub u8);

impl Allocator for Fixed {
    fn block_for(&mut self, _trace: &Trace, _index: usize, verify_capacity: u8) -> u8 {
        self.0.min(verify_capacity)
    }

    fn name(&self) -> &'static str {
        "fixed"
    }
}

/// PRAX, using only the current position's entropy — no lookahead.
pub struct PraxAllocator {
    prax: Prax,
    slot_cost: f64,
}

impl PraxAllocator {
    #[must_use]
    pub fn new() -> Self {
        Self { prax: Prax::new(), slot_cost: CostModel::default().draft_cost() }
    }

    #[must_use]
    pub fn with_thresholds(t: EntropyThresholds) -> Self {
        Self { prax: Prax::with_thresholds(t), slot_cost: CostModel::default().draft_cost() }
    }
}

impl Default for PraxAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl PraxAllocator {
    /// Tell PRAX what a drafted slot costs, so it can price its own blocks.
    /// Without this the allocator is blind to load and keeps drafting into a
    /// busy engine — the defect this harness found.
    #[must_use]
    pub fn at_slot_cost(slot_cost: f64) -> Self {
        Self { prax: Prax::new(), slot_cost }
    }
}

impl Allocator for PraxAllocator {
    fn block_for(&mut self, trace: &Trace, index: usize, verify_capacity: u8) -> u8 {
        let signals = Signals {
            entropy_nats: trace.get(index).copied(),
            slot_cost: self.slot_cost,
            ..Signals::unknown(verify_capacity)
        };
        self.prax.allocate(&signals).block
    }

    fn record(&mut self, accepted: u8, block: u8) {
        self.prax.record_outcome(accepted, block);
    }

    fn name(&self) -> &'static str {
        "prax"
    }
}

/// An allocator with perfect foresight: it knows exactly how many of the next
/// positions will be accepted.
///
/// Not achievable — it is the ceiling. Included so a result can be read as a
/// *fraction of what was available*, which is far more informative than PRAX
/// beating a baseline by some margin with no sense of the room remaining.
pub struct Oracle;

impl Allocator for Oracle {
    fn block_for(&mut self, trace: &Trace, index: usize, verify_capacity: u8) -> u8 {
        // Draft exactly as far as acceptance will actually hold.
        let mut n = 0u8;
        while n < verify_capacity {
            match trace.get(index + n as usize) {
                Some(&nats) if accepts(nats, index + n as usize) => n += 1,
                _ => break,
            }
        }
        n
    }

    fn name(&self) -> &'static str {
        "oracle"
    }
}

/// Run one allocator over one trace.
///
/// Acceptance is drawn deterministically from the trace, identically for every
/// allocator, so any difference in the result is attributable to allocation
/// alone.
pub fn run(alloc: &mut dyn Allocator, trace: &Trace, verify_capacity: u8) -> Outcome {
    run_with_cost(alloc, trace, verify_capacity, &CostModel::default())
}

/// Run one allocator under an explicit cost model.
pub fn run_with_cost(
    alloc: &mut dyn Allocator,
    trace: &Trace,
    verify_capacity: u8,
    cost: &CostModel,
) -> Outcome {
    let mut pos = 0usize;
    let mut steps = 0u32;
    let mut total_tokens = 0.0f64;
    let mut wasted = 0u32;
    let mut saved = 0.0f64;

    while pos < trace.len() {
        let block = alloc.block_for(trace, pos, verify_capacity);
        steps += 1;

        if block == 0 {
            // No draft: one token, the target's own. Nothing saved, nothing
            // spent — the baseline every allocator is measured against.
            total_tokens += 1.0;
            alloc.record(0, 0);
            pos += 1;
            continue;
        }

        // How far acceptance actually holds, by the same rule for everyone.
        let mut accepted = 0u8;
        while accepted < block {
            match trace.get(pos + accepted as usize) {
                Some(&nats) if accepts(nats, pos + accepted as usize) => accepted += 1,
                _ => break,
            }
        }
        wasted += u32::from(block - accepted);

        // Count what was actually emitted: the accepted prefix plus the bonus
        // token the target supplies regardless.
        //
        // NOT expected_tokens(mean_acceptance, block). That scores an allocator
        // for *asking* for a large block rather than for landing one, so a
        // fixed maximum "wins" by always asking and a perfect oracle "loses" by
        // honestly declining to draft at a branch point. Measured that way the
        // oracle came out below both — which is impossible, and was the signal
        // the metric was wrong rather than the allocators.
        total_tokens += f64::from(accepted) + 1.0;

        // dT = accepted*t_a - drafted*t_v. Every drafted position is verified;
        // only accepted ones save anything.
        saved += f64::from(accepted).mul_add(cost.t_a, -(f64::from(block) * cost.draft_cost()));

        alloc.record(accepted, block);
        // Advance by what was accepted, plus the bonus token.
        pos += accepted as usize + 1;
    }

    Outcome {
        tokens_per_step: if steps == 0 { 0.0 } else { total_tokens / f64::from(steps) },
        net_saving: saved,
        steps,
        wasted_draft_slots: wasted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAP: u8 = 8;
    const LEN: usize = 2000;

    /// Best fixed block on a trace under a given cost model — the strongest
    /// honest baseline. Comparing against a badly chosen constant would rig the
    /// result, so the baseline is always the best constant available.
    fn best_fixed(trace: &Trace, cost: &CostModel) -> (u8, Outcome) {
        (1..=CAP)
            .map(|b| (b, run_with_cost(&mut Fixed(b), trace, CAP, cost)))
            .max_by(|a, b| a.1.net_saving.total_cmp(&b.1.net_saving))
            .expect("at least one block size")
    }

    fn prax_on(trace: &Trace, cost: &CostModel) -> Outcome {
        run_with_cost(&mut PraxAllocator::new(), trace, CAP, cost)
    }

    #[test]
    fn traces_have_the_structure_they_claim() {
        // If the generators are wrong, every result below is meaningless.
        let bursty = bursty_trace(LEN, 1);
        let uniform = uniform_trace(LEN, 1);
        let low = |t: &Trace| t.iter().filter(|&&n| n < 1e-2).count() as f64 / t.len() as f64;

        assert!(low(&bursty) > 0.4, "bursty should be ~half low-entropy: {}", low(&bursty));
        assert!(low(&uniform) < 0.01, "uniform should have no low-entropy mass");
    }

    #[test]
    fn drafting_is_not_free() {
        // The defect that made an earlier harness untestable: with no cost per
        // drafted token, asking for the maximum was always optimal and no
        // adaptive policy could ever win. Assert the cost exists.
        let c = CostModel::default();
        assert!(c.draft_cost() > 0.0, "a drafted token must cost something");
        assert!(
            CostModel::at_concurrency(16).draft_cost() > c.draft_cost(),
            "shared verify capacity must get dearer under load"
        );
    }

    #[test]
    fn prax_beats_the_best_fixed_block_on_bursty_text() {
        // THE CLAIM. Failing this refutes PRAX's reason to exist.
        let trace = bursty_trace(LEN, 42);
        let cost = CostModel::default();
        let (bb, fixed) = best_fixed(&trace, &cost);
        let prax = prax_on(&trace, &cost);

        assert!(
            prax.net_saving > fixed.net_saving,
            "PRAX saved {:.1} against the best fixed block ({bb}) at {:.1}",
            prax.net_saving,
            fixed.net_saving
        );
    }

    #[test]
    fn prax_does_not_lose_on_uniform_text() {
        // The honest limit: no regime structure means nothing to exploit, so
        // PRAX should roughly tie. Losing badly would mean the machinery costs
        // more than it returns.
        let trace = uniform_trace(LEN, 7);
        let cost = CostModel::default();
        let (_, fixed) = best_fixed(&trace, &cost);
        let mut alloc = PraxAllocator::at_slot_cost(cost.draft_cost());
        let prax = run_with_cost(&mut alloc, &trace, CAP, &cost);

        // A 10% tolerance, which an earlier version used, let a 69% loss pass
        // because the floor scaled with a positive baseline. Compare against the
        // room actually available instead: on structureless text a fixed block
        // is already near the oracle, so any real loss is a large fraction of a
        // small gap and must be caught.
        let oracle = run_with_cost(&mut Oracle, &trace, CAP, &cost).net_saving;
        let shortfall = fixed.net_saving - prax.net_saving;
        let room = (oracle - fixed.net_saving).abs().max(1.0);
        assert!(
            shortfall <= room,
            "PRAX {:.0} lost {:.0} to fixed {:.0} on uniform text; only {:.0} was ever available",
            prax.net_saving,
            shortfall,
            fixed.net_saving,
            room
        );
    }

    #[test]
    fn prax_does_not_lose_on_uniform_text_under_load() {
        // The case a lenient tolerance hid: at concurrency the uniform-text
        // regression was 69%, and the test still passed.
        let trace = uniform_trace(LEN, 7);
        let cost = CostModel::at_concurrency(32);
        let (_, fixed) = best_fixed(&trace, &cost);
        let mut alloc = PraxAllocator::at_slot_cost(cost.draft_cost());
        let prax = run_with_cost(&mut alloc, &trace, CAP, &cost).net_saving;

        // On structureless text there is nothing for any allocator to find, so
        // matching a fixed block is the right answer rather than a shortfall.
        // Measured both at -49 under load: identical, which is the outcome to
        // protect. Tolerance is absolute, not proportional — a proportional
        // floor around a negative baseline inverts and silently passes a loss.
        let shortfall = fixed.net_saving - prax;
        assert!(
            shortfall <= fixed.net_saving.abs().mul_add(0.05, 1.0),
            "PRAX {prax:.0} lost {shortfall:.0} to fixed {:.0} under load on uniform text",
            fixed.net_saving
        );
    }

    #[test]
    fn margin_over_fixed_is_recorded_across_load() {
        // The mechanism PRAX actually exploits: when verify slots are shared,
        // not wasting them is worth more. Published speedups collapse from ~3x
        // at concurrency 1 to ~1x at 32 precisely because fixed blocks keep
        // spending slots they cannot land.
        let trace = bursty_trace(LEN, 13);

        let idle = CostModel::at_concurrency(1);
        let busy = CostModel::at_concurrency(32);

        let margin = |c: &CostModel| {
            let (_, f) = best_fixed(&trace, c);
            // PRAX is told what a slot costs, exactly as the fixed baseline is
            // charged for one. Withholding it would test a blind allocator.
            let mut alloc = PraxAllocator::at_slot_cost(c.draft_cost());
            run_with_cost(&mut alloc, &trace, CAP, c).net_saving - f.net_saving
        };

        let idle_margin = margin(&idle);
        let busy_margin = margin(&busy);
        println!("\n  margin over best-fixed: c=1 {idle_margin:+.0}   c=32 {busy_margin:+.0}");

        // The claim, stated as what the measurement supports: PRAX must still
        // beat the best fixed block under load. Requiring the margin to GROW
        // was a stronger claim than the evidence carries once run-length sizing
        // was removed, and a test should assert what is true rather than what
        // would be flattering.
        // MEASURED, and it does not go PRAX's way: +69 at concurrency 1,
        // **-194 at 32**. The earlier result showing the margin growing with
        // load came from a run-length discount that was removed after its
        // justification — acceptance autocorrelation — measured at +0.189
        // against a pre-registered 0.2 threshold.
        //
        // So the discount was carrying the concurrency win while resting on a
        // premise the measurement does not support. Both cannot stand. This is
        // recorded rather than asserted, because the honest state is "entropy
        // routing plus slot pricing wins when idle and loses under load", and a
        // green test claiming otherwise would be the expensive kind of wrong.
        // Recorded, not asserted. The per-curve-best baseline is chosen with
        // hindsight no deployment gets; the comparison that decides whether to
        // enable this is in `mod deployed`, against ONE fixed block held across
        // a varied workload. Against that baseline the headroom is large — a
        // perfect allocator is worth +185% — and three of eight workloads send
        // a single fixed block NEGATIVE, i.e. slower than not speculating.
        println!("  (per-curve-best is a hindsight baseline; see mod deployed)");
    }

    #[test]
    fn prax_wastes_fewer_slots_on_chaotic_text() {
        let trace = chaotic_trace(LEN, 3);
        let cost = CostModel::default();
        let prax = run_with_cost(&mut PraxAllocator::new(), &trace, CAP, &cost);
        let fixed = run_with_cost(&mut Fixed(CAP), &trace, CAP, &cost);

        assert!(
            prax.wasted_draft_slots < fixed.wasted_draft_slots,
            "PRAX wasted {} slots vs fixed {}",
            prax.wasted_draft_slots,
            fixed.wasted_draft_slots
        );
    }

    #[test]
    fn the_oracle_beats_everything_as_it_must() {
        // Sanity check on the harness: if anything beats perfect foresight the
        // measurement is wrong, and every other result is suspect. This is how
        // the first broken metric was caught.
        let trace = bursty_trace(LEN, 11);
        let cost = CostModel::default();
        let oracle = run_with_cost(&mut Oracle, &trace, CAP, &cost);
        let prax = prax_on(&trace, &cost);
        let (_, fixed) = best_fixed(&trace, &cost);

        assert!(oracle.net_saving >= prax.net_saving, "oracle beaten by PRAX");
        assert!(oracle.net_saving >= fixed.net_saving, "oracle beaten by fixed");
    }

    #[test]
    fn prax_closes_part_of_the_gap_to_the_oracle() {
        // How much of the available room does allocation recover? Stated as a
        // fraction so the number means something on its own.
        let trace = bursty_trace(LEN, 5);
        let cost = CostModel::default();
        let oracle = run_with_cost(&mut Oracle, &trace, CAP, &cost).net_saving;
        let (_, fixed) = best_fixed(&trace, &cost);
        let prax = prax_on(&trace, &cost).net_saving;

        let room = oracle - fixed.net_saving;
        assert!(room > 0.0, "no room between fixed and oracle; the trace is uninformative");
        let closed = (prax - fixed.net_saving) / room;
        assert!(closed > 0.0, "PRAX closed none of the gap ({closed:.3})");
    }

    #[test]
    fn results_hold_across_seeds() {
        // One favourable seed proves nothing.
        let cost = CostModel::default();
        let mut wins = 0;
        for seed in 0..8u64 {
            let trace = bursty_trace(LEN, seed);
            let (_, fixed) = best_fixed(&trace, &cost);
            if prax_on(&trace, &cost).net_saving > fixed.net_saving {
                wins += 1;
            }
        }
        // Recorded rather than thresholded. The count moves as the cost model
        // and the reach cap are tuned, and a threshold here would be a ratchet
        // that quietly discourages honest changes to either.
        println!("\n  PRAX beat the best fixed block on {wins}/8 bursty seeds");
        assert!(wins > 0, "PRAX beat the best fixed block on no seed at all");
    }

    #[test]
    fn every_allocator_covers_the_whole_trace() {
        // A harness bug that stopped early would flatter whichever allocator
        // stopped earliest.
        let trace = bursty_trace(500, 2);
        let cost = CostModel::default();
        for alloc in &mut [
            Box::new(Fixed(4)) as Box<dyn Allocator>,
            Box::new(PraxAllocator::new()),
            Box::new(Oracle),
        ] {
            let o = run_with_cost(alloc.as_mut(), &trace, CAP, &cost);
            assert!(o.steps > 0);
            assert!(o.tokens_per_step >= 1.0, "{} fell below plain decoding", alloc.name());
        }
    }

    #[test]
    fn no_allocator_exceeds_the_theoretical_ceiling() {
        let trace = bursty_trace(LEN, 9);
        let cost = CostModel::default();
        for alloc in &mut [
            Box::new(Fixed(CAP)) as Box<dyn Allocator>,
            Box::new(PraxAllocator::new()),
            Box::new(Oracle),
        ] {
            let o = run_with_cost(alloc.as_mut(), &trace, CAP, &cost);
            assert!(
                o.tokens_per_step <= f64::from(CAP) + 1.0,
                "{} reported {:.3}, above the ceiling",
                alloc.name(),
                o.tokens_per_step
            );
        }
    }
}

#[cfg(test)]
mod report {
    use super::*;

    /// Prints the comparison rather than asserting it. Run with
    /// `cargo test -p praecise-runtime report -- --nocapture` to see the
    /// numbers behind the assertions in `tests`.
    #[test]
    fn print_margins() {
        const CAP: u8 = 8;
        const LEN: usize = 2000;
        println!("\n  trace      conc   fixed    prax   oracle   prax-vs-fixed");
        for (name, mk) in [
            ("bursty ", bursty_trace as fn(usize, u64) -> Trace),
            ("uniform", uniform_trace),
            ("chaotic", chaotic_trace),
        ] {
            for conc in [1u32, 8, 32] {
                let trace = mk(LEN, 42);
                let cost = CostModel::at_concurrency(conc);
                let fixed = (1..=CAP)
                    .map(|b| run_with_cost(&mut Fixed(b), &trace, CAP, &cost).net_saving)
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut pa = PraxAllocator::at_slot_cost(cost.draft_cost());
                let prax = run_with_cost(&mut pa, &trace, CAP, &cost).net_saving;
                let oracle = run_with_cost(&mut Oracle, &trace, CAP, &cost).net_saving;
                println!(
                    "  {name}  {conc:>4}  {fixed:>7.0} {prax:>7.0}  {oracle:>7.0}   {:>+7.0}",
                    prax - fixed
                );
            }
        }
        println!();
    }
}

/// Does regime memory add anything over entropy alone?
///
/// This is the experiment that decides whether [`crate::prax`]'s central claim
/// survives. PRAX conditions on *how long the current low-entropy run has
/// lasted*, on the argument that entropy is autocorrelated rather than i.i.d.
/// and that run length therefore predicts continuation. Published bounds on
/// speculative decoding assume i.i.d. acceptance, and if that assumption is
/// close enough to true then run length carries no information over entropy and
/// the whole idea is machinery for nothing.
///
/// So: measure the autocorrelation directly, and — more importantly — measure
/// whether an allocator *with* regime memory beats an otherwise identical one
/// *without* it. The second is the honest test, because a signal can be
/// statistically real and still worthless once entropy has been accounted for.
#[cfg(test)]
mod autocorrelation {
    use super::*;

    /// Lag-1 autocorrelation of a series. Zero means memoryless.
    fn lag1(xs: &[f64]) -> f64 {
        if xs.len() < 3 {
            return 0.0;
        }
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        if var <= f64::EPSILON {
            return 0.0;
        }
        let cov: f64 = xs.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum();
        cov / var
    }

    /// An allocator identical to PRAX except that it forgets everything between
    /// positions — entropy routing with no regime memory.
    ///
    /// This is the control. Comparing PRAX against a *fixed* block conflates two
    /// claims: that entropy routing helps, and that regime memory helps on top.
    /// Only this comparison isolates the second, which is the novel one.
    struct Memoryless {
        thresholds: EntropyThresholds,
        slot_cost: f64,
    }

    impl Allocator for Memoryless {
        fn block_for(&mut self, trace: &Trace, index: usize, verify_capacity: u8) -> u8 {
            // A fresh Prax per position: entropy routing applies, history cannot.
            let mut fresh = Prax::with_thresholds(self.thresholds);
            let signals = Signals {
                entropy_nats: trace.get(index).copied(),
                slot_cost: self.slot_cost,
                ..Signals::unknown(verify_capacity)
            };
            fresh.allocate(&signals).block
        }

        fn name(&self) -> &'static str {
            "memoryless"
        }
    }

    #[test]
    fn bursty_traces_are_autocorrelated_and_uniform_ones_are_not() {
        // Sanity check on the generators before drawing any conclusion from
        // them. If a "bursty" trace is not actually autocorrelated, every
        // result below is measuring nothing.
        let bursty = lag1(&bursty_trace(4000, 1));
        let uniform = lag1(&uniform_trace(4000, 1));
        println!("\n  lag-1 autocorrelation: bursty {bursty:+.3}   uniform {uniform:+.3}");

        // The uniform control must be memoryless, or the measurement itself is
        // broken and the bursty figure means nothing.
        assert!(uniform.abs() < 0.1, "uniform control is not memoryless: {uniform:.3}");

        // The bursty figure is RECORDED, not required. A pre-registered
        // threshold of 0.2 was set before measuring; the result was +0.189 on a
        // trace built specifically to be bursty, and PRAX's run-length sizing
        // was removed as a result. Asserting the threshold now would either
        // fail permanently or invite quietly lowering it — so the finding is
        // pinned as an upper bound instead, and a future change that made
        // burstiness strong would trip it and prompt a re-think.
        assert!(
            bursty < 0.5,
            "burstiness is far stronger than when run-length sizing was dropped ({bursty:.3}); \
             revisit that decision rather than assuming it still holds"
        );
    }

    #[test]
    fn report_whether_regime_memory_earns_its_place() {
        // THE DECISIVE MEASUREMENT. Prints rather than asserts, because the
        // useful outcome here is a number to act on — including a number that
        // says the claim is dead.
        const CAP: u8 = 8;
        const LEN: usize = 4000;

        println!("\n  Does regime memory beat entropy routing alone?");
        println!("  trace      conc   memoryless      prax    delta      %");
        for (name, mk) in [
            ("bursty ", bursty_trace as fn(usize, u64) -> Trace),
            ("uniform", uniform_trace),
        ] {
            for conc in [1u32, 8, 32] {
                let cost = CostModel::at_concurrency(conc);
                // Average over seeds: one trace proves nothing either way.
                let (mut mem_total, mut prax_total) = (0.0, 0.0);
                for seed in 0..8u64 {
                    let trace = mk(LEN, seed);
                    let mut memoryless = Memoryless {
                        thresholds: EntropyThresholds::default(),
                        slot_cost: cost.draft_cost(),
                    };
                    mem_total += run_with_cost(&mut memoryless, &trace, CAP, &cost).net_saving;
                    let mut prax = PraxAllocator::at_slot_cost(cost.draft_cost());
                    prax_total += run_with_cost(&mut prax, &trace, CAP, &cost).net_saving;
                }
                let (m, p) = (mem_total / 8.0, prax_total / 8.0);
                let pct = if m.abs() > f64::EPSILON { (p - m) / m.abs() * 100.0 } else { 0.0 };
                println!("  {name}  {conc:>4}   {m:>10.0} {p:>9.0} {:>8.0} {pct:>6.1}", p - m);
            }
        }
        println!();
    }

    #[test]
    fn regime_memory_does_not_hurt() {
        // The weakest defensible claim, and the one worth enforcing. If regime
        // memory cannot be shown to help, it must at least not cost anything —
        // otherwise it is strictly worse than entropy routing alone and should
        // be deleted rather than defended.
        const CAP: u8 = 8;
        const LEN: usize = 4000;
        let cost = CostModel::at_concurrency(8);

        for seed in 0..4u64 {
            let trace = bursty_trace(LEN, seed);
            let mut memoryless = Memoryless {
                thresholds: EntropyThresholds::default(),
                slot_cost: cost.draft_cost(),
            };
            let m = run_with_cost(&mut memoryless, &trace, CAP, &cost).net_saving;
            let mut prax = PraxAllocator::at_slot_cost(cost.draft_cost());
            let p = run_with_cost(&mut prax, &trace, CAP, &cost).net_saving;
            assert!(
                p >= m - m.abs() * 0.05,
                "seed {seed}: regime memory cost {:.0} against entropy routing alone",
                m - p
            );
        }
    }
}

/// Where PRAX wins, where it loses, and where it breaks.
///
/// The comparisons in `tests` assert individual claims. This module maps the
/// whole operating envelope, so the answer to "should I turn this on?" is a
/// lookup rather than a guess — and so a regression shows up as a changed
/// boundary rather than as one red assertion.
///
/// Two things are deliberately included that flatter nothing: a trace type PRAX
/// is expected to lose on, and a search for the load at which it stops paying.
/// A characterisation that only covers the favourable cases is marketing.
#[cfg(test)]
mod envelope {
    use super::*;

    const CAP: u8 = 8;
    const LEN: usize = 4000;
    const SEEDS: u64 = 8;

    /// Mean net saving over several seeds — one trace is noise.
    fn mean(mk: fn(usize, u64) -> Trace, cost: &CostModel, prax: bool) -> f64 {
        let mut total = 0.0;
        for seed in 0..SEEDS {
            let trace = mk(LEN, seed);
            total += if prax {
                let mut a = PraxAllocator::at_slot_cost(cost.draft_cost());
                run_with_cost(&mut a, &trace, CAP, cost).net_saving
            } else {
                (1..=CAP)
                    .map(|b| run_with_cost(&mut Fixed(b), &trace, CAP, cost).net_saving)
                    .fold(f64::NEG_INFINITY, f64::max)
            };
        }
        total / SEEDS as f64
    }

    /// A trace with *smoothly varying* entropy — no runs, no plateaus, just a
    /// slow drift. Included because it is the shape PRAX has least to offer on:
    /// there is structure, but not the kind a run-length brake can read.
    fn drifting_trace(len: usize, seed: u64) -> Trace {
        let phase = (seed % 8) as f64 * 0.4;
        (0..len)
            .map(|i| {
                let t = i as f64 / 90.0 + phase;
                // 0.02 .. 0.72 nats: crosses every regime boundary slowly.
                0.35f64.mul_add(t.sin(), 0.37)
            })
            .collect()
    }

    #[test]
    fn operating_envelope() {
        println!("\n  PRAX vs best fixed block, mean of {SEEDS} seeds, net units");
        println!("  trace       conc     fixed      prax      delta        %");
        for (name, mk) in [
            ("bursty  ", bursty_trace as fn(usize, u64) -> Trace),
            ("uniform ", uniform_trace),
            ("chaotic ", chaotic_trace),
            ("drifting", drifting_trace),
        ] {
            for conc in [1u32, 4, 8, 16, 32, 64] {
                let cost = CostModel::at_concurrency(conc);
                let f = mean(mk, &cost, false);
                let p = mean(mk, &cost, true);
                let pct = if f.abs() > 1.0 { (p - f) / f.abs() * 100.0 } else { 0.0 };
                println!("  {name}   {conc:>4}  {f:>8.0}  {p:>8.0}  {:>9.0}  {pct:>7.1}", p - f);
            }
        }
        println!();
    }

    #[test]
    fn prax_never_makes_a_server_slower_than_not_speculating() {
        // The property that decides whether this is safe to enable by default.
        // Plain decoding saves exactly 0; a fixed block can go well below that
        // by spending verification on drafts nobody takes. PRAX must not.
        for (name, mk) in [
            ("bursty", bursty_trace as fn(usize, u64) -> Trace),
            ("uniform", uniform_trace),
            ("chaotic", chaotic_trace),
            ("drifting", drifting_trace),
        ] {
            for conc in [1u32, 8, 32, 64] {
                let cost = CostModel::at_concurrency(conc);
                let p = mean(mk, &cost, true);
                // Allow a small negative: one drafted position is floored on
                // to keep drafter KV state live (see prax.rs), and that floor
                // costs a little where nothing is acceptable.
                assert!(
                    p > -f64::from(LEN as u32) * 0.05,
                    "{name} at concurrency {conc}: PRAX net {p:.0}, worse than not speculating"
                );
            }
        }
    }

    #[test]
    fn a_fixed_block_does_make_a_server_slower() {
        // The failure PRAX exists to avoid, demonstrated rather than asserted.
        // Without this, "PRAX never goes negative" is unimpressive — it needs
        // to be true that the alternative does.
        let cost = CostModel::at_concurrency(1);
        let worst = (1..=CAP)
            .map(|b| {
                let mut total = 0.0;
                for seed in 0..SEEDS {
                    let t = chaotic_trace(LEN, seed);
                    total += run_with_cost(&mut Fixed(b), &t, CAP, &cost).net_saving;
                }
                total / SEEDS as f64
            })
            .fold(f64::INFINITY, f64::min);
        assert!(worst < 0.0, "a badly chosen fixed block should lose; got {worst:.0}");
        println!("\n  worst fixed block on chaotic text: {worst:.0} net units");
    }

    #[test]
    fn find_the_load_where_speculation_stops_paying() {
        // Every allocator has a load beyond which drafting cannot pay. Knowing
        // where PRAX's is, and that it is no earlier than a fixed block's, is
        // what makes it deployable rather than a gamble.
        let mut prax_last = 0u32;
        let mut fixed_last = 0u32;
        for conc in 1..=128u32 {
            let cost = CostModel::at_concurrency(conc);
            if mean(bursty_trace, &cost, true) > 0.0 {
                prax_last = conc;
            }
            if mean(bursty_trace, &cost, false) > 0.0 {
                fixed_last = conc;
            }
        }
        println!(
            "\n  last concurrency still paying: prax {prax_last}, best-fixed {fixed_last}"
        );
        assert!(
            prax_last >= fixed_last,
            "PRAX gives up earlier than a fixed block ({prax_last} vs {fixed_last})"
        );
    }

    #[test]
    fn the_win_is_not_an_artifact_of_one_verify_capacity() {
        // A result that only holds at capacity 8 would be a tuning artifact.
        println!("\n  bursty text at concurrency 8, by verify capacity");
        for cap in [2u8, 4, 8, 16] {
            let cost = CostModel::at_concurrency(8);
            let mut fixed_best = f64::NEG_INFINITY;
            let mut prax_total = 0.0;
            for seed in 0..SEEDS {
                let t = bursty_trace(LEN, seed);
                let f = (1..=cap)
                    .map(|b| run_with_cost(&mut Fixed(b), &t, cap, &cost).net_saving)
                    .fold(f64::NEG_INFINITY, f64::max);
                fixed_best = fixed_best.max(f);
                let mut a = PraxAllocator::at_slot_cost(cost.draft_cost());
                prax_total += run_with_cost(&mut a, &t, cap, &cost).net_saving;
            }
            let p = prax_total / SEEDS as f64;
            println!("    capacity {cap:>2}: fixed {fixed_best:>7.0}   prax {p:>7.0}");
            assert!(p > 0.0, "PRAX should still pay at capacity {cap}");
        }
        println!();
    }
}

/// Validation against llama.cpp's own synthetic acceptance model.
///
/// The traces elsewhere in this module are mine, which means a result on them
/// could be an artifact of how I chose to model acceptance. llama.cpp ships a
/// synthetic mode built for exactly this purpose — benchmarking draft-block
/// policy without a real drafter — and adopting its model removes that
/// objection.
///
/// Its shape, from upstream source: `--spec-synth-rates P0,P1,...` gives
/// *unconditional* per-position acceptance probabilities, validated to be
/// finite, within `[0,1]`, and **monotonically non-increasing**; internally they
/// are converted to conditional probabilities as `rate[i] / rate[i-1]`, and
/// acceptance at position `i` is a Bernoulli draw against that. Position `i` is
/// only reached if every earlier position was accepted, which is why the
/// unconditional curve must be non-increasing.
///
/// That is a stricter and more realistic model than a flat per-position
/// probability: it makes the cost of an over-long block explicit, since the tail
/// of a block is reached rarely but paid for every time.
#[cfg(test)]
mod synthetic {
    use super::*;

    const CAP: u8 = 8;
    const STEPS: usize = 4000;

    /// An acceptance curve in llama.cpp's `--spec-synth-rates` form:
    /// unconditional probability that position `i` is accepted, non-increasing.
    struct SynthCurve {
        name: &'static str,
        rates: Vec<f64>,
    }

    impl SynthCurve {
        /// Geometric decay, the shape a real drafter produces: each further
        /// position needs all its predecessors to have matched.
        fn geometric(name: &'static str, first: f64, n: usize) -> Self {
            let rates = (0..n).map(|i| first.powi(i as i32 + 1)).collect();
            Self { name, rates }
        }

        /// A drafter that is strong for a few positions then falls off a cliff —
        /// the profile of a shallow draft head.
        fn cliff(name: &'static str, plateau: f64, hold: usize, n: usize) -> Self {
            let rates = (0..n)
                .map(|i| if i < hold { plateau } else { plateau * 0.15f64.powi((i - hold) as i32 + 1) })
                .collect();
            Self { name, rates }
        }

        /// Conditional probability of accepting position `i`, given all earlier
        /// positions were accepted. This is llama.cpp's `rate / rate_prev`.
        fn conditional(&self, i: usize) -> f64 {
            match (self.rates.get(i), i.checked_sub(1).and_then(|j| self.rates.get(j))) {
                (Some(&r), Some(&prev)) if prev > 0.0 => (r / prev).clamp(0.0, 1.0),
                (Some(&r), None) => r.clamp(0.0, 1.0),
                _ => 0.0,
            }
        }

        /// Mean accepted length for a given block size, in closed form: the sum
        /// of unconditional acceptance probabilities over the block.
        fn mean_accepted(&self, block: u8) -> f64 {
            self.rates.iter().take(block as usize).sum()
        }

        /// Net saving of a fixed block under a cost model. Every drafted
        /// position is verified; only accepted ones save anything.
        fn net_for_block(&self, block: u8, cost: &CostModel) -> f64 {
            let accepted = self.mean_accepted(block);
            accepted.mul_add(cost.t_a, -(f64::from(block) * cost.draft_cost()))
        }

        /// The best a fixed block can do — the honest baseline.
        fn best_fixed(&self, cost: &CostModel) -> (u8, f64) {
            (1..=CAP)
                .map(|b| (b, self.net_for_block(b, cost)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("a block size exists")
        }

        /// What PRAX achieves, given the same curve.
        ///
        /// PRAX is driven by entropy, so the curve's first-position acceptance
        /// is inverted through `a = exp(-nats)` to give it a comparable signal.
        /// That inversion is the one bridge between the two models, and it is
        /// applied identically at every load.
        fn net_for_prax(&self, cost: &CostModel) -> f64 {
            let a0 = self.rates.first().copied().unwrap_or(0.0).clamp(1e-6, 1.0);
            let nats = -a0.ln();
            let mut prax = Prax::new();
            let mut total = 0.0;
            for _ in 0..STEPS {
                let signals = Signals {
                    entropy_nats: Some(nats),
                    slot_cost: cost.draft_cost(),
                    ..Signals::unknown(CAP)
                };
                let block = prax.allocate(&signals).block;
                if block == 0 {
                    prax.record_outcome(0, 0);
                    continue;
                }
                let accepted = self.mean_accepted(block);
                total += accepted.mul_add(cost.t_a, -(f64::from(block) * cost.draft_cost()));
                // Report the accepted PREFIX length, not the expected total.
                // The profile learns per-position reach, and a rounded mean
                // would tell it the block succeeded uniformly when in fact it
                // succeeded up to a point and then stopped -- which is exactly
                // the cliff it exists to detect.
                let mut prefix = 0u8;
                let mut cum = 1.0f64;
                for i in 0..block as usize {
                    cum *= self.conditional(i);
                    if cum < 0.5 {
                        break;
                    }
                    prefix += 1;
                }
                prax.record_outcome(prefix, block);
            }
            total / STEPS as f64
        }
    }

    fn curves() -> Vec<SynthCurve> {
        vec![
            SynthCurve::geometric("strong  ", 0.90, CAP as usize),
            SynthCurve::geometric("moderate", 0.75, CAP as usize),
            SynthCurve::geometric("weak    ", 0.50, CAP as usize),
            SynthCurve::cliff("cliff-3 ", 0.85, 3, CAP as usize),
            SynthCurve::cliff("cliff-1 ", 0.80, 1, CAP as usize),
        ]
    }

    #[test]
    fn curves_match_upstream_validation_rules() {
        // llama.cpp requires rates finite, in [0,1], and monotonically
        // non-increasing. A curve that violated those would be testing against
        // a model upstream would reject.
        for c in curves() {
            assert_eq!(c.rates.len(), CAP as usize, "{}: wrong length", c.name);
            for w in c.rates.windows(2) {
                assert!(w[1] <= w[0] + 1e-12, "{}: rates must not increase", c.name);
            }
            for &r in &c.rates {
                assert!(r.is_finite() && (0.0..=1.0).contains(&r), "{}: rate out of range", c.name);
            }
            // The conditional form must also be a probability.
            for i in 0..c.rates.len() {
                let p = c.conditional(i);
                assert!((0.0..=1.0).contains(&p), "{}: conditional out of range at {i}", c.name);
            }
        }
    }

    #[test]
    fn prax_against_upstream_acceptance_curves() {
        println!("\n  PRAX vs best fixed block, on llama.cpp --spec-synth-rates curves");
        println!("  curve      conc   bestN    fixed     prax    delta       %");
        for c in curves() {
            for conc in [1u32, 8, 32] {
                let cost = CostModel::at_concurrency(conc);
                let (bn, f) = c.best_fixed(&cost);
                let p = c.net_for_prax(&cost);
                let pct = if f.abs() > 1e-6 { (p - f) / f.abs() * 100.0 } else { 0.0 };
                println!(
                    "  {}  {conc:>4}   {bn:>4}  {f:>7.3}  {p:>7.3}  {:>7.3}  {pct:>6.1}",
                    c.name,
                    p - f
                );
            }
        }
        println!();
    }

    #[test]
    fn prax_never_loses_more_than_it_could_gain() {
        // The deployment question: is turning this on safe? PRAX may trail the
        // best fixed block on some curves — the baseline is chosen with perfect
        // hindsight per curve, which no deployment gets — but it must never be
        // catastrophically worse, and it must never be worse than not
        // speculating at all.
        for c in curves() {
            for conc in [1u32, 4, 8, 16, 32] {
                let cost = CostModel::at_concurrency(conc);
                let (_, f) = c.best_fixed(&cost);
                let p = c.net_for_prax(&cost);
                assert!(
                    p >= -0.05,
                    "{} at {conc}: PRAX net {p:.3}, worse than not speculating",
                    c.name
                );
                // NOT asserted, and deliberately so. On cliff-shaped curves
                // PRAX trails the best fixed block -- measured worst case
                // `cliff-1` at concurrency 16: 0.020 against 0.350. The cause is
                // structural, not a tuning miss: PRAX prices a block as `a^k`,
                // geometric decay, while a cliff drafter holds flat and then
                // collapses. Entropy cannot see that, because the cliff belongs
                // to the drafter rather than to the text.
                //
                // Recorded here instead of asserted, because an assertion would
                // either fail permanently or have to be loosened until it
                // proved nothing. What IS asserted above is the property that
                // decides deployability: PRAX is never worse than not
                // speculating.
                if p < f {
                    println!("    trails best-fixed: {} at {conc}: {p:.3} vs {f:.3}", c.name);
                }
            }
        }
    }

    #[test]
    fn a_fixed_block_chosen_for_one_curve_fails_on_another() {
        // Why per-position allocation is worth anything at all. A deployment
        // picks ONE block size; acceptance curves vary by model, prompt and
        // load. Tuning for a strong drafter and meeting a weak one is the
        // common case, and it is what an adaptive policy is insurance against.
        let strong = SynthCurve::geometric("strong", 0.90, CAP as usize);
        let weak = SynthCurve::geometric("weak", 0.50, CAP as usize);
        let cost = CostModel::at_concurrency(8);

        let (tuned_for_strong, _) = strong.best_fixed(&cost);
        let on_weak = weak.net_for_block(tuned_for_strong, &cost);
        let (_, weak_best) = weak.best_fixed(&cost);

        println!(
            "\n  block {tuned_for_strong} tuned on a strong drafter scores {on_weak:.3} on a weak one; \
             best there is {weak_best:.3}"
        );
        assert!(on_weak < weak_best, "a mistuned fixed block should underperform");
    }
}

/// The comparison that decides whether this is worth turning on.
///
/// Everything else in this module compares against the *best* fixed block for
/// each curve — chosen with perfect hindsight, per workload. No deployment gets
/// that. A real server picks one number, once, and then meets whatever arrives:
/// a strong drafter on one request, a weak one on the next, idle at 3am and
/// saturated at noon.
///
/// So the honest baseline is a **single fixed block held constant across
/// everything**, which is what every shipped engine actually does. That is the
/// bar to beat, and it is the only comparison that answers "should I enable
/// this".
#[cfg(test)]
mod deployed {
    use super::*;
    use crate::prax::{Prax, Signals};

    const CAP: u8 = 8;
    const STEPS: usize = 3000;

    /// The mix a server actually sees: drafter quality varies by model and
    /// prompt, load varies by hour. Each entry is (name, first-position
    /// acceptance, concurrency).
    fn workload_mix() -> Vec<(&'static str, f64, u32)> {
        vec![
            ("strong drafter, idle    ", 0.90, 1),
            ("strong drafter, busy    ", 0.90, 16),
            ("moderate drafter, idle  ", 0.75, 1),
            ("moderate drafter, busy  ", 0.75, 16),
            ("weak drafter, idle      ", 0.50, 1),
            ("weak drafter, busy      ", 0.50, 16),
            ("weak drafter, saturated ", 0.50, 32),
            ("poor drafter, busy      ", 0.30, 16),
        ]
    }

    /// Net saving for a fixed block on a geometric acceptance curve.
    fn fixed_net(block: u8, a0: f64, cost: &CostModel) -> f64 {
        let accepted: f64 = (0..block).map(|i| a0.powi(i32::from(i) + 1)).sum();
        accepted.mul_add(cost.t_a, -(f64::from(block) * cost.draft_cost()))
    }

    /// Net saving for PRAX on the same curve.
    fn prax_net(a0: f64, cost: &CostModel) -> f64 {
        let nats = -a0.clamp(1e-6, 1.0).ln();
        let mut prax = Prax::new();
        let mut total = 0.0;
        for _ in 0..STEPS {
            let signals = Signals {
                entropy_nats: Some(nats),
                slot_cost: cost.draft_cost(),
                ..Signals::unknown(CAP)
            };
            let block = prax.allocate(&signals).block;
            if block == 0 {
                prax.record_outcome(0, 0);
                continue;
            }
            let accepted: f64 = (0..block).map(|i| a0.powi(i32::from(i) + 1)).sum();
            total += accepted.mul_add(cost.t_a, -(f64::from(block) * cost.draft_cost()));
            // Report the accepted prefix, which is what a real verify returns.
            let mut prefix = 0u8;
            let mut cum = 1.0;
            for _ in 0..block {
                cum *= a0;
                if cum < 0.5 {
                    break;
                }
                prefix += 1;
            }
            prax.record_outcome(prefix, block);
        }
        total / STEPS as f64
    }

    /// The single fixed block that does best *averaged over the whole mix* —
    /// the choice a careful operator would make with full knowledge of their
    /// own traffic. Still one number, as a deployment must be.
    fn best_deployed_fixed() -> (u8, f64) {
        (1..=CAP)
            .map(|b| {
                let total: f64 = workload_mix()
                    .iter()
                    .map(|&(_, a0, conc)| fixed_net(b, a0, &CostModel::at_concurrency(conc)))
                    .sum();
                (b, total / workload_mix().len() as f64)
            })
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .expect("a block size exists")
    }

    #[test]
    fn against_a_single_deployed_block_size() {
        let (chosen, fixed_mean) = best_deployed_fixed();
        println!(
            "\n  Against ONE fixed block ({chosen}) held across a varied workload —\n  \
             the choice a real deployment makes, not per-curve hindsight.\n"
        );
        println!("  workload                   fixed     prax     delta       %");

        let mut prax_total = 0.0;
        for (name, a0, conc) in workload_mix() {
            let cost = CostModel::at_concurrency(conc);
            let f = fixed_net(chosen, a0, &cost);
            let p = prax_net(a0, &cost);
            prax_total += p;
            let pct = if f.abs() > 1e-6 { (p - f) / f.abs() * 100.0 } else { 0.0 };
            println!("  {name}  {f:>7.3}  {p:>7.3}  {:>8.3}  {pct:>6.1}", p - f);
        }
        let prax_mean = prax_total / workload_mix().len() as f64;
        let pct = (prax_mean - fixed_mean) / fixed_mean.abs() * 100.0;
        println!(
            "  {:26} {fixed_mean:>7.3}  {prax_mean:>7.3}  {:>8.3}  {pct:>6.1}\n",
            "MEAN", prax_mean - fixed_mean
        );
    }

    #[test]
    fn prax_beats_a_single_deployed_block_size() {
        // The bar that matters: averaged over a varied workload, does adapting
        // beat committing to one number? If not, this should not ship.
        let (_, fixed_mean) = best_deployed_fixed();
        let prax_mean: f64 = workload_mix()
            .iter()
            .map(|&(_, a0, conc)| prax_net(a0, &CostModel::at_concurrency(conc)))
            .sum::<f64>()
            / workload_mix().len() as f64;

        assert!(
            prax_mean > fixed_mean,
            "PRAX {prax_mean:.3} vs one deployed block {fixed_mean:.3} — not worth enabling"
        );
    }

    #[test]
    fn no_single_block_size_is_good_everywhere() {
        // Why adapting can win at all. If one block were near-optimal on every
        // workload, there would be nothing to adapt to and PRAX would be pure
        // overhead — so this establishes that the premise holds before the
        // comparison above means anything.
        let mut best_blocks = std::collections::BTreeSet::new();
        for (_, a0, conc) in workload_mix() {
            let cost = CostModel::at_concurrency(conc);
            let best = (1..=CAP)
                .max_by(|&x, &y| fixed_net(x, a0, &cost).total_cmp(&fixed_net(y, a0, &cost)))
                .expect("a block size exists");
            best_blocks.insert(best);
        }
        println!("\n  optimal block size varies across the mix: {best_blocks:?}");
        assert!(
            best_blocks.len() > 1,
            "one block is optimal everywhere; adapting cannot help"
        );
    }
}
