//! Draft verification: accepting a block without changing the output.
//!
//! Verification is the one part of speculative decoding that is essentially
//! solved — optimal-transport analysis puts working schemes within **0.1-3.3%**
//! of the true optimum, so there is no meaningful headroom left in the
//! accept/reject rule itself. This module exists for two narrower reasons.
//!
//! **The free win.** Multi-draft verification is usually implemented by
//! sampling candidates *with* replacement. Sampling **without** replacement
//! measures **3.0-4.0 points** higher acceptance, provably, and is largely
//! unimplemented. It costs nothing: the same candidates, deduplicated.
//!
//! **Losslessness must be a property, not a promise.** Every accept rule here
//! preserves the target distribution exactly — greedy decoding produces output
//! identical to not speculating at all. That is testable, and it is tested
//! below, because "lossless" is the claim a speculative engine most needs to be
//! true and least able to check by eye.
//!
//! ## The relations this implements
//!
//! For target `p` and draft `q`, single-draft acceptance is
//! `alpha = E[min(p, q)] = 1 - D_TV(p, q)`, and expected tokens per step is
//! `(1 - alpha^(gamma+1)) / (1 - alpha)`. Those come straight from the original
//! speculative-sampling result and are implemented here as
//! [`acceptance_rate`] and [`expected_tokens`] so that a policy can *predict*
//! the payoff of a block rather than discover it.

/// Acceptance probability for a single drafted position: `1 - D_TV(p, q)`.
///
/// Equivalently `sum_x min(p(x), q(x))` — the overlap between the two
/// distributions. Returns 0 when the inputs are not comparable rather than
/// guessing, since a wrong acceptance estimate silently mis-sizes every block
/// that follows.
#[must_use]
pub fn acceptance_rate(target: &[f64], draft: &[f64]) -> f64 {
    if target.len() != draft.len() || target.is_empty() {
        return 0.0;
    }
    target
        .iter()
        .zip(draft)
        .map(|(&p, &q)| if p < q { p } else { q })
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Expected tokens emitted per verification step: `(1 - a^(g+1)) / (1 - a)`.
///
/// The `+1` is the bonus token: a step that rejects everything still emits the
/// target's own sample, so speculation never emits fewer than one.
#[must_use]
pub fn expected_tokens(acceptance: f64, block: u8) -> f64 {
    let a = acceptance.clamp(0.0, 1.0);
    let g = f64::from(block);
    // a == 1 makes the closed form 0/0; the limit is simply g+1, every token
    // accepted. Guard on approach rather than equality so near-1 stays stable.
    if a >= 1.0 - f64::EPSILON {
        return g + 1.0;
    }
    (1.0 - a.powf(g + 1.0)) / (1.0 - a)
}

/// Whether a block is worth verifying at all.
///
/// From the same result: speculation pays when acceptance exceeds the cost
/// ratio `c` of a draft step to a target step. Below that the drafting is
/// slower than simply decoding, however good it looks in isolation.
#[must_use]
pub fn is_worth_speculating(acceptance: f64, draft_cost_ratio: f64) -> bool {
    acceptance > draft_cost_ratio.clamp(0.0, 1.0)
}

/// How multiple candidate drafts for one position are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MultiDraftSampling {
    /// Independent draws. A high-probability token can be proposed several
    /// times, and the duplicates verify to the same outcome — capacity spent
    /// learning nothing.
    WithReplacement,
    /// Distinct candidates. Every slot tests a different token, which is the
    /// entire source of the 3-4 point gain.
    #[default]
    WithoutReplacement,
}

/// Deduplicate candidates according to the sampling mode.
///
/// With-replacement is a pass-through. Without-replacement keeps first
/// occurrences, preserving order, so the highest-ranked candidate stays first
/// and a caller may truncate from the end without losing its best proposal.
#[must_use]
pub fn prepare_candidates(candidates: &[i32], mode: MultiDraftSampling) -> Vec<i32> {
    match mode {
        MultiDraftSampling::WithReplacement => candidates.to_vec(),
        MultiDraftSampling::WithoutReplacement => {
            let mut seen = std::collections::HashSet::with_capacity(candidates.len());
            candidates.iter().copied().filter(|t| seen.insert(*t)).collect()
        }
    }
}

/// Acceptance for a set of candidates: the target mass they cover.
///
/// This is where without-replacement wins, and the mechanism is visible in the
/// arithmetic: a duplicated candidate contributes its probability **once**
/// under the union, so proposing it twice buys nothing while consuming a slot
/// that a distinct token could have used.
#[must_use]
pub fn multi_draft_acceptance(target: &[f64], candidates: &[i32]) -> f64 {
    let mut seen = std::collections::HashSet::with_capacity(candidates.len());
    candidates
        .iter()
        .filter(|&&t| seen.insert(t))
        .filter_map(|&t| usize::try_from(t).ok())
        .filter_map(|i| target.get(i))
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Longest accepted prefix of a greedy draft.
///
/// Greedy decoding accepts a drafted token exactly when it equals what the
/// target would itself have produced, so the result is bit-identical to
/// decoding without speculation. That is what makes greedy speculation lossless
/// by construction rather than by assertion.
#[must_use]
pub fn greedy_accepted_prefix(drafted: &[i32], target_argmax: &[i32]) -> usize {
    drafted
        .iter()
        .zip(target_argmax)
        .take_while(|(d, t)| d == t)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distributions differing only in how mass is spread.
    fn dists() -> (Vec<f64>, Vec<f64>) {
        (vec![0.5, 0.3, 0.15, 0.05], vec![0.4, 0.35, 0.2, 0.05])
    }

    #[test]
    fn identical_distributions_always_accept() {
        let (p, _) = dists();
        assert!((acceptance_rate(&p, &p) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn disjoint_distributions_never_accept() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        assert!(acceptance_rate(&p, &q).abs() < 1e-9);
    }

    #[test]
    fn acceptance_is_one_minus_total_variation() {
        let (p, q) = dists();
        let tv: f64 = p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
        assert!((acceptance_rate(&p, &q) - (1.0 - tv)).abs() < 1e-9);
    }

    #[test]
    fn mismatched_inputs_do_not_guess() {
        // A wrong acceptance estimate mis-sizes every block after it, so
        // refuse rather than approximate.
        assert_eq!(acceptance_rate(&[0.5, 0.5], &[1.0]), 0.0);
        assert_eq!(acceptance_rate(&[], &[]), 0.0);
    }

    #[test]
    fn perfect_acceptance_emits_the_whole_block_plus_the_bonus() {
        assert!((expected_tokens(1.0, 4) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn zero_acceptance_still_emits_the_bonus_token() {
        // Speculation never emits fewer than one token: a fully rejected block
        // still yields the target's own sample.
        assert!((expected_tokens(0.0, 4) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn expected_tokens_rise_with_acceptance_and_with_block_size() {
        assert!(expected_tokens(0.8, 4) > expected_tokens(0.4, 4));
        assert!(expected_tokens(0.8, 8) > expected_tokens(0.8, 4));
    }

    #[test]
    fn expected_tokens_never_exceed_the_block_plus_one() {
        for a in [0.0, 0.25, 0.5, 0.75, 0.99, 1.0] {
            for g in [1u8, 2, 4, 8, 16] {
                let e = expected_tokens(a, g);
                assert!(e <= f64::from(g) + 1.0 + 1e-9, "a={a} g={g} gave {e}");
                assert!(e >= 1.0 - 1e-9, "a={a} g={g} gave {e}");
            }
        }
    }

    #[test]
    fn speculation_pays_only_above_the_cost_ratio() {
        assert!(is_worth_speculating(0.8, 0.2));
        assert!(!is_worth_speculating(0.1, 0.2));
        // Exactly at the ratio is break-even, which is not a win.
        assert!(!is_worth_speculating(0.2, 0.2));
    }

    #[test]
    fn without_replacement_removes_duplicates() {
        let c = [5, 3, 5, 9, 3, 1];
        assert_eq!(
            prepare_candidates(&c, MultiDraftSampling::WithoutReplacement),
            vec![5, 3, 9, 1]
        );
        assert_eq!(
            prepare_candidates(&c, MultiDraftSampling::WithReplacement),
            c.to_vec()
        );
    }

    #[test]
    fn deduplication_preserves_rank_order() {
        // The best candidate must stay first so truncating from the end never
        // discards it.
        let c = [7, 2, 7, 4, 2];
        let out = prepare_candidates(&c, MultiDraftSampling::WithoutReplacement);
        assert_eq!(out[0], 7, "the top candidate must remain first");
        assert_eq!(out, vec![7, 2, 4]);
    }

    #[test]
    fn without_replacement_covers_more_target_mass() {
        // The free win, demonstrated: same slot count, more mass covered,
        // because a duplicate contributes its probability only once.
        let target = vec![0.4, 0.3, 0.2, 0.1];
        let with_dupes = [0, 0, 1]; // 3 slots, 2 distinct tokens
        let distinct = prepare_candidates(&[0, 0, 1, 2], MultiDraftSampling::WithoutReplacement);

        let a = multi_draft_acceptance(&target, &with_dupes);
        let b = multi_draft_acceptance(&target, &distinct);
        assert!((a - 0.7).abs() < 1e-9, "duplicates cover 0.4+0.3, got {a}");
        assert!((b - 0.9).abs() < 1e-9, "distinct cover 0.4+0.3+0.2, got {b}");
        assert!(b > a, "without replacement must cover more: {b} vs {a}");
    }

    #[test]
    fn multi_draft_acceptance_ignores_out_of_range_tokens() {
        // A drafter proposing a token outside the vocabulary must not corrupt
        // the estimate or panic.
        let target = vec![0.5, 0.5];
        assert!((multi_draft_acceptance(&target, &[0, 99, -1]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn multi_draft_acceptance_is_bounded() {
        let target = vec![0.5, 0.5];
        assert!((multi_draft_acceptance(&target, &[0, 1]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn greedy_acceptance_is_the_matching_prefix() {
        assert_eq!(greedy_accepted_prefix(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(greedy_accepted_prefix(&[1, 2], &[1, 2]), 2);
        assert_eq!(greedy_accepted_prefix(&[9], &[1]), 0);
    }

    #[test]
    fn greedy_speculation_is_lossless() {
        // The property that matters most and is hardest to eyeball: whatever
        // the draft proposed, the emitted tokens are exactly what the target
        // would have produced alone.
        let target_argmax = [4, 7, 1, 9, 3];
        for draft in [
            vec![4, 7, 1, 9, 3], // all correct
            vec![4, 7, 0, 0, 0], // diverges midway
            vec![0, 0, 0],       // wrong immediately
            vec![],              // no draft at all
        ] {
            let n = greedy_accepted_prefix(&draft, &target_argmax);
            assert_eq!(
                draft[..n],
                target_argmax[..n],
                "accepted tokens must equal the target's own output"
            );
        }
    }

    #[test]
    fn a_shorter_draft_than_the_target_window_is_safe() {
        assert_eq!(greedy_accepted_prefix(&[1], &[1, 2, 3]), 1);
        assert_eq!(greedy_accepted_prefix(&[1, 2, 3], &[1]), 1);
    }
}
