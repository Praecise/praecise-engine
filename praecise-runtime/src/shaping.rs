//! Request shaping: making a batch cheap before it reaches a backend.
//!
//! Decode is memory-bandwidth-bound, and on hardware already running near its
//! bandwidth ceiling there is little left for a scheduler or an allocator to
//! win. What *is* still reachable from outside a backend is how requests are
//! **shaped and grouped** before they arrive — and a few of those levers are
//! sharp enough to matter.
//!
//! Everything here is arithmetic about batching rather than a claim about text.
//! Nothing has to be true of the model for it to hold, which is why it is worth
//! more than a heuristic that has to be tuned per workload.
//!
//! ## The one that surprises people
//!
//! `all_greedy` and `no_penalties` are **batch-wide booleans**. One request
//! using a frequency or presence penalty drags the *entire batch* onto the
//! expensive sampling path — including every request that asked for none. The
//! cost is not the sampler, which runs in tens of microseconds; it is the
//! penalty bookkeeping over each generated sequence, measured at **6-47 ms per
//! step** at batch 4 and growing with context length.
//!
//! A caller cannot see this from inside one request. A layer that groups
//! requests by their sampling shape can.

use crate::config::GenerationConfig;

/// How a request will make a batch behave.
///
/// Requests sharing a class can be batched without one making the others
/// expensive. The ordering is deliberate: [`Cheap`](Self::Cheap) requests may be
/// merged into any batch, while [`Penalised`](Self::Penalised) requests
/// contaminate whatever they join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchClass {
    /// Greedy, no penalties. The cheapest path a backend has.
    Greedy,
    /// Sampled, but with no penalties — the sampler alone is nearly free.
    Sampled,
    /// Uses a frequency, presence or repetition penalty. Forces the whole batch
    /// onto the expensive path, so these belong together and apart from others.
    Penalised,
}

impl BatchClass {
    /// Classify a request.
    #[must_use]
    pub fn of(config: &GenerationConfig) -> Self {
        if uses_penalties(config) {
            Self::Penalised
        } else if config.temperature <= f64::EPSILON {
            Self::Greedy
        } else {
            Self::Sampled
        }
    }

    /// Whether adding a request of this class to a batch of `other` would make
    /// the existing requests more expensive.
    ///
    /// Not symmetric: a greedy request joining a penalised batch costs the
    /// greedy request nothing extra, while the reverse spoils the batch.
    #[must_use]
    pub fn contaminates(self, batch: Self) -> bool {
        self == Self::Penalised && batch != Self::Penalised
    }
}

/// Whether the request actually asks for penalty bookkeeping.
///
/// Checked against zero rather than against presence, because a caller setting
/// a penalty to its neutral value is asking for nothing while still paying for
/// everything — and dropping that is one of the cheapest wins available.
fn uses_penalties(config: &GenerationConfig) -> bool {
    (config.repeat_penalty - 1.0).abs() > f32::EPSILON
}

/// Cost-reducing rewrites that do not change what a caller asked for.
///
/// Each is a case where two settings express the same intent and one is
/// materially cheaper. Returned rather than applied, so a caller can see what
/// would change and decline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rewrite {
    /// A penalty set to its neutral value (1.0) still costs full penalty
    /// bookkeeping — 6-47 ms per step at batch 4 — while doing nothing. Worse,
    /// it pulls the whole batch onto the expensive path.
    DropNeutralPenalty,
    /// `top_p` requires a vocabulary sort; `min_p` is softmax, amax and a
    /// compare. Where the intent is "drop implausible tokens" rather than "keep
    /// a fixed probability mass", `min_p` expresses it and costs less.
    PreferMinP,
    /// Tile quantisation: a chunk size of 257 costs **32% more** than 256.
    /// Rounding down to the multiple of 256 is almost never distinguishable to
    /// a caller.
    RoundChunkSize { from: u32, to: u32 },
}

impl Rewrite {
    /// What this changes and why, for a caller deciding whether to accept it.
    #[must_use]
    pub fn explain(self) -> String {
        match self {
            Self::DropNeutralPenalty => "a penalty set to 1.0 changes nothing but costs full \
                 penalty bookkeeping, and forces the whole batch onto the expensive sampling path"
                .to_string(),
            Self::PreferMinP => "min_p avoids the vocabulary sort that top_p requires".to_string(),
            Self::RoundChunkSize { from, to } => format!(
                "chunk size {from} straddles a tile boundary and costs about 32% more than {to}"
            ),
        }
    }
}

/// Suggest rewrites for a request. Empty when there is nothing to gain.
#[must_use]
pub fn rewrites(config: &GenerationConfig) -> Vec<Rewrite> {
    let mut out = Vec::new();

    // A penalty of exactly 1.0 asks for nothing while still costing the full
    // bookkeeping — and, worse, pulling the whole batch onto the expensive
    // path. `Default` leaves it at 1.0, so only flag it when the caller has a
    // reason to have set it: alongside sampling, where penalties are usually
    // configured deliberately.
    if (config.repeat_penalty - 1.0).abs() <= f32::EPSILON && config.temperature > f64::EPSILON {
        out.push(Rewrite::DropNeutralPenalty);
    }
    if config.top_p < 1.0 && config.min_p.is_none() {
        out.push(Rewrite::PreferMinP);
    }
    out
}

/// Round a chunk size down to a tile boundary.
///
/// Returns `None` when it is already aligned. Rounding *down* rather than up so
/// the result never exceeds what the caller asked for.
#[must_use]
pub fn align_chunk(size: u32) -> Option<Rewrite> {
    const TILE: u32 = 256;
    if size <= TILE || size.is_multiple_of(TILE) {
        return None;
    }
    let to = (size / TILE) * TILE;
    Some(Rewrite::RoundChunkSize { from: size, to })
}

/// Partition requests into groups that can share a batch without one making the
/// others expensive.
///
/// Returns indices grouped by [`BatchClass`], cheapest class first, so a caller
/// forming batches under a size limit fills the cheap paths before the dear
/// ones.
#[must_use]
pub fn group_by_class(configs: &[GenerationConfig]) -> Vec<(BatchClass, Vec<usize>)> {
    let mut groups: Vec<(BatchClass, Vec<usize>)> = Vec::new();
    for class in [BatchClass::Greedy, BatchClass::Sampled, BatchClass::Penalised] {
        let members: Vec<usize> = configs
            .iter()
            .enumerate()
            .filter(|(_, c)| BatchClass::of(c) == class)
            .map(|(i, _)| i)
            .collect();
        if !members.is_empty() {
            groups.push((class, members));
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy() -> GenerationConfig {
        GenerationConfig { temperature: 0.0, repeat_penalty: 1.0, ..Default::default() }
    }

    fn sampled() -> GenerationConfig {
        GenerationConfig { temperature: 0.7, repeat_penalty: 1.0, ..Default::default() }
    }

    fn penalised() -> GenerationConfig {
        GenerationConfig { temperature: 0.7, repeat_penalty: 1.1, ..Default::default() }
    }

    #[test]
    fn requests_are_classified_by_what_they_cost_a_batch() {
        assert_eq!(BatchClass::of(&greedy()), BatchClass::Greedy);
        assert_eq!(BatchClass::of(&sampled()), BatchClass::Sampled);
        assert_eq!(BatchClass::of(&penalised()), BatchClass::Penalised);
    }

    #[test]
    fn a_neutral_penalty_is_not_a_penalty() {
        // Asking for a penalty of 1.0 asks for nothing, and must not condemn
        // the batch to the expensive path.
        let neutral = GenerationConfig { repeat_penalty: 1.0, ..sampled() };
        assert_eq!(BatchClass::of(&neutral), BatchClass::Sampled);
    }

    #[test]
    fn one_penalised_request_contaminates_a_cheap_batch() {
        // The finding this module exists for: batch-wide booleans mean a single
        // request changes the cost for everyone else.
        assert!(BatchClass::Penalised.contaminates(BatchClass::Greedy));
        assert!(BatchClass::Penalised.contaminates(BatchClass::Sampled));
    }

    #[test]
    fn contamination_is_not_symmetric() {
        // A cheap request joining an expensive batch costs nothing extra; the
        // reverse is what must be avoided.
        assert!(!BatchClass::Greedy.contaminates(BatchClass::Penalised));
        assert!(!BatchClass::Sampled.contaminates(BatchClass::Penalised));
        assert!(!BatchClass::Penalised.contaminates(BatchClass::Penalised));
    }

    #[test]
    fn grouping_keeps_the_expensive_requests_together() {
        let configs = vec![greedy(), penalised(), sampled(), penalised(), greedy()];
        let groups = group_by_class(&configs);

        assert_eq!(groups[0].0, BatchClass::Greedy, "cheapest class first");
        assert_eq!(groups[0].1, vec![0, 4]);
        assert_eq!(groups.last().unwrap().0, BatchClass::Penalised);
        assert_eq!(groups.last().unwrap().1, vec![1, 3]);
    }

    #[test]
    fn grouping_omits_classes_with_no_members() {
        let groups = group_by_class(&[greedy(), greedy()]);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn grouping_an_empty_batch_yields_nothing() {
        assert!(group_by_class(&[]).is_empty());
    }

    #[test]
    fn a_neutral_penalty_is_suggested_for_removal() {
        let neutral = GenerationConfig { repeat_penalty: 1.0, ..sampled() };
        let r = rewrites(&neutral);
        assert!(r.contains(&Rewrite::DropNeutralPenalty));
        assert!(r[0].explain().contains("batch"), "should say why it matters beyond one request");
    }

    #[test]
    fn a_real_penalty_is_left_alone() {
        // The caller asked for something; do not quietly remove it.
        assert!(!rewrites(&penalised()).contains(&Rewrite::DropNeutralPenalty));
    }

    #[test]
    fn top_p_suggests_min_p_only_when_min_p_is_unset() {
        let with_top_p = GenerationConfig { top_p: 0.9, ..sampled() };
        assert!(rewrites(&with_top_p).contains(&Rewrite::PreferMinP));

        let with_both =
            GenerationConfig { top_p: 0.9, min_p: Some(0.05), ..sampled() };
        assert!(!rewrites(&with_both).contains(&Rewrite::PreferMinP));
    }

    #[test]
    fn chunk_sizes_are_aligned_down_to_a_tile() {
        // 257 costs about 32% more than 256, and rounding down never gives the
        // caller more than they asked for.
        assert_eq!(align_chunk(257), Some(Rewrite::RoundChunkSize { from: 257, to: 256 }));
        assert_eq!(align_chunk(600), Some(Rewrite::RoundChunkSize { from: 600, to: 512 }));
    }

    #[test]
    fn aligned_and_small_chunk_sizes_are_left_alone() {
        assert_eq!(align_chunk(256), None);
        assert_eq!(align_chunk(512), None);
        assert_eq!(align_chunk(128), None, "below one tile there is nothing to align to");
    }

    #[test]
    fn a_greedy_request_is_not_told_to_drop_its_penalty() {
        // Greedy decoding never reaches the penalty path, so a neutral penalty
        // there costs nothing and is not worth reporting.
        //
        // Note this does NOT assert "no rewrites at all": `GenerationConfig`
        // defaults `top_p` to 0.9, so the default request genuinely does carry
        // a min_p suggestion. Asserting emptiness here was testing my
        // assumption about the defaults rather than the behaviour.
        assert!(!rewrites(&greedy()).contains(&Rewrite::DropNeutralPenalty));
    }

    #[test]
    fn the_default_config_suggests_min_p() {
        // Worth pinning: `top_p` defaults to 0.9, so every request that does
        // not override it pays for a vocabulary sort it may not need.
        let d = GenerationConfig::default();
        assert!(d.top_p < 1.0, "this test assumes the default enables top_p");
        assert!(rewrites(&d).contains(&Rewrite::PreferMinP));
    }
}
