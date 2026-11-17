//! TOPLOC-style inference commitments.
//!
//! During generation the provider records, for every output token, the
//! top-k raw logits (token id + value) that produced the sample. The
//! canonical serialization of those records is the commitment blob; its
//! SHA-256 is the commitment hash that rides on inference results and
//! receipts.
//!
//! A verifier holding the prompt, the output tokens, and the same model
//! weights re-executes the whole sequence as a single prefill — roughly
//! two orders of magnitude cheaper than the original decode — reads the
//! logits at each output position, and fuzzy-compares them against the
//! committed records. Exact float equality is not required: kernels
//! differ across GPUs, drivers, and batch shapes, so verification
//! tolerates bounded index churn and bounded logit drift per step, and
//! requires a minimum fraction of steps to pass overall, short of a small
//! fixed allowance of failing steps.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Domain tag prefixed to the canonical commitment encoding.
const COMMITMENT_DOMAIN_TAG: &[u8] = b"toploc/inference-commitment";

/// Default number of logits committed per generated token.
pub const DEFAULT_COMMITMENT_K: u8 = 16;

/// Maximum accepted `commitment_k` — bounds blob size per token.
pub const MAX_COMMITMENT_K: u8 = 64;

/// Minimum accepted `commitment_k`. With one or two logits per step the
/// overlap test says almost nothing (a single committed id is either present
/// or not), so a commitment below this is refused outright and producers
/// raise a smaller request to it.
pub const MIN_COMMITMENT_K: u8 = 4;

/// The `k` a producer commits with for a requested `k`.
pub fn commitment_k_for(requested: u8) -> u8 {
    requested.clamp(MIN_COMMITMENT_K, MAX_COMMITMENT_K)
}

/// Minimum fraction of committed top-k token ids that must reappear in
/// the recomputed top-k for a step to pass.
pub const MIN_INDEX_OVERLAP: f32 = 0.75;

/// Maximum mean absolute logit difference over the shared token ids for
/// a step to pass. An honest run on the CPU, decoded one token at a time and
/// verified in one prefill, differs by up to 0.505 in a single top-k logit
/// (Qwen3 0.6B, 24 steps; 0.31 for Qwen3.5 0.8B), so the bound sits above
/// that.
pub const MAX_MEAN_LOGIT_DELTA: f32 = 0.6;

/// Minimum fraction of steps that must pass for the commitment as a
/// whole to verify, counting [`ALLOWED_FAILING_STEPS`] as passing.
pub const MIN_PASSING_STEP_FRACTION: f32 = 0.9;

/// Failing steps a commitment may have beyond what
/// [`MIN_PASSING_STEP_FRACTION`] allows. Honest CPU runs fail a few steps
/// where two candidates nearly tie and the rounding of decode and prefill
/// swaps them: 4 of 33 (Qwen3 0.6B), 1 of 24, 1 of 9 (Qwen3.5 0.8B). Under
/// the fraction alone a commitment of fewer than ten steps tolerates none.
/// More steps must still pass than fail.
pub const ALLOWED_FAILING_STEPS: usize = 2;

/// The thresholds a commitment is compared under. A verifier whose numbers
/// differ from the provider's by more than rounding (another quantisation,
/// another precision) passes its own; [`Tolerance::default`] is the three
/// constants above.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerance {
    /// See [`MIN_INDEX_OVERLAP`].
    pub min_index_overlap: f32,
    /// See [`MAX_MEAN_LOGIT_DELTA`].
    pub max_mean_logit_delta: f32,
    /// See [`MIN_PASSING_STEP_FRACTION`].
    pub min_passing_step_fraction: f32,
    /// See [`ALLOWED_FAILING_STEPS`]. Absent from a serialized tolerance of
    /// an earlier release, it reads as 0, the rule that release applied.
    #[serde(default)]
    pub allowed_failing_steps: usize,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            min_index_overlap: MIN_INDEX_OVERLAP,
            max_mean_logit_delta: MAX_MEAN_LOGIT_DELTA,
            min_passing_step_fraction: MIN_PASSING_STEP_FRACTION,
            allowed_failing_steps: ALLOWED_FAILING_STEPS,
        }
    }
}

/// One committed logit: token id and raw (pre-sampler) logit value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopKEntry {
    pub token_id: u32,
    pub logit: f32,
}

/// Commitment record for one generated token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// The token the provider actually sampled at this step.
    pub token_id: u32,
    /// Top-k raw logits at this step, sorted by logit descending
    /// (ties broken by ascending token id).
    pub top_k: Vec<TopKEntry>,
}

/// Full commitment blob for one inference call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCommitment {
    /// Number of logits committed per step.
    pub k: u8,
    /// Prompt length in tokens (BOS included), fixing the positions a
    /// verifier must read logits from: step `j` maps to sequence index
    /// `prompt_tokens - 1 + j`.
    pub prompt_tokens: u32,
    /// One record per generated token, in generation order.
    pub steps: Vec<StepRecord>,
}

impl InferenceCommitment {
    /// Deterministic byte encoding hashed into the commitment.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            COMMITMENT_DOMAIN_TAG.len() + 9 + self.steps.len() * (5 + self.k as usize * 8),
        );
        out.extend_from_slice(COMMITMENT_DOMAIN_TAG);
        out.push(self.k);
        out.extend_from_slice(&self.prompt_tokens.to_le_bytes());
        out.extend_from_slice(&(self.steps.len() as u32).to_le_bytes());
        for step in &self.steps {
            out.extend_from_slice(&step.token_id.to_le_bytes());
            out.push(step.top_k.len() as u8);
            for entry in &step.top_k {
                out.extend_from_slice(&entry.token_id.to_le_bytes());
                out.extend_from_slice(&entry.logit.to_le_bytes());
            }
        }
        out
    }

    /// SHA-256 of the canonical encoding.
    pub fn commitment_hash(&self) -> [u8; 32] {
        let digest = Sha256::digest(self.canonical_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&digest);
        hash
    }

    /// Hex form of [`commitment_hash`](Self::commitment_hash).
    pub fn hash_hex(&self) -> String {
        hex::encode(self.commitment_hash())
    }

    /// The generated token ids, in order — what a verifier feeds back
    /// through the model after the prompt.
    pub fn output_token_ids(&self) -> Vec<u32> {
        self.steps.iter().map(|s| s.token_id).collect()
    }
}

/// Per-step comparison between a committed record and recomputed logits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepComparison {
    /// Whether the token the provider says it emitted is among the
    /// recomputed top-k. A substituted token that the model would not have
    /// produced fails here even when the committed logits are genuine.
    pub token_supported: bool,
    /// Fraction of committed token ids present in the recomputed top-k.
    pub index_overlap: f32,
    /// Mean absolute logit difference over the shared token ids.
    pub mean_logit_delta: f32,
    pub pass: bool,
}

/// Outcome of verifying a full commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationOutcome {
    pub steps_total: usize,
    pub steps_passed: usize,
    /// Indices of failing steps (capped at 32 for reporting).
    pub failing_steps: Vec<usize>,
    pub pass: bool,
}

/// Select the top-k logits from a full logit row, sorted by logit
/// descending with ties broken by ascending token id — deterministic
/// regardless of the input's memory order.
pub fn top_k_from_logits(logits: &[f32], k: usize) -> Vec<TopKEntry> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if k < indexed.len() {
        indexed.select_nth_unstable_by(k - 1, cmp);
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(cmp);
    indexed
        .into_iter()
        .map(|(token_id, logit)| TopKEntry { token_id, logit })
        .collect()
}

/// Compare one committed step against the recomputed top-k at the same
/// position.
///
/// A step passes only when the emitted token is among the recomputed top-k
/// and the committed top-k matches the recomputed one within tolerance. A
/// token sampled from outside the verifier's top-k therefore fails its step;
/// the whole commitment tolerates that at the rate
/// [`MIN_PASSING_STEP_FRACTION`] and [`ALLOWED_FAILING_STEPS`] allow.
pub fn compare_step(committed: &StepRecord, recomputed: &[TopKEntry]) -> StepComparison {
    compare_step_with(committed, recomputed, &Tolerance::default())
}

/// [`compare_step`] under the given thresholds.
pub fn compare_step_with(committed: &StepRecord, recomputed: &[TopKEntry], tolerance: &Tolerance) -> StepComparison {
    let token_supported = recomputed.iter().any(|r| r.token_id == committed.token_id);
    if committed.top_k.is_empty() {
        return StepComparison {
            token_supported,
            index_overlap: 0.0,
            mean_logit_delta: f32::INFINITY,
            pass: false,
        };
    }
    let mut shared = 0usize;
    let mut delta_sum = 0f64;
    for entry in &committed.top_k {
        if let Some(rec) = recomputed.iter().find(|r| r.token_id == entry.token_id) {
            shared += 1;
            delta_sum += (entry.logit - rec.logit).abs() as f64;
        }
    }
    let index_overlap = shared as f32 / committed.top_k.len() as f32;
    let mean_logit_delta = if shared > 0 {
        (delta_sum / shared as f64) as f32
    } else {
        f32::INFINITY
    };
    let pass = token_supported
        && index_overlap >= tolerance.min_index_overlap
        && mean_logit_delta <= tolerance.max_mean_logit_delta;
    StepComparison {
        token_supported,
        index_overlap,
        mean_logit_delta,
        pass,
    }
}

/// Verify a full commitment against recomputed top-k rows, one per
/// step. Fails outright when the verifier could not reproduce the committed
/// shape: a step count that differs, a `k` outside
/// [`MIN_COMMITMENT_K`]`..=`[`MAX_COMMITMENT_K`], or a step that commits a
/// number of logits other than `k` (which would also make the canonical
/// encoding ambiguous).
pub fn verify_commitment(
    committed: &InferenceCommitment,
    recomputed: &[Vec<TopKEntry>],
) -> VerificationOutcome {
    verify_commitment_with(committed, recomputed, &Tolerance::default())
}

/// [`verify_commitment`] under the given thresholds.
pub fn verify_commitment_with(
    committed: &InferenceCommitment,
    recomputed: &[Vec<TopKEntry>],
    tolerance: &Tolerance,
) -> VerificationOutcome {
    let well_formed = (MIN_COMMITMENT_K..=MAX_COMMITMENT_K).contains(&committed.k)
        && committed.steps.iter().all(|s| s.top_k.len() == usize::from(committed.k));
    if !well_formed || committed.steps.is_empty() || committed.steps.len() != recomputed.len() {
        return VerificationOutcome {
            steps_total: committed.steps.len(),
            steps_passed: 0,
            failing_steps: Vec::new(),
            pass: false,
        };
    }
    let mut steps_passed = 0usize;
    let mut failing_steps = Vec::new();
    for (i, (step, rec)) in committed.steps.iter().zip(recomputed.iter()).enumerate() {
        if compare_step_with(step, rec, tolerance).pass {
            steps_passed += 1;
        } else if failing_steps.len() < 32 {
            failing_steps.push(i);
        }
    }
    let total = committed.steps.len();
    let failing = total - steps_passed;
    let credited = steps_passed + tolerance.allowed_failing_steps.min(failing);
    VerificationOutcome {
        steps_total: total,
        steps_passed,
        failing_steps,
        pass: credited as f32 / total as f32 >= tolerance.min_passing_step_fraction
            && (tolerance.allowed_failing_steps == 0 || steps_passed > failing),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(token_id: u32, logit: f32) -> TopKEntry {
        TopKEntry { token_id, logit }
    }

    fn commitment_with(steps: Vec<StepRecord>) -> InferenceCommitment {
        InferenceCommitment {
            k: 4,
            prompt_tokens: 8,
            steps,
        }
    }

    #[test]
    fn top_k_selects_and_orders_deterministically() {
        let logits = vec![0.1, 5.0, 3.0, 5.0, -1.0, 4.0];
        let top = top_k_from_logits(&logits, 4);
        // Tie between ids 1 and 3 at 5.0 breaks toward the lower id.
        assert_eq!(
            top.iter().map(|e| e.token_id).collect::<Vec<_>>(),
            vec![1, 3, 5, 2]
        );
    }

    #[test]
    fn top_k_handles_short_rows() {
        let logits = vec![2.0, 1.0];
        let top = top_k_from_logits(&logits, 16);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].token_id, 0);
    }

    #[test]
    fn canonical_hash_is_stable_and_sensitive() {
        let c1 = commitment_with(vec![StepRecord {
            token_id: 42,
            top_k: vec![entry(42, 9.5), entry(7, 8.0)],
        }]);
        let c2 = c1.clone();
        assert_eq!(c1.hash_hex(), c2.hash_hex());

        let mut c3 = c1.clone();
        c3.steps[0].top_k[1].logit = 8.0001;
        assert_ne!(c1.hash_hex(), c3.hash_hex());

        let mut c4 = c1.clone();
        c4.prompt_tokens = 9;
        assert_ne!(c1.hash_hex(), c4.hash_hex());
    }

    #[test]
    fn identical_step_passes() {
        let step = StepRecord {
            token_id: 5,
            top_k: vec![entry(5, 10.0), entry(9, 9.0), entry(2, 8.5), entry(1, 7.0)],
        };
        let cmp = compare_step(&step, &step.top_k);
        assert!(cmp.pass);
        assert_eq!(cmp.index_overlap, 1.0);
        assert_eq!(cmp.mean_logit_delta, 0.0);
    }

    #[test]
    fn small_drift_within_tolerance_passes() {
        let step = StepRecord {
            token_id: 5,
            top_k: vec![entry(5, 10.0), entry(9, 9.0), entry(2, 8.5), entry(1, 7.0)],
        };
        // One index swapped out (3/4 = 0.75 overlap) + 0.1 logit drift.
        let recomputed = vec![entry(5, 10.1), entry(9, 8.9), entry(2, 8.6), entry(77, 7.1)];
        let cmp = compare_step(&step, &recomputed);
        assert!(
            cmp.pass,
            "overlap {} delta {}",
            cmp.index_overlap, cmp.mean_logit_delta
        );
    }

    #[test]
    fn large_logit_drift_fails() {
        let step = StepRecord {
            token_id: 5,
            top_k: vec![entry(5, 10.0), entry(9, 9.0)],
        };
        let recomputed = vec![entry(5, 12.0), entry(9, 6.0)];
        assert!(!compare_step(&step, &recomputed).pass);
    }

    #[test]
    fn disjoint_indices_fail() {
        let step = StepRecord {
            token_id: 5,
            top_k: vec![entry(5, 10.0), entry(9, 9.0)],
        };
        let recomputed = vec![entry(1, 10.0), entry(2, 9.0)];
        let cmp = compare_step(&step, &recomputed);
        assert!(!cmp.pass);
        assert_eq!(cmp.index_overlap, 0.0);
    }

    fn four(first: u32) -> Vec<TopKEntry> {
        vec![entry(first, 5.0), entry(first + 1, 4.0), entry(first + 2, 3.0), entry(first + 3, 2.0)]
    }

    /// `passed` exact steps followed by `failed` disjoint ones.
    fn mixed(passed: usize, failed: usize) -> (InferenceCommitment, Vec<Vec<TopKEntry>>) {
        let good = StepRecord { token_id: 1, top_k: four(1) };
        let bad = StepRecord { token_id: 3, top_k: four(3) };
        let mut steps = vec![good.clone(); passed];
        steps.extend(vec![bad; failed]);
        let mut recomputed = vec![good.top_k; passed];
        recomputed.extend(vec![four(100); failed]);
        (commitment_with(steps), recomputed)
    }

    #[test]
    fn the_default_allows_a_few_failing_steps_beyond_the_fraction() {
        for (passed, failed, pass) in [(29, 4, true), (8, 1, true), (23, 1, true), (7, 3, true), (6, 4, false), (2, 1, true), (1, 1, false), (0, 1, false)] {
            let (committed, recomputed) = mixed(passed, failed);
            let outcome = verify_commitment(&committed, &recomputed);
            assert_eq!(outcome.pass, pass, "{passed} passing, {failed} failing");
            assert_eq!(outcome.steps_passed, passed);
        }
    }

    #[test]
    fn without_an_allowance_the_fraction_alone_decides() {
        let tolerance = Tolerance { allowed_failing_steps: 0, ..Tolerance::default() };
        for (passed, failed, pass) in [(9, 1, true), (8, 2, false), (8, 1, false), (29, 4, false)] {
            let (committed, recomputed) = mixed(passed, failed);
            assert_eq!(verify_commitment_with(&committed, &recomputed, &tolerance).pass, pass, "{passed} passing, {failed} failing");
        }
        let earlier: Tolerance =
            serde_json::from_str(r#"{"min_index_overlap":0.75,"max_mean_logit_delta":0.25,"min_passing_step_fraction":0.9}"#).unwrap();
        assert_eq!(earlier.allowed_failing_steps, 0);
    }

    #[test]
    fn verify_requires_passing_fraction() {
        let good = StepRecord { token_id: 1, top_k: four(1) };
        let bad = StepRecord { token_id: 3, top_k: four(3) };
        // 10 steps: 9 exact matches + 1 disjoint recompute = 0.9 passes.
        let mut steps = vec![good.clone(); 9];
        steps.push(bad.clone());
        let committed = commitment_with(steps);
        let mut recomputed: Vec<Vec<TopKEntry>> = vec![good.top_k.clone(); 9];
        recomputed.push(four(100));
        let outcome = verify_commitment(&committed, &recomputed);
        assert!(outcome.pass);
        assert_eq!(outcome.steps_passed, 9);
        assert_eq!(outcome.failing_steps, vec![9]);

        // Without the allowance, 10 steps: 8 passes = 0.8 fails.
        let mut steps = vec![good.clone(); 8];
        steps.push(bad.clone());
        steps.push(bad.clone());
        let committed = commitment_with(steps);
        let mut recomputed: Vec<Vec<TopKEntry>> = vec![good.top_k.clone(); 8];
        recomputed.push(four(100));
        recomputed.push(four(100));
        let fraction_only = Tolerance { allowed_failing_steps: 0, ..Tolerance::default() };
        assert!(!verify_commitment_with(&committed, &recomputed, &fraction_only).pass);
    }

    #[test]
    fn a_substituted_token_fails_its_step_even_with_genuine_logits() {
        // The committed logits are exactly what the verifier recomputes, but
        // the emitted token is one the model does not rank at all.
        let honest = StepRecord { token_id: 1, top_k: four(1) };
        let forged = StepRecord { token_id: 999, top_k: four(1) };
        assert!(compare_step(&honest, &four(1)).pass);
        let cmp = compare_step(&forged, &four(1));
        assert!(!cmp.token_supported);
        assert!(!cmp.pass);
        // A whole output of substituted tokens does not verify.
        let committed = commitment_with(vec![forged; 10]);
        assert!(!verify_commitment(&committed, &vec![four(1); 10]).pass);
    }

    #[test]
    fn a_commitment_below_the_minimum_k_is_refused() {
        let step = StepRecord { token_id: 1, top_k: vec![entry(1, 5.0)] };
        let committed = InferenceCommitment { k: 1, prompt_tokens: 8, steps: vec![step.clone(); 10] };
        assert!(!verify_commitment(&committed, &vec![step.top_k.clone(); 10]).pass);
        assert_eq!(commitment_k_for(1), MIN_COMMITMENT_K);
        assert_eq!(commitment_k_for(200), MAX_COMMITMENT_K);
        assert_eq!(commitment_k_for(16), 16);
    }

    #[test]
    fn a_step_with_other_than_k_logits_is_refused() {
        let short = StepRecord { token_id: 1, top_k: four(1)[..3].to_vec() };
        let committed = commitment_with(vec![StepRecord { token_id: 1, top_k: four(1) }, short]);
        assert!(!verify_commitment(&committed, &[four(1), four(1)]).pass);
    }

    #[test]
    fn verify_fails_on_length_mismatch() {
        let step = StepRecord { token_id: 1, top_k: four(1) };
        let committed = commitment_with(vec![step.clone(), step.clone()]);
        let recomputed = vec![step.top_k.clone()];
        assert!(!verify_commitment(&committed, &recomputed).pass);
    }

    #[test]
    fn empty_commitment_fails() {
        let committed = commitment_with(Vec::new());
        assert!(!verify_commitment(&committed, &[]).pass);
    }

    #[test]
    fn a_wider_tolerance_passes_what_the_default_fails() {
        let committed = commitment_with(vec![StepRecord {
            token_id: 1,
            top_k: vec![entry(1, 9.0), entry(2, 8.0), entry(3, 7.0), entry(4, 6.0)],
        }]);
        // Every logit 0.75 off: past the default mean delta of 0.6.
        let recomputed = vec![vec![entry(1, 9.75), entry(2, 8.75), entry(3, 7.75), entry(4, 6.75)]];
        assert_eq!(Tolerance::default().max_mean_logit_delta, MAX_MEAN_LOGIT_DELTA);
        assert!(!verify_commitment(&committed, &recomputed).pass);
        let with_default = verify_commitment_with(&committed, &recomputed, &Tolerance::default());
        assert_eq!((with_default.pass, with_default.steps_passed), (false, 0));
        let wide = Tolerance { max_mean_logit_delta: 0.8, ..Tolerance::default() };
        assert!(verify_commitment_with(&committed, &recomputed, &wide).pass);
        assert!(compare_step_with(&committed.steps[0], &recomputed[0], &wide).pass);
    }
}
