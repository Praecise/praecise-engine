//! Published speculative drafters, and which of them we may actually ship.
//!
//! A drafter is a second set of weights, so it carries its own licence — which
//! is frequently *not* the licence of the model it drafts for. The two are
//! published separately, by different orgs, and nothing in an engine's
//! configuration surface makes the difference visible: a checkpoint path is
//! just a string.
//!
//! That is the failure this module exists to prevent. `Qwen3.8-27B-DFlash2` is
//! Apache-2.0 and fine to ship. `GLM-5.3-Flash-DFlash2` is CC BY-NC-ND — no
//! commercial use without contacting its publisher, and no derivatives at all.
//! Both are on the same hub, both advertise similar speedups, and confusing
//! them is a licensing problem discovered long after deployment rather than a
//! runtime error caught at load.
//!
//! So the catalogue records the licence alongside the checkpoint, and
//! [`Drafter::ensure_permissive`] refuses the ones we cannot use. Entries for
//! non-permissive drafters are kept rather than omitted, because a caller who
//! names one deserves to be told *why* it is refused — silence would read as
//! "unknown drafter" and send them looking in the wrong place.
//!
//! ## What is recorded, and what is not
//!
//! These are catalogue facts — names, licences, block sizes, engine support —
//! not measurements taken here. Speedup figures are deliberately absent: they
//! are published at a specific concurrency on specific hardware, and a number
//! carried around without those qualifiers is worse than no number. Measure
//! acceptance in your own deployment; [`crate::ngram::NgramCache`] and
//! [`crate::spec_policy`] are built to let you.

use crate::error::{Error, Result};

/// Licence terms, reduced to the question that matters: may we ship this?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Licence {
    /// Apache-2.0 — permissive, patent grant, safe to redistribute.
    Apache2,
    /// MIT — permissive.
    Mit,
    /// Creative Commons non-commercial, no derivatives. Not usable here.
    CcByNcNd4,
    /// A bespoke licence needing a human to read it. Treated as non-permissive
    /// until someone has, because the safe default for an unread licence is
    /// "no".
    Custom(&'static str),
    /// The publisher stated no licence at all.
    ///
    /// Not the same as permissive. Absent terms mean no grant of rights, so
    /// this is refused exactly like an explicitly restrictive licence — and
    /// several real checkpoints are in this state, which is why it needs its
    /// own variant rather than being folded into `Custom`.
    Unstated,
}

impl Licence {
    /// Whether this licence permits commercial use and redistribution.
    #[must_use]
    pub fn is_permissive(self) -> bool {
        matches!(self, Licence::Apache2 | Licence::Mit)
    }

    /// Human-readable name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Licence::Apache2 => "Apache-2.0",
            Licence::Mit => "MIT",
            Licence::CcByNcNd4 => "CC-BY-NC-ND-4.0",
            Licence::Custom(s) => s,
            Licence::Unstated => "unstated",
        }
    }
}

/// The drafting algorithm a checkpoint implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftAlgorithm {
    /// Block-diffusion drafting: the whole block is emitted in one forward
    /// pass, so draft cost is roughly independent of block size. (Verification
    /// cost is not — see [`crate::spec_policy`].)
    DFlash,
    /// DFlash plus a candidate selector and a two-tap dynamic convolution that
    /// addresses accuracy decay toward the end of a block. Same engine flag as
    /// DFlash; the version is selected by the checkpoint, not the flag.
    DFlash2,
    /// Autoregressive drafter fusing target features into its input
    /// embeddings. The signal dilutes with draft depth, which is why these are
    /// kept shallow.
    Eagle3,
}

impl DraftAlgorithm {
    /// The identifier engines accept. DFlash and DFlash2 share it deliberately.
    #[must_use]
    pub fn engine_id(self) -> &'static str {
        match self {
            DraftAlgorithm::DFlash | DraftAlgorithm::DFlash2 => "dflash",
            DraftAlgorithm::Eagle3 => "eagle3",
        }
    }
}

/// A published drafter checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drafter {
    /// Hub repository, as it would be given to an engine.
    pub repo: &'static str,
    /// The model this drafts for.
    pub target: &'static str,
    pub algorithm: DraftAlgorithm,
    pub licence: Licence,
    /// Block size the checkpoint was trained at.
    ///
    /// Not a free tuning knob: running a checkpoint at a block size other than
    /// its trained one measurably degrades acceptance, so this travels with the
    /// checkpoint rather than with the request.
    pub trained_block_size: u8,
}

impl Drafter {
    /// Fail unless this drafter's licence permits shipping it.
    ///
    /// # Errors
    /// [`Error::DrafterLicence`] naming the licence, so the refusal is
    /// actionable rather than a bare denial.
    pub fn ensure_permissive(&self) -> Result<()> {
        if self.licence.is_permissive() {
            return Ok(());
        }
        Err(Error::DrafterLicence { repo: self.repo, licence: self.licence.as_str() })
    }

    /// The `num_speculative_tokens` value for a vLLM-style config.
    ///
    /// vLLM counts speculative tokens as block size minus one, where SGLang's
    /// `--speculative-num-draft-tokens` is the block size itself. The
    /// off-by-one is real and belongs here rather than in every caller.
    #[must_use]
    pub fn vllm_speculative_tokens(&self) -> u8 {
        self.trained_block_size.saturating_sub(1)
    }
}

/// Every drafter the catalogue knows about, permissive or not.
///
/// Non-permissive entries are present on purpose — see the module docs. Use
/// [`permissive`] for the ones that can actually be shipped.
pub const CATALOGUE: &[Drafter] = &[
    // ---- DFlash2: selector + dynamic convolution, newest generation --------
    Drafter {
        repo: "z-lab/Qwen3.8-27B-DFlash2",
        target: "Qwen/Qwen3.8-27B",
        algorithm: DraftAlgorithm::DFlash2,
        licence: Licence::Apache2,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/Muse-Glimmer-30B-DFlash2",
        target: "Muse-Glimmer-30B",
        algorithm: DraftAlgorithm::DFlash2,
        licence: Licence::Apache2,
        trained_block_size: 16,
    },
    // Kept so that naming it produces a licence refusal rather than a
    // not-found: commercial use requires contacting the publisher, and "no
    // derivatives" rules out any fine-tuning.
    Drafter {
        repo: "incoai/GLM-5.3-Flash-DFlash2",
        target: "GLM-5.3-Flash",
        algorithm: DraftAlgorithm::DFlash2,
        licence: Licence::CcByNcNd4,
        trained_block_size: 8,
    },

    // ---- DFlash v1 ---------------------------------------------------------
    Drafter {
        repo: "z-lab/gemma-4-31B-it-DFlash",
        target: "google/gemma-4-31B-it",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Apache2,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/gemma-4-26B-A4B-it-DFlash",
        target: "google/gemma-4-26B-A4B-it",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Apache2,
        trained_block_size: 8,
    },
    // Published, but the card states no licence at all. Absent terms grant
    // nothing, so this is refused like a restrictive one — see Licence::Unstated.
    Drafter {
        repo: "z-lab/gemma4-12B-it-DFlash",
        target: "google/gemma-4-12B-it",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Unstated,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/Qwen3.6-27B-DFlash",
        target: "Qwen/Qwen3.6-27B",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Mit,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/Qwen3.5-27B-DFlash",
        target: "Qwen/Qwen3.5-27B",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Apache2,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/Qwen3-Coder-30B-A3B-DFlash",
        target: "Qwen/Qwen3-Coder-30B-A3B",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Mit,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/Kimi-K2.6-DFlash",
        target: "moonshotai/Kimi-K2.6",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Mit,
        trained_block_size: 8,
    },
    Drafter {
        repo: "z-lab/gpt-oss-120b-DFlash",
        target: "openai/gpt-oss-120b",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Mit,
        trained_block_size: 8,
    },
    // Also unlicensed on the hub.
    Drafter {
        repo: "z-lab/GLM-5.1-FP8-DFlash",
        target: "GLM-5.1-FP8",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Unstated,
        trained_block_size: 8,
    },
    // b16 in the name is the trained block size, and it is not interchangeable
    // with the b8 checkpoints above — running either at the other's size is
    // Trap::BlockSizeMismatch.
    Drafter {
        repo: "z-lab/Qwen3-8B-DFlash-b16",
        target: "Qwen/Qwen3-8B",
        algorithm: DraftAlgorithm::DFlash,
        licence: Licence::Apache2,
        trained_block_size: 16,
    },
];

/// The drafters that may be shipped.
pub fn permissive() -> impl Iterator<Item = &'static Drafter> {
    CATALOGUE.iter().filter(|d| d.licence.is_permissive())
}

/// Find a permissive drafter for a target model.
///
/// Returns `None` both when nothing is published and when the only published
/// drafter is non-permissive — from the engine's point of view those are the
/// same situation, and [`find`] is available when the difference matters.
#[must_use]
pub fn for_target(target: &str) -> Option<&'static Drafter> {
    permissive().find(|d| d.target.eq_ignore_ascii_case(target))
}

/// Look a drafter up by repository, permissive or not.
#[must_use]
pub fn find(repo: &str) -> Option<&'static Drafter> {
    CATALOGUE.iter().find(|d| d.repo.eq_ignore_ascii_case(repo))
}

/// A known way a drafter deployment goes wrong *without* reporting an error.
///
/// Every variant here is a documented failure that produces plausible output or
/// a silently reduced speedup rather than a crash — which is why they are worth
/// enumerating in code. A trap that announced itself would not need this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    /// A checkpoint run at a block size other than the one it was trained at.
    /// Acceptance drops measurably and nothing reports it, because both values
    /// are individually valid.
    BlockSizeMismatch { trained: u8, requested: u8 },
    /// vLLM silently degrades a DFlash2 checkpoint to DFlash v1 behaviour when
    /// its v2 model runner is disabled: the selector and convolution are
    /// skipped, no error is raised, and the only symptom is a smaller speedup.
    V2RunnerDisabled,
    /// Structured output and this drafter cannot both be active. Grammar
    /// masking and the drafter's candidate selection disagree about the token
    /// set, and the combination is documented as unsupported.
    StructuredOutputConflict,
    /// A sampler restricted to part of the vocabulary. These drafters score
    /// over the full vocabulary; a partial one makes the scores incomparable.
    PartialVocabularySampling,
}

impl Trap {
    /// What went wrong and what to do about it.
    #[must_use]
    pub fn explain(self) -> String {
        match self {
            Trap::BlockSizeMismatch { trained, requested } => format!(
                "block size {requested} but this checkpoint was trained at {trained}; \
                 acceptance degrades and nothing reports it. Use {trained}, or a \
                 checkpoint trained at {requested}."
            ),
            Trap::V2RunnerDisabled => "the v2 model runner is disabled, so a DFlash2 \
                 checkpoint runs as DFlash v1: no selector, no dynamic convolution, and \
                 no error. Enable it or use a v1 checkpoint deliberately."
                .to_string(),
            Trap::StructuredOutputConflict => "grammar-constrained output is active. \
                 Constrained decoding and this drafter are documented as incompatible; \
                 the grammar masks tokens the drafter has already scored over."
                .to_string(),
            Trap::PartialVocabularySampling => "sampling is restricted to part of the \
                 vocabulary. These drafters score over the full vocabulary, so the \
                 comparison against the target is not meaningful."
                .to_string(),
        }
    }
}

/// How a drafter is about to be run.
#[derive(Clone, Copy, Debug)]
pub struct DeploymentIntent {
    /// Block size the caller intends to use, if overriding the trained one.
    pub block_size: Option<u8>,
    /// Whether the engine's v2 model runner is active (vLLM-specific).
    pub v2_model_runner: bool,
    /// Whether a grammar or JSON schema constrains the output.
    pub structured_output: bool,
    /// Whether sampling sees the whole vocabulary.
    pub full_vocabulary: bool,
}

impl Default for DeploymentIntent {
    /// The configuration in which these drafters actually work: trained block
    /// size, v2 runner on, no grammar, full vocabulary.
    fn default() -> Self {
        Self {
            block_size: None,
            v2_model_runner: true,
            structured_output: false,
            full_vocabulary: true,
        }
    }
}

/// Check an intended deployment against the known traps.
///
/// Returns every trap that applies rather than the first, because these are
/// independent: fixing a block size does not resolve a grammar conflict, and a
/// caller shown one problem at a time will fix one and redeploy.
///
/// This is advisory by design — it reports, and the caller decides. Refusing
/// outright would be wrong for the cases where someone knowingly accepts a
/// smaller speedup.
#[must_use]
pub fn preflight(drafter: &Drafter, intent: &DeploymentIntent) -> Vec<Trap> {
    let mut traps = Vec::new();

    if let Some(requested) = intent.block_size
        && requested != drafter.trained_block_size
    {
        traps.push(Trap::BlockSizeMismatch {
            trained: drafter.trained_block_size,
            requested,
        });
    }

    // Only DFlash2 has the v2-only components to lose; flagging this for a v1
    // checkpoint would be noise.
    if drafter.algorithm == DraftAlgorithm::DFlash2 && !intent.v2_model_runner {
        traps.push(Trap::V2RunnerDisabled);
    }

    if intent.structured_output {
        traps.push(Trap::StructuredOutputConflict);
    }

    if !intent.full_vocabulary {
        traps.push(Trap::PartialVocabularySampling);
    }

    traps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_is_not_empty_and_has_permissive_entries() {
        assert!(!CATALOGUE.is_empty());
        assert!(permissive().count() > 0);
    }

    #[test]
    fn non_commercial_drafters_are_refused() {
        let glm = find("incoai/GLM-5.3-Flash-DFlash2").expect("kept in the catalogue");
        assert!(!glm.licence.is_permissive());
        let msg = glm.ensure_permissive().unwrap_err().to_string();
        assert!(msg.contains("CC-BY-NC-ND"), "the refusal must name the licence: {msg}");
    }

    #[test]
    fn permissive_drafters_are_allowed() {
        let q = find("z-lab/Qwen3.8-27B-DFlash2").expect("in the catalogue");
        assert_eq!(q.licence, Licence::Apache2);
        assert!(q.ensure_permissive().is_ok());
    }

    #[test]
    fn lookup_by_target_never_returns_a_non_permissive_drafter() {
        // The trap this module exists for: GLM-5.3-Flash HAS a published
        // drafter, and it must not be handed back by a convenience lookup.
        assert!(for_target("GLM-5.3-Flash").is_none());
        assert!(for_target("Qwen/Qwen3.8-27B").is_some());
    }

    #[test]
    fn every_permissive_entry_actually_passes_its_own_check() {
        for d in permissive() {
            assert!(d.ensure_permissive().is_ok(), "{} claims permissive but fails", d.repo);
        }
    }

    #[test]
    fn an_unread_custom_licence_is_treated_as_non_permissive() {
        // The safe default for a licence nobody has read is "no".
        assert!(!Licence::Custom("some-vendor-terms").is_permissive());
    }

    #[test]
    fn vllm_token_count_is_one_below_the_block_size() {
        // vLLM's num_speculative_tokens is block size - 1; SGLang's
        // num_draft_tokens is the block size. Getting this wrong silently
        // changes the block a checkpoint runs at.
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        assert_eq!(q.trained_block_size, 8);
        assert_eq!(q.vllm_speculative_tokens(), 7);
    }

    #[test]
    fn dflash_versions_share_an_engine_identifier() {
        // Both versions are selected by checkpoint, not by flag. A caller that
        // expected separate identifiers would build an invalid config.
        assert_eq!(DraftAlgorithm::DFlash.engine_id(), DraftAlgorithm::DFlash2.engine_id());
    }

    #[test]
    fn a_default_deployment_hits_no_traps() {
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        assert!(preflight(q, &DeploymentIntent::default()).is_empty());
    }

    #[test]
    fn the_trained_block_size_is_not_flagged() {
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        let intent = DeploymentIntent {
            block_size: Some(q.trained_block_size),
            ..DeploymentIntent::default()
        };
        assert!(preflight(q, &intent).is_empty(), "the trained size must not warn");
    }

    #[test]
    fn a_mismatched_block_size_is_caught() {
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        let intent = DeploymentIntent { block_size: Some(16), ..DeploymentIntent::default() };
        let traps = preflight(q, &intent);
        assert_eq!(traps, vec![Trap::BlockSizeMismatch { trained: 8, requested: 16 }]);
        assert!(traps[0].explain().contains('8'), "should name the trained size");
    }

    #[test]
    fn a_disabled_v2_runner_is_caught_for_dflash2_only() {
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        let intent = DeploymentIntent { v2_model_runner: false, ..DeploymentIntent::default() };
        assert!(preflight(q, &intent).contains(&Trap::V2RunnerDisabled));

        // A v1 checkpoint has nothing to silently lose, so it must not warn.
        let v1 = Drafter { algorithm: DraftAlgorithm::DFlash, ..*q };
        assert!(!preflight(&v1, &intent).contains(&Trap::V2RunnerDisabled));
    }

    #[test]
    fn structured_output_and_partial_vocab_are_caught() {
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        let intent = DeploymentIntent {
            structured_output: true,
            full_vocabulary: false,
            ..DeploymentIntent::default()
        };
        let traps = preflight(q, &intent);
        assert!(traps.contains(&Trap::StructuredOutputConflict));
        assert!(traps.contains(&Trap::PartialVocabularySampling));
    }

    #[test]
    fn all_applicable_traps_are_reported_together() {
        // Independent problems: a caller shown one at a time fixes one and
        // redeploys into the next.
        let q = find("z-lab/Qwen3.8-27B-DFlash2").unwrap();
        let intent = DeploymentIntent {
            block_size: Some(4),
            v2_model_runner: false,
            structured_output: true,
            full_vocabulary: false,
        };
        assert_eq!(preflight(q, &intent).len(), 4);
    }

    #[test]
    fn every_trap_explains_itself_in_actionable_terms() {
        for t in [
            Trap::BlockSizeMismatch { trained: 8, requested: 16 },
            Trap::V2RunnerDisabled,
            Trap::StructuredOutputConflict,
            Trap::PartialVocabularySampling,
        ] {
            let e = t.explain();
            assert!(e.len() > 40, "too terse to act on: {e}");
        }
    }

    #[test]
    fn an_unstated_licence_is_refused() {
        // Absent terms grant nothing. This is a real state on the hub, not a
        // hypothetical: two catalogued checkpoints publish no licence.
        assert!(!Licence::Unstated.is_permissive());
        let g = find("z-lab/gemma4-12B-it-DFlash").expect("catalogued");
        assert!(g.ensure_permissive().is_err(), "no licence must not mean yes");
    }

    #[test]
    fn gemma4_has_usable_drafters() {
        // The 31B and 26B-A4B cards state Apache-2.0; the 12B states nothing.
        assert!(for_target("google/gemma-4-31B-it").is_some());
        assert!(for_target("google/gemma-4-26B-A4B-it").is_some());
        assert!(
            for_target("google/gemma-4-12B-it").is_none(),
            "the 12B drafter states no licence and must not be offered"
        );
    }

    #[test]
    fn block_sizes_differ_across_checkpoints() {
        // A catalogue where every entry shared a block size would make
        // Trap::BlockSizeMismatch untestable against real data.
        let sizes: std::collections::BTreeSet<u8> =
            CATALOGUE.iter().map(|d| d.trained_block_size).collect();
        assert!(sizes.len() > 1, "expected varied block sizes, got {sizes:?}");
    }

    #[test]
    fn no_two_entries_share_a_repo() {
        let mut seen = std::collections::BTreeSet::new();
        for d in CATALOGUE {
            assert!(seen.insert(d.repo), "duplicate catalogue entry: {}", d.repo);
        }
    }

    #[test]
    fn no_catalogue_entry_has_a_zero_block_size() {
        // Zero would mean "no speculation" while looking like a configured
        // drafter, which is exactly the kind of silent misconfiguration the
        // catalogue is meant to remove.
        for d in CATALOGUE {
            assert!(d.trained_block_size > 0, "{} has block size 0", d.repo);
        }
    }
}
