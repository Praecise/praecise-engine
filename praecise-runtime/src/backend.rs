//! Backend selection: which inference runtime executes a request.
//!
//! Praecise Engine is an acceleration layer, not a runtime. Everything above
//! this module — speculation policy, n-gram drafting, sampling configuration,
//! stop handling — is expressed without reference to any particular engine.
//! This module is where that abstraction meets a concrete one.
//!
//! ## Two kinds of backend, and why the difference matters
//!
//! Backends do not all offer the same control surface, and pretending they do
//! is the mistake this module exists to prevent.
//!
//! - **Linked** ([`Integration::Linked`]) — the runtime is compiled in and
//!   called through FFI. The engine drives the decode loop itself, so it can
//!   propose a draft block, verify it, and inspect per-position logits.
//!   llama.cpp is linked.
//! - **Served** ([`Integration::Served`]) — the runtime is a separate process
//!   reached over HTTP. It owns its own decode loop. The engine can choose a
//!   model, sampling parameters and a schema, but it cannot interpose on
//!   token-by-token decoding, because there is no seam to interpose at. vLLM
//!   and SGLang are served.
//!
//! That distinction is load-bearing. Against a served backend, this engine's
//! speculation is **not** available — the remote runtime does its own, with its
//! own drafters. Reporting a speculation plan for such a backend would be a
//! lie the caller could not detect, so [`Backend::supports`] answers it up
//! front and [`plan_for`] refuses rather than pretends.
//!
//! ## What is actually implemented
//!
//! [`Backend::LlamaCpp`] works today. The served backends are **described but
//! not implemented**: they are enumerated here so that selection, capability
//! reporting and error messages are honest, and so adding one is filling in a
//! function rather than reshaping the crate. A caller that selects one gets
//! [`Error::BackendUnavailable`] naming exactly what is missing — never a
//! silent fallback to a different runtime, which would make a benchmark
//! meaningless without any visible sign.

use std::fmt;

use crate::error::{Error, Result};
use crate::spec_policy::{self, LoadState, ModelProfile, SpecPlan, SpecPolicy};

/// An inference runtime the engine can drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// llama.cpp, linked and driven through FFI. The only backend implemented.
    LlamaCpp,
    /// vLLM over its OpenAI-compatible HTTP API. Not implemented.
    Vllm,
    /// SGLang over its OpenAI-compatible HTTP API. Not implemented.
    SgLang,
    /// TensorRT-LLM via `trtllm-serve`, its OpenAI-compatible server.
    TensorRtLlm,
    /// MLX via `mlx_lm.server`. Apple Silicon.
    MlxLm,
}

/// How the engine reaches a backend — see the module docs on why this decides
/// what acceleration is possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integration {
    /// Compiled in and called through FFI; the engine owns the decode loop.
    Linked,
    /// A separate process over HTTP; the runtime owns its own decode loop.
    Served,
}

/// What a backend can and cannot do.
///
/// Deliberately reported rather than assumed: a caller that asks before acting
/// gets a truthful answer, and one that acts without asking gets an error
/// instead of a silent downgrade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub integration: Integration,
    /// Whether **this engine** can run its own speculative decoding here. False
    /// for served backends — they speculate internally, which is not the same
    /// thing and must not be counted as ours.
    pub engine_speculation: bool,
    /// Whether per-position logits are visible, which speculation verification
    /// requires and which nothing served exposes.
    pub logit_access: bool,
    /// Whether grammar-constrained decoding is available.
    pub structured_output: bool,
    /// Whether the backend is implemented at all today.
    pub implemented: bool,
}

impl Backend {
    /// Every backend the engine knows about, implemented or not.
    #[must_use]
    pub fn all() -> &'static [Backend] {
        &[Backend::LlamaCpp, Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm, Backend::MlxLm]
    }

    /// Stable identifier used in configuration and logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::LlamaCpp => "llama.cpp",
            Backend::Vllm => "vllm",
            Backend::SgLang => "sglang",
            Backend::TensorRtLlm => "tensorrt-llm",
            Backend::MlxLm => "mlx",
        }
    }

    /// Parse a backend name. Accepts the spellings people actually write.
    ///
    /// # Errors
    /// [`Error::BackendUnknown`] if the name matches nothing, listing what is
    /// valid — an unknown backend must not quietly become the default.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace(['-', '_'], ".").as_str() {
            "llama.cpp" | "llamacpp" | "llama" => Ok(Backend::LlamaCpp),
            "vllm" => Ok(Backend::Vllm),
            "sglang" | "sgl" => Ok(Backend::SgLang),
            "tensorrt.llm" | "tensorrtllm" | "trtllm" | "tensorrt" => Ok(Backend::TensorRtLlm),
            "mlx" | "mlx.lm" | "mlxlm" => Ok(Backend::MlxLm),
            _ => Err(Error::BackendUnknown {
                name: s.to_string(),
                known: Backend::all().iter().map(|b| b.as_str()).collect::<Vec<_>>().join(", "),
            }),
        }
    }

    /// What this backend supports.
    #[must_use]
    pub fn supports(self) -> Capabilities {
        match self {
            Backend::LlamaCpp => Capabilities {
                integration: Integration::Linked,
                engine_speculation: true,
                logit_access: true,
                structured_output: true,
                implemented: cfg!(feature = "bundled-llama"),
            },
            // Both served backends expose an OpenAI-compatible API: a model,
            // sampling parameters, and a JSON schema. None of that reaches
            // inside their decode loop, so engine-side speculation is
            // impossible by construction rather than merely unimplemented.
            Backend::Vllm | Backend::SgLang | Backend::TensorRtLlm => Capabilities {
                integration: Integration::Served,
                engine_speculation: false,
                logit_access: false,
                structured_output: true,
                implemented: true,
            },
            // MLX is the outlier and the reason `structured_output` is a
            // capability rather than an assumption: `mlx_lm.server` never reads
            // `response_format`, so a schema-constrained request degrades to
            // free text with no error. Reporting that up front is the only way
            // a caller finds out before the output is wrong.
            Backend::MlxLm => Capabilities {
                integration: Integration::Served,
                engine_speculation: false,
                logit_access: false,
                structured_output: false,
                implemented: true,
            },
        }
    }

    /// Whether this backend can be used right now.
    #[must_use]
    pub fn is_available(self) -> bool {
        self.supports().implemented
    }

    /// Fail unless this backend can actually serve a request.
    ///
    /// # Errors
    /// [`Error::BackendUnavailable`] with the reason — a missing feature flag
    /// reads differently from a backend nobody has written yet, and a caller
    /// deserves to know which.
    pub fn ensure_available(self) -> Result<()> {
        if self.is_available() {
            return Ok(());
        }
        let reason = match self {
            Backend::LlamaCpp => {
                "the `bundled-llama` feature is not enabled; build with it, or pass in a \
                 backend the host application already links"
            }
            Backend::Vllm | Backend::SgLang | Backend::TensorRtLlm | Backend::MlxLm => {
                "served backends are reachable but need an endpoint; build one with \
                 `served::Endpoint` and drive it with the caller's HTTP client"
            }
        };
        Err(Error::BackendUnavailable { backend: self.as_str(), reason })
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Default for Backend {
    /// llama.cpp: the only implemented backend, and the one whose linked
    /// integration the acceleration paths are written against.
    fn default() -> Self {
        Backend::LlamaCpp
    }
}

/// Plan speculation for a specific backend.
///
/// Wraps [`spec_policy::plan`] with the one question that policy cannot answer
/// on its own: whether this engine is even in a position to speculate here. A
/// served backend runs its own decode loop, so the honest plan is
/// [`SpecMethod::None`](crate::spec_policy::SpecMethod::None) with a reason
/// saying why — not a block size the caller would have no way to apply.
#[must_use]
pub fn plan_for(
    backend: Backend,
    model: &ModelProfile,
    load: &LoadState,
    config: &crate::config::GenerationConfig,
    policy: &SpecPolicy,
) -> SpecPlan {
    if !backend.supports().engine_speculation {
        return SpecPlan::none("backend owns its own decode loop; speculation is not ours to do");
    }
    spec_policy::plan(model, load, config, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GenerationConfig;

    #[test]
    fn names_round_trip() {
        for b in Backend::all() {
            assert_eq!(Backend::parse(b.as_str()).unwrap(), *b);
        }
    }

    #[test]
    fn common_spellings_parse() {
        for (s, want) in [
            ("llama.cpp", Backend::LlamaCpp),
            ("llamacpp", Backend::LlamaCpp),
            ("LLAMA-CPP", Backend::LlamaCpp),
            ("  vLLM  ", Backend::Vllm),
            ("sglang", Backend::SgLang),
            ("SGL", Backend::SgLang),
        ] {
            assert_eq!(Backend::parse(s).unwrap(), want, "parsing {s:?}");
        }
    }

    #[test]
    fn an_unknown_backend_is_refused_and_lists_the_known_ones() {
        // Never silently fall back to a default: a typo must be visible, not
        // quietly served by a different runtime than the caller asked for.
        // "tensorrt" used to be the example here and is now a valid alias --
        // pick something that is not a backend at all.
        let e = Backend::parse("gpt4all").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("gpt4all"), "should name the bad input: {msg}");
        assert!(msg.contains("llama.cpp"), "should list what is valid: {msg}");
    }

    #[test]
    fn served_backends_do_not_offer_engine_speculation() {
        for b in [Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm, Backend::MlxLm] {
            let c = b.supports();
            assert_eq!(c.integration, Integration::Served);
            assert!(!c.engine_speculation, "{b} must not claim our speculation");
            assert!(!c.logit_access, "{b} cannot expose per-position logits over HTTP");
        }
    }

    #[test]
    fn served_backends_are_available() {
        for b in [Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm, Backend::MlxLm] {
            assert!(b.is_available(), "{b} has an adapter");
        }
    }

    #[test]
    fn mlx_reports_that_it_cannot_constrain_output() {
        // The trap this capability exists for: mlx_lm.server never reads
        // `response_format`, so a schema-constrained request silently returns
        // free text. A caller must be able to learn that before sending.
        assert!(!Backend::MlxLm.supports().structured_output);
        for b in [Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm] {
            assert!(b.supports().structured_output, "{b} does constrain output");
        }
    }

    #[test]
    fn planning_against_a_served_backend_never_speculates() {
        // The important case: policy alone would happily return a 4-token
        // block for an idle dense model. Against a backend we cannot interpose
        // on, that number is unusable and reporting it would be a lie.
        let p = plan_for(
            Backend::Vllm,
            &ModelProfile::dense(),
            &LoadState::idle(),
            &GenerationConfig { max_tokens: 256, ..Default::default() },
            &SpecPolicy::default(),
        );
        assert!(!p.speculates());
        assert_eq!(p.draft_n, 0);
        assert!(!p.reason.is_empty());
    }

    #[test]
    fn planning_against_llama_cpp_defers_to_policy() {
        let cfg = GenerationConfig { max_tokens: 256, ..Default::default() };
        let direct = spec_policy::plan(
            &ModelProfile::dense(),
            &LoadState::idle(),
            &cfg,
            &SpecPolicy::default(),
        );
        let viaback = plan_for(
            Backend::LlamaCpp,
            &ModelProfile::dense(),
            &LoadState::idle(),
            &cfg,
            &SpecPolicy::default(),
        );
        assert_eq!(direct, viaback, "the linked backend must not alter the policy decision");
    }

    #[test]
    fn the_default_backend_is_the_implemented_one() {
        assert_eq!(Backend::default(), Backend::LlamaCpp);
    }

    #[test]
    fn every_backend_reports_its_integration_kind() {
        // A backend added later must decide linked-vs-served deliberately,
        // because that is what determines whether acceleration applies at all.
        for b in Backend::all() {
            let c = b.supports();
            match c.integration {
                Integration::Linked => assert!(c.logit_access, "{b}: linked implies logit access"),
                Integration::Served => {
                    assert!(!c.engine_speculation, "{b}: served cannot host our speculation");
                }
            }
        }
    }
}
