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
//! - **Hosted** ([`Integration::Hosted`]) — the runtime runs in a runtime host:
//!   a confined child process the application starts and supervises, spoken
//!   to over its standard streams (the `praecise-host` crate). The runtime
//!   owns its own decode loop. The engine chooses the model and the sampling
//!   parameters, but it cannot interpose on token-by-token decoding. vLLM,
//!   SGLang, TensorRT-LLM and transformers are hosted.
//!
//! That distinction is load-bearing. Against a hosted backend, this engine's
//! speculation is **not** available — the runtime does its own, with its own
//! drafters. Reporting a speculation plan for such a backend would be a lie
//! the caller could not detect, so [`Backend::supports`] answers it up front
//! and [`plan_for`] refuses rather than pretends.
//!
//! A backend that cannot run here is refused with
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
    /// llama.cpp, linked and driven through FFI.
    LlamaCpp,
    /// vLLM, hosted through its offline engine API.
    Vllm,
    /// SGLang, hosted through its offline engine API.
    SgLang,
    /// TensorRT-LLM, hosted through its LLM API.
    TensorRtLlm,
    /// Hugging Face transformers, hosted; runs on the CPU as well as the GPU.
    Transformers,
}

/// How the engine reaches a backend — see the module docs on why this decides
/// what acceleration is possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integration {
    /// Compiled in and called through FFI; the engine owns the decode loop.
    Linked,
    /// A confined child process the application supervises; the runtime owns
    /// its own decode loop.
    Hosted,
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
    /// for hosted backends — they speculate internally, which is not the same
    /// thing and must not be counted as ours.
    pub engine_speculation: bool,
    /// Whether per-position logits are visible, which speculation verification
    /// requires and which no hosted runtime exposes.
    pub logit_access: bool,
    /// Whether grammar-constrained decoding is available.
    pub structured_output: bool,
    /// Whether the backend can run in this build on this platform.
    pub implemented: bool,
}

impl Backend {
    /// Every backend the engine knows about.
    #[must_use]
    pub fn all() -> &'static [Backend] {
        &[Backend::LlamaCpp, Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm, Backend::Transformers]
    }

    /// Stable identifier used in configuration, logs and engine claims.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::LlamaCpp => "llama.cpp",
            Backend::Vllm => "vllm",
            Backend::SgLang => "sglang",
            Backend::TensorRtLlm => "tensorrt-llm",
            Backend::Transformers => "transformers",
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
            "transformers" | "hf" => Ok(Backend::Transformers),
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
            // A hosted runtime receives a prompt and sampling parameters and
            // streams text back. Nothing reaches inside its decode loop, so
            // engine-side speculation is impossible by construction, and the
            // host protocol carries no schema, so output is not constrained.
            Backend::Vllm | Backend::SgLang | Backend::TensorRtLlm | Backend::Transformers => Capabilities {
                integration: Integration::Hosted,
                engine_speculation: false,
                logit_access: false,
                structured_output: false,
                implemented: cfg!(target_os = "linux"),
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
    /// reads differently from a platform that cannot confine a runtime host,
    /// and a caller deserves to know which.
    pub fn ensure_available(self) -> Result<()> {
        if self.is_available() {
            return Ok(());
        }
        let reason = match self.supports().integration {
            Integration::Linked => {
                "the `bundled-llama` feature is not enabled; build with it, or pass in a \
                 backend the host application already links"
            }
            Integration::Hosted => "runtime hosts need Linux namespaces, Landlock and seccomp",
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
    /// llama.cpp: the linked backend the acceleration paths are written
    /// against.
    fn default() -> Self {
        Backend::LlamaCpp
    }
}

/// Plan speculation for a specific backend.
///
/// Wraps [`spec_policy::plan`] with the one question that policy cannot answer
/// on its own: whether this engine is even in a position to speculate here. A
/// hosted backend runs its own decode loop, so the honest plan is
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

    const HOSTED: [Backend; 4] = [Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm, Backend::Transformers];

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
            ("trtllm", Backend::TensorRtLlm),
            ("HF", Backend::Transformers),
        ] {
            assert_eq!(Backend::parse(s).unwrap(), want, "parsing {s:?}");
        }
    }

    #[test]
    fn an_unknown_backend_is_refused_and_lists_the_known_ones() {
        // Never silently fall back to a default: a typo must be visible, not
        // quietly served by a different runtime than the caller asked for.
        let e = Backend::parse("gpt4all").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("gpt4all"), "should name the bad input: {msg}");
        assert!(msg.contains("llama.cpp"), "should list what is valid: {msg}");
    }

    #[test]
    fn hosted_backends_do_not_offer_engine_speculation() {
        for b in HOSTED {
            let c = b.supports();
            assert_eq!(c.integration, Integration::Hosted);
            assert!(!c.engine_speculation, "{b} must not claim our speculation");
            assert!(!c.logit_access, "{b} exposes no per-position logits through a host");
            assert!(!c.structured_output, "the host protocol carries no schema");
        }
    }

    #[test]
    fn hosted_backends_run_where_hosts_can_be_confined() {
        for b in HOSTED {
            assert_eq!(b.is_available(), cfg!(target_os = "linux"), "{b}");
            if !cfg!(target_os = "linux") {
                assert!(b.ensure_available().unwrap_err().to_string().contains("Landlock"));
            }
        }
    }

    #[test]
    fn planning_against_a_hosted_backend_never_speculates() {
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
    fn the_default_backend_is_the_linked_one() {
        assert_eq!(Backend::default(), Backend::LlamaCpp);
    }

    #[test]
    fn every_backend_reports_its_integration_kind() {
        // A backend added later must decide linked-vs-hosted deliberately,
        // because that is what determines whether acceleration applies at all.
        for b in Backend::all() {
            let c = b.supports();
            match c.integration {
                Integration::Linked => assert!(c.logit_access, "{b}: linked implies logit access"),
                Integration::Hosted => {
                    assert!(!c.engine_speculation, "{b}: hosted cannot host our speculation");
                }
            }
        }
    }
}
