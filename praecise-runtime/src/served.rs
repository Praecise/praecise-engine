//! Adapters for backends reached over HTTP.
//!
//! vLLM, SGLang, TensorRT-LLM and MLX all expose an OpenAI-compatible surface,
//! which makes one client shape workable — but only if the differences between
//! them are handled explicitly. They are not interchangeable: the same concept
//! has different field names, some parameters live outside the standard body,
//! and a field that works on one server is silently ignored by another.
//!
//! Silently ignored is the danger. An unknown key in a JSON body does not
//! error; it is dropped, and the request runs with different sampling than the
//! caller asked for. So this module translates deliberately per backend rather
//! than sending one body everywhere and hoping.
//!
//! ## What a served backend can and cannot do
//!
//! It owns its own decode loop, so this engine cannot interpose on
//! token-by-token generation — no engine-side speculative decoding, no
//! per-position logit inspection. Those runtimes do their own speculation
//! internally, and theirs is better than ours would be over a socket.
//!
//! Everything else the layer does still applies: model selection, request
//! shaping, batch composition, drafter licence checks, structured-output
//! routing. That is the majority of the surface, and it is why a served
//! adapter is worth having rather than dismissing as "not really integrated".
//!
//! ## Transport is deliberately absent
//!
//! This module builds and parses requests; it does not send them. The caller
//! supplies the HTTP client. That keeps the crate free of a networking
//! dependency, keeps this testable without a server, and lets a host reuse the
//! connection pool, timeouts, retries and tracing it already has — which for
//! anything running in production it does.

use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::config::GenerationConfig;
use crate::error::{Error, Result};

/// Where a served backend is reachable, and how to address it.
#[derive(Clone, Debug)]
pub struct Endpoint {
    /// Base URL, without a trailing slash — e.g. `http://127.0.0.1:8000`.
    pub base_url: String,
    /// Model identifier the server expects. Required by the OpenAI schema even
    /// where a server has only one model loaded and ignores the value.
    pub model: String,
    /// Bearer token, where the server requires one. Most local deployments do
    /// not, so this is optional rather than an empty-string convention.
    pub api_key: Option<String>,
}

impl Endpoint {
    /// An endpoint with no authentication, the usual local case.
    #[must_use]
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let mut base = base_url.into();
        while base.ends_with('/') {
            base.pop();
        }
        Self { base_url: base, model: model.into(), api_key: None }
    }

    /// Add a bearer token.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Full URL for a path such as `/v1/completions`.
    #[must_use]
    pub fn url(&self, path: &str) -> String {
        if path.starts_with('/') {
            format!("{}{path}", self.base_url)
        } else {
            format!("{}/{path}", self.base_url)
        }
    }
}

/// One HTTP request, ready for a caller's client to send.
///
/// Returned rather than sent, so this crate needs no HTTP dependency and the
/// host keeps control of timeouts, retries and connection reuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: &'static str,
    pub url: String,
    /// Headers as `(name, value)`. `Authorization` appears only when the
    /// endpoint carries a key.
    pub headers: Vec<(String, String)>,
    /// JSON body, or `None` for a GET.
    pub body: Option<String>,
}

/// What a completion returned, normalised across servers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Server-reported reason, verbatim. Not mapped to an enum: servers use
    /// values outside the OpenAI set, and silently folding an unrecognised one
    /// into `Stop` would hide a truncation.
    pub finish_reason: Option<String>,
}

/// How each server wants non-standard sampling parameters passed.
///
/// `top_k`, `min_p` and `repetition_penalty` are not in the OpenAI schema, and
/// the servers disagree about where they belong. Sending them in the wrong
/// place is silent: the key is dropped and generation proceeds with different
/// sampling than was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtraParams {
    /// Accepted at the top level of the request body alongside the standard
    /// fields.
    TopLevel,
    /// Must be nested under a named object.
    Nested(&'static str),
    /// The server takes only the standard OpenAI fields; anything else is
    /// dropped. A caller asking for one gets an error rather than silence.
    Unsupported,
}

/// Per-backend translation rules.
///
/// Deliberately data rather than a trait implementation per backend: the
/// differences are a handful of names and one placement rule, and a table makes
/// them auditable side by side instead of scattered across four files.
#[derive(Clone, Copy, Debug)]
pub struct Dialect {
    pub backend: Backend,
    /// Path for text completion.
    pub completions_path: &'static str,
    /// Path for chat completion.
    pub chat_path: &'static str,
    /// Path listing available models.
    pub models_path: &'static str,
    /// Health or readiness path, where one exists.
    pub health_path: Option<&'static str>,
    /// Prometheus metrics path, where one exists.
    pub metrics_path: Option<&'static str>,
    /// Where non-standard sampling parameters go.
    pub extra_params: ExtraParams,
    /// Field name for a JSON-schema constraint on the response.
    pub structured_field: Option<&'static str>,
}

impl Dialect {
    /// The dialect for a backend.
    ///
    /// # Errors
    /// [`Error::BackendUnavailable`] for a backend that is not reached over
    /// HTTP — llama.cpp is linked, and asking for its HTTP dialect is a
    /// programming error rather than a configuration one.
    pub fn for_backend(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Vllm => Ok(Self {
                backend,
                completions_path: "/v1/completions",
                chat_path: "/v1/chat/completions",
                models_path: "/v1/models",
                health_path: Some("/health"),
                metrics_path: Some("/metrics"),
                // vLLM accepts non-standard sampling at the top level of the
                // body; OpenAI *client libraries* have to smuggle them through
                // `extra_body`, but that is a client concern, not the wire.
                extra_params: ExtraParams::TopLevel,
                // `guided_json` and friends were removed in v0.12.0. The
                // replacement object accepts exactly ONE of json / regex /
                // choice / grammar / json_object / structural_tag -- zero or
                // two is a hard validation error, not a preference.
                structured_field: Some("structured_outputs"),
            }),
            Backend::SgLang => Ok(Self {
                backend,
                completions_path: "/v1/completions",
                chat_path: "/v1/chat/completions",
                models_path: "/v1/models",
                // `/health` is liveness only; `/health_generate` actually runs
                // a generation and is the real readiness check.
                health_path: Some("/health"),
                // Present, but only with `--enable-metrics`. Off by default, so
                // a scraper cannot assume it.
                metrics_path: Some("/metrics"),
                // SGLang wants non-standard sampling nested under `extra_body`,
                // unlike vLLM which takes it flat. Sending it flat here is
                // silently dropped.
                extra_params: ExtraParams::Nested("extra_body"),
                structured_field: Some("response_format"),
            }),
            Backend::TensorRtLlm => Ok(Self {
                backend,
                completions_path: "/v1/completions",
                chat_path: "/v1/chat/completions",
                models_path: "/v1/models",
                health_path: Some("/health"),
                metrics_path: Some("/metrics"),
                extra_params: ExtraParams::TopLevel,
                structured_field: Some("response_format"),
            }),
            Backend::MlxLm => Ok(Self {
                backend,
                completions_path: "/v1/completions",
                chat_path: "/v1/chat/completions",
                models_path: "/v1/models",
                health_path: Some("/health"),
                // No metrics endpoint exists at all.
                metrics_path: None,
                extra_params: ExtraParams::TopLevel,
                // `mlx_lm.server` never reads `response_format`. Declaring a
                // field here would let a caller send a schema that is silently
                // ignored, which is worse than refusing.
                structured_field: None,
            }),
            Backend::LlamaCpp => Err(Error::BackendUnavailable {
                backend: backend.as_str(),
                reason: "llama.cpp is linked, not served; it has no HTTP dialect",
            }),
        }
    }
}

/// Build a completion request for a served backend.
///
/// # Errors
/// [`Error::BackendUnavailable`] if the backend is not served, or if the caller
/// asked for a parameter the target does not accept — refused rather than
/// dropped, because a silently ignored sampling parameter produces plausible
/// output from the wrong distribution.
pub fn completion_request(
    dialect: &Dialect,
    endpoint: &Endpoint,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<HttpRequest> {
    let mut body = serde_json::json!({
        "model": endpoint.model,
        "prompt": prompt,
        // MLX checks `max_completion_tokens` first and falls back to this;
        // every other server reads `max_tokens`. Sending both is harmless and
        // avoids a per-backend branch for one field.
        "max_tokens": config.max_tokens,
        "max_completion_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "stream": false,
    });

    if !config.stop.is_empty() {
        body["stop"] = serde_json::json!(config.stop);
    }

    // Non-standard sampling. Collected first so placement is decided once.
    let mut extra = serde_json::Map::new();
    if let Some(top_k) = config.top_k {
        extra.insert("top_k".into(), serde_json::json!(top_k));
    }
    if let Some(min_p) = config.min_p {
        extra.insert("min_p".into(), serde_json::json!(min_p));
    }
    if (config.repeat_penalty - 1.0).abs() > f32::EPSILON {
        extra.insert("repetition_penalty".into(), serde_json::json!(config.repeat_penalty));
    }

    if !extra.is_empty() {
        match dialect.extra_params {
            ExtraParams::TopLevel => {
                for (k, v) in extra {
                    body[k] = v;
                }
            }
            ExtraParams::Nested(field) => {
                body[field] = serde_json::Value::Object(extra);
            }
            ExtraParams::Unsupported => {
                return Err(Error::BackendUnavailable {
                    backend: dialect.backend.as_str(),
                    reason: "this backend accepts only standard OpenAI sampling parameters; \
                             top_k, min_p and repetition_penalty would be silently dropped",
                });
            }
        }
    }

    Ok(HttpRequest {
        method: "POST",
        url: endpoint.url(dialect.completions_path),
        headers: headers_for(endpoint),
        body: Some(body.to_string()),
    })
}

/// Build a request listing the server's models.
#[must_use]
pub fn models_request(dialect: &Dialect, endpoint: &Endpoint) -> HttpRequest {
    HttpRequest {
        method: "GET",
        url: endpoint.url(dialect.models_path),
        headers: headers_for(endpoint),
        body: None,
    }
}

/// Build a health-check request, where the server has one.
#[must_use]
pub fn health_request(dialect: &Dialect, endpoint: &Endpoint) -> Option<HttpRequest> {
    dialect.health_path.map(|p| HttpRequest {
        method: "GET",
        url: endpoint.url(p),
        headers: headers_for(endpoint),
        body: None,
    })
}

fn headers_for(endpoint: &Endpoint) -> Vec<(String, String)> {
    let mut h = vec![("Content-Type".to_string(), "application/json".to_string())];
    if let Some(key) = &endpoint.api_key {
        h.push(("Authorization".to_string(), format!("Bearer {key}")));
    }
    h
}

/// Minimal shape of an OpenAI-style completion response.
///
/// Only the fields this crate uses are modelled, and every one is optional:
/// servers differ in what they populate, and a missing `usage` block should
/// yield zero counts rather than a parse failure that loses the generated text.
#[derive(Debug, Deserialize, Serialize)]
struct RawCompletion {
    #[serde(default)]
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Option<RawUsage>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct RawChoice {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    message: Option<RawMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct RawMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct RawUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Parse a completion response.
///
/// Handles both the text-completion shape (`choices[].text`) and the chat shape
/// (`choices[].message.content`), since a caller may hit either path and the
/// difference is not worth pushing onto them.
///
/// # Errors
/// [`Error::Inference`] if the body is not JSON or carries no choices — an
/// empty `choices` array is a server-side failure, not an empty completion, and
/// returning empty text would hide it.
pub fn parse_completion(body: &str) -> Result<Completion> {
    let raw: RawCompletion = serde_json::from_str(body)
        .map_err(|e| Error::Inference(format!("malformed completion response: {e}")))?;

    let choice = raw
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Inference("completion response carried no choices".into()))?;

    let text = choice
        .text
        .or_else(|| choice.message.and_then(|m| m.content))
        .unwrap_or_default();

    let usage = raw.usage.unwrap_or_default();
    Ok(Completion {
        text,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        finish_reason: choice.finish_reason,
    })
}

/// Model ids from a `/v1/models` response.
///
/// # Errors
/// [`Error::Inference`] if the body is not the expected shape.
pub fn parse_models(body: &str) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Resp {
        #[serde(default)]
        data: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        id: String,
    }
    let r: Resp = serde_json::from_str(body)
        .map_err(|e| Error::Inference(format!("malformed models response: {e}")))?;
    Ok(r.data.into_iter().map(|e| e.id).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep() -> Endpoint {
        Endpoint::new("http://127.0.0.1:8000", "test-model")
    }

    fn cfg() -> GenerationConfig {
        GenerationConfig { max_tokens: 128, temperature: 0.7, ..Default::default() }
    }

    #[test]
    fn a_trailing_slash_does_not_produce_a_double_slash() {
        let e = Endpoint::new("http://host:8000/", "m");
        assert_eq!(e.url("/v1/models"), "http://host:8000/v1/models");
    }

    #[test]
    fn a_path_without_a_leading_slash_still_joins_correctly() {
        assert_eq!(ep().url("v1/models"), "http://127.0.0.1:8000/v1/models");
    }

    #[test]
    fn llama_cpp_has_no_http_dialect() {
        // It is linked, not served. Asking for its dialect is a programming
        // error and should say so rather than inventing endpoints.
        let e = Dialect::for_backend(Backend::LlamaCpp).unwrap_err();
        assert!(e.to_string().contains("linked"), "{e}");
    }

    #[test]
    fn served_backends_have_dialects() {
        for b in [Backend::Vllm, Backend::SgLang] {
            let d = Dialect::for_backend(b).expect("served backend has a dialect");
            assert_eq!(d.backend, b);
            assert!(d.completions_path.starts_with('/'));
        }
    }

    #[test]
    fn sglang_nests_extras_where_vllm_sends_them_flat() {
        // Verified against both servers: sending SGLang a flat `top_k` drops it
        // silently, and generation proceeds with different sampling than asked.
        let c = GenerationConfig { top_k: Some(40), ..cfg() };

        let v = Dialect::for_backend(Backend::Vllm).unwrap();
        let vb: serde_json::Value =
            serde_json::from_str(completion_request(&v, &ep(), "x", &c).unwrap().body.as_ref().unwrap()).unwrap();
        assert_eq!(vb["top_k"], 40, "vLLM takes it flat");

        let g = Dialect::for_backend(Backend::SgLang).unwrap();
        let gb: serde_json::Value =
            serde_json::from_str(completion_request(&g, &ep(), "x", &c).unwrap().body.as_ref().unwrap()).unwrap();
        assert_eq!(gb["extra_body"]["top_k"], 40, "SGLang wants it nested");
        assert!(gb.get("top_k").is_none(), "and must not also be flat");
    }

    #[test]
    fn mlx_declares_no_structured_output() {
        // The silent-degradation trap: mlx_lm.server never reads
        // `response_format`, so a schema-constrained request returns free text
        // with no error. Declaring None is what lets a caller find out first.
        assert!(Dialect::for_backend(Backend::MlxLm).unwrap().structured_field.is_none());
        for b in [Backend::Vllm, Backend::SgLang, Backend::TensorRtLlm] {
            assert!(Dialect::for_backend(b).unwrap().structured_field.is_some(), "{b}");
        }
    }

    #[test]
    fn mlx_has_no_metrics_endpoint() {
        assert!(Dialect::for_backend(Backend::MlxLm).unwrap().metrics_path.is_none());
    }

    #[test]
    fn every_served_backend_has_a_dialect() {
        for b in Backend::all() {
            let served = b.supports().integration == crate::backend::Integration::Served;
            assert_eq!(
                Dialect::for_backend(*b).is_ok(),
                served,
                "{b}: a served backend needs a dialect and a linked one must not haveated"
            );
        }
    }

    #[test]
    fn max_tokens_is_sent_under_both_names() {
        // MLX reads `max_completion_tokens` first; the rest read `max_tokens`.
        let d = Dialect::for_backend(Backend::MlxLm).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(completion_request(&d, &ep(), "x", &cfg()).unwrap().body.as_ref().unwrap()).unwrap();
        assert_eq!(b["max_tokens"], 128);
        assert_eq!(b["max_completion_tokens"], 128);
    }

    #[test]
    fn a_completion_request_carries_the_standard_fields() {
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let r = completion_request(&d, &ep(), "hello", &cfg()).unwrap();
        assert_eq!(r.method, "POST");
        assert_eq!(r.url, "http://127.0.0.1:8000/v1/completions");

        let b: serde_json::Value = serde_json::from_str(r.body.as_ref().unwrap()).unwrap();
        assert_eq!(b["model"], "test-model");
        assert_eq!(b["prompt"], "hello");
        assert_eq!(b["max_tokens"], 128);
        assert_eq!(b["stream"], false);
    }

    #[test]
    fn non_standard_sampling_lands_where_the_server_expects_it() {
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let c = GenerationConfig { top_k: Some(40), min_p: Some(0.05), ..cfg() };
        let r = completion_request(&d, &ep(), "x", &c).unwrap();
        let b: serde_json::Value = serde_json::from_str(r.body.as_ref().unwrap()).unwrap();
        assert_eq!(b["top_k"], 40, "vLLM takes top_k at the top level");
        assert_eq!(b["min_p"], 0.05);
    }

    #[test]
    fn a_neutral_penalty_is_not_sent() {
        // Sending repetition_penalty=1.0 asks for nothing while committing the
        // server to its penalty path. See `crate::shaping`.
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let c = GenerationConfig { repeat_penalty: 1.0, ..cfg() };
        let r = completion_request(&d, &ep(), "x", &c).unwrap();
        let b: serde_json::Value = serde_json::from_str(r.body.as_ref().unwrap()).unwrap();
        assert!(b.get("repetition_penalty").is_none());

        // The default is 1.1, so it IS sent -- worth pinning, since a reader
        // may assume the default is neutral.
        let d2 = completion_request(&d, &ep(), "x", &cfg()).unwrap();
        let b2: serde_json::Value = serde_json::from_str(d2.body.as_ref().unwrap()).unwrap();
        assert!(b2.get("repetition_penalty").is_some());
    }

    #[test]
    fn a_real_penalty_is_sent() {
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let c = GenerationConfig { repeat_penalty: 1.15, ..cfg() };
        let r = completion_request(&d, &ep(), "x", &c).unwrap();
        let b: serde_json::Value = serde_json::from_str(r.body.as_ref().unwrap()).unwrap();
        assert!(b["repetition_penalty"].as_f64().unwrap() > 1.0);
    }

    #[test]
    fn stop_sequences_are_sent_only_when_present() {
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let bare = completion_request(&d, &ep(), "x", &cfg()).unwrap();
        let b: serde_json::Value = serde_json::from_str(bare.body.as_ref().unwrap()).unwrap();
        assert!(b.get("stop").is_none(), "an empty stop list should be omitted");

        let c = GenerationConfig { stop: vec!["\n\n".into()], ..cfg() };
        let with = completion_request(&d, &ep(), "x", &c).unwrap();
        let b2: serde_json::Value = serde_json::from_str(with.body.as_ref().unwrap()).unwrap();
        assert_eq!(b2["stop"][0], "\n\n");
    }

    #[test]
    fn an_unsupported_extra_parameter_is_refused_not_dropped() {
        // The failure this module exists to prevent: an unknown JSON key is
        // ignored by the server, so the request runs with different sampling
        // than the caller asked for and nothing reports it.
        let d = Dialect { extra_params: ExtraParams::Unsupported, ..Dialect::for_backend(Backend::Vllm).unwrap() };
        let c = GenerationConfig { top_k: Some(40), ..cfg() };
        let e = completion_request(&d, &ep(), "x", &c).unwrap_err();
        assert!(e.to_string().contains("silently dropped"), "{e}");
    }

    #[test]
    fn an_unsupported_backend_still_accepts_a_genuinely_plain_request() {
        // Refusal must be limited to what is actually unsupported. Note
        // `GenerationConfig` defaults `repeat_penalty` to **1.1**, not 1.0, so
        // a default config is NOT parameter-free -- it carries a real penalty
        // and is correctly refused by a server that cannot express one.
        let d = Dialect { extra_params: ExtraParams::Unsupported, ..Dialect::for_backend(Backend::Vllm).unwrap() };
        let plain = GenerationConfig { repeat_penalty: 1.0, ..cfg() };
        assert!(completion_request(&d, &ep(), "x", &plain).is_ok());
        // ...and the default, which is not plain, is refused.
        assert!(completion_request(&d, &ep(), "x", &cfg()).is_err());
    }

    #[test]
    fn nested_extra_parameters_are_grouped_under_their_field() {
        let d = Dialect { extra_params: ExtraParams::Nested("sampling"), ..Dialect::for_backend(Backend::Vllm).unwrap() };
        let c = GenerationConfig { top_k: Some(7), ..cfg() };
        let r = completion_request(&d, &ep(), "x", &c).unwrap();
        let b: serde_json::Value = serde_json::from_str(r.body.as_ref().unwrap()).unwrap();
        assert_eq!(b["sampling"]["top_k"], 7);
        assert!(b.get("top_k").is_none(), "must not also appear at the top level");
    }

    #[test]
    fn an_api_key_becomes_a_bearer_header() {
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let e = ep().with_api_key("sk-test");
        let r = completion_request(&d, &e, "x", &cfg()).unwrap();
        assert!(r.headers.iter().any(|(k, v)| k == "Authorization" && v == "Bearer sk-test"));
    }

    #[test]
    fn no_api_key_means_no_authorization_header() {
        let d = Dialect::for_backend(Backend::Vllm).unwrap();
        let r = completion_request(&d, &ep(), "x", &cfg()).unwrap();
        assert!(!r.headers.iter().any(|(k, _)| k == "Authorization"));
    }

    #[test]
    fn a_text_completion_response_parses() {
        let c = parse_completion(
            r#"{"choices":[{"text":"hi there","finish_reason":"length"}],
                "usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
        )
        .unwrap();
        assert_eq!(c.text, "hi there");
        assert_eq!(c.prompt_tokens, 5);
        assert_eq!(c.completion_tokens, 2);
        assert_eq!(c.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn a_chat_completion_response_parses_too() {
        // Same function, different shape -- a caller should not have to know
        // which endpoint produced the body.
        let c = parse_completion(
            r#"{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        assert_eq!(c.text, "hello");
    }

    #[test]
    fn a_missing_usage_block_yields_zero_counts_not_an_error() {
        // Losing the generated text over absent bookkeeping would be a poor
        // trade; servers differ in what they populate.
        let c = parse_completion(r#"{"choices":[{"text":"ok"}]}"#).unwrap();
        assert_eq!(c.text, "ok");
        assert_eq!(c.prompt_tokens, 0);
    }

    #[test]
    fn an_empty_choices_array_is_an_error_not_empty_text() {
        // A server returning no choices has failed; reporting empty text would
        // present that as a successful empty completion.
        assert!(parse_completion(r#"{"choices":[]}"#).is_err());
    }

    #[test]
    fn a_malformed_body_is_an_error() {
        assert!(parse_completion("not json").is_err());
        assert!(parse_completion("").is_err());
    }

    #[test]
    fn an_unrecognised_finish_reason_is_preserved_verbatim() {
        // Servers use values outside the OpenAI set; folding one into `stop`
        // would hide a truncation.
        let c = parse_completion(
            r#"{"choices":[{"text":"x","finish_reason":"abort_by_watchdog"}]}"#,
        )
        .unwrap();
        assert_eq!(c.finish_reason.as_deref(), Some("abort_by_watchdog"));
    }

    #[test]
    fn models_parse_into_ids() {
        let ids = parse_models(r#"{"object":"list","data":[{"id":"a"},{"id":"b"}]}"#).unwrap();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn an_empty_model_list_is_not_an_error() {
        // A server with nothing loaded is a valid state, unlike a completion
        // with no choices.
        assert!(parse_models(r#"{"data":[]}"#).unwrap().is_empty());
    }

    #[test]
    fn health_and_models_requests_are_gets_with_no_body() {
        let d = Dialect::for_backend(Backend::SgLang).unwrap();
        let m = models_request(&d, &ep());
        assert_eq!(m.method, "GET");
        assert!(m.body.is_none());

        let h = health_request(&d, &ep()).expect("sglang exposes health");
        assert_eq!(h.method, "GET");
        assert!(h.body.is_none());
    }
}
