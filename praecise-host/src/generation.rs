//! Text generation on a hosted engine, in the runtime's own types.
//!
//! [`generate`] turns a [`GenerationConfig`] and a prompt into the host
//! protocol's `generate` request, passes the streamed text to the caller, and
//! reads the reply back as an [`InferenceResult`], so a hosted engine answers
//! with the same types, token counts and stop reasons as the linked one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use praecise_runtime::{ChatMessage, GenerationConfig, InferenceResult, StopReason};
use serde_json::{Value, json};

use crate::engines::{HostedModel, Usage};
use crate::{Error, Result};

/// What to generate from.
#[derive(Debug, Clone, Copy)]
pub enum Prompt<'a> {
    /// Raw text, passed to the engine as it is.
    Text(&'a str),
    /// Chat turns, rendered by the engine with the model's chat template.
    Chat(&'a [ChatMessage]),
}

/// Generate on `model`, passing each streamed piece of text to `on_text`.
/// `on_text` returns false when its caller has gone, which abandons the
/// request. Blocking.
///
/// # Errors
/// A request for an inference commitment, which a hosted engine cannot
/// produce; the engine's refusal or exit; an abandoned request; a reply
/// without token usage.
pub fn generate(
    model: &Arc<HostedModel>,
    prompt: Prompt<'_>,
    config: &GenerationConfig,
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<InferenceResult> {
    let started = Instant::now();
    let request = request(prompt, config)?;
    let (reply, usage) = model.generate(&request, |event| {
        let text = event.get("text").and_then(Value::as_str).unwrap_or_default();
        if text.is_empty() || on_text(text) {
            Ok(())
        } else {
            Err(Error::Failed(format!("{}: the caller went away", model.model_id())))
        }
    })?;
    Ok(result(&reply, usage, config, started.elapsed()))
}

/// The protocol request for `prompt` under `config`.
fn request(prompt: Prompt<'_>, config: &GenerationConfig) -> Result<Value> {
    if config.commitment_k.is_some() {
        return Err(Error::Refused("a hosted engine does not produce inference commitments".into()));
    }
    let mut req = json!({
        "op": "generate",
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
    });
    match prompt {
        Prompt::Text(text) => req["prompt"] = text.into(),
        Prompt::Chat(messages) => req["messages"] = serde_json::to_value(messages)?,
    }
    if !config.stop.is_empty() {
        req["stop"] = serde_json::to_value(&config.stop)?;
    }
    Ok(req)
}

/// The reply read back as an [`InferenceResult`].
fn result(reply: &Value, usage: Usage, config: &GenerationConfig, elapsed: Duration) -> InferenceResult {
    let output_tokens = u32::try_from(usage.completion_tokens).unwrap_or(u32::MAX);
    let generation_time_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    let stop_reason = match reply.get("finish_reason").and_then(Value::as_str) {
        Some("length") => StopReason::Length,
        _ if output_tokens >= config.max_tokens => StopReason::Length,
        _ => StopReason::Eos,
    };
    InferenceResult {
        text: reply.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
        thinking: None,
        input_tokens: u32::try_from(usage.prompt_tokens).unwrap_or(u32::MAX),
        output_tokens,
        generation_time_ms,
        tokens_per_second: if elapsed.is_zero() { 0.0 } else { f64::from(output_tokens) / elapsed.as_secs_f64() },
        stop_reason,
        commitment: None,
        cached_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chat_request_carries_the_sampling_the_caller_asked_for() {
        let config = GenerationConfig {
            max_tokens: 16,
            temperature: 0.2,
            top_p: 0.9,
            seed: 7,
            stop: vec!["END".into()],
            ..Default::default()
        };
        let messages = [ChatMessage::new("user", "hi")];
        let req = request(Prompt::Chat(&messages), &config).unwrap();
        assert_eq!(req["op"], "generate");
        assert_eq!(req["messages"][0]["content"], "hi");
        assert_eq!((req["max_tokens"].as_u64(), req["seed"].as_u64()), (Some(16), Some(7)));
        assert!((req["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-9);
        assert_eq!(req["stop"][0], "END");
        assert!(req.get("prompt").is_none());
    }

    #[test]
    fn a_commitment_is_refused_rather_than_silently_missing() {
        let config = GenerationConfig { commitment_k: Some(4), ..Default::default() };
        assert!(request(Prompt::Text("x"), &config).is_err());
    }

    #[test]
    fn the_reply_reads_back_as_an_inference_result() {
        let config = GenerationConfig { max_tokens: 10, ..Default::default() };
        let usage = Usage { prompt_tokens: 4, completion_tokens: 10 };
        let r = result(&json!({"ok": true, "text": "abc", "finish_reason": "stop"}), usage, &config, Duration::from_millis(500));
        assert_eq!((r.text.as_str(), r.input_tokens, r.output_tokens), ("abc", 4, 10));
        assert_eq!(r.stop_reason, StopReason::Length, "a full budget is a length stop");
        assert!((r.tokens_per_second - 20.0).abs() < 1e-9);
        let usage = Usage { prompt_tokens: 4, completion_tokens: 3 };
        assert_eq!(result(&json!({"text": "abc", "finish_reason": "stop"}), usage, &config, Duration::ZERO).stop_reason, StopReason::Eos);
    }
}
