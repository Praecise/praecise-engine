//! Inference result and stop-reason types.

use serde::{Deserialize, Serialize};

/// Why generation stopped. The text alone cannot distinguish a trimmed stop
/// sequence from an end-of-generation token, so this reports the cause beside
/// the token counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model emitted an end-of-generation token.
    Eos,
    /// The token budget in [`crate::config::GenerationConfig::max_tokens`] was
    /// exhausted.
    Length,
    /// Decoded text ended with one of the configured stop sequences.
    StopSequence,
}

impl StopReason {
    /// Termination cause for a loop that ran to completion: a stop sequence
    /// wins over an exhausted budget, since the sequence is what halted
    /// decoding.
    pub fn from_loop(hit_stop: bool, output_tokens: u32, max_tokens: u32) -> Self {
        if hit_stop {
            Self::StopSequence
        } else if output_tokens >= max_tokens {
            Self::Length
        } else {
            Self::Eos
        }
    }

    /// Wire spelling, matching the serde representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::Length => "length",
            Self::StopSequence => "stop_sequence",
        }
    }
}

/// Result from running inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// What the caller is meant to see. Reasoning spans are already removed, so
    /// a caller never has to strip `<think>` itself.
    pub text: String,
    /// The model's reasoning, when it produced any. Separate from [`Self::text`]
    /// so a surface can render, collapse or drop it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Prompt token count.
    pub input_tokens: u32,
    /// Tokens the model generated. Tokens the engine forced into the output
    /// (the close marker of a reasoning block whose budget ran out) are not
    /// counted, though they appear in the text and, when a commitment was
    /// requested, in its steps.
    pub output_tokens: u32,
    /// Wall-clock generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Decode throughput in tokens per second.
    pub tokens_per_second: f64,
    /// Engine-observed termination cause.
    pub stop_reason: StopReason,
    /// Verifiable-inference commitment, present when the request set
    /// [`crate::config::GenerationConfig::commitment_k`]. Carries the per-step
    /// top-k logit records a verifier can recompute against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commitment: Option<crate::toploc::InferenceCommitment>,
    /// Prompt tokens served from the engine's prefix cache instead of being
    /// prefilled: already counted in [`Self::input_tokens`], and computed by an
    /// earlier request.
    pub cached_tokens: u32,
    /// Positions the model ran forward, when that is not the prompt once and
    /// each output token once: a diffusion model reads its canvas once per
    /// denoising step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_positions: Option<u64>,
    /// Denoising steps a diffusion model took.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denoise_steps: Option<u32>,
    /// Tool calls the model made, read in the chat format of the turn it was
    /// answering. Empty when the generation had no chat format or made none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ParsedToolCall>,
}

/// A reply split into its parts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParsedReply {
    /// The answer, without reasoning or call markup.
    pub content: String,
    /// The reasoning, when the model wrote any.
    pub reasoning: Option<String>,
    /// The tool calls, in the order written.
    pub tool_calls: Vec<ParsedToolCall>,
}

/// One tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    /// The id the model gave the call, or a generated one.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments. An object for a complete call; the argument text so far
    /// when it is not valid JSON (a reply cut short).
    pub arguments: serde_json::Value,
}

/// One step of a reply read as it is generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplyDelta {
    /// More of the answer.
    Content(String),
    /// More of the reasoning.
    Reasoning(String),
    /// A step of the tool call at `index`: its id and name when first seen,
    /// then argument text as it arrives.
    ToolCall {
        /// The call's position in the reply.
        index: usize,
        /// The call's id, on the step that starts it.
        id: Option<String>,
        /// The tool's name, on the step that names it.
        name: Option<String>,
        /// Argument text added by this step.
        arguments: String,
    },
}

/// A fresh tool-call id, for a call the model did not name.
pub(crate) fn new_call_id() -> String {
    let mut bytes = [0u8; 12];
    // An id only has to be unique within a conversation; on the rare
    // platform without an entropy source the zero id is still well formed.
    let _ = getrandom::getrandom(&mut bytes);
    format!("call_{}", hex::encode(bytes))
}

/// Call arguments as written: a JSON value when they parse, else the text.
pub(crate) fn call_arguments(text: &str) -> serde_json::Value {
    if text.trim().is_empty() {
        return serde_json::Value::Object(Default::default());
    }
    serde_json::from_str(text).unwrap_or_else(|_| serde_json::Value::String(text.to_string()))
}

/// A chat message with role and content, for chat-template formatting.
///
/// The tool fields carry a turn's tool structure to templates that render tool
/// history themselves. A model whose template has tool branches — muse-glimmer
/// renders an assistant tool call as `<|start|>assistant to=NAME<|message|>`
/// plus ATEM markup, and a result as `<|start|>tool NAME<|message|><tool_output
/// …>` — must be shown its own dialect, because it will imitate whatever its
/// history shows it. Flattening tool history into text taught it a foreign
/// dialect, which it then reproduced instead of calling a tool: measured,
/// giving it `<tool_use …>` XML made it emit `<tool_use …>` XML, and giving it
/// prose made it emit prose.
///
/// Every field is optional and skipped when absent, so a template that has no
/// tool branches sees exactly what it saw before.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role (`system`, `user`, `assistant`, `tool`, …).
    pub role: String,
    /// Message content.
    pub content: String,
    /// Tool calls this assistant turn made, in the OpenAI shape templates
    /// expect: `[{id, type: "function", function: {name, arguments}}]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// For `role == "tool"`: the tool that produced this result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// For `role == "tool"`: the call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For `role == "assistant"`: the reasoning the model produced before this
    /// reply, when the client sends it back. Templates that render reasoning
    /// into history (Qwen's read `reasoning_content`) then reproduce the
    /// previous turn exactly as it was generated, so the engine's cache of that
    /// turn is an exact prefix of the next request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// A plain turn with no tool structure — the shape every caller used before
    /// the tool fields existed.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }
}
