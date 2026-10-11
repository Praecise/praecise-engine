//! A chat turn rendered through the model's own template, and the reply read
//! back in the same format.
//!
//! Instruct models do not share one output syntax. A tool call is a JSON object
//! in `<tool_call>` tags, `<function=name><parameter=k>v</parameter>`,
//! `<arg_key>`/`<arg_value>` pairs, a `[TOOL_CALLS]` array, DeepSeek's
//! `<｜tool▁call▁begin｜>` blocks, Kimi's `<|tool_call_begin|>` sections,
//! MiniMax's `<invoke>` elements or a harmony `to=functions.name` channel;
//! reasoning arrives in `<think>` blocks, `<|channel|>analysis` segments or not
//! at all. The template that wrote the prompt is the one thing that knows which
//! of these the model will answer in, so the reply is read by the parser
//! llama.cpp's `common_chat` builds from that same template. Nothing here
//! branches on a model's name or family: a model is supported by its template.
//!
//! [`ChatTemplateSource`] is a template plus the token text it refers to,
//! taken from a loaded model or from a Hugging Face model directory.
//! [`ChatFormat::render`] turns a turn into a prompt (and, when tools are
//! offered, a lazy grammar for the calls), and [`ChatFormat::parse`] splits the
//! generated text into answer, reasoning and tool calls.

use std::path::Path;

use llama_cpp_2::model::{ChatTemplateResult, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::token::LlamaToken;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::result::{
    ChatMessage, ParsedReply, ParsedToolCall, ReplyDelta, call_arguments, new_call_id,
};
use crate::stream::{ReplyFormat, ReplyReader};

/// A chat template and the token text it refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateSource {
    template: String,
    bos_token: String,
    eos_token: String,
    tokenizer_adds_bos: bool,
}

impl ChatTemplateSource {
    /// A template given as source text. `tokenizer_adds_bos` says whether the
    /// tokenizer that will read the prompt prepends the beginning-of-sequence
    /// token itself, in which case a rendered prompt does not repeat it.
    #[must_use]
    pub fn new(
        template: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
        tokenizer_adds_bos: bool,
    ) -> Self {
        Self {
            template: template.into(),
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
            tokenizer_adds_bos,
        }
    }

    /// The template a GGUF model carries, with its vocabulary's special tokens.
    ///
    /// # Errors
    /// A model without a chat template.
    pub fn from_model(model: &LlamaModel) -> Result<Self> {
        let template = model
            .chat_template(None)
            .map_err(|e| Error::ChatFormat(format!("the model has no chat template: {e}")))?
            .to_string()
            .map_err(|e| Error::ChatFormat(format!("the chat template is not UTF-8: {e}")))?;
        // A vocabulary without the token reports a negative id.
        let token_text = |token: LlamaToken| {
            if token.0 < 0 {
                return String::new();
            }
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            model
                .token_to_piece(token, &mut decoder, true, None)
                .unwrap_or_default()
        };
        Ok(Self {
            template,
            bos_token: token_text(model.token_bos()),
            eos_token: token_text(model.token_eos()),
            tokenizer_adds_bos: model.vocab_adds_bos(),
        })
    }

    /// The template of a Hugging Face model directory: `chat_template.jinja`
    /// when present, else `chat_template` in `tokenizer_config.json`, with the
    /// special tokens and `add_bos_token` from that config.
    ///
    /// # Errors
    /// A directory with neither, or an unreadable config.
    pub fn from_hf_dir(dir: &Path) -> Result<Self> {
        let config: Value = match std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(e.into()),
        };
        let template = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                default_template(&config["chat_template"]).ok_or_else(|| {
                    Error::ChatFormat(format!("{} has no chat template", dir.display()))
                })?
            }
            Err(e) => return Err(e.into()),
        };
        Ok(Self {
            template,
            bos_token: token_field(&config["bos_token"]),
            eos_token: token_field(&config["eos_token"]),
            tokenizer_adds_bos: config["add_bos_token"].as_bool().unwrap_or(false),
        })
    }

    /// The template source text.
    #[must_use]
    pub fn template(&self) -> &str {
        &self.template
    }
}

/// `chat_template` is either one template or a list of named ones.
fn default_template(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(named) => named
            .iter()
            .find(|t| t["name"] == "default")
            .or_else(|| named.first())
            .and_then(|t| t["template"].as_str())
            .map(str::to_string),
        _ => None,
    }
}

/// A special token is a string or an object with a `content` string.
fn token_field(value: &Value) -> String {
    value
        .as_str()
        .or_else(|| value["content"].as_str())
        .unwrap_or_default()
        .to_string()
}

/// One turn to render.
#[derive(Debug, Clone, Copy)]
pub struct ChatTurn<'a> {
    /// The conversation so far, in the OpenAI message shape.
    pub messages: &'a [ChatMessage],
    /// The tools offered, as an OpenAI tool array. `None` or an empty array
    /// offers none, and the reply is not searched for calls.
    pub tools: Option<&'a Value>,
    /// The template's reasoning switch, for templates that have one.
    pub enable_thinking: bool,
    /// Extra template variables, as a JSON object (a reasoning budget, for
    /// example).
    pub template_kwargs: Option<&'a Value>,
    /// Allow more than one tool call in a reply.
    pub parallel_tool_calls: bool,
    /// Render the template as plain Jinja and read the reply as plain text,
    /// for a template whose format `common_chat` identifies wrongly.
    pub force_pure_content: bool,
    /// End with the prompt that opens the model's reply. Off renders the
    /// conversation as history, as the next turn's prompt begins.
    pub add_generation_prompt: bool,
}

impl<'a> ChatTurn<'a> {
    /// A turn with reasoning on, no tools and no extra variables.
    #[must_use]
    pub fn new(messages: &'a [ChatMessage]) -> Self {
        Self {
            messages,
            tools: None,
            enable_thinking: true,
            template_kwargs: None,
            parallel_tool_calls: false,
            force_pure_content: false,
            add_generation_prompt: true,
        }
    }
}

/// A rendered turn: the prompt, the grammar for the tools it offers, and the
/// parser for the reply.
#[derive(Debug)]
pub struct ChatFormat {
    prompt: String,
    render: ChatTemplateResult,
}

impl ChatFormat {
    /// Render `turn` through `source`.
    ///
    /// # Errors
    /// A template that fails to render, or renders nothing.
    pub fn render(source: &ChatTemplateSource, turn: &ChatTurn<'_>) -> Result<Self> {
        let messages_json = serde_json::to_string(turn.messages)?;
        let tools_json = match turn.tools {
            Some(Value::Array(tools)) if !tools.is_empty() => Some(serde_json::to_string(tools)?),
            _ => None,
        };
        let kwargs_json = turn
            .template_kwargs
            .map(serde_json::to_string)
            .transpose()?;
        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json: tools_json.as_deref(),
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: Some("auto"),
            chat_template_kwargs: kwargs_json.as_deref(),
            add_generation_prompt: turn.add_generation_prompt,
            use_jinja: true,
            parallel_tool_calls: turn.parallel_tool_calls,
            enable_thinking: turn.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: tools_json.is_some(),
            force_pure_content: turn.force_pure_content,
        };
        let render = llama_cpp_2::model::apply_chat_template_source_oaicompat(
            &source.template,
            &source.bos_token,
            &source.eos_token,
            &params,
        )
        .map_err(|e| Error::ChatFormat(format!("rendering the chat template: {e}")))?;
        // A tokenizer that prepends the beginning-of-sequence token would
        // otherwise see it twice.
        let prompt = match render.prompt.strip_prefix(source.bos_token.as_str()) {
            Some(rest) if source.tokenizer_adds_bos && !source.bos_token.is_empty() => {
                rest.to_string()
            }
            _ => render.prompt.clone(),
        };
        if prompt.trim().is_empty() {
            return Err(Error::ChatFormat(
                "the chat template rendered an empty prompt".into(),
            ));
        }
        Ok(Self { prompt, render })
    }

    /// The prompt to tokenize.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// The full render: the tool grammar and its lazy triggers, the tokens the
    /// grammar needs kept whole, and stop strings the template adds.
    #[must_use]
    pub fn template_result(&self) -> &ChatTemplateResult {
        &self.render
    }

    /// Read a complete reply.
    ///
    /// # Errors
    /// Text the format's parser cannot read.
    pub fn parse(&self, text: &str) -> Result<ParsedReply> {
        self.parse_with(text, false)
    }

    /// Read a reply still being generated: a call whose arguments are not yet
    /// complete is returned with the arguments so far.
    ///
    /// # Errors
    /// As [`parse`](Self::parse).
    pub fn parse_partial(&self, text: &str) -> Result<ParsedReply> {
        self.parse_with(text, true)
    }

    /// A reader for the reply as it is generated, piece by piece.
    ///
    /// # Errors
    /// A parser the backend cannot load.
    pub fn stream(&self) -> Result<ChatStream> {
        let state = self
            .render
            .streaming_state_oaicompat()
            .map_err(|e| Error::ChatFormat(format!("starting the reply reader: {e}")))?;
        Ok(ChatStream { state })
    }

    fn parse_with(&self, text: &str, partial: bool) -> Result<ParsedReply> {
        let json = self
            .render
            .parse_response_oaicompat(text, partial)
            .map_err(|e| Error::ChatFormat(format!("reading the reply: {e}")))?;
        ParsedReply::from_oaicompat(&serde_json::from_str(&json)?)
    }
}

impl ParsedReply {
    pub(crate) fn from_oaicompat(msg: &Value) -> Result<Self> {
        let content = match &msg["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .concat(),
            _ => String::new(),
        };
        let reasoning = msg["reasoning_content"]
            .as_str()
            .filter(|r| !r.trim().is_empty())
            .map(str::to_string);
        let mut tool_calls = Vec::new();
        for call in msg["tool_calls"].as_array().into_iter().flatten() {
            let function = &call["function"];
            let name = function["name"]
                .as_str()
                .ok_or_else(|| Error::ChatFormat("a tool call without a name".into()))?
                .to_string();
            let arguments = match &function["arguments"] {
                Value::String(s) => call_arguments(s),
                Value::Null => Value::Object(Default::default()),
                other => other.clone(),
            };
            tool_calls.push(ParsedToolCall {
                id: call["id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .map_or_else(new_call_id, str::to_string),
                name,
                arguments,
            });
        }
        Ok(Self {
            content,
            reasoning,
            tool_calls,
        })
    }
}

/// Reads a reply as it is generated. Each piece of generated text yields the
/// parts of the reply it completes: answer text, reasoning text, or a step of
/// a tool call. Markup never reaches the answer or the reasoning.
pub struct ChatStream {
    state: llama_cpp_2::openai::ChatParseStateOaicompat,
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatStream").finish_non_exhaustive()
    }
}

impl ChatStream {
    /// Read the next piece of generated text.
    ///
    /// # Errors
    /// Text the format's parser cannot read.
    pub fn push(&mut self, piece: &str) -> Result<Vec<ReplyDelta>> {
        self.update(piece, true)
    }

    /// The reply is complete: read whatever the last pieces left pending.
    ///
    /// # Errors
    /// As [`push`](Self::push).
    pub fn finish(&mut self) -> Result<Vec<ReplyDelta>> {
        self.update("", false)
    }

    fn update(&mut self, piece: &str, partial: bool) -> Result<Vec<ReplyDelta>> {
        let diffs = self
            .state
            .update(piece, partial)
            .map_err(|e| Error::ChatFormat(format!("reading the reply: {e}")))?;
        let mut deltas = Vec::new();
        for diff in diffs {
            let diff: Value = serde_json::from_str(&diff)?;
            if let Some(r) = diff["reasoning_content"].as_str() {
                deltas.push(ReplyDelta::Reasoning(r.to_string()));
            }
            if let Some(c) = diff["content"].as_str() {
                deltas.push(ReplyDelta::Content(c.to_string()));
            }
            for call in diff["tool_calls"].as_array().into_iter().flatten() {
                deltas.push(ReplyDelta::ToolCall {
                    index: call["index"].as_u64().unwrap_or_default() as usize,
                    id: call["id"].as_str().map(str::to_string),
                    name: call["function"]["name"].as_str().map(str::to_string),
                    arguments: call["function"]["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                });
            }
        }
        Ok(deltas)
    }
}

impl ReplyFormat for ChatFormat {
    fn reader(&self) -> Result<Box<dyn ReplyReader>> {
        Ok(Box::new(self.stream()?))
    }

    fn reasoning_end(&self) -> Option<&str> {
        self.render.thinking_end_tag.as_deref()
    }
}

impl ReplyReader for ChatStream {
    fn read(&mut self, text: &str) -> Result<Vec<ReplyDelta>> {
        self.push(text)
    }

    fn finish(&mut self) -> Result<Vec<ReplyDelta>> {
        ChatStream::finish(self)
    }
}
