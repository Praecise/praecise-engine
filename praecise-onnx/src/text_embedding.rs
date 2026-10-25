//! Text embedding on ONNX Runtime, and the request types every embedder
//! shares: [`EmbedInput`] (text, pictures, video), [`TextEmbedConfig`], the
//! [`TextEncoder`] trait and the query/document [`EmbeddingTemplate`]s.
//!
//! # Scope
//!
//! Foundation text encoders in 2026 share the `[B, L]` int64 token tensor
//! + `[B, L]` int64 attention mask interface, but disagree on:
//!
//! - **Pooling**: decoder-only Qwen3-Embedding takes the *last* non-pad
//!   token; bidirectional BGE-M3 takes the CLS (first) token; some
//!   exports (EmbeddingGemma) ship a pre-pooled `sentence_embedding`
//!   output and need no pooling at all.
//! - **Padding side**: Qwen3-Embedding is a causal LM repurposed for
//!   embedding — it needs left-padding so the last token is always real.
//!   Bidirectional encoders use right-padding.
//! - **Output naming**: `sentence_embedding` (Optimum exports of
//!   sentence-transformers checkpoints), `last_hidden_state`
//!   (transformers.js exports), or the only output (legacy single-output
//!   exports). The runtime prefers `sentence_embedding` when present and
//!   otherwise pools `last_hidden_state` per-family.
//! - **Matryoshka**: EmbeddingGemma supports 768 → 512 → 256 → 128.
//!   The runtime truncates and re-normalizes after pooling when
//!   `requested_dim` is smaller than the native dim.
//!
//! # Threading
//!
//! Models are `Send + Sync`. ORT sessions are not safe to call
//! concurrently, so each session lives behind a `parking_lot::Mutex`.
//! Inference is synchronous; async callers run it on a blocking thread.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// How an embedding family expects queries and documents to be prefixed
/// before tokenization.
///
/// Embedding families disagree on instruction formatting the way chat
/// families disagree on chat templates; the template is data handed to the
/// encoder at load, never a per-model branch in the encoder itself. Each
/// variant follows the family's own model card:
///
/// - Qwen3-Embedding: an unprefixed query embeds noticeably worse on
///   retrieval (`Instruct: …\nQuery:…` for queries; documents take no
///   prefix).
/// - EmbeddingGemma: queries and documents both take a prefix, two
///   different ones; mixing them up degrades retrieval without an error.
/// - BGE-M3: no instruction for either role.
/// - ModernBERT-embed and other Nomic-lineage encoders: the
///   `search_query:` / `search_document:` prefix pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmbeddingTemplate {
    /// No prefix for either role. BGE-M3.
    None,
    /// Qwen3-Embedding: query → `"Instruct: {task}\nQuery:{text}"` (no
    /// space after the colon, as in the model's own `get_detailed_instruct`
    /// example); document → no prefix. `default_task` is used when a
    /// request does not supply one.
    InstructQuery { default_task: String },
    /// EmbeddingGemma: query → `"task: {task} | query: {text}"`;
    /// document → `"title: {title|none} | text: {text}"`. `default_task`
    /// is used when a request does not supply one.
    GemmaTask { default_task: String },
    /// Nomic-lineage (ModernBERT-embed, nomic-embed-*): query →
    /// `"search_query: {text}"`; document → `"search_document: {text}"`.
    NomicPrefix,
    /// Query/passage prefixes (Nemotron embed): query → `"query: {text}"`;
    /// document → `"passage: {text}"`.
    QueryPassagePrefix,
    /// Chat-formatted instruction (decoder embedders on a GGUF engine): the
    /// instruction is the system turn and the input the user turn of the
    /// checkpoint's own chat template, which the engine applies. Query → the
    /// caller's task or `default_task`; document → `document_task`.
    ChatInstruct {
        default_task: String,
        document_task: String,
    },
}

/// Which role a batch of inputs plays, for families whose template
/// distinguishes queries from documents (see [`EmbeddingTemplate`]).
///
/// Defaults to `Document` rather than `Query`: a caller that forgets to
/// set this on an asymmetric family (Qwen3-Embedding, EmbeddingGemma) gets
/// the document-side treatment, which is a same-quality-or-slightly-lower
/// retrieval hit — not the wrong-instruction corruption that defaulting to
/// `Query` would silently apply to every document a caller embeds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum EmbedRole {
    /// Being indexed for later retrieval. `title` is only consumed by
    /// families whose template wants one (EmbeddingGemma); ignored
    /// elsewhere.
    Document {
        #[serde(default)]
        title: Option<String>,
    },
    /// Being embedded to search an existing index. `task` is only
    /// consumed by families whose template wants one (Qwen3-Embedding,
    /// EmbeddingGemma) — falls back to the template's own
    /// `default_task` when not set. Ignored on families with no
    /// instruction template (BGE-M3) or a fixed one (ModernBERT).
    Query {
        #[serde(default)]
        task: Option<String>,
    },
}

impl Default for EmbedRole {
    fn default() -> Self {
        Self::Document { title: None }
    }
}

/// Apply a family's [`EmbeddingTemplate`] to one input string for the given
/// role. Pure data-in-data-out — the encoder calls this once per input
/// before tokenization; no model-name branching lives here or anywhere
/// downstream of it.
pub fn apply_embedding_template(
    template: &EmbeddingTemplate,
    role: &EmbedRole,
    text: &str,
) -> String {
    match (template, role) {
        (EmbeddingTemplate::None, _) => text.to_string(),
        (EmbeddingTemplate::InstructQuery { .. }, EmbedRole::Document { .. }) => text.to_string(),
        (EmbeddingTemplate::InstructQuery { default_task }, EmbedRole::Query { task }) => {
            let task = task.as_deref().unwrap_or(default_task);
            // Verified byte-for-byte against Qwen3-Embedding's own
            // `get_detailed_instruct` example: no space after "Query:".
            format!("Instruct: {}\nQuery:{}", task, text)
        }
        (EmbeddingTemplate::GemmaTask { .. }, EmbedRole::Document { title }) => {
            format!(
                "title: {} | text: {}",
                title.as_deref().unwrap_or("none"),
                text
            )
        }
        (EmbeddingTemplate::GemmaTask { default_task }, EmbedRole::Query { task }) => {
            let task = task.as_deref().unwrap_or(default_task);
            format!("task: {} | query: {}", task, text)
        }
        (EmbeddingTemplate::NomicPrefix, EmbedRole::Document { .. }) => {
            format!("search_document: {}", text)
        }
        (EmbeddingTemplate::NomicPrefix, EmbedRole::Query { .. }) => {
            format!("search_query: {}", text)
        }
        (EmbeddingTemplate::QueryPassagePrefix, EmbedRole::Document { .. }) => {
            format!("passage: {}", text)
        }
        (EmbeddingTemplate::QueryPassagePrefix, EmbedRole::Query { .. }) => {
            format!("query: {}", text)
        }
        // The engine writes the instruction into the checkpoint's own chat
        // template (see `chat_instruction`); the input goes in as it is.
        (EmbeddingTemplate::ChatInstruct { .. }, _) => text.to_string(),
    }
}

/// The instruction a chat-instruct template gives the engine for `role`:
/// the caller's task or the template's default for a query, the document
/// task for a document. `None` for templates applied as a text prefix.
pub fn chat_instruction<'a>(
    template: &'a EmbeddingTemplate,
    role: &'a EmbedRole,
) -> Option<&'a str> {
    match (template, role) {
        (EmbeddingTemplate::ChatInstruct { document_task, .. }, EmbedRole::Document { .. }) => {
            Some(document_task.as_str())
        }
        (EmbeddingTemplate::ChatInstruct { default_task, .. }, EmbedRole::Query { task }) => {
            Some(task.as_deref().unwrap_or(default_task))
        }
        _ => None,
    }
}

/// One input to embed: text, a picture, a video (as frames), or any of
/// them together, which the model embeds as one item. On the wire a plain
/// string is text; otherwise an object with `text`, `image` (base64 bytes of
/// a PNG, JPEG or WebP picture) and `video` (base64 frames, in order).
/// Pictures and video are served only by vision-language embedders.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbedInput {
    pub text: Option<String>,
    pub image: Option<Vec<u8>>,
    pub video: Option<Vec<Vec<u8>>>,
}

impl EmbedInput {
    /// A text-only input.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::default()
        }
    }

    /// Whether the input carries a picture or a video.
    pub fn has_media(&self) -> bool {
        self.image.is_some() || self.video.is_some()
    }

    /// Bytes the caller submitted (text, picture and frame bytes).
    pub fn byte_len(&self) -> u64 {
        let text = self.text.as_ref().map_or(0, |t| t.len());
        let image = self.image.as_ref().map_or(0, |b| b.len());
        let video: usize = self.video.iter().flatten().map(|f| f.len()).sum();
        (text + image + video) as u64
    }
}

impl From<String> for EmbedInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for EmbedInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbedInputWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    video: Option<Vec<String>>,
}

impl Serialize for EmbedInput {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use base64::Engine as _;
        let b64 = |b: &Vec<u8>| base64::engine::general_purpose::STANDARD.encode(b);
        EmbedInputWire {
            text: self.text.clone(),
            image: self.image.as_ref().map(b64),
            video: self.video.as_ref().map(|v| v.iter().map(b64).collect()),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EmbedInput {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        use base64::Engine as _;
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Text(String),
            Parts(EmbedInputWire),
        }
        let decode = |field: &str, s: &str| {
            base64::engine::general_purpose::STANDARD
                .decode(s)
                .map_err(|e| D::Error::custom(format!("'{field}' is not base64: {e}")))
        };
        match Wire::deserialize(deserializer).map_err(|_| {
            D::Error::custom(
                "an input is a string or an object with 'text', 'image' (base64) and/or 'video' (base64 frames)",
            )
        })? {
            Wire::Text(text) => Ok(Self::text(text)),
            Wire::Parts(w) => {
                let image = w.image.as_deref().map(|s| decode("image", s)).transpose()?;
                let video = w
                    .video
                    .map(|frames| frames.iter().map(|f| decode("video", f)).collect::<std::result::Result<Vec<_>, _>>())
                    .transpose()?;
                Ok(Self {
                    text: w.text,
                    image,
                    video,
                })
            }
        }
    }
}

/// Configuration for a text-embedding request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TextEmbedConfig {
    /// Optional Matryoshka dimension. Must be `<= embedding_dim()`.
    /// Models that support MRL list valid dims in their
    /// `matryoshka_dims` field; the runtime accepts any value
    /// `<= embedding_dim` and truncates+renormalizes.
    #[serde(default)]
    pub requested_dim: Option<u32>,
    /// L2-normalize the output embedding. Most retrieval pipelines want
    /// this on; downstream classification heads want it off.
    #[serde(default)]
    pub normalize: bool,
    /// Query vs. document role, for families whose template distinguishes
    /// them. See [`EmbedRole`].
    #[serde(default)]
    pub role: EmbedRole,
}

/// Result of a text-embedding call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedResult {
    /// Embedding vectors, one per input string. Outer length =
    /// `inputs.len()`, inner length = effective embedding dim
    /// (post-Matryoshka if requested).
    pub embeddings: Vec<Vec<f32>>,
    /// Effective embedding dim used (after Matryoshka truncation).
    pub dim: usize,
    /// Tokens consumed across the batch, excluding padding. Batching pads every
    /// input up to the longest one in the batch; those pad positions are an
    /// artifact of how the tensor is shaped, not work the caller asked for, so
    /// this counts attention-mask positions rather than `batch × seq_len`.
    pub prompt_tokens: u32,
    /// Total inference wall time in milliseconds (tokenize + ORT run).
    pub generation_time_ms: u64,
}

/// Family of text encoder. Drives padding side and pooling strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TextEncoderFamily {
    /// Decoder-only causal LM repurposed as an embedding model
    /// (Qwen3-Embedding). Left-padding, last-token pooling.
    Qwen3,
    /// Bidirectional encoder, CLS pooling (BGE-M3, Snowflake Arctic
    /// Embed L v2.0 use [CLS] at position 0).
    Cls,
    /// Bidirectional encoder, mean pooling over non-pad positions
    /// (ModernBERT-embed base/large).
    Mean,
    /// Pre-pooled `sentence_embedding` output (EmbeddingGemma's Optimum
    /// export). The runtime asserts the `sentence_embedding` output is
    /// present at load time when this family is chosen.
    SentenceEmbedding,
}

/// Trait for text encoders. Implementations adapt model-specific
/// tokenization and pooling to a common signature.
pub trait TextEncoder: Send + Sync {
    /// Encode a batch of inputs to dense vectors, one per input. Encoders
    /// without a vision tower refuse inputs that carry a picture or video.
    fn embed(&self, inputs: &[EmbedInput], config: &TextEmbedConfig) -> Result<TextEmbedResult>;

    /// Native embedding dimension (before any Matryoshka truncation).
    fn embedding_dim(&self) -> usize;

    /// Maximum sequence length the tokenizer will produce.
    fn max_sequence_length(&self) -> usize;
}

/// Stub text encoder. Returns `ProviderNotAvailable` from every method.
mod onnx_backend {
    use super::*;
    use ndarray::{Array2, Array4};
    use ort::session::Session;
    use ort::value::{Tensor, ValueType};
    use std::time::Instant;
    use tokenizers::{
        PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    };

    /// ORT-backed text encoder covering Qwen3-Embedding, BGE-M3,
    /// Snowflake Arctic Embed, and EmbeddingGemma. Pooling and padding
    /// side are picked per-family at load time.
    ///
    /// `Session::run` requires `&mut self` in ort 2.x, so the session is
    /// wrapped in a `Mutex` to expose a `&self` API. ONNX Runtime sessions
    /// are not safe to call concurrently from multiple threads regardless,
    /// so the mutex matches the underlying contract rather than restricting it.
    pub struct GenericTextEncoder {
        session: parking_lot::Mutex<Session>,
        tokenizer: Tokenizer,
        family: TextEncoderFamily,
        embedding_template: EmbeddingTemplate,
        embedding_dim: usize,
        max_sequence_length: usize,
        /// Cached input names. Most exports use `input_ids` /
        /// `attention_mask`, but a few rename them — captured at load.
        input_ids_name: String,
        attention_mask_name: String,
        /// Optional `position_ids` input — required by some Qwen3 exports.
        position_ids_name: Option<String>,
        /// Optional `token_type_ids` input — required by some BERT-style
        /// exports (BGE-M3 declares it but accepts all-zeros).
        token_type_ids_name: Option<String>,
        /// Output name to read. Either `sentence_embedding` (pre-pooled)
        /// or `last_hidden_state` (needs pooling).
        output_name: String,
        /// Whether `output_name` is `sentence_embedding` (already pooled).
        output_is_pooled: bool,
        /// `(key_name, value_name)` per layer, when the graph declares
        /// `past_key_values.{i}.key`/`.value` inputs. Some decoder-repurposed
        /// embedding exports (Qwen3-Embedding) keep these even though
        /// embedding only ever does a single forward pass with no prior
        /// cache — ONNX Runtime still requires every declared input to be
        /// fed, so an empty (`past_sequence_length = 0`) tensor is supplied
        /// for each at inference time. Empty when the graph has none.
        past_kv_names: Vec<(String, String)>,
        /// `(num_kv_heads, head_dim)` for the empty past-KV tensors above,
        /// read off the graph's own declared static shape — never guessed
        /// or hardcoded to one model. `None` iff `past_kv_names` is empty.
        kv_shape: Option<(usize, usize)>,
    }

    impl GenericTextEncoder {
        /// Load an ONNX file and a tokenizer.json file. The family hint
        /// drives padding side + pooling strategy. The runtime introspects
        /// the session's input/output names and picks the right
        /// `sentence_embedding`/`last_hidden_state` output.
        pub fn from_onnx(
            onnx_path: impl AsRef<Path>,
            tokenizer_path: impl AsRef<Path>,
            family: TextEncoderFamily,
            embedding_template: EmbeddingTemplate,
            embedding_dim: usize,
            max_sequence_length: usize,
        ) -> Result<Self> {
            if matches!(embedding_template, EmbeddingTemplate::ChatInstruct { .. }) {
                return Err(Error::InvalidModel(
                    "a chat-instruct template is written by the Praecise engine, not an ONNX encoder".to_string(),
                ));
            }
            let session = crate::session::build_onnx_session(onnx_path.as_ref(), "model")?;

            // Introspect inputs.
            let input_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
            let input_ids_name = pick_input(&input_names, &["input_ids"]).ok_or_else(|| {
                Error::InvalidModel("ONNX model is missing 'input_ids' input".to_string())
            })?;
            let attention_mask_name =
                pick_input(&input_names, &["attention_mask"]).ok_or_else(|| {
                    Error::InvalidModel("ONNX model is missing 'attention_mask' input".to_string())
                })?;
            let position_ids_name = pick_input(&input_names, &["position_ids"]);
            let token_type_ids_name = pick_input(&input_names, &["token_type_ids"]);

            // Detect an optional per-layer KV-cache input pair. Generic over
            // however many layers the graph declares — found by name
            // pattern, not hardcoded to one model's layer count.
            let mut past_kv_names: Vec<(String, String)> = Vec::new();
            let mut kv_shape: Option<(usize, usize)> = None;
            let mut layer = 0usize;
            loop {
                let key_name = format!("past_key_values.{layer}.key");
                let value_name = format!("past_key_values.{layer}.value");
                if !input_names.contains(&key_name) || !input_names.contains(&value_name) {
                    break;
                }
                if kv_shape.is_none() {
                    let key_input = session
                        .inputs()
                        .iter()
                        .find(|i| i.name() == key_name)
                        .expect("just matched by name");
                    if let ValueType::Tensor { shape, .. } = &key_input.dtype()
                        && shape.len() == 4
                        && shape[1] > 0
                        && shape[3] > 0
                    {
                        kv_shape = Some((shape[1] as usize, shape[3] as usize));
                    }
                }
                past_kv_names.push((key_name, value_name));
                layer += 1;
            }
            if !past_kv_names.is_empty() && kv_shape.is_none() {
                return Err(Error::InvalidModel(format!(
                    "ONNX model declares {} past_key_values input(s) but their static \
                     (num_kv_heads, head_dim) shape could not be read",
                    past_kv_names.len()
                )));
            }

            // Introspect outputs. Prefer `sentence_embedding` when the
            // family asks for it; otherwise read `last_hidden_state` and
            // pool here.
            let output_names: Vec<String> =
                session.outputs().iter().map(|o| o.name().to_string()).collect();
            let (output_name, output_is_pooled) = match family {
                TextEncoderFamily::SentenceEmbedding => {
                    let n =
                        pick_input(&output_names, &["sentence_embedding"]).ok_or_else(|| {
                            Error::InvalidModel(
                                "SentenceEmbedding family requires a 'sentence_embedding' output"
                                    .to_string(),
                            )
                        })?;
                    (n, true)
                }
                _ => {
                    // Prefer last_hidden_state; fall back to first output.
                    let n = pick_input(&output_names, &["last_hidden_state", "hidden_states"])
                        .or_else(|| output_names.first().cloned())
                        .ok_or_else(|| {
                            Error::InvalidModel("ONNX model has no outputs".to_string())
                        })?;
                    (n, false)
                }
            };

            // Load + configure tokenizer.
            let mut tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
                .map_err(|e| Error::InvalidModel(format!("tokenizer load: {}", e)))?;

            // Padding side per family.
            let padding_direction = match family {
                TextEncoderFamily::Qwen3 => PaddingDirection::Left,
                _ => PaddingDirection::Right,
            };
            let pad_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);
            let pad_token = tokenizer
                .get_padding()
                .map(|p| p.pad_token.clone())
                .unwrap_or_else(|| "[PAD]".to_string());
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: padding_direction,
                pad_to_multiple_of: None,
                pad_id,
                pad_type_id: 0,
                pad_token,
            }));
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: max_sequence_length,
                    ..Default::default()
                }))
                .map_err(|e| Error::InvalidModel(format!("tokenizer truncation: {}", e)))?;

            Ok(Self {
                session: parking_lot::Mutex::new(session),
                tokenizer,
                family,
                embedding_template,
                embedding_dim,
                max_sequence_length,
                input_ids_name,
                attention_mask_name,
                position_ids_name,
                token_type_ids_name,
                output_name,
                output_is_pooled,
                past_kv_names,
                kv_shape,
            })
        }

        /// Tokenize a batch and return `(input_ids, attention_mask)`
        /// tensors as `[B, L]` int64 ndarrays.
        fn tokenize(&self, inputs: &[String]) -> Result<(Array2<i64>, Array2<i64>)> {
            let encodings = self
                .tokenizer
                .encode_batch(inputs.to_vec(), true)
                .map_err(|e| Error::InferenceError(format!("tokenize: {}", e)))?;

            let batch_size = encodings.len();
            let seq_len = encodings
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0);

            let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
            let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));

            for (b, enc) in encodings.iter().enumerate() {
                for (i, (id, m)) in enc
                    .get_ids()
                    .iter()
                    .zip(enc.get_attention_mask())
                    .enumerate()
                {
                    input_ids[[b, i]] = *id as i64;
                    attention_mask[[b, i]] = *m as i64;
                }
            }
            Ok((input_ids, attention_mask))
        }
    }

    impl TextEncoder for GenericTextEncoder {
        fn embed(
            &self,
            inputs: &[EmbedInput],
            config: &TextEmbedConfig,
        ) -> Result<TextEmbedResult> {
            if inputs.is_empty() {
                return Ok(TextEmbedResult {
                    embeddings: vec![],
                    dim: 0,
                    prompt_tokens: 0,
                    generation_time_ms: 0,
                });
            }
            let start = Instant::now();
            if inputs.iter().any(EmbedInput::has_media) {
                return Err(Error::InvalidInput(
                    "this model embeds text only; pictures and video need a vision-language embedder"
                        .to_string(),
                ));
            }
            let templated: Vec<String> = inputs
                .iter()
                .map(|input| {
                    apply_embedding_template(
                        &self.embedding_template,
                        &config.role,
                        input.text.as_deref().unwrap_or(""),
                    )
                })
                .collect();
            let (input_ids, attention_mask) = self.tokenize(&templated)?;
            let (batch_size, seq_len) = (input_ids.shape()[0], input_ids.shape()[1]);
            let prompt_tokens = attention_mask.iter().filter(|m| **m != 0).count() as u32;

            // Build extra optional inputs before the lock.
            let position_ids: Option<Array2<i64>> = if self.position_ids_name.is_some() {
                let mut pids = Array2::<i64>::zeros((batch_size, seq_len));
                for b in 0..batch_size {
                    for i in 0..seq_len {
                        pids[[b, i]] = i as i64;
                    }
                }
                Some(pids)
            } else {
                None
            };
            let token_type_ids: Option<Array2<i64>> = if self.token_type_ids_name.is_some() {
                Some(Array2::<i64>::zeros((batch_size, seq_len)))
            } else {
                None
            };
            // Empty (`past_sequence_length = 0`) KV-cache tensors, one pair
            // per declared `past_key_values.{i}` layer — see the field docs
            // on `past_kv_names`/`kv_shape`. Every layer shares the same
            // shape, so build it once and clone per layer.
            let empty_past_kv: Option<Array4<f32>> =
                self.kv_shape.map(|(num_kv_heads, head_dim)| {
                    Array4::<f32>::zeros((batch_size, num_kv_heads, 0, head_dim))
                });

            // input_ids is consumed by the tensor; attention_mask must be
            // cloned because the pooling step below still needs it.
            let input_ids_tensor = Tensor::from_array(input_ids)
                .map_err(|e| Error::InferenceError(format!("ORT tensor input_ids: {}", e)))?;
            let attention_mask_tensor = Tensor::from_array(attention_mask.clone())
                .map_err(|e| Error::InferenceError(format!("ORT tensor attention_mask: {}", e)))?;

            // Build the input feed dynamically — we may have 2, 3, or 4
            // inputs depending on whether the export declares position_ids
            // and/or token_type_ids. `Vec<(K, V)>` where K: Into<Cow<str>>
            // and V: Into<SessionInputValue> auto-converts to SessionInputs
            // (see ort::session::input::SessionInputs::From<Vec<...>>).
            let (out_shape, out_data): (Vec<i64>, Vec<f32>) = {
                let mut session = self.session.lock();
                let mut feed: Vec<(String, ort::session::SessionInputValue<'_>)> =
                    Vec::with_capacity(4 + self.past_kv_names.len() * 2);
                feed.push((self.input_ids_name.clone(), input_ids_tensor.into()));
                feed.push((
                    self.attention_mask_name.clone(),
                    attention_mask_tensor.into(),
                ));
                if let (Some(name), Some(arr)) = (&self.position_ids_name, position_ids) {
                    let t = Tensor::from_array(arr).map_err(|e| {
                        Error::InferenceError(format!("ORT tensor position_ids: {}", e))
                    })?;
                    feed.push((name.clone(), t.into()));
                }
                if let (Some(name), Some(arr)) = (&self.token_type_ids_name, token_type_ids) {
                    let t = Tensor::from_array(arr).map_err(|e| {
                        Error::InferenceError(format!("ORT tensor token_type_ids: {}", e))
                    })?;
                    feed.push((name.clone(), t.into()));
                }
                if let Some(empty) = &empty_past_kv {
                    for (key_name, value_name) in &self.past_kv_names {
                        let key_t = Tensor::from_array(empty.clone()).map_err(|e| {
                            Error::InferenceError(format!("ORT tensor {key_name}: {e}"))
                        })?;
                        let value_t = Tensor::from_array(empty.clone()).map_err(|e| {
                            Error::InferenceError(format!("ORT tensor {value_name}: {e}"))
                        })?;
                        feed.push((key_name.clone(), key_t.into()));
                        feed.push((value_name.clone(), value_t.into()));
                    }
                }

                let outputs = session
                    .run(feed)
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;

                let out_value = outputs.get(self.output_name.as_str()).ok_or_else(|| {
                    Error::InferenceError(format!(
                        "missing output '{}' in run result",
                        self.output_name
                    ))
                })?;
                let (shape, data) = out_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract: {}", e)))?;
                (shape.iter().copied().collect(), data.to_vec())
            };

            // Pool to per-row [D] vectors. Two cases:
            //  * already pooled: shape is [B, D] — split into rows.
            //  * needs pooling: shape is [B, L, D] — apply family pool.
            let mut embeddings: Vec<Vec<f32>> = if self.output_is_pooled {
                if out_shape.len() != 2 {
                    return Err(Error::InferenceError(format!(
                        "expected pooled output shape [B, D], got {:?}",
                        out_shape
                    )));
                }
                let b = out_shape[0] as usize;
                let d = out_shape[1] as usize;
                if b != batch_size {
                    return Err(Error::InferenceError(format!(
                        "batch mismatch: expected {}, got {}",
                        batch_size, b
                    )));
                }
                (0..b)
                    .map(|row| out_data[row * d..(row + 1) * d].to_vec())
                    .collect()
            } else {
                if out_shape.len() != 3 {
                    return Err(Error::InferenceError(format!(
                        "expected hidden-state shape [B, L, D], got {:?}",
                        out_shape
                    )));
                }
                let b = out_shape[0] as usize;
                let l = out_shape[1] as usize;
                let d = out_shape[2] as usize;
                if b != batch_size || l != seq_len {
                    return Err(Error::InferenceError(format!(
                        "shape mismatch: expected [{}, {}, D], got [{}, {}, {}]",
                        batch_size, seq_len, b, l, d
                    )));
                }
                pool_hidden_state(&out_data, &attention_mask, b, l, d, self.family)
            };

            // Matryoshka: truncate then renormalize when smaller dim
            // is requested.
            let mut effective_dim = self.embedding_dim;
            if let Some(req) = config.requested_dim {
                let req = req as usize;
                if req == 0 || req > self.embedding_dim {
                    return Err(Error::InvalidModel(format!(
                        "requested_dim {} out of range (1..={})",
                        req, self.embedding_dim
                    )));
                }
                if req < self.embedding_dim {
                    for v in embeddings.iter_mut() {
                        v.truncate(req);
                    }
                    effective_dim = req;
                }
            }

            // L2 normalize. Always on for Matryoshka (truncation
            // breaks norm); optional otherwise.
            let must_normalize = config.normalize || config.requested_dim.is_some();
            if must_normalize {
                for v in embeddings.iter_mut() {
                    l2_normalize(v);
                }
            }

            Ok(TextEmbedResult {
                embeddings,
                dim: effective_dim,
                prompt_tokens,
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }

        fn embedding_dim(&self) -> usize {
            self.embedding_dim
        }

        fn max_sequence_length(&self) -> usize {
            self.max_sequence_length
        }
    }

    /// Pool `[B, L, D]` hidden states down to `[B, D]` per the family's
    /// pooling rule. Returns a `Vec<Vec<f32>>` of length B.
    fn pool_hidden_state(
        data: &[f32],
        attention_mask: &Array2<i64>,
        batch_size: usize,
        seq_len: usize,
        dim: usize,
        family: TextEncoderFamily,
    ) -> Vec<Vec<f32>> {
        let row_stride = seq_len * dim;
        let mut out = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let row = &data[b * row_stride..(b + 1) * row_stride];
            let pooled = match family {
                TextEncoderFamily::Qwen3 => {
                    // Last non-pad token. With left-padding the last
                    // token is always real, but we still respect the
                    // mask in case the export uses right-padding.
                    let mut last_idx = seq_len.saturating_sub(1);
                    for i in (0..seq_len).rev() {
                        if attention_mask[[b, i]] == 1 {
                            last_idx = i;
                            break;
                        }
                    }
                    row[last_idx * dim..(last_idx + 1) * dim].to_vec()
                }
                TextEncoderFamily::Cls => {
                    // Position 0.
                    row[0..dim].to_vec()
                }
                TextEncoderFamily::Mean => {
                    let mut acc = vec![0.0f32; dim];
                    let mut count = 0usize;
                    for i in 0..seq_len {
                        if attention_mask[[b, i]] == 1 {
                            for d in 0..dim {
                                acc[d] += row[i * dim + d];
                            }
                            count += 1;
                        }
                    }
                    if count > 0 {
                        let inv = 1.0 / count as f32;
                        for v in acc.iter_mut().take(dim) {
                            *v *= inv;
                        }
                    }
                    acc
                }
                TextEncoderFamily::SentenceEmbedding => {
                    // Should never reach this branch — caller checks
                    // `output_is_pooled` first. Fall back to mean as a
                    // safety net rather than panic.
                    let mut acc = vec![0.0f32; dim];
                    for i in 0..seq_len {
                        for d in 0..dim {
                            acc[d] += row[i * dim + d];
                        }
                    }
                    let inv = 1.0 / seq_len as f32;
                    for v in acc.iter_mut().take(dim) {
                        *v *= inv;
                    }
                    acc
                }
            };
            out.push(pooled);
        }
        out
    }

    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    /// Pick the first matching name from `candidates` that appears in
    /// `available`. Returns `None` if no candidate matches.
    fn pick_input(available: &[String], candidates: &[&str]) -> Option<String> {
        for c in candidates {
            for a in available {
                if a == c {
                    return Some(a.clone());
                }
            }
        }
        None
    }
}

pub use onnx_backend::GenericTextEncoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruct_query_prefixes_only_the_query_role() {
        let t = EmbeddingTemplate::InstructQuery {
            default_task: "retrieve relevant passages".into(),
        };
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Query { task: None }, "capital of France"),
            "Instruct: retrieve relevant passages\nQuery:capital of France"
        );
        assert_eq!(
            apply_embedding_template(
                &t,
                &EmbedRole::Query {
                    task: Some("find code examples".into())
                },
                "capital of France"
            ),
            "Instruct: find code examples\nQuery:capital of France"
        );
        // Document role: no prefix at all, per Qwen3-Embedding's own docs.
        assert_eq!(
            apply_embedding_template(
                &t,
                &EmbedRole::Document { title: None },
                "Paris is the capital of France."
            ),
            "Paris is the capital of France."
        );
    }

    #[test]
    fn gemma_task_prefixes_both_roles_differently() {
        let t = EmbeddingTemplate::GemmaTask {
            default_task: "search result".into(),
        };
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Query { task: None }, "capital of France"),
            "task: search result | query: capital of France"
        );
        assert_eq!(
            apply_embedding_template(
                &t,
                &EmbedRole::Document { title: None },
                "Paris is the capital of France."
            ),
            "title: none | text: Paris is the capital of France."
        );
        assert_eq!(
            apply_embedding_template(
                &t,
                &EmbedRole::Document {
                    title: Some("France".into())
                },
                "Paris is the capital of France."
            ),
            "title: France | text: Paris is the capital of France."
        );
    }

    #[test]
    fn nomic_prefix_differs_by_role() {
        let t = EmbeddingTemplate::NomicPrefix;
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Query { task: None }, "x"),
            "search_query: x"
        );
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Document { title: None }, "x"),
            "search_document: x"
        );
    }

    #[test]
    fn chat_instruct_names_the_instruction_and_leaves_the_input() {
        let t = EmbeddingTemplate::ChatInstruct {
            default_task: "Retrieve relevant documents for the query.".into(),
            document_task: "Represent the user's input.".into(),
        };
        let query = EmbedRole::Query { task: None };
        let document = EmbedRole::Document { title: None };
        let asked = EmbedRole::Query {
            task: Some("Find code".into()),
        };
        assert_eq!(
            chat_instruction(&t, &query),
            Some("Retrieve relevant documents for the query.")
        );
        assert_eq!(
            chat_instruction(&t, &document),
            Some("Represent the user's input.")
        );
        assert_eq!(chat_instruction(&t, &asked), Some("Find code"));
        assert_eq!(apply_embedding_template(&t, &query, "x"), "x");
        assert_eq!(chat_instruction(&EmbeddingTemplate::None, &query), None);
    }

    #[test]
    fn query_passage_prefix_differs_by_role() {
        let t = EmbeddingTemplate::QueryPassagePrefix;
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Query { task: None }, "x"),
            "query: x"
        );
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Document { title: None }, "x"),
            "passage: x"
        );
    }

    #[test]
    fn none_template_never_prefixes() {
        let t = EmbeddingTemplate::None;
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Query { task: None }, "x"),
            "x"
        );
        assert_eq!(
            apply_embedding_template(&t, &EmbedRole::Document { title: None }, "x"),
            "x"
        );
    }

    #[test]
    fn default_role_is_document() {
        assert_eq!(EmbedRole::default(), EmbedRole::Document { title: None });
    }

    /// An ONNX text encoder against its source checkpoint, from vectors the
    /// reference script wrote. Each text already carries any instruction or
    /// role prefix, so it is embedded as a document with no template.
    /// `PRAECISE_ENCODER_POOLING` is `sentence` (pre-pooled output) or `mean`.
    #[test]
    #[ignore = "needs an encoder and its reference vectors: PRAECISE_ENCODER_*"]
    fn an_onnx_encoder_matches_its_source_checkpoint() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
        let (onnx, tokenizer, expected, dim, pooling) = (
            var("PRAECISE_ENCODER_ONNX"),
            var("PRAECISE_ENCODER_TOKENIZER"),
            var("PRAECISE_ENCODER_EXPECTED"),
            var("PRAECISE_ENCODER_DIM"),
            var("PRAECISE_ENCODER_POOLING"),
        );
        crate::session::install(
            crate::session::OnnxSettings::resolve(&crate::session::OnnxConfig::default(), 8),
            crate::session::Gpus::default(),
        );
        let family = match pooling.as_str() {
            "sentence" => TextEncoderFamily::SentenceEmbedding,
            "mean" => TextEncoderFamily::Mean,
            other => panic!("unknown pooling {other}"),
        };
        let exp: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(expected).unwrap()).unwrap();
        let texts: Vec<String> = serde_json::from_value(exp["texts"].clone()).unwrap();
        let want: Vec<Vec<f32>> = serde_json::from_value(exp["embeddings"].clone()).unwrap();
        let encoder = GenericTextEncoder::from_onnx(
            onnx,
            tokenizer,
            family,
            EmbeddingTemplate::None,
            dim.parse().unwrap(),
            8192,
        )
        .unwrap();
        let mut worst = 1f32;
        for (text, want) in texts.iter().zip(&want) {
            let cfg = TextEmbedConfig {
                normalize: true,
                ..Default::default()
            };
            let got = encoder.embed(&[text.as_str().into()], &cfg).unwrap();
            let cos: f32 = got.embeddings[0].iter().zip(want).map(|(a, b)| a * b).sum();
            worst = worst.min(cos);
        }
        eprintln!("worst cosine {worst} over {} texts", texts.len());
        assert!(worst > 0.999, "worst cosine {worst}");
    }
}
