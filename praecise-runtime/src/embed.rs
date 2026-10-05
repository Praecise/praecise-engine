//! Pooled embeddings from decoder embedders.
//!
//! One input (text, or text carrying pictures and video frames) becomes one
//! vector: the input is written into the checkpoint's chat template with an
//! instruction as the system turn, the input as the user turn and the
//! assistant turn opened, and the end-of-text token after it when the
//! checkpoint's tokenizer appends one; the hidden state of the last position
//! is read (LAST pooling) and L2-normalised. Every input is decoded on its own, in a
//! fresh sequence, so its vector does not depend on the other inputs in the
//! request.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};

use crate::error::{Error, Result};

/// What an input with no text and no pictures is embedded as, as the
/// checkpoints' own pipelines do.
pub const EMPTY_INPUT: &str = "NULL";

/// One input's vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The input's position in the request.
    pub index: usize,
    /// Unit-length vector.
    pub vector: Vec<f32>,
    /// Tokens the input occupied, template and pictures included.
    pub tokens: usize,
}

/// Options for an embedding call.
#[derive(Debug, Clone, Copy)]
pub struct EmbedOptions {
    /// Context length per input; a longer input is refused, never truncated.
    pub n_ctx: u32,
    /// Threads for the batch.
    pub n_threads: i32,
    /// Keep only the first `dimensions` components (for checkpoints trained
    /// with nested dimensions), normalised again; all of them when `None`.
    pub dimensions: Option<usize>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            n_ctx: 8192,
            n_threads: std::thread::available_parallelism()
                .map(|n| i32::try_from(n.get()).unwrap_or(1))
                .unwrap_or(1),
            dimensions: None,
        }
    }
}

fn inference(what: impl std::fmt::Display) -> Error {
    Error::Inference(what.to_string())
}

/// Unicode punctuation (general category P) that an instruction may end in.
fn is_punctuation(c: char) -> bool {
    matches!(
        c,
        '!' | '"' | '#' | '%' | '&' | '\'' | '(' | ')' | '*' | ',' | '-' | '.' | '/' | ':' | ';' | '?' | '@' | '[' | '\\' | ']' | '_' | '{' | '}'
            | '\u{00A1}' | '\u{00A7}' | '\u{00AB}' | '\u{00B6}' | '\u{00B7}' | '\u{00BB}' | '\u{00BF}'
            | '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{2043}' | '\u{2045}'..='\u{2051}' | '\u{2053}'..='\u{205E}'
            | '\u{3001}'..='\u{3003}' | '\u{3008}'..='\u{3011}' | '\u{3014}'..='\u{301F}'
            | '\u{FF01}'..='\u{FF03}' | '\u{FF05}'..='\u{FF0A}' | '\u{FF0C}'..='\u{FF0F}' | '\u{FF1A}' | '\u{FF1B}' | '\u{FF1F}' | '\u{FF20}'
    )
}

/// The instruction as the checkpoints' pipelines use it: trimmed, with a
/// full stop added when it does not end in punctuation.
#[must_use]
pub fn instruction(text: &str) -> String {
    let t = text.trim();
    match t.chars().last() {
        Some(c) if !is_punctuation(c) => format!("{t}."),
        _ => t.to_owned(),
    }
}

/// The prompt for one input: the checkpoint's chat template with
/// `instruction` as the system turn and `content` as the user turn, the
/// assistant turn opened.
///
/// # Errors
/// When the model has no chat template or the template cannot be applied.
pub fn prompt(model: &LlamaModel, instruction_text: &str, content: &str) -> Result<String> {
    let template = model.chat_template(None).map_err(|e| inference(format!("the model has no chat template: {e}")))?;
    let content = if content.is_empty() { EMPTY_INPUT } else { content };
    let chat = [
        LlamaChatMessage::new("system".into(), instruction(instruction_text)).map_err(inference)?,
        LlamaChatMessage::new("user".into(), content.into()).map_err(inference)?,
    ];
    model.apply_chat_template(&template, &chat, true).map_err(inference)
}

/// The end-of-text token, when the checkpoint's tokenizer appends one to
/// every input (the pooled position is then that token).
fn end_token(model: &LlamaModel) -> Option<llama_cpp_2::token::LlamaToken> {
    let eos = model.token_eos();
    (model.vocab_adds_eos() && eos.0 >= 0).then_some(eos)
}

/// Context parameters for embedding one input at a time in sequence 0.
fn context(n_ctx: u32, n_ubatch: u32, n_threads: i32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ubatch)
        .with_n_seq_max(1)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Last)
}

/// The pooled vector of the input just decoded, cut and normalised.
fn pooled(ctx: &LlamaContext<'_>, dimensions: Option<usize>) -> Result<Vec<f32>> {
    let v = ctx.embeddings_seq_ith(0).map_err(inference)?;
    let keep = match dimensions {
        Some(0) => return Err(inference("zero embedding dimensions requested")),
        Some(d) if d > v.len() => return Err(inference(format!("{d} dimensions requested of a {}-dimensional model", v.len()))),
        Some(d) => d,
        None => v.len(),
    };
    Ok(normalise(&v[..keep]))
}

/// `v` scaled to unit length (left as is when it is all zeros).
#[must_use]
pub fn normalise(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
}

/// A context of at least `floor` tokens that fits `longest`, within the
/// requested and the trained context.
fn fit(longest: usize, floor: u32, options: &EmbedOptions, model: &LlamaModel) -> Result<u32> {
    let limit = options.n_ctx.min(model.n_ctx_train().max(1));
    if longest > limit as usize {
        return Err(inference(format!("an input of {longest} tokens does not fit the context of {limit}")));
    }
    Ok(u32::try_from(longest.next_multiple_of(256)).unwrap_or(u32::MAX).max(floor).min(limit))
}

/// Embed text inputs with the decoder embedder `model`.
///
/// # Errors
/// When the model has no chat template, an input does not fit the context,
/// or the backend fails.
pub fn embed(backend: &LlamaBackend, model: &LlamaModel, instruction_text: &str, inputs: &[&str], options: EmbedOptions) -> Result<Vec<Embedding>> {
    let mut tokenized = Vec::with_capacity(inputs.len());
    for text in inputs {
        let p = prompt(model, instruction_text, text)?;
        let mut tokens = model.str_to_token(&p, AddBos::Never).map_err(inference)?;
        tokens.extend(end_token(model));
        tokenized.push(tokens);
    }
    let longest = tokenized.iter().map(Vec::len).max().unwrap_or(0);
    let n_ctx = fit(longest, 256, &options, model)?;
    let mut ctx = model.new_context(backend, context(n_ctx, n_ctx, options.n_threads)).map_err(inference)?;
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut out = Vec::with_capacity(inputs.len());
    for (index, tokens) in tokenized.iter().enumerate() {
        ctx.clear_kv_cache();
        batch.clear();
        batch.add_sequence(tokens, 0, false).map_err(inference)?;
        ctx.decode(&mut batch).map_err(inference)?;
        out.push(Embedding { index, vector: pooled(&ctx, options.dimensions)?, tokens: tokens.len() });
    }
    Ok(out)
}

/// Micro-batch a multimodal input is decoded in; a longer input is decoded
/// in several, which a causal decoder pools the same.
#[cfg(feature = "mtmd")]
const MEDIA_UBATCH: u32 = 2048;

/// Embed inputs that may carry pictures and video frames with the decoder
/// embedder `model` and its vision projector: each input becomes one
/// vector covering its text, pictures and frames together.
///
/// Each input is written into the chat template with its media markers in
/// place, tokenized into text and media chunks, and evaluated into a fresh
/// sequence; positions come from the chunk evaluation, so multi-axis rotary
/// positions are laid out by the projector.
///
/// # Errors
/// When the model has no chat template, the projector has no vision tower
/// and an input carries pictures, an input's markers and pictures disagree,
/// an input does not fit the context, or the backend fails.
#[cfg(feature = "mtmd")]
pub fn embed_media(
    backend: &LlamaBackend,
    model: &LlamaModel,
    projector: &llama_cpp_2::mtmd::MtmdContext,
    instruction_text: &str,
    inputs: &[crate::media::MediaInput],
    options: EmbedOptions,
) -> Result<Vec<Embedding>> {
    if inputs.iter().any(|i| !i.pictures.is_empty()) && !projector.support_vision() {
        return Err(inference("the projector has no vision tower"));
    }
    for input in inputs {
        input.check("an input")?;
    }
    let end = match end_token(model) {
        Some(t) => String::from_utf8(model.token_to_piece_bytes(t, 64, true, None).map_err(inference)?).map_err(inference)?,
        None => String::new(),
    };
    let tokenize = |input: &crate::media::MediaInput| -> Result<llama_cpp_2::mtmd::MtmdInputChunks> {
        let text = prompt(model, instruction_text, &input.text)? + &end;
        let bitmaps = input.bitmaps()?;
        let refs: Vec<&llama_cpp_2::mtmd::MtmdBitmap> = bitmaps.iter().collect();
        projector
            .tokenize(llama_cpp_2::mtmd::MtmdInputText { text, add_special: false, parse_special: true }, &refs)
            .map_err(|e| inference(format!("multimodal tokenization failed: {e}")))
    };
    // The context fits the longest input; the micro-batch is the same for
    // every request, so an input is split, and pooled, identically whatever
    // else the request carries.
    let mut lengths = Vec::with_capacity(inputs.len());
    for input in inputs {
        lengths.push(tokenize(input)?.total_tokens());
    }
    let n_ctx = fit(lengths.iter().copied().max().unwrap_or(0), MEDIA_UBATCH, &options, model)?;
    let n_ubatch = n_ctx.min(MEDIA_UBATCH);
    let mut ctx = model.new_context(backend, context(n_ctx, n_ubatch, options.n_threads)).map_err(inference)?;
    let n_batch = i32::try_from(n_ubatch).map_err(inference)?;
    let mut out = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        let chunks = tokenize(input)?;
        ctx.clear_kv_cache();
        chunks
            .eval_chunks(projector, &mut ctx, 0, 0, n_batch, true)
            .map_err(|e| inference(format!("the input could not be evaluated: {e}")))?;
        out.push(Embedding { index, vector: pooled(&ctx, options.dimensions)?, tokens: lengths[index] });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instructions_end_in_punctuation() {
        assert_eq!(instruction("  Represent the user's input "), "Represent the user's input.");
        assert_eq!(instruction("Retrieve documents."), "Retrieve documents.");
        assert_eq!(instruction("Find this?"), "Find this?");
        assert_eq!(instruction("検索してください。"), "検索してください。");
        // Symbols are not punctuation.
        assert_eq!(instruction("cost in $"), "cost in $.");
    }

    #[test]
    fn normalised_vectors_have_unit_length() {
        let v = normalise(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-7 && (v[1] - 0.8).abs() < 1e-7);
        assert_eq!(normalise(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}
