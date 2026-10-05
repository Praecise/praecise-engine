//! Cross-encoder reranking.
//!
//! A reranker reads a query and one document together and scores how well
//! the document answers the query. The model's graph ends in a classification
//! head (RANK pooling); the head's first output is the score.
//!
//! Every pair is decoded on its own, in a fresh sequence: a pair's score does
//! not depend on which other documents arrived in the same request or in what
//! order, so anyone holding the same weights reproduces it exactly.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::error::{Error, Result};

/// One document's score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankScore {
    /// The document's position in the request.
    pub index: usize,
    /// The classification head's first output for the pair.
    pub score: f32,
    /// Tokens the pair occupied.
    pub tokens: usize,
}

/// Options for a reranking call.
#[derive(Debug, Clone, Copy)]
pub struct RerankOptions {
    /// Context length per pair; a longer pair is refused, never truncated.
    pub n_ctx: u32,
    /// Threads for the batch.
    pub n_threads: i32,
}

impl Default for RerankOptions {
    fn default() -> Self {
        Self {
            n_ctx: 8192,
            n_threads: std::thread::available_parallelism()
                .map(|n| i32::try_from(n.get()).unwrap_or(1))
                .unwrap_or(1),
        }
    }
}

fn inference(what: impl std::fmt::Display) -> Error {
    Error::Inference(what.to_string())
}

/// The tokens of one (query, document) pair. A model that ships a `rerank`
/// prompt template gets the pair written into it; otherwise the pair is the
/// query and the document joined by the vocabulary's own markers
/// (beginning, end and separator tokens, each where the vocabulary adds it).
///
/// # Errors
///
/// When the model's vocabulary cannot tokenize the text.
pub fn pair_tokens(model: &LlamaModel, query: &str, document: &str) -> Result<Vec<LlamaToken>> {
    if let Ok(template) = model.chat_template(Some("rerank")) {
        let template = template.to_str().map_err(inference)?;
        let prompt = template.replace("{query}", query).replace("{document}", document);
        return model.str_to_token(&prompt, AddBos::Never).map_err(inference);
    }
    let eos = {
        let eos = model.token_eos();
        if eos.0 < 0 { model.token_sep() } else { eos }
    };
    let mut out = Vec::new();
    if model.vocab_adds_bos() {
        out.push(model.token_bos());
    }
    out.extend(model.str_to_token(query, AddBos::Never).map_err(inference)?);
    if model.vocab_adds_eos() {
        out.push(eos);
    }
    if model.vocab_adds_sep() {
        out.push(model.token_sep());
    }
    out.extend(model.str_to_token(document, AddBos::Never).map_err(inference)?);
    if model.vocab_adds_eos() {
        out.push(eos);
    }
    Ok(out)
}

/// Score every document against `query` with the reranker `model`.
///
/// # Errors
///
/// When the model has no classification head, a pair does not fit the
/// context, or the backend fails to decode.
pub fn rerank(
    backend: &LlamaBackend,
    model: &LlamaModel,
    query: &str,
    documents: &[&str],
    options: RerankOptions,
) -> Result<Vec<RerankScore>> {
    if model.n_cls_out() == 0 {
        return Err(inference("the model has no classification head to score with"));
    }
    let pairs = documents
        .iter()
        .map(|d| pair_tokens(model, query, d))
        .collect::<Result<Vec<_>>>()?;
    let longest = pairs.iter().map(Vec::len).max().unwrap_or(0);
    let n_ctx = options.n_ctx.min(model.n_ctx_train().max(1));
    if longest > n_ctx as usize {
        return Err(inference(format!(
            "a pair of {longest} tokens does not fit the context of {n_ctx}"
        )));
    }
    let mut ctx = model
        .new_context(backend, pair_context(n_ctx, n_ctx, options.n_threads))
        .map_err(inference)?;
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut out = Vec::with_capacity(pairs.len());
    for (index, tokens) in pairs.iter().enumerate() {
        ctx.clear_kv_cache();
        batch.clear();
        batch.add_sequence(tokens, 0, false).map_err(inference)?;
        ctx.decode(&mut batch).map_err(inference)?;
        out.push(RerankScore {
            index,
            score: head_score(&ctx)?,
            tokens: tokens.len(),
        });
    }
    Ok(out)
}

/// Context parameters for scoring one pair at a time in sequence 0.
fn pair_context(n_ctx: u32, n_ubatch: u32, n_threads: i32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ubatch)
        .with_n_seq_max(1)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Rank)
}

/// The classification head's first output for the pair just decoded.
fn head_score(ctx: &llama_cpp_2::context::LlamaContext<'_>) -> Result<f32> {
    let head = ctx.embeddings_seq_ith(0).map_err(inference)?;
    head.first()
        .copied()
        .ok_or_else(|| inference("the classification head returned nothing"))
}


/// Micro-batch a multimodal pair is decoded in; a longer pair is decoded in
/// several, which a causal decoder scores the same.
#[cfg(feature = "mtmd")]
const MEDIA_UBATCH: u32 = 2048;

/// Score every document against `query`, where either side may carry
/// pictures and video, with the reranker `model` and its vision projector.
///
/// Each pair is written into the model's `rerank` template with the media
/// markers in place, tokenized into text and media chunks, and evaluated
/// into a fresh sequence; positions come from the chunk evaluation, so
/// multi-axis rotary positions are laid out by the projector. A pair's score
/// does not depend on the other documents in the request.
///
/// # Errors
///
/// When the model has no classification head or no `rerank` template, the
/// projector has no vision tower, a side's markers and pictures disagree, a
/// pair does not fit the context, or the backend fails.
#[cfg(feature = "mtmd")]
pub fn rerank_media(
    backend: &LlamaBackend,
    model: &LlamaModel,
    projector: &llama_cpp_2::mtmd::MtmdContext,
    query: &crate::media::MediaInput,
    documents: &[crate::media::MediaInput],
    options: RerankOptions,
) -> Result<Vec<RerankScore>> {
    if model.n_cls_out() == 0 {
        return Err(inference("the model has no classification head to score with"));
    }
    let template = model
        .chat_template(Some("rerank"))
        .map_err(|_| inference("the model has no rerank template to place pictures in"))?;
    let template = template.to_str().map_err(inference)?.to_owned();
    let has_media = !query.pictures.is_empty() || documents.iter().any(|d| !d.pictures.is_empty());
    if has_media && !projector.support_vision() {
        return Err(inference("the projector has no vision tower"));
    }
    for (side, input) in std::iter::once(("the query", query)).chain(documents.iter().map(|d| ("a document", d))) {
        input.check(side)?;
    }
    let query_first = match (template.find("{query}"), template.find("{document}")) {
        (Some(q), Some(d)) => q < d,
        _ => return Err(inference("the rerank template lacks a query or document slot")),
    };

    let tokenize = |document: &crate::media::MediaInput| -> Result<llama_cpp_2::mtmd::MtmdInputChunks> {
        let prompt = template
            .replace("{query}", &query.text)
            .replace("{document}", &document.text);
        let (first, second) = if query_first { (query, document) } else { (document, query) };
        let mut bitmaps = first.bitmaps()?;
        bitmaps.extend(second.bitmaps()?);
        let refs: Vec<&llama_cpp_2::mtmd::MtmdBitmap> = bitmaps.iter().collect();
        projector
            .tokenize(
                llama_cpp_2::mtmd::MtmdInputText {
                    text: prompt,
                    add_special: false,
                    parse_special: true,
                },
                &refs,
            )
            .map_err(|e| inference(format!("multimodal tokenization failed: {e}")))
    };

    // Size the context to the longest pair, so a request of short pairs does
    // not reserve the room a long video would need. The micro-batch is the
    // same for every request, so a pair is split, and scored, identically
    // whatever else the request carries.
    let mut lengths = Vec::with_capacity(documents.len());
    for document in documents {
        lengths.push(tokenize(document)?.total_tokens());
    }
    let longest = lengths.iter().copied().max().unwrap_or(0);
    let limit = options.n_ctx.min(model.n_ctx_train().max(1));
    if longest > limit as usize {
        return Err(inference(format!(
            "a pair of {longest} tokens does not fit the context of {limit}"
        )));
    }
    let n_ctx = u32::try_from(longest.next_multiple_of(256))
        .unwrap_or(u32::MAX)
        .max(MEDIA_UBATCH)
        .min(limit);
    let n_ubatch = n_ctx.min(MEDIA_UBATCH);
    let mut ctx = model
        .new_context(backend, pair_context(n_ctx, n_ubatch, options.n_threads))
        .map_err(inference)?;
    let n_batch = i32::try_from(n_ubatch).map_err(inference)?;
    let mut out = Vec::with_capacity(documents.len());
    for (index, document) in documents.iter().enumerate() {
        let chunks = tokenize(document)?;
        ctx.clear_kv_cache();
        chunks
            .eval_chunks(projector, &mut ctx, 0, 0, n_batch, true)
            .map_err(|e| inference(format!("the pair could not be evaluated: {e}")))?;
        out.push(RerankScore {
            index,
            score: head_score(&ctx)?,
            tokens: lengths[index],
        });
    }
    Ok(out)
}
