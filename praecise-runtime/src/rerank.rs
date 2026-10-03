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
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_n_seq_max(1)
        .with_n_threads(options.n_threads)
        .with_n_threads_batch(options.n_threads)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Rank);
    let mut ctx = model.new_context(backend, params).map_err(inference)?;
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut out = Vec::with_capacity(pairs.len());
    for (index, tokens) in pairs.iter().enumerate() {
        ctx.clear_kv_cache();
        batch.clear();
        batch.add_sequence(tokens, 0, false).map_err(inference)?;
        ctx.decode(&mut batch).map_err(inference)?;
        let head = ctx.embeddings_seq_ith(0).map_err(inference)?;
        let score = *head.first().ok_or_else(|| inference("the classification head returned nothing"))?;
        out.push(RerankScore {
            index,
            score,
            tokens: tokens.len(),
        });
    }
    Ok(out)
}
