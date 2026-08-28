//! N-gram speculative decode: one model, no drafter, drafts from token history.
//!
//! Same shape as [`crate::speculative::generate_speculative`] — propose a block,
//! verify it in one target decode, accept the longest matching prefix — but the
//! proposals come from an [`NgramCache`] instead of a second model. That removes
//! the second context, the second set of weights, and the KV bookkeeping needed
//! to keep two contexts in step.
//!
//! The trade is acceptance rate: a draft model understands the text, an n-gram
//! table only remembers it. That is a good trade exactly when text repeats —
//! agent loops re-reading a file, code with recurring identifiers, structured
//! output with a fixed skeleton — and a poor one on prose it has never seen.
//! Since an unaccepted draft costs only the verify slot it occupied, the floor
//! is ordinary decode speed rather than a slowdown, provided the block stays
//! short.
//!
//! ## Acceptance is the metric to watch
//!
//! llama.cpp's comparable path documents 0.703 acceptance; below roughly 0.3 a
//! block of this size stops paying for itself. [`NgramStats`] is returned so a
//! caller can measure rather than assume — the whole point of putting this in a
//! layer is being able to decide per workload whether to use it.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::config::GenerationConfig;
use crate::error::{Error, Result};
use crate::ngram::NgramCache;
use crate::result::{InferenceResult, StopReason};
use crate::sampling::build_sampler_chain;
use crate::stream::StopStream;

/// Fallback context length when neither the model nor the caller gives one.
const DEFAULT_CONTEXT_LENGTH: u32 = 8192;

/// How many tokens to propose per step by default.
///
/// Four, because acceptance decays along a block — each position needs every
/// earlier one to have matched — and the verify batch costs the full block
/// whether or not the tail is accepted. Longer blocks pay off only where
/// acceptance is very high, which is a per-workload judgement a caller can make
/// by passing its own value.
pub const DEFAULT_DRAFT_N: usize = 4;

/// What the n-gram path actually did, so callers can measure instead of assume.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NgramStats {
    /// Tokens proposed across the whole generation.
    pub drafted: u64,
    /// Proposed tokens the target agreed with.
    pub accepted: u64,
    /// Steps where the cache knew nothing and we decoded one token normally.
    pub cold_steps: u64,
}

impl NgramStats {
    /// Accepted ÷ drafted, or `None` before anything was drafted.
    ///
    /// Compare against ~0.703 (llama.cpp's `ngram-mod`). Well below that, the
    /// verify slots are being wasted and plain decode is the better path.
    #[must_use]
    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }
}

/// Generate with n-gram self-speculation over a single model.
///
/// `cache` is borrowed mutably and updated with accepted tokens as generation
/// proceeds, so passing the same cache across calls is what gives cross-request
/// reuse. Callers sharing one cache between tenants must not — see
/// [`crate::ngram`] on why.
///
/// Returns the result alongside [`NgramStats`] so acceptance can be measured.
#[allow(clippy::too_many_arguments)]
pub fn generate_ngram_speculative(
    model: &LlamaModel,
    backend: &LlamaBackend,
    context_length: u32,
    prompt: &str,
    config: &GenerationConfig,
    token_tx: Option<&tokio::sync::mpsc::Sender<String>>,
    draft_n: usize,
    cache: &mut NgramCache,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(InferenceResult, NgramStats)> {
    let start = std::time::Instant::now();
    let mut stats = NgramStats::default();

    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| Error::Other(format!("Tokenization failed: {e}")))?;
    if tokens_list.is_empty() {
        return Err(Error::Inference("prompt tokenized to zero tokens".to_string()));
    }
    let input_tokens = tokens_list.len() as u32;

    // The prompt is the best material the cache will get: it is the text most
    // likely to be echoed back in the completion.
    cache.observe(&tokens_list.iter().map(|t| t.0).collect::<Vec<_>>());

    let n_ctx = NonZeroU32::new(context_length)
        .unwrap_or_else(|| NonZeroU32::new(DEFAULT_CONTEXT_LENGTH).expect("non-zero"));
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Other(format!("Context creation failed: {e}")))?;

    // The verify batch holds id_last plus a full block of drafts.
    let batch_capacity = 1 + draft_n.max(1);
    let mut batch = LlamaBatch::new(batch_capacity.max(512), 1);

    // Prefill.
    batch.clear();
    let last_idx = tokens_list.len() - 1;
    for (i, tok) in tokens_list.iter().enumerate() {
        batch
            .add(*tok, i as i32, &[0], i == last_idx)
            .map_err(|e| Error::Other(format!("Batch add failed: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| Error::Other(format!("Prefill decode failed: {e}")))?;

    let mut sampler = build_sampler_chain(config, model.n_vocab());
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut stream = StopStream::new(config.stop.clone());

    let mut n_past = tokens_list.len() as i32;
    let mut id_last = sampler.sample(&ctx, -1);
    sampler.accept(id_last);

    let mut output_tokens: u32 = 0;
    let mut history: Vec<i32> = tokens_list.iter().map(|t| t.0).collect();

    // The first sampled token is output like any other before the loop verifies
    // anything, so emit it here rather than letting the loop re-sample it.
    let mut stop_now = false;
    if model.is_eog_token(id_last) {
        stop_now = true;
    } else {
        output_tokens += 1;
        history.push(id_last.0);
        if let Ok(piece) = model.token_to_piece(id_last, &mut decoder, true, None)
            && !(stream.push(&piece, token_tx) && !stream.hit_stop())
        {
            stop_now = true;
        }
    }

    let max_pos = n_ctx.get() as i32 - 1;

    'genloop: while !stop_now && output_tokens < config.max_tokens && n_past < max_pos {
        // Stop before more GPU work if the receiver dropped or the caller
        // cancelled; the context drops at return.
        if token_tx.is_some_and(tokio::sync::mpsc::Sender::is_closed)
            || cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        {
            break 'genloop;
        }

        // 1. Draft from history. Empty is normal and simply means plain decode.
        let drafts: Vec<LlamaToken> = cache
            .draft(&history, draft_n)
            .into_iter()
            .map(LlamaToken)
            .collect();
        if drafts.is_empty() {
            stats.cold_steps += 1;
        } else {
            stats.drafted += drafts.len() as u64;
        }

        // 2. Verify: id_last first, then the drafts, logits at every position.
        batch.clear();
        batch
            .add(id_last, n_past, &[0], true)
            .map_err(|e| Error::Other(format!("Batch add failed: {e}")))?;
        for (i, d) in drafts.iter().enumerate() {
            batch
                .add(*d, n_past + 1 + i as i32, &[0], true)
                .map_err(|e| Error::Other(format!("Batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("Decode failed: {e}")))?;

        // 3. Accept the longest matching prefix. Sampling at index i gives the
        //    token that truly follows position i, so a draft is accepted only
        //    where it equals what the target would itself have produced —
        //    which is what keeps output identical to non-speculative decode.
        let mut n_accepted: usize = 0;
        let mut idx: i32 = 0;
        loop {
            let sampled = sampler.sample(&ctx, idx);
            sampler.accept(sampled);
            id_last = sampled;
            history.push(sampled.0);

            if model.is_eog_token(sampled) {
                stop_now = true;
                break;
            }
            output_tokens += 1;
            if let Ok(piece) = model.token_to_piece(sampled, &mut decoder, true, None)
                && !(stream.push(&piece, token_tx) && !stream.hit_stop())
            {
                stop_now = true;
            }

            let matched = (idx as usize) < drafts.len() && sampled == drafts[idx as usize];
            if !matched {
                break;
            }
            n_accepted += 1;
            idx += 1;
            if stop_now || output_tokens >= config.max_tokens {
                break;
            }
        }
        stats.accepted += n_accepted as u64;

        // 4. Learn from what was accepted only. Feeding rejected drafts back
        //    would teach the cache its own mistakes and depress acceptance.
        cache.observe(&history[history.len().saturating_sub(n_accepted + MAX_LEARN_TAIL)..]);

        // 5. Advance past the accepted run and drop rejected drafts' KV. The
        //    verify wrote KV for every drafted position; anything past the
        //    accepted prefix describes tokens that were never emitted, so it
        //    must go before the next step reuses those positions.
        n_past += 1 + n_accepted as i32;
        let _ = ctx.kv_cache_seq_rm(0, Some(n_past as u32), None);
    }

    let generation_time_ms = start.elapsed().as_millis() as u64;
    let tokens_per_second = if generation_time_ms > 0 {
        (output_tokens as f64) / (generation_time_ms as f64 / 1000.0)
    } else {
        0.0
    };
    let stop_reason = StopReason::from_loop(stream.hit_stop(), output_tokens, config.max_tokens);
    let (text, thinking) = stream.finish_parts(token_tx);

    Ok((
        InferenceResult {
            text,
            thinking,
            input_tokens,
            output_tokens,
            generation_time_ms,
            tokens_per_second,
            stop_reason,
            // Same as the drafted-model path: the top-k capture a commitment
            // needs lives in the standard decode loop, not here.
            commitment: None,
        },
        stats,
    ))
}

/// How much history to re-observe after each step.
///
/// Enough to include the n-grams ending at the accepted tokens without walking
/// the whole sequence every step: the longest key is [`crate::ngram::MAX_NGRAM`]
/// tokens, so that many tokens of lead-in is all a new observation can use.
const MAX_LEARN_TAIL: usize = crate::ngram::MAX_NGRAM;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_rate_is_none_before_any_draft() {
        assert_eq!(NgramStats::default().acceptance_rate(), None);
    }

    #[test]
    fn acceptance_rate_is_accepted_over_drafted() {
        let s = NgramStats { drafted: 10, accepted: 7, cold_steps: 2 };
        assert_eq!(s.acceptance_rate(), Some(0.7));
    }

    #[test]
    fn a_cold_step_does_not_count_as_a_failed_draft() {
        // Cold steps drafted nothing, so they must not drag the rate down —
        // otherwise a cache that correctly declines to guess looks like one
        // that guesses badly.
        let s = NgramStats { drafted: 4, accepted: 4, cold_steps: 100 };
        assert_eq!(s.acceptance_rate(), Some(1.0));
    }
}
