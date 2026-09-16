//! Continuous batching engine for GGUF text generation.
//!
//! Backend-gated (llama.cpp): compiled under the `bundled-llama` feature.
//!
//! Holds a single long-lived `llama_context` per model with a fixed pool of
//! sequence slots and interleaves every active request into one `llama_decode`
//! per step. Throughput scales with the number of active sequences because a
//! decode over K sequences costs close to a decode over one — the same weight
//! matrices are read once and applied to K token rows.
//!
//! ## Slot model
//!
//! The context is built with `n_seq_max = max_slots()` KV-cache sequence slots.
//! Each in-flight request owns one slot (its `seq_id`); the number of admitted
//! requests never exceeds the slot count, so a `seq_id` is always a valid slot
//! index. When a request finishes its slot's KV is cleared so a waiting request
//! can take it.
//!
//! ## Scheduler loop
//!
//! One dedicated OS thread per model owns the `LlamaModel` and its
//! `LlamaContext`. Each iteration admits waiting requests into free slots,
//! extends every running sequence by its last sampled token, spends the
//! remaining batch capacity prefilling prompts, runs one `llama_decode`, then
//! samples each sequence from its own logits with its own sampler.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
#[cfg(feature = "mtmd")]
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdInputChunks, MtmdInputText};
use tracing::{info, warn};

use crate::config::GenerationConfig;
use crate::error::{Error, Result};
use crate::prompt::render_chatml_prompt;
use crate::result::{ChatMessage, InferenceResult, StopReason};
use crate::stream::{ReasoningFrame, StopStream};
use llama_cpp_2::context::params::LlamaContextType;
use llama_cpp_2::speculative::{MtpSpeculativeParams, SpeculativeBatch};

/// Number of concurrent sequence slots a batched context serves by default.
/// This is the KV-cache `n_seq_max` and the ceiling on requests decoded in one
/// step. llama.cpp divides the context across it, so the per-request window is
/// `n_ctx / n_seq_max`.
const MAX_SLOTS_DEFAULT: usize = 32;

/// Most sequences generating at once for which the engine still drafts,
/// overridable with `TENZRO_BATCH_SPEC_MAX_ACTIVE` (0 turns drafting off).
///
/// Speculation pays when a step's decode is bound by reading the weights and
/// verifying extra tokens is nearly free. Batching many sequences spends that
/// same headroom, so whether drafts still pay at a given width depends on how
/// many of them are accepted. Measured on qwen3.8-27b on a GB10, tok/s at 1, 4
/// and 16 streams:
///
/// * MTP head, 4 drafted: 23.1, 40.9, 87.6 against 13.2, 43.6, 92.4 without
///   speculation. Short drafts at ~35% acceptance only pay for a couple of
///   streams, so the head drafts up to two.
/// * DFlash2 block drafter, 7 drafted: 32.8, 67.4, 102.5. Long accepted blocks
///   keep paying at sixteen, so a block drafter drafts at every width.
pub fn spec_max_active(spec_type: i32) -> usize {
    std::env::var("TENZRO_BATCH_SPEC_MAX_ACTIVE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(if spec_type == 0 { 2 } else { usize::MAX })
}

/// Sequence slots, overridable with `TENZRO_MAX_SLOTS`.
///
/// This is `n_seq_max`. On a device that cannot afford `32 x` a useful window,
/// fewer slots buy back per-request context. The host's admission layer must
/// not advertise more concurrency than this — a request is only ever admitted
/// into one of these slots, so `n_seq_max` is the true concurrent capacity.
pub fn max_slots() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TENZRO_MAX_SLOTS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MAX_SLOTS_DEFAULT)
    })
}

/// Physical batch capacity for a single `llama_decode`, overridable with
/// `TENZRO_PHYSICAL_BATCH`. Sets `n_batch`/`n_ubatch`; the compute buffer
/// scales with it.
const PHYSICAL_BATCH_DEFAULT: usize = 2048;

/// Physical batch size (see [`PHYSICAL_BATCH_DEFAULT`]).
fn physical_batch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TENZRO_PHYSICAL_BATCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v >= 32)
            .unwrap_or(PHYSICAL_BATCH_DEFAULT)
    })
}

/// Prompt tokens one slot may contribute to a single batch. Bounding the
/// per-slot share lets several prefills advance together instead of one long
/// prompt consuming every step's spare capacity until it is done.
const PREFILL_CHUNK: usize = 512;

/// How long the scheduler parks waiting for the first request before looping to
/// re-check the shutdown signal. Bounds shutdown latency when idle without
/// spinning the CPU.
const IDLE_POLL: Duration = Duration::from_millis(50);

/// The prompt for a batch request, before tokenization.
///
/// `Raw` is a fully-formed prompt string; `Chat` is a message list the
/// scheduler renders through the model's GGUF chat template (falling back to
/// ChatML). A host with model-specific templating renders its own prompt and
/// submits it as `Raw`.
pub enum BatchPrompt {
    /// A prompt string that is already fully formed (no templating applied).
    Raw(String),
    /// Chat messages rendered through the model's built-in chat template at
    /// admission time, with the generation prompt appended.
    Chat(Vec<ChatMessage>),
}

/// One unit of work submitted to a model's batch engine.
pub struct BatchRequest {
    /// The prompt to serve.
    pub prompt: BatchPrompt,
    /// Sampling / generation configuration.
    pub config: GenerationConfig,
    /// Per-token streaming sink. `None` for non-streaming callers; the final
    /// aggregate still returns via `result_tx`.
    pub token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    /// Where the terminal [`InferenceResult`] (or error) is delivered.
    pub result_tx: tokio::sync::oneshot::Sender<Result<InferenceResult>>,
    /// Images or audio, in the order their markers appear in the prompt. Empty
    /// for text. An engine spawned without a projector refuses a request that
    /// carries any, rather than dropping them and answering as if it had seen
    /// them.
    pub media: Vec<Vec<u8>>,
}

/// The multimodal projector an engine may hold. A unit type in a build without
/// mtmd, so the engine's signatures are the same either way.
#[cfg(feature = "mtmd")]
pub type Projector = MtmdContext;
/// The multimodal projector an engine may hold. A unit type in a build without
/// mtmd, so the engine's signatures are the same either way.
#[cfg(not(feature = "mtmd"))]
pub type Projector = ();

/// Handle to a per-model continuous-batching engine. Cloneable; every clone
/// submits to the same scheduler thread.
#[derive(Clone)]
pub struct BatchEngine {
    tx: Sender<BatchRequest>,
    inner: Arc<EngineInner>,
}

struct EngineInner {
    model_id: String,
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
    /// Closing this drops the scheduler's receiver, ending the loop.
    shutdown: Sender<()>,
}

/// Speculative decoding for the batch engine.
///
/// Every generating sequence drafts a block each step, every block is verified
/// in that step's one target decode, and each sequence keeps the longest prefix
/// its own sampler agrees with. On a model whose decode is bound by memory
/// bandwidth, verifying a few tokens costs little more than decoding one, so
/// the accepted drafts are close to free.
pub struct BatchSpeculation {
    /// 0 drafts with the target's own MTP head, 1 with a DFlash block-diffusion
    /// drafter.
    pub spec_type: i32,
    /// Most tokens a sequence drafts in one step.
    pub n_max: u8,
    /// A separately shipped drafter. `None` drafts with the target model.
    pub draft_model: Option<LlamaModel>,
}

impl BatchEngine {
    /// Spawn a batch engine for an already-loaded model. Takes ownership of the
    /// `LlamaModel` — it is moved onto the scheduler thread and lives there for
    /// the engine's lifetime. `context_length` is the effective (host-capped)
    /// per-sequence context window. `enable_thinking` is passed to templates
    /// that support a reasoning toggle.
    pub fn spawn(
        model_id: String,
        model: LlamaModel,
        backend: Arc<LlamaBackend>,
        context_length: u32,
        enable_thinking: bool,
    ) -> Result<Self> {
        Self::spawn_inner(model_id, model, backend, context_length, enable_thinking, None, None)
    }

    /// [`spawn`](Self::spawn), with the model's multimodal projector.
    ///
    /// The projector moves onto the scheduler thread with the model. Requests
    /// carrying media are prefilled one at a time on the shared context and
    /// then decode in the batch with everything else, so a vision model no
    /// longer has to be served one request at a time.
    #[cfg(feature = "mtmd")]
    pub fn spawn_with_projector(
        model_id: String,
        model: LlamaModel,
        backend: Arc<LlamaBackend>,
        context_length: u32,
        enable_thinking: bool,
        projector: Option<MtmdContext>,
    ) -> Result<Self> {
        Self::spawn_inner(model_id, model, backend, context_length, enable_thinking, projector, None)
    }

    /// [`spawn_with_projector`](Self::spawn_with_projector), drafting and
    /// verifying speculatively on every sequence.
    #[cfg(feature = "mtmd")]
    pub fn spawn_speculative(
        model_id: String,
        model: LlamaModel,
        backend: Arc<LlamaBackend>,
        context_length: u32,
        enable_thinking: bool,
        projector: Option<MtmdContext>,
        speculation: Option<BatchSpeculation>,
    ) -> Result<Self> {
        Self::spawn_inner(model_id, model, backend, context_length, enable_thinking, projector, speculation)
    }

    fn spawn_inner(
        model_id: String,
        model: LlamaModel,
        backend: Arc<LlamaBackend>,
        context_length: u32,
        enable_thinking: bool,
        projector: Option<Projector>,
        speculation: Option<BatchSpeculation>,
    ) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<BatchRequest>();
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();

        let model_id_thread = model_id.clone();
        let handle = std::thread::Builder::new()
            .name(format!("batch-{}", model_id))
            .spawn(move || {
                if let Err(e) = scheduler_loop(
                    &model_id_thread,
                    model,
                    &backend,
                    context_length,
                    enable_thinking,
                    projector,
                    speculation,
                    &rx,
                    &shutdown_rx,
                ) {
                    warn!(
                        "batch engine for {} exited with error: {}",
                        model_id_thread, e
                    );
                }
            })
            .map_err(|e| Error::Other(format!("failed to spawn batch scheduler thread: {}", e)))?;

        Ok(Self {
            tx,
            inner: Arc::new(EngineInner {
                model_id,
                handle: std::sync::Mutex::new(Some(handle)),
                shutdown: shutdown_tx,
            }),
        })
    }

    /// Submit a request to the scheduler. Returns immediately; the caller awaits
    /// the request's `result_tx` (and drains `token_tx` if streaming).
    pub fn submit(&self, req: BatchRequest) -> Result<()> {
        self.tx.send(req).map_err(|_| {
            Error::Other(format!(
                "batch engine for {} is no longer running",
                self.inner.model_id
            ))
        })
    }

    /// The model this engine serves.
    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    /// Signal the scheduler to stop and join its thread. In-flight requests
    /// receive an error on their `result_tx`.
    pub fn shutdown(&self) {
        let _ = self.inner.shutdown.send(());
        if let Some(handle) = self.inner.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        let _ = self.shutdown.send(());
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

/// What a slot's KV cache still holds after its request finished.
///
/// The whole point of prefix reuse: an agent's next turn repeats the system
/// prompt, the tool schemas and the entire conversation so far, and re-reading
/// those through the model is the single largest cost in an agent loop. Keeping
/// the tokens that are already in KV lets the next request start where the last
/// one diverged instead of at zero.
#[derive(Default, Clone)]
struct CachedPrefix {
    /// Tokens resident in this sequence's KV, in order.
    tokens: Vec<LlamaToken>,
}

/// Length of the shared prefix of two token runs.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A running sequence occupying one slot.
struct Sequence {
    /// KV-cache sequence id == slot index.
    seq_id: i32,
    sampler: LlamaSampler,
    token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    result_tx: Option<tokio::sync::oneshot::Sender<Result<InferenceResult>>>,
    decoder: encoding_rs::Decoder,
    /// Prompt tokens staged by `admit`, waiting for the scheduler to prefill
    /// them. `None` once the whole prompt has been committed to the KV cache.
    pending_prompt: Option<Vec<LlamaToken>>,
    /// How many tokens of `pending_prompt` are already committed.
    prefill_cursor: usize,
    /// Next KV position for this sequence.
    n_past: i32,
    /// Every token committed to this sequence's KV — prompt then generated —
    /// so the next request on this slot can measure its shared prefix.
    resident: Vec<LlamaToken>,
    input_tokens: u32,
    output_tokens: u32,
    /// Absolute position ceiling: `input_tokens + max_tokens`, capped at context.
    max_pos: i32,
    /// Accumulates decoded pieces, trims a configured stop sequence out of the
    /// text, and holds back bytes that could still turn out to be the start of
    /// one so a delimiter never reaches a streaming client.
    stream: StopStream,
    started: Instant,
    /// The token to feed at the next decode step (the one just sampled). `None`
    /// during prefill, where logits come from the prompt tail instead.
    pending_token: Option<LlamaToken>,
    /// Whether this sequence's KV holds media embeddings. Such a KV is never
    /// offered for prefix reuse: image positions are not tokens, and a token
    /// list that matched would describe a cache that holds something else.
    multimodal: bool,
    /// Whether the drafter has been started on this sequence, so it drafts.
    speculate: bool,
    /// Tokens spent inside the reasoning block so far.
    reasoning_tokens: u32,
    /// Reasoning tokens allowed before the block is closed for the model.
    reasoning_budget: Option<u32>,
    /// The template's close marker, tokenized once.
    close_tokens: Vec<LlamaToken>,
    /// Tokens the engine feeds instead of sampling: the close marker, once the
    /// budget is spent.
    forced: std::collections::VecDeque<LlamaToken>,
    /// Whether the budget has already closed the block once.
    budget_closed: bool,
    /// Media prefill waiting for the scheduler.
    #[cfg(feature = "mtmd")]
    pending_media: Option<MtmdInputChunks>,
}

/// One row of a step's batch, recorded so the step can be undone.
///
/// A decode that finds no room in the shared KV pool leaves the cache as it
/// was — each step is one micro-batch, so nothing was partly applied — but the
/// sequences have already advanced their positions and consumed their pending
/// tokens. Undoing that is what lets the step be retried after room is made,
/// instead of failing every sequence in it.
enum StepRow {
    Extend {
        slot: usize,
        token: LlamaToken,
    },
    PrefillPart {
        slot: usize,
        start: usize,
        added: usize,
    },
    PrefillDone {
        slot: usize,
        prompt: Vec<LlamaToken>,
        start: usize,
        added: usize,
    },
}

fn rollback_step(slots: &mut [Option<Sequence>], step: Vec<StepRow>) {
    for row in step.into_iter().rev() {
        match row {
            StepRow::Extend { slot, token } => {
                if let Some(s) = slots[slot].as_mut() {
                    s.resident.pop();
                    s.n_past -= 1;
                    s.pending_token = Some(token);
                }
            }
            StepRow::PrefillPart { slot, start, added } => {
                if let Some(s) = slots[slot].as_mut() {
                    let keep = s.resident.len().saturating_sub(added);
                    s.resident.truncate(keep);
                    s.n_past -= added as i32;
                    s.prefill_cursor = start;
                }
            }
            StepRow::PrefillDone { slot, prompt, start, added } => {
                if let Some(s) = slots[slot].as_mut() {
                    let keep = s.resident.len().saturating_sub(added);
                    s.resident.truncate(keep);
                    s.n_past -= added as i32;
                    s.prefill_cursor = start;
                    s.pending_prompt = Some(prompt);
                }
            }
        }
    }
}

/// Free the KV held by idle slots' cached prefixes. True when anything was
/// freed. Reuse is an optimisation; a request that cannot run is not.
fn evict_idle_prefixes(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    slots: &[Option<Sequence>],
    cached: &mut [CachedPrefix],
) -> bool {
    let mut freed = false;
    for (i, c) in cached.iter_mut().enumerate() {
        if slots[i].is_none() && !c.tokens.is_empty() {
            let _ = ctx.clear_kv_cache_seq(Some(i as u32), None, None);
            draft_seq_rm(draft, i as i32, None);
            c.tokens.clear();
            freed = true;
        }
    }
    freed
}

/// Stop the running sequence holding the most of the shared pool, so the
/// others can continue. Used only when evicting idle prefixes freed nothing.
fn fail_largest_sequence(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    model_id: &str,
) {
    let Some(idx) = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_ref().map(|q| (i, q.n_past)))
        .max_by_key(|(_, n)| *n)
        .map(|(i, _)| i)
    else {
        return;
    };
    if let Some(mut seq) = slots[idx].take() {
        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
        draft_seq_rm(draft, seq.seq_id, None);
        cached[idx].tokens.clear();
        warn!(
            "shared KV pool for {} is full: stopping the request holding {} positions so the others can continue",
            model_id, seq.n_past
        );
        seq.fail(Error::Inference(
            "the model's shared context is full, and this request held the most of it; \
             retry with a shorter prompt or a smaller max_tokens"
                .into(),
        ));
    }
}

/// Sample the next token for one sequence from logits row `logits_idx`, feed it
/// to the stream, and say whether the sequence is finished.
fn sample_into(
    model: &LlamaModel,
    ctx: &llama_cpp_2::context::LlamaContext,
    slots: &mut [Option<Sequence>],
    slot_idx: usize,
    logits_idx: i32,
    model_id: &str,
) -> bool {
    let Some(seq) = slots[slot_idx].as_mut() else {
        return false;
    };
    let token = next_token(seq, ctx, logits_idx);
    seq.sampler.accept(token);

    if model.is_eog_token(token) {
        return true;
    }
    let mut free_slot = false;
    match model.token_to_piece(token, &mut seq.decoder, true, None) {
        Ok(piece) => {
            let open = seq.stream.push(&piece, seq.token_tx.as_ref());
            seq.output_tokens += 1;
            if !open || seq.stream.hit_stop() {
                free_slot = true;
            }
        }
        Err(e) => {
            warn!("token decode failed on {}: {}", model_id, e);
            seq.output_tokens += 1;
        }
    }
    spend_reasoning(seq);
    if !free_slot {
        if seq.input_tokens as i32 + seq.output_tokens as i32 >= seq.max_pos {
            free_slot = true;
        } else {
            seq.pending_token = Some(token);
        }
    }
    free_slot
}

/// How many tokens a request may spend reasoning: what it asked for, or all of
/// `max_tokens` but a reserve for the answer.
///
/// Without a budget a reasoning model given a small `max_tokens` spends every
/// token inside the block and returns empty content, measured on qwen3.8-27b at
/// `max_tokens: 256`, where the reply was 1,053 characters of reasoning and no
/// answer. llama.cpp's server bounds this with `--reasoning-budget`; this is the
/// same bound, per request.
fn reasoning_budget(config: &GenerationConfig) -> Option<u32> {
    config.reasoning_budget.or_else(|| {
        let reserve = (config.max_tokens / 4).clamp(64, 2048);
        Some(config.max_tokens.saturating_sub(reserve))
    })
}

/// The tokens that close a reasoning block in this template: `</think>` and the
/// blank line Qwen's template puts after it, or Gemma 4's `<channel|>`.
fn close_marker_tokens(model: &LlamaModel, frame: ReasoningFrame) -> Vec<LlamaToken> {
    let text = if frame == ReasoningFrame::GEMMA4_THOUGHT || frame.close == ReasoningFrame::GEMMA4_THOUGHT.close {
        frame.close.to_string()
    } else {
        format!("{}\n\n", frame.close)
    };
    model.str_to_token(&text, AddBos::Never).unwrap_or_default()
}

/// The next token for a sequence: a forced one when the engine is closing its
/// reasoning block, otherwise a sample from its logits row.
fn next_token(seq: &mut Sequence, ctx: &llama_cpp_2::context::LlamaContext, logits_idx: i32) -> LlamaToken {
    match seq.forced.pop_front() {
        Some(token) => token,
        None => seq.sampler.sample(ctx, logits_idx),
    }
}

/// Count an emitted token against the reasoning budget, and once the budget is
/// spent queue the close marker, so the tokens after it are the answer.
fn spend_reasoning(seq: &mut Sequence) {
    if seq.budget_closed || !seq.forced.is_empty() || !seq.stream.in_reasoning() {
        return;
    }
    seq.reasoning_tokens += 1;
    if seq.reasoning_budget.is_some_and(|b| seq.reasoning_tokens >= b) && !seq.close_tokens.is_empty() {
        seq.forced.extend(seq.close_tokens.iter().copied());
        seq.budget_closed = true;
    }
}

/// Drop a sequence's positions from `p0` on in the draft context, which must
/// hold exactly what the target holds for every speculating sequence.
fn draft_seq_rm(draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>, seq_id: i32, p0: Option<u32>) {
    if let Some(d) = draft.as_mut() {
        let _ = d.clear_kv_cache_seq(Some(seq_id as u32), p0, None);
    }
}

/// Verify one sequence's drafted block.
///
/// Row `first_idx` holds the logits after the sequence's last committed token
/// and row `first_idx + i + 1` the logits after draft `i`. The sampler walks the
/// rows, and every token it samples is emitted, so the output is exactly what
/// plain decoding would have produced; a draft only saves the decode for the
/// tokens the sampler agreed with. Accepted drafts become committed KV; the
/// first disagreement (or the bonus token after a fully accepted block) becomes
/// the next pending token. Returns whether the sequence finished and how many
/// drafts it accepted.
fn sample_block_into(
    model: &LlamaModel,
    ctx: &llama_cpp_2::context::LlamaContext,
    slots: &mut [Option<Sequence>],
    slot_idx: usize,
    first_idx: i32,
    drafts: &[LlamaToken],
    model_id: &str,
) -> (bool, u16) {
    let Some(seq) = slots[slot_idx].as_mut() else {
        return (false, 0);
    };
    let mut accepted: u16 = 0;
    for i in 0..=drafts.len() {
        let token = next_token(seq, ctx, first_idx + i as i32);
        seq.sampler.accept(token);
        if model.is_eog_token(token) {
            return (true, accepted);
        }
        let matched = i < drafts.len() && token == drafts[i];
        let mut finished = false;
        match model.token_to_piece(token, &mut seq.decoder, true, None) {
            Ok(piece) => {
                let open = seq.stream.push(&piece, seq.token_tx.as_ref());
                if !open || seq.stream.hit_stop() {
                    finished = true;
                }
            }
            Err(e) => warn!("token decode failed on {}: {}", model_id, e),
        }
        seq.output_tokens += 1;
        spend_reasoning(seq);
        if matched {
            // Its KV row was written by the verify decode; keep it.
            accepted += 1;
            seq.resident.push(token);
            seq.n_past += 1;
        }
        if !finished && seq.input_tokens as i32 + seq.output_tokens as i32 >= seq.max_pos {
            finished = true;
        }
        if finished {
            return (true, accepted);
        }
        if !matched {
            seq.pending_token = Some(token);
            return (false, accepted);
        }
    }
    unreachable!("the row after the last draft never matches a draft")
}

impl Sequence {
    fn finish(mut self) {
        let elapsed = self.started.elapsed();
        let generation_time_ms = elapsed.as_millis() as u64;
        let tokens_per_second = if generation_time_ms > 0 {
            (self.output_tokens as f64) / (generation_time_ms as f64 / 1000.0)
        } else {
            0.0
        };
        // A stop sequence outranks the position ceiling, which outranks
        // end-of-generation: the sequence is what halted decoding, and the
        // trimmed text alone cannot tell the caller which of the three it was.
        let stop_reason = if self.stream.hit_stop() {
            StopReason::StopSequence
        } else if self.input_tokens as i32 + self.output_tokens as i32 >= self.max_pos {
            StopReason::Length
        } else {
            StopReason::Eos
        };
        let token_tx = self.token_tx.take();
        let (text, thinking) = self.stream.finish_parts(token_tx.as_ref());
        if let Some(result_tx) = self.result_tx.take() {
            let _ = result_tx.send(Ok(InferenceResult {
                text,
                thinking,
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                generation_time_ms,
                tokens_per_second,
                stop_reason,
                commitment: None,
            }));
        }
    }

    fn fail(&mut self, err: Error) {
        if let Some(result_tx) = self.result_tx.take() {
            let _ = result_tx.send(Err(err));
        }
    }
}

fn build_sampler(config: &GenerationConfig, n_vocab: i32) -> LlamaSampler {
    // The same chain the serial and speculative paths use. This engine used to
    // build its own, which dropped top_k, min_p and the frequency and presence
    // penalties a caller sent, so one request sampled differently depending on
    // which path served it.
    crate::sampling::build_sampler_chain(config, n_vocab)
}

#[allow(clippy::too_many_lines)]
fn scheduler_loop(
    model_id: &str,
    model: LlamaModel,
    backend: &LlamaBackend,
    context_length: u32,
    enable_thinking: bool,
    projector: Option<Projector>,
    speculation: Option<BatchSpeculation>,
    rx: &Receiver<BatchRequest>,
    shutdown_rx: &Receiver<()>,
) -> Result<()> {
    use std::num::NonZeroU32;

    let n_ctx = NonZeroU32::new(context_length).unwrap_or(NonZeroU32::new(8192).unwrap());

    // One long-lived context with max_slots() sequence slots. n_batch/n_ubatch
    // cover the interleaved prefill+extend batch.
    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_seq_max(max_slots() as u32)
        // One KV pool shared by every sequence. Without it `n_ctx` is split
        // evenly, so each request is capped at `n_ctx / slots` however little
        // the others use — at 32 slots a 131k model answers in 4k.
        .with_kv_unified(true)
        .with_n_batch(physical_batch() as u32)
        .with_n_ubatch(physical_batch() as u32);
    // Rejected drafts are rolled back out of the target's KV. A model whose
    // layers carry recurrent state cannot rewind by position, so it keeps this
    // many snapshots per sequence to roll back to.
    if let Some(s) = speculation.as_ref() {
        ctx_params = ctx_params.with_n_rs_seq(u32::from(s.n_max));
    }

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Other(format!("batch context init failed: {}", e)))?;

    let ctx_size = ctx.n_ctx() as i32;

    info!(
        "batch engine for {} online: {} slots, ctx={}",
        model_id,
        max_slots(),
        ctx_size
    );

    // Speculation: a draft context beside the target, one sequence per slot.
    // Anything that fails here costs speed, never serving: the engine runs
    // without speculation and says why.
    let (draft_model, spec_type, spec_n_max) = match speculation {
        Some(s) => (s.draft_model, Some(s.spec_type), s.n_max),
        None => (None, None, 0),
    };
    let mut draft_ctx = match spec_type {
        Some(kind) if spec_n_max > 0 => {
            let drafter: &LlamaModel = draft_model.as_ref().unwrap_or(&model);
            let mut params = LlamaContextParams::default()
                .with_n_ctx(Some(n_ctx))
                .with_n_seq_max(max_slots() as u32)
                .with_kv_unified(true)
                .with_n_batch(physical_batch() as u32)
                .with_n_ubatch(physical_batch() as u32);
            if kind == 0 {
                // The MTP head is a graph over the target's weights, built only
                // for an MTP-typed context.
                params = params.with_context_type(LlamaContextType::Mtp);
            }
            match drafter.new_context_with_ctx_other(backend, params, &ctx) {
                Ok(c) => Some(c),
                Err(e) => {
                    warn!("batch engine for {}: no draft context ({}), serving without speculation", model_id, e);
                    None
                }
            }
        }
        _ => None,
    };
    let mut spec = match (draft_ctx.as_ref(), spec_type) {
        (Some(dctx), Some(kind)) => {
            let params = MtpSpeculativeParams {
                n_max: i32::from(spec_n_max),
                n_min: 0,
                p_min: 0.0,
                spec_type: kind,
            };
            // SAFETY: both contexts live in this function and are dropped only
            // after `spec`, explicitly, at the end of it.
            match unsafe { SpeculativeBatch::new(&ctx, dctx, params, max_slots() as u32) } {
                Ok(s) => {
                    info!(
                        "batch engine for {}: speculative decoding on ({}, up to {} drafted tokens per sequence)",
                        model_id,
                        match kind {
                            1 => "DFlash drafter",
                            2 => "DSpark drafter",
                            _ => "MTP head",
                        },
                        spec_n_max
                    );
                    Some(s)
                }
                Err(e) => {
                    warn!("batch engine for {}: speculation unavailable ({}), serving without it", model_id, e);
                    None
                }
            }
        }
        _ => None,
    };
    if spec.is_none() {
        draft_ctx = None;
    }

    // Slots: None == free.
    let mut slots: Vec<Option<Sequence>> = (0..max_slots()).map(|_| None).collect();
    // What each slot's KV still holds between requests, so a follow-up turn can
    // start from the divergence instead of from zero.
    let mut cached: Vec<CachedPrefix> = (0..max_slots()).map(|_| CachedPrefix::default()).collect();
    let mut batch = LlamaBatch::new(physical_batch(), max_slots() as i32);

    loop {
        if shutdown_rx.try_recv().is_ok() {
            break;
        }

        // Cancellation sweep: free any slot whose client is gone before more GPU
        // work. A dropped result/stream receiver closes `result_tx`, covering
        // both streaming and non-streaming cancellation. Matches the consumer's
        // verified scheduler.
        for slot_idx in 0..slots.len() {
            let client_gone = slots[slot_idx]
                .as_ref()
                .and_then(|s| s.result_tx.as_ref())
                .is_some_and(tokio::sync::oneshot::Sender::is_closed);
            if client_gone && let Some(seq) = slots[slot_idx].take() {
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                draft_seq_rm(&mut draft_ctx, seq.seq_id, None);
                // The KV is gone, so the cache record must go with it. Leaving
                // it would promise the next request a prefix that is no longer
                // resident, and it would prefill from a position the cache
                // cannot satisfy.
                cached[slot_idx].tokens.clear();
            }
        }

        let active = slots.iter().filter(|s| s.is_some()).count();

        // Admit new requests into free slots. When idle, block (bounded) on the
        // first one so the thread parks instead of spinning; then drain the rest
        // non-blocking.
        if active == 0 {
            match rx.recv_timeout(IDLE_POLL) {
                Ok(req) => {
                    if let Some(r) = admit(
                        &model,
                        ctx_size,
                        &mut slots,
                        &mut cached,
                        req,
                        enable_thinking,
                        projector.as_ref(),
                    ) {
                        let reuse_slot = r.slot_idx;
                        apply_prefix_reuse(&mut ctx, &mut slots, &mut cached, r);
                        let held = cached[reuse_slot].tokens.len() as u32;
                        draft_seq_rm(&mut draft_ctx, reuse_slot as i32, Some(held));
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        // Fill any remaining free slots without blocking.
        while slots.iter().any(|s| s.is_none()) {
            match rx.try_recv() {
                Ok(req) => {
                    if let Some(r) = admit(
                        &model,
                        ctx_size,
                        &mut slots,
                        &mut cached,
                        req,
                        enable_thinking,
                        projector.as_ref(),
                    ) {
                        let reuse_slot = r.slot_idx;
                        apply_prefix_reuse(&mut ctx, &mut slots, &mut cached, r);
                        let held = cached[reuse_slot].tokens.len() as u32;
                        draft_seq_rm(&mut draft_ctx, reuse_slot as i32, Some(held));
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // Media prefill, at most one per step. `eval_chunks` encodes the media
        // and decodes its embeddings for that one sequence on the shared
        // context, switching to non-causal attention where the projector needs
        // it; that switch is context-wide, which is why media cannot share a
        // batch with other sequences' tokens. One per step bounds how long
        // decoding sequences wait behind an image.
        #[cfg(feature = "mtmd")]
        if let Some(projector) = projector.as_ref() {
            let next = slots
                .iter()
                .position(|s| s.as_ref().is_some_and(|q| q.pending_media.is_some()));
            if let Some(slot_idx) = next {
                let (chunks, n_past, seq_id) = {
                    let seq = slots[slot_idx].as_mut().expect("found above");
                    (
                        seq.pending_media.take().expect("filtered above"),
                        seq.n_past,
                        seq.seq_id,
                    )
                };
                let n_batch = ctx.n_batch() as i32;
                match chunks.eval_chunks(projector, &ctx, n_past, seq_id, n_batch, true) {
                    Ok(new_n_past) => {
                        if let Some(seq) = slots[slot_idx].as_mut() {
                            seq.n_past = new_n_past;
                        }
                        if sample_into(&model, &ctx, &mut slots, slot_idx, -1, model_id) {
                            finalize_and_free(&mut ctx, &mut draft_ctx, &mut slots, &mut cached, slot_idx);
                        }
                    }
                    Err(e) => {
                        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                        draft_seq_rm(&mut draft_ctx, seq_id, None);
                        if evict_idle_prefixes(&mut ctx, &mut draft_ctx, &slots, &mut cached) {
                            // Room was made; try this request again next step.
                            if let Some(seq) = slots[slot_idx].as_mut() {
                                seq.n_past = 0;
                                seq.pending_media = Some(chunks);
                            }
                        } else if let Some(mut seq) = slots[slot_idx].take() {
                            cached[slot_idx].tokens.clear();
                            seq.fail(Error::Inference(format!(
                                "multimodal prefill failed: {}",
                                e
                            )));
                        }
                    }
                }
            }
        }

        // Draft for every sequence that is generating. The drafter writes its
        // block into the draft context, which is then trimmed back to what the
        // target holds before the verify batch reaches it.
        let mut drafts: Vec<Vec<LlamaToken>> = vec![Vec::new(); slots.len()];
        let generating = slots
            .iter()
            .flatten()
            .filter(|s| s.pending_token.is_some())
            .count();
        if let Some(sp) = spec.as_mut().filter(|_| generating <= spec_max_active(spec_type.unwrap_or(0))) {
            let mut asked = false;
            for seq in slots.iter().flatten() {
                let Some(id_last) = seq.pending_token else {
                    continue;
                };
                // A sequence whose block is being closed feeds fixed tokens next.
                if !seq.speculate || !seq.forced.is_empty() {
                    continue;
                }
                // Leave room for the pending token and stay under the ceiling.
                let room = seq.max_pos - seq.n_past - 1;
                let n = i32::from(spec_n_max).min(room);
                if n > 0 && sp.request(seq.seq_id, n, seq.n_past, id_last, &seq.resident).is_ok() {
                    asked = true;
                }
            }
            if asked {
                match sp.draft() {
                    Ok(()) => {
                        for (slot_idx, maybe_seq) in slots.iter().enumerate() {
                            if let Some(seq) = maybe_seq
                                && seq.speculate
                                && seq.pending_token.is_some()
                            {
                                drafts[slot_idx] = sp.result(seq.seq_id).unwrap_or_default();
                                draft_seq_rm(&mut draft_ctx, seq.seq_id, Some(seq.n_past as u32));
                            }
                        }
                    }
                    Err(e) => warn!("draft step failed on {}: {}", model_id, e),
                }
            }
        }

        // Build the interleaved batch. `logits_slot` maps a batch logits index
        // back to the slot that owns it.
        batch.clear();
        let mut logits_slot: Vec<(i32, usize)> = Vec::with_capacity(max_slots());
        let mut step: Vec<StepRow> = Vec::new();

        // Extension first: every running sequence contributes exactly one token,
        // so a slot mid-prefill can never hold back the slots already generating.
        for (slot_idx, maybe_seq) in slots.iter_mut().enumerate() {
            let Some(seq) = maybe_seq.as_mut() else {
                continue;
            };
            let Some(tok) = seq.pending_token.take() else {
                continue;
            };
            if batch.add(tok, seq.n_past, &[seq.seq_id], true).is_err() {
                seq.pending_token = Some(tok);
                continue;
            }
            // Generated tokens land in KV too, so a follow-up turn that repeats
            // this answer as history reuses it rather than re-decoding it.
            seq.resident.push(tok);
            seq.n_past += 1;
            logits_slot.push((batch.n_tokens() - 1, slot_idx));
            step.push(StepRow::Extend {
                slot: slot_idx,
                token: tok,
            });
            // The drafted block follows its token, every row with logits, so the
            // one decode scores the pending token and each draft.
            let mut n_added = 0usize;
            for (k, draft_tok) in drafts[slot_idx].iter().enumerate() {
                if batch.add(*draft_tok, seq.n_past + k as i32, &[seq.seq_id], true).is_err() {
                    break;
                }
                n_added += 1;
            }
            drafts[slot_idx].truncate(n_added);
        }

        // Prefill with what capacity is left, capped per slot.
        for (slot_idx, maybe_seq) in slots.iter_mut().enumerate() {
            let Some(seq) = maybe_seq.as_mut() else {
                continue;
            };
            let Some(prompt) = seq.pending_prompt.take() else {
                continue;
            };

            let room = physical_batch()
                .saturating_sub(batch.n_tokens() as usize)
                .min(PREFILL_CHUNK);
            let start = seq.prefill_cursor;
            let end = prompt.len().min(start + room);
            let last = prompt.len() - 1;

            let mut cursor = start;
            while cursor < end {
                if batch
                    .add(prompt[cursor], seq.n_past, &[seq.seq_id], cursor == last)
                    .is_err()
                {
                    break;
                }
                seq.resident.push(prompt[cursor]);
                seq.n_past += 1;
                cursor += 1;
            }

            let added = cursor - start;
            if cursor > last {
                // Whole prompt committed; its tail carries this slot's logits.
                seq.prefill_cursor = 0;
                logits_slot.push((batch.n_tokens() - 1, slot_idx));
                step.push(StepRow::PrefillDone {
                    slot: slot_idx,
                    prompt,
                    start,
                    added,
                });
            } else {
                // More prompt to go — no logits from this slot this step.
                seq.prefill_cursor = cursor;
                seq.pending_prompt = Some(prompt);
                step.push(StepRow::PrefillPart {
                    slot: slot_idx,
                    start,
                    added,
                });
            }
        }

        if batch.n_tokens() == 0 {
            continue;
        }

        if let Err(e) = ctx.decode(&mut batch) {
            if matches!(e, llama_cpp_2::DecodeError::NoKvCacheSlot) {
                // The shared pool is full. Undo the step, make room by dropping
                // idle cached prefixes, and retry next step. If there was nothing
                // idle to drop, stop the request holding the most of the pool
                // rather than every request in the batch.
                rollback_step(&mut slots, step);
                if evict_idle_prefixes(&mut ctx, &mut draft_ctx, &slots, &mut cached) {
                    info!("shared KV pool for {} was full: dropped idle cached prefixes", model_id);
                } else {
                    fail_largest_sequence(&mut ctx, &mut draft_ctx, &mut slots, &mut cached, model_id);
                }
                continue;
            }
            // Any other decode failure is fatal to every sequence in this step.
            for (_idx, slot_idx) in &logits_slot {
                if let Some(mut seq) = slots[*slot_idx].take() {
                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                    draft_seq_rm(&mut draft_ctx, seq.seq_id, None);
                    // Same reason as the cancellation sweep: a discarded KV must
                    // not leave a cache record behind, or one decode failure
                    // becomes a permanent one for every later request on the
                    // slot.
                    cached[*slot_idx].tokens.clear();
                    seq.fail(Error::Inference(format!("decode failed: {}", e)));
                }
            }
            continue;
        }

        // The drafter follows every decode the target makes, prompts included.
        let lost_step = match spec.as_mut() {
            Some(sp) => sp.process(&batch).err(),
            None => None,
        };
        if let Some(e) = lost_step {
            warn!(
                "the drafter for {} lost step with the target ({}); serving without speculation",
                model_id, e
            );
            spec = None;
            draft_ctx = None;
            for seq in slots.iter_mut().flatten() {
                seq.speculate = false;
            }
            for block in &mut drafts {
                block.clear();
            }
        }
        // A prompt that finished prefilling this step starts the drafter on it.
        // Media sequences never speculate: their image positions were decoded
        // outside any batch the drafter saw.
        if let Some(sp) = spec.as_mut() {
            for row in &step {
                if let StepRow::PrefillDone { slot, .. } = row
                    && let Some(seq) = slots[*slot].as_mut()
                    && !seq.multimodal
                {
                    seq.speculate = sp.begin(seq.seq_id, &seq.resident).is_ok();
                }
            }
        }

        // Sample each sequence from its own logits index.
        for (logits_idx, slot_idx) in logits_slot {
            let block = std::mem::take(&mut drafts[slot_idx]);
            let done = if block.is_empty() {
                sample_into(&model, &ctx, &mut slots, slot_idx, logits_idx, model_id)
            } else {
                let (done, accepted) =
                    sample_block_into(&model, &ctx, &mut slots, slot_idx, logits_idx, &block, model_id);
                if let Some(seq) = slots[slot_idx].as_ref() {
                    // Rejected drafts were written into both caches; take them out.
                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), Some(seq.n_past as u32), None);
                    draft_seq_rm(&mut draft_ctx, seq.seq_id, Some(seq.n_past as u32));
                    if let Some(sp) = spec.as_mut() {
                        let _ = sp.accept(seq.seq_id, accepted);
                    }
                }
                done
            };
            if done {
                finalize_and_free(&mut ctx, &mut draft_ctx, &mut slots, &mut cached, slot_idx);
            }
        }
    }

    // Drain remaining slots on shutdown.
    for slot in slots.iter_mut() {
        if let Some(mut seq) = slot.take() {
            seq.fail(Error::Other("batch engine shutting down".into()));
        }
    }

    // The speculator points into both contexts, and the draft context into
    // the drafter model; release them in that order.
    drop(spec);
    drop(draft_ctx);
    drop(draft_model);
    // The projector points into the model; release it first.
    drop(projector);
    info!("batch engine for {} stopped", model_id);
    Ok(())
}

/// Finalize a finished sequence and free its slot, keeping its KV for reuse.
fn finalize_and_free(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    slot_idx: usize,
) {
    if let Some(seq) = slots[slot_idx].take() {
        if seq.multimodal {
            // Media embeddings occupy positions no token list describes, so this
            // KV is dropped rather than offered to the next request as a prefix.
            let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
            draft_seq_rm(draft, seq.seq_id, None);
            cached[slot_idx].tokens.clear();
        } else {
            // Keep the KV. The next request on this slot is very often the same
            // conversation one turn later, and re-decoding a prefix we already
            // hold is the largest avoidable cost in an agent loop. `admit` trims
            // whatever the next prompt diverges from.
            cached[slot_idx].tokens = seq.resident.clone();
        }
        seq.finish();
    }
}

/// Drop the part of a slot's KV cache that the new prompt diverges from, and
/// record what the slot now holds.
///
/// Order matters: the trim must happen before the scheduler decodes anything
/// into this sequence, or the new tokens would be written on top of positions
/// still holding the previous request's.
fn apply_prefix_reuse(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    r: PrefixReuse,
) {
    // Everything from the divergence onward was computed under a different
    // prefix, so it is wrong rather than merely old. `None` for the end means
    // "to the end of the sequence".
    let seq = r.slot_idx as u32;

    // Nothing past the reused span means nothing to drop. This is the ordinary
    // agent turn — the conversation grew, so the new prompt starts with the
    // whole of the old one — and it is also the only case that works on a model
    // whose layers carry recurrent state, because such a cache can be cleared
    // but not rewound to an arbitrary position. Asking it to trim here would be
    // refused and would cost us the reuse we already have.
    if r.reused > 0 && r.reused == r.cached_len {
        cached[r.slot_idx].tokens = r.prompt[..r.reused].to_vec();
        tracing::info!(
            slot = r.slot_idx,
            reused_tokens = r.reused,
            prompt_tokens = r.prompt.len(),
            "prefix cache hit (extension, no trim needed)"
        );
        return;
    }

    let removed = ctx
        .clear_kv_cache_seq(Some(seq), Some(r.reused as u32), None)
        .unwrap_or(false);

    // Trust the trim only after confirming it. `llama_memory_seq_rm` returns
    // true without removing anything when a cache shares cells, and a cache
    // that kept its old positions is not a slow path — on an M-RoPE model the
    // next batch must start strictly beyond the highest cached position, so a
    // stale entry makes the request unschedulable and it fails outright.
    // Verify against the cache itself rather than the return value.
    let stale = ctx.kv_cache_seq_pos_max(r.slot_idx as i32) >= r.reused as i32;

    if !removed || stale {
        // Could not trim to the divergence. Drop the sequence's KV entirely and
        // prefill from zero: strictly slower, but correct, and self-healing
        // because the slot starts clean on the next turn either way.
        let _ = ctx.clear_kv_cache_seq(Some(seq), None, None);
        cached[r.slot_idx].tokens.clear();
        if let Some(s) = slots[r.slot_idx].as_mut() {
            s.prefill_cursor = 0;
            s.n_past = 0;
            s.resident.clear();
        }
        tracing::info!(
            slot = r.slot_idx,
            wanted_reuse = r.reused,
            removed,
            "prefix trim refused by the KV cache; prefilling from zero"
        );
        return;
    }

    cached[r.slot_idx].tokens = r.prompt[..r.reused].to_vec();
    if r.reused > 0 {
        tracing::info!(
            slot = r.slot_idx,
            reused_tokens = r.reused,
            prompt_tokens = r.prompt.len(),
            "prefix cache hit"
        );
    }
}

/// Tokenize an admitted request into the first free slot and stage its prompt
/// for prefill.
fn admit(
    model: &LlamaModel,
    ctx_size: i32,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    req: BatchRequest,
    enable_thinking: bool,
    projector: Option<&Projector>,
) -> Option<PrefixReuse> {
    let Some(slot_idx) = slots.iter().position(|s| s.is_none()) else {
        // No free slot — reject rather than block. Caller sheds load.
        let _ = req.result_tx.send(Err(Error::QueueFull {
            model_id: "batched".into(),
            waiting: slots.len(),
            max: slots.len(),
        }));
        return None;
    };
    let seq_id = slot_idx as i32;

    // The engine's thinking mode is the default; a request may override it.
    // This is what lets one served model answer plainly to API callers and
    // show its reasoning to a client that asked to see it.
    let enable_thinking = req.config.enable_thinking.unwrap_or(enable_thinking);
    // The effort level is a thinking-mode knob; with thinking off the template
    // never reads it, and some templates reject it outright.
    let reasoning = if enable_thinking {
        req.config.reasoning_effort.as_deref().map(|value| {
            (
                req.config.reasoning_kwarg.as_deref().unwrap_or("reasoning_effort"),
                value,
            )
        })
    } else {
        None
    };
    let prompt = match render_prompt(model, &req.prompt, enable_thinking, reasoning) {
        Ok(p) => p,
        Err(e) => {
            let _ = req.result_tx.send(Err(e));
            return None;
        }
    };
    let frame = ReasoningFrame::for_prompt(&prompt);
    let reasoning_budget = reasoning_budget(&req.config);
    let close_tokens = close_marker_tokens(model, frame);

    if !req.media.is_empty() {
        return admit_media(model, ctx_size, slots, cached, req, prompt, projector, slot_idx);
    }

    let tokens = match model.str_to_token(&prompt, AddBos::Always) {
        Ok(t) => t,
        Err(e) => {
            let _ = req
                .result_tx
                .send(Err(Error::Other(format!("tokenization failed: {}", e))));
            return None;
        }
    };
    if tokens.is_empty() {
        let _ = req.result_tx.send(Err(Error::Other("empty prompt".into())));
        return None;
    }

    let input_tokens = tokens.len() as u32;
    // A prompt that alone fills the context leaves no room to generate.
    if input_tokens as i32 >= ctx_size {
        let _ = req.result_tx.send(Err(Error::Inference(format!(
            "prompt of {} tokens exceeds context window {}",
            input_tokens, ctx_size
        ))));
        return None;
    }
    let max_pos = ctx_size.min(input_tokens as i32 + req.config.max_tokens as i32);

    // Reuse whatever of this slot's KV already matches. The tokens are
    // identical up to `reuse`, so those positions are already correct in the
    // cache and only the divergent tail needs decoding. Everything at or past
    // the divergence is dropped, because a KV entry computed under a different
    // prefix is wrong, not merely stale.
    let cached_len = cached[slot_idx].tokens.len();
    let reuse = common_prefix_len(&cached[slot_idx].tokens, &tokens);
    // A one-token overlap (the BOS) is not worth the bookkeeping, and reusing
    // the entire prompt would leave nothing to decode and no logits to sample
    // from — so always leave at least the final token to be processed.
    let reuse = if reuse < 2 {
        0
    } else {
        reuse.min(tokens.len() - 1)
    };

    slots[slot_idx] = Some(Sequence {
        seq_id,
        sampler: build_sampler(&req.config, model.n_vocab()),
        token_tx: req.token_tx,
        result_tx: Some(req.result_tx),
        decoder: encoding_rs::UTF_8.new_decoder(),
        pending_prompt: Some(tokens.clone()),
        // Prefill resumes at the divergence rather than at zero.
        prefill_cursor: reuse,
        n_past: reuse as i32,
        // The reused prefix is already in KV, so it counts as resident.
        resident: tokens[..reuse].to_vec(),
        input_tokens,
        output_tokens: 0,
        max_pos,
        stream: StopStream::new(req.config.stop).framed(frame),
        started: Instant::now(),
        speculate: false,
        reasoning_tokens: 0,
        reasoning_budget,
        close_tokens,
        forced: std::collections::VecDeque::new(),
        budget_closed: false,
        pending_token: None,
        multimodal: false,
        #[cfg(feature = "mtmd")]
        pending_media: None,
    });

    Some(PrefixReuse {
        slot_idx,
        reused: reuse,
        cached_len,
        prompt: tokens,
    })
}

/// Admit a request that carries media into `slot_idx`.
///
/// The slot's KV is cleared rather than reused, and the prompt is tokenized by
/// the projector into text and media chunks that the scheduler prefills on its
/// own step.
#[cfg(feature = "mtmd")]
#[allow(clippy::too_many_arguments)]
fn admit_media(
    model: &LlamaModel,
    ctx_size: i32,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    req: BatchRequest,
    prompt: String,
    projector: Option<&Projector>,
    slot_idx: usize,
) -> Option<PrefixReuse> {
    let frame = ReasoningFrame::for_prompt(&prompt);
    let reasoning_budget = reasoning_budget(&req.config);
    let close_tokens = close_marker_tokens(model, frame);
    let Some(projector) = projector else {
        let _ = req.result_tx.send(Err(Error::Inference(
            "this model is served text-only: it has no projector, so it cannot take image or \
             audio attachments"
                .into(),
        )));
        return None;
    };
    let mut bitmaps = Vec::with_capacity(req.media.len());
    for (i, bytes) in req.media.iter().enumerate() {
        let bitmap = match MtmdBitmap::from_buffer(projector, bytes, false) {
            Ok(b) => b,
            Err(e) => {
                let _ = req.result_tx.send(Err(Error::Inference(format!(
                    "attachment {} could not be decoded: {}",
                    i, e
                ))));
                return None;
            }
        };
        let (kind, supported) = if bitmap.is_audio() {
            ("audio", projector.support_audio())
        } else {
            ("an image", projector.support_vision())
        };
        if !supported {
            let _ = req.result_tx.send(Err(Error::Inference(format!(
                "attachment {} is {}, which this projector has no tower for",
                i, kind
            ))));
            return None;
        }
        bitmaps.push(bitmap);
    }
    let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    let chunks = match projector.tokenize(
        MtmdInputText {
            text: prompt,
            add_special: true,
            parse_special: true,
        },
        &refs,
    ) {
        Ok(c) => c,
        Err(e) => {
            let _ = req
                .result_tx
                .send(Err(Error::Inference(format!("multimodal tokenization failed: {}", e))));
            return None;
        }
    };
    let input_tokens = chunks.total_tokens() as u32;
    if input_tokens == 0 {
        let _ = req.result_tx.send(Err(Error::Other("empty prompt".into())));
        return None;
    }
    if input_tokens as i32 >= ctx_size {
        let _ = req.result_tx.send(Err(Error::Inference(format!(
            "prompt of {} tokens exceeds context window {}",
            input_tokens, ctx_size
        ))));
        return None;
    }
    let max_pos = ctx_size.min(input_tokens as i32 + req.config.max_tokens as i32);
    let cached_len = cached[slot_idx].tokens.len();
    slots[slot_idx] = Some(Sequence {
        seq_id: slot_idx as i32,
        sampler: build_sampler(&req.config, model.n_vocab()),
        token_tx: req.token_tx,
        result_tx: Some(req.result_tx),
        decoder: encoding_rs::UTF_8.new_decoder(),
        pending_prompt: None,
        prefill_cursor: 0,
        n_past: 0,
        resident: Vec::new(),
        input_tokens,
        output_tokens: 0,
        max_pos,
        stream: StopStream::new(req.config.stop).framed(frame),
        started: Instant::now(),
        speculate: false,
        reasoning_tokens: 0,
        reasoning_budget,
        close_tokens,
        forced: std::collections::VecDeque::new(),
        budget_closed: false,
        pending_token: None,
        multimodal: true,
        pending_media: Some(chunks),
    });
    // A reuse of zero clears the slot's KV and records it as holding nothing.
    Some(PrefixReuse {
        slot_idx,
        reused: 0,
        cached_len,
        prompt: Vec::new(),
    })
}

/// Refuse media in a build without mtmd.
#[cfg(not(feature = "mtmd"))]
#[allow(clippy::too_many_arguments)]
fn admit_media(
    _model: &LlamaModel,
    _ctx_size: i32,
    _slots: &mut [Option<Sequence>],
    _cached: &mut [CachedPrefix],
    req: BatchRequest,
    _prompt: String,
    _projector: Option<&Projector>,
    _slot_idx: usize,
) -> Option<PrefixReuse> {
    let _ = req.result_tx.send(Err(Error::Inference(
        "this engine was built without multimodal support".into(),
    )));
    None
}

/// What `admit` decided about an incoming request's prefix.
///
/// Returned rather than acted on inside `admit`, because trimming the KV cache
/// needs the context and `admit` deliberately does not take it — tokenizing
/// must not be able to touch the cache.
struct PrefixReuse {
    slot_idx: usize,
    /// Tokens taken from cache instead of re-decoded.
    reused: usize,
    /// What the slot held before this request. When the whole of it is reused
    /// the prompt merely extends the cache and nothing has to be dropped, which
    /// is the difference between a trim we can skip and one that must succeed.
    cached_len: usize,
    prompt: Vec<LlamaToken>,
}

/// Render a [`BatchPrompt`] to the final prompt string fed to the tokenizer.
///
/// Backend-generic: uses the model's own embedded GGUF chat template, falling
/// back to ChatML. A host with model-specific templating (e.g. tool-calling
/// grammars or arch-specific native chat formats) renders upstream and submits
/// a [`BatchPrompt::Raw`]. `_enable_thinking` is reserved for templates that
/// take a reasoning-toggle variable; this binding's `apply_chat_template` does
/// not, so the model's template default applies.
/// Render the prompt the engine will prefill.
///
/// The thinking switch and effort level are template variables, so they only
/// exist if the render passes them: the model's own Jinja template reads
/// `enable_thinking` (a Qwen-family template prefills `<think>\n\n</think>\n\n`
/// when it is false, and opens a `<think>` block when it is true or absent)
/// and `reasoning_effort`. The legacy three-argument render carries neither,
/// which is how a model served here thought on every request whatever the
/// caller sent — the flag was plumbed all the way to a function that dropped
/// it. Measured on qwen3.8-flash-next: 1,200-3,800 characters of reasoning
/// with `enable_thinking: false`, and an empty answer at a 300-token budget.
///
/// Rendering goes through the OpenAI-compatible entry point with the
/// variables set; a template that cannot be applied that way falls back to
/// the legacy render, then to ChatML, exactly as before.
fn render_prompt(
    model: &LlamaModel,
    prompt: &BatchPrompt,
    enable_thinking: bool,
    reasoning: Option<(&str, &str)>,
) -> Result<String> {
    match prompt {
        BatchPrompt::Raw(s) => Ok(s.clone()),
        BatchPrompt::Chat(messages) => {
            if let Ok(tmpl) = model.chat_template(None) {
                if let Ok(messages_json) = serde_json::to_string(messages) {
                    let kwargs = reasoning.and_then(|(name, value)| {
                        let mut map = serde_json::Map::new();
                        map.insert(name.to_string(), serde_json::Value::String(value.to_string()));
                        serde_json::to_string(&serde_json::Value::Object(map)).ok()
                    });
                    let params = OpenAIChatTemplateParams {
                        messages_json: &messages_json,
                        tools_json: None,
                        tool_choice: None,
                        json_schema: None,
                        grammar: None,
                        reasoning_format: None,
                        chat_template_kwargs: kwargs.as_deref(),
                        add_generation_prompt: true,
                        use_jinja: true,
                        parallel_tool_calls: false,
                        enable_thinking,
                        // The tokenizer adds BOS at prefill (`AddBos::Always`);
                        // adding it here as well would double it.
                        add_bos: false,
                        add_eos: false,
                        parse_tool_calls: false,
                        force_pure_content: false,
                    };
                    if let Ok(rendered) = model.apply_chat_template_oaicompat(&tmpl, &params)
                        && !rendered.prompt.trim().is_empty()
                    {
                        return Ok(rendered.prompt);
                    }
                }

                // Legacy render: no template variables, so thinking follows the
                // template's own default. Kept only as the fallback for a
                // template the Jinja path cannot apply.
                let llama_messages: Vec<LlamaChatMessage> = messages
                    .iter()
                    .map(|m| {
                        LlamaChatMessage::new(m.role.clone(), m.content.clone())
                            .map_err(|e| Error::Other(format!("Invalid chat message: {}", e)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if let Ok(rendered) = model.apply_chat_template(&tmpl, &llama_messages, true)
                    && !rendered.trim().is_empty()
                {
                    return Ok(rendered);
                }
            }
            Ok(render_chatml_prompt(messages))
        }
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    /// The decision `admit` makes, extracted so it can be tested without a
    /// model, a context, or a GPU.
    fn reuse_for(cached: &[LlamaToken], prompt: &[LlamaToken]) -> usize {
        let r = common_prefix_len(cached, prompt);
        if r < 2 { 0 } else { r.min(prompt.len() - 1) }
    }

    #[test]
    fn a_follow_up_turn_reuses_the_conversation_so_far() {
        // The agent case: turn two repeats turn one verbatim and appends.
        let turn1 = toks(&[1, 2, 3, 4, 5]);
        let turn2 = toks(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            reuse_for(&turn1, &turn2),
            5,
            "the whole prior turn is already in KV"
        );
    }

    #[test]
    fn a_different_conversation_reuses_nothing() {
        let cached = toks(&[1, 2, 3, 4]);
        let other = toks(&[9, 8, 7, 6]);
        assert_eq!(reuse_for(&cached, &other), 0);
    }

    /// Whether the cache has to be trimmed at all, which decides whether reuse
    /// survives on a model whose state cannot be rewound.
    fn needs_trim(cached: &[LlamaToken], prompt: &[LlamaToken]) -> bool {
        let reuse = reuse_for(cached, prompt);
        !(reuse > 0 && reuse == cached.len())
    }

    #[test]
    fn extending_a_conversation_needs_no_trim() {
        // Turn two repeats turn one verbatim and appends: everything cached is
        // still a prefix, so there is nothing to drop and the reuse holds even
        // where a partial trim would be refused.
        let turn1 = toks(&[1, 2, 3, 4, 5]);
        let turn2 = toks(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(!needs_trim(&turn1, &turn2));
    }

    #[test]
    fn diverging_from_the_cache_needs_a_trim() {
        // The cached tail (4, 5) is not in the new prompt, so those entries are
        // wrong rather than stale and cannot simply be kept.
        let cached = toks(&[1, 2, 3, 4, 5]);
        let prompt = toks(&[1, 2, 3, 9, 9, 9]);
        assert!(needs_trim(&cached, &prompt));
    }

    #[test]
    fn divergence_mid_prompt_stops_the_reuse_there() {
        // Everything past the divergence was computed under a different prefix,
        // so it is wrong rather than stale and must not be reused.
        let cached = toks(&[1, 2, 3, 40, 50]);
        let prompt = toks(&[1, 2, 3, 41, 51]);
        assert_eq!(reuse_for(&cached, &prompt), 3);
    }

    #[test]
    fn an_identical_prompt_still_leaves_a_token_to_decode() {
        // Reusing everything would leave no token to run through the model and
        // so no logits to sample the reply from.
        let same = toks(&[1, 2, 3, 4, 5]);
        assert_eq!(reuse_for(&same, &same), 4, "must hold back the final token");
    }

    #[test]
    fn a_bos_only_overlap_is_not_worth_reusing() {
        let cached = toks(&[1, 77, 78]);
        let prompt = toks(&[1, 90, 91]);
        assert_eq!(reuse_for(&cached, &prompt), 0);
    }

    #[test]
    fn a_cold_slot_reuses_nothing() {
        assert_eq!(reuse_for(&[], &toks(&[1, 2, 3])), 0);
    }

    #[test]
    fn a_shorter_prompt_than_the_cache_is_bounded_by_the_prompt() {
        // Cache holds a long conversation; the new prompt is a prefix of it.
        let cached = toks(&[1, 2, 3, 4, 5, 6, 7]);
        let prompt = toks(&[1, 2, 3]);
        assert_eq!(
            reuse_for(&cached, &prompt),
            2,
            "never past the prompt's own end"
        );
    }
}
