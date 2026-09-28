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
//! Each in-flight request owns one slot, and its `seq_id` is that slot's index:
//! `slots[i]` always holds the sequence decoding into KV sequence `i`. When a
//! request finishes, its slot keeps the KV (and the checkpoints taken while it
//! prefilled) for the next request that shares its prefix; see
//! [`crate::prefix_cache`] for what can be reused on which kind of model.
//!
//! ## Scheduler loop
//!
//! One dedicated OS thread per model owns the `LlamaModel` and its
//! `LlamaContext`. Each iteration admits waiting requests into free slots,
//! extends every running sequence by its last sampled token, spends the
//! remaining batch capacity prefilling prompts, runs one `llama_decode`, then
//! samples each sequence from its own logits with its own sampler.
//!
//! ## Moving a sequence between engines
//!
//! An engine spawned with [`BatchEngine::spawn_with_migration`] reserves one
//! extra KV sequence id past the slots as a staging area, and can move a
//! running sequence to another engine serving the same weights without the
//! destination re-reading its prompt. A request submitted with
//! [`BatchEngine::submit_tracked`] gets a [`SequenceTicket`];
//! [`BatchEngine::export_sequence`] returns the positions its slot gained since
//! the previous export as an encoded [`KvBlob`](crate::KvBlob) (the attention
//! cells, and on a model with recurrent layers the recurrent state at the last
//! position), and [`BatchEngine::detach_sequence`] exports the final delta and
//! takes the sequence out of the engine. [`BatchEngine::resume`] rebuilds it in
//! a free slot of the destination from the chain of blobs and continues
//! decoding from the token it had sampled last.
//!
//! Exports are taken by the scheduler thread between two decode steps, when
//! every slot's cache is consistent, and cost the other slots only the time to
//! copy the new positions out. A sequence still prefilling its prompt, closing
//! a reasoning block, or whose last speculative step was rolled back on a
//! recurrent model is exported at the first step boundary where it is not.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError, TrySendError};
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
use crate::kv_migration::{ModelFingerprint, SequenceExport, SequenceImport};
use crate::prefix_cache::{
    CheckpointBudget, CheckpointStore, MediaSpan, MemoryTraits, Namespace, PrefixId, Reuse, Rewind, TurnEnds,
};
use crate::prompt::render_chatml_prompt;
use crate::result::{ChatMessage, InferenceResult, StopReason};
use crate::stream::{ReasoningFrame, StopStream};
use crate::toploc::{InferenceCommitment, StepRecord};
use llama_cpp_2::context::params::LlamaContextType;
use llama_cpp_2::speculative::{MtpSpeculativeParams, SpeculativeBatch};

/// Number of concurrent sequence slots a batched context serves by default.
/// This is the KV-cache `n_seq_max` and the ceiling on requests decoded in one
/// step. llama.cpp divides the context across it, so the per-request window is
/// `n_ctx / n_seq_max`.
const MAX_SLOTS_DEFAULT: usize = 32;

/// Most sequences generating at once for which the engine still drafts,
/// overridable with `BATCH_SPEC_MAX_ACTIVE` (0 turns drafting off).
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
    std::env::var("BATCH_SPEC_MAX_ACTIVE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(if spec_type == 0 { 2 } else { usize::MAX })
}

/// Sequence slots, overridable with `BATCH_MAX_SLOTS`.
///
/// This is `n_seq_max`. On a device that cannot afford `32 x` a useful window,
/// fewer slots buy back per-request context. The host's admission layer must
/// not advertise more concurrency than this — a request is only ever admitted
/// into one of these slots, so `n_seq_max` is the true concurrent capacity.
pub fn max_slots() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BATCH_MAX_SLOTS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MAX_SLOTS_DEFAULT)
    })
}

/// Physical batch capacity for a single `llama_decode`, overridable with
/// `BATCH_PHYSICAL_BATCH`. Sets `n_batch`/`n_ubatch`; the compute buffer
/// scales with it.
const PHYSICAL_BATCH_DEFAULT: usize = 2048;

/// Physical batch size (see [`PHYSICAL_BATCH_DEFAULT`]).
fn physical_batch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BATCH_PHYSICAL_BATCH")
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
    /// Sampling / generation configuration. Its `reasoning_tx` streams the
    /// model's reasoning apart from `token_tx`.
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

/// Names a sequence submitted with [`BatchEngine::submit_tracked`] or
/// [`BatchEngine::resume`], so it can be exported while it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceTicket(u64);

/// A running sequence taken out of its engine by
/// [`BatchEngine::detach_sequence`], with what another engine needs to resume
/// it. The source request's `result_tx` receives
/// [`Error::SequenceHandedOff`].
pub struct SequenceHandoff {
    /// The final delta of the sequence's blob chain: every position not
    /// carried by an earlier [`BatchEngine::export_sequence`] of it.
    pub blob: Vec<u8>,
    /// The token sampled last. Its text was already streamed; it is not yet in
    /// the cache, and the destination decodes it first.
    pub next_token: i32,
    /// Tokens generated before the handoff, for the destination's budget.
    pub generated_tokens: u32,
    /// All visible text generated before the handoff.
    pub text: String,
    /// All reasoning generated before the handoff.
    pub thinking: Option<String>,
    /// The reasoning markers the sequence was split on, with
    /// `open_at_start` set when it was inside a reasoning span.
    pub reasoning: ReasoningFrame,
}

/// A sequence to rebuild from its blob chain and continue, for
/// [`BatchEngine::resume`].
pub struct SequenceResume {
    /// Every encoded blob of the chain, in export order, starting at position 0.
    pub blobs: Vec<Vec<u8>>,
    /// [`SequenceHandoff::next_token`].
    pub next_token: i32,
    /// [`SequenceHandoff::reasoning`].
    pub reasoning: ReasoningFrame,
    /// Sampling for the continuation. `max_tokens` counts tokens generated
    /// after the resume; the imported positions are the request's input.
    pub config: GenerationConfig,
    /// Per-token streaming sink for the continuation. The continuation's
    /// reasoning streams to the config's `reasoning_tx`.
    pub token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    /// Where the continuation's [`InferenceResult`] (or error) is delivered.
    pub result_tx: tokio::sync::oneshot::Sender<Result<InferenceResult>>,
}

/// A request on its way to the scheduler, with the ticket it was given.
struct Queued {
    req: BatchRequest,
    ticket: Option<u64>,
}

/// Where an export's answer goes.
enum ExportReply {
    Delta(tokio::sync::oneshot::Sender<Result<Vec<u8>>>),
    Handoff(tokio::sync::oneshot::Sender<Result<SequenceHandoff>>),
}

impl ExportReply {
    fn is_closed(&self) -> bool {
        match self {
            Self::Delta(tx) => tx.is_closed(),
            Self::Handoff(tx) => tx.is_closed(),
        }
    }

    fn fail(self, err: Error) {
        match self {
            Self::Delta(tx) => drop(tx.send(Err(err))),
            Self::Handoff(tx) => drop(tx.send(Err(err))),
        }
    }
}

struct PendingExport {
    ticket: u64,
    reply: ExportReply,
}

/// Migration requests, served by the scheduler between decode steps.
enum Control {
    Export(PendingExport),
    Resume(Box<SequenceResume>, u64),
}

/// Handle to a per-model continuous-batching engine. Cloneable; every clone
/// submits to the same scheduler thread.
#[derive(Clone)]
pub struct BatchEngine {
    tx: SyncSender<Queued>,
    inner: Arc<EngineInner>,
}

struct EngineInner {
    model_id: String,
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
    /// Closing this drops the scheduler's receiver, ending the loop.
    shutdown: Sender<()>,
    /// Exports and resumes, for an engine spawned with migration.
    control: Sender<Control>,
    /// The weights this engine serves, when it can move sequences.
    migration: Option<ModelFingerprint>,
    next_ticket: std::sync::atomic::AtomicU64,
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
        Self::spawn_inner(model_id, model, backend, context_length, enable_thinking, None, None, None)
    }

    /// Spawn an engine that can export its running sequences to, and resume
    /// sequences from, other engines serving the same weights. `fingerprint`
    /// identifies those weights (see [`ModelFingerprint::of_file`]); a blob
    /// made from other weights is refused. The context reserves one KV
    /// sequence id beyond [`max_slots`] to stage copies through, which on a
    /// model with recurrent layers costs one more sequence's recurrent state.
    ///
    /// # Errors
    /// As [`spawn`](Self::spawn), and for an encoder-decoder model, whose
    /// decoder cache depends on an encoder output that is not exported.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_migration(
        model_id: String,
        model: LlamaModel,
        backend: Arc<LlamaBackend>,
        context_length: u32,
        enable_thinking: bool,
        projector: Option<Projector>,
        speculation: Option<BatchSpeculation>,
        fingerprint: ModelFingerprint,
    ) -> Result<Self> {
        Self::spawn_inner(
            model_id,
            model,
            backend,
            context_length,
            enable_thinking,
            projector,
            speculation,
            Some(fingerprint),
        )
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
        Self::spawn_inner(model_id, model, backend, context_length, enable_thinking, projector, None, None)
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
        Self::spawn_inner(model_id, model, backend, context_length, enable_thinking, projector, speculation, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_inner(
        model_id: String,
        model: LlamaModel,
        backend: Arc<LlamaBackend>,
        context_length: u32,
        enable_thinking: bool,
        projector: Option<Projector>,
        speculation: Option<BatchSpeculation>,
        migration: Option<ModelFingerprint>,
    ) -> Result<Self> {
        // This engine generates text a token at a time from a causal cache.
        // A model that does something else is refused here, with the reason,
        // rather than served into garbage or a decode error on every request.
        let traits = memory_traits(&model, swa_full());
        if traits.reuse() == Reuse::None {
            return Err(Error::Other(format!(
                "{} cannot be served by the batch engine: {}",
                model_id,
                if traits.diffusion {
                    "it decodes by diffusion over the whole output, not one token at a time"
                } else {
                    "it is an embedding or reranking model (bidirectional or pooled), which \
                     produces vectors, not text"
                }
            )));
        }
        if migration.is_some() && traits.reuse() == Reuse::EncoderOutput {
            return Err(Error::Other(format!(
                "{model_id} cannot move sequences between engines: an encoder-decoder's decoder cache \
                 depends on an encoder output that is not exported"
            )));
        }
        // Bounded: each queued request holds its whole prompt and any media
        // bytes, so a queue without a bound is memory without a bound. One
        // slot's worth of requests may wait behind the running ones; past
        // that, `submit` refuses and the caller's admission holds or sheds.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Queued>(max_slots());
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        let (control_tx, control_rx) = std::sync::mpsc::channel::<Control>();

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
                    traits,
                    migration,
                    &rx,
                    &control_rx,
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
                control: control_tx,
                migration,
                next_ticket: std::sync::atomic::AtomicU64::new(1),
            }),
        })
    }

    /// Submit a request to the scheduler. Returns immediately; the caller awaits
    /// the request's `result_tx` (and drains `token_tx` if streaming).
    ///
    /// Never blocks: when as many requests as there are slots are already
    /// queued behind the running ones, the request is refused with
    /// [`Error::QueueFull`] for the caller to hold or shed.
    pub fn submit(&self, req: BatchRequest) -> Result<()> {
        self.enqueue(Queued { req, ticket: None })
    }

    /// [`submit`](Self::submit), returning a ticket that names the sequence
    /// for [`export_sequence`](Self::export_sequence) and
    /// [`detach_sequence`](Self::detach_sequence).
    ///
    /// # Errors
    /// As [`submit`](Self::submit), and when the engine was not spawned with
    /// [`spawn_with_migration`](Self::spawn_with_migration).
    pub fn submit_tracked(&self, req: BatchRequest) -> Result<SequenceTicket> {
        self.migration()?;
        let ticket = self.new_ticket();
        self.enqueue(Queued { req, ticket: Some(ticket) })?;
        Ok(SequenceTicket(ticket))
    }

    /// Export the positions the ticket's sequence gained since its previous
    /// export (all of them the first time) as an encoded [`KvBlob`](crate::KvBlob).
    /// The sequence keeps decoding. The blob is taken at the next step
    /// boundary where the sequence is exportable (see the module docs), and
    /// the other slots are not paused beyond the copy.
    ///
    /// # Errors
    /// Immediately when the engine cannot migrate; through the receiver when
    /// no running sequence holds the ticket (it finished, failed or was
    /// detached), when it holds media positions, or when the backend refuses.
    pub fn export_sequence(
        &self,
        ticket: SequenceTicket,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<Vec<u8>>>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.control(Control::Export(PendingExport { ticket: ticket.0, reply: ExportReply::Delta(tx) }))?;
        Ok(rx)
    }

    /// Export the ticket's sequence's final delta and take it out of this
    /// engine, freeing its slot. Its request's `result_tx` receives
    /// [`Error::SequenceHandedOff`]; the returned handoff, with the earlier
    /// blobs of the chain, resumes it elsewhere through [`resume`](Self::resume).
    ///
    /// # Errors
    /// As [`export_sequence`](Self::export_sequence).
    pub fn detach_sequence(
        &self,
        ticket: SequenceTicket,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<SequenceHandoff>>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.control(Control::Export(PendingExport { ticket: ticket.0, reply: ExportReply::Handoff(tx) }))?;
        Ok(rx)
    }

    /// Rebuild a sequence exported by another engine of the same weights in a
    /// free slot, and continue decoding it without re-reading its prompt. It
    /// waits for a slot ahead of submitted requests. A continuation does not
    /// draft speculatively: the drafter never saw its prefix.
    ///
    /// Refusals arrive on `resume.result_tx`: a corrupt blob, a blob of other
    /// weights, a broken or empty chain, or a chain longer than the context.
    ///
    /// # Errors
    /// When the engine was not spawned with migration, or has stopped.
    pub fn resume(&self, resume: SequenceResume) -> Result<SequenceTicket> {
        let ticket = self.new_ticket();
        self.control(Control::Resume(Box::new(resume), ticket))?;
        Ok(SequenceTicket(ticket))
    }

    /// The weights this engine serves, when it was spawned with migration.
    pub fn fingerprint(&self) -> Option<ModelFingerprint> {
        self.inner.migration
    }

    fn migration(&self) -> Result<ModelFingerprint> {
        self.inner.migration.ok_or_else(|| {
            Error::KvSequence(format!(
                "the batch engine for {} was not spawned with migration",
                self.inner.model_id
            ))
        })
    }

    fn new_ticket(&self) -> u64 {
        self.inner.next_ticket.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn control(&self, c: Control) -> Result<()> {
        self.migration()?;
        self.inner.control.send(c).map_err(|_| {
            Error::Other(format!("batch engine for {} is no longer running", self.inner.model_id))
        })
    }

    fn enqueue(&self, q: Queued) -> Result<()> {
        self.tx.try_send(q).map_err(|e| match e {
            TrySendError::Full(_) => Error::QueueFull {
                model_id: self.inner.model_id.clone(),
                waiting: max_slots(),
                max: max_slots(),
            },
            TrySendError::Disconnected(_) => {
                Error::Other(format!("batch engine for {} is no longer running", self.inner.model_id))
            }
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
/// An agent's next turn repeats the system prompt, the tool schemas and the
/// entire conversation so far, and re-reading those through the model is the
/// largest avoidable cost in an agent loop. Keeping what is already in KV lets
/// the next request start where the two diverge instead of at zero.
#[derive(Default)]
struct CachedPrefix {
    /// Identity of every position resident in this sequence's KV, in order:
    /// prompt, then what was generated.
    ids: Vec<PrefixId>,
    /// Media chunks among `ids`.
    spans: Vec<MediaSpan>,
    /// Snapshots of the parts of the sequence's memory that cannot be trimmed
    /// (recurrent state, a sliding window), keyed by identity index.
    checkpoints: CheckpointStore<llama_cpp_2::SeqState>,
    /// Whose prompt filled this cache.
    namespace: Namespace,
    /// Scheduler tick of the last request served from this slot, for
    /// evicting the least recently used idle prefix first.
    last_used: u64,
}

impl CachedPrefix {
    /// The slot's KV was dropped: nothing it recorded describes the cache now.
    fn forget(&mut self) {
        self.ids.clear();
        self.spans.clear();
        self.checkpoints.clear();
    }

    /// Keep only the first `n` identity positions (and what describes them).
    fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        self.spans.retain(|s| s.at + s.len <= n);
        self.checkpoints.truncate_after(n);
    }
}

/// How this engine reuses prefixes, fixed for its model at spawn.
struct ReusePolicy {
    kind: Reuse,
    /// Tokens the chat template closes a turn with.
    turn_ends: TurnEnds,
    /// Checkpoints each slot may hold; 0 when the model needs none or memory
    /// affords none.
    per_slot: usize,
}

/// Fewest prompt tokens between two checkpoints at earlier turn starts: a
/// turn shorter than this is cheaper to re-prefill than to keep a snapshot
/// for. The last turn start and the prompt's end are always kept.
const CHECKPOINT_MIN_SPACING: usize = 256;

/// Most checkpoints per slot the operator allows, from
/// `BATCH_PROMPT_CHECKPOINTS` (0 turns them off). Memory may allow fewer.
fn checkpoint_cap() -> usize {
    std::env::var("BATCH_PROMPT_CHECKPOINTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(crate::prefix_cache::MAX_CHECKPOINTS_PER_SLOT)
}

/// Whether a sliding-window model keeps every position of its window layers
/// (`BATCH_SWA_FULL`, on unless set to 0). On, the window layers trim like
/// full attention and need no checkpoints; off, they hold only the window,
/// which costs far less memory per slot and is rewound through checkpoints.
fn swa_full() -> bool {
    std::env::var("BATCH_SWA_FULL").map_or(true, |v| v.trim() != "0")
}

/// Read what the loaded model's memory can do.
fn memory_traits(model: &LlamaModel, swa_full: bool) -> MemoryTraits {
    let arch = model.meta_val_str("general.architecture").unwrap_or_default();
    let meta = |key: &str| model.meta_val_str(&format!("{arch}.{key}")).ok();
    // An encoder attends both ways, and a pooled model is an embedder or a
    // reranker whatever its attention: neither produces text a token at a time.
    let bidirectional = meta("attention.causal").is_some_and(|v| v.trim() == "false");
    let pooled = meta("pooling_type").is_some_and(|v| !matches!(v.trim(), "" | "0"));
    MemoryTraits {
        recurrent: model.is_recurrent(),
        hybrid: model.is_hybrid(),
        n_swa: model.n_swa(),
        swa_full,
        causal: !bidirectional && !pooled,
        encoder_decoder: model.has_encoder() && model.has_decoder(),
        diffusion: model.is_diffusion(),
    }
}

/// Learn which tokens the model's chat template closes a turn with, by
/// rendering a short probe conversation and reading the first control token
/// after each message's content. The vocabulary's end-of-generation tokens are
/// always included: a template closes the assistant's turn with one.
fn probe_turn_ends(model: &LlamaModel) -> TurnEnds {
    let mut ends: Vec<i32> = Vec::new();
    let probes = ["turnprobeuser", "turnprobereply", "turnprobenext"];
    let messages = vec![
        ChatMessage::new("user", probes[0]),
        ChatMessage::new("assistant", probes[1]),
        ChatMessage::new("user", probes[2]),
    ];
    if let Ok(rendered) = render_prompt(model, &BatchPrompt::Chat(messages), false, None) {
        for probe in &probes[..2] {
            let Some(at) = rendered.find(probe) else { continue };
            let after = &rendered[at + probe.len()..];
            let Ok(tokens) = model.str_to_token(after, AddBos::Never) else { continue };
            if let Some(t) = tokens.iter().find(|t| {
                model.is_eog_token(**t) || model.token_attr(**t).contains(llama_cpp_2::token_type::LlamaTokenAttr::Control)
            }) {
                ends.push(t.0);
            }
        }
    }
    for id in 0..model.n_vocab() {
        if model.is_eog_token(LlamaToken(id)) {
            ends.push(id);
        }
    }
    TurnEnds::new(ends)
}

/// What one checkpoint and one token of KV cost on this context.
struct StateSizes {
    /// Bytes of one checkpoint of a slot with a full window (or the fixed
    /// recurrent state).
    checkpoint: u64,
    /// Bytes of KV each cached token takes across every layer.
    kv_per_token: u64,
}

/// Measure [`StateSizes`] on the live context by decoding two tokens into
/// slot 0, reading the serialized state sizes after each, and clearing the
/// slot again. Measured rather than derived from metadata: the layouts differ
/// by architecture (gated delta nets, Mamba, RWKV, sliding windows), and the
/// context already knows what it allocated.
fn measure_state_sizes(ctx: &mut llama_cpp_2::context::LlamaContext, model: &LlamaModel, kind: Reuse) -> StateSizes {
    use llama_cpp_2::LlamaStateSeqFlags;
    let mut sizes = [(0u64, 0u64); 2];
    let mut batch = LlamaBatch::new(1, 1);
    for (pos, size) in sizes.iter_mut().enumerate() {
        batch.clear();
        if batch.add(model.token_bos(), pos as i32, &[0], false).is_err() || ctx.decode(&mut batch).is_err() {
            break;
        }
        *size = (
            ctx.state_seq_get_size_ext(0, LlamaStateSeqFlags::PARTIAL_ONLY) as u64,
            ctx.state_seq_get_size_ext(0, LlamaStateSeqFlags::empty()) as u64,
        );
    }
    let _ = ctx.clear_kv_cache_seq(Some(0), None, None);
    let [(partial1, full1), (partial2, full2)] = sizes;
    let checkpoint = match kind {
        // The window holds up to `n_swa` positions plus one batch in flight.
        Reuse::Window { n_swa } => {
            partial1 + (u64::from(n_swa) + physical_batch() as u64).saturating_mul(partial2.saturating_sub(partial1))
        }
        _ => partial2,
    };
    StateSizes {
        checkpoint,
        kv_per_token: full2.saturating_sub(full1),
    }
}

/// The checkpoint budget this host affords, and whether checkpoints draw on
/// the same memory as the KV cache.
///
/// Checkpoints are host memory. They share it with the KV cache when there is
/// no accelerator (CPU serving) or the accelerator's memory is the host's
/// (integrated GPUs, Apple silicon, GB10-class superchips whose GPU reports the
/// host's memory as its own); there, room for checkpoints can be bought with
/// context. On a discrete GPU the KV cache lives in device memory and
/// checkpoints do not compete with it.
///
/// Both are read from the backend's device list at load. An operator can
/// override either: `BATCH_CHECKPOINT_MEMORY` is the bytes checkpoints may use
/// across every slot, and `BATCH_UNIFIED_MEMORY` (1 or 0) says whether they
/// share memory with the KV cache.
fn host_checkpoint_budget(checkpoint_bytes: u64) -> (CheckpointBudget, bool) {
    use llama_cpp_2::LlamaBackendDeviceType as Kind;
    let env = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
    let devices = llama_cpp_2::list_llama_ggml_backend_devices();
    let host = devices.iter().find(|d| d.device_type == Kind::Cpu);
    let (free, total) = host.map_or((0, 0), |d| (d.memory_free as u64, d.memory_total as u64));
    let accelerators: Vec<_> = devices
        .iter()
        .filter(|d| matches!(d.device_type, Kind::Gpu | Kind::IntegratedGpu | Kind::Accelerator))
        .collect();
    let unified = env("BATCH_UNIFIED_MEMORY").map_or_else(
        || {
            accelerators.is_empty()
                || accelerators.iter().any(|d| {
                    d.device_type == Kind::IntegratedGpu
                        || d.backend.starts_with("Metal")
                        || d.backend.starts_with("MTL")
                        || (total > 0 && d.memory_total as u64 >= total / 10 * 9)
                })
        },
        |v| v != 0,
    );
    let budget = match env("BATCH_CHECKPOINT_MEMORY") {
        Some(bytes) => crate::prefix_cache::budget_of(bytes, max_slots(), checkpoint_bytes),
        None => crate::prefix_cache::checkpoint_budget(free, total, max_slots(), checkpoint_bytes),
    };
    (budget, unified)
}

/// Identity of a text-only prompt.
fn text_ids(tokens: &[LlamaToken]) -> Vec<PrefixId> {
    tokens.iter().map(|t| crate::prefix_cache::text_id(t.0)).collect()
}

/// A running sequence occupying one slot.
struct Sequence {
    /// KV-cache sequence id, always the index of the slot holding this
    /// sequence (asserted wherever a slot is read).
    seq_id: i32,
    sampler: LlamaSampler,
    token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    result_tx: Option<tokio::sync::oneshot::Sender<Result<InferenceResult>>>,
    decoder: encoding_rs::Decoder,
    /// Prompt tokens staged by `admit`, waiting for the scheduler to prefill
    /// them. `None` once the whole prompt has been committed to the KV cache.
    /// A media chunk's positions hold a placeholder token; the media step
    /// evaluates them, never the text prefill.
    pending_prompt: Option<Vec<LlamaToken>>,
    /// How many positions of the prompt are already committed.
    prefill_cursor: usize,
    /// Next KV position for this sequence.
    n_past: i32,
    /// Every token committed to this sequence's KV — prompt then generated —
    /// so the next request on this slot can measure its shared prefix.
    resident: Vec<LlamaToken>,
    /// Identity of the prompt, position for position.
    prompt_ids: Vec<PrefixId>,
    /// Media chunks in the prompt, in order.
    media_spans: Vec<MediaSpan>,
    /// Whose prompt this is.
    namespace: Namespace,
    /// Prompt positions at which to snapshot the sequence's memory, ascending.
    cuts: Vec<usize>,
    /// Prompt positions taken from the slot's cache instead of prefilled.
    cached_tokens: u32,
    /// Activation commitment being recorded, when the request asked for one.
    commitment: Option<CommitmentLog>,
    input_tokens: u32,
    /// Every token emitted after the prompt, sampled or forced: what the
    /// sequence occupies in the cache and what the commitment records.
    output_tokens: u32,
    /// Of those, the tokens the engine forced (a reasoning block's close
    /// marker). They were never generated by the model, so they are not
    /// reported as generated output.
    forced_tokens: u32,
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
    /// Media chunks waiting for the scheduler's media step.
    #[cfg(feature = "mtmd")]
    pending_media: Option<PendingMedia>,
    /// The name callers export this sequence by, when it has one.
    ticket: Option<u64>,
    /// How far this sequence's cache has been exported.
    exporter: Option<SequenceExport>,
    /// The last step rolled back rejected drafts on a model with recurrent
    /// layers. The rollback selects an earlier recurrent snapshot that only the
    /// next decode applies, so until then the state cannot be copied.
    rollback_pending: bool,
}

impl Sequence {
    /// Whether the cache holds exactly `resident` and nothing is half done:
    /// the prompt is in, a sampled token waits, no forced tokens are queued,
    /// and no recurrent rollback is waiting on the next decode.
    fn exportable_now(&self) -> bool {
        self.pending_prompt.is_none()
            && self.pending_token.is_some()
            && self.forced.is_empty()
            && !self.rollback_pending
            && self.n_past as usize == self.resident.len()
    }

    /// Close this sequence's stream for a handoff and tell its caller.
    fn hand_off(mut self, blob: Vec<u8>) -> SequenceHandoff {
        let reasoning = ReasoningFrame { open_at_start: self.stream.in_reasoning(), ..self.stream.frame() };
        let token_tx = self.token_tx.take();
        let (text, thinking) = self.stream.close(token_tx.as_ref());
        let _ = self.stream.flush(token_tx.as_ref());
        self.fail(Error::SequenceHandedOff);
        SequenceHandoff {
            blob,
            next_token: self.pending_token.map_or(-1, |t| t.0),
            generated_tokens: self.output_tokens - self.forced_tokens,
            text,
            thinking,
            reasoning,
        }
    }
}

/// A multimodal prompt's chunks, and which chunk each media span is.
#[cfg(feature = "mtmd")]
struct PendingMedia {
    chunks: MtmdInputChunks,
    /// Chunk index of each entry of the sequence's `media_spans`.
    chunk_of_span: Vec<usize>,
}

/// The top-k logits recorded at every generated token, for a verifiable
/// request.
struct CommitmentLog {
    k: usize,
    steps: Vec<StepRecord>,
    /// A step whose logits could not be read (a prompt that ended in a media
    /// chunk leaves them only in llama.cpp's last row). Such a log cannot be
    /// verified, so none is returned.
    incomplete: bool,
}

impl CommitmentLog {
    fn new(config: &GenerationConfig) -> Option<Self> {
        config.commitment_k.map(|k| Self {
            k: usize::from(crate::toploc::commitment_k_for(k)),
            steps: Vec::new(),
            incomplete: false,
        })
    }

    /// Record the logits row a token is about to be sampled from.
    fn top_k(&self, ctx: &llama_cpp_2::context::LlamaContext, logits_idx: i32) -> Option<Vec<crate::toploc::TopKEntry>> {
        (logits_idx >= 0).then(|| crate::toploc::top_k_from_logits(ctx.get_logits_ith(logits_idx), self.k))
    }

    fn push(&mut self, token: LlamaToken, top_k: Option<Vec<crate::toploc::TopKEntry>>) {
        match top_k {
            Some(top_k) => self.steps.push(StepRecord {
                token_id: token.0 as u32,
                top_k,
            }),
            None => self.incomplete = true,
        }
    }

    fn finish(self, prompt_tokens: u32) -> Option<InferenceCommitment> {
        (!self.incomplete && !self.steps.is_empty()).then(|| InferenceCommitment {
            k: self.k as u8,
            prompt_tokens,
            steps: self.steps,
        })
    }
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

/// Free the KV held by the least recently used idle slot's cached prefix.
/// True when one was freed. Called once per failed step, so under pressure
/// the coldest conversations go one at a time until the step fits, and the
/// warm ones survive. Reuse is an optimisation; a request that cannot run is
/// not.
fn evict_idle_prefix(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    slots: &[Option<Sequence>],
    cached: &mut [CachedPrefix],
) -> bool {
    let Some(i) = (0..cached.len())
        .filter(|&i| slots[i].is_none() && !cached[i].ids.is_empty())
        .min_by_key(|&i| cached[i].last_used)
    else {
        return false;
    };
    let _ = ctx.clear_kv_cache_seq(Some(i as u32), None, None);
    draft_seq_rm(draft, i as i32, None);
    cached[i].forget();
    true
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
        cached[idx].forget();
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
    debug_assert_eq!(seq.seq_id as usize, slot_idx, "a slot decodes only its own KV sequence");
    let top_k = seq.commitment.as_ref().map(|c| c.top_k(ctx, logits_idx));
    let token = next_token(seq, ctx, logits_idx);
    seq.sampler.accept(token);

    if model.is_eog_token(token) {
        return true;
    }
    if let (Some(log), Some(top_k)) = (seq.commitment.as_mut(), top_k) {
        log.push(token, top_k);
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
        Some(token) => {
            seq.forced_tokens += 1;
            token
        }
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
    debug_assert_eq!(seq.seq_id as usize, slot_idx, "a slot decodes only its own KV sequence");
    let mut accepted: u16 = 0;
    for i in 0..=drafts.len() {
        let row = first_idx + i as i32;
        let top_k = seq.commitment.as_ref().map(|c| c.top_k(ctx, row));
        let token = next_token(seq, ctx, row);
        seq.sampler.accept(token);
        if model.is_eog_token(token) {
            return (true, accepted);
        }
        if let (Some(log), Some(top_k)) = (seq.commitment.as_mut(), top_k) {
            log.push(token, top_k);
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

/// A finished sequence whose streaming receiver has not taken every chunk
/// yet. Its result is delivered once the stream is, so a caller that reads
/// the stream to its end and then the result sees all of the text.
struct Draining {
    tx: Option<tokio::sync::mpsc::Sender<String>>,
    stream: StopStream,
    result_tx: tokio::sync::oneshot::Sender<Result<InferenceResult>>,
    result: InferenceResult,
    since: Instant,
}

/// How long a finished sequence's stream may take to drain before its
/// receiver is given up on.
const STREAM_DRAIN_TIMEOUT: Duration = Duration::from_secs(60);

/// The error a sequence ends with when its streaming receiver stops reading.
fn stalled_stream() -> Error {
    Error::Inference(
        "the streaming client stopped reading; the request was ended so it no longer holds \
         back other requests"
            .into(),
    )
}

/// Hand queued chunks to every draining receiver, and deliver each result
/// whose stream has been fully taken, whose receiver is gone, or which has
/// waited past [`STREAM_DRAIN_TIMEOUT`].
fn drain_streams(draining: &mut Vec<Draining>) {
    let mut i = 0;
    while i < draining.len() {
        let d = &mut draining[i];
        // A receiver that is gone has nothing left to wait for.
        let delivered = !d.stream.flush(d.tx.as_ref()) || d.stream.delivered();
        if delivered || d.since.elapsed() >= STREAM_DRAIN_TIMEOUT {
            let d = draining.swap_remove(i);
            let _ = d.result_tx.send(if delivered { Ok(d.result) } else { Err(stalled_stream()) });
        } else {
            i += 1;
        }
    }
}

impl Sequence {
    /// Deliver the result, or, when the stream still has chunks the receiver
    /// has not taken, hand the sequence's stream to the drain list and deliver
    /// the result once it has.
    fn finish(mut self) -> Option<Draining> {
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
        let stalled = self.stream.stalled();
        let (text, thinking) = self.stream.close(token_tx.as_ref());
        let result_tx = self.result_tx.take()?;
        if stalled {
            let _ = result_tx.send(Err(stalled_stream()));
            return None;
        }
        let result = InferenceResult {
            text,
            thinking,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens - self.forced_tokens,
            generation_time_ms,
            tokens_per_second,
            stop_reason,
            commitment: self.commitment.take().and_then(|c| c.finish(self.input_tokens)),
            cached_tokens: self.cached_tokens,
        };
        if self.stream.delivered() {
            let _ = result_tx.send(Ok(result));
            return None;
        }
        Some(Draining {
            tx: token_tx,
            stream: self.stream,
            result_tx,
            result,
            since: Instant::now(),
        })
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
    traits: MemoryTraits,
    migration: Option<ModelFingerprint>,
    rx: &Receiver<Queued>,
    control_rx: &Receiver<Control>,
    shutdown_rx: &Receiver<()>,
) -> Result<()> {
    use std::num::NonZeroU32;

    let n_ctx = NonZeroU32::new(context_length).unwrap_or(NonZeroU32::new(8192).unwrap());
    let kind = traits.reuse();
    // An engine that moves sequences stages copies through one sequence id
    // past the slots, which no request ever decodes into.
    let scratch = max_slots() as i32;
    let seq_ids = max_slots() as u32 + u32::from(migration.is_some());
    // Rolling back rejected drafts on a recurrent layer defers the restore to
    // the next decode (see `Sequence::rollback_pending`).
    let recurrent_state = model.is_recurrent() || model.is_hybrid();

    // One long-lived context with max_slots() sequence slots. n_batch/n_ubatch
    // cover the interleaved prefill+extend batch.
    let context_params = |n_ctx: NonZeroU32| {
        let mut params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_seq_max(seq_ids)
            // One KV pool shared by every sequence. Without it `n_ctx` is split
            // evenly, so each request is capped at `n_ctx / slots` however little
            // the others use — at 32 slots a 131k model answers in 4k.
            .with_kv_unified(true)
            .with_swa_full(traits.swa_full)
            .with_n_batch(physical_batch() as u32)
            .with_n_ubatch(physical_batch() as u32);
        // Rejected drafts are rolled back out of the target's KV. A model whose
        // layers carry recurrent state cannot rewind by position, so it keeps this
        // many snapshots per sequence to roll back to.
        if let Some(s) = speculation.as_ref() {
            params = params.with_n_rs_seq(u32::from(s.n_max));
        }
        params
    };

    let mut ctx = model
        .new_context(backend, context_params(n_ctx))
        .map_err(|e| Error::Other(format!("batch context init failed: {}", e)))?;

    // What prefix reuse costs and affords on this model and this host.
    let mut per_slot = 0;
    if kind.needs_checkpoints() && checkpoint_cap() > 0 {
        let sizes = measure_state_sizes(&mut ctx, &model, kind);
        let (mut budget, unified) = host_checkpoint_budget(sizes.checkpoint);
        // Where checkpoints and the KV cache draw on the same memory, a
        // context too large to leave room for the fewest useful checkpoints is
        // shrunk until it does, never below a quarter of what was asked or 8K.
        if unified
            && let Some(release) = crate::prefix_cache::context_to_release(
                budget,
                max_slots(),
                sizes.checkpoint,
                sizes.kv_per_token,
                ctx.n_ctx(),
                (n_ctx.get() / 4).max(8192),
            )
        {
            let smaller = NonZeroU32::new(ctx.n_ctx() - release).expect("above the floor");
            drop(ctx);
            ctx = model
                .new_context(backend, context_params(smaller))
                .map_err(|e| Error::Other(format!("batch context init failed: {}", e)))?;
            budget = host_checkpoint_budget(sizes.checkpoint).0;
            info!(
                "batch engine for {}: context {} -> {} tokens, so every slot can keep its checkpoints",
                model_id, n_ctx, smaller
            );
        }
        per_slot = budget.per_slot.min(checkpoint_cap());
        info!(
            "batch engine for {}: {:?} memory, {} checkpoints per slot of {} bytes each ({} bytes budgeted{})",
            model_id,
            kind,
            per_slot,
            sizes.checkpoint,
            budget.bytes,
            if unified { ", shared with the KV cache" } else { "" },
        );
    }
    let policy = ReusePolicy {
        kind,
        turn_ends: if per_slot > 0 { probe_turn_ends(&model) } else { TurnEnds::default() },
        per_slot,
    };

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
    // Finished sequences whose streams are still being handed over.
    let mut draining: Vec<Draining> = Vec::new();
    // An encoder-decoder model keeps one encoder output in its context, so it
    // decodes one request at a time; the others wait here, in order.
    let mut waiting: std::collections::VecDeque<Queued> = std::collections::VecDeque::new();
    // Exports waiting for their sequence to reach a step boundary it can be
    // copied at, and sequences waiting for a slot to be rebuilt in.
    let mut exports: Vec<PendingExport> = Vec::new();
    let mut resumes: std::collections::VecDeque<(Box<SequenceResume>, u64)> = std::collections::VecDeque::new();
    // The encoder input whose output the context currently holds.
    let mut encoded: Option<(Namespace, Vec<LlamaToken>)> = None;
    // Counts admissions, for least-recently-used eviction.
    let mut tick: u64 = 0;
    let mut batch = LlamaBatch::new(physical_batch(), max_slots() as i32);

    loop {
        if shutdown_rx.try_recv().is_ok() {
            break;
        }

        drain_streams(&mut draining);
        // A receiver that fell behind gets what is queued for it now; one that
        // has stalled ends its own sequence, and no one else's.
        for slot_idx in 0..slots.len() {
            let stalled = slots[slot_idx].as_mut().is_some_and(|s| {
                s.stream.flush(s.token_tx.as_ref());
                s.stream.stalled()
            });
            if stalled && let Some(mut seq) = slots[slot_idx].take() {
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                draft_seq_rm(&mut draft_ctx, seq.seq_id, None);
                cached[slot_idx].forget();
                seq.fail(stalled_stream());
            }
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
                cached[slot_idx].forget();
            }
        }

        // Migration: take new exports and resumes, then answer every export
        // whose sequence is at a consistent point. Between two decode steps
        // nothing is in flight, so the copy sees one step's state throughout.
        while let Ok(c) = control_rx.try_recv() {
            match c {
                Control::Export(e) => exports.push(e),
                Control::Resume(r, ticket) => resumes.push_back((r, ticket)),
            }
        }
        if let Some(fp) = migration {
            serve_exports(&mut ctx, &mut draft_ctx, &mut slots, &mut cached, &mut exports, fp, scratch);
        }

        // An encoder-decoder model decodes one request at a time.
        let capacity = if policy.kind == Reuse::EncoderOutput { 1 } else { slots.len() };

        // A moved sequence already holds a client mid-answer, so it takes a
        // free slot ahead of requests that have not started.
        if let Some(fp) = migration {
            while slots.iter().filter(|s| s.is_some()).count() < capacity {
                let Some((resume, ticket)) = resumes.pop_front() else { break };
                tick += 1;
                admit_resume(&model, &mut ctx, &mut draft_ctx, ctx_size, &mut slots, &mut cached, fp, scratch, *resume, ticket, tick);
            }
        }
        let active = slots.iter().filter(|s| s.is_some()).count();

        // Admit new requests into free slots. When idle, block (bounded) on the
        // first one so the thread parks instead of spinning; then drain the rest
        // non-blocking.
        if active == 0 && waiting.is_empty() && resumes.is_empty() {
            match rx.recv_timeout(IDLE_POLL) {
                Ok(q) => waiting.push_back(q),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        // Fill any remaining free slots without blocking.
        while slots.iter().filter(|s| s.is_some()).count() < capacity {
            let Queued { req, ticket } = match waiting.pop_front() {
                Some(q) => q,
                None => match rx.try_recv() {
                    Ok(q) => q,
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                },
            };
            tick += 1;
            if policy.kind == Reuse::EncoderOutput {
                admit_encoder_decoder(&model, &mut ctx, ctx_size, &mut slots, &mut cached, req, enable_thinking, &mut encoded);
                continue;
            }
            if let Some(r) = admit(
                &model,
                ctx_size,
                &mut slots,
                &mut cached,
                &policy,
                req,
                enable_thinking,
                projector.as_ref(),
            ) {
                let reuse_slot = r.slot_idx;
                cached[reuse_slot].last_used = tick;
                if let Some(seq) = slots[reuse_slot].as_mut() {
                    seq.ticket = ticket;
                }
                let held = apply_prefix_reuse(&mut ctx, &mut slots, &mut cached, &policy, r);
                draft_seq_rm(&mut draft_ctx, reuse_slot as i32, Some(held as u32));
            }
        }

        // Media prefill, at most one chunk per step. A media chunk is encoded
        // and its embeddings decoded for that one sequence on the shared
        // context, switching to non-causal attention where the projector needs
        // it; that switch is context-wide, which is why media cannot share a
        // batch with other sequences' tokens. The text around it is prefilled
        // in the ordinary batch, and a chunk the slot's cache already holds is
        // never evaluated at all. One chunk per step bounds how long decoding
        // sequences wait behind an image.
        #[cfg(feature = "mtmd")]
        if let Some(projector) = projector.as_ref() {
            media_step(
                &model,
                &mut ctx,
                &mut draft_ctx,
                projector,
                &mut slots,
                &mut cached,
                policy.kind,
                &mut draining,
                model_id,
            );
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
            // Stop at the next checkpoint cut, so the state after this step is
            // the state at the cut, and before the next media chunk, which the
            // media step evaluates on its own.
            let next_cut = seq.cuts.iter().copied().find(|&c| c > start).unwrap_or(usize::MAX);
            let next_media = seq.media_spans.iter().map(|s| s.at).find(|&a| a >= start).unwrap_or(usize::MAX);
            let end = prompt.len().min(start + room).min(next_cut).min(next_media);
            let last = prompt.len() - 1;
            if end <= start {
                // Waiting on the media step.
                seq.pending_prompt = Some(prompt);
                continue;
            }

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
                if evict_idle_prefix(&mut ctx, &mut draft_ctx, &slots, &mut cached) {
                    info!("shared KV pool for {} was full: dropped the least recently used idle prefix", model_id);
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
                    cached[*slot_idx].forget();
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

        // A prefill that stopped at a checkpoint cut leaves the sequence's
        // memory exactly as it stands after the cut: nothing past it has been
        // decoded yet. Keep the part that cannot be trimmed (recurrent state, a
        // sliding window), so a later request on this slot that diverges past
        // the cut can rewind here. Taken only after the decode succeeded: a
        // step rolled back for want of KV room never reaches this point.
        if policy.per_slot > 0 {
            for row in &step {
                let StepRow::PrefillPart { slot, .. } = row else { continue };
                let Some(seq) = slots[*slot].as_ref() else { continue };
                let at = seq.prefill_cursor;
                if !seq.cuts.contains(&at) {
                    continue;
                }
                match ctx.state_seq_get(seq.seq_id, llama_cpp_2::LlamaStateSeqFlags::PARTIAL_ONLY) {
                    Ok(state) => cached[*slot].checkpoints.insert(at, state, policy.per_slot),
                    Err(e) => warn!("checkpoint at {} for slot {} not taken: {}", at, slot, e),
                }
            }
        }

        // Sample each sequence from its own logits index.
        for (logits_idx, slot_idx) in logits_slot {
            // This decode applied any recurrent rollback still pending.
            if let Some(seq) = slots[slot_idx].as_mut() {
                seq.rollback_pending = false;
            }
            let block = std::mem::take(&mut drafts[slot_idx]);
            let done = if block.is_empty() {
                sample_into(&model, &ctx, &mut slots, slot_idx, logits_idx, model_id)
            } else {
                let (done, accepted) =
                    sample_block_into(&model, &ctx, &mut slots, slot_idx, logits_idx, &block, model_id);
                if let Some(seq) = slots[slot_idx].as_mut() {
                    // Rejected drafts were written into both caches; take them out.
                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), Some(seq.n_past as u32), None);
                    seq.rollback_pending = recurrent_state;
                    draft_seq_rm(&mut draft_ctx, seq.seq_id, Some(seq.n_past as u32));
                    if let Some(sp) = spec.as_mut() {
                        let _ = sp.accept(seq.seq_id, accepted);
                    }
                }
                done
            };
            if done {
                finalize_and_free(&mut ctx, &mut draft_ctx, &mut slots, &mut cached, policy.kind, slot_idx, &mut draining);
            }
        }
    }

    // Drain remaining slots on shutdown.
    for slot in slots.iter_mut() {
        if let Some(mut seq) = slot.take() {
            seq.fail(Error::Other("batch engine shutting down".into()));
        }
    }
    for q in waiting {
        let _ = q.req.result_tx.send(Err(Error::Other("batch engine shutting down".into())));
    }
    for (resume, _) in resumes {
        let _ = resume.result_tx.send(Err(Error::Other("batch engine shutting down".into())));
    }
    for e in exports {
        e.reply.fail(Error::Other("batch engine shutting down".into()));
    }
    // Finished results still waiting on their streams are delivered as they
    // are: the engine will not flush them again.
    for d in draining {
        let _ = d.result_tx.send(Ok(d.result));
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
#[allow(clippy::too_many_arguments)]
fn finalize_and_free(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    kind: Reuse,
    slot_idx: usize,
    draining: &mut Vec<Draining>,
) {
    let Some(seq) = slots[slot_idx].take() else {
        return;
    };
    debug_assert_eq!(seq.seq_id as usize, slot_idx, "a slot decodes only its own KV sequence");
    let c = &mut cached[slot_idx];
    if kind.keeps_cache() {
        // Keep the KV. The next request on this slot is very often the same
        // conversation one turn later, and re-decoding a prefix already held is
        // the largest avoidable cost in an agent loop. The next admission
        // rewinds whatever its prompt diverges from. What the slot holds is the
        // prompt exactly as it was prefilled (media included, by content), then
        // every generated token that reached KV.
        let prompt_len = seq.prompt_ids.len();
        c.ids.clear();
        c.ids.extend_from_slice(&seq.prompt_ids);
        c.ids.extend(seq.resident.get(prompt_len..).unwrap_or_default().iter().map(|t| crate::prefix_cache::text_id(t.0)));
        c.spans.clone_from(&seq.media_spans);
        c.namespace = seq.namespace;
        c.checkpoints.truncate_after(c.ids.len());
    } else {
        // An encoder-decoder's decoder cache depends on the encoder output,
        // which the next request may replace; it is never offered as a prefix.
        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
        draft_seq_rm(draft, seq.seq_id, None);
        c.forget();
    }
    if let Some(d) = seq.finish() {
        draining.push(d);
    }
}

/// Answer every export whose sequence can be copied now; keep the rest for a
/// later step boundary. Runs on the scheduler thread between decode steps.
#[allow(clippy::too_many_arguments)]
fn serve_exports(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    exports: &mut Vec<PendingExport>,
    fp: ModelFingerprint,
    scratch: i32,
) {
    let mut i = 0;
    while i < exports.len() {
        if exports[i].reply.is_closed() {
            exports.remove(i);
            continue;
        }
        let ticket = exports[i].ticket;
        let Some(slot_idx) = slots.iter().position(|s| s.as_ref().is_some_and(|s| s.ticket == Some(ticket))) else {
            exports.remove(i).reply.fail(Error::KvSequence(format!(
                "no running sequence holds ticket {ticket}: it finished, failed or was handed off"
            )));
            continue;
        };
        let seq = slots[slot_idx].as_mut().expect("slot found above");
        if seq.multimodal {
            exports.remove(i).reply.fail(Error::KvSequence(
                "a sequence holding media positions cannot be exported".into(),
            ));
            continue;
        }
        if !seq.exportable_now() {
            i += 1;
            continue;
        }
        let seq_id = seq.seq_id;
        let blob = seq
            .exporter
            .get_or_insert_with(|| SequenceExport::new(fp, seq_id, scratch))
            .export(ctx, &seq.resident);
        match (exports.remove(i).reply, blob) {
            (ExportReply::Delta(tx), blob) => {
                let _ = tx.send(blob);
            }
            (ExportReply::Handoff(tx), Err(e)) => {
                let _ = tx.send(Err(e));
            }
            (ExportReply::Handoff(tx), Ok(blob)) => {
                let seq = slots[slot_idx].take().expect("slot found above");
                let _ = ctx.clear_kv_cache_seq(Some(slot_idx as u32), None, None);
                draft_seq_rm(draft, slot_idx as i32, None);
                cached[slot_idx].forget();
                let _ = tx.send(Ok(seq.hand_off(blob)));
            }
        }
    }
}

/// Rebuild a moved sequence in a free slot from its blob chain and set it to
/// decode its last sampled token next. A refusal goes to the resume's
/// `result_tx` and leaves the slot empty.
#[allow(clippy::too_many_arguments)]
fn admit_resume(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    ctx_size: i32,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    fp: ModelFingerprint,
    scratch: i32,
    resume: SequenceResume,
    ticket: u64,
    tick: u64,
) {
    // The free slot whose cached prefix is worth least: an empty one, else
    // the one least recently used.
    let Some(slot_idx) = (0..slots.len())
        .filter(|&i| slots[i].is_none())
        .min_by_key(|&i| (!cached[i].ids.is_empty(), cached[i].last_used))
    else {
        let _ = resume.result_tx.send(Err(Error::KvSequence("no free slot to resume into".into())));
        return;
    };
    let _ = ctx.clear_kv_cache_seq(Some(slot_idx as u32), None, None);
    draft_seq_rm(draft, slot_idx as i32, None);
    cached[slot_idx].forget();

    let refuse = |ctx: &mut llama_cpp_2::context::LlamaContext, tx: tokio::sync::oneshot::Sender<Result<InferenceResult>>, e: Error| {
        let _ = ctx.clear_kv_cache_seq(Some(slot_idx as u32), None, None);
        let _ = tx.send(Err(e));
    };
    let mut importer = SequenceImport::new(fp, slot_idx as i32, scratch);
    let mut tokens: Vec<LlamaToken> = Vec::new();
    for encoded in &resume.blobs {
        match importer.apply(ctx, encoded) {
            Ok(blob) => tokens.extend(blob.tokens.into_iter().map(LlamaToken)),
            Err(e) => return refuse(ctx, resume.result_tx, e),
        }
    }
    if tokens.is_empty() {
        return refuse(ctx, resume.result_tx, Error::KvSequence("the blob chain carries no positions".into()));
    }
    let n = tokens.len();
    if n as i32 >= ctx_size {
        return refuse(
            ctx,
            resume.result_tx,
            Error::Inference(format!("resumed sequence of {n} positions exceeds context window {ctx_size}")),
        );
    }
    let staged = Staged {
        ids: text_ids(&tokens),
        tokens: tokens.clone(),
        spans: Vec::new(),
        #[cfg(feature = "mtmd")]
        media: None,
    };
    let namespace = Namespace::of(resume.config.cache_salt.as_deref());
    let max_pos = ctx_size.min(n as i32 + resume.config.max_tokens as i32);
    let mut seq = Sequence::new(
        model,
        resume.config,
        resume.token_tx,
        resume.result_tx,
        resume.reasoning,
        staged,
        namespace,
        max_pos,
    );
    // The cache already holds every position: nothing to prefill, and the
    // first decode feeds the token the source sampled last.
    seq.pending_prompt = None;
    seq.resident = tokens;
    seq.n_past = n as i32;
    seq.cached_tokens = n as u32;
    seq.pending_token = Some(LlamaToken(resume.next_token));
    seq.ticket = Some(ticket);
    cached[slot_idx].last_used = tick;
    install(slots, slot_idx, seq);
}

/// Drop the part of a slot's KV cache that the new prompt diverges from, and
/// record what the slot now holds.
///
/// Order matters: the trim must happen before the scheduler decodes anything
/// into this sequence, or the new tokens would be written on top of positions
/// still holding the previous request's.
///
/// Returns the KV position prefill resumes from, which is also how much of
/// the draft context still matches.
fn apply_prefix_reuse(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    policy: &ReusePolicy,
    r: PrefixReuse,
) -> usize {
    let slot = r.slot_idx;
    let c = &mut cached[slot];
    // Where a sliding window has moved to: positions below it are gone.
    let pos_min = i64::from(ctx.kv_cache_seq_pos_min(slot as i32));
    let planned =
        crate::prefix_cache::plan_rewind(policy.kind, c.ids.len(), r.shared, &c.checkpoints.positions(), pos_min);

    // Everything from the divergence onward was computed under a different
    // prefix, so it is wrong rather than merely old. Each rewind is trusted
    // only once the cache confirms it: `llama_memory_seq_rm` returns true
    // without removing anything when a cache shares cells, and a cache that
    // kept its old positions is not a slow path — on an M-RoPE model the next
    // batch must start strictly beyond the highest cached position, so a stale
    // entry makes the request unschedulable and it fails outright.
    let drop_from = |ctx: &mut llama_cpp_2::context::LlamaContext, pos: usize| {
        ctx.clear_kv_cache_seq(Some(slot as u32), Some(pos as u32), None)
            .unwrap_or(false)
            && ctx.kv_cache_seq_pos_max(slot as i32) < pos as i32
    };
    let done = match planned {
        // The prompt extends everything the slot holds: nothing to drop. On a
        // model with recurrent state this is the one rewind that needs no
        // checkpoint.
        Rewind::Keep(_) | Rewind::Reset => true,
        Rewind::Trim(n) => drop_from(ctx, crate::prefix_cache::kv_pos(&c.spans, n)),
        // Restore the untrimmable part as it stood at the checkpoint, then
        // drop attention from there on. In that order: with the recurrent
        // state back at the checkpoint, the trim touches attention only, which
        // is the part that can rewind.
        Rewind::Restore(n) => {
            c.checkpoints.get(n).is_some_and(|state| ctx.state_seq_set(state, slot as i32).is_ok())
                && drop_from(ctx, crate::prefix_cache::kv_pos(&c.spans, n))
        }
    };
    let rewind = if done { planned } else { Rewind::Reset };
    if rewind == Rewind::Reset {
        // Nothing reusable, or a rewind that did not take and left the
        // sequence in an unknown state: clear it and prefill from zero.
        let _ = ctx.clear_kv_cache_seq(Some(slot as u32), None, None);
        c.forget();
    }
    let reused = rewind.reused();
    c.truncate(reused);
    let kv = crate::prefix_cache::kv_pos(&c.spans, reused);

    let Some(s) = slots[slot].as_mut() else {
        return kv;
    };
    debug_assert_eq!(s.seq_id as usize, slot, "a slot decodes only its own KV sequence");
    let prompt = s.pending_prompt.as_deref().unwrap_or_default();
    s.prefill_cursor = reused;
    s.n_past = kv as i32;
    s.resident = prompt[..reused.min(prompt.len())].to_vec();
    s.cached_tokens = reused as u32;
    if policy.per_slot > 0 {
        let raw: Vec<i32> = prompt.iter().map(|t| t.0).collect();
        s.cuts = crate::prefix_cache::checkpoint_cuts(
            &policy.turn_ends.turn_starts(&raw),
            reused,
            prompt.len(),
            policy.per_slot,
            CHECKPOINT_MIN_SPACING,
        );
        let spans = &s.media_spans;
        s.cuts.retain(|&cut| crate::prefix_cache::snap_out_of_media(spans, cut) == cut);
    }
    match (planned, rewind) {
        (Rewind::Reset, _) if r.shared > 0 => tracing::info!(
            slot,
            shared = r.shared,
            prompt_tokens = prompt.len(),
            "prefix shared but not reachable (no checkpoint at or below the divergence); prefilling from zero"
        ),
        (_, Rewind::Reset) if planned != Rewind::Reset => tracing::warn!(
            slot,
            wanted_reuse = planned.reused(),
            "prefix rewind refused by the cache; prefilling from zero"
        ),
        (_, Rewind::Reset) => {}
        (_, rewind) => tracing::info!(
            slot,
            reused_tokens = reused,
            prompt_tokens = prompt.len(),
            how = match rewind {
                Rewind::Keep(_) => "extension",
                Rewind::Trim(_) => "trim",
                _ => "checkpoint",
            },
            "prefix cache hit"
        ),
    }
    kv
}

/// A request's prompt, tokenized and ready to prefill.
struct Staged {
    /// One token per identity position; a media chunk's positions hold a
    /// placeholder, since the media step evaluates them.
    tokens: Vec<LlamaToken>,
    ids: Vec<PrefixId>,
    spans: Vec<MediaSpan>,
    #[cfg(feature = "mtmd")]
    media: Option<PendingMedia>,
}

/// The placeholder standing in for a media position among prompt tokens.
const MEDIA_PLACEHOLDER: LlamaToken = LlamaToken(-1);

/// Render the request's prompt with its thinking mode and effort level.
fn render_request(model: &LlamaModel, req: &BatchRequest, enable_thinking: bool) -> Result<String> {
    render_with_config(model, &req.prompt, &req.config, enable_thinking)
}

/// Render `prompt` with the thinking mode and effort level `config` asks for,
/// `enable_thinking` being the engine's default.
pub(crate) fn render_with_config(
    model: &LlamaModel,
    prompt: &BatchPrompt,
    config: &GenerationConfig,
    enable_thinking: bool,
) -> Result<String> {
    // The engine's thinking mode is the default; a request may override it.
    // This is what lets one served model answer plainly to API callers and
    // show its reasoning to a client that asked to see it.
    let enable_thinking = config.enable_thinking.unwrap_or(enable_thinking);
    // The effort level is a thinking-mode knob; with thinking off the template
    // never reads it, and some templates reject it outright.
    let reasoning = if enable_thinking {
        config.reasoning_effort.as_deref().map(|value| {
            (
                config.reasoning_kwarg.as_deref().unwrap_or("reasoning_effort"),
                value,
            )
        })
    } else {
        None
    };
    render_prompt(model, prompt, enable_thinking, reasoning)
}

/// Tokenize a text prompt.
fn stage_text(model: &LlamaModel, prompt: &str) -> Result<Staged> {
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| Error::Other(format!("tokenization failed: {}", e)))?;
    Ok(Staged {
        ids: text_ids(&tokens),
        tokens,
        spans: Vec::new(),
        #[cfg(feature = "mtmd")]
        media: None,
    })
}

/// The free slot to serve a prompt from.
///
/// The one whose cache can be rewound to the longest prefix of this prompt,
/// counting only caches filled from the request's own namespace. With no
/// reachable prefix anywhere, an empty slot, so no other conversation's cache
/// is thrown away; failing that, the least recently used one.
///
/// Routing a conversation back to the slot that holds it matters: an agent's
/// next turn repeats everything it has said so far, and on a busy node the
/// first free slot is usually somebody else's. On a model with recurrent
/// state a partial match is also worth nothing without a checkpoint below it,
/// which is why reachability, not the raw match, decides.
fn choose_slot(free: &[bool], cached: &[CachedPrefix], kind: Reuse, namespace: Namespace, ids: &[PrefixId], spans: &[MediaSpan]) -> Option<usize> {
    let reach = |i: usize| {
        let c = &cached[i];
        if c.namespace != namespace || c.ids.is_empty() {
            return 0;
        }
        let shared = crate::prefix_cache::snap_out_of_media(spans, crate::prefix_cache::shared_prefix(&c.ids, ids));
        crate::prefix_cache::plan_rewind(kind, c.ids.len(), shared, &c.checkpoints.positions(), i64::MIN).reused()
    };
    let candidates = || (0..free.len()).filter(|&i| free[i]);
    let best = candidates().max_by_key(|&i| (reach(i), std::cmp::Reverse(i)))?;
    if reach(best) > 0 {
        return Some(best);
    }
    candidates()
        .find(|&i| cached[i].ids.is_empty())
        .or_else(|| candidates().min_by_key(|&i| cached[i].last_used))
}

/// Put `seq` into `slots[slot_idx]`, as KV sequence `slot_idx`. The one place
/// a sequence enters a slot, so a sequence can never decode into another
/// slot's KV.
fn install(slots: &mut [Option<Sequence>], slot_idx: usize, mut seq: Sequence) {
    seq.seq_id = slot_idx as i32;
    slots[slot_idx] = Some(seq);
}

/// Tokenize an admitted request, choose the slot whose cache serves it best,
/// and stage its prompt for prefill. The slot's cache is rewound by
/// [`apply_prefix_reuse`] once the returned plan reaches the scheduler.
#[allow(clippy::too_many_arguments)]
fn admit(
    model: &LlamaModel,
    ctx_size: i32,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    policy: &ReusePolicy,
    req: BatchRequest,
    enable_thinking: bool,
    projector: Option<&Projector>,
) -> Option<PrefixReuse> {
    let free: Vec<bool> = slots.iter().map(Option::is_none).collect();
    if !free.contains(&true) {
        // No free slot — reject rather than block. Caller sheds load.
        let _ = req.result_tx.send(Err(Error::QueueFull {
            model_id: "batched".into(),
            waiting: slots.len(),
            max: slots.len(),
        }));
        return None;
    }
    let prompt = match render_request(model, &req, enable_thinking) {
        Ok(p) => p,
        Err(e) => {
            let _ = req.result_tx.send(Err(e));
            return None;
        }
    };
    let frame = ReasoningFrame::for_prompt(&prompt);
    let staged = if req.media.is_empty() {
        stage_text(model, &prompt)
    } else {
        stage_media(prompt, &req.media, projector)
    };
    let staged = match staged {
        Ok(s) if s.ids.is_empty() => {
            let _ = req.result_tx.send(Err(Error::Other("empty prompt".into())));
            return None;
        }
        Ok(s) => s,
        Err(e) => {
            let _ = req.result_tx.send(Err(e));
            return None;
        }
    };
    let input_tokens = staged.ids.len() as u32;
    // A prompt that alone fills the context leaves no room to generate.
    if input_tokens as i32 >= ctx_size {
        let _ = req.result_tx.send(Err(Error::Inference(format!(
            "prompt of {} tokens exceeds context window {}",
            input_tokens, ctx_size
        ))));
        return None;
    }
    let namespace = Namespace::of(req.config.cache_salt.as_deref());
    let slot_idx = choose_slot(&free, cached, policy.kind, namespace, &staged.ids, &staged.spans)?;
    let shared = if cached[slot_idx].namespace == namespace {
        crate::prefix_cache::snap_out_of_media(
            &staged.spans,
            crate::prefix_cache::shared_prefix(&cached[slot_idx].ids, &staged.ids),
        )
    } else {
        0
    };
    let max_pos = ctx_size.min(input_tokens as i32 + req.config.max_tokens as i32);
    let seq = Sequence::new(model, req.config, req.token_tx, req.result_tx, frame, staged, namespace, max_pos);
    install(slots, slot_idx, seq);
    Some(PrefixReuse { slot_idx, shared })
}

impl Sequence {
    /// A sequence about to prefill `staged`, not yet placed in a slot (see
    /// [`install`]) and not yet rewound (see [`apply_prefix_reuse`]).
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &LlamaModel,
        config: GenerationConfig,
        token_tx: Option<tokio::sync::mpsc::Sender<String>>,
        result_tx: tokio::sync::oneshot::Sender<Result<InferenceResult>>,
        frame: ReasoningFrame,
        staged: Staged,
        namespace: Namespace,
        max_pos: i32,
    ) -> Self {
        Sequence {
            seq_id: -1,
            sampler: build_sampler(&config, model.n_vocab()),
            token_tx,
            result_tx: Some(result_tx),
            decoder: encoding_rs::UTF_8.new_decoder(),
            prefill_cursor: 0,
            n_past: 0,
            resident: Vec::new(),
            input_tokens: staged.ids.len() as u32,
            prompt_ids: staged.ids,
            multimodal: !staged.spans.is_empty(),
            media_spans: staged.spans,
            pending_prompt: Some(staged.tokens),
            namespace,
            cuts: Vec::new(),
            cached_tokens: 0,
            commitment: CommitmentLog::new(&config),
            output_tokens: 0,
            forced_tokens: 0,
            max_pos,
            reasoning_budget: reasoning_budget(&config),
            close_tokens: close_marker_tokens(model, frame),
            stream: StopStream::new(config.stop).framed(frame).nonblocking().with_reasoning(config.reasoning_tx),
            started: Instant::now(),
            speculate: false,
            reasoning_tokens: 0,
            forced: std::collections::VecDeque::new(),
            budget_closed: false,
            pending_token: None,
            #[cfg(feature = "mtmd")]
            pending_media: staged.media,
            ticket: None,
            exporter: None,
            rollback_pending: false,
        }
    }
}

/// Admit a request to an encoder-decoder model into slot 0, and run its
/// encoder, unless the context already holds the encoder output for exactly
/// this input in this namespace; then the encoder pass is skipped and the
/// request only decodes.
///
/// The context holds one encoder output at a time and the decoder's cache
/// cross-attends to it, so an encoder-decoder model decodes one request at a
/// time, and its decoder cache is never reused across requests.
#[allow(clippy::too_many_arguments)]
fn admit_encoder_decoder(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    ctx_size: i32,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    req: BatchRequest,
    enable_thinking: bool,
    encoded: &mut Option<(Namespace, Vec<LlamaToken>)>,
) {
    if !req.media.is_empty() {
        let _ = req.result_tx.send(Err(Error::Inference(
            "this encoder-decoder model takes text only".into(),
        )));
        return;
    }
    let prompt = match render_request(model, &req, enable_thinking).and_then(|p| stage_text(model, &p)) {
        Ok(s) if !s.tokens.is_empty() => s.tokens,
        Ok(_) => {
            let _ = req.result_tx.send(Err(Error::Other("empty prompt".into())));
            return;
        }
        Err(e) => {
            let _ = req.result_tx.send(Err(e));
            return;
        }
    };
    if prompt.len() as i32 >= ctx_size {
        let _ = req.result_tx.send(Err(Error::Inference(format!(
            "prompt of {} tokens exceeds context window {}",
            prompt.len(),
            ctx_size
        ))));
        return;
    }
    let namespace = Namespace::of(req.config.cache_salt.as_deref());
    let hit = encoded.as_ref().is_some_and(|(ns, tokens)| *ns == namespace && *tokens == prompt);
    if !hit {
        *encoded = None;
        let mut batch = LlamaBatch::new(prompt.len(), 1);
        let encoded_ok = batch.add_sequence(&prompt, 0, false).is_ok() && ctx.encode(&mut batch).is_ok();
        if !encoded_ok {
            let _ = req.result_tx.send(Err(Error::Inference("the encoder pass failed".into())));
            return;
        }
        *encoded = Some((namespace, prompt.clone()));
    }
    let _ = ctx.clear_kv_cache_seq(Some(0), None, None);
    cached[0].forget();
    let start = match model.decode_start_token() {
        t if t.0 >= 0 => t,
        _ => model.token_bos(),
    };
    let frame = ReasoningFrame::default();
    let staged = Staged {
        ids: text_ids(&[start]),
        tokens: vec![start],
        spans: Vec::new(),
        #[cfg(feature = "mtmd")]
        media: None,
    };
    let max_pos = ctx_size.min(1 + req.config.max_tokens as i32);
    let mut seq = Sequence::new(model, req.config, req.token_tx, req.result_tx, frame, staged, namespace, max_pos);
    // Billed and reported against the encoder input, which is the prompt.
    seq.input_tokens = prompt.len() as u32;
    seq.max_pos = max_pos + seq.input_tokens as i32 - 1;
    seq.cached_tokens = if hit { prompt.len() as u32 } else { 0 };
    if hit {
        tracing::info!(prompt_tokens = prompt.len(), "encoder output reused; encoder pass skipped");
    }
    install(slots, 0, seq);
}

/// Tokenize a prompt that carries media into text and media chunks.
///
/// Each attachment is identified by its content: its bitmap's id is the hash
/// of its bytes, so its positions in the prompt's identity match another
/// prompt's only where that prompt carries the same bytes there. That is what
/// lets a conversation about an image reuse the image's embeddings on its next
/// turn, and never reuse them for a different image.
#[cfg(feature = "mtmd")]
fn stage_media(prompt: String, media: &[Vec<u8>], projector: Option<&Projector>) -> Result<Staged> {
    let Some(projector) = projector else {
        return Err(Error::Inference(
            "this model is served text-only: it has no projector, so it cannot take image or \
             audio attachments"
                .into(),
        ));
    };
    let mut bitmaps = Vec::with_capacity(media.len());
    for (i, bytes) in media.iter().enumerate() {
        let mut bitmap = MtmdBitmap::from_buffer(projector, bytes, false)
            .map_err(|e| Error::Inference(format!("attachment {} could not be decoded: {}", i, e)))?;
        let (kind, supported) = if bitmap.is_audio() {
            ("audio", projector.support_audio())
        } else {
            ("an image", projector.support_vision())
        };
        if !supported {
            return Err(Error::Inference(format!(
                "attachment {} is {}, which this projector has no tower for",
                i, kind
            )));
        }
        bitmap
            .set_id(&hex::encode(crate::prefix_cache::media_hash(bytes)))
            .map_err(|e| Error::Inference(format!("attachment {} could not be identified: {}", i, e)))?;
        bitmaps.push(bitmap);
    }
    let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    let chunks = projector
        .tokenize(
            MtmdInputText {
                text: prompt,
                add_special: true,
                parse_special: true,
            },
            &refs,
        )
        .map_err(|e| Error::Inference(format!("multimodal tokenization failed: {}", e)))?;

    let mut tokens = Vec::with_capacity(chunks.total_tokens());
    let mut ids = Vec::with_capacity(chunks.total_tokens());
    let mut spans = Vec::new();
    let mut chunk_of_span = Vec::new();
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for index in 0..chunks.len() {
        let Some(chunk) = chunks.get(index) else { continue };
        if chunk.chunk_type() == llama_cpp_2::mtmd::MtmdInputChunkType::Text {
            let text = chunk.text_tokens().unwrap_or_default();
            ids.extend(text.iter().map(|t| crate::prefix_cache::text_id(t.0)));
            tokens.extend_from_slice(text);
            continue;
        }
        let len = chunk.n_tokens();
        // One attachment may span several chunks (long audio); each chunk of
        // it has its own identity.
        let id = chunk.id().unwrap_or_default();
        let ordinal = seen.entry(id.clone()).or_insert(0);
        let content = crate::prefix_cache::media_hash(format!("{id}/{ordinal}").as_bytes());
        *ordinal += 1;
        spans.push(MediaSpan {
            at: ids.len(),
            len,
            n_pos: usize::try_from(chunk.n_positions()).unwrap_or(len),
        });
        chunk_of_span.push(index);
        ids.extend(crate::prefix_cache::media_ids(&content, len));
        tokens.extend(std::iter::repeat_n(MEDIA_PLACEHOLDER, len));
    }
    Ok(Staged {
        tokens,
        ids,
        spans,
        media: Some(PendingMedia { chunks, chunk_of_span }),
    })
}

/// Refuse media in a build without mtmd.
#[cfg(not(feature = "mtmd"))]
fn stage_media(_prompt: String, _media: &[Vec<u8>], _projector: Option<&Projector>) -> Result<Staged> {
    Err(Error::Inference("this engine was built without multimodal support".into()))
}

/// Evaluate the next media chunk of at most one sequence: the first whose
/// prefill has reached one. Text around media is prefilled in the ordinary
/// batch; a media chunk the slot's cache already held was never staged here,
/// because prefill resumes past it.
#[cfg(feature = "mtmd")]
#[allow(clippy::too_many_arguments)]
fn media_step(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    draft: &mut Option<llama_cpp_2::context::LlamaContext<'_>>,
    projector: &Projector,
    slots: &mut [Option<Sequence>],
    cached: &mut [CachedPrefix],
    kind: Reuse,
    draining: &mut Vec<Draining>,
    model_id: &str,
) {
    let next = slots.iter().enumerate().find_map(|(i, s)| {
        let q = s.as_ref()?;
        q.pending_prompt.as_ref()?;
        q.pending_media.as_ref()?;
        let k = q.media_spans.iter().position(|sp| sp.at == q.prefill_cursor)?;
        Some((i, k))
    });
    let Some((slot_idx, k)) = next else {
        return;
    };
    let evaluated = {
        let seq = slots[slot_idx].as_mut().expect("found above");
        let span = seq.media_spans[k];
        let prompt_len = seq.pending_prompt.as_ref().map_or(0, Vec::len);
        // A prompt that ends in media takes its logits from the media chunk.
        let last = span.at + span.len >= prompt_len;
        let media = seq.pending_media.as_ref().expect("found above");
        let n_batch = ctx.n_batch() as i32;
        match media.chunks.eval_chunk(media.chunk_of_span[k], projector, ctx, seq.n_past, seq.seq_id, n_batch, last) {
            Ok(n_past) => {
                seq.n_past = n_past;
                seq.prefill_cursor += span.len;
                seq.resident.extend(std::iter::repeat_n(MEDIA_PLACEHOLDER, span.len));
                if last {
                    seq.pending_prompt = None;
                    seq.pending_media = None;
                }
                Ok(last)
            }
            Err(e) => Err(e),
        }
    };
    match evaluated {
        Ok(true) => {
            if sample_into(model, ctx, slots, slot_idx, -1, model_id) {
                finalize_and_free(ctx, draft, slots, cached, kind, slot_idx, draining);
            }
        }
        Ok(false) => {}
        Err(e) => {
            // The chunk may have written part of its positions: start this
            // request over from nothing, after making room if there is any
            // to make, or fail it.
            let _ = ctx.clear_kv_cache_seq(Some(slot_idx as u32), None, None);
            draft_seq_rm(draft, slot_idx as i32, None);
            cached[slot_idx].forget();
            if evict_idle_prefix(ctx, draft, slots, cached) {
                if let Some(seq) = slots[slot_idx].as_mut() {
                    seq.prefill_cursor = 0;
                    seq.n_past = 0;
                    seq.resident.clear();
                    seq.cached_tokens = 0;
                }
            } else if let Some(mut seq) = slots[slot_idx].take() {
                seq.fail(Error::Inference(format!("multimodal prefill failed: {}", e)));
            }
        }
    }
}

/// What `admit` decided about an incoming request's prefix.
///
/// Returned rather than acted on inside `admit`, because rewinding the KV
/// cache needs the context and `admit` deliberately does not take it —
/// tokenizing must not be able to touch the cache.
struct PrefixReuse {
    slot_idx: usize,
    /// Identity positions the prompt shares with the slot's cache, never
    /// splitting a media chunk. How much of it is reachable depends on the
    /// model's memory, which [`apply_prefix_reuse`] decides.
    shared: usize,
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

    fn ids(v: &[i32]) -> Vec<PrefixId> {
        v.iter().map(|t| crate::prefix_cache::text_id(*t)).collect()
    }

    fn slot_with(v: &[i32], namespace: Namespace, last_used: u64) -> CachedPrefix {
        CachedPrefix {
            ids: ids(v),
            namespace,
            last_used,
            ..Default::default()
        }
    }

    fn pick(cached: &[CachedPrefix], free: &[bool], prompt: &[i32]) -> usize {
        choose_slot(free, cached, Reuse::Trim, Namespace::default(), &ids(prompt), &[]).expect("a free slot")
    }

    #[test]
    fn a_turn_goes_back_to_the_slot_holding_its_conversation() {
        // Slot 0 is free and empty; slot 2 holds this conversation. Taking the
        // first free slot would prefill the whole prompt from zero.
        let ns = Namespace::default();
        let cached = [slot_with(&[], ns, 0), slot_with(&[9, 9, 9], ns, 0), slot_with(&[1, 2, 3, 4, 5], ns, 0)];
        assert_eq!(pick(&cached, &[true, false, true], &[1, 2, 3, 4, 5, 6, 7]), 2);
    }

    #[test]
    fn a_busy_better_slot_is_not_stolen() {
        let ns = Namespace::default();
        let cached = [slot_with(&[], ns, 0), slot_with(&[1, 2, 3, 4, 5], ns, 0), slot_with(&[1, 2], ns, 0)];
        assert_eq!(pick(&cached, &[true, false, true], &[1, 2, 3, 4, 5, 6]), 2, "the best FREE slot");
    }

    #[test]
    fn with_nothing_reachable_an_empty_slot_is_taken_before_anyone_elses_cache() {
        // Slot 0 holds another conversation, slot 2 nothing: taking slot 0
        // would throw that conversation away for no gain.
        let ns = Namespace::default();
        let cached = [slot_with(&[8, 8], ns, 5), slot_with(&[7, 7], ns, 1), slot_with(&[], ns, 0)];
        assert_eq!(pick(&cached, &[true, true, true], &[1, 2, 3]), 2);
        // No empty slot: the least recently used cache goes.
        let cached = [slot_with(&[8, 8], ns, 5), slot_with(&[7, 7], ns, 1)];
        assert_eq!(pick(&cached, &[true, true], &[1, 2, 3]), 1);
    }

    #[test]
    fn a_recurrent_model_does_not_chase_a_match_it_cannot_reach() {
        // Slot 0 shares 4 tokens but holds 6 and has no checkpoint at or below
        // the divergence: on a recurrent model that is worth nothing, so the
        // empty slot is taken and slot 0's conversation survives.
        let ns = Namespace::default();
        let cached = [slot_with(&[1, 2, 3, 4, 5, 6], ns, 0), slot_with(&[], ns, 0)];
        let prompt = ids(&[1, 2, 3, 4, 50, 60]);
        assert_eq!(choose_slot(&[true, true], &cached, Reuse::Recurrent, ns, &prompt, &[]), Some(1));
        assert_eq!(choose_slot(&[true, true], &cached, Reuse::Trim, ns, &prompt, &[]), Some(0));
    }

    #[test]
    fn another_namespaces_cache_is_never_matched() {
        // Slot 0 holds exactly this prompt, filled under another key. It is
        // not reused, and the choice does not depend on it: the request lands
        // on the empty slot exactly as it would if slot 0 held anything else.
        let alice = Namespace::of(Some("alice"));
        let bob = Namespace::of(Some("bob"));
        let prompt = [1, 2, 3, 4, 5, 6];
        let cached = [slot_with(&prompt, alice, 9), slot_with(&[], bob, 0)];
        assert_eq!(choose_slot(&[true, true], &cached, Reuse::Trim, bob, &ids(&prompt), &[]), Some(1));
        let unrelated = [slot_with(&[40, 41], alice, 9), slot_with(&[], bob, 0)];
        assert_eq!(choose_slot(&[true, true], &unrelated, Reuse::Trim, bob, &ids(&prompt), &[]), Some(1));
        // The owner does reuse it.
        assert_eq!(choose_slot(&[true, true], &cached, Reuse::Trim, alice, &ids(&prompt), &[]), Some(0));
    }

    #[test]
    fn no_free_slot_is_no_choice() {
        let cached = [slot_with(&[1, 2], Namespace::default(), 0)];
        assert_eq!(choose_slot(&[false], &cached, Reuse::Trim, Namespace::default(), &ids(&[1, 2]), &[]), None);
    }

    /// A sequence with nothing staged, for placement tests.
    fn bare_sequence(seq_id: i32) -> Sequence {
        let (result_tx, _) = tokio::sync::oneshot::channel();
        Sequence {
            seq_id,
            sampler: LlamaSampler::greedy(),
            token_tx: None,
            result_tx: Some(result_tx),
            decoder: encoding_rs::UTF_8.new_decoder(),
            pending_prompt: None,
            prefill_cursor: 0,
            n_past: 0,
            resident: Vec::new(),
            prompt_ids: Vec::new(),
            media_spans: Vec::new(),
            namespace: Namespace::default(),
            cuts: Vec::new(),
            cached_tokens: 0,
            commitment: None,
            input_tokens: 0,
            output_tokens: 0,
            forced_tokens: 0,
            max_pos: 0,
            stream: StopStream::new(vec![]),
            started: Instant::now(),
            pending_token: None,
            multimodal: false,
            speculate: false,
            reasoning_tokens: 0,
            reasoning_budget: None,
            close_tokens: Vec::new(),
            forced: std::collections::VecDeque::new(),
            budget_closed: false,
            #[cfg(feature = "mtmd")]
            pending_media: None,
            ticket: None,
            exporter: None,
            rollback_pending: false,
        }
    }

    #[test]
    fn the_chosen_slot_is_the_kv_sequence_the_request_decodes_into() {
        // The request is routed to slot 2 while slot 0 is the first free one.
        // Whatever sequence id it arrived with, it must decode into slot 2's
        // KV, the one whose prefix was reused.
        let ns = Namespace::default();
        let cached = [slot_with(&[], ns, 0), slot_with(&[9, 9], ns, 0), slot_with(&[1, 2, 3, 4, 5], ns, 0)];
        let free = [true, false, true];
        let chosen = pick(&cached, &free, &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(chosen, 2);
        let mut slots: Vec<Option<Sequence>> = (0..3).map(|_| None).collect();
        install(&mut slots, chosen, bare_sequence(0));
        let seq = slots[chosen].as_ref().expect("installed");
        assert_eq!(seq.seq_id as usize, chosen, "slots[pick].seq_id == pick");
        assert!(slots[0].is_none(), "the first free slot is left free");
    }
}
