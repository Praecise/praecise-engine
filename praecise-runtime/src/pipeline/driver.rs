//! The first stage of a pipeline: owns the token embedding and the first blocks, feeds
//! micro-batches down the chain and collects logits from the last stage.

use std::collections::{BTreeSet, HashMap};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use super::stage::{check_model, context_params, truncate, StageOptions};
use super::transport::{self, handshake, HandshakeError, LinkCloser, Role};
use super::wire::{Forward, Message, ALL_TICKETS};
use super::{PipelineError, PipelinePlan, StageAuthenticator};

/// Driver settings.
#[derive(Debug, Clone)]
pub struct DriverOptions {
    /// Engine settings shared with every stage.
    pub stage: StageOptions,
    /// Micro-batches allowed in flight before `submit` waits for one to finish.
    pub max_in_flight: usize,
    /// How long to wait for any answer from the chain before declaring it stalled.
    pub answer_timeout: Duration,
}

impl Default for DriverOptions {
    fn default() -> Self {
        Self { stage: StageOptions::default(), max_in_flight: 8, answer_timeout: Duration::from_secs(120) }
    }
}

/// Logits rows returned for one ticket, in the order of the rows asked for.
pub type LogitsRows = Vec<Vec<f32>>;

enum Upstream {
    Logits { ticket: u64, rows: LogitsRows },
    Failure { ticket: u64, stage: usize, reason: String },
    Closed(String),
}

/// A connected pipeline, driven from the machine that holds the first blocks.
pub struct Pipeline<'m> {
    ctx: LlamaContext<'m>,
    n_embd: usize,
    n_stages: usize,
    max_rows: usize,
    max_in_flight: usize,
    timeout: Duration,
    down: Option<mpsc::Sender<Vec<u8>>>,
    up: mpsc::Receiver<Upstream>,
    closer: LinkCloser,
    next_ticket: u64,
    pending: BTreeSet<u64>,
    done: HashMap<u64, Result<LogitsRows, (usize, String)>>,
    failed: Option<(usize, String)>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for Pipeline<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("n_stages", &self.n_stages)
            .field("in_flight", &self.pending.len())
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}

impl<'m> Pipeline<'m> {
    /// Connect to stage 1 (which connects onward) and prepare the local blocks.
    /// `model` is this machine's stage, loaded with the plan's first layer range.
    ///
    /// # Errors
    ///
    /// A plan that does not fit the model, an identity the plan does not give stage 0,
    /// an unreachable or unauthenticated stage 1, or a context the engine refuses.
    pub fn connect(
        backend: &LlamaBackend,
        model: &'m LlamaModel,
        plan: &PipelinePlan,
        auth: &dyn StageAuthenticator,
        opts: &DriverOptions,
    ) -> Result<Self, PipelineError> {
        check_model(model, plan, 0)?;
        if plan.stages[0].identity != auth.identity() {
            return Err(PipelineError::Plan("this machine's identity is not the one the plan gives stage 0".into()));
        }
        let next = &plan.stages[1];
        let stream = transport::dial(&next.address, opts.stage.connect_timeout).map_err(|e| PipelineError::StageFailed {
            stage: 1,
            reason: format!("cannot reach {}: {e}", next.address),
        })?;
        let (mut reader, mut writer) =
            handshake(stream, auth, Role::Initiator, &next.identity, &plan.digest()).map_err(|e| match e {
                HandshakeError::Rejected(r) => PipelineError::Authentication { stage: 1, reason: r },
                HandshakeError::Io(e) => PipelineError::StageFailed { stage: 1, reason: format!("handshake: {e}") },
            })?;
        let ctx = model
            .new_context(backend, context_params(&opts.stage, false))
            .map_err(|e| PipelineError::Engine(format!("context: {e}")))?;
        let closer = writer.closer()?;

        let (down_tx, down_rx) = mpsc::channel::<Vec<u8>>();
        let (up_tx, up_rx) = mpsc::channel::<Upstream>();
        let mut threads = Vec::new();
        {
            let up_tx = up_tx.clone();
            threads.push(std::thread::spawn(move || {
                for frame in down_rx {
                    if let Err(e) = writer.send(&frame) {
                        let _ = up_tx.send(Upstream::Closed(format!("link to stage 1 failed: {e}")));
                        return;
                    }
                }
                writer.shutdown();
            }));
        }
        threads.push(std::thread::spawn(move || loop {
            let item = match reader.recv() {
                Ok(Some(bytes)) => match Message::decode(&bytes) {
                    Ok(Message::Logits { ticket, rows, width, data }) => {
                        let w = width as usize;
                        let rows = (0..rows as usize).map(|r| data[r * w..(r + 1) * w].to_vec()).collect();
                        Upstream::Logits { ticket, rows }
                    }
                    Ok(Message::Failure { ticket, stage, reason }) => Upstream::Failure { ticket, stage: stage as usize, reason },
                    Ok(_) => Upstream::Closed("stage 1 sent an unexpected message".into()),
                    Err(e) => Upstream::Closed(format!("stage 1: {e}")),
                },
                Ok(None) => Upstream::Closed("stage 1 closed its link".into()),
                Err(e) => Upstream::Closed(format!("link to stage 1 failed: {e}")),
            };
            let end = matches!(item, Upstream::Closed(_));
            if up_tx.send(item).is_err() || end {
                return;
            }
        }));

        Ok(Self {
            ctx,
            n_embd: plan.n_embd as usize,
            n_stages: plan.stages.len(),
            max_rows: opts.stage.max_rows as usize,
            max_in_flight: opts.max_in_flight.max(1),
            timeout: opts.answer_timeout,
            down: Some(down_tx),
            up: up_rx,
            closer,
            next_ticket: 0,
            pending: BTreeSet::new(),
            done: HashMap::new(),
            failed: None,
            threads,
        })
    }

    /// Number of stages, this one included.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    fn failure(&self) -> Option<PipelineError> {
        self.failed.as_ref().map(|(stage, reason)| PipelineError::StageFailed { stage: *stage, reason: reason.clone() })
    }

    /// Run `tokens` (positions `start_pos..`) of sequence `seq` through the local blocks
    /// and send the hidden states on, without waiting for the rest of the chain. The
    /// last stage returns logits for the rows marked in `outputs`. Returns the ticket to
    /// [`Self::wait`] on.
    ///
    /// # Errors
    ///
    /// An earlier stage failure, an oversized micro-batch, or a local engine error.
    pub fn submit(&mut self, seq: i32, start_pos: i32, tokens: &[LlamaToken], outputs: &[bool]) -> Result<u64, PipelineError> {
        if let Some(e) = self.failure() {
            return Err(e);
        }
        let rows = tokens.len();
        if rows == 0 || rows > self.max_rows || outputs.len() != rows {
            return Err(PipelineError::Engine(format!("micro-batch of {rows} rows; 1..={} with one output flag each", self.max_rows)));
        }
        while self.pending.len() >= self.max_in_flight {
            self.pump()?;
        }

        let mut batch = LlamaBatch::new(rows, 1);
        let mut positions = Vec::with_capacity(rows);
        for (i, t) in tokens.iter().enumerate() {
            let pos = start_pos + i as i32;
            batch.add(*t, pos, &[seq], true).map_err(|e| PipelineError::Engine(e.to_string()))?;
            positions.push(pos);
        }
        self.ctx.decode(&mut batch).map_err(|e| PipelineError::Engine(format!("decode: {e}")))?;
        let mut hidden = Vec::with_capacity(rows * self.n_embd);
        for i in 0..rows {
            let h = self.ctx.embeddings_ith(i as i32).map_err(|e| PipelineError::Engine(format!("hidden state {i}: {e}")))?;
            hidden.extend_from_slice(h);
        }

        let ticket = self.next_ticket;
        self.next_ticket += 1;
        let msg = Message::Forward(Forward {
            ticket,
            positions,
            seqs: vec![seq; rows],
            outputs: outputs.iter().map(|o| u8::from(*o)).collect(),
            width: self.n_embd as u32,
            hidden,
        });
        self.send(msg.encode())?;
        self.pending.insert(ticket);
        Ok(ticket)
    }

    fn send(&mut self, frame: Vec<u8>) -> Result<(), PipelineError> {
        let ok = self.down.as_ref().is_some_and(|d| d.send(frame).is_ok());
        if ok {
            return Ok(());
        }
        let _ = self.pump_for(Duration::from_millis(100));
        Err(self.failure().unwrap_or(PipelineError::StageFailed { stage: 1, reason: "link to stage 1 is closed".into() }))
    }

    /// Wait for one answer from the chain.
    fn pump(&mut self) -> Result<(), PipelineError> {
        let timeout = self.timeout;
        self.pump_for(timeout)
    }

    fn pump_for(&mut self, timeout: Duration) -> Result<(), PipelineError> {
        if let Some(e) = self.failure() {
            return Err(e);
        }
        let deadline = Instant::now() + timeout;
        let item = match self.up.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(item) => item,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(PipelineError::Timeout(format!(
                    "no answer from stages 1..{} within {timeout:?} with {} micro-batches in flight",
                    self.n_stages - 1,
                    self.pending.len()
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Upstream::Closed("link to stage 1 is closed".into()),
        };
        match item {
            Upstream::Logits { ticket, rows } => {
                if self.pending.remove(&ticket) {
                    self.done.insert(ticket, Ok(rows));
                }
            }
            Upstream::Failure { ticket, stage, reason } if ticket != ALL_TICKETS => {
                if self.pending.remove(&ticket) {
                    self.done.insert(ticket, Err((stage, reason)));
                }
            }
            Upstream::Failure { stage, reason, .. } => self.fail(stage, reason),
            Upstream::Closed(reason) => {
                // a stage further down reports its failure before stage 1 closes the link;
                // prefer that report over the bare closed link
                let mut culprit = (1, reason);
                while let Ok(item) = self.up.recv_timeout(Duration::from_millis(200)) {
                    if let Upstream::Failure { ticket: ALL_TICKETS, stage, reason } = item {
                        culprit = (stage, reason);
                        break;
                    }
                }
                self.fail(culprit.0, culprit.1);
            }
        }
        self.failure().map_or(Ok(()), Err)
    }

    fn fail(&mut self, stage: usize, reason: String) {
        if self.failed.is_none() {
            tracing::warn!(stage, %reason, "pipeline stage failed");
            self.failed = Some((stage, reason));
            self.down = None;
            self.closer.close();
        }
    }

    /// Wait for a ticket's logits rows.
    ///
    /// # Errors
    ///
    /// The failure of the stage that failed this ticket or the whole chain, or
    /// [`PipelineError::Timeout`] when the chain stops answering.
    pub fn wait(&mut self, ticket: u64) -> Result<LogitsRows, PipelineError> {
        loop {
            if let Some(r) = self.done.remove(&ticket) {
                return r.map_err(|(stage, reason)| PipelineError::StageFailed { stage, reason });
            }
            if !self.pending.contains(&ticket) {
                return Err(PipelineError::Protocol(format!("ticket {ticket} is not in flight")));
            }
            self.pump()?;
        }
    }

    /// Run a prompt through the pipeline in micro-batches of `micro_batch` rows; every
    /// stage works on a different micro-batch at once. Returns the logits of the last
    /// position.
    ///
    /// # Errors
    ///
    /// As [`Self::submit`] and [`Self::wait`].
    pub fn prefill(&mut self, seq: i32, start_pos: i32, tokens: &[LlamaToken], micro_batch: usize) -> Result<Vec<f32>, PipelineError> {
        let step = micro_batch.clamp(1, self.max_rows);
        let n_chunks = tokens.len().div_ceil(step);
        if n_chunks == 0 {
            return Err(PipelineError::Engine("empty prompt".into()));
        }
        let mut tickets = Vec::with_capacity(n_chunks);
        for (c, chunk) in tokens.chunks(step).enumerate() {
            let mut outputs = vec![false; chunk.len()];
            if c + 1 == n_chunks {
                outputs[chunk.len() - 1] = true;
            }
            tickets.push(self.submit(seq, start_pos + (c * step) as i32, chunk, &outputs)?);
        }
        let mut last = Vec::new();
        for t in tickets {
            last = self.wait(t)?;
        }
        last.pop().ok_or_else(|| PipelineError::Protocol("last stage returned no logits".into()))
    }

    /// Decode one token at `pos` and return its logits.
    ///
    /// # Errors
    ///
    /// As [`Self::submit`] and [`Self::wait`].
    pub fn step(&mut self, seq: i32, pos: i32, token: LlamaToken) -> Result<Vec<f32>, PipelineError> {
        let t = self.submit(seq, pos, &[token], &[true])?;
        self.wait(t)?.pop().ok_or_else(|| PipelineError::Protocol("last stage returned no logits".into()))
    }

    /// Drop positions `from_pos..` of `seq` (all of it for a negative `from_pos`) from
    /// every stage's cache. Ordered after every micro-batch already submitted.
    ///
    /// # Errors
    ///
    /// An earlier stage failure or a local engine error.
    pub fn truncate(&mut self, seq: i32, from_pos: i32) -> Result<(), PipelineError> {
        if let Some(e) = self.failure() {
            return Err(e);
        }
        truncate(&mut self.ctx, seq, from_pos).map_err(PipelineError::Engine)?;
        self.send(Message::Truncate { seq, from_pos }.encode())
    }
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        self.down = None;
        self.closer.close();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
