//! A pipeline stage after the first: receives hidden states from the stage before it,
//! runs its blocks, and passes the result on (or, as the last stage, returns logits).

use std::net::{TcpListener, TcpStream};
use std::num::NonZeroU32;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;

use super::transport::{self, handshake, HandshakeError, Role, SecureWriter};
use super::wire::{Forward, Message, ALL_TICKETS};
use super::{PipelineError, PipelinePlan, StageAuthenticator};

/// Engine settings every stage of one pipeline must share.
#[derive(Debug, Clone)]
pub struct StageOptions {
    /// Context length per stage.
    pub n_ctx: u32,
    /// Largest micro-batch, in rows.
    pub max_rows: u32,
    /// Sequences that may be in flight at once.
    pub n_seq_max: u32,
    /// CPU threads for the local blocks; `None` keeps the engine default.
    pub n_threads: Option<i32>,
    /// How long to wait when dialing the next stage.
    pub connect_timeout: Duration,
}

impl Default for StageOptions {
    fn default() -> Self {
        Self { n_ctx: 4096, max_rows: 64, n_seq_max: 4, n_threads: None, connect_timeout: Duration::from_secs(10) }
    }
}

/// Context parameters for a stage. A stage without the output head returns hidden
/// states through the embeddings output, one row per position. One micro-batch is one
/// engine batch, so a stage computes the same shapes whatever its neighbours do.
#[must_use]
pub fn context_params(opts: &StageOptions, has_output_head: bool) -> LlamaContextParams {
    let mut p = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(opts.n_ctx))
        .with_n_batch(opts.max_rows)
        .with_n_ubatch(opts.max_rows)
        .with_n_seq_max(opts.n_seq_max)
        .with_kv_unified(true);
    if let Some(t) = opts.n_threads {
        p = p.with_n_threads(t).with_n_threads_batch(t);
    }
    if !has_output_head {
        p = p.with_embeddings(true).with_pooling_type(LlamaPoolingType::None);
    }
    p
}

/// Check that a loaded stage model matches its place in the plan.
///
/// # Errors
///
/// [`PipelineError::Plan`] when the model's width or block count disagrees.
pub fn check_model(model: &LlamaModel, plan: &PipelinePlan, index: usize) -> Result<(), PipelineError> {
    plan.validate()?;
    let spec = plan.stages.get(index).ok_or_else(|| PipelineError::Plan(format!("no stage {index}")))?;
    if u32::try_from(model.n_embd()).ok() != Some(plan.n_embd) {
        return Err(PipelineError::Plan(format!("model width {} differs from the plan's {}", model.n_embd(), plan.n_embd)));
    }
    if model.n_layer() != spec.layers.len() {
        return Err(PipelineError::Plan(format!(
            "stage {index} model holds {} blocks, the plan gives it {}..{}",
            model.n_layer(),
            spec.layers.begin,
            spec.layers.end
        )));
    }
    Ok(())
}

/// Accept links from the previous stage forever, serving one session at a time. Each
/// session starts with an empty cache. A peer that fails authentication is dropped and
/// the stage keeps listening; a session that fails is logged and ended.
///
/// # Errors
///
/// A plan that does not fit this model, or an error accepting on `listener`.
pub fn serve(
    listener: &TcpListener,
    backend: &LlamaBackend,
    model: &LlamaModel,
    plan: &PipelinePlan,
    index: usize,
    auth: &dyn StageAuthenticator,
    opts: &StageOptions,
) -> Result<(), PipelineError> {
    if index == 0 {
        return Err(PipelineError::Plan("stage 0 is the driver; it does not listen".into()));
    }
    check_model(model, plan, index)?;
    if plan.stages[index].identity != auth.identity() {
        return Err(PipelineError::Plan(format!("this machine's identity is not the one the plan gives stage {index}")));
    }
    loop {
        let (stream, peer) = listener.accept()?;
        match serve_session(stream, backend, model, plan, index, auth, opts) {
            Ok(()) => tracing::info!(stage = index, %peer, "pipeline session ended"),
            Err(e) => tracing::warn!(stage = index, %peer, error = %e, "pipeline session failed"),
        }
    }
}

enum Inbound {
    Msg(Message),
    UpstreamClosed(Option<String>),
    DownstreamFailed,
}

/// Serve one session on an accepted link from the previous stage.
///
/// # Errors
///
/// Authentication failure, an unreachable next stage, or a failed link.
pub fn serve_session(
    stream: TcpStream,
    backend: &LlamaBackend,
    model: &LlamaModel,
    plan: &PipelinePlan,
    index: usize,
    auth: &dyn StageAuthenticator,
    opts: &StageOptions,
) -> Result<(), PipelineError> {
    let digest = plan.digest();
    let last = index + 1 == plan.stages.len();
    let (mut up_r, up_w) = handshake(stream, auth, Role::Responder, &plan.stages[index - 1].identity, &digest)
        .map_err(|e| PipelineError::Authentication { stage: index - 1, reason: e.to_string() })?;
    let up_closer = up_w.closer()?;
    let up_w = Arc::new(Mutex::new(up_w));

    let fail_all = |stage: usize, reason: String| {
        let msg = Message::Failure { ticket: ALL_TICKETS, stage: stage as u32, reason };
        let _ = up_w.lock().map(|mut w| w.send(&msg.encode()));
    };

    let down = if last {
        None
    } else {
        let next = &plan.stages[index + 1];
        let link = transport::dial(&next.address, opts.connect_timeout)
            .map_err(HandshakeError::Io)
            .and_then(|s| handshake(s, auth, Role::Initiator, &next.identity, &digest));
        match link {
            Ok(l) => Some(l),
            Err(e) => {
                let reason = format!("cannot reach stage {} at {}: {e}", index + 1, next.address);
                fail_all(index + 1, reason.clone());
                up_closer.close();
                return Err(PipelineError::StageFailed { stage: index + 1, reason });
            }
        }
    };

    let mut ctx = match model.new_context(backend, context_params(opts, last)) {
        Ok(c) => c,
        Err(e) => {
            let reason = format!("context: {e}");
            fail_all(index, reason.clone());
            up_closer.close();
            return Err(PipelineError::Engine(reason));
        }
    };

    let (tx, rx) = mpsc::channel::<Inbound>();
    std::thread::scope(|scope| -> Result<(), PipelineError> {
        {
            let tx = tx.clone();
            scope.spawn(move || loop {
                match up_r.recv() {
                    Ok(Some(bytes)) => match Message::decode(&bytes) {
                        Ok(m) => {
                            if tx.send(Inbound::Msg(m)).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Inbound::UpstreamClosed(Some(e.to_string())));
                            return;
                        }
                    },
                    Ok(None) => {
                        let _ = tx.send(Inbound::UpstreamClosed(None));
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(Inbound::UpstreamClosed(Some(e.to_string())));
                        return;
                    }
                }
            });
        }

        let mut down_tx: Option<mpsc::Sender<Vec<u8>>> = None;
        let mut down_closer = None;
        if let Some((mut dr, mut dw)) = down {
            down_closer = Some(dw.closer()?);
            let (dtx, drx) = mpsc::channel::<Vec<u8>>();
            down_tx = Some(dtx);
            let tx_w = tx.clone();
            let fail_w = &fail_all;
            scope.spawn(move || {
                for frame in drx {
                    if let Err(e) = dw.send(&frame) {
                        fail_w(index + 1, format!("link to stage {} failed: {e}", index + 1));
                        let _ = tx_w.send(Inbound::DownstreamFailed);
                        return;
                    }
                }
                dw.shutdown();
            });
            let tx_r = tx.clone();
            let up_relay = Arc::clone(&up_w);
            let fail_r = &fail_all;
            scope.spawn(move || {
                let reason = loop {
                    match dr.recv() {
                        Ok(Some(bytes)) => {
                            // results and failures from further down pass straight up
                            if up_relay.lock().map(|mut w| w.send(&bytes)).map_or(true, |r| r.is_err()) {
                                return;
                            }
                        }
                        Ok(None) => break format!("stage {} closed its link", index + 1),
                        Err(e) => break format!("link to stage {} failed: {e}", index + 1),
                    }
                };
                fail_r(index + 1, reason);
                let _ = tx_r.send(Inbound::DownstreamFailed);
            });
        }
        drop(tx);

        let result = compute_loop(&rx, &mut ctx, plan, index, opts, &up_w, down_tx.as_ref());

        drop(down_tx);
        if let Some(c) = &down_closer {
            c.close();
        }
        up_closer.close();
        result
    })
}

fn send_up(up: &Mutex<SecureWriter>, msg: &Message) -> Result<(), PipelineError> {
    up.lock()
        .map_err(|_| PipelineError::Protocol("upstream writer poisoned".into()))?
        .send(&msg.encode())
        .map_err(PipelineError::Io)
}

fn compute_loop(
    rx: &mpsc::Receiver<Inbound>,
    ctx: &mut LlamaContext<'_>,
    plan: &PipelinePlan,
    index: usize,
    opts: &StageOptions,
    up: &Mutex<SecureWriter>,
    down: Option<&mpsc::Sender<Vec<u8>>>,
) -> Result<(), PipelineError> {
    let n_embd = plan.n_embd as usize;
    loop {
        let Ok(inbound) = rx.recv() else { return Ok(()) };
        match inbound {
            Inbound::UpstreamClosed(None) => return Ok(()),
            Inbound::UpstreamClosed(Some(e)) => return Err(PipelineError::Protocol(format!("upstream link: {e}"))),
            Inbound::DownstreamFailed => {
                return Err(PipelineError::StageFailed { stage: index + 1, reason: "link to the next stage failed".into() })
            }
            Inbound::Msg(Message::Forward(f)) => {
                let ticket = f.ticket;
                match run_forward(ctx, f, n_embd, opts, down.is_none()) {
                    Ok(Message::Forward(out)) => {
                        let tx = down.ok_or_else(|| PipelineError::Protocol("no next stage".into()))?;
                        if tx.send(Message::Forward(out).encode()).is_err() {
                            return Err(PipelineError::StageFailed { stage: index + 1, reason: "link to the next stage closed".into() });
                        }
                    }
                    Ok(reply) => send_up(up, &reply)?,
                    Err(reason) => {
                        tracing::warn!(stage = index, ticket, %reason, "pipeline stage failed a micro-batch");
                        send_up(up, &Message::Failure { ticket, stage: index as u32, reason })?;
                    }
                }
            }
            Inbound::Msg(Message::Truncate { seq, from_pos }) => {
                truncate(ctx, seq, from_pos).map_err(PipelineError::Engine)?;
                if let Some(tx) = down {
                    let _ = tx.send(Message::Truncate { seq, from_pos }.encode());
                }
            }
            Inbound::Msg(other) => {
                return Err(PipelineError::Protocol(format!("unexpected message from upstream: {other:?}")));
            }
        }
    }
}

pub(crate) fn truncate(ctx: &mut LlamaContext<'_>, seq: i32, from_pos: i32) -> Result<(), String> {
    let seq = u32::try_from(seq).map_err(|_| format!("bad sequence {seq}"))?;
    let p0 = u32::try_from(from_pos).ok();
    ctx.clear_kv_cache_seq(Some(seq), p0, None).map_err(|e| e.to_string())?;
    Ok(())
}

/// Run one micro-batch of hidden states through the local blocks.
fn run_forward(ctx: &mut LlamaContext<'_>, f: Forward, n_embd: usize, opts: &StageOptions, last: bool) -> Result<Message, String> {
    let rows = f.positions.len();
    if f.width as usize != n_embd || f.hidden.len() != rows * n_embd || f.seqs.len() != rows || f.outputs.len() != rows {
        return Err(format!("micro-batch shape mismatch: {rows} rows of width {}", f.width));
    }
    if rows == 0 || rows > opts.max_rows as usize {
        return Err(format!("micro-batch of {rows} rows; this stage takes 1..={}", opts.max_rows));
    }
    let mut batch = LlamaBatch::new_hidden(rows, n_embd, 1);
    for i in 0..rows {
        let want = if last { f.outputs[i] != 0 } else { true };
        batch
            .add_hidden(&f.hidden[i * n_embd..(i + 1) * n_embd], f.positions[i], &[f.seqs[i]], want)
            .map_err(|e| e.to_string())?;
    }
    ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
    if last {
        let mut data = Vec::new();
        let mut n = 0u32;
        let mut width = 0u32;
        for i in 0..rows {
            if f.outputs[i] != 0 {
                let l = ctx.get_logits_ith(i as i32);
                width = l.len() as u32;
                data.extend_from_slice(l);
                n += 1;
            }
        }
        Ok(Message::Logits { ticket: f.ticket, rows: n, width, data })
    } else {
        let mut hidden = Vec::with_capacity(rows * n_embd);
        for i in 0..rows {
            hidden.extend_from_slice(ctx.embeddings_ith(i as i32).map_err(|e| format!("hidden state {i}: {e}"))?);
        }
        Ok(Message::Forward(Forward { hidden, ..f }))
    }
}
