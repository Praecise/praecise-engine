//! Serving a diffusion language model through the batch engine.
//!
//! [`BatchEngine`](crate::BatchEngine) spawns this scheduler for a model that
//! decodes by diffusion, so a host submits the same [`BatchRequest`] and gets
//! the same [`InferenceResult`] whichever way the model writes its answer.
//!
//! **Batching.** Every running sequence contributes one forward pass's worth
//! of positions to each step (see [`MaskedCanvas::next_chunk`]), and the step
//! decodes them together, each under its own sequence id, as long as they fit
//! in one physical batch: a model that attends both ways must see a whole
//! chunk in one pass. A request whose prompt and answer cannot fit is refused
//! at admission with the sizes, never cut short silently.
//!
//! **Streaming.** A position is streamed once it can no longer change: when
//! it and every position before it are revealed and, with token editing, its
//! block is closed. A masked model with blocks therefore streams block by
//! block, one without blocks as its leftmost positions fill in, and a
//! uniform-noise model one canvas at a time.
//!
//! **Metering.** `input_tokens` is the prompt and `output_tokens` the answer
//! delivered, up to and without an end token: positions left after the end,
//! and the masks never revealed, are not output. What the model actually ran
//! is reported apart, in `forward_positions` and `denoise_steps`, because a
//! diffusion model reads its canvas once per step, which costs far more than
//! its output token count says.

use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use tracing::{info, warn};

use crate::batching::{drain_streams, render_with_config, BatchRequest, Draining, Queued};
use crate::denoise::{
    noise_row, CanvasEnd, Chunk, MaskedCanvas, MaskedPlan, NoiseCanvas, NoisePlan, NoiseRow, Pace, Rng, RowSampling,
    Unmask, DEFAULT_TOP_K,
};
use crate::error::{Error, Result};
use crate::result::{InferenceResult, StopReason};
use crate::stream::StopStream;

/// How a model is decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum Plan {
    Masked(MaskedPlan),
    Noise(NoisePlan),
}

/// Positions one step may decode across all sequences, overridable with
/// `DENOISE_WINDOW`. A sequence's prompt and answer must fit in it when the
/// model reads its whole canvas each step, so it bounds the longest request;
/// the attention it costs grows with its square.
const WINDOW_DEFAULT: u32 = 2048;

const IDLE_POLL: Duration = Duration::from_millis(50);

fn window(context_length: u32) -> u32 {
    std::env::var("DENOISE_WINDOW")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v >= 64)
        .unwrap_or(WINDOW_DEFAULT)
        .min(context_length.max(64))
}

fn meta<T: std::str::FromStr>(model: &LlamaModel, key: &str) -> Option<T> {
    model.meta_val_str(key).ok().and_then(|v| v.trim().parse().ok())
}

/// The decoding plan for a model, from its architecture and whatever the
/// GGUF declares about it.
///
/// # Errors
/// For an architecture with no known plan, or a masked model whose
/// vocabulary declares no mask token.
pub fn plan_for(model: &LlamaModel) -> Result<Plan> {
    let arch = model.meta_val_str("general.architecture").unwrap_or_default();
    // A fixed-length canvas marks a uniform-noise model; its sampler's
    // settings travel in the GGUF, with the reference values as defaults.
    if let Some(canvas) = meta::<u32>(model, "diffusion.canvas_length").filter(|c| *c > 0) {
        return Ok(Plan::Noise(NoisePlan {
            canvas,
            max_steps: meta(model, "diffusion.eb_max_steps").unwrap_or(48).max(1),
            t_min: meta(model, "diffusion.eb_t_min").unwrap_or(0.4),
            t_max: meta(model, "diffusion.eb_t_max").unwrap_or(0.8),
            entropy_bound: meta(model, "diffusion.eb_entropy_bound").unwrap_or(0.1),
            stability: meta(model, "diffusion.eb_stability_threshold").unwrap_or(1),
            confidence: meta(model, "diffusion.eb_confidence_threshold").unwrap_or(0.005),
        }));
    }
    let mask = model.token_mask().ok_or_else(|| {
        Error::Other(format!("{arch} decodes by diffusion but its vocabulary declares no mask token"))
    })?;
    let mut plan = match arch.as_str() {
        // Whole-answer canvases revealed lowest-entropy first on a noise schedule.
        "dream" | "rnd1" => MaskedPlan {
            mask: mask.0,
            block: 0,
            steps: 0,
            unmask: Unmask::Entropy,
            pace: Pace::Timestep { eps: 1e-3 },
            threshold: None,
            edit_threshold: None,
            // Both are trained from causal checkpoints and keep their
            // next-token alignment; a GGUF that says otherwise wins below.
            shift_logits: true,
            cached_blocks: false,
        },
        // Blocks of 32, one position per step, surest first.
        "llada" | "llada-moe" => MaskedPlan {
            mask: mask.0,
            block: 32,
            steps: 0,
            unmask: Unmask::Confidence,
            pace: Pace::Even,
            threshold: None,
            edit_threshold: None,
            shift_logits: false,
            cached_blocks: false,
        },
        // Block-causal: blocks of 32 in the cache, every position above 0.95
        // revealed at once.
        "llada2" => MaskedPlan {
            mask: mask.0,
            block: 32,
            steps: 0,
            unmask: Unmask::Confidence,
            pace: Pace::Even,
            threshold: Some(0.95),
            edit_threshold: None,
            shift_logits: false,
            cached_blocks: true,
        },
        other => {
            return Err(Error::Other(format!(
                "{other} decodes by diffusion, but no decoding plan is known for that architecture"
            )));
        }
    };
    // What the GGUF declares outranks the architecture's defaults.
    if let Ok(v) = model.meta_val_str("diffusion.shift_logits") {
        plan.shift_logits = v.trim() == "true";
    }
    if let Some(b) = meta::<u32>(model, "diffusion.block_length") {
        plan.block = b;
    }
    if let Some(s) = meta::<u32>(model, "diffusion.steps") {
        plan.steps = s;
    }
    if let Some(t) = meta::<f32>(model, "diffusion.threshold") {
        plan.threshold = (t > 0.0).then_some(t);
    }
    if let Some(t) = meta::<f32>(model, "diffusion.edit_threshold") {
        plan.edit_threshold = (t > 0.0).then_some(t);
    }
    plan.check().map_err(|e| Error::Other(format!("{arch}: {e}")))?;
    Ok(Plan::Masked(plan))
}

/// A sequence's canvas, of either family.
enum Decode {
    Masked(MaskedCanvas),
    Noise {
        canvas: NoiseCanvas,
        /// The previous step's canvas logits, which condition the next.
        conditioning: Vec<f32>,
    },
}

impl Decode {
    fn next_chunk(&self) -> Option<Chunk> {
        match self {
            Self::Masked(c) => c.next_chunk(),
            Self::Noise { canvas, .. } => canvas.next_input().map(|tokens| {
                let n = tokens.len();
                let rows = (n - canvas.canvas_len()..n).collect();
                Chunk { start: 0, tokens, rows, clear_from: None }
            }),
        }
    }

    fn settled(&self) -> &[i32] {
        match self {
            Self::Masked(c) => c.settled(),
            Self::Noise { canvas, .. } => canvas.settled(),
        }
    }

    fn is_done(&self) -> bool {
        match self {
            Self::Masked(c) => c.is_done(),
            Self::Noise { canvas, .. } => canvas.is_done(),
        }
    }

    fn finish(&mut self) {
        match self {
            Self::Masked(c) => c.finish(),
            Self::Noise { canvas, .. } => canvas.finish(),
        }
    }

    fn forward_positions(&self) -> u64 {
        match self {
            Self::Masked(c) => c.forward_positions(),
            Self::Noise { canvas, .. } => canvas.forward_positions(),
        }
    }

    fn steps_taken(&self) -> u32 {
        match self {
            Self::Masked(c) => c.steps_taken(),
            Self::Noise { canvas, .. } => canvas.steps_taken(),
        }
    }
}

/// Read one uniform-noise step's canvas rows: each position's prediction,
/// and its logits kept to condition the next step. Positions are read in
/// parallel; each is a pass over the whole vocabulary.
fn noise_rows(rows: &[&[f32]], temp_inv: f32, u: &[f32], conditioning: &mut Vec<f32>) -> Vec<NoiseRow> {
    let n_vocab = rows.first().map_or(0, |r| r.len());
    conditioning.resize(rows.len() * n_vocab, 0.0);
    for (i, r) in rows.iter().enumerate() {
        conditioning[i * n_vocab..(i + 1) * n_vocab].copy_from_slice(r);
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 32);
    let per = rows.len().div_ceil(threads).max(1);
    let mut out = vec![NoiseRow { argmax: 0, sampled: 0, entropy: 0.0 }; rows.len()];
    std::thread::scope(|scope| {
        for (k, slot) in out.chunks_mut(per).enumerate() {
            let rows = &rows[k * per..];
            let u = &u[k * per..];
            scope.spawn(move || {
                for (j, o) in slot.iter_mut().enumerate() {
                    *o = noise_row(rows[j], temp_inv, u[j]);
                }
            });
        }
    });
    out
}

/// One request being decoded.
struct Running {
    seq_id: i32,
    canvas: Decode,
    rng: Rng,
    sampling: RowSampling,
    /// Settled answer tokens already streamed.
    emitted: usize,
    output_tokens: u32,
    input_tokens: u32,
    /// The answer length the canvas was given; reaching it is `Length`.
    answer_len: usize,
    hit_end: bool,
    decoder: encoding_rs::Decoder,
    stream: StopStream,
    token_tx: Option<tokio::sync::mpsc::Sender<String>>,
    result_tx: Option<tokio::sync::oneshot::Sender<Result<InferenceResult>>>,
    started: Instant,
}

impl Running {
    /// Stream what has settled since the last call. False when the answer
    /// is complete: an end token, a stop sequence, or a receiver gone.
    fn stream_settled(&mut self, model: &LlamaModel) -> bool {
        let settled = self.canvas.settled();
        while self.emitted < settled.len() {
            let token = LlamaToken(settled[self.emitted]);
            self.emitted += 1;
            if model.is_eog_token(token) {
                self.hit_end = true;
                return false;
            }
            self.output_tokens += 1;
            match model.token_to_piece(token, &mut self.decoder, true, None) {
                Ok(piece) => {
                    if !self.stream.push(&piece, self.token_tx.as_ref()) || self.stream.hit_stop() {
                        return false;
                    }
                }
                Err(e) => warn!("token decode failed: {e}"),
            }
        }
        !self.canvas.is_done()
    }

    fn finish(mut self) -> Option<Draining> {
        let generation_time_ms = self.started.elapsed().as_millis() as u64;
        let tokens_per_second = if generation_time_ms > 0 {
            f64::from(self.output_tokens) / (generation_time_ms as f64 / 1000.0)
        } else {
            0.0
        };
        let stop_reason = if self.stream.hit_stop() {
            StopReason::StopSequence
        } else if self.hit_end {
            StopReason::Eos
        } else if self.emitted >= self.answer_len {
            StopReason::Length
        } else {
            StopReason::Eos
        };
        let stalled = self.stream.stalled();
        let token_tx = self.token_tx.take();
        let reply = self.stream.close(token_tx.as_ref());
        let result_tx = self.result_tx.take()?;
        if stalled {
            let _ = result_tx.send(Err(Error::Inference(
                "the streaming client stopped reading; the request was ended so it no longer holds back other \
                 requests"
                    .into(),
            )));
            return None;
        }
        let reply = match reply {
            Ok(reply) => reply,
            Err(e) => {
                let _ = result_tx.send(Err(e));
                return None;
            }
        };
        let result = InferenceResult {
            text: reply.text,
            thinking: reply.thinking,
            tool_calls: reply.tool_calls,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            generation_time_ms,
            tokens_per_second,
            stop_reason,
            commitment: None,
            cached_tokens: 0,
            forward_positions: Some(self.canvas.forward_positions()),
            denoise_steps: Some(self.canvas.steps_taken()),
        };
        if self.stream.delivered() {
            let _ = result_tx.send(Ok(result));
            return None;
        }
        Some(Draining { tx: token_tx, stream: self.stream, result_tx, result, since: Instant::now() })
    }
}

/// Render, tokenize and size one request, or refuse it with the reason.
#[allow(clippy::too_many_arguments)]
fn admit(
    model: &LlamaModel,
    plan: &Plan,
    req: BatchRequest,
    enable_thinking: bool,
    seq_id: i32,
    window: usize,
    n_ctx: usize,
) -> std::result::Result<Running, (tokio::sync::oneshot::Sender<Result<InferenceResult>>, Error)> {
    let BatchRequest { prompt, mut config, token_tx, result_tx, media } = req;
    if !media.is_empty() {
        return Err((result_tx, Error::Other("this diffusion model reads text only; the request carries media".into())));
    }
    let rendered = match render_with_config(model, &prompt, &config, enable_thinking) {
        Ok((p, format)) => {
            if config.reply_format.is_none() {
                config.reply_format = format;
            }
            p
        }
        Err(e) => return Err((result_tx, e)),
    };
    let tokens = match model.str_to_token(&rendered, AddBos::Always) {
        Ok(t) => t,
        Err(e) => return Err((result_tx, Error::Other(format!("tokenization failed: {e}")))),
    };
    let prompt_len = tokens.len();
    // A block-causal model holds its canvas in the cache; any other reads the
    // whole of it in one pass. A uniform-noise model reads its prompt, every
    // canvas it finished and the open one.
    let (room, least) = match plan {
        Plan::Masked(p) if p.cached_blocks => (n_ctx, prompt_len + 1),
        Plan::Masked(_) => (window, prompt_len + 1),
        Plan::Noise(p) => (window, prompt_len + p.canvas as usize),
    };
    if least > room {
        return Err((
            result_tx,
            Error::Other(format!(
                "a prompt of {prompt_len} tokens leaves no room to answer: this diffusion model reads the prompt and \
                 its answer together, at most {room} positions"
            )),
        ));
    }
    let prompt_ids: Vec<i32> = tokens.iter().map(|t| t.0).collect();
    let mut rng = Rng::new(config.seed);
    let (canvas, answer_len, forbid) = match plan {
        Plan::Masked(p) => {
            let answer_len = (config.max_tokens.max(1) as usize).min(room - prompt_len);
            (Decode::Masked(MaskedCanvas::new(p.clone(), &prompt_ids, answer_len)), answer_len, Some(p.mask))
        }
        Plan::Noise(p) => {
            // Each finished canvas joins the prefix of the next, so the answer
            // is bounded by the canvases that fit after the prompt.
            let c = p.canvas as usize;
            let fits = (room - prompt_len) / c * c;
            let answer_len = (config.max_tokens.max(1) as usize).min(fits);
            let canvas = NoiseCanvas::new(*p, model.n_vocab() as u32, &prompt_ids, answer_len, &mut rng);
            (Decode::Noise { canvas, conditioning: Vec::new() }, answer_len, None)
        }
    };
    Ok(Running {
        seq_id,
        canvas,
        rng,
        sampling: RowSampling {
            temperature: config.temperature as f32,
            top_k: config.top_k.unwrap_or(DEFAULT_TOP_K),
            top_p: config.top_p as f32,
            forbid,
        },
        emitted: 0,
        output_tokens: 0,
        input_tokens: prompt_len as u32,
        answer_len,
        hit_end: false,
        decoder: encoding_rs::UTF_8.new_decoder(),
        stream: StopStream::new(config.stop).reading(config.reply_format.as_ref()).nonblocking().with_reasoning(config.reasoning_tx),
        token_tx,
        result_tx: Some(result_tx),
        started: Instant::now(),
    })
}

/// The scheduler thread's loop: admit, step every running sequence together,
/// stream what settled, finish what completed.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn scheduler_loop(
    model_id: &str,
    model: LlamaModel,
    backend: &LlamaBackend,
    context_length: u32,
    enable_thinking: bool,
    plan: Plan,
    rx: &Receiver<Queued>,
    shutdown_rx: &Receiver<()>,
) -> Result<()> {
    use std::num::NonZeroU32;

    let cached_blocks = matches!(&plan, Plan::Masked(p) if p.cached_blocks);
    // The conditioning state belongs to the model: one uniform-noise canvas
    // per pass.
    let one_per_pass = matches!(plan, Plan::Noise(_));
    let slots = crate::batching::max_slots();
    let window = window(context_length);
    let n_ctx = NonZeroU32::new(context_length.max(window)).unwrap_or(NonZeroU32::MIN);
    let mut params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_seq_max(slots as u32)
        .with_kv_unified(true)
        .with_n_batch(window)
        .with_n_ubatch(window);
    // A uniform-noise pass needs the canvas's logits, not the prompt's: at a
    // vocabulary of a quarter of a million, the difference is gigabytes.
    if let Plan::Noise(p) = &plan {
        params = params.with_n_outputs_max(p.canvas);
    }
    let mut ctx = model
        .new_context(backend, params)
        .map_err(|e| Error::Other(format!("denoising context init failed: {e}")))?;
    // Positions attend both ways; a block-causal model's cache still holds
    // only the blocks before the one being decoded.
    ctx.set_causal_attn(false);
    info!("denoising engine for {model_id}: window {window}, {slots} sequences, {plan:?}");

    let mut running: Vec<Running> = Vec::new();
    let mut draining: Vec<Draining> = Vec::new();
    let mut batch = LlamaBatch::new(window as usize, 1);
    // Which sequence goes first in the next step, so a long canvas cannot
    // keep a later one out of every step.
    let mut turn = 0usize;

    loop {
        match shutdown_rx.try_recv() {
            Ok(()) | Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }
        // Admit while there is a free sequence id.
        loop {
            if running.len() >= slots {
                break;
            }
            let queued = if running.is_empty() && draining.is_empty() {
                match rx.recv_timeout(IDLE_POLL) {
                    Ok(q) => q,
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return Ok(()),
                }
            } else {
                match rx.try_recv() {
                    Ok(q) => q,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            };
            let used: Vec<i32> = running.iter().map(|r| r.seq_id).collect();
            let seq_id = (0..slots as i32).find(|i| !used.contains(i)).unwrap_or(0);
            match admit(&model, &plan, queued.req, enable_thinking, seq_id, window as usize, n_ctx.get() as usize) {
                Ok(r) => running.push(r),
                Err((tx, e)) => {
                    let _ = tx.send(Err(e));
                }
            }
        }
        drain_streams(&mut draining);
        if running.is_empty() {
            if !draining.is_empty() {
                std::thread::sleep(Duration::from_millis(2));
            }
            continue;
        }

        // Gather this step's chunks.
        batch.clear();
        let mut members: Vec<(usize, usize)> = Vec::new(); // (running index, first batch index)
        let mut used = 0usize;
        let n = running.len();
        for k in 0..n {
            let i = (turn + k) % n;
            let Some(chunk) = running[i].canvas.next_chunk() else { continue };
            if used + chunk.tokens.len() > window as usize || (one_per_pass && !members.is_empty()) {
                continue;
            }
            if let Some(p0) = chunk.clear_from {
                let _ = ctx.clear_kv_cache_seq(Some(running[i].seq_id as u32), Some(p0 as u32), None);
            }
            let first = used;
            let mut wanted = vec![false; chunk.tokens.len()];
            for &r in &chunk.rows {
                wanted[r] = true;
            }
            for (j, &t) in chunk.tokens.iter().enumerate() {
                batch
                    .add(LlamaToken(t), (chunk.start + j) as i32, &[running[i].seq_id], wanted[j])
                    .map_err(|e| Error::Other(format!("denoising batch: {e}")))?;
            }
            used += chunk.tokens.len();
            members.push((i, first));
            if let Decode::Noise { canvas, conditioning } = &mut running[i].canvas {
                // A canvas's first step reads zeros, gated off.
                if conditioning.is_empty() {
                    conditioning.resize(canvas.canvas_len() * model.n_vocab() as usize, 0.0);
                }
                let (on, temp_inv) = canvas.conditioning();
                model.diffusion_set_self_conditioning(Some(conditioning.as_slice()), if on { 1.0 } else { 0.0 }, temp_inv);
            }
        }
        turn = turn.wrapping_add(1);
        if members.is_empty() {
            // Nothing fits beside nothing: admission sized every request to
            // the window, so this is a canvas that grew past it.
            for r in running.drain(..) {
                if let Some(tx) = r.result_tx {
                    let _ = tx.send(Err(Error::Other("a denoising step outgrew the window".into())));
                }
            }
            continue;
        }

        if let Err(e) = ctx.decode(&mut batch) {
            warn!("denoising step failed on {model_id}: {e}");
            for (i, _) in &members {
                if let Some(tx) = running[*i].result_tx.take() {
                    let _ = tx.send(Err(Error::Inference(format!("denoising step failed: {e}"))));
                }
            }
            running.retain(|r| r.result_tx.is_some());
            continue;
        }

        let mut finished: Vec<usize> = Vec::new();
        for &(i, first) in &members {
            let r = &mut running[i];
            let chunk = r.canvas.next_chunk().unwrap_or_else(|| unreachable!("a member had a chunk"));
            let rows: Vec<&[f32]> = chunk.rows.iter().map(|&j| ctx.get_logits_ith((first + j) as i32)).collect();
            match &mut r.canvas {
                Decode::Masked(c) => c.apply(&rows, &r.sampling, &mut r.rng),
                Decode::Noise { canvas, conditioning } => {
                    let temp_inv = canvas.temp_inv();
                    let (u, renoise) = canvas.draws(&mut r.rng);
                    let picked = noise_rows(&rows, temp_inv, &u, conditioning);
                    if canvas.apply(&picked, &renoise, |t| model.is_eog_token(LlamaToken(t)), &mut r.rng)
                        == CanvasEnd::Closed
                    {
                        // A new canvas starts unconditioned.
                        conditioning.clear();
                    }
                }
            }
            if !r.stream_settled(&model) {
                r.canvas.finish();
                finished.push(i);
            } else if !r.stream.flush(r.token_tx.as_ref()) {
                // The client is gone.
                r.canvas.finish();
                finished.push(i);
            }
        }
        finished.sort_unstable();
        for i in finished.into_iter().rev() {
            let r = running.swap_remove(i);
            if cached_blocks {
                let _ = ctx.clear_kv_cache_seq(Some(r.seq_id as u32), None, None);
            }
            if let Some(d) = r.finish() {
                draining.push(d);
            }
        }
    }
    for r in running {
        if let Some(tx) = r.result_tx {
            let _ = tx.send(Err(Error::Other(format!("the engine for {model_id} shut down"))));
        }
    }
    Ok(())
}
