//! Moving a running sequence between two batch engines of one model.
//! Needs `PRAECISE_TEST_KV_MODEL` (a small dense GGUF) and, optionally,
//! `PRAECISE_TEST_HYBRID_MODEL` (attention plus recurrent layers); ignored
//! by default and fails without them when run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::toploc::StepRecord;
use praecise_runtime::{
    BatchEngine, BatchPrompt, BatchRequest, Error, GenerationConfig, InferenceResult, KvBlob, ModelFingerprint,
    SequenceResume,
};
use tokio::sync::{mpsc, oneshot};

const MAX_TOKENS: u32 = 40;
const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step and \
                      listening to the sea below. One night the lamp would not light, and";
const OTHER: &str = "Write a long list of the names of rivers, towns and mountains you know:";

fn greedy(max_tokens: u32) -> GenerationConfig {
    GenerationConfig {
        temperature: 0.0,
        top_k: Some(1),
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        max_tokens,
        commitment_k: Some(8),
        ..GenerationConfig::default()
    }
}

fn request(
    prompt: &str,
    config: GenerationConfig,
    token_tx: Option<mpsc::Sender<String>>,
) -> (BatchRequest, oneshot::Receiver<praecise_runtime::Result<InferenceResult>>) {
    let (result_tx, result_rx) = oneshot::channel();
    let req = BatchRequest {
        prompt: BatchPrompt::Raw(prompt.to_string()),
        config,
        token_tx,
        reasoning_tx: None,
        result_tx,
        media: Vec::new(),
    };
    (req, result_rx)
}

fn load(backend: &LlamaBackend, path: &str) -> LlamaModel {
    LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load")
}

fn engine(backend: &Arc<LlamaBackend>, path: &str, fp: ModelFingerprint) -> BatchEngine {
    BatchEngine::spawn_with_migration("m".into(), load(backend, path), backend.clone(), 1024, false, None, None, fp)
        .expect("spawn")
}

/// Wait for `n` more streamed chunks.
fn take_chunks(rx: &mut mpsc::Receiver<String>, n: usize) {
    for _ in 0..n {
        rx.blocking_recv().expect("stream ended early");
    }
}

/// Chunks of `rx` received so far without waiting.
fn drain_count(rx: &mut mpsc::Receiver<String>) -> usize {
    let mut n = 0;
    while rx.try_recv().is_ok() {
        n += 1;
    }
    n
}

fn steps(r: &InferenceResult) -> &[StepRecord] {
    &r.commitment.as_ref().expect("commitment recorded").steps
}

fn run(var: &str, path: &str, backend: &Arc<LlamaBackend>) {
    let fp = ModelFingerprint::of_file(path).expect("fingerprint");
    let src = engine(backend, path, fp);
    let dst = engine(backend, path, fp);

    // Reference: the sequence decoded start to finish, alone, on an engine
    // whose caches are as fresh as the source's. Greedy decoding is compared
    // step for step, so both runs decode alone: a batch of a different width
    // may round differently and turn a near-tie the other way.
    let (req, rx) = request(PROMPT, greedy(MAX_TOKENS), None);
    dst.submit(req).expect("submit");
    let reference = rx.blocking_recv().unwrap().expect("reference");
    let ref_steps = steps(&reference).to_vec();
    assert!(ref_steps.len() >= 20, "{var}: reference too short ({} tokens)", ref_steps.len());

    // The sequence to move: exported while it runs, then detached.
    let (a_tx, mut a_rx) = mpsc::channel(256);
    let (req, a_result) = request(PROMPT, greedy(MAX_TOKENS), Some(a_tx));
    let ticket = src.submit_tracked(req).expect("submit tracked");
    take_chunks(&mut a_rx, 4);
    let blob1 = src.export_sequence(ticket).unwrap().blocking_recv().unwrap().expect("first export");
    take_chunks(&mut a_rx, 6);
    let blob2 = src.export_sequence(ticket).unwrap().blocking_recv().unwrap().expect("delta export");
    take_chunks(&mut a_rx, 2);
    let handoff = src.detach_sequence(ticket).unwrap().blocking_recv().unwrap().expect("detach");
    assert!(matches!(a_result.blocking_recv().unwrap(), Err(Error::SequenceHandedOff)), "{var}");
    // The ticket names nothing on the source any more.
    assert!(src.export_sequence(ticket).unwrap().blocking_recv().unwrap().is_err(), "{var}");

    let (k1, k2, k3) =
        (KvBlob::decode(&blob1).unwrap(), KvBlob::decode(&blob2).unwrap(), KvBlob::decode(&handoff.blob).unwrap());
    assert_eq!(k1.base_pos, 0, "{var}");
    assert_eq!(k2.base_pos, k1.end_pos(), "{var}: a delta starts where the last export ended");
    assert_eq!(k3.base_pos, k2.end_pos(), "{var}");
    assert!(!k2.tokens.is_empty() && !k3.tokens.is_empty(), "{var}");
    eprintln!(
        "{var}: blobs {} + {} + {} positions, {} + {} + {} bytes",
        k1.tokens.len(),
        k2.tokens.len(),
        k3.tokens.len(),
        blob1.len(),
        blob2.len(),
        handoff.blob.len()
    );

    // The chain holds the prompt and every generated token but the last,
    // which is the handoff's next token.
    let g = handoff.generated_tokens as usize;
    let chain: Vec<i32> = [&k1.tokens, &k2.tokens, &k3.tokens].into_iter().flatten().copied().collect();
    let prompt_len = reference.input_tokens as usize;
    assert_eq!(chain.len(), prompt_len + g - 1, "{var}");
    let generated: Vec<u32> = chain[prompt_len..].iter().map(|&t| t as u32).collect();
    assert_eq!(generated, ref_steps[..g - 1].iter().map(|s| s.token_id).collect::<Vec<_>>(), "{var}");
    assert_eq!(handoff.next_token as u32, ref_steps[g - 1].token_id, "{var}");
    assert!(g + 5 < ref_steps.len(), "{var}: handed off too late ({g} of {})", ref_steps.len());

    // Resume on the second engine.
    let (result_tx, resumed_rx) = oneshot::channel();
    let started = Instant::now();
    dst.resume(SequenceResume {
        blobs: vec![blob1, blob2, handoff.blob.clone()],
        next_token: handoff.next_token,
        reasoning: handoff.reasoning,
        config: greedy(MAX_TOKENS - handoff.generated_tokens),
        token_tx: None,
        reasoning_tx: None,
        result_tx,
    })
    .expect("resume");
    let resumed = resumed_rx.blocking_recv().unwrap().expect("resumed");
    let first_blob = handoff.blob.clone();
    assert_eq!(resumed.input_tokens as usize, prompt_len + g - 1, "{var}: nothing was prefilled again");
    assert_eq!(resumed.cached_tokens, resumed.input_tokens, "{var}");

    // The first logits the destination computes are those the source would
    // have computed next, and the continuation is the reference's.
    let got = steps(&resumed);
    let want = &ref_steps[g..];
    let max_diff = got[0]
        .top_k
        .iter()
        .zip(&want[0].top_k)
        .map(|(a, b)| {
            assert_eq!(a.token_id, b.token_id, "{var}: top-k order differs");
            (a.logit - b.logit).abs()
        })
        .fold(0f32, f32::max);
    eprintln!(
        "{var}: handed off after {g} tokens; next-token top-{} logits max |diff| {max_diff:e} (bitwise identical: {}); \
         resumed in {:?}",
        got[0].top_k.len(),
        got[0].top_k == want[0].top_k,
        started.elapsed()
    );
    assert!(max_diff <= 1e-3, "{var}: next-token logits diverged by {max_diff}");
    assert_eq!(
        got.iter().map(|s| s.token_id).collect::<Vec<_>>(),
        want.iter().map(|s| s.token_id).collect::<Vec<_>>(),
        "{var}: the continuation differs from the uninterrupted run"
    );

    // Exporting one slot does not hold up the others: a request beside a
    // tracked one keeps streaming across its exports and finishes, and the
    // moved sequence resumes beside a request on a busy destination.
    let (a_tx, mut a_rx) = mpsc::channel(256);
    let (req, _a_result) = request(PROMPT, greedy(MAX_TOKENS), Some(a_tx));
    let ticket = src.submit_tracked(req).expect("submit tracked");
    let (b_tx, mut b_rx) = mpsc::channel(256);
    let (req, b_result) = request(OTHER, greedy(200), Some(b_tx));
    src.submit(req).expect("submit other");
    take_chunks(&mut a_rx, 4);
    take_chunks(&mut b_rx, 1);
    let mut chain = vec![src.export_sequence(ticket).unwrap().blocking_recv().unwrap().expect("export")];
    let mut b_streamed = Vec::new();
    for _ in 0..3 {
        drain_count(&mut b_rx);
        take_chunks(&mut a_rx, 3);
        b_streamed.push(drain_count(&mut b_rx));
        chain.push(src.export_sequence(ticket).unwrap().blocking_recv().unwrap().expect("export"));
    }
    assert!(b_streamed.iter().all(|&n| n > 0), "{var}: the other slot stalled across exports: {b_streamed:?}");
    let handoff = src.detach_sequence(ticket).unwrap().blocking_recv().unwrap().expect("detach");
    chain.push(handoff.blob);
    let (c_tx, _c_rx) = mpsc::channel(256);
    let (req, c_result) = request(OTHER, greedy(60), Some(c_tx));
    dst.submit(req).expect("submit on destination");
    let (result_tx, resumed_rx) = oneshot::channel();
    dst.resume(SequenceResume {
        blobs: chain,
        next_token: handoff.next_token,
        reasoning: handoff.reasoning,
        config: greedy(MAX_TOKENS - handoff.generated_tokens),
        token_tx: None,
        reasoning_tx: None,
        result_tx,
    })
    .expect("resume");
    let r = resumed_rx.blocking_recv().unwrap().expect("resumed beside another request");
    assert!(r.output_tokens > 0, "{var}");
    let b = b_result.blocking_recv().unwrap().expect("the other request finishes");
    assert!(b.output_tokens > 0, "{var}");
    c_result.blocking_recv().unwrap().expect("the destination's other request finishes");
    eprintln!("{var}: the neighbouring slot streamed {b_streamed:?} chunks between exports");

    // A chain from other weights is refused on the resume's result.
    let mut forged = fp.0;
    forged[0] ^= 0xff;
    let other_engine = engine(backend, path, ModelFingerprint::from_digest(forged));
    let (result_tx, refused) = oneshot::channel();
    other_engine
        .resume(SequenceResume {
            blobs: vec![first_blob],
            next_token: handoff.next_token,
            reasoning: handoff.reasoning,
            config: greedy(4),
            token_tx: None,
            reasoning_tx: None,
            result_tx,
        })
        .expect("resume");
    let refused = refused.blocking_recv().unwrap();
    assert!(matches!(refused, Err(Error::ModelMismatch { .. })), "{var}: {:?}", refused.err());

    other_engine.shutdown();
    src.shutdown();
    dst.shutdown();
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL"]
fn a_running_sequence_moves_to_another_engine_and_continues_where_it_stopped() {
    // Few slots keep the test's contexts small; set before any engine reads it.
    // SAFETY: set before this test binary starts any other thread.
    unsafe { std::env::set_var("BATCH_MAX_SLOTS", "4") };
    let models: Vec<(&str, String)> = ["PRAECISE_TEST_KV_MODEL", "PRAECISE_TEST_HYBRID_MODEL"]
        .into_iter()
        .filter_map(|var| std::env::var(var).ok().map(|p| (var, p)))
        .collect();
    assert!(!models.is_empty(), "set PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    for (var, path) in models {
        run(var, &path, &backend);
    }
}
