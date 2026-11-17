//! A batched request asked to yield its slot returns what it has generated.
//! Needs `PRAECISE_TEST_KV_MODEL` (a small GGUF) and, optionally,
//! `PRAECISE_TEST_HYBRID_MODEL`; ignored by default and fails without them when
//! run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::result::StopReason;
use praecise_runtime::toploc::{compare_step, verify_commitment};
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult};
use tokio::sync::{mpsc, oneshot};

const PROMPT: &str = "Write a long list of the names of rivers, towns and mountains you know:";

fn request(
    max_tokens: u32,
    token_tx: Option<mpsc::Sender<String>>,
) -> (BatchRequest, oneshot::Receiver<praecise_runtime::Result<InferenceResult>>) {
    let (result_tx, rx) = oneshot::channel();
    let config = GenerationConfig {
        temperature: 0.0,
        top_k: Some(1),
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        max_tokens,
        commitment_k: Some(8),
        ..GenerationConfig::default()
    };
    (BatchRequest { prompt: BatchPrompt::Raw(PROMPT.into()), config, token_tx, result_tx, media: Vec::new() }, rx)
}

fn run(var: &str, path: &str, backend: &Arc<LlamaBackend>) {
    let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    let engine = BatchEngine::spawn("m".into(), model, backend.clone(), 1024, false).expect("spawn");

    // Asked to yield partway, the request ends at a token boundary with its
    // partial output. Far enough in that one step rounding apart between decode
    // and prefill (as in batch_scoring) stays within the passing fraction: under
    // ten steps, a single such step fails the commitment.
    let (tx, mut chunks) = mpsc::channel(256);
    let (req, rx) = request(300, Some(tx));
    let ticket = engine.submit_tracked(req).expect("submit tracked");
    for _ in 0..32 {
        chunks.blocking_recv().expect("stream ended early");
    }
    engine.yield_sequence(ticket).expect("yield");
    while chunks.blocking_recv().is_some() {}
    let r = rx.blocking_recv().unwrap().expect("a yielded request returns its result");
    assert_eq!(r.stop_reason, StopReason::Preempted, "{var}");
    assert!(r.output_tokens >= 32 && r.output_tokens < 300, "{var}: {} tokens", r.output_tokens);
    assert!(!r.text.is_empty(), "{var}");

    // Its commitment covers exactly the tokens it returned and verifies step
    // by step against the same continuation scored on the engine.
    let commitment = r.commitment.expect("commitment");
    assert_eq!(commitment.steps.len(), r.output_tokens as usize, "{var}");
    let tokens: Vec<i32> = commitment.steps.iter().map(|s| s.token_id as i32).collect();
    let rows = engine
        .score_continuation(BatchPrompt::Raw(PROMPT.into()), &tokens, 8)
        .unwrap()
        .blocking_recv()
        .unwrap()
        .expect("scored");

    // The same steps an uninterrupted run starts with: tokens, top-k ids and
    // logits, bit for bit. The run is on a fresh engine, so that it prefills
    // the prompt as the yielded request did; this engine would reuse the
    // prompt it has cached and compute the first logits from another batch.
    let fresh = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    let fresh = BatchEngine::spawn("m".into(), fresh, backend.clone(), 1024, false).expect("spawn");
    let (req, rx) = request(r.output_tokens, None);
    fresh.submit(req).expect("submit");
    let full = rx.blocking_recv().unwrap().expect("reference");
    fresh.shutdown();
    let reference = full.commitment.expect("commitment");
    let want: Vec<u32> = reference.steps.iter().map(|s| s.token_id).collect();
    assert_eq!(tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(), want, "{var}: the partial output is not the run's start");
    for (i, (a, b)) in commitment.steps.iter().zip(&reference.steps).enumerate() {
        let a: Vec<(u32, u32)> = a.top_k.iter().map(|e| (e.token_id, e.logit.to_bits())).collect();
        let b: Vec<(u32, u32)> = b.top_k.iter().map(|e| (e.token_id, e.logit.to_bits())).collect();
        assert_eq!(a, b, "{var}: step {i} differs from the uninterrupted run");
    }

    let outcome = verify_commitment(&commitment, &rows);
    eprintln!("{var}: yielded after {} tokens; {} of {} steps verify", r.output_tokens, outcome.steps_passed, outcome.steps_total);
    for &i in &outcome.failing_steps {
        let c = compare_step(&commitment.steps[i], &rows[i]);
        eprintln!(
            "{var}: step {i}: token supported {}, index overlap {}, mean logit delta {}",
            c.token_supported, c.index_overlap, c.mean_logit_delta
        );
    }
    assert!(outcome.pass, "{var}: {outcome:?}");

    // A ticket that names nothing any more is ignored.
    engine.yield_sequence(ticket).expect("a stale yield is not an error");
    engine.shutdown();
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL"]
fn a_yielded_request_returns_its_partial_output_with_a_commitment_that_verifies() {
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
