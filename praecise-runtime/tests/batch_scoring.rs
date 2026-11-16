//! Verifying a commitment on the batch engine: a fixed continuation scored in
//! one prefill. Needs `PRAECISE_TEST_KV_MODEL` (a small GGUF) and, optionally,
//! `PRAECISE_TEST_HYBRID_MODEL`; ignored by default and fails without them when
//! run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::toploc::{Tolerance, verify_commitment, verify_commitment_with};
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult};
use tokio::sync::oneshot;

const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step and \
                      listening to the sea below. One night the lamp would not light, and";

fn generate(engine: &BatchEngine, prompt: &str, max_tokens: u32) -> oneshot::Receiver<praecise_runtime::Result<InferenceResult>> {
    let (result_tx, rx) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(prompt.into()),
            config: GenerationConfig {
                temperature: 0.0,
                top_k: Some(1),
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                max_tokens,
                commitment_k: Some(8),
                ..GenerationConfig::default()
            },
            token_tx: None,
            result_tx,
            media: Vec::new(),
        })
        .expect("submit");
    rx
}

fn run(var: &str, path: &str, backend: &Arc<LlamaBackend>) {
    let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    let engine = BatchEngine::spawn("m".into(), model, backend.clone(), 1024, false).expect("spawn");

    let served = generate(&engine, PROMPT, 24).blocking_recv().unwrap().expect("generate");
    let commitment = served.commitment.expect("the batch engine records a commitment");
    let tokens: Vec<i32> = commitment.steps.iter().map(|s| s.token_id as i32).collect();
    assert!(tokens.len() >= 8, "{var}: too short ({} tokens)", tokens.len());

    // Scored beside a request that is generating, as a verifier would be.
    let busy = generate(&engine, "Write a long list of the names of rivers, towns and mountains you know:", 120);
    let rows = engine
        .score_continuation(BatchPrompt::Raw(PROMPT.into()), &tokens, 8)
        .expect("score")
        .blocking_recv()
        .unwrap()
        .expect("scored");
    assert_eq!(rows.len(), tokens.len(), "{var}: one row per continuation token");
    let outcome = verify_commitment(&commitment, &rows);
    let max_diff = commitment
        .steps
        .iter()
        .zip(&rows)
        .flat_map(|(s, r)| s.top_k.iter().zip(r).map(|(a, b)| (a.logit - b.logit).abs()))
        .fold(0f32, f32::max);
    eprintln!(
        "{var}: {} of {} steps pass, top-k logits max |diff| {max_diff:e}",
        outcome.steps_passed, outcome.steps_total
    );
    assert!(outcome.pass, "{var}: the engine's own commitment does not verify: {outcome:?}");
    // Decoded one token at a time and prefilled at once, the logits round
    // apart (measured up to 0.5 on CPU), so a near tie may swap; each committed
    // token still lies in its scored row's top-k.
    let supported = commitment.steps.iter().zip(&rows).filter(|(s, r)| r.iter().any(|e| e.token_id == s.token_id)).count();
    assert_eq!(supported, rows.len(), "{var}: a committed token is outside its scored top-k");

    // Another continuation is not this commitment's.
    let wrong: Vec<i32> = tokens.iter().map(|t| 100 + (t + 1) % 900).collect();
    let rows = engine
        .score_continuation(BatchPrompt::Raw(PROMPT.into()), &wrong, 8)
        .unwrap()
        .blocking_recv()
        .unwrap()
        .expect("scored");
    assert!(!verify_commitment(&commitment, &rows).pass, "{var}: a wrong continuation verified");
    // Under a tolerance no row can meet, even the right one fails.
    let strict = Tolerance { min_index_overlap: 1.1, ..Tolerance::default() };
    let rows = engine
        .score_continuation(BatchPrompt::Raw(PROMPT.into()), &tokens, 8)
        .unwrap()
        .blocking_recv()
        .unwrap()
        .expect("scored");
    assert!(!verify_commitment_with(&commitment, &rows, &strict).pass, "{var}");

    assert!(engine.score_continuation(BatchPrompt::Raw(PROMPT.into()), &[], 8).is_err(), "{var}: empty continuation");
    busy.blocking_recv().unwrap().expect("the neighbouring request finishes");
    engine.shutdown();
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL"]
fn a_batch_commitment_verifies_against_a_continuation_scored_on_the_batch_engine() {
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
