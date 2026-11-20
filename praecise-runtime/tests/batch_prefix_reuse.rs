//! A request that extends one the engine already served must answer as a cold
//! engine does: the prompt cache is an optimisation, not a change of output.
//! Needs `PRAECISE_TEST_KV_MODEL` (a small GGUF) and, optionally,
//! `PRAECISE_TEST_HYBRID_MODEL` (attention and recurrent layers); ignored by
//! default and fails without them when run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult};
use tokio::sync::oneshot;

const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step and \
                      listening to the sea below. One night the lamp would not light, and";

fn generate(engine: &BatchEngine, prompt: &str, max_tokens: u32) -> InferenceResult {
    let (result_tx, rx): (_, oneshot::Receiver<praecise_runtime::Result<InferenceResult>>) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(prompt.into()),
            config: GenerationConfig {
                temperature: 0.0,
                top_k: Some(1),
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                max_tokens,
                commitment_k: Some(4),
                ..GenerationConfig::default()
            },
            token_tx: None,
            result_tx,
            media: Vec::new(),
        })
        .expect("submit");
    rx.blocking_recv().unwrap().expect("generate")
}

fn engine(backend: &Arc<LlamaBackend>, path: &str) -> BatchEngine {
    let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    BatchEngine::spawn("m".into(), model, backend.clone(), 1024, false).expect("spawn")
}

/// The committed tokens and every committed logit, which must be finite.
fn tokens(var: &str, what: &str, r: &InferenceResult) -> Vec<u32> {
    let steps = &r.commitment.as_ref().expect("the batch engine records a commitment").steps;
    for (i, s) in steps.iter().enumerate() {
        assert!(s.top_k.iter().all(|e| e.logit.is_finite()), "{var}: {what}: a non-finite logit at step {i}");
    }
    steps.iter().map(|s| s.token_id).collect()
}

fn run(var: &str, path: &str, backend: &Arc<LlamaBackend>) {
    let warm = engine(backend, path);
    let first = generate(&warm, PROMPT, 24);
    tokens(var, "first answer", &first);

    // The exact extension: the prompt, what was answered, and a little more.
    let extended = format!("{PROMPT}{} Nobody in the village had ever seen", first.text);
    let again = generate(&warm, &extended, 16);
    let warm_tokens = tokens(var, "warm extension", &again);
    // The same request once more hits the cache in full.
    let repeat = generate(&warm, &extended, 16);
    assert_eq!(warm_tokens, tokens(var, "warm repeat", &repeat), "{var}: the same request answered differently from a warm cache");
    warm.shutdown();

    let cold = engine(backend, path);
    let cold_tokens = tokens(var, "cold extension", &generate(&cold, &extended, 16));
    cold.shutdown();
    assert_eq!(warm_tokens, cold_tokens, "{var}: an exact extension answered differently from a cold engine");
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL"]
fn an_exact_extension_answers_as_a_cold_engine_does() {
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
