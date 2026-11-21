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
use praecise_runtime::toploc::StepRecord;
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult};
use tokio::sync::oneshot;

const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step and \
                      listening to the sea below. One night the lamp would not light, and";

fn generate(engine: &BatchEngine, prompt: &str, salt: &str) -> InferenceResult {
    let (result_tx, rx): (_, oneshot::Receiver<praecise_runtime::Result<InferenceResult>>) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(prompt.into()),
            config: GenerationConfig {
                temperature: 0.0,
                top_k: Some(1),
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                max_tokens: 12,
                seed: 7,
                enable_thinking: Some(false),
                cache_salt: Some(salt.into()),
                commitment_k: Some(8),
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
    BatchEngine::spawn("m".into(), model, backend.clone(), 4096, false).expect("spawn")
}

/// The committed steps; every committed logit must be finite.
fn steps<'a>(var: &str, what: &str, r: &'a InferenceResult) -> &'a [StepRecord] {
    let steps = &r.commitment.as_ref().expect("the batch engine records a commitment").steps;
    for (i, s) in steps.iter().enumerate() {
        assert!(s.top_k.iter().all(|e| e.logit.is_finite()), "{var}: {what}: a non-finite logit at step {i}");
    }
    steps
}

/// The score gap, in one step's committed top-k, between its best token and `token`.
fn gap(step: &StepRecord, token: u32) -> Option<f32> {
    let best = step.top_k.iter().map(|e| e.logit).fold(f32::MIN, f32::max);
    step.top_k.iter().find(|e| e.token_id == token).map(|e| best - e.logit)
}

/// Decoded one token at a time and prefilled at once, the logits round apart
/// (measured up to 0.5 on CPU, as in `batch_scoring`), so a cache hit of another
/// length may swap a near tie. Two answers are the same up to that rounding when
/// they are the same tokens, or when at the step where they part each one's token
/// is within the rounding of the other's best score.
const ROUNDING: f32 = 0.5;

fn same(var: &str, what: &str, a: &[StepRecord], b: &[StepRecord]) {
    let at = match a.iter().zip(b).position(|(x, y)| x.token_id != y.token_id) {
        None => {
            assert_eq!(a.len(), b.len(), "{var}: {what}: one answer is longer");
            return;
        }
        Some(at) => at,
    };
    let (x, y) = (&a[at], &b[at]);
    let (ga, gb) = (gap(x, y.token_id), gap(y, x.token_id));
    let near = |g: Option<f32>| g.is_some_and(|g| g <= ROUNDING);
    let top = |s: &StepRecord| s.top_k.iter().map(|e| (e.token_id, e.logit)).collect::<Vec<_>>();
    assert!(near(ga) && near(gb), "{var}: {what}: the answers part at step {at} beyond rounding: {:?} vs {:?}", top(x), top(y));
}

fn run(var: &str, path: &str, backend: &Arc<LlamaBackend>) {
    let warm = engine(backend, path);
    let first = generate(&warm, PROMPT, "tenant-a");
    steps(var, "first answer", &first);
    // The same request again: its whole prompt is in the cache, and the answer
    // still carries the commitment that was asked for.
    let again = generate(&warm, PROMPT, "tenant-a");
    steps(var, "identical re-send", &again);
    assert!(again.cached_tokens + 1 >= again.input_tokens, "{var}: an identical re-send reused {} of {}", again.cached_tokens, again.input_tokens);

    // The exact extension: the prompt, what was answered, and one more turn.
    let extended = format!("{PROMPT}{}\n\nAnd one more.", again.text);
    let warm_answer = generate(&warm, &extended, "tenant-a");
    let repeat = generate(&warm, &extended, "tenant-a");
    same(var, "the same request answered differently from a warm cache", steps(var, "warm extension", &warm_answer), steps(var, "warm repeat", &repeat));
    warm.shutdown();

    let cold = engine(backend, path);
    let cold_answer = generate(&cold, &extended, "cold-a");
    cold.shutdown();
    same(var, "an exact extension answered differently from a cold engine", steps(var, "warm extension", &warm_answer), steps(var, "cold extension", &cold_answer));
    assert_eq!(warm_answer.text.is_empty(), cold_answer.text.is_empty(), "{var}: one answer is empty");
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
