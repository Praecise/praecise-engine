//! Batch engines of one process, each under limits of its own.
//! Needs `PRAECISE_TEST_KV_MODEL` (a small GGUF); ignored by default and fails
//! without it when run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::{BatchEngine, BatchLimits, BatchPrompt, BatchRequest, GenerationConfig};
use tokio::sync::oneshot;

fn load(backend: &LlamaBackend, path: &str) -> LlamaModel {
    LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load")
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_KV_MODEL"]
fn two_engines_of_one_process_run_with_their_own_slot_counts() {
    let path = std::env::var("PRAECISE_TEST_KV_MODEL").expect("set PRAECISE_TEST_KV_MODEL");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    let spawn = |slots: usize| {
        let limits = BatchLimits { physical_batch: 256, prompt_checkpoints: 0, ..BatchLimits::default() }
            .with_max_slots(slots);
        BatchEngine::spawn_with_limits("m".into(), load(&backend, &path), backend.clone(), 512, false, None, None, None, limits)
            .expect("spawn")
    };
    let (two, three) = (spawn(2), spawn(3));
    for (engine, want) in [(&two, 2), (&three, 3)] {
        let (result_tx, rx) = oneshot::channel();
        engine
            .submit(BatchRequest {
                prompt: BatchPrompt::Raw("The capital of France is".into()),
                config: GenerationConfig { temperature: 0.0, top_k: Some(1), max_tokens: 4, ..GenerationConfig::default() },
                token_tx: None,
                result_tx,
                media: Vec::new(),
            })
            .expect("submit");
        let r = rx.blocking_recv().unwrap().expect("served");
        assert!(r.output_tokens > 0);
        // Read once the scheduler has fitted its slots, which it does before
        // serving the first request.
        assert_eq!(engine.slots(), want, "each engine has the slot count it was given");
    }
    two.shutdown();
    three.shutdown();
}
