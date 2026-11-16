//! An encoder-decoder model on the batch engine: one sequence slot, requests
//! served in turn. Needs `PRAECISE_TEST_ENCODER_DECODER_MODEL` (a small T5-style
//! GGUF); ignored by default and fails without it when run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::{BatchEngine, BatchLimits, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult};
use tokio::sync::oneshot;

fn submit(engine: &BatchEngine, prompt: String) -> oneshot::Receiver<praecise_runtime::Result<InferenceResult>> {
    let (result_tx, rx) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(prompt),
            config: GenerationConfig { temperature: 0.0, top_k: Some(1), max_tokens: 8, ..GenerationConfig::default() },
            token_tx: None,
            result_tx,
            media: Vec::new(),
        })
        .expect("submit");
    rx
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_ENCODER_DECODER_MODEL"]
fn an_encoder_decoder_model_is_served_from_one_slot() {
    let path = std::env::var("PRAECISE_TEST_ENCODER_DECODER_MODEL").expect("set PRAECISE_TEST_ENCODER_DECODER_MODEL");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    let model = LlamaModel::load_from_file(&backend, &path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    assert!(model.has_encoder() && model.has_decoder(), "not an encoder-decoder model");
    let limits = BatchLimits::default().with_physical_batch(64).with_max_slots(8);
    let engine =
        BatchEngine::spawn_with_limits("m".into(), model, backend.clone(), 512, false, None, None, None, limits).expect("spawn");

    // Several at once: each waits for the one slot instead of aborting the
    // process in the backend's cross-attention.
    let pending: Vec<_> = ["translate English to German: The house is wonderful.", "summarize: The sun is a star.", "translate English to French: Good morning."]
        .into_iter()
        .map(|p| submit(&engine, p.to_string()))
        .collect();
    for rx in pending {
        let r = rx.blocking_recv().unwrap().expect("served");
        assert!(r.output_tokens > 0);
    }
    assert_eq!(engine.slots(), 1, "an encoder-decoder engine has one slot whatever its limit");

    // An encoder input past one physical batch is refused, not aborted on.
    let long = "word ".repeat(200);
    let err = submit(&engine, long).blocking_recv().unwrap().unwrap_err();
    assert!(err.to_string().contains("physical batch"), "{err}");
    engine.shutdown();
}
