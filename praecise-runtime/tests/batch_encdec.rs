//! An encoder-decoder model on the batch engine: it decodes one request at a
//! time, however many slots the limits allow, and two requests sent together
//! both finish. Needs `PRAECISE_TEST_ENCDEC_MODEL` (a T5-class GGUF, such as
//! the catalog's flan-t5-small); ignored by default and fails without it when
//! run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult};
use tokio::sync::oneshot;

fn submit(engine: &BatchEngine, prompt: &str) -> oneshot::Receiver<praecise_runtime::Result<InferenceResult>> {
    let (result_tx, rx) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(prompt.into()),
            config: GenerationConfig {
                temperature: 0.0,
                top_k: Some(1),
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                max_tokens: 16,
                ..GenerationConfig::default()
            },
            token_tx: None,
            result_tx,
            media: Vec::new(),
        })
        .expect("submit");
    rx
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_ENCDEC_MODEL"]
fn an_encoder_decoder_model_serves_two_concurrent_requests_one_at_a_time() {
    let path = std::env::var("PRAECISE_TEST_ENCDEC_MODEL").expect("set PRAECISE_TEST_ENCDEC_MODEL");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    let model = LlamaModel::load_from_file(&backend, &path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    let engine = BatchEngine::spawn("m".into(), model, backend.clone(), 512, false).expect("spawn");
    assert_eq!(engine.slots(), 1, "an encoder-decoder model runs one sequence whatever the limit");

    let a = submit(&engine, "Translate English to German: The house is wonderful.");
    let b = submit(&engine, "Answer the question: what is the capital of France?");
    let a = a.blocking_recv().unwrap().expect("first request");
    let b = b.blocking_recv().unwrap().expect("second request");
    assert!(!a.text.trim().is_empty(), "the first request answered nothing");
    assert!(!b.text.trim().is_empty(), "the second request answered nothing");
    engine.shutdown();
}
