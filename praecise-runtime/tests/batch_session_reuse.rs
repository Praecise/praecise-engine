//! A checkpoint carried to another engine is reused by the next turn of its
//! session: the node resumes the chain for one token (the first token of the
//! next history), then submits the turn, which must find the resumed positions
//! in the slot's prompt cache and prefill only what follows. Needs
//! `PRAECISE_TEST_KV_MODEL` (a small GGUF) and, optionally,
//! `PRAECISE_TEST_HYBRID_MODEL` (attention and recurrent layers); ignored by
//! default and fails without them when run with `--ignored`.
#![cfg(feature = "bundled-llama")]

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::{
    BatchEngine, BatchPrompt, BatchRequest, GenerationConfig, InferenceResult, KvBlob, ModelFingerprint, SequenceResume,
};
use tokio::sync::oneshot;

const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step and \
                      listening to the sea below. One night the lamp would not light, and";

fn config(max_tokens: u32) -> GenerationConfig {
    GenerationConfig {
        temperature: 0.0,
        top_k: Some(1),
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        max_tokens,
        commitment_k: Some(4),
        cache_salt: Some("session/reuse".into()),
        ..GenerationConfig::default()
    }
}

fn request(prompt: &str, max_tokens: u32) -> (BatchRequest, oneshot::Receiver<praecise_runtime::Result<InferenceResult>>) {
    let (result_tx, rx) = oneshot::channel();
    let req = BatchRequest { prompt: BatchPrompt::Raw(prompt.into()), config: config(max_tokens), token_tx: None, result_tx, media: Vec::new() };
    (req, rx)
}

fn engine(backend: &Arc<LlamaBackend>, path: &str, fp: ModelFingerprint) -> BatchEngine {
    let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default().with_n_gpu_layers(0)).expect("load");
    BatchEngine::spawn_with_migration("m".into(), model, backend.clone(), 1024, false, None, None, fp).expect("spawn")
}

fn run(var: &str, path: &str, backend: &Arc<LlamaBackend>) {
    let fp = ModelFingerprint::of_file(path).expect("fingerprint");
    let (src, dst) = (engine(backend, path, fp), engine(backend, path, fp));

    // The first engine answers one token and hands over its checkpoint: the prompt.
    let (req, one) = request(PROMPT, 1);
    let ticket = src.submit_tracked(req).expect("submit tracked");
    let blob = src.export_sequence(ticket).unwrap().blocking_recv().unwrap().expect("export at the free");
    let one = one.blocking_recv().unwrap().expect("the first request finishes");
    let carried = KvBlob::decode(&blob).unwrap().tokens.len();
    let first = one.commitment.as_ref().expect("a commitment").steps[0].token_id as i32;

    // The second engine resumes the chain for one token, as the node does.
    let (result_tx, resumed) = oneshot::channel();
    dst.resume(SequenceResume {
        blobs: vec![blob],
        next_token: first,
        generated: String::new(),
        config: config(1),
        token_tx: None,
        result_tx,
    })
    .expect("resume");
    resumed.blocking_recv().unwrap().expect("resumed");

    // The next turn extends what was carried and must reuse it.
    let (req, turn) = request(&format!("{PROMPT}{} and then", one.text), 8);
    dst.submit(req).expect("submit");
    let turn = turn.blocking_recv().unwrap().expect("the turn");
    eprintln!("{var}: carried {carried}, reused {} of {} prompt tokens", turn.cached_tokens, turn.input_tokens);
    assert!(
        turn.cached_tokens as usize + 2 >= carried,
        "{var}: the turn prefilled again what the checkpoint carried: reused {} of {carried}",
        turn.cached_tokens
    );
    src.shutdown();
    dst.shutdown();
}

#[test]
#[ignore = "needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL"]
fn the_next_turn_reuses_the_checkpoint_a_session_was_resumed_from() {
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
