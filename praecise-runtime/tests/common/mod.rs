//! Shared helpers for the model-backed runtime tests.
#![allow(dead_code, missing_docs)]
#![cfg(feature = "bundled-llama")]

use std::num::NonZeroU32;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

/// Path of a small dense GGUF model, or `None` to skip.
pub fn model_path() -> Option<String> {
    let p = std::env::var("PRAECISE_TEST_KV_MODEL").ok();
    if p.is_none() {
        eprintln!("PRAECISE_TEST_KV_MODEL not set; skipping model-backed test");
    }
    p
}

pub fn backend() -> &'static LlamaBackend {
    static B: OnceLock<LlamaBackend> = OnceLock::new();
    B.get_or_init(|| {
        let mut b = LlamaBackend::init().expect("backend");
        b.void_logs();
        b
    })
}

pub fn cpu_params() -> LlamaModelParams {
    LlamaModelParams::default().with_n_gpu_layers(0)
}

pub fn load(path: &str) -> LlamaModel {
    LlamaModel::load_from_file(backend(), path, &cpu_params()).expect("load model")
}

/// Two sequences (one serving, one scratch) over a unified KV buffer.
pub fn context(model: &LlamaModel) -> LlamaContext<'_> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_n_batch(512)
        .with_n_seq_max(2)
        .with_kv_unified(true)
        .with_n_threads(4)
        .with_n_threads_batch(4);
    model.new_context(backend(), params).expect("context")
}

/// Decode `tokens` into sequence `seq` starting at `start`; logits of the last.
pub fn decode(ctx: &mut LlamaContext<'_>, seq: i32, start: usize, tokens: &[LlamaToken]) -> Vec<f32> {
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    for (i, t) in tokens.iter().enumerate() {
        batch.add(*t, (start + i) as i32, &[seq], i + 1 == tokens.len()).expect("batch add");
    }
    ctx.decode(&mut batch).expect("decode");
    ctx.get_logits_ith(tokens.len() as i32 - 1).to_vec()
}
