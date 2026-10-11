//! Aggregate decode throughput of the batch engine at several concurrencies,
//! for comparing engine layouts on one host.
//!
//! Needs `PRAECISE_TEST_THROUGHPUT_MODEL` (a GGUF). `PRAECISE_TEST_THROUGHPUT_CONCURRENCY`
//! lists the concurrencies (default `1,4,8`), `PRAECISE_TEST_THROUGHPUT_TOKENS`
//! the tokens per request (default 128), `PRAECISE_TEST_THROUGHPUT_CTX` the
//! engine context (default 8192). `PRAECISE_TEST_GPU_LAYERS` offloads layers.
//! Ignored by default; it reports and asserts only that every request finished.
#![cfg(feature = "bundled-llama")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, ChatMessage, GenerationConfig};
use tokio::sync::oneshot;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "needs a model: PRAECISE_TEST_THROUGHPUT_MODEL"]
fn decode_throughput_by_concurrency() {
    let path = std::env::var("PRAECISE_TEST_THROUGHPUT_MODEL").expect("PRAECISE_TEST_THROUGHPUT_MODEL");
    let tokens: u32 = env_or("PRAECISE_TEST_THROUGHPUT_TOKENS", 128);
    let ctx: u32 = env_or("PRAECISE_TEST_THROUGHPUT_CTX", 8192);
    let levels: Vec<usize> = std::env::var("PRAECISE_TEST_THROUGHPUT_CONCURRENCY")
        .unwrap_or_else(|_| "1,4,8".into())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    let model = LlamaModel::load_from_file(&backend, &path, &common::params()).expect("load model");
    let engine = BatchEngine::spawn("throughput".into(), model, backend.clone(), ctx, false).expect("engine");
    for n in levels {
        let started = Instant::now();
        let mut pending = vec![];
        for i in 0..n {
            let (result_tx, result_rx) = oneshot::channel();
            let config = GenerationConfig {
                temperature: 0.0,
                repeat_penalty: 1.0,
                max_tokens: tokens,
                cache_salt: Some(format!("{n}-{i}")),
                ..Default::default()
            };
            let prompt = format!("Write a long story about lighthouse number {i} and the keeper who lived there.");
            engine
                .submit(BatchRequest {
                    prompt: BatchPrompt::Chat(vec![ChatMessage::new("user", prompt)]),
                    config,
                    token_tx: None,
                    result_tx,
                    media: vec![],
                })
                .expect("submit");
            pending.push(result_rx);
        }
        let mut out = 0u64;
        for rx in pending {
            out += u64::from(rx.blocking_recv().expect("result").expect("generation").output_tokens);
        }
        let secs = started.elapsed().as_secs_f64();
        eprintln!("concurrency {n}: {out} tokens in {secs:.2} s, {:.1} tok/s aggregate", out as f64 / secs);
        assert!(out > 0);
    }
    engine.shutdown();
}
