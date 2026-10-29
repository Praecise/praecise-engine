//! Aggregate decode throughput of the batch engine at several concurrencies,
//! for comparing engine layouts on one host.
//!
//! Needs `PRAECISE_TEST_THROUGHPUT_MODEL` (a GGUF). `PRAECISE_TEST_THROUGHPUT_CONCURRENCY`
//! lists the concurrencies (default `1,4,8`), `PRAECISE_TEST_THROUGHPUT_TOKENS`
//! the tokens per request (default 128), `PRAECISE_TEST_THROUGHPUT_CTX` the
//! engine context (default 8192). `PRAECISE_TEST_GPU_LAYERS` offloads layers.
//! Ignored by default; it reports and asserts only that every request finished.
//!
//! Opt-in sweep report: set `PRAECISE_TEST_THROUGHPUT_SWEEP=1` to also score
//! every level through the validity gate and print valid tokens per hour, the
//! marginal gain of each step and the chosen concurrency
//! (`PRAECISE_TEST_THROUGHPUT_GAIN`, default 0.05). `PRAECISE_TEST_THROUGHPUT_PIN`
//! gives the configuration as `key=value;key=value` and must carry
//! `model_digest`, `build`, `shape` and `hardware`. Latency here is the
//! engine's generation time, which excludes queue wait. On the shared Spark,
//! follow the host rules (capped wrapper, memory floor, GPU lock) before running.
#![cfg(feature = "bundled-llama")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use praecise_runtime::concurrency_sweep::{score_level, Attempt, LevelRun, LlmGate, Payload, Report, RunPin, SweepPlan};
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
    let sweep = std::env::var("PRAECISE_TEST_THROUGHPUT_SWEEP").is_ok_and(|v| v == "1");
    let mut scored = vec![];
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
        let mut attempts = vec![];
        for rx in pending {
            let r = rx.blocking_recv().expect("result").expect("generation");
            out += u64::from(r.output_tokens);
            attempts.push(Attempt {
                latency_s: r.generation_time_ms as f64 / 1000.0,
                units: f64::from(r.output_tokens),
                payload: Payload::Text(r.text),
                error: None,
            });
        }
        let secs = started.elapsed().as_secs_f64();
        if sweep {
            scored.push(score_level(n, &LevelRun { wall_s: secs, attempts }, &LlmGate::default()));
        }
        eprintln!("concurrency {n}: {out} tokens in {secs:.2} s, {:.1} tok/s aggregate", out as f64 / secs);
        assert!(out > 0);
    }
    if sweep {
        let mut pin = RunPin::new();
        for kv in std::env::var("PRAECISE_TEST_THROUGHPUT_PIN").unwrap_or_default().split(';') {
            if let Some((k, v)) = kv.split_once('=') {
                pin = pin.with(k.trim(), v.trim());
            }
        }
        let plan = SweepPlan {
            levels: scored.iter().map(|l| l.concurrency).collect(),
            requests_per_level: scored.iter().map(|l| l.attempts).max().unwrap_or(0),
            gain_threshold: env_or("PRAECISE_TEST_THROUGHPUT_GAIN", 0.05),
            latency_budget_s: None,
        };
        let missing = pin.missing();
        assert!(missing.is_empty(), "PRAECISE_TEST_THROUGHPUT_PIN lacks {missing:?}");
        eprintln!("{}", Report::from_levels(&pin, &plan, scored).table());
    }
    engine.shutdown();
}
