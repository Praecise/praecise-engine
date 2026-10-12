//! Speculative decoding in the batch engine gives exactly what plain decoding
//! gives: the same tokens, greedy or seeded, alone or beside other requests,
//! streamed or whole, and whatever the drafter proposes.
//!
//! Needs `PRAECISE_TEST_SPEC_MODEL` (a GGUF). `PRAECISE_TEST_SPEC_METHOD`
//! picks the drafting method: `prompt-lookup` (default) or `mtp-head` for a
//! model that ships one. `PRAECISE_TEST_SPEC_DRAFT` is the most tokens drafted
//! per step (default 6), `PRAECISE_TEST_SPEC_TOKENS` the tokens per request
//! (default 96), `PRAECISE_TEST_GPU_LAYERS` offloads layers. The speed table
//! takes `PRAECISE_TEST_SPEC_CONCURRENCY` (default `1,4,8`).
//! `PRAECISE_TEST_SPEC_NO_REPACK` loads the weights without the CPU's repacked
//! layouts, whose multi-row kernel rounds differently from the one-row kernel.
#![cfg(feature = "bundled-llama")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use praecise_runtime::{
    BatchEngine, BatchPrompt, BatchRequest, BatchSpeculation, ChatMessage, GenerationConfig, InferenceResult,
};
use tokio::sync::{mpsc, oneshot};

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn model_path() -> String {
    std::env::var("PRAECISE_TEST_SPEC_MODEL").expect("PRAECISE_TEST_SPEC_MODEL names a GGUF")
}

fn method() -> i32 {
    match std::env::var("PRAECISE_TEST_SPEC_METHOD").as_deref() {
        Ok("mtp-head") => BatchSpeculation::MTP_HEAD,
        Ok("dflash") => BatchSpeculation::DFLASH,
        _ => BatchSpeculation::PROMPT_LOOKUP,
    }
}

const CTX: u32 = 4096;

const CODE: &str = "fn total_price(items: &[Item], discount: f64) -> f64 {\n    let mut total = 0.0;\n    \
for item in items {\n        let line = item.unit_price * item.quantity as f64;\n        total += line;\n    }\n    \
if discount > 0.0 {\n        total = total * (1.0 - discount);\n    }\n    total\n}\n\nfn total_weight(items: &[Item]) -> f64 {\n    \
let mut total = 0.0;\n    for item in items {\n        total += item.unit_weight * item.quantity as f64;\n    }\n    total\n}\n";

/// Prompts whose answers draw on their own text, and prompts that do not.
fn prompts() -> Vec<(&'static str, String)> {
    vec![
        ("edit", format!("Rename `items` to `basket` everywhere in this Rust code and print the whole result:\n\n{CODE}")),
        ("quote", format!("Copy the following code exactly, then explain it in two sentences:\n\n{CODE}")),
        (
            "list",
            "List the days of the week, each on its own line as `Day N: name`, twice in a row.".to_string(),
        ),
        ("story", "Write a short story about a lighthouse keeper who finds a letter in a bottle.".to_string()),
        ("facts", "Explain how a refrigerator moves heat from inside to outside.".to_string()),
    ]
}

fn greedy(tokens: u32) -> GenerationConfig {
    GenerationConfig {
        temperature: 0.0,
        top_k: Some(1),
        repeat_penalty: 1.0,
        max_tokens: tokens,
        commitment_k: Some(4),
        ..GenerationConfig::default()
    }
}

fn seeded(tokens: u32, seed: u64) -> GenerationConfig {
    GenerationConfig {
        temperature: 0.8,
        top_k: Some(40),
        top_p: 0.95,
        repeat_penalty: 1.0,
        seed,
        max_tokens: tokens,
        commitment_k: Some(4),
        ..GenerationConfig::default()
    }
}

/// The tokens a result committed to, in order.
fn tokens(r: &InferenceResult) -> Vec<u32> {
    r.commitment.as_ref().expect("commitment requested").steps.iter().map(|s| s.token_id).collect()
}

fn load(backend: &LlamaBackend) -> LlamaModel {
    let mut params = common::params();
    if std::env::var("PRAECISE_TEST_SPEC_NO_REPACK").is_ok() {
        params = params.with_use_extra_bufts(false);
    }
    if method() == BatchSpeculation::MTP_HEAD {
        params = params.with_load_mtp(true);
    }
    LlamaModel::load_from_file(backend, &model_path(), &params).expect("load model")
}

fn backend() -> Arc<LlamaBackend> {
    let mut b = LlamaBackend::init().expect("backend");
    b.void_logs();
    Arc::new(b)
}

fn plain(backend: &Arc<LlamaBackend>) -> BatchEngine {
    BatchEngine::spawn_with_speculation("plain".into(), load(backend), backend.clone(), CTX, false, None)
        .expect("plain engine")
}

fn speculative(backend: &Arc<LlamaBackend>, spec_type: i32) -> BatchEngine {
    let n_max: u8 = env_or("PRAECISE_TEST_SPEC_DRAFT", 6);
    let engine = BatchEngine::spawn_with_speculation(
        "spec".into(),
        load(backend),
        backend.clone(),
        CTX,
        false,
        Some(BatchSpeculation { spec_type, n_max, draft_model: None }),
    )
    .expect("speculative engine");
    engine
}

struct Pending {
    result: oneshot::Receiver<praecise_runtime::Result<InferenceResult>>,
    stream: Option<mpsc::Receiver<String>>,
}

fn submit(engine: &BatchEngine, prompt: &str, config: GenerationConfig, streamed: bool) -> Pending {
    let (result_tx, result) = oneshot::channel();
    let (token_tx, stream) = if streamed {
        let (tx, rx) = mpsc::channel(4096);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Chat(vec![ChatMessage::new("user", prompt)]),
            config,
            token_tx,
            result_tx,
            media: vec![],
        })
        .expect("submit");
    Pending { result, stream }
}

/// The result, and for a streamed request the text its stream carried.
fn finish(p: Pending) -> (InferenceResult, Option<String>) {
    let mut stream = p.stream;
    let mut text = String::new();
    if let Some(rx) = stream.as_mut() {
        while let Some(chunk) = rx.blocking_recv() {
            text.push_str(&chunk);
        }
    }
    let r = p.result.blocking_recv().expect("result").expect("generation");
    (r, stream.map(|_| text))
}

fn run_alone(engine: &BatchEngine, cases: &[(String, GenerationConfig)], streamed: bool) -> Vec<(InferenceResult, Option<String>)> {
    cases.iter().map(|(p, c)| finish(submit(engine, p, c.clone(), streamed))).collect()
}

fn run_together(engine: &BatchEngine, cases: &[(String, GenerationConfig)]) -> Vec<(InferenceResult, Option<String>)> {
    // Every other request streams, so streamed and whole share steps.
    let pending: Vec<Pending> =
        cases.iter().enumerate().map(|(i, (p, c))| submit(engine, p, c.clone(), i % 2 == 1)).collect();
    pending.into_iter().map(finish).collect()
}

fn cases(tokens_per: u32) -> Vec<(String, GenerationConfig)> {
    let mut out = Vec::new();
    for (i, (_, p)) in prompts().into_iter().enumerate() {
        out.push((p.clone(), greedy(tokens_per)));
        out.push((p, seeded(tokens_per, 1000 + i as u64)));
    }
    out
}

fn assert_same(label: &str, want: &[(InferenceResult, Option<String>)], got: &[(InferenceResult, Option<String>)]) {
    assert_eq!(want.len(), got.len());
    for (i, ((w, _), (g, streamed))) in want.iter().zip(got).enumerate() {
        assert_eq!(tokens(w), tokens(g), "{label}: case {i} decoded different tokens");
        assert_eq!(w.text, g.text, "{label}: case {i} text differs");
        assert_eq!(w.output_tokens, g.output_tokens, "{label}: case {i} token count differs");
        if let Some(s) = streamed {
            let joined = match &g.thinking {
                Some(_) => continue,
                None => s,
            };
            assert_eq!(joined.trim(), g.text.trim(), "{label}: case {i} stream differs from the result");
        }
    }
}

/// Plain decoding one request at a time, then speculative decoding of the same
/// requests one at a time (whole, streamed, and again over cached prefixes,
/// each after the engine has served the others), then all at once when
/// `concurrent` is set.
fn check_matches(spec_type: i32, concurrent: bool) {
    let tokens_per: u32 = env_or("PRAECISE_TEST_SPEC_TOKENS", 96);
    let backend = backend();
    let cases = cases(tokens_per);

    let reference = {
        let engine = plain(&backend);
        let r = run_alone(&engine, &cases, false);
        if concurrent {
            // Plain decoding must not depend on what it is batched with, or
            // no concurrent comparison can hold.
            let together = run_together(&engine, &cases);
            assert_same("plain, together", &r, &together);
        }
        let report = engine.speculation();
        assert!(report.method.is_none(), "the plain engine reports no speculation");
        assert_eq!(report.off_reason.as_deref(), Some("not requested"));
        engine.shutdown();
        r
    };

    let engine = speculative(&backend, spec_type);
    let alone_whole = run_alone(&engine, &cases, false);
    // The engine has served requests, so it has set up speculation or said why not.
    let report = engine.speculation();
    assert!(report.method.is_some(), "speculation is on: {report:?}");
    assert_same("alone, whole", &reference, &alone_whole);
    let alone_streamed = run_alone(&engine, &cases, true);
    assert_same("alone, streamed", &reference, &alone_streamed);
    let alone_cached = run_alone(&engine, &cases, false);
    assert_same("alone, cached prefixes", &reference, &alone_cached);
    if concurrent {
        let together = run_together(&engine, &cases);
        assert_same("together", &reference, &together);
    }

    let report = engine.speculation();
    eprintln!(
        "{}: drafted {}, accepted {}, acceptance {:.3}, verify steps {}",
        report.method.unwrap_or("none"),
        report.drafted,
        report.accepted,
        report.acceptance_rate().unwrap_or(0.0),
        report.verify_steps
    );
    assert!(report.drafted > 0, "drafts were made and verified");
    engine.shutdown();
}

#[test]
#[ignore = "needs a model: PRAECISE_TEST_SPEC_MODEL"]
fn speculative_matches_plain() {
    check_matches(method(), false);
}

/// Drafts that are wrong at a varying position of every block, and padded with
/// wrong tokens, still produce exactly the plain output.
#[test]
#[ignore = "needs a model: PRAECISE_TEST_SPEC_MODEL"]
fn bad_drafts_never_change_the_output() {
    check_matches(BatchSpeculation::CORRUPTED_LOOKUP, false);
}

/// The same, with every request in flight at once. Holds only where plain
/// decoding gives a sequence the same tokens whatever it is batched with,
/// which this checks first.
#[test]
#[ignore = "needs a model: PRAECISE_TEST_SPEC_MODEL"]
fn speculative_matches_plain_concurrent() {
    check_matches(method(), true);
}

/// Aggregate tok/s with and without speculation at several concurrencies, and
/// the acceptance rate, per prompt kind, and how many outputs equal plain.
#[test]
#[ignore = "needs a model: PRAECISE_TEST_SPEC_MODEL"]
fn speculative_speed_by_concurrency() {
    let tokens_per: u32 = env_or("PRAECISE_TEST_SPEC_TOKENS", 128);
    let levels: Vec<usize> = std::env::var("PRAECISE_TEST_SPEC_CONCURRENCY")
        .unwrap_or_else(|_| "1,4,8".into())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let backend = backend();
    let groups: Vec<(&str, Vec<String>)> = {
        let all = prompts();
        let grounded = all.iter().filter(|(k, _)| matches!(*k, "edit" | "quote" | "list")).map(|(_, p)| p.clone()).collect();
        let open = all.iter().filter(|(k, _)| matches!(*k, "story" | "facts")).map(|(_, p)| p.clone()).collect();
        vec![("grounded", grounded), ("open", open)]
    };
    // One engine at a time, so a large model is never held twice.
    let batches: Vec<(String, usize, Vec<(String, GenerationConfig)>)> = groups
        .iter()
        .flat_map(|(kind, pool)| {
            levels.iter().map(move |&n| {
                // n distinct requests, each with its own salt so none reuses
                // another's cache.
                let batch = (0..n)
                    .map(|i| {
                        let mut c = greedy(tokens_per);
                        c.commitment_k = None;
                        c.cache_salt = Some(format!("{kind}-{n}-{i}"));
                        (format!("{}\n\n(Request {}.)", pool[i % pool.len()], i + 1), c)
                    })
                    .collect();
                (kind.to_string(), n, batch)
            })
        })
        .collect();
    let plain_engine = plain(&backend);
    let plain_runs: Vec<(Vec<(InferenceResult, Option<String>)>, f64)> = batches
        .iter()
        .map(|(_, _, batch)| {
            let t = Instant::now();
            let r = run_together(&plain_engine, batch);
            (r, t.elapsed().as_secs_f64())
        })
        .collect();
    plain_engine.shutdown();
    drop(plain_engine);

    let spec_engine = speculative(&backend, method());
    let mut table = Vec::new();
    for ((kind, n, batch), (want, plain_secs)) in batches.iter().zip(&plain_runs) {
        let before = spec_engine.speculation();
        let t = Instant::now();
        let got = run_together(&spec_engine, batch);
        let spec_secs = t.elapsed().as_secs_f64();
        let after = spec_engine.speculation();
        // Identity is proven by the tests above; the table records it for
        // the measured runs too.
        let same = want.iter().zip(&got).filter(|((w, _), (g, _))| w.text == g.text).count();
        let out: u64 = got.iter().map(|(r, _)| u64::from(r.output_tokens)).sum();
        let drafted = after.drafted - before.drafted;
        let accepted = after.accepted - before.accepted;
        table.push((
            kind.clone(),
            *n,
            out as f64 / plain_secs,
            out as f64 / spec_secs,
            if drafted > 0 { accepted as f64 / drafted as f64 } else { 0.0 },
            format!("{same}/{}", got.len()),
        ));
    }
    let method_name = spec_engine.speculation().method.unwrap_or("none");
    eprintln!("| method | prompts | concurrency | plain tok/s | speculative tok/s | speedup | acceptance | same as plain |");
    eprintln!("|---|---|---|---|---|---|---|---|");
    for (kind, n, p, s, a, same) in &table {
        eprintln!("| {method_name} | {kind} | {n} | {p:.1} | {s:.1} | {:.2}x | {a:.3} | {same} |", s / p);
    }
    spec_engine.shutdown();
}

/// The precondition for every test above: a token's logits are the same bits
/// whether it is decoded alone or as one row of a verify block. Decodes the
/// same continuation one token per step in one context and as blocks of
/// `PRAECISE_TEST_SPEC_DRAFT + 1` in another, and compares every row.
#[test]
#[ignore = "needs a model: PRAECISE_TEST_SPEC_MODEL"]
fn verify_rows_match_single_decode() {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;
    use std::num::NonZeroU32;

    let block: usize = env_or::<usize>("PRAECISE_TEST_SPEC_DRAFT", 6) + 1;
    let backend = backend();
    let model = load(&backend);
    let params = || {
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048))
            .with_n_batch(512)
            .with_n_ubatch(512)
            .with_n_seq_max(1)
            .with_kv_unified(false)
            .with_n_rs_seq(block as u32)
    };
    let text = format!("Copy this code exactly:\n\n{CODE}");
    let all = model.str_to_token(&text, AddBos::Always).expect("tokenize");
    let split = all.len() / 2;
    let (prompt, rest) = all.split_at(split);
    let rest = &rest[..rest.len().min(4 * block)];

    let prefill = |ctx: &mut llama_cpp_2::context::LlamaContext<'_>| {
        let mut b = LlamaBatch::new(prompt.len(), 1);
        for (i, t) in prompt.iter().enumerate() {
            b.add(*t, i as i32, &[0], false).unwrap();
        }
        ctx.decode(&mut b).expect("prefill");
    };
    let mut single = model.new_context(&backend, params()).expect("ctx");
    prefill(&mut single);
    let mut want: Vec<Vec<f32>> = Vec::new();
    for (i, t) in rest.iter().enumerate() {
        let mut b = LlamaBatch::new(1, 1);
        b.add(*t, (split + i) as i32, &[0], true).unwrap();
        single.decode(&mut b).expect("single");
        want.push(single.get_logits_ith(0).to_vec());
    }
    drop(single);

    let mut blocked = model.new_context(&backend, params()).expect("ctx");
    prefill(&mut blocked);
    let mut got: Vec<Vec<f32>> = Vec::new();
    for (c, chunk) in rest.chunks(block).enumerate() {
        let mut b = LlamaBatch::new(chunk.len(), 1);
        for (i, t) in chunk.iter().enumerate() {
            b.add(*t, (split + c * block + i) as i32, &[0], true).unwrap();
        }
        blocked.decode(&mut b).expect("block");
        for i in 0..chunk.len() {
            got.push(blocked.get_logits_ith(i as i32).to_vec());
        }
    }
    let mut mismatched = 0;
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        let max = w.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let same_bits = w.iter().zip(g).all(|(a, b)| a.to_bits() == b.to_bits());
        if !same_bits {
            mismatched += 1;
        }
        eprintln!("row {i} (block row {}): bit-identical {same_bits}, max |diff| {max:e}", i % block);
    }
    assert_eq!(mismatched, 0, "{mismatched} of {} rows differ from single-token decode", want.len());
}

/// What plain concurrent decoding needs: a sequence's logits do not depend on
/// how its prompt was cut into prefill chunks, nor on another sequence's rows
/// in the same decode. Reports each separately.
#[test]
#[ignore = "needs a model: PRAECISE_TEST_SPEC_MODEL"]
fn rows_match_across_batch_composition() {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;
    use std::num::NonZeroU32;

    let backend = backend();
    let model = load(&backend);
    let params = || {
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(4096))
            .with_n_batch(512)
            .with_n_ubatch(512)
            .with_n_seq_max(2)
            .with_kv_unified(false)
    };
    let a = model.str_to_token(&format!("Copy this code exactly:\n\n{CODE}"), AddBos::Always).unwrap();
    let b = model.str_to_token("Explain how a refrigerator moves heat from inside to outside.", AddBos::Always).unwrap();
    let last_logits = |ctx: &llama_cpp_2::context::LlamaContext<'_>, i: i32| ctx.get_logits_ith(i).to_vec();
    let same = |x: &[f32], y: &[f32]| {
        let max = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        (x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), max)
    };

    // Seq 0's prompt in one prefill, alone.
    let mut whole = model.new_context(&backend, params()).unwrap();
    let mut bt = LlamaBatch::new(a.len(), 1);
    for (i, t) in a.iter().enumerate() {
        bt.add(*t, i as i32, &[0], i + 1 == a.len()).unwrap();
    }
    whole.decode(&mut bt).unwrap();
    let want = last_logits(&whole, a.len() as i32 - 1);
    drop(whole);

    // The same prompt in chunks of 7.
    let mut chunked = model.new_context(&backend, params()).unwrap();
    for (c, chunk) in a.chunks(7).enumerate() {
        let mut bt = LlamaBatch::new(chunk.len(), 1);
        for (i, t) in chunk.iter().enumerate() {
            let pos = c * 7 + i;
            bt.add(*t, pos as i32, &[0], pos + 1 == a.len()).unwrap();
        }
        chunked.decode(&mut bt).unwrap();
    }
    let got = last_logits(&chunked, ((a.len() - 1) % 7) as i32);
    drop(chunked);
    let (chunk_same, chunk_max) = same(&want, &got);
    eprintln!("prefill in chunks of 7 vs whole: bit-identical {chunk_same}, max |diff| {chunk_max:e}");

    // The same prompt in one prefill beside another sequence's prompt.
    let mut beside = model.new_context(&backend, params()).unwrap();
    let mut bt = LlamaBatch::new(a.len() + b.len(), 2);
    for (i, t) in b.iter().enumerate() {
        bt.add(*t, i as i32, &[1], false).unwrap();
    }
    for (i, t) in a.iter().enumerate() {
        bt.add(*t, i as i32, &[0], i + 1 == a.len()).unwrap();
    }
    beside.decode(&mut bt).unwrap();
    let got = last_logits(&beside, (a.len() + b.len()) as i32 - 1);
    let (beside_same, beside_max) = same(&want, &got);
    eprintln!("prefill beside another sequence vs alone: bit-identical {beside_same}, max |diff| {beside_max:e}");

    assert!(chunk_same && beside_same, "a sequence's logits depend on how it is batched");
}
