//! A fact placed deep in a long prompt is recalled through the batch engine:
//! a needle at several depths of a prompt that fills most of the context.
//!
//! Needs `PRAECISE_TEST_LONG_MODEL` (a GGUF with a chat template) and
//! `PRAECISE_TEST_LONG_CTX` (the context length, in tokens).
//! `PRAECISE_TEST_GPU_LAYERS` offloads layers; `PRAECISE_TEST_LONG_DEPTHS`
//! lists the depths in percent (default `10,50,90`). Ignored by default.
#![cfg(feature = "bundled-llama")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{AddBos, LlamaModel};
use praecise_runtime::{
    BatchEngine, BatchPrompt, BatchRequest, ChatFormat, ChatMessage, ChatTemplateSource, ChatTurn,
    GenerationConfig,
};
use tokio::sync::oneshot;

const FILLER: [&str; 6] = [
    "The archive kept its ledgers in long rows of oak shelves, each labelled by year and by the clerk who filed it.",
    "Rain fell on the harbour through most of the autumn, and the ferries ran late more often than not.",
    "A surveyor walked the northern ridge twice a year to check the boundary stones against the old maps.",
    "In the market square the bakers opened before dawn and the fishmongers shortly after.",
    "The council met on the first Tuesday of every month to hear petitions about roads and water.",
    "Most of the town's letters still went by the morning coach to the river crossing and on to the city.",
];

fn needle(depth: u32) -> (String, String) {
    let code = format!(
        "{}-{}",
        4000 + depth * 37,
        ["ORCHID", "LANTERN", "QUARRY", "MERIDIAN"][(depth % 4) as usize]
    );
    (
        format!("Remember this: the vault access code is {code}."),
        code,
    )
}

#[test]
#[ignore = "needs a long-context model: PRAECISE_TEST_LONG_MODEL, PRAECISE_TEST_LONG_CTX"]
fn needle_is_recalled_at_depth() {
    let path = std::env::var("PRAECISE_TEST_LONG_MODEL").expect("PRAECISE_TEST_LONG_MODEL");
    let n_ctx: u32 = std::env::var("PRAECISE_TEST_LONG_CTX")
        .expect("PRAECISE_TEST_LONG_CTX")
        .parse()
        .unwrap();
    let depths: Vec<u32> = std::env::var("PRAECISE_TEST_LONG_DEPTHS")
        .unwrap_or_else(|_| "10,50,90".into())
        .split(',')
        .map(|d| d.trim().parse().unwrap())
        .collect();
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    let model = LlamaModel::load_from_file(&backend, &path, &common::params()).expect("load model");
    let source = ChatTemplateSource::from_model(&model).expect("chat template");
    eprintln!("model {path}: trained context {}", model.n_ctx_train());

    // Size the haystack to leave room for the template, the question and the answer.
    let per_sentence = {
        let toks = model
            .str_to_token(&FILLER.join(" "), AddBos::Never)
            .unwrap();
        toks.len() as f64 / FILLER.len() as f64
    };
    let budget = (n_ctx as f64 * 0.92) as usize - 400;
    let sentences = (budget as f64 / per_sentence) as usize;
    let engine =
        BatchEngine::spawn("long".into(), model, backend.clone(), n_ctx, false).expect("engine");

    let mut failed = vec![];
    for depth in depths {
        let (fact, code) = needle(depth);
        let at = sentences * depth as usize / 100;
        let mut text = String::new();
        for i in 0..sentences {
            if i == at {
                text.push_str(&fact);
                text.push(' ');
            }
            text.push_str(FILLER[i % FILLER.len()]);
            text.push(' ');
        }
        let messages = vec![ChatMessage::new(
            "user",
            format!("{text}\n\nWhat is the vault access code mentioned above? Reply with the code only."),
        )];
        let turn = ChatTurn {
            enable_thinking: false,
            ..ChatTurn::new(&messages)
        };
        let format = Arc::new(ChatFormat::render(&source, &turn).expect("render"));
        let mut config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 1.0,
            max_tokens: 48,
            ..Default::default()
        };
        config
            .stop
            .extend(format.template_result().additional_stops.iter().cloned());
        config.reply_format = Some(format.clone());
        let (result_tx, result_rx) = oneshot::channel();
        let started = Instant::now();
        engine
            .submit(BatchRequest {
                prompt: BatchPrompt::Raw(format.prompt().to_string()),
                config,
                token_tx: None,
                result_tx,
                media: vec![],
            })
            .expect("submit");
        let r = result_rx
            .blocking_recv()
            .expect("result")
            .expect("generation");
        let secs = started.elapsed().as_secs_f64();
        let ok = r.text.contains(&code);
        eprintln!(
            "depth {depth}%: prompt {} tokens, {:.1} s ({:.0} prompt tok/s), answer {:?}, want {code}: {}",
            r.input_tokens,
            secs,
            r.input_tokens as f64 / secs,
            r.text.trim(),
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            failed.push(depth);
        }
    }
    engine.shutdown();
    assert!(failed.is_empty(), "needle missed at depths {failed:?}");
}
