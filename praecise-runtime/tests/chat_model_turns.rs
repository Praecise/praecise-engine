//! Chat turns through a real model and the batch engine: a reply that calls a
//! tool comes back as a parsed call, the reasoning apart from the answer, and
//! the streamed reply carries the same parts as the whole one, with no
//! template markup in either.
//!
//! Needs `PRAECISE_TEST_CHAT_MODELS`, a list of GGUF paths separated by `;`,
//! each with a chat template that offers tools. `PRAECISE_TEST_GPU_LAYERS`
//! offloads layers. Ignored by default; fails without the models when run with
//! `--ignored`.
#![cfg(feature = "bundled-llama")]

mod common;

use std::sync::Arc;
use std::time::Duration;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use praecise_runtime::{
    BatchEngine, BatchPrompt, BatchRequest, ChatFormat, ChatMessage, ChatTemplateSource, ChatTurn,
    GenerationConfig, InferenceResult,
};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

const MARKUP: &[&str] = &[
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<|channel|>",
    "<|im_end|>",
    "<|im_start|>",
    "[TOOL_CALLS]",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|tool_call>",
];

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string", "description": "City name" } },
                "required": ["city"],
            },
        },
    }])
}

fn ask_weather() -> Vec<ChatMessage> {
    vec![
        ChatMessage::new(
            "system",
            "You are a helpful assistant. Use the tools when they help.",
        ),
        ChatMessage::new(
            "user",
            "What is the weather in Paris right now? Use the get_weather tool.",
        ),
    ]
}

static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What one request brought back: the streamed visible text and reasoning,
/// and the final result.
struct Turn {
    streamed_text: String,
    streamed_reasoning: String,
    result: InferenceResult,
}

fn run(engine: &BatchEngine, format: &Arc<ChatFormat>, stream: bool, max_tokens: u32) -> Turn {
    run_with(engine, format, stream, stream, max_tokens)
}

/// One request, with the visible text streamed when `text` and the reasoning
/// streamed when `reasoning`.
fn run_with(
    engine: &BatchEngine,
    format: &Arc<ChatFormat>,
    text: bool,
    reasoning: bool,
    max_tokens: u32,
) -> Turn {
    let stream = text;
    let mut config = GenerationConfig {
        temperature: 0.0,
        repeat_penalty: 1.0,
        max_tokens,
        // Each request computes its prompt afresh, so the whole and the
        // streamed reply decode the same numbers.
        cache_salt: Some(format!(
            "{}-{stream}",
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )),
        ..Default::default()
    };
    config
        .stop
        .extend(format.template_result().additional_stops.iter().cloned());
    config.reply_format = Some(format.clone());
    let (token_tx, mut token_rx) = mpsc::channel::<String>(4096);
    let (reasoning_tx, mut reasoning_rx) = mpsc::channel::<String>(4096);
    if reasoning {
        config.reasoning_tx = Some(reasoning_tx);
    } else {
        drop(reasoning_tx);
    }
    let (result_tx, result_rx) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(format.prompt().to_string()),
            config,
            token_tx: stream.then_some(token_tx),
            result_tx,
            media: vec![],
        })
        .expect("submit");
    let text = std::thread::spawn(move || {
        let mut s = String::new();
        while let Some(c) = token_rx.blocking_recv() {
            s.push_str(&c);
        }
        s
    });
    let reasoning = std::thread::spawn(move || {
        let mut s = String::new();
        while let Some(c) = reasoning_rx.blocking_recv() {
            s.push_str(&c);
        }
        s
    });
    let result = result_rx
        .blocking_recv()
        .expect("result")
        .expect("generation");
    Turn {
        streamed_text: text.join().unwrap(),
        streamed_reasoning: reasoning.join().unwrap(),
        result,
    }
}

fn no_markup(what: &str, s: &str) {
    for m in MARKUP {
        assert!(!s.contains(m), "{what} carries {m:?}: {s:?}");
    }
}

fn render(
    source: &ChatTemplateSource,
    messages: &[ChatMessage],
    tools: Option<&Value>,
    thinking: bool,
) -> Arc<ChatFormat> {
    let turn = ChatTurn {
        tools,
        enable_thinking: thinking,
        ..ChatTurn::new(messages)
    };
    Arc::new(ChatFormat::render(source, &turn).expect("render"))
}

fn check_model(backend: &Arc<LlamaBackend>, path: &str) {
    let name = std::path::Path::new(path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let model = LlamaModel::load_from_file(backend, path, &common::params()).expect("load model");
    let source = ChatTemplateSource::from_model(&model).expect("chat template");
    let engine =
        BatchEngine::spawn(name.clone(), model, backend.clone(), 8192, true).expect("engine");
    let tools = tools();

    // 1. A turn offering a tool: the whole reply and the streamed one agree,
    //    and both name the tool with a city argument.
    let messages = ask_weather();
    let format = render(&source, &messages, Some(&tools), true);
    eprintln!(
        "[{name}] chat format: {}",
        format.template_result().chat_format
    );
    let whole = run(&engine, &format, false, 1024);
    let streamed = run(&engine, &format, true, 1024);
    let w = &whole.result;
    eprintln!(
        "[{name}] tool turn: {} tokens, {:.1} tok/s, calls {:?}, reasoning {} chars, text {:?}",
        w.output_tokens,
        w.tokens_per_second,
        w.tool_calls
            .iter()
            .map(|c| (&c.name, c.arguments.to_string()))
            .collect::<Vec<_>>(),
        w.thinking.as_deref().unwrap_or("").len(),
        w.text
    );
    assert_eq!(w.tool_calls.len(), 1, "[{name}] one call expected: {w:?}");
    let call = &w.tool_calls[0];
    assert_eq!(call.name, "get_weather", "[{name}]");
    let city = call
        .arguments
        .get("city")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        city.to_lowercase().contains("paris"),
        "[{name}] city argument: {}",
        call.arguments
    );
    no_markup("whole text", &w.text);
    no_markup("whole reasoning", w.thinking.as_deref().unwrap_or(""));
    let s = &streamed.result;
    let names = |r: &InferenceResult| {
        r.tool_calls
            .iter()
            .map(|c| (c.name.clone(), c.arguments.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        (s.output_tokens, &s.text, &s.thinking, names(s)),
        (w.output_tokens, &w.text, &w.thinking, names(w)),
        "[{name}] streamed and whole replies differ"
    );
    no_markup("streamed text", &streamed.streamed_text);
    no_markup("streamed reasoning", &streamed.streamed_reasoning);
    assert_eq!(
        streamed.streamed_text.trim(),
        s.text.trim(),
        "[{name}] streamed text vs its result"
    );
    assert_eq!(
        streamed.streamed_reasoning.trim(),
        s.thinking.as_deref().unwrap_or("").trim(),
        "[{name}] streamed reasoning vs its result"
    );

    // 2. The tool's answer goes back: the model answers in plain text with
    //    the result, and no call.
    let mut followup = messages.clone();
    followup.push(ChatMessage {
        role: "assistant".into(),
        content: String::new(),
        tool_calls: Some(json!([{
            "id": call.id,
            "type": "function",
            "function": { "name": call.name, "arguments": call.arguments.to_string() },
        }])),
        ..Default::default()
    });
    followup.push(ChatMessage {
        role: "tool".into(),
        content: json!({ "city": "Paris", "temperature_c": 17, "sky": "light rain" }).to_string(),
        name: Some(call.name.clone()),
        tool_call_id: Some(call.id.clone()),
        ..Default::default()
    });
    let format = render(&source, &followup, Some(&tools), true);
    let answer = run(&engine, &format, true, 1024);
    eprintln!(
        "[{name}] answer turn: calls {}, text {:?}",
        answer.result.tool_calls.len(),
        answer.result.text
    );
    assert!(
        answer.result.tool_calls.is_empty(),
        "[{name}] the answer calls again"
    );
    assert!(
        answer.result.text.contains("17"),
        "[{name}] the answer misses the tool result"
    );
    no_markup("answer text", &answer.streamed_text);
    no_markup("answer reasoning", &answer.streamed_reasoning);

    // 3. Reasoning on, no tools: the reasoning and the answer come apart.
    let q = vec![ChatMessage::new(
        "user",
        "What is 17 times 23? Answer with the number.",
    )];
    let format = render(&source, &q, None, true);
    let think = run(&engine, &format, true, 2048);
    let r = think.result.thinking.as_deref().unwrap_or("");
    eprintln!(
        "[{name}] reasoning turn: reasoning {} chars, streamed reasoning {} chars, text {:?}",
        r.len(),
        think.streamed_reasoning.len(),
        think.result.text
    );
    assert!(
        think.result.text.contains("391"),
        "[{name}] wrong or missing answer"
    );
    no_markup("reasoning-turn text", &think.streamed_text);
    no_markup("reasoning-turn reasoning", &think.streamed_reasoning);
    if format.template_result().thinking_end_tag.is_some() {
        assert!(
            !r.is_empty(),
            "[{name}] a reasoning template gave no reasoning"
        );
        assert_eq!(
            think.streamed_reasoning.trim(),
            r.trim(),
            "[{name}] streamed reasoning vs result"
        );
    }
    engine.shutdown();
    std::thread::sleep(Duration::from_millis(200));
}

/// Four slots per engine: the KV cache stays small while later requests
/// still run beside the cached KV of earlier ones.
fn four_slots() {
    // SAFETY: set before any thread of this test process reads the environment.
    unsafe { std::env::set_var("BATCH_MAX_SLOTS", "4") };
}

/// The same request decodes to the same reply however it is delivered: whole,
/// streamed, with only its text or only its reasoning streamed, and in any
/// order on one engine.
fn same_reply_however_delivered(backend: &Arc<LlamaBackend>, path: &str) {
    let name = std::path::Path::new(path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let modes = [
        ("whole", false, false),
        ("streamed", true, true),
        ("whole", false, false),
        ("text only", true, false),
        ("reasoning only", false, true),
        ("streamed", true, true),
    ];
    type Reply = (u32, u32, String, Option<String>, Vec<(String, Value)>);
    let one =
        |engine: &BatchEngine, format: &Arc<ChatFormat>, text: bool, reasoning: bool| -> Reply {
            let r = run_with(engine, format, text, reasoning, 512).result;
            let calls = r
                .tool_calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect();
            (r.input_tokens, r.output_tokens, r.text, r.thinking, calls)
        };
    let mut replies: Vec<(String, Reply)> = vec![];
    // Each mode on an engine of its own, then every mode in turn on one engine.
    for fresh in [true, false] {
        let mut shared: Option<(BatchEngine, Arc<ChatFormat>)> = None;
        for (label, text, reasoning) in modes {
            let new_engine = || {
                let model = LlamaModel::load_from_file(backend, path, &common::params())
                    .expect("load model");
                let source = ChatTemplateSource::from_model(&model).expect("chat template");
                let engine = BatchEngine::spawn(name.clone(), model, backend.clone(), 8192, true)
                    .expect("engine");
                let tools = tools();
                let format = render(&source, &ask_weather(), Some(&tools), true);
                (engine, format)
            };
            let r = if fresh {
                let (engine, format) = new_engine();
                let r = one(&engine, &format, text, reasoning);
                engine.shutdown();
                r
            } else {
                let (engine, format) = shared.get_or_insert_with(new_engine);
                one(engine, format, text, reasoning)
            };
            let label = format!(
                "{} {label}",
                if fresh {
                    "fresh engine"
                } else {
                    "shared engine"
                }
            );
            eprintln!(
                "[{name}] {label}: {} prompt tokens, {} tokens, {} reasoning chars, calls {}",
                r.0,
                r.1,
                r.3.as_deref().unwrap_or("").len(),
                r.4.len()
            );
            replies.push((label, r));
        }
        if let Some((engine, _)) = shared {
            engine.shutdown();
        }
    }
    let first = &replies[0];
    let differing: Vec<&str> = replies[1..]
        .iter()
        .filter(|r| r.1 != first.1)
        .map(|r| r.0.as_str())
        .collect();
    assert!(
        differing.is_empty(),
        "[{name}] differ from the first reply: {differing:?}"
    );
}

/// Submit a whole (not streamed) greedy request and hand back its receiver.
fn submit_whole(
    engine: &BatchEngine,
    format: &ChatFormat,
    salt: String,
    max_tokens: u32,
) -> oneshot::Receiver<praecise_runtime::Result<InferenceResult>> {
    let mut config = GenerationConfig {
        temperature: 0.0,
        repeat_penalty: 1.0,
        max_tokens,
        cache_salt: Some(salt),
        ..Default::default()
    };
    config.stop.extend(format.template_result().additional_stops.iter().cloned());
    let (result_tx, result_rx) = oneshot::channel();
    engine
        .submit(BatchRequest {
            prompt: BatchPrompt::Raw(format.prompt().to_string()),
            config,
            token_tx: None,
            result_tx,
            media: vec![],
        })
        .expect("submit");
    result_rx
}

/// The same request decodes to the same reply when other requests decode in
/// the same batches as it: started together with it, and already running when
/// it arrives.
fn same_reply_beside_other_requests(backend: &Arc<LlamaBackend>, path: &str) {
    let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
    let model = LlamaModel::load_from_file(backend, path, &common::params()).expect("load model");
    let source = ChatTemplateSource::from_model(&model).expect("chat template");
    let engine = BatchEngine::spawn(name.clone(), model, backend.clone(), 8192, true).expect("engine");
    let tools = tools();
    let format = render(&source, &ask_weather(), Some(&tools), true);
    let others: Vec<_> = [
        "Write a long poem about the sea.",
        "Explain how a steam engine works, step by step.",
        "List twenty cities and one fact about each.",
    ]
    .iter()
    .map(|q| render(&source, &[ChatMessage::new("user", *q)], None, true))
    .collect();
    let reply = |r: InferenceResult| (r.output_tokens, r.text, r.thinking);
    let alone = reply(submit_whole(&engine, &format, "alone".into(), 512).blocking_recv().unwrap().unwrap());
    // Started together.
    let rx = submit_whole(&engine, &format, "together".into(), 512);
    let side: Vec<_> = others.iter().enumerate().map(|(i, f)| submit_whole(&engine, f, format!("t{i}"), 384)).collect();
    let together = reply(rx.blocking_recv().unwrap().unwrap());
    for s in side {
        let _ = s.blocking_recv();
    }
    // Arriving while the others are mid-generation.
    let side: Vec<_> = others.iter().enumerate().map(|(i, f)| submit_whole(&engine, f, format!("m{i}"), 384)).collect();
    std::thread::sleep(Duration::from_millis(1500));
    let midway = reply(submit_whole(&engine, &format, "midway".into(), 512).blocking_recv().unwrap().unwrap());
    for s in side {
        let _ = s.blocking_recv();
    }
    engine.shutdown();
    eprintln!("[{name}] beside others: alone {} tokens, started together {} tokens, arriving midway {} tokens", alone.0, together.0, midway.0);
    assert_eq!(together, alone, "[{name}] the reply changed when started together with other requests");
    assert_eq!(midway, alone, "[{name}] the reply changed when other requests were already running");
}

#[test]
#[ignore = "needs chat models: PRAECISE_TEST_CHAT_MODELS"]
fn same_tokens_beside_other_requests() {
    four_slots();
    let paths = std::env::var("PRAECISE_TEST_CHAT_MODELS").expect("PRAECISE_TEST_CHAT_MODELS lists GGUF paths");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    for path in paths.split(';').filter(|p| !p.is_empty()) {
        same_reply_beside_other_requests(&backend, path);
    }
}

#[test]
#[ignore = "needs chat models: PRAECISE_TEST_CHAT_MODELS"]
fn same_tokens_however_delivered() {
    four_slots();
    let paths = std::env::var("PRAECISE_TEST_CHAT_MODELS")
        .expect("PRAECISE_TEST_CHAT_MODELS lists GGUF paths");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    for path in paths.split(';').filter(|p| !p.is_empty()) {
        same_reply_however_delivered(&backend, path);
    }
}

#[test]
#[ignore = "needs chat models: PRAECISE_TEST_CHAT_MODELS"]
fn tool_calls_and_reasoning_from_real_models() {
    four_slots();
    let paths = std::env::var("PRAECISE_TEST_CHAT_MODELS")
        .expect("PRAECISE_TEST_CHAT_MODELS lists GGUF paths");
    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let backend = Arc::new(backend);
    let mut failed = vec![];
    for path in paths.split(';').filter(|p| !p.is_empty()) {
        let b = backend.clone();
        let p = path.to_string();
        if let Err(e) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || check_model(&b, &p)))
        {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            eprintln!("FAILED {path}: {msg}");
            failed.push(path.to_string());
        } else {
            eprintln!("PASSED {path}");
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}
