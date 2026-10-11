//! Serving diffusion language models through the batch engine.
//!
//! `tiny_models_decode_as_the_reference_decoder_does` needs
//! `PRAECISE_DENOISE_FIXTURES`, a directory holding the GGUFs written by
//! `tests/parity/make_denoise_fixtures.py` (with `--block-causal` for a
//! backend that has the llada2 architecture) and `expected.json`, the reference
//! decoder's answers for them (`tests/parity/denoise_reference.py`).
//! `a_served_model_answers` needs `PRAECISE_DENOISE_MODEL`, a real diffusion
//! model, and offloads `PRAECISE_DENOISE_GPU_LAYERS` layers (default all).
//! Both are ignored by default and fail without their inputs.
#![cfg(feature = "bundled-llama")]

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use praecise_runtime::{BatchEngine, BatchPrompt, BatchRequest, ChatMessage, GenerationConfig, InferenceResult, StopReason};
use tokio::sync::{mpsc, oneshot};

fn backend() -> Arc<LlamaBackend> {
    static B: OnceLock<Arc<LlamaBackend>> = OnceLock::new();
    B.get_or_init(|| {
        let mut b = LlamaBackend::init().expect("backend");
        b.void_logs();
        Arc::new(b)
    })
    .clone()
}

fn greedy(max_tokens: u32) -> GenerationConfig {
    GenerationConfig { temperature: 0.0, top_k: Some(1), max_tokens, ..GenerationConfig::default() }
}

type Reply = oneshot::Receiver<praecise_runtime::Result<InferenceResult>>;

fn request(prompt: BatchPrompt, config: GenerationConfig, token_tx: Option<mpsc::Sender<String>>) -> (BatchRequest, Reply) {
    let (result_tx, result_rx) = oneshot::channel();
    (BatchRequest { prompt, config, token_tx, result_tx, media: Vec::new() }, result_rx)
}

fn engine(path: &str, gpu_layers: u32) -> BatchEngine {
    let backend = backend();
    let model = LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default().with_n_gpu_layers(gpu_layers))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(model.is_diffusion(), "{path} is not a diffusion model");
    BatchEngine::spawn("m".into(), model, backend, 4096, false).expect("spawn")
}

fn answer(engine: &BatchEngine, prompt: &str, config: GenerationConfig) -> InferenceResult {
    let (req, rx) = request(BatchPrompt::Raw(prompt.into()), config, None);
    engine.submit(req).expect("submit");
    rx.blocking_recv().expect("reply").expect("answer")
}

const PROMPTS: [&str; 4] = [
    "The lighthouse keeper climbed the stairs and",
    "List three rivers:",
    "x",
    "A much longer prompt that runs past the first block of thirty-two positions, so the decode starts later.",
];

#[test]
#[ignore = "needs PRAECISE_DENOISE_FIXTURES: tiny diffusion GGUFs and the reference decoder's answers"]
fn tiny_models_decode_as_the_reference_decoder_does() {
    let dir = std::env::var("PRAECISE_DENOISE_FIXTURES").expect("PRAECISE_DENOISE_FIXTURES names the fixture directory");
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/expected.json")).expect("expected.json"))
            .expect("json");
    let cases = expected.as_array().expect("a list of cases");
    let mut checked = 0;
    // The block-causal fixture is written only for a backend that has its
    // architecture; the reference decoder has no block-causal mode, so it is
    // held to everything below but the comparison.
    for arch in ["dream", "llada", "llada-moe", "rnd1", "llada2"] {
        let path = format!("{dir}/{arch}-tiny.gguf");
        if arch == "llada2" && !std::path::Path::new(&path).exists() {
            continue;
        }
        let e = engine(&path, 0);

        // 1. Token for token with the reference decoder. It decodes the whole
        // canvas whatever it holds; this engine stops at the first end token,
        // so its answer is the reference's up to there.
        for case in cases.iter().filter(|c| c["arch"] == arch) {
            let prompt = case["prompt"].as_str().expect("prompt");
            let max_tokens = case["max_tokens"].as_u64().expect("max_tokens") as u32;
            let want = case["text"].as_str().expect("text");
            let got = answer(&e, prompt, greedy(max_tokens));
            eprintln!("{arch} {prompt:?}: {} of {} characters, {:?}", got.text.len(), want.len(), got.stop_reason);
            assert!(
                want.starts_with(&got.text),
                "{arch} {prompt:?}: the engine answered {:?}, the reference {want:?}",
                got.text
            );
            if got.stop_reason == StopReason::Length {
                assert_eq!(got.text, want, "{arch} {prompt:?}: a full canvas is the reference's whole answer");
            }
            checked += 1;
        }

        // 2. Batching changes nothing: the same requests decoded together
        // answer exactly as they do one at a time.
        let alone: Vec<String> = PROMPTS.iter().map(|p| answer(&e, p, greedy(40)).text).collect();
        let replies: Vec<Reply> = PROMPTS
            .iter()
            .map(|p| {
                let (req, rx) = request(BatchPrompt::Raw((*p).into()), greedy(40), None);
                e.submit(req).expect("submit");
                rx
            })
            .collect();
        let together: Vec<String> =
            replies.into_iter().map(|rx| rx.blocking_recv().expect("reply").expect("answer").text).collect();
        assert_eq!(alone, together, "{arch}: batched answers differ from single ones");

        // 3. The stream is the answer, and the work is reported.
        let (tx, mut rx) = mpsc::channel(4096);
        let (req, reply) = request(BatchPrompt::Raw(PROMPTS[0].into()), greedy(64), Some(tx));
        e.submit(req).expect("submit");
        let result = reply.blocking_recv().expect("reply").expect("answer");
        let mut streamed = String::new();
        while let Ok(chunk) = rx.try_recv() {
            streamed.push_str(&chunk);
        }
        assert_eq!(streamed, result.text, "{arch}: the stream differs from the answer");
        assert!(result.output_tokens <= 64);
        let steps = result.denoise_steps.expect("denoise steps");
        let forward = result.forward_positions.expect("forward positions");
        assert!(steps >= 1 && forward >= u64::from(result.input_tokens + result.output_tokens), "{arch}: {result:?}");

        // 4. What cannot be served is refused with the reason.
        let (mut req, rx) = request(BatchPrompt::Raw("look".into()), greedy(8), None);
        req.media.push(vec![0u8; 4]);
        e.submit(req).expect("submit");
        let err = rx.blocking_recv().expect("reply").expect_err("media refused");
        assert!(err.to_string().contains("text only"), "{err}");
        let long = "y".repeat(5000);
        let (req, rx) = request(BatchPrompt::Raw(long), greedy(8), None);
        e.submit(req).expect("submit");
        let err = rx.blocking_recv().expect("reply").expect_err("too long refused");
        assert!(err.to_string().contains("leaves no room"), "{err}");
        assert!(e.fingerprint().is_none());
        e.shutdown();
    }
    assert!(checked >= 8, "expected.json held {checked} cases");
}

#[test]
#[ignore = "needs PRAECISE_DENOISE_MODEL: a real diffusion language model"]
fn a_served_model_answers() {
    let path = std::env::var("PRAECISE_DENOISE_MODEL").expect("PRAECISE_DENOISE_MODEL names a diffusion GGUF");
    let layers = std::env::var("PRAECISE_DENOISE_GPU_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(999);
    let e = engine(&path, layers);
    let chat = |q: &str| BatchPrompt::Chat(vec![ChatMessage::new("user", q)]);
    let questions = [
        ("What is the capital of France? Answer with one word.", "Paris"),
        ("What is 12 plus 30? Answer with the number only.", "42"),
        ("Name the largest planet in the solar system in one word.", "Jupiter"),
    ];
    // One at a time, then all together.
    for (q, want) in questions {
        let started = Instant::now();
        let (req, rx) = request(chat(q), greedy(64), None);
        e.submit(req).expect("submit");
        let r = rx.blocking_recv().expect("reply").expect("answer");
        eprintln!(
            "{q:?} -> {:?} ({} output tokens, {} steps, {} forward positions, {:.0} ms)",
            r.text,
            r.output_tokens,
            r.denoise_steps.unwrap_or(0),
            r.forward_positions.unwrap_or(0),
            started.elapsed().as_secs_f64() * 1e3
        );
        assert!(r.text.contains(want), "{q:?}: {:?}", r.text);
    }
    let started = Instant::now();
    let replies: Vec<(&str, Reply)> = questions
        .iter()
        .map(|(q, want)| {
            let (req, rx) = request(chat(q), greedy(64), None);
            e.submit(req).expect("submit");
            (*want, rx)
        })
        .collect();
    for (want, rx) in replies {
        let r = rx.blocking_recv().expect("reply").expect("answer");
        assert!(r.text.contains(want), "batched: {:?}", r.text);
    }
    eprintln!("three together: {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
    e.shutdown();
}
