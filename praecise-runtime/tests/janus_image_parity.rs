//! Janus-Pro image generation against the checkpoint's own transformers model.
//!
//! `tests/parity/janus_image/expected.json` (written by
//! `tests/parity/make_janus_image_reference.py`, greedy under guidance) holds the prompt and
//! its unconditioned copy as token ids, the guidance scale and the 576 image tokens the
//! reference chose; `logits.f32` the guided logits of the kept steps and `pixels.f32` the VQ
//! decoder's output for those tokens.
//!
//! The engine replays the reference tokens (so one early near-tie cannot derail the rest) and
//! must give the same guided logits at the kept steps, pick the reference token at (almost)
//! every step, and decode the grid to the same pixels.
//!
//! The weights are not committed: the test is ignored by default and, run with `--ignored`,
//! needs `PRAECISE_JANUS_GGUF` (the language backbone converted from the checkpoint) and
//! `PRAECISE_JANUS_DIR` (the checkpoint directory, for the image head and the VQ decoder).
//! `PRAECISE_JANUS_FIXTURES` points at another fixture directory of the same layout;
//! `PRAECISE_JANUS_GPU=1` runs backbone, head and decoder on the GPU;
//! `PRAECISE_JANUS_TOLERANCE` bounds the largest logit difference relative to the largest
//! reference logit (default 1e-3) and `PRAECISE_JANUS_PIXEL_TOLERANCE` the largest pixel
//! difference (default 2e-2).
#![cfg(feature = "image-gen")]

use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use praecise_diffusion::ggml::{Backend, Device};
use praecise_diffusion::janus::{JanusImageHead, JanusVqDecoder};
use praecise_diffusion::pipeline::Precision;
use praecise_runtime::image_gen::{generate_janus, JanusRequest, TokenSampling};

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ints(v: &serde_json::Value) -> Vec<i64> {
    v.as_array().expect("list").iter().map(|t| t.as_i64().expect("integer")).collect()
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().map_or(default, |t| t.parse().unwrap_or_else(|_| panic!("{name} is a number")))
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "needs the Janus-Pro weights: PRAECISE_JANUS_GGUF and PRAECISE_JANUS_DIR"]
fn janus_image_matches_the_source_checkpoint() {
    let gguf = std::env::var("PRAECISE_JANUS_GGUF").expect("PRAECISE_JANUS_GGUF names the backbone GGUF");
    let ckpt = std::env::var("PRAECISE_JANUS_DIR").expect("PRAECISE_JANUS_DIR names the checkpoint directory");
    let default_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/parity/janus_image"));
    let fixtures = std::env::var("PRAECISE_JANUS_FIXTURES").ok();
    let dir = fixtures.as_deref().map_or(default_dir, Path::new);
    let gpu = std::env::var("PRAECISE_JANUS_GPU").as_deref() == Ok("1");
    let tolerance = env_f32("PRAECISE_JANUS_TOLERANCE", 1e-3);
    let pixel_tolerance = env_f32("PRAECISE_JANUS_PIXEL_TOLERANCE", 2e-2);
    let expected = dir.join("expected.json");
    let exp: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&expected).unwrap_or_else(|e| panic!("{}: {e}", expected.display())),
    )
    .expect("json");
    let cond: Vec<LlamaToken> = ints(&exp["cond"]).into_iter().map(|t| LlamaToken(t as i32)).collect();
    let uncond: Vec<LlamaToken> = ints(&exp["uncond"]).into_iter().map(|t| LlamaToken(t as i32)).collect();
    let tokens: Vec<u32> = ints(&exp["tokens"]).into_iter().map(|t| t as u32).collect();
    let kept: Vec<usize> = ints(&exp["kept_steps"]).into_iter().map(|t| t as usize).collect();
    let codebook = exp["codebook"].as_u64().expect("codebook") as usize;
    let scale = exp["guidance_scale"].as_f64().expect("guidance_scale") as f32;
    let ref_logits = f32s(&dir.join("logits.f32"));
    let ref_pixels = f32s(&dir.join("pixels.f32"));
    assert_eq!(ref_logits.len(), kept.len() * codebook, "logits.f32 size");

    let mut backend = LlamaBackend::init().expect("backend");
    if std::env::var("PRAECISE_TEST_LOGS").as_deref() != Ok("1") {
        backend.void_logs();
    }
    let model_params =
        if gpu { LlamaModelParams::default().with_n_gpu_layers(1000) } else { LlamaModelParams::default() };
    let model = LlamaModel::load_from_file(&backend, &gguf, &model_params).expect("load");
    let n_ctx = (cond.len() + tokens.len() + 16) as u32;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2 * n_ctx))
        .with_n_batch(2 * n_ctx)
        .with_n_ubatch(2 * n_ctx)
        .with_n_seq_max(2)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::None);
    let mut ctx = model.new_context(&backend, ctx_params).expect("context");

    let device = if gpu { Device::Auto } else { Device::Cpu };
    let side_backend = || Backend::select_device(device, 8).expect("image backend");
    let dir_ckpt = Path::new(&ckpt);
    let mut head = JanusImageHead::load(dir_ckpt, side_backend(), Precision::F32, 2).expect("image head");
    let decoder = JanusVqDecoder::load(dir_ckpt, side_backend(), true).expect("VQ decoder");

    let req = JanusRequest {
        cond: &cond,
        uncond: Some(&uncond),
        sampling: TokenSampling { guidance_scale: scale, temperature: 0.0, top_k: 0, top_p: 1.0, seed: 0 },
        forced: Some(&tokens),
        keep_logits: true,
    };
    let image = generate_janus(&mut ctx, &mut head, &decoder, &req).expect("generate");

    let mut worst = 0.0f32;
    for (k, &step) in kept.iter().enumerate() {
        let r = &ref_logits[k * codebook..(k + 1) * codebook];
        let e = &image.logits[step];
        let peak = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let diff = r.iter().zip(e).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let rel = diff / peak.max(1e-6);
        eprintln!("step {step}: largest logit difference {diff:.3e} ({rel:.3e} of the peak {peak:.3})");
        worst = worst.max(rel);
    }
    let agree = image.logits.iter().zip(&tokens).filter(|(l, &t)| argmax(l) == t as usize).count();
    eprintln!("greedy choice agrees on {agree}/{} steps", tokens.len());
    let pix = ref_pixels.iter().zip(&image.pixels).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let mean = ref_pixels.iter().zip(&image.pixels).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>()
        / ref_pixels.len() as f64;
    eprintln!("pixels: largest difference {pix:.3e}, mean {mean:.3e}");
    if let Ok(dump) = std::env::var("PRAECISE_JANUS_DUMP") {
        let bytes: Vec<u8> = image.pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(Path::new(&dump).join("engine_pixels.f32"), bytes).expect("dump");
    }

    assert_eq!(image.pixels.len(), ref_pixels.len(), "pixel count");
    assert!(worst <= tolerance, "guided logits differ by {worst:.3e} of the peak (bound {tolerance:.1e})");
    assert!(agree * 100 >= tokens.len() * 99, "greedy choice agrees on only {agree}/{} steps", tokens.len());
    assert!(pix <= pixel_tolerance, "pixels differ by {pix:.3e} (bound {pixel_tolerance:.1e})");
}
