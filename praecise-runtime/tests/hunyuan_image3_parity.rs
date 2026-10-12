//! HunyuanImage 3.0 image path against the checkpoint's own model classes, at a tiny size.
//!
//! The full checkpoint does not fit one desktop GPU, so `tests/parity/make_hunyuan_image3_tiny.py`
//! builds the reference model (the `hunyuan.py` family published with the weights) with a tiny
//! seeded random configuration, saves it as a checkpoint and records one denoising forward over a
//! conditioned and an unconditioned prompt and one latent decode. The checkpoint goes through the
//! converter like the real one (backbone GGUF plus `<name>.image.safetensors`).
//!
//! Checked, each against the largest reference magnitude: the timestep token, the image tokens
//! from the patch embedding, the final layer on the reference hidden states, the velocity of both
//! guidance sequences through the backbone, the decoder's last frame; and the flow schedule
//! against the reference scheduler.
//!
//! The converted checkpoint is not committed: the test is ignored by default and, run with
//! `--ignored`, needs `PRAECISE_HY3_GGUF` (the converted tiny backbone; its image side is read
//! from `<name>.image.safetensors` beside it unless `PRAECISE_HY3_IMAGE` names it) and
//! `PRAECISE_HY3_FIXTURES` (the script's output directory). `PRAECISE_HY3_GPU=1` runs everything
//! on the GPU; `PRAECISE_HY3_TOLERANCE` bounds every difference relative to the reference peak
//! (default 1e-3).
#![cfg(feature = "image-gen")]

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use praecise_diffusion::ggml::{Backend, Device};
use praecise_diffusion::hunyuan_image3::{HunyuanImage3Head, HunyuanImage3Vae};
use praecise_runtime::image_gen::{flow_sigmas, hunyuan_image3_prefill, hunyuan_image3_velocity};

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ints(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .expect("list")
        .iter()
        .map(|t| t.as_i64().expect("integer"))
        .collect()
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().map_or(default, |t| {
        t.parse().unwrap_or_else(|_| panic!("{name} is a number"))
    })
}

/// Largest difference relative to the reference peak, reported under `what`.
fn compare(what: &str, reference: &[f32], engine: &[f32]) -> f32 {
    assert_eq!(engine.len(), reference.len(), "{what}: size");
    let peak = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = reference
        .iter()
        .zip(engine)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let rel = diff / peak.max(1e-6);
    eprintln!("{what}: largest difference {diff:.3e} ({rel:.3e} of the peak {peak:.3})");
    rel
}

#[test]
#[ignore = "needs the converted tiny checkpoint: PRAECISE_HY3_GGUF and PRAECISE_HY3_FIXTURES"]
fn hunyuan_image3_matches_the_reference_model() {
    let gguf =
        std::env::var("PRAECISE_HY3_GGUF").expect("PRAECISE_HY3_GGUF names the converted backbone");
    let image = std::env::var("PRAECISE_HY3_IMAGE").map_or_else(
        |_| {
            PathBuf::from(format!(
                "{}.image.safetensors",
                gguf.strip_suffix(".gguf").unwrap_or(&gguf)
            ))
        },
        PathBuf::from,
    );
    let fixtures = std::env::var("PRAECISE_HY3_FIXTURES")
        .expect("PRAECISE_HY3_FIXTURES names the reference output");
    let dir = Path::new(&fixtures);
    let gpu = std::env::var("PRAECISE_HY3_GPU").as_deref() == Ok("1");
    let tolerance = env_f32("PRAECISE_HY3_TOLERANCE", 1e-3);
    let expected = dir.join("expected.json");
    let exp: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&expected)
            .unwrap_or_else(|e| panic!("{}: {e}", expected.display())),
    )
    .expect("json");
    let cond: Vec<LlamaToken> = ints(&exp["cond"])
        .into_iter()
        .map(|t| LlamaToken(t as i32))
        .collect();
    let uncond: Vec<LlamaToken> = ints(&exp["uncond"])
        .into_iter()
        .map(|t| LlamaToken(t as i32))
        .collect();
    let grid = ints(&exp["grid"]);
    let (h, w) = (grid[0] as usize, grid[1] as usize);
    let vae_grid = ints(&exp["vae_grid"]);
    let (vh, vw) = (vae_grid[0] as usize, vae_grid[1] as usize);
    let t = exp["timestep"].as_f64().expect("timestep") as f32;
    let steps = exp["steps"].as_u64().expect("steps") as usize;
    let shift = exp["shift"].as_f64().expect("shift") as f32;
    let ref_sigmas: Vec<f32> = exp["sigmas"]
        .as_array()
        .expect("sigmas")
        .iter()
        .map(|v| v.as_f64().expect("sigma") as f32)
        .collect();

    let device = if gpu { Device::Auto } else { Device::Cpu };
    let side_backend = || Backend::select_device(device, 8).expect("image backend");
    let head = HunyuanImage3Head::load(&image, side_backend(), true).expect("image head");
    let vae = HunyuanImage3Vae::load(&image, side_backend(), true).expect("latent decoder");

    let mut worst = Vec::new();
    let sig = flow_sigmas(steps, shift);
    let sig_diff = ref_sigmas
        .iter()
        .zip(&sig)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("flow schedule: largest difference {sig_diff:.3e}");
    assert_eq!(sig.len(), ref_sigmas.len(), "flow schedule length");

    let tt = head.timestep_token(t).expect("timestep token");
    worst.push((
        "timestep token",
        compare("timestep token", &f32s(&dir.join("timestep.f32")), &tt),
    ));
    let latent = f32s(&dir.join("latent.f32"));
    let tokens = head
        .embed_latents(&latent, 1, (h, w), t)
        .expect("patch embedding");
    worst.push((
        "image tokens",
        compare("image tokens", &f32s(&dir.join("tokens.f32")), &tokens),
    ));
    let ref_velocity = f32s(&dir.join("velocity.f32"));
    let from_ref_hidden = head
        .predict(&f32s(&dir.join("hidden.f32")), 2, (h, w), t)
        .expect("final layer");
    worst.push((
        "final layer",
        compare(
            "final layer on the reference hidden states",
            &ref_velocity,
            &from_ref_hidden,
        ),
    ));

    let mut backend = LlamaBackend::init().expect("backend");
    if std::env::var("PRAECISE_TEST_LOGS").as_deref() != Ok("1") {
        backend.void_logs();
    }
    // the default offloads every layer when a GPU is present
    let model_params = LlamaModelParams::default().with_n_gpu_layers(if gpu { 1000 } else { 0 });
    let model = LlamaModel::load_from_file(&backend, &gguf, &model_params).expect("load");
    let per_seq = (cond.len() + 1 + h * w) as u32;
    // the reference runs in float32 throughout; so does the backbone's attention here
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2 * per_seq + 16))
        .with_n_batch(2 * per_seq)
        .with_n_ubatch(2 * per_seq)
        .with_n_seq_max(2)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::None)
        .with_type_k(KvCacheType::F32)
        .with_type_v(KvCacheType::F32)
        .with_flash_attention(false);
    let mut ctx = model.new_context(&backend, ctx_params).expect("context");
    let seqs = hunyuan_image3_prefill(&mut ctx, &cond, Some(&uncond)).expect("prefill");
    let velocity = hunyuan_image3_velocity(&mut ctx, &head, cond.len(), seqs, &latent, (h, w), t)
        .expect("velocity");
    worst.push((
        "velocity",
        compare("velocity, both sequences", &ref_velocity, &velocity),
    ));

    let pixels = vae
        .decode(&f32s(&dir.join("vae_latent.f32")), (vh, vw))
        .expect("decode");
    worst.push((
        "pixels",
        compare("decoded pixels", &f32s(&dir.join("pixels.f32")), &pixels),
    ));

    assert!(sig_diff <= 1e-6, "flow schedule differs by {sig_diff:.3e}");
    for (what, rel) in worst {
        assert!(
            rel <= tolerance,
            "{what} differs by {rel:.3e} of the peak (bound {tolerance:.1e})"
        );
    }
}
