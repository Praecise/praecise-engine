//! Agreement of the image transformer and time grid with the reference
//! implementation, on fixtures produced by
//! `tests/parity/make_flux3_image_fixtures.py` (random diffusers image
//! transformers at tiny widths, in the release's tensor names, with and
//! without a reference image, as a flow map and not).
//!
//! Run with `PRAECISE_FLUX3_IMAGE_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored flux3_image`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "tensor and kernel math: sizes, indices and counts move between usize, i64, u32 and f32 within bounds the tensor shapes fix; a checked conversion at each site would change no value"
)]

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::flux3::parity::{agreement, f32s, ids};
use crate::pipeline::{CheckpointFiles, Precision, RgbImage};

fn root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_FLUX3_IMAGE_PARITY").expect("PRAECISE_FLUX3_IMAGE_PARITY names the fixture dir"))
}

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    let meta: Value = serde_json::from_slice(&std::fs::read(root().join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    for (tag, m) in meta.as_object().unwrap() {
        let dir = root().join(tag);
        let tf = Flux3Transformer::load(&[dir.join("model.safetensors")], "", opts).unwrap().with_spatial_rope_scale(IMAGE_ROPE_SCALE);
        assert_eq!(tf.config().flow_map, m["flow_map"].as_bool().unwrap(), "{tag}");
        let (img, img_ids, reference, ref_ids) = (f32s(&dir, "img"), ids(&dir, "img_ids"), f32s(&dir, "ref"), ids(&dir, "ref_ids"));
        let (ctx, ctx_ids) = (f32s(&dir, "ctx"), ids(&dir, "ctx_ids"));
        let ctx_t = vec![0f32; ctx_ids.len()];
        let text = TextInput { tokens: &ctx, ids: &ctx_ids, timesteps: &ctx_t };
        let img_t = vec![m["t"].as_f64().unwrap() as f32; img_ids.len()];
        let ref_t = vec![0f32; ref_ids.len()];
        let target = m["target"].as_f64().map(|v| v as f32);
        assert!(tf.forward_to(text, None, &[], if target.is_some() { None } else { Some(0.5) }).is_err(), "{tag}: a target time against the model");
        for case in ["ref", "noref"] {
            let mut streams = vec![StreamInput { name: "image", tokens: &img, ids: &img_ids, timesteps: &img_t }];
            if case == "ref" {
                streams.push(StreamInput { name: "image_cond", tokens: &reference, ids: &ref_ids, timesteps: &ref_t });
            }
            let out = tf.forward_to(text, None, &streams, target).unwrap();
            let (cos, rel) = agreement(&out[0], &f32s(&dir, &format!("out_{case}")));
            eprintln!("{tag} {case}: cosine {cos:.6}, max relative error {rel:.6}");
            assert!(cos >= min_cos && rel <= max_rel, "{tag} {case}: cosine {cos}, max relative error {rel}");
        }
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_image_parity_f32() {
    run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_image_parity_bf16() {
    run(Precision::Bf16, 0.9999, 2e-2);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_image_time_grid_is_the_reference_schedulers() {
    for steps in [DISTILLED_STEPS, BASE_STEPS] {
        let ours = euler_times(steps as usize, SCHEDULE_SHIFT);
        let reference = f32s(&root(), &format!("sigmas_{steps}"));
        assert_eq!(ours.len(), reference.len(), "{steps} steps");
        let worst = ours.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-6, "{steps} steps: grids differ by {worst}");
    }
}

/// The release itself against BFL's reference code. Needs the gated weights
/// (`PRAECISE_FLUX3_RELEASE`: a directory with `model.safetensors`, `vae/` and
/// `text_encoder/`) and fixtures from
/// `tests/parity/make_flux3_image_release_fixtures.py`
/// (`PRAECISE_FLUX3_RELEASE_FIXTURES`). Without either it prints why and
/// passes; the weights go in `D:\work\models\flux-3-image-klein` on the
/// Windows PC and `~/models/flux-3-image-klein` on the spark.
#[test]
fn flux3_image_release_parity() {
    let (Ok(weights), Ok(fixtures)) = (std::env::var("PRAECISE_FLUX3_RELEASE"), std::env::var("PRAECISE_FLUX3_RELEASE_FIXTURES")) else {
        eprintln!(
            "SKIPPED flux3_image_release_parity: set PRAECISE_FLUX3_RELEASE to the downloaded flux-3-image-klein directory \
             (model.safetensors, vae/, text_encoder/) and PRAECISE_FLUX3_RELEASE_FIXTURES to the output of \
             tests/parity/make_flux3_image_release_fixtures.py; the release weights are gated and not on this machine"
        );
        return;
    };
    let opts = LoadOptions { precision: Precision::Bf16, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let pipeline = Flux3Image::load_release(&CheckpointFiles::new(&weights), opts).expect("release loads");
    eprintln!("loaded: {pipeline:?}");
    let mut cases: Vec<_> = std::fs::read_dir(&fixtures).unwrap().flatten().filter(|e| e.path().join("meta.json").is_file()).collect();
    cases.sort_by_key(std::fs::DirEntry::file_name);
    assert!(!cases.is_empty(), "no cases in {fixtures}");
    for case in cases {
        let (dir, tag) = (case.path(), case.file_name().to_string_lossy().into_owned());
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        let (width, height) = (meta["width"].as_u64().unwrap() as u32, meta["height"].as_u64().unwrap() as u32);
        assert_eq!(meta["model"] == "distilled", pipeline.is_distilled(), "{tag}: fixtures and weights are different models");

        // Reference preparation, byte for byte.
        let sizes = meta["ref_sizes"].as_array().unwrap();
        let mut references = Vec::new();
        for (i, s) in sizes.iter().enumerate() {
            let n = |k: usize| s[k].as_u64().unwrap() as u32;
            let orig = RgbImage { width: n(0), height: n(1), rgb: std::fs::read(dir.join(format!("ref{i}_orig.rgb"))).unwrap() };
            let want = std::fs::read(dir.join(format!("ref{i}_prepared.rgb"))).unwrap();
            let got = if sizes.len() > 1 { prepare::prepare_multi_reference(&orig).unwrap() } else { prepare::prepare_reference(&orig).unwrap().0 };
            assert_eq!((got.width, got.height), (n(2), n(3)), "{tag} ref {i}: prepared size");
            let worst = got.rgb.iter().zip(&want).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
            eprintln!("{tag} ref {i}: prepared image differs by at most {worst} levels");
            assert!(worst <= 1, "{tag} ref {i}: preparation differs by {worst} levels");
            references.push(got);
        }

        // Text context.
        let prompt = meta["prompt"].as_str().unwrap();
        let (ctx, _) = pipeline.te.encode(prompt, None).unwrap();
        let (cos, rel) = agreement(&ctx, &f32s(&dir, "ctx"));
        eprintln!("{tag}: text context cosine {cos:.6}, max relative error {rel:.4}");
        assert!(cos >= 0.999, "{tag}: text context cosine {cos}");

        // Denoised latents from the reference's own noise, then the decoded image.
        let req = Request {
            prompt: prompt.into(),
            references,
            width,
            height,
            steps: meta["steps"].as_u64().unwrap() as u32,
            guidance_scale: meta["guidance"].as_f64().map_or(1.0, |g| g as f32),
            seed: meta["seed"].as_u64().unwrap(),
        };
        let latents = pipeline.denoise(&req, f32s(&dir, "noise")).unwrap();
        let (cos, rel) = agreement(&latents.tokens, &f32s(&dir, "out"));
        eprintln!("{tag}: denoised latents cosine {cos:.6}, max relative error {rel:.4}");
        assert!(cos >= 0.98, "{tag}: denoised latents cosine {cos}");
        let cell = pipeline.ae.cell() as usize;
        let rgb = pipeline.ae.decode(&pipeline.backend, &latents.tokens, height as usize / cell, width as usize / cell).unwrap();
        let want = std::fs::read(dir.join("out.rgb")).unwrap();
        let mean = rgb.iter().zip(&want).map(|(&a, &b)| f64::from(a.abs_diff(b))).sum::<f64>() / want.len() as f64;
        eprintln!("{tag}: decoded image differs from the reference by {mean:.3} levels on average");
        assert!(mean < 4.0, "{tag}: decoded image differs by {mean} levels");
    }
}
