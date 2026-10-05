//! Agreement with the reference implementation, stage by stage and end to
//! end, on fixtures produced by `tests/parity/make_cosmos3_fixtures.py` (a
//! small random checkpoint, or the released one over a short clip).
//!
//! Run with `PRAECISE_COSMOS3_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored cosmos3_parity`. The older checkpoint
//! layout (Nano, Super) is checked from `PRAECISE_COSMOS3_OLDER_PARITY`, a
//! directory holding one fixture directory per layout (`nano`, `super`).

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_COSMOS3_PARITY").expect("PRAECISE_COSMOS3_PARITY names the fixture dir"))
}

fn meta(d: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap()
}

fn bin(d: &Path, name: &str) -> Vec<f32> {
    std::fs::read(d.join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Full precision unless `PRAECISE_COSMOS3_PRECISION` names a faster format
/// to measure against the same reference.
fn load(d: &Path) -> Cosmos3 {
    let precision = match std::env::var("PRAECISE_COSMOS3_PRECISION").as_deref() {
        Ok("bf16") => Precision::Bf16,
        Ok("q8_0") => Precision::Q8_0,
        _ => Precision::F32,
    };
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    Cosmos3::load(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap()
}

fn request(d: &Path, m: &Value, image: bool, guided: bool) -> VideoRequest {
    let u = |k: &str| m[k].as_u64().unwrap() as u32;
    let (width, height) = (u("width"), u("height"));
    VideoRequest {
        prompt: m["prompt"].as_str().unwrap().into(),
        negative_prompt: Some(m["negative"].as_str().unwrap().into()),
        image: image.then(|| RgbImage { width, height, rgb: std::fs::read(d.join("image.bin")).unwrap() }),
        width,
        height,
        num_frames: u("frames"),
        fps: m["fps"].as_f64().unwrap() as f32,
        steps: u("steps"),
        guidance_scale: if guided { m["guidance"].as_f64().unwrap() as f32 } else { 1.0 },
        seed: 0,
    }
}

fn ids(m: &Value, k: &str) -> Vec<i32> {
    m[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect()
}

/// Cosine similarity and max error relative to the reference's largest
/// magnitude.
fn agreement(ours: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(ours.len(), reference.len(), "length");
    let (mut dot, mut na, mut nb, mut maxerr, mut maxref) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (a, b) in ours.iter().zip(reference) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        dot += a * b;
        na += a * a;
        nb += b * b;
        maxerr = maxerr.max((a - b).abs());
        maxref = maxref.max(b.abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr / maxref)
}

fn assert_close(what: &str, ours: &[f32], reference: &[f32], min_cos: f64, max_rel: f64) {
    let (cos, rel) = agreement(ours, reference);
    eprintln!("{what}: cosine {cos:.6}, max relative error {rel:.5}");
    assert!(cos >= min_cos && rel <= max_rel, "{what}: cosine {cos}, max relative error {rel}");
}

/// The reference noise with this implementation's encoding of the first
/// frame in place.
fn start(d: &Path, p: &Cosmos3, req: &VideoRequest) -> Vec<f32> {
    let (lt, lh, lw) = grid(req);
    let mut x = bin(d, "noise");
    if let Some(img) = &req.image {
        let first = p.encode_image(img).unwrap();
        let plane = lh * lw;
        for (c, chunk) in first.chunks_exact(plane).enumerate() {
            x[c * lt * plane..c * lt * plane + plane].copy_from_slice(chunk);
        }
    }
    x
}

fn tokens_match(d: &Path) {
    let m = meta(d);
    let p = load(d);
    let req = request(d, &m, true, true);
    let (c, u) = Cosmos3::prompts(&req);
    assert_eq!(p.tokens(&c, false).unwrap(), ids(&m, "cond_ids"));
    assert_eq!(p.tokens(&u, false).unwrap(), ids(&m, "uncond_ids"));
}

fn one_step(d: &Path) {
    let m = meta(d);
    let p = load(d);
    let mut req = request(d, &m, true, false);
    req.steps = 1;
    let (c, u) = Cosmos3::prompts(&req);
    let (cid, uid) = (p.tokens(&c, false).unwrap(), p.tokens(&u, false).unwrap());
    let (x, evals) = p.denoise(&cid, &uid, start(d, &p, &req), 1, grid(&req), req.fps, 1, 1.0).unwrap();
    assert_eq!(evals, 1);
    assert_close("one step", &x, &bin(d, "one_step"), 0.9999, 2e-3);
}

fn whole_pipeline(d: &Path) {
    let m = meta(d);
    let mut p = load(d);
    for (name, image, guided) in [("i2v", true, true), ("t2v", false, false)] {
        let req = request(d, &m, image, guided);
        let (c, u) = Cosmos3::prompts(&req);
        let (cid, uid) = (p.tokens(&c, false).unwrap(), p.tokens(&u, false).unwrap());
        let shape = grid(&req);
        let (lat, evals) = p
            .denoise(&cid, &uid, start(d, &p, &req), usize::from(image), shape, req.fps, req.steps as usize, req.guidance_scale)
            .unwrap();
        assert_eq!(evals, req.steps * if guided { 2 } else { 1 });
        assert_close(&format!("{name} latents"), &lat, &bin(d, &format!("{name}_latents")), 0.9999, 5e-3);
        let frames = p.decode(&lat, shape).unwrap();
        assert_close(&format!("{name} frames"), &frames, &bin(d, &format!("{name}_frames")), 0.9999, 5e-3);
        let video = p.generate_from(&req, bin(d, "noise")).unwrap();
        assert_eq!(video.frames, req.num_frames);
        assert_eq!(video.evaluations, evals);
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_tokens_match_the_reference_tokenizer() {
    tokens_match(&dir());
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_encoder() {
    let d = dir();
    let m = meta(&d);
    let p = load(&d);
    let req = request(&d, &m, true, false);
    let ours = p.encode_image(req.image.as_ref().unwrap()).unwrap();
    assert_close("first-frame latent", &ours, &bin(&d, "x0_first"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_one_transformer_step() {
    one_step(&dir());
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_decoder() {
    let d = dir();
    let m = meta(&d);
    let p = load(&d);
    let req = request(&d, &m, false, false);
    let frames = p.decode(&bin(&d, "noise"), grid(&req)).unwrap();
    assert_eq!(frames.len() as u32, req.num_frames * 3 * req.width * req.height);
    assert_close("decoder", &frames, &bin(&d, "decoded"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_whole_pipeline() {
    whole_pipeline(&dir());
}

/// The Nano and Super layouts: small random checkpoints written in the
/// older repository layout, end to end against the reference.
#[test]
#[ignore = "needs PRAECISE_COSMOS3_OLDER_PARITY fixtures"]
fn cosmos3_older_layout_parity() {
    let root = PathBuf::from(std::env::var("PRAECISE_COSMOS3_OLDER_PARITY").expect("PRAECISE_COSMOS3_OLDER_PARITY names the fixture dir"));
    for layout in ["nano", "super"] {
        let d = root.join(layout);
        let index: Value = serde_json::from_slice(&std::fs::read(d.join("checkpoint/model_index.json")).unwrap()).unwrap();
        assert_eq!(index["_class_name"], PIPELINE_CLASSES[1], "{layout}: not the older layout");
        eprintln!("{layout}:");
        tokens_match(&d);
        one_step(&d);
        whole_pipeline(&d);
    }
}
