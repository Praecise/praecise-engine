//! Agreement with the reference implementation, stage by stage and end to
//! end, on fixtures produced by `tests/parity/make_cosmos3_fixtures.py` (a
//! small random checkpoint, or the released one over a short clip).
//!
//! Run with `PRAECISE_COSMOS3_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored cosmos3_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_COSMOS3_PARITY").expect("PRAECISE_COSMOS3_PARITY names the fixture dir"))
}

fn meta() -> Value {
    serde_json::from_slice(&std::fs::read(dir().join("meta.json")).unwrap()).unwrap()
}

fn bin(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Full precision unless `PRAECISE_COSMOS3_PRECISION` names a faster format
/// to measure against the same reference.
fn load() -> Cosmos3 {
    let precision = match std::env::var("PRAECISE_COSMOS3_PRECISION").as_deref() {
        Ok("bf16") => Precision::Bf16,
        Ok("q8_0") => Precision::Q8_0,
        _ => Precision::F32,
    };
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    Cosmos3::load(&CheckpointFiles::new(dir().join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap()
}

fn request(m: &Value, image: bool, guided: bool) -> VideoRequest {
    let u = |k: &str| m[k].as_u64().unwrap() as u32;
    let (width, height) = (u("width"), u("height"));
    VideoRequest {
        prompt: m["prompt"].as_str().unwrap().into(),
        negative_prompt: Some(m["negative"].as_str().unwrap().into()),
        image: image.then(|| RgbImage { width, height, rgb: std::fs::read(dir().join("image.bin")).unwrap() }),
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
fn start(p: &Cosmos3, req: &VideoRequest) -> Vec<f32> {
    let (lt, lh, lw) = grid(req);
    let mut x = bin("noise");
    if let Some(img) = &req.image {
        let first = p.encode_image(img).unwrap();
        let plane = lh * lw;
        for (c, chunk) in first.chunks_exact(plane).enumerate() {
            x[c * lt * plane..c * lt * plane + plane].copy_from_slice(chunk);
        }
    }
    x
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_tokens_match_the_reference_tokenizer() {
    let m = meta();
    let p = load();
    let req = request(&m, true, true);
    let (c, u) = Cosmos3::prompts(&req);
    assert_eq!(p.tokens(&c, false).unwrap(), ids(&m, "cond_ids"));
    assert_eq!(p.tokens(&u, false).unwrap(), ids(&m, "uncond_ids"));
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_encoder() {
    let m = meta();
    let p = load();
    let req = request(&m, true, false);
    let ours = p.encode_image(req.image.as_ref().unwrap()).unwrap();
    assert_close("first-frame latent", &ours, &bin("x0_first"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_one_transformer_step() {
    let m = meta();
    let p = load();
    let mut req = request(&m, true, false);
    req.steps = 1;
    let (c, u) = Cosmos3::prompts(&req);
    let (cid, uid) = (p.tokens(&c, false).unwrap(), p.tokens(&u, false).unwrap());
    let (x, evals) = p.denoise(&cid, &uid, start(&p, &req), 1, grid(&req), req.fps, 1, 1.0).unwrap();
    assert_eq!(evals, 1);
    assert_close("one step", &x, &bin("one_step"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_decoder() {
    let m = meta();
    let p = load();
    let req = request(&m, false, false);
    let frames = p.decode(&bin("noise"), grid(&req)).unwrap();
    assert_eq!(frames.len() as u32, req.num_frames * 3 * req.width * req.height);
    assert_close("decoder", &frames, &bin("decoded"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn cosmos3_parity_whole_pipeline() {
    let m = meta();
    let mut p = load();
    for (name, image, guided) in [("i2v", true, true), ("t2v", false, false)] {
        let req = request(&m, image, guided);
        let (c, u) = Cosmos3::prompts(&req);
        let (cid, uid) = (p.tokens(&c, false).unwrap(), p.tokens(&u, false).unwrap());
        let shape = grid(&req);
        let (lat, evals) =
            p.denoise(&cid, &uid, start(&p, &req), usize::from(image), shape, req.fps, req.steps as usize, req.guidance_scale).unwrap();
        assert_eq!(evals, req.steps * if guided { 2 } else { 1 });
        assert_close(&format!("{name} latents"), &lat, &bin(&format!("{name}_latents")), 0.9999, 5e-3);
        let frames = p.decode(&lat, shape).unwrap();
        assert_close(&format!("{name} frames"), &frames, &bin(&format!("{name}_frames")), 0.9999, 5e-3);
        let video = p.generate_from(&req, bin("noise")).unwrap();
        assert_eq!(video.frames, req.num_frames);
        assert_eq!(video.evaluations, evals);
    }
}
