//! Agreement with the reference implementation, stage by stage and end to
//! end, on fixtures produced by `tests/parity/make_zimage_fixtures.py`.
//!
//! Run with `PRAECISE_ZIMAGE_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored zimage_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::pipeline::Precision;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_ZIMAGE_PARITY").expect("PRAECISE_ZIMAGE_PARITY names the fixture dir"))
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

/// Full precision unless `PRAECISE_ZIMAGE_PRECISION` names a faster format
/// to measure against the same reference.
fn load() -> ZImage {
    let precision = match std::env::var("PRAECISE_ZIMAGE_PRECISION").as_deref() {
        Ok("bf16") => Precision::Bf16,
        Ok("q8_0") => Precision::Q8_0,
        _ => Precision::F32,
    };
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    // `PRAECISE_ZIMAGE_PARTS=text|image` loads one side only, for checkpoints
    // staged a part at a time.
    let (text, image) = match std::env::var("PRAECISE_ZIMAGE_PARTS").as_deref() {
        Ok("text") => (true, false),
        Ok("image") => (false, true),
        _ => (true, true),
    };
    ZImage::load_parts(&CheckpointFiles::new(dir().join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }, text, image)
        .unwrap()
}

fn request(m: &Value, guidance: f32) -> Request {
    Request {
        prompt: m["prompt"].as_str().unwrap().into(),
        references: Vec::new(),
        width: m["width"].as_u64().unwrap() as u32,
        height: m["height"].as_u64().unwrap() as u32,
        steps: m["steps"].as_u64().unwrap() as u32,
        guidance_scale: guidance,
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

fn shape(req: &Request) -> (usize, usize) {
    ((req.height / 8) as usize, (req.width / 8) as usize)
}

#[test]
#[ignore = "needs the reference fixtures"]
fn zimage_parity_tokens_and_caption_features() {
    let m = meta();
    let p = load();
    let req = request(&m, 0.0);
    let cond = p.tokens(&req.prompt).unwrap();
    assert_eq!(cond, ids(&m, "cond_ids"));
    assert_eq!(p.tokens("").unwrap(), ids(&m, "uncond_ids"));
    assert_close("caption features", &p.encode(&cond).unwrap(), &bin("cap"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn zimage_parity_one_transformer_step() {
    let m = meta();
    let p = load();
    let req = request(&m, 0.0);
    let (x, evals) = p.denoise(&bin("cap"), &[], bin("noise"), shape(&req), 1, 0.0).unwrap();
    assert_eq!(evals, 1);
    assert_close("one step", &x, &bin("one_step"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn zimage_parity_decoder() {
    let m = meta();
    let p = load();
    let req = request(&m, 0.0);
    assert_close("decoder", &p.decode(&bin("noise"), shape(&req)).unwrap(), &bin("decoded"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn zimage_parity_whole_pipeline() {
    let m = meta();
    let mut p = load();
    for run in m["runs"].as_array().unwrap() {
        let name = run[0].as_str().unwrap();
        let guidance = run[1].as_f64().unwrap() as f32;
        let req = request(&m, guidance);
        let (lat, evals) = p.denoise(&bin("cap"), &bin("cap_uncond"), bin("noise"), shape(&req), req.steps as usize, guidance).unwrap();
        assert_eq!(evals, req.steps * if guidance > 0.0 { 2 } else { 1 });
        assert_close(&format!("{name} latents"), &lat, &bin(&format!("{name}_latents")), 0.9999, 5e-3);
        let px = p.decode(&lat, shape(&req)).unwrap();
        // The decoder carries the latents' small differences through two
        // dozen convolutions; at full image sizes that is a few tenths of a
        // percent of the peak.
        assert_close(&format!("{name} pixels"), &px, &bin(&format!("{name}_pixels")), 0.9999, 1e-2);
        if m["has_text_encoder"].as_bool().unwrap_or(true) {
            let img = p.generate_from(&req, bin("noise")).unwrap();
            assert_eq!(img.evaluations, evals);
        }
    }
}
