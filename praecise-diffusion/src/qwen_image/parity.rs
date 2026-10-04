//! Agreement with the reference implementation on fixtures produced by
//! `tests/parity/make_qwen_image_fixtures.py`.
//!
//! Run with `PRAECISE_QWEN_IMAGE_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored qwen_image_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_QWEN_IMAGE_PARITY").expect("PRAECISE_QWEN_IMAGE_PARITY names the fixture dir"))
}

pub(crate) fn bin(dir: &std::path::Path, name: &str) -> Vec<f32> {
    std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Cosine similarity and max error relative to the reference's largest
/// magnitude, asserted against the limits.
pub(crate) fn assert_close(what: &str, ours: &[f32], reference: &[f32], min_cos: f64, max_rel: f64) {
    assert_eq!(ours.len(), reference.len(), "{what}: length");
    let (mut dot, mut na, mut nb, mut maxerr, mut maxref) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (a, b) in ours.iter().zip(reference) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        dot += a * b;
        na += a * a;
        nb += b * b;
        maxerr = maxerr.max((a - b).abs());
        maxref = maxref.max(b.abs());
    }
    let (cos, rel) = (dot / (na.sqrt() * nb.sqrt()), maxerr / maxref);
    eprintln!("{what}: cosine {cos:.6}, max relative error {rel:.6}");
    assert!(cos >= min_cos && rel <= max_rel, "{what}: cosine {cos}, max relative error {rel}");
}

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = dir();
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let tf = QwenImageTransformer::load(&CheckpointFiles::new(d.join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None }).unwrap();
    let text = bin(&d, "text");
    let t = m["t"].as_f64().unwrap() as f32;
    for case in m["cases"].as_array().unwrap() {
        let tag = case["tag"].as_str().unwrap();
        let images: Vec<(usize, usize)> =
            case["images"].as_array().unwrap().iter().map(|s| (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize)).collect();
        let out = tf.forward(&bin(&d, &format!("img_{tag}")), &text, &images, t).unwrap();
        assert_close(tag, &out, &bin(&d, &format!("out_{tag}")), min_cos, max_rel);
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image_parity_transformer_f32() {
    run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image_parity_transformer_bf16() {
    run(Precision::Bf16, 0.9999, 2e-2);
}

fn vae_run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = dir();
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let backend = LoadOptions { precision, cpu_threads: threads, device: None }.backend().unwrap();
    let vae = vae::QwenImageVae::load(&CheckpointFiles::new(d.join("checkpoint")), &backend, precision, true).unwrap();
    let (lh, lw) = (m["latent"][0].as_u64().unwrap() as usize, m["latent"][1].as_u64().unwrap() as usize);
    let s = m["scale"].as_u64().unwrap() as usize;
    let px = vae.decode(&backend, &bin(&d, "vae_latent"), (lh, lw)).unwrap();
    assert_close("decoded", &px, &bin(&d, "vae_decoded"), min_cos, max_rel);
    let z = vae.encode(&backend, &bin(&d, "vae_pixels"), (lh * s, lw * s)).unwrap();
    assert_close("encoded", &z, &bin(&d, "vae_encoded"), min_cos, max_rel);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image_parity_vae_f32() {
    vae_run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image_parity_vae_f16() {
    vae_run(Precision::Bf16, 0.9999, 5e-3);
}
