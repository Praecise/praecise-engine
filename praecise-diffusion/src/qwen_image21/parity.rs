//! Agreement with the reference implementation on fixtures produced by
//! `tests/parity/make_qwen_image21_fixtures.py`.
//!
//! Run with `PRAECISE_QWEN_IMAGE21_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored qwen_image21_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::qwen_image::parity::{assert_close, bin};

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_QWEN_IMAGE21_PARITY").expect("PRAECISE_QWEN_IMAGE21_PARITY names the fixture dir"))
}

pub(crate) fn layout(case: &Value) -> Vec<Segment> {
    case["layout"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| match s.as_array() {
            Some(rc) => Segment::Image { rows: rc[0].as_u64().unwrap() as usize, cols: rc[1].as_u64().unwrap() as usize },
            None => Segment::Text(s.as_u64().unwrap() as usize),
        })
        .collect()
}

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = dir();
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let opts = LoadOptions { precision, cpu_threads: threads, device: None };
    let tf = QwenImage21Transformer::load(&CheckpointFiles::new(d.join("checkpoint")), opts).unwrap();
    let times: Vec<f32> = m["times"].as_array().unwrap().iter().map(|t| t.as_f64().unwrap() as f32).collect();
    for case in m["cases"].as_array().unwrap() {
        let tag = case["tag"].as_str().unwrap();
        let prefix = tf.prefill(&bin(&d, &format!("{tag}_text")), &bin(&d, &format!("{tag}_cond")), &layout(case)).unwrap();
        for (k, &t) in times.iter().enumerate() {
            let out = tf.forward(&prefix, &bin(&d, &format!("{tag}_target{k}")), t).unwrap();
            assert_close(&format!("{tag} t{k}"), &out, &bin(&d, &format!("{tag}_out{k}")), min_cos, max_rel);
        }
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image21_parity_transformer_f32() {
    run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image21_parity_transformer_bf16() {
    run(Precision::Bf16, 0.9999, 2e-2);
}

fn vae_run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = dir();
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let backend = LoadOptions { precision, cpu_threads: threads, device: None }.backend().unwrap();
    let vae = vae::QwenImage21Vae::load(&CheckpointFiles::new(d.join("checkpoint")), &backend, precision, true).unwrap();
    let (lh, lw) = (m["latent"][0].as_u64().unwrap() as usize, m["latent"][1].as_u64().unwrap() as usize);
    let s = m["scale"].as_u64().unwrap() as usize;
    let px = vae.decode(&backend, &bin(&d, "vae_latent"), (lh, lw)).unwrap();
    assert_close("decoded", &px, &bin(&d, "vae_decoded"), min_cos, max_rel);
    let z = vae.encode(&backend, &bin(&d, "vae_pixels"), (lh * s, lw * s)).unwrap();
    assert_close("encoded", &z, &bin(&d, "vae_encoded"), min_cos, max_rel);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image21_parity_vae_f32() {
    vae_run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn qwen_image21_parity_vae_f16() {
    vae_run(Precision::Bf16, 0.9999, 5e-3);
}
