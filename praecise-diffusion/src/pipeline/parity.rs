//! Agreement with the reference implementation, stage by stage and end to
//! end, on a small random checkpoint produced by
//! `tests/parity/make_fixtures.py`.
//!
//! Run with `PRAECISE_DIFFUSION_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_DIFFUSION_PARITY").expect("PRAECISE_DIFFUSION_PARITY names the fixture dir"))
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

fn load() -> Flux2Klein {
    // Full precision: these tests check the computation against the
    // reference, and the small random checkpoint amplifies rounding across its
    // 27 encoder layers far more than a trained one does. The faster formats
    // are measured on the real checkpoint.
    Flux2Klein::load(&CheckpointFiles::new(dir().join("checkpoint")), LoadOptions { precision: Precision::F32, cpu_threads: 8 })
        .unwrap()
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

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn parity_prompt_tokens_match_the_reference_tokenizer() {
    let p = load();
    let m = meta();
    let (ours, _) = p.tokens(m["prompt"].as_str().unwrap()).unwrap();
    let want: Vec<i32> = m["tokens"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
    assert_eq!(ours, want);
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn parity_prompt_embedding_matches_the_reference_encoder() {
    let p = load();
    let ours = p.encode(meta()["prompt"].as_str().unwrap()).unwrap();
    assert_close("prompt embeds", &ours, &bin("prompt_embeds"), 0.999, 0.03);
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn parity_one_transformer_evaluation_matches_the_reference() {
    let p = load();
    let m = meta();
    let emb = bin("prompt_embeds");
    let noise = bin("noise");
    let cell = p.vae_cfg.scale_factor() as usize * 2;
    let (gh, gw) = (m["height"].as_u64().unwrap() as usize / cell, m["width"].as_u64().unwrap() as usize / cell);
    let mut g = Graph::new(&p.backend).unwrap();
    let io = flux2::build(&mut g, &p.dit_cfg, &p.dit, MAX_PROMPT_TOKENS as i64, (gh * gw) as i64);
    g.finish(&[io.out]).unwrap();
    g.set_i32(io.pos, &flux2::rope_positions(&flux2::positions(MAX_PROMPT_TOKENS, &[(0.0, gh, gw)])));
    g.set_f32(io.freq_factors, &flux2::rope_freq_factors(&p.dit_cfg));
    let sigma = m["dit_sigma"].as_f64().unwrap() as f32;
    g.set_f32(io.t_feat, &flux2::timestep_features(sigma * 1000.0, p.dit_cfg.timestep_guidance_channels as usize));
    g.set_f32(io.img, &noise);
    g.set_f32(io.txt, &emb);
    g.compute().unwrap();
    assert_close("transformer", &g.read_f32(io.out), &bin("dit_out"), 0.999, 0.03);
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn parity_decode_matches_the_reference_autoencoder() {
    let p = load();
    let m = meta();
    let cell = p.vae_cfg.scale_factor() as usize * 2;
    let (h, w) = (m["height"].as_u64().unwrap() as usize, m["width"].as_u64().unwrap() as usize);
    let (gh, gw) = (h / cell, w / cell);
    let lat = p.unpatch(&bin("noise"), gh, gw);
    let mut g = Graph::new(&p.backend).unwrap();
    let io = vae::build_decoder(&mut g, &p.vae_cfg, &p.vae, (2 * gw) as i64, (2 * gh) as i64);
    g.finish(&[io.out]).unwrap();
    g.set_f32(io.latents, &lat);
    g.compute().unwrap();
    assert_close("decode", &g.read_f32(io.out), &bin("decoded"), 0.999, 0.03);
}

fn request(references: Vec<RgbImage>) -> Request {
    let m = meta();
    Request {
        prompt: m["prompt"].as_str().unwrap().into(),
        references,
        width: m["width"].as_u64().unwrap() as u32,
        height: m["height"].as_u64().unwrap() as u32,
        steps: m["steps"].as_u64().unwrap() as u32,
        guidance_scale: 1.0,
        seed: 0,
    }
}

/// PSNR of our 8-bit image against the reference `[H, W, 3]` floats in `[0, 1]`.
fn psnr(ours: &Image, reference_bin: &str) -> f64 {
    let reference: Vec<f32> = bin(reference_bin);
    let ours: Vec<f32> = ours.rgb.iter().map(|v| f32::from(*v) / 255.0).collect();
    assert_eq!(ours.len(), reference.len());
    let mse: f64 = ours.iter().zip(&reference).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / ours.len() as f64;
    10.0 * (1.0 / mse.max(1e-12)).log10()
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn parity_whole_pipeline_matches_the_reference_image() {
    let mut p = load();
    let img = p.run(&request(Vec::new()), bin("noise")).unwrap();
    let db = psnr(&img, "pipeline");
    eprintln!("pipeline: PSNR {db:.2} dB");
    assert!(db >= 35.0, "PSNR {db}");
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn parity_reference_conditioned_generation_matches_the_reference_image() {
    let mut p = load();
    let m = meta();
    let (h, w) = (m["reference"][0].as_u64().unwrap() as u32, m["reference"][1].as_u64().unwrap() as u32);
    let rgb = std::fs::read(dir().join("reference.bin")).unwrap();
    let with_ref = p.run(&request(vec![RgbImage { width: w, height: h, rgb }]), bin("noise")).unwrap();
    let db = psnr(&with_ref, "pipeline_ref");
    eprintln!("pipeline with a reference: PSNR {db:.2} dB");
    assert!(db >= 35.0, "PSNR {db}");
    // The reference must actually change the result.
    assert!(psnr(&with_ref, "pipeline") < db, "conditioning had no effect");
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn every_layer_norm_is_followed_by_its_scale_and_shift() {
    // The backend fuses norm, scale and shift into one kernel only when the
    // three are adjacent in the graph.
    let p = load();
    let mut g = Graph::new(&p.backend).unwrap();
    let io = flux2::build(&mut g, &p.dit_cfg, &p.dit, MAX_PROMPT_TOKENS as i64, 16);
    g.finish(&[io.out]).unwrap();
    let ops = g.ops();
    let norms: Vec<usize> = ops.iter().enumerate().filter(|(_, o)| *o == "NORM").map(|(i, _)| i).collect();
    assert!(!norms.is_empty());
    for i in norms {
        assert_eq!(&ops[i + 1..i + 3], ["MUL", "ADD"], "after node {i}: {:?}", &ops[i.saturating_sub(3)..(i + 4).min(ops.len())]);
    }
}

#[test]
#[ignore = "needs fixtures from tests/parity/make_fixtures.py"]
fn every_group_norm_is_followed_by_its_scale_and_shift() {
    let p = load();
    let mut g = Graph::new(&p.backend).unwrap();
    let io = vae::build_decoder(&mut g, &p.vae_cfg, &p.vae, 8, 8);
    g.finish(&[io.out]).unwrap();
    let ops = g.ops();
    let norms: Vec<usize> = ops.iter().enumerate().filter(|(_, o)| *o == "GROUP_NORM").map(|(i, _)| i).collect();
    assert!(!norms.is_empty());
    for i in norms {
        assert_eq!(&ops[i + 1..i + 3], ["MUL", "ADD"], "after node {i}: {:?}", &ops[i.saturating_sub(3)..(i + 5).min(ops.len())]);
    }
}
