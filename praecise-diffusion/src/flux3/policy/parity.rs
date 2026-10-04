//! Agreement of the policy's packing and chunk sampler with the reference, on
//! fixtures from `tests/parity/make_flux3_policy_fixtures.py` (a frame and a
//! history configuration with tiny random transformers).
//!
//! Run with `PRAECISE_FLUX3_POLICY_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored flux3_policy_parity`.

use std::path::{Path, PathBuf};

use super::*;
use crate::pipeline::Precision;

fn f32s(dir: &Path, name: &str) -> Vec<f32> {
    std::fs::read(dir.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn ids(dir: &Path, name: &str) -> Vec<[i32; 4]> {
    let v: Vec<i32> =
        std::fs::read(dir.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    v.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

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

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    let root = PathBuf::from(std::env::var("PRAECISE_FLUX3_POLICY_PARITY").expect("PRAECISE_FLUX3_POLICY_PARITY names the fixture dir"));
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    for tag in ["frame", "history"] {
        let dir = root.join(tag);
        let json: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        let cfg = PolicyConfig::from_json(&json).unwrap();

        // Positions of the predicted streams.
        let time_ids: Vec<i32> = ids_flat(&dir, "video_ids");
        assert_eq!(cfg.predicted_video_time_ids(), time_ids, "{tag}: predicted video times");
        let action_ids: Vec<i32> = ids_flat(&dir, "action_ids");
        let ours: Vec<i32> = packing::sequence_ids(&cfg.action_times(), 0).iter().map(|i| i[0]).collect();
        assert_eq!(ours, action_ids, "{tag}: action times");

        // Conditioning packing.
        let past = (dir.join("past.bin").exists()).then(|| f32s(&dir, "past"));
        let (action, action_pos) = pack_actions(&cfg, &f32s(&dir, "states"), past.as_deref()).unwrap();
        let (cos, rel) = agreement(&action, &f32s(&dir, "action_cond"));
        assert!(cos > 0.999_999 && rel < 1e-6, "{tag}: action conditioning {cos} {rel}");
        assert_eq!(action_pos, ids(&dir, "action_cond_ids"), "{tag}: action conditioning ids");
        let full: Vec<usize> = serde_json::from_value(meta["lat_full"].clone()).unwrap();
        let picks: Vec<usize> = serde_json::from_value(meta["picks"].clone()).unwrap();
        let latents: Vec<(Vec<f32>, [usize; 3])> = (0..picks.len()).map(|k| (f32s(&dir, &format!("lat{k}")), [1, full[0], full[1]])).collect();
        let seconds: Vec<f32> = picks
            .iter()
            .map(|&i| match cfg.conditioning {
                Conditioning::Frame => 0.0,
                Conditioning::History => i as f32 / cfg.video_position_fps.unwrap_or(cfg.fps),
            })
            .collect();
        let (video, video_pos) = pack_video(&cfg, &latents, &seconds);
        assert_eq!(video, f32s(&dir, "video_cond"), "{tag}: video conditioning");
        assert_eq!(video_pos, ids(&dir, "video_cond_ids"), "{tag}: video conditioning ids");

        // The chunk sampler.
        let dit = Flux3Transformer::load(&[dir.join("model.safetensors")], "dit.", opts).unwrap();
        let cond = Conditions { video, video_ids: video_pos, action, action_ids: action_pos };
        let noise = Noise { video: f32s(&dir, "noise_video"), action: f32s(&dir, "noise_action") };
        let chunk = sample_chunk(&dit, &cfg, &cond, &f32s(&dir, "ctx_c"), Some(&f32s(&dir, "ctx_uc")), &noise).unwrap();
        let (cos, rel) = agreement(&chunk, &f32s(&dir, "chunk"));
        eprintln!("{tag} {precision:?}: chunk cosine {cos:.6}, max relative error {rel:.2e}");
        assert!(cos >= min_cos && rel <= max_rel, "{tag}: chunk cosine {cos}, max relative error {rel}");
    }
}

fn ids_flat(dir: &Path, name: &str) -> Vec<i32> {
    std::fs::read(dir.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
#[ignore = "needs PRAECISE_FLUX3_POLICY_PARITY fixtures"]
fn flux3_policy_parity_f32() {
    run(Precision::F32, 0.999_99, 1e-3);
}

#[test]
#[ignore = "needs PRAECISE_FLUX3_POLICY_PARITY fixtures"]
fn flux3_policy_parity_bf16() {
    run(Precision::Bf16, 0.999, 5e-2);
}
