//! Agreement of the multi-stream transformer with the reference
//! implementation, on fixtures produced by
//! `tests/parity/make_flux3_fixtures.py` (random checkpoints with the
//! released layout at tiny widths: video, video conditioning, action and
//! action conditioning streams, per-token timesteps, a global vector).
//!
//! Run with `PRAECISE_FLUX3_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored flux3_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_FLUX3_PARITY").expect("PRAECISE_FLUX3_PARITY names the fixture dir"))
}

fn raw(dir: &std::path::Path, name: &str) -> Vec<[u8; 4]> {
    std::fs::read(dir.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]).collect()
}

fn f32s(dir: &std::path::Path, name: &str) -> Vec<f32> {
    raw(dir, name).into_iter().map(f32::from_le_bytes).collect()
}

fn ids(dir: &std::path::Path, name: &str) -> Vec<[i32; 4]> {
    raw(dir, name).into_iter().map(i32::from_le_bytes).collect::<Vec<_>>().chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
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

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    let meta: Value = serde_json::from_slice(&std::fs::read(root().join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from) };
    for (tag, m) in meta.as_object().unwrap() {
        let dir = root().join(tag);
        let tf = Flux3Transformer::load(&[dir.join("model.safetensors")], "dit.", opts).unwrap();
        assert_eq!(tf.config().streams.len(), 4);
        let (ctx, ctx_ids, ctx_t, vector) = (f32s(&dir, "ctx"), ids(&dir, "ctx_ids"), f32s(&dir, "ctx_t"), f32s(&dir, "vector"));
        let text = TextInput { tokens: &ctx, ids: &ctx_ids, timesteps: &ctx_t };
        for (case, keys) in m["cases"].as_object().unwrap() {
            let keys: Vec<&str> = keys.as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
            let data: Vec<(Vec<f32>, Vec<[i32; 4]>, Vec<f32>)> =
                keys.iter().map(|k| (f32s(&dir, k), ids(&dir, &format!("{k}_ids")), f32s(&dir, &format!("{k}_t")))).collect();
            let streams: Vec<StreamInput<'_>> = keys
                .iter()
                .zip(&data)
                .map(|(k, (x, i, t))| StreamInput { name: k.strip_prefix("x_").unwrap(), tokens: x, ids: i, timesteps: t })
                .collect();
            let outs = tf.forward(text, Some(&vector), &streams).unwrap();
            for (k, o) in keys.iter().zip(&outs) {
                let (cos, rel) = agreement(o, &f32s(&dir, &format!("out_{case}_{k}")));
                eprintln!("{tag} {case} {k}: cosine {cos:.6}, max relative error {rel:.6}");
                assert!(cos >= min_cos && rel <= max_rel, "{tag} {case} {k}: cosine {cos}, max relative error {rel}");
            }
        }
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_parity_f32() {
    run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_parity_bf16() {
    run(Precision::Bf16, 0.9999, 2e-2);
}
