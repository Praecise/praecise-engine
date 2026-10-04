//! Agreement of the audio-video transformer with the reference
//! implementation, on fixtures produced by
//! `tests/parity/make_ltx2_fixtures.py`.
//!
//! The released transformer does not fit a parity host (22B parameters), so
//! the fixture is a random checkpoint with the released LTX-2.3 layout at tiny
//! widths; it exercises every weight, every modulation path and every rotary
//! table of the real one.
//!
//! Run with `PRAECISE_LTX2_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored ltx2_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_LTX2_PARITY").expect("PRAECISE_LTX2_PARITY names the fixture dir"))
}

fn bin(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
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
    eprintln!("{what}: cosine {cos:.6}, max relative error {rel:.6}");
    assert!(cos >= min_cos && rel <= max_rel, "{what}: cosine {cos}, max relative error {rel}");
}

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    run_from(precision, false, min_cos, max_rel);
}

fn run_from(precision: Precision, single_file: bool, min_cos: f64, max_rel: f64) {
    let m: Value = serde_json::from_slice(&std::fs::read(dir().join("meta.json")).unwrap()).unwrap();
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let opts = LoadOptions { precision, cpu_threads: threads, device: None };
    let tf = if single_file {
        Ltx2Transformer::load_single_file(&dir().join("single.safetensors"), opts).unwrap()
    } else {
        Ltx2Transformer::load(&CheckpointFiles::new(dir().join("checkpoint")), opts).unwrap()
    };
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let s = AvShape { frames: u("frames"), height: u("height"), width: u("width"), audio_frames: u("audio_frames"), fps: m["fps"].as_f64().unwrap() as f32 };
    let (video, audio, text, audio_text) = (bin("video"), bin("audio"), bin("text"), bin("audio_text"));
    for case in m["cases"].as_array().unwrap() {
        let tag = case["tag"].as_str().unwrap();
        let (tv, ta) = (case["video_t"].as_f64().unwrap() as f32, case["audio_t"].as_f64().unwrap() as f32);
        let pass = Pass {
            perturbed_blocks: case["stg_blocks"].as_array().map_or_else(Vec::new, |v| v.iter().map(|b| b.as_u64().unwrap() as usize).collect()),
            isolate_modalities: case["isolate_modalities"].as_bool().unwrap_or(false),
        };
        let (ov, oa) = tf.forward(&video, &audio, &text, &audio_text, s, (tv, ta), &pass).unwrap();
        assert_close(&format!("{tag} video"), &ov, &bin(&format!("out_video_{tag}")), min_cos, max_rel);
        assert_close(&format!("{tag} audio"), &oa, &bin(&format!("out_audio_{tag}")), min_cos, max_rel);
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_transformer_f32() {
    run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_transformer_bf16() {
    run(Precision::Bf16, 0.9999, 2e-2);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_transformer_single_file_f32() {
    run_from(Precision::F32, true, 0.999_999, 1e-4);
}

fn connectors_run(precision: Precision, single_file: bool, min_cos: f64, max_rel: f64) {
    let d = PathBuf::from(std::env::var("PRAECISE_LTX2_CONNECTORS_PARITY").expect("PRAECISE_LTX2_CONNECTORS_PARITY names the fixture dir"));
    let read = |name: &str| -> Vec<f32> { std::fs::read(d.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let c = if single_file {
        connectors::Ltx2Connectors::load_single_file(&d.join("single.safetensors"), opts).unwrap()
    } else {
        connectors::Ltx2Connectors::load(&CheckpointFiles::new(d.join("checkpoint")), opts).unwrap()
    };
    let seq = m["seq_len"].as_u64().unwrap() as usize;
    for n in m["valid"].as_array().unwrap() {
        let n = n.as_u64().unwrap();
        let (v, a) = c.forward(&read(&format!("hidden_{n}")), seq).unwrap();
        assert_close(&format!("{n} valid video"), &v, &read(&format!("out_video_{n}")), min_cos, max_rel);
        assert_close(&format!("{n} valid audio"), &a, &read(&format!("out_audio_{n}")), min_cos, max_rel);
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_connectors_f32() {
    connectors_run(Precision::F32, false, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_connectors_single_file_f32() {
    connectors_run(Precision::F32, true, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_connectors_bf16() {
    connectors_run(Precision::Bf16, false, 0.9999, 2e-2);
}

fn video_decoder_run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = PathBuf::from(std::env::var("PRAECISE_LTX2_VAE_PARITY").expect("PRAECISE_LTX2_VAE_PARITY names the fixture dir"));
    let read = |name: &str| -> Vec<f32> { std::fs::read(d.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let dec = vae::Ltx2VideoDecoder::load_single_file(&d.join("single.safetensors"), opts).unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let frames = u("frames");
    assert_eq!(dec.config().video_frames(frames) as u64, m["out_shape"][1].as_u64().unwrap());
    let px = dec.decode(&read("latent"), frames, u("height"), u("width")).unwrap();
    assert_close("video decoder", &px, &read("video"), min_cos, max_rel);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_video_decoder_f32() {
    video_decoder_run(Precision::F32, 0.999_999, 1e-4);
}

fn video_decoder_tiled_run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = PathBuf::from(std::env::var("PRAECISE_LTX2_VAE_PARITY").expect("PRAECISE_LTX2_VAE_PARITY names the fixture dir"));
    let read = |name: &str| -> Vec<f32> { std::fs::read(d.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let m = &m["tiled"];
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let dec = vae::Ltx2VideoDecoder::load_single_file(&d.join("single.safetensors"), opts).unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let tiling = vae::Tiling {
        spatial: Some(vae::Tile { min: u("min_px"), stride: u("stride_px") }),
        temporal: Some(vae::Tile { min: u("min_frames"), stride: u("stride_frames") }),
    };
    let px = dec.decode_tiled(&read("tiled_latent"), u("frames"), u("height"), u("width"), tiling).unwrap();
    let want = read("tiled_video");
    assert_eq!(px.len(), want.len(), "tiled output size");
    assert_close("tiled video decoder", &px, &want, min_cos, max_rel);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_video_decoder_tiled_f32() {
    video_decoder_tiled_run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_video_decoder_bf16() {
    video_decoder_run(Precision::Bf16, 0.9999, 2e-2);
}

fn audio_decoder_run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = PathBuf::from(std::env::var("PRAECISE_LTX2_AUDIO_VAE_PARITY").expect("PRAECISE_LTX2_AUDIO_VAE_PARITY names the fixture dir"));
    let read = |name: &str| -> Vec<f32> { std::fs::read(d.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let dec = audio_vae::Ltx2AudioDecoder::load_single_file(&d.join("single.safetensors"), opts).unwrap();
    let frames = m["frames"].as_u64().unwrap() as usize;
    assert_eq!(dec.config().spectrogram_frames(frames) as u64, m["out_shape"][1].as_u64().unwrap());
    let mel = dec.decode(&read("packed"), frames).unwrap();
    assert_close("audio decoder", &mel, &read("mel"), min_cos, max_rel);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_audio_decoder_f32() {
    audio_decoder_run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_audio_decoder_bf16() {
    audio_decoder_run(Precision::Bf16, 0.9999, 2e-2);
}

fn vocoder_run(precision: Precision, min_cos: f64, max_rel: f64) {
    let d = PathBuf::from(std::env::var("PRAECISE_LTX2_VOCODER_PARITY").expect("PRAECISE_LTX2_VOCODER_PARITY names the fixture dir"));
    let read = |name: &str| -> Vec<f32> { std::fs::read(d.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let voc = vocoder::Ltx2Vocoder::load_single_file(&d.join("single.safetensors"), opts).unwrap();
    let frames = m["frames"].as_u64().unwrap() as usize;
    assert_eq!(voc.config().samples(frames) as u64, m["out_shape"][1].as_u64().unwrap());
    let wave = voc.synthesize(&read("mel"), frames).unwrap();
    assert_close("vocoder", &wave, &read("wave"), min_cos, max_rel);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_vocoder_f32() {
    vocoder_run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_vocoder_bf16() {
    vocoder_run(Precision::Bf16, 0.999_999, 1e-4);
}
