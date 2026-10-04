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

fn pipeline_run(precision: Precision, latents: (f64, f64), frames: (f64, f64), waveform: (f64, f64)) {
    let d = PathBuf::from(std::env::var("PRAECISE_LTX2_PIPELINE_PARITY").expect("PRAECISE_LTX2_PIPELINE_PARITY names the fixture dir"));
    let read = |name: &str| -> Vec<f32> { std::fs::read(d.join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let list = |k: &str| -> Vec<usize> { m[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect() };
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    let p = pipeline::Ltx2Pipeline::load(&d.join("single.safetensors"), &crate::pipeline::CheckpointFiles::new(d.join("text")), opts).unwrap();
    let req = pipeline::Ltx2Request {
        width: u("width"),
        height: u("height"),
        num_frames: u("frames"),
        fps: m["fps"].as_f64().unwrap() as f32,
        steps: u("steps"),
        stg_blocks: list("stg_blocks"),
        max_sequence_length: u("seq_len"),
        ..Default::default()
    };
    // Packed starting latents as the reference loop received them.
    let (video, audio) = (read("video_start"), read("audio_start"));
    let n = list("latent").iter().product::<usize>();
    let af = u("audio_frames");
    let sig = pipeline::sigmas(req.steps, n);
    let want: Vec<f32> = m["sigmas"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    assert_eq!(sig.len(), want.len());
    for (a, b) in sig.iter().zip(&want) {
        assert!((a - b).abs() < 1e-6, "schedule {sig:?} vs {want:?}");
    }
    let to_u32 = |k: &str| -> Vec<u32> { list(k).into_iter().map(|v| v as u32).collect() };
    let (out, s, _) = p.latents_from(&to_u32("positive"), &to_u32("negative"), video, audio, &req).unwrap();
    assert_eq!(s.audio_frames, af);
    assert_close("final video latents", &out.video, &read("final_video"), latents.0, latents.1);
    assert_close("final audio latents", &out.audio, &read("final_audio"), latents.0, latents.1);
    let (px, wave) = p.decode(&out, s).unwrap();
    // Reference frames are `[T][3][H][W]` in [0, 1].
    let shape = list("frames_shape");
    let (t, hw) = (shape[0], shape[2] * shape[3]);
    assert_eq!(px.len(), t * 3 * hw);
    let mut ours = vec![0.0; px.len()];
    for c in 0..3 {
        for f in 0..t {
            for i in 0..hw {
                ours[(f * 3 + c) * hw + i] = (px[(c * t + f) * hw + i] / 2.0 + 0.5).clamp(0.0, 1.0);
            }
        }
    }
    assert_close("frames", &ours, &read("frames"), frames.0, frames.1);
    assert_close("waveform", &wave, &read("wave"), waveform.0, waveform.1);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_pipeline_f32() {
    pipeline_run(Precision::F32, (0.999_999, 1e-4), (0.999_999, 1e-4), (0.999_999, 1e-3));
}

#[test]
#[ignore = "needs the reference fixtures"]
fn ltx2_parity_pipeline_bf16() {
    // Reduced-precision mel spectrograms move the waveform most: the
    // vocoder's second stage re-analyses its own output through a log-mel.
    pipeline_run(Precision::Bf16, (0.999_99, 1e-2), (0.999_99, 5e-2), (0.999, 0.5));
}

