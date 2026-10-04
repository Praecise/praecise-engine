//! Agreement with the reference implementation, stage by stage and end to
//! end, on a small random checkpoint produced by
//! `tests/parity/make_acestep_fixtures.py`.
//!
//! Run with `PRAECISE_ACESTEP_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored acestep_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_ACESTEP_PARITY").expect("PRAECISE_ACESTEP_PARITY names the fixture dir"))
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

/// Full precision unless `PRAECISE_ACESTEP_PRECISION` names a faster format
/// to measure against the same reference.
fn load() -> AceStep {
    let precision = match std::env::var("PRAECISE_ACESTEP_PRECISION").as_deref() {
        Ok("bf16") => Precision::Bf16,
        Ok("q8_0") => Precision::Q8_0,
        _ => Precision::F32,
    };
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    AceStep::load(&CheckpointFiles::new(dir().join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None })
        .unwrap()
}

fn request(m: &Value, guidance: f32) -> MusicRequest {
    let s = |k: &str| m[k].as_str().unwrap().to_string();
    MusicRequest {
        prompt: s("prompt"),
        lyrics: s("lyrics"),
        language: s("language"),
        duration_secs: m["duration"].as_f64().unwrap() as f32,
        steps: Some(m["steps"].as_u64().unwrap() as u32),
        guidance_scale: Some(guidance),
        shift: Some(m["shift"].as_f64().unwrap() as f32),
        seed: 0,
        bpm: Some(m["bpm"].as_u64().unwrap() as u32),
        keyscale: Some(s("keyscale")),
        timesignature: Some(s("timesignature")),
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

#[test]
#[ignore = "needs the reference fixtures"]
fn acestep_parity_tokens_match_the_reference_tokenizer() {
    let m = meta();
    let p = load();
    let (text, lyrics) = AceStep::format(&request(&m, 1.0));
    assert_eq!(p.tokens(&text, MAX_TEXT_TOKENS).unwrap(), ids(&m, "text_tokens"));
    assert_eq!(p.tokens(&lyrics, MAX_LYRIC_TOKENS).unwrap(), ids(&m, "lyric_tokens"));
}

#[test]
#[ignore = "needs the reference fixtures"]
fn acestep_parity_text_and_condition_encoders() {
    let m = meta();
    let p = load();
    let (th, lh) = p.encode_text(&ids(&m, "text_tokens"), &ids(&m, "lyric_tokens")).unwrap();
    assert_close("text hidden states", &th, &bin("text_hidden"), 0.9999, 2e-3);
    assert_close("lyric embeddings", &lh, &bin("lyric_embeds"), 0.9999, 2e-3);
    // From the reference's own inputs, so this stage is measured alone.
    let enc = p.condition(&bin("text_hidden"), &bin("lyric_embeds")).unwrap();
    assert_close("condition encoder", &enc, &bin("encoder_hidden"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn acestep_parity_one_transformer_step() {
    let m = meta();
    let p = load();
    let t = m["dit_t"].as_f64().unwrap() as f32;
    // One step from t to 0 returns x - t v, so v is recovered exactly.
    let noise = bin("noise");
    let (x, evals) = p.denoise_at(&bin("encoder_hidden"), &noise, t).unwrap();
    assert_eq!(evals, 1);
    let v: Vec<f32> = x.iter().zip(&noise).map(|(x, n)| (n - x) / t).collect();
    assert_close("transformer velocity", &v, &bin("dit_out"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn acestep_parity_decoder() {
    let p = load();
    let audio = p.decode(&bin("noise")).unwrap();
    assert_close("decoder", &audio, &bin("decoded"), 0.9999, 2e-3);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn acestep_parity_whole_pipeline() {
    let m = meta();
    let mut p = load();
    for run in m["runs"].as_array().unwrap() {
        let name = run[0].as_str().unwrap();
        let guidance = run[1].as_f64().unwrap() as f32;
        let req = request(&m, guidance);
        let (text, lyrics) = AceStep::format(&req);
        let (th, lh) = p.encode_text(&p.tokens(&text, MAX_TEXT_TOKENS).unwrap(), &p.tokens(&lyrics, MAX_LYRIC_TOKENS).unwrap()).unwrap();
        let ctx = p.condition(&th, &lh).unwrap();
        let (lat, evals) = p.denoise(&ctx, &bin("noise"), req.steps.unwrap() as usize, req.shift.unwrap(), guidance).unwrap();
        assert_eq!(evals, req.steps.unwrap() * if guidance > 1.0 { 2 } else { 1 });
        assert_close(&format!("{name} latents"), &lat, &bin(&format!("{name}_latents")), 0.9999, 5e-3);
        let audio = p.generate_from(&req, &bin("noise")).unwrap();
        assert_eq!(audio.channels, 2);
        assert_close(&format!("{name} audio"), &audio.samples, &bin(name), 0.9999, 5e-3);
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn acestep_parity_tiled_decode_matches_one_pass() {
    let p = load();
    let a = p.acoustic();
    // Long enough for three tiles.
    let frames = 1200;
    let latents: Vec<f32> = (0..frames * a).map(|i| ((i * 2_654_435_761) % 1009) as f32 / 1009.0 - 0.5).collect();
    let tiled = p.decode(&latents).unwrap();
    let whole = p.decode_once(&latents, frames).unwrap();
    assert_close("tiled decode", &tiled, &whole, 0.999_999, 1e-5);
}
