//! Pooled embeddings of text, pictures and video against reference vectors
//! from the source checkpoint's own pipeline (`reference/embed_media.py`).
//! Ignored by default; run with `--ignored`, it needs `PRAECISE_EMBED_GGUF`
//! (a vision-language embedder GGUF), `PRAECISE_EMBED_MMPROJ` (its projector)
//! and `PRAECISE_EMBED_MEDIA` (the reference output directory), and fails
//! without them. `PRAECISE_EMBED_CASES` (comma-separated case names) runs a
//! subset. Every case, a 1280x960 picture (1222 tokens) and a 64-frame video
//! (4780 tokens) among them, must reach a cosine of [`BAR`].
#![cfg(feature = "mtmd")]

use std::path::Path;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
use praecise_runtime::embed::{embed, embed_media, EmbedOptions};
use praecise_runtime::media::{MediaInput, Picture};

/// Least cosine to the reference vector, for every case.
const BAR: f64 = 0.9999;

fn input(dir: &Path, case: &serde_json::Value) -> MediaInput {
    let pictures = case["pictures"]
        .as_array()
        .expect("pictures")
        .iter()
        .map(|p| Picture {
            width: u32::try_from(p["width"].as_u64().expect("width")).expect("width"),
            height: u32::try_from(p["height"].as_u64().expect("height")).expect("height"),
            rgb: std::fs::read(dir.join(p["file"].as_str().expect("file"))).expect("picture"),
            video_frame: p["video_frame"].as_bool().expect("video_frame"),
        })
        .collect();
    // The empty case goes in empty: the engine writes the placeholder.
    let text = if case["name"] == "empty" { String::new() } else { case["text"].as_str().expect("text").to_owned() };
    MediaInput { text, pictures }
}

fn cosine(a: &[f32], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * y).sum();
    let na = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "needs the embedder weights and reference: PRAECISE_EMBED_GGUF, PRAECISE_EMBED_MMPROJ and PRAECISE_EMBED_MEDIA"]
fn pooled_embeddings_match_the_source_checkpoint() {
    let (Ok(gguf), Ok(mmproj), Ok(dir)) = (
        std::env::var("PRAECISE_EMBED_GGUF"),
        std::env::var("PRAECISE_EMBED_MMPROJ"),
        std::env::var("PRAECISE_EMBED_MEDIA"),
    ) else {
        panic!("PRAECISE_EMBED_GGUF, PRAECISE_EMBED_MMPROJ and PRAECISE_EMBED_MEDIA must name the embedder, its projector and the reference");
    };
    let only: Option<Vec<String>> = std::env::var("PRAECISE_EMBED_CASES").ok().map(|s| s.split(',').map(str::to_owned).collect());
    let dir = Path::new(&dir);
    let exp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).expect("expected")).expect("json");

    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let model = LlamaModel::load_from_file(&backend, &gguf, &LlamaModelParams::default()).expect("load");
    let params = MtmdContextParams { use_gpu: false, image_min_tokens: 4, ..MtmdContextParams::default() };
    let projector = unsafe { MtmdContext::init_from_file(&mmproj, &model, &params) }.expect("projector");
    let options = EmbedOptions { n_ctx: 8192, ..EmbedOptions::default() };

    let mut worst = 1.0f64;
    let mut compared = 0usize;
    for case in exp["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        if only.as_ref().is_some_and(|o| !o.iter().any(|n| n == name)) {
            continue;
        }
        let want: Vec<f64> = case["embedding"].as_array().expect("embedding").iter().map(|v| v.as_f64().expect("f64")).collect();
        let instruction = case["instruction"].as_str().expect("instruction");
        let inp = input(dir, case);
        let got = embed_media(&backend, &model, &projector, instruction, std::slice::from_ref(&inp), options).expect("embed").remove(0);
        let c = cosine(&got.vector, &want);
        println!("{name}: tokens {} (reference {}), cosine {c:.7}", got.tokens, case["tokens"]);
        assert_eq!(got.tokens as u64, case["tokens"].as_u64().expect("tokens"), "{name}: token count");
        assert!(c >= BAR, "{name}: cosine {c}");
        worst = worst.min(c);
        compared += 1;
        if inp.pictures.is_empty() {
            // Text goes the same way without the projector.
            let t = embed(&backend, &model, instruction, &[inp.text.as_str()], options).expect("embed").remove(0);
            let ct = cosine(&t.vector, &want);
            assert_eq!(t.tokens, got.tokens, "{name}: text-only token count");
            assert!(ct >= BAR, "{name}: text-only cosine {ct}");
        }
    }
    assert!(compared > 0, "no case was compared");
    println!("worst cosine {worst:.7}");
}
