//! Multimodal cross-encoder scores against reference probabilities from the
//! source checkpoint's own pipeline. The reference scores and the raw RGB
//! pictures they were computed from are committed under
//! `tests/parity/rerank_media` (written by
//! `tests/parity/make_rerank_media_fixtures.py`). The weights are not: the
//! test is ignored by default and, run with `--ignored`, needs
//! `PRAECISE_RERANK_GGUF` (the vision-language reranker GGUF converted from
//! the checkpoint revision recorded in `expected.json`) and
//! `PRAECISE_RERANK_MMPROJ` (its projector), and fails without them.
//! `PRAECISE_RERANK_CASES` (comma-separated case names) runs a subset.
#![cfg(feature = "mtmd")]

use std::path::Path;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
use praecise_runtime::media::{MediaInput, Picture};
use praecise_runtime::rerank::{rerank_media, RerankOptions};

fn side(dir: &Path, v: &serde_json::Value) -> MediaInput {
    let pictures = v["pictures"]
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
    MediaInput {
        text: v["text"].as_str().expect("text").to_owned(),
        pictures,
    }
}

#[test]
#[ignore = "needs the reranker weights: PRAECISE_RERANK_GGUF and PRAECISE_RERANK_MMPROJ"]
fn multimodal_reranker_scores_match_the_source_checkpoint() {
    let gguf = std::env::var("PRAECISE_RERANK_GGUF").expect("PRAECISE_RERANK_GGUF names the reranker GGUF");
    let mmproj = std::env::var("PRAECISE_RERANK_MMPROJ").expect("PRAECISE_RERANK_MMPROJ names its projector");
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/parity/rerank_media"));
    let expected = dir.join("expected.json");
    let exp: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&expected).unwrap_or_else(|e| panic!("{}: {e}", expected.display())),
    )
    .expect("json");

    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let model = LlamaModel::load_from_file(&backend, &gguf, &LlamaModelParams::default()).expect("load");
    // The pictures arrive sized by the checkpoint's processor; the smallest
    // it produces is 4 merged patches.
    let params = MtmdContextParams {
        use_gpu: false,
        image_min_tokens: 4,
        ..MtmdContextParams::default()
    };
    let projector = unsafe { MtmdContext::init_from_file(&mmproj, &model, &params) }.expect("projector");
    let options = RerankOptions {
        n_ctx: 32768,
        ..RerankOptions::default()
    };

    let mut worst = 0f32;
    let mut pairs = 0usize;
    let only = std::env::var("PRAECISE_RERANK_CASES").ok();
    let cases = exp["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "{} holds no cases", expected.display());
    for case in cases {
        let name = case["name"].as_str().expect("name");
        if only.as_deref().is_some_and(|o| !o.split(',').any(|c| c == name)) {
            continue;
        }
        let query = side(dir, &case["query"]);
        let documents: Vec<MediaInput> = case["documents"]
            .as_array()
            .expect("documents")
            .iter()
            .map(|d| side(dir, d))
            .collect();
        let want: Vec<f32> = serde_json::from_value(case["p_yes"].clone()).expect("p_yes");
        let tokens: Vec<usize> = serde_json::from_value(case["n_tokens"].clone()).expect("n_tokens");
        let got = rerank_media(&backend, &model, &projector, &query, &documents, options).expect("rerank");
        assert_eq!(got.len(), want.len(), "{name}: {} scores for {} reference scores", got.len(), want.len());
        assert_eq!(tokens.len(), want.len(), "{name}: token counts and scores disagree");
        pairs += got.len();
        for (s, (w, t)) in got.iter().zip(want.iter().zip(&tokens)) {
            let d = (s.score - w).abs();
            worst = worst.max(d);
            eprintln!(
                "pair {name}[{}]: P(yes) {:.6} vs {:.6} (|d| {:.6}), tokens {} vs {}",
                s.index, s.score, w, d, s.tokens, t
            );
            assert_eq!(s.tokens, *t, "{name}[{}]: token count differs from the checkpoint pipeline", s.index);
        }
        // A document's score does not depend on the other documents.
        if case == &exp["cases"][0] {
            let alone = rerank_media(&backend, &model, &projector, &query, &documents[1..2], options).expect("alone");
            assert_eq!(alone[0].score.to_bits(), got[1].score.to_bits(), "{name}: score depends on the request");
        }
    }
    assert!(pairs > 0, "no pair was scored (PRAECISE_RERANK_CASES={only:?})");
    eprintln!("{pairs} pairs, worst |P(yes) difference| {worst:.6}");
    assert!(worst < 1e-2, "worst P(yes) difference {worst} exceeds 1e-2");
}
