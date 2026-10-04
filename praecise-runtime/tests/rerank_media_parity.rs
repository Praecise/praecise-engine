//! Multimodal cross-encoder scores against reference probabilities from the
//! source checkpoint's own pipeline. Needs `PRAECISE_RERANK_GGUF` (a
//! vision-language reranker GGUF), `PRAECISE_RERANK_MMPROJ` (its projector)
//! and `PRAECISE_RERANK_MEDIA` (a directory holding `expected.json` and the
//! raw RGB pictures it names); skips without them.
#![cfg(feature = "mtmd")]

use std::path::Path;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
use praecise_runtime::rerank::{rerank_media, RerankInput, RerankOptions, RerankPicture};

fn side(dir: &Path, v: &serde_json::Value) -> RerankInput {
    let pictures = v["pictures"]
        .as_array()
        .expect("pictures")
        .iter()
        .map(|p| RerankPicture {
            width: u32::try_from(p["width"].as_u64().expect("width")).expect("width"),
            height: u32::try_from(p["height"].as_u64().expect("height")).expect("height"),
            rgb: std::fs::read(dir.join(p["file"].as_str().expect("file"))).expect("picture"),
            video_frame: p["video_frame"].as_bool().expect("video_frame"),
        })
        .collect();
    RerankInput {
        text: v["text"].as_str().expect("text").to_owned(),
        pictures,
    }
}

#[test]
fn multimodal_reranker_scores_match_the_source_checkpoint() {
    let (Ok(gguf), Ok(mmproj), Ok(dir)) = (
        std::env::var("PRAECISE_RERANK_GGUF"),
        std::env::var("PRAECISE_RERANK_MMPROJ"),
        std::env::var("PRAECISE_RERANK_MEDIA"),
    ) else {
        eprintln!("PRAECISE_RERANK_GGUF / PRAECISE_RERANK_MMPROJ / PRAECISE_RERANK_MEDIA not set; skipping");
        return;
    };
    let dir = Path::new(&dir);
    let exp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).expect("expected")).expect("json");

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
    for case in exp["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let query = side(dir, &case["query"]);
        let documents: Vec<RerankInput> = case["documents"]
            .as_array()
            .expect("documents")
            .iter()
            .map(|d| side(dir, d))
            .collect();
        let want: Vec<f32> = serde_json::from_value(case["p_yes"].clone()).expect("p_yes");
        let tokens: Vec<usize> = serde_json::from_value(case["n_tokens"].clone()).expect("n_tokens");
        let got = rerank_media(&backend, &model, &projector, &query, &documents, options).expect("rerank");
        for (s, (w, t)) in got.iter().zip(want.iter().zip(&tokens)) {
            let d = (s.score - w).abs();
            worst = worst.max(d);
            eprintln!(
                "{name}[{}]: P(yes) {:.6} vs {:.6} (|d| {:.6}), tokens {} vs {}",
                s.index, s.score, w, d, s.tokens, t
            );
            assert_eq!(s.tokens, *t, "{name}[{}]: token count differs from the checkpoint pipeline", s.index);
        }
        // A document's score does not depend on the other documents.
        let alone = rerank_media(&backend, &model, &projector, &query, &documents[1..2], options).expect("alone");
        assert_eq!(alone[0].score.to_bits(), got[1].score.to_bits(), "{name}: score depends on the request");
    }
    eprintln!("worst |P(yes) difference| {worst:.6}");
    assert!(worst < 1e-2, "worst P(yes) difference {worst} exceeds 1e-2");
}
