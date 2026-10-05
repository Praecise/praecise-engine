//! Cross-encoder scores against reference probabilities from the source checkpoint.
//! Ignored by default; run with `--ignored`, it needs `PRAECISE_RERANK_GGUF` (a
//! reranker GGUF) and `PRAECISE_RERANK_EXPECTED` (JSON with `query`,
//! `documents` and `p_yes`) and fails without them.
#![cfg(feature = "bundled-llama")]

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use praecise_runtime::rerank::{pair_tokens, rerank, RerankOptions};

fn ranking(v: &[f32]) -> Vec<usize> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
    i
}

#[test]
#[ignore = "needs a reranker GGUF and its reference scores: PRAECISE_RERANK_GGUF and PRAECISE_RERANK_EXPECTED"]
fn reranker_scores_match_the_source_checkpoint() {
    let gguf = std::env::var("PRAECISE_RERANK_GGUF").expect("PRAECISE_RERANK_GGUF names the reranker GGUF");
    let expected =
        std::env::var("PRAECISE_RERANK_EXPECTED").expect("PRAECISE_RERANK_EXPECTED names the reference scores");
    let exp: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&expected).unwrap_or_else(|e| panic!("{expected}: {e}")),
    )
    .expect("json");
    let query = exp["query"].as_str().expect("query");
    let documents: Vec<String> = serde_json::from_value(exp["documents"].clone()).expect("documents");
    let want: Vec<f32> = serde_json::from_value(exp["p_yes"].clone()).expect("p_yes");
    let docs: Vec<&str> = documents.iter().map(String::as_str).collect();

    let mut backend = LlamaBackend::init().expect("backend");
    backend.void_logs();
    let model = LlamaModel::load_from_file(&backend, &gguf, &LlamaModelParams::default()).expect("load");

    let first = pair_tokens(&model, query, docs[0]).expect("tokens");
    eprintln!("pair 0: {} tokens, head {:?}, tail {:?}", first.len(), &first[..8.min(first.len())], &first[first.len().saturating_sub(8)..]);

    // The same pair read as a causal language model: P(yes) from the yes/no logits
    // at the final position, independent of the classification head.
    let yes = model.str_to_token("yes", AddBos::Never).expect("yes");
    let no = model.str_to_token("no", AddBos::Never).expect("no");
    assert!(yes.len() == 1 && no.len() == 1, "\"yes\" and \"no\" must be single tokens");
    let params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048)).with_n_batch(2048);
    let mut ctx = model.new_context(&backend, params).expect("context");
    let mut lm = Vec::new();
    for d in &docs {
        let t = pair_tokens(&model, query, d).expect("tokens");
        ctx.clear_kv_cache();
        let mut b = LlamaBatch::new(2048, 1);
        b.add_sequence(&t, 0, false).expect("batch");
        ctx.decode(&mut b).expect("decode");
        let l = ctx.get_logits_ith(b.n_tokens() - 1);
        let m = l[yes[0].0 as usize] - l[no[0].0 as usize];
        lm.push(1.0 / (1.0 + (-m).exp()));
    }
    eprintln!("causal-lm p_yes {lm:?}");
    drop(ctx);

    let got = rerank(&backend, &model, query, &docs, RerankOptions::default()).expect("rerank");
    let scores: Vec<f32> = got.iter().map(|s| s.score).collect();
    eprintln!("rerank  p_yes {scores:?}\nwant    p_yes {want:?}");
    assert!(!want.is_empty(), "{expected} holds no scores");
    assert_eq!(scores.len(), want.len(), "{} scores for {} reference scores", scores.len(), want.len());
    let worst = scores.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    assert_eq!(ranking(&scores), ranking(&want), "ranking differs");
    assert!(worst < 0.05, "max probability error {worst}");
}
