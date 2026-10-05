//! Staged-weights activation against a cold load, on a real model.
//! Ignored by default; run with `--ignored`, it needs `PRAECISE_TEST_KV_MODEL`
//! and fails without it. Runs in its own process so
//! no other test holds the file mapped while it is evicted.
#![cfg(all(feature = "bundled-llama", target_os = "linux"))]

mod common;

use std::time::Instant;

use llama_cpp_2::model::AddBos;
use praecise_runtime::staging::{evict_from_page_cache, page_cache_fraction};
use praecise_runtime::StagedWeights;

/// Load, build a context, and decode a short prompt: time to first logits.
fn serve_first_token(model: &llama_cpp_2::model::LlamaModel) -> usize {
    let tokens = model.str_to_token("Staged weights activate", AddBos::Always).expect("tokenize");
    let mut ctx = common::context(model);
    common::decode(&mut ctx, 0, 0, &tokens).len()
}

#[test]
#[ignore = "needs a small dense GGUF: PRAECISE_TEST_KV_MODEL"]
fn staged_activation_is_faster_than_a_cold_load() {
    let path = common::model_path();
    common::backend();

    // Cold: the file is evicted from the page cache first.
    evict_from_page_cache(&path).expect("evict");
    let cold_resident = page_cache_fraction(&path).expect("residency");
    let t = Instant::now();
    let model = common::load(&path);
    assert!(serve_first_token(&model) > 0);
    let cold = t.elapsed();
    drop(model);

    // Staged: evict again, stage (untimed standby work), then activate.
    evict_from_page_cache(&path).expect("evict");
    let evicted_again = page_cache_fraction(&path).expect("residency");
    let t = Instant::now();
    let staged = StagedWeights::stage(&path).expect("stage");
    let stage_time = t.elapsed();
    let staged_resident = staged.resident_fraction().expect("residency");
    let t = Instant::now();
    let model = staged.activate(common::backend(), &common::cpu_params()).expect("activate");
    assert!(serve_first_token(&model) > 0);
    let warm = t.elapsed();

    eprintln!(
        "model {} ({} MiB): cold load+first token {:.1} ms (resident before {:.1}%); \
         staging {:.1} ms (pinned {}, resident {:.1}%, before {:.1}%); activate+first token {:.1} ms; speedup {:.1}x",
        path,
        staged.len() >> 20,
        cold.as_secs_f64() * 1e3,
        cold_resident * 100.0,
        stage_time.as_secs_f64() * 1e3,
        staged.pinned(),
        staged_resident * 100.0,
        evicted_again * 100.0,
        warm.as_secs_f64() * 1e3,
        cold.as_secs_f64() / warm.as_secs_f64()
    );
    assert!(staged_resident > 0.99, "staged weights not resident");
    if cold_resident < 0.05 {
        assert!(warm < cold, "activation ({warm:?}) not faster than cold load ({cold:?})");
    }
}
