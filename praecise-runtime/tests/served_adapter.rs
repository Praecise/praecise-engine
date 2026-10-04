//! A LoRA adapter pinned by content hash, served on every context of its base model.
#![cfg(feature = "bundled-llama")]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::float_cmp)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::{GgufMetadata, LoraWeight, write_lora_gguf};
use praecise_runtime::{AdapterSpec, Error, attach_adapter};
use sha2::{Digest, Sha256};

const N_VOCAB: usize = 32;
const PROBE: [i32; 6] = [3, 9, 1, 4, 7, 2];

fn backend() -> &'static LlamaBackend {
    static B: OnceLock<LlamaBackend> = OnceLock::new();
    B.get_or_init(|| LlamaBackend::init().unwrap())
}

fn build_model() -> LlamaModel {
    let mut meta = GgufMetadata::new();
    meta.set_str("general.architecture", "llama")
        .set_u32("llama.vocab_size", N_VOCAB as u32)
        .set_u32("llama.context_length", 32)
        .set_u32("llama.embedding_length", 32)
        .set_u32("llama.block_count", 2)
        .set_u32("llama.feed_forward_length", 64)
        .set_u32("llama.attention.head_count", 4)
        .set_u32("llama.attention.head_count_kv", 2)
        .set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("llama.rope.dimension_count", 8)
        .set_f32("llama.rope.freq_base", 10000.0)
        .set_str("tokenizer.ggml.model", "no_vocab");
    let mut seed = 5u64;
    LlamaModel::from_metadata(
        backend(),
        &meta,
        |name, n| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let u = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    if name.contains("norm") { 1.0 } else { 0.5 * u }
                })
                .collect()
        },
        &LlamaModelParams::default(),
    )
    .unwrap()
}

fn adapter_file() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("praecise-runtime-adapter-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapter.gguf");
    let weights: Vec<LoraWeight> = (0..2)
        .map(|il| {
            let (n_in, n_out, rank) = (32, 32, 2);
            let a = (0..n_in * rank).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.05).collect();
            let b = (0..rank * n_out).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.05).collect();
            LoraWeight { target: format!("blk.{il}.attn_q.weight"), n_in, n_out, rank, a, b }
        })
        .collect();
    write_lora_gguf(&path, "llama", 4.0, &weights).unwrap();
    path
}

fn logits(model: &LlamaModel, manual: Option<&std::path::Path>) -> Vec<f32> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(32))
        .with_n_threads(2)
        .with_n_threads_batch(2);
    let mut ctx = model.new_context(backend(), params).unwrap();
    let mut adapter = manual.map(|p| model.lora_adapter_init(p).unwrap());
    if let Some(a) = adapter.as_mut() {
        ctx.lora_adapter_set(a, 1.0).unwrap();
    }
    let mut b = LlamaBatch::new(PROBE.len(), 1);
    for (i, &t) in PROBE.iter().enumerate() {
        b.add(LlamaToken(t), i as i32, &[0], i + 1 == PROBE.len()).unwrap();
    }
    ctx.decode(&mut b).unwrap();
    ctx.get_logits_ith(PROBE.len() as i32 - 1).to_vec()
}

#[test]
fn served_adapter_is_pinned_by_hash_and_applied_to_every_context() {
    let path = adapter_file();
    let sha256: [u8; 32] = Sha256::digest(std::fs::read(&path).unwrap()).into();

    let mut model = build_model();
    let base = logits(&model, None);

    let mut wrong = sha256;
    wrong[0] ^= 1;
    let refused = attach_adapter(&mut model, &AdapterSpec { path: path.clone(), sha256: wrong, scale: 1.0 });
    assert!(matches!(refused, Err(Error::Adapter(_))), "{refused:?}");
    assert!(!model.has_served_lora());
    assert_eq!(logits(&model, None), base);

    attach_adapter(&mut model, &AdapterSpec { path: path.clone(), sha256, scale: 1.0 }).unwrap();
    let served = logits(&model, None);
    assert_ne!(served, base, "the adapter changes the outputs");
    assert_eq!(logits(&model, None), served, "every new context serves the adapter");

    // the same adapter set by hand on a context of a separately built base gives the same logits
    let reference = build_model();
    assert_eq!(logits(&reference, Some(&path)), served);

    let again = attach_adapter(&mut model, &AdapterSpec { path, sha256, scale: 1.0 });
    assert!(matches!(again, Err(Error::Adapter(_))), "{again:?}");
}
