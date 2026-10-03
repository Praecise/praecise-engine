//! Gradient-only training through the safe API: a random dense model built from metadata, a `LoRA`
//! adapter written as GGUF and loaded, gradients for the adapter only, checked by finite
//! differences.

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::num::NonZeroU32;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::{GgufMetadata, LoraWeight, TrainError, write_lora_gguf};

const N_VOCAB: usize = 40;
const N_EMBD: usize = 32;
const N_CTX: u32 = 32;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    #[allow(clippy::cast_precision_loss)]
    let u = (*seed >> 40) as f32 / (1u64 << 24) as f32;
    u - 0.5
}

fn metadata() -> GgufMetadata {
    let mut m = GgufMetadata::new();
    m.set_str("general.architecture", "llama")
        .set_u32("llama.vocab_size", N_VOCAB as u32)
        .set_u32("llama.context_length", N_CTX)
        .set_u32("llama.embedding_length", N_EMBD as u32)
        .set_u32("llama.block_count", 2)
        .set_u32("llama.feed_forward_length", 48)
        .set_u32("llama.attention.head_count", 2)
        .set_u32("llama.attention.head_count_kv", 2)
        .set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("llama.rope.dimension_count", (N_EMBD / 2) as u32)
        .set_f32("llama.rope.freq_base", 10000.0)
        .set_str("tokenizer.ggml.model", "no_vocab");
    m
}

fn ce_loss(logits: &[f32], targets: &[usize]) -> f64 {
    let mut loss = 0.0;
    for (row, &t) in logits.chunks(N_VOCAB).zip(targets) {
        let mx = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
        let lse = mx + row.iter().map(|&x| (f64::from(x) - mx).exp()).sum::<f64>().ln();
        loss -= f64::from(row[t]) - lse;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = targets.len() as f64;
    loss / n
}

#[test]
fn lora_gradients_match_finite_differences() {
    let backend = LlamaBackend::init().unwrap();
    let mut seed = 99u64;
    let model = LlamaModel::from_metadata(
        &backend,
        &metadata(),
        |name, n| {
            let norm = name.contains("norm");
            (0..n).map(|_| if norm { 1.0 } else { lcg(&mut seed) }).collect()
        },
        &LlamaModelParams::default(),
    )
    .unwrap();
    assert!(model.tensor("blk.0.attn_q.weight").is_some());
    assert!(model.tensor("no.such.weight").is_none());

    let rank = 4;
    let weights: Vec<LoraWeight> = ["blk.0.attn_q.weight", "blk.1.attn_v.weight", "blk.1.ffn_down.weight"]
        .iter()
        .map(|target| {
            let shape = model.tensor(target).unwrap().shape();
            let (n_in, n_out) = (usize::try_from(shape[0]).unwrap(), usize::try_from(shape[1]).unwrap());
            LoraWeight {
                target: (*target).to_string(),
                n_in,
                n_out,
                rank,
                a: (0..n_in * rank).map(|_| 0.5 * lcg(&mut seed)).collect(),
                b: (0..rank * n_out).map(|_| 0.5 * lcg(&mut seed)).collect(),
            }
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("llama-train-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapter.gguf");
    write_lora_gguf(&path, "llama", 8.0, &weights).unwrap();
    let mut adapter = model.lora_adapter_init(&path).unwrap();
    let lora_tensors = adapter.tensors();
    assert_eq!(lora_tensors.len(), 6);

    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX)
        .with_n_ubatch(N_CTX)
        .with_n_threads(4)
        .with_n_threads_batch(4)
        .with_type_k(KvCacheType::F32)
        .with_type_v(KvCacheType::F32)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED);
    let mut ctx = model.new_context(&backend, params).unwrap();
    ctx.lora_adapter_set(&mut adapter, 1.0).unwrap();
    ctx.grad_init(|name| name.ends_with(".lora_a") || name.ends_with(".lora_b")).unwrap();
    assert_eq!(ctx.grad_init(|_| true), Err(TrainError::AlreadyTraining));

    let tokens: Vec<LlamaToken> = (0..10).map(|i| LlamaToken((i * 7 + 1) % N_VOCAB as i32)).collect();
    let target_ids: Vec<usize> = (0..10).map(|i| (i * 5 + 3) % N_VOCAB).collect();
    let mut targets = vec![0.0f32; tokens.len() * N_VOCAB];
    for (i, &t) in target_ids.iter().enumerate() {
        targets[i * N_VOCAB + t] = 1.0;
    }
    let mut logits = vec![0.0f32; tokens.len() * N_VOCAB];
    ctx.grad_sequence(&tokens, Some(&targets), Some(&mut logits)).unwrap();

    // the base weight is frozen: it has no gradient
    assert!(ctx.grad(&model.tensor("blk.0.attn_q.weight").unwrap()).is_none());

    for t in &lora_tensors {
        let grad = ctx.grad(t).unwrap().read_f32().unwrap();
        let mut values = t.read_f32().unwrap();
        let i = grad
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        let eps = 1e-2f32;
        let x = values[i];
        let mut l = [0.0f64; 2];
        for (s, sign) in [1.0f32, -1.0].iter().enumerate() {
            values[i] = x + sign * eps;
            t.write_f32(&values).unwrap();
            ctx.grad_sequence(&tokens, None, Some(&mut logits)).unwrap();
            l[s] = ce_loss(&logits, &target_ids);
        }
        values[i] = x;
        t.write_f32(&values).unwrap();
        let fd = (l[0] - l[1]) / (2.0 * f64::from(eps));
        let g = f64::from(grad[i]);
        assert!(g.abs() > 1e-6, "{}: gradient vanished", t.name());
        assert!((fd - g).abs() <= 2e-2 * g.abs() + 2e-6, "{}: fd {fd} vs grad {g}", t.name());
    }

    ctx.grad_reset();
    assert!(ctx.grad(&lora_tensors[0]).unwrap().read_f32().unwrap().iter().all(|&x| x == 0.0));
    std::fs::remove_dir_all(&dir).ok();
}
