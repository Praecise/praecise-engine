//! LoRA SFT on a real dense language model, and the exported adapter served by the runtime
//! against the trainer on many prompts. Ignored by default: run with
//! `PRAECISE_TRAIN_MODEL=<model.gguf> cargo test -p praecise-train --features engine --test real_model -- --ignored`.
//! `PRAECISE_TRAIN_PROMPTS` sets the number of served prompts (default 1000).
#![cfg(feature = "engine")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::write_lora_gguf;
use praecise_train::engine::{EngineConfig, LoraTrainer, SftExample, StepBatch, init_lora_weights, log_softmax_at};
use praecise_train::hash::sha256;
use praecise_train::kernel_class::Backend;
use praecise_train::philox::Philox;
use praecise_train::recipe::{AdapterSpec, Objective, OptimizerSpec, Precision, Recipe};

const N_CTX: u32 = 64;
const SEQ: usize = 32;
const THREADS: i32 = 16;

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_TRAIN_MODEL").expect("PRAECISE_TRAIN_MODEL names the model GGUF"))
}

fn recipe() -> Recipe {
    Recipe {
        objective: Objective::Sft,
        optimizer: OptimizerSpec::AdamW { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 },
        adapter: Some(AdapterSpec { rank: 8, alpha: 16.0, targets: vec!["attn_q.weight".into(), "attn_v.weight".into()] }),
        precision: Precision::F32,
        seq_len: SEQ as u32,
        batch: 2,
        grad_clip: 1.0,
    }
}

/// Token ids of stream `stream` of the counter PRNG, fixed for a seed.
fn tokens(seed: u64, stream: u32, n_vocab: u32, n: usize) -> Vec<i32> {
    let rng = Philox::new(seed);
    (0..n).map(|i| (rng.word(stream, 0, i as u64) % (n_vocab.min(30_000) - 1)) as i32 + 1).collect()
}

#[test]
#[ignore = "needs PRAECISE_TRAIN_MODEL"]
fn lora_sft_on_a_real_model_and_served_logprobs_match() {
    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default().with_use_extra_bufts(false)).unwrap();
    let arch = model.meta_val_str("general.architecture").unwrap();
    let n_vocab = model.n_vocab() as u32;
    let n_prompts: usize = std::env::var("PRAECISE_TRAIN_PROMPTS").map_or(1000, |v| v.parse().unwrap());

    let dir = std::env::temp_dir().join(format!("praecise-train-real-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let r = recipe();
    let spec = r.adapter.clone().unwrap();
    let init = dir.join("init.gguf");
    write_lora_gguf(&init, &arch, spec.alpha, &init_lora_weights(&model, &spec, 7).unwrap()).unwrap();
    let config = EngineConfig { n_ctx: N_CTX, n_threads: THREADS, deterministic: Some(Backend::Cpu), pooling: None };

    // a fixed pair of sequences, memorized by the adapter
    let data: Vec<SftExample> = (0..2).map(|k| SftExample::all_targets(tokens(3, k, n_vocab, SEQ))).collect();
    let batch = StepBatch::Sft(data);

    // repacked CPU weights run only the forward pass: a trainer on them is refused
    {
        let repacked = LlamaModel::load_from_file(&backend, model_path(), &LlamaModelParams::default()).unwrap();
        let refused = LoraTrainer::new(&backend, &repacked, &init, recipe(), config);
        assert!(refused.is_err(), "a trainer on repacked weights is refused");
    }

    let run = |n_steps: usize| {
        let mut t = LoraTrainer::new(&backend, &model, &init, recipe(), config).unwrap();
        let losses: Vec<f64> = (0..n_steps).map(|_| t.step(&batch).unwrap().loss).collect();
        (t, losses)
    };
    let (t, losses) = run(30);
    let (first, last) = (losses[0], losses[losses.len() - 1]);
    println!("loss {first:.4} -> {last:.4}");
    assert!(first.is_finite() && first > 1.0, "the base model cannot already predict random tokens: {first}");
    assert!(last < 0.5 * first, "loss {first} -> {last}");

    let (_, again) = run(5);
    assert_eq!(&losses[..5], &again[..], "repeated runs differ");

    let exported = dir.join("adapter.gguf");
    let digest = t.export_lora(&exported, &arch).unwrap();
    assert_eq!(digest, sha256(&std::fs::read(&exported).unwrap()));
    let mut trainer = t;

    let mut adapter = model.lora_adapter_init(&exported).unwrap();
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX)
        .with_n_ubatch(N_CTX)
        .with_n_threads(THREADS)
        .with_n_threads_batch(THREADS)
        .with_type_k(KvCacheType::F32)
        .with_type_v(KvCacheType::F32)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED);
    let mut ctx = model.new_context(&backend, params).unwrap();
    ctx.lora_adapter_set(&mut adapter, 1.0).unwrap();

    let nv = n_vocab as usize;
    let mut max_diff = 0.0f64;
    for p in 0..n_prompts {
        let len = 4 + (p % (SEQ - 4));
        let prompt = tokens(5, p as u32, n_vocab, len);
        let trained = trainer.logits(&prompt).unwrap();
        ctx.clear_kv_cache();
        let mut b = LlamaBatch::new(len, 1);
        for (i, &tok) in prompt.iter().enumerate() {
            b.add(LlamaToken(tok), i as i32, &[0], true).unwrap();
        }
        ctx.decode(&mut b).unwrap();
        for i in 0..len - 1 {
            let next = prompt[i + 1] as usize;
            let served = log_softmax_at(ctx.get_logits_ith(i as i32), next);
            let row = log_softmax_at(&trained[i * nv..(i + 1) * nv], next);
            max_diff = max_diff.max((served - row).abs());
        }
    }
    println!("{n_prompts} prompts, max |served - trained| log-prob {max_diff:e}");
    assert!(max_diff == 0.0, "served log-probs differ from the trainer by up to {max_diff}");
}
