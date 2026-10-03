//! `LoRA` SFT steps on the engine: the loss falls, steps are bitwise reproducible across thread
//! counts and replays from a checkpoint, refusals hold, and the exported adapter serves the same
//! log-probs the trainer computes.

#![cfg(feature = "engine")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::OnceLock;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::{GgufMetadata, write_lora_gguf};
use praecise_train::Error;
use praecise_train::engine::{EngineConfig, LoraSft, SftExample, init_lora_weights, log_softmax_at};
use praecise_train::hash::sha256;
use praecise_train::kernel_class::{Backend, KernelClass};
use praecise_train::recipe::{AdapterSpec, Objective, OptimizerSpec, Precision, Recipe};
use praecise_train::steplog::{StepLog, StepSpec};

const N_VOCAB: usize = 32;
const N_CTX: u32 = 32;

fn backend() -> &'static LlamaBackend {
    static B: OnceLock<LlamaBackend> = OnceLock::new();
    B.get_or_init(|| LlamaBackend::init().unwrap())
}

fn model() -> &'static LlamaModel {
    static M: OnceLock<LlamaModel> = OnceLock::new();
    M.get_or_init(|| {
        let mut meta = GgufMetadata::new();
        meta.set_str("general.architecture", "llama")
            .set_u32("llama.vocab_size", N_VOCAB as u32)
            .set_u32("llama.context_length", N_CTX)
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
    })
}

fn recipe(optimizer: OptimizerSpec) -> Recipe {
    Recipe {
        objective: Objective::Sft,
        optimizer,
        adapter: Some(AdapterSpec {
            rank: 4,
            alpha: 8.0,
            targets: vec!["attn_q.weight".into(), "attn_v.weight".into(), "ffn_down.weight".into()],
        }),
        precision: Precision::F32,
        seq_len: 16,
        batch: 2,
        grad_clip: 1.0,
    }
}

fn adamw() -> OptimizerSpec {
    OptimizerSpec::AdamW { lr: 2e-2, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 }
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("praecise-train-engine-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn adapter_file(recipe: &Recipe, seed: u64) -> PathBuf {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = tmp(&format!("init-{seed}-{n}.gguf"));
    let spec = recipe.adapter.as_ref().unwrap();
    let weights = init_lora_weights(model(), spec, seed).unwrap();
    write_lora_gguf(&path, "llama", spec.alpha, &weights).unwrap();
    path
}

/// A periodic sequence the base model cannot predict and an adapter can learn.
fn batch(step: u64) -> Vec<SftExample> {
    (0..2u64)
        .map(|r| {
            let off = (step * 2 + r) as i32;
            SftExample::all_targets((0..16).map(|i| (off + i * 3) % 9 + 1).collect())
        })
        .collect()
}

const PROBE: [i32; 8] = [1, 4, 7, 1, 4, 7, 1, 4];

fn trainer(threads: i32, opt: OptimizerSpec) -> LoraSft<'static> {
    let r = recipe(opt);
    let path = adapter_file(&r, 11);
    LoraSft::new(backend(), model(), &path, r, EngineConfig { n_ctx: N_CTX, n_threads: threads, deterministic: Some(Backend::Cpu) }).unwrap()
}

#[test]
fn loss_falls() {
    let mut t = trainer(4, adamw());
    let first = t.step(&batch(0)).unwrap().loss;
    let mut last = first;
    for s in 1..40 {
        last = t.step(&batch(s % 4)).unwrap().loss;
    }
    assert!(last < 0.5 * first, "loss {first} -> {last}");

    let mut sgd = trainer(4, OptimizerSpec::Sgd { lr: 0.5, weight_decay: 0.0 });
    let first = sgd.step(&batch(0)).unwrap().loss;
    let mut last = first;
    for s in 1..40 {
        last = sgd.step(&batch(s % 4)).unwrap().loss;
    }
    assert!(last < first, "sgd loss {first} -> {last}");
}

fn run(threads: i32, n_steps: u64) -> (StepLog, Vec<praecise_train::checkpoint::TrainState>) {
    let mut t = trainer(threads, adamw());
    let r = recipe(adamw());
    let mut log = StepLog::new();
    let mut states = vec![t.state().unwrap()];
    for s in 0..n_steps {
        let spec = StepSpec {
            base_root: sha256(b"base"),
            state_root: states.last().unwrap().state_root(),
            data_root: sha256(b"data"),
            sample_seed: 3,
            step_index: s,
            recipe_hash: r.hash(),
            kernel_class: KernelClass::host_cpu().id(),
        };
        let result = t.run_step(&spec, &batch(s), &PROBE).unwrap();
        log.append(spec, result).unwrap();
        states.push(t.state().unwrap());
    }
    (log, states)
}

#[test]
fn steps_are_bitwise_reproducible() {
    let (a, sa) = run(1, 3);
    let (b, _) = run(6, 3);
    a.verify().unwrap();
    assert_eq!(a.head(), b.head(), "step logs differ between 1 and 6 threads");
    for (x, y) in a.records().iter().zip(b.records()) {
        assert!(x.result.bitwise_eq(&y.result));
    }

    // replay step 2 from the checkpoint after step 1 on a fresh trainer
    let mut t = trainer(3, adamw());
    t.load_state(&sa[2], 2).unwrap();
    let rec = &a.records()[2];
    let replayed = t.run_step(&rec.spec, &batch(2), &PROBE).unwrap();
    assert!(replayed.bitwise_eq(&rec.result));
    assert_eq!(a.first_divergence(&a.records().iter().map(|r| r.result.clone()).collect::<Vec<_>>()), None);

    // a spec that does not start from the trainer's state is refused
    let mut t = trainer(3, adamw());
    assert!(matches!(t.run_step(&rec.spec, &batch(2), &PROBE), Err(Error::Mismatch(_))));
}

#[test]
fn refusals() {
    let mut r = recipe(adamw());
    r.objective = Objective::Dpo { beta: 0.1 };
    let path = adapter_file(&recipe(adamw()), 1);
    let cfg = EngineConfig { n_ctx: N_CTX, n_threads: 2, deterministic: None };
    assert!(matches!(LoraSft::new(backend(), model(), &path, r, cfg), Err(Error::Refused(_))));
    let mut r = recipe(adamw());
    r.optimizer = OptimizerSpec::Muon { lr: 0.01, momentum: 0.9, ns_steps: 5, weight_decay: 0.0 };
    assert!(matches!(LoraSft::new(backend(), model(), &path, r, cfg), Err(Error::Refused(_))));
    let mut t = trainer(2, adamw());
    let empty = vec![SftExample { tokens: vec![1, 2, 3], target_mask: vec![false, false, false] }];
    assert!(matches!(t.step(&empty), Err(Error::Refused(_))));
    // deterministic mode on a backend whose kernel set is not audited refuses the step and
    // leaves the state unchanged
    let r = recipe(adamw());
    let path = adapter_file(&r, 1);
    let gpu = EngineConfig { n_ctx: N_CTX, n_threads: 2, deterministic: Some(Backend::Cuda) };
    let mut t = LoraSft::new(backend(), model(), &path, r, gpu).unwrap();
    let before = t.state().unwrap().state_root();
    assert!(matches!(t.step(&batch(0)), Err(Error::Refused(m)) if m.contains("deterministic")));
    assert_eq!(t.state().unwrap().state_root(), before);
    assert_eq!(t.steps_taken(), 0);

    let mut spec = recipe(adamw()).adapter.unwrap();
    spec.targets = vec!["nothing.weight".into()];
    assert!(init_lora_weights(model(), &spec, 1).is_err());
}

#[test]
fn exported_adapter_serves_the_trainer_logprobs() {
    let mut t = trainer(4, adamw());
    for s in 0..5 {
        t.step(&batch(s)).unwrap();
    }
    let trained = t.logits(&PROBE).unwrap();

    let path = tmp("export.gguf");
    let digest = t.export_lora(&path, "llama").unwrap();
    assert_eq!(digest, sha256(&std::fs::read(&path).unwrap()));

    let mut adapter = model().lora_adapter_init(&path).unwrap();
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX)
        .with_n_ubatch(N_CTX)
        .with_n_threads(4)
        .with_n_threads_batch(4)
        .with_type_k(KvCacheType::F32)
        .with_type_v(KvCacheType::F32)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED);
    let mut ctx = model().new_context(backend(), params).unwrap();
    ctx.lora_adapter_set(&mut adapter, 1.0).unwrap();
    let mut b = LlamaBatch::new(PROBE.len(), 1);
    for (i, &tok) in PROBE.iter().enumerate() {
        b.add(LlamaToken(tok), i as i32, &[0], true).unwrap();
    }
    ctx.decode(&mut b).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..PROBE.len() - 1 {
        let served = ctx.get_logits_ith(i as i32);
        let row = &trained[i * N_VOCAB..(i + 1) * N_VOCAB];
        let t = PROBE[i + 1] as usize;
        max_diff = max_diff.max((log_softmax_at(served, t) - log_softmax_at(row, t)).abs());
    }
    assert!(max_diff == 0.0, "served log-probs differ from the trainer by up to {max_diff}");
}
