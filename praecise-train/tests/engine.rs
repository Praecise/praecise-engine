//! `LoRA` SFT steps on the engine: the loss falls, steps are bitwise reproducible across thread
//! counts and replays from a checkpoint, refusals hold, and the exported adapter serves the same
//! log-probs the trainer computes.

#![cfg(feature = "engine")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::float_cmp, clippy::case_sensitive_file_extension_comparisons)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::OnceLock;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::{GgufMetadata, write_lora_gguf};
use praecise_train::Error;
use praecise_train::engine::{
    ContrastivePair, RegressionExample, RerankGroup,
    DistillExample, EngineConfig, LoraTrainer, PreferencePair, Rollout, RolloutGroup, SftExample, StepBatch, init_lora_weights,
    log_softmax_at, sequence_logprobs,
};
use praecise_train::hash::sha256;
use praecise_train::reward::{Reward, TokenFraction};
use praecise_train::philox::{Philox, uniform_f64};
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
    M.get_or_init(|| build_model(5))
}

/// A second model on the same vocabulary, used as a distillation teacher.
fn teacher() -> &'static LlamaModel {
    static M: OnceLock<LlamaModel> = OnceLock::new();
    M.get_or_init(|| build_model(17))
}

fn build_model(seed: u64) -> LlamaModel {
    {
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
        let mut seed = seed;
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
fn batch(step: u64) -> StepBatch {
    StepBatch::Sft(sft(step))
}

fn sft(step: u64) -> Vec<SftExample> {
    (0..2u64)
        .map(|r| {
            let off = (step * 2 + r) as i32;
            SftExample::all_targets((0..16).map(|i| (off + i * 3) % 9 + 1).collect())
        })
        .collect()
}

const PROBE: [i32; 8] = [1, 4, 7, 1, 4, 7, 1, 4];

fn trainer(threads: i32, opt: OptimizerSpec) -> LoraTrainer<'static> {
    let r = recipe(opt);
    let path = adapter_file(&r, 11);
    LoraTrainer::new(backend(), model(), &path, r, EngineConfig { n_ctx: N_CTX, n_threads: threads, deterministic: Some(Backend::Cpu), pooling: None }).unwrap()
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
    r.adapter = None;
    let path = adapter_file(&recipe(adamw()), 1);
    let cfg = EngineConfig { n_ctx: N_CTX, n_threads: 2, deterministic: None, pooling: None };
    assert!(matches!(LoraTrainer::new(backend(), model(), &path, r, cfg), Err(Error::Refused(_))));
    // a generative trainer has logits; an embedding trainer refuses them
    let mut e = trainer_for_pooled(Objective::RerankBce, adamw());
    assert!(matches!(e.logits(&[1, 2]), Err(Error::Refused(_))));
    let mut t = trainer(2, adamw());
    let empty = StepBatch::Sft(vec![SftExample { tokens: vec![1, 2, 3], target_mask: vec![false, false, false] }]);
    assert!(matches!(t.step(&empty), Err(Error::Refused(_))));
    assert!(matches!(t.step(&StepBatch::Dpo(vec![])), Err(Error::Refused(m)) if m.contains("objective")));
    // deterministic mode on a backend whose kernel set is not audited refuses the step and
    // leaves the state unchanged
    let r = recipe(adamw());
    let path = adapter_file(&r, 1);
    let gpu = EngineConfig { n_ctx: N_CTX, n_threads: 2, deterministic: Some(Backend::Cuda), pooling: None };
    let mut t = LoraTrainer::new(backend(), model(), &path, r, gpu).unwrap();
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

fn recipe_for(objective: Objective, optimizer: OptimizerSpec) -> Recipe {
    let mut r = recipe(optimizer);
    r.objective = objective;
    r
}

fn trainer_for(objective: Objective, optimizer: OptimizerSpec) -> LoraTrainer<'static> {
    let r = recipe_for(objective, optimizer);
    let path = adapter_file(&r, 23);
    let cfg = EngineConfig { n_ctx: N_CTX, n_threads: 4, deterministic: Some(Backend::Cpu), pooling: None };
    LoraTrainer::new(backend(), model(), &path, r, cfg).unwrap()
}

/// A serving context of `m` without adapters: the reference policy, or a teacher.
fn serving(m: &'static LlamaModel) -> LlamaContext<'static> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX)
        .with_n_ubatch(N_CTX)
        .with_n_threads(4)
        .with_n_threads_batch(4)
        .with_type_k(KvCacheType::F32)
        .with_type_v(KvCacheType::F32)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED);
    m.new_context(backend(), params).unwrap()
}

/// Samples `n` completion tokens after `prompt` from the trainer's current policy, keyed by
/// `(seed, stream)`; returns the sequence and the sampler's log-probs of the completion.
fn sample(t: &mut LoraTrainer<'_>, prompt: &[i32], n: usize, seed: u64, stream: u32) -> (Vec<i32>, Vec<f64>) {
    let rng = Philox::new(seed);
    let mut seq = prompt.to_vec();
    let mut lps = Vec::new();
    for k in 0..n {
        let logits = t.logits(&seq).unwrap();
        let row = &logits[(seq.len() - 1) * N_VOCAB..];
        let u = uniform_f64(rng.word(stream, 0, 2 * k as u64), rng.word(stream, 0, 2 * k as u64 + 1));
        let lp: Vec<f64> = (0..N_VOCAB).map(|j| log_softmax_at(row, j)).collect();
        let mut acc = 0.0;
        let mut tok = N_VOCAB - 1;
        for (j, l) in lp.iter().enumerate() {
            acc += l.exp();
            if u < acc {
                tok = j;
                break;
            }
        }
        seq.push(tok as i32);
        lps.push(lp[tok]);
    }
    (seq, lps)
}

fn completion_logprobs(ctx: &mut LlamaContext<'_>, tokens: &[i32], prompt_len: usize) -> Vec<f64> {
    sequence_logprobs(ctx, tokens).unwrap()[prompt_len - 1..].to_vec()
}

#[test]
fn muon_loss_falls() {
    let mut t = trainer(4, OptimizerSpec::Muon { lr: 0.05, momentum: 0.9, ns_steps: 5, weight_decay: 0.0 });
    let first = t.step(&batch(0)).unwrap().loss;
    let mut last = first;
    for s in 1..40 {
        last = t.step(&batch(s % 4)).unwrap().loss;
    }
    assert!(last < 0.7 * first, "muon loss {first} -> {last}");
    assert!(t.state().unwrap().iter().any(|(n, _)| n.ends_with(".m")));
}

#[test]
fn dpo_raises_the_preference_margin() {
    let mut reference = serving(model());
    let mut pairs = Vec::new();
    for prompt in [[1, 2, 3], [2, 3, 1]] {
        let chosen: Vec<i32> = prompt.iter().copied().chain([4, 4, 4]).collect();
        let rejected: Vec<i32> = prompt.iter().copied().chain([5, 5, 5]).collect();
        let ref_chosen = completion_logprobs(&mut reference, &chosen, 3).iter().sum();
        let ref_rejected = completion_logprobs(&mut reference, &rejected, 3).iter().sum();
        pairs.push(PreferencePair { prompt_len: 3, chosen, rejected, ref_chosen, ref_rejected });
    }
    let mut t = trainer_for(Objective::Dpo { beta: 0.5 }, adamw());
    let margin = |t: &mut LoraTrainer<'_>| -> f64 {
        pairs.iter().map(|p| {
            let c: f64 = t.token_logprobs(&p.chosen).unwrap()[2..].iter().sum();
            let r: f64 = t.token_logprobs(&p.rejected).unwrap()[2..].iter().sum();
            c - r
        }).sum()
    };
    let m0 = margin(&mut t);
    let batch = StepBatch::Dpo(pairs.clone());
    // the adapter starts as the identity, so the policy equals the reference exactly
    let first = t.step(&batch).unwrap().loss;
    assert_eq!(first, std::f64::consts::LN_2, "initial loss {first}");
    let mut last = first;
    for _ in 1..20 {
        last = t.step(&batch).unwrap().loss;
    }
    let m1 = margin(&mut t);
    assert!(m1 > m0 + 1.0 && last < 0.5 * first, "margin {m0} -> {m1}, loss {first} -> {last}");
}

#[test]
fn grpo_raises_a_verifiable_reward() {
    let mut reference = serving(model());
    let mut t = trainer_for(Objective::Grpo { clip: 0.2, kl_weight: 0.01 }, adamw());
    let prompt = [1, 2];
    let rewarder = TokenFraction { tokens: vec![7] };
    let reward = |seq: &[i32]| rewarder.score(&seq[..2], &seq[2..]).unwrap();
    let mut means = Vec::new();
    for step in 0..30u64 {
        let mut rollouts = Vec::new();
        for r in 0..8u32 {
            let (seq, sampler_logprobs) = sample(&mut t, &prompt, 6, 1000 + step, r);
            // the trainer's log-probs of a sampled sequence equal the sampler's exactly
            let lp = t.token_logprobs(&seq).unwrap();
            assert_eq!(&lp[1..], &sampler_logprobs[..], "sampler and trainer log-probs differ");
            let ref_logprobs = completion_logprobs(&mut reference, &seq, 2);
            rollouts.push(Rollout { reward: reward(&seq), tokens: seq, sampler_logprobs, ref_logprobs });
        }
        means.push(rollouts.iter().map(|r| r.reward).sum::<f64>() / 8.0);
        t.step(&StepBatch::Grpo(vec![RolloutGroup { prompt_len: 2, rollouts }])).unwrap();
    }
    let early: f64 = means[..5].iter().sum::<f64>() / 5.0;
    let late: f64 = means[25..].iter().sum::<f64>() / 5.0;
    assert!(late > early + 0.2, "mean reward {early} -> {late}");
}

/// Exact reverse KL(student || teacher) over the vocabulary at the last position of `prefix`.
fn exact_kl(t: &mut LoraTrainer<'_>, teacher: &mut LlamaContext<'_>, prefix: &[i32]) -> f64 {
    let s = t.logits(prefix).unwrap();
    let srow = &s[(prefix.len() - 1) * N_VOCAB..];
    let mut ext = prefix.to_vec();
    ext.push(0);
    sequence_logprobs(teacher, &ext).unwrap();
    let trow = teacher.get_logits_ith(prefix.len() as i32 - 1).to_vec();
    (0..N_VOCAB)
        .map(|j| {
            let (ls, lt) = (log_softmax_at(srow, j), log_softmax_at(&trow, j));
            ls.exp() * (ls - lt)
        })
        .sum()
}

#[test]
fn distillation_pulls_the_student_to_the_teacher() {
    let mut teach = serving(teacher());
    let mut t = trainer_for(Objective::Distill, adamw());
    let prefixes: [&[i32]; 3] = [&[1, 2], &[3], &[5, 6, 7]];
    let kl = |t: &mut LoraTrainer<'_>, teach: &mut LlamaContext<'_>| -> f64 {
        prefixes.iter().map(|p| exact_kl(t, teach, p)).sum()
    };
    let kl0 = kl(&mut t, &mut teach);
    for step in 0..30u64 {
        let mut batch = Vec::new();
        for (r, p) in prefixes.iter().enumerate() {
            let (tokens, _) = sample(&mut t, p, 4, 500 + step, r as u32);
            let teacher_logprobs = completion_logprobs(&mut teach, &tokens, p.len());
            batch.push(DistillExample { prompt_len: p.len(), tokens, teacher_logprobs });
        }
        t.step(&StepBatch::Distill(batch)).unwrap();
    }
    let kl1 = kl(&mut t, &mut teach);
    assert!(kl1 < 0.7 * kl0, "reverse KL {kl0} -> {kl1}");
}

fn trainer_for_pooled(objective: Objective, optimizer: OptimizerSpec) -> LoraTrainer<'static> {
    let r = recipe_for(objective, optimizer);
    let path = adapter_file(&r, 29);
    let cfg = EngineConfig { n_ctx: N_CTX, n_threads: 4, deterministic: Some(Backend::Cpu), pooling: Some(LlamaPoolingType::Mean) };
    LoraTrainer::new(backend(), model(), &path, r, cfg).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// Mean nDCG of the positive document over all queries (one relevant document each).
fn ndcg(t: &mut LoraTrainer<'_>, pairs: &[ContrastivePair]) -> f64 {
    let q: Vec<Vec<f32>> = pairs.iter().map(|p| t.outputs(&p.query).unwrap()).collect();
    let d: Vec<Vec<f32>> = pairs.iter().map(|p| t.outputs(&p.positive).unwrap()).collect();
    let mut total = 0.0;
    for i in 0..pairs.len() {
        let si = cosine(&q[i], &d[i]);
        let rank = (0..pairs.len()).filter(|&j| j != i && cosine(&q[i], &d[j]) > si).count();
        total += 1.0 / ((rank + 2) as f64).log2();
    }
    total / pairs.len() as f64
}

#[test]
fn contrastive_embeddings_raise_retrieval_ndcg() {
    let pairs: Vec<ContrastivePair> = (0..6)
        .map(|i: i32| ContrastivePair {
            query: vec![(i * 5 + 1) % 31 + 1, (i * 7 + 2) % 31 + 1, (i * 3 + 4) % 31 + 1],
            positive: vec![(i * 11 + 3) % 31 + 1, (i * 13 + 5) % 31 + 1, (i * 17 + 6) % 31 + 1, (i * 2 + 9) % 31 + 1],
        })
        .collect();
    let mut t = trainer_for_pooled(Objective::InfoNce { temperature: 0.1 }, adamw());
    let before = ndcg(&mut t, &pairs);
    let batch = StepBatch::Contrastive(pairs.clone());
    let first = t.step(&batch).unwrap().loss;
    let mut last = first;
    for _ in 1..30 {
        last = t.step(&batch).unwrap().loss;
    }
    let after = ndcg(&mut t, &pairs);
    assert!(after > before && after > 0.9 && last < first, "nDCG {before} -> {after}, loss {first} -> {last}");
}

#[test]
fn reranking_learns_the_relevant_candidate() {
    let groups: Vec<RerankGroup> = (0..3)
        .map(|g: i32| RerankGroup {
            candidates: (0..4).map(|c: i32| vec![g + 1, 20, (g * 4 + c) % 8 + 2, if c == g % 4 { 9 } else { 10 }]).collect(),
            labels: (0..4).map(|c| if c == g % 4 { 1.0 } else { 0.0 }).collect(),
        })
        .collect();
    for objective in [Objective::RerankListwise, Objective::RerankBce] {
        let mut t = trainer_for_pooled(objective.clone(), adamw());
        let batch = StepBatch::Rerank(groups.clone());
        let first = t.step(&batch).unwrap().loss;
        let mut last = first;
        for _ in 1..30 {
            last = t.step(&batch).unwrap().loss;
        }
        let mut correct = 0;
        for grp in &groups {
            let scores: Vec<f32> = grp.candidates.iter().map(|c| t.outputs(c).unwrap()[0]).collect();
            let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap();
            correct += usize::from(grp.labels[best] == 1.0);
        }
        assert!(last < 0.5 * first && correct == groups.len(), "{objective:?}: loss {first} -> {last}, top-1 {correct}/3");
    }
}

#[test]
fn quantile_regression_lowers_the_pinball_loss() {
    let batch: Vec<RegressionExample> = (0..6)
        .map(|i: i32| {
            let tokens: Vec<i32> = (0..5).map(|k| if k <= i % 5 { 3 } else { (i + k) % 7 + 4 }).collect();
            let target = f64::from(tokens.iter().filter(|&&x| x == 3).count() as u8) / 5.0;
            RegressionExample { tokens, target }
        })
        .collect();
    let mut t = trainer_for_pooled(Objective::Pinball { quantiles: vec![0.1, 0.5, 0.9] }, adamw());
    let b = StepBatch::Regression(batch);
    let first = t.step(&b).unwrap().loss;
    let mut last = first;
    for _ in 1..40 {
        last = t.step(&b).unwrap().loss;
    }
    assert!(last < 0.5 * first, "pinball {first} -> {last}");
}
