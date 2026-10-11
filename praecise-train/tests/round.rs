//! Supervised rounds on a real model, on the CPU or on a CUDA device: two runs of the same rounds
//! agree bitwise, a round continued from a saved state agrees with the uninterrupted run, and
//! the loss falls. Ignored by default: run with
//! `PRAECISE_TRAIN_MODEL=<model.gguf> PRAECISE_TRAIN_DEVICE=cpu|cuda cargo test -p praecise-train --test round -- --ignored`
//! (`--features cuda` for `cuda`).
#![cfg(feature = "engine")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::train::write_lora_gguf;
use praecise_train::checkpoint::TrainState;
use praecise_train::engine::{EngineConfig, LoraTrainer, SftExample, init_lora_weights};
use praecise_train::hash::{Digest, sha256};
use praecise_train::kernel_class::Backend;
use praecise_train::philox::Philox;
use praecise_train::recipe::{AdapterSpec, Objective, OptimizerSpec, Precision, Recipe};
use praecise_train::round::{RoundOutcome, SftRound, run_sft_round};

const N_CTX: u32 = 64;
const SEQ: usize = 32;

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

fn tokens(seed: u64, stream: u32, n_vocab: u32, n: usize) -> Vec<i32> {
    let rng = Philox::new(seed);
    (0..n).map(|i| (rng.word(stream, 0, i as u64) % (n_vocab.min(30_000) - 1)) as i32 + 1).collect()
}

struct Run<'a> {
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
    init: &'a Path,
    config: EngineConfig,
    round: SftRound<'a>,
}

impl Run<'_> {
    /// Two rounds of `steps` on one trainer.
    fn straight(&self) -> (RoundOutcome, RoundOutcome) {
        let mut t = LoraTrainer::new(self.backend, self.model, self.init, recipe(), self.config).unwrap();
        let a = run_sft_round(&mut t, &self.round).unwrap();
        let b = run_sft_round(&mut t, &self.round).unwrap();
        (a, b)
    }

    /// The second round on a new trainer, from the first round's saved state.
    fn resumed(&self) -> RoundOutcome {
        let mut t = LoraTrainer::new(self.backend, self.model, self.init, recipe(), self.config).unwrap();
        run_sft_round(&mut t, &self.round).unwrap();
        let saved = t.state().unwrap().payload();
        let steps = t.steps_taken();
        drop(t);
        let mut t = LoraTrainer::new(self.backend, self.model, self.init, recipe(), self.config).unwrap();
        t.load_state(&TrainState::from_payload(&saved).unwrap(), steps).unwrap();
        run_sft_round(&mut t, &self.round).unwrap()
    }
}

#[test]
#[ignore = "needs PRAECISE_TRAIN_MODEL"]
fn rounds_replay_bitwise_and_resume_from_a_saved_state() {
    let path = PathBuf::from(std::env::var("PRAECISE_TRAIN_MODEL").expect("PRAECISE_TRAIN_MODEL names the model GGUF"));
    let device = std::env::var("PRAECISE_TRAIN_DEVICE").unwrap_or_else(|_| "cpu".into());
    let (kind, gpu_layers) = match device.as_str() {
        "cpu" => (Backend::Cpu, 0),
        "cuda" => (Backend::Cuda, u32::MAX),
        other => panic!("PRAECISE_TRAIN_DEVICE is cpu or cuda, not {other}"),
    };
    let backend = LlamaBackend::init().unwrap();
    let params = LlamaModelParams::default().with_use_extra_bufts(false).with_n_gpu_layers(gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &path, &params).unwrap();
    let arch = model.meta_val_str("general.architecture").unwrap();
    let n_vocab = model.n_vocab() as u32;

    let dir = std::env::temp_dir().join(format!("praecise-train-round-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = recipe().adapter.unwrap();
    let init = dir.join("init.gguf");
    write_lora_gguf(&init, &arch, spec.alpha, &init_lora_weights(&model, &spec, 7).unwrap()).unwrap();

    let records: Vec<SftExample> = (0..4).map(|k| SftExample::all_targets(tokens(3, k, n_vocab, SEQ))).collect();
    let probe = tokens(5, 0, n_vocab, 16);
    let run = Run {
        backend: &backend,
        model: &model,
        init: &init,
        config: EngineConfig { n_ctx: N_CTX, n_threads: 8, deterministic: Some(kind), pooling: None },
        round: SftRound {
            base_root: sha256(&std::fs::read(&path).unwrap()),
            sample_seed: 11,
            kernel_class: format!("{device}/test"),
            steps: 10,
            records: &records,
            probe: &probe,
        },
    };

    let started = Instant::now();
    let (a1, a2) = run.straight();
    let elapsed = started.elapsed();
    let (b1, b2) = run.straight();
    let c2 = run.resumed();

    let losses: Vec<f64> = a1.losses().into_iter().chain(a2.losses()).collect();
    println!(
        "{device}: 20 steps in {:.1} s, loss {:.4} -> {:.4}, commitments {} {}",
        elapsed.as_secs_f64(),
        losses[0],
        losses[losses.len() - 1],
        a1.commitment(),
        a2.commitment()
    );
    assert_eq!(a1.commitment(), b1.commitment(), "the first round does not replay");
    assert_eq!(a2.commitment(), b2.commitment(), "the second round does not replay");
    assert_eq!(a2.commitment(), c2.commitment(), "a resumed round differs from the uninterrupted one");
    assert_eq!(a2.start_root, a1.end_root);
    assert_eq!(a2.first_step, 10);
    assert_ne!(a1.commitment(), Digest::default());
    a1.log.verify().unwrap();
    assert!(losses[0].is_finite() && losses[0] > 1.0, "{losses:?}");
    assert!(losses[losses.len() - 1] < 0.5 * losses[0], "{losses:?}");
}
