//! `LoRA` supervised fine-tuning on the serving graph.
//!
//! [`LoraSft`] owns a context with a `LoRA` adapter whose tensors are the only trainable
//! parameters. A step zeroes the gradients, runs every sequence of the batch forward and backward
//! on the engine, reduces the loss and the global gradient norm in a fixed order, clips, and
//! applies the optimizer element by element in a fixed order, so a step is a pure function of
//! the starting state, the batch and the recipe on one kernel class.

use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::{LoraWeight, TrainTensor, write_lora_gguf};

use crate::Error;
use crate::checkpoint::{Tensor, TrainState};
use crate::hash::{Digest, domain_hash, sha256};
use crate::kernel_class::{Backend, check_deterministic};
use crate::philox::{Philox, normal_f64};
use crate::recipe::{AdapterSpec, Objective, OptimizerSpec, Recipe};
use crate::steplog::{StepResult, StepSpec};
use crate::update::{OPT_PREFIX, PARAM_PREFIX};

fn engine(e: impl std::fmt::Display) -> Error {
    Error::Refused(format!("engine: {e}"))
}

/// One training sequence: `tokens[i]` is a target exactly when `target_mask[i]`; position 0 is
/// never a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftExample {
    /// Token ids.
    pub tokens: Vec<i32>,
    /// Which tokens count in the loss.
    pub target_mask: Vec<bool>,
}

impl SftExample {
    /// A sequence whose every token after the first is a target.
    #[must_use]
    pub fn all_targets(tokens: Vec<i32>) -> Self {
        let target_mask = (0..tokens.len()).map(|i| i > 0).collect();
        Self { tokens, target_mask }
    }

    fn n_targets(&self) -> usize {
        self.target_mask.iter().skip(1).filter(|&&m| m).count()
    }
}

/// Initial adapter weights: `A` from a seeded normal scaled by `1/sqrt(n_in)`, `B` zero, so the
/// adapted model starts equal to the base model.
///
/// # Errors
/// [`Error::Refused`] when a target matches no weight of the model.
pub fn init_lora_weights(model: &LlamaModel, spec: &AdapterSpec, seed: u64) -> Result<Vec<LoraWeight>, Error> {
    let rng = Philox::new(seed);
    let rank = spec.rank as usize;
    let mut out = Vec::new();
    for il in 0..model.n_layer() {
        for suffix in &spec.targets {
            let target = format!("blk.{il}.{suffix}");
            let Some(t) = model.tensor(&target) else { continue };
            let shape = t.shape();
            let n_in = usize::try_from(shape[0]).map_err(engine)?;
            let n_out = usize::try_from(shape[1]).map_err(engine)?;
            let tensor_id = u32::try_from(out.len()).map_err(engine)?;
            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / (n_in as f64).sqrt();
            #[allow(clippy::cast_possible_truncation)]
            let a = (0..n_in * rank)
                .map(|i| {
                    let i = i as u64;
                    (normal_f64(rng.word(tensor_id, 0, 2 * i), rng.word(tensor_id, 0, 2 * i + 1)) * scale) as f32
                })
                .collect();
            out.push(LoraWeight { target, n_in, n_out, rank, a, b: vec![0.0; rank * n_out] });
        }
    }
    if out.is_empty() {
        return Err(Error::Refused(format!("no weight matches the adapter targets {:?}", spec.targets)));
    }
    Ok(out)
}

/// What one step measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    /// Mean token cross-entropy over the batch's targets, before the update.
    pub loss: f64,
    /// Global gradient norm before clipping.
    pub grad_norm: f64,
}

/// Engine settings of a training context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    /// Longest sequence, which is also the context and ubatch size.
    pub n_ctx: u32,
    /// CPU threads.
    pub n_threads: i32,
    /// Deterministic mode on the backend the model runs on: every op of every training graph
    /// must be in that backend's deterministic kernel set, or the step is refused before its
    /// update. `None` turns the check off.
    pub deterministic: Option<Backend>,
}

/// `LoRA` SFT on one model.
#[derive(Debug)]
pub struct LoraSft<'m> {
    // the context refers to the adapter, so it is declared (and dropped) first
    ctx: LlamaContext<'m>,
    _adapter: LlamaLoraAdapter,
    params: Vec<(String, TrainTensor)>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    step: u64,
    recipe: Recipe,
    n_vocab: usize,
    deterministic: Option<Backend>,
}

impl<'m> LoraSft<'m> {
    /// Loads the adapter at `adapter_path` (see [`write_lora_gguf`]) onto a new context of
    /// `model` and makes its tensors the trainable parameters.
    ///
    /// # Errors
    /// [`Error::Refused`] for a recipe this trainer does not run (not SFT, no adapter, Muon) and
    /// for any engine refusal.
    pub fn new(
        backend: &LlamaBackend,
        model: &'m LlamaModel,
        adapter_path: &Path,
        recipe: Recipe,
        config: EngineConfig,
    ) -> Result<Self, Error> {
        if recipe.objective != Objective::Sft {
            return Err(Error::Refused("LoraSft runs the SFT objective only".into()));
        }
        if recipe.adapter.is_none() {
            return Err(Error::Refused("LoraSft needs an adapter in the recipe".into()));
        }
        if matches!(recipe.optimizer, OptimizerSpec::Muon { .. }) {
            return Err(Error::Refused("LoraSft supports AdamW and SGD".into()));
        }
        let mut adapter = model.lora_adapter_init(adapter_path).map_err(engine)?;
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(config.n_ctx))
            .with_n_batch(config.n_ctx)
            .with_n_ubatch(config.n_ctx)
            .with_n_seq_max(1)
            .with_n_threads(config.n_threads)
            .with_n_threads_batch(config.n_threads)
            .with_type_k(KvCacheType::F32)
            .with_type_v(KvCacheType::F32)
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED);
        let mut ctx = model.new_context(backend, params).map_err(engine)?;
        let scale = 1.0;
        ctx.lora_adapter_set(&mut adapter, scale).map_err(engine)?;
        ctx.grad_init(|name| name.ends_with(".lora_a") || name.ends_with(".lora_b")).map_err(engine)?;

        let mut params: Vec<(String, TrainTensor)> = adapter.tensors().into_iter().map(|t| (t.name(), t)).collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let m = params.iter().map(|(_, t)| vec![0.0; t.n_elements()]).collect();
        let v = params.iter().map(|(_, t)| vec![0.0; t.n_elements()]).collect();
        let n_vocab = usize::try_from(model.n_vocab()).map_err(engine)?;
        Ok(Self { ctx, _adapter: adapter, params, m, v, step: 0, recipe, n_vocab, deterministic: config.deterministic })
    }

    /// Optimizer steps taken.
    #[must_use]
    pub fn steps_taken(&self) -> u64 {
        self.step
    }

    /// Forward logits of one sequence (`tokens.len() x n_vocab`), with the current parameters.
    ///
    /// # Errors
    /// Engine refusals.
    pub fn logits(&mut self, tokens: &[i32]) -> Result<Vec<f32>, Error> {
        let toks: Vec<LlamaToken> = tokens.iter().map(|&t| LlamaToken(t)).collect();
        let mut logits = vec![0.0f32; tokens.len() * self.n_vocab];
        self.ctx.grad_sequence(&toks, None, Some(&mut logits)).map_err(engine)?;
        Ok(logits)
    }

    /// One optimizer step on `batch`.
    ///
    /// # Errors
    /// [`Error::Refused`] for an empty batch, a batch without targets, a malformed example, an op
    /// outside the deterministic kernel set in deterministic mode, or an engine refusal; a refused
    /// step leaves the parameters unchanged.
    pub fn step(&mut self, batch: &[SftExample]) -> Result<StepOutcome, Error> {
        let n_targets: usize = batch.iter().map(SftExample::n_targets).sum();
        if n_targets == 0 {
            return Err(Error::Refused("the batch has no targets".into()));
        }
        self.ctx.grad_reset();

        let n_vocab = self.n_vocab;
        let mut loss = 0.0f64;
        for ex in batch {
            if ex.tokens.len() != ex.target_mask.len() || ex.tokens.len() < 2 {
                return Err(Error::Refused("an example needs two tokens and one mask entry per token".into()));
            }
            // inputs are tokens[..n-1]; row i predicts tokens[i+1]
            let n = ex.tokens.len() - 1;
            let toks: Vec<LlamaToken> = ex.tokens[..n].iter().map(|&t| LlamaToken(t)).collect();
            // the engine divides by the rows of a sequence: weighting each target row by
            // n / n_targets makes the sum over the batch the mean over all targets
            #[allow(clippy::cast_precision_loss)]
            let w = n as f32 / n_targets as f32;
            let mut targets = vec![0.0f32; n * n_vocab];
            for i in 0..n {
                if ex.target_mask[i + 1] {
                    let t = usize::try_from(ex.tokens[i + 1]).map_err(engine)?;
                    if t >= n_vocab {
                        return Err(Error::Refused(format!("token {t} outside the vocabulary")));
                    }
                    targets[i * n_vocab + t] = w;
                }
            }
            let mut logits = vec![0.0f32; n * n_vocab];
            self.ctx.grad_sequence(&toks, Some(&targets), Some(&mut logits)).map_err(engine)?;
            if let Some(backend) = self.deterministic {
                let ops = self.ctx.grad_graph_ops();
                check_deterministic(backend, ops.iter().map(String::as_str))?;
            }
            for i in 0..n {
                if ex.target_mask[i + 1] {
                    let row = &logits[i * n_vocab..(i + 1) * n_vocab];
                    let t = usize::try_from(ex.tokens[i + 1]).map_err(engine)?;
                    loss -= log_softmax_at(row, t);
                }
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let loss = loss / n_targets as f64;

        let mut grads = Vec::with_capacity(self.params.len());
        let mut sq = 0.0f64;
        for (_, t) in &self.params {
            let g = self.ctx.grad(t).ok_or_else(|| Error::Refused("a parameter has no gradient".into()))?;
            let g = g.read_f32().map_err(engine)?;
            sq += g.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
            grads.push(g);
        }
        let grad_norm = sq.sqrt();
        #[allow(clippy::cast_possible_truncation)]
        let clip = if self.recipe.grad_clip > 0.0 && grad_norm > f64::from(self.recipe.grad_clip) {
            (f64::from(self.recipe.grad_clip) / grad_norm) as f32
        } else {
            1.0
        };

        self.step += 1;
        for (k, (_, t)) in self.params.iter().enumerate() {
            let mut p = t.read_f32().map_err(engine)?;
            let g = &grads[k];
            match self.recipe.optimizer {
                OptimizerSpec::AdamW { lr, beta1, beta2, eps, weight_decay } => {
                    let step = i32::try_from(self.step).map_err(engine)?;
                    let bc1 = 1.0 / (1.0 - f64::from(beta1).powi(step));
                    let bc2 = 1.0 / (1.0 - f64::from(beta2).powi(step));
                    #[allow(clippy::cast_possible_truncation)]
                    let (bc1, bc2) = (bc1 as f32, bc2 as f32);
                    let (m, v) = (&mut self.m[k], &mut self.v[k]);
                    for i in 0..p.len() {
                        let gi = g[i] * clip;
                        m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
                        v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                        let mh = m[i] * bc1;
                        let vh = v[i] * bc2;
                        p[i] = p[i] * (1.0 - lr * weight_decay) - lr * mh / (vh.sqrt() + eps);
                    }
                }
                OptimizerSpec::Sgd { lr, weight_decay } => {
                    for i in 0..p.len() {
                        p[i] = p[i] * (1.0 - lr * weight_decay) - lr * g[i] * clip;
                    }
                }
                OptimizerSpec::Muon { .. } => unreachable!("refused at construction"),
            }
            t.write_f32(&p).map_err(engine)?;
        }
        Ok(StepOutcome { loss, grad_norm })
    }

    /// Runs `spec` on `batch` and returns the step result: the new state root, the loss, the
    /// gradient norm and a digest of the per-token log-probs of `probe` after the step.
    ///
    /// # Errors
    /// [`Error::Mismatch`] when the current state is not `spec.state_root` or the step index is
    /// not the next one; step and engine refusals.
    pub fn run_step(&mut self, spec: &StepSpec, batch: &[SftExample], probe: &[i32]) -> Result<StepResult, Error> {
        if spec.step_index != self.step {
            return Err(Error::Mismatch(format!("step index {} but {} steps taken", spec.step_index, self.step)));
        }
        if spec.recipe_hash != self.recipe.hash() {
            return Err(Error::Mismatch("recipe hash".into()));
        }
        if self.state()?.state_root() != spec.state_root {
            return Err(Error::Mismatch("starting state root".into()));
        }
        let out = self.step(batch)?;
        let logits = self.logits(probe)?;
        let mut bytes = Vec::with_capacity(probe.len() * 8);
        for (i, row) in logits.chunks(self.n_vocab).enumerate().take(probe.len().saturating_sub(1)) {
            let t = usize::try_from(probe[i + 1]).map_err(engine)?;
            bytes.extend_from_slice(&log_softmax_at(row, t).to_le_bytes());
        }
        Ok(StepResult {
            state_root: self.state()?.state_root(),
            loss: out.loss,
            grad_norm: out.grad_norm,
            probe_digest: domain_hash("praecise.probe.logprobs.v1", &[&bytes]),
            reward_digest: None,
        })
    }

    /// The trainable state: `param.<tensor>` for every adapter tensor and, for `AdamW`,
    /// `opt.<tensor>.m` and `opt.<tensor>.v`.
    ///
    /// # Errors
    /// Engine refusals.
    pub fn state(&self) -> Result<TrainState, Error> {
        let mut s = TrainState::new();
        for (k, (name, t)) in self.params.iter().enumerate() {
            let shape: Vec<u64> = shape_of(*t);
            s.insert(format!("{PARAM_PREFIX}{name}"), Tensor::from_f32(shape.clone(), &t.read_f32().map_err(engine)?))?;
            if matches!(self.recipe.optimizer, OptimizerSpec::AdamW { .. }) {
                s.insert(format!("{OPT_PREFIX}{name}.m"), Tensor::from_f32(shape.clone(), &self.m[k]))?;
                s.insert(format!("{OPT_PREFIX}{name}.v"), Tensor::from_f32(shape, &self.v[k]))?;
            }
        }
        Ok(s)
    }

    /// Installs a state produced by [`Self::state`] after `steps_taken` optimizer steps.
    ///
    /// # Errors
    /// [`Error::Mismatch`] for a missing tensor or a shape that differs.
    pub fn load_state(&mut self, state: &TrainState, steps_taken: u64) -> Result<(), Error> {
        for (k, (name, t)) in self.params.iter().enumerate() {
            let get = |key: String| -> Result<Vec<f32>, Error> {
                let tensor = state.get(&key).ok_or_else(|| Error::Mismatch(format!("missing {key}")))?;
                if tensor.shape != shape_of(*t) {
                    return Err(Error::Mismatch(format!("shape of {key}")));
                }
                tensor.to_f32()
            };
            t.write_f32(&get(format!("{PARAM_PREFIX}{name}"))?).map_err(engine)?;
            if matches!(self.recipe.optimizer, OptimizerSpec::AdamW { .. }) {
                self.m[k] = get(format!("{OPT_PREFIX}{name}.m"))?;
                self.v[k] = get(format!("{OPT_PREFIX}{name}.v"))?;
            }
        }
        self.step = steps_taken;
        Ok(())
    }

    /// Writes the current adapter as a GGUF `LoRA` file for serving and returns its SHA-256.
    ///
    /// # Errors
    /// Engine refusals and filesystem failures.
    pub fn export_lora(&self, path: &Path, arch: &str) -> Result<Digest, Error> {
        let alpha = self.recipe.adapter.as_ref().map_or(0.0, |a| a.alpha);
        let mut weights: Vec<LoraWeight> = Vec::new();
        for pair in self.params.chunks(2) {
            let [(a_name, a), (b_name, b)] = pair else {
                return Err(Error::Mismatch("unpaired adapter tensor".into()));
            };
            let target = a_name
                .strip_suffix(".lora_a")
                .filter(|t| b_name.strip_suffix(".lora_b") == Some(*t))
                .ok_or_else(|| Error::Mismatch(format!("unpaired adapter tensors {a_name} and {b_name}")))?;
            let (sa, sb) = (a.shape(), b.shape());
            weights.push(LoraWeight {
                target: target.to_string(),
                n_in: usize::try_from(sa[0]).map_err(engine)?,
                n_out: usize::try_from(sb[1]).map_err(engine)?,
                rank: usize::try_from(sa[1]).map_err(engine)?,
                a: a.read_f32().map_err(engine)?,
                b: b.read_f32().map_err(engine)?,
            });
        }
        write_lora_gguf(path, arch, alpha, &weights).map_err(engine)?;
        Ok(sha256(&std::fs::read(path)?))
    }
}

fn shape_of(t: TrainTensor) -> Vec<u64> {
    let ne = t.shape();
    let n_dims = ne.iter().rposition(|&d| d != 1).map_or(1, |i| i + 1);
    ne[..n_dims].iter().map(|&d| u64::try_from(d).unwrap_or(0)).collect()
}

/// `log_softmax(row)[t]` in f64 with a fixed summation order.
#[must_use]
pub fn log_softmax_at(row: &[f32], t: usize) -> f64 {
    let mx = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
    let sum: f64 = row.iter().map(|&x| (f64::from(x) - mx).exp()).sum();
    f64::from(row[t]) - mx - sum.ln()
}
