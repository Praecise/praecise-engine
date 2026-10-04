//! `LoRA` training on the serving graph.
//!
//! [`LoraTrainer`] owns a context with a `LoRA` adapter whose tensors are the only trainable
//! parameters. A step zeroes the gradients, runs the batch forward and backward on the engine,
//! reduces the loss and the global gradient norm in a fixed order, clips, and applies the
//! optimizer element by element in a fixed order, so a step is a pure function of the starting
//! state, the batch and the recipe on one kernel class.
//!
//! Supervised fine-tuning differentiates the cross entropy inside the graph. Preference,
//! group-relative policy and distillation objectives are computed from the forward log-probs
//! (see [`crate::objective`]) and their gradient with respect to the logits is fed back through
//! the engine's weighted-sum pass.

use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::train::{GradLoss, LoraWeight, TrainTensor, write_lora_gguf};

use crate::Error;
use crate::checkpoint::{Tensor, TrainState};
use crate::hash::{Digest, domain_hash, sha256};
use crate::kernel_class::{Backend, check_deterministic};
use crate::objective::{
    dlogits, distill_token, dpo, group_advantages, grpo_token, info_nce, newton_schulz, pinball, rerank_bce, rerank_listwise,
    token_logprobs,
};
use crate::philox::{Philox, normal_f64};
use crate::recipe::{AdapterSpec, Objective, OptimizerSpec, Recipe};
use crate::steplog::{StepResult, StepSpec};
use crate::update::{OPT_PREFIX, PARAM_PREFIX};

pub use crate::objective::log_softmax_at;

fn engine(e: impl std::fmt::Display) -> Error {
    Error::Refused(format!("engine: {e}"))
}

/// One supervised sequence: `tokens[i]` is a target exactly when `target_mask[i]`; position 0 is
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

/// A preference pair: two sequences sharing a prompt of `prompt_len` tokens, with the reference
/// policy's summed log-probs of each completion.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferencePair {
    /// Prompt length; completion tokens are `prompt_len..`.
    pub prompt_len: usize,
    /// Prompt and preferred completion.
    pub chosen: Vec<i32>,
    /// Prompt and dispreferred completion.
    pub rejected: Vec<i32>,
    /// Reference log-prob of the chosen completion.
    pub ref_chosen: f64,
    /// Reference log-prob of the rejected completion.
    pub ref_rejected: f64,
}

/// One sampled completion with its reward and per-completion-token log-probs.
#[derive(Debug, Clone, PartialEq)]
pub struct Rollout {
    /// Prompt and completion.
    pub tokens: Vec<i32>,
    /// Scalar reward.
    pub reward: f64,
    /// Log-probs of the completion tokens under the policy that sampled them.
    pub sampler_logprobs: Vec<f64>,
    /// Log-probs of the completion tokens under the reference policy.
    pub ref_logprobs: Vec<f64>,
}

/// Completions of one prompt; advantages are relative within the group.
#[derive(Debug, Clone, PartialEq)]
pub struct RolloutGroup {
    /// Prompt length shared by the rollouts.
    pub prompt_len: usize,
    /// The rollouts.
    pub rollouts: Vec<Rollout>,
}

/// A completion the student sampled, with the teacher's log-probs of its tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillExample {
    /// Prompt length; completion tokens are `prompt_len..`.
    pub prompt_len: usize,
    /// Prompt and completion.
    pub tokens: Vec<i32>,
    /// Teacher log-probs of the completion tokens.
    pub teacher_logprobs: Vec<f64>,
}

/// A query and its positive document; the other documents of the batch are its negatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContrastivePair {
    /// Query tokens.
    pub query: Vec<i32>,
    /// Positive document tokens.
    pub positive: Vec<i32>,
}

/// Candidates for one query with relevance labels in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankGroup {
    /// Candidate sequences (query and candidate together, as the model scores them).
    pub candidates: Vec<Vec<i32>>,
    /// Relevance of each candidate.
    pub labels: Vec<f64>,
}

/// A sequence and the value whose quantiles the model predicts.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionExample {
    /// Input tokens.
    pub tokens: Vec<i32>,
    /// Target value.
    pub target: f64,
}

/// The batch of one step, matching the recipe's objective.
#[derive(Debug, Clone, PartialEq)]
pub enum StepBatch {
    /// Supervised sequences.
    Sft(Vec<SftExample>),
    /// Preference pairs.
    Dpo(Vec<PreferencePair>),
    /// Rollout groups.
    Grpo(Vec<RolloutGroup>),
    /// Student samples with teacher log-probs.
    Distill(Vec<DistillExample>),
    /// Query and positive pairs for contrastive embeddings.
    Contrastive(Vec<ContrastivePair>),
    /// Candidate lists for pointwise or listwise reranking.
    Rerank(Vec<RerankGroup>),
    /// Sequences with targets for quantile regression.
    Regression(Vec<RegressionExample>),
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

/// Per-token log-probs of `tokens[1..]` under a serving context (any model, any adapters set on
/// it), from one forward pass at positions `0..tokens.len()` on a cleared memory.
///
/// # Errors
/// Engine refusals.
pub fn sequence_logprobs(ctx: &mut LlamaContext<'_>, tokens: &[i32]) -> Result<Vec<f64>, Error> {
    ctx.clear_kv_cache();
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (i, &t) in tokens.iter().enumerate() {
        batch.add(LlamaToken(t), i32::try_from(i).map_err(engine)?, &[0], i + 1 < tokens.len()).map_err(engine)?;
    }
    ctx.decode(&mut batch).map_err(engine)?;
    (0..tokens.len() - 1)
        .map(|i| {
            let row = ctx.get_logits_ith(i32::try_from(i).map_err(engine)?);
            Ok(log_softmax_at(row, usize::try_from(tokens[i + 1]).map_err(engine)?))
        })
        .collect()
}

/// What one step measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    /// The objective over the batch, before the update.
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
    /// Pooling of the embedding context that embedding, reranking and regression objectives
    /// train; `None` uses the model's own. A score is the first pooled output and the quantile
    /// predictions are the first outputs, one per quantile.
    pub pooling: Option<LlamaPoolingType>,
}

/// `LoRA` training of one model.
#[derive(Debug)]
pub struct LoraTrainer<'m> {
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
    embedding: bool,
}

impl<'m> LoraTrainer<'m> {
    /// Loads the adapter at `adapter_path` (see [`write_lora_gguf`]) onto a new context of
    /// `model` and makes its tensors the trainable parameters.
    ///
    /// # Errors
    /// [`Error::Refused`] for a recipe without an adapter, and engine refusals.
    pub fn new(
        backend: &LlamaBackend,
        model: &'m LlamaModel,
        adapter_path: &Path,
        recipe: Recipe,
        config: EngineConfig,
    ) -> Result<Self, Error> {
        if recipe.adapter.is_none() {
            return Err(Error::Refused("the recipe has no adapter".into()));
        }
        let loss = match recipe.objective {
            Objective::Sft => GradLoss::CrossEntropy,
            _ => GradLoss::WeightedSum,
        };
        let embedding = matches!(
            recipe.objective,
            Objective::InfoNce { .. } | Objective::RerankBce | Objective::RerankListwise | Objective::Pinball { .. }
        );
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
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED)
            .with_embeddings(embedding)
            .with_pooling_type(config.pooling.unwrap_or(LlamaPoolingType::Unspecified));
        let mut ctx = model.new_context(backend, params).map_err(engine)?;
        ctx.lora_adapter_set(&mut adapter, 1.0).map_err(engine)?;
        ctx.grad_init(loss, |name| name.ends_with(".lora_a") || name.ends_with(".lora_b")).map_err(engine)?;

        let mut params: Vec<(String, TrainTensor)> = adapter.tensors().into_iter().map(|t| (t.name(), t)).collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let m = params.iter().map(|(_, t)| vec![0.0; t.n_elements()]).collect();
        let v = params.iter().map(|(_, t)| vec![0.0; t.n_elements()]).collect();
        let n_vocab = usize::try_from(model.n_vocab()).map_err(engine)?;
        Ok(Self { ctx, _adapter: adapter, params, m, v, step: 0, recipe, n_vocab, deterministic: config.deterministic, embedding })
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
        if self.embedding {
            return Err(Error::Refused("an embedding trainer has no logits".into()));
        }
        self.outputs(tokens)
    }

    /// Forward outputs of one sequence with the current parameters: the logits of a token
    /// objective, the pooled outputs of an embedding, reranking or regression objective.
    ///
    /// # Errors
    /// Engine refusals.
    pub fn outputs(&mut self, tokens: &[i32]) -> Result<Vec<f32>, Error> {
        let toks: Vec<LlamaToken> = tokens.iter().map(|&t| LlamaToken(t)).collect();
        let mut out = vec![0.0f32; self.ctx.grad_output_size(tokens.len())];
        self.ctx.grad_sequence(&toks, None, Some(&mut out)).map_err(engine)?;
        Ok(out)
    }

    /// Adds the gradient of `sum(d * outputs(tokens))`.
    fn backward_outputs(&mut self, tokens: &[i32], d: &[f64]) -> Result<(), Error> {
        #[allow(clippy::cast_possible_truncation)]
        let d: Vec<f32> = d.iter().map(|&x| x as f32).collect();
        self.backward(tokens, &d, None)
    }

    /// The first `n` pooled outputs of every sequence.
    fn heads(&mut self, seqs: &[Vec<i32>], n: usize) -> Result<Vec<Vec<f64>>, Error> {
        seqs.iter()
            .map(|t| {
                self.check_tokens(t)?;
                let out = self.outputs(t)?;
                if out.len() < n {
                    return Err(Error::Refused(format!("the pooled output has {} values, {n} needed", out.len())));
                }
                Ok(out[..n].iter().map(|&x| f64::from(x)).collect())
            })
            .collect()
    }

    fn head_grad(&self, tokens: &[i32], g: &[f64]) -> Vec<f64> {
        let mut d = vec![0.0; self.ctx.grad_output_size(tokens.len())];
        d[..g.len()].copy_from_slice(g);
        d
    }

    /// Log-probs of `tokens[1..]` with the current parameters.
    ///
    /// # Errors
    /// Engine refusals and tokens outside the vocabulary.
    pub fn token_logprobs(&mut self, tokens: &[i32]) -> Result<Vec<f64>, Error> {
        self.check_tokens(tokens)?;
        let logits = self.logits(&tokens[..tokens.len() - 1])?;
        Ok(token_logprobs(&logits, self.n_vocab, tokens))
    }

    fn check_tokens(&self, tokens: &[i32]) -> Result<(), Error> {
        if tokens.len() < 2 {
            return Err(Error::Refused("a sequence needs at least two tokens".into()));
        }
        if let Some(t) = tokens.iter().find(|&&t| usize::try_from(t).map_or(true, |t| t >= self.n_vocab)) {
            return Err(Error::Refused(format!("token {t} outside the vocabulary")));
        }
        Ok(())
    }

    /// One backward pass, then the deterministic-mode check of the graph it ran.
    fn backward(&mut self, inputs: &[i32], targets: &[f32], logits: Option<&mut [f32]>) -> Result<(), Error> {
        let toks: Vec<LlamaToken> = inputs.iter().map(|&t| LlamaToken(t)).collect();
        self.ctx.grad_sequence(&toks, Some(targets), logits).map_err(engine)?;
        if let Some(backend) = self.deterministic {
            let ops = self.ctx.grad_graph_ops();
            check_deterministic(backend, ops.iter().map(String::as_str))?;
        }
        Ok(())
    }

    /// Adds the gradient of `sum_i weights[i] * -log p(tokens[i + 1])` (weighted-sum objectives).
    fn backward_weighted(&mut self, tokens: &[i32], weights: &[f64]) -> Result<(), Error> {
        let inputs = &tokens[..tokens.len() - 1];
        let logits = self.logits(inputs)?;
        let d = dlogits(&logits, self.n_vocab, tokens, weights);
        self.backward(inputs, &d, None)
    }

    /// One optimizer step on `batch`, which must match the recipe's objective.
    ///
    /// # Errors
    /// [`Error::Refused`] for a batch of another objective, an empty or malformed batch, an op
    /// outside the deterministic kernel set in deterministic mode, or an engine refusal; a refused
    /// step leaves the parameters unchanged.
    pub fn step(&mut self, batch: &StepBatch) -> Result<StepOutcome, Error> {
        self.ctx.grad_reset();
        let loss = match (&self.recipe.objective, batch) {
            (Objective::Sft, StepBatch::Sft(b)) => self.accumulate_sft(b)?,
            (Objective::Dpo { beta }, StepBatch::Dpo(b)) => self.accumulate_dpo(*beta, b)?,
            (Objective::Grpo { clip, kl_weight }, StepBatch::Grpo(b)) => self.accumulate_grpo(*clip, *kl_weight, b)?,
            (Objective::Distill, StepBatch::Distill(b)) => self.accumulate_distill(b)?,
            (Objective::InfoNce { temperature }, StepBatch::Contrastive(b)) => self.accumulate_contrastive(*temperature, b)?,
            (Objective::RerankBce, StepBatch::Rerank(b)) => self.accumulate_rerank(false, b)?,
            (Objective::RerankListwise, StepBatch::Rerank(b)) => self.accumulate_rerank(true, b)?,
            (Objective::Pinball { quantiles }, StepBatch::Regression(b)) => {
                let q = quantiles.clone();
                self.accumulate_regression(&q, b)?
            }
            _ => return Err(Error::Refused("the batch does not match the recipe's objective".into())),
        };
        let grad_norm = self.apply_update()?;
        Ok(StepOutcome { loss, grad_norm })
    }

    fn accumulate_sft(&mut self, batch: &[SftExample]) -> Result<f64, Error> {
        let n_targets: usize = batch.iter().map(SftExample::n_targets).sum();
        if n_targets == 0 {
            return Err(Error::Refused("the batch has no targets".into()));
        }
        let n_vocab = self.n_vocab;
        let mut loss = 0.0f64;
        for ex in batch {
            if ex.tokens.len() != ex.target_mask.len() {
                return Err(Error::Refused("an example needs one mask entry per token".into()));
            }
            self.check_tokens(&ex.tokens)?;
            // inputs are tokens[..n-1]; row i predicts tokens[i+1]. The engine divides by the
            // rows of a sequence: weighting each target row by n / n_targets makes the sum over
            // the batch the mean over all targets.
            let n = ex.tokens.len() - 1;
            #[allow(clippy::cast_precision_loss)]
            let w = n as f32 / n_targets as f32;
            let mut targets = vec![0.0f32; n * n_vocab];
            for i in 0..n {
                if ex.target_mask[i + 1] {
                    targets[i * n_vocab + usize::try_from(ex.tokens[i + 1]).map_err(engine)?] = w;
                }
            }
            let mut logits = vec![0.0f32; n * n_vocab];
            self.backward(&ex.tokens[..n], &targets, Some(&mut logits))?;
            let lp = token_logprobs(&logits, n_vocab, &ex.tokens);
            loss -= (0..n).filter(|&i| ex.target_mask[i + 1]).map(|i| lp[i]).sum::<f64>();
        }
        #[allow(clippy::cast_precision_loss)]
        Ok(loss / n_targets as f64)
    }

    fn completion_weights(n: usize, prompt_len: usize, per_token: &[f64]) -> Vec<f64> {
        // row i predicts token i+1; completion tokens are prompt_len..n
        let mut w = vec![0.0; n - 1];
        for (k, &x) in per_token.iter().enumerate() {
            w[prompt_len - 1 + k] = x;
        }
        w
    }

    fn check_completion(&self, tokens: &[i32], prompt_len: usize) -> Result<usize, Error> {
        self.check_tokens(tokens)?;
        if prompt_len == 0 || prompt_len >= tokens.len() {
            return Err(Error::Refused("a completion needs a prompt and at least one token".into()));
        }
        Ok(tokens.len() - prompt_len)
    }

    fn accumulate_dpo(&mut self, beta: f64, batch: &[PreferencePair]) -> Result<f64, Error> {
        if batch.is_empty() {
            return Err(Error::Refused("the batch has no pairs".into()));
        }
        #[allow(clippy::cast_precision_loss)]
        let n_pairs = batch.len() as f64;
        let mut loss = 0.0;
        for p in batch {
            let nc = self.check_completion(&p.chosen, p.prompt_len)?;
            let nr = self.check_completion(&p.rejected, p.prompt_len)?;
            let lc: f64 = self.token_logprobs(&p.chosen)?[p.prompt_len - 1..].iter().sum();
            let lr: f64 = self.token_logprobs(&p.rejected)?[p.prompt_len - 1..].iter().sum();
            let (l, w) = dpo(beta, lc, lr, p.ref_chosen, p.ref_rejected);
            loss += l / n_pairs;
            let wc = Self::completion_weights(p.chosen.len(), p.prompt_len, &vec![w / n_pairs; nc]);
            let wr = Self::completion_weights(p.rejected.len(), p.prompt_len, &vec![-w / n_pairs; nr]);
            self.backward_weighted(&p.chosen, &wc)?;
            self.backward_weighted(&p.rejected, &wr)?;
        }
        Ok(loss)
    }

    fn accumulate_grpo(&mut self, clip: f64, kl_weight: f64, batch: &[RolloutGroup]) -> Result<f64, Error> {
        let n_rollouts: usize = batch.iter().map(|g| g.rollouts.len()).sum();
        if n_rollouts == 0 {
            return Err(Error::Refused("the batch has no rollouts".into()));
        }
        #[allow(clippy::cast_precision_loss)]
        let n_rollouts = n_rollouts as f64;
        let mut loss = 0.0;
        for g in batch {
            let adv = group_advantages(&g.rollouts.iter().map(|r| r.reward).collect::<Vec<_>>());
            for (r, a) in g.rollouts.iter().zip(adv) {
                let nc = self.check_completion(&r.tokens, g.prompt_len)?;
                if r.sampler_logprobs.len() != nc || r.ref_logprobs.len() != nc {
                    return Err(Error::Refused("one sampler and one reference log-prob per completion token".into()));
                }
                let lp = self.token_logprobs(&r.tokens)?;
                #[allow(clippy::cast_precision_loss)]
                let scale = 1.0 / (nc as f64 * n_rollouts);
                let mut per_token = Vec::with_capacity(nc);
                for k in 0..nc {
                    let (l, w) = grpo_token(lp[g.prompt_len - 1 + k], r.sampler_logprobs[k], r.ref_logprobs[k], a, clip, kl_weight);
                    loss += l * scale;
                    per_token.push(w * scale);
                }
                let w = Self::completion_weights(r.tokens.len(), g.prompt_len, &per_token);
                self.backward_weighted(&r.tokens, &w)?;
            }
        }
        Ok(loss)
    }

    fn accumulate_distill(&mut self, batch: &[DistillExample]) -> Result<f64, Error> {
        let mut n_tokens = 0usize;
        for ex in batch {
            n_tokens += self.check_completion(&ex.tokens, ex.prompt_len)?;
            if ex.teacher_logprobs.len() != ex.tokens.len() - ex.prompt_len {
                return Err(Error::Refused("one teacher log-prob per completion token".into()));
            }
        }
        if n_tokens == 0 {
            return Err(Error::Refused("the batch has no completion tokens".into()));
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / n_tokens as f64;
        let mut loss = 0.0;
        for ex in batch {
            let lp = self.token_logprobs(&ex.tokens)?;
            let mut per_token = Vec::with_capacity(ex.teacher_logprobs.len());
            for (k, &q) in ex.teacher_logprobs.iter().enumerate() {
                let (l, w) = distill_token(lp[ex.prompt_len - 1 + k], q);
                loss += l * scale;
                per_token.push(w * scale);
            }
            let w = Self::completion_weights(ex.tokens.len(), ex.prompt_len, &per_token);
            self.backward_weighted(&ex.tokens, &w)?;
        }
        Ok(loss)
    }

    fn accumulate_contrastive(&mut self, temperature: f64, batch: &[ContrastivePair]) -> Result<f64, Error> {
        if batch.len() < 2 {
            return Err(Error::Refused("in-batch negatives need at least two pairs".into()));
        }
        let mut q = Vec::with_capacity(batch.len());
        let mut d = Vec::with_capacity(batch.len());
        for p in batch {
            self.check_tokens(&p.query)?;
            self.check_tokens(&p.positive)?;
            q.push(self.outputs(&p.query)?);
            d.push(self.outputs(&p.positive)?);
        }
        let (loss, gq, gd) = info_nce(&q, &d, temperature);
        for (p, (gq, gd)) in batch.iter().zip(gq.iter().zip(&gd)) {
            self.backward_outputs(&p.query, gq)?;
            self.backward_outputs(&p.positive, gd)?;
        }
        Ok(loss)
    }

    fn accumulate_rerank(&mut self, listwise: bool, batch: &[RerankGroup]) -> Result<f64, Error> {
        if batch.is_empty() {
            return Err(Error::Refused("the batch has no candidate lists".into()));
        }
        #[allow(clippy::cast_precision_loss)]
        let n_groups = batch.len() as f64;
        let mut loss = 0.0;
        for grp in batch {
            if grp.candidates.is_empty() || grp.candidates.len() != grp.labels.len() {
                return Err(Error::Refused("one label per candidate".into()));
            }
            if listwise && grp.labels.iter().sum::<f64>() <= 0.0 {
                return Err(Error::Refused("a listwise group needs a relevant candidate".into()));
            }
            let scores: Vec<f64> = self.heads(&grp.candidates, 1)?.into_iter().map(|h| h[0]).collect();
            let (l, g) = if listwise { rerank_listwise(&scores, &grp.labels) } else { rerank_bce(&scores, &grp.labels) };
            loss += l / n_groups;
            for (c, gi) in grp.candidates.iter().zip(g) {
                let d = self.head_grad(c, &[gi / n_groups]);
                self.backward_outputs(c, &d)?;
            }
        }
        Ok(loss)
    }

    fn accumulate_regression(&mut self, quantiles: &[f64], batch: &[RegressionExample]) -> Result<f64, Error> {
        if batch.is_empty() || quantiles.is_empty() {
            return Err(Error::Refused("the batch has no examples or the recipe no quantiles".into()));
        }
        #[allow(clippy::cast_precision_loss)]
        let n = batch.len() as f64;
        let seqs: Vec<Vec<i32>> = batch.iter().map(|e| e.tokens.clone()).collect();
        let preds = self.heads(&seqs, quantiles.len())?;
        let mut loss = 0.0;
        for (ex, p) in batch.iter().zip(&preds) {
            let (l, g) = pinball(p, ex.target, quantiles);
            loss += l / n;
            let g: Vec<f64> = g.iter().map(|x| x / n).collect();
            let d = self.head_grad(&ex.tokens, &g);
            self.backward_outputs(&ex.tokens, &d)?;
        }
        Ok(loss)
    }

    /// Reads the gradients, clips them by the global norm and applies the optimizer.
    fn apply_update(&mut self) -> Result<f64, Error> {
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
                    #[allow(clippy::cast_possible_truncation)]
                    let bc1 = (1.0 / (1.0 - f64::from(beta1).powi(step))) as f32;
                    #[allow(clippy::cast_possible_truncation)]
                    let bc2 = (1.0 / (1.0 - f64::from(beta2).powi(step))) as f32;
                    let (m, v) = (&mut self.m[k], &mut self.v[k]);
                    for i in 0..p.len() {
                        let gi = g[i] * clip;
                        m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
                        v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                        p[i] = p[i] * (1.0 - lr * weight_decay) - lr * (m[i] * bc1) / ((v[i] * bc2).sqrt() + eps);
                    }
                }
                OptimizerSpec::Sgd { lr, weight_decay } => {
                    for i in 0..p.len() {
                        p[i] = p[i] * (1.0 - lr * weight_decay) - lr * g[i] * clip;
                    }
                }
                OptimizerSpec::Muon { lr, momentum, ns_steps, weight_decay } => {
                    // momentum on the matrix, then an orthogonalized update scaled by its aspect
                    let shape = t.shape();
                    let cols = usize::try_from(shape[0]).map_err(engine)?;
                    let rows = p.len() / cols;
                    let m = &mut self.m[k];
                    for i in 0..p.len() {
                        m[i] = momentum * m[i] + g[i] * clip;
                    }
                    let o = newton_schulz(m, rows, cols, ns_steps);
                    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                    let scale = (rows as f64 / cols as f64).max(1.0).sqrt() as f32;
                    for i in 0..p.len() {
                        p[i] = p[i] * (1.0 - lr * weight_decay) - lr * scale * o[i];
                    }
                }
            }
            t.write_f32(&p).map_err(engine)?;
        }
        Ok(grad_norm)
    }

    /// Runs `spec` on `batch` and returns the step result: the new state root, the loss, the
    /// gradient norm and a digest of the per-token log-probs of `probe` after the step (of its
    /// pooled outputs for an embedding objective).
    ///
    /// # Errors
    /// [`Error::Mismatch`] when the current state is not `spec.state_root`, the step index is not
    /// the next one or the recipe hash differs; step and engine refusals.
    pub fn run_step(&mut self, spec: &StepSpec, batch: &StepBatch, probe: &[i32]) -> Result<StepResult, Error> {
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
        let mut bytes = Vec::with_capacity(probe.len() * 8);
        if self.embedding {
            for x in self.outputs(probe)? {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        } else {
            for lp in self.token_logprobs(probe)? {
                bytes.extend_from_slice(&lp.to_le_bytes());
            }
        }
        let reward_digest = match batch {
            StepBatch::Grpo(groups) => {
                let rewards: Vec<f64> = groups.iter().flat_map(|g| g.rollouts.iter().map(|r| r.reward)).collect();
                Some(crate::reward::reward_digest(&rewards))
            }
            _ => None,
        };
        Ok(StepResult {
            state_root: self.state()?.state_root(),
            loss: out.loss,
            grad_norm: out.grad_norm,
            probe_digest: domain_hash("praecise.probe.logprobs.v1", &[&bytes]),
            reward_digest,
        })
    }

    fn has_m(&self) -> bool {
        matches!(self.recipe.optimizer, OptimizerSpec::AdamW { .. } | OptimizerSpec::Muon { .. })
    }

    fn has_v(&self) -> bool {
        matches!(self.recipe.optimizer, OptimizerSpec::AdamW { .. })
    }

    /// The trainable state: `param.<tensor>` for every adapter tensor, `opt.<tensor>.m` for
    /// `AdamW` and Muon, and `opt.<tensor>.v` for `AdamW`.
    ///
    /// # Errors
    /// Engine refusals.
    pub fn state(&self) -> Result<TrainState, Error> {
        let mut s = TrainState::new();
        for (k, (name, t)) in self.params.iter().enumerate() {
            let shape: Vec<u64> = shape_of(*t);
            s.insert(format!("{PARAM_PREFIX}{name}"), Tensor::from_f32(shape.clone(), &t.read_f32().map_err(engine)?))?;
            if self.has_m() {
                s.insert(format!("{OPT_PREFIX}{name}.m"), Tensor::from_f32(shape.clone(), &self.m[k]))?;
            }
            if self.has_v() {
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
        let (has_m, has_v) = (self.has_m(), self.has_v());
        for (k, (name, t)) in self.params.iter().enumerate() {
            let get = |key: String| -> Result<Vec<f32>, Error> {
                let tensor = state.get(&key).ok_or_else(|| Error::Mismatch(format!("missing {key}")))?;
                if tensor.shape != shape_of(*t) {
                    return Err(Error::Mismatch(format!("shape of {key}")));
                }
                tensor.to_f32()
            };
            t.write_f32(&get(format!("{PARAM_PREFIX}{name}"))?).map_err(engine)?;
            if has_m {
                self.m[k] = get(format!("{OPT_PREFIX}{name}.m"))?;
            }
            if has_v {
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
