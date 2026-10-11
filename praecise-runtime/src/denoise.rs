//! Decoding a diffusion language model: the canvas, the order positions are
//! revealed in, and when an answer is final.
//!
//! A diffusion model does not produce one token per forward pass. It reads a
//! canvas (the prompt, then the answer still to be written) and predicts every
//! position at once; the decoder keeps some of the predictions and runs the
//! canvas again until it is complete. Two families are served:
//!
//! * **Masked.** The answer starts as mask tokens and each step reveals the
//!   positions the model is surest of. A model trained on blocks reveals one
//!   block at a time from left to right, and a block-causal model keeps the
//!   blocks it has finished in the cache instead of reading them again. With
//!   token editing, a revealed token in the open block is rewritten when the
//!   model later prefers another one strongly enough.
//! * **Uniform noise.** The answer starts as random tokens. Each step keeps the
//!   lowest-entropy predictions within an entropy budget, re-noises the rest
//!   and feeds the step's logits back to the model (self-conditioning). A long
//!   answer is written one fixed-length canvas after another, each finished
//!   canvas joining the prefix of the next.
//!
//! This module is the decoding logic alone, with no backend, so schedules and
//! reveal orders are tested against hand-made logits. The engine that runs it
//! against a model is [`crate::denoise_engine`].

use std::ops::Range;

/// Candidates a position samples from when the request names no `top_k`.
/// The same truncation the token-at-a-time engine applies, for the same
/// reason: a full-vocabulary sort per position is the sampler's whole cost.
pub const DEFAULT_TOP_K: u32 = 40;

/// A small deterministic generator: a request's seed reproduces its answer.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n.max(1))) as u32
    }
}

/// How one position's prediction is drawn.
#[derive(Debug, Clone, Copy)]
pub struct RowSampling {
    /// `0` takes the most likely token; above it, samples at that temperature.
    pub temperature: f32,
    /// Candidates kept before sampling.
    pub top_k: u32,
    /// Nucleus mass kept among those candidates; `1.0` keeps them all.
    pub top_p: f32,
    /// A token never predicted, such as the mask itself.
    pub forbid: Option<i32>,
}

/// A position's prediction and how sure of it the model is, over the
/// candidates it was drawn from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pick {
    pub token: i32,
    /// Probability of `token`.
    pub prob: f32,
    /// Most likely candidate's probability minus the runner-up's.
    pub margin: f32,
    /// Entropy of the candidate distribution.
    pub entropy: f32,
}

/// Draw one position's prediction from its logits.
pub fn pick(logits: &[f32], s: &RowSampling, rng: &mut Rng) -> Pick {
    let k = (s.top_k.max(1) as usize).min(logits.len());
    // The k largest logits, best first, ties broken towards the lower id.
    let mut top: Vec<(f32, i32)> = Vec::with_capacity(k + 1);
    for (id, &l) in logits.iter().enumerate() {
        if s.forbid == Some(id as i32) || l.is_nan() {
            continue;
        }
        if top.len() == k && l <= top[k - 1].0 {
            continue;
        }
        let at = top.partition_point(|&(v, _)| v >= l);
        top.insert(at, (l, id as i32));
        top.truncate(k);
    }
    if top.is_empty() {
        return Pick { token: 0, prob: 0.0, margin: 0.0, entropy: 0.0 };
    }
    let tau = if s.temperature > 0.0 { s.temperature } else { 1.0 };
    let m = top[0].0 / tau;
    let mut p: Vec<f32> = top.iter().map(|&(l, _)| (l / tau - m).exp()).collect();
    let z: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= z);
    if s.top_p < 1.0 {
        let mut mass = 0.0;
        let mut keep = p.len();
        for (i, &x) in p.iter().enumerate() {
            mass += x;
            if mass >= s.top_p {
                keep = i + 1;
                break;
            }
        }
        p.truncate(keep);
        top.truncate(keep);
        let z: f32 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= z);
    }
    let chosen = if s.temperature > 0.0 {
        let u = rng.unit();
        let mut acc = 0.0;
        let mut at = p.len() - 1;
        for (i, &x) in p.iter().enumerate() {
            acc += x;
            if u < acc {
                at = i;
                break;
            }
        }
        at
    } else {
        0
    };
    let entropy = -p.iter().filter(|&&x| x > 0.0).map(|&x| x * x.ln()).sum::<f32>();
    let margin = if p.len() > 1 { p[0] - p[1] } else { p[0] };
    Pick { token: top[chosen].1, prob: p[chosen], margin, entropy }
}

/// Which masked positions a step reveals first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unmask {
    /// Highest probability of the predicted token.
    Confidence,
    /// Widest gap between the two most likely tokens.
    Margin,
    /// Lowest entropy.
    Entropy,
    /// Uniformly at random.
    Random,
}

/// How many masked positions each step reveals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pace {
    /// The fraction a noise schedule leaves at each step: few early, the rest
    /// at the last step. `eps` is where the schedule stops short of zero.
    Timestep { eps: f32 },
    /// The same number at every step of a block.
    Even,
}

/// A masked-diffusion model's decoding plan.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskedPlan {
    pub mask: i32,
    /// Positions per block, revealed left to right; `0` treats the whole
    /// answer as one block.
    pub block: u32,
    /// Steps per block (per answer when `block` is `0`); `0` takes one step
    /// per position.
    pub steps: u32,
    pub unmask: Unmask,
    pub pace: Pace,
    /// Reveal every position above this confidence, when that is more than
    /// the step's share.
    pub threshold: Option<f32>,
    /// Rewrite a revealed token in the open block when the model prefers
    /// another one with more than this probability.
    pub edit_threshold: Option<f32>,
    /// Position `i` is predicted by the logits at `i - 1`.
    pub shift_logits: bool,
    /// Block-causal attention: a block sees itself and the blocks before it,
    /// so finished blocks stay in the cache. Blocks are aligned to absolute
    /// positions from the start of the prompt.
    pub cached_blocks: bool,
}

impl MaskedPlan {
    /// Why this plan cannot be run, if it cannot.
    pub fn check(&self) -> Result<(), String> {
        if self.cached_blocks && self.block == 0 {
            return Err("block-causal decoding needs a block length".into());
        }
        if self.cached_blocks && self.shift_logits {
            return Err("shifted logits need the position before each block, which a cached block does not read".into());
        }
        if self.edit_threshold.is_some() && self.block == 0 {
            return Err("token editing closes a block before streaming it, so it needs a block length".into());
        }
        Ok(())
    }
}

/// One forward pass's worth of a sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Position of the first token.
    pub start: usize,
    pub tokens: Vec<i32>,
    /// Indices into `tokens` whose logits the decoder reads, in the order
    /// [`MaskedCanvas::apply`] expects them.
    pub rows: Vec<usize>,
    /// Drop cached positions from here on before the pass.
    pub clear_from: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Prompt blocks still to be read into the cache: the next one starts here.
    Prefill(usize),
    /// Denoising the open block.
    Denoise,
    /// The open block is final; read it into the cache once more so the next
    /// block sees its final tokens.
    Commit,
    Done,
}

/// A masked-diffusion sequence in flight.
#[derive(Debug, Clone)]
pub struct MaskedCanvas {
    plan: MaskedPlan,
    prompt_len: usize,
    tokens: Vec<i32>,
    /// The open block, as absolute positions.
    open: Range<usize>,
    step: u32,
    /// Positions to reveal at each step of the open block.
    schedule: Vec<usize>,
    /// Steps taken after the open block ran out of masks, editing only.
    edit_rounds: u32,
    phase: Phase,
    /// Forward positions run so far, the decode's actual work.
    forward_positions: u64,
    steps_taken: u32,
}

impl MaskedCanvas {
    /// A canvas of `answer_len` masks after `prompt`.
    pub fn new(mut plan: MaskedPlan, prompt: &[i32], answer_len: usize) -> Self {
        if plan.steps == 0 {
            plan.steps = if plan.block > 0 { plan.block } else { answer_len.max(1) as u32 };
        }
        let prompt_len = prompt.len();
        let mut tokens = prompt.to_vec();
        tokens.resize(prompt_len + answer_len, plan.mask);
        let mut c = Self {
            prompt_len,
            open: 0..0,
            step: 0,
            schedule: Vec::new(),
            edit_rounds: 0,
            phase: Phase::Denoise,
            forward_positions: 0,
            steps_taken: 0,
            tokens,
            plan,
        };
        let first = if c.plan.cached_blocks {
            prompt_len / c.plan.block as usize * c.plan.block as usize
        } else {
            prompt_len
        };
        if c.plan.cached_blocks && first > 0 {
            c.phase = Phase::Prefill(0);
        }
        c.open_block(first);
        if answer_len == 0 {
            c.phase = Phase::Done;
        }
        c
    }

    fn open_block(&mut self, start: usize) {
        let end = if self.plan.block == 0 {
            self.tokens.len()
        } else {
            (start + self.plan.block as usize).min(self.tokens.len())
        };
        self.open = start..end;
        self.step = 0;
        self.edit_rounds = 0;
        let masked = self.masked_in_open().count();
        self.schedule = match self.plan.pace {
            Pace::Even => {
                let steps = self.plan.steps as usize;
                (0..steps).map(|i| masked / steps + usize::from(i < masked % steps)).collect()
            }
            Pace::Timestep { .. } => Vec::new(),
        };
    }

    fn masked_in_open(&self) -> impl Iterator<Item = usize> + '_ {
        self.open.clone().filter(|&i| self.tokens[i] == self.plan.mask)
    }

    /// Positions written by the model in the open block, which editing may
    /// rewrite.
    fn editable_in_open(&self) -> impl Iterator<Item = usize> + '_ {
        self.open
            .clone()
            .filter(|&i| i >= self.prompt_len && self.tokens[i] != self.plan.mask)
    }

    pub fn is_done(&self) -> bool {
        self.phase == Phase::Done
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Forward positions run so far.
    pub fn forward_positions(&self) -> u64 {
        self.forward_positions
    }

    /// Denoising steps taken so far.
    pub fn steps_taken(&self) -> u32 {
        self.steps_taken
    }

    /// The answer positions that can no longer change: revealed, left to
    /// right without a gap, and outside a block that editing may still
    /// rewrite.
    pub fn settled(&self) -> &[i32] {
        let limit = if self.plan.edit_threshold.is_some() && self.phase != Phase::Done {
            self.open.start.max(self.prompt_len)
        } else {
            self.tokens.len()
        };
        let answer = &self.tokens[self.prompt_len..limit];
        let n = answer.iter().position(|&t| t == self.plan.mask).unwrap_or(answer.len());
        &answer[..n]
    }

    /// Stop decoding: what is settled is the answer.
    pub fn finish(&mut self) {
        self.phase = Phase::Done;
    }

    /// The next forward pass, or `None` once the canvas is done.
    pub fn next_chunk(&self) -> Option<Chunk> {
        let block = self.plan.block as usize;
        match self.phase {
            Phase::Done => None,
            Phase::Prefill(at) => Some(Chunk {
                start: at,
                tokens: self.tokens[at..at + block].to_vec(),
                rows: Vec::new(),
                clear_from: Some(at),
            }),
            Phase::Commit => Some(Chunk {
                start: self.open.start,
                tokens: self.tokens[self.open.clone()].to_vec(),
                rows: Vec::new(),
                clear_from: Some(self.open.start),
            }),
            Phase::Denoise => {
                let targets = self.targets();
                if self.plan.cached_blocks {
                    let start = self.open.start;
                    Some(Chunk {
                        start,
                        tokens: self.tokens[self.open.clone()].to_vec(),
                        rows: targets.iter().map(|&p| p - start).collect(),
                        clear_from: Some(start),
                    })
                } else {
                    let rows = targets
                        .iter()
                        .map(|&p| if self.plan.shift_logits { p.saturating_sub(1) } else { p })
                        .collect();
                    Some(Chunk { start: 0, tokens: self.tokens.clone(), rows, clear_from: None })
                }
            }
        }
    }

    /// The positions a denoising step predicts: the open block's masks, then
    /// its editable tokens when editing is on.
    fn targets(&self) -> Vec<usize> {
        let mut t: Vec<usize> = self.masked_in_open().collect();
        if self.plan.edit_threshold.is_some() {
            t.extend(self.editable_in_open());
        }
        t
    }

    /// How many masked positions this step reveals.
    fn share(&self, masked: usize) -> usize {
        let steps = self.plan.steps;
        if self.step + 1 >= steps {
            return masked;
        }
        match self.plan.pace {
            Pace::Even => self.schedule.get(self.step as usize).copied().unwrap_or(masked).min(masked),
            Pace::Timestep { eps } => {
                let t = 1.0 - self.step as f32 / steps as f32 * (1.0 - eps);
                let s = 1.0 - (self.step + 1) as f32 / steps as f32 * (1.0 - eps);
                (masked as f32 * (1.0 - s / t)) as usize
            }
        }
    }

    /// Take the logits of the last [`next_chunk`](Self::next_chunk), one row
    /// per entry of its `rows`, and advance.
    pub fn apply(&mut self, rows: &[&[f32]], sampling: &RowSampling, rng: &mut Rng) {
        let block = self.plan.block as usize;
        match self.phase {
            Phase::Done => {}
            Phase::Prefill(at) => {
                self.forward_positions += block as u64;
                let next = at + block;
                self.phase = if next >= self.open.start { Phase::Denoise } else { Phase::Prefill(next) };
            }
            Phase::Commit => {
                self.forward_positions += self.open.len() as u64;
                let next = self.open.end;
                self.open_block(next);
                self.phase = Phase::Denoise;
            }
            Phase::Denoise => {
                self.forward_positions +=
                    if self.plan.cached_blocks { self.open.len() } else { self.tokens.len() } as u64;
                self.steps_taken += 1;
                self.denoise(rows, sampling, rng);
            }
        }
    }

    fn denoise(&mut self, rows: &[&[f32]], sampling: &RowSampling, rng: &mut Rng) {
        let targets = self.targets();
        debug_assert_eq!(rows.len(), targets.len());
        let masked: Vec<usize> = self.masked_in_open().collect();
        let picks: Vec<Pick> = rows.iter().map(|r| pick(r, sampling, rng)).collect();
        let (mask_picks, edit_picks) = picks.split_at(masked.len());

        // Reveal.
        let share = self.share(masked.len());
        let score = |p: &Pick, rng: &mut Rng| match self.plan.unmask {
            Unmask::Confidence => p.prob,
            Unmask::Margin => p.margin,
            Unmask::Entropy => -p.entropy,
            Unmask::Random => rng.unit(),
        };
        let mut ranked: Vec<(f32, usize)> = mask_picks.iter().enumerate().map(|(i, p)| (score(p, rng), i)).collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let above = self
            .plan
            .threshold
            .map_or(0, |th| mask_picks.iter().filter(|p| p.prob > th).count());
        let reveal: Vec<usize> = if self.plan.threshold.is_some() && above >= share {
            (0..masked.len()).filter(|&i| mask_picks[i].prob > self.plan.threshold.unwrap_or(1.0)).collect()
        } else {
            ranked.iter().take(share).map(|&(_, i)| i).collect()
        };

        // Edit tokens revealed by earlier steps.
        let mut edited = false;
        if let Some(th) = self.plan.edit_threshold {
            let editable: Vec<usize> = self.editable_in_open().collect();
            for (i, &pos) in editable.iter().enumerate() {
                let p = &edit_picks[i];
                if p.token != self.tokens[pos] && p.prob > th {
                    self.tokens[pos] = p.token;
                    edited = true;
                }
            }
        }
        for i in reveal {
            self.tokens[masked[i]] = mask_picks[i].token;
        }
        self.step += 1;

        let masks_left = self.masked_in_open().next().is_some();
        if masks_left {
            return;
        }
        if self.plan.edit_threshold.is_some() && edited && self.edit_rounds < self.plan.steps {
            self.edit_rounds += 1;
            return;
        }
        // The open block is final.
        if self.open.end >= self.tokens.len() {
            self.phase = Phase::Done;
        } else if self.plan.cached_blocks {
            self.phase = Phase::Commit;
        } else {
            let next = self.open.end;
            self.open_block(next);
        }
    }
}

/// A uniform-noise model's decoding plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoisePlan {
    /// Positions per canvas.
    pub canvas: u32,
    /// Most steps one canvas takes.
    pub max_steps: u32,
    /// Sampling temperature at the last step and at the first, annealed
    /// linearly between them.
    pub t_min: f32,
    pub t_max: f32,
    /// Entropy (in nats, summed over positions) a step may accept: positions
    /// are kept lowest-entropy first until the next one would exceed it.
    pub entropy_bound: f32,
    /// Steps the most likely canvas must hold unchanged to count as stable.
    pub stability: u32,
    /// Mean entropy per position below which a stable canvas is final.
    pub confidence: f32,
}

/// One position of a uniform-noise step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseRow {
    pub argmax: i32,
    pub sampled: i32,
    pub entropy: f32,
}

/// Read one canvas position at `temp_inv` (one over the temperature): its
/// most likely token, a token drawn with the uniform `u`, and its entropy,
/// over the whole vocabulary.
pub fn noise_row(logits: &[f32], temp_inv: f32, u: f32) -> NoiseRow {
    let mut m = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (v, &l) in logits.iter().enumerate() {
        let z = l * temp_inv;
        if z > m {
            m = z;
            argmax = v;
        }
    }
    let mut z_sum = 0.0f32;
    for &l in logits {
        z_sum += (l * temp_inv - m).exp();
    }
    let target = u * z_sum;
    let mut cum = 0.0f32;
    let mut entropy = 0.0f32;
    let mut sampled = None;
    for (v, &l) in logits.iter().enumerate() {
        let e = (l * temp_inv - m).exp();
        let p = e / z_sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        cum += e;
        if sampled.is_none() && cum >= target {
            sampled = Some(v);
        }
    }
    NoiseRow {
        argmax: argmax as i32,
        sampled: sampled.unwrap_or(logits.len() - 1) as i32,
        entropy,
    }
}

/// A uniform-noise sequence in flight.
#[derive(Debug, Clone)]
pub struct NoiseCanvas {
    plan: NoisePlan,
    n_vocab: u32,
    prompt_len: usize,
    /// The prompt and every finished canvas.
    prefix: Vec<i32>,
    /// The canvas being denoised, as fed to the next pass.
    current: Vec<i32>,
    argmax: Vec<i32>,
    previous: Option<Vec<i32>>,
    held: u32,
    step: u32,
    /// One over the temperature of the last step, which conditioned it.
    prev_temp_inv: f32,
    /// Answer tokens that are final.
    answer: Vec<i32>,
    budget: usize,
    done: bool,
    forward_positions: u64,
    steps_taken: u32,
}

/// What a finished uniform-noise canvas holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanvasEnd {
    /// Still denoising.
    Open,
    /// A canvas finished; its tokens joined the answer.
    Closed,
}

impl NoiseCanvas {
    pub fn new(plan: NoisePlan, n_vocab: u32, prompt: &[i32], budget: usize, rng: &mut Rng) -> Self {
        let mut c = Self {
            plan,
            n_vocab,
            prompt_len: prompt.len(),
            prefix: prompt.to_vec(),
            current: Vec::new(),
            argmax: Vec::new(),
            previous: None,
            held: 0,
            step: 0,
            prev_temp_inv: 1.0,
            answer: Vec::new(),
            budget,
            done: budget == 0,
            forward_positions: 0,
            steps_taken: 0,
        };
        c.fresh_canvas(rng);
        c
    }

    fn fresh_canvas(&mut self, rng: &mut Rng) {
        let n = self.plan.canvas as usize;
        self.current = (0..n).map(|_| rng.below(self.n_vocab) as i32).collect();
        self.argmax = vec![0; n];
        self.previous = None;
        self.held = 0;
        self.step = 0;
        self.prev_temp_inv = 1.0;
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// Answer tokens that are final.
    pub fn settled(&self) -> &[i32] {
        &self.answer
    }

    /// Positions per canvas.
    pub fn canvas_len(&self) -> usize {
        self.plan.canvas as usize
    }

    pub fn finish(&mut self) {
        self.done = true;
    }

    pub fn forward_positions(&self) -> u64 {
        self.forward_positions
    }

    pub fn steps_taken(&self) -> u32 {
        self.steps_taken
    }

    /// Positions the next pass reads: the prefix, then the canvas. The
    /// canvas rows are the last `canvas` positions.
    pub fn next_input(&self) -> Option<Vec<i32>> {
        if self.done {
            return None;
        }
        let mut t = self.prefix.clone();
        t.extend_from_slice(&self.current);
        Some(t)
    }

    /// Whether the next pass conditions on the previous step's logits, and
    /// the temperature those were read at.
    pub fn conditioning(&self) -> (bool, f32) {
        (self.step > 0, self.prev_temp_inv)
    }

    /// One over the temperature of the next step.
    pub fn temp_inv(&self) -> f32 {
        let s = self.plan.max_steps.max(1);
        let cur = s - self.step.min(s - 1);
        let t = self.plan.t_min + (self.plan.t_max - self.plan.t_min) * (cur as f32 / s as f32);
        1.0 / t
    }

    /// Draw the uniforms and re-noise tokens a step uses, in a fixed order so
    /// a seed reproduces the answer however the rows are computed.
    pub fn draws(&self, rng: &mut Rng) -> (Vec<f32>, Vec<i32>) {
        let n = self.plan.canvas as usize;
        let mut u = Vec::with_capacity(n);
        let mut renoise = Vec::with_capacity(n);
        for _ in 0..n {
            u.push(rng.unit());
            renoise.push(rng.below(self.n_vocab) as i32);
        }
        (u, renoise)
    }

    /// Take a step's rows (one per canvas position, from [`noise_row`] at
    /// [`temp_inv`](Self::temp_inv) with the uniforms of
    /// [`draws`](Self::draws)) and advance.
    pub fn apply(&mut self, rows: &[NoiseRow], renoise: &[i32], is_end: impl Fn(i32) -> bool, rng: &mut Rng) -> CanvasEnd {
        let n = self.plan.canvas as usize;
        debug_assert_eq!(rows.len(), n);
        self.forward_positions += (self.prefix.len() + n) as u64;
        self.steps_taken += 1;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| rows[a].entropy.total_cmp(&rows[b].entropy).then(a.cmp(&b)));
        let mut accepted = vec![false; n];
        let mut cum = 0.0f64;
        for &pos in &order {
            let before = cum;
            cum += f64::from(rows[pos].entropy);
            if before <= f64::from(self.plan.entropy_bound) {
                accepted[pos] = true;
            }
        }
        let mut entropy_sum = 0.0f32;
        for pos in 0..n {
            self.current[pos] = if accepted[pos] { rows[pos].sampled } else { renoise[pos] };
            self.argmax[pos] = rows[pos].argmax;
            entropy_sum += rows[pos].entropy;
        }
        self.held = if self.previous.as_deref() == Some(&self.argmax[..]) { self.held + 1 } else { 0 };
        self.previous = Some(self.argmax.clone());
        self.prev_temp_inv = self.temp_inv();
        self.step += 1;
        let stable = self.held >= self.plan.stability && entropy_sum / (n as f32) < self.plan.confidence;
        if !stable && self.step < self.plan.max_steps {
            return CanvasEnd::Open;
        }
        // The canvas is final: its most likely tokens, up to an end token.
        let canvas = std::mem::take(&mut self.argmax);
        let end = canvas.iter().position(|&t| is_end(t));
        let kept = &canvas[..end.unwrap_or(n)];
        let room = self.budget.saturating_sub(self.answer.len());
        self.answer.extend_from_slice(&kept[..kept.len().min(room)]);
        if end.is_some() || self.answer.len() >= self.budget {
            self.done = true;
        } else {
            self.prefix.extend_from_slice(&canvas);
            self.fresh_canvas(rng);
        }
        CanvasEnd::Closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASK: i32 = 9;

    fn greedy() -> RowSampling {
        RowSampling { temperature: 0.0, top_k: 1000, top_p: 1.0, forbid: Some(MASK) }
    }

    /// Logits over a 10-token vocabulary preferring `token` with `strength`.
    fn row(token: i32, strength: f32) -> Vec<f32> {
        let mut r = vec![0.0; 10];
        r[token as usize] = strength;
        r
    }

    fn plan(block: u32, steps: u32) -> MaskedPlan {
        MaskedPlan {
            mask: MASK,
            block,
            steps,
            unmask: Unmask::Confidence,
            pace: Pace::Even,
            threshold: None,
            edit_threshold: None,
            shift_logits: false,
            cached_blocks: false,
        }
    }

    /// Run a canvas to the end against a fixed answer: each position's
    /// logits prefer its answer token, more strongly further right.
    fn run(mut c: MaskedCanvas, answer: &[i32]) -> (MaskedCanvas, Vec<Vec<usize>>) {
        let mut rng = Rng::new(1);
        let mut chunks = Vec::new();
        while let Some(ch) = c.next_chunk() {
            let owned: Vec<Vec<f32>> = ch
                .rows
                .iter()
                .map(|&r| {
                    let pos = ch.start + r;
                    let pos = if c.plan.shift_logits { pos + 1 } else { pos };
                    row(answer[pos - c.prompt_len()], 1.0 + pos as f32)
                })
                .collect();
            let rows: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
            chunks.push(ch.rows.iter().map(|&r| ch.start + r).collect());
            c.apply(&rows, &greedy(), &mut rng);
        }
        (c, chunks)
    }

    #[test]
    fn a_greedy_pick_is_the_most_likely_token_with_its_probability() {
        let p = pick(&row(3, 2.0), &greedy(), &mut Rng::new(0));
        assert_eq!(p.token, 3);
        let expected = 2f32.exp() / (2f32.exp() + 8.0);
        assert!((p.prob - expected).abs() < 1e-6);
        assert!(p.margin > 0.0 && p.entropy > 0.0);
    }

    #[test]
    fn the_mask_is_never_predicted() {
        let p = pick(&row(MASK, 50.0), &greedy(), &mut Rng::new(0));
        assert_ne!(p.token, MASK);
    }

    #[test]
    fn one_candidate_is_certain() {
        let s = RowSampling { top_k: 1, ..greedy() };
        let p = pick(&row(4, 0.5), &s, &mut Rng::new(0));
        assert_eq!((p.token, p.prob, p.entropy), (4, 1.0, 0.0));
    }

    #[test]
    fn sampling_follows_the_seed() {
        let s = RowSampling { temperature: 1.0, ..greedy() };
        let r = row(2, 1.0);
        let a: Vec<i32> = { let mut g = Rng::new(7); (0..32).map(|_| pick(&r, &s, &mut g).token).collect() };
        let b: Vec<i32> = { let mut g = Rng::new(7); (0..32).map(|_| pick(&r, &s, &mut g).token).collect() };
        assert_eq!(a, b);
        assert!(a.iter().any(|&t| t != 2), "a temperature of 1 over these logits samples more than the top token");
    }

    #[test]
    fn blocks_are_revealed_left_to_right_one_position_per_step() {
        let answer = [1, 2, 3, 4, 5, 6, 7, 8];
        let (c, chunks) = run(MaskedCanvas::new(plan(4, 4), &[0, 0], 8), &answer);
        assert!(c.is_done());
        assert_eq!(c.settled(), &answer);
        assert_eq!(c.steps_taken(), 8);
        // Each step predicts only the open block's remaining masks.
        // The surest position goes first: here the rightmost, so the last
        // step of the block is left with the leftmost.
        assert_eq!(chunks[0], vec![2, 3, 4, 5]);
        assert_eq!(chunks[1], vec![2, 3, 4]);
        assert_eq!(chunks[3], vec![2]);
        assert_eq!(chunks[4], vec![6, 7, 8, 9]);
        // Without a cache every step reads the whole canvas.
        assert_eq!(c.forward_positions(), 8 * 10);
    }

    #[test]
    fn the_surest_position_is_revealed_first() {
        let mut c = MaskedCanvas::new(plan(0, 3), &[0], 3);
        let ch = c.next_chunk().unwrap();
        assert_eq!(ch.rows, vec![1, 2, 3]);
        let rows = [row(5, 1.0), row(6, 3.0), row(7, 2.0)];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        c.apply(&refs, &greedy(), &mut Rng::new(0));
        assert_eq!(c.tokens(), &[0, MASK, 6, MASK]);
        // Nothing is settled until the first answer position is.
        assert!(c.settled().is_empty());
    }

    #[test]
    fn the_timestep_pace_leaves_the_rest_to_the_last_step() {
        let mut p = plan(0, 4);
        p.pace = Pace::Timestep { eps: 1e-3 };
        let (c, chunks) = run(MaskedCanvas::new(p, &[0], 8), &[1; 8]);
        assert!(c.is_done());
        assert!(chunks.len() <= 4);
        assert_eq!(c.settled(), &[1; 8]);
    }

    #[test]
    fn shifted_logits_read_the_row_before_each_position() {
        let mut p = plan(0, 1);
        p.shift_logits = true;
        let c = MaskedCanvas::new(p, &[0, 0], 2);
        assert_eq!(c.next_chunk().unwrap().rows, vec![1, 2]);
        let (c, _) = run(c, &[3, 4]);
        assert_eq!(c.settled(), &[3, 4]);
    }

    #[test]
    fn above_the_threshold_a_step_reveals_more_than_its_share() {
        let mut p = plan(4, 4);
        p.threshold = Some(0.5);
        // Strong logits: every position is above 0.5, so one step fills the block.
        let mut c = MaskedCanvas::new(p, &[0], 4);
        let rows: Vec<Vec<f32>> = (0..4).map(|i| row(i + 1, 10.0)).collect();
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        c.apply(&refs, &greedy(), &mut Rng::new(0));
        assert_eq!(c.settled(), &[1, 2, 3, 4]);
        assert!(c.is_done());
    }

    #[test]
    fn block_causal_decoding_prefills_whole_prompt_blocks_then_commits_each_block() {
        let mut p = plan(4, 4);
        p.cached_blocks = true;
        p.threshold = Some(0.5);
        // Six prompt tokens: one whole block is prefilled, the open block
        // holds the last two prompt tokens and the first two answer slots.
        let c = MaskedCanvas::new(p, &[1, 1, 1, 1, 1, 1], 6);
        let first = c.next_chunk().unwrap();
        assert_eq!((first.start, first.tokens.len(), first.rows.len(), first.clear_from), (0, 4, 0, Some(0)));
        let (c, chunks) = run(c, &[2, 3, 4, 5, 6, 7]);
        assert_eq!(c.settled(), &[2, 3, 4, 5, 6, 7]);
        // prefill, block [4,8) in one step, commit, block [8,12) in one step.
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[1], vec![6, 7]);
        assert_eq!(chunks[3], vec![8, 9, 10, 11]);
        // Each pass reads one block, never the whole canvas.
        assert_eq!(c.forward_positions(), 4 * 4);
    }

    #[test]
    fn editing_rewrites_a_revealed_token_and_holds_the_block_back_until_it_settles() {
        let mut p = plan(2, 2);
        p.edit_threshold = Some(0.5);
        let mut c = MaskedCanvas::new(p, &[0], 4);
        let mut rng = Rng::new(0);
        // Step 1: reveal position 2 (stronger) as 5.
        let r1 = [row(3, 1.0), row(5, 2.0)];
        c.apply(&r1.iter().map(Vec::as_slice).collect::<Vec<_>>(), &greedy(), &mut rng);
        assert_eq!(c.tokens()[1..3], [MASK, 5]);
        assert!(c.settled().is_empty(), "the open block is not streamed while it can be edited");
        // Step 2: reveal position 1, and the model now prefers 6 at position 2.
        assert_eq!(c.next_chunk().unwrap().rows, vec![1, 2]);
        let r2 = [row(3, 1.0), row(6, 10.0)];
        c.apply(&r2.iter().map(Vec::as_slice).collect::<Vec<_>>(), &greedy(), &mut rng);
        assert_eq!(c.tokens()[1..3], [3, 6]);
        // An edit keeps the block open for another look.
        assert!(c.settled().is_empty());
        let r3 = [row(3, 10.0), row(6, 10.0)];
        c.apply(&r3.iter().map(Vec::as_slice).collect::<Vec<_>>(), &greedy(), &mut rng);
        assert_eq!(c.settled(), &[3, 6]);
    }

    #[test]
    fn impossible_plans_are_refused() {
        let mut p = plan(0, 4);
        p.cached_blocks = true;
        assert!(p.check().is_err());
        let mut p = plan(4, 4);
        p.cached_blocks = true;
        p.shift_logits = true;
        assert!(p.check().is_err());
        let mut p = plan(0, 4);
        p.edit_threshold = Some(0.9);
        assert!(p.check().is_err());
        assert!(plan(4, 0).check().is_ok(), "no step count means one step per position");
        assert!(plan(4, 4).check().is_ok());
    }

    #[test]
    fn a_noise_row_reads_argmax_entropy_and_a_draw() {
        let r = noise_row(&row(3, 5.0), 1.0, 0.0);
        assert_eq!(r.argmax, 3);
        assert_eq!(r.sampled, 0, "a draw of zero takes the first token with any mass");
        let r = noise_row(&row(3, 50.0), 1.0, 0.5);
        assert_eq!(r.sampled, 3);
        assert!(r.entropy < 1e-3);
    }

    fn noise_plan() -> NoisePlan {
        NoisePlan { canvas: 4, max_steps: 8, t_min: 0.4, t_max: 0.8, entropy_bound: 0.1, stability: 1, confidence: 0.005 }
    }

    #[test]
    fn a_confident_stable_canvas_closes_and_an_end_token_finishes_the_answer() {
        let mut rng = Rng::new(3);
        let mut c = NoiseCanvas::new(noise_plan(), 10, &[1, 2], 16, &mut rng);
        assert_eq!(c.next_input().unwrap().len(), 6);
        assert_eq!(c.conditioning().0, false);
        let rows: Vec<NoiseRow> = [4, 5, 0, 6].iter().map(|&t| NoiseRow { argmax: t, sampled: t, entropy: 0.0 }).collect();
        let (_, renoise) = c.draws(&mut rng);
        assert_eq!(c.apply(&rows, &renoise, |t| t == 0, &mut rng), CanvasEnd::Open, "one step is not yet stable");
        assert!(c.conditioning().0);
        assert_eq!(c.apply(&rows, &renoise, |t| t == 0, &mut rng), CanvasEnd::Closed);
        assert_eq!(c.settled(), &[4, 5]);
        assert!(c.is_done());
    }

    #[test]
    fn an_uncertain_canvas_runs_its_steps_then_the_next_canvas_follows_it() {
        let mut rng = Rng::new(3);
        let mut c = NoiseCanvas::new(noise_plan(), 10, &[1], 8, &mut rng);
        let rows: Vec<NoiseRow> = (0..4).map(|i| NoiseRow { argmax: 4 + i, sampled: 4 + i, entropy: 1.0 }).collect();
        let mut closed = 0;
        for _ in 0..8 {
            let (_, renoise) = c.draws(&mut rng);
            if c.apply(&rows, &renoise, |_| false, &mut rng) == CanvasEnd::Closed {
                closed += 1;
            }
        }
        assert_eq!(closed, 1);
        assert_eq!(c.settled(), &[4, 5, 6, 7]);
        // The finished canvas joined the prefix of the next.
        assert_eq!(c.next_input().unwrap().len(), 1 + 4 + 4);
        assert!(!c.is_done());
    }

    #[test]
    fn the_entropy_bound_keeps_the_surest_positions_and_renoises_the_rest() {
        let mut rng = Rng::new(3);
        let mut c = NoiseCanvas::new(noise_plan(), 10, &[1], 8, &mut rng);
        let rows = [
            NoiseRow { argmax: 4, sampled: 4, entropy: 0.05 },
            NoiseRow { argmax: 5, sampled: 5, entropy: 2.0 },
            NoiseRow { argmax: 6, sampled: 6, entropy: 0.0 },
            NoiseRow { argmax: 7, sampled: 7, entropy: 0.08 },
        ];
        let renoise = [9, 9, 9, 9];
        c.apply(&rows, &renoise, |_| false, &mut rng);
        // Sorted by entropy: 0.0, 0.05, 0.08 (sum before it 0.05 <= 0.1), then 2.0 (sum before 0.13).
        assert_eq!(&c.next_input().unwrap()[1..], &[4, 9, 6, 7]);
    }
}
