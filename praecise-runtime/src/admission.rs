//! Admission and queueing for a bandwidth-bound decoder.
//!
//! Backend-agnostic: nothing here touches a model. It is the arithmetic a
//! serving layer needs to decide, for each arriving request, whether to run it
//! now, hold it until a slot frees, or turn it away — and to be honest about
//! how long "until a slot frees" is.
//!
//! ## What was wrong before this existed
//!
//! The host that first used Praecise admitted on a *predicted end-to-end
//! time* against a fixed 30-second deadline, and it predicted that time from
//! a single learned rate of milliseconds per kilotoken **of prompt**. On a
//! decoder running at ~30 tok/s that fails in three independent ways, and
//! all three were observed on a DGX Spark serving a 125B MoE:
//!
//! 1. **Output was never in the estimate.** A 20-token prompt that generates
//!    1,000 tokens costs 32 s of decode. Charged as 20 tokens of prompt, one
//!    such request taught the model "128 s per kilotoken". Every request after
//!    it — including a 200-token prompt asking for 200 tokens — was predicted
//!    at 30–50 s and refused.
//! 2. **The deadline was fixed while the work was not.** A caller asking for
//!    4,096 tokens at 30 tok/s has asked for a two-minute answer. Refusing it
//!    because thirty is a nice number does not protect anyone; it converts
//!    every long generation into an error.
//! 3. **Refusal instead of a queue.** The batching engine underneath holds a
//!    channel and admits into slots as they free. Turning callers away above
//!    it, with a "retry after" they cannot act on, threw away the queue that
//!    already existed.
//!
//! ## The model this module uses instead
//!
//! - **Cost has two terms.** Prefill is priced per prompt token and decode per
//!   generated token, learned separately, because on this hardware they differ
//!   by more than an order of magnitude (measured on GB10: ~730 tok/s prefill,
//!   ~31 tok/s decode for the same model). A request's service time is
//!   `prompt × prefill_rate + expected_output × decode_rate`.
//! - **The SLO is on the wait, not the answer.** Interactive traffic gets a
//!   budget for how long it may sit in the queue before its first token; the
//!   generation itself runs for as long as `max_tokens` asks. This is the
//!   convention every large serving deployment uses (time-to-first-token and
//!   queue timeout, never a cap on total generation) and it is the one that
//!   makes long completions possible at all on a slow decoder.
//! - **Queue, then refuse.** When every slot is busy the answer is an ETA,
//!   derived from the remaining work of the requests already running, and the
//!   host waits that long. Only an ETA beyond the class budget, or a queue
//!   already at its depth limit, is a refusal — and the refusal says why.
//!
//! The host owns locking, timers and the actual wait; this module is pure
//! state and arithmetic so it can be tested with the numbers above.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

/// Which kind of caller is waiting, and therefore how long they will tolerate
/// sitting in the queue before their first token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// A person or an agent loop is blocked on the answer. Short queue budget.
    Interactive,
    /// Throughput work. Long queue budget; yields to interactive traffic.
    Batch,
}

/// The size of a request as far as admission cares: how much must be read,
/// and how much may be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Shape {
    /// Tokens in the prompt. A byte-length estimate is fine; this drives an
    /// ETA, not a bill.
    pub prompt_tokens: u64,
    /// The most tokens the caller allowed the model to generate.
    pub max_output_tokens: u64,
}

impl Shape {
    pub fn new(prompt_tokens: u64, max_output_tokens: u64) -> Self {
        Self { prompt_tokens, max_output_tokens }
    }
}

/// What a completed request actually cost, fed back to the cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    pub prompt_tokens: u64,
    /// Tokens the model produced. Zero means the request did not run and
    /// teaches nothing.
    pub generated_tokens: u64,
    /// The most tokens the caller allowed (`Shape::max_output_tokens`). With
    /// `generated_tokens` this teaches how much of a budget requests really
    /// use, which is what turns a ceiling into an expected length.
    pub budget_tokens: u64,
    /// Wall time from start of service (not arrival) to completion.
    pub service_ms: u64,
    /// Time to the first generated token, when the host can measure it. This
    /// is what separates the two rates cleanly; without it the prefill term is
    /// only inferred.
    pub first_token_ms: Option<u64>,
    /// How many requests shared the decoder while this one ran, including
    /// itself. Batched decode is slower per request than solo decode, so the
    /// learned decode rate is normalised to a solo-equivalent before storage
    /// and re-scaled at prediction time.
    pub concurrency: u32,
}

/// Per-model cost model: microseconds per token, for prefill and decode.
///
/// Stored per token rather than per request so that a value learned from a
/// long agent turn transfers to a short chat message. Stored as microseconds
/// so that integer arithmetic keeps four digits of precision on a decode step
/// that costs ~32,000 µs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostModel {
    prefill_us_per_tok: u64,
    /// Solo-equivalent decode cost per token.
    decode_us_per_tok: u64,
    /// Share of the caller's output budget that requests actually use, in
    /// percent. `max_output_tokens` is a ceiling and most generations stop
    /// well short of it; pricing every request at its ceiling makes a
    /// 2,000-token budget look like a minute of decode when the answer takes
    /// ten seconds, and refuses the caller behind it for a wait that never
    /// happens. Learned per model, since chat and agent traffic differ.
    budget_use_pct: u64,
    samples: u32,
}

impl CostModel {
    /// Prior for prefill: 500 tok/s. Deliberately pessimistic for a GPU; a
    /// prior that is too optimistic under-estimates ETAs and makes the host
    /// promise waits it cannot keep.
    pub const DEFAULT_PREFILL_US_PER_TOK: u64 = 2_000;
    /// Prior for decode: 25 tok/s. Same reasoning.
    pub const DEFAULT_DECODE_US_PER_TOK: u64 = 40_000;
    /// Prior for budget use: the whole budget. Pessimistic on purpose, for
    /// the same reason as the rates — an ETA promised short and kept long
    /// is worse than one promised long. The truth arrives with the first
    /// completions.
    pub const DEFAULT_BUDGET_USE_PCT: u64 = 100;
    /// Never predict less output than this, however small the learned share:
    /// a run of one-word answers must not make the next real question look
    /// free.
    const MIN_EXPECTED_OUTPUT_TOKENS: u64 = 32;
    /// Weight of the newest sample in the moving average. A quarter converges
    /// in a handful of requests without one outlier owning the estimate.
    const NEW_SAMPLE_PCT: u64 = 25;

    pub fn new() -> Self {
        Self {
            prefill_us_per_tok: Self::DEFAULT_PREFILL_US_PER_TOK,
            decode_us_per_tok: Self::DEFAULT_DECODE_US_PER_TOK,
            budget_use_pct: Self::DEFAULT_BUDGET_USE_PCT,
            samples: 0,
        }
    }

    /// Seed from a known throughput instead of the prior — e.g. a benchmark
    /// the operator ran on this hardware.
    pub fn from_rates(prefill_tok_per_s: u64, decode_tok_per_s: u64) -> Self {
        Self {
            prefill_us_per_tok: 1_000_000 / prefill_tok_per_s.max(1),
            decode_us_per_tok: 1_000_000 / decode_tok_per_s.max(1),
            budget_use_pct: Self::DEFAULT_BUDGET_USE_PCT,
            samples: 1,
        }
    }

    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Learned solo decode throughput, tokens per second.
    pub fn decode_tok_per_s(&self) -> u64 {
        1_000_000 / self.decode_us_per_tok.max(1)
    }

    /// Learned prefill throughput, tokens per second.
    pub fn prefill_tok_per_s(&self) -> u64 {
        1_000_000 / self.prefill_us_per_tok.max(1)
    }

    /// Learned share of the output budget that requests use, in percent.
    pub fn budget_use_pct(&self) -> u64 {
        self.budget_use_pct
    }

    fn ema(old: u64, new: u64, first: bool) -> u64 {
        if first {
            new
        } else {
            (new * Self::NEW_SAMPLE_PCT + old * (100 - Self::NEW_SAMPLE_PCT)) / 100
        }
    }

    /// Fold a completed request into the estimate.
    ///
    /// Only a request that generated something teaches anything; a refused or
    /// aborted request carries no service time and is ignored.
    pub fn observe(&mut self, o: Observation) {
        if o.generated_tokens == 0 || o.service_ms == 0 {
            return;
        }
        let first = self.samples == 0;
        let conc = u64::from(o.concurrency.max(1));

        // Prefill: measured directly when the host reports first-token time,
        // otherwise left alone. Inferring it from a total that is 95% decode
        // would only add noise to the smaller term.
        let prefill_ms = match o.first_token_ms {
            Some(ttft) if o.prompt_tokens > 0 => {
                let us_per_tok = ttft.saturating_mul(1_000) / o.prompt_tokens;
                if us_per_tok > 0 {
                    self.prefill_us_per_tok = Self::ema(self.prefill_us_per_tok, us_per_tok, first);
                }
                ttft
            }
            _ => self.prefill_us_per_tok.saturating_mul(o.prompt_tokens) / 1_000,
        };

        // Decode: whatever was not prefill, per generated token, normalised to
        // a solo-equivalent by the concurrency it ran under. At c=2 on a
        // bandwidth-bound decoder each request sees roughly half the solo
        // rate, so dividing the per-token wall time by c recovers the solo
        // figure this model stores.
        let decode_ms = o.service_ms.saturating_sub(prefill_ms);
        let us_per_tok = decode_ms.saturating_mul(1_000) / o.generated_tokens / conc;
        if us_per_tok > 0 {
            self.decode_us_per_tok = Self::ema(self.decode_us_per_tok, us_per_tok, first);
        }

        // Budget use: how much of what the caller allowed was actually
        // generated. A request that hit its ceiling counts as 100%, not more.
        if o.budget_tokens > 0 {
            let pct = (o.generated_tokens.saturating_mul(100) / o.budget_tokens).min(100);
            self.budget_use_pct = Self::ema(self.budget_use_pct, pct, first);
        }
        self.samples = self.samples.saturating_add(1);
    }

    /// Predicted service time for `shape` if it ran alongside `concurrency - 1`
    /// other requests.
    ///
    /// `max_output_tokens` is an upper bound, and most generations stop
    /// early, so the decode term is priced at the share of the budget this
    /// model's requests have been using — the whole budget until something
    /// has been observed. Over-estimating an ETA costs a caller a little
    /// extra wait; under-estimating it costs them a broken promise, so the
    /// prior is the ceiling and the floor is never zero.
    pub fn predict(&self, shape: Shape, concurrency: u32) -> Estimate {
        let conc = u64::from(concurrency.max(1));
        let prefill_ms = self.prefill_us_per_tok.saturating_mul(shape.prompt_tokens) / 1_000;
        let decode_ms =
            self.decode_us_per_tok.saturating_mul(self.expected_output(shape)).saturating_mul(conc) / 1_000;
        Estimate { prefill_ms, decode_ms }
    }

    /// Tokens this request is expected to generate: its budget scaled by the
    /// learned use, floored so a run of short answers cannot make the next
    /// request look free, and never more than the budget itself.
    fn expected_output(&self, shape: Shape) -> u64 {
        let scaled = shape.max_output_tokens.saturating_mul(self.budget_use_pct) / 100;
        scaled.max(Self::MIN_EXPECTED_OUTPUT_TOKENS).min(shape.max_output_tokens)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

/// A predicted service time, split so a host can report time-to-first-token
/// separately from total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Estimate {
    pub prefill_ms: u64,
    pub decode_ms: u64,
}

impl Estimate {
    pub fn total_ms(&self) -> u64 {
        self.prefill_ms.saturating_add(self.decode_ms)
    }
}

/// How long each class may wait in the queue, and how deep the queue may get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// Queue budget for interactive callers. Thirty seconds is the point past
    /// which a person has usually given up on a blank screen — but this is a
    /// wait for the *first* token, not for the whole answer.
    pub interactive_wait: Duration,
    /// Queue budget for batch callers.
    pub batch_wait: Duration,
    /// Requests allowed to wait at once. Beyond this, refuse immediately so a
    /// burst fails fast instead of timing out one by one.
    pub max_queue_depth: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            interactive_wait: Duration::from_secs(30),
            batch_wait: Duration::from_secs(600),
            max_queue_depth: 64,
        }
    }
}

impl Policy {
    pub fn wait_budget(&self, class: Class) -> Duration {
        match class {
            Class::Interactive => self.interactive_wait,
            Class::Batch => self.batch_wait,
        }
    }
}

/// Why a request was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Every slot is busy and the queue is full. Come back after roughly one
    /// service time.
    QueueFull { depth: u32, limit: u32, retry_after_ms: u64 },
    /// A slot will free, but not within what this class will wait.
    WaitTooLong { eta_ms: u64, budget_ms: u64 },
}

impl Refusal {
    pub fn retry_after_ms(&self) -> u64 {
        match self {
            Refusal::QueueFull { retry_after_ms, .. } => *retry_after_ms,
            Refusal::WaitTooLong { eta_ms, .. } => *eta_ms,
        }
    }
}

/// The scheduler's answer for one arriving request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// A slot is free. Start now.
    Admit,
    /// Every slot is busy; one is expected to free in about `eta`. The host
    /// should hold the request and ask again — the ETA shrinks as running
    /// requests finish.
    Wait { eta: Duration, position: u32 },
    Refuse(Refusal),
}

/// Who a request is for, as far as sharing the decoder fairly is concerned.
///
/// `id` tells callers apart: an API key's id, a lease, `"anonymous"` for
/// everyone without one. `weight` is that caller's share relative to others:
/// while both are waiting, a weight-2 caller is served about twice as much as
/// a weight-1 caller. Zero is read as one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Client {
    pub id: String,
    pub weight: u32,
}

impl Client {
    pub fn new(id: impl Into<String>, weight: u32) -> Self {
        Self {
            id: id.into(),
            weight: weight.max(1),
        }
    }
}

/// Service units charged per prompt token. Prefill runs the prompt in parallel.
const W_INPUT: u64 = 1;
/// Service units charged per generated token. Decode is one step per token and
/// costs the decoder about twice what a prompt token does.
const W_OUTPUT: u64 = 2;
/// Fixed-point scale, so dividing a charge by a caller's weight keeps precision.
const SCALE: u64 = 1_000;

/// A caller holding a place in the fair queue.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Waiter {
    client: String,
    class: Class,
    arrived_ms: u64,
}

/// One request the decoder is currently serving, as the scheduler sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Running {
    shape: Shape,
    /// Milliseconds since the scheduler's epoch when service began.
    started_ms: u64,
    /// Whose request this is, when the host said. `None` is admission that
    /// named no caller, which is never charged.
    client: Option<String>,
    weight: u32,
    /// Output tokens charged in advance, reconciled at `finish`.
    charged_output: u64,
}

/// Slot accounting for one model.
///
/// Time is passed in by the host as milliseconds on any monotonic clock, so
/// this stays deterministic under test.
#[derive(Debug, Clone)]
pub struct Scheduler {
    slots: u32,
    policy: Policy,
    cost: CostModel,
    running: BTreeMap<u64, Running>,
    next_id: u64,
    waiting: u32,
    /// Callers queued by identity. Ranked on demand, never kept sorted, so a
    /// counter changing while someone waits can reorder the line.
    fair: BTreeMap<u64, Waiter>,
    next_waiter: u64,
    /// Service received per caller, in `SCALE`ths of a unit divided by the
    /// caller's weight. The lowest counter is served next.
    counters: HashMap<String, u64>,
    /// Counter of the last caller to leave, for lifting a caller that arrives
    /// when nobody else is active.
    last_departed: u64,
}

impl Scheduler {
    pub fn new(slots: u32, policy: Policy) -> Self {
        Self {
            slots: slots.max(1),
            policy,
            cost: CostModel::new(),
            running: BTreeMap::new(),
            next_id: 1,
            waiting: 0,
            fair: BTreeMap::new(),
            next_waiter: 1,
            counters: HashMap::new(),
            last_departed: 0,
        }
    }

    pub fn with_cost(mut self, cost: CostModel) -> Self {
        self.cost = cost;
        self
    }

    /// Replace the cost model in place — e.g. seeding from an operator's
    /// declared throughput after the scheduler already exists.
    pub fn set_cost(&mut self, cost: CostModel) {
        self.cost = cost;
    }

    pub fn cost(&self) -> &CostModel {
        &self.cost
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Change the slot count — e.g. after a model reload with a different
    /// `n_seq_max`. Running requests are unaffected.
    pub fn set_slots(&mut self, slots: u32) {
        self.slots = slots.max(1);
    }

    /// Replace the queue policy — e.g. a host that bounds this queue's depth
    /// below the default. Callers already waiting are unaffected.
    pub fn set_policy(&mut self, policy: Policy) {
        self.policy = policy;
    }

    pub fn running(&self) -> u32 {
        self.running.len() as u32
    }

    pub fn waiting(&self) -> u32 {
        self.waiting.saturating_add(self.fair.len() as u32)
    }

    /// Milliseconds until the `k`-th slot frees (0-indexed), from the remaining
    /// work of what is running. Requests that overran their estimate count
    /// as freeing now: the estimate was wrong, not the request.
    fn eta_ms(&self, k: u32, now_ms: u64) -> u64 {
        let conc = self.running.len() as u32;
        let mut remaining: Vec<u64> = self
            .running
            .values()
            .map(|r| {
                let predicted = self.cost.predict(r.shape, conc).total_ms();
                predicted.saturating_sub(now_ms.saturating_sub(r.started_ms))
            })
            .collect();
        remaining.sort_unstable();
        if remaining.is_empty() {
            return 0;
        }
        // Slot k frees when the k-th running request does; past the running
        // set, every further position costs one more mean service time on
        // the slot that frees first.
        let n = remaining.len() as u32;
        if k < n {
            remaining[k as usize]
        } else {
            let mean = remaining.iter().sum::<u64>() / u64::from(n);
            remaining[0].saturating_add(mean.saturating_mul(u64::from(k - n + 1)))
        }
    }

    /// Decide for a request of `shape` and `class` arriving at `now_ms`.
    ///
    /// Pure: nothing changes until the host calls [`Scheduler::start`] or
    /// [`Scheduler::enqueue`]. A host that got `Wait` should enqueue, sleep
    /// for about the ETA (or until something finishes), and decide again.
    pub fn decide(&self, class: Class, shape: Shape, now_ms: u64) -> Decision {
        let _ = shape; // shape is not yet used for placement; kept for the API
        if self.running() < self.slots {
            return Decision::Admit;
        }
        if self.waiting >= self.policy.max_queue_depth {
            let conc = self.running.len() as u32;
            let mean = self
                .running
                .values()
                .map(|r| self.cost.predict(r.shape, conc).total_ms())
                .sum::<u64>()
                / u64::from(conc.max(1));
            return Decision::Refuse(Refusal::QueueFull {
                depth: self.waiting,
                limit: self.policy.max_queue_depth,
                retry_after_ms: mean.max(1_000),
            });
        }
        let position = self.waiting;
        let eta_ms = self.eta_ms(position, now_ms);
        let budget_ms = self.policy.wait_budget(class).as_millis() as u64;
        if eta_ms > budget_ms {
            return Decision::Refuse(Refusal::WaitTooLong { eta_ms, budget_ms });
        }
        Decision::Wait { eta: Duration::from_millis(eta_ms.max(50)), position }
    }

    /// A charge of `tokens` at `per_token` units each, scaled and divided by the
    /// caller's weight.
    fn units(tokens: u64, per_token: u64, weight: u32) -> u64 {
        tokens.saturating_mul(per_token).saturating_mul(SCALE) / u64::from(weight.max(1))
    }

    /// Output the cost model expects a request of this shape to produce: the
    /// learned share of `max_tokens` callers actually use, with a floor so a
    /// short history never prices a request at nothing.
    fn expected_output(&self, shape: Shape) -> u64 {
        let pct = self.cost.budget_use_pct().clamp(1, 100);
        let learned = shape.max_output_tokens.saturating_mul(pct) / 100;
        learned
            .max(shape.max_output_tokens.min(32))
            .min(shape.max_output_tokens)
    }

    fn counter(&self, client: &str) -> u64 {
        self.counters.get(client).copied().unwrap_or(0)
    }

    fn is_active(&self, client: &str) -> bool {
        self.fair.values().any(|w| w.client == client)
            || self
                .running
                .values()
                .any(|r| r.client.as_deref() == Some(client))
    }

    /// The lowest counter among callers with work queued or running.
    fn active_min(&self) -> Option<u64> {
        self.fair
            .values()
            .map(|w| w.client.as_str())
            .chain(self.running.values().filter_map(|r| r.client.as_deref()))
            .map(|c| self.counter(c))
            .min()
    }

    /// Service a caller has received, as the fair queue counts it. For an
    /// operator asking why one caller is waiting behind another.
    pub fn service_of(&self, client: &str) -> u64 {
        self.counter(client)
    }

    /// Join the fair queue as `client`.
    ///
    /// A caller with nothing queued or running has its counter lifted to the
    /// least among callers that do, or, with nobody active, to the last one to
    /// leave. Without the lift a caller idle for an hour would come back
    /// holding an hour of unused credit and take every slot until it had spent
    /// it. The lift only raises: stepping away cannot shed service already
    /// received.
    ///
    /// Returns the handle for [`Scheduler::decide_waiter`] and
    /// [`Scheduler::leave`]. Pair every join with a leave.
    pub fn join(&mut self, client: &Client, class: Class, now_ms: u64) -> u64 {
        if !self.is_active(&client.id) {
            let floor = self.active_min().unwrap_or(self.last_departed);
            let c = self.counters.entry(client.id.clone()).or_insert(0);
            if *c < floor {
                *c = floor;
            }
        }
        let id = self.next_waiter;
        self.next_waiter += 1;
        self.fair.insert(
            id,
            Waiter {
                client: client.id.clone(),
                class,
                arrived_ms: now_ms,
            },
        );
        id
    }

    /// Give up a place in the fair queue: admitted, refused or gone.
    pub fn leave(&mut self, waiter: u64) {
        self.fair.remove(&waiter);
    }

    /// How many waiters are ahead of this one: interactive before batch, then
    /// the least service received, then arrival.
    fn rank(&self, waiter: u64) -> Option<u32> {
        let key = |id: u64, w: &Waiter| {
            let class = match w.class {
                Class::Interactive => 0u8,
                Class::Batch => 1u8,
            };
            (class, self.counter(&w.client), w.arrived_ms, id)
        };
        let mine = key(waiter, self.fair.get(&waiter)?);
        Some(
            self.fair
                .iter()
                .filter(|(id, w)| key(**id, w) < mine)
                .count() as u32,
        )
    }

    /// Decide for a waiter that has [`joined`](Scheduler::join).
    ///
    /// Admits when a slot is free and fewer waiters outrank this one than there
    /// are free slots. A caller that has had little service goes ahead of one
    /// that has had a lot, however long the heavy one has been queued, which is
    /// what stops a single steady caller holding a one-slot model against
    /// everyone else. ETA and refusal follow [`Scheduler::decide`], from this
    /// waiter's rank.
    pub fn decide_waiter(&self, waiter: u64, shape: Shape, now_ms: u64) -> Decision {
        let (Some(me), Some(rank)) = (self.fair.get(&waiter), self.rank(waiter)) else {
            return self.decide(Class::Interactive, shape, now_ms);
        };
        let free = self.slots.saturating_sub(self.running());
        if rank < free {
            return Decision::Admit;
        }
        if rank >= self.policy.max_queue_depth {
            let conc = self.running.len() as u32;
            let mean = self
                .running
                .values()
                .map(|r| self.cost.predict(r.shape, conc).total_ms())
                .sum::<u64>()
                / u64::from(conc.max(1));
            return Decision::Refuse(Refusal::QueueFull {
                depth: self.fair.len() as u32,
                limit: self.policy.max_queue_depth,
                retry_after_ms: mean.max(1_000),
            });
        }
        let eta_ms = self.eta_ms(rank - free, now_ms);
        let budget_ms = self.policy.wait_budget(me.class).as_millis() as u64;
        if eta_ms > budget_ms {
            return Decision::Refuse(Refusal::WaitTooLong { eta_ms, budget_ms });
        }
        Decision::Wait {
            eta: Duration::from_millis(eta_ms.max(50)),
            position: rank,
        }
    }

    fn note_departure(&mut self, client: &str) {
        if !self.is_active(client) {
            self.last_departed = self.counter(client);
        }
    }

    /// Record that a request has joined the queue. Pair with
    /// [`Scheduler::dequeue`] whether it is later admitted or abandoned.
    pub fn enqueue(&mut self) {
        self.waiting = self.waiting.saturating_add(1);
    }

    pub fn dequeue(&mut self) {
        self.waiting = self.waiting.saturating_sub(1);
    }

    /// Record that service began. Returns a ticket for [`Scheduler::finish`].
    ///
    /// Charged to nobody; see [`Scheduler::start_for`].
    pub fn start(&mut self, shape: Shape, now_ms: u64) -> u64 {
        self.start_for(shape, None, now_ms)
    }

    /// Record that service began for `client`, and charge it.
    ///
    /// The prompt is charged in full and the output at what the cost model
    /// expects this request to produce, reconciled at [`Scheduler::finish`]
    /// against what it actually produced. Charging only at the end would let a
    /// caller start any number of long requests before the first had cost it
    /// anything.
    pub fn start_for(&mut self, shape: Shape, client: Option<&Client>, now_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let expected = self.expected_output(shape);
        let (client_id, weight) = match client {
            Some(c) => {
                let w = c.weight.max(1);
                let charge = Self::units(shape.prompt_tokens, W_INPUT, w)
                    .saturating_add(Self::units(expected, W_OUTPUT, w));
                let counter = self.counters.entry(c.id.clone()).or_insert(0);
                *counter = counter.saturating_add(charge);
                (Some(c.id.clone()), w)
            }
            None => (None, 1),
        };
        self.running.insert(
            id,
            Running {
                shape,
                started_ms: now_ms,
                client: client_id,
                weight,
                charged_output: expected,
            },
        );
        id
    }

    /// Free a slot for a request that never ran, refunding what its caller was
    /// charged. Teaches the cost model nothing.
    pub fn abort(&mut self, ticket: u64) {
        let Some(r) = self.running.remove(&ticket) else {
            return;
        };
        if let Some(client) = r.client.as_deref() {
            let refund = Self::units(r.shape.prompt_tokens, W_INPUT, r.weight)
                .saturating_add(Self::units(r.charged_output, W_OUTPUT, r.weight));
            let counter = self.counters.entry(client.to_string()).or_insert(0);
            *counter = counter.saturating_sub(refund);
            self.note_departure(client);
        }
    }

    /// Record completion and teach the cost model. `generated_tokens == 0`
    /// frees the slot without learning anything (the request never ran).
    ///
    /// A caller is charged for what its request actually produced. When the
    /// handler reported nothing, the advance charge stands: an unreported
    /// request still held the slot, and refunding it would make not reporting
    /// the cheapest way to use the model.
    pub fn finish(&mut self, ticket: u64, generated_tokens: u64, first_token_ms: Option<u64>, now_ms: u64) {
        let concurrency = self.running.len() as u32;
        let Some(r) = self.running.remove(&ticket) else {
            return;
        };
        if let Some(client) = r.client.as_deref() {
            if generated_tokens > 0 {
                let counter = self.counters.entry(client.to_string()).or_insert(0);
                if generated_tokens >= r.charged_output {
                    let extra = Self::units(generated_tokens - r.charged_output, W_OUTPUT, r.weight);
                    *counter = counter.saturating_add(extra);
                } else {
                    let unused = Self::units(r.charged_output - generated_tokens, W_OUTPUT, r.weight);
                    *counter = counter.saturating_sub(unused);
                }
            }
            self.note_departure(client);
        }
        self.cost.observe(Observation {
            prompt_tokens: r.shape.prompt_tokens,
            generated_tokens,
            budget_tokens: r.shape.max_output_tokens,
            service_ms: now_ms.saturating_sub(r.started_ms),
            first_token_ms,
            concurrency,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bandwidth-bound decoder: 730 tok/s prefill, 31.6 tok/s decode.
    fn slow_decoder() -> CostModel {
        CostModel::from_rates(730, 31)
    }

    #[test]
    fn output_tokens_dominate_the_estimate() {
        let e = slow_decoder().predict(Shape::new(20, 1_000), 1);
        // 1,000 tokens at 31 tok/s is ~32 s; 20 tokens of prefill is nothing.
        assert!((31_000..34_000).contains(&e.total_ms()), "{e:?}");
        assert!(e.prefill_ms < 100);
    }

    #[test]
    fn a_long_generation_does_not_poison_the_next_short_one() {
        // The failure this module exists to fix: one 1,000-token answer must
        // not make a 200-token request look like 30 seconds of work.
        let mut c = slow_decoder();
        c.observe(Observation {
            prompt_tokens: 20,
            generated_tokens: 1_000,
            budget_tokens: 1_000,
            service_ms: 32_300,
            first_token_ms: Some(30),
            concurrency: 1,
        });
        let short = c.predict(Shape::new(200, 200), 1).total_ms();
        assert!(short < 8_000, "200 tokens predicted at {short} ms");
    }

    #[test]
    fn decode_rate_is_learned_from_scratch() {
        let mut c = CostModel::new();
        for _ in 0..8 {
            c.observe(Observation {
                prompt_tokens: 100,
                generated_tokens: 500,
                budget_tokens: 500,
                service_ms: 16_300, // 500 / 31 tok/s + a little prefill
                first_token_ms: Some(140),
                concurrency: 1,
            });
        }
        assert!((29..=33).contains(&c.decode_tok_per_s()), "{}", c.decode_tok_per_s());
        assert!((600..=800).contains(&c.prefill_tok_per_s()), "{}", c.prefill_tok_per_s());
    }

    #[test]
    fn concurrency_is_normalised_out_of_the_learned_rate() {
        // Two requests sharing the decoder each see ~half the solo rate.
        let mut c = CostModel::new();
        for _ in 0..8 {
            c.observe(Observation {
                prompt_tokens: 50,
                generated_tokens: 500,
                budget_tokens: 500,
                service_ms: 32_300,
                first_token_ms: Some(100),
                concurrency: 2,
            });
        }
        assert!((29..=33).contains(&c.decode_tok_per_s()), "{}", c.decode_tok_per_s());
        // And predicting for c=2 gives the wall time actually seen.
        let e = c.predict(Shape::new(50, 500), 2).total_ms();
        assert!((30_000..35_000).contains(&e), "{e}");
    }

    #[test]
    fn the_budget_asked_for_is_not_the_time_it_takes() {
        // Callers ask for 2,000 tokens and answer in 200. Until that has been
        // seen, the ceiling is the estimate; once seen, the estimate follows
        // the answers, and the caller behind a big budget is told a wait it
        // will actually get.
        let mut c = slow_decoder();
        let ceiling = c.predict(Shape::new(20, 2_000), 1).total_ms();
        assert!(ceiling > 60_000, "prior should price the whole budget: {ceiling}");
        for _ in 0..8 {
            c.observe(Observation {
                prompt_tokens: 20,
                generated_tokens: 200,
                budget_tokens: 2_000,
                service_ms: 6_500,
                first_token_ms: Some(30),
                concurrency: 1,
            });
        }
        assert!(c.budget_use_pct() < 30, "{}", c.budget_use_pct());
        let learned = c.predict(Shape::new(20, 2_000), 1).total_ms();
        assert!((5_000..20_000).contains(&learned), "{learned}");
    }

    #[test]
    fn short_answers_never_make_the_next_request_free() {
        let mut c = slow_decoder();
        for _ in 0..16 {
            c.observe(Observation {
                prompt_tokens: 20,
                generated_tokens: 1,
                budget_tokens: 4_000,
                service_ms: 60,
                first_token_ms: Some(30),
                concurrency: 1,
            });
        }
        // Floor: at least MIN_EXPECTED_OUTPUT_TOKENS of decode, ~1 s here.
        let e = c.predict(Shape::new(20, 4_000), 1).decode_ms;
        assert!(e >= 900, "{e}");
        // And never more than the budget asks for.
        let tiny = c.predict(Shape::new(20, 8), 1).decode_ms;
        assert!(tiny <= 300, "{tiny}");
    }

    #[test]
    fn a_free_slot_admits_regardless_of_the_estimate() {
        let s = Scheduler::new(2, Policy::default()).with_cost(slow_decoder());
        // 8k tokens is over four minutes of decode. It is still admitted:
        // the caller asked for it and a slot is free.
        assert_eq!(s.decide(Class::Interactive, Shape::new(1_000, 8_192), 0), Decision::Admit);
    }

    #[test]
    fn full_slots_give_an_eta_not_a_refusal() {
        let mut s = Scheduler::new(2, Policy::default()).with_cost(slow_decoder());
        s.start(Shape::new(100, 300), 0); // ~19 s at c=2
        s.start(Shape::new(100, 600), 0); // ~39 s at c=2
        match s.decide(Class::Interactive, Shape::new(200, 200), 5_000) {
            Decision::Wait { eta, position } => {
                assert_eq!(position, 0);
                // The first slot frees when the 300-token request does:
                // ~19.5 s predicted, 5 s elapsed.
                let ms = eta.as_millis() as u64;
                assert!((12_000..17_000).contains(&ms), "eta {ms}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn interactive_refuses_only_when_the_wait_itself_is_too_long() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        s.start(Shape::new(100, 4_000), 0); // ~2 min of decode
        match s.decide(Class::Interactive, Shape::new(10, 10), 1_000) {
            Decision::Refuse(Refusal::WaitTooLong { eta_ms, budget_ms }) => {
                assert_eq!(budget_ms, 30_000);
                assert!(eta_ms > 100_000, "{eta_ms}");
            }
            other => panic!("{other:?}"),
        }
        // Batch will wait for it.
        assert!(matches!(s.decide(Class::Batch, Shape::new(10, 10), 1_000), Decision::Wait { .. }));
    }

    #[test]
    fn queue_positions_stack_behind_each_other() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        s.start(Shape::new(100, 300), 0);
        let first = match s.decide(Class::Batch, Shape::new(10, 10), 0) {
            Decision::Wait { eta, .. } => eta,
            o => panic!("{o:?}"),
        };
        s.enqueue();
        let second = match s.decide(Class::Batch, Shape::new(10, 10), 0) {
            Decision::Wait { eta, position } => {
                assert_eq!(position, 1);
                eta
            }
            o => panic!("{o:?}"),
        };
        assert!(second > first, "{second:?} should follow {first:?}");
    }

    #[test]
    fn a_full_queue_fails_fast() {
        let mut s = Scheduler::new(1, Policy { max_queue_depth: 2, ..Policy::default() });
        s.start(Shape::new(10, 10), 0);
        s.enqueue();
        s.enqueue();
        assert!(matches!(
            s.decide(Class::Batch, Shape::new(10, 10), 0),
            Decision::Refuse(Refusal::QueueFull { depth: 2, limit: 2, .. })
        ));
    }

    #[test]
    fn a_replaced_policy_bounds_the_queue_from_then_on() {
        let mut s = Scheduler::new(1, Policy::default());
        s.start(Shape::new(10, 10), 0);
        s.enqueue();
        assert!(matches!(
            s.decide(Class::Batch, Shape::new(10, 10), 0),
            Decision::Wait { .. }
        ));
        s.set_policy(Policy {
            max_queue_depth: 1,
            ..Policy::default()
        });
        assert!(matches!(
            s.decide(Class::Batch, Shape::new(10, 10), 0),
            Decision::Refuse(Refusal::QueueFull {
                depth: 1,
                limit: 1,
                ..
            })
        ));
    }

    #[test]
    fn overrunning_requests_count_as_about_to_free() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        s.start(Shape::new(10, 100), 0); // ~3 s predicted
        // 60 s later it is still running: the estimate was wrong. Do not
        // punish the next caller for it.
        match s.decide(Class::Interactive, Shape::new(10, 10), 60_000) {
            Decision::Wait { eta, .. } => assert!(eta.as_millis() <= 50),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn finish_frees_the_slot_and_teaches() {
        let mut s = Scheduler::new(1, Policy::default());
        let t = s.start(Shape::new(100, 500), 0);
        s.finish(t, 500, Some(140), 16_300);
        assert_eq!(s.running(), 0);
        assert_eq!(s.cost().samples(), 1);
        assert!((29..=33).contains(&s.cost().decode_tok_per_s()));
    }
    fn caller(id: &str) -> Client {
        Client::new(id, 1)
    }

    /// Serve whichever waiter the scheduler admits, to completion.
    fn serve_one(s: &mut Scheduler, waiters: &mut Vec<(u64, Client)>, shape: Shape, now: u64) -> String {
        let idx = waiters
            .iter()
            .position(|(w, _)| s.decide_waiter(*w, shape, now) == Decision::Admit)
            .expect("a free slot admits someone");
        let (w, c) = waiters.remove(idx);
        s.leave(w);
        let t = s.start_for(shape, Some(&c), now);
        s.finish(t, shape.max_output_tokens, Some(10), now + 1_000);
        c.id
    }

    /// The case this was written for: one caller sending steadily to a
    /// one-slot model while another arrives. Under arrival order the newcomer
    /// waits behind every queued request of the steady caller; here it is
    /// served within two.
    #[test]
    fn a_steady_caller_does_not_hold_a_one_slot_model_against_a_newcomer() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        let shape = Shape::new(200, 300);
        let sweep = caller("key:sweep");
        let issa = caller("key:issa");
        let mut waiters: Vec<(u64, Client)> = Vec::new();
        for i in 0..5 {
            waiters.push((s.join(&sweep, Class::Interactive, i), sweep.clone()));
        }
        for t in 0..3 {
            serve_one(&mut s, &mut waiters, shape, 10 + t);
            waiters.push((s.join(&sweep, Class::Interactive, 20 + t), sweep.clone()));
        }
        waiters.push((s.join(&issa, Class::Interactive, 100), issa.clone()));
        let first = serve_one(&mut s, &mut waiters, shape, 200);
        let second = serve_one(&mut s, &mut waiters, shape, 300);
        assert!(first == "key:issa" || second == "key:issa", "served {first} then {second}");
    }

    /// Weight 2 against weight 1, both always backlogged: service splits 2:1.
    #[test]
    fn a_heavier_weight_is_served_in_proportion() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        let shape = Shape::new(100, 100);
        let standard = Client::new("key:standard", 2);
        let free = Client::new("key:free", 1);
        let mut waiters = Vec::new();
        for c in [&standard, &free, &standard, &free] {
            waiters.push((s.join(c, Class::Batch, 0), c.clone()));
        }
        let mut served: HashMap<String, u32> = HashMap::new();
        for t in 1..=300 {
            let who = serve_one(&mut s, &mut waiters, shape, t);
            *served.entry(who.clone()).or_default() += 1;
            let c = if who == "key:standard" { standard.clone() } else { free.clone() };
            waiters.push((s.join(&c, Class::Batch, t), c));
        }
        let ratio = f64::from(served["key:standard"]) / f64::from(served["key:free"]);
        assert!((1.8..2.2).contains(&ratio), "ratio {ratio} from {served:?}");
    }

    /// A caller arriving after idling does not bring banked credit with it.
    #[test]
    fn a_caller_back_from_idle_brings_no_banked_credit() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        let shape = Shape::new(100, 100);
        let busy = caller("key:busy");
        let mut waiters = vec![
            (s.join(&busy, Class::Batch, 0), busy.clone()),
            (s.join(&busy, Class::Batch, 0), busy.clone()),
        ];
        for t in 1..=20 {
            serve_one(&mut s, &mut waiters, shape, t);
            waiters.push((s.join(&busy, Class::Batch, t), busy.clone()));
        }
        s.join(&caller("key:idle"), Class::Batch, 100);
        assert!(s.service_of("key:idle") >= s.service_of("key:busy"));
    }

    #[test]
    fn interactive_callers_rank_ahead_of_batch() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        let shape = Shape::new(10, 10);
        let t = s.start(shape, 0);
        let batch = s.join(&caller("key:batch"), Class::Batch, 0);
        let chat = s.join(&caller("key:chat"), Class::Interactive, 5);
        s.finish(t, 10, Some(5), 1_000);
        assert_eq!(s.decide_waiter(chat, shape, 1_000), Decision::Admit);
        assert!(matches!(s.decide_waiter(batch, shape, 1_000), Decision::Wait { .. }));
    }

    #[test]
    fn a_request_that_never_ran_costs_its_caller_nothing() {
        let mut s = Scheduler::new(1, Policy::default()).with_cost(slow_decoder());
        let c = caller("key:x");
        let t = s.start_for(Shape::new(500, 500), Some(&c), 0);
        assert!(s.service_of("key:x") > 0);
        s.abort(t);
        assert_eq!(s.service_of("key:x"), 0);
        assert_eq!(s.running(), 0);
    }

    #[test]
    fn free_slots_admit_the_best_placed_waiters_only() {
        let mut s = Scheduler::new(2, Policy::default()).with_cost(slow_decoder());
        let shape = Shape::new(10, 10);
        let a = s.join(&caller("a"), Class::Batch, 0);
        let b = s.join(&caller("b"), Class::Batch, 1);
        let c = s.join(&caller("c"), Class::Batch, 2);
        assert_eq!(s.decide_waiter(a, shape, 3), Decision::Admit);
        assert_eq!(s.decide_waiter(b, shape, 3), Decision::Admit);
        assert!(!matches!(s.decide_waiter(c, shape, 3), Decision::Admit));
        assert_eq!(s.waiting(), 3);
        s.leave(a);
        s.leave(b);
        s.leave(c);
        assert_eq!(s.waiting(), 0);
    }
}
