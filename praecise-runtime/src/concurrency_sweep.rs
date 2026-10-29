//! Concurrency sweep with a marginal-gain threshold and a validity gate.
//!
//! Backend-agnostic: nothing here touches a model. A [`SweepBackend`] runs a
//! fixed number of requests at a stated concurrency and hands back what each
//! one did; this module decides what counts, what the numbers are, and which
//! concurrency to serve at.
//!
//! The method is the one Nebius published for serving a video model on 8-GPU
//! nodes ("Serving NVIDIA Cosmos 3 Super: Request Latency and Node
//! Throughput", 2026-10-26), carried over to an engine whose work is decoding:
//!
//! 1. **A validity gate runs before throughput is counted.** An attempt adds
//!    to the rate only if it succeeded and its output passes a [`Gate`]. A
//!    rejected attempt adds nothing but its time stays in the wall clock, so a
//!    configuration that produces garbage fast scores worse, not better.
//! 2. **The rate is valid output per hour**, `sum(valid units) / wall * 3600`,
//!    where the wall clock runs from the first dispatch to the last finish
//!    and `units` is whatever the workload produces (generated tokens here).
//! 3. **Concurrency is chosen by marginal gain against a threshold fixed in
//!    advance** (default 5%). Walk the levels in ascending order; take the
//!    last level whose step up from the one before it still gained at least
//!    the threshold in valid-output rate. Latency is reported beside every
//!    step so the cost of the gain is visible, and an optional latency budget
//!    caps the choice.
//! 4. **The result carries a pin of the exact configuration** (model digest,
//!    build, shape, hardware, anything else the caller adds) and refuses to
//!    be produced without the required keys, so a number cannot be quoted
//!    without saying what it was measured on.
//!
//! What does not carry over from the video case is stated in
//! `docs/admission-and-queueing.md`: for a bandwidth-bound decoder
//! the first steps above one request in flight usually do gain, because
//! slots share one pass over the weights, so the curve is expected to turn
//! over at a different place than a diffusion replica's.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The marginal gain below which a higher concurrency is not worth its
/// latency, as a fraction of the lower level's rate.
pub const DEFAULT_GAIN_THRESHOLD: f64 = 0.05;

/// Keys a [`RunPin`] must carry before a report is produced.
pub const REQUIRED_PIN_KEYS: [&str; 4] = ["model_digest", "build", "shape", "hardware"];

/// What an attempt produced, for the gate to look at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Payload {
    /// Nothing kept (the gate can only check the counts).
    None,
    /// Generated token ids.
    Tokens(Vec<u32>),
    /// Generated text.
    Text(String),
}

/// One request's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    /// Dispatch to completion, seconds.
    pub latency_s: f64,
    /// Output units produced (generated tokens for an LLM workload).
    pub units: f64,
    /// The output itself, for the gate.
    pub payload: Payload,
    /// Set when the request failed outright.
    pub error: Option<String>,
}

/// Everything one level of the sweep did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelRun {
    /// First dispatch to last finish, seconds. Includes failed attempts.
    pub wall_s: f64,
    /// One entry per request, failed ones included.
    pub attempts: Vec<Attempt>,
}

/// Something that can run `requests` requests with `concurrency` in flight.
pub trait SweepBackend {
    /// Called before each level. A real adapter checks its host rules here
    /// (the shared-GPU lock, the free-memory floor) and returns the reason it
    /// must not run; the sweep stops there.
    ///
    /// # Errors
    /// With the reason the level must not run.
    fn admit(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Run the level. Warmup is the backend's business and is not timed.
    ///
    /// # Errors
    /// When the level could not be run at all (not for a failed request,
    /// which is an [`Attempt`] with an error).
    fn run_level(&mut self, concurrency: usize, requests: usize) -> Result<LevelRun, String>;
}

/// Decides whether an attempt's output may count toward throughput.
pub trait Gate {
    /// `Ok` when the output counts, otherwise the reason it does not.
    ///
    /// # Errors
    /// With the rejection reason.
    fn check(&self, attempt: &Attempt) -> Result<(), String>;
}

/// The gate for a text-generation workload: the request succeeded, it made
/// at least `min_units`, and the output is not degenerate. The degenerate
/// checks are the decoding counterparts of "blank frame" and "frozen frame":
/// one token making up most of the output (a stuck loop), or replacement
/// characters (broken decoding). Like the video gate, it checks integrity,
/// not quality: a fluent wrong answer passes.
#[derive(Debug, Clone)]
pub struct LlmGate {
    /// Fewest units an attempt may produce and still count.
    pub min_units: f64,
    /// Largest share one token id may take of the output, from 16 tokens up.
    pub max_single_token_share: f64,
    /// Largest share of replacement characters in text output.
    pub max_replacement_share: f64,
}

impl Default for LlmGate {
    fn default() -> Self {
        Self { min_units: 1.0, max_single_token_share: 0.9, max_replacement_share: 0.01 }
    }
}

impl Gate for LlmGate {
    fn check(&self, a: &Attempt) -> Result<(), String> {
        if let Some(e) = &a.error {
            return Err(format!("request failed: {e}"));
        }
        if a.units.is_nan() || a.units < self.min_units {
            return Err(format!("produced {} units, below the minimum {}", a.units, self.min_units));
        }
        match &a.payload {
            Payload::None => {}
            Payload::Tokens(t) => {
                if t.is_empty() {
                    return Err("empty output".into());
                }
                if t.len() >= 16 {
                    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
                    for id in t {
                        *counts.entry(*id).or_default() += 1;
                    }
                    let top = counts.values().copied().max().unwrap_or(0);
                    if top as f64 / t.len() as f64 > self.max_single_token_share {
                        return Err("one token makes up the output (stuck loop)".into());
                    }
                }
            }
            Payload::Text(s) => {
                if s.trim().is_empty() {
                    return Err("blank output".into());
                }
                let total = s.chars().count();
                let bad = s.chars().filter(|c| *c == '\u{fffd}').count();
                if bad as f64 / total as f64 > self.max_replacement_share {
                    return Err("replacement characters in the output".into());
                }
            }
        }
        Ok(())
    }
}

/// The exact configuration a result was measured on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunPin(BTreeMap<String, String>);

impl RunPin {
    /// An empty pin.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Set one key (a digest, a version, a shape, a flag).
    #[must_use]
    pub fn with(mut self, key: &str, value: &str) -> Self {
        self.0.insert(key.to_string(), value.to_string());
        self
    }
    /// Required keys that are absent or empty.
    #[must_use]
    pub fn missing(&self) -> Vec<&'static str> {
        REQUIRED_PIN_KEYS
            .iter()
            .copied()
            .filter(|k| self.0.get(*k).is_none_or(|v| v.trim().is_empty()))
            .collect()
    }
    /// SHA-256 over the length-prefixed sorted keys and values, hex. Two
    /// reports with the same digest were measured on the same stated
    /// configuration.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        for (k, v) in &self.0 {
            h.update(k.len().to_le_bytes());
            h.update(k.as_bytes());
            h.update(v.len().to_le_bytes());
            h.update(v.as_bytes());
        }
        hex::encode(h.finalize())
    }
}

/// How to run the sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepPlan {
    /// Concurrency levels, run in ascending order.
    pub levels: Vec<usize>,
    /// Requests at every level. At least the highest level, so the slots fill.
    pub requests_per_level: usize,
    /// Marginal gain below which a higher level is not taken.
    pub gain_threshold: f64,
    /// If set, no level whose mean latency exceeds this is chosen.
    pub latency_budget_s: Option<f64>,
}

impl Default for SweepPlan {
    fn default() -> Self {
        Self { levels: vec![1, 2, 4, 8], requests_per_level: 16, gain_threshold: DEFAULT_GAIN_THRESHOLD, latency_budget_s: None }
    }
}

/// One level's numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelResult {
    /// Requests in flight.
    pub concurrency: usize,
    /// Requests sent.
    pub attempts: usize,
    /// Requests that passed the gate.
    pub valid: usize,
    /// Rejection reasons and how often each occurred.
    pub rejected: BTreeMap<String, usize>,
    /// Wall clock for the level, seconds.
    pub wall_s: f64,
    /// Valid output units per hour.
    pub valid_units_per_hour: f64,
    /// Mean latency of the attempts that passed, seconds.
    pub mean_latency_s: f64,
    /// Median latency of the attempts that passed, seconds.
    pub p50_latency_s: f64,
    /// 95th-percentile latency of the attempts that passed, seconds.
    pub p95_latency_s: f64,
}

/// A step from one level to the next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// Lower level.
    pub from: usize,
    /// Higher level.
    pub to: usize,
    /// Change in valid output rate, as a fraction of the lower level's.
    pub throughput_gain: f64,
    /// Change in mean latency, as a fraction of the lower level's.
    pub latency_increase: f64,
    /// Whether the step cleared the threshold (and the latency budget).
    pub taken: bool,
}

/// The sweep's answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// The configuration the numbers belong to.
    pub pin: RunPin,
    /// SHA-256 of the pin.
    pub pin_digest: String,
    /// The plan that produced them.
    pub plan: SweepPlan,
    /// One result per level.
    pub levels: Vec<LevelResult>,
    /// One step per adjacent pair.
    pub steps: Vec<Step>,
    /// The chosen concurrency, or `None` when no level qualified.
    pub chosen: Option<usize>,
}

/// Why a sweep did not produce a report.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SweepError {
    /// The pin lacks required keys.
    #[error("the run pin lacks {0:?}; a result needs the exact configuration it was measured on")]
    PinIncomplete(Vec<&'static str>),
    /// The plan cannot be run.
    #[error("bad plan: {0}")]
    BadPlan(String),
    /// The backend refused a level.
    #[error("level {level} not run: {reason}")]
    Refused {
        /// The level.
        level: usize,
        /// The backend's reason.
        reason: String,
    },
    /// The backend failed to run a level.
    #[error("level {level} failed: {reason}")]
    Backend {
        /// The level.
        level: usize,
        /// The backend's reason.
        reason: String,
    },
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// Score one level's run through the gate.
#[must_use]
pub fn score_level(concurrency: usize, run: &LevelRun, gate: &dyn Gate) -> LevelResult {
    let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
    let mut units = 0.0;
    let mut latencies = vec![];
    let mut valid = 0;
    for a in &run.attempts {
        match gate.check(a) {
            Ok(()) => {
                valid += 1;
                units += a.units;
                latencies.push(a.latency_s);
            }
            Err(why) => *rejected.entry(why).or_default() += 1,
        }
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let mean = if latencies.is_empty() { 0.0 } else { latencies.iter().sum::<f64>() / latencies.len() as f64 };
    let rate = if run.wall_s > 0.0 { units / run.wall_s * 3600.0 } else { 0.0 };
    LevelResult {
        concurrency,
        attempts: run.attempts.len(),
        valid,
        rejected,
        wall_s: run.wall_s,
        valid_units_per_hour: rate,
        mean_latency_s: mean,
        p50_latency_s: percentile(&latencies, 0.50),
        p95_latency_s: percentile(&latencies, 0.95),
    }
}

/// Choose a level from scored results (ascending by concurrency).
///
/// Starts at the lowest level that produced valid output within the latency
/// budget, and moves up one level at a time while the step gains at least
/// `plan.gain_threshold` in valid output rate and the higher level is within
/// the budget. The first step that fails either test ends the walk, even if
/// a later level would clear it: the threshold is a stopping rule fixed in
/// advance, not a search for the global maximum.
#[must_use]
pub fn choose(levels: &[LevelResult], plan: &SweepPlan) -> (Vec<Step>, Option<usize>) {
    let within = |l: &LevelResult| plan.latency_budget_s.is_none_or(|b| l.mean_latency_s <= b);
    let Some(start) = levels.iter().position(|l| l.valid > 0 && within(l)) else {
        return (vec![], None);
    };
    let mut steps = vec![];
    let mut chosen = levels[start].concurrency;
    let mut walking = true;
    for pair in levels[start..].windows(2) {
        let (lo, hi) = (&pair[0], &pair[1]);
        let gain = if lo.valid_units_per_hour > 0.0 { hi.valid_units_per_hour / lo.valid_units_per_hour - 1.0 } else { f64::INFINITY };
        let lat = if lo.mean_latency_s > 0.0 { hi.mean_latency_s / lo.mean_latency_s - 1.0 } else { 0.0 };
        let taken = walking && hi.valid > 0 && gain >= plan.gain_threshold && within(hi);
        if taken {
            chosen = hi.concurrency;
        } else {
            walking = false;
        }
        steps.push(Step { from: lo.concurrency, to: hi.concurrency, throughput_gain: gain, latency_increase: lat, taken });
    }
    (steps, Some(chosen))
}

/// Run the sweep.
///
/// # Errors
/// [`SweepError`] when the pin is incomplete, the plan is unusable, or the
/// backend refuses or fails a level.
pub fn run_sweep(
    backend: &mut dyn SweepBackend,
    gate: &dyn Gate,
    plan: &SweepPlan,
    pin: &RunPin,
) -> Result<Report, SweepError> {
    let missing = pin.missing();
    if !missing.is_empty() {
        return Err(SweepError::PinIncomplete(missing));
    }
    if plan.levels.is_empty() || plan.levels.contains(&0) || plan.levels.windows(2).any(|w| w[0] >= w[1]) {
        return Err(SweepError::BadPlan("levels must be non-empty, positive and strictly ascending".into()));
    }
    let top = *plan.levels.last().unwrap_or(&0);
    if plan.requests_per_level < top {
        return Err(SweepError::BadPlan(format!(
            "requests_per_level {} is below the top level {top}, so the slots never fill",
            plan.requests_per_level
        )));
    }
    if plan.gain_threshold.is_nan() || plan.gain_threshold < 0.0 {
        return Err(SweepError::BadPlan("gain_threshold must be non-negative".into()));
    }
    let mut results = vec![];
    for &level in &plan.levels {
        backend.admit().map_err(|reason| SweepError::Refused { level, reason })?;
        let run = backend
            .run_level(level, plan.requests_per_level)
            .map_err(|reason| SweepError::Backend { level, reason })?;
        results.push(score_level(level, &run, gate));
    }
    Ok(Report::from_levels(pin, plan, results))
}

impl Report {
    /// Assemble a report from levels already scored with [`score_level`], for
    /// callers that run their own levels (the batch throughput test does).
    #[must_use]
    pub fn from_levels(pin: &RunPin, plan: &SweepPlan, levels: Vec<LevelResult>) -> Self {
        let (steps, chosen) = choose(&levels, plan);
        Self { pin: pin.clone(), pin_digest: pin.digest(), plan: plan.clone(), levels, steps, chosen }
    }

    /// A plain-text table, one line per level, then one per step.
    #[must_use]
    pub fn table(&self) -> String {
        let mut out = format!("pin {}\n", self.pin_digest);
        out.push_str("conc  valid/sent  units/hour  mean_s  p50_s  p95_s  rejected\n");
        for l in &self.levels {
            let rej: usize = l.rejected.values().sum();
            out.push_str(&format!(
                "{:>4}  {:>4}/{:<5}  {:>10.0}  {:>6.2}  {:>5.2}  {:>5.2}  {}\n",
                l.concurrency, l.valid, l.attempts, l.valid_units_per_hour, l.mean_latency_s, l.p50_latency_s, l.p95_latency_s, rej
            ));
        }
        for s in &self.steps {
            out.push_str(&format!(
                "step {}->{}: throughput {:+.1}%, latency {:+.1}%, {}\n",
                s.from,
                s.to,
                s.throughput_gain * 100.0,
                s.latency_increase * 100.0,
                if s.taken { "taken" } else { "not taken" }
            ));
        }
        match self.chosen {
            Some(c) => out.push_str(&format!("chosen concurrency: {c}\n")),
            None => out.push_str("chosen concurrency: none (no level qualified)\n"),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin() -> RunPin {
        RunPin::new()
            .with("model_digest", "sha256:aaaa")
            .with("build", "praecise-engine 0.1.0 abc1234")
            .with("shape", "prompt=512 max_tokens=128")
            .with("hardware", "gb10")
    }

    fn good(latency_s: f64, tokens: u32) -> Attempt {
        Attempt { latency_s, units: f64::from(tokens), payload: Payload::Tokens((0..tokens).collect()), error: None }
    }

    /// Aggregate rate rises until `slots`, then stays flat; latency grows
    /// with concurrency. Each request makes 100 tokens.
    struct Curve {
        slots: usize,
        per_slot_tok_s: f64,
        calls: Vec<usize>,
        broken_at: Option<usize>,
    }

    impl SweepBackend for Curve {
        fn run_level(&mut self, c: usize, requests: usize) -> Result<LevelRun, String> {
            self.calls.push(c);
            let eff = c.min(self.slots) as f64;
            let aggregate = eff * self.per_slot_tok_s;
            let latency = 100.0 * c as f64 / aggregate;
            let wall = requests as f64 * 100.0 / aggregate;
            let attempts = (0..requests)
                .map(|_| {
                    if self.broken_at == Some(c) {
                        Attempt { latency_s: latency, units: 100.0, payload: Payload::Tokens(vec![7; 100]), error: None }
                    } else {
                        good(latency, 100)
                    }
                })
                .collect();
            Ok(LevelRun { wall_s: wall, attempts })
        }
    }

    fn curve(slots: usize) -> Curve {
        Curve { slots, per_slot_tok_s: 20.0, calls: vec![], broken_at: None }
    }

    fn level(c: usize, rate: f64, latency: f64) -> LevelResult {
        LevelResult {
            concurrency: c,
            attempts: 4,
            valid: 4,
            rejected: BTreeMap::new(),
            wall_s: 1.0,
            valid_units_per_hour: rate,
            mean_latency_s: latency,
            p50_latency_s: latency,
            p95_latency_s: latency,
        }
    }

    #[test]
    fn stops_at_the_level_where_the_gain_falls_below_the_threshold() {
        let plan = SweepPlan { levels: vec![1, 2, 4, 8], requests_per_level: 8, ..Default::default() };
        let r = run_sweep(&mut curve(4), &LlmGate::default(), &plan, &pin()).unwrap();
        assert_eq!(r.chosen, Some(4));
        assert!(r.steps[0].taken && r.steps[1].taken && !r.steps[2].taken);
        assert!(r.steps[2].throughput_gain.abs() < 1e-9);
        assert!(r.steps[2].latency_increase > 0.9, "doubling past saturation doubles latency");
    }

    #[test]
    fn a_linear_curve_takes_every_step() {
        let plan = SweepPlan { levels: vec![1, 2, 4], requests_per_level: 4, ..Default::default() };
        let r = run_sweep(&mut curve(64), &LlmGate::default(), &plan, &pin()).unwrap();
        assert_eq!(r.chosen, Some(4));
    }

    #[test]
    fn a_gain_just_under_the_threshold_is_not_taken_and_just_over_is() {
        let plan = SweepPlan::default();
        assert_eq!(choose(&[level(1, 1000.0, 1.0), level(2, 1049.0, 1.0)], &plan).1, Some(1));
        assert_eq!(choose(&[level(1, 1000.0, 1.0), level(2, 1051.0, 1.0)], &plan).1, Some(2));
    }

    #[test]
    fn the_walk_does_not_resume_after_a_failed_step() {
        let ls = [level(1, 1000.0, 1.0), level(2, 1010.0, 1.0), level(4, 5000.0, 1.0)];
        let (steps, chosen) = choose(&ls, &SweepPlan::default());
        assert_eq!(chosen, Some(1));
        assert!(!steps[1].taken, "a later jump does not reopen a closed walk");
    }

    #[test]
    fn the_latency_budget_caps_the_choice() {
        let ls = [level(1, 1000.0, 2.0), level(2, 1800.0, 4.0), level(4, 3000.0, 9.0)];
        let plan = SweepPlan { latency_budget_s: Some(5.0), ..Default::default() };
        let (steps, chosen) = choose(&ls, &plan);
        assert_eq!(chosen, Some(2));
        assert!(steps[0].taken && !steps[1].taken, "level 4 gains 67% but is over budget");
        let plan = SweepPlan { latency_budget_s: Some(1.0), ..Default::default() };
        assert_eq!(choose(&ls, &plan).1, None, "no level fits a 1 s budget");
    }

    #[test]
    fn rejected_output_adds_no_throughput_but_keeps_its_wall_time() {
        let mut backend = curve(4);
        backend.broken_at = Some(2);
        let plan = SweepPlan { levels: vec![1, 2], requests_per_level: 4, ..Default::default() };
        let r = run_sweep(&mut backend, &LlmGate::default(), &plan, &pin()).unwrap();
        let l2 = &r.levels[1];
        assert_eq!(l2.valid, 0);
        assert_eq!(l2.valid_units_per_hour, 0.0);
        assert!(l2.wall_s > 0.0);
        assert_eq!(l2.rejected.values().sum::<usize>(), 4);
        assert_eq!(r.chosen, Some(1), "a level that only produces garbage is never chosen");
    }

    #[test]
    fn the_rate_is_valid_units_over_wall_time_per_hour() {
        let run = LevelRun {
            wall_s: 10.0,
            attempts: vec![
                good(5.0, 100),
                good(5.0, 100),
                Attempt { latency_s: 9.0, units: 0.0, payload: Payload::None, error: Some("timeout".into()) },
            ],
        };
        let l = score_level(2, &run, &LlmGate::default());
        assert_eq!(l.valid, 2);
        assert!((l.valid_units_per_hour - 200.0 / 10.0 * 3600.0).abs() < 1e-6);
        assert_eq!(l.rejected.get("request failed: timeout"), Some(&1));
    }

    #[test]
    fn the_gate_catches_loops_blank_text_replacement_characters_and_short_output() {
        let g = LlmGate { min_units: 8.0, ..Default::default() };
        let attempt = |units, payload| Attempt { latency_s: 1.0, units, payload, error: None };
        assert!(g.check(&attempt(100.0, Payload::Tokens(vec![5; 100]))).is_err());
        assert!(g.check(&attempt(100.0, Payload::Text("  \n".into()))).is_err());
        assert!(g.check(&attempt(100.0, Payload::Text("\u{fffd}\u{fffd}ok".into()))).is_err());
        assert!(g.check(&attempt(3.0, Payload::Tokens(vec![1, 2, 3]))).is_err());
        assert!(g.check(&attempt(100.0, Payload::Text("a real answer".into()))).is_ok());
        assert!(g.check(&attempt(100.0, Payload::Tokens((0..100).collect()))).is_ok());
    }

    #[test]
    fn a_result_without_its_configuration_is_refused() {
        let plan = SweepPlan::default();
        let incomplete = RunPin::new().with("model_digest", "sha256:aaaa").with("shape", " ");
        let err = run_sweep(&mut curve(4), &LlmGate::default(), &plan, &incomplete).unwrap_err();
        assert_eq!(err, SweepError::PinIncomplete(vec!["build", "shape", "hardware"]));
    }

    #[test]
    fn the_pin_digest_ignores_insertion_order_and_changes_with_any_value() {
        let a = RunPin::new().with("b", "2").with("a", "1");
        let b = RunPin::new().with("a", "1").with("b", "2");
        assert_eq!(a.digest(), b.digest());
        assert_ne!(a.digest(), RunPin::new().with("a", "1").with("b", "3").digest());
        // Key/value boundaries are part of the hash: "ab"="c" is not "a"="bc".
        assert_ne!(RunPin::new().with("ab", "c").digest(), RunPin::new().with("a", "bc").digest());
    }

    #[test]
    fn bad_plans_are_refused_before_anything_runs() {
        let mut backend = curve(4);
        for plan in [
            SweepPlan { levels: vec![], ..Default::default() },
            SweepPlan { levels: vec![0, 1], ..Default::default() },
            SweepPlan { levels: vec![2, 2], ..Default::default() },
            SweepPlan { levels: vec![1, 8], requests_per_level: 4, ..Default::default() },
        ] {
            assert!(matches!(run_sweep(&mut backend, &LlmGate::default(), &plan, &pin()), Err(SweepError::BadPlan(_))));
        }
        assert!(backend.calls.is_empty());
    }

    struct Locked(u32);
    impl SweepBackend for Locked {
        fn admit(&mut self) -> Result<(), String> {
            if self.0 == 0 {
                Err("gpu-big lock held by another tenant".into())
            } else {
                self.0 -= 1;
                Ok(())
            }
        }
        fn run_level(&mut self, _c: usize, requests: usize) -> Result<LevelRun, String> {
            Ok(LevelRun { wall_s: 1.0, attempts: (0..requests).map(|_| good(1.0, 10)).collect() })
        }
    }

    #[test]
    fn a_refusal_from_the_host_stops_the_sweep_at_that_level() {
        let plan = SweepPlan { levels: vec![1, 2, 4], requests_per_level: 4, ..Default::default() };
        let err = run_sweep(&mut Locked(1), &LlmGate::default(), &plan, &pin()).unwrap_err();
        assert_eq!(err, SweepError::Refused { level: 2, reason: "gpu-big lock held by another tenant".into() });
    }

    #[test]
    fn the_report_round_trips_through_json_and_prints_a_table() {
        let plan = SweepPlan { levels: vec![1, 2], requests_per_level: 2, ..Default::default() };
        let r = run_sweep(&mut curve(4), &LlmGate::default(), &plan, &pin()).unwrap();
        let back: Report = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
        let t = r.table();
        assert!(t.contains(&r.pin_digest) && t.contains("chosen concurrency: 2"));
    }
}
