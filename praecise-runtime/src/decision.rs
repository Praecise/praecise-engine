//! Decision models.
//!
//! A decision model reads a piece of text (the state) and a set of typed
//! questions and answers each of them in one forward pass: no text is
//! generated, so there is nothing to parse. The model is an encoder whose
//! output head scores every option of a question at the position of the
//! option's marker token; a softmax over the scores, calibrated by a
//! per-question-type temperature stored in the model file, gives the
//! probabilities.
//!
//! Three question types exist:
//!
//! * `choice`: one option out of several; the answer is the most probable
//!   key, with the probability of every key and a confidence;
//! * `score`: 2 to 10 ordered levels; the answer is the expected level, with
//!   the probability of every level and a confidence;
//! * `noul`: yes or no ("does the statement hold"); the answer is the
//!   probability of yes.
//!
//! Every question is decoded on its own, in a fresh sequence, so an answer
//! does not depend on which other questions the request carried.
//!
//! The prompt is written with the model's own `systemone` template. That
//! template is data in the model file, and a decision model whose template is
//! not one of the templates [`KNOWN_TEMPLATES`] was verified against is
//! refused rather than rendered by guess.

use std::collections::HashMap;
use std::num::NonZeroU32;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// Most questions one request may carry.
pub const MAX_QUESTIONS: usize = 64;
/// Most options one question may carry.
pub const MAX_OPTIONS: usize = 255;
/// Fewest levels of a score question.
pub const MIN_SCORE_LEVELS: usize = 2;
/// Most levels of a score question.
pub const MAX_SCORE_LEVELS: usize = 10;
/// Tokens kept of one option's text before the head is shared out.
const MAX_OPTION_TOKENS: usize = 48;

/// The `systemone` templates the renderer in this file reproduces exactly.
pub const KNOWN_TEMPLATES: &[&str] = &[
    r#"[CLS]{{ type }} question: {{ instructions if instructions is string else instructions | tojson }}[SEP]{% for o in options %}[MASK] {% if type == 'choice' %}{{ o.key }}{% if o.description %}: {{ o.description if o.description is string else o.description | tojson }}{% endif %}{% elif type == 'score' %}level {{ o.key }}: {{ o.description if o.description is string else o.description | tojson }}{% else %}{{ o.key }}: {% if o.description %}{{ o.description if o.description is string else o.description | tojson }}{% elif o.key == 'true' %}yes, the statement holds{% else %}no, the statement does not hold{% endif %}{% endif %}{% endfor %}[SEP]{{ state if state is string else state | tojson }}[SEP]"#,
];

/// The kind of a question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    /// One option out of several.
    Choice,
    /// An ordered scale of 2 to 10 levels.
    Score,
    /// Yes or no.
    Noul,
}

impl QuestionKind {
    /// The name the template and the answer use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// The column of the model's output row that scores this kind.
    fn column(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }
}

/// One option of a question: a key and an optional description (text, or any
/// JSON value, which the template writes as JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionOption {
    /// The key the answer names.
    pub key: String,
    /// What the option means; `null` when not given.
    #[serde(default)]
    pub description: Value,
}

/// One question of a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// The id the answer is returned under.
    pub id: String,
    /// What the question is: text, or any JSON value.
    pub instructions: Value,
    /// The kind of the question.
    pub kind: QuestionKind,
    /// The options, in the order the prompt lists them. A `noul` question
    /// has exactly two (`false`, `true`), a `score` question one per level
    /// (keys `0`, `1`, ...).
    pub options: Vec<DecisionOption>,
}

impl Question {
    /// A choice question.
    pub fn choice(
        id: impl Into<String>,
        instructions: impl Into<Value>,
        options: Vec<DecisionOption>,
    ) -> Self {
        Self {
            id: id.into(),
            instructions: instructions.into(),
            kind: QuestionKind::Choice,
            options,
        }
    }

    /// A score question with one level per description.
    pub fn score(
        id: impl Into<String>,
        instructions: impl Into<Value>,
        levels: Vec<Value>,
    ) -> Self {
        Self {
            id: id.into(),
            instructions: instructions.into(),
            kind: QuestionKind::Score,
            options: levels
                .into_iter()
                .enumerate()
                .map(|(i, description)| DecisionOption {
                    key: i.to_string(),
                    description,
                })
                .collect(),
        }
    }

    /// A yes or no question, with optional descriptions of the two answers.
    pub fn noul(
        id: impl Into<String>,
        instructions: impl Into<Value>,
        description_false: Value,
        description_true: Value,
    ) -> Self {
        Self {
            id: id.into(),
            instructions: instructions.into(),
            kind: QuestionKind::Noul,
            options: vec![
                DecisionOption {
                    key: "false".into(),
                    description: description_false,
                },
                DecisionOption {
                    key: "true".into(),
                    description: description_true,
                },
            ],
        }
    }

    /// Check the question against the limits of the model.
    ///
    /// # Errors
    ///
    /// When the options do not fit the kind or exceed [`MAX_OPTIONS`].
    pub fn validate(&self) -> Result<()> {
        let err = |m: &str| Error::Inference(format!("question {}: {m}", self.id));
        if self.instructions.is_null() {
            return Err(err("instructions must be provided"));
        }
        if self.options.len() > MAX_OPTIONS {
            return Err(err(&format!(
                "too many options ({}), at most {MAX_OPTIONS}",
                self.options.len()
            )));
        }
        match self.kind {
            QuestionKind::Choice => {
                if self.options.is_empty() {
                    return Err(err("a choice needs at least one option"));
                }
                let mut keys: Vec<&str> = self.options.iter().map(|o| o.key.as_str()).collect();
                keys.sort_unstable();
                keys.dedup();
                if keys.len() != self.options.len() {
                    return Err(err("option keys must be distinct"));
                }
            }
            QuestionKind::Score => {
                let n = self.options.len();
                if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&n) {
                    return Err(err(&format!(
                        "a score has {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {n}"
                    )));
                }
            }
            QuestionKind::Noul => {
                let keys: Vec<&str> = self.options.iter().map(|o| o.key.as_str()).collect();
                if keys != ["false", "true"] {
                    return Err(err("a yes/no question has the options false and true"));
                }
            }
        }
        Ok(())
    }
}

/// The answer to one question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// The question's id.
    pub id: String,
    /// The question's kind.
    pub kind: QuestionKind,
    /// Choice: the most probable key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,
    /// Score: the expected level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Yes or no: the probability of yes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,
    /// The probability of every key, in option order (choice and score).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub probabilities: Vec<(String, f64)>,
    /// How decided the model is, from 0 to 1 (choice and score).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Score: what each level means.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub legend: Vec<(String, Value)>,
    /// Tokens the question's prompt took.
    pub tokens: usize,
}

/// The calibration a decision model carries in its metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionHead {
    /// Tokens shared out between a question and its options.
    pub max_head_tokens: usize,
    /// Softmax temperatures by name (`choice`, `choice.3_5`, `score`, ...).
    pub temperatures: HashMap<String, f32>,
    /// Additive biases of the logits of a score question, by level count.
    pub score_bias: HashMap<usize, Vec<f32>>,
}

fn inference(what: impl std::fmt::Display) -> Error {
    Error::Inference(what.to_string())
}

impl DecisionHead {
    /// Read the head of `model`.
    ///
    /// # Errors
    ///
    /// When the model is not a Laya-type decision model, its `systemone`
    /// template is not one this renderer was verified against, or its
    /// calibration is malformed.
    pub fn from_model(model: &LlamaModel) -> Result<Self> {
        let arch = model
            .meta_val_str("general.architecture")
            .map_err(inference)?;
        let prefix = format!("{arch}.decision.");
        let kind = model
            .meta_val_str(&format!("{prefix}type"))
            .map_err(|_| inference("the model is not a decision model"))?;
        if kind != "laya" {
            return Err(inference(format!(
                "decision model type {kind} is not supported by this renderer"
            )));
        }
        let template = model
            .chat_template(Some("systemone"))
            .map_err(|_| inference("the decision model has no systemone template"))?;
        let template = template.to_str().map_err(inference)?;
        if !KNOWN_TEMPLATES.contains(&template) {
            return Err(inference(
                "the systemone template differs from the ones this renderer reproduces; refusing to render it by guess",
            ));
        }
        let max_head_tokens: usize = model
            .meta_val_str(&format!("{prefix}max_head_tokens"))
            .map_err(|_| inference("the decision model has no max_head_tokens"))?
            .trim()
            .parse()
            .map_err(inference)?;
        if max_head_tokens == 0 {
            return Err(inference("max_head_tokens is zero"));
        }
        let mut temperatures = HashMap::new();
        let mut score_bias: HashMap<usize, Vec<f32>> = HashMap::new();
        let temp_prefix = format!("{prefix}temperature.");
        let bias_prefix = format!("{prefix}score_bias.");
        for i in 0..model.meta_count() {
            let Ok(key) = model.meta_key_by_index(i) else {
                continue;
            };
            if let Some(name) = key.strip_prefix(&temp_prefix) {
                let value: f32 = model
                    .meta_val_str_by_index(i)
                    .map_err(inference)?
                    .trim()
                    .parse()
                    .map_err(inference)?;
                if value.is_nan() || value <= 0.0 {
                    return Err(inference(format!("invalid decision temperature {key}")));
                }
                temperatures.insert(name.to_string(), value);
            } else if let Some(rest) = key.strip_prefix(&bias_prefix) {
                let mut parts = rest.split('.');
                let n: usize = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
                let level: usize = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
                if n < 2 || n > 255 || level >= n || parts.next().is_some() {
                    return Err(inference(format!("invalid decision score bias {key}")));
                }
                let value: f32 = model
                    .meta_val_str_by_index(i)
                    .map_err(inference)?
                    .trim()
                    .parse()
                    .map_err(inference)?;
                score_bias.entry(n).or_insert_with(|| vec![0.0; n])[level] = value;
            }
        }
        Ok(Self {
            max_head_tokens,
            temperatures,
            score_bias,
        })
    }

    /// The softmax temperature of a question: by kind and option-count
    /// bucket, then by kind, then 1.
    pub fn temperature(&self, kind: QuestionKind, n_options: usize) -> f32 {
        let bucket = match n_options {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        };
        [format!("{}.{bucket}", kind.name()), kind.name().to_string()]
            .iter()
            .find_map(|name| self.temperatures.get(name).copied())
            .unwrap_or(1.0)
    }
}

/// `value | tojson`: Python's `json.dumps` separators, keys as given, text
/// not ASCII-escaped; a string is written as is.
fn jinja_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => json_dumps(other),
    }
}

fn json_dumps(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(json_dumps).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    serde_json::to_string(k).unwrap_or_default(),
                    json_dumps(v)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Replace the model's marker text in a string of the input, so that text
/// cannot add an option of its own.
fn strip_marker(text: &str, marker: &str) -> String {
    if marker.is_empty() {
        text.to_string()
    } else {
        text.replace(marker, " ")
    }
}

/// The prompt of one question: the `systemone` template, written out.
pub fn render_prompt(question: &Question, state: &Value, marker: &str) -> String {
    let clean = |v: &Value| -> String { strip_marker(&jinja_text(v), marker) };
    let mut out = String::new();
    out.push_str("[CLS]");
    out.push_str(question.kind.name());
    out.push_str(" question: ");
    out.push_str(&clean(&question.instructions));
    out.push_str("[SEP]");
    for o in &question.options {
        out.push_str("[MASK] ");
        let key = strip_marker(&o.key, marker);
        // The template's `{% if o.description %}`: Jinja truthiness.
        let described = match &o.description {
            Value::Null => false,
            Value::String(s) => !s.is_empty(),
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
            Value::Array(a) => !a.is_empty(),
            Value::Object(m) => !m.is_empty(),
        };
        match question.kind {
            QuestionKind::Choice => {
                out.push_str(&key);
                if described {
                    out.push_str(": ");
                    out.push_str(&clean(&o.description));
                }
            }
            QuestionKind::Score => {
                out.push_str("level ");
                out.push_str(&key);
                out.push_str(": ");
                out.push_str(&clean(&o.description));
            }
            QuestionKind::Noul => {
                out.push_str(&key);
                out.push_str(": ");
                if described {
                    out.push_str(&clean(&o.description));
                } else if o.key == "true" {
                    out.push_str("yes, the statement holds");
                } else {
                    out.push_str("no, the statement does not hold");
                }
            }
        }
    }
    out.push_str("[SEP]");
    out.push_str(&clean(state));
    out.push_str("[SEP]");
    out
}

/// Cut the question and the options to the head budget the model was
/// trained with. Returns the tokens and the position of every option's
/// marker.
///
/// # Errors
///
/// When the tokens are not laid out as the template writes them.
pub fn fit_head(
    tokens: &[LlamaToken],
    marker: LlamaToken,
    sep: LlamaToken,
    n_options: usize,
    max_head_tokens: usize,
) -> Result<(Vec<LlamaToken>, Vec<usize>)> {
    let layout = || inference("unexpected layout of the decision prompt");
    let markers: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == marker)
        .map(|(i, _)| i)
        .collect();
    if markers.len() != n_options
        || n_options == 0
        || markers[0] < 2
        || tokens[markers[0] - 1] != sep
        || tokens.last() != Some(&sep)
    {
        return Err(layout());
    }
    let head_end = markers[0] - 1;
    let last = *markers.last().ok_or_else(layout)?;
    let opts_end = tokens[last..]
        .iter()
        .position(|t| *t == sep)
        .map(|p| p + last)
        .ok_or_else(layout)?;
    if opts_end + 1 >= tokens.len() {
        return Err(layout());
    }
    let mut options: Vec<Vec<LlamaToken>> = (0..n_options)
        .map(|i| {
            let end = if i + 1 < n_options {
                markers[i + 1]
            } else {
                opts_end
            };
            tokens[markers[i]..end].to_vec()
        })
        .collect();
    let set_max = |options: &mut Vec<Vec<LlamaToken>>, n_max: usize| -> usize {
        let mut total = 0;
        for o in options.iter_mut() {
            o.truncate(n_max);
            total += o.len();
        }
        total
    };
    let mut n_options_tokens = set_max(&mut options, MAX_OPTION_TOKENS + 1);
    if n_options_tokens + 16 > max_head_tokens {
        // too many or too long options: shrink them evenly
        n_options_tokens = set_max(
            &mut options,
            4.max((max_head_tokens - 16.min(max_head_tokens)) / n_options),
        );
    }
    let n_question_max = 8.max(max_head_tokens - n_options_tokens.min(max_head_tokens));
    let mut out = Vec::with_capacity(tokens.len());
    out.push(tokens[0]);
    out.extend_from_slice(&tokens[1..head_end.min(1 + n_question_max)]);
    out.push(sep);
    let mut positions = Vec::with_capacity(n_options);
    for o in &options {
        positions.push(out.len());
        out.extend_from_slice(o);
    }
    out.extend_from_slice(&tokens[opts_end..]);
    Ok((out, positions))
}

/// Softmax of `scores` at `temperature`, after the score bias of a question
/// with that many levels when it has one.
pub fn probabilities(scores: &[f32], temperature: f32, bias: Option<&[f32]>) -> Vec<f64> {
    let logits: Vec<f64> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| f64::from(*s) + bias.map_or(0.0, |b| f64::from(b[i])))
        .collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|l| ((l - max) / f64::from(temperature)).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn confidence_choice(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / p.len() as f64;
    let p_max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

fn confidence_score(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let n = p.len();
    let mode = p
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (i, v)| {
            if *v > best.1 { (i, *v) } else { best }
        })
        .0;
    let mut dist = 0.0;
    let mut dist_uniform = 0.0;
    for (i, v) in p.iter().enumerate() {
        dist += v * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n as f64 - 1.0) / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

/// The answer to `question` from the scores of its options.
pub fn answer(
    head: &DecisionHead,
    question: &Question,
    scores: &[f32],
    tokens: usize,
) -> Result<Answer> {
    let n = question.options.len();
    if scores.len() != n {
        return Err(inference(
            "decision result does not match the number of options",
        ));
    }
    if scores.iter().any(|s| s.is_nan()) {
        return Err(inference("the model could not evaluate the decision"));
    }
    let bias = (question.kind == QuestionKind::Score)
        .then(|| head.score_bias.get(&n))
        .flatten();
    let p = probabilities(
        scores,
        head.temperature(question.kind, n),
        bias.map(Vec::as_slice),
    );
    let mut out = Answer {
        id: question.id.clone(),
        kind: question.kind,
        choice: None,
        score: None,
        noul: None,
        probabilities: Vec::new(),
        confidence: None,
        legend: Vec::new(),
        tokens,
    };
    match question.kind {
        QuestionKind::Noul => {
            out.noul = question
                .options
                .iter()
                .position(|o| o.key == "true")
                .map(|i| p[i]);
        }
        QuestionKind::Choice => {
            let best = p
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |b, (i, v)| {
                    if *v > b.1 { (i, *v) } else { b }
                })
                .0;
            out.choice = Some(question.options[best].key.clone());
            out.probabilities = question
                .options
                .iter()
                .zip(&p)
                .map(|(o, p)| (o.key.clone(), *p))
                .collect();
            out.confidence = Some(confidence_choice(&p));
        }
        QuestionKind::Score => {
            out.score = Some(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum());
            out.probabilities = question
                .options
                .iter()
                .zip(&p)
                .map(|(o, p)| (o.key.clone(), *p))
                .collect();
            out.legend = question
                .options
                .iter()
                .map(|o| (o.key.clone(), o.description.clone()))
                .collect();
            out.confidence = Some(confidence_score(&p));
        }
    }
    Ok(out)
}

/// Options for a decision call.
#[derive(Debug, Clone, Copy)]
pub struct DecisionOptions {
    /// Longest prompt, in tokens; a longer one is refused, never cut (the
    /// head is cut to its trained budget, the state is not).
    pub n_ctx: u32,
    /// Threads for the batch.
    pub n_threads: i32,
}

impl Default for DecisionOptions {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_threads: std::thread::available_parallelism()
                .map(|n| i32::try_from(n.get()).unwrap_or(1))
                .unwrap_or(1),
        }
    }
}

/// Answer every question about `state` with the decision model `model`.
///
/// # Errors
///
/// When the request exceeds [`MAX_QUESTIONS`] or a question breaks its
/// limits, a prompt does not fit `options.n_ctx`, or the backend fails.
pub fn decide(
    backend: &LlamaBackend,
    model: &LlamaModel,
    head: &DecisionHead,
    state: &Value,
    questions: &[Question],
    options: DecisionOptions,
) -> Result<Vec<Answer>> {
    if state.is_null() {
        return Err(inference("state must be provided"));
    }
    if questions.is_empty() || questions.len() > MAX_QUESTIONS {
        return Err(inference(format!(
            "a request carries 1 to {MAX_QUESTIONS} questions, got {}",
            questions.len()
        )));
    }
    let mut ids: Vec<&str> = questions.iter().map(|q| q.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != questions.len() {
        return Err(inference("question ids must be distinct"));
    }
    for q in questions {
        q.validate()?;
    }
    let marker = model
        .token_mask()
        .ok_or_else(|| inference("the decision model has no mask token"))?;
    let sep = model.token_sep();
    let marker_text = String::from_utf8_lossy(
        &model
            .token_to_piece_bytes(marker, 32, true, None)
            .map_err(inference)?,
    )
    .into_owned();
    let n_ctx = options.n_ctx.min(model.n_ctx_train().max(1));
    let n_columns = usize::try_from(model.n_embd_out()).unwrap_or(0);
    if n_columns < 3 {
        return Err(inference(
            "the model's output row is too narrow for a decision head",
        ));
    }
    let mut ctx = model
        .new_context(
            backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(n_ctx)
                .with_n_ubatch(n_ctx)
                .with_n_seq_max(1)
                .with_n_threads(options.n_threads)
                .with_n_threads_batch(options.n_threads)
                .with_embeddings(true)
                .with_pooling_type(LlamaPoolingType::None),
        )
        .map_err(inference)?;
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut out = Vec::with_capacity(questions.len());
    for q in questions {
        let prompt = render_prompt(q, state, &marker_text);
        let raw = model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(inference)?;
        let (tokens, positions) =
            fit_head(&raw, marker, sep, q.options.len(), head.max_head_tokens)?;
        if tokens.len() > n_ctx as usize {
            return Err(inference(format!(
                "question {}: the prompt of {} tokens does not fit the context of {n_ctx}",
                q.id,
                tokens.len()
            )));
        }
        ctx.clear_kv_cache();
        batch.clear();
        batch.add_sequence(&tokens, 0, true).map_err(inference)?;
        ctx.decode(&mut batch).map_err(inference)?;
        let column = q.kind.column();
        let mut scores = Vec::with_capacity(positions.len());
        for pos in &positions {
            let row = ctx
                .embeddings_ith(i32::try_from(*pos).map_err(inference)?)
                .map_err(inference)?;
            scores.push(
                *row.get(column)
                    .ok_or_else(|| inference("the output row has no column for this question"))?,
            );
        }
        out.push(answer(head, q, &scores, tokens.len())?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opt(key: &str, description: Value) -> DecisionOption {
        DecisionOption {
            key: key.into(),
            description,
        }
    }

    fn head() -> DecisionHead {
        DecisionHead {
            max_head_tokens: 192,
            temperatures: HashMap::from([
                ("choice".to_string(), 1.5),
                ("choice.3_5".to_string(), 2.0),
                ("noul".to_string(), 2.0),
                ("score".to_string(), 1.0),
            ]),
            score_bias: HashMap::new(),
        }
    }

    #[test]
    fn the_choice_prompt_is_the_template_written_out() {
        let q = Question::choice(
            "route",
            "Where should this go?",
            vec![
                opt("billing", json!("invoices and payments")),
                opt("other", Value::Null),
            ],
        );
        assert_eq!(
            render_prompt(&q, &json!("My card was charged twice"), "[MASK]"),
            "[CLS]choice question: Where should this go?[SEP][MASK] billing: invoices and payments[MASK] other[SEP]My card was charged twice[SEP]"
        );
    }

    #[test]
    fn score_and_yes_no_prompts_and_json_values() {
        let s = Question::score(
            "tone",
            json!({"rate": "politeness"}),
            vec![json!("rude"), json!("polite")],
        );
        assert_eq!(
            render_prompt(&s, &json!({"a": [1, "x"]}), "[MASK]"),
            "[CLS]score question: {\"rate\": \"politeness\"}[SEP][MASK] level 0: rude[MASK] level 1: polite[SEP]{\"a\": [1, \"x\"]}[SEP]"
        );
        let n = Question::noul("spam", "Is this spam?", Value::Null, Value::Null);
        assert_eq!(
            render_prompt(&n, &json!("buy now"), "[MASK]"),
            "[CLS]noul question: Is this spam?[SEP][MASK] false: no, the statement does not hold[MASK] true: yes, the statement holds[SEP]buy now[SEP]"
        );
    }

    #[test]
    fn text_cannot_add_an_option_of_its_own() {
        let q = Question::noul("x", "q [MASK] q", Value::Null, Value::Null);
        let p = render_prompt(&q, &json!("s [MASK] s"), "[MASK]");
        assert_eq!(p.matches("[MASK]").count(), 2, "{p}");
    }

    #[test]
    fn limits_are_enforced_per_question() {
        let too_many = Question::choice(
            "c",
            "q",
            (0..=MAX_OPTIONS)
                .map(|i| opt(&i.to_string(), Value::Null))
                .collect(),
        );
        assert!(too_many.validate().is_err());
        let ok = Question::choice(
            "c",
            "q",
            (0..MAX_OPTIONS)
                .map(|i| opt(&i.to_string(), Value::Null))
                .collect(),
        );
        assert!(ok.validate().is_ok());
        assert!(
            Question::score("s", "q", vec![json!("a")])
                .validate()
                .is_err()
        );
        assert!(
            Question::score("s", "q", (0..11).map(|i| json!(i)).collect())
                .validate()
                .is_err()
        );
        assert!(
            Question::score("s", "q", (0..10).map(|i| json!(i)).collect())
                .validate()
                .is_ok()
        );
        assert!(
            Question::score("s", "q", (0..2).map(|i| json!(i)).collect())
                .validate()
                .is_ok()
        );
        assert!(
            Question::choice("c", Value::Null, vec![opt("a", Value::Null)])
                .validate()
                .is_err()
        );
        let dup = Question::choice("c", "q", vec![opt("a", Value::Null), opt("a", Value::Null)]);
        assert!(dup.validate().is_err());
    }

    #[test]
    fn probabilities_are_a_calibrated_softmax() {
        let p = probabilities(&[2.0, 0.0], 1.0, None);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p[0] - 0.880797077977882).abs() < 1e-12);
        // a higher temperature flattens it
        let q = probabilities(&[2.0, 0.0], 2.0, None);
        assert!(q[0] < p[0] && q[0] > 0.5);
        // a bias moves a score level
        let b = probabilities(&[0.0, 0.0], 1.0, Some(&[0.0, 1.0]));
        assert!(b[1] > b[0]);
    }

    #[test]
    fn temperature_falls_back_from_the_bucket_to_the_kind_to_one() {
        let h = head();
        assert_eq!(h.temperature(QuestionKind::Choice, 4), 2.0);
        assert_eq!(h.temperature(QuestionKind::Choice, 7), 1.5);
        assert_eq!(h.temperature(QuestionKind::Noul, 2), 2.0);
        assert_eq!(h.temperature(QuestionKind::Score, 5), 1.0);
        let empty = DecisionHead {
            temperatures: HashMap::new(),
            ..h
        };
        assert_eq!(empty.temperature(QuestionKind::Choice, 3), 1.0);
    }

    #[test]
    fn answers_carry_the_fields_of_their_kind() {
        let h = head();
        let c = Question::choice("c", "q", vec![opt("a", Value::Null), opt("b", Value::Null)]);
        let a = answer(&h, &c, &[0.0, 3.0], 20).unwrap();
        assert_eq!(a.choice.as_deref(), Some("b"));
        assert_eq!(a.probabilities.len(), 2);
        assert!(a.confidence.unwrap() > 0.5);
        let n = Question::noul("n", "q", Value::Null, Value::Null);
        let a = answer(&h, &n, &[0.0, 4.0], 20).unwrap();
        assert!(a.noul.unwrap() > 0.8 && a.choice.is_none() && a.probabilities.is_empty());
        let s = Question::score("s", "q", vec![json!("lo"), json!("mid"), json!("hi")]);
        let a = answer(&h, &s, &[0.0, 0.0, 10.0], 20).unwrap();
        assert!(a.score.unwrap() > 1.9 && a.score.unwrap() <= 2.0);
        assert_eq!(a.legend.len(), 3);
        assert!(answer(&h, &s, &[0.0, f32::NAN, 1.0], 1).is_err());
        assert!(answer(&h, &s, &[0.0], 1).is_err());
    }

    #[test]
    fn the_head_is_shared_out_as_the_model_was_trained() {
        let t = |n: i32| LlamaToken(n);
        let (marker, sep) = (t(9), t(8));
        // [cls] q q q [sep] [mark] a a [mark] b [sep] s s [sep]
        let tokens = vec![
            t(1),
            t(2),
            t(2),
            t(2),
            sep,
            marker,
            t(3),
            t(3),
            marker,
            t(4),
            sep,
            t(5),
            t(5),
            sep,
        ];
        let (out, pos) = fit_head(&tokens, marker, sep, 2, 192).unwrap();
        assert_eq!(out, tokens);
        assert_eq!(pos, vec![5, 8]);
        // a long question is cut to what the options leave of the budget
        let mut long = vec![t(1)];
        long.extend(std::iter::repeat_n(t(2), 40));
        long.extend([sep, marker, t(3), t(3), marker, t(4), sep, t(5), sep]);
        let (cut, pos) = fit_head(&long, marker, sep, 2, 18).unwrap();
        // budget 18, options take 5 tokens: the start token, 13 question tokens, a separator, the options, the state
        assert_eq!(cut.len(), 1 + 13 + 1 + 5 + 3);
        assert_eq!(cut[pos[0]], marker);
        assert_eq!(cut[pos[1]], marker);
        assert_eq!(*cut.last().unwrap(), sep);
        // many long options are shrunk evenly, never below four tokens each
        let mut wide = vec![t(1), t(2), sep];
        for _ in 0..8 {
            wide.push(marker);
            wide.extend(std::iter::repeat_n(t(3), 30));
        }
        wide.extend([sep, t(5), sep]);
        let (cut, pos) = fit_head(&wide, marker, sep, 8, 40).unwrap();
        assert_eq!(pos.len(), 8);
        assert!(cut.len() < wide.len());
        for w in pos.windows(2) {
            assert!(w[1] - w[0] <= 4, "each option kept to its share");
        }
        // a layout the template cannot have produced is refused
        assert!(fit_head(&tokens[..10], marker, sep, 2, 192).is_err());
        assert!(fit_head(&tokens, marker, sep, 3, 192).is_err());
    }
    /// The real English Laya model (`ggml-org/Laya-GGUF`, Q8_0), whose path is
    /// in `LAYA_GGUF`. Prints the answers as JSON so they can be compared
    /// with the same requests sent to the llama.cpp server's decision route.
    #[test]
    #[ignore = "needs LAYA_GGUF=<path to Laya-Q8_0.gguf>"]
    fn the_real_laya_model_decides() {
        let Ok(path) = std::env::var("LAYA_GGUF") else {
            eprintln!("skip: LAYA_GGUF not set");
            return;
        };
        let backend = LlamaBackend::init().expect("backend");
        let model = LlamaModel::load_from_file(&backend, &path, &Default::default()).expect("load");
        let head = DecisionHead::from_model(&model).expect("a Laya head");
        assert_eq!(head.max_head_tokens, 192);
        assert!(head.temperatures.contains_key("choice"));
        let state = json!(
            "WIN A FREE IPHONE now!!! Click the link below to claim your prize before midnight"
        );
        let questions = vec![
            Question::noul("spam", "Is this message spam?", Value::Null, Value::Null),
            Question::choice(
                "category",
                "Which kind of message is this?",
                vec![
                    opt("promotion", json!("an advertisement or prize offer")),
                    opt("support", json!("a customer asking for help")),
                    opt("personal", Value::Null),
                ],
            ),
            Question::score(
                "urgency",
                json!({"rate": "how urgent the message tries to sound"}),
                vec![
                    json!("not urgent"),
                    json!("somewhat urgent"),
                    json!("very urgent"),
                ],
            ),
        ];
        let answers = decide(
            &backend,
            &model,
            &head,
            &state,
            &questions,
            DecisionOptions::default(),
        )
        .expect("decide");
        let out = serde_json::to_string(&answers).unwrap();
        eprintln!("LAYA-RESULT {out}");
        if let Ok(p) = std::env::var("LAYA_OUT") {
            std::fs::write(p, out).unwrap();
        }
        let spam = answers[0].noul.unwrap();
        assert!(spam > 0.5, "a prize scam is spam: {spam}");
        assert_eq!(answers[1].choice.as_deref(), Some("promotion"));
        for a in &answers[1..] {
            let sum: f64 = a.probabilities.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-9, "{sum}");
        }
        // a question's answer does not depend on the others in the request
        let alone = decide(
            &backend,
            &model,
            &head,
            &state,
            &questions[1..2],
            DecisionOptions::default(),
        )
        .unwrap();
        assert_eq!(alone[0].probabilities, answers[1].probabilities);
        // a request over the question limit is refused
        let many: Vec<Question> = (0..=MAX_QUESTIONS)
            .map(|i| Question::noul(format!("q{i}"), "x?", Value::Null, Value::Null))
            .collect();
        assert!(
            decide(
                &backend,
                &model,
                &head,
                &state,
                &many,
                DecisionOptions::default()
            )
            .is_err()
        );
    }
}
