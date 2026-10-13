//! The MolmoAct2 policy's inputs as its release prepares them for inference.
//!
//! Each camera frame is resized (torchvision bilinear, no antialiasing) to
//! the single crop the release's processor uses, mapped to `[-1, 1]` and cut
//! into patches; each image token pools a 2x2 (by the release config) block
//! of patches, the grid padded with empty slots at the bottom and right.
//! The prompt names the normalised task, the setup and control mode in their
//! marker tokens and the state as one discrete token per value, after one
//! image placeholder per camera; each placeholder becomes the image's layout
//! tokens (a start, rows of patch tokens each closed by a row token, an end)
//! and the tokenizer's output is prefixed with BOS.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use super::read_json;
use super::resample::{resize_bilinear_bytes, Frame8};
use crate::error::{Error, Result};

const IMAGE_PROMPT: &str = "<|image|>";
const QUESTION_PUNCTUATION: &[char] = &['.', ',', '!', '?', ';', ':', '\u{2026}'];
const QUESTION_CLOSERS: &[char] = &['"', '\'', '\u{201d}', '\u{2019}', ')', ']', '}'];
const QUESTION_DELIMITERS: &[char] = &['"', '\'', '`', '\u{201c}', '\u{201d}', '\u{2018}', '\u{2019}', '[', ']', '(', ')', '{', '}'];

/// `s` without a leading `words` (ASCII case-insensitive, whitespace
/// between them) followed by whitespace, if it starts so.
fn strip_words<'a>(s: &'a str, words: &[&str]) -> Option<&'a str> {
    let mut rest = s;
    for w in words {
        if rest.len() < w.len() || !rest.is_char_boundary(w.len()) || !rest[..w.len()].eq_ignore_ascii_case(w) {
            return None;
        }
        rest = &rest[w.len()..];
        let trimmed = rest.trim_start();
        if trimmed.len() == rest.len() {
            return None;
        }
        rest = trimmed;
    }
    Some(rest)
}

/// `s` without a leading `task:`-style label (`task`, `instruction`,
/// `language instruction`, `language_instruction` or `goal`, then `:` or
/// `-`), if it has one.
fn strip_label(s: &str) -> Option<&str> {
    for label in ["task", "instruction", "language_instruction", "language instruction", "goal"] {
        if s.len() >= label.len() && s.is_char_boundary(label.len()) && s[..label.len()].eq_ignore_ascii_case(label) {
            let rest = s[label.len()..].trim_start();
            if let Some(r) = rest.strip_prefix([':', '-']) {
                return Some(r.trim_start());
            }
        }
    }
    None
}

/// The task as the policy writes it into the prompt: whitespace collapsed,
/// surrounding quotes and brackets, a leading `task:` label or `the task is
/// to`, and trailing punctuation removed (repeatedly), several sentences
/// joined by `; `, lower case.
#[must_use]
pub fn normalize_task(text: &str) -> String {
    let mut s = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut previous = String::new();
    while !s.is_empty() && s != previous {
        previous.clone_from(&s);
        let mut t: &str = s.trim().trim_matches(QUESTION_DELIMITERS).trim();
        if let Some(r) = strip_label(t) {
            t = r.trim();
        }
        if let Some(r) = strip_words(t, &["the", "task", "is", "to"]).or_else(|| strip_words(t, &["your", "task", "is", "to"])) {
            t = r.trim();
        }
        let t = t.trim_end_matches(QUESTION_PUNCTUATION).trim_end();
        let t = t.trim_end_matches(QUESTION_CLOSERS).trim_end();
        let t = t.trim_end_matches(QUESTION_PUNCTUATION).trim_end();
        s = t.to_owned();
    }
    let chunks: Vec<&str> = s.split(['.', '!', '?']).map(str::trim).filter(|c| !c.is_empty()).collect();
    if chunks.len() > 1 {
        s = chunks.join("; ");
    }
    s.to_lowercase()
}

/// The discrete token of each normalised state value: `[-1, 1]` (clamped;
/// NaN as 0) onto `bins` levels, rounded half to even.
#[must_use]
pub fn state_bins(state: &[f32], bins: usize) -> Vec<usize> {
    let top = bins.saturating_sub(1);
    state
        .iter()
        .map(|&x| {
            let x = if x.is_nan() { 0.0 } else { x.clamp(-1.0, 1.0) };
            let v = ((x + 1.0) / 2.0 * top as f32).round_ties_even();
            (v.max(0.0) as usize).min(top)
        })
        .collect()
}

/// The robot prompt: one image placeholder per camera (numbered when there
/// are several), then the user turn naming the task, the setup, the state
/// tokens and the control mode, and the open assistant turn with the action
/// trigger. Empty setup or control texts stay empty and unmarked.
#[must_use]
pub fn robot_prompt(task: &str, state: &[usize], setup: &str, control: &str, marked: (bool, bool), images: usize) -> String {
    let wrap = |text: &str, tag: &str, on: bool| {
        let (start, end) = (format!("<{tag}_start>"), format!("<{tag}_end>"));
        if text.is_empty() || !on || (text.starts_with(&start) && text.ends_with(&end)) {
            text.to_owned()
        } else {
            format!("{start}{text}{end}")
        }
    };
    let state_clause = if state.is_empty() {
        String::new()
    } else {
        let tokens: String = state.iter().map(|b| format!("<state_{b}>")).collect();
        format!(" The current state of the robot is <state_start>{tokens}<state_end>.")
    };
    let prompt = format!(
        "The task is to {task}. The setup is {}.{state_clause} The expected control mode is {}. Given these, what action should the robot take to complete the task?",
        wrap(setup, "setup", marked.0),
        wrap(control, "control", marked.1),
    );
    let prefix = match images {
        0 => String::new(),
        1 => IMAGE_PROMPT.to_owned(),
        n => (1..=n).map(|i| format!("Image {i}{IMAGE_PROMPT}")).collect(),
    };
    format!("{prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<action_output>")
}

/// For each image token of a `h x w` patch grid pooled in `ph x pw` blocks,
/// the patch indices it pools (`-1` for the slots the grid is padded with,
/// half the padding before and the rest after), rows of tokens in order.
#[must_use]
pub fn pooling_indices(h: usize, w: usize, ph: usize, pw: usize) -> (usize, usize, Vec<i32>) {
    let (th, tw) = (h.div_ceil(ph), w.div_ceil(pw));
    let (top, left) = ((th * ph - h) / 2, (tw * pw - w) / 2);
    let mut idx = Vec::with_capacity(th * tw * ph * pw);
    for ty in 0..th {
        for tx in 0..tw {
            for dy in 0..ph {
                for dx in 0..pw {
                    let (y, x) = ((ty * ph + dy).wrapping_sub(top), (tx * pw + dx).wrapping_sub(left));
                    idx.push(if y < h && x < w { (y * w + x) as i32 } else { -1 });
                }
            }
        }
    }
    (th, tw, idx)
}

/// One observation's model inputs.
#[derive(Debug, Clone)]
pub struct MolmoAct2Inputs {
    /// The prompt text before the image placeholders are expanded.
    pub prompt: String,
    /// Prompt token ids, BOS first.
    pub ids: Vec<i32>,
    /// Crops `[crops][patches][patch_len]` in `[-1, 1]`.
    pub pixels: Vec<f32>,
    /// Crops.
    pub crops: usize,
    /// Per image token, `pool` patch indices into all crops (`-1` pads).
    pub pooling: Vec<i32>,
    /// Patches pooled per image token.
    pub pool: usize,
}

/// The release's inference preprocessing.
pub struct MolmoAct2Processor {
    tokenizer: Tokenizer,
    bos: u32,
    crop: (usize, usize),
    patch: usize,
    pool: (usize, usize),
    col_tokens: bool,
    state_tokens: usize,
    marked: (bool, bool),
    /// The robot setup named in the prompt (the policy config's
    /// `setup_type`).
    pub setup_type: String,
    /// The control mode named in the prompt (the policy config's
    /// `control_mode`).
    pub control_mode: String,
    /// Normalise the task text (the policy config's `normalize_language`).
    pub normalize_language: bool,
}

impl std::fmt::Debug for MolmoAct2Processor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MolmoAct2Processor").field("crop", &self.crop).field("setup_type", &self.setup_type).finish_non_exhaustive()
    }
}

fn pair(v: Option<&Value>, a: &str, b: &str) -> Option<(usize, usize)> {
    let v = v?;
    match v {
        Value::Array(x) => Some((x.first()?.as_u64()? as usize, x.get(1)?.as_u64()? as usize)),
        Value::Object(_) => Some((v.get(a)?.as_u64()? as usize, v.get(b)?.as_u64()? as usize)),
        _ => None,
    }
}

impl MolmoAct2Processor {
    /// The preprocessing of a release directory (`config.json`,
    /// `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`)
    /// for a robot described by `setup_type` and `control_mode`.
    ///
    /// # Errors
    /// On a missing or malformed file, or a processor variant this engine
    /// does not implement (crop modes other than one resized crop,
    /// resampling other than bilinear, other pixel normalisation).
    pub fn load(dir: &Path, setup_type: &str, control_mode: &str) -> Result<Self> {
        let c = read_json(&dir.join("config.json"))?;
        let pc = read_json(&dir.join("processor_config.json"))?;
        let ip = pc.get("image_processor").ok_or_else(|| Error::Config("processor config gives no image_processor".into()))?;
        let unsupported = |what: &str| Err(Error::Config(format!("MolmoAct2 image processing with {what} is not implemented")));
        if ip.get("crop_mode").and_then(Value::as_str) != Some("resize") {
            return unsupported("overlapping crops");
        }
        if ip.get("resample").and_then(Value::as_u64).unwrap_or(2) != 2 {
            return unsupported("resampling other than bilinear");
        }
        for k in ["image_mean", "image_std"] {
            let ok = ip.get(k).and_then(Value::as_array).is_some_and(|v| v.len() == 3 && v.iter().all(|x| x.as_f64() == Some(0.5)));
            if !ok {
                return unsupported("normalisation other than to [-1, 1]");
            }
        }
        let crop = pair(ip.get("size"), "height", "width").ok_or_else(|| Error::Config("image processor gives no size".into()))?;
        let pool = pair(ip.get("pooling_size"), "", "").unwrap_or((2, 2));
        let patch = ip.get("patch_size").and_then(Value::as_u64).unwrap_or(14) as usize;
        if patch == 0 || crop.0 % patch != 0 || crop.1 % patch != 0 || pool.0 == 0 || pool.1 == 0 {
            return Err(Error::Config("malformed MolmoAct2 image processor sizes".into()));
        }
        let col = pc.get("image_use_col_tokens").and_then(Value::as_bool).unwrap_or(true);
        let col_tokens = pc.get("use_single_crop_col_tokens").and_then(Value::as_bool).unwrap_or(col);

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        let tc = read_json(&dir.join("tokenizer_config.json"))?;
        let token_name = |k: &str| match tc.get(k) {
            Some(Value::String(s)) => Some(s.clone()),
            Some(v) => v.get("content").and_then(Value::as_str).map(str::to_owned),
            None => None,
        };
        let bos_name = token_name("bos_token").or_else(|| token_name("eos_token")).ok_or_else(|| Error::Config("tokenizer config gives no BOS or EOS token".into()))?;
        let bos = tokenizer.token_to_id(&bos_name).ok_or_else(|| Error::Tokenizer(format!("{bos_name} is not in the vocabulary")))?;
        let flag = |k: &str| c.get(k).and_then(Value::as_bool).unwrap_or(true);
        Ok(Self {
            tokenizer,
            bos,
            crop,
            patch,
            pool,
            col_tokens,
            state_tokens: c.get("num_state_tokens").and_then(Value::as_u64).unwrap_or(256) as usize,
            marked: (flag("add_setup_tokens"), flag("add_control_tokens")),
            setup_type: setup_type.to_owned(),
            control_mode: control_mode.to_owned(),
            normalize_language: true,
        })
    }

    /// Crop height and width in pixels.
    #[must_use]
    pub fn crop(&self) -> (usize, usize) {
        self.crop
    }

    /// The layout tokens of one image: a start, per row of image tokens the
    /// patch tokens (each closed by a row token when the release uses them),
    /// an end.
    #[must_use]
    pub fn image_token_text(&self) -> String {
        let (gh, gw) = (self.crop.0 / self.patch, self.crop.1 / self.patch);
        let (th, tw) = (gh.div_ceil(self.pool.0), gw.div_ceil(self.pool.1));
        let mut row = "<im_patch>".repeat(tw);
        if self.col_tokens {
            row.push_str("<im_col>");
        }
        format!("<im_start>{}<im_end>", row.repeat(th))
    }

    /// The prompt ids of `prompt` with its image placeholders expanded,
    /// BOS first.
    ///
    /// # Errors
    /// When the tokenizer fails.
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        let text = prompt.replace(IMAGE_PROMPT, &self.image_token_text());
        let enc = self.tokenizer.encode(text, true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids: Vec<i32> = Vec::with_capacity(enc.get_ids().len() + 1);
        if enc.get_ids().first() != Some(&self.bos) {
            ids.push(self.bos as i32);
        }
        ids.extend(enc.get_ids().iter().map(|&i| i as i32));
        Ok(ids)
    }

    /// One frame's crop as patches `[patches][patch * patch * 3]` (rows of
    /// pixels, channels innermost) in `[-1, 1]`.
    #[must_use]
    pub fn crop_patches(&self, frame: &Frame8) -> Vec<f32> {
        let (h, w) = self.crop;
        let p = self.patch;
        let r = resize_bilinear_bytes(frame, h, w);
        let mut out = Vec::with_capacity(3 * h * w);
        for py in 0..h / p {
            for px in 0..w / p {
                for dy in 0..p {
                    for dx in 0..p {
                        for c in 0..3 {
                            let v = f32::from(r.data[(c * h + py * p + dy) * w + px * p + dx]) / 255.0;
                            out.push(v * 2.0 - 1.0);
                        }
                    }
                }
            }
        }
        out
    }

    /// The model inputs of one observation: a frame per camera, the state
    /// normalised to `[-1, 1]` and the task.
    ///
    /// # Errors
    /// When there is no frame, or the tokenizer fails.
    pub fn prepare(&self, frames: &[Frame8], state: &[f32], task: &str) -> Result<MolmoAct2Inputs> {
        if frames.is_empty() {
            return Err(Error::Request("MolmoAct2 needs at least one camera frame".into()));
        }
        let task = if self.normalize_language { normalize_task(task) } else { task.to_owned() };
        let bins = state_bins(state, self.state_tokens);
        let prompt = robot_prompt(&task, &bins, &self.setup_type, &self.control_mode, self.marked, frames.len());
        let ids = self.tokenize(&prompt)?;
        let (gh, gw) = (self.crop.0 / self.patch, self.crop.1 / self.patch);
        let (_, _, local) = pooling_indices(gh, gw, self.pool.0, self.pool.1);
        let patches = (gh * gw) as i32;
        let mut pixels = Vec::new();
        let mut pooling = Vec::with_capacity(local.len() * frames.len());
        for (i, f) in frames.iter().enumerate() {
            pixels.extend(self.crop_patches(f));
            pooling.extend(local.iter().map(|&j| if j < 0 { -1 } else { j + i as i32 * patches }));
        }
        Ok(MolmoAct2Inputs { prompt, ids, pixels, crops: frames.len(), pooling, pool: self.pool.0 * self.pool.1 })
    }
}
