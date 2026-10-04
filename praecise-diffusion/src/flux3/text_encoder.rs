//! FLUX 3 text context from a Qwen3-VL language model.
//!
//! The prompt is wrapped in the chat template as a user turn followed by the
//! assistant header, tokenized, and right-padded to a fixed length (or to the
//! next multiple of [`PAD_MULTIPLE`]). The context is the hidden states after
//! layers 4, 8, ..., 32 stacked per token, so `[tokens][8 * hidden]`.
//!
//! Only the text decoder is used. With text alone, the multi-axis rotary
//! positions of Qwen3-VL are equal on every axis and reduce to ordinary
//! rotary positions; positions run sequentially through the padding, which
//! attends only to the prompt tokens, as the reference padding mask makes it.

use std::path::Path;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::qwen3::{self, Layout, Qwen3Config};
use crate::safetensors::SafeTensors;
use tokenizers::Tokenizer;

/// Hidden states stacked into the context, 1-based layer numbers.
pub const CONTEXT_LAYERS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
/// Bucketed prompts are padded to a multiple of this many tokens.
pub const PAD_MULTIPLE: usize = 80;
/// Longest prompt, template included.
pub const MAX_TOKENS: usize = 8192;
const LAYOUT: Layout = Layout { prefix: "model.language_model.", final_norm: false };

/// Prompt token ids padded to `ids.len()`, the first `real` being the prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTokens {
    /// Token ids including padding.
    pub ids: Vec<i32>,
    /// Number of prompt (non-padding) tokens.
    pub real: usize,
}

/// The loaded text encoder.
pub struct TextEncoder {
    backend: Backend,
    cfg: Qwen3Config,
    w: Weights,
    theta: f32,
    tokenizer: Tokenizer,
    pad: u32,
    layers: Vec<usize>,
    exact: bool,
}

impl std::fmt::Debug for TextEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextEncoder").field("device", &self.backend.name()).field("layers", &self.layers).finish_non_exhaustive()
    }
}

impl TextEncoder {
    /// Load from a directory holding `config.json` (with a `text_config`),
    /// `tokenizer.json`, `tokenizer_config.json` and the safetensors shards,
    /// reading the hidden states after `layers` (1-based; the released
    /// policies use [`CONTEXT_LAYERS`]).
    ///
    /// # Errors
    /// On missing or malformed files, missing weights or no usable backend.
    pub fn load(dir: &Path, layers: &[usize], opts: LoadOptions) -> Result<Self> {
        let read = |n: &str| std::fs::read(dir.join(n)).map_err(|e| Error::Config(format!("{n}: {e}")));
        let config: serde_json::Value = serde_json::from_slice(&read("config.json")?).map_err(|e| Error::Config(format!("config.json: {e}")))?;
        let text = config.get("text_config").cloned().ok_or_else(|| Error::Config("config.json has no text_config".into()))?;
        let cfg: Qwen3Config = serde_json::from_value(text).map_err(|e| Error::Config(format!("text_config: {e}")))?;
        let theta = cfg.theta()? as f32;
        let last = layers.iter().copied().max().ok_or_else(|| Error::Config("no context layers requested".into()))?;
        if layers.contains(&0) || last > cfg.num_hidden_layers {
            return Err(Error::Config(format!("context layers {layers:?} outside 1..={}", cfg.num_hidden_layers)));
        }
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let tc: serde_json::Value = serde_json::from_slice(&read("tokenizer_config.json")?).map_err(|e| Error::Config(format!("tokenizer_config.json: {e}")))?;
        let pad_token = tc.get("pad_token").and_then(|v| v.as_str().or_else(|| v.get("content").and_then(|c| c.as_str()))).ok_or_else(|| Error::Config("tokenizer_config.json names no pad_token".into()))?;
        let pad = tokenizer.token_to_id(pad_token).ok_or_else(|| Error::Tokenizer(format!("pad token {pad_token} is not in the vocabulary")))?;
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| Error::Config(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        let st = SafeTensors::open(&shards)?;
        let backend = Backend::select(opts.cpu_threads)?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "text encoder backend selected");
        let w = Weights::load(&backend, &st, &cfg.weight_specs(LAYOUT, last, opts.precision.wtype())?)?;
        Ok(Self { backend, cfg, w, theta, tokenizer, pad, layers: layers.to_vec(), exact: opts.precision == Precision::F32 })
    }

    /// Width of one context token.
    #[must_use]
    pub fn context_width(&self) -> usize {
        self.cfg.hidden_size as usize * self.layers.len()
    }

    /// Template, tokenize and pad `prompt`: to `fixed_length` tokens when
    /// given (truncating longer prompts), else to the next multiple of
    /// [`PAD_MULTIPLE`] (at most [`MAX_TOKENS`]).
    ///
    /// # Errors
    /// On a tokenizer failure or a zero `fixed_length`.
    pub fn tokens(&self, prompt: &str, fixed_length: Option<usize>) -> Result<PromptTokens> {
        if fixed_length == Some(0) {
            return Err(Error::Config("fixed text length must be positive".into()));
        }
        let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let enc = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&t| t as i32).collect();
        ids.truncate(fixed_length.unwrap_or(MAX_TOKENS));
        let real = ids.len();
        let n = fixed_length.unwrap_or_else(|| (real.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE).min(MAX_TOKENS));
        ids.resize(n, self.pad as i32);
        Ok(PromptTokens { ids, real })
    }

    /// Context `[tokens][context_width]` for already padded tokens.
    ///
    /// # Errors
    /// On a backend failure.
    pub fn encode_tokens(&self, t: &PromptTokens) -> Result<Vec<f32>> {
        let n = t.ids.len();
        let mut g = Graph::new(&self.backend)?;
        let io = qwen3::build(&mut g, &self.cfg, &self.w, LAYOUT, n as i64, &self.layers, self.theta, self.exact);
        g.finish(&[io.out])?;
        g.set_i32(io.tokens, &t.ids);
        // Text-only prompts take plain sequential positions, padding included.
        let pos: Vec<i32> = (0..n as i32).collect();
        g.set_i32(io.positions, &pos);
        g.set_f16(io.mask, &qwen3::mask(n, t.real));
        g.compute()?;
        Ok(g.read_f32(io.out))
    }

    /// Context for `prompt` (see [`TextEncoder::tokens`]); returns the
    /// context and its token count.
    ///
    /// # Errors
    /// On a tokenizer or backend failure.
    pub fn encode(&self, prompt: &str, fixed_length: Option<usize>) -> Result<(Vec<f32>, usize)> {
        let t = self.tokens(prompt, fixed_length)?;
        Ok((self.encode_tokens(&t)?, t.ids.len()))
    }
}

#[cfg(test)]
mod parity {
    //! Against fixtures from `tests/parity/make_flux3_text_fixtures.py`.
    use super::*;

    fn run(precision: Precision, min_cos: f64) {
        let root = std::path::PathBuf::from(std::env::var("PRAECISE_FLUX3_TEXT_PARITY").expect("PRAECISE_FLUX3_TEXT_PARITY names the fixture dir"));
        let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("cases.json")).unwrap()).unwrap();
        let layers: Vec<usize> = serde_json::from_value(cases["layers"].clone()).unwrap();
        let te = TextEncoder::load(&root.join("model"), &layers, LoadOptions { precision, cpu_threads: 8, device: None }).unwrap();
        for (i, c) in cases["cases"].as_array().unwrap().iter().enumerate() {
            let fixed = c["fixed_length"].as_u64().map(|v| v as usize);
            let t = te.tokens(c["prompt"].as_str().unwrap(), fixed).unwrap();
            let want_ids: Vec<i32> = serde_json::from_value(c["ids"].clone()).unwrap();
            assert_eq!(t.ids, want_ids, "case {i}: token ids");
            assert_eq!(t.real, c["real"].as_u64().unwrap() as usize);
            let ours = te.encode_tokens(&t).unwrap();
            let want: Vec<f32> = std::fs::read(root.join(format!("ctx{i}.f32"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            assert_eq!(ours.len(), want.len());
            let (mut dot, mut a2, mut b2, mut d2) = (0f64, 0f64, 0f64, 0f64);
            for (&a, &b) in ours.iter().zip(&want) {
                let (a, b) = (f64::from(a), f64::from(b));
                dot += a * b;
                a2 += a * a;
                b2 += b * b;
                d2 += (a - b) * (a - b);
            }
            let cos = dot / (a2.sqrt() * b2.sqrt());
            let width = ours.len() / t.ids.len();
            let real_end = t.real * width;
            let rc = {
                let (o, w) = (&ours[..real_end], &want[..real_end]);
                let d: f64 = o.iter().zip(w).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                let na: f64 = o.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
                let nb: f64 = w.iter().map(|&b| f64::from(b) * f64::from(b)).sum();
                d / (na.sqrt() * nb.sqrt())
            };
            println!("case {i} {precision:?}: prompt rows cosine {rc:.6}");
            println!("case {i} {precision:?}: {} tokens ({} real), cosine {cos:.6} rel {:.2e}", t.ids.len(), t.real, (d2 / b2).sqrt());
            assert!(cos >= min_cos);
        }
    }

    #[test]
    #[ignore = "needs PRAECISE_FLUX3_TEXT_PARITY fixtures"]
    fn flux3_text_parity_f32() {
        run(Precision::F32, 0.999_99);
    }

    #[test]
    #[ignore = "needs PRAECISE_FLUX3_TEXT_PARITY fixtures"]
    fn flux3_text_parity_bf16() {
        run(Precision::Bf16, 0.999);
    }
}
