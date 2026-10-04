//! LTX-2.3 prompt connectors.
//!
//! The text encoder's hidden states from every layer (embeddings included)
//! are normalised per token and layer over the feature axis, flattened and
//! projected once per stream to that stream's width. Each stream then runs
//! its own small transformer: the valid prompt tokens are moved to the front
//! and every padded slot after them is filled with a learned register (the
//! table repeats along the sequence), so every slot attends to every other
//! with no mask. Blocks are plain pre-norm self-attention (gated, split
//! rotary embeddings over token index) and feed-forward; a final RMS norm
//! closes each stream.

use std::path::Path;

use serde::Deserialize;

use super::single_file::{header_config, open_part, Part};
use super::{attention, feed_forward, rope_tables, Ctx};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Norm epsilon of the per-token feature norm and of every block norm.
const EPS: f32 = 1e-6;

/// `connectors/config.json`.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct ConnectorsConfig {
    pub caption_channels: u64,
    pub text_proj_in_factor: u64,
    pub video_connector_num_attention_heads: u64,
    pub video_connector_attention_head_dim: u64,
    pub video_connector_num_layers: u64,
    pub video_connector_num_learnable_registers: Option<u64>,
    pub video_gated_attn: bool,
    pub audio_connector_num_attention_heads: u64,
    pub audio_connector_attention_head_dim: u64,
    pub audio_connector_num_layers: u64,
    pub audio_connector_num_learnable_registers: Option<u64>,
    pub audio_gated_attn: bool,
    pub connector_rope_base_seq_len: u64,
    pub rope_theta: f64,
    pub rope_type: String,
    pub per_modality_projections: bool,
    pub video_hidden_dim: u64,
    pub audio_hidden_dim: u64,
    pub proj_bias: bool,
}

/// One stream's connector shape.
#[derive(Debug, Clone, Copy)]
struct Stream {
    heads: u64,
    head_dim: u64,
    layers: u64,
    registers: u64,
}

impl Stream {
    fn width(self) -> u64 {
        self.heads * self.head_dim
    }
}

impl ConnectorsConfig {
    /// Refuse layouts other than the released LTX-2.3 one.
    ///
    /// # Errors
    /// When the configuration names a different layout.
    pub fn validate(&self) -> Result<()> {
        let (v, a) = (self.video(), self.audio());
        let checks = [
            (self.per_modality_projections && self.proj_bias, "a shared prompt projection"),
            (self.video_gated_attn && self.audio_gated_attn, "ungated attention"),
            (self.rope_type == "split", "rotary layouts other than split"),
            (v.registers > 0 && a.registers > 0, "connectors without learned registers"),
            (v.width() == self.video_hidden_dim && a.width() == self.audio_hidden_dim, "stream widths other than the connector widths"),
            (v.head_dim % 2 == 0 && a.head_dim % 2 == 0, "odd head widths"),
        ];
        for (ok, what) in checks {
            if !ok {
                return Err(Error::Config(format!("prompt connectors: {what} not implemented")));
            }
        }
        Ok(())
    }

    fn video(&self) -> Stream {
        Stream {
            heads: self.video_connector_num_attention_heads,
            head_dim: self.video_connector_attention_head_dim,
            layers: self.video_connector_num_layers,
            registers: self.video_connector_num_learnable_registers.unwrap_or(0),
        }
    }

    fn audio(&self) -> Stream {
        Stream {
            heads: self.audio_connector_num_attention_heads,
            head_dim: self.audio_connector_attention_head_dim,
            layers: self.audio_connector_num_layers,
            registers: self.audio_connector_num_learnable_registers.unwrap_or(0),
        }
    }

    /// Flattened text encoder feature width per token.
    #[must_use]
    pub fn feature_width(&self) -> u64 {
        self.caption_channels * self.text_proj_in_factor
    }

    /// Every connector weight.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let k = self.feature_width();
        let mut v = Vec::new();
        for (p, s) in [("video", self.video()), ("audio", self.audio())] {
            let d = s.width();
            v.push(WeightSpec::new(format!("{p}_text_proj_in.weight"), &[d, k], linear));
            v.push(WeightSpec::new(format!("{p}_text_proj_in.bias"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{p}_connector.learnable_registers"), &[s.registers, d], WType::F32));
            for i in 0..s.layers {
                let b = format!("{p}_connector.transformer_blocks.{i}");
                for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                    v.push(WeightSpec::new(format!("{b}.attn1.{n}.weight"), &[d, d], linear));
                    v.push(WeightSpec::new(format!("{b}.attn1.{n}.bias"), &[d], WType::F32));
                }
                v.push(WeightSpec::new(format!("{b}.attn1.norm_q.weight"), &[d], WType::F32));
                v.push(WeightSpec::new(format!("{b}.attn1.norm_k.weight"), &[d], WType::F32));
                v.push(WeightSpec::new(format!("{b}.attn1.to_gate_logits.weight"), &[s.heads, d], WType::F32));
                v.push(WeightSpec::new(format!("{b}.attn1.to_gate_logits.bias"), &[s.heads], WType::F32));
                v.push(WeightSpec::new(format!("{b}.ff.net.0.proj.weight"), &[4 * d, d], linear));
                v.push(WeightSpec::new(format!("{b}.ff.net.0.proj.bias"), &[4 * d], WType::F32));
                v.push(WeightSpec::new(format!("{b}.ff.net.2.weight"), &[d, 4 * d], linear));
                v.push(WeightSpec::new(format!("{b}.ff.net.2.bias"), &[d], WType::F32));
            }
        }
        v
    }

    /// Rotary tables `(cos, sin)` over `[token][head][head width]` for a
    /// sequence of `n` slots.
    fn rotary(&self, s: Stream, n: usize) -> (Vec<f32>, Vec<f32>) {
        let pos: Vec<f32> = (0..n).map(|i| i as f32).collect();
        rope_tables(&[(&pos, self.connector_rope_base_seq_len as f32)], s.width() as usize, s.heads as usize, self.rope_theta)
    }

    /// Normalise per-layer text encoder states `[token][caption][layer]` of
    /// the valid prompt tokens: RMS over the caption axis for every token and
    /// layer, flattened caption-major.
    #[must_use]
    pub fn normalise(&self, hidden: &[f32]) -> Vec<f32> {
        let (c, l) = (self.caption_channels as usize, self.text_proj_in_factor as usize);
        let mut out = hidden.to_vec();
        for tok in out.chunks_exact_mut(c * l) {
            for layer in 0..l {
                let ms = (0..c).map(|i| f64::from(tok[i * l + layer]).powi(2)).sum::<f64>() / c as f64;
                let r = (1.0 / (ms + f64::from(EPS)).sqrt()) as f32;
                for i in 0..c {
                    tok[i * l + layer] *= r;
                }
            }
        }
        out
    }
}

fn stream(g: &mut Graph, w: &Weights, p: &str, cfg: &ConnectorsConfig, s: Stream, (feats, ids, rope): (Tn, Option<Tn>, (Tn, Tn)), exact: bool) -> Tn {
    let c = Ctx { eps: EPS, exact };
    let scale = ((s.width() as f64) / cfg.caption_channels as f64).sqrt() as f32;
    let x = g.scale_bias(feats, scale, 0.0);
    let mut x = g.linear_b(w.get(&format!("{p}_text_proj_in.weight")), w.get(&format!("{p}_text_proj_in.bias")), x);
    if let Some(ids) = ids {
        let r = g.get_rows(w.get(&format!("{p}_connector.learnable_registers")), ids);
        x = g.concat(x, r, 1);
    }
    let (hd, heads) = (s.head_dim as i64, s.heads as i64);
    for i in 0..s.layers {
        let b = format!("{p}_connector.transformer_blocks.{i}");
        let h = g.rms_norm(x, c.eps);
        let h = attention(g, w, &format!("{b}.attn1"), c, (hd, heads), h, h, Some((rope, rope)));
        x = g.add(x, h);
        let h = g.rms_norm(x, c.eps);
        let h = feed_forward(g, w, &format!("{b}.ff"), h, exact);
        x = g.add(x, h);
    }
    g.rms_norm(x, c.eps)
}

/// The two prompt connectors.
pub struct Ltx2Connectors {
    backend: Backend,
    cfg: ConnectorsConfig,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for Ltx2Connectors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Connectors").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl Ltx2Connectors {
    /// Load `connectors/` of a checkpoint.
    ///
    /// # Errors
    /// On an unsupported configuration, missing weights or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: ConnectorsConfig = parse(files.json("connectors/config.json")?, "connectors config")?;
        cfg.validate()?;
        let backend = opts.backend()?;
        let st = SafeTensors::open(&files.weights("connectors")?)?;
        let w = Weights::load(&backend, &st, &cfg.weight_specs(opts.precision.wtype()))?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32 })
    }

    /// Load the connectors of a single-file checkpoint.
    ///
    /// # Errors
    /// As [`Self::load`].
    pub fn load_single_file(path: &Path, opts: LoadOptions) -> Result<Self> {
        let st = SafeTensors::open(&[path.to_path_buf()])?;
        let header = header_config(&st)?;
        let k = st
            .get("text_embedding_projection.video_aggregate_embed.weight")
            .and_then(|v| v.shape.get(1).copied())
            .ok_or_else(|| Error::MissingTensor("text_embedding_projection.video_aggregate_embed.weight".into()))?;
        let cfg = ConnectorsConfig::from_single_file(&header["transformer"], k)?;
        let st = open_part(st, Part::Connectors)?;
        let backend = opts.backend()?;
        let w = Weights::load(&backend, &st, &cfg.weight_specs(opts.precision.wtype()))?;
        Ok(Self { backend, cfg, w, exact: opts.precision == Precision::F32 })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &ConnectorsConfig {
        &self.cfg
    }

    /// Prompt features for the video and audio streams, each
    /// `[seq_len][stream width]`, from the text encoder states of the valid
    /// prompt tokens `[token][caption][layer]` (padding dropped) and the
    /// padded prompt length the transformer will see.
    ///
    /// # Errors
    /// When the states are not whole tokens, there are none or more than
    /// `seq_len`, `seq_len` is not a whole number of register tables, or the
    /// backend fails.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.cfg;
        let k = cfg.feature_width() as usize;
        let n = hidden.len() / k;
        let (v, a) = (cfg.video(), cfg.audio());
        if hidden.len() % k != 0 || n == 0 || n > seq_len || seq_len % v.registers as usize != 0 || seq_len % a.registers as usize != 0 {
            return Err(Error::Request("prompt connector inputs disagree with the sequence length".into()));
        }
        let mut g = Graph::new(&self.backend)?;
        let feats = g.input(sys::GGML_TYPE_F32, &[k as i64, n as i64]);
        let pad = (seq_len - n) as i64;
        let mut io = Vec::new();
        let mut outs = Vec::new();
        for (p, s) in [("video", v), ("audio", a)] {
            let ids = (pad > 0).then(|| g.input(sys::GGML_TYPE_I32, &[pad]));
            let hd = s.head_dim as i64;
            let rope = (g.input(sys::GGML_TYPE_F32, &[hd, s.heads as i64, seq_len as i64]), g.input(sys::GGML_TYPE_F32, &[hd, s.heads as i64, seq_len as i64]));
            outs.push(stream(&mut g, &self.w, p, cfg, s, (feats, ids, rope), self.exact));
            io.push((s, ids, rope));
        }
        g.finish(&outs)?;
        g.set_f32(feats, &cfg.normalise(hidden));
        for (s, ids, (cos, sin)) in io {
            if let Some(ids) = ids {
                let r = s.registers as usize;
                g.set_i32(ids, &(n..seq_len).map(|i| (i % r) as i32).collect::<Vec<_>>());
            }
            let (c, sn) = cfg.rotary(s, seq_len);
            g.set_f32(cos, &c);
            g.set_f32(sin, &sn);
        }
        g.compute()?;
        Ok((g.read_f32(outs[0]), g.read_f32(outs[1])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_is_per_token_and_layer() {
        let cfg: ConnectorsConfig = serde_json::from_value(serde_json::json!({
            "caption_channels": 2, "text_proj_in_factor": 2,
            "video_connector_num_attention_heads": 1, "video_connector_attention_head_dim": 2,
            "video_connector_num_layers": 1, "video_connector_num_learnable_registers": 2, "video_gated_attn": true,
            "audio_connector_num_attention_heads": 1, "audio_connector_attention_head_dim": 2,
            "audio_connector_num_layers": 1, "audio_connector_num_learnable_registers": 2, "audio_gated_attn": true,
            "connector_rope_base_seq_len": 4096, "rope_theta": 10000.0, "rope_type": "split",
            "per_modality_projections": true, "video_hidden_dim": 2, "audio_hidden_dim": 2, "proj_bias": true
        }))
        .unwrap();
        cfg.validate().unwrap();
        // One token, [caption][layer]: layer 0 is (3, 4), layer 1 is (1, 1).
        let y = cfg.normalise(&[3.0, 1.0, 4.0, 1.0]);
        let r0 = (12.5f32 + 1e-6).sqrt();
        assert!((y[0] - 3.0 / r0).abs() < 1e-6 && (y[2] - 4.0 / r0).abs() < 1e-6);
        assert!((y[1] - 1.0).abs() < 1e-5 && (y[3] - 1.0).abs() < 1e-5);
    }
}
