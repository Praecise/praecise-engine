//! Tensor names of the single-file LTX-2.3 release.
//!
//! The release ships one safetensors file holding every component under its
//! original module names, with the component configurations as JSON in the
//! header metadata. The native components load per-component names; this
//! maps each original name to its component and the name that component
//! loads.

use super::connectors::ConnectorsConfig;
use crate::error::{Error, Result};
use crate::safetensors::SafeTensors;

/// A component of the single-file checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// The audio-video transformer.
    Transformer,
    /// The prompt connectors with their input projections.
    Connectors,
    /// The audio autoencoder.
    AudioVae,
    /// The video autoencoder.
    VideoVae,
}

const TRANSFORMER: &str = "model.diffusion_model.";

/// Substring renames of transformer names, applied in order.
const TRANSFORMER_RENAMES: [(&str, &str); 9] = [
    ("patchify_proj", "proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
];

/// Leading timestep-embedding modules, renamed whole.
const TIME_EMBEDS: [(&str, &str); 4] = [
    ("adaln_single.", "time_embed."),
    ("audio_adaln_single.", "audio_time_embed."),
    ("prompt_adaln_single.", "prompt_adaln."),
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
];

fn connector(stream: &str, rest: &str) -> String {
    let rest = rest.replace("transformer_1d_blocks", "transformer_blocks").replace("q_norm", "norm_q").replace("k_norm", "norm_k");
    format!("{stream}_connector.{rest}")
}

/// The component and component-local name of one tensor of the single file,
/// or `None` for a tensor no native component loads yet.
#[must_use]
pub fn component_name(key: &str) -> Option<(Part, String)> {
    for stream in ["video", "audio"] {
        if let Some(rest) = key.strip_prefix(&format!("{TRANSFORMER}{stream}_embeddings_connector.")) {
            return Some((Part::Connectors, connector(stream, rest)));
        }
        if let Some(rest) = key.strip_prefix(&format!("text_embedding_projection.{stream}_aggregate_embed.")) {
            return Some((Part::Connectors, format!("{stream}_text_proj_in.{rest}")));
        }
    }
    if let Some(rest) = key.strip_prefix(TRANSFORMER) {
        for (from, to) in TIME_EMBEDS {
            if let Some(tail) = rest.strip_prefix(from) {
                return Some((Part::Transformer, format!("{to}{tail}")));
            }
        }
        let name = TRANSFORMER_RENAMES.iter().fold(rest.to_string(), |n, (from, to)| n.replace(from, to));
        return Some((Part::Transformer, name));
    }
    if let Some(rest) = key.strip_prefix("vae.") {
        let name = rest.replace("per_channel_statistics.mean-of-means", "latents_mean").replace("per_channel_statistics.std-of-means", "latents_std");
        return Some((Part::VideoVae, name));
    }
    if let Some(rest) = key.strip_prefix("audio_vae.") {
        let name = rest.replace("per_channel_statistics.mean-of-means", "latents_mean").replace("per_channel_statistics.std-of-means", "latents_std");
        return Some((Part::AudioVae, name));
    }
    None
}

/// Open the single file as one component's tensor namespace.
///
/// # Errors
/// When the file cannot be opened or two names collide.
pub fn open_part(files: SafeTensors, part: Part) -> Result<SafeTensors> {
    files.renamed(|k| component_name(k).filter(|(p, _)| *p == part).map(|(_, n)| n))
}

/// The component configurations stored in the file header.
///
/// # Errors
/// When the header has no configuration.
pub fn header_config(files: &SafeTensors) -> Result<serde_json::Value> {
    let raw = files.metadata("config").ok_or_else(|| Error::Config("single-file checkpoint has no config metadata".into()))?;
    serde_json::from_str(raw).map_err(|e| Error::Config(format!("single-file config: {e}")))
}

impl ConnectorsConfig {
    /// The connector configuration of a single-file checkpoint, from the
    /// transformer section of its header and the width of the stacked text
    /// encoder features (`feature_width`, read off the input projection).
    ///
    /// # Errors
    /// When a field is missing or the layout is not the released one.
    pub fn from_single_file(transformer: &serde_json::Value, feature_width: u64) -> Result<Self> {
        let u = |k: &str| transformer[k].as_u64().ok_or_else(|| Error::Config(format!("single-file transformer config lacks {k}")));
        let b = |k: &str| transformer[k].as_bool().unwrap_or(false);
        let caption = u("caption_channels")?;
        let (heads, hd, aheads, ahd) = (u("connector_num_attention_heads")?, u("connector_attention_head_dim")?, u("audio_connector_num_attention_heads")?, u("audio_connector_attention_head_dim")?);
        let (layers, registers) = (u("connector_num_layers")?, u("connector_num_learnable_registers")?);
        let base = transformer["connector_positional_embedding_max_pos"][0].as_u64().ok_or_else(|| Error::Config("single-file transformer config lacks connector_positional_embedding_max_pos".into()))?;
        if feature_width % caption != 0 || transformer["text_encoder_norm_type"] != "per_token_rms" || !b("caption_proj_before_connector") {
            return Err(Error::Config("single-file prompt projection: layout not implemented".into()));
        }
        let cfg = Self {
            caption_channels: caption,
            text_proj_in_factor: feature_width / caption,
            video_connector_num_attention_heads: heads,
            video_connector_attention_head_dim: hd,
            video_connector_num_layers: layers,
            video_connector_num_learnable_registers: Some(registers),
            video_gated_attn: b("connector_apply_gated_attention"),
            audio_connector_num_attention_heads: aheads,
            audio_connector_attention_head_dim: ahd,
            audio_connector_num_layers: layers,
            audio_connector_num_learnable_registers: Some(registers),
            audio_gated_attn: b("connector_apply_gated_attention"),
            connector_rope_base_seq_len: base,
            rope_theta: transformer["positional_embedding_theta"].as_f64().unwrap_or(10_000.0),
            rope_type: transformer["rope_type"].as_str().unwrap_or("").to_string(),
            per_modality_projections: true,
            video_hidden_dim: heads * hd,
            audio_hidden_dim: aheads * ahd,
            proj_bias: true,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(k: &str) -> (Part, String) {
        component_name(k).unwrap()
    }

    #[test]
    fn transformer_names() {
        let t = |k: &str| (Part::Transformer, k.to_string());
        assert_eq!(name("model.diffusion_model.adaln_single.linear.weight"), t("time_embed.linear.weight"));
        assert_eq!(name("model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_1.bias"), t("audio_time_embed.emb.timestep_embedder.linear_1.bias"));
        assert_eq!(name("model.diffusion_model.prompt_adaln_single.linear.bias"), t("prompt_adaln.linear.bias"));
        assert_eq!(name("model.diffusion_model.audio_prompt_adaln_single.linear.bias"), t("audio_prompt_adaln.linear.bias"));
        assert_eq!(name("model.diffusion_model.av_ca_a2v_gate_adaln_single.linear.weight"), t("av_cross_attn_video_a2v_gate.linear.weight"));
        assert_eq!(name("model.diffusion_model.audio_patchify_proj.bias"), t("audio_proj_in.bias"));
        assert_eq!(name("model.diffusion_model.transformer_blocks.3.audio_to_video_attn.k_norm.weight"), t("transformer_blocks.3.audio_to_video_attn.norm_k.weight"));
        assert_eq!(name("model.diffusion_model.transformer_blocks.0.scale_shift_table_a2v_ca_audio"), t("transformer_blocks.0.audio_a2v_cross_attn_scale_shift_table"));
    }

    #[test]
    fn connector_and_vae_names() {
        let c = |k: &str| (Part::Connectors, k.to_string());
        assert_eq!(name("model.diffusion_model.video_embeddings_connector.learnable_registers"), c("video_connector.learnable_registers"));
        assert_eq!(name("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.7.attn1.q_norm.weight"), c("audio_connector.transformer_blocks.7.attn1.norm_q.weight"));
        assert_eq!(name("text_embedding_projection.video_aggregate_embed.weight"), c("video_text_proj_in.weight"));
        assert_eq!(name("audio_vae.per_channel_statistics.std-of-means"), (Part::AudioVae, "latents_std".into()));
        assert_eq!(name("vae.decoder.up_blocks.1.conv.conv.weight"), (Part::VideoVae, "decoder.up_blocks.1.conv.conv.weight".into()));
        assert_eq!(name("vae.per_channel_statistics.mean-of-means"), (Part::VideoVae, "latents_mean".into()));
        assert_eq!(component_name("vocoder.mel_stft.window"), None);
    }

    #[test]
    fn released_connector_config() {
        let t = serde_json::json!({
            "caption_channels": 3840, "connector_num_attention_heads": 32, "connector_attention_head_dim": 128,
            "audio_connector_num_attention_heads": 32, "audio_connector_attention_head_dim": 64,
            "connector_num_layers": 8, "connector_num_learnable_registers": 128,
            "connector_positional_embedding_max_pos": [4096], "connector_apply_gated_attention": true,
            "positional_embedding_theta": 10000.0, "rope_type": "split",
            "text_encoder_norm_type": "per_token_rms", "caption_proj_before_connector": true
        });
        let cfg = ConnectorsConfig::from_single_file(&t, 188_160).unwrap();
        assert_eq!(cfg.text_proj_in_factor, 49);
        assert_eq!((cfg.video_hidden_dim, cfg.audio_hidden_dim), (4096, 2048));
    }
}
