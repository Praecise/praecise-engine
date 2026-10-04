//! Tensor names of the single-file LTX-2.3 release.
//!
//! The release ships one safetensors file holding every component under its
//! original module names, with the component configurations as JSON in the
//! header metadata. The native components load per-component names; this
//! maps each original name to its component and the name that component
//! loads.

use super::audio_vae::AudioVaeConfig;
use super::connectors::ConnectorsConfig;
use super::vae::VideoVaeConfig;
use super::Ltx2Config;
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
    /// The vocoder with its bandwidth extension.
    Vocoder,
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
    if let Some(rest) = key.strip_prefix("vocoder.") {
        return Some((Part::Vocoder, rest.to_string()));
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

impl Ltx2Config {
    /// The transformer configuration of a single-file checkpoint, from its
    /// header (the transformer section plus the compression factors of the
    /// two autoencoders) and the audio latent width (read off the audio input
    /// projection, which the header does not record).
    ///
    /// # Errors
    /// When a field is missing or the layout is not the released one.
    pub fn from_single_file(header: &serde_json::Value, audio_in_channels: u64) -> Result<Self> {
        let t = &header["transformer"];
        let missing = |k: &str| Error::Config(format!("single-file transformer config lacks {k}"));
        let u = |k: &str| t[k].as_u64().ok_or_else(|| missing(k));
        let f = |k: &str| t[k].as_f64().ok_or_else(|| missing(k));
        let b = |k: &str| t[k].as_bool().ok_or_else(|| missing(k));
        let s = |k: &str| t[k].as_str().ok_or_else(|| missing(k));
        let first = |k: &str, i: usize| t[k][i].as_u64().ok_or_else(|| missing(k));
        let released = [
            ("use_audio_video_cross_attention", b("use_audio_video_cross_attention")?),
            ("av_cross_ada_norm", b("av_cross_ada_norm")?),
            ("use_embeddings_connector", b("use_embeddings_connector")?),
            ("caption_proj_before_connector", b("caption_proj_before_connector")?),
            ("causal_temporal_positioning", b("causal_temporal_positioning")?),
            ("use_middle_indices_grid", b("use_middle_indices_grid")?),
            ("double_self_attention", !b("double_self_attention")?),
            ("only_cross_attention", !b("only_cross_attention")?),
            ("share_ff", !b("share_ff")?),
            ("upcast_attention", !b("upcast_attention")?),
            ("positional_embedding_type", s("positional_embedding_type")? == "rope"),
            ("frequencies_precision", s("frequencies_precision")? == "float64"),
            ("standardization_norm", s("standardization_norm")? == "rms_norm"),
            ("attention_type", s("attention_type")? == "default"),
        ];
        if let Some((k, _)) = released.iter().find(|(_, ok)| !ok) {
            return Err(Error::Config(format!("single-file transformer: {k} other than the released layout not implemented")));
        }
        let qk_norm = match s("qk_norm")? {
            "rms_norm" => "rms_norm_across_heads".to_string(),
            other => other.to_string(),
        };
        let vae = VideoVaeConfig::from_single_file(&header["vae"])?;
        let (spatial, temporal) = vae.factors();
        let audio = &header["audio_vae"];
        let avae = AudioVaeConfig::from_single_file(audio)?;
        let pre = &audio["preprocessing"];
        let sampling_rate = pre["audio"]["sampling_rate"].as_u64().ok_or_else(|| missing("audio_vae preprocessing sampling_rate"))?;
        let hop_length = pre["stft"]["hop_length"].as_u64().ok_or_else(|| missing("audio_vae preprocessing hop_length"))?;
        let gated = b("apply_gated_attention")?;
        let prompt_mod = b("cross_attention_adaln")?;
        let bias = b("attention_bias")?;
        let cfg = Self {
            in_channels: u("in_channels")?,
            out_channels: u("out_channels")?,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: u("num_attention_heads")?,
            attention_head_dim: u("attention_head_dim")?,
            cross_attention_dim: u("cross_attention_dim")?,
            vae_scale_factors: vec![temporal, spatial, spatial],
            pos_embed_max_pos: first("positional_embedding_max_pos", 0)?,
            base_height: first("positional_embedding_max_pos", 1)?,
            base_width: first("positional_embedding_max_pos", 2)?,
            gated_attn: gated,
            cross_attn_mod: prompt_mod,
            audio_in_channels,
            audio_out_channels: u("audio_out_channels")?,
            audio_patch_size: 1,
            audio_patch_size_t: 1,
            audio_num_attention_heads: u("audio_num_attention_heads")?,
            audio_attention_head_dim: u("audio_attention_head_dim")?,
            audio_cross_attention_dim: u("audio_cross_attention_dim")?,
            audio_scale_factor: 1 << (avae.ch_mult.len() - 1),
            audio_pos_embed_max_pos: first("audio_positional_embedding_max_pos", 0)?,
            audio_sampling_rate: sampling_rate,
            audio_hop_length: hop_length,
            audio_gated_attn: gated,
            audio_cross_attn_mod: prompt_mod,
            num_layers: usize::try_from(u("num_layers")?).map_err(|_| missing("num_layers"))?,
            activation_fn: s("activation_fn")?.to_string(),
            qk_norm,
            norm_elementwise_affine: b("norm_elementwise_affine")?,
            norm_eps: f("norm_eps")?,
            rope_theta: f("positional_embedding_theta")?,
            causal_offset: 1,
            timestep_scale_multiplier: u("timestep_scale_multiplier")?,
            cross_attn_timestep_scale_multiplier: t["av_ca_timestep_scale_multiplier"].as_f64().map(|v| v as u64).ok_or_else(|| missing("av_ca_timestep_scale_multiplier"))?,
            rope_type: s("rope_type")?.to_string(),
            use_prompt_embeddings: false,
            use_prompt_adaln_single: prompt_mod,
            attention_bias: bias,
            attention_out_bias: bias,
            ff_bias: true,
            audio_ff_bias: true,
            use_keyframes_abs_pos_embedding: false,
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

    /// The header sections of the released single file.
    pub(crate) fn released_header() -> serde_json::Value {
        serde_json::json!({
            "transformer": {"activation_fn": "gelu-approximate", "apply_gated_attention": true, "attention_bias": true,
                "attention_head_dim": 128, "attention_type": "default", "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048, "audio_num_attention_heads": 32, "audio_out_channels": 128,
                "audio_positional_embedding_max_pos": [20], "av_ca_timestep_scale_multiplier": 1000.0, "av_cross_ada_norm": true,
                "caption_proj_before_connector": true, "causal_temporal_positioning": true, "cross_attention_adaln": true,
                "cross_attention_dim": 4096, "double_self_attention": false, "frequencies_precision": "float64", "in_channels": 128,
                "norm_elementwise_affine": false, "norm_eps": 1e-06, "num_attention_heads": 32, "num_layers": 48,
                "only_cross_attention": false, "out_channels": 128, "positional_embedding_max_pos": [20, 2048, 2048],
                "positional_embedding_theta": 10000.0, "positional_embedding_type": "rope", "qk_norm": "rms_norm",
                "rope_type": "split", "share_ff": false, "standardization_norm": "rms_norm", "timestep_scale_multiplier": 1000,
                "upcast_attention": false, "use_audio_video_cross_attention": true, "use_embeddings_connector": true,
                "use_middle_indices_grid": true},
            "vae": {"dims": 3, "latent_channels": 128, "patch_size": 4, "decoder_base_channels": 128, "causal_decoder": false,
                "decoder_blocks": [["res_x", {"num_layers": 4}], ["compress_space", {"multiplier": 2}], ["res_x", {"num_layers": 6}],
                ["compress_time", {"multiplier": 2}], ["res_x", {"num_layers": 4}], ["compress_all", {"multiplier": 1}],
                ["res_x", {"num_layers": 2}], ["compress_all", {"multiplier": 2}], ["res_x", {"num_layers": 2}]]},
            "audio_vae": {"model": {"params": {"ddconfig": {"double_z": true, "mel_bins": 64, "z_channels": 8,
                "resolution": 256, "downsample_time": false, "in_channels": 2, "out_ch": 2, "ch": 128, "ch_mult": [1, 2, 4],
                "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0, "mid_block_add_attention": false,
                "norm_type": "pixel", "causality_axis": "height"}}},
                "preprocessing": {"audio": {"sampling_rate": 16000}, "stft": {"hop_length": 160}}}
        })
    }

    #[test]
    fn released_transformer_config() {
        let c = Ltx2Config::from_single_file(&released_header(), 128).unwrap();
        assert_eq!((c.inner(), c.audio_inner(), c.num_layers), (4096, 2048, 48));
        assert_eq!(c.vae_scale_factors, vec![8, 32, 32]);
        assert_eq!((c.audio_scale_factor, c.audio_sampling_rate, c.audio_hop_length), (4, 16000, 160));
        assert_eq!((c.pos_embed_max_pos, c.base_height, c.base_width, c.audio_pos_embed_max_pos), (20, 2048, 2048, 20));
        assert_eq!(c.cross_attn_timestep_scale_multiplier, 1000);
        let mut h = released_header();
        h["transformer"]["share_ff"] = true.into();
        assert!(Ltx2Config::from_single_file(&h, 128).is_err());
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
        assert_eq!(name("vocoder.vocoder.ups.0.weight"), (Part::Vocoder, "vocoder.ups.0.weight".into()));
        assert_eq!(component_name("unknown.tensor"), None);
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
