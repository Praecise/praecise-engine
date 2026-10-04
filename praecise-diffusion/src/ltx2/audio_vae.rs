//! LTX-2 audio autoencoder: the decoder, latents to a mel spectrogram.
//!
//! The latent is a small image, time by mel bins, with a few channels; the
//! decoder is a 2D residual network over it (pixel norm, SiLU, convolution,
//! twice, with a 1x1 shortcut on a width change) with nearest-neighbour 2x
//! upsampling between levels. Every convolution is causal in time: two zero
//! rows of padding before the first frame and none after, so an output row
//! never sees a later frame; the mel axis is zero padded on both sides. Each
//! upsampling drops its first output row, so `T` latent frames decode to
//! `4 (T - 1) + 1` spectrogram frames for two upsamplings.
//!
//! Activations are laid out `[mel, time, C]`.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::single_file::{header_config, open_part, Part};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Pixel norm epsilon (fixed in the reference).
const NORM_EPS: f32 = 1e-6;

/// Decoder configuration, from the `audio_vae` section of a single-file
/// header.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeConfig {
    /// Base width.
    pub ch: u64,
    /// Width multiples per level.
    pub ch_mult: Vec<u64>,
    /// Residual blocks per level (the decoder runs one more).
    pub num_res_blocks: usize,
    /// Latent channels.
    pub z_channels: u64,
    /// Spectrogram channels (2 for stereo).
    pub out_ch: u64,
    /// Mel bins of the spectrogram.
    pub mel_bins: u64,
    #[serde(default)]
    norm_type: String,
    #[serde(default)]
    causality_axis: String,
    #[serde(default)]
    attn_resolutions: Vec<u64>,
    #[serde(default)]
    mid_block_add_attention: bool,
}

impl AudioVaeConfig {
    /// Parse the `audio_vae` section of a single-file header.
    ///
    /// # Errors
    /// A layout the native decoder does not implement.
    pub fn from_single_file(section: &Value) -> Result<Self> {
        let dd = &section["model"]["params"]["ddconfig"];
        let c: Self = serde_json::from_value(dd.clone()).map_err(|e| Error::Config(format!("audio autoencoder config: {e}")))?;
        let bad = |w: &str| Err(Error::Config(format!("audio autoencoder: {w} is not supported")));
        if c.norm_type != "pixel" {
            return bad(&format!("norm {}", c.norm_type));
        }
        if c.causality_axis != "height" {
            return bad(&format!("causality axis {}", c.causality_axis));
        }
        if !c.attn_resolutions.is_empty() || c.mid_block_add_attention {
            return bad("attention");
        }
        if c.ch_mult.is_empty() {
            return bad("an empty level list");
        }
        let down = 1u64 << (c.ch_mult.len() - 1);
        if !c.mel_bins.is_multiple_of(down) {
            return bad("a mel bin count the levels do not divide");
        }
        Ok(c)
    }

    /// Mel bins of the latent.
    #[must_use]
    pub fn latent_mel_bins(&self) -> u64 {
        self.mel_bins >> (self.ch_mult.len() - 1)
    }

    /// Width of one packed latent frame (channels times latent mel bins).
    #[must_use]
    pub fn packed_width(&self) -> u64 {
        self.z_channels * self.latent_mel_bins()
    }

    /// Spectrogram frames decoded from `frames` latent frames.
    #[must_use]
    pub fn spectrogram_frames(&self, frames: usize) -> usize {
        let up = 1usize << (self.ch_mult.len() - 1);
        (frames * up).saturating_sub(up - 1).max(1)
    }

    /// `(level, block, cin, cout)` of the decoder's residual blocks, in run
    /// order, and the level width after each level.
    fn levels(&self) -> Vec<(usize, Vec<(u64, u64)>)> {
        let mut cin = self.ch * self.ch_mult.last().copied().unwrap_or(1);
        let mut out = Vec::new();
        for level in (0..self.ch_mult.len()).rev() {
            let cout = self.ch * self.ch_mult[level];
            let mut blocks = Vec::new();
            for _ in 0..=self.num_res_blocks {
                blocks.push((cin, cout));
                cin = cout;
            }
            out.push((level, blocks));
        }
        out
    }

    fn host_tensors(&self, st: &SafeTensors, kt: WType) -> Result<Vec<HostTensor>> {
        let mut v = Vec::new();
        let mut conv = |p: &str, cin: u64, cout: u64, k: u64| -> Result<()> {
            let w = st.require(&format!("{p}.weight"), &[cout, cin, k, k])?.to_f32();
            v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, k, k], ty: kt, data: w });
            v.push(HostTensor { name: format!("{p}.bias"), shape: vec![cout], ty: WType::F32, data: st.require(&format!("{p}.bias"), &[cout])?.to_f32() });
            Ok(())
        };
        let top = self.ch * self.ch_mult.last().copied().unwrap_or(1);
        conv("decoder.conv_in.conv", self.z_channels, top, 3)?;
        for b in ["block_1", "block_2"] {
            conv(&format!("decoder.mid.{b}.conv1.conv"), top, top, 3)?;
            conv(&format!("decoder.mid.{b}.conv2.conv"), top, top, 3)?;
        }
        let mut last = top;
        for (level, blocks) in self.levels() {
            for (i, (cin, cout)) in blocks.into_iter().enumerate() {
                let p = format!("decoder.up.{level}.block.{i}");
                conv(&format!("{p}.conv1.conv"), cin, cout, 3)?;
                conv(&format!("{p}.conv2.conv"), cout, cout, 3)?;
                if cin != cout {
                    conv(&format!("{p}.nin_shortcut.conv"), cin, cout, 1)?;
                }
                last = cout;
            }
            if level != 0 {
                conv(&format!("decoder.up.{level}.upsample.conv.conv"), last, last, 3)?;
            }
        }
        conv("decoder.conv_out.conv", last, self.out_ch, 3)?;
        Ok(v)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
}

impl Net<'_, '_> {
    fn bias(&mut self, p: &str, y: Tn) -> Tn {
        let b = self.w.get(&format!("{p}.bias"));
        let b = self.g.reshape(b, &[1, 1, b.ne(0), 1]);
        self.g.add(y, b)
    }

    /// A 3x3 convolution causal in time: a zero row is rolled to the front,
    /// so the symmetric padding of one row becomes two before and none after.
    fn causal(&mut self, p: &str, x: Tn) -> Tn {
        let t = x.ne(1);
        let padded = self.g.pad_end(x, 0, 1);
        let shifted = self.g.roll(padded, 0, 1);
        let y = self.g.conv2d(self.w.get(&format!("{p}.weight")), shifted, 1);
        let v = self.g.view_4d(y, [y.ne(0), t, y.ne(2), 1], y.nb(1), y.nb(2), y.nb(3), 0);
        let v = self.g.cont(v);
        self.bias(p, v)
    }

    fn pointwise(&mut self, p: &str, x: Tn) -> Tn {
        let y = self.g.conv2d(self.w.get(&format!("{p}.weight")), x, 0);
        self.bias(p, y)
    }

    fn norm_silu(&mut self, x: Tn) -> Tn {
        let c = self.g.permute(x, [1, 2, 0, 3]);
        let c = self.g.cont(c);
        let n = self.g.rms_norm(c, NORM_EPS);
        let n = self.g.silu(n);
        let back = self.g.permute(n, [2, 0, 1, 3]);
        self.g.cont(back)
    }

    fn resnet(&mut self, p: &str, x: Tn, cin: u64, cout: u64) -> Tn {
        let h = self.norm_silu(x);
        let h = self.causal(&format!("{p}.conv1.conv"), h);
        let h = self.norm_silu(h);
        let h = self.causal(&format!("{p}.conv2.conv"), h);
        let skip = if cin == cout { x } else { self.pointwise(&format!("{p}.nin_shortcut.conv"), x) };
        self.g.add(h, skip)
    }

    fn upsample(&mut self, p: &str, x: Tn) -> Tn {
        let up = self.g.upscale_nearest(x, 2);
        let y = self.causal(p, up);
        let v = self.g.view_4d(y, [y.ne(0), y.ne(1) - 1, y.ne(2), 1], y.nb(1), y.nb(2), y.nb(3), y.nb(1));
        self.g.cont(v)
    }
}

fn build(g: &mut Graph, cfg: &AudioVaeConfig, w: &Weights, latent: Tn) -> Tn {
    let mut n = Net { g, w };
    let top = cfg.ch * cfg.ch_mult.last().copied().unwrap_or(1);
    let mut x = n.causal("decoder.conv_in.conv", latent);
    x = n.resnet("decoder.mid.block_1", x, top, top);
    x = n.resnet("decoder.mid.block_2", x, top, top);
    for (level, blocks) in cfg.levels() {
        for (i, (cin, cout)) in blocks.into_iter().enumerate() {
            x = n.resnet(&format!("decoder.up.{level}.block.{i}"), x, cin, cout);
        }
        if level != 0 {
            x = n.upsample(&format!("decoder.up.{level}.upsample.conv.conv"), x);
        }
    }
    let x = n.norm_silu(x);
    n.causal("decoder.conv_out.conv", x)
}

/// The audio decoder with its resident weights.
pub struct Ltx2AudioDecoder {
    backend: Backend,
    cfg: AudioVaeConfig,
    w: Weights,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl std::fmt::Debug for Ltx2AudioDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2AudioDecoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl Ltx2AudioDecoder {
    /// Load the audio decoder of a single-file checkpoint.
    ///
    /// # Errors
    /// On an unsupported layout, missing weights or no usable backend.
    pub fn load_single_file(path: &Path, opts: LoadOptions) -> Result<Self> {
        let st = SafeTensors::open(&[path.to_path_buf()])?;
        let header = header_config(&st)?;
        let cfg = AudioVaeConfig::from_single_file(&header["audio_vae"])?;
        let st = open_part(st, Part::AudioVae)?;
        let backend = opts.backend()?;
        let kt = if opts.precision == Precision::F32 { WType::F32 } else { WType::F16 };
        let w = Weights::from_host(&backend, &cfg.host_tensors(&st, kt)?)?;
        let p = cfg.packed_width();
        let mean = st.require("latents_mean", &[p])?.to_f32();
        let std = st.require("latents_std", &[p])?.to_f32();
        Ok(Self { backend, cfg, w, mean, std })
    }

    /// The decoder configuration.
    #[must_use]
    pub fn config(&self) -> &AudioVaeConfig {
        &self.cfg
    }

    /// Decode normalised packed latents `[T][C * mel]` (as the transformer
    /// produces them) to a mel spectrogram `[channels][T'][mel bins]`.
    ///
    /// # Errors
    /// A latent size that disagrees with the frame count, or a backend
    /// failure.
    pub fn decode(&self, packed: &[f32], frames: usize) -> Result<Vec<f32>> {
        let (c, m) = (self.cfg.z_channels as usize, self.cfg.latent_mel_bins() as usize);
        if frames == 0 || packed.len() != frames * c * m {
            return Err(Error::Request("audio latent size disagrees with its frame count".into()));
        }
        // Denormalise per packed slot, then unpack to the graph's [C][T][mel].
        let mut z = vec![0f32; packed.len()];
        for t in 0..frames {
            for ch in 0..c {
                for b in 0..m {
                    let j = ch * m + b;
                    z[(ch * frames + t) * m + b] = packed[t * c * m + j] * self.std[j] + self.mean[j];
                }
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[m as i64, frames as i64, c as i64, 1]);
        let out = build(&mut g, &self.cfg, &self.w, input);
        g.finish(&[out])?;
        g.set_f32(input, &z);
        g.compute()?;
        Ok(g.read_f32(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_layout() {
        let v = serde_json::json!({"model": {"params": {"ddconfig": {"double_z": true, "mel_bins": 64, "z_channels": 8,
            "resolution": 256, "downsample_time": false, "in_channels": 2, "out_ch": 2, "ch": 128, "ch_mult": [1, 2, 4],
            "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0, "mid_block_add_attention": false,
            "norm_type": "pixel", "causality_axis": "height"}}}});
        let c = AudioVaeConfig::from_single_file(&v).unwrap();
        assert_eq!((c.latent_mel_bins(), c.packed_width()), (16, 128));
        assert_eq!(c.spectrogram_frames(126), 501);
        let levels = c.levels();
        assert_eq!(levels[0].0, 2);
        assert_eq!(levels[0].1[0], (512, 512));
        assert_eq!(levels[1].1[0], (512, 256));
        assert_eq!(levels[2].1[0], (256, 128));
    }
}
