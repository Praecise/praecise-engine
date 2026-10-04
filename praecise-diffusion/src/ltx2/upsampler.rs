//! LTX-2 latent spatial upsampler: doubles the width and height of
//! unnormalised video latents.
//!
//! Activations are laid out `[W, H, C, T]`. The network is a 3x3x3
//! convolution, group norm and SiLU, residual blocks (convolution, group
//! norm, SiLU, convolution, group norm, then SiLU of the sum with the
//! input), a per-frame 3x3 convolution to four times the channels folded
//! into 2x2 pixel blocks, more residual blocks and a final 3x3x3
//! convolution. Every convolution is zero padded on all three axes; group
//! norm statistics span the channels of a group over all frames and pixels.

use std::path::Path;

use serde::Deserialize;

use super::vae::depth_to_space;
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Channel groups of every group norm.
const GROUPS: u64 = 32;
/// Group norm epsilon (the reference default).
const NORM_EPS: f32 = 1e-5;

/// Shape of the upsampler, from the checkpoint's `config` metadata.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct UpsamplerConfig {
    /// Latent channels in and out.
    pub in_channels: u64,
    /// Working width.
    pub mid_channels: u64,
    /// Residual blocks before and after the upsampling step.
    pub num_blocks_per_stage: u64,
    #[serde(default = "three")]
    dims: u64,
    #[serde(default)]
    spatial_upsample: bool,
    #[serde(default)]
    temporal_upsample: bool,
    #[serde(default = "two")]
    spatial_scale: f64,
    #[serde(default)]
    rational_resampler: bool,
}

fn three() -> u64 {
    3
}

fn two() -> f64 {
    2.0
}

impl UpsamplerConfig {
    /// Parse and check the `config` metadata entry.
    ///
    /// # Errors
    /// When the entry is malformed or describes a variant other than the
    /// released 3D, spatial x2, plain pixel-shuffle upsampler.
    pub fn from_metadata(raw: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(raw).map_err(|e| Error::Config(format!("latent upsampler config: {e}")))?;
        if cfg.dims != 3 || !cfg.spatial_upsample || cfg.temporal_upsample || cfg.rational_resampler || cfg.spatial_scale != 2.0 {
            return Err(Error::Config("latent upsampler: only the 3D spatial x2 variant without a rational resampler is supported".into()));
        }
        if cfg.mid_channels % GROUPS != 0 || cfg.in_channels == 0 {
            return Err(Error::Config("latent upsampler: channel counts must suit 32 norm groups".into()));
        }
        Ok(cfg)
    }

    fn host_tensors(&self, st: &SafeTensors, kt: WType) -> Result<Vec<HostTensor>> {
        let (c, m) = (self.in_channels, self.mid_channels);
        let mut v = Vec::new();
        let vec = |v: &mut Vec<HostTensor>, name: String, n: u64| -> Result<()> {
            v.push(HostTensor { data: st.require(&name, &[n])?.to_f32(), name, shape: vec![n], ty: WType::F32 });
            Ok(())
        };
        let conv3 = |v: &mut Vec<HostTensor>, p: &str, cin: u64, cout: u64| -> Result<()> {
            let w = st.require(&format!("{p}.weight"), &[cout, cin, 3, 3, 3])?.to_f32();
            for k in 0..3 {
                let d: Vec<f32> = w.chunks_exact(27).flat_map(|c| (0..9).map(move |j| c[k * 9 + j])).collect();
                v.push(HostTensor { name: format!("{p}.t{k}"), shape: vec![cout, cin, 3, 3], ty: kt, data: d });
            }
            Ok(())
        };
        conv3(&mut v, "initial_conv", c, m)?;
        vec(&mut v, "initial_conv.bias".into(), m)?;
        vec(&mut v, "initial_norm.weight".into(), m)?;
        vec(&mut v, "initial_norm.bias".into(), m)?;
        for stage in ["res_blocks", "post_upsample_res_blocks"] {
            for b in 0..self.num_blocks_per_stage {
                for i in 1..=2 {
                    let p = format!("{stage}.{b}");
                    conv3(&mut v, &format!("{p}.conv{i}"), m, m)?;
                    vec(&mut v, format!("{p}.conv{i}.bias"), m)?;
                    vec(&mut v, format!("{p}.norm{i}.weight"), m)?;
                    vec(&mut v, format!("{p}.norm{i}.bias"), m)?;
                }
            }
        }
        let up = st.require("upsampler.0.weight", &[4 * m, m, 3, 3])?.to_f32();
        v.push(HostTensor { name: "upsampler.0.weight".into(), shape: vec![4 * m, m, 3, 3], ty: kt, data: up });
        vec(&mut v, "upsampler.0.bias".into(), 4 * m)?;
        conv3(&mut v, "final_conv", m, c)?;
        vec(&mut v, "final_conv.bias".into(), c)?;
        Ok(v)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
}

impl Net<'_, '_> {
    fn bias(&mut self, x: Tn, name: &str) -> Tn {
        let b = self.w.get(name);
        let b = self.g.reshape(b, &[1, 1, b.ne(0), 1]);
        self.g.add(x, b)
    }

    /// A 3x3x3 convolution, zero padded on every axis.
    fn conv3(&mut self, p: &str, x: Tn) -> Tn {
        let t = x.ne(3);
        let v = self.g.view_4d(x, [x.ne(0), x.ne(1), x.ne(2), 1], x.nb(1), x.nb(2), x.nb(3), 0);
        let first = self.g.cont(v);
        let zero = self.g.scale_bias(first, 0.0, 0.0);
        let a = self.g.concat(zero, x, 3);
        let padded = self.g.concat(a, zero, 3);
        let shape = [x.ne(0), x.ne(1), x.ne(2), t];
        let mut acc: Option<Tn> = None;
        for k in 0..3 {
            let v = self.g.view_4d(padded, shape, padded.nb(1), padded.nb(2), padded.nb(3), k as usize * padded.nb(3));
            let y = self.g.conv2d(self.w.get(&format!("{p}.t{k}")), v, 1);
            acc = Some(match acc {
                None => y,
                Some(a) => self.g.add(a, y),
            });
        }
        let acc = acc.expect("three taps");
        self.bias(acc, &format!("{p}.bias"))
    }

    /// Affine group norm with statistics over each group's channels, all
    /// frames and all pixels.
    fn norm(&mut self, p: &str, x: Tn) -> Tn {
        let (w, h, c, t) = (x.ne(0), x.ne(1), x.ne(2), x.ne(3));
        let a = self.g.permute(x, [0, 1, 3, 2]);
        let a = self.g.cont(a);
        let a = self.g.reshape(a, &[w * h, t, c, 1]);
        let a = self.g.group_norm(a, GROUPS as i32, NORM_EPS);
        let a = self.g.reshape(a, &[w, h, t, c]);
        let a = self.g.permute(a, [0, 1, 3, 2]);
        let a = self.g.cont(a);
        let s = self.w.get(&format!("{p}.weight"));
        let s = self.g.reshape(s, &[1, 1, c, 1]);
        let a = self.g.mul(a, s);
        self.bias(a, &format!("{p}.bias"))
    }

    fn block(&mut self, p: &str, x: Tn) -> Tn {
        let h = self.conv3(&format!("{p}.conv1"), x);
        let h = self.norm(&format!("{p}.norm1"), h);
        let h = self.g.silu(h);
        let h = self.conv3(&format!("{p}.conv2"), h);
        let h = self.norm(&format!("{p}.norm2"), h);
        let h = self.g.add(h, x);
        self.g.silu(h)
    }
}

fn build(g: &mut Graph, cfg: &UpsamplerConfig, w: &Weights, latent: Tn) -> Tn {
    let mut n = Net { g, w };
    let x = n.conv3("initial_conv", latent);
    let x = n.norm("initial_norm", x);
    let mut x = n.g.silu(x);
    for b in 0..cfg.num_blocks_per_stage {
        x = n.block(&format!("res_blocks.{b}"), x);
    }
    let y = n.g.conv2d(n.w.get("upsampler.0.weight"), x, 1);
    let y = n.bias(y, "upsampler.0.bias");
    let mut x = depth_to_space(n.g, y, 1, 2);
    for b in 0..cfg.num_blocks_per_stage {
        x = n.block(&format!("post_upsample_res_blocks.{b}"), x);
    }
    n.conv3("final_conv", x)
}

/// The latent upsampler with its resident weights.
pub struct Ltx2LatentUpsampler {
    backend: Backend,
    cfg: UpsamplerConfig,
    w: Weights,
}

impl std::fmt::Debug for Ltx2LatentUpsampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2LatentUpsampler").field("cfg", &self.cfg).finish_non_exhaustive()
    }
}

impl Ltx2LatentUpsampler {
    /// Device bytes held.
    pub(crate) fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Whether `st` holds an upsampler checkpoint.
    pub(crate) fn recognises(st: &SafeTensors) -> bool {
        st.get("upsampler.0.weight").is_some() && st.get("initial_conv.weight").is_some()
    }

    /// Load an upsampler checkpoint file.
    ///
    /// # Errors
    /// On an unsupported variant, missing weights or no usable backend.
    pub fn load(path: &Path, opts: LoadOptions) -> Result<Self> {
        let st = SafeTensors::open(&[path.to_path_buf()])?;
        Self::from_files(&st, opts)
    }

    pub(crate) fn from_files(st: &SafeTensors, opts: LoadOptions) -> Result<Self> {
        let raw = st.metadata("config").ok_or_else(|| Error::Config("latent upsampler checkpoint has no config metadata".into()))?;
        let cfg = UpsamplerConfig::from_metadata(raw)?;
        let backend = opts.backend()?;
        let kt = if opts.precision == Precision::F32 { WType::F32 } else { WType::F16 };
        let w = Weights::from_host(&backend, &cfg.host_tensors(st, kt)?)?;
        Ok(Self { backend, cfg, w })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &UpsamplerConfig {
        &self.cfg
    }

    /// Upsample unnormalised latents `[C][T][H][W]` to `[C][T][2H][2W]`.
    ///
    /// # Errors
    /// When the latent size disagrees with its shape, or on a backend
    /// failure.
    pub fn upsample(&self, latent: &[f32], frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        let c = self.cfg.in_channels as usize;
        let hw = height * width;
        if frames == 0 || hw == 0 || latent.len() != c * frames * hw {
            return Err(Error::Request("latent size disagrees with its shape".into()));
        }
        let mut zt = vec![0f32; latent.len()];
        for ch in 0..c {
            for t in 0..frames {
                zt[(t * c + ch) * hw..][..hw].copy_from_slice(&latent[(ch * frames + t) * hw..][..hw]);
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[width as i64, height as i64, c as i64, frames as i64]);
        let out = build(&mut g, &self.cfg, &self.w, input);
        g.finish(&[out])?;
        g.set_f32(input, &zt);
        g.compute()?;
        let y = g.read_f32(out);
        let ohw = 4 * hw;
        let mut d = vec![0f32; y.len()];
        for t in 0..frames {
            for ch in 0..c {
                d[(ch * frames + t) * ohw..][..ohw].copy_from_slice(&y[(t * c + ch) * ohw..][..ohw]);
            }
        }
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_released_variant_is_accepted() {
        let ok = r#"{"_class_name": "LatentUpsampler", "in_channels": 128, "mid_channels": 1024, "num_blocks_per_stage": 4, "dims": 3, "spatial_upsample": true, "temporal_upsample": false, "spatial_scale": 2.0, "rational_resampler": false}"#;
        assert_eq!(UpsamplerConfig::from_metadata(ok).unwrap().mid_channels, 1024);
        assert!(UpsamplerConfig::from_metadata(&ok.replace(r#""rational_resampler": false"#, r#""rational_resampler": true"#)).is_err());
        assert!(UpsamplerConfig::from_metadata(&ok.replace(r#""dims": 3"#, r#""dims": 2"#)).is_err());
    }
}
