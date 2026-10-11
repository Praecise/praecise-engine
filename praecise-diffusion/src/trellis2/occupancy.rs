//! Occupancy decoder: a dense 3D convolutional decoder from the `8 x 16^3`
//! structure latent to `64^3` occupancy logits, upsampling twice by a
//! convolution to eight times the channels followed by a 3D pixel shuffle.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::safetensors::SafeTensors;
use crate::sparse3d::{conv3, kernel_from_dense, ConvIndex, Voxels};

const EPS: f32 = 1e-5;

/// Occupancy decoder configuration (`args` of the checkpoint's JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct OccupancyConfig {
    /// Output channels (one logit).
    pub out_channels: usize,
    /// Latent channels.
    pub latent_channels: usize,
    /// Residual blocks per resolution.
    pub num_res_blocks: usize,
    /// Residual blocks before the first upsampling.
    #[serde(default = "two")]
    pub num_res_blocks_middle: usize,
    /// Width per resolution.
    pub channels: Vec<usize>,
    /// Normalisation; only per-channel layer norm is implemented.
    #[serde(default = "layer")]
    pub norm_type: String,
}

fn two() -> usize {
    2
}
fn layer() -> String {
    "layer".into()
}

/// A loaded occupancy decoder.
pub struct OccupancyDecoder {
    /// Configuration.
    pub cfg: OccupancyConfig,
    w: Weights,
}

impl std::fmt::Debug for OccupancyDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccupancyDecoder").field("bytes", &self.w.bytes()).finish_non_exhaustive()
    }
}

impl OccupancyDecoder {
    /// Load from the checkpoint, re-laying out every convolution kernel for
    /// the gather convolution.
    ///
    /// # Errors
    /// Unsupported configuration, missing tensors, or an upload failure.
    pub fn load(backend: &Backend, cfg: OccupancyConfig, st: &SafeTensors) -> Result<Self> {
        if cfg.norm_type != "layer" || cfg.out_channels != 1 || cfg.channels.is_empty() {
            return Err(Error::Config("occupancy decoder: only a one-logit, layer-normalised decoder is implemented".into()));
        }
        let mut host = Vec::new();
        let mut conv = |name: &str, co: usize, ci: usize, host: &mut Vec<HostTensor>| -> Result<()> {
            let w = st.require(&format!("{name}.weight"), &[co as u64, ci as u64, 3, 3, 3])?.to_f32();
            let b = st.require(&format!("{name}.bias"), &[co as u64])?.to_f32();
            host.push(HostTensor { name: format!("{name}.weight"), shape: vec![co as u64, 27 * ci as u64], ty: WType::F16, data: kernel_from_dense(&w, co, ci) });
            host.push(HostTensor { name: format!("{name}.bias"), shape: vec![co as u64], ty: WType::F32, data: b });
            Ok(())
        };
        let norm = |name: &str, c: usize, host: &mut Vec<HostTensor>| -> Result<()> {
            for p in ["weight", "bias"] {
                let v = st.require(&format!("{name}.{p}"), &[c as u64])?.to_f32();
                host.push(HostTensor { name: format!("{name}.{p}"), shape: vec![c as u64], ty: WType::F32, data: v });
            }
            Ok(())
        };
        let ch = &cfg.channels;
        conv("input_layer", ch[0], cfg.latent_channels, &mut host)?;
        let res = |p: &str, c: usize, host: &mut Vec<HostTensor>| -> Result<()> {
            norm(&format!("{p}.norm1"), c, host)?;
            norm(&format!("{p}.norm2"), c, host)?;
            conv(&format!("{p}.conv1"), c, c, host)?;
            conv(&format!("{p}.conv2"), c, c, host)
        };
        for i in 0..cfg.num_res_blocks_middle {
            res(&format!("middle_block.{i}"), ch[0], &mut host)?;
        }
        let mut b = 0;
        for (i, &c) in ch.iter().enumerate() {
            for _ in 0..cfg.num_res_blocks {
                res(&format!("blocks.{b}"), c, &mut host)?;
                b += 1;
            }
            if i + 1 < ch.len() {
                conv(&format!("blocks.{b}.conv"), ch[i + 1] * 8, c, &mut host)?;
                b += 1;
            }
        }
        norm("out_layer.0", *ch.last().unwrap(), &mut host)?;
        conv("out_layer.2", 1, *ch.last().unwrap(), &mut host)?;
        Ok(Self { cfg, w: Weights::from_host(backend, &host)? })
    }

    /// Occupancy logits on the `(16 * 2^(stages - 1))^3` grid, `x` slowest,
    /// for a latent given as `[latent_channels, 16, 16, 16]` row-major.
    ///
    /// # Errors
    /// A mis-sized latent or backend failures.
    pub fn decode(&self, backend: &Backend, latent: &[f32], res: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let ch = &cfg.channels;
        let n0 = res * res * res;
        if latent.len() != cfg.latent_channels * n0 {
            return Err(Error::Request("structure latent size does not match its grid".into()));
        }
        // Channel-major volume to voxel-major features.
        let mut feats = vec![0f32; latent.len()];
        for c in 0..cfg.latent_channels {
            for v in 0..n0 {
                feats[v * cfg.latent_channels + c] = latent[c * n0 + v];
            }
        }
        let mut r = res;
        let mut b = 0;
        for (i, &c) in ch.iter().enumerate() {
            let vox = Voxels::dense(r as i32);
            let mut g = Graph::new(backend)?;
            let idx = ConvIndex::declare(&mut g, &vox);
            let in_c = if i == 0 { cfg.latent_channels } else { c };
            let x = g.input(llama_cpp_sys_2::GGML_TYPE_F32, &[in_c as i64, vox.len() as i64]);
            let mut h = x;
            if i == 0 {
                h = conv3(&mut g, h, idx, self.w.get("input_layer.weight"), Some(self.w.get("input_layer.bias")));
                for m in 0..cfg.num_res_blocks_middle {
                    h = self.res_block(&mut g, &format!("middle_block.{m}"), h, idx);
                }
            }
            for _ in 0..cfg.num_res_blocks {
                h = self.res_block(&mut g, &format!("blocks.{b}"), h, idx);
                b += 1;
            }
            let last = i + 1 == ch.len();
            let out = if last {
                let y = self.norm(&mut g, "out_layer.0", h);
                let y = g.silu(y);
                conv3(&mut g, y, idx, self.w.get("out_layer.2.weight"), Some(self.w.get("out_layer.2.bias")))
            } else {
                let y = conv3(&mut g, h, idx, self.w.get(&format!("blocks.{b}.conv.weight")), Some(self.w.get(&format!("blocks.{b}.conv.bias"))));
                b += 1;
                y
            };
            g.finish(&[out])?;
            idx.set(&g, &vox);
            g.set_f32(x, &feats);
            g.compute()?;
            let y = g.read_f32(out);
            if last {
                return Ok(y);
            }
            feats = pixel_shuffle(&y, ch[i + 1], r);
            r *= 2;
        }
        unreachable!("the last resolution returns")
    }

    fn norm(&self, g: &mut Graph, p: &str, x: Tn) -> Tn {
        let y = g.norm(x, EPS);
        let y = g.mul(y, self.w.get(&format!("{p}.weight")));
        g.add(y, self.w.get(&format!("{p}.bias")))
    }

    fn res_block(&self, g: &mut Graph, p: &str, x: Tn, idx: ConvIndex) -> Tn {
        let h = self.norm(g, &format!("{p}.norm1"), x);
        let h = g.silu(h);
        let h = conv3(g, h, idx, self.w.get(&format!("{p}.conv1.weight")), Some(self.w.get(&format!("{p}.conv1.bias"))));
        let h = self.norm(g, &format!("{p}.norm2"), h);
        let h = g.silu(h);
        let h = conv3(g, h, idx, self.w.get(&format!("{p}.conv2.weight")), Some(self.w.get(&format!("{p}.conv2.bias"))));
        g.add(h, x)
    }
}

/// 3D pixel shuffle of voxel-major features `[r^3, 8 * c]` (channel
/// `k * 8 + a * 4 + b * 2 + d` goes to child `(2x + a, 2y + b, 2z + d)`) into
/// `[(2r)^3, c]`.
#[must_use]
pub fn pixel_shuffle(y: &[f32], c: usize, r: usize) -> Vec<f32> {
    let r2 = 2 * r;
    let mut out = vec![0f32; y.len()];
    for x in 0..r {
        for yy in 0..r {
            for z in 0..r {
                let v = (x * r + yy) * r + z;
                for k in 0..c {
                    for s in 0..8 {
                        let (a, b, d) = (s >> 2, (s >> 1) & 1, s & 1);
                        let o = ((2 * x + a) * r2 + (2 * yy + b)) * r2 + (2 * z + d);
                        out[o * c + k] = y[v * 8 * c + k * 8 + s];
                    }
                }
            }
        }
    }
    out
}

/// Occupied cells of the `res / factor` grid: a cell is occupied when any of
/// its `factor^3` children has a positive logit (`x` slowest).
#[must_use]
pub fn occupied_cells(logits: &[f32], res: usize, factor: usize) -> Vec<[i32; 3]> {
    let r = res / factor;
    let mut out = Vec::new();
    for x in 0..r {
        for y in 0..r {
            for z in 0..r {
                let mut any = false;
                'scan: for a in 0..factor {
                    for b in 0..factor {
                        for d in 0..factor {
                            let i = ((x * factor + a) * res + (y * factor + b)) * res + (z * factor + d);
                            if logits[i] > 0.0 {
                                any = true;
                                break 'scan;
                            }
                        }
                    }
                }
                if any {
                    out.push([x as i32, y as i32, z as i32]);
                }
            }
        }
    }
    out
}
