//! Structured-latent decoders: a sparse convolutional U-Net that refines the
//! latent voxels through ConvNeXt blocks and doubles their resolution four
//! times, each time splitting every voxel into the children a subdivision
//! mask keeps. The shape decoder predicts its own masks and ends in
//! per-voxel dual-grid vertices, edge-crossing flags and quad split weights;
//! the material decoder reuses the shape decoder's masks so both land on the
//! same voxels, and ends in base colour, metallic, roughness and alpha.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::sparse3d::{conv3, ConvIndex, Voxels};

const EPS: f32 = 1e-6;
const OUT_EPS: f32 = 1e-5;

/// Decoder configuration (`args` of the checkpoint's JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    /// Width per stage.
    pub model_channels: Vec<usize>,
    /// Latent channels.
    pub latent_channels: usize,
    /// ConvNeXt blocks per stage.
    pub num_blocks: Vec<usize>,
    /// Block kind per stage.
    pub block_type: Vec<String>,
    /// Upsampling block kind between stages.
    pub up_block_type: Vec<String>,
    /// Output channels (the material decoder names them; the shape decoder's
    /// are fixed at 7).
    #[serde(default)]
    pub out_channels: Option<usize>,
    /// Whether the decoder predicts its own subdivision masks.
    #[serde(default = "yes")]
    pub pred_subdiv: bool,
    /// Output grid side (shape decoder).
    #[serde(default)]
    pub resolution: Option<usize>,
}

fn yes() -> bool {
    true
}

impl DecoderConfig {
    /// Output channels per voxel.
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.out_channels.unwrap_or(7)
    }

    /// Refuse what this implementation does not run.
    ///
    /// # Errors
    /// [`Error::Config`] naming the unsupported setting.
    pub fn validate(&self) -> Result<()> {
        let s = self.model_channels.len();
        if s == 0 || self.num_blocks.len() != s || self.block_type.len() != s || self.up_block_type.len() + 1 != s {
            return Err(Error::Config("decoder stage lists disagree in length".into()));
        }
        if self.block_type.iter().any(|b| b != "SparseConvNeXtBlock3d") {
            return Err(Error::Config("decoder blocks other than SparseConvNeXtBlock3d are not implemented".into()));
        }
        if self.up_block_type.iter().any(|b| b != "SparseResBlockC2S3d") {
            return Err(Error::Config("decoder upsampling other than SparseResBlockC2S3d is not implemented".into()));
        }
        for w in self.model_channels.windows(2) {
            if w[0] % 8 != 0 || w[1] % (w[0] / 8) != 0 {
                return Err(Error::Config("decoder widths do not split into 8 children".into()));
            }
        }
        Ok(())
    }

    /// Every weight, in the checkpoint's names; convolution kernels are read
    /// as `[C_out, 27 * C_in]`.
    #[must_use]
    pub fn weight_specs(&self) -> Vec<WeightSpec> {
        let mut v = vec![
            WeightSpec::new("from_latent.weight", &[self.model_channels[0] as u64, self.latent_channels as u64], WType::F32),
            WeightSpec::new("from_latent.bias", &[self.model_channels[0] as u64], WType::F32),
            WeightSpec::new("output_layer.weight", &[self.outputs() as u64, *self.model_channels.last().unwrap() as u64], WType::F32),
            WeightSpec::new("output_layer.bias", &[self.outputs() as u64], WType::F32),
        ];
        for (s, &c) in self.model_channels.iter().enumerate() {
            let c = c as u64;
            for j in 0..self.num_blocks[s] {
                let p = format!("blocks.{s}.{j}");
                v.push(WeightSpec::new(format!("{p}.conv.weight"), &[c, 27 * c], WType::F16));
                v.push(WeightSpec::new(format!("{p}.conv.bias"), &[c], WType::F32));
                v.push(WeightSpec::new(format!("{p}.norm.weight"), &[c], WType::F32));
                v.push(WeightSpec::new(format!("{p}.norm.bias"), &[c], WType::F32));
                v.push(WeightSpec::new(format!("{p}.mlp.0.weight"), &[4 * c, c], WType::F16));
                v.push(WeightSpec::new(format!("{p}.mlp.0.bias"), &[4 * c], WType::F32));
                v.push(WeightSpec::new(format!("{p}.mlp.2.weight"), &[c, 4 * c], WType::F16));
                v.push(WeightSpec::new(format!("{p}.mlp.2.bias"), &[c], WType::F32));
            }
            if s + 1 < self.model_channels.len() {
                let o = self.model_channels[s + 1] as u64;
                let p = format!("blocks.{s}.{}", self.num_blocks[s]);
                v.push(WeightSpec::new(format!("{p}.norm1.weight"), &[c], WType::F32));
                v.push(WeightSpec::new(format!("{p}.norm1.bias"), &[c], WType::F32));
                v.push(WeightSpec::new(format!("{p}.conv1.weight"), &[8 * o, 27 * c], WType::F16));
                v.push(WeightSpec::new(format!("{p}.conv1.bias"), &[8 * o], WType::F32));
                v.push(WeightSpec::new(format!("{p}.conv2.weight"), &[o, 27 * o], WType::F16));
                v.push(WeightSpec::new(format!("{p}.conv2.bias"), &[o], WType::F32));
                if self.pred_subdiv {
                    v.push(WeightSpec::new(format!("{p}.to_subdiv.weight"), &[8, c], WType::F32));
                    v.push(WeightSpec::new(format!("{p}.to_subdiv.bias"), &[8], WType::F32));
                }
            }
        }
        v
    }

    /// Names of the five-dimensional convolution kernels and the
    /// `[C_out, 27 * C_in]` shape they are read as.
    #[must_use]
    pub fn kernel_shapes(&self) -> Vec<(String, Vec<u64>)> {
        self.weight_specs()
            .into_iter()
            .filter(|s| s.name.ends_with("conv.weight") || s.name.ends_with("conv1.weight") || s.name.ends_with("conv2.weight"))
            .map(|s| (s.name, s.shape))
            .collect()
    }
}

/// Decoded voxels.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// Output voxels on the finest grid.
    pub voxels: Voxels,
    /// Per-voxel outputs, `outputs` values per voxel.
    pub feats: Vec<f32>,
    /// Subdivision masks applied at each upsampling (8 flags per voxel).
    pub subdivisions: Vec<Vec<bool>>,
}

/// A loaded decoder.
pub struct SparseDecoder {
    /// Configuration.
    pub cfg: DecoderConfig,
    w: Weights,
}

impl std::fmt::Debug for SparseDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseDecoder").field("bytes", &self.w.bytes()).finish_non_exhaustive()
    }
}

/// The state carried from one stage graph to the next: the parent voxels'
/// pre-split features and skip input, and which children survive.
struct Carry {
    parent: Voxels,
    h8: Vec<f32>,
    skip: Vec<f32>,
    children: Voxels,
    rows: Vec<i32>,
}

impl SparseDecoder {
    /// Wrap loaded weights.
    #[must_use]
    pub fn new(cfg: DecoderConfig, w: Weights) -> Self {
        Self { cfg, w }
    }

    /// Device bytes of the weights.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Decode `latent` (`[voxels, latent_channels]` row-major) at `voxels`.
    /// A decoder that predicts no masks takes `guide` (one mask list per
    /// upsampling, as a predicting decoder returns them).
    ///
    /// # Errors
    /// Missing or mis-sized masks, or backend failures.
    pub fn decode(&self, backend: &Backend, voxels: &Voxels, latent: &[f32], guide: Option<&[Vec<bool>]>) -> Result<Decoded> {
        let cfg = &self.cfg;
        if latent.len() != voxels.len() * cfg.latent_channels {
            return Err(Error::Request("latent size does not match its voxels".into()));
        }
        if !cfg.pred_subdiv && guide.is_none_or(|g| g.len() + 1 != cfg.model_channels.len()) {
            return Err(Error::Request("this decoder needs one subdivision mask per upsampling".into()));
        }
        let stages = cfg.model_channels.len();
        let mut carry: Option<Carry> = None;
        let mut subdivisions = Vec::new();
        for s in 0..stages {
            let here = carry.as_ref().map_or_else(|| voxels.clone(), |c| c.children.clone());
            let last = s + 1 == stages;
            let mut g = Graph::new(backend)?;
            let f32t = llama_cpp_sys_2::GGML_TYPE_F32;
            let idx = ConvIndex::declare(&mut g, &here);
            let c = cfg.model_channels[s] as i64;
            let n = here.len() as i64;
            // Stage input.
            let (h, inputs) = match &carry {
                None => {
                    let x = g.input(f32t, &[cfg.latent_channels as i64, n]);
                    let h = g.linear_b(self.w.get("from_latent.weight"), self.w.get("from_latent.bias"), x);
                    (h, StageInputs::Latent(x))
                }
                Some(cr) => {
                    let pc = cfg.model_channels[s - 1] as i64;
                    let np = cr.parent.len() as i64;
                    let h8 = g.input(f32t, &[8 * c, np]);
                    let skip = g.input(f32t, &[pc, np]);
                    let rows = g.input(llama_cpp_sys_2::GGML_TYPE_I32, &[n]);
                    let h = self.split_finish(&mut g, s - 1, h8, skip, rows, idx, c, pc, np, n);
                    (h, StageInputs::Split { h8, skip, rows })
                }
            };
            let mut h = h;
            for j in 0..cfg.num_blocks[s] {
                h = self.convnext(&mut g, &format!("blocks.{s}.{j}"), h, idx);
            }
            let outs: Vec<Tn>;
            let (o_h8, o_skip, o_sub, o_out);
            if last {
                let y = g.norm(h, OUT_EPS);
                let y = g.linear_b(self.w.get("output_layer.weight"), self.w.get("output_layer.bias"), y);
                o_out = Some(y);
                (o_h8, o_skip, o_sub) = (None, None, None);
                outs = vec![y];
            } else {
                let p = format!("blocks.{s}.{}", cfg.num_blocks[s]);
                let sub = cfg.pred_subdiv.then(|| g.linear_b(self.w.get(&format!("{p}.to_subdiv.weight")), self.w.get(&format!("{p}.to_subdiv.bias")), h));
                let y = g.norm(h, EPS);
                let y = g.mul(y, self.w.get(&format!("{p}.norm1.weight")));
                let y = g.add(y, self.w.get(&format!("{p}.norm1.bias")));
                let y = g.silu(y);
                let y = conv3(&mut g, y, idx, self.w.get(&format!("{p}.conv1.weight")), Some(self.w.get(&format!("{p}.conv1.bias"))));
                o_h8 = Some(y);
                o_skip = Some(h);
                o_sub = sub;
                o_out = None;
                outs = [Some(y), Some(h), sub].into_iter().flatten().collect();
            }
            g.finish(&outs)?;
            idx.set(&g, &here);
            match (&inputs, &carry) {
                (StageInputs::Latent(x), _) => g.set_f32(*x, latent),
                (StageInputs::Split { h8, skip, rows }, Some(cr)) => {
                    g.set_f32(*h8, &cr.h8);
                    g.set_f32(*skip, &cr.skip);
                    g.set_i32(*rows, &cr.rows);
                }
                _ => unreachable!("a split stage always has a carry"),
            }
            g.compute()?;
            if let Some(y) = o_out {
                return Ok(Decoded { voxels: here, feats: g.read_f32(y), subdivisions });
            }
            let mask: Vec<bool> = match o_sub {
                Some(sub) => g.read_f32(sub).into_iter().map(|v| v > 0.0).collect(),
                None => {
                    let m = guide.expect("checked above")[s].clone();
                    if m.len() != here.len() * 8 {
                        return Err(Error::Request(format!("subdivision mask {s} holds {} flags for {} voxels", m.len(), here.len())));
                    }
                    m
                }
            };
            let (children, rows) = here.subdivide(&mask);
            if children.is_empty() {
                return Err(Error::Request(format!("decoder stage {s} kept no voxels")));
            }
            subdivisions.push(mask);
            carry = Some(Carry {
                parent: here,
                h8: g.read_f32(o_h8.expect("set above")),
                skip: g.read_f32(o_skip.expect("set above")),
                children,
                rows,
            });
            let _ = c;
        }
        unreachable!("the last stage returns")
    }

    /// `conv -> norm -> MLP -> + x`.
    fn convnext(&self, g: &mut Graph, p: &str, x: Tn, idx: ConvIndex) -> Tn {
        let w = |s: &str| self.w.get(&format!("{p}.{s}"));
        let h = conv3(g, x, idx, w("conv.weight"), Some(w("conv.bias")));
        let h = g.norm(h, EPS);
        let h = g.mul(h, w("norm.weight"));
        let h = g.add(h, w("norm.bias"));
        let h = g.linear_b(w("mlp.0.weight"), w("mlp.0.bias"), h);
        let h = g.silu(h);
        let h = g.linear_b(w("mlp.2.weight"), w("mlp.2.bias"), h);
        g.add(h, x)
    }

    /// Second half of an upsampling block: move each kept child's channels
    /// out of its parent's `8 * C` columns, then `norm -> silu -> conv`, plus
    /// the parent's input repeated across the wider channels.
    #[allow(clippy::too_many_arguments)]
    fn split_finish(&self, g: &mut Graph, stage: usize, h8: Tn, skip: Tn, rows: Tn, idx: ConvIndex, c: i64, pc: i64, np: i64, n: i64) -> Tn {
        let p = format!("blocks.{stage}.{}", self.cfg.num_blocks[stage]);
        let table = g.reshape(h8, &[c, 8 * np]);
        let h = g.get_rows(table, rows);
        let h = g.norm(h, EPS);
        let h = g.silu(h);
        let h = conv3(g, h, idx, self.w.get(&format!("{p}.conv2.weight")), Some(self.w.get(&format!("{p}.conv2.bias"))));
        let sc = pc / 8;
        let st = g.reshape(skip, &[sc, 8 * np]);
        let sx = g.get_rows(st, rows);
        let r = c / sc;
        let sx = g.reshape(sx, &[1, sc, n]);
        let sx = g.repeat_to(sx, [r, sc, n, 1]);
        let sx = g.reshape(sx, &[c, n]);
        g.add(h, sx)
    }
}

enum StageInputs {
    Latent(Tn),
    Split { h8: Tn, skip: Tn, rows: Tn },
}
