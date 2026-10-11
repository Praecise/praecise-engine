//! SigLIP vision transformer: square patches embedded by a strided
//! convolution, a learned position per patch, pre-norm encoder layers
//! (LayerNorm, biased multi-head attention, tanh-GELU MLP) and a final
//! LayerNorm. Every patch attends to every other.

use serde::Deserialize;

use crate::flux3::packing::Frame;
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use llama_cpp_sys_2 as sys;

/// Vision-tower configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct SiglipConfig {
    /// Width.
    pub hidden_size: u64,
    /// Encoder layers.
    #[serde(default = "default_layers")]
    pub num_hidden_layers: usize,
    /// Attention heads.
    pub num_attention_heads: u64,
    /// MLP width.
    #[serde(default = "default_intermediate")]
    pub intermediate_size: u64,
    /// Patch side in pixels.
    pub patch_size: usize,
    /// Image side in pixels.
    pub image_size: usize,
    /// LayerNorm epsilon.
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
}

fn default_layers() -> usize {
    12
}
fn default_intermediate() -> u64 {
    3072
}
fn default_eps() -> f64 {
    1e-6
}

impl SiglipConfig {
    /// Patches per image side.
    #[must_use]
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Patches per image.
    #[must_use]
    pub fn patches(&self) -> usize {
        self.grid() * self.grid()
    }

    /// Values per patch.
    #[must_use]
    pub fn patch_len(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// Every weight of the tower under `prefix` (ending in `.`).
    #[must_use]
    pub fn weight_specs(&self, prefix: &str, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size;
        let ff = self.intermediate_size;
        let p = self.patch_size as u64;
        let f = WType::F32;
        // Kernel rows are p * p values, not a multiple of the 8-bit block.
        let kernel = if linear == WType::Q8_0 { WType::Bf16 } else { linear };
        let mut v = vec![
            WeightSpec::new(format!("{prefix}embeddings.patch_embedding.weight"), &[d, 3, p, p], kernel),
            WeightSpec::new(format!("{prefix}embeddings.patch_embedding.bias"), &[d], f),
            WeightSpec::new(format!("{prefix}embeddings.position_embedding.weight"), &[self.patches() as u64, d], f),
            WeightSpec::new(format!("{prefix}post_layernorm.weight"), &[d], f),
            WeightSpec::new(format!("{prefix}post_layernorm.bias"), &[d], f),
        ];
        for i in 0..self.num_hidden_layers {
            let l = format!("{prefix}encoder.layers.{i}");
            for n in ["layer_norm1", "layer_norm2"] {
                v.push(WeightSpec::new(format!("{l}.{n}.weight"), &[d], f));
                v.push(WeightSpec::new(format!("{l}.{n}.bias"), &[d], f));
            }
            for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                v.push(WeightSpec::new(format!("{l}.self_attn.{n}.weight"), &[d, d], linear));
                v.push(WeightSpec::new(format!("{l}.self_attn.{n}.bias"), &[d], f));
            }
            v.push(WeightSpec::new(format!("{l}.mlp.fc1.weight"), &[ff, d], linear));
            v.push(WeightSpec::new(format!("{l}.mlp.fc1.bias"), &[ff], f));
            v.push(WeightSpec::new(format!("{l}.mlp.fc2.weight"), &[d, ff], linear));
            v.push(WeightSpec::new(format!("{l}.mlp.fc2.bias"), &[d], f));
        }
        v
    }

    /// The patches `[patches][3 * p * p]` of a frame already at
    /// `image_size` (values as the tower reads them), each patch channel
    /// major then row then column, as the convolution kernel is laid out.
    #[must_use]
    pub fn patchify(&self, f: &Frame) -> Vec<f32> {
        let (p, g, s) = (self.patch_size, self.grid(), self.image_size);
        let mut out = vec![0f32; self.patches() * self.patch_len()];
        for py in 0..g {
            for px in 0..g {
                let base = (py * g + px) * self.patch_len();
                for c in 0..3 {
                    for y in 0..p {
                        let src = (c * s + py * p + y) * s + px * p;
                        let dst = base + (c * p + y) * p;
                        out[dst..dst + p].copy_from_slice(&f.data[src..src + p]);
                    }
                }
            }
        }
        out
    }
}

fn layer_norm(g: &mut Graph, w: &Weights, name: &str, x: Tn, eps: f32) -> Tn {
    let h = g.norm(x, eps);
    let h = g.mul(h, w.get(&format!("{name}.weight")));
    g.add(h, w.get(&format!("{name}.bias")))
}

/// Encode one image: `patches` is a `[patch_len, patches]` input; the result
/// is `[hidden_size, patches]` after the final LayerNorm. `exact` computes
/// attention and the GELU entirely in float32.
#[must_use]
pub fn build(g: &mut Graph, cfg: &SiglipConfig, w: &Weights, prefix: &str, patches: Tn, exact: bool) -> Tn {
    let d = cfg.hidden_size as i64;
    let nh = cfg.num_attention_heads as i64;
    let hd = d / nh;
    let n = cfg.patches() as i64;
    let eps = cfg.layer_norm_eps as f32;
    let wn = |s: &str| w.get(&format!("{prefix}{s}"));
    let kernel = g.reshape(wn("embeddings.patch_embedding.weight"), &[cfg.patch_len() as i64, d]);
    let x = g.linear(kernel, patches);
    let x = g.add(x, wn("embeddings.patch_embedding.bias"));
    let mut x = g.add(x, wn("embeddings.position_embedding.weight"));
    for i in 0..cfg.num_hidden_layers {
        let l = format!("{prefix}encoder.layers.{i}");
        let lw = |s: &str| w.get(&format!("{l}.{s}"));
        let h = layer_norm(g, w, &format!("{l}.layer_norm1"), x, eps);
        let q = g.linear_b(lw("self_attn.q_proj.weight"), lw("self_attn.q_proj.bias"), h);
        let k = g.linear_b(lw("self_attn.k_proj.weight"), lw("self_attn.k_proj.bias"), h);
        let v = g.linear_b(lw("self_attn.v_proj.weight"), lw("self_attn.v_proj.bias"), h);
        let q = g.reshape(q, &[hd, nh, n]);
        let k = g.reshape(k, &[hd, nh, n]);
        let v = g.reshape(v, &[hd, nh, n]);
        let q = g.permute(q, [0, 2, 1, 3]);
        let k = g.permute(k, [0, 2, 1, 3]);
        let v = g.permute(v, [0, 2, 1, 3]);
        let scale = 1.0 / (hd as f32).sqrt();
        let o = if exact {
            g.attention_exact(q, k, v, None, scale)
        } else {
            let k = g.cast(k, sys::GGML_TYPE_F16);
            let v = g.cast(v, sys::GGML_TYPE_F16);
            g.attention(q, k, v, None, scale, true)
        };
        let o = g.reshape(o, &[d, n]);
        let o = g.linear_b(lw("self_attn.out_proj.weight"), lw("self_attn.out_proj.bias"), o);
        x = g.add(x, o);
        let h = layer_norm(g, w, &format!("{l}.layer_norm2"), x, eps);
        let h = g.linear_b(lw("mlp.fc1.weight"), lw("mlp.fc1.bias"), h);
        let h = if exact { g.gelu_tanh_exact(h) } else { g.gelu_tanh(h) };
        let h = g.linear_b(lw("mlp.fc2.weight"), lw("mlp.fc2.bias"), h);
        x = g.add(x, h);
    }
    layer_norm(g, w, &format!("{prefix}post_layernorm"), x, eps)
}
