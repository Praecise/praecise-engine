//! DINOv3 vision transformer used as an image feature encoder: patch
//! embedding, a class token and register tokens, pre-norm blocks with
//! layer scale and a 2D rotary embedding over patch centres, returning every
//! token's last hidden state layer-normalised without affine parameters.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};

/// Encoder configuration (`config.json` of the checkpoint).
#[derive(Debug, Clone, Deserialize)]
pub struct Dinov3Config {
    /// Width.
    pub hidden_size: usize,
    /// Depth.
    pub num_hidden_layers: usize,
    /// Attention heads.
    pub num_attention_heads: usize,
    /// Feed-forward width.
    pub intermediate_size: usize,
    /// Patch side in pixels.
    pub patch_size: usize,
    /// Input channels.
    #[serde(default = "three")]
    pub num_channels: usize,
    /// Register tokens after the class token.
    #[serde(default)]
    pub num_register_tokens: usize,
    /// Rotary base.
    pub rope_theta: f64,
    /// Layer norm epsilon.
    #[serde(default = "eps")]
    pub layer_norm_eps: f64,
    /// Gated feed-forward.
    #[serde(default)]
    pub use_gated_mlp: bool,
    /// Feed-forward activation.
    #[serde(default = "gelu")]
    pub hidden_act: String,
    /// Query bias.
    #[serde(default = "yes")]
    pub query_bias: bool,
    /// Key bias.
    #[serde(default)]
    pub key_bias: bool,
    /// Value bias.
    #[serde(default = "yes")]
    pub value_bias: bool,
    /// Output projection bias.
    #[serde(default = "yes")]
    pub proj_bias: bool,
    /// Feed-forward biases.
    #[serde(default = "yes")]
    pub mlp_bias: bool,
}

fn three() -> usize {
    3
}
fn eps() -> f64 {
    1e-5
}
fn gelu() -> String {
    "gelu".into()
}
fn yes() -> bool {
    true
}

impl Dinov3Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Refuse what this implementation does not run.
    ///
    /// # Errors
    /// [`Error::Config`] naming the unsupported setting.
    pub fn validate(&self) -> Result<()> {
        if self.use_gated_mlp {
            return Err(Error::Config("image encoder: the gated feed-forward is not implemented".into()));
        }
        if self.hidden_act != "gelu" {
            return Err(Error::Config(format!("image encoder activation {:?} is not implemented", self.hidden_act)));
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 || self.head_dim() % 4 != 0 {
            return Err(Error::Config("image encoder heads do not divide the width into multiples of 4".into()));
        }
        Ok(())
    }

    /// Every weight, in the checkpoint's names.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size as u64;
        let f = self.intermediate_size as u64;
        let p = self.patch_size as u64;
        let mut v = vec![
            WeightSpec::new("embeddings.cls_token", &[1, 1, d], WType::F32),
            WeightSpec::new("embeddings.patch_embeddings.weight", &[d, self.num_channels as u64, p, p], WType::F32),
            WeightSpec::new("embeddings.patch_embeddings.bias", &[d], WType::F32),
        ];
        if self.num_register_tokens > 0 {
            v.push(WeightSpec::new("embeddings.register_tokens", &[1, self.num_register_tokens as u64, d], WType::F32));
        }
        for i in 0..self.num_hidden_layers {
            let l = format!("layer.{i}");
            for n in ["norm1", "norm2"] {
                v.push(WeightSpec::new(format!("{l}.{n}.weight"), &[d], WType::F32));
                v.push(WeightSpec::new(format!("{l}.{n}.bias"), &[d], WType::F32));
            }
            for (proj, bias) in [("q_proj", self.query_bias), ("k_proj", self.key_bias), ("v_proj", self.value_bias), ("o_proj", self.proj_bias)] {
                v.push(WeightSpec::new(format!("{l}.attention.{proj}.weight"), &[d, d], linear));
                if bias {
                    v.push(WeightSpec::new(format!("{l}.attention.{proj}.bias"), &[d], WType::F32));
                }
            }
            v.push(WeightSpec::new(format!("{l}.layer_scale1.lambda1"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{l}.layer_scale2.lambda1"), &[d], WType::F32));
            v.push(WeightSpec::new(format!("{l}.mlp.up_proj.weight"), &[f, d], linear));
            v.push(WeightSpec::new(format!("{l}.mlp.down_proj.weight"), &[d, f], linear));
            if self.mlp_bias {
                v.push(WeightSpec::new(format!("{l}.mlp.up_proj.bias"), &[f], WType::F32));
                v.push(WeightSpec::new(format!("{l}.mlp.down_proj.bias"), &[d], WType::F32));
            }
        }
        v
    }

    /// Rotary tables `(cos, sin)` for `prefix` unrotated tokens followed by an
    /// `h x w` patch grid, each `[head_dim, tokens]`.
    #[must_use]
    pub fn rope_tables(&self, prefix: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim();
        let q = hd / 4;
        let inv: Vec<f32> = (0..q).map(|j| 1.0 / (self.rope_theta as f32).powf((4 * j) as f32 / hd as f32)).collect();
        let n = prefix + h * w;
        let mut cos = vec![1f32; hd * n];
        let mut sin = vec![0f32; hd * n];
        for py in 0..h {
            for px in 0..w {
                let t = prefix + py * w + px;
                let cy = 2.0 * ((py as f32 + 0.5) / h as f32) - 1.0;
                let cx = 2.0 * ((px as f32 + 0.5) / w as f32) - 1.0;
                for j in 0..q {
                    let ay = 2.0 * std::f32::consts::PI * cy * inv[j];
                    let ax = 2.0 * std::f32::consts::PI * cx * inv[j];
                    for rep in 0..2 {
                        let base = t * hd + rep * 2 * q;
                        cos[base + j] = ay.cos();
                        sin[base + j] = ay.sin();
                        cos[base + q + j] = ax.cos();
                        sin[base + q + j] = ax.sin();
                    }
                }
            }
        }
        (cos, sin)
    }
}

/// A loaded image encoder.
pub struct Dinov3 {
    /// Configuration.
    pub cfg: Dinov3Config,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for Dinov3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dinov3").field("layers", &self.cfg.num_hidden_layers).finish_non_exhaustive()
    }
}

impl Dinov3 {
    /// Wrap loaded weights; `exact` keeps attention unfused in float32.
    #[must_use]
    pub fn new(cfg: Dinov3Config, w: Weights, exact: bool) -> Self {
        Self { cfg, w, exact }
    }

    /// Features `[tokens, hidden]` (row-major) of an image given as
    /// normalised `[channels, height, width]` row-major floats; height and
    /// width must be multiples of the patch size.
    ///
    /// # Errors
    /// Mis-sized images or backend failures.
    pub fn encode(&self, backend: &Backend, pixels: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        if height % p != 0 || width % p != 0 || pixels.len() != cfg.num_channels * height * width {
            return Err(Error::Request("image size is not a whole number of patches".into()));
        }
        let (gh, gw) = (height / p, width / p);
        let np = gh * gw;
        let k = cfg.num_channels * p * p;
        // Patches as columns in the kernel's (channel, row, column) order.
        let mut patches = vec![0f32; k * np];
        for py in 0..gh {
            for px in 0..gw {
                let col = (py * gw + px) * k;
                for c in 0..cfg.num_channels {
                    for y in 0..p {
                        for x in 0..p {
                            patches[col + (c * p + y) * p + x] = pixels[(c * height + py * p + y) * width + px * p + x];
                        }
                    }
                }
            }
        }
        let prefix = 1 + cfg.num_register_tokens;
        let n = prefix + np;
        let d = cfg.hidden_size as i64;
        let heads = cfg.num_attention_heads as i64;
        let hd = cfg.head_dim() as i64;
        let mut g = Graph::new(backend)?;
        let f32t = llama_cpp_sys_2::GGML_TYPE_F32;
        let pin = g.input(f32t, &[k as i64, np as i64]);
        let cos_in = g.input(f32t, &[hd, 1, n as i64]);
        let sin_in = g.input(f32t, &[hd, 1, n as i64]);
        let w = |s: &str| self.w.get(s);
        let pw = g.reshape(w("embeddings.patch_embeddings.weight"), &[k as i64, d]);
        let pe = g.linear_b(pw, w("embeddings.patch_embeddings.bias"), pin);
        let cls = g.reshape(w("embeddings.cls_token"), &[d, 1]);
        let mut h = cls;
        if cfg.num_register_tokens > 0 {
            let reg = g.reshape(w("embeddings.register_tokens"), &[d, cfg.num_register_tokens as i64]);
            h = g.concat(h, reg, 1);
        }
        let mut h = g.concat(h, pe, 1);
        let eps = cfg.layer_norm_eps as f32;
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..cfg.num_hidden_layers {
            let l = format!("layer.{i}");
            let ln = |g: &mut Graph, x: Tn, nm: &str| {
                let y = g.norm(x, eps);
                let y = g.mul(y, w(&format!("{l}.{nm}.weight")));
                g.add(y, w(&format!("{l}.{nm}.bias")))
            };
            let x = ln(&mut g, h, "norm1");
            let proj = |g: &mut Graph, x: Tn, nm: &str, bias: bool| {
                let y = g.linear(w(&format!("{l}.attention.{nm}.weight")), x);
                if bias { g.add(y, w(&format!("{l}.attention.{nm}.bias"))) } else { y }
            };
            let q = proj(&mut g, x, "q_proj", cfg.query_bias);
            let kk = proj(&mut g, x, "k_proj", cfg.key_bias);
            let v = proj(&mut g, x, "v_proj", cfg.value_bias);
            let q = g.reshape(q, &[hd, heads, n as i64]);
            let kk = g.reshape(kk, &[hd, heads, n as i64]);
            let v = g.reshape(v, &[hd, heads, n as i64]);
            let q = g.rotate_half_rope(q, cos_in, sin_in);
            let kk = g.rotate_half_rope(kk, cos_in, sin_in);
            let q = g.permute(q, [0, 2, 1, 3]);
            let kk = g.permute(kk, [0, 2, 1, 3]);
            let v = g.permute(v, [0, 2, 1, 3]);
            let o = if self.exact {
                let (q, kk, v) = (g.cont(q), g.cont(kk), g.cont(v));
                g.attention_exact(q, kk, v, None, scale)
            } else {
                let kk = g.cast(kk, llama_cpp_sys_2::GGML_TYPE_F16);
                let v = g.cast(v, llama_cpp_sys_2::GGML_TYPE_F16);
                g.attention(q, kk, v, None, scale, true)
            };
            let o = g.reshape(o, &[d, n as i64]);
            let o = proj(&mut g, o, "o_proj", cfg.proj_bias);
            let o = g.mul(o, w(&format!("{l}.layer_scale1.lambda1")));
            h = g.add(h, o);
            let x = ln(&mut g, h, "norm2");
            let x = g.linear(w(&format!("{l}.mlp.up_proj.weight")), x);
            let x = if cfg.mlp_bias { g.add(x, w(&format!("{l}.mlp.up_proj.bias"))) } else { x };
            let x = g.gelu_erf(x);
            let x = g.linear(w(&format!("{l}.mlp.down_proj.weight")), x);
            let x = if cfg.mlp_bias { g.add(x, w(&format!("{l}.mlp.down_proj.bias"))) } else { x };
            let x = g.mul(x, w(&format!("{l}.layer_scale2.lambda1")));
            h = g.add(h, x);
        }
        let out = g.norm(h, 1e-5);
        g.finish(&[out])?;
        g.set_f32(pin, &patches);
        let (cos, sin) = cfg.rope_tables(prefix, gh, gw);
        g.set_f32(cos_in, &cos);
        g.set_f32(sin_in, &sin);
        g.compute()?;
        Ok(g.read_f32(out))
    }
}
