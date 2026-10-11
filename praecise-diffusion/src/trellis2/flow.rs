//! The flow transformer shared by every TRELLIS.2 sampling stage: tokens are
//! voxels (all `16^3` cells of the occupancy latent, or the occupied cells of
//! a structured latent), each block is adaptive-norm self-attention with a
//! 3-axis rotary embedding over the voxel coordinates, cross-attention to the
//! image features, and a GELU feed-forward, all modulated by one shared
//! timestep projection plus a learned per-block offset.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::flux2::timestep_features;
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};

const TIME_FEATURES: usize = 256;
const BLOCK_EPS: f32 = 1e-6;
const OUT_EPS: f32 = 1e-5;
/// `F.normalize` divides by `max(|x|, 1e-12)`; as an RMS norm that floor is
/// negligible.
const QK_EPS: f32 = 1e-12;

/// Flow transformer configuration (`args` of the checkpoint's JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    /// Grid side of the latent the tokens live on.
    pub resolution: usize,
    /// Input channels (noisy latent plus any concatenated condition).
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Width.
    pub model_channels: usize,
    /// Width of the image features.
    pub cond_channels: usize,
    /// Depth.
    pub num_blocks: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Feed-forward width over model width.
    pub mlp_ratio: f64,
    /// Position encoding; only `rope` is implemented.
    pub pe_mode: String,
    /// One modulation projection shared by all blocks.
    #[serde(default)]
    pub share_mod: bool,
    /// RMS-normalised queries and keys in self-attention.
    #[serde(default)]
    pub qk_rms_norm: bool,
    /// RMS-normalised queries and keys in cross-attention.
    #[serde(default)]
    pub qk_rms_norm_cross: bool,
    /// Rotary frequency range `(scale, base)`.
    #[serde(default = "default_rope_freq")]
    pub rope_freq: (f64, f64),
}

fn default_rope_freq() -> (f64, f64) {
    (1.0, 10000.0)
}

impl FlowConfig {
    /// Feed-forward width, as the reference truncates it.
    #[must_use]
    pub fn mlp_width(&self) -> usize {
        (self.model_channels as f64 * self.mlp_ratio) as usize
    }

    /// Per-head width.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.model_channels / self.num_heads
    }

    /// Refuse what this implementation does not run.
    ///
    /// # Errors
    /// [`Error::Config`] naming the unsupported setting.
    pub fn validate(&self) -> Result<()> {
        if self.pe_mode != "rope" {
            return Err(Error::Config(format!("flow transformer position encoding {:?}: only rope is implemented", self.pe_mode)));
        }
        if !self.share_mod {
            return Err(Error::Config("flow transformer without a shared modulation is not implemented".into()));
        }
        if self.num_heads == 0 || self.model_channels % self.num_heads != 0 || self.head_dim() % 2 != 0 {
            return Err(Error::Config("flow transformer heads do not divide the width into even head widths".into()));
        }
        if self.head_dim() / 2 / 3 == 0 {
            return Err(Error::Config("flow transformer head width too small for three rotary axes".into()));
        }
        Ok(())
    }

    /// Every weight, in the checkpoint's names.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let c = self.model_channels as u64;
        let cc = self.cond_channels as u64;
        let m = self.mlp_width() as u64;
        let h = self.num_heads as u64;
        let hd = self.head_dim() as u64;
        let mut v = vec![
            WeightSpec::new("t_embedder.mlp.0.weight", &[c, TIME_FEATURES as u64], linear),
            WeightSpec::new("t_embedder.mlp.0.bias", &[c], WType::F32),
            WeightSpec::new("t_embedder.mlp.2.weight", &[c, c], linear),
            WeightSpec::new("t_embedder.mlp.2.bias", &[c], WType::F32),
            WeightSpec::new("adaLN_modulation.1.weight", &[6 * c, c], linear),
            WeightSpec::new("adaLN_modulation.1.bias", &[6 * c], WType::F32),
            WeightSpec::new("input_layer.weight", &[c, self.in_channels as u64], WType::F32),
            WeightSpec::new("input_layer.bias", &[c], WType::F32),
            WeightSpec::new("out_layer.weight", &[self.out_channels as u64, c], WType::F32),
            WeightSpec::new("out_layer.bias", &[self.out_channels as u64], WType::F32),
        ];
        for i in 0..self.num_blocks {
            let p = format!("blocks.{i}");
            v.push(WeightSpec::new(format!("{p}.modulation"), &[6 * c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.norm2.weight"), &[c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.norm2.bias"), &[c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.self_attn.to_qkv.weight"), &[3 * c, c], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.to_qkv.bias"), &[3 * c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.self_attn.to_out.weight"), &[c, c], linear));
            v.push(WeightSpec::new(format!("{p}.self_attn.to_out.bias"), &[c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.cross_attn.to_q.weight"), &[c, c], linear));
            v.push(WeightSpec::new(format!("{p}.cross_attn.to_q.bias"), &[c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.cross_attn.to_kv.weight"), &[2 * c, cc], linear));
            v.push(WeightSpec::new(format!("{p}.cross_attn.to_kv.bias"), &[2 * c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.cross_attn.to_out.weight"), &[c, c], linear));
            v.push(WeightSpec::new(format!("{p}.cross_attn.to_out.bias"), &[c], WType::F32));
            v.push(WeightSpec::new(format!("{p}.mlp.mlp.0.weight"), &[m, c], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.mlp.0.bias"), &[m], WType::F32));
            v.push(WeightSpec::new(format!("{p}.mlp.mlp.2.weight"), &[c, m], linear));
            v.push(WeightSpec::new(format!("{p}.mlp.mlp.2.bias"), &[c], WType::F32));
            if self.qk_rms_norm {
                v.push(WeightSpec::new(format!("{p}.self_attn.q_rms_norm.gamma"), &[h, hd], WType::F32));
                v.push(WeightSpec::new(format!("{p}.self_attn.k_rms_norm.gamma"), &[h, hd], WType::F32));
            }
            if self.qk_rms_norm_cross {
                v.push(WeightSpec::new(format!("{p}.cross_attn.q_rms_norm.gamma"), &[h, hd], WType::F32));
                v.push(WeightSpec::new(format!("{p}.cross_attn.k_rms_norm.gamma"), &[h, hd], WType::F32));
            }
        }
        v
    }

    /// Rotary tables for tokens at `coords`: `(cos, sin)`, each
    /// `[head_dim / 2, tokens]`, pair `j` of axis `a` turning by
    /// `coord[a] * scale / base^(j / f)` with `f = head_dim / 6` pairs per
    /// axis; pairs past the three axes do not turn.
    #[must_use]
    pub fn rope_tables(&self, coords: &[[i32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_dim() / 2;
        let f = half / 3;
        let freqs: Vec<f32> = (0..f)
            .map(|j| (self.rope_freq.0 as f32) / (self.rope_freq.1 as f32).powf(j as f32 / f as f32))
            .collect();
        let mut cos = vec![1f32; half * coords.len()];
        let mut sin = vec![0f32; half * coords.len()];
        for (t, c) in coords.iter().enumerate() {
            for a in 0..3 {
                for j in 0..f {
                    let ang = c[a] as f32 * freqs[j];
                    cos[t * half + a * f + j] = ang.cos();
                    sin[t * half + a * f + j] = ang.sin();
                }
            }
        }
        (cos, sin)
    }
}

/// The inputs of one forward pass, declared on a graph.
struct Inputs {
    x: Tn,
    tfeat: Tn,
    cond: Tn,
    cos: Tn,
    sin: Tn,
}

/// A loaded flow transformer.
pub struct FlowModel {
    /// Configuration.
    pub cfg: FlowConfig,
    w: Weights,
    exact: bool,
}

impl std::fmt::Debug for FlowModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowModel").field("blocks", &self.cfg.num_blocks).field("bytes", &self.w.bytes()).finish_non_exhaustive()
    }
}

impl FlowModel {
    /// Wrap loaded weights. `exact` keeps attention unfused and in float32.
    #[must_use]
    pub fn new(cfg: FlowConfig, w: Weights, exact: bool) -> Self {
        Self { cfg, w, exact }
    }

    /// Device bytes of the weights.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Predicted velocity `[out_channels, tokens]` (token-major on the host:
    /// `out[t * out_channels + c]`) for latent `x` (`[tokens, in_channels]`
    /// row-major) at flow time `t` (already scaled by 1000), conditioned on
    /// image features `cond` (`[cond_tokens, cond_channels]` row-major).
    ///
    /// # Errors
    /// Shape mismatches or backend failures.
    pub fn forward(&self, backend: &Backend, x: &[f32], coords: &[[i32; 3]], t: f32, cond: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let n = coords.len();
        if x.len() != n * cfg.in_channels {
            return Err(Error::Request(format!("flow input holds {} values for {n} tokens of {}", x.len(), cfg.in_channels)));
        }
        if cond.is_empty() || cond.len() % cfg.cond_channels != 0 {
            return Err(Error::Request("image features do not match the condition width".into()));
        }
        let lc = cond.len() / cfg.cond_channels;
        let half = cfg.head_dim() / 2;
        let mut g = Graph::new(backend)?;
        let f32t = llama_cpp_sys_2::GGML_TYPE_F32;
        let inp = Inputs {
            x: g.input(f32t, &[cfg.in_channels as i64, n as i64]),
            tfeat: g.input(f32t, &[TIME_FEATURES as i64]),
            cond: g.input(f32t, &[cfg.cond_channels as i64, lc as i64]),
            cos: g.input(f32t, &[1, half as i64, 1, n as i64]),
            sin: g.input(f32t, &[1, half as i64, 1, n as i64]),
        };
        let out = self.build(&mut g, &inp, n as i64, lc as i64);
        g.finish(&[out])?;
        g.set_f32(inp.x, x);
        g.set_f32(inp.tfeat, &timestep_features(t, TIME_FEATURES));
        g.set_f32(inp.cond, cond);
        let (cos, sin) = cfg.rope_tables(coords);
        g.set_f32(inp.cos, &cos);
        g.set_f32(inp.sin, &sin);
        g.compute()?;
        Ok(g.read_f32(out))
    }

    fn build(&self, g: &mut Graph, inp: &Inputs, n: i64, lc: i64) -> Tn {
        let cfg = &self.cfg;
        let w = |s: &str| self.w.get(s);
        let c = cfg.model_channels as i64;
        let heads = cfg.num_heads as i64;
        let hd = cfg.head_dim() as i64;

        // Timestep embedding and the shared modulation.
        let te = g.linear_b(w("t_embedder.mlp.0.weight"), w("t_embedder.mlp.0.bias"), inp.tfeat);
        let te = g.silu(te);
        let te = g.linear_b(w("t_embedder.mlp.2.weight"), w("t_embedder.mlp.2.bias"), te);
        let te = g.silu(te);
        let shared = g.linear_b(w("adaLN_modulation.1.weight"), w("adaLN_modulation.1.bias"), te);

        let mut h = g.linear_b(w("input_layer.weight"), w("input_layer.bias"), inp.x);
        for i in 0..cfg.num_blocks {
            let p = format!("blocks.{i}");
            let m = g.add(shared, w(&format!("{p}.modulation")));
            let chunk = |g: &mut Graph, k: i64| g.view_1d(m, c, (k * c) as usize);
            let (shift_msa, scale_msa, gate_msa) = (chunk(g, 0), chunk(g, 1), chunk(g, 2));
            let (shift_mlp, scale_mlp, gate_mlp) = (chunk(g, 3), chunk(g, 4), chunk(g, 5));

            // Self-attention.
            let x = g.norm(h, BLOCK_EPS);
            let x = modulate(g, x, shift_msa, scale_msa);
            let qkv = g.linear_b(w(&format!("{p}.self_attn.to_qkv.weight")), w(&format!("{p}.self_attn.to_qkv.bias")), x);
            let q = g.view_heads(qkv, 0, hd, heads);
            let k = g.view_heads(qkv, c, hd, heads);
            let v = g.view_heads(qkv, 2 * c, hd, heads);
            let (q, k) = if cfg.qk_rms_norm {
                (
                    head_rms_norm(g, q, w(&format!("{p}.self_attn.q_rms_norm.gamma"))),
                    head_rms_norm(g, k, w(&format!("{p}.self_attn.k_rms_norm.gamma"))),
                )
            } else {
                (g.cont(q), g.cont(k))
            };
            let q = rope_pairs(g, q, inp.cos, inp.sin);
            let k = rope_pairs(g, k, inp.cos, inp.sin);
            let v = g.cont(v);
            let a = self.attend(g, q, k, v, n);
            let a = g.linear_b(w(&format!("{p}.self_attn.to_out.weight")), w(&format!("{p}.self_attn.to_out.bias")), a);
            let a = g.mul(a, gate_msa);
            h = g.add(h, a);

            // Cross-attention to the image features (affine norm, no modulation).
            let x = g.norm(h, BLOCK_EPS);
            let x = g.mul(x, w(&format!("{p}.norm2.weight")));
            let x = g.add(x, w(&format!("{p}.norm2.bias")));
            let q = g.linear_b(w(&format!("{p}.cross_attn.to_q.weight")), w(&format!("{p}.cross_attn.to_q.bias")), x);
            let kv = g.linear_b(w(&format!("{p}.cross_attn.to_kv.weight")), w(&format!("{p}.cross_attn.to_kv.bias")), inp.cond);
            let q = g.reshape(q, &[hd, heads, n]);
            let k = g.view_heads(kv, 0, hd, heads);
            let v = g.view_heads(kv, c, hd, heads);
            let (q, k) = if cfg.qk_rms_norm_cross {
                (
                    head_rms_norm(g, q, w(&format!("{p}.cross_attn.q_rms_norm.gamma"))),
                    head_rms_norm(g, k, w(&format!("{p}.cross_attn.k_rms_norm.gamma"))),
                )
            } else {
                (q, g.cont(k))
            };
            let v = g.cont(v);
            let _ = lc;
            let a = self.attend(g, q, k, v, n);
            let a = g.linear_b(w(&format!("{p}.cross_attn.to_out.weight")), w(&format!("{p}.cross_attn.to_out.bias")), a);
            h = g.add(h, a);

            // Feed-forward.
            let x = g.norm(h, BLOCK_EPS);
            let x = modulate(g, x, shift_mlp, scale_mlp);
            let x = g.linear_b(w(&format!("{p}.mlp.mlp.0.weight")), w(&format!("{p}.mlp.mlp.0.bias")), x);
            let x = if self.exact { g.gelu_tanh_exact(x) } else { g.gelu_tanh(x) };
            let x = g.linear_b(w(&format!("{p}.mlp.mlp.2.weight")), w(&format!("{p}.mlp.mlp.2.bias")), x);
            let x = g.mul(x, gate_mlp);
            h = g.add(h, x);
        }
        let h = g.norm(h, OUT_EPS);
        g.linear_b(w("out_layer.weight"), w("out_layer.bias"), h)
    }

    /// Attention of `q` (`[hd, heads, n]`) over `k`, `v` (`[hd, heads, m]`):
    /// result `[hd * heads, n]`.
    fn attend(&self, g: &mut Graph, q: Tn, k: Tn, v: Tn, n: i64) -> Tn {
        let hd = q.ne(0);
        let heads = q.ne(1);
        let scale = 1.0 / (hd as f32).sqrt();
        let q = g.permute(q, [0, 2, 1, 3]);
        let k = g.permute(k, [0, 2, 1, 3]);
        let v = g.permute(v, [0, 2, 1, 3]);
        let o = if self.exact {
            let q = g.cont(q);
            let k = g.cont(k);
            let v = g.cont(v);
            g.attention_exact(q, k, v, None, scale)
        } else {
            let k = g.cast(k, llama_cpp_sys_2::GGML_TYPE_F16);
            let v = g.cast(v, llama_cpp_sys_2::GGML_TYPE_F16);
            g.attention(q, k, v, None, scale, true)
        };
        g.reshape(o, &[hd * heads, n])
    }
}

/// `x * (1 + scale) + shift` with per-channel `scale`, `shift`.
fn modulate(g: &mut Graph, x: Tn, shift: Tn, scale: Tn) -> Tn {
    let s = g.scale_bias(scale, 1.0, 1.0);
    let y = g.mul(x, s);
    g.add(y, shift)
}

/// Per-head RMS normalisation of `x` (`[hd, heads, n]`) with a learned
/// `[heads, hd]` gain.
fn head_rms_norm(g: &mut Graph, x: Tn, gamma: Tn) -> Tn {
    let x = g.cont(x);
    let y = g.rms_norm(x, QK_EPS);
    g.mul(y, gamma)
}

/// Rotate adjacent pairs `(x[2j], x[2j + 1])` of each head by the angle
/// tables (`[1, hd / 2, 1, n]`).
fn rope_pairs(g: &mut Graph, x: Tn, cos: Tn, sin: Tn) -> Tn {
    let (hd, heads, n) = (x.ne(0), x.ne(1), x.ne(2));
    let x4 = g.reshape(x, &[2, hd / 2, heads, n]);
    let es = x4.nb(0);
    let a = g.view_4d(x4, [1, hd / 2, heads, n], x4.nb(1), x4.nb(2), x4.nb(3), 0);
    let b = g.view_4d(x4, [1, hd / 2, heads, n], x4.nb(1), x4.nb(2), x4.nb(3), es);
    let a = g.cont(a);
    let b = g.cont(b);
    let ac = g.mul(a, cos);
    let bs = g.mul(b, sin);
    let ra = g.sub(ac, bs);
    let as_ = g.mul(a, sin);
    let bc = g.mul(b, cos);
    let rb = g.add(as_, bc);
    let r = g.concat(ra, rb, 0);
    g.reshape(r, &[hd, heads, n])
}
