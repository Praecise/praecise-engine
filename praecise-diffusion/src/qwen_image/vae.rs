//! The Qwen-Image autoencoder on single images.
//!
//! The checkpoint is a causal 3D video autoencoder; on one frame each causal
//! convolution sees two zero frames of padding before the image, so only its
//! last temporal tap reads data, and the temporal resampling convolutions do
//! not run on a first frame. A single image is therefore exactly a 2D network
//! over the last taps, which is what this module builds. Activations are laid
//! out `[W, H, C, 1]`.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::{parse, CheckpointFiles, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// The channel norm divides by the L2 norm (clamped at 1e-12), which the RMS
/// norm matches up to this term.
const NORM_EPS: f32 = 1e-24;

/// `vae/config.json` of a Qwen-Image checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenImageVaeConfig {
    pub base_dim: u64,
    pub z_dim: u64,
    pub dim_mult: Vec<u64>,
    pub num_res_blocks: u64,
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    pub temperal_downsample: Vec<bool>,
    #[serde(default)]
    pub is_residual: bool,
    #[serde(default)]
    pub decoder_base_dim: Option<u64>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl QwenImageVaeConfig {
    /// Refuse variants this implementation does not compute.
    ///
    /// # Errors
    /// [`Error::Config`] naming the first unsupported setting.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("image autoencoder: {m}")));
        if self.is_residual || self.decoder_base_dim.is_some() || !self.attn_scales.is_empty() {
            return bad("only the plain layout without stage shortcuts or attention stages is implemented");
        }
        if self.dim_mult.len() < 2 || self.temperal_downsample.len() + 1 != self.dim_mult.len() || self.num_res_blocks == 0 {
            return bad("expected one downsampling stage fewer than resolutions");
        }
        if self.latents_mean.len() != self.z_dim as usize || self.latents_std.len() != self.z_dim as usize {
            return bad("latent statistics do not match the latent channels");
        }
        Ok(())
    }

    /// Pixels per latent along each side.
    #[must_use]
    pub fn scale(&self) -> usize {
        1 << (self.dim_mult.len() - 1)
    }

    fn enc_dims(&self) -> Vec<u64> {
        std::iter::once(1).chain(self.dim_mult.iter().copied()).map(|m| self.base_dim * m).collect()
    }

    /// `(in, out, upsample)` per decoder stage; every stage after one that
    /// upsamples starts at half its predecessor's width.
    fn up_stages(&self) -> Vec<(u64, u64, bool)> {
        let last = *self.dim_mult.last().expect("validated");
        let d: Vec<u64> = std::iter::once(last).chain(self.dim_mult.iter().rev().copied()).map(|m| self.base_dim * m).collect();
        let n = self.dim_mult.len();
        (0..n).map(|i| (if i > 0 { d[i] / 2 } else { d[i] }, d[i + 1], i != n - 1)).collect()
    }

    /// Encoder entries of the flat `down_blocks` list: `(index, in, out)` per
    /// residual block and `(index, out)` per downsampling convolution.
    fn down_layout(&self) -> (Vec<(usize, u64, u64)>, Vec<(usize, u64)>) {
        let d = self.enc_dims();
        let n = self.dim_mult.len();
        let (mut res, mut down, mut k) = (Vec::new(), Vec::new(), 0);
        for i in 0..n {
            let mut c = d[i];
            for _ in 0..self.num_res_blocks {
                res.push((k, c, d[i + 1]));
                c = d[i + 1];
                k += 1;
            }
            if i != n - 1 {
                down.push((k, d[i + 1]));
                k += 1;
            }
        }
        (res, down)
    }

    fn host_tensors(&self, f: &SafeTensors, exact: bool, encoder: bool) -> Result<Vec<HostTensor>> {
        let mut h = Hosts { f, v: Vec::new(), kt: if exact { WType::F32 } else { WType::F16 } };
        let z = self.z_dim;
        if encoder {
            let ed = self.enc_dims();
            h.last_tap("encoder.conv_in", 3, ed[0])?;
            let (res, down) = self.down_layout();
            let mut entries: Vec<(usize, Option<(u64, u64)>, u64)> = res.iter().map(|&(k, i, o)| (k, Some((i, o)), o)).collect();
            entries.extend(down.iter().map(|&(k, c)| (k, None, c)));
            for (k, r, c) in entries {
                let p = format!("encoder.down_blocks.{k}");
                match r {
                    Some((i, o)) => h.resnet(&p, i, o)?,
                    None => h.conv2d(&format!("{p}.resample.1"), c, c)?,
                }
            }
            let top = *ed.last().expect("validated");
            h.mid("encoder.mid_block", top)?;
            h.gamma("encoder.norm_out", top)?;
            h.last_tap("encoder.conv_out", top, 2 * z)?;
            h.pointwise("quant_conv", 2 * z, 2 * z)?;
        } else {
            h.pointwise("post_quant_conv", z, z)?;
            let st = self.up_stages();
            h.last_tap("decoder.conv_in", z, st[0].0)?;
            h.mid("decoder.mid_block", st[0].0)?;
            for (i, &(cin, cout, up)) in st.iter().enumerate() {
                let p = format!("decoder.up_blocks.{i}");
                let mut c = cin;
                for r in 0..=self.num_res_blocks {
                    h.resnet(&format!("{p}.resnets.{r}"), c, cout)?;
                    c = cout;
                }
                if up {
                    h.conv2d_to(&format!("{p}.upsamplers.0.resample.1"), cout, cout / 2)?;
                }
            }
            let last = st.last().expect("validated").1;
            h.gamma("decoder.norm_out", last)?;
            h.last_tap("decoder.conv_out", last, 3)?;
        }
        Ok(h.v)
    }
}

struct Hosts<'a> {
    f: &'a SafeTensors,
    v: Vec<HostTensor>,
    kt: WType,
}

impl Hosts<'_> {
    fn bias(&mut self, p: &str, c: u64) -> Result<()> {
        let b = self.f.require(&format!("{p}.bias"), &[c])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.bias"), shape: vec![c], ty: WType::F32, data: b });
        Ok(())
    }

    /// The last temporal tap of a causal 3x3x3 convolution as a 3x3 kernel.
    fn last_tap(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.f.require(&format!("{p}.weight"), &[cout, cin, 3, 3, 3])?.to_f32();
        let (ci, co) = (cin as usize, cout as usize);
        let mut d = Vec::with_capacity(co * ci * 9);
        for o in 0..co {
            for i in 0..ci {
                let base = ((o * ci + i) * 3 + 2) * 9;
                d.extend_from_slice(&w[base..base + 9]);
            }
        }
        self.v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, 3, 3], ty: self.kt, data: d });
        self.bias(p, cout)
    }

    fn pointwise(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.f.require(&format!("{p}.weight"), &[cout, cin, 1, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, 1, 1], ty: self.kt, data: w });
        self.bias(p, cout)
    }

    fn conv2d(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        self.conv2d_to(p, cin, cout)
    }

    fn conv2d_to(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        let w = self.f.require(&format!("{p}.weight"), &[cout, cin, 3, 3])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, 3, 3], ty: self.kt, data: w });
        self.bias(p, cout)
    }

    fn gamma(&mut self, p: &str, c: u64) -> Result<()> {
        let g = self.f.require(&format!("{p}.gamma"), &[c, 1, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.gamma"), shape: vec![c], ty: WType::F32, data: g });
        Ok(())
    }

    fn resnet(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        self.gamma(&format!("{p}.norm1"), cin)?;
        self.gamma(&format!("{p}.norm2"), cout)?;
        self.last_tap(&format!("{p}.conv1"), cin, cout)?;
        self.last_tap(&format!("{p}.conv2"), cout, cout)?;
        if cin != cout {
            self.pointwise(&format!("{p}.conv_shortcut"), cin, cout)?;
        }
        Ok(())
    }

    fn mid(&mut self, p: &str, c: u64) -> Result<()> {
        self.resnet(&format!("{p}.resnets.0"), c, c)?;
        let a = format!("{p}.attentions.0");
        let g = self.f.require(&format!("{a}.norm.gamma"), &[c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.norm.gamma"), shape: vec![c], ty: WType::F32, data: g });
        let qkv = self.f.require(&format!("{a}.to_qkv.weight"), &[3 * c, c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.to_qkv.weight"), shape: vec![3 * c, c], ty: WType::F32, data: qkv });
        self.bias(&format!("{a}.to_qkv"), 3 * c)?;
        let proj = self.f.require(&format!("{a}.proj.weight"), &[c, c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.proj.weight"), shape: vec![c, c], ty: WType::F32, data: proj });
        self.bias(&format!("{a}.proj"), c)?;
        self.resnet(&format!("{p}.resnets.1"), c, c)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
}

impl Net<'_, '_> {
    fn add_bias(&mut self, y: Tn, p: &str) -> Tn {
        let b = self.w.get(&format!("{p}.bias"));
        let b = self.g.reshape(b, &[1, 1, b.ne(0), 1]);
        self.g.add(y, b)
    }

    fn conv(&mut self, p: &str, x: Tn, pad: i32) -> Tn {
        let y = self.g.conv2d(self.w.get(&format!("{p}.weight")), x, pad);
        self.add_bias(y, p)
    }

    /// Channel RMS norm with gain, then SiLU.
    fn norm_silu(&mut self, p: &str, x: Tn) -> Tn {
        let c = self.g.permute(x, [1, 2, 0, 3]);
        let c = self.g.cont(c);
        let n = self.g.rms_norm(c, NORM_EPS);
        let n = self.g.mul(n, self.w.get(&format!("{p}.gamma")));
        let n = self.g.silu(n);
        let back = self.g.permute(n, [2, 0, 1, 3]);
        self.g.cont(back)
    }

    fn resnet(&mut self, p: &str, x: Tn, cin: u64, cout: u64) -> Tn {
        let h = if cin == cout { x } else { self.conv(&format!("{p}.conv_shortcut"), x, 0) };
        let y = self.norm_silu(&format!("{p}.norm1"), x);
        let y = self.conv(&format!("{p}.conv1"), y, 1);
        let y = self.norm_silu(&format!("{p}.norm2"), y);
        let y = self.conv(&format!("{p}.conv2"), y, 1);
        self.g.add(y, h)
    }

    /// Single-head self-attention over the image.
    fn attention(&mut self, p: &str, x: Tn) -> Tn {
        let (w, h, c) = (x.ne(0), x.ne(1), x.ne(2));
        let n = w * h;
        let xc = self.g.permute(x, [1, 2, 0, 3]);
        let xc = self.g.cont(xc);
        let xn = self.g.rms_norm(xc, NORM_EPS);
        let xn = self.g.mul(xn, self.w.get(&format!("{p}.norm.gamma")));
        let xn = self.g.reshape(xn, &[c, n, 1]);
        let qkv = self.g.linear_b(self.w.get(&format!("{p}.to_qkv.weight")), self.w.get(&format!("{p}.to_qkv.bias")), xn);
        let es = qkv.nb(0);
        let part = |g: &mut Graph, i: i64| {
            let v = g.view_4d(qkv, [c, n, 1, 1], qkv.nb(1), qkv.nb(2), qkv.nb(3), (i * c) as usize * es);
            g.cont(v)
        };
        let q = part(self.g, 0);
        let k = part(self.g, 1);
        let v = part(self.g, 2);
        let o = self.g.attention_exact(q, k, v, None, 1.0 / (c as f32).sqrt());
        let o = self.g.permute(o, [0, 2, 1, 3]);
        let o = self.g.cont(o);
        let o = self.g.linear_b(self.w.get(&format!("{p}.proj.weight")), self.w.get(&format!("{p}.proj.bias")), o);
        let o = self.g.reshape(o, &[c, w, h, 1]);
        let o = self.g.permute(o, [2, 0, 1, 3]);
        let o = self.g.cont(o);
        self.g.add(o, x)
    }

    fn mid(&mut self, p: &str, x: Tn, c: u64) -> Tn {
        let x = self.resnet(&format!("{p}.resnets.0"), x, c, c);
        let x = self.attention(&format!("{p}.attentions.0"), x);
        self.resnet(&format!("{p}.resnets.1"), x, c, c)
    }
}

/// A loaded autoencoder (either half, or both).
pub struct QwenImageVae {
    cfg: QwenImageVaeConfig,
    enc: Option<Weights>,
    dec: Option<Weights>,
}

impl std::fmt::Debug for QwenImageVae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImageVae").field("z_dim", &self.cfg.z_dim).finish_non_exhaustive()
    }
}

impl QwenImageVae {
    /// Load `vae/` of a checkpoint onto `backend`: the decoder, and the
    /// encoder when `encoder` is set (editing needs it for the reference
    /// images).
    ///
    /// # Errors
    /// A missing or malformed config or weight.
    pub fn load(files: &CheckpointFiles, backend: &Backend, precision: Precision, encoder: bool) -> Result<Self> {
        let cfg: QwenImageVaeConfig = parse(files.json("vae/config.json")?, "autoencoder config")?;
        cfg.validate()?;
        let st = SafeTensors::open(&files.weights("vae")?)?;
        let exact = precision == Precision::F32;
        let dec = Some(Weights::from_host(backend, &cfg.host_tensors(&st, exact, false)?)?);
        let enc = if encoder { Some(Weights::from_host(backend, &cfg.host_tensors(&st, exact, true)?)?) } else { None };
        Ok(Self { cfg, enc, dec })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &QwenImageVaeConfig {
        &self.cfg
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.enc.as_ref().map_or(0, Weights::bytes) + self.dec.as_ref().map_or(0, Weights::bytes)
    }

    /// Decode a latent `[z][lh][lw]` (as the transformer sees it, before
    /// denormalisation) to pixels `[3][H][W]` in `[-1, 1]`.
    ///
    /// # Errors
    /// A latent of the wrong size, or a backend failure.
    pub fn decode(&self, backend: &Backend, latent: &[f32], (lh, lw): (usize, usize)) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let z = cfg.z_dim as usize;
        if latent.len() != z * lh * lw || lh == 0 || lw == 0 {
            return Err(Error::Request("latent disagrees with its size".into()));
        }
        let w = self.dec.as_ref().ok_or_else(|| Error::Request("decoder not loaded".into()))?;
        let mut x = latent.to_vec();
        for (c, plane) in x.chunks_exact_mut(lh * lw).enumerate() {
            for v in plane {
                *v = *v * cfg.latents_std[c] + cfg.latents_mean[c];
            }
        }
        let mut g = Graph::new(backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[lw as i64, lh as i64, z as i64, 1]);
        let mut n = Net { g: &mut g, w };
        let mut h = n.conv("post_quant_conv", input, 0);
        let st = cfg.up_stages();
        h = n.conv("decoder.conv_in", h, 1);
        h = n.mid("decoder.mid_block", h, st[0].0);
        for (i, &(cin, cout, up)) in st.iter().enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            let mut c = cin;
            for r in 0..=cfg.num_res_blocks {
                h = n.resnet(&format!("{p}.resnets.{r}"), h, c, cout);
                c = cout;
            }
            if up {
                let u = n.g.upscale_nearest(h, 2);
                h = n.conv(&format!("{p}.upsamplers.0.resample.1"), u, 1);
            }
        }
        h = n.norm_silu("decoder.norm_out", h);
        let out = n.conv("decoder.conv_out", h, 1);
        let out = g.clamp(out, -1.0, 1.0);
        g.finish(&[out])?;
        g.set_f32(input, &x);
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// Encode pixels `[3][H][W]` in `[-1, 1]` (sides multiples of the
    /// scale) to the normalised latent mean `[z][H/s][W/s]`.
    ///
    /// # Errors
    /// Pixels of the wrong size, no encoder, or a backend failure.
    pub fn encode(&self, backend: &Backend, pixels: &[f32], (h, w): (usize, usize)) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let s = cfg.scale();
        if pixels.len() != 3 * h * w || h % s != 0 || w % s != 0 || h == 0 || w == 0 {
            return Err(Error::Request(format!("image sides must be positive multiples of {s}")));
        }
        let wt = self.enc.as_ref().ok_or_else(|| Error::Request("encoder not loaded".into()))?;
        let mut g = Graph::new(backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[w as i64, h as i64, 3, 1]);
        let mut n = Net { g: &mut g, w: wt };
        let mut x = n.conv("encoder.conv_in", input, 1);
        let (res, down) = cfg.down_layout();
        let mut steps: Vec<(usize, Option<(u64, u64)>)> = res.iter().map(|&(k, i, o)| (k, Some((i, o)))).collect();
        steps.extend(down.iter().map(|&(k, _)| (k, None)));
        steps.sort_by_key(|s| s.0);
        for (k, r) in steps {
            let p = format!("encoder.down_blocks.{k}");
            x = match r {
                Some((i, o)) => n.resnet(&p, x, i, o),
                None => {
                    let padded = n.g.pad_end(x, 1, 1);
                    let y = n.g.conv2d_stride2(n.w.get(&format!("{p}.resample.1.weight")), padded);
                    n.add_bias(y, &format!("{p}.resample.1"))
                }
            };
        }
        let top = *cfg.enc_dims().last().expect("validated");
        x = n.mid("encoder.mid_block", x, top);
        x = n.norm_silu("encoder.norm_out", x);
        x = n.conv("encoder.conv_out", x, 1);
        let q = n.conv("quant_conv", x, 0);
        g.finish(&[q])?;
        g.set_f32(input, pixels);
        g.compute()?;
        let all = g.read_f32(q);
        let plane = (h / s) * (w / s);
        let z = cfg.z_dim as usize;
        let mut mean = all[..z * plane].to_vec();
        for (c, p) in mean.chunks_exact_mut(plane).enumerate() {
            for v in p {
                *v = (*v - cfg.latents_mean[c]) / cfg.latents_std[c];
            }
        }
        Ok(mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QwenImageVaeConfig {
        serde_json::from_value(serde_json::json!({
            "attn_scales": [], "base_dim": 96, "dim_mult": [1, 2, 4, 4], "dropout": 0.0,
            "latents_mean": vec![0.0; 16], "latents_std": vec![1.0; 16], "num_res_blocks": 2,
            "temperal_downsample": [false, true, true], "z_dim": 16
        }))
        .unwrap()
    }

    #[test]
    fn the_released_layout_has_the_reference_stage_widths() {
        let c = cfg();
        c.validate().unwrap();
        assert_eq!(c.scale(), 8);
        assert_eq!(c.up_stages(), vec![(384, 384, true), (192, 384, true), (192, 192, true), (96, 96, false)]);
        let (res, down) = c.down_layout();
        assert_eq!(res.len(), 8);
        assert_eq!(down, vec![(2, 96), (5, 192), (8, 384)]);
        assert_eq!(res[2], (3, 96, 192));
    }
}
