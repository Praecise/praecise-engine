//! The Qwen-Image 2.1 autoencoder: a residual image autoencoder (averaging
//! and duplicating shortcuts around every resolution stage, 16x spatial, 64
//! latent channels, RGBA pixels).
//!
//! Its convolutions are 2D, and on a single image the temporal parts of its
//! stages never run: the averaging shortcut of a stage that halves time sees
//! a zero frame before the image (so it averages over half the group), and
//! the duplicating shortcut keeps only the last temporal phase. Activations
//! are laid out `[W, H, C, 1]`.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::{parse, CheckpointFiles, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// The channel norm divides by the L2 norm (clamped at 1e-12), which the RMS
/// norm matches up to this term.
const NORM_EPS: f32 = 1e-24;

/// `vae/config.json` of a Qwen-Image 2.1 checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenImage21VaeConfig {
    pub base_dim: u64,
    #[serde(default)]
    pub decoder_base_dim: Option<u64>,
    pub z_dim: u64,
    pub dim_mult: Vec<u64>,
    pub num_res_blocks: u64,
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    pub temperal_downsample: Vec<bool>,
    #[serde(default)]
    pub is_residual: bool,
    pub in_channels: u64,
    pub out_channels: u64,
    #[serde(default)]
    pub patch_size: Option<u64>,
    pub scale_factor_spatial: u64,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

/// One resolution stage: `(in, out, halves time, changes resolution)`.
type Stage = (u64, u64, bool, bool);

impl QwenImage21VaeConfig {
    /// Refuse variants this implementation does not compute.
    ///
    /// # Errors
    /// [`Error::Config`] naming the unsupported setting.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("autoencoder: {m}")));
        if !self.is_residual || self.patch_size.is_some() || !self.attn_scales.is_empty() {
            return bad("only the residual layout without pixel patches or attention stages is implemented");
        }
        let n = self.dim_mult.len();
        if n < 2 || self.temperal_downsample.len() != n - 1 || self.num_res_blocks == 0 {
            return bad("one fewer downsampling stage than resolutions expected");
        }
        if !(3..=4).contains(&self.in_channels) || self.out_channels != self.in_channels {
            return bad("expected RGB or RGBA in and out");
        }
        if self.scale_factor_spatial != 1 << (n - 1) {
            return bad("spatial scale disagrees with the stages");
        }
        if self.latents_mean.len() != self.z_dim as usize || self.latents_std.len() != self.z_dim as usize {
            return bad("latent statistics do not match the latent channels");
        }
        Ok(())
    }

    /// Pixels per latent along each side.
    #[must_use]
    pub fn scale(&self) -> usize {
        self.scale_factor_spatial as usize
    }

    fn enc_dims(&self) -> Vec<u64> {
        std::iter::once(1).chain(self.dim_mult.iter().copied()).map(|m| self.base_dim * m).collect()
    }

    fn dec_dims(&self) -> Vec<u64> {
        let b = self.decoder_base_dim.unwrap_or(self.base_dim);
        let last = *self.dim_mult.last().expect("validated");
        std::iter::once(last).chain(self.dim_mult.iter().rev().copied()).map(|m| b * m).collect()
    }

    fn down_stages(&self) -> Vec<Stage> {
        let d = self.enc_dims();
        let n = self.dim_mult.len();
        (0..n).map(|i| (d[i], d[i + 1], i != n - 1 && self.temperal_downsample[i], i != n - 1)).collect()
    }

    fn up_stages(&self) -> Vec<Stage> {
        let d = self.dec_dims();
        let n = self.dim_mult.len();
        let up: Vec<bool> = self.temperal_downsample.iter().rev().copied().collect();
        (0..n).map(|i| (d[i], d[i + 1], i != n - 1 && up[i], i != n - 1)).collect()
    }

    fn host_tensors(&self, f: &SafeTensors, exact: bool, encoder: bool) -> Result<Vec<HostTensor>> {
        let mut h = Hosts { f, v: Vec::new(), kt: if exact { WType::F32 } else { WType::F16 } };
        let z = self.z_dim;
        if encoder {
            let ed = self.enc_dims();
            h.conv("encoder.conv_in", self.in_channels, ed[0], 3)?;
            for (i, (cin, cout, temporal, spatial)) in self.down_stages().into_iter().enumerate() {
                let p = format!("encoder.down_blocks.{i}");
                let mut c = cin;
                for r in 0..self.num_res_blocks {
                    h.resnet(&format!("{p}.resnets.{r}"), c, cout)?;
                    c = cout;
                }
                if spatial {
                    h.conv(&format!("{p}.downsampler.resample.1"), cout, cout, 3)?;
                }
                if let Some(k) = avg_down_kernel(cin, cout, temporal, spatial) {
                    h.v.push(HostTensor { name: format!("{p}.avg_shortcut"), shape: vec![cout, cin, 2, 2], ty: h.kt, data: k });
                }
            }
            let top = *ed.last().expect("validated");
            h.mid("encoder.mid_block", top)?;
            h.gamma("encoder.norm_out", top)?;
            h.conv("encoder.conv_out", top, 2 * z, 3)?;
            h.conv("quant_conv", 2 * z, 2 * z, 1)?;
        } else {
            let dd = self.dec_dims();
            h.conv("post_quant_conv", z, z, 1)?;
            h.conv("decoder.conv_in", z, dd[0], 3)?;
            h.mid("decoder.mid_block", dd[0])?;
            for (i, (cin, cout, _, spatial)) in self.up_stages().into_iter().enumerate() {
                let p = format!("decoder.up_blocks.{i}");
                let mut c = cin;
                for r in 0..=self.num_res_blocks {
                    h.resnet(&format!("{p}.resnets.{r}"), c, cout)?;
                    c = cout;
                }
                if spatial {
                    h.conv(&format!("{p}.upsampler.resample.1"), cout, cout, 3)?;
                }
            }
            let last = *dd.last().expect("validated");
            h.gamma("decoder.norm_out", last)?;
            h.conv("decoder.conv_out", last, self.out_channels, 3)?;
        }
        Ok(h.v)
    }
}

/// The encoder's averaging shortcut on a single image as a 2x2, stride-2
/// kernel `[out, in, 2, 2]`: output channel `o` averages the `group`
/// space-to-depth channels `o * group ..`; where the stage halves time the
/// image is the second frame of its pair (the first is zero padding), so only
/// those channels carry weight. `None` when the shortcut is the identity.
fn avg_down_kernel(cin: u64, cout: u64, temporal: bool, spatial: bool) -> Option<Vec<f32>> {
    let (ft, fs) = (if temporal { 2 } else { 1 }, if spatial { 2 } else { 1 });
    if ft == 1 && fs == 1 && cin == cout {
        return None;
    }
    assert_eq!(fs, 2, "a channel-changing shortcut always downsamples space");
    let (cin, cout) = (cin as usize, cout as usize);
    let factor = ft * fs * fs;
    let group = cin * factor / cout;
    let mut k = vec![0f32; cout * cin * 4];
    for o in 0..cout {
        for gi in 0..group {
            let j = o * group + gi;
            let (c, sub) = (j / factor, j % factor);
            let (it, rem) = (sub / (fs * fs), sub % (fs * fs));
            if it != ft - 1 {
                continue;
            }
            let (ih, iw) = (rem / fs, rem % fs);
            k[((o * cin + c) * 2 + ih) * 2 + iw] += 1.0 / group as f32;
        }
    }
    Some(k)
}

struct Hosts<'a> {
    f: &'a SafeTensors,
    v: Vec<HostTensor>,
    kt: WType,
}

impl Hosts<'_> {
    fn conv(&mut self, p: &str, cin: u64, cout: u64, k: u64) -> Result<()> {
        let w = self.f.require(&format!("{p}.weight"), &[cout, cin, k, k])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.weight"), shape: vec![cout, cin, k, k], ty: self.kt, data: w });
        let b = self.f.require(&format!("{p}.bias"), &[cout])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.bias"), shape: vec![cout], ty: WType::F32, data: b });
        Ok(())
    }

    fn gamma(&mut self, p: &str, c: u64) -> Result<()> {
        let g = self.f.require(&format!("{p}.gamma"), &[c, 1, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{p}.gamma"), shape: vec![c], ty: WType::F32, data: g });
        Ok(())
    }

    fn resnet(&mut self, p: &str, cin: u64, cout: u64) -> Result<()> {
        self.gamma(&format!("{p}.norm1"), cin)?;
        self.gamma(&format!("{p}.norm2"), cout)?;
        self.conv(&format!("{p}.conv1"), cin, cout, 3)?;
        self.conv(&format!("{p}.conv2"), cout, cout, 3)?;
        if cin != cout {
            self.conv(&format!("{p}.conv_shortcut"), cin, cout, 1)?;
        }
        Ok(())
    }

    fn mid(&mut self, p: &str, c: u64) -> Result<()> {
        self.resnet(&format!("{p}.resnets.0"), c, c)?;
        let a = format!("{p}.attentions.0");
        let g = self.f.require(&format!("{a}.norm.gamma"), &[c, 1, 1])?.to_f32();
        self.v.push(HostTensor { name: format!("{a}.norm.gamma"), shape: vec![c], ty: WType::F32, data: g });
        for (name, out) in [("to_qkv", 3 * c), ("proj", c)] {
            let w = self.f.require(&format!("{a}.{name}.weight"), &[out, c, 1, 1])?.to_f32();
            self.v.push(HostTensor { name: format!("{a}.{name}.weight"), shape: vec![out, c], ty: WType::F32, data: w });
            let b = self.f.require(&format!("{a}.{name}.bias"), &[out])?.to_f32();
            self.v.push(HostTensor { name: format!("{a}.{name}.bias"), shape: vec![out], ty: WType::F32, data: b });
        }
        self.resnet(&format!("{p}.resnets.1"), c, c)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
    /// Gather indices to set before the run.
    feeds: Vec<(Tn, Vec<i32>)>,
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

    /// The decoder's duplicating shortcut on a single image: output channel
    /// `o` at sub-position `(ih, iw)` repeats input channel `(o * factor +
    /// (ft - 1) * 4 + ih * 2 + iw) / repeats`.
    fn dup_up(&mut self, x: Tn, cout: u64, temporal: bool) -> Tn {
        let (w, h, cin) = (x.ne(0), x.ne(1), x.ne(2) as usize);
        let cout = cout as usize;
        let ft = if temporal { 2 } else { 1 };
        let factor = ft * 4;
        let repeats = cout * factor / cin;
        let flat = self.g.reshape(x, &[w * h, cin as i64]);
        let ids: Vec<i32> = (0..cout * 4).map(|s| ((s / 4 * factor + (ft - 1) * 4 + s % 4) / repeats) as i32).collect();
        let idt = self.g.input(sys::GGML_TYPE_I32, &[(cout * 4) as i64]);
        self.feeds.push((idt, ids));
        let z = self.g.get_rows(flat, idt);
        let z = self.g.reshape(z, &[w, h, (cout * 4) as i64, 1]);
        pixel_shuffle(self.g, z)
    }
}

/// `[W, H, C * 4, 1]` with channel `c * 4 + ih * 2 + iw` to `[2W, 2H, C, 1]`.
fn pixel_shuffle(g: &mut Graph, z: Tn) -> Tn {
    let (w, h, c4) = (z.ne(0), z.ne(1), z.ne(2));
    let r = c4 / 2;
    let a = g.reshape(z, &[w, h, 2, r]);
    let a = g.permute(a, [1, 2, 0, 3]);
    let a = g.cont(a);
    let a = g.reshape(a, &[2 * w, h, 2, r / 2]);
    let a = g.permute(a, [0, 2, 1, 3]);
    let a = g.cont(a);
    g.reshape(a, &[2 * w, 2 * h, c4 / 4, 1])
}

/// A loaded autoencoder (either half, or both).
pub struct QwenImage21Vae {
    cfg: QwenImage21VaeConfig,
    enc: Option<Weights>,
    dec: Option<Weights>,
}

impl std::fmt::Debug for QwenImage21Vae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImage21Vae").field("z_dim", &self.cfg.z_dim).finish_non_exhaustive()
    }
}

impl QwenImage21Vae {
    /// Load `vae/` of a checkpoint onto `backend`: the decoder, and the
    /// encoder when `encoder` is set (condition images need it).
    ///
    /// # Errors
    /// A missing or malformed config or weight.
    pub fn load(files: &CheckpointFiles, backend: &Backend, precision: Precision, encoder: bool) -> Result<Self> {
        let cfg: QwenImage21VaeConfig = parse(files.json("vae/config.json")?, "autoencoder config")?;
        cfg.validate()?;
        let st = SafeTensors::open(&files.weights("vae")?)?;
        let exact = precision == Precision::F32;
        let dec = Some(Weights::from_host(backend, &cfg.host_tensors(&st, exact, false)?)?);
        let enc = if encoder { Some(Weights::from_host(backend, &cfg.host_tensors(&st, exact, true)?)?) } else { None };
        Ok(Self { cfg, enc, dec })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &QwenImage21VaeConfig {
        &self.cfg
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.enc.as_ref().map_or(0, Weights::bytes) + self.dec.as_ref().map_or(0, Weights::bytes)
    }

    /// Decode a latent `[z][lh][lw]` (as the transformer sees it, before
    /// denormalisation) to pixels `[out channels][H][W]` in `[-1, 1]`.
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
        let mut n = Net { g: &mut g, w, feeds: Vec::new() };
        let mut h = n.conv("post_quant_conv", input, 0);
        h = n.conv("decoder.conv_in", h, 1);
        let st = cfg.up_stages();
        h = n.mid("decoder.mid_block", h, st[0].0);
        for (i, &(cin, cout, temporal, spatial)) in st.iter().enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            let skip = h;
            let mut c = cin;
            for r in 0..=cfg.num_res_blocks {
                h = n.resnet(&format!("{p}.resnets.{r}"), h, c, cout);
                c = cout;
            }
            if spatial {
                let u = n.g.upscale_nearest(h, 2);
                h = n.conv(&format!("{p}.upsampler.resample.1"), u, 1);
                let s = n.dup_up(skip, cout, temporal);
                h = n.g.add(h, s);
            }
        }
        h = n.norm_silu("decoder.norm_out", h);
        let out = n.conv("decoder.conv_out", h, 1);
        let feeds = std::mem::take(&mut n.feeds);
        let out = g.clamp(out, -1.0, 1.0);
        g.finish(&[out])?;
        g.set_f32(input, &x);
        for (t, ids) in &feeds {
            g.set_i32(*t, ids);
        }
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// Encode pixels `[in channels][H][W]` in `[-1, 1]` (sides multiples of
    /// the scale) to the normalised latent mean `[z][H/s][W/s]`.
    ///
    /// # Errors
    /// Pixels of the wrong size, no encoder, or a backend failure.
    pub fn encode(&self, backend: &Backend, pixels: &[f32], (h, w): (usize, usize)) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (s, ch) = (cfg.scale(), cfg.in_channels as usize);
        if pixels.len() != ch * h * w || h % s != 0 || w % s != 0 || h == 0 || w == 0 {
            return Err(Error::Request(format!("image sides must be positive multiples of {s}")));
        }
        let wt = self.enc.as_ref().ok_or_else(|| Error::Request("encoder not loaded".into()))?;
        let mut g = Graph::new(backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[w as i64, h as i64, ch as i64, 1]);
        let mut n = Net { g: &mut g, w: wt, feeds: Vec::new() };
        let mut x = n.conv("encoder.conv_in", input, 1);
        for (i, (cin, cout, temporal, spatial)) in cfg.down_stages().into_iter().enumerate() {
            let p = format!("encoder.down_blocks.{i}");
            let skip = x;
            let mut c = cin;
            for r in 0..cfg.num_res_blocks {
                x = n.resnet(&format!("{p}.resnets.{r}"), x, c, cout);
                c = cout;
            }
            if spatial {
                let padded = n.g.pad_end(x, 1, 1);
                let y = n.g.conv2d_stride2(n.w.get(&format!("{p}.downsampler.resample.1.weight")), padded);
                x = n.add_bias(y, &format!("{p}.downsampler.resample.1"));
            }
            let shortcut = spatial || temporal || cin != cout;
            let s = if shortcut { n.g.conv2d_strided(n.w.get(&format!("{p}.avg_shortcut")), skip, 2) } else { skip };
            x = n.g.add(x, s);
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

    fn cfg() -> QwenImage21VaeConfig {
        serde_json::from_value(serde_json::json!({
            "attn_scales": [], "base_dim": 96, "decoder_base_dim": 144, "dim_mult": [1, 2, 4, 8, 8],
            "in_channels": 4, "out_channels": 4, "is_residual": true, "patch_size": null,
            "latents_mean": vec![0.0; 64], "latents_std": vec![1.0; 64], "num_res_blocks": 2,
            "scale_factor_spatial": 16, "temperal_downsample": [false, true, true, true], "z_dim": 64
        }))
        .unwrap()
    }

    #[test]
    fn the_released_layout_has_the_reference_stage_widths() {
        let c = cfg();
        c.validate().unwrap();
        assert_eq!(c.scale(), 16);
        assert_eq!(c.down_stages()[0], (96, 96, false, true));
        assert_eq!(c.down_stages()[4], (768, 768, false, false));
        assert_eq!(c.up_stages()[0], (1152, 1152, true, true));
        assert_eq!(c.up_stages()[3], (576, 288, false, true));
        assert_eq!(c.up_stages()[4], (288, 144, false, false));
    }

    #[test]
    fn the_averaging_shortcut_reads_only_the_image_frame_of_a_halved_pair() {
        let k = avg_down_kernel(2, 4, true, true).unwrap();
        // factor 8, group 4: half of each group is the zero frame.
        let total: f32 = k.iter().sum();
        assert!((total - 4.0 * 0.5).abs() < 1e-6, "{total}");
        assert!(avg_down_kernel(8, 8, false, false).is_none());
    }
}
