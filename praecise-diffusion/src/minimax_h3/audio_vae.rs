//! MiniMax-H3 audio autoencoder: a mono waveform to latents and back.
//!
//! The encoder is a strided convolutional stack: an input convolution, then
//! per stage three residual units (snake `x + sin^2(a x) / a`, a dilated
//! convolution at dilations 1, 3 and 9, snake, a pointwise convolution) and a
//! strided convolution that doubles the width, then snake and an output
//! convolution. A projection block maps the trunk to the latent width: a
//! normed linear path plus causal attention whose heads are averaged and
//! adaptively pooled to the latent width, followed by a pre-norm GeGLU MLP;
//! the posterior mean head finishes it. The decoder is a pointwise input
//! projection and an anti-aliased AMP generator (shared with the LTX-2
//! vocoder) whose output is clamped to `[-1, 1]`. Every convolution is
//! weight-normed in the checkpoint and folded at load. Everything runs in
//! f32, as the reference keeps it.
//!
//! Activations are laid out `[C, T]` (channels innermost). Latents are
//! normalised with the checkpoint's `latents_mean` / `latents_std`.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, Weights};
use crate::ltx2::vocoder::{GeneratorConfig, Hosts, Net, WeightSource};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Snake amplitude epsilon (fixed in the reference).
const SNAKE_EPS: f32 = 1e-9;
/// Layer norm epsilon of the projection block (the torch default).
const LN_EPS: f32 = 1e-5;
/// Residual unit dilations of every encoder stage (fixed in the reference).
const DILATIONS: [i64; 3] = [1, 3, 9];

/// `audio_vae/config.json` of a MiniMax-H3 checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct H3AudioVaeConfig {
    pub encoder_dim: u64,
    pub encoder_rates: Vec<u64>,
    pub latent_dim: u64,
    pub latent_channels: u64,
    pub num_attention_heads: u64,
    pub decoder_dim: u64,
    pub decoder_rates: Vec<u64>,
    pub decoder_kernel_sizes: Vec<u64>,
    pub resblock_kernel_sizes: Vec<u64>,
    pub resblock_dilation_sizes: Vec<Vec<u64>>,
    pub sampling_rate: u64,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl H3AudioVaeConfig {
    /// Refuse layouts this implementation does not compute.
    ///
    /// # Errors
    /// The first unsupported or inconsistent field.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("MiniMax-H3 audio autoencoder: {m}")));
        if self.encoder_rates.is_empty() || self.encoder_rates.contains(&0) {
            return bad("empty or zero encoder strides");
        }
        if self.decoder_rates.iter().product::<u64>() != self.hop() as u64 {
            return bad("decoder upsampling differs from the encoder hop");
        }
        let (d, l, h) = (self.latent_dim, self.latent_channels, self.num_attention_heads);
        if l == 0 || h == 0 || d % l != 0 || d % h != 0 || d / h < l {
            return bad("latent and head widths");
        }
        if self.latents_mean.len() != l as usize || self.latents_std.len() != l as usize {
            return bad("latent statistics disagree with the latent width");
        }
        self.generator().map(|_| ())
    }

    /// Waveform samples per latent frame.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.encoder_rates.iter().product::<u64>() as usize
    }

    fn generator(&self) -> Result<GeneratorConfig> {
        GeneratorConfig::amp(
            self.decoder_dim,
            self.decoder_rates.clone(),
            self.decoder_kernel_sizes.clone(),
            self.resblock_kernel_sizes.clone(),
            self.resblock_dilation_sizes.clone(),
        )
    }

    /// Encoder widths: the input convolution's, then one per stage.
    fn widths(&self) -> Vec<u64> {
        (0..=self.encoder_rates.len()).map(|i| self.encoder_dim << i).collect()
    }
}

/// The checkpoint under the vocoder's names, with weight norm folded:
/// `ups.i` is `ups.i.0`, `acts1.d` / `acts2.d` are `activations.2d` /
/// `activations.2d+1`, `act_post` is `activation_post`.
struct Source<'a>(&'a SafeTensors);

fn checkpoint_name(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut out: Vec<String> = Vec::with_capacity(parts.len() + 1);
    let mut i = 0;
    while i < parts.len() {
        let p = parts[i];
        let next = parts.get(i + 1).and_then(|n| n.parse::<usize>().ok());
        match (p, next) {
            ("ups", Some(n)) => {
                out.extend(["ups".into(), n.to_string(), "0".into()]);
                i += 2;
            }
            ("acts1" | "acts2", Some(d)) => {
                out.extend(["activations".into(), (2 * d + usize::from(p == "acts2")).to_string()]);
                i += 2;
            }
            ("act_post", _) => {
                out.push("activation_post".into());
                i += 1;
            }
            _ => {
                out.push(p.into());
                i += 1;
            }
        }
    }
    out.join(".")
}

impl WeightSource for Source<'_> {
    fn values(&self, name: &str, shape: &[u64]) -> Result<Vec<f32>> {
        let n = checkpoint_name(name);
        let Some(stem) = n.strip_suffix(".weight").filter(|s| self.0.get(&format!("{s}.weight_v")).is_some()) else {
            return Ok(self.0.require(&n, shape)?.to_f32());
        };
        let v = self.0.require(&format!("{stem}.weight_v"), shape)?.to_f32();
        let mut gs = vec![1u64; shape.len()];
        gs[0] = shape[0];
        let g = self.0.require(&format!("{stem}.weight_g"), &gs)?.to_f32();
        let row = v.len() / shape[0] as usize;
        let mut w = Vec::with_capacity(v.len());
        for (r, gr) in v.chunks_exact(row).zip(&g) {
            let norm = r.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt() as f32;
            let s = gr / norm;
            w.extend(r.iter().map(|x| x * s));
        }
        Ok(w)
    }

    fn any(&self, name: &str) -> Result<Vec<f32>> {
        let n = checkpoint_name(name);
        Ok(self.0.get(&n).ok_or_else(|| Error::MissingTensor(n.clone()))?.to_f32())
    }
}

/// A loaded MiniMax-H3 audio autoencoder.
pub struct H3AudioVae {
    backend: Backend,
    cfg: H3AudioVaeConfig,
    amp: GeneratorConfig,
    w: Weights,
    filters: HashMap<String, Vec<f32>>,
}

impl std::fmt::Debug for H3AudioVae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H3AudioVae").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl H3AudioVae {
    /// Load `audio_vae/` of a checkpoint (always f32).
    ///
    /// # Errors
    /// A missing or malformed config or weight, or no usable backend.
    pub fn load(files: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let cfg: H3AudioVaeConfig = parse(files.json("audio_vae/config.json")?, "audio autoencoder config")?;
        cfg.validate()?;
        let amp = cfg.generator()?;
        let st = SafeTensors::open(&files.weights("audio_vae")?)?;
        let src = Source(&st);
        let mut h = Hosts { st: &src, v: Vec::new(), filters: HashMap::new() };
        let widths = cfg.widths();
        h.conv("encoder.block.0", 1, widths[0], 7, true)?;
        let n = cfg.encoder_rates.len();
        for (b, &s) in cfg.encoder_rates.iter().enumerate() {
            let (half, dim) = (widths[b], widths[b + 1]);
            let p = format!("encoder.block.{}", b + 1);
            for u in 0..DILATIONS.len() {
                let q = format!("{p}.block.{u}");
                snake(&mut h, &format!("{q}.block.0"), half)?;
                h.conv(&format!("{q}.block.1"), half, half, 7, true)?;
                snake(&mut h, &format!("{q}.block.2"), half)?;
                h.conv(&format!("{q}.block.3"), half, half, 1, true)?;
            }
            snake(&mut h, &format!("{p}.block.3"), half)?;
            h.conv(&format!("{p}.block.4"), half, dim, 2 * s, true)?;
        }
        let top = widths[n];
        snake(&mut h, &format!("encoder.block.{}", n + 1), top)?;
        let d = cfg.latent_dim;
        h.conv(&format!("encoder.block.{}", n + 2), top, d, 3, true)?;

        let l = cfg.latent_channels;
        let plain = |h: &mut Hosts<'_>, name: &str, shape: &[u64]| -> Result<()> {
            let v = st.require(name, shape)?.to_f32();
            h.f32(name.into(), shape.to_vec(), v);
            Ok(())
        };
        for norm in ["norm1", "norm3"] {
            plain(&mut h, &format!("pre_block.{norm}.weight"), &[d])?;
            plain(&mut h, &format!("pre_block.{norm}.bias"), &[d])?;
        }
        for norm in ["pre_block.norm2", "pre_block.mlp.norm"] {
            plain(&mut h, &format!("{norm}.weight"), &[l])?;
            plain(&mut h, &format!("{norm}.bias"), &[l])?;
        }
        plain(&mut h, "pre_block.proj.weight", &[l, d])?;
        plain(&mut h, "pre_block.proj.bias", &[l])?;
        plain(&mut h, "pre_block.attn.qkv.weight", &[3 * d, d])?;
        let q = st.require("pre_block.attn.q_bias", &[d])?.to_f32();
        let k = st.require("pre_block.attn.zero_k_bias", &[d])?.to_f32();
        let v = st.require("pre_block.attn.v_bias", &[d])?.to_f32();
        h.f32("pre_block.attn.qkv.bias".into(), vec![3 * d], [q, k, v].concat());
        h.f32("pre_block.attn.pool".into(), vec![l, d], head_pool(d as usize, cfg.num_attention_heads as usize, l as usize));
        plain(&mut h, "pre_block.attn.proj.weight", &[l, l])?;
        plain(&mut h, "pre_block.attn.proj.bias", &[l])?;
        for (name, din, dout) in [("w0", l, 2 * l), ("w1", l, 2 * l), ("w2", 2 * l, l)] {
            plain(&mut h, &format!("pre_block.mlp.{name}.weight"), &[dout, din])?;
            plain(&mut h, &format!("pre_block.mlp.{name}.bias"), &[dout])?;
        }
        let w = st.require("mean_proj.weight", &[l, l, 1])?.to_f32();
        h.f32("mean_proj.weight".into(), vec![l, l], w);
        plain(&mut h, "mean_proj.bias", &[l])?;
        let w = st.require("dec_in_proj.weight", &[d, l, 1])?.to_f32();
        h.f32("dec_in_proj.weight".into(), vec![d, l], w);
        plain(&mut h, "dec_in_proj.bias", &[d])?;
        h.generator("decoder", &amp, d, 1)?;
        let (v, filters) = (h.v, h.filters);
        let backend = opts.backend()?;
        let w = Weights::from_host(&backend, &v)?;
        Ok(Self { backend, cfg, amp, w, filters })
    }

    /// The layout.
    #[must_use]
    pub fn config(&self) -> &H3AudioVaeConfig {
        &self.cfg
    }

    /// Bytes of weights held on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes()
    }

    /// Encode a mono waveform (zero padded to whole hops) to normalised
    /// latents `[C][T]` (the posterior mean); returns them with `T`.
    ///
    /// # Errors
    /// An empty waveform or a backend failure.
    pub fn encode(&self, wave: &[f32]) -> Result<(Vec<f32>, usize)> {
        if wave.is_empty() {
            return Err(Error::Request("empty waveform".into()));
        }
        let cfg = &self.cfg;
        let hop = cfg.hop();
        let samples = wave.len().div_ceil(hop) * hop;
        let (d, l, heads) = (cfg.latent_dim as i64, cfg.latent_channels as i64, cfg.num_attention_heads as i64);
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[1, samples as i64]);
        let t = (samples / hop) as i64;
        let mask = g.input(sys::GGML_TYPE_F16, &[t, t]);
        let mut n = Net { g: &mut g, w: &self.w, filters: &self.filters };
        let mut x = n.conv("encoder.block.0", input, 1, true);
        for (b, &s) in cfg.encoder_rates.iter().enumerate() {
            let p = format!("encoder.block.{}", b + 1);
            for (u, &dil) in DILATIONS.iter().enumerate() {
                let q = format!("{p}.block.{u}");
                let r = snake_apply(n.g, n.w, &format!("{q}.block.0"), x);
                let r = n.conv(&format!("{q}.block.1"), r, dil, true);
                let r = snake_apply(n.g, n.w, &format!("{q}.block.2"), r);
                let r = n.conv(&format!("{q}.block.3"), r, 1, true);
                x = n.g.add(x, r);
            }
            let y = snake_apply(n.g, n.w, &format!("{p}.block.3"), x);
            x = strided_conv(&mut n, &format!("{p}.block.4"), y, s as i64);
        }
        let k = cfg.encoder_rates.len();
        let y = snake_apply(n.g, n.w, &format!("encoder.block.{}", k + 1), x);
        let x = n.conv(&format!("encoder.block.{}", k + 2), y, 1, true);

        let w = &self.w;
        let g = n.g;
        let ln = |g: &mut Graph, p: &str, x: Tn| {
            let y = g.norm(x, LN_EPS);
            let y = g.mul(y, w.get(&format!("{p}.weight")));
            g.add(y, w.get(&format!("{p}.bias")))
        };
        let lin = |g: &mut Graph, p: &str, x: Tn| g.linear_b(w.get(&format!("{p}.weight")), w.get(&format!("{p}.bias")), x);
        let a = ln(g, "pre_block.norm3", x);
        let a = lin(g, "pre_block.proj", a);
        let h1 = ln(g, "pre_block.norm1", x);
        let qkv = lin(g, "pre_block.attn.qkv", h1);
        let hd = d / heads;
        let mut parts = [qkv; 3];
        for (j, o) in parts.iter_mut().enumerate() {
            let v = g.view_heads(qkv, j as i64 * d, hd, heads);
            let v = g.cont(v);
            *o = g.permute(v, [0, 2, 1, 3]);
        }
        let [q, kk, v] = parts;
        let kk = g.cont(kk);
        let v = g.cont(v);
        let o = g.attention_exact(q, kk, v, Some(mask), 1.0 / (hd as f32).sqrt());
        let o = g.reshape(o, &[d, t]);
        let o = g.linear(w.get("pre_block.attn.pool"), o);
        let o = lin(g, "pre_block.attn.proj", o);
        let hcur = g.add(a, o);
        let m = ln(g, "pre_block.norm2", hcur);
        let m = ln(g, "pre_block.mlp.norm", m);
        let m0 = lin(g, "pre_block.mlp.w0", m);
        let m0 = g.gelu_tanh_exact(m0);
        let m1 = lin(g, "pre_block.mlp.w1", m);
        let m = g.mul(m0, m1);
        let m = lin(g, "pre_block.mlp.w2", m);
        let hcur = g.add(hcur, m);
        let mean = lin(g, "mean_proj", hcur);
        g.finish(&[mean])?;
        let mut pcm = wave.to_vec();
        pcm.resize(samples, 0.0);
        g.set_f32(input, &pcm);
        let tu = t as usize;
        let mut mk = vec![0f32; tu * tu];
        for qi in 0..tu {
            for ki in qi + 1..tu {
                mk[qi * tu + ki] = f32::NEG_INFINITY;
            }
        }
        g.set_f16(mask, &mk);
        g.compute()?;
        let out = g.read_f32(mean);
        let lu = l as usize;
        let mut z = vec![0f32; lu * tu];
        for (f, row) in out.chunks_exact(lu).enumerate() {
            for (c, v) in row.iter().enumerate() {
                z[c * tu + f] = (v - cfg.latents_mean[c]) / cfg.latents_std[c];
            }
        }
        Ok((z, tu))
    }

    /// Decode normalised latents `[C][frames]` to a mono waveform of
    /// `frames x hop` samples in `[-1, 1]`.
    ///
    /// # Errors
    /// A latent size that disagrees with `frames`, or a backend failure.
    pub fn decode(&self, latent: &[f32], frames: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let l = cfg.latent_channels as usize;
        if frames == 0 || latent.len() != l * frames {
            return Err(Error::Request("audio latent size disagrees with its shape".into()));
        }
        let mut z = vec![0f32; l * frames];
        for c in 0..l {
            for f in 0..frames {
                z[f * l + c] = latent[c * frames + f] * cfg.latents_std[c] + cfg.latents_mean[c];
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[l as i64, frames as i64]);
        let x = g.linear_b(self.w.get("dec_in_proj.weight"), self.w.get("dec_in_proj.bias"), input);
        let mut n = Net { g: &mut g, w: &self.w, filters: &self.filters };
        let y = n.generator("decoder", &self.amp, x);
        let y = g.clamp(y, -1.0, 1.0);
        g.finish(&[y])?;
        g.set_f32(input, &z);
        g.compute()?;
        Ok(g.read_f32(y))
    }
}

/// A snake activation's per-channel frequency and gain.
fn snake(h: &mut Hosts<'_>, p: &str, c: u64) -> Result<()> {
    let a = h.st.values(&format!("{p}.alpha"), &[1, c, 1])?;
    h.f32(format!("{p}.gain"), vec![c], a.iter().map(|x| 1.0 / (x + SNAKE_EPS)).collect());
    h.f32(format!("{p}.freq"), vec![c], a);
    Ok(())
}

fn snake_apply(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let s = g.mul(x, w.get(&format!("{p}.freq")));
    let s = g.sin(s);
    let s = g.sqr(s);
    let s = g.mul(s, w.get(&format!("{p}.gain")));
    g.add(x, s)
}

/// Convolution of kernel `2 s`, stride `s`, padding `ceil(s / 2)`.
fn strided_conv(n: &mut Net<'_, '_>, p: &str, x: Tn, s: i64) -> Tn {
    let taps = n.w.get(&format!("{p}.taps"));
    let (cin, cout, k) = (taps.ne(0), taps.ne(1), taps.ne(2));
    let pad = (s + 1) / 2;
    let xp = n.zero_pad(x, pad, pad);
    let out = (xp.ne(1) - k) / s + 1;
    let mut y: Option<Tn> = None;
    for j in 0..k {
        let wj = n.g.view_4d(taps, [cin, cout, 1, 1], taps.nb(1), taps.nb(2), taps.nb(3), j as usize * taps.nb(2));
        let xj = n.g.view_4d(xp, [cin, out, 1, 1], xp.nb(1) * s as usize, xp.nb(2), xp.nb(3), j as usize * xp.nb(1));
        let xj = n.g.cont(xj);
        let yj = n.g.linear(wj, xj);
        y = Some(match y {
            None => yj,
            Some(acc) => n.g.add(acc, yj),
        });
    }
    let y = y.expect("at least one tap");
    n.g.add(y, n.w.get(&format!("{p}.bias")))
}

/// The head average followed by adaptive average pooling of each head's
/// `d / heads` channels down to `out`, as one `[out][d]` matrix.
fn head_pool(d: usize, heads: usize, out: usize) -> Vec<f32> {
    let hd = d / heads;
    let mut m = vec![0f32; out * d];
    for o in 0..out {
        let (a, b) = (o * hd / out, ((o + 1) * hd).div_ceil(out));
        let wgt = 1.0 / (heads * (b - a)) as f32;
        for h in 0..heads {
            for j in a..b {
                m[o * d + h * hd + j] = wgt;
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::pipeline::Precision;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn checkpoint_names() {
        assert_eq!(checkpoint_name("decoder.ups.3.weight"), "decoder.ups.3.0.weight");
        assert_eq!(checkpoint_name("decoder.resblocks.4.acts2.1.act.alpha"), "decoder.resblocks.4.activations.3.act.alpha");
        assert_eq!(checkpoint_name("decoder.act_post.upsample.filter"), "decoder.activation_post.upsample.filter");
        assert_eq!(checkpoint_name("encoder.block.1.block.0.block.1.weight"), "encoder.block.1.block.0.block.1.weight");
    }

    #[test]
    fn pooling_averages_heads_and_windows() {
        let m = head_pool(8, 2, 2);
        assert_eq!(m[..8], [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0]);
        let m = head_pool(3, 1, 2);
        assert_eq!(m, [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_audio_vae() {
        run("checkpoint");
    }

    /// The single-file release: weight norms folded into plain weights and
    /// the latent statistics stored beside them.
    #[test]
    #[ignore = "needs the reference fixtures"]
    fn minimax_h3_parity_audio_vae_single_file() {
        run("single");
    }

    fn run(checkpoint: &str) {
        let d = PathBuf::from(std::env::var("PRAECISE_MINIMAX_H3_AUDIO_VAE_PARITY").expect("PRAECISE_MINIMAX_H3_AUDIO_VAE_PARITY names the fixture dir"));
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let vae = H3AudioVae::load(&CheckpointFiles::new(d.join(checkpoint)), LoadOptions { precision: Precision::Bf16, cpu_threads: threads, device: None }).unwrap();
        let (z, t) = vae.encode(&bin(&d, "enc_in")).unwrap();
        assert_eq!(t as u64, m["cases"]["enc"]["output"].as_u64().unwrap());
        assert_close("encode", &z, &bin(&d, "enc_out"), 0.999_999, 1e-4);
        let frames = m["cases"]["dec"]["input"].as_u64().unwrap() as usize;
        let wave = vae.decode(&bin(&d, "dec_in"), frames).unwrap();
        assert_close("decode", &wave, &bin(&d, "dec_out"), 0.999_999, 1e-4);
    }
}
