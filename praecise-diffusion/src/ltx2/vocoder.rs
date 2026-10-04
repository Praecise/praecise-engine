//! LTX-2.3 vocoder: mel spectrogram to a 48 kHz waveform.
//!
//! Two generators of the same kind run in sequence. The first turns the
//! decoded mel spectrogram (both channels' bins stacked as input features)
//! into a 16 kHz waveform: an input convolution, then per stage a transposed
//! convolution that upsamples time and the mean of a few residual blocks
//! with different kernels, each a chain of dilated convolutions behind
//! anti-aliased periodic activations (2x low-pass upsampling, `x + sin^2(a
//! x) / b`, low-pass 2x downsampling). The second, bandwidth extension,
//! takes the log-mel spectrogram of that waveform (a causal windowed DFT as
//! a strided matrix product, magnitudes, a mel filter bank) and predicts a
//! 48 kHz residual, added to a windowed-sinc 3x resampling of the 16 kHz
//! waveform and clamped to `[-1, 1]`.
//!
//! Activations are laid out `[C, T]` (channels innermost); every
//! convolution is a sum of per-tap matrix products.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::single_file::{header_config, open_part, Part};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, Weights};
use crate::pipeline::LoadOptions;
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Snake amplitude epsilon (fixed in the reference).
const SNAKE_EPS: f32 = 1e-9;
/// Anti-aliasing ratio and low-pass length (fixed in the reference).
const AA_RATIO: usize = 2;
/// Log-mel floor.
const MEL_FLOOR: f32 = 1e-5;

/// One generator's configuration, from the single-file header.
#[derive(Debug, Clone, Deserialize)]
pub struct GeneratorConfig {
    /// Width after the input convolution; each stage halves it.
    pub upsample_initial_channel: u64,
    /// Time upsampling per stage.
    pub upsample_rates: Vec<u64>,
    /// Transposed convolution length per stage.
    pub upsample_kernel_sizes: Vec<u64>,
    /// Residual block kernels (one block each, averaged).
    pub resblock_kernel_sizes: Vec<u64>,
    /// Dilations per residual block.
    pub resblock_dilation_sizes: Vec<Vec<u64>>,
    #[serde(default)]
    resblock: String,
    #[serde(default)]
    activation: String,
    /// Bias on the output convolution.
    #[serde(default)]
    pub use_bias_at_final: bool,
    /// Tanh on the output.
    #[serde(default)]
    pub use_tanh_at_final: bool,
    #[serde(default = "yes")]
    apply_final_activation: bool,
}

fn yes() -> bool {
    true
}

impl GeneratorConfig {
    fn validate(&self) -> Result<()> {
        let bad = |w: String| Err(Error::Config(format!("vocoder: {w} is not supported")));
        if self.resblock != "AMP1" || self.activation != "snakebeta" {
            return bad(format!("{} blocks with {} activations", self.resblock, self.activation));
        }
        if self.upsample_rates.len() != self.upsample_kernel_sizes.len() || self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return bad("mismatched stage lists".into());
        }
        if self.upsample_rates.iter().zip(&self.upsample_kernel_sizes).any(|(s, k)| *s == 0 || k < s || !(k - s).is_multiple_of(2)) {
            return bad("an upsampling kernel shorter than its stride or of odd excess".into());
        }
        if self.resblock_kernel_sizes.iter().any(|k| k.is_multiple_of(2)) {
            return bad("an even residual kernel".into());
        }
        if self.upsample_initial_channel >> self.upsample_rates.len() == 0 {
            return bad("more stages than the width allows".into());
        }
        Ok(())
    }

    fn total_upsampling(&self) -> u64 {
        self.upsample_rates.iter().product()
    }

    fn tanh(&self) -> bool {
        self.use_tanh_at_final && self.apply_final_activation
    }
}

/// Bandwidth extension settings.
#[derive(Debug, Clone, Deserialize)]
pub struct BweConfig {
    /// The generator.
    #[serde(flatten)]
    pub generator: GeneratorConfig,
    /// STFT hop.
    pub hop_length: u64,
    /// DFT length.
    pub n_fft: u64,
    /// Window length.
    pub win_size: u64,
    /// Mel bands.
    pub num_mels: u64,
    /// Rate of the first generator's waveform.
    pub input_sampling_rate: u64,
    /// Output rate.
    pub output_sampling_rate: u64,
}

/// Vocoder configuration, from the `vocoder` section of a single-file header.
#[derive(Debug, Clone, Deserialize)]
pub struct VocoderConfig {
    /// The mel-to-waveform generator.
    pub vocoder: GeneratorConfig,
    /// The bandwidth extension.
    pub bwe: BweConfig,
}

impl VocoderConfig {
    /// Parse the `vocoder` section of a single-file header.
    ///
    /// # Errors
    /// A layout the native vocoder does not implement.
    pub fn from_single_file(section: &Value) -> Result<Self> {
        let c: Self = serde_json::from_value(section.clone()).map_err(|e| Error::Config(format!("vocoder config: {e}")))?;
        c.vocoder.validate()?;
        c.bwe.generator.validate()?;
        let b = &c.bwe;
        if b.win_size != b.n_fft || b.hop_length == 0 || b.hop_length > b.n_fft {
            return Err(Error::Config("vocoder: a window unequal to the DFT length is not supported".into()));
        }
        if b.input_sampling_rate == 0 || !b.output_sampling_rate.is_multiple_of(b.input_sampling_rate) {
            return Err(Error::Config("vocoder: a non-integer resampling ratio is not supported".into()));
        }
        if b.generator.total_upsampling() != b.hop_length * b.output_sampling_rate / b.input_sampling_rate {
            return Err(Error::Config("vocoder: bandwidth extension length disagrees with its hop".into()));
        }
        Ok(c)
    }

    /// Resampling ratio of the bandwidth extension.
    #[must_use]
    pub fn ratio(&self) -> u64 {
        self.bwe.output_sampling_rate / self.bwe.input_sampling_rate
    }

    /// Output samples for `frames` mel frames.
    #[must_use]
    pub fn samples(&self, frames: usize) -> usize {
        frames * (self.vocoder.total_upsampling() * self.ratio()) as usize
    }
}

/// The 3x resampler's windowed-sinc filter, gain included.
fn hann_sinc(ratio: usize) -> (Vec<f32>, usize, usize) {
    let rolloff = 0.99f64;
    let lowpass_width = 6.0f64;
    let width = (lowpass_width / rolloff).ceil() as usize;
    let k = 2 * width * ratio + 1;
    let r = ratio as f64;
    let f = (0..k)
        .map(|i| {
            let t = (i as f64 / r - width as f64) * rolloff;
            let tc = t.clamp(-lowpass_width, lowpass_width);
            let window = (tc * std::f64::consts::PI / lowpass_width / 2.0).cos().powi(2);
            let sinc = if t == 0.0 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
            (sinc * window * rolloff / r * r) as f32
        })
        .collect();
    (f, width, 2 * width * ratio)
}

struct Hosts<'a> {
    st: &'a SafeTensors,
    v: Vec<HostTensor>,
    filters: HashMap<String, Vec<f32>>,
}

impl Hosts<'_> {
    fn f32(&mut self, name: String, shape: Vec<u64>, data: Vec<f32>) {
        self.v.push(HostTensor { name, shape, ty: WType::F32, data });
    }

    /// Kernel `[cout, cin, k]` as per-tap matrices `[k, cout, cin]`.
    fn conv(&mut self, p: &str, cin: u64, cout: u64, k: u64, bias: bool) -> Result<()> {
        let w = self.st.require(&format!("{p}.weight"), &[cout, cin, k])?.to_f32();
        let (ci, co, kk) = (cin as usize, cout as usize, k as usize);
        let mut taps = vec![0f32; w.len()];
        for o in 0..co {
            for i in 0..ci {
                for j in 0..kk {
                    taps[(j * co + o) * ci + i] = w[(o * ci + i) * kk + j];
                }
            }
        }
        self.v.push(HostTensor { name: format!("{p}.taps"), shape: vec![k, cout, cin], ty: WType::F32, data: taps });
        if bias {
            let b = self.st.require(&format!("{p}.bias"), &[cout])?.to_f32();
            self.f32(format!("{p}.bias"), vec![cout], b);
        }
        Ok(())
    }

    /// Transposed kernel `[cin, cout, k]` of stride `s` as `ceil(k / s)`
    /// matrices `[s cout, cin]`: block `b`, row `r cout + o` is tap `s b + r`.
    fn conv_t(&mut self, p: &str, cin: u64, cout: u64, k: u64, s: u64) -> Result<()> {
        let w = self.st.require(&format!("{p}.weight"), &[cin, cout, k])?.to_f32();
        let (ci, co, kk, ss) = (cin as usize, cout as usize, k as usize, s as usize);
        let nb = kk.div_ceil(ss);
        let mut m = vec![0f32; nb * ss * co * ci];
        for b in 0..nb {
            for r in 0..ss {
                let j = ss * b + r;
                if j >= kk {
                    continue;
                }
                for o in 0..co {
                    for i in 0..ci {
                        m[((b * ss + r) * co + o) * ci + i] = w[(i * co + o) * kk + j];
                    }
                }
            }
        }
        self.v.push(HostTensor { name: format!("{p}.blocks"), shape: vec![nb as u64, s * cout, cin], ty: WType::F32, data: m });
        let b = self.st.require(&format!("{p}.bias"), &[cout])?.to_f32();
        self.f32(format!("{p}.bias"), vec![cout], b);
        Ok(())
    }

    /// An anti-aliased snake activation: per-channel frequency and gain, and
    /// the two low-pass filters (kept on the host as constants).
    fn act(&mut self, p: &str, c: u64) -> Result<()> {
        let a = self.st.require(&format!("{p}.act.alpha"), &[c])?.to_f32();
        let b = self.st.require(&format!("{p}.act.beta"), &[c])?.to_f32();
        self.f32(format!("{p}.freq"), vec![c], a.iter().map(|x| x.exp()).collect());
        self.f32(format!("{p}.gain"), vec![c], b.iter().map(|x| 1.0 / (x.exp() + SNAKE_EPS)).collect());
        for (key, file) in [("up", "upsample.filter"), ("down", "downsample.lowpass.filter")] {
            let t = self.st.get(&format!("{p}.{file}")).ok_or_else(|| Error::MissingTensor(format!("{p}.{file}")))?;
            let mut f = t.to_f32();
            if key == "up" {
                for x in &mut f {
                    *x *= AA_RATIO as f32;
                }
            }
            self.filters.insert(format!("{p}.{key}"), f);
        }
        Ok(())
    }

    fn generator(&mut self, p: &str, cfg: &GeneratorConfig, cin: u64, cout: u64) -> Result<()> {
        let mut c = cfg.upsample_initial_channel;
        self.conv(&format!("{p}.conv_pre"), cin, c, 7, true)?;
        let nr = cfg.resblock_kernel_sizes.len();
        for (i, (s, k)) in cfg.upsample_rates.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            self.conv_t(&format!("{p}.ups.{i}"), c, c / 2, *k, *s)?;
            c /= 2;
            for (j, (rk, dil)) in cfg.resblock_kernel_sizes.iter().zip(&cfg.resblock_dilation_sizes).enumerate() {
                let r = format!("{p}.resblocks.{}", i * nr + j);
                for d in 0..dil.len() {
                    self.act(&format!("{r}.acts1.{d}"), c)?;
                    self.conv(&format!("{r}.convs1.{d}"), c, c, *rk, true)?;
                    self.act(&format!("{r}.acts2.{d}"), c)?;
                    self.conv(&format!("{r}.convs2.{d}"), c, c, *rk, true)?;
                }
            }
        }
        self.act(&format!("{p}.act_post"), c)?;
        self.conv(&format!("{p}.conv_post"), c, cout, 7, cfg.use_bias_at_final)
    }
}

struct Net<'a, 'g> {
    g: &'g mut Graph,
    w: &'a Weights,
    filters: &'a HashMap<String, Vec<f32>>,
}

impl Net<'_, '_> {
    /// `[C, T]` with `left` and `right` zero columns.
    fn zero_pad(&mut self, x: Tn, left: i64, right: i64) -> Tn {
        if left + right == 0 {
            return x;
        }
        let y = self.g.pad_end(x, 0, (left + right) as i32);
        if left == 0 { y } else { self.g.roll(y, 0, left as i32) }
    }

    /// `[C, T]` with its edge columns repeated `left` and `right` times.
    fn edge_pad(&mut self, x: Tn, left: i64, right: i64) -> Tn {
        let (c, t) = (x.ne(0), x.ne(1));
        let mut y = x;
        if left > 0 {
            let first = self.g.view_cols(x, 0, 1);
            let first = self.g.repeat_to(first, [c, left, 1, 1]);
            y = self.g.concat(first, y, 1);
        }
        if right > 0 {
            let last = self.g.view_cols(x, t - 1, 1);
            let last = self.g.repeat_to(last, [c, right, 1, 1]);
            y = self.g.concat(y, last, 1);
        }
        y
    }

    fn conv(&mut self, p: &str, x: Tn, dil: i64, bias: bool) -> Tn {
        let taps = self.w.get(&format!("{p}.taps"));
        let (cin, cout, k) = (taps.ne(0), taps.ne(1), taps.ne(2));
        let t = x.ne(1);
        let pad = dil * (k - 1) / 2;
        let xp = self.zero_pad(x, pad, pad);
        let mut y: Option<Tn> = None;
        for j in 0..k {
            let wj = self.g.view_4d(taps, [cin, cout, 1, 1], taps.nb(1), taps.nb(2), taps.nb(3), j as usize * taps.nb(2));
            let xj = self.g.view_cols(xp, j * dil, t);
            let yj = self.g.linear(wj, xj);
            y = Some(match y {
                None => yj,
                Some(acc) => self.g.add(acc, yj),
            });
        }
        let y = y.expect("at least one tap");
        if bias { self.g.add(y, self.w.get(&format!("{p}.bias"))) } else { y }
    }

    /// Transposed convolution of stride `s`, kernel `k`, padding `(k - s) /
    /// 2`: `t` columns to `s t`.
    fn conv_t(&mut self, p: &str, x: Tn, s: i64, k: i64) -> Tn {
        let m = self.w.get(&format!("{p}.blocks"));
        let (cin, rows, nb) = (m.ne(0), m.ne(1), m.ne(2));
        let cout = rows / s;
        let t = x.ne(1);
        let mut full: Option<Tn> = None;
        for b in 0..nb {
            let mb = self.g.view_4d(m, [cin, rows, 1, 1], m.nb(1), m.nb(2), m.nb(3), b as usize * m.nb(2));
            let a = self.g.linear(mb, x);
            let a = self.zero_pad(a, b, nb - 1 - b);
            full = Some(match full {
                None => a,
                Some(acc) => self.g.add(acc, a),
            });
        }
        let full = full.expect("at least one block");
        let full = self.g.reshape(full, &[cout, s * (t + nb - 1)]);
        let y = self.g.view_cols(full, (k - s) / 2, s * t);
        let y = self.g.cont(y);
        self.g.add(y, self.w.get(&format!("{p}.bias")))
    }

    /// Depthwise upsampling by `ratio` with filter `f` (gain included): the
    /// input edge-padded by `pad` columns, transposed-convolved, cropped from
    /// `crop` to `ratio t` columns.
    fn upsample(&mut self, x: Tn, f: &[f32], ratio: i64, pad: i64, crop: i64) -> Tn {
        let (c, t) = (x.ne(0), x.ne(1));
        let nb = (f.len() as i64 + ratio - 1) / ratio;
        let xp = self.edge_pad(x, pad, pad);
        let z = self.zero_pad(xp, nb - 1, nb - 1);
        let q = t + 2 * pad + nb - 1;
        let mut phases: Option<Tn> = None;
        for r in 0..ratio {
            let mut acc: Option<Tn> = None;
            for b in 0..nb {
                let Some(&coef) = f.get((ratio * b + r) as usize) else { continue };
                let v = self.g.view_cols(z, nb - 1 - b, q);
                let v = self.g.cont(v);
                let v = self.g.scale_bias(v, coef, 0.0);
                acc = Some(match acc {
                    None => v,
                    Some(a) => self.g.add(a, v),
                });
            }
            let phase = self.g.reshape(acc.expect("a tap per phase"), &[c, 1, q]);
            phases = Some(match phases {
                None => phase,
                Some(p) => self.g.concat(p, phase, 1),
            });
        }
        let full = self.g.reshape(phases.expect("ratio > 0"), &[c, ratio * q]);
        let y = self.g.view_cols(full, crop, ratio * t);
        self.g.cont(y)
    }

    /// Depthwise low-pass and 2x decimation with edge padding.
    fn downsample(&mut self, x: Tn, f: &[f32]) -> Tn {
        let (c, t) = (x.ne(0), x.ne(1));
        let k = f.len() as i64;
        let r = AA_RATIO as i64;
        let (left, right) = (k / 2 + k % 2 - 1, k / 2);
        let xp = self.edge_pad(x, left, right);
        let m = t + left + right;
        let n_out = (m - k) / r + 1;
        let mr = (m + r - 1) / r * r;
        let xp = self.zero_pad(xp, 0, mr - m);
        let planes = self.g.reshape(xp, &[c, r, mr / r]);
        let mut acc: Option<Tn> = None;
        for (j, &coef) in f.iter().enumerate() {
            let j = j as i64;
            let off = (j % r) as usize * planes.nb(1) + (j / r) as usize * planes.nb(2);
            let v = self.g.view_4d(planes, [c, n_out, 1, 1], planes.nb(2), planes.nb(2) * n_out as usize, planes.nb(2) * n_out as usize, off);
            let v = self.g.cont(v);
            let v = self.g.scale_bias(v, coef, 0.0);
            acc = Some(match acc {
                None => v,
                Some(a) => self.g.add(a, v),
            });
        }
        acc.expect("a filter tap")
    }

    fn act(&mut self, p: &str, x: Tn) -> Tn {
        let fl = self.filters;
        let up = &fl[&format!("{p}.up")];
        let k = up.len() as i64;
        let r = AA_RATIO as i64;
        let pad = k / r - 1;
        let crop = pad * r + (k - r) / 2;
        let x = self.upsample(x, up, r, pad, crop);
        let s = self.g.mul(x, self.w.get(&format!("{p}.freq")));
        let s = self.g.sin(s);
        let s = self.g.sqr(s);
        let s = self.g.mul(s, self.w.get(&format!("{p}.gain")));
        let x = self.g.add(x, s);
        self.downsample(x, &fl[&format!("{p}.down")])
    }

    fn generator(&mut self, p: &str, cfg: &GeneratorConfig, x: Tn) -> Tn {
        let mut x = self.conv(&format!("{p}.conv_pre"), x, 1, true);
        let nr = cfg.resblock_kernel_sizes.len();
        for (i, (s, k)) in cfg.upsample_rates.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            x = self.conv_t(&format!("{p}.ups.{i}"), x, *s as i64, *k as i64);
            let mut sum: Option<Tn> = None;
            for (j, dil) in cfg.resblock_dilation_sizes.iter().enumerate() {
                let r = format!("{p}.resblocks.{}", i * nr + j);
                let mut h = x;
                for (d, &dv) in dil.iter().enumerate() {
                    let a = self.act(&format!("{r}.acts1.{d}"), h);
                    let a = self.conv(&format!("{r}.convs1.{d}"), a, dv as i64, true);
                    let a = self.act(&format!("{r}.acts2.{d}"), a);
                    let a = self.conv(&format!("{r}.convs2.{d}"), a, 1, true);
                    h = self.g.add(h, a);
                }
                sum = Some(match sum {
                    None => h,
                    Some(acc) => self.g.add(acc, h),
                });
            }
            x = self.g.scale_bias(sum.expect("a residual block"), 1.0 / nr as f32, 0.0);
        }
        x = self.act(&format!("{p}.act_post"), x);
        let y = self.conv(&format!("{p}.conv_post"), x, 1, cfg.use_bias_at_final);
        if cfg.tanh() { self.g.tanh(y) } else { y }
    }

    /// Causal log-mel spectrogram `[mels, n / hop]` of one channel `[n, 1]`.
    fn log_mel(&mut self, wave: Tn, hop: i64, blocks: i64) -> Tn {
        let n = wave.ne(0);
        let frames = n / hop;
        let left = hop * (blocks - 1);
        let y = self.g.pad_end(wave, left as i32, 0);
        let y = self.g.roll(y, left as i32, 0);
        let cols = self.g.reshape(y, &[hop, frames + blocks - 1]);
        let mut spec: Option<Tn> = None;
        for j in 0..blocks {
            let b = self.w.get(&format!("mel_stft.basis.{j}"));
            let v = self.g.view_cols(cols, j, frames);
            let s = self.g.linear(b, v);
            spec = Some(match spec {
                None => s,
                Some(a) => self.g.add(a, s),
            });
        }
        let spec = spec.expect("a basis block");
        let nf = spec.ne(0) / 2;
        let re = self.g.view_rows(spec, 0, nf);
        let re = self.g.cont(re);
        let im = self.g.view_rows(spec, nf, nf);
        let im = self.g.cont(im);
        let re = self.g.sqr(re);
        let im = self.g.sqr(im);
        let mag = self.g.add(re, im);
        let mag = self.g.sqrt(mag);
        let mel = self.g.linear(self.w.get("mel_stft.mel_basis"), mag);
        let mel = self.g.clamp(mel, MEL_FLOOR, f32::MAX);
        self.g.log(mel)
    }
}

/// The vocoder with its resident weights.
pub struct Ltx2Vocoder {
    backend: Backend,
    cfg: VocoderConfig,
    w: Weights,
    filters: HashMap<String, Vec<f32>>,
    /// Input features (channels times mel bins).
    in_features: u64,
    /// Output channels.
    channels: u64,
    /// Basis blocks of the STFT.
    blocks: u64,
}

impl std::fmt::Debug for Ltx2Vocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Vocoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl Ltx2Vocoder {
    /// Load the vocoder of a single-file checkpoint.
    ///
    /// # Errors
    /// On an unsupported layout, missing weights or no usable backend.
    pub fn load_single_file(path: &Path, opts: LoadOptions) -> Result<Self> {
        let st = SafeTensors::open(&[path.to_path_buf()])?;
        let header = header_config(&st)?;
        let cfg = VocoderConfig::from_single_file(&header["vocoder"])?;
        let st = open_part(st, Part::Vocoder)?;
        let shape = |n: &str| st.get(n).map(|t| t.shape.to_vec()).ok_or_else(|| Error::MissingTensor(n.into()));
        let in_features = shape("vocoder.conv_pre.weight")?[1];
        let channels = shape("vocoder.conv_post.weight")?[0];
        let bwe_in = shape("bwe_generator.conv_pre.weight")?[1];
        let b = &cfg.bwe;
        if bwe_in != channels * b.num_mels || shape("bwe_generator.conv_post.weight")?[0] != channels {
            return Err(Error::Config("vocoder: bandwidth extension widths disagree with the channels".into()));
        }
        // The vocoder is small and its output feeds a log and a second
        // generator, where reduced-precision products drift audibly: it runs
        // in f32 at every precision (lossless for the bf16 file).
        let mut h = Hosts { st: &st, v: Vec::new(), filters: HashMap::new() };
        h.generator("vocoder", &cfg.vocoder, in_features, channels)?;
        h.generator("bwe_generator", &b.generator, bwe_in, channels)?;
        // DFT basis blocks: the window shifted right so the causal left
        // padding is a whole number of hops.
        let (l, hop) = (b.n_fft as usize, b.hop_length as usize);
        let nf2 = 2 * (l / 2 + 1);
        let basis = st.require("mel_stft.stft_fn.forward_basis", &[nf2 as u64, 1, l as u64])?.to_f32();
        let blocks = l.div_ceil(hop);
        let shift = blocks * hop - l;
        for j in 0..blocks {
            let mut m = vec![0f32; nf2 * hop];
            for k in 0..nf2 {
                for i in 0..hop {
                    let col = j * hop + i;
                    if col >= shift {
                        m[k * hop + i] = basis[k * l + col - shift];
                    }
                }
            }
            h.f32(format!("mel_stft.basis.{j}"), vec![nf2 as u64, hop as u64], m);
        }
        let mel = st.require("mel_stft.mel_basis", &[b.num_mels, (l / 2 + 1) as u64])?.to_f32();
        h.f32("mel_stft.mel_basis".into(), vec![b.num_mels, (l / 2 + 1) as u64], mel);
        let (v, filters) = (h.v, h.filters);
        let backend = opts.backend()?;
        let w = Weights::from_host(&backend, &v)?;
        Ok(Self { backend, cfg, w, filters, in_features, channels, blocks: blocks as u64 })
    }

    /// The vocoder configuration.
    #[must_use]
    pub fn config(&self) -> &VocoderConfig {
        &self.cfg
    }

    /// Output sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u64 {
        self.cfg.bwe.output_sampling_rate
    }

    /// A mel spectrogram `[channels][frames][mel bins]` (the audio decoder's
    /// output) to a waveform `[channels][samples]`.
    ///
    /// # Errors
    /// A spectrogram size that disagrees with the frame count, or a backend
    /// failure.
    pub fn synthesize(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let feats = self.in_features as usize;
        let ch = self.channels as usize;
        if frames == 0 || !feats.is_multiple_of(ch) || mel.len() != frames * feats {
            return Err(Error::Request("mel spectrogram size disagrees with its frame count".into()));
        }
        let bins = feats / ch;
        // [C][T][M] to the graph's [T][C * M].
        let mut x = vec![0f32; mel.len()];
        for c in 0..ch {
            for t in 0..frames {
                for m in 0..bins {
                    x[t * feats + c * bins + m] = mel[(c * frames + t) * bins + m];
                }
            }
        }
        let b = &self.cfg.bwe;
        let hop = b.hop_length as i64;
        let ratio = self.cfg.ratio() as i64;
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[feats as i64, frames as i64]);
        let mut n = Net { g: &mut g, w: &self.w, filters: &self.filters };
        let wave = n.generator("vocoder", &self.cfg.vocoder, input);
        let samples = wave.ne(1);
        let padded = (samples + hop - 1) / hop * hop;
        let wave_p = n.zero_pad(wave, 0, padded - samples);
        let by_channel = n.g.permute(wave_p, [1, 0, 2, 3]);
        let by_channel = n.g.cont(by_channel);
        let mut feats_bwe: Option<Tn> = None;
        for c in 0..ch as i64 {
            let one = n.g.view_cols(by_channel, c, 1);
            let one = n.g.cont(one);
            let lm = n.log_mel(one, hop, self.blocks as i64);
            feats_bwe = Some(match feats_bwe {
                None => lm,
                Some(a) => n.g.concat(a, lm, 0),
            });
        }
        let residual = n.generator("bwe_generator", &b.generator, feats_bwe.expect("a channel"));
        let (f, width, crop) = hann_sinc(ratio as usize);
        let skip = n.upsample(wave_p, &f, ratio, width as i64, crop as i64);
        let sum = n.g.add(residual, skip);
        let out = n.g.clamp(sum, -1.0, 1.0);
        let keep = samples * ratio;
        let out = n.g.view_cols(out, 0, keep);
        let out = n.g.cont(out);
        g.finish(&[out])?;
        g.set_f32(input, &x);
        g.compute()?;
        let y = g.read_f32(out);
        let keep = keep as usize;
        let mut wav = vec![0f32; y.len()];
        for (t, row) in y.chunks_exact(ch).enumerate() {
            for (c, v) in row.iter().enumerate() {
                wav[c * keep + t] = *v;
            }
        }
        Ok(wav)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_layout() {
        let v = serde_json::json!({"vocoder": {"upsample_initial_channel": 1536, "resblock": "AMP1", "upsample_rates": [5, 2, 2, 2, 2, 2],
            "resblock_kernel_sizes": [3, 7, 11], "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "stereo": true, "use_tanh_at_final": false,
            "activation": "snakebeta", "use_bias_at_final": false},
            "bwe": {"upsample_initial_channel": 512, "resblock": "AMP1", "upsample_rates": [6, 5, 2, 2, 2],
            "resblock_kernel_sizes": [3, 7, 11], "upsample_kernel_sizes": [12, 11, 4, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "stereo": true, "use_tanh_at_final": false,
            "activation": "snakebeta", "use_bias_at_final": false, "apply_final_activation": false,
            "input_sampling_rate": 16000, "output_sampling_rate": 48000, "hop_length": 80, "n_fft": 512, "win_size": 512, "num_mels": 64}});
        let c = VocoderConfig::from_single_file(&v).unwrap();
        assert_eq!(c.ratio(), 3);
        assert_eq!(c.samples(501), 501 * 480);
        let (f, width, crop) = hann_sinc(3);
        assert_eq!((f.len(), width, crop), (43, 7, 42));
        assert!((f[21] - 0.99).abs() < 1e-6);
    }
}
