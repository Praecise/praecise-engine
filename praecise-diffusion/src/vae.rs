//! The FLUX.2 autoencoder's decoder: latents back to pixels.
//!
//! A KL-autoencoder decoder (residual blocks with group norm, one
//! self-attention block in the middle, nearest-neighbour upsampling) behind a
//! 1x1 post-quantisation convolution. The latent normalisation (per-channel
//! mean and variance over 2x2-patched latents) is undone on the host before the
//! graph runs.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Graph, Tn, WType, WeightSpec, Weights};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Autoencoder configuration, read from the VAE's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    /// Image channels in.
    pub in_channels: u64,
    /// Channel widths per resolution level, shallowest first.
    pub block_out_channels: Vec<u64>,
    /// Latent channels.
    pub latent_channels: u64,
    /// Residual blocks per encoder level; the decoder has one more.
    pub layers_per_block: usize,
    /// Group-norm groups.
    pub norm_num_groups: u64,
    /// Output channels.
    pub out_channels: u64,
    /// Latent patch size `[h, w]`.
    pub patch_size: Vec<u64>,
    /// Batch-norm epsilon for the latent statistics.
    pub batch_norm_eps: f64,
    /// Whether a 1x1 convolution precedes the decoder.
    pub use_post_quant_conv: bool,
    /// Whether the middle block has attention.
    pub mid_block_add_attention: bool,
}

const GN_EPS: f32 = 1e-6;

/// Queries per score block in the middle attention.
const ATTN_QUERY_BLOCK: i64 = 4096;

impl VaeConfig {
    /// Validate the parts this implementation depends on.
    ///
    /// # Errors
    /// [`Error::Config`] for an unsupported layout.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size != [2, 2] {
            return Err(Error::Config(format!("latent patch size {:?} is not implemented", self.patch_size)));
        }
        if !self.use_post_quant_conv || !self.mid_block_add_attention {
            return Err(Error::Config("expected a post-quantisation convolution and mid-block attention".into()));
        }
        if self.block_out_channels.is_empty() {
            return Err(Error::Config("no decoder levels".into()));
        }
        Ok(())
    }

    /// Pixel size of one latent cell before patching.
    #[must_use]
    pub fn scale_factor(&self) -> u64 {
        1 << (self.block_out_channels.len() - 1)
    }

    fn decoder_channels(&self) -> Vec<u64> {
        self.block_out_channels.iter().rev().copied().collect()
    }

    /// Every decoder weight. Convolution kernels are f16 (the convolution
    /// runs as an implicit matrix product on tensor cores), everything else
    /// f32.
    #[must_use]
    pub fn weight_specs(&self) -> Vec<WeightSpec> {
        let lc = self.latent_channels;
        let ch = self.decoder_channels();
        let top = ch[0];
        let mut v = vec![
            WeightSpec::new("post_quant_conv.weight", &[lc, lc, 1, 1], WType::F16),
            WeightSpec::new("post_quant_conv.bias", &[lc], WType::F32),
            WeightSpec::new("decoder.conv_in.weight", &[top, lc, 3, 3], WType::F16),
            WeightSpec::new("decoder.conv_in.bias", &[top], WType::F32),
        ];
        mid_specs(&mut v, "decoder", top);
        let mut prev = top;
        for (i, &c) in ch.iter().enumerate() {
            for r in 0..=self.layers_per_block {
                let cin = if r == 0 { prev } else { c };
                resnet_specs(&mut v, &format!("decoder.up_blocks.{i}.resnets.{r}"), cin, c);
            }
            if i + 1 < ch.len() {
                v.push(WeightSpec::new(format!("decoder.up_blocks.{i}.upsamplers.0.conv.weight"), &[c, c, 3, 3], WType::F16));
                v.push(WeightSpec::new(format!("decoder.up_blocks.{i}.upsamplers.0.conv.bias"), &[c], WType::F32));
            }
            prev = c;
        }
        let last = *ch.last().expect("validated non-empty");
        v.push(WeightSpec::new("decoder.conv_norm_out.weight", &[last], WType::F32));
        v.push(WeightSpec::new("decoder.conv_norm_out.bias", &[last], WType::F32));
        v.push(WeightSpec::new("decoder.conv_out.weight", &[self.out_channels, last, 3, 3], WType::F16));
        v.push(WeightSpec::new("decoder.conv_out.bias", &[self.out_channels], WType::F32));
        v
    }
}

impl VaeConfig {
    /// Every encoder weight (used for reference images).
    #[must_use]
    pub fn encoder_weight_specs(&self) -> Vec<WeightSpec> {
        let lc = self.latent_channels;
        let ch = &self.block_out_channels;
        let first = ch[0];
        let top = *ch.last().expect("validated non-empty");
        let mut v = vec![
            WeightSpec::new("encoder.conv_in.weight", &[first, self.in_channels, 3, 3], WType::F16),
            WeightSpec::new("encoder.conv_in.bias", &[first], WType::F32),
        ];
        let mut prev = first;
        for (i, &c) in ch.iter().enumerate() {
            for r in 0..self.layers_per_block {
                let cin = if r == 0 { prev } else { c };
                resnet_specs(&mut v, &format!("encoder.down_blocks.{i}.resnets.{r}"), cin, c);
            }
            if i + 1 < ch.len() {
                v.push(WeightSpec::new(format!("encoder.down_blocks.{i}.downsamplers.0.conv.weight"), &[c, c, 3, 3], WType::F16));
                v.push(WeightSpec::new(format!("encoder.down_blocks.{i}.downsamplers.0.conv.bias"), &[c], WType::F32));
            }
            prev = c;
        }
        mid_specs(&mut v, "encoder", top);
        v.push(WeightSpec::new("encoder.conv_norm_out.weight", &[top], WType::F32));
        v.push(WeightSpec::new("encoder.conv_norm_out.bias", &[top], WType::F32));
        v.push(WeightSpec::new("encoder.conv_out.weight", &[2 * lc, top, 3, 3], WType::F16));
        v.push(WeightSpec::new("encoder.conv_out.bias", &[2 * lc], WType::F32));
        v.push(WeightSpec::new("quant_conv.weight", &[2 * lc, 2 * lc, 1, 1], WType::F16));
        v.push(WeightSpec::new("quant_conv.bias", &[2 * lc], WType::F32));
        v
    }
}

fn mid_specs(v: &mut Vec<WeightSpec>, side: &str, c: u64) {
    for r in 0..2 {
        resnet_specs(v, &format!("{side}.mid_block.resnets.{r}"), c, c);
    }
    let a = format!("{side}.mid_block.attentions.0");
    v.push(WeightSpec::new(format!("{a}.group_norm.weight"), &[c], WType::F32));
    v.push(WeightSpec::new(format!("{a}.group_norm.bias"), &[c], WType::F32));
    for n in ["to_q", "to_k", "to_v", "to_out.0"] {
        v.push(WeightSpec::new(format!("{a}.{n}.weight"), &[c, c], WType::F32));
        v.push(WeightSpec::new(format!("{a}.{n}.bias"), &[c], WType::F32));
    }
}

fn resnet_specs(v: &mut Vec<WeightSpec>, p: &str, cin: u64, cout: u64) {
    v.push(WeightSpec::new(format!("{p}.norm1.weight"), &[cin], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm1.bias"), &[cin], WType::F32));
    v.push(WeightSpec::new(format!("{p}.conv1.weight"), &[cout, cin, 3, 3], WType::F16));
    v.push(WeightSpec::new(format!("{p}.conv1.bias"), &[cout], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm2.weight"), &[cout], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm2.bias"), &[cout], WType::F32));
    v.push(WeightSpec::new(format!("{p}.conv2.weight"), &[cout, cout, 3, 3], WType::F16));
    v.push(WeightSpec::new(format!("{p}.conv2.bias"), &[cout], WType::F32));
    if cin != cout {
        v.push(WeightSpec::new(format!("{p}.conv_shortcut.weight"), &[cout, cin, 1, 1], WType::F16));
        v.push(WeightSpec::new(format!("{p}.conv_shortcut.bias"), &[cout], WType::F32));
    }
}

/// Per-channel latent statistics `(mean, std)` over the patched channels.
///
/// # Errors
/// When the batch-norm statistics are missing.
pub fn latent_stats(files: &SafeTensors, cfg: &VaeConfig) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = cfg.latent_channels * cfg.patch_size[0] * cfg.patch_size[1];
    let mean = files.require("bn.running_mean", &[n])?.to_f32();
    let var = files.require("bn.running_var", &[n])?.to_f32();
    let std = var.iter().map(|v| (f64::from(*v) + cfg.batch_norm_eps).sqrt() as f32).collect();
    Ok((mean, std))
}

fn bias4(g: &mut Graph, b: Tn) -> Tn {
    let c = b.ne(0);
    g.reshape(b, &[1, 1, c, 1])
}

fn conv(g: &mut Graph, w: &Weights, p: &str, x: Tn, pad: i32) -> Tn {
    let y = g.conv2d(w.get(&format!("{p}.weight")), x, pad);
    let b = bias4(g, w.get(&format!("{p}.bias")));
    g.add(y, b)
}

fn gn(g: &mut Graph, w: &Weights, p: &str, x: Tn, groups: i32) -> Tn {
    // Scale and shift are placed first, so the norm, scale, shift (and a
    // following SiLU) are adjacent in the graph and run as one fused kernel.
    let s = bias4(g, w.get(&format!("{p}.weight")));
    let b = bias4(g, w.get(&format!("{p}.bias")));
    g.expand(s);
    g.expand(b);
    g.expand(x);
    let y = g.group_norm(x, groups, GN_EPS);
    let y = g.mul(y, s);
    g.add(y, b)
}

fn resnet(g: &mut Graph, w: &Weights, p: &str, x: Tn, groups: i32) -> Tn {
    let h = gn(g, w, &format!("{p}.norm1"), x, groups);
    let h = g.silu(h);
    let h = conv(g, w, &format!("{p}.conv1"), h, 1);
    let h = gn(g, w, &format!("{p}.norm2"), h, groups);
    let h = g.silu(h);
    let h = conv(g, w, &format!("{p}.conv2"), h, 1);
    let skip = if x.ne(2) == h.ne(2) { x } else { conv(g, w, &format!("{p}.conv_shortcut"), x, 0) };
    g.add(skip, h)
}

fn mid_attention(g: &mut Graph, w: &Weights, a: &str, x: Tn, groups: i32) -> Tn {
    let (wd, ht, c) = (x.ne(0), x.ne(1), x.ne(2));
    let h = gn(g, w, &format!("{a}.group_norm"), x, groups);
    // [W, H, C] -> tokens [C, W*H]
    let h = g.reshape(h, &[wd * ht, c]);
    let h = g.permute(h, [1, 0, 2, 3]);
    let h = g.cont(h);
    let q = g.linear_b(w.get(&format!("{a}.to_q.weight")), w.get(&format!("{a}.to_q.bias")), h);
    let k = g.linear_b(w.get(&format!("{a}.to_k.weight")), w.get(&format!("{a}.to_k.bias")), h);
    let v = g.linear_b(w.get(&format!("{a}.to_v.weight")), w.get(&format!("{a}.to_v.bias")), h);
    // One head as wide as the channel count; fused attention kernels do not
    // cover that width, so scores are materialised a block of queries at a
    // time to bound the score matrix.
    let n = wd * ht;
    let vt = g.permute(v, [1, 0, 2, 3]);
    let vt = g.cont(vt);
    let scale = 1.0 / (c as f32).sqrt();
    let mut o: Option<Tn> = None;
    let mut from = 0;
    while from < n {
        let m = ATTN_QUERY_BLOCK.min(n - from);
        let qb = g.view_cols(q, from, m);
        let s = g.linear(k, qb);
        let p = g.soft_max(s, scale);
        let ob = g.linear(vt, p);
        o = Some(match o {
            None => ob,
            Some(prev) => g.concat(prev, ob, 1),
        });
        from += m;
    }
    let o = o.expect("at least one latent cell");
    let o = g.linear_b(w.get(&format!("{a}.to_out.0.weight")), w.get(&format!("{a}.to_out.0.bias")), o);
    // tokens [C, W*H] -> [W, H, C]
    let o = g.permute(o, [1, 0, 2, 3]);
    let o = g.cont(o);
    let o = g.reshape(o, &[wd, ht, c, 1]);
    g.add(x, o)
}

/// Graph input and output of one decode.
#[derive(Debug, Clone, Copy)]
pub struct VaeIo {
    /// Latents `[W, H, latent_channels, 1]`.
    pub latents: Tn,
    /// Pixels `[W * f, H * f, out_channels, 1]` in `[-1, 1]`.
    pub out: Tn,
}

/// Build the decoder for a `lat_w x lat_h` latent into `g`.
#[must_use]
pub fn build_decoder(g: &mut Graph, cfg: &VaeConfig, w: &Weights, lat_w: i64, lat_h: i64) -> VaeIo {
    let groups = cfg.norm_num_groups as i32;
    let latents = g.input(sys::GGML_TYPE_F32, &[lat_w, lat_h, cfg.latent_channels as i64, 1]);
    let x = conv(g, w, "post_quant_conv", latents, 0);
    let mut x = conv(g, w, "decoder.conv_in", x, 1);
    x = resnet(g, w, "decoder.mid_block.resnets.0", x, groups);
    x = mid_attention(g, w, "decoder.mid_block.attentions.0", x, groups);
    x = resnet(g, w, "decoder.mid_block.resnets.1", x, groups);
    let levels = cfg.block_out_channels.len();
    for i in 0..levels {
        for r in 0..=cfg.layers_per_block {
            x = resnet(g, w, &format!("decoder.up_blocks.{i}.resnets.{r}"), x, groups);
        }
        if i + 1 < levels {
            x = g.upscale_nearest(x, 2);
            x = conv(g, w, &format!("decoder.up_blocks.{i}.upsamplers.0.conv"), x, 1);
        }
    }
    x = gn(g, w, "decoder.conv_norm_out", x, groups);
    x = g.silu(x);
    let out = conv(g, w, "decoder.conv_out", x, 1);
    VaeIo { latents, out }
}

/// Graph input and output of one encode.
#[derive(Debug, Clone, Copy)]
pub struct VaeEncodeIo {
    /// Pixels `[W, H, in_channels, 1]` in `[-1, 1]`.
    pub pixels: Tn,
    /// Posterior mean `[W / f, H / f, latent_channels, 1]`.
    pub mean: Tn,
}

/// Build the encoder for a `width x height` image into `g`. The output is the
/// posterior mean (the mode), which is what reference conditioning uses.
#[must_use]
pub fn build_encoder(g: &mut Graph, cfg: &VaeConfig, w: &Weights, width: i64, height: i64) -> VaeEncodeIo {
    let groups = cfg.norm_num_groups as i32;
    let pixels = g.input(sys::GGML_TYPE_F32, &[width, height, cfg.in_channels as i64, 1]);
    let mut x = conv(g, w, "encoder.conv_in", pixels, 1);
    let levels = cfg.block_out_channels.len();
    for i in 0..levels {
        for r in 0..cfg.layers_per_block {
            x = resnet(g, w, &format!("encoder.down_blocks.{i}.resnets.{r}"), x, groups);
        }
        if i + 1 < levels {
            // Downsampling pads one pixel on the right and bottom only, then
            // convolves with stride 2.
            let p = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
            let padded = g.pad_end(x, 1, 1);
            let y = g.conv2d_stride2(w.get(&format!("{p}.weight")), padded);
            let b = bias4(g, w.get(&format!("{p}.bias")));
            x = g.add(y, b);
        }
    }
    x = resnet(g, w, "encoder.mid_block.resnets.0", x, groups);
    x = mid_attention(g, w, "encoder.mid_block.attentions.0", x, groups);
    x = resnet(g, w, "encoder.mid_block.resnets.1", x, groups);
    x = gn(g, w, "encoder.conv_norm_out", x, groups);
    x = g.silu(x);
    x = conv(g, w, "encoder.conv_out", x, 1);
    let moments = conv(g, w, "quant_conv", x, 0);
    // The first half of the channels is the mean; the second the log-variance.
    let lc = cfg.latent_channels as i64;
    let (mw, mh) = (moments.ne(0), moments.ne(1));
    let mean = g.view_4d(moments, [mw, mh, lc, 1], moments.nb(1), moments.nb(2), moments.nb(3), 0);
    let mean = g.cont(mean);
    VaeEncodeIo { pixels, mean }
}
