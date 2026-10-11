//! Janus-Pro image side: the image-token head and embedder that run beside the language
//! backbone, and the VQ decoder that turns a grid of image tokens into pixels.
//!
//! Generation alternates with the backbone: the backbone's last hidden state goes through
//! [`JanusImageHead::logits_n`] to score the next image token (a codebook index), and the chosen
//! token comes back through [`JanusImageHead::embed_n`] as the backbone's next input. When the
//! grid is complete, [`JanusVqDecoder::decode`] reads each token's codebook vector
//! (L2-normalised), applies the post-quantisation convolution and the decoder (residual blocks
//! with group norm, single-head attention at the lowest resolution, nearest-neighbour
//! upsampling).
//!
//! Weights are read from the checkpoint's own safetensors files (`model.generation_*` and
//! `model.vqmodel.*`).

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Tn, WType, WeightSpec, Weights};
use crate::pipeline::Precision;
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

const GN_EPS: f32 = 1e-6;
const GN_GROUPS: i32 = 32;
const ATTN_QUERY_BLOCK: i64 = 4096;

/// The VQ model and generation head configuration (`vq_config` of the checkpoint).
#[derive(Debug, Clone, Deserialize)]
pub struct JanusVqConfig {
    /// Codebook vector width.
    pub embed_dim: u64,
    /// Codebook size (image-token vocabulary).
    pub num_embeddings: u64,
    /// Decoder base width.
    pub base_channels: u64,
    /// Width multiplier per level, shallowest first.
    pub channel_multiplier: Vec<u64>,
    /// Residual blocks per encoder level; the decoder has one more.
    pub num_res_blocks: usize,
    /// Channels after the post-quantisation convolution.
    pub latent_channels: u64,
    /// Image channels out.
    pub out_channels: u64,
    /// Width of the generation aligner and head.
    pub projection_dim: u64,
    /// Width of the backbone's hidden state.
    pub image_token_embed_dim: u64,
    /// Layers of the generation aligner.
    pub num_hidden_layers: usize,
    /// Activation of the aligner and head.
    pub hidden_act: String,
    /// Token grid side.
    pub num_patches: u64,
}

#[derive(Debug, Deserialize)]
struct Top {
    vq_config: JanusVqConfig,
}

impl JanusVqConfig {
    /// Read `vq_config` from the checkpoint's `config.json`.
    ///
    /// # Errors
    /// [`Error::Config`] when the file is missing, malformed, or describes an unsupported layout.
    pub fn from_checkpoint(dir: &Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("config.json")).map_err(|e| Error::Config(format!("config.json: {e}")))?;
        let top: Top = serde_json::from_slice(&raw).map_err(|e| Error::Config(format!("config.json: {e}")))?;
        top.vq_config.validate()?;
        Ok(top.vq_config)
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_act != "gelu" {
            return Err(Error::Config(format!("Janus head activation {} is not implemented", self.hidden_act)));
        }
        if self.channel_multiplier.is_empty() || self.num_hidden_layers == 0 {
            return Err(Error::Config("Janus VQ config has no levels".into()));
        }
        Ok(())
    }

    /// Pixels per token side.
    #[must_use]
    pub fn scale_factor(&self) -> u64 {
        1 << (self.channel_multiplier.len() - 1)
    }

    /// Tokens per image.
    #[must_use]
    pub fn tokens_per_image(&self) -> usize {
        (self.num_patches * self.num_patches) as usize
    }

    fn head_specs(&self, ty: WType) -> Vec<WeightSpec> {
        let (e, p, h, n) = (self.embed_dim, self.projection_dim, self.image_token_embed_dim, self.num_embeddings);
        let mut v = vec![
            WeightSpec::new("model.generation_embeddings.weight", &[n, e], WType::F32),
            WeightSpec::new("model.generation_aligner.fc1.weight", &[p, e], WType::F32),
            WeightSpec::new("model.generation_aligner.fc1.bias", &[p], WType::F32),
            WeightSpec::new("model.generation_head.proj_out.weight", &[p, h], ty),
            WeightSpec::new("model.generation_head.proj_out.bias", &[p], WType::F32),
            WeightSpec::new("model.generation_head.vision_head.weight", &[n, p], ty),
            WeightSpec::new("model.generation_head.vision_head.bias", &[n], WType::F32),
        ];
        for i in 0..self.num_hidden_layers - 1 {
            v.push(WeightSpec::new(format!("model.generation_aligner.hidden_layers.{i}.weight"), &[p, p], ty));
            v.push(WeightSpec::new(format!("model.generation_aligner.hidden_layers.{i}.bias"), &[p], WType::F32));
        }
        v
    }

    fn decoder_level(&self, k: usize) -> u64 {
        let levels = self.channel_multiplier.len();
        self.base_channels * self.channel_multiplier[levels - 1 - k]
    }

    fn decoder_specs(&self, kt: WType) -> Vec<WeightSpec> {
        let d = "model.vqmodel.decoder";
        let lc = self.latent_channels;
        let top = self.decoder_level(0);
        let mut v = vec![
            WeightSpec::new("model.vqmodel.post_quant_conv.weight", &[lc, self.embed_dim, 1, 1], kt),
            WeightSpec::new("model.vqmodel.post_quant_conv.bias", &[lc], WType::F32),
            WeightSpec::new(format!("{d}.conv_in.weight"), &[top, lc, 3, 3], kt),
            WeightSpec::new(format!("{d}.conv_in.bias"), &[top], WType::F32),
        ];
        resnet_specs(&mut v, &format!("{d}.mid.block_1"), top, top, kt);
        attn_specs(&mut v, &format!("{d}.mid.attn_1"), top, kt);
        resnet_specs(&mut v, &format!("{d}.mid.block_2"), top, top, kt);
        let levels = self.channel_multiplier.len();
        let mut cin = top;
        for k in 0..levels {
            let c = self.decoder_level(k);
            for b in 0..=self.num_res_blocks {
                resnet_specs(&mut v, &format!("{d}.up.{k}.block.{b}"), cin, c, kt);
                cin = c;
                if k == 0 {
                    attn_specs(&mut v, &format!("{d}.up.{k}.attn.{b}"), c, kt);
                }
            }
            if k + 1 < levels {
                v.push(WeightSpec::new(format!("{d}.up.{k}.upsample.conv.weight"), &[c, c, 3, 3], kt));
                v.push(WeightSpec::new(format!("{d}.up.{k}.upsample.conv.bias"), &[c], WType::F32));
            }
        }
        v.push(WeightSpec::new(format!("{d}.norm_out.weight"), &[cin], WType::F32));
        v.push(WeightSpec::new(format!("{d}.norm_out.bias"), &[cin], WType::F32));
        v.push(WeightSpec::new(format!("{d}.conv_out.weight"), &[self.out_channels, cin, 3, 3], kt));
        v.push(WeightSpec::new(format!("{d}.conv_out.bias"), &[self.out_channels], WType::F32));
        v
    }
}

fn resnet_specs(v: &mut Vec<WeightSpec>, p: &str, cin: u64, cout: u64, kt: WType) {
    v.push(WeightSpec::new(format!("{p}.norm1.weight"), &[cin], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm1.bias"), &[cin], WType::F32));
    v.push(WeightSpec::new(format!("{p}.conv1.weight"), &[cout, cin, 3, 3], kt));
    v.push(WeightSpec::new(format!("{p}.conv1.bias"), &[cout], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm2.weight"), &[cout], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm2.bias"), &[cout], WType::F32));
    v.push(WeightSpec::new(format!("{p}.conv2.weight"), &[cout, cout, 3, 3], kt));
    v.push(WeightSpec::new(format!("{p}.conv2.bias"), &[cout], WType::F32));
    if cin != cout {
        v.push(WeightSpec::new(format!("{p}.nin_shortcut.weight"), &[cout, cin, 1, 1], kt));
        v.push(WeightSpec::new(format!("{p}.nin_shortcut.bias"), &[cout], WType::F32));
    }
}

fn attn_specs(v: &mut Vec<WeightSpec>, p: &str, c: u64, kt: WType) {
    v.push(WeightSpec::new(format!("{p}.norm.weight"), &[c], WType::F32));
    v.push(WeightSpec::new(format!("{p}.norm.bias"), &[c], WType::F32));
    for n in ["q", "k", "v", "proj_out"] {
        v.push(WeightSpec::new(format!("{p}.{n}.weight"), &[c, c, 1, 1], kt));
        v.push(WeightSpec::new(format!("{p}.{n}.bias"), &[c], WType::F32));
    }
}

fn open(dir: &Path) -> Result<SafeTensors> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| Error::Weights(format!("{}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(Error::Weights(format!("no safetensors files in {}", dir.display())));
    }
    SafeTensors::open(&paths)
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

fn gn(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let s = bias4(g, w.get(&format!("{p}.weight")));
    let b = bias4(g, w.get(&format!("{p}.bias")));
    let y = g.group_norm(x, GN_GROUPS, GN_EPS);
    let y = g.mul(y, s);
    g.add(y, b)
}

fn resnet(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let h = gn(g, w, &format!("{p}.norm1"), x);
    let h = g.silu(h);
    let h = conv(g, w, &format!("{p}.conv1"), h, 1);
    let h = gn(g, w, &format!("{p}.norm2"), h);
    let h = g.silu(h);
    let h = conv(g, w, &format!("{p}.conv2"), h, 1);
    let skip = if x.ne(2) == h.ne(2) { x } else { conv(g, w, &format!("{p}.nin_shortcut"), x, 0) };
    g.add(skip, h)
}

/// Single-head attention over every cell, as wide as the channel count; scores are formed a
/// block of queries at a time.
fn attn(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let (wd, ht, c) = (x.ne(0), x.ne(1), x.ne(2));
    let h = gn(g, w, &format!("{p}.norm"), x);
    let h = g.reshape(h, &[wd * ht, c]);
    let h = g.permute(h, [1, 0, 2, 3]);
    let h = g.cont(h);
    let proj = |g: &mut Graph, n: &str, t: Tn| {
        let wt = w.get(&format!("{p}.{n}.weight"));
        let wt = g.reshape(wt, &[c, c]);
        g.linear_b(wt, w.get(&format!("{p}.{n}.bias")), t)
    };
    let q = proj(g, "q", h);
    let k = proj(g, "k", h);
    let v = proj(g, "v", h);
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
        let pr = g.soft_max(s, scale);
        let ob = g.linear(vt, pr);
        o = Some(match o {
            None => ob,
            Some(prev) => g.concat(prev, ob, 1),
        });
        from += m;
    }
    let o = o.expect("at least one cell");
    let o = proj(g, "proj_out", o);
    let o = g.permute(o, [1, 0, 2, 3]);
    let o = g.cont(o);
    let o = g.reshape(o, &[wd, ht, c, 1]);
    g.add(x, o)
}

/// The generation embedder and image-token head, resident on one backend.
pub struct JanusImageHead {
    cfg: JanusVqConfig,
    weights: Weights,
    backend: Backend,
    batch: usize,
    head_graph: Graph,
    head_in: Tn,
    head_out: Tn,
    embed_graph: Graph,
    embed_in: Tn,
    embed_out: Tn,
}

impl std::fmt::Debug for JanusImageHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JanusImageHead").field("batch", &self.batch).field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl JanusImageHead {
    /// Load the head from a Janus checkpoint directory, for `batch` rows per call (2 for
    /// guided generation: the conditioned and the unconditioned sequence).
    ///
    /// # Errors
    /// Missing or mis-shaped weights, or a backend that cannot run the graphs.
    pub fn load(dir: &Path, backend: Backend, precision: Precision, batch: usize) -> Result<Self> {
        let cfg = JanusVqConfig::from_checkpoint(dir)?;
        let files = open(dir)?;
        let ty = match precision {
            Precision::F32 => WType::F32,
            Precision::Bf16 => WType::Bf16,
            Precision::Q8_0 => WType::Q8_0,
        };
        let weights = Weights::load(&backend, &files, &cfg.head_specs(ty))?;
        let b = batch as i64;

        let mut head_graph = Graph::new(&backend)?;
        let head_in = head_graph.input(sys::GGML_TYPE_F32, &[cfg.image_token_embed_dim as i64, b]);
        let x = head_graph.linear_b(weights.get("model.generation_head.proj_out.weight"), weights.get("model.generation_head.proj_out.bias"), head_in);
        let x = head_graph.gelu_erf(x);
        let head_out = head_graph.linear_b(weights.get("model.generation_head.vision_head.weight"), weights.get("model.generation_head.vision_head.bias"), x);
        head_graph.finish(&[head_out])?;

        let mut embed_graph = Graph::new(&backend)?;
        let embed_in = embed_graph.input(sys::GGML_TYPE_I32, &[b]);
        let e = embed_graph.get_rows(weights.get("model.generation_embeddings.weight"), embed_in);
        let mut x = embed_graph.linear_b(weights.get("model.generation_aligner.fc1.weight"), weights.get("model.generation_aligner.fc1.bias"), e);
        for i in 0..cfg.num_hidden_layers - 1 {
            x = embed_graph.gelu_erf(x);
            x = embed_graph.linear_b(
                weights.get(&format!("model.generation_aligner.hidden_layers.{i}.weight")),
                weights.get(&format!("model.generation_aligner.hidden_layers.{i}.bias")),
                x,
            );
        }
        let embed_out = x;
        embed_graph.finish(&[embed_out])?;
        Ok(Self { cfg, weights, backend, batch, head_graph, head_in, head_out, embed_graph, embed_in, embed_out })
    }

    /// The configuration read from the checkpoint.
    #[must_use]
    pub fn config(&self) -> &JanusVqConfig {
        &self.cfg
    }

    /// Image-token logits for `n` hidden states (`n * hidden` floats, row-major); `n` is the
    /// batch the head was loaded for.
    ///
    /// # Errors
    /// A wrong count or input size, or a failed compute.
    pub fn logits_n(&mut self, hidden: &[f32], n: usize) -> Result<Vec<f32>> {
        if n != self.batch {
            return Err(Error::Config(format!("image head was loaded for {} rows, called with {n}", self.batch)));
        }
        let want = self.batch * self.cfg.image_token_embed_dim as usize;
        if hidden.len() != want {
            return Err(Error::Config(format!("image head takes {want} floats, got {}", hidden.len())));
        }
        self.head_graph.set_f32(self.head_in, hidden);
        self.head_graph.compute()?;
        Ok(self.head_graph.read_f32(self.head_out))
    }

    /// Backbone input embeddings for `batch` image tokens.
    ///
    /// # Errors
    /// A wrong count, an out-of-range token, or a failed compute.
    pub fn embed_n(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.len() != self.batch {
            return Err(Error::Config(format!("image embedder takes {} tokens, got {}", self.batch, tokens.len())));
        }
        if let Some(t) = tokens.iter().find(|&&t| u64::from(t) >= self.cfg.num_embeddings) {
            return Err(Error::Config(format!("image token {t} is outside the codebook")));
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.embed_graph.set_i32(self.embed_in, &ids);
        self.embed_graph.compute()?;
        Ok(self.embed_graph.read_f32(self.embed_out))
    }

    /// Device bytes held by the head weights.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.weights.bytes()
    }
}

/// The VQ decoder: a grid of image tokens to pixels.
pub struct JanusVqDecoder {
    cfg: JanusVqConfig,
    codebook: Vec<f32>,
    weights: Weights,
    backend: Backend,
}

impl std::fmt::Debug for JanusVqDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JanusVqDecoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

impl JanusVqDecoder {
    /// Load from a Janus checkpoint directory. `exact` keeps the convolution kernels in f32.
    ///
    /// # Errors
    /// Missing or mis-shaped weights.
    pub fn load(dir: &Path, backend: Backend, exact: bool) -> Result<Self> {
        let cfg = JanusVqConfig::from_checkpoint(dir)?;
        let files = open(dir)?;
        let n = cfg.num_embeddings;
        let e = cfg.embed_dim as usize;
        let mut codebook = files.require("model.vqmodel.quantize.embedding.weight", &[n, cfg.embed_dim])?.to_f32();
        for row in codebook.chunks_mut(e) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|v| *v /= norm);
        }
        let kt = if exact { WType::F32 } else { WType::F16 };
        let weights = Weights::load(&backend, &files, &cfg.decoder_specs(kt))?;
        Ok(Self { cfg, codebook, weights, backend })
    }

    /// Decode a full grid of tokens (row-major) to planar RGB `[3, H, W]` in `[-1, 1]`.
    ///
    /// # Errors
    /// A wrong token count, an out-of-range token, or a failed compute.
    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let side = self.cfg.num_patches as usize;
        if tokens.len() != side * side {
            return Err(Error::Config(format!("the decoder takes {} tokens, got {}", side * side, tokens.len())));
        }
        let e = self.cfg.embed_dim as usize;
        // Planar [C, H, W] latent.
        let mut z = vec![0f32; e * side * side];
        for (i, &t) in tokens.iter().enumerate() {
            let t = t as usize;
            if t as u64 >= self.cfg.num_embeddings {
                return Err(Error::Config(format!("image token {t} is outside the codebook")));
            }
            for c in 0..e {
                z[c * side * side + i] = self.codebook[t * e + c];
            }
        }
        let mut g = Graph::new(&self.backend)?;
        let s = side as i64;
        let latents = g.input(sys::GGML_TYPE_F32, &[s, s, e as i64, 1]);
        let d = "model.vqmodel.decoder";
        let w = &self.weights;
        let x = conv(&mut g, w, "model.vqmodel.post_quant_conv", latents, 0);
        let mut x = conv(&mut g, w, &format!("{d}.conv_in"), x, 1);
        x = resnet(&mut g, w, &format!("{d}.mid.block_1"), x);
        x = attn(&mut g, w, &format!("{d}.mid.attn_1"), x);
        x = resnet(&mut g, w, &format!("{d}.mid.block_2"), x);
        let levels = self.cfg.channel_multiplier.len();
        for k in 0..levels {
            for b in 0..=self.cfg.num_res_blocks {
                x = resnet(&mut g, w, &format!("{d}.up.{k}.block.{b}"), x);
                if k == 0 {
                    x = attn(&mut g, w, &format!("{d}.up.{k}.attn.{b}"), x);
                }
            }
            if k + 1 < levels {
                x = g.upscale_nearest(x, 2);
                x = conv(&mut g, w, &format!("{d}.up.{k}.upsample.conv"), x, 1);
            }
        }
        x = gn(&mut g, w, &format!("{d}.norm_out"), x);
        x = g.silu(x);
        let out = conv(&mut g, w, &format!("{d}.conv_out"), x, 1);
        g.finish(&[out])?;
        g.set_f32(latents, &z);
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// Output image side in pixels.
    #[must_use]
    pub fn image_side(&self) -> usize {
        (self.cfg.num_patches * self.cfg.scale_factor()) as usize
    }
}
