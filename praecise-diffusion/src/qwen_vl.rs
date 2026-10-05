//! Qwen2.5-VL as a prompt encoder: a windowed vision transformer turns each
//! image into tokens spliced into the prompt, and the language model's last,
//! normalised hidden state conditions the image generator.
//!
//! Image tokens take three-axis positions (time, row, column of the merged
//! patch grid) and the rotary frequencies are split between the axes in
//! contiguous sections; text tokens share one position on all three axes.

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Per-channel mean the vision tower's inputs are normalised by.
#[allow(clippy::excessive_precision)]
pub const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// Per-channel standard deviation the vision tower's inputs are normalised by.
#[allow(clippy::excessive_precision)]
pub const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];
const VISION_THETA: f64 = 10_000.0;

/// The vision tower's layout.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Blocks.
    pub depth: usize,
    /// Width.
    pub hidden_size: u64,
    /// Feed-forward width.
    pub intermediate_size: u64,
    /// Attention heads.
    pub num_heads: u64,
    /// Width of the merged tokens handed to the language model.
    pub out_hidden_size: u64,
    /// Patch side in pixels.
    pub patch_size: u64,
    /// Patches merged per side into one language-model token.
    pub spatial_merge_size: u64,
    /// Frames per patch (a still image is repeated).
    pub temporal_patch_size: u64,
    /// Attention window side in pixels.
    pub window_size: u64,
    /// Blocks that attend over the whole image instead of a window.
    pub fullatt_block_indexes: Vec<usize>,
    /// Input channels. Released configs may carry both this key and the
    /// older `in_chans`; read [`VisionConfig::in_channels`].
    #[serde(default)]
    in_channels: Option<u64>,
    /// Input channels under the older key.
    #[serde(default)]
    in_chans: Option<u64>,
}

impl VisionConfig {
    /// Input channels (3 when the config names none).
    #[must_use]
    pub fn in_channels(&self) -> u64 {
        self.in_channels.or(self.in_chans).unwrap_or(3)
    }

    fn validate(&self) -> Result<()> {
        if let (Some(a), Some(b)) = (self.in_channels, self.in_chans) {
            if a != b {
                return Err(Error::Config(format!("vision config: in_channels {a} and in_chans {b} disagree")));
            }
        }
        Ok(())
    }
}

/// Multi-axis rotary sections.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Frequency pairs given to the time, row and column positions.
    pub mrope_section: Vec<u64>,
}

/// The language model's layout plus its vision tower.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenVlConfig {
    /// Width.
    pub hidden_size: u64,
    /// Feed-forward width.
    pub intermediate_size: u64,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Layers.
    pub num_hidden_layers: usize,
    /// Key/value heads.
    pub num_key_value_heads: u64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Rotary base.
    pub rope_theta: f64,
    /// Rotary sections.
    pub rope_scaling: RopeScaling,
    /// Vocabulary size.
    pub vocab_size: u64,
    /// Placeholder token each image token replaces.
    pub image_token_id: u32,
    /// The vision tower.
    pub vision_config: VisionConfig,
}

impl QwenVlConfig {
    /// Parse and check a `config.json`.
    ///
    /// # Errors
    /// A malformed or unsupported configuration.
    pub fn from_json(v: Value) -> Result<Self> {
        let c: Self = parse(v, "vision-language encoder config")?;
        c.validate()?;
        Ok(c)
    }

    fn head_dim(&self) -> u64 {
        self.hidden_size / self.num_attention_heads
    }

    fn sections(&self) -> Result<[usize; 3]> {
        let s = &self.rope_scaling.mrope_section;
        let a = <[u64; 3]>::try_from(s.as_slice()).map_err(|_| Error::Config(format!("{} rotary sections, expected 3", s.len())))?;
        Ok(a.map(|x| x as usize))
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Config(m));
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) || !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return bad("heads do not divide the width".into());
        }
        let s = self.sections()?;
        if (s.iter().sum::<usize>() * 2) as u64 != self.head_dim() {
            return bad(format!("rotary sections {s:?} do not cover the head width {}", self.head_dim()));
        }
        let v = &self.vision_config;
        v.validate()?;
        if !v.hidden_size.is_multiple_of(v.num_heads) || !(v.hidden_size / v.num_heads).is_multiple_of(4) {
            return bad("vision heads do not split into row and column halves".into());
        }
        let unit = v.patch_size * v.spatial_merge_size;
        if !v.window_size.is_multiple_of(unit) {
            return bad(format!("window {} is not a multiple of the merged patch {unit}", v.window_size));
        }
        if v.fullatt_block_indexes.iter().any(|&i| i >= v.depth) {
            return bad("full-attention block outside the tower".into());
        }
        Ok(())
    }

    /// Pixel side the image sides must be multiples of.
    #[must_use]
    pub fn image_unit(&self) -> usize {
        (self.vision_config.patch_size * self.vision_config.spatial_merge_size) as usize
    }

    fn weight_specs(&self, embed: WType, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size;
        let hd = self.head_dim();
        let q = self.num_attention_heads * hd;
        let kv = self.num_key_value_heads * hd;
        let ff = self.intermediate_size;
        let f = WType::F32;
        let mut s = vec![
            WeightSpec::new("model.embed_tokens.weight", &[self.vocab_size, d], embed),
            WeightSpec::new("model.norm.weight", &[d], f),
        ];
        for i in 0..self.num_hidden_layers {
            let p = format!("model.layers.{i}");
            s.push(WeightSpec::new(format!("{p}.input_layernorm.weight"), &[d], f));
            s.push(WeightSpec::new(format!("{p}.post_attention_layernorm.weight"), &[d], f));
            for (n, rows) in [("q", q), ("k", kv), ("v", kv)] {
                s.push(WeightSpec::new(format!("{p}.self_attn.{n}_proj.weight"), &[rows, d], linear));
                s.push(WeightSpec::new(format!("{p}.self_attn.{n}_proj.bias"), &[rows], f));
            }
            s.push(WeightSpec::new(format!("{p}.self_attn.o_proj.weight"), &[d, q], linear));
            s.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[ff, d], linear));
            s.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[ff, d], linear));
            s.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[d, ff], linear));
        }
        let v = &self.vision_config;
        let (vd, vf) = (v.hidden_size, v.intermediate_size);
        let merged = vd * v.spatial_merge_size * v.spatial_merge_size;
        for i in 0..v.depth {
            let p = format!("visual.blocks.{i}");
            s.push(WeightSpec::new(format!("{p}.norm1.weight"), &[vd], f));
            s.push(WeightSpec::new(format!("{p}.norm2.weight"), &[vd], f));
            s.push(WeightSpec::new(format!("{p}.attn.qkv.weight"), &[3 * vd, vd], linear));
            s.push(WeightSpec::new(format!("{p}.attn.qkv.bias"), &[3 * vd], f));
            s.push(WeightSpec::new(format!("{p}.attn.proj.weight"), &[vd, vd], linear));
            s.push(WeightSpec::new(format!("{p}.attn.proj.bias"), &[vd], f));
            for (n, shape) in [("gate", [vf, vd]), ("up", [vf, vd]), ("down", [vd, vf])] {
                s.push(WeightSpec::new(format!("{p}.mlp.{n}_proj.weight"), &shape, linear));
                s.push(WeightSpec::new(format!("{p}.mlp.{n}_proj.bias"), &[shape[0]], f));
            }
        }
        s.push(WeightSpec::new("visual.merger.ln_q.weight", &[vd], f));
        s.push(WeightSpec::new("visual.merger.mlp.0.weight", &[merged, merged], linear));
        s.push(WeightSpec::new("visual.merger.mlp.0.bias", &[merged], f));
        s.push(WeightSpec::new("visual.merger.mlp.2.weight", &[v.out_hidden_size, merged], linear));
        s.push(WeightSpec::new("visual.merger.mlp.2.bias", &[v.out_hidden_size], f));
        // 8-bit blocks hold 32 columns; narrower rows (the vision feed-forward
        // width 3420 of the released tower) stay float32.
        for w in &mut s {
            if w.ty == WType::Q8_0 && !w.shape.last().copied().unwrap_or(0).is_multiple_of(32) {
                w.ty = f;
            }
        }
        s
    }

    fn patch_in(&self) -> u64 {
        let v = &self.vision_config;
        v.in_channels() * v.temporal_patch_size * v.patch_size * v.patch_size
    }
}

/// One image prepared for the vision tower.
#[derive(Debug, Clone)]
pub struct VlImage {
    /// Patches `[patches][channels * frames * patch * patch]` in the tower's
    /// order (each merged 2x2 block contiguous).
    pub patches: Vec<f32>,
    /// Patch grid `(rows, cols)`.
    pub grid: (usize, usize),
}

/// The prompt encoder.
pub struct QwenVlEncoder {
    backend: Backend,
    cfg: QwenVlConfig,
    w: Weights,
    patch: Weights,
    exact: bool,
}

impl std::fmt::Debug for QwenVlEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenVlEncoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

/// Additive mask: `allow(q, k)` keeps a pair, everything else is `-inf`.
fn mask(n: usize, allow: impl Fn(usize, usize) -> bool) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; n * n];
    for q in 0..n {
        for k in 0..n {
            if allow(q, k) {
                m[q * n + k] = 0.0;
            }
        }
    }
    m
}

impl QwenVlEncoder {
    /// Load `dir/` (its `config.json` and safetensors files) of a checkpoint.
    ///
    /// # Errors
    /// An unsupported configuration, missing weights or no usable backend.
    pub fn load(files: &CheckpointFiles, dir: &str, opts: LoadOptions) -> Result<Self> {
        let cfg = QwenVlConfig::from_json(files.json(&format!("{dir}/config.json"))?)?;
        let st = SafeTensors::open(&files.weights(dir)?)?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let embed = if exact { WType::F32 } else { WType::F16 };
        let w = Weights::load(&backend, &st, &cfg.weight_specs(embed, opts.precision.wtype()))?;
        let v = &cfg.vision_config;
        let name = "visual.patch_embed.proj.weight";
        let shape = [v.hidden_size, v.in_channels(), v.temporal_patch_size, v.patch_size, v.patch_size];
        let data = st.require(name, &shape)?.to_f32();
        let patch = Weights::from_host(
            &backend,
            &[HostTensor { name: name.into(), shape: vec![v.hidden_size, cfg.patch_in()], ty: WType::F32, data }],
        )?;
        Ok(Self { backend, cfg, w, patch, exact })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &QwenVlConfig {
        &self.cfg
    }

    /// Device bytes held.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes() + self.patch.bytes()
    }

    /// Patches of an image `[3][h][w]` with values in `[0, 1]` (sides
    /// multiples of [`QwenVlConfig::image_unit`]), normalised and laid out
    /// for [`Self::encode_image`].
    ///
    /// # Errors
    /// Sides that are not multiples of the merged patch.
    pub fn image(&self, pixels: &[f32], (h, w): (usize, usize)) -> Result<VlImage> {
        let v = &self.cfg.vision_config;
        let (p, m, tp) = (v.patch_size as usize, v.spatial_merge_size as usize, v.temporal_patch_size as usize);
        let unit = p * m;
        if h % unit != 0 || w % unit != 0 || h == 0 || w == 0 || pixels.len() != 3 * h * w {
            return Err(Error::Request(format!("image {w}x{h} is not a multiple of {unit} pixels")));
        }
        let (gh, gw) = (h / p, w / p);
        let mut out = Vec::with_capacity(gh * gw * 3 * tp * p * p);
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let (r0, c0) = ((bh * m + mh) * p, (bw * m + mw) * p);
                        for c in 0..3 {
                            for _ in 0..tp {
                                for y in 0..p {
                                    for x in 0..p {
                                        let px = pixels[(c * h + r0 + y) * w + c0 + x];
                                        out.push((px - IMAGE_MEAN[c]) / IMAGE_STD[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(VlImage { patches: out, grid: (gh, gw) })
    }

    /// Merged image tokens `[tokens][language width]` of one image, in the
    /// order the prompt holds them (row-major over the merged grid).
    ///
    /// # Errors
    /// Patches that disagree with the grid, or a backend failure.
    pub fn encode_image(&self, img: &VlImage) -> Result<Vec<f32>> {
        let v = &self.cfg.vision_config;
        let (gh, gw) = img.grid;
        let n = gh * gw;
        let m = v.spatial_merge_size as usize;
        if n == 0 || gh % m != 0 || gw % m != 0 || img.patches.len() as u64 != n as u64 * self.cfg.patch_in() {
            return Err(Error::Request("image patches disagree with their grid".into()));
        }
        let (d, nh) = (v.hidden_size as i64, v.num_heads as i64);
        let hd = d / nh;
        let ni = n as i64;
        let eps = 1e-6;
        let w = &self.w;
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[self.cfg.patch_in() as i64, ni]);
        let cos = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let sin = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let window = g.input(sys::GGML_TYPE_F16, &[ni, ni]);
        let mut x = g.linear(self.patch.get("visual.patch_embed.proj.weight"), input);
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..v.depth {
            let p = format!("visual.blocks.{i}");
            let wn = |s: &str| w.get(&format!("{p}.{s}"));
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("norm1.weight"));
            let qkv = g.linear_b(wn("attn.qkv.weight"), wn("attn.qkv.bias"), h);
            let heads: Vec<Tn> = (0..3)
                .map(|j| {
                    let t = g.view_heads(qkv, j * d, hd, nh);
                    g.cont(t)
                })
                .collect();
            let q = g.rotate_half_rope(heads[0], cos, sin);
            let k = g.rotate_half_rope(heads[1], cos, sin);
            let q = g.permute(q, [0, 2, 1, 3]);
            let k = g.permute(k, [0, 2, 1, 3]);
            let vv = g.permute(heads[2], [0, 2, 1, 3]);
            let full = v.fullatt_block_indexes.contains(&i);
            let o = if full {
                self.attend(&mut g, q, k, vv, None, scale)
            } else {
                self.attend(&mut g, q, k, vv, Some(window), scale)
            };
            let o = g.reshape(o, &[d, ni]);
            let o = g.linear_b(wn("attn.proj.weight"), wn("attn.proj.bias"), o);
            x = g.add(x, o);
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("norm2.weight"));
            let gate = g.linear_b(wn("mlp.gate_proj.weight"), wn("mlp.gate_proj.bias"), h);
            let up = g.linear_b(wn("mlp.up_proj.weight"), wn("mlp.up_proj.bias"), h);
            let f = g.swiglu_split(gate, up);
            let f = g.linear_b(wn("mlp.down_proj.weight"), wn("mlp.down_proj.bias"), f);
            x = g.add(x, f);
        }
        let h = g.rms_norm(x, eps);
        let h = g.mul(h, w.get("visual.merger.ln_q.weight"));
        let h = g.reshape(h, &[d * (m * m) as i64, ni / (m * m) as i64]);
        let h = g.linear_b(w.get("visual.merger.mlp.0.weight"), w.get("visual.merger.mlp.0.bias"), h);
        let h = g.gelu_erf(h);
        let out = g.linear_b(w.get("visual.merger.mlp.2.weight"), w.get("visual.merger.mlp.2.bias"), h);
        g.finish(&[out])?;

        g.set_f32(input, &img.patches);
        let (c, s) = self.vision_tables(img.grid);
        g.set_f32(cos, &c);
        g.set_f32(sin, &s);
        let wins = self.windows(img.grid);
        g.set_f16(window, &mask(n, |q, k| wins[q] == wins[k]));
        g.compute()?;
        Ok(g.read_f32(out))
    }

    fn attend(&self, g: &mut Graph, q: Tn, k: Tn, v: Tn, mask: Option<Tn>, scale: f32) -> Tn {
        if self.exact {
            g.attention_exact(q, k, v, mask, scale)
        } else {
            let k = g.cast(k, sys::GGML_TYPE_F16);
            let v = g.cast(v, sys::GGML_TYPE_F16);
            g.attention(q, k, v, mask, scale, true)
        }
    }

    /// Row and column of every patch in the tower's order.
    fn patch_positions(&self, (gh, gw): (usize, usize)) -> Vec<(usize, usize)> {
        let m = self.cfg.vision_config.spatial_merge_size as usize;
        let mut out = Vec::with_capacity(gh * gw);
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        out.push((bh * m + mh, bw * m + mw));
                    }
                }
            }
        }
        out
    }

    /// Attention window of every patch: windows tile the merged grid from
    /// the top-left corner, partial at the right and bottom edges.
    fn windows(&self, grid: (usize, usize)) -> Vec<usize> {
        let v = &self.cfg.vision_config;
        let m = v.spatial_merge_size as usize;
        let side = (v.window_size / v.patch_size) as usize / m;
        let across = (grid.1 / m).div_ceil(side);
        self.patch_positions(grid).iter().map(|&(r, c)| (r / m / side) * across + (c / m / side)).collect()
    }

    /// Rotary tables `[tokens][head width]`: the first quarter of the pairs
    /// rotate by the row, the second by the column, mirrored into the
    /// second half.
    fn vision_tables(&self, grid: (usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let v = &self.cfg.vision_config;
        let hd = (v.hidden_size / v.num_heads) as usize;
        let q = hd / 4;
        let inv: Vec<f64> = (0..q).map(|j| 1.0 / VISION_THETA.powf((2 * j) as f64 / (hd / 2) as f64)).collect();
        let pos = self.patch_positions(grid);
        let mut cos = Vec::with_capacity(pos.len() * hd);
        let mut sin = Vec::with_capacity(pos.len() * hd);
        for &(r, c) in &pos {
            let ang: Vec<f32> = inv.iter().map(|f| (r as f32) * (*f as f32)).chain(inv.iter().map(|f| (c as f32) * (*f as f32))).collect();
            for _ in 0..2 {
                cos.extend(ang.iter().map(|a| a.cos()));
                sin.extend(ang.iter().map(|a| a.sin()));
            }
        }
        (cos, sin)
    }

    /// Three-axis positions of a prompt whose image placeholders are already
    /// expanded to one token per merged patch: image `k`'s tokens take the
    /// row and column of their merged patch offset by the next free
    /// position, which then advances past the larger grid side.
    fn positions(&self, tokens: &[u32], grids: &[(usize, usize)]) -> Result<Vec<[f32; 3]>> {
        let m = self.cfg.vision_config.spatial_merge_size as usize;
        let mut out = Vec::with_capacity(tokens.len());
        let mut next = 0usize;
        let mut i = 0usize;
        let mut img = 0usize;
        while i < tokens.len() {
            if tokens[i] == self.cfg.image_token_id {
                let &(gh, gw) = grids.get(img).ok_or_else(|| Error::Request("more image placeholders than images".into()))?;
                let (rh, rw) = (gh / m, gw / m);
                if tokens.len() < i + rh * rw || tokens[i..i + rh * rw].iter().any(|&t| t != self.cfg.image_token_id) {
                    return Err(Error::Request(format!("image {img}: placeholder run shorter than its {} tokens", rh * rw)));
                }
                for r in 0..rh {
                    for c in 0..rw {
                        out.push([next as f32, (next + r) as f32, (next + c) as f32]);
                    }
                }
                next += rh.max(rw);
                i += rh * rw;
                img += 1;
            } else {
                out.push([next as f32; 3]);
                next += 1;
                i += 1;
            }
        }
        if img != grids.len() {
            return Err(Error::Request("fewer image placeholders than images".into()));
        }
        Ok(out)
    }

    /// Expand every single image placeholder in `tokens` into one per merged
    /// patch of the matching image.
    #[must_use]
    pub fn expand_placeholders(&self, tokens: &[u32], grids: &[(usize, usize)]) -> Vec<u32> {
        let m = self.cfg.vision_config.spatial_merge_size as usize;
        let mut out = Vec::with_capacity(tokens.len());
        let mut img = 0;
        for &t in tokens {
            if let (true, Some(&(gh, gw))) = (t == self.cfg.image_token_id, grids.get(img)) {
                out.extend(std::iter::repeat_n(t, (gh / m) * (gw / m)));
                img += 1;
                continue;
            }
            out.push(t);
        }
        out
    }

    /// The final normalised hidden state `[tokens][width]` of a prompt whose
    /// placeholders are expanded (see [`Self::expand_placeholders`]), with
    /// the images in placeholder order.
    ///
    /// # Errors
    /// Tokens outside the vocabulary, images that disagree with the
    /// placeholders, or a backend failure.
    pub fn forward(&self, tokens: &[u32], images: &[VlImage]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let n = tokens.len();
        if n == 0 || tokens.iter().any(|&t| u64::from(t) >= cfg.vocab_size) {
            return Err(Error::Request("prompt tokens outside the vocabulary".into()));
        }
        let grids: Vec<(usize, usize)> = images.iter().map(|i| i.grid).collect();
        let pos = self.positions(tokens, &grids)?;
        let feats: Vec<Vec<f32>> = images.iter().map(|i| self.encode_image(i)).collect::<Result<_>>()?;

        let (d, hd) = (cfg.hidden_size as i64, cfg.head_dim() as i64);
        let (nh, nkv) = (cfg.num_attention_heads as i64, cfg.num_key_value_heads as i64);
        let eps = cfg.rms_norm_eps as f32;
        let ni = n as i64;
        let w = &self.w;
        let mut g = Graph::new(&self.backend)?;
        // Runs of text tokens and image features, in prompt order.
        let mut runs: Vec<(bool, usize, usize)> = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            let is_img = t == cfg.image_token_id;
            match runs.last_mut() {
                Some((k, _, len)) if *k == is_img => *len += 1,
                _ => runs.push((is_img, i, 1)),
            }
        }
        let mut inputs: Vec<(bool, usize, usize, Tn)> = Vec::new();
        let mut x: Option<Tn> = None;
        for &(is_img, start, len) in &runs {
            let t = if is_img {
                let t = g.input(sys::GGML_TYPE_F32, &[d, len as i64]);
                inputs.push((true, start, len, t));
                t
            } else {
                let ids = g.input(sys::GGML_TYPE_I32, &[len as i64]);
                inputs.push((false, start, len, ids));
                g.get_rows(w.get("model.embed_tokens.weight"), ids)
            };
            x = Some(match x {
                None => t,
                Some(prev) => g.concat(prev, t, 1),
            });
        }
        let Some(mut x) = x else { return Err(Error::Request("empty prompt".into())) };
        let cos = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let sin = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let causal = g.input(sys::GGML_TYPE_F16, &[ni, ni]);
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let wn = |s: &str| w.get(&format!("{p}.{s}"));
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("input_layernorm.weight"));
            let q = g.linear_b(wn("self_attn.q_proj.weight"), wn("self_attn.q_proj.bias"), h);
            let k = g.linear_b(wn("self_attn.k_proj.weight"), wn("self_attn.k_proj.bias"), h);
            let v = g.linear_b(wn("self_attn.v_proj.weight"), wn("self_attn.v_proj.bias"), h);
            let q = g.reshape(q, &[hd, nh, ni]);
            let k = g.reshape(k, &[hd, nkv, ni]);
            let v = g.reshape(v, &[hd, nkv, ni]);
            let q = g.rotate_half_rope(q, cos, sin);
            let k = g.rotate_half_rope(k, cos, sin);
            let q = g.permute(q, [0, 2, 1, 3]);
            let k = g.permute(k, [0, 2, 1, 3]);
            let v = g.permute(v, [0, 2, 1, 3]);
            let o = self.attend(&mut g, q, k, v, Some(causal), scale);
            let o = g.reshape(o, &[hd * nh, ni]);
            let o = g.linear(wn("self_attn.o_proj.weight"), o);
            x = g.add(x, o);
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("post_attention_layernorm.weight"));
            let gate = g.linear(wn("mlp.gate_proj.weight"), h);
            let up = g.linear(wn("mlp.up_proj.weight"), h);
            let f = g.swiglu_split(gate, up);
            let f = g.linear(wn("mlp.down_proj.weight"), f);
            x = g.add(x, f);
        }
        let h = g.rms_norm(x, eps);
        let out = g.mul(h, w.get("model.norm.weight"));
        g.finish(&[out])?;

        let mut img = 0;
        for &(is_img, start, len, t) in &inputs {
            if is_img {
                let f = &feats[img];
                if f.len() != len * d as usize {
                    return Err(Error::Request(format!("image {img}: {} features for {len} placeholders", f.len() / d as usize)));
                }
                g.set_f32(t, f);
                img += 1;
            } else {
                g.set_i32(t, &tokens[start..start + len].iter().map(|&t| t as i32).collect::<Vec<_>>());
            }
        }
        let (c, s) = self.text_tables(&pos)?;
        g.set_f32(cos, &c);
        g.set_f32(sin, &s);
        g.set_f16(causal, &mask(n, |q, k| k <= q));
        g.compute()?;
        Ok(g.read_f32(out))
    }

    /// Rotary tables `[tokens][head width]`: pair `i` rotates by the time,
    /// row or column position according to the section it falls in.
    fn text_tables(&self, pos: &[[f32; 3]]) -> Result<(Vec<f32>, Vec<f32>)> {
        let hd = self.cfg.head_dim() as usize;
        let half = hd / 2;
        let sec = self.cfg.sections()?;
        let axis: Vec<usize> = (0..half).map(|i| if i < sec[0] { 0 } else if i < sec[0] + sec[1] { 1 } else { 2 }).collect();
        let inv: Vec<f32> = (0..half).map(|i| (1.0 / self.cfg.rope_theta.powf((2 * i) as f64 / hd as f64)) as f32).collect();
        let mut cos = Vec::with_capacity(pos.len() * hd);
        let mut sin = Vec::with_capacity(pos.len() * hd);
        for p in pos {
            let ang: Vec<f32> = (0..half).map(|i| p[axis[i]] * inv[i]).collect();
            for _ in 0..2 {
                cos.extend(ang.iter().map(|a| a.cos()));
                sin.extend(ang.iter().map(|a| a.sin()));
            }
        }
        Ok((cos, sin))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::qwen_image::parity::{assert_close, bin};

    fn vision(extra: Value) -> Result<VisionConfig> {
        let mut v = serde_json::json!({
            "depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "out_hidden_size": 32,
            "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 112,
            "fullatt_block_indexes": [1]
        });
        v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let c: VisionConfig = serde_json::from_value(v).map_err(|e| Error::Config(e.to_string()))?;
        c.validate()?;
        Ok(c)
    }

    #[test]
    fn eight_bit_weights_keep_narrow_rows_in_float32() {
        let c = serde_json::json!({
            "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_hidden_layers": 1,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "rope_scaling": {"mrope_section": [2, 3, 3]},
            "vocab_size": 256, "image_token_id": 7,
            "vision_config": {
                "depth": 1, "hidden_size": 64, "intermediate_size": 3420, "num_heads": 4, "out_hidden_size": 64,
                "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 112,
                "fullatt_block_indexes": [0]
            }
        });
        let cfg = QwenVlConfig::from_json(c).unwrap();
        let specs = cfg.weight_specs(WType::F16, WType::Q8_0);
        let ty = |n: &str| specs.iter().find(|w| w.name == n).unwrap().ty;
        assert_eq!(ty("visual.blocks.0.mlp.down_proj.weight"), WType::F32);
        assert_eq!(ty("visual.blocks.0.mlp.up_proj.weight"), WType::Q8_0);
        assert_eq!(ty("model.layers.0.mlp.down_proj.weight"), WType::Q8_0);
    }

    #[test]
    fn vision_config_reads_either_channel_key() {
        assert_eq!(vision(serde_json::json!({"in_channels": 3, "in_chans": 3})).unwrap().in_channels(), 3);
        assert_eq!(vision(serde_json::json!({"in_chans": 4})).unwrap().in_channels(), 4);
        assert_eq!(vision(serde_json::json!({})).unwrap().in_channels(), 3);
        assert!(vision(serde_json::json!({"in_channels": 3, "in_chans": 4})).is_err());
    }

    fn dir() -> PathBuf {
        PathBuf::from(std::env::var("PRAECISE_QWEN_VL_PARITY").expect("PRAECISE_QWEN_VL_PARITY names the fixture dir"))
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = dir();
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let opts = LoadOptions { precision, cpu_threads: threads, device: None };
        let enc = QwenVlEncoder::load(&CheckpointFiles::new(d.join("checkpoint")), "text_encoder", opts).unwrap();
        let grids: Vec<(usize, usize)> =
            m["grids"].as_array().unwrap().iter().map(|g| (g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize)).collect();
        let mut images = Vec::new();
        for (i, &(gh, gw)) in grids.iter().enumerate() {
            let img = enc.image(&bin(&d, &format!("pixels_{i}")), (gh * 14, gw * 14)).unwrap();
            assert_close(&format!("patches {i}"), &img.patches, &bin(&d, &format!("patches_{i}")), 0.999_999, 1e-5);
            let tokens = enc.encode_image(&img).unwrap();
            assert_close(&format!("image tokens {i}"), &tokens, &bin(&d, &format!("tokens_{i}")), min_cos, max_rel);
            images.push(img);
        }
        let ids = |k: &str| m[k].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
        let tokens = ids("ids");
        let collapsed: Vec<u32> = tokens.iter().enumerate().filter(|&(i, &t)| t != enc.cfg.image_token_id || tokens[i - 1] != t).map(|(_, &t)| t).collect();
        assert_eq!(enc.expand_placeholders(&collapsed, &grids), tokens);
        assert_close("prompt", &enc.forward(&tokens, &images).unwrap(), &bin(&d, "hidden"), min_cos, max_rel);
        assert_close("text prompt", &enc.forward(&ids("text_ids"), &[]).unwrap(), &bin(&d, "hidden_text"), min_cos, max_rel);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen_vl_parity_f32() {
        run(Precision::F32, 0.999_999, 1e-4);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen_vl_parity_bf16() {
        run(Precision::Bf16, 0.9999, 1e-2);
    }
}
