//! Qwen3-VL as a prompt encoder: a vision transformer turns each image into
//! tokens spliced into the prompt, and features from three of its blocks are
//! added to the image tokens after the first language-model layers.
//!
//! The vision tower adds a learned position grid, interpolated bilinearly to
//! each image's patch grid, and rotates queries and keys by patch row and
//! column. Image tokens take three-axis positions (time, row, column of the
//! merged patch grid) whose rotary frequencies interleave across the axes;
//! text tokens share one position on all three axes.

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Per-channel mean and deviation the vision tower's inputs are normalised by.
pub const IMAGE_MEAN: [f32; 3] = [0.5; 3];
pub const IMAGE_STD: [f32; 3] = [0.5; 3];
const VISION_THETA: f32 = 10_000.0;
const VISION_EPS: f32 = 1e-6;

const TEXT: &str = "model.language_model";
const VISUAL: &str = "model.visual";

/// The vision tower's layout.
#[derive(Debug, Clone, Deserialize)]
pub struct Vision3Config {
    pub depth: usize,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_heads: u64,
    pub out_hidden_size: u64,
    pub patch_size: u64,
    pub spatial_merge_size: u64,
    pub temporal_patch_size: u64,
    #[serde(default = "three")]
    pub in_channels: u64,
    /// Learned position grid entries (a square).
    pub num_position_embeddings: u64,
    /// Blocks whose output also feeds the first language-model layers.
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

fn three() -> u64 {
    3
}

/// Rotary settings of the language model.
#[derive(Debug, Clone, Deserialize)]
pub struct Rope3 {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    pub mrope_section: Vec<u64>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

/// The language model's layout.
#[derive(Debug, Clone, Deserialize)]
pub struct Text3Config {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_attention_heads: u64,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_scaling: Option<Rope3>,
    #[serde(default)]
    pub rope_parameters: Option<Rope3>,
    pub vocab_size: u64,
}

/// `config.json` of a Qwen3-VL checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VlConfig {
    pub text_config: Text3Config,
    pub vision_config: Vision3Config,
    pub image_token_id: u32,
}

impl Qwen3VlConfig {
    /// Parse and check a `config.json`.
    ///
    /// # Errors
    /// A malformed or unsupported configuration.
    pub fn from_json(v: Value) -> Result<Self> {
        let c: Self = parse(v, "vision-language encoder config")?;
        c.validate()?;
        Ok(c)
    }

    fn rope(&self) -> Result<(f64, [usize; 3])> {
        let t = &self.text_config;
        let r = t.rope_parameters.as_ref().or(t.rope_scaling.as_ref()).ok_or_else(|| Error::Config("no rotary sections".into()))?;
        let theta = r.rope_theta.or(t.rope_theta).ok_or_else(|| Error::Config("no rotary base".into()))?;
        let s = <[u64; 3]>::try_from(r.mrope_section.as_slice()).map_err(|_| Error::Config("expected three rotary sections".into()))?;
        if r.mrope_interleaved == Some(false) {
            return Err(Error::Config("only interleaved rotary sections are supported".into()));
        }
        Ok((theta, s.map(|x| x as usize)))
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Config(m));
        let t = &self.text_config;
        if !t.num_attention_heads.is_multiple_of(t.num_key_value_heads) {
            return bad("query heads are not a multiple of the key/value heads".into());
        }
        let (_, s) = self.rope()?;
        if (s.iter().sum::<usize>() * 2) as u64 != t.head_dim {
            return bad(format!("rotary sections {s:?} do not cover the head width {}", t.head_dim));
        }
        let v = &self.vision_config;
        if !v.hidden_size.is_multiple_of(v.num_heads) || !(v.hidden_size / v.num_heads).is_multiple_of(4) {
            return bad("vision heads do not split into row and column halves".into());
        }
        let side = (v.num_position_embeddings as f64).sqrt() as u64;
        if side * side != v.num_position_embeddings || side < 2 {
            return bad("the position grid is not square".into());
        }
        if v.deepstack_visual_indexes.iter().any(|&i| i >= v.depth) || v.deepstack_visual_indexes.len() > t.num_hidden_layers {
            return bad("deepstack blocks outside the towers".into());
        }
        if v.hidden_act.as_deref().is_some_and(|a| a != "gelu_pytorch_tanh") {
            return bad("only the tanh GELU vision feed-forward is supported".into());
        }
        Ok(())
    }

    /// Pixel side the image sides must be multiples of.
    #[must_use]
    pub fn image_unit(&self) -> usize {
        (self.vision_config.patch_size * self.vision_config.spatial_merge_size) as usize
    }

    fn patch_in(&self) -> u64 {
        let v = &self.vision_config;
        v.in_channels * v.temporal_patch_size * v.patch_size * v.patch_size
    }

    fn weight_specs(&self, embed: WType, linear: WType) -> Vec<WeightSpec> {
        let t = &self.text_config;
        let (d, hd, ff) = (t.hidden_size, t.head_dim, t.intermediate_size);
        let (q, kv) = (t.num_attention_heads * hd, t.num_key_value_heads * hd);
        let f = WType::F32;
        // 8-bit blocks hold 32 columns; narrower rows stay float32.
        let q8 = |cols: u64| if linear == WType::Q8_0 && cols % 32 != 0 { f } else { linear };
        let mut s = vec![WeightSpec::new(format!("{TEXT}.embed_tokens.weight"), &[t.vocab_size, d], embed)];
        for i in 0..t.num_hidden_layers {
            let p = format!("{TEXT}.layers.{i}");
            s.push(WeightSpec::new(format!("{p}.input_layernorm.weight"), &[d], f));
            s.push(WeightSpec::new(format!("{p}.post_attention_layernorm.weight"), &[d], f));
            s.push(WeightSpec::new(format!("{p}.self_attn.q_norm.weight"), &[hd], f));
            s.push(WeightSpec::new(format!("{p}.self_attn.k_norm.weight"), &[hd], f));
            for (n, rows) in [("q", q), ("k", kv), ("v", kv)] {
                s.push(WeightSpec::new(format!("{p}.self_attn.{n}_proj.weight"), &[rows, d], q8(d)));
            }
            s.push(WeightSpec::new(format!("{p}.self_attn.o_proj.weight"), &[d, q], q8(q)));
            s.push(WeightSpec::new(format!("{p}.mlp.gate_proj.weight"), &[ff, d], q8(d)));
            s.push(WeightSpec::new(format!("{p}.mlp.up_proj.weight"), &[ff, d], q8(d)));
            s.push(WeightSpec::new(format!("{p}.mlp.down_proj.weight"), &[d, ff], q8(ff)));
        }
        let v = &self.vision_config;
        let (vd, vf) = (v.hidden_size, v.intermediate_size);
        let merged = vd * v.spatial_merge_size * v.spatial_merge_size;
        let lin = |s: &mut Vec<WeightSpec>, p: String, out: u64, inp: u64| {
            s.push(WeightSpec::new(format!("{p}.weight"), &[out, inp], q8(inp)));
            s.push(WeightSpec::new(format!("{p}.bias"), &[out], f));
        };
        let ln = |s: &mut Vec<WeightSpec>, p: String, c: u64| {
            s.push(WeightSpec::new(format!("{p}.weight"), &[c], f));
            s.push(WeightSpec::new(format!("{p}.bias"), &[c], f));
        };
        s.push(WeightSpec::new(format!("{VISUAL}.pos_embed.weight"), &[v.num_position_embeddings, vd], f));
        for i in 0..v.depth {
            let p = format!("{VISUAL}.blocks.{i}");
            ln(&mut s, format!("{p}.norm1"), vd);
            ln(&mut s, format!("{p}.norm2"), vd);
            lin(&mut s, format!("{p}.attn.qkv"), 3 * vd, vd);
            lin(&mut s, format!("{p}.attn.proj"), vd, vd);
            lin(&mut s, format!("{p}.mlp.linear_fc1"), vf, vd);
            lin(&mut s, format!("{p}.mlp.linear_fc2"), vd, vf);
        }
        let mergers = std::iter::once((format!("{VISUAL}.merger"), vd))
            .chain((0..v.deepstack_visual_indexes.len()).map(|k| (format!("{VISUAL}.deepstack_merger_list.{k}"), merged)));
        for (p, norm) in mergers {
            ln(&mut s, format!("{p}.norm"), norm);
            lin(&mut s, format!("{p}.linear_fc1"), merged, merged);
            lin(&mut s, format!("{p}.linear_fc2"), v.out_hidden_size, merged);
        }
        s
    }
}

/// One image prepared for the vision tower.
#[derive(Debug, Clone)]
pub struct Vl3Image {
    /// Patches `[patches][channels * frames * patch * patch]` in the tower's
    /// order (each merged 2x2 block contiguous).
    pub patches: Vec<f32>,
    /// Patch grid `(rows, cols)`.
    pub grid: (usize, usize),
}

/// Merged tokens of one image and the features its deepstack blocks add.
#[derive(Debug, Clone)]
pub struct Vl3Features {
    /// `[tokens][language width]`.
    pub tokens: Vec<f32>,
    /// One `[tokens][language width]` per deepstack block, in layer order.
    pub deepstack: Vec<Vec<f32>>,
}

/// The prompt encoder.
pub struct Qwen3VlEncoder {
    backend: Backend,
    cfg: Qwen3VlConfig,
    w: Weights,
    patch: Weights,
    exact: bool,
}

impl std::fmt::Debug for Qwen3VlEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3VlEncoder").field("backend", &self.backend.name()).finish_non_exhaustive()
    }
}

fn layer_norm(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    let h = g.norm(x, VISION_EPS);
    let h = g.mul(h, w.get(&format!("{p}.weight")));
    g.add(h, w.get(&format!("{p}.bias")))
}

fn linear(g: &mut Graph, w: &Weights, p: &str, x: Tn) -> Tn {
    g.linear_b(w.get(&format!("{p}.weight")), w.get(&format!("{p}.bias")), x)
}

/// Bilinear taps (align-corners) of `index` on a grid of `size` samples
/// over `side` learned positions: `[(tap, weight); 2]`.
fn taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = index as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
    let floor = src.floor();
    let mut out = [(0, 0.0); 2];
    for (o, slot) in out.iter_mut().enumerate() {
        let raw = floor as i64 + o as i64;
        let tap = raw.clamp(0, side as i64 - 1) as usize;
        let dist = (src - floor - o as f32).abs();
        *slot = (tap, (1.0 - dist).max(0.0));
    }
    out
}

impl Qwen3VlEncoder {
    /// Load `dir/` (its `config.json` and safetensors files) of a checkpoint.
    ///
    /// # Errors
    /// An unsupported configuration, missing weights or no usable backend.
    pub fn load(files: &CheckpointFiles, dir: &str, opts: LoadOptions) -> Result<Self> {
        Self::load_layers(files, dir, opts, None)
    }

    /// Load as [`Self::load`] but keep only the first `layers` decoder
    /// layers, so [`Self::forward`] returns the hidden state after layer
    /// `layers` (`hidden_states[layers]` of the reference), before any norm.
    ///
    /// # Errors
    /// As [`Self::load`], or a layer count of zero or beyond the model's.
    pub fn load_layers(files: &CheckpointFiles, dir: &str, opts: LoadOptions, layers: Option<usize>) -> Result<Self> {
        let mut cfg = Qwen3VlConfig::from_json(files.json(&format!("{dir}/config.json"))?)?;
        if let Some(n) = layers {
            let t = &mut cfg.text_config;
            if n == 0 || n > t.num_hidden_layers {
                return Err(Error::Config(format!("Qwen3-VL: cannot stop after {n} of {} layers", t.num_hidden_layers)));
            }
            t.num_hidden_layers = n;
        }
        let st = SafeTensors::open(&files.weights(dir)?)?;
        let backend = opts.backend()?;
        let exact = opts.precision == Precision::F32;
        let embed = if exact { WType::F32 } else { WType::F16 };
        let w = Weights::load(&backend, &st, &cfg.weight_specs(embed, opts.precision.wtype()))?;
        let v = &cfg.vision_config;
        let name = format!("{VISUAL}.patch_embed.proj");
        let shape = [v.hidden_size, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size];
        let data = st.require(&format!("{name}.weight"), &shape)?.to_f32();
        let bias = st.require(&format!("{name}.bias"), &[v.hidden_size])?.to_f32();
        let patch = Weights::from_host(
            &backend,
            &[
                HostTensor { name: format!("{name}.weight"), shape: vec![v.hidden_size, cfg.patch_in()], ty: WType::F32, data },
                HostTensor { name: format!("{name}.bias"), shape: vec![v.hidden_size], ty: WType::F32, data: bias },
            ],
        )?;
        Ok(Self { backend, cfg, w, patch, exact })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &Qwen3VlConfig {
        &self.cfg
    }

    /// The backend the weights live on.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Device bytes held.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes() + self.patch.bytes()
    }

    /// Patches of an image `[3][h][w]` with values in `[0, 1]` (sides
    /// multiples of [`Qwen3VlConfig::image_unit`]), normalised and laid out
    /// for [`Self::encode_image`].
    ///
    /// # Errors
    /// Sides that are not multiples of the merged patch.
    pub fn image(&self, pixels: &[f32], (h, w): (usize, usize)) -> Result<Vl3Image> {
        let v = &self.cfg.vision_config;
        let (p, m, tp) = (v.patch_size as usize, v.spatial_merge_size as usize, v.temporal_patch_size as usize);
        let unit = p * m;
        if h % unit != 0 || w % unit != 0 || h == 0 || w == 0 || pixels.len() != 3 * h * w {
            return Err(Error::Request(format!("image {w}x{h} is not a multiple of {unit} pixels")));
        }
        let (gh, gw) = (h / p, w / p);
        let mut out = Vec::with_capacity(gh * gw * 3 * tp * p * p);
        for (r, c) in self.patch_positions((gh, gw)) {
            let (r0, c0) = (r * p, c * p);
            for ch in 0..3 {
                for _ in 0..tp {
                    for y in 0..p {
                        for x in 0..p {
                            let px = pixels[(ch * h + r0 + y) * w + c0 + x];
                            out.push((px - IMAGE_MEAN[ch]) / IMAGE_STD[ch]);
                        }
                    }
                }
            }
        }
        Ok(Vl3Image { patches: out, grid: (gh, gw) })
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

    /// Rotary tables `[tokens][head width]`: the first quarter of the pairs
    /// rotate by the row, the second by the column, mirrored into the second
    /// half.
    fn vision_tables(&self, grid: (usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let v = &self.cfg.vision_config;
        let hd = (v.hidden_size / v.num_heads) as usize;
        let q = hd / 4;
        let inv: Vec<f32> = (0..q).map(|j| 1.0 / VISION_THETA.powf((2 * j) as f32 / (hd / 2) as f32)).collect();
        let pos = self.patch_positions(grid);
        let mut cos = Vec::with_capacity(pos.len() * hd);
        let mut sin = Vec::with_capacity(pos.len() * hd);
        for &(r, c) in &pos {
            let ang: Vec<f32> = inv.iter().map(|f| r as f32 * f).chain(inv.iter().map(|f| c as f32 * f)).collect();
            for _ in 0..2 {
                cos.extend(ang.iter().map(|a| a.cos()));
                sin.extend(ang.iter().map(|a| a.sin()));
            }
        }
        (cos, sin)
    }

    /// The four learned-grid rows and weights of every patch.
    fn interpolation(&self, (gh, gw): (usize, usize)) -> ([Vec<i32>; 4], [Vec<f32>; 4]) {
        let side = (self.cfg.vision_config.num_position_embeddings as f64).sqrt() as usize;
        let mut ids: [Vec<i32>; 4] = Default::default();
        let mut wts: [Vec<f32>; 4] = Default::default();
        for (r, c) in self.patch_positions((gh, gw)) {
            let (th, tw) = (taps(r, gh, side), taps(c, gw, side));
            for (a, &(hr, hwt)) in th.iter().enumerate() {
                for (b, &(wc, wwt)) in tw.iter().enumerate() {
                    ids[a * 2 + b].push((hr * side + wc) as i32);
                    wts[a * 2 + b].push(hwt * wwt);
                }
            }
        }
        (ids, wts)
    }

    fn attend(&self, g: &mut Graph, q: Tn, k: Tn, v: Tn, mask: Option<Tn>, scale: f32) -> Tn {
        if self.exact {
            let k = g.cont(k);
            let v = g.cont(v);
            g.attention_exact(q, k, v, mask, scale)
        } else {
            let k = g.cast(k, sys::GGML_TYPE_F16);
            let v = g.cast(v, sys::GGML_TYPE_F16);
            g.attention(q, k, v, mask, scale, true)
        }
    }

    /// Merged image tokens of one image and its deepstack features, in the
    /// order the prompt holds them (row-major over the merged grid).
    ///
    /// # Errors
    /// Patches that disagree with the grid, or a backend failure.
    pub fn encode_image(&self, img: &Vl3Image) -> Result<Vl3Features> {
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
        let mm = (m * m) as i64;
        let w = &self.w;
        let mut g = Graph::new(&self.backend)?;
        let input = g.input(sys::GGML_TYPE_F32, &[self.cfg.patch_in() as i64, ni]);
        let cos = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let sin = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let pname = format!("{VISUAL}.patch_embed.proj");
        let mut x = g.linear_b(self.patch.get(&format!("{pname}.weight")), self.patch.get(&format!("{pname}.bias")), input);
        let table = w.get(&format!("{VISUAL}.pos_embed.weight"));
        let mut interp = Vec::new();
        let mut pos: Option<Tn> = None;
        for _ in 0..4 {
            let ids = g.input(sys::GGML_TYPE_I32, &[ni]);
            let wt = g.input(sys::GGML_TYPE_F32, &[1, ni]);
            interp.push((ids, wt));
            let rows = g.get_rows(table, ids);
            let rows = g.mul(rows, wt);
            pos = Some(match pos {
                None => rows,
                Some(p) => g.add(p, rows),
            });
        }
        x = g.add(x, pos.expect("four taps"));
        let scale = 1.0 / (hd as f32).sqrt();
        let merger = |g: &mut Graph, p: &str, x: Tn, post: bool| {
            let h = if post {
                let h = g.reshape(x, &[d * mm, ni / mm]);
                layer_norm(g, w, &format!("{p}.norm"), h)
            } else {
                let h = layer_norm(g, w, &format!("{p}.norm"), x);
                g.reshape(h, &[d * mm, ni / mm])
            };
            let h = linear(g, w, &format!("{p}.linear_fc1"), h);
            let h = g.gelu_erf(h);
            linear(g, w, &format!("{p}.linear_fc2"), h)
        };
        let mut deep = Vec::new();
        for i in 0..v.depth {
            let p = format!("{VISUAL}.blocks.{i}");
            let h = layer_norm(&mut g, w, &format!("{p}.norm1"), x);
            let qkv = linear(&mut g, w, &format!("{p}.attn.qkv"), h);
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
            let o = self.attend(&mut g, q, k, vv, None, scale);
            let o = g.reshape(o, &[d, ni]);
            let o = linear(&mut g, w, &format!("{p}.attn.proj"), o);
            x = g.add(x, o);
            let h = layer_norm(&mut g, w, &format!("{p}.norm2"), x);
            let h = linear(&mut g, w, &format!("{p}.mlp.linear_fc1"), h);
            let h = if self.exact { g.gelu_tanh_exact(h) } else { g.gelu_tanh(h) };
            let h = linear(&mut g, w, &format!("{p}.mlp.linear_fc2"), h);
            x = g.add(x, h);
            if let Some(k) = v.deepstack_visual_indexes.iter().position(|&b| b == i) {
                deep.push((k, merger(&mut g, &format!("{VISUAL}.deepstack_merger_list.{k}"), x, true)));
            }
        }
        let out = merger(&mut g, &format!("{VISUAL}.merger"), x, false);
        deep.sort_by_key(|(k, _)| *k);
        let mut outputs = vec![out];
        outputs.extend(deep.iter().map(|(_, t)| *t));
        g.finish(&outputs)?;

        g.set_f32(input, &img.patches);
        let (c, s) = self.vision_tables(img.grid);
        g.set_f32(cos, &c);
        g.set_f32(sin, &s);
        let (ids, wts) = self.interpolation(img.grid);
        for (k, &(it, wt)) in interp.iter().enumerate() {
            g.set_i32(it, &ids[k]);
            g.set_f32(wt, &wts[k]);
        }
        g.compute()?;
        Ok(Vl3Features { tokens: g.read_f32(out), deepstack: deep.iter().map(|(_, t)| g.read_f32(*t)).collect() })
    }

    /// Three-axis positions of a prompt whose image placeholders are already
    /// expanded to one token per merged patch: image `k`'s tokens take the
    /// row and column of their merged patch offset by the next free
    /// position, which then advances past the larger grid side.
    fn positions(&self, tokens: &[u32], grids: &[(usize, usize)]) -> Result<Vec<[f32; 3]>> {
        let m = self.cfg.vision_config.spatial_merge_size as usize;
        let image = self.cfg.image_token_id;
        let mut out = Vec::with_capacity(tokens.len());
        let (mut next, mut i, mut img) = (0usize, 0usize, 0usize);
        while i < tokens.len() {
            if tokens[i] == image {
                let &(gh, gw) = grids.get(img).ok_or_else(|| Error::Request("more image placeholders than images".into()))?;
                let (rh, rw) = (gh / m, gw / m);
                if tokens.len() < i + rh * rw || tokens[i..i + rh * rw].iter().any(|&t| t != image) {
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

    /// Rotary tables `[tokens][head width]`: pair `i` rotates by the row
    /// position when `i % 3 == 1`, by the column when `i % 3 == 2` (within
    /// those axes' sections), and by the time position otherwise.
    fn text_tables(&self, pos: &[[f32; 3]]) -> Result<(Vec<f32>, Vec<f32>)> {
        let hd = self.cfg.text_config.head_dim as usize;
        let half = hd / 2;
        let (theta, sec) = self.cfg.rope()?;
        let axis: Vec<usize> = (0..half)
            .map(|i| match i % 3 {
                1 if i < 3 * sec[1] => 1,
                2 if i < 3 * sec[2] => 2,
                _ => 0,
            })
            .collect();
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / hd as f32)).collect();
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

    /// The last layer's hidden state `[tokens][width]`, before the final
    /// norm, of a prompt whose placeholders are expanded (see
    /// [`Self::expand_placeholders`]), with the images in placeholder order.
    ///
    /// # Errors
    /// Tokens outside the vocabulary, images that disagree with the
    /// placeholders, or a backend failure.
    pub fn forward(&self, tokens: &[u32], images: &[Vl3Image]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let t = &cfg.text_config;
        let n = tokens.len();
        if n == 0 || tokens.iter().any(|&tk| u64::from(tk) >= t.vocab_size) {
            return Err(Error::Request("prompt tokens outside the vocabulary".into()));
        }
        let grids: Vec<(usize, usize)> = images.iter().map(|i| i.grid).collect();
        let pos = self.positions(tokens, &grids)?;
        let feats: Vec<Vl3Features> = images.iter().map(|i| self.encode_image(i)).collect::<Result<_>>()?;
        let (d, hd) = (t.hidden_size as i64, t.head_dim as i64);
        let (nh, nkv) = (t.num_attention_heads as i64, t.num_key_value_heads as i64);
        let eps = t.rms_norm_eps as f32;
        let ni = n as i64;
        let w = &self.w;
        let mut g = Graph::new(&self.backend)?;
        let mut runs: Vec<(bool, usize, usize)> = Vec::new();
        for (i, &tk) in tokens.iter().enumerate() {
            let is_img = tk == cfg.image_token_id;
            match runs.last_mut() {
                Some((k, _, len)) if *k == is_img => *len += 1,
                _ => runs.push((is_img, i, 1)),
            }
        }
        let mut inputs: Vec<(bool, usize, usize, Tn)> = Vec::new();
        let mut x: Option<Tn> = None;
        for &(is_img, start, len) in &runs {
            let tn = if is_img {
                let tn = g.input(sys::GGML_TYPE_F32, &[d, len as i64]);
                inputs.push((true, start, len, tn));
                tn
            } else {
                let ids = g.input(sys::GGML_TYPE_I32, &[len as i64]);
                inputs.push((false, start, len, ids));
                g.get_rows(w.get(&format!("{TEXT}.embed_tokens.weight")), ids)
            };
            x = Some(match x {
                None => tn,
                Some(prev) => g.concat(prev, tn, 1),
            });
        }
        let Some(mut x) = x else { return Err(Error::Request("empty prompt".into())) };
        // The reference records a layer's output before the deepstack
        // features join it, so the last layer never takes them.
        let n_deep = if images.is_empty() { 0 } else { cfg.vision_config.deepstack_visual_indexes.len().min(t.num_hidden_layers - 1) };
        let deep: Vec<Tn> = (0..n_deep).map(|_| g.input(sys::GGML_TYPE_F32, &[d, ni])).collect();
        let cos = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let sin = g.input(sys::GGML_TYPE_F32, &[hd, 1, ni]);
        let causal = g.input(sys::GGML_TYPE_F16, &[ni, ni]);
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..t.num_hidden_layers {
            let p = format!("{TEXT}.layers.{i}");
            let wn = |s: &str| w.get(&format!("{p}.{s}"));
            let h = g.rms_norm(x, eps);
            let h = g.mul(h, wn("input_layernorm.weight"));
            let q = g.linear(wn("self_attn.q_proj.weight"), h);
            let k = g.linear(wn("self_attn.k_proj.weight"), h);
            let v = g.linear(wn("self_attn.v_proj.weight"), h);
            let q = g.reshape(q, &[hd, nh, ni]);
            let k = g.reshape(k, &[hd, nkv, ni]);
            let v = g.reshape(v, &[hd, nkv, ni]);
            let q = g.rms_norm(q, eps);
            let q = g.mul(q, wn("self_attn.q_norm.weight"));
            let k = g.rms_norm(k, eps);
            let k = g.mul(k, wn("self_attn.k_norm.weight"));
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
            if let Some(&ds) = deep.get(i) {
                x = g.add(x, ds);
            }
        }
        g.finish(&[x])?;

        let du = d as usize;
        let mut deep_host = vec![vec![0f32; n * du]; n_deep];
        let mut img = 0;
        for &(is_img, start, len, tn) in &inputs {
            if is_img {
                let f = &feats[img];
                if f.tokens.len() != len * du {
                    return Err(Error::Request(format!("image {img}: {} features for {len} placeholders", f.tokens.len() / du)));
                }
                g.set_f32(tn, &f.tokens);
                for (k, host) in deep_host.iter_mut().enumerate() {
                    host[start * du..(start + len) * du].copy_from_slice(&f.deepstack[k]);
                }
                img += 1;
            } else {
                g.set_i32(tn, &tokens[start..start + len].iter().map(|&tk| tk as i32).collect::<Vec<_>>());
            }
        }
        for (tn, host) in deep.iter().zip(&deep_host) {
            g.set_f32(*tn, host);
        }
        let (c, s) = self.text_tables(&pos)?;
        g.set_f32(cos, &c);
        g.set_f32(sin, &s);
        let mut causal_mask = vec![0f32; n * n];
        for q in 0..n {
            for k in q + 1..n {
                causal_mask[q * n + k] = f32::NEG_INFINITY;
            }
        }
        g.set_f16(causal, &causal_mask);
        g.compute()?;
        Ok(g.read_f32(x))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::qwen_image::parity::{assert_close, bin};

    #[test]
    fn grid_taps_meet_the_corners() {
        assert_eq!(taps(0, 8, 4), [(0, 1.0), (1, 0.0)]);
        let last = taps(7, 8, 4);
        assert_eq!(last[0], (3, 1.0));
        let mid = taps(1, 3, 4);
        assert!((mid[0].1 - 0.5).abs() < 1e-6 && mid[0].0 == 1 && mid[1].0 == 2);
    }

    fn dir() -> PathBuf {
        PathBuf::from(std::env::var("PRAECISE_QWEN3_VL_PARITY").expect("PRAECISE_QWEN3_VL_PARITY names the fixture dir"))
    }

    fn run(precision: Precision, min_cos: f64, max_rel: f64) {
        let d = dir();
        let m: Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let opts = LoadOptions { precision, cpu_threads: threads, device: None };
        let enc = Qwen3VlEncoder::load(&CheckpointFiles::new(d.join("checkpoint")), "text_encoder", opts).unwrap();
        let unit = enc.cfg.vision_config.patch_size as usize;
        let grids: Vec<(usize, usize)> =
            m["grids"].as_array().unwrap().iter().map(|g| (g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize)).collect();
        let mut images = Vec::new();
        for (i, &(gh, gw)) in grids.iter().enumerate() {
            let img = enc.image(&bin(&d, &format!("pixels_{i}")), (gh * unit, gw * unit)).unwrap();
            assert_close(&format!("patches {i}"), &img.patches, &bin(&d, &format!("patches_{i}")), 0.999_999, 1e-5);
            let f = enc.encode_image(&img).unwrap();
            assert_close(&format!("image tokens {i}"), &f.tokens, &bin(&d, &format!("tokens_{i}")), min_cos, max_rel);
            for (k, ds) in f.deepstack.iter().enumerate() {
                assert_close(&format!("deepstack {k} of image {i}"), ds, &bin(&d, &format!("deepstack_{i}_{k}")), min_cos, max_rel);
            }
            images.push(img);
        }
        let ids = |k: &str| m[k].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<_>>();
        let tokens = ids("ids");
        let image = enc.cfg.image_token_id;
        let collapsed: Vec<u32> = tokens.iter().enumerate().filter(|&(i, &t)| t != image || tokens[i - 1] != t).map(|(_, &t)| t).collect();
        assert_eq!(enc.expand_placeholders(&collapsed, &grids), tokens);
        assert_close("prompt", &enc.forward(&tokens, &images).unwrap(), &bin(&d, "hidden"), min_cos, max_rel);
        assert_close("text prompt", &enc.forward(&ids("text_ids"), &[]).unwrap(), &bin(&d, "hidden_text"), min_cos, max_rel);
        let at = m["layer"].as_u64().unwrap() as usize;
        let cut = Qwen3VlEncoder::load_layers(&CheckpointFiles::new(d.join("checkpoint")), "text_encoder", opts, Some(at)).unwrap();
        assert_close("prompt at a layer", &cut.forward(&tokens, &images).unwrap(), &bin(&d, "hidden_at"), min_cos, max_rel);
        assert_close("text prompt at a layer", &cut.forward(&ids("text_ids"), &[]).unwrap(), &bin(&d, "hidden_text_at"), min_cos, max_rel);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen3_vl_parity_f32() {
        run(Precision::F32, 0.999_999, 1e-4);
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn qwen3_vl_parity_bf16() {
        run(Precision::Bf16, 0.9999, 1e-2);
    }
}
