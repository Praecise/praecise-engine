//! FLUX 3 multi-stream flow transformer.
//!
//! One network jointly denoises any set of content streams (video latents,
//! action chunks, their conditioning counterparts) given a text context.
//! Every token carries its own timestep, so clean conditioning tokens and
//! noised targets share one sequence. Each stream first runs through its own
//! parallel attention + MLP blocks (attention within the stream), then all
//! active streams and the text run through shared single-stream blocks with
//! joint attention, and each stream leaves through its own output head.
//!
//! Action policies built on it add an action stream (and its conditioning
//! stream) beside the video streams; the configuration is read from the
//! checkpoint itself, so any embodiment head loads without a config file.

use crate::error::{Error, Result};
use crate::flux2::{axis_rope_freq_factors, rope_head_order, rope_positions, timestep_features};
use crate::ggml::{Backend, Graph, HostTensor, Tn, WType, WeightSpec, Weights};
use crate::pipeline::{LoadOptions, Precision};
use crate::safetensors::SafeTensors;
use llama_cpp_sys_2 as sys;

/// Width of the sinusoidal timestep features.
pub const TIME_FEATURES: usize = 256;
/// Timesteps in `[0, 1]` are scaled by this before the sinusoid.
pub const TIME_FACTOR: f32 = 1000.0;
const EPS: f32 = 1e-6;
const THETA: f64 = 10000.0;

/// Hyper-parameters, all read from checkpoint tensor shapes.
#[derive(Debug, Clone, PartialEq)]
pub struct Flux3Config {
    /// Model width.
    pub hidden_size: u64,
    /// Attention head width.
    pub head_dim: u64,
    /// MLP hidden width.
    pub mlp_hidden: u64,
    /// Per-stream blocks before the joint blocks.
    pub depth: usize,
    /// Joint single-stream blocks.
    pub depth_single_blocks: usize,
    /// Per-stream blocks after the joint blocks.
    pub depth_late_blocks: usize,
    /// Text context width.
    pub context_in_dim: u64,
    /// Width of the global conditioning vector, if the model takes one.
    pub vec_in_dim: Option<u64>,
    /// Content streams and their token widths, sorted by name.
    pub streams: Vec<(String, u64)>,
    /// The streams with an output head (a prediction), sorted by name; the
    /// others only condition.
    pub outputs: Vec<String>,
    /// Rotary widths of the four position axes `(t, h, w, l)`.
    pub axes_dim: [u64; 4],
    /// A flow map: each evaluation also takes the time the step jumps to,
    /// embedded by its own MLP and added to every token's conditioning.
    pub flow_map: bool,
}

impl Flux3Config {
    /// Read the configuration from tensor names and shapes.
    ///
    /// # Errors
    /// When a required tensor is missing or the layout is one this
    /// implementation does not cover (biased or gated projections).
    pub fn from_tensors(st: &SafeTensors) -> Result<Self> {
        let shape = |n: &str| -> Result<Vec<u64>> { st.get(n).map(|v| v.shape.to_vec()).ok_or_else(|| Error::MissingTensor(n.into())) };
        let txt_in = shape("txt_in.weight")?;
        let hidden_size = txt_in[0];
        let head_dim = shape("single_blocks.0.norm.query_norm.scale")?[0];
        let mlp_hidden = shape("single_blocks.0.mlp_out.weight")?[1];
        let count = |prefix: &str| (0..).take_while(|i| st.get(&format!("{prefix}.{i}.q_proj.weight")).is_some()).count();
        let mut streams: Vec<(String, u64)> = st
            .names()
            .filter_map(|n| n.strip_prefix("emb_in.").and_then(|r| r.strip_suffix(".weight")).map(str::to_owned))
            .map(|m| shape(&format!("emb_in.{m}.weight")).map(|s| (m, s[1])))
            .collect::<Result<_>>()?;
        streams.sort();
        if st.names().any(|n| n.ends_with(".gate_proj.weight") || (n.ends_with("_proj.bias"))) {
            return Err(Error::Config("gated or biased attention projections are not supported".into()));
        }
        if head_dim % 8 != 0 || hidden_size % head_dim != 0 {
            return Err(Error::Config(format!("head width {head_dim} does not split hidden width {hidden_size}")));
        }
        let mut cfg = Self {
            hidden_size,
            head_dim,
            mlp_hidden,
            depth: count("txt_mode_blocks"),
            depth_single_blocks: count("single_blocks"),
            depth_late_blocks: count("late_txt_mode_blocks"),
            context_in_dim: txt_in[1],
            vec_in_dim: st.get("vector_in.in_layer.weight").map(|v| v.shape[1]),
            streams,
            axes_dim: [head_dim / 4; 4],
            flow_map: st.get("ref_time_in.in_layer.weight").is_some(),
            outputs: Vec::new(),
        };
        cfg.outputs = cfg.streams.iter().map(|(m, _)| m.clone()).filter(|m| st.get(&format!("final_layer.{m}.linear.weight")).is_some()).collect();
        if cfg.streams.is_empty() || cfg.outputs.is_empty() || cfg.depth_single_blocks == 0 {
            return Err(Error::Config("checkpoint has no content streams, no output head or no single-stream blocks".into()));
        }
        Ok(cfg)
    }

    /// Attention heads.
    #[must_use]
    pub fn heads(&self) -> u64 {
        self.hidden_size / self.head_dim
    }

    /// Token width of `stream`.
    #[must_use]
    pub fn channels(&self, stream: &str) -> Option<u64> {
        self.streams.iter().find(|(m, _)| m == stream).map(|(_, c)| *c)
    }

    fn block_specs(&self, v: &mut Vec<WeightSpec>, p: &str, linear: WType) {
        let (d, h, hd) = (self.hidden_size, self.mlp_hidden, self.head_dim as usize);
        let head = rope_head_order(hd);
        let du = d as usize;
        // Query and key heads in rotary order (see `rope_head_order`).
        let mut rows: Vec<usize> = (0..3 * du).collect();
        for part in 0..2 {
            for hi in 0..du / hd {
                for (new, old) in head.iter().enumerate() {
                    rows[part * du + hi * hd + new] = part * du + hi * hd + old;
                }
            }
        }
        let parts: Vec<String> = ["q_proj", "k_proj", "v_proj"].iter().map(|n| format!("{p}.{n}.weight")).collect();
        v.push(WeightSpec::stacked(format!("{p}.qkv"), &parts, &[d, d], linear).with_rows(rows));
        v.push(WeightSpec::new(format!("{p}.mlp_in.weight"), &[2 * h, d], linear));
        v.push(WeightSpec::new(format!("{p}.attn_out.weight"), &[d, d], linear));
        v.push(WeightSpec::new(format!("{p}.mlp_out.weight"), &[d, h], linear));
        for n in ["query_norm", "key_norm"] {
            v.push(WeightSpec::new(format!("{p}.norm.{n}.scale"), &[self.head_dim], WType::F32).with_rows(head.clone()));
        }
    }

    /// Every weight, named as in the checkpoint.
    #[must_use]
    pub fn weight_specs(&self, linear: WType) -> Vec<WeightSpec> {
        let d = self.hidden_size;
        let mut v = vec![
            WeightSpec::new("txt_in.weight", &[d, self.context_in_dim], linear),
            WeightSpec::new("time_in.in_layer.weight", &[d, TIME_FEATURES as u64], WType::F32),
            WeightSpec::new("time_in.out_layer.weight", &[d, d], WType::F32),
        ];
        if self.flow_map {
            v.push(WeightSpec::new("ref_time_in.in_layer.weight", &[d, TIME_FEATURES as u64], WType::F32));
            v.push(WeightSpec::new("ref_time_in.out_layer.weight", &[d, d], WType::F32));
        }
        if let Some(vd) = self.vec_in_dim {
            v.push(WeightSpec::new("vector_in.in_layer.weight", &[d, vd], WType::F32));
            v.push(WeightSpec::new("vector_in.out_layer.weight", &[d, d], WType::F32));
        }
        let mut mods = vec!["early_stream_modulations", "single_stream_modulations"];
        if self.depth_late_blocks > 0 {
            mods.push("late_stream_modulations");
        }
        let names: Vec<&str> = self.streams.iter().map(|(m, _)| m.as_str()).chain(["txt"]).collect();
        for kind in &mods {
            for m in &names {
                v.push(WeightSpec::new(format!("{kind}.{m}.lin.weight"), &[3 * d, d], linear));
            }
        }
        for (m, c) in &self.streams {
            v.push(WeightSpec::new(format!("emb_in.{m}.weight"), &[d, *c], WType::F32));
            if self.outputs.contains(m) {
                v.push(WeightSpec::new(format!("final_layer.{m}.adaLN_modulation.1.weight"), &[2 * d, d], linear));
                v.push(WeightSpec::new(format!("final_layer.{m}.linear.weight"), &[*c, d], WType::F32));
            }
            for i in 0..self.depth {
                self.block_specs(&mut v, &format!("content_mode_blocks.{m}.{i}"), linear);
            }
            for i in 0..self.depth_late_blocks {
                self.block_specs(&mut v, &format!("late_content_mode_blocks.{m}.{i}"), linear);
            }
        }
        for i in 0..self.depth {
            self.block_specs(&mut v, &format!("txt_mode_blocks.{i}"), linear);
        }
        for i in 0..self.depth_late_blocks {
            self.block_specs(&mut v, &format!("late_txt_mode_blocks.{i}"), linear);
        }
        for i in 0..self.depth_single_blocks {
            self.block_specs(&mut v, &format!("single_blocks.{i}"), linear);
        }
        v
    }
}

/// One content stream of a forward call.
#[derive(Debug, Clone, Copy)]
pub struct StreamInput<'a> {
    /// Stream name, as in the checkpoint (`video`, `video_cond`, an action stream, ...).
    pub name: &'a str,
    /// Tokens `[n][channels]`.
    pub tokens: &'a [f32],
    /// Per-token positions `(t, h, w, l)`.
    pub ids: &'a [[i32; 4]],
    /// Per-token timesteps in `[0, 1]` (0 = clean).
    pub timesteps: &'a [f32],
}

/// The text context of a forward call.
#[derive(Debug, Clone, Copy)]
pub struct TextInput<'a> {
    /// Context features `[n][context_in_dim]`.
    pub tokens: &'a [f32],
    /// Per-token positions `(t, h, w, l)`.
    pub ids: &'a [[i32; 4]],
    /// Per-token timesteps (0 for a clean context).
    pub timesteps: &'a [f32],
}

/// Per-token shift, `1 + scale` and gate, each `[d, n]`.
struct Mod {
    shift: Tn,
    scale1: Tn,
    gate: Tn,
}

/// `x` through the linear weight `name`, then through the per-row scale a
/// block-quantised weight carries beside its blocks (`<name>.out_scale`).
fn linear(g: &mut Graph, w: &Weights, scales: Option<&Weights>, name: &str, x: Tn) -> Tn {
    let y = g.linear(w.get(name), x);
    match scales.and_then(|s| s.find(&format!("{name}.out_scale"))) {
        Some(s) => g.mul(y, s),
        None => y,
    }
}

fn modulation(g: &mut Graph, c: &Ctx<'_>, name: &str, vec_act: Tn, d: i64) -> Mod {
    let m = linear(g, c.w, c.scales, name, vec_act);
    let shift = g.view_rows(m, 0, d);
    let shift = g.cont(shift);
    let scale = g.view_rows(m, d, d);
    let scale = g.cont(scale);
    let scale1 = g.scale_bias(scale, 1.0, 1.0);
    let gate = g.view_rows(m, 2 * d, d);
    let gate = g.cont(gate);
    Mod { shift, scale1, gate }
}

fn concat_mods(g: &mut Graph, mods: &[&Mod]) -> Mod {
    let cat = |g: &mut Graph, f: fn(&Mod) -> Tn| mods[1..].iter().fold(f(mods[0]), |acc, m| g.concat(acc, f(m), 1));
    Mod { shift: cat(g, |m| m.shift), scale1: cat(g, |m| m.scale1), gate: cat(g, |m| m.gate) }
}

struct Ctx<'a> {
    cfg: &'a Flux3Config,
    w: &'a Weights,
    scales: Option<&'a Weights>,
    freq_factors: Tn,
    exact: bool,
}

/// `x + gate * (attn_out(attention) + mlp_out(swiglu(mlp_in)))` on the
/// modulated pre-norm of `x`; attention runs over all of `x`'s tokens.
fn block(g: &mut Graph, c: &Ctx<'_>, p: &str, x: Tn, pos: Tn, m: &Mod) -> Tn {
    let cfg = c.cfg;
    let (d, hd) = (cfg.hidden_size as i64, cfg.head_dim as i64);
    let heads = d / hd;
    let wn = |n: &str| c.w.get(&format!("{p}.{n}"));
    let lin = |g: &mut Graph, n: &str, x: Tn| linear(g, c.w, c.scales, &format!("{p}.{n}"), x);
    let xn = g.norm(x, EPS);
    let xn = g.mul(xn, m.scale1);
    let xn = g.add(xn, m.shift);
    let qkv = lin(g, "qkv", xn);
    let q = g.view_heads(qkv, 0, hd, heads);
    let k = g.view_heads(qkv, d, hd, heads);
    let v = g.view_heads(qkv, 2 * d, hd, heads);
    let q = g.rms_norm(q, EPS);
    let q = g.mul(q, wn("norm.query_norm.scale"));
    let k = g.rms_norm(k, EPS);
    let k = g.mul(k, wn("norm.key_norm.scale"));
    let mut sections = [0i32; 4];
    for (s, a) in sections.iter_mut().zip(&cfg.axes_dim) {
        *s = (*a / 2) as i32;
    }
    let q = g.rope_multi(q, pos, c.freq_factors, hd as i32, sections, THETA as f32);
    let k = g.rope_multi(k, pos, c.freq_factors, hd as i32, sections, THETA as f32);
    let v = g.cont(v);
    let q = g.permute(q, [0, 2, 1, 3]);
    let k = g.permute(k, [0, 2, 1, 3]);
    let v = g.permute(v, [0, 2, 1, 3]);
    let scale = 1.0 / (hd as f32).sqrt();
    let o = if c.exact {
        g.attention_exact(q, k, v, None, scale)
    } else {
        let k = g.cast(k, sys::GGML_TYPE_F16);
        let v = g.cast(v, sys::GGML_TYPE_F16);
        g.attention(q, k, v, None, scale, false)
    };
    let n = o.ne(2);
    let o = g.reshape(o, &[d, n]);
    let o = lin(g, "attn_out.weight", o);
    let f = lin(g, "mlp_in.weight", xn);
    let f = g.swiglu(f);
    let f = lin(g, "mlp_out.weight", f);
    let out = g.add(o, f);
    let out = g.mul(out, m.gate);
    g.add(x, out)
}

/// Graph inputs of one stream (or of the text).
#[derive(Debug, Clone, Copy)]
struct StreamIo {
    tokens: Tn,
    t_feat: Tn,
    pos: Tn,
}

/// The loaded transformer.
pub struct Flux3Transformer {
    backend: Backend,
    cfg: Flux3Config,
    w: Weights,
    /// Per-row output scales of block-quantised weights, when any carry one.
    scales: Option<Weights>,
    position_scale: [f32; 4],
    exact: bool,
}

impl std::fmt::Debug for Flux3Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flux3Transformer").field("device", &self.backend.name()).field("streams", &self.cfg.streams).finish_non_exhaustive()
    }
}

impl Flux3Transformer {
    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.bytes() + self.scales.as_ref().map_or(0, Weights::bytes)
    }

    /// Name of the compute device.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    /// Load from safetensors files whose tensor names may carry `prefix`
    /// (stripped; tensors without it are ignored).
    ///
    /// # Errors
    /// On an unsupported layout, missing weights or no usable backend.
    pub fn load(files: &[std::path::PathBuf], prefix: &str, opts: LoadOptions) -> Result<Self> {
        let st = SafeTensors::open(files)?.renamed(|n| n.strip_prefix(prefix).map(str::to_owned))?;
        Self::from_safetensors(st, opts.precision.wtype(), opts)
    }

    /// Load from opened tensors named as in the released checkpoints, storing
    /// the projection matrices as `linear`. Quantised checkpoints load too:
    /// NVFP4 matrices (ModelOpt layout: packed E2M1 pairs, E4M3 scales per 16
    /// values, a float32 scale per tensor) keep their 4-bit blocks, and the
    /// per-tensor scale is applied to the product; float8 matrices with a
    /// per-row or per-tensor scale are scaled and stored as `linear`.
    ///
    /// # Errors
    /// On an unsupported layout, missing weights or no usable backend.
    pub fn from_safetensors(st: SafeTensors, linear: WType, opts: LoadOptions) -> Result<Self> {
        let st = st.nvfp4_to_blocks()?.dequantize_scaled_f8()?;
        let cfg = Flux3Config::from_tensors(&st)?;
        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), streams = ?cfg.streams, flow_map = cfg.flow_map, "multi-stream transformer backend selected");
        let specs = cfg.weight_specs(linear);
        let scales = output_scales(&st, &specs)?;
        let w = Weights::load(&backend, &st, &specs)?;
        let scales = if scales.is_empty() { None } else { Some(Weights::from_host(&backend, &scales)?) };
        Ok(Self { backend, cfg, w, scales, position_scale: [1.0; 4], exact: opts.precision == Precision::F32 })
    }

    /// Scale every token's position on each axis `(t, h, w, l)` by these
    /// factors before the rotary embedding (fractional positions, which an
    /// integer position grid cannot hold, become a change of rotary
    /// frequency on their axis).
    #[must_use]
    pub fn with_position_scale(mut self, scale: [f32; 4]) -> Self {
        self.position_scale = scale;
        self
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &Flux3Config {
        &self.cfg
    }

    /// One evaluation: the prediction for every stream in `streams`, in the
    /// same order, each `[n][channels]` (empty for a stream without an
    /// output head). Streams absent from `streams` take no part; the joint
    /// sequence is the text followed by `streams` in order.
    ///
    /// # Errors
    /// On an unknown stream, mismatched input lengths, a flow-map model
    /// (which needs [`Flux3Transformer::forward_to`]) or a backend failure.
    pub fn forward(&self, text: TextInput<'_>, vector: Option<&[f32]>, streams: &[StreamInput<'_>]) -> Result<Vec<Vec<f32>>> {
        self.forward_to(text, vector, streams, None)
    }

    /// [`Flux3Transformer::forward`] for a flow map, which also takes
    /// `target`, the time in `[0, 1]` the step jumps to. `target` is given
    /// to, and only to, a flow-map model.
    ///
    /// # Errors
    /// As [`Flux3Transformer::forward`], or a `target` the model does not
    /// take (or a missing one it needs).
    pub fn forward_to(&self, text: TextInput<'_>, vector: Option<&[f32]>, streams: &[StreamInput<'_>], target: Option<f32>) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.cfg;
        if cfg.flow_map != target.is_some() {
            return Err(Error::Request(if cfg.flow_map { "a flow-map model needs the time each step jumps to".into() } else { "only a flow-map model takes a target time".into() }));
        }
        let d = cfg.hidden_size as i64;
        let n_txt = text.ids.len();
        if n_txt == 0 || text.tokens.len() != n_txt * cfg.context_in_dim as usize || text.timesteps.len() != n_txt {
            return Err(Error::Request("text context disagrees with its ids or timesteps".into()));
        }
        let mut chans = Vec::with_capacity(streams.len());
        for s in streams {
            let c = cfg.channels(s.name).ok_or_else(|| Error::Request(format!("unknown stream {}", s.name)))? as usize;
            let n = s.ids.len();
            if n == 0 || s.tokens.len() != n * c || s.timesteps.len() != n {
                return Err(Error::Request(format!("stream {} disagrees with its ids or timesteps", s.name)));
            }
            if streams.iter().filter(|o| o.name == s.name).count() > 1 {
                return Err(Error::Request(format!("stream {} given twice", s.name)));
            }
            chans.push(c);
        }
        if vector.map(<[f32]>::len) != cfg.vec_in_dim.map(|v| v as usize) {
            return Err(Error::Request("global conditioning vector disagrees with the model".into()));
        }
        if streams.is_empty() {
            return Err(Error::Request("no content streams".into()));
        }

        let mut g = Graph::new(&self.backend)?;
        let freq_factors = g.input(sys::GGML_TYPE_F32, &[cfg.head_dim as i64 / 2]);
        let c = Ctx { cfg, w: &self.w, scales: self.scales.as_ref(), freq_factors, exact: self.exact };
        let w = &self.w;
        let lin = |g: &mut Graph, name: &str, x: Tn| linear(g, &self.w, self.scales.as_ref(), name, x);
        let input = |g: &mut Graph, width: i64, n: usize| StreamIo {
            tokens: g.input(sys::GGML_TYPE_F32, &[width, n as i64]),
            t_feat: g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64, n as i64]),
            pos: g.input(sys::GGML_TYPE_I32, &[4 * n as i64]),
        };
        let txt_io = input(&mut g, cfg.context_in_dim as i64, n_txt);
        let s_io: Vec<StreamIo> = streams.iter().zip(&chans).map(|(s, &ch)| input(&mut g, ch as i64, s.ids.len())).collect();
        let n_all: usize = n_txt + streams.iter().map(|s| s.ids.len()).sum::<usize>();
        let joint_pos = g.input(sys::GGML_TYPE_I32, &[4 * n_all as i64]);
        let vec_in = cfg.vec_in_dim.map(|vd| g.input(sys::GGML_TYPE_F32, &[vd as i64]));
        let target_feat = target.map(|_| g.input(sys::GGML_TYPE_F32, &[TIME_FEATURES as i64]));
        let target_emb = target_feat.map(|f| {
            let h = g.linear(w.get("ref_time_in.in_layer.weight"), f);
            let h = g.silu(h);
            g.linear(w.get("ref_time_in.out_layer.weight"), h)
        });

        let vector_emb = vec_in.map(|v| {
            let h = g.linear(w.get("vector_in.in_layer.weight"), v);
            let h = g.silu(h);
            g.linear(w.get("vector_in.out_layer.weight"), h)
        });
        // Per-token conditioning: time embedding plus the global vector, then SiLU.
        let cond = |g: &mut Graph, t_feat: Tn| {
            let h = g.linear(w.get("time_in.in_layer.weight"), t_feat);
            let h = g.silu(h);
            let mut v = g.linear(w.get("time_in.out_layer.weight"), h);
            if let Some(e) = vector_emb {
                v = g.add(v, e);
            }
            if let Some(e) = target_emb {
                v = g.add(v, e);
            }
            v
        };
        let txt_vec = cond(&mut g, txt_io.t_feat);
        let s_vec: Vec<Tn> = s_io.iter().map(|io| cond(&mut g, io.t_feat)).collect();
        let txt_act = g.silu(txt_vec);
        let s_act: Vec<Tn> = s_vec.iter().map(|&v| g.silu(v)).collect();
        let mods = |g: &mut Graph, kind: &str| -> (Mod, Vec<Mod>) {
            let t = modulation(g, &c, &format!("{kind}.txt.lin.weight"), txt_act, d);
            let s = streams.iter().zip(&s_act).map(|(s, &a)| modulation(g, &c, &format!("{kind}.{}.lin.weight", s.name), a, d)).collect();
            (t, s)
        };
        let (early_txt, early_s) = mods(&mut g, "early_stream_modulations");
        let (single_txt, single_s) = mods(&mut g, "single_stream_modulations");

        let mut txt = lin(&mut g, "txt_in.weight", txt_io.tokens);
        let mut xs: Vec<Tn> = streams.iter().zip(&s_io).map(|(s, io)| lin(&mut g, &format!("emb_in.{}.weight", s.name), io.tokens)).collect();
        for i in 0..cfg.depth {
            for (j, s) in streams.iter().enumerate() {
                xs[j] = block(&mut g, &c, &format!("content_mode_blocks.{}.{i}", s.name), xs[j], s_io[j].pos, &early_s[j]);
            }
            txt = block(&mut g, &c, &format!("txt_mode_blocks.{i}"), txt, txt_io.pos, &early_txt);
        }
        let mut all = vec![&single_txt];
        all.extend(single_s.iter());
        let joint_mod = concat_mods(&mut g, &all);
        let mut joint = xs.iter().fold(txt, |acc, &x| g.concat(acc, x, 1));
        for i in 0..cfg.depth_single_blocks {
            joint = block(&mut g, &c, &format!("single_blocks.{i}"), joint, joint_pos, &joint_mod);
        }
        let mut off = n_txt as i64;
        for (j, s) in streams.iter().enumerate() {
            let n = s.ids.len() as i64;
            let v = g.view_cols(joint, off, n);
            xs[j] = g.cont(v);
            off += n;
        }
        if cfg.depth_late_blocks > 0 {
            let (_, late_s) = mods(&mut g, "late_stream_modulations");
            for i in 0..cfg.depth_late_blocks {
                for (j, s) in streams.iter().enumerate() {
                    xs[j] = block(&mut g, &c, &format!("late_content_mode_blocks.{}.{i}", s.name), xs[j], s_io[j].pos, &late_s[j]);
                }
            }
        }
        let mut outs = Vec::with_capacity(streams.len());
        let mut heads = Vec::with_capacity(streams.len());
        for (j, s) in streams.iter().enumerate() {
            heads.push(cfg.outputs.iter().any(|m| m == s.name).then_some(outs.len()));
            if heads[j].is_none() {
                continue;
            }
            // Final head: shift then scale, both from the stream's own conditioning.
            let m = lin(&mut g, &format!("final_layer.{}.adaLN_modulation.1.weight", s.name), s_act[j]);
            let shift = g.view_rows(m, 0, d);
            let shift = g.cont(shift);
            let scale = g.view_rows(m, d, d);
            let scale = g.cont(scale);
            let scale1 = g.scale_bias(scale, 1.0, 1.0);
            let xn = g.norm(xs[j], EPS);
            let xn = g.mul(xn, scale1);
            let xn = g.add(xn, shift);
            outs.push(lin(&mut g, &format!("final_layer.{}.linear.weight", s.name), xn));
        }
        g.finish(&outs)?;

        let mut factors = axis_rope_freq_factors(&cfg.axes_dim, cfg.head_dim, THETA);
        let mut pair = 0;
        for (axis, width) in cfg.axes_dim.iter().enumerate() {
            for _ in 0..width / 2 {
                factors[pair] /= self.position_scale[axis];
                pair += 1;
            }
        }
        g.set_f32(freq_factors, &factors);
        if let (Some(io), Some(t)) = (target_feat, target) {
            g.set_f32(io, &timestep_features(t * TIME_FACTOR, TIME_FEATURES));
        }
        let feats = |ts: &[f32]| ts.iter().flat_map(|&t| timestep_features(t * TIME_FACTOR, TIME_FEATURES)).collect::<Vec<f32>>();
        let pos = |ids: &[[i32; 4]]| rope_positions(&ids.iter().map(|p| p.map(|v| v as f32)).collect::<Vec<_>>());
        g.set_f32(txt_io.tokens, text.tokens);
        g.set_f32(txt_io.t_feat, &feats(text.timesteps));
        g.set_i32(txt_io.pos, &pos(text.ids));
        let mut all_ids: Vec<[i32; 4]> = text.ids.to_vec();
        for (s, io) in streams.iter().zip(&s_io) {
            g.set_f32(io.tokens, s.tokens);
            g.set_f32(io.t_feat, &feats(s.timesteps));
            g.set_i32(io.pos, &pos(s.ids));
            all_ids.extend_from_slice(s.ids);
        }
        g.set_i32(joint_pos, &pos(&all_ids));
        if let (Some(io), Some(v)) = (vec_in, vector) {
            g.set_f32(io, v);
        }
        g.compute()?;
        Ok(heads.iter().map(|h| h.map_or_else(Vec::new, |i| g.read_f32(outs[i]))).collect())
    }
}

/// Per-row output scales of the NVFP4 matrices among `specs`, one host
/// tensor `<spec>.out_scale` each: a stacked matrix takes each part's
/// per-tensor scale over that part's rows, in the spec's device row order.
fn output_scales(st: &SafeTensors, specs: &[WeightSpec]) -> Result<Vec<HostTensor>> {
    let mut out = Vec::new();
    for spec in specs {
        let parts: Vec<&str> = if spec.parts.is_empty() { vec![spec.name.as_str()] } else { spec.parts.iter().map(String::as_str).collect() };
        let scale = |p: &str| -> Option<f32> {
            let layer = p.strip_suffix(".weight")?;
            st.get(&format!("{layer}.weight_scale_2")).filter(|v| v.numel() == 1).map(|v| v.to_f32()[0])
        };
        let found: Vec<Option<f32>> = parts.iter().map(|p| scale(p)).collect();
        if found.iter().all(Option::is_none) {
            continue;
        }
        let Some(per_part) = found.into_iter().collect::<Option<Vec<f32>>>() else {
            return Err(Error::Weights(format!("{}: only some stacked parts carry a tensor scale", spec.name)));
        };
        let rows = spec.shape[0] as usize;
        let per = rows / parts.len();
        let file_order: Vec<f32> = (0..rows).map(|r| per_part[r / per]).collect();
        let data = match &spec.rows {
            Some(order) => order.iter().map(|&old| file_order[old]).collect(),
            None => file_order,
        };
        out.push(HostTensor { name: format!("{}.out_scale", spec.name), shape: vec![rows as u64], ty: WType::F32, data });
    }
    Ok(out)
}

pub mod image;
pub mod packing;
pub mod policy;
pub mod release;
pub mod sampling;
pub mod text_encoder;
pub mod video_vae;

#[cfg(test)]
mod parity;
