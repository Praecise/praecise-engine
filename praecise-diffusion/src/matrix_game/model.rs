//! The action world model as a [`WorldModel`]: the world transformer beside
//! the Wan2.2 text encoder and autoencoder.
//!
//! A session starts from a first frame (latent 0). The first chunk is
//! generated after it; every later chunk continues the last
//! [`CONTINUITY`] latent frames of the previous one and also sees
//! [`VIEW_PICKS`] earlier latent frames chosen by field of view along the
//! camera path, each with its own camera rays. Action row `f` moves the
//! camera from pixel frame `f` to `f + 1`; frame 0 has no action leading
//! into it, so a neutral row stands for it.

use serde_json::Value;

use super::camera::{self, MemoryPick, Pose};
use super::{action_rotary, config, patchify_rays, rename, windows};
use crate::error::{Error, Result};
use crate::ggml::{Backend, Graph, Weights};
use crate::pipeline::{parse, CheckpointFiles, LoadOptions, Precision, RgbImage};
use crate::safetensors::SafeTensors;
use crate::schedule;
use crate::umt5::Umt5Config;
use crate::unipc::{flow_sigmas_schedule, UniPc, UniPcConfig};
use crate::wan::{self, WanVaeConfig};
use crate::wan_dit::{self, WanDitConfig};
use crate::wan_video::{place_frame, text_states, tokens, zero_frames, TEXT_TOKENS};
use crate::world::{ChunkRequest, LatentFrame, WorldModel, FRAMES_PER_LATENT};

/// Latent frames of the previous chunk each later chunk continues.
pub const CONTINUITY: usize = 4;
/// Earlier latent frames chosen by field of view for each later chunk.
pub const VIEW_PICKS: usize = 5;
/// Pixels per latent cell side.
const STRIDE: usize = 16;

type Mat4 = [[f64; 4]; 4];

/// A loaded action world model.
pub struct MatrixGame {
    backend: Backend,
    cfg: WanDitConfig,
    tf: Weights,
    pe: Weights,
    te_cfg: Umt5Config,
    te: Weights,
    vae_cfg: WanVaeConfig,
    vae: Weights,
    sched: UniPcConfig,
    tokenizer: tokenizers::Tokenizer,
    exact: bool,
}

impl std::fmt::Debug for MatrixGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixGame").field("layers", &self.cfg.num_layers).field("device", &self.backend.name()).finish_non_exhaustive()
    }
}

/// Per-session state: the prompt's text states (then the negative
/// prompt's under guidance), every action row so far and the camera path.
#[derive(Debug, Clone, Default)]
pub struct Session {
    states: Vec<Vec<f32>>,
    rows: Vec<f32>,
    poses: Vec<Pose>,
    path: Vec<Mat4>,
}

/// Pixel and latent spans of one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Clip {
    /// First pixel frame (inclusive).
    start: usize,
    /// Last pixel frame (exclusive).
    end: usize,
    /// First rays target frame.
    tgt_first: usize,
    /// First latent frame.
    latent_start: usize,
    /// Latent frames, held then new.
    latents: usize,
    /// Held latent frames at the start of the clip.
    held: usize,
}

impl Clip {
    fn of(first_index: usize, new_frames: usize) -> Result<Self> {
        if first_index == 0 {
            return Err(Error::Request("this model starts from a first frame".into()));
        }
        let end = FRAMES_PER_LATENT * (first_index + new_frames - 1) + 1;
        if first_index == 1 {
            return Ok(Self { start: 0, end, tgt_first: 0, latent_start: 0, latents: new_frames + 1, held: 1 });
        }
        if first_index <= CONTINUITY {
            return Err(Error::Request(format!("a continuing chunk needs {CONTINUITY} latent frames before it")));
        }
        let start = end - FRAMES_PER_LATENT * (new_frames + CONTINUITY);
        Ok(Self { start, end, tgt_first: start + 3, latent_start: first_index - CONTINUITY, latents: new_frames + CONTINUITY, held: CONTINUITY })
    }
}

impl MatrixGame {
    /// Load the world transformer from `world` (its
    /// `base_distilled_model/` directory) and the text encoder,
    /// tokenizer, autoencoder and scheduler from the Wan2.2 checkpoint
    /// `base` (diffusers layout).
    ///
    /// # Errors
    /// Configuration, weight or backend failures, each named.
    pub fn load(world: &CheckpointFiles, base: &CheckpointFiles, opts: LoadOptions) -> Result<Self> {
        let class = world.json("model_index.json")?.get("_class_name").and_then(Value::as_str).unwrap_or_default().to_owned();
        if class != "MatrixGame3I2VPipeline" {
            return Err(Error::Config(format!("pipeline class {class:?} is not MatrixGame3I2VPipeline")));
        }
        let raw = serde_json::to_vec(&world.json("base_distilled_model/config.json")?).map_err(|e| Error::Config(e.to_string()))?;
        let cfg = config(&raw)?;
        let te_cfg: Umt5Config = parse(base.json("text_encoder/config.json")?, "text encoder config")?;
        te_cfg.validate()?;
        let vae_cfg: WanVaeConfig = parse(base.json("vae/config.json")?, "vae config")?;
        vae_cfg.validate()?;
        if vae_cfg.z_dim != cfg.in_channels || cfg.in_channels != cfg.out_channels || te_cfg.d_model != cfg.text_dim {
            return Err(Error::Config("transformer, text encoder and autoencoder widths disagree".into()));
        }
        let sched: UniPcConfig = parse(base.json("scheduler/scheduler_config.json")?, "scheduler config")?;
        sched.validate()?;
        if !sched.use_flow_sigmas || sched.use_karras_sigmas || sched.use_dynamic_shifting || sched.shift_terminal.is_some() || sched.final_sigmas_type != "zero" {
            return Err(Error::Config("only shifted flow sigmas ending at zero are supported".into()));
        }

        let backend = opts.backend()?;
        tracing::info!(backend = backend.name(), gpu = backend.is_gpu(), "world model backend selected");
        let exact = opts.precision == Precision::F32;
        let wt = opts.precision.wtype();
        let tf_files = SafeTensors::open(&world.weights("base_distilled_model")?)?.renamed(rename)?;
        let tf = Weights::load(&backend, &tf_files, &cfg.weight_specs(wt))?;
        let pe = Weights::from_host(&backend, &cfg.patch_weights(&tf_files)?)?;
        drop(tf_files);
        let te_files = SafeTensors::open(&base.weights("text_encoder")?)?;
        let te = Weights::load(&backend, &te_files, &te_cfg.weight_specs(wt))?;
        drop(te_files);
        let vae_files = SafeTensors::open(&base.weights("vae")?)?;
        let vae = Weights::from_host(&backend, &vae_cfg.host_tensors(&vae_files, exact)?)?;
        drop(vae_files);
        let tok_path = base.root.join("tokenizer/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
        Ok(Self { backend, cfg, tf, pe, te_cfg, te, vae_cfg, vae, sched, tokenizer, exact })
    }

    /// Device bytes held by the weights.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.tf.bytes() + self.pe.bytes() + self.te.bytes() + self.vae.bytes()
    }

    /// The backend device name.
    #[must_use]
    pub fn device(&self) -> &str {
        self.backend.name()
    }

    fn dims(&self) -> (usize, usize) {
        let a = &self.cfg.world.as_ref().expect("world config").action;
        (a.keyboard_dim_in as usize, a.mouse_dim_in as usize)
    }

    /// Run one transformer pass over `frames` latent frames.
    #[allow(clippy::too_many_arguments)]
    fn pass(&self, g: &mut Graph, io: &wan_dit::DitIo, latents: &[f32], frames: usize, grid: (usize, usize), time: &[f32], ctx: &[f32], positions: &[usize], act_pos: &[usize], kb: &[f32], mo: &[f32], rays: &[f32]) -> Result<Vec<f32>> {
        let wd = self.cfg.world.as_ref().expect("world config");
        let (lh, lw) = grid;
        let (cos, sin) = self.cfg.rotary_tables_at(positions, lh / 2, lw / 2);
        g.set_f32(io.patches, &wan_dit::patchify(&self.cfg, latents, frames, lh, lw));
        g.set_f32(io.time, time);
        g.set_f32(io.context, ctx);
        g.set_f32(io.cos, &cos);
        g.set_f32(io.sin, &sin);
        let aio = io.actions.as_ref().expect("action inputs");
        let (acos, asin) = action_rotary(&wd.action, act_pos);
        g.set_f32(aio.keyboard, kb);
        g.set_f32(aio.mouse, mo);
        g.set_f32(aio.cos, &acos);
        g.set_f32(aio.sin, &asin);
        g.set_f32(io.camera.expect("camera input"), &patchify_rays(rays, wd.camera_channels as usize, frames, lh, lw));
        g.compute()?;
        Ok(wan_dit::unpatchify(&self.cfg, &g.read_f32(io.out), frames, lh, lw))
    }

    fn graph(&self, frames: usize, grid: (usize, usize)) -> Result<(Graph, wan_dit::DitIo)> {
        let n = frames * (grid.0 / 2) * (grid.1 / 2);
        let mut g = Graph::new(&self.backend)?;
        let io = wan_dit::build(&mut g, &self.cfg, &self.tf, &self.pe, frames as i64, n as i64, TEXT_TOKENS as i64, n as i64, self.exact);
        g.finish(&[io.out])?;
        Ok((g, io))
    }
}

/// Per-token time features: `clean` frames at 0, the rest at `t`.
fn times(frames: usize, clean: usize, hw: usize, t: f32) -> Vec<f32> {
    (0..frames * hw).flat_map(|i| wan_dit::time_features(if i < clean * hw { 0.0 } else { t })).collect()
}

/// `[z][a + b][plane]` from `[z][a][plane]` and `[z][b][plane]`.
fn join(a: &[f32], b: &[f32], z: usize) -> Vec<f32> {
    let (fa, fb) = (a.len() / z, b.len() / z);
    (0..z).flat_map(|c| a[c * fa..(c + 1) * fa].iter().chain(&b[c * fb..(c + 1) * fb]).copied()).collect()
}

/// Frames `from..` of `[z][frames][plane]`.
fn tail(v: &[f32], z: usize, frames: usize, from: usize) -> Vec<f32> {
    let f = v.len() / (z * frames);
    (0..z).flat_map(|c| v[(c * frames + from) * f..(c + 1) * frames * f].iter().copied()).collect()
}

/// `[z][frames][plane]` from frames `[z][plane]` each.
fn stack(frames: &[LatentFrame], z: usize) -> Vec<f32> {
    let plane = frames.first().map_or(0, |f| f.data.len() / z);
    (0..z).flat_map(|c| frames.iter().flat_map(move |f| f.data[c * plane..(c + 1) * plane].iter().copied())).collect()
}

impl WorldModel for MatrixGame {
    type Context = Session;
    type Stream = wan::Decoder;

    fn latent_channels(&self) -> usize {
        self.cfg.in_channels as usize
    }

    fn action_dims(&self) -> usize {
        let (k, m) = self.dims();
        k + m
    }

    fn spatial_stride(&self) -> usize {
        STRIDE
    }

    fn context(&self, prompt: &str, negative: &str, guided: bool) -> Result<Self::Context> {
        let mut states = vec![text_states(&self.backend, &self.te_cfg, &self.te, &tokens(&self.tokenizer, prompt)?)?];
        if guided {
            states.push(text_states(&self.backend, &self.te_cfg, &self.te, &tokens(&self.tokenizer, negative)?)?);
        }
        Ok(Session { states, ..Session::default() })
    }

    fn encode_image(&self, image: &RgbImage) -> Result<Vec<f32>> {
        wan::encode_frames(&self.backend, &self.vae_cfg, &self.vae, std::slice::from_ref(image))
    }

    fn advance(&self, ctx: &mut Self::Context, actions: &[f32], first_pixel: usize) -> Result<()> {
        let dims = self.action_dims();
        let (kdim, _) = self.dims();
        if first_pixel == 0 {
            return Err(Error::Request("this model starts from a first frame".into()));
        }
        if ctx.poses.is_empty() {
            ctx.rows = vec![0.0; dims];
            ctx.poses.push([0.0; 5]);
            ctx.path.push(camera::extrinsic(&[0.0; 5]));
        }
        if ctx.rows.len() != first_pixel * dims {
            return Err(Error::Request(format!("actions must continue at pixel frame {}", ctx.rows.len() / dims)));
        }
        ctx.rows.extend_from_slice(actions);
        for f in first_pixel..ctx.rows.len() / dims {
            let row = &ctx.rows[(f - 1) * dims..f * dims];
            let pose = camera::next_pose(&ctx.poses[f - 1], &row[..kdim], &row[kdim..]);
            ctx.poses.push(pose);
            ctx.path.push(camera::extrinsic(&pose));
        }
        Ok(())
    }

    fn select_memory(&self, ctx: &Self::Context, _held: &[usize], first_index: usize, _count: usize) -> Option<Vec<usize>> {
        if first_index == 1 {
            return Some(vec![0]);
        }
        let n = (ctx.path.len() - 1) / FRAMES_PER_LATENT + 1 - first_index;
        let clip = Clip::of(first_index, n).ok()?;
        let mut chosen: Vec<usize> = camera::memory_by_view(&ctx.path, clip.start, clip.end, VIEW_PICKS).iter().map(|p| p.latent).collect();
        chosen.extend(first_index - CONTINUITY..first_index);
        Some(chosen)
    }

    fn pinned(&self, index: usize) -> bool {
        index == 1
    }

    fn rollout(&self, ctx: &Self::Context, req: &ChunkRequest<'_>) -> Result<(Vec<f32>, u32)> {
        let dims = self.action_dims();
        let (kdim, mdim) = self.dims();
        let wd = self.cfg.world.as_ref().expect("world config");
        let clip = Clip::of(req.first_index, req.new_frames)?;
        if ctx.path.len() != clip.end || req.actions.len() != dims * (clip.end - 1 - FRAMES_PER_LATENT * (req.first_index - 1)) {
            return Err(Error::Request("the chunk's actions were not taken in".into()));
        }
        let picks: Vec<MemoryPick> = if req.first_index == 1 { Vec::new() } else { camera::memory_by_view(&ctx.path, clip.start, clip.end, VIEW_PICKS) };
        let want: Vec<usize> = picks.iter().map(|p| p.latent).chain(clip.latent_start..clip.latent_start + clip.held).collect();
        if req.memory.iter().map(|f| f.index).collect::<Vec<_>>() != want {
            return Err(Error::Request(format!("the chunk needs memory frames {want:?}; the session holds {} memory frames (raise its memory and history)", req.memory.len())));
        }
        let (lh, lw) = req.grid;
        let plane = lh * lw;
        let hw = plane / 4;
        let z = self.latent_channels();
        let (mem, held) = req.memory.split_at(picks.len());
        let mem = stack(mem, z);

        let mut latents = schedule::gaussian(req.seed, z * clip.latents * plane);
        for (i, f) in held.iter().enumerate() {
            place_frame(&mut latents, &f.data, i, clip.latents, plane);
        }

        let rows = &ctx.rows[clip.start * dims..clip.end * dims];
        let split = |r: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let k = r.chunks_exact(dims).flat_map(|x| x[..kdim].iter().copied()).collect();
            let m = r.chunks_exact(dims).flat_map(|x| x[kdim..].iter().copied()).collect();
            (k, m)
        };
        let (kb_rows, mo_rows) = split(rows);
        let (mem_kb, mem_mo) = (vec![-1.0; picks.len() * kdim], vec![1.0; picks.len() * mdim]);
        let kb = windows(&wd.action, &kb_rows, kdim, &mem_kb)?;
        let mo = windows(&wd.action, &mo_rows, mdim, &mem_mo)?;
        let clip_rays = camera::clip_rays(&ctx.path, clip.start, clip.end, clip.tgt_first, clip.latents, (lh, lw), STRIDE);
        let mut rays = Vec::new();
        for p in &picks {
            rays.extend(camera::memory_rays(&ctx.path, p.pixel, p.reference, (lh, lw), STRIDE));
        }
        let rays = join(&rays, &clip_rays, 6 * STRIDE * STRIDE);
        let clip_pos: Vec<usize> = (clip.latent_start..clip.latent_start + clip.latents).collect();
        let positions: Vec<usize> = picks.iter().map(|p| p.latent).chain(clip_pos.iter().copied()).collect();
        let act_pos: Vec<usize> = std::iter::repeat_n(0, picks.len()).chain(0..clip.latents).collect();
        let full_frames = picks.len() + clip.latents;

        let guided = ctx.states.len() == 2 && req.guidance_scale > 1.0;
        let (mut gf, iof) = self.graph(full_frames, req.grid)?;
        let mut null = if guided {
            let null_kb = windows(&wd.action, &vec![-1.0; kb_rows.len()], kdim, &[])?;
            let null_mo = windows(&wd.action, &vec![1.0; mo_rows.len()], mdim, &[])?;
            Some((self.graph(clip.latents, req.grid)?, null_kb, null_mo))
        } else {
            None
        };
        let act_clip: Vec<usize> = (0..clip.latents).collect();

        let (sigmas, timesteps) = flow_sigmas_schedule(req.steps as usize, self.sched.flow_shift, self.sched.num_train_timesteps);
        let mut sampler = UniPc::new(sigmas);
        let mut evaluations = 0u32;
        for &t in &timesteps {
            let t = t as f32;
            let x = join(&mem, &latents, z);
            let out = self.pass(&mut gf, &iof, &x, full_frames, req.grid, &times(full_frames, full_frames - clip.latents + clip.held, hw, t), &ctx.states[0], &positions, &act_pos, &kb, &mo, &rays)?;
            evaluations += 1;
            let full = tail(&out, z, full_frames, picks.len());
            let mut v = match &mut null {
                Some(((g, io), nkb, nmo)) => {
                    let u = self.pass(g, io, &latents, clip.latents, req.grid, &times(clip.latents, clip.held, hw, t), &ctx.states[1], &clip_pos, &act_clip, nkb, nmo, &clip_rays)?;
                    evaluations += 1;
                    u.iter().zip(&full).map(|(u, c)| u + req.guidance_scale * (c - u)).collect()
                }
                None => full,
            };
            zero_frames(&mut v, clip.held, clip.latents, plane);
            latents = sampler.step(&v, &latents)?;
        }
        Ok((tail(&latents, z, clip.latents, clip.held), evaluations))
    }

    fn open_stream(&self, (lh, lw): (usize, usize)) -> Result<Self::Stream> {
        wan::Decoder::new(&self.backend, &self.vae_cfg, lh, lw)
    }

    fn decode_next(&self, stream: &mut Self::Stream, latent: &[f32]) -> Result<Vec<f32>> {
        stream.push(&self.backend, &self.vae_cfg, &self.vae, latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_follow_the_reference_schedule() {
        let first = Clip::of(1, 14).unwrap();
        assert_eq!(first, Clip { start: 0, end: 57, tgt_first: 0, latent_start: 0, latents: 15, held: 1 });
        let next = Clip::of(15, 10).unwrap();
        assert_eq!(next, Clip { start: 41, end: 97, tgt_first: 44, latent_start: 11, latents: 14, held: 4 });
        assert_eq!(Clip::of(25, 10).unwrap().end, 137);
        assert!(Clip::of(0, 10).is_err());
        assert!(Clip::of(3, 10).is_err());
    }

    #[test]
    fn frames_join_and_split_per_channel() {
        let (a, b) = (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0]);
        let j = join(&a, &b, 2);
        assert_eq!(j, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
        assert_eq!(tail(&j, 2, 3, 2), vec![5.0, 6.0]);
        let s = stack(&[LatentFrame { index: 0, data: vec![1.0, 3.0] }, LatentFrame { index: 1, data: vec![2.0, 4.0] }], 2);
        assert_eq!(s, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
