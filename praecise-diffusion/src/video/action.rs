//! Cosmos3 action-conditioned generation for robot embodiments.
//!
//! An action run pairs a video of `chunk_size + 1` frames with `chunk_size`
//! action vectors (one per transition) in one generation stream. Three tasks:
//!
//! - policy: from a first frame and a task description, roll out the future
//!   video and the actions that produce it;
//! - forward dynamics: from a first frame and given actions, roll out the
//!   video they produce;
//! - inverse dynamics: from a whole video, infer the actions connecting its
//!   frames.
//!
//! The embodiment selects the per-domain action projections and fixes the
//! action width; narrower actions are zero-padded to the model's width and
//! the padding is held at zero throughout sampling. The task description is
//! wrapped in the structured JSON caption the model was trained on.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{grid_of, Cosmos3, Video};
use crate::cosmos3::{self, ActionSpan};
use crate::error::{Error, Result};
use crate::ggml::Graph;
use crate::pipeline::{RgbImage, Timings};
use crate::schedule;
use crate::unipc::UniPc;

/// The task an action run solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionMode {
    /// Future video and actions from a first frame.
    Policy,
    /// Future video from a first frame and given actions.
    ForwardDynamics,
    /// Actions connecting the frames of a given video.
    InverseDynamics,
}

/// An embodiment: its domain index in the action projections and its action
/// width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Embodiment {
    /// Row of the per-domain action projections.
    pub domain: u32,
    /// Action width before padding.
    pub action_width: usize,
}

/// Embodiments with a fixed action width, by name.
const EMBODIMENTS: &[(&str, u32, usize)] = &[
    ("av", 1, 9),
    ("camera_pose", 2, 9),
    ("hand_pose", 3, 57),
    ("pusht", 4, 2),
    ("umi", 6, 10),
    ("bridge_orig_lerobot", 7, 10),
    ("droid_lerobot", 8, 10),
    ("robomind-franka", 8, 10),
    ("galbot", 9, 30),
    ("robomind-franka-dual", 12, 20),
    ("robomind-ur", 13, 10),
    ("agibotworld", 15, 29),
    ("agibot_gear_gripper", 15, 29),
    ("agibot_gear_gripper_ext", 15, 29),
    ("fractal", 20, 10),
];

impl Embodiment {
    /// The embodiment registered under `name`.
    ///
    /// # Errors
    /// [`Error::Request`] for a name without a fixed action width.
    pub fn named(name: &str) -> Result<Self> {
        EMBODIMENTS
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|&(_, domain, action_width)| Self { domain, action_width })
            .ok_or_else(|| {
                let known: Vec<&str> = EMBODIMENTS.iter().map(|(n, _, _)| *n).collect();
                Error::Request(format!("unknown embodiment {name:?}; expected one of {known:?}"))
            })
    }
}

/// Conditioning canvases `(height, width)` per resolution tier. Frames must
/// already be one of the tier's canvases.
const RESOLUTION_TIERS: &[(u32, [(u32, u32); 5])] = &[
    (256, [(256, 256), (256, 320), (320, 256), (192, 320), (320, 192)]),
    (480, [(640, 640), (544, 736), (736, 544), (480, 832), (832, 480)]),
    (704, [(960, 960), (832, 1088), (1088, 832), (704, 1280), (1280, 704)]),
    (720, [(960, 960), (832, 1104), (1104, 832), (720, 1280), (1280, 720)]),
];

/// Camera framing sentences, by viewpoint name.
const VIEWPOINTS: &[(&str, &str)] = &[
    ("ego_view", "This video is captured from a first-person perspective looking at the scene."),
    ("third_person_view", "This video is captured from a third-person perspective looking towards the agent from the front."),
    ("wrist_view", "This video is captured from a wrist-mounted camera."),
    ("concat_view", "This video contains concatenated views from multiple camera perspectives."),
];

fn default_tier() -> u32 {
    480
}

fn default_view() -> String {
    "ego_view".into()
}

/// An action-conditioned generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    /// The task.
    pub mode: ActionMode,
    /// Embodiment name, for example `droid_lerobot`.
    pub embodiment: String,
    /// What the agent does.
    pub prompt: String,
    /// What to avoid; empty when absent.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Camera viewpoint: `ego_view`, `third_person_view`, `wrist_view` or
    /// `concat_view`.
    #[serde(default = "default_view")]
    pub view_point: String,
    /// Resolution tier: 256, 480, 704 or 720.
    #[serde(default = "default_tier")]
    pub resolution_tier: u32,
    /// Action transitions in the chunk, a positive multiple of 4; the video
    /// spans one frame more.
    pub chunk_size: u32,
    /// Conditioning frames, each one of the tier's canvases: the first frame
    /// for policy and forward dynamics (further frames are ignored), all
    /// `chunk_size + 1` frames for inverse dynamics.
    pub frames: Vec<RgbImage>,
    /// Forward dynamics only: actions `[steps][action width]`, one or more;
    /// fewer than `chunk_size` repeat the last, more are cut.
    #[serde(default)]
    pub actions: Option<Vec<Vec<f32>>>,
    /// Frame rate the video is generated for.
    pub fps: f32,
    /// Denoising steps.
    pub steps: u32,
    /// Classifier-free guidance scale; 1 disables guidance.
    pub guidance_scale: f32,
    /// Seed for the starting noise.
    pub seed: u64,
}

/// The result of an action run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutput {
    /// The generated (or, for inverse dynamics, reconstructed) video.
    pub video: Video,
    /// Predicted actions `[chunk_size][action width]`; `None` for forward
    /// dynamics, whose actions are given.
    pub actions: Option<Vec<Vec<f32>>>,
}

/// A JSON string literal as the reference serialiser writes it: ASCII only,
/// everything else escaped as UTF-16 units.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A float as the reference serialiser writes it (shortest round trip, with
/// a fractional part).
fn json_float(v: f32) -> String {
    let s = format!("{v}");
    if s.contains(['.', 'e', 'E']) || !v.is_finite() { s } else { format!("{s}.0") }
}

/// The structured caption of an action run.
#[must_use]
pub fn action_caption(description: &str, view_point: &str, num_frames: u32, fps: f32, height: u32, width: u32) -> String {
    let seconds = if fps > 0.0 { f64::from(num_frames) / f64::from(fps) } else { 0.0 };
    let duration = seconds as u64;
    let end = seconds.round_ties_even() as u64;
    let (minutes, secs) = (end / 60, end % 60);
    let mut desc = description.trim().to_string();
    if !desc.is_empty() && !desc.ends_with(['.', '!', '?']) {
        desc.push('.');
    }
    let ratio = if height > 0 { f64::from(width) / f64::from(height) } else { 1.0 };
    let mut aspect = "1,1";
    let mut best = f64::INFINITY;
    for (name, a, b) in [("1,1", 1.0, 1.0), ("4,3", 4.0, 3.0), ("3,4", 3.0, 4.0), ("16,9", 16.0, 9.0), ("9,16", 9.0, 16.0)] {
        let d = (a / b - ratio).abs();
        if d < best {
            best = d;
            aspect = name;
        }
    }
    let mut out = String::from("{");
    if let Some((_, framing)) = VIEWPOINTS.iter().find(|(n, _)| *n == view_point) {
        out.push_str(&format!("\"cinematography\": {{\"framing\": {}}}, ", json_str(framing)));
    }
    out.push_str(&format!(
        "\"actions\": [{{\"time\": \"0:00-{minutes}:{secs:02}\", \"description\": {}}}], \"duration\": \"{duration}s\", \
         \"fps\": {}, \"resolution\": {{\"H\": {height}, \"W\": {width}}}, \"aspect_ratio\": \"{aspect}\"}}",
        json_str(&desc),
        json_float(fps)
    ));
    out
}

/// One transformer pass and its rotary tables.
struct Pass {
    g: Graph,
    io: cosmos3::GenIo,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Cosmos3 {
    fn check_action(&self, req: &ActionRequest) -> Result<(Embodiment, usize)> {
        let bad = |m: String| Err(Error::Request(m));
        let model_width = match (self.cfg.action_gen, self.cfg.action_dim) {
            (true, Some(w)) => w as usize,
            _ => return bad("this checkpoint has no action head".into()),
        };
        let emb = Embodiment::named(&req.embodiment)?;
        if emb.action_width > model_width {
            return bad(format!("embodiment {} needs {} action channels, the model has {model_width}", req.embodiment, emb.action_width));
        }
        let domains = self.tf.get("action_proj_in.fc.weight").ne(1) as u32;
        if emb.domain >= domains {
            return bad(format!("embodiment {} uses domain {}, the model has {domains}", req.embodiment, emb.domain));
        }
        if req.chunk_size == 0 || req.chunk_size % 4 != 0 {
            return bad("chunk_size must be a positive multiple of 4".into());
        }
        if !(req.fps > 0.0 && req.fps.is_finite()) {
            return bad("fps must be positive".into());
        }
        if req.steps == 0 {
            return bad("steps must be at least 1".into());
        }
        let Some((_, canvases)) = RESOLUTION_TIERS.iter().find(|(t, _)| *t == req.resolution_tier) else {
            return bad(format!("resolution tier {} is not one of 256, 480, 704, 720", req.resolution_tier));
        };
        let need = if req.mode == ActionMode::InverseDynamics { req.chunk_size as usize + 1 } else { 1 };
        if req.frames.len() < need {
            return bad(format!("{:?} needs {need} conditioning frames, got {}", req.mode, req.frames.len()));
        }
        let first = &req.frames[0];
        for f in &req.frames[..need] {
            if (f.width, f.height) != (first.width, first.height) || f.rgb.len() != (f.width * f.height * 3) as usize {
                return bad("conditioning frames must share one size and match their pixel buffers".into());
            }
        }
        if !canvases.contains(&(first.height, first.width)) {
            return bad(format!(
                "frames are {}x{}; tier {} takes (height, width) {canvases:?}",
                first.width, first.height, req.resolution_tier
            ));
        }
        if first.width % 32 != 0 || first.height % 32 != 0 {
            return bad(format!("canvas {}x{} is not a multiple of 32", first.width, first.height));
        }
        match (&req.mode, &req.actions) {
            (ActionMode::ForwardDynamics, None) => return bad("forward dynamics needs actions".into()),
            (ActionMode::ForwardDynamics, Some(a)) if a.is_empty() => return bad("forward dynamics needs at least one action".into()),
            (ActionMode::ForwardDynamics, Some(a)) => {
                if let Some(row) = a.iter().find(|r| r.len() != emb.action_width) {
                    return bad(format!("an action has width {}, embodiment {} takes {}", row.len(), req.embodiment, emb.action_width));
                }
            }
            (_, Some(_)) => return bad(format!("{:?} predicts actions; none may be given", req.mode)),
            (_, None) => {}
        }
        Ok((emb, model_width))
    }

    /// Positions `[time, height, width]` of `n` action tokens after a text of
    /// `text` tokens: one per transition, starting one frame in.
    fn action_positions(&self, text: usize, n: usize, fps: f32) -> Vec<[f32; 3]> {
        let offset = (text as u64 + self.cfg.unified_3d_mrope_temporal_modality_margin) as f32;
        let spatial = if self.cfg.unified_3d_mrope_reset_spatial_ids { 0.0 } else { offset };
        let base_tps = (self.cfg.base_fps / self.vae_cfg.scale_factor_temporal as f64) as f32;
        let modulate = self.cfg.enable_fps_modulation && n > 1;
        (0..n)
            .map(|i| {
                let f = (i + 1) as f32;
                let t = if modulate { f / fps * base_tps + offset } else { f + offset };
                [t, spatial, spatial]
            })
            .collect()
    }

    /// Encode a whole video to normalised latents `[z][T][H/16][W/16]`.
    fn encode_video(&self, _frames: &[RgbImage]) -> Result<Vec<f32>> {
        Err(Error::Request("inverse dynamics needs the full-video encoder, which this build does not have yet".into()))
    }

    /// Run one action task.
    ///
    /// # Errors
    /// [`Error::Request`] for an embodiment, size, frame set or action set the
    /// model cannot serve; backend failures otherwise.
    pub fn generate_action(&mut self, req: &ActionRequest) -> Result<ActionOutput> {
        let (emb, width) = self.check_action(req)?;
        let first = &req.frames[0];
        let (lt, lh, lw) = grid_of(req.chunk_size + 1, first.height, first.width);
        let na = req.chunk_size as usize;
        let mut noise = schedule::gaussian(req.seed, self.cfg.latent_channel as usize * lt * lh * lw + na * width);
        let action_noise = noise.split_off(self.cfg.latent_channel as usize * lt * lh * lw);
        self.generate_action_from(req, emb, width, noise, action_noise)
    }

    /// [`Self::generate_action`] from given video noise `[z][T][H][W]` and
    /// action noise `[chunk][model action width]`.
    pub(crate) fn generate_action_from(
        &mut self,
        req: &ActionRequest,
        emb: Embodiment,
        width: usize,
        mut latents: Vec<f32>,
        mut actions: Vec<f32>,
    ) -> Result<ActionOutput> {
        let t0 = Instant::now();
        let first = &req.frames[0];
        let num_frames = req.chunk_size + 1;
        let shape = grid_of(num_frames, first.height, first.width);
        let (lt, lh, lw) = shape;
        let (gh, gw) = (lh / 2, lw / 2);
        let na = req.chunk_size as usize;
        let plane = lh * lw;

        let caption = action_caption(&req.prompt, &req.view_point, num_frames, req.fps, first.height, first.width);
        let cond_ids = self.tokens(&caption, false)?;
        let uncond_ids = self.tokens(req.negative_prompt.as_deref().unwrap_or(""), false)?;

        let cond_frames = if req.mode == ActionMode::InverseDynamics {
            latents = self.encode_video(&req.frames[..num_frames as usize])?;
            lt
        } else {
            let enc = self.encode_image(first)?;
            for (c, chunk) in enc.chunks_exact(plane).enumerate() {
                latents[c * lt * plane..c * lt * plane + plane].copy_from_slice(chunk);
            }
            1
        };
        let cond_actions = if let (ActionMode::ForwardDynamics, Some(given)) = (req.mode, &req.actions) {
            for i in 0..na {
                let row = &given[i.min(given.len() - 1)];
                actions[i * width..(i + 1) * width].fill(0.0);
                actions[i * width..i * width + row.len()].copy_from_slice(row);
            }
            na
        } else {
            0
        };
        for row in actions.chunks_exact_mut(width) {
            row[emb.action_width..].fill(0.0);
        }
        let encode_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let n = (lt * gh * gw) as i64;
        let cond = (cond_frames * gh * gw) as i64;
        let span = ActionSpan { tokens: na as i64, cond: cond_actions as i64 };
        let use_cfg = req.guidance_scale != 1.0;
        let mut passes = Vec::new();
        let mut caches = Vec::new();
        for ids in if use_cfg { vec![&cond_ids, &uncond_ids] } else { vec![&cond_ids] } {
            let cache = self.text_cache(ids)?;
            let mut g = Graph::new(&self.backend)?;
            let io = cosmos3::build_gen(&mut g, &self.cfg, &self.tf, &cache, n, cond, Some(span), self.exact);
            let a = io.actions.as_ref().expect("action graph");
            let outs: Vec<_> = io.out.iter().chain(a.out.iter()).copied().collect();
            g.finish(&outs)?;
            let mut pos = self.video_positions(ids.len(), lt, gh, gw, req.fps);
            pos.extend(self.action_positions(ids.len(), na, req.fps));
            let (cos, sin) = self.cfg.rotary_tables(&pos);
            passes.push(Pass { g, io, cos, sin });
            caches.push(cache);
        }
        let domain = [emb.domain as i32];
        let (sigmas, timesteps) = self.sched.schedule(req.steps as usize);
        let mut video_sampler = UniPc::new(sigmas.clone());
        let mut action_sampler = UniPc::new(sigmas);
        let mut evaluations = 0u32;
        for &t in &timesteps {
            let patches = self.patches(&latents, shape);
            let time = self.cfg.time_features(t);
            let mut vpreds = Vec::with_capacity(passes.len());
            let mut apreds = Vec::with_capacity(passes.len());
            for p in &passes {
                let a = p.io.actions.as_ref().expect("action graph");
                p.g.set_f32(p.io.patches, &patches);
                if cond < n {
                    p.g.set_f32(p.io.time, &time);
                }
                p.g.set_f32(p.io.cos, &p.cos);
                p.g.set_f32(p.io.sin, &p.sin);
                p.g.set_f32(a.values, &actions);
                if cond_actions < na {
                    p.g.set_f32(a.time, &time);
                }
                p.g.set_i32(a.domain, &domain);
                p.g.compute()?;
                evaluations += 1;
                if let Some(out) = p.io.out {
                    vpreds.push(self.velocity(&p.g.read_f32(out), cond_frames, shape));
                }
                if let Some(out) = a.out {
                    let mut v = vec![0f32; na * width];
                    let pred = p.g.read_f32(out);
                    for (i, row) in pred.chunks_exact(width).enumerate() {
                        let dst = &mut v[(cond_actions + i) * width..(cond_actions + i + 1) * width];
                        dst[..emb.action_width].copy_from_slice(&row[..emb.action_width]);
                    }
                    apreds.push(v);
                }
            }
            let guide = |mut preds: Vec<Vec<f32>>| -> Vec<f32> {
                if use_cfg {
                    preds[1].iter().zip(&preds[0]).map(|(u, c)| u + req.guidance_scale * (c - u)).collect()
                } else {
                    preds.pop().expect("one pass")
                }
            };
            let zeros = vec![0f32; latents.len()];
            let v = if vpreds.is_empty() { zeros } else { guide(vpreds) };
            latents = video_sampler.step(&v, &latents)?;
            if !apreds.is_empty() {
                actions = action_sampler.step(&guide(apreds), &actions)?;
                for row in actions.chunks_exact_mut(width) {
                    row[emb.action_width..].fill(0.0);
                }
            }
        }
        drop(passes);
        drop(caches);
        let denoise_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let px = self.decode(&latents, shape)?;
        let decode_ms = t2.elapsed().as_millis() as u64;
        let (w, h) = (first.width as usize, first.height as usize);
        let frames = px.len() / (3 * w * h);
        let mut rgb = vec![0u8; frames * w * h * 3];
        for f in 0..frames {
            for c in 0..3 {
                for i in 0..w * h {
                    let v = px[(f * 3 + c) * w * h + i];
                    rgb[(f * w * h + i) * 3 + c] = ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        let predicted = (req.mode != ActionMode::ForwardDynamics)
            .then(|| actions.chunks_exact(width).map(|r| r[..emb.action_width].to_vec()).collect());
        Ok(ActionOutput {
            video: Video {
                width: first.width,
                height: first.height,
                frames: frames as u32,
                fps: req.fps,
                rgb,
                seed: req.seed,
                evaluations,
                timings: Timings { encode_ms, denoise_ms, decode_ms },
            },
            actions: predicted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_caption_matches_the_reference_serialisation() {
        let c = action_caption("pick up the cup", "ego_view", 17, 15.0, 640, 640);
        assert_eq!(
            c,
            "{\"cinematography\": {\"framing\": \"This video is captured from a first-person perspective looking at the scene.\"}, \
             \"actions\": [{\"time\": \"0:00-0:01\", \"description\": \"pick up the cup.\"}], \"duration\": \"1s\", \
             \"fps\": 15.0, \"resolution\": {\"H\": 640, \"W\": 640}, \"aspect_ratio\": \"1,1\"}"
        );
        let c = action_caption(" Wave! ", "unknown", 33, 12.5, 480, 832);
        assert!(c.starts_with("{\"actions\": [{\"time\": \"0:00-0:03\", \"description\": \"Wave!\"}]"), "{c}");
        assert!(c.ends_with("\"fps\": 12.5, \"resolution\": {\"H\": 480, \"W\": 832}, \"aspect_ratio\": \"16,9\"}"), "{c}");
        assert_eq!(json_str("caf\u{e9} \u{1f600}"), "\"caf\\u00e9 \\ud83d\\ude00\"");
    }

    #[test]
    fn embodiments_resolve_and_unknown_names_are_refused() {
        assert_eq!(Embodiment::named("droid_lerobot").unwrap(), Embodiment { domain: 8, action_width: 10 });
        assert_eq!(Embodiment::named("agibotworld").unwrap(), Embodiment { domain: 15, action_width: 29 });
        assert!(Embodiment::named("libero").is_err());
    }
}
