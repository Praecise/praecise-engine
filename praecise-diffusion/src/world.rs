//! Interactive world-model sessions: a video model rolled forward chunk by
//! chunk, each chunk conditioned on the frames kept from before and on the
//! actions supplied for it, with the frames streamed out as each chunk is
//! decoded.
//!
//! State is bounded however long a session runs:
//!
//! - the session holds at most [`SessionConfig::history_latent_frames`]
//!   latent frames (plus any the model pins), each at its own time position;
//!   each chunk is conditioned on [`SessionConfig::memory_latent_frames`] of
//!   them, so the attention sequence (and with it the keys and values every
//!   denoising step computes) never grows past `memory + chunk` latent
//!   frames;
//! - which held frames condition a chunk is the model's choice
//!   ([`WorldModel::select_memory`]); a model without a selector takes the
//!   most recent;
//! - the prompt is encoded once per session and its states reused for every
//!   chunk and step;
//! - the autoencoder decoder is causal and streams with a fixed-size cache.
//!
//! The transformers served here attend bidirectionally within a chunk, so
//! the memory frames' keys and values are recomputed with the chunk rather
//! than appended to a growing cache; the memory bound is what keeps that
//! work constant.

use std::collections::VecDeque;

use crate::error::{Error, Result};
use crate::pipeline::RgbImage;
use crate::video::to_rgb8;

/// Pixel frames each latent frame after the first decodes to.
pub const FRAMES_PER_LATENT: usize = 4;

/// One latent frame `[z][rows][cols]` and its time position in the stream.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentFrame {
    /// Position of this frame in the session's latent timeline.
    pub index: usize,
    /// Normalised latent values.
    pub data: Vec<f32>,
}

/// What a session asks of the model for one chunk.
#[derive(Debug)]
pub struct ChunkRequest<'a> {
    /// Frames kept from earlier chunks, oldest first. They stay clean.
    pub memory: &'a [LatentFrame],
    /// Latent frames to generate after the memory.
    pub new_frames: usize,
    /// Time position of the first new frame.
    pub first_index: usize,
    /// Actions for the new frames, `[pixel frame][action dims]`, empty for
    /// a model without action input.
    pub actions: &'a [f32],
    /// Latent grid `(rows, cols)`.
    pub grid: (usize, usize),
    /// Denoising steps.
    pub steps: u32,
    /// Classifier-free guidance scale (1 or below disables it).
    pub guidance_scale: f32,
    /// Seed of the chunk's starting noise.
    pub seed: u64,
}

/// A video model that can be rolled forward in chunks.
pub trait WorldModel {
    /// Encoded prompt and per-session state, created once per session.
    type Context;
    /// Open decoder stream.
    type Stream;

    /// Latent channels.
    fn latent_channels(&self) -> usize;
    /// Action values per pixel frame; 0 for a model without action input.
    fn action_dims(&self) -> usize;
    /// Pixels per latent cell along each axis.
    fn spatial_stride(&self) -> usize;
    /// Encode the prompt (and, under guidance, the negative prompt).
    ///
    /// # Errors
    /// Tokenizer or backend failures.
    fn context(&self, prompt: &str, negative: &str, guided: bool) -> Result<Self::Context>;
    /// Encode one image into a latent frame `[z][rows][cols]`.
    ///
    /// # Errors
    /// Backend failures.
    fn encode_image(&self, image: &RgbImage) -> Result<Vec<f32>>;
    /// Generate `req.new_frames` latent frames `[z][new][rows][cols]` after
    /// the memory, returning them and the transformer evaluations run.
    ///
    /// # Errors
    /// Backend failures.
    fn rollout(&self, ctx: &Self::Context, req: &ChunkRequest<'_>) -> Result<(Vec<f32>, u32)>;
    /// Take in the next chunk's actions (`[pixel frame][dims]`, the first
    /// at pixel frame `first_pixel`) before its memory is chosen.
    ///
    /// # Errors
    /// Actions the model cannot follow.
    fn advance(&self, _ctx: &mut Self::Context, _actions: &[f32], _first_pixel: usize) -> Result<()> {
        Ok(())
    }
    /// The `count` memory frames for the chunk starting at latent position
    /// `first_index`, chosen from the time positions held (oldest first),
    /// in the order the model takes them; repeats are allowed. `None` takes
    /// the most recent `count`.
    fn select_memory(&self, _ctx: &Self::Context, _held: &[usize], _first_index: usize, _count: usize) -> Option<Vec<usize>> {
        None
    }
    /// Whether the frame at latent position `index` stays held however old.
    fn pinned(&self, _index: usize) -> bool {
        false
    }
    /// Open a decoder stream for a latent grid.
    ///
    /// # Errors
    /// Backend failures.
    fn open_stream(&self, grid: (usize, usize)) -> Result<Self::Stream>;
    /// Decode the next latent frame to pixels `[3][T][H][W]` in [-1, 1].
    ///
    /// # Errors
    /// Backend failures.
    fn decode_next(&self, stream: &mut Self::Stream, latent: &[f32]) -> Result<Vec<f32>>;
}

/// Fixed settings of a session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per simulated second.
    pub fps: f32,
    /// Latent frames generated per chunk.
    pub chunk_latent_frames: usize,
    /// Latent frames conditioning each chunk.
    pub memory_latent_frames: usize,
    /// Latent frames held between chunks, from which the memory is chosen
    /// (at least `memory_latent_frames`).
    pub history_latent_frames: usize,
    /// Denoising steps per chunk.
    pub steps: u32,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
    /// Seed of the first chunk; chunk `k` uses `seed + k`.
    pub seed: u64,
}

impl SessionConfig {
    /// # Errors
    /// [`Error::Request`] naming the first setting no session can serve.
    pub fn validate(&self, stride: usize) -> Result<()> {
        let bad = |m: &str| Err(Error::Request(m.into()));
        let s = stride as u32 * 2;
        if self.width == 0 || self.height == 0 || self.width % s != 0 || self.height % s != 0 {
            return bad("width and height must be positive multiples of the patch stride");
        }
        if !(self.fps > 0.0 && self.fps.is_finite()) {
            return bad("fps must be positive");
        }
        if self.chunk_latent_frames == 0 {
            return bad("chunks must hold at least one latent frame");
        }
        if self.memory_latent_frames == 0 {
            return bad("the memory must keep at least one latent frame");
        }
        if self.history_latent_frames < self.memory_latent_frames {
            return bad("the history must hold at least the memory");
        }
        if self.steps == 0 {
            return bad("steps must be at least 1");
        }
        Ok(())
    }
}

/// Frames produced by one step of a session.
#[derive(Debug, Clone)]
pub struct WorldChunk {
    /// Index of the chunk in the session, from 0.
    pub chunk: u64,
    /// Index of the first pixel frame here in the session's output.
    pub first_frame: u64,
    /// Pixel frames in this chunk.
    pub frames: u32,
    /// Interleaved RGB, `frames x height x width x 3`.
    pub rgb: Vec<u8>,
    /// Simulated seconds this chunk covers.
    pub simulated_secs: f64,
    /// Transformer evaluations run.
    pub evaluations: u32,
}

/// A running world-model session.
pub struct WorldSession<'m, M: WorldModel> {
    model: &'m M,
    cfg: SessionConfig,
    ctx: M::Context,
    memory: VecDeque<LatentFrame>,
    stream: M::Stream,
    next_index: usize,
    chunks: u64,
    frames_out: u64,
    pending: Vec<f32>,
}

impl<M: WorldModel> std::fmt::Debug for WorldSession<'_, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldSession").field("chunks", &self.chunks).field("frames_out", &self.frames_out).finish_non_exhaustive()
    }
}

impl<'m, M: WorldModel> WorldSession<'m, M> {
    /// Start a session from a prompt and optionally a first frame (which
    /// becomes the first memory frame and the first frame streamed out).
    ///
    /// # Errors
    /// [`Error::Request`] for settings the session cannot serve; model
    /// failures otherwise.
    pub fn start(model: &'m M, cfg: SessionConfig, prompt: &str, negative: &str, first_frame: Option<&RgbImage>) -> Result<Self> {
        cfg.validate(model.spatial_stride())?;
        let grid = Self::grid_of(&cfg, model.spatial_stride());
        let ctx = model.context(prompt, negative, cfg.guidance_scale > 1.0)?;
        let mut stream = model.open_stream(grid)?;
        let mut memory = VecDeque::with_capacity(cfg.history_latent_frames + cfg.chunk_latent_frames);
        let mut pending = Vec::new();
        let mut next_index = 0;
        if let Some(img) = first_frame {
            if img.width != cfg.width || img.height != cfg.height {
                return Err(Error::Request("the first frame must match the session size".into()));
            }
            let data = model.encode_image(img)?;
            pending = model.decode_next(&mut stream, &data)?;
            memory.push_back(LatentFrame { index: 0, data });
            next_index = 1;
        }
        Ok(Self { model, cfg, ctx, memory, stream, next_index, chunks: 0, frames_out: 0, pending })
    }

    fn grid_of(cfg: &SessionConfig, stride: usize) -> (usize, usize) {
        (cfg.height as usize / stride, cfg.width as usize / stride)
    }

    /// Action values the next chunk takes: one row per pixel frame it adds.
    #[must_use]
    pub fn actions_needed(&self) -> usize {
        self.model.action_dims() * self.pixel_frames_next()
    }

    /// Pixel frames the next chunk adds.
    #[must_use]
    pub fn pixel_frames_next(&self) -> usize {
        let n = self.cfg.chunk_latent_frames;
        if self.next_index == 0 {
            1 + FRAMES_PER_LATENT * (n - 1)
        } else {
            FRAMES_PER_LATENT * n
        }
    }

    /// Latent frames held now, oldest first.
    #[must_use]
    pub fn memory(&self) -> impl Iterator<Item = &LatentFrame> {
        self.memory.iter()
    }

    /// Simulated seconds streamed so far.
    #[must_use]
    pub fn simulated_secs(&self) -> f64 {
        self.frames_out as f64 / f64::from(self.cfg.fps)
    }

    /// Chunks generated so far.
    #[must_use]
    pub fn chunks(&self) -> u64 {
        self.chunks
    }

    /// Generate the next chunk under `actions` (`[pixel frame][dims]`, see
    /// [`Self::actions_needed`]) and stream its frames out.
    ///
    /// # Errors
    /// [`Error::Request`] for the wrong number of action values; model
    /// failures otherwise.
    pub fn step(&mut self, actions: &[f32]) -> Result<WorldChunk> {
        if actions.len() != self.actions_needed() {
            return Err(Error::Request(format!("this chunk takes {} action values, got {}", self.actions_needed(), actions.len())));
        }
        if actions.iter().any(|a| !a.is_finite()) {
            return Err(Error::Request("action values must be finite".into()));
        }
        let stride = self.model.spatial_stride();
        let grid = Self::grid_of(&self.cfg, stride);
        let plane = grid.0 * grid.1;
        let z = self.model.latent_channels();
        let new_frames = self.cfg.chunk_latent_frames;
        let first_pixel = if self.next_index == 0 { 0 } else { 1 + FRAMES_PER_LATENT * (self.next_index - 1) };
        self.model.advance(&mut self.ctx, actions, first_pixel)?;
        let held: Vec<usize> = self.memory.iter().map(|f| f.index).collect();
        let count = self.cfg.memory_latent_frames.min(held.len());
        let chosen = self.model.select_memory(&self.ctx, &held, self.next_index, count).unwrap_or_else(|| held[held.len() - count..].to_vec());
        let memory = chosen
            .iter()
            .map(|i| self.memory.iter().find(|f| f.index == *i).cloned().ok_or_else(|| Error::Request(format!("the model chose memory frame {i}, which is not held"))))
            .collect::<Result<Vec<LatentFrame>>>()?;
        if memory.len() > self.cfg.memory_latent_frames {
            return Err(Error::Request("the model chose more memory frames than the session holds".into()));
        }
        let req = ChunkRequest {
            memory: &memory,
            new_frames,
            first_index: self.next_index,
            actions,
            grid,
            steps: self.cfg.steps,
            guidance_scale: self.cfg.guidance_scale,
            seed: self.cfg.seed.wrapping_add(self.chunks),
        };
        let (latents, evaluations) = self.model.rollout(&self.ctx, &req)?;
        if latents.len() != z * new_frames * plane {
            return Err(Error::Request("the model returned a chunk of the wrong size".into()));
        }
        let mut px = std::mem::take(&mut self.pending);
        for t in 0..new_frames {
            let data: Vec<f32> = (0..z).flat_map(|c| latents[(c * new_frames + t) * plane..(c * new_frames + t + 1) * plane].iter().copied()).collect();
            px.extend(self.model.decode_next(&mut self.stream, &data)?);
            self.memory.push_back(LatentFrame { index: self.next_index, data });
            self.next_index += 1;
        }
        let mut unpinned = self.memory.iter().filter(|f| !self.model.pinned(f.index)).count();
        let pinned = self.memory.len() - unpinned;
        while pinned + unpinned > self.cfg.history_latent_frames && unpinned > 0 {
            let at = self.memory.iter().position(|f| !self.model.pinned(f.index)).unwrap_or(0);
            self.memory.remove(at);
            unpinned -= 1;
        }
        let (rgb, frames) = to_rgb8(&px, self.cfg.width as usize, self.cfg.height as usize);
        let chunk = WorldChunk {
            chunk: self.chunks,
            first_frame: self.frames_out,
            frames: frames as u32,
            rgb,
            simulated_secs: frames as f64 / f64::from(self.cfg.fps),
            evaluations,
        };
        self.chunks += 1;
        self.frames_out += frames as u64;
        Ok(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// One channel, 16-pixel cells; the latent value of a new frame is its
    /// time index; decoding emits frames filled with that value.
    struct Toy {
        dims: usize,
        seen: RefCell<Vec<(Vec<usize>, usize, usize, u64)>>,
        /// Choose the first held frame (pinned) and the most recent.
        pick: bool,
    }

    impl WorldModel for Toy {
        type Context = String;
        type Stream = usize;
        fn latent_channels(&self) -> usize {
            1
        }
        fn action_dims(&self) -> usize {
            self.dims
        }
        fn spatial_stride(&self) -> usize {
            16
        }
        fn context(&self, prompt: &str, _: &str, _: bool) -> Result<String> {
            Ok(prompt.to_owned())
        }
        fn encode_image(&self, _: &RgbImage) -> Result<Vec<f32>> {
            Ok(vec![0.0; 4])
        }
        fn rollout(&self, _: &String, req: &ChunkRequest<'_>) -> Result<(Vec<f32>, u32)> {
            self.seen.borrow_mut().push((req.memory.iter().map(|f| f.index).collect(), req.first_index, req.actions.len(), req.seed));
            let plane = req.grid.0 * req.grid.1;
            Ok(((0..req.new_frames).flat_map(|t| vec![(req.first_index + t) as f32; plane]).collect(), req.steps))
        }
        fn select_memory(&self, _: &String, held: &[usize], _: usize, count: usize) -> Option<Vec<usize>> {
            let last = *held.last()?;
            self.pick.then(|| if count == 1 { vec![last] } else { vec![held[0], last] })
        }
        fn pinned(&self, index: usize) -> bool {
            self.pick && index == 0
        }
        fn open_stream(&self, _: (usize, usize)) -> Result<usize> {
            Ok(0)
        }
        fn decode_next(&self, pushed: &mut usize, latent: &[f32]) -> Result<Vec<f32>> {
            let t = if *pushed == 0 { 1 } else { FRAMES_PER_LATENT };
            *pushed += 1;
            let v = latent[0] / 100.0;
            Ok(vec![v; 3 * t * 32 * 32])
        }
    }

    fn cfg() -> SessionConfig {
        SessionConfig { width: 32, height: 32, fps: 16.0, chunk_latent_frames: 3, memory_latent_frames: 4, history_latent_frames: 4, steps: 2, guidance_scale: 1.0, seed: 7 }
    }

    fn image() -> RgbImage {
        RgbImage { width: 32, height: 32, rgb: vec![0; 32 * 32 * 3] }
    }

    #[test]
    fn memory_stays_bounded_and_keeps_time_positions() {
        let toy = Toy { dims: 0, seen: RefCell::new(Vec::new()), pick: false };
        let mut s = WorldSession::start(&toy, cfg(), "a road", "", Some(&image())).unwrap();
        for _ in 0..5 {
            s.step(&[]).unwrap();
            assert!(s.memory().count() <= 4);
        }
        let seen = toy.seen.borrow();
        assert_eq!(seen[0].0, vec![0]);
        assert_eq!(seen[0].1, 1);
        assert_eq!(seen[1].0, vec![0, 1, 2, 3]);
        assert_eq!(seen[2].0, vec![3, 4, 5, 6]);
        assert_eq!(seen[4].0, vec![9, 10, 11, 12]);
        assert_eq!(seen[4].1, 13);
        assert_eq!(s.memory().map(|f| f.index).collect::<Vec<_>>(), vec![12, 13, 14, 15]);
        assert_eq!(seen.iter().map(|r| r.3).collect::<Vec<_>>(), vec![7, 8, 9, 10, 11]);
    }

    #[test]
    fn a_model_can_choose_its_memory_and_pin_frames() {
        let toy = Toy { dims: 0, seen: RefCell::new(Vec::new()), pick: true };
        let c = SessionConfig { memory_latent_frames: 2, history_latent_frames: 3, ..cfg() };
        let mut s = WorldSession::start(&toy, c, "a road", "", Some(&image())).unwrap();
        for _ in 0..3 {
            s.step(&[]).unwrap();
            assert!(s.memory().count() <= 3);
        }
        let seen: Vec<Vec<usize>> = toy.seen.borrow().iter().map(|r| r.0.clone()).collect();
        assert_eq!(seen, vec![vec![0], vec![0, 3], vec![0, 6]]);
        assert_eq!(s.memory().map(|f| f.index).collect::<Vec<_>>(), vec![0, 8, 9]);
    }

    #[test]
    fn frames_stream_per_chunk_and_count_simulated_time() {
        let toy = Toy { dims: 0, seen: RefCell::new(Vec::new()), pick: false };
        let mut s = WorldSession::start(&toy, cfg(), "a road", "", Some(&image())).unwrap();
        let a = s.step(&[]).unwrap();
        assert_eq!(a.frames, 1 + 12);
        assert_eq!(a.rgb.len(), 13 * 32 * 32 * 3);
        let b = s.step(&[]).unwrap();
        assert_eq!((b.first_frame, b.frames), (13, 12));
        assert!((s.simulated_secs() - 25.0 / 16.0).abs() < 1e-12);
        assert!((b.simulated_secs - 0.75).abs() < 1e-12);
    }

    #[test]
    fn a_text_only_session_opens_with_a_single_frame() {
        let toy = Toy { dims: 0, seen: RefCell::new(Vec::new()), pick: false };
        let mut s = WorldSession::start(&toy, cfg(), "a road", "", None).unwrap();
        assert_eq!(s.pixel_frames_next(), 9);
        assert_eq!(s.step(&[]).unwrap().frames, 9);
        assert_eq!(s.pixel_frames_next(), 12);
        assert!(toy.seen.borrow()[0].0.is_empty());
    }

    #[test]
    fn actions_are_one_row_per_pixel_frame() {
        let toy = Toy { dims: 8, seen: RefCell::new(Vec::new()), pick: false };
        let mut s = WorldSession::start(&toy, cfg(), "a road", "", Some(&image())).unwrap();
        assert_eq!(s.actions_needed(), 8 * 12);
        assert!(matches!(s.step(&[0.0; 8]), Err(Error::Request(_))));
        assert!(matches!(s.step(&[f32::NAN; 96]), Err(Error::Request(_))));
        s.step(&[0.5; 96]).unwrap();
        assert_eq!(toy.seen.borrow()[0].2, 96);
        let none = Toy { dims: 0, seen: RefCell::new(Vec::new()), pick: false };
        let mut s = WorldSession::start(&none, cfg(), "a road", "", None).unwrap();
        assert!(matches!(s.step(&[1.0]), Err(Error::Request(_))));
    }

    #[test]
    fn settings_are_checked() {
        let toy = Toy { dims: 0, seen: RefCell::new(Vec::new()), pick: false };
        for bad in [
            SessionConfig { width: 40, ..cfg() },
            SessionConfig { memory_latent_frames: 0, ..cfg() },
            SessionConfig { history_latent_frames: 3, ..cfg() },
            SessionConfig { chunk_latent_frames: 0, ..cfg() },
            SessionConfig { fps: 0.0, ..cfg() },
        ] {
            assert!(matches!(WorldSession::start(&toy, bad, "a", "", None), Err(Error::Request(_))));
        }
        let small = RgbImage { width: 16, height: 32, rgb: vec![0; 16 * 32 * 3] };
        assert!(WorldSession::start(&toy, cfg(), "a", "", Some(&small)).is_err());
    }
}
