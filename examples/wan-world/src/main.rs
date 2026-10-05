//! Run a world-model session: a prompt (and optionally a first frame)
//! rolled forward chunk by chunk from a bounded memory, each chunk
//! streamed out as it is decoded. With `--world` the action world model
//! runs on the Wan2.2 checkpoint's text encoder and autoencoder, from a
//! first frame (a P6 PPM, or a plain horizon) and a scripted walk that
//! moves forward and turns. Reports per-chunk timings and pixel
//! statistics and writes the stream as MP4.
//!
//! ```text
//! cargo run --release -p wan-world -- --model <checkpoint dir> \
//!     --prompt "a car driving down a coastal road" --chunks 3 --out out.mp4
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::wan_video::Wan22;
use praecise_diffusion::{CheckpointFiles, LoadOptions, MatrixGame, Precision, RgbImage, SessionConfig, WorldModel, WorldSession};

#[derive(Parser, Debug)]
struct Args {
    /// Run one plain clip of this many frames instead of a session and
    /// write its frames as a binary PPM strip to `--out`.
    #[arg(long)]
    plain_frames: Option<u32>,
    /// Checkpoint directory (holding `model_index.json`).
    #[arg(long)]
    model: PathBuf,
    /// What to show.
    #[arg(long)]
    prompt: String,
    /// What to avoid.
    #[arg(long, default_value = "")]
    negative: String,
    /// Width in pixels, a multiple of 32.
    #[arg(long, default_value_t = 256)]
    width: u32,
    /// Height in pixels, a multiple of 32.
    #[arg(long, default_value_t = 256)]
    height: u32,
    /// Latent frames per chunk.
    #[arg(long, default_value_t = 2)]
    chunk: usize,
    /// Latent frames kept between chunks.
    #[arg(long, default_value_t = 2)]
    memory: usize,
    /// Chunks to generate.
    #[arg(long, default_value_t = 3)]
    chunks: usize,
    /// Denoising steps per chunk.
    #[arg(long, default_value_t = 8)]
    steps: u32,
    /// Guidance scale (1 disables it).
    #[arg(long, default_value_t = 5.0)]
    guidance: f32,
    /// Frames per simulated second.
    #[arg(long, default_value_t = 24.0)]
    fps: f32,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Weight precision: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// MP4 output.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Action world-model checkpoint directory; `--model` then gives the
    /// Wan2.2 text encoder and autoencoder.
    #[arg(long)]
    world: Option<PathBuf>,
    /// First frame (binary PPM) for the action world model.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Latent frames held between chunks (defaults to the memory).
    #[arg(long)]
    history: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let a = Args::parse();
    let precision = match a.precision.as_str() {
        "f32" => Precision::F32,
        "q8_0" => Precision::Q8_0,
        _ => Precision::Bf16,
    };
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let opts = LoadOptions { precision, cpu_threads: threads, device: None };
    let t = Instant::now();
    if let Some(world) = &a.world {
        let m = MatrixGame::load(&CheckpointFiles::new(world), &CheckpointFiles::new(&a.model), opts)?;
        println!("loaded on {} in {:.1}s, {:.2} GB resident", m.device(), t.elapsed().as_secs_f64(), m.resident_bytes() as f64 / 1e9);
        let first = match &a.image {
            Some(p) => ppm(p)?,
            None => horizon(a.width, a.height),
        };
        return run(&m, &a, Some(&first));
    }
    let mut wan = Wan22::load(&CheckpointFiles::new(&a.model), opts)?;
    println!("loaded on {} in {:.1}s, {:.2} GB resident", wan.device(), t.elapsed().as_secs_f64(), wan.resident_bytes() as f64 / 1e9);
    if let Some(n) = a.plain_frames {
        let req = praecise_diffusion::VideoRequest { prompt: a.prompt.clone(), negative_prompt: Some(a.negative.clone()), image: None, width: a.width, height: a.height, num_frames: n, fps: a.fps, steps: a.steps, guidance_scale: a.guidance, seed: a.seed };
        let v = wan.generate(&req)?;
        let px = v.rgb.iter().map(|&b| f64::from(b)).collect::<Vec<_>>();
        let mean = px.iter().sum::<f64>() / px.len() as f64;
        let sd = (px.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / px.len() as f64).sqrt();
        println!("plain: {} frames {}x{} in {:.1}s, {} evaluations, pixel mean {mean:.1} sd {sd:.1}", v.frames, v.width, v.height, t.elapsed().as_secs_f64(), v.evaluations);
        let mut f = std::fs::File::create(a.out.as_ref().expect("--out names the PPM"))?;
        write!(f, "P6\n{} {}\n255\n", v.width, v.height * v.frames)?;
        f.write_all(&v.rgb)?;
        return Ok(());
    }
    run(&wan, &a, None)
}

/// A plain sky over ground.
fn horizon(w: u32, h: u32) -> RgbImage {
    let rgb = (0..h).flat_map(|y| (0..w).flat_map(move |_| if y < h / 2 { [120, 170, 230] } else { [90, 120, 60] })).collect();
    RgbImage { width: w, height: h, rgb }
}

/// Read a binary PPM (P6, 8-bit).
fn ppm(p: &PathBuf) -> anyhow::Result<RgbImage> {
    let b = std::fs::read(p)?;
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while b.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        let s = i;
        while b.get(i).is_some_and(|c| !c.is_ascii_whitespace()) {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&b[s..i]).into_owned());
    }
    anyhow::ensure!(fields[0] == "P6" && fields[3] == "255", "only 8-bit P6 images are read");
    let (width, height): (u32, u32) = (fields[1].parse()?, fields[2].parse()?);
    let rgb = b.get(i + 1..i + 1 + (width * height * 3) as usize).ok_or_else(|| anyhow::anyhow!("short image"))?.to_vec();
    Ok(RgbImage { width, height, rgb })
}

/// One scripted action row per pixel frame: forward, turning right.
fn walk(dims: usize, rows: usize) -> Vec<f32> {
    if dims == 0 {
        return Vec::new();
    }
    (0..rows).flat_map(|_| (0..dims).map(|d| match d { 0 => 1.0, _ if d == dims - 1 => 0.1, _ => 0.0 })).collect()
}

fn run<M: WorldModel>(model: &M, a: &Args, first: Option<&RgbImage>) -> anyhow::Result<()> {
    let cfg = SessionConfig {
        width: a.width,
        height: a.height,
        fps: a.fps,
        chunk_latent_frames: a.chunk,
        memory_latent_frames: a.memory,
        history_latent_frames: a.history.unwrap_or(a.memory),
        steps: a.steps,
        guidance_scale: a.guidance,
        seed: a.seed,
    };
    let mut s = WorldSession::start(model, cfg, &a.prompt, &a.negative, first)?;
    let mut out = match &a.out {
        Some(_) => Some(praecise_diffusion::codec::VideoWriter::new(praecise_diffusion::codec::EncodeOptions::default(), a.width, a.height, a.fps)?),
        None => None,
    };
    for _ in 0..a.chunks {
        let t = Instant::now();
        let c = s.step(&walk(model.action_dims(), s.pixel_frames_next()))?;
        let n = c.rgb.len() as f64;
        let mean = c.rgb.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let sd = (c.rgb.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n).sqrt();
        println!(
            "chunk {}: frames {}..{} ({} frames, {:.3} simulated s) in {:.1}s, {} evaluations, pixel mean {mean:.1} sd {sd:.1}",
            c.chunk,
            c.first_frame,
            c.first_frame + u64::from(c.frames),
            c.frames,
            c.simulated_secs,
            t.elapsed().as_secs_f64(),
            c.evaluations
        );
        if let Some(w) = out.as_mut() {
            for fr in c.rgb.chunks_exact((a.width * a.height) as usize * 3) {
                w.push(fr)?;
            }
        }
    }
    if let (Some(w), Some(p)) = (out, &a.out) {
        std::fs::write(p, w.finish(None)?)?;
    }
    println!("simulated {:.3}s over {} chunks, memory {:?}", s.simulated_secs(), s.chunks(), s.memory().map(|f| f.index).collect::<Vec<_>>());
    Ok(())
}
