//! Run a Wan2.2 world-model session: a prompt (and optionally a first
//! frame) rolled forward chunk by chunk from a bounded memory, each chunk
//! streamed out as it is decoded. Reports per-chunk timings and pixel
//! statistics and writes the stream as Y4M.
//!
//! ```text
//! cargo run --release -p wan-world -- --model <checkpoint dir> \
//!     --prompt "a car driving down a coastal road" --chunks 3 --out out.y4m
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::wan_video::Wan22;
use praecise_diffusion::{CheckpointFiles, LoadOptions, Precision, SessionConfig, WorldSession};

#[derive(Parser, Debug)]
struct Args {
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
    /// Y4M output.
    #[arg(long)]
    out: Option<PathBuf>,
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
    let t = Instant::now();
    let wan = Wan22::load(&CheckpointFiles::new(&a.model), LoadOptions { precision, cpu_threads: threads, device: None })?;
    println!("loaded on {} in {:.1}s, {:.2} GB resident", wan.device(), t.elapsed().as_secs_f64(), wan.resident_bytes() as f64 / 1e9);
    let cfg = SessionConfig {
        width: a.width,
        height: a.height,
        fps: a.fps,
        chunk_latent_frames: a.chunk,
        memory_latent_frames: a.memory,
        history_latent_frames: a.memory,
        steps: a.steps,
        guidance_scale: a.guidance,
        seed: a.seed,
    };
    let mut s = WorldSession::start(&wan, cfg, &a.prompt, &a.negative, None)?;
    let mut out = match &a.out {
        Some(p) => {
            let mut f = std::io::BufWriter::new(std::fs::File::create(p)?);
            write!(f, "YUV4MPEG2 W{} H{} F{}:1 Ip A1:1 C444\n", a.width, a.height, a.fps.round() as u32)?;
            Some(f)
        }
        None => None,
    };
    for _ in 0..a.chunks {
        let t = Instant::now();
        let c = s.step(&[])?;
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
        if let Some(f) = out.as_mut() {
            let plane = (a.width * a.height) as usize;
            for fr in c.rgb.chunks_exact(plane * 3) {
                f.write_all(b"FRAME\n")?;
                for ch in 0..3 {
                    let rgb_to = |p: &[u8]| -> u8 {
                        let (r, g, b) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
                        let v = match ch {
                            0 => 0.299 * r + 0.587 * g + 0.114 * b,
                            1 => 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b,
                            _ => 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b,
                        };
                        v.round().clamp(0.0, 255.0) as u8
                    };
                    let row: Vec<u8> = fr.chunks_exact(3).map(rgb_to).collect();
                    f.write_all(&row)?;
                }
            }
        }
    }
    println!("simulated {:.3}s over {} chunks, memory {:?}", s.simulated_secs(), s.chunks(), s.memory().map(|f| f.index).collect::<Vec<_>>());
    Ok(())
}
