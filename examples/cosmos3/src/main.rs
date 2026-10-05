//! Generate a video with a Cosmos3 checkpoint, optionally from a first frame,
//! and report per-stage timings. The first run warms the backend; later runs
//! are the steady state.
//!
//! ```text
//! cargo run --release -p cosmos3 --features cuda -- \
//!     --model <checkpoint dir> --prompt "a red kite over a beach" \
//!     --image first.ppm --frames 121 --runs 2 --out out.mp4
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::{CheckpointFiles, Cosmos3, LoadOptions, Precision, RgbImage, VideoRequest};

#[derive(Parser, Debug)]
struct Args {
    /// Checkpoint directory (holding `model_index.json`).
    #[arg(long)]
    model: PathBuf,
    /// What to show.
    #[arg(long)]
    prompt: String,
    /// What to avoid.
    #[arg(long)]
    negative: Option<String>,
    /// First frame (binary PPM at exactly the output size); text-to-video
    /// when absent.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Width in pixels, a multiple of 32.
    #[arg(long, default_value_t = 832)]
    width: u32,
    /// Height in pixels, a multiple of 32.
    #[arg(long, default_value_t = 480)]
    height: u32,
    /// Frames: 1 or 4k + 1.
    #[arg(long, default_value_t = 121)]
    frames: u32,
    /// Frame rate.
    #[arg(long, default_value_t = 24.0)]
    fps: f32,
    /// Denoising steps.
    #[arg(long, default_value_t = 35)]
    steps: u32,
    /// Guidance scale; 1 disables guidance.
    #[arg(long, default_value_t = 6.0)]
    guidance: f32,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Weight format: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// Number of generations; the first is reported as warm-up.
    #[arg(long, default_value_t = 2)]
    runs: u32,
    /// Where to write the last result (MP4).
    #[arg(long)]
    out: Option<PathBuf>,
}

/// A binary PPM (P6, 8-bit) image.
fn read_ppm(path: &Path) -> anyhow::Result<RgbImage> {
    let bytes = std::fs::read(path)?;
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) == Some(&b'#') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        anyhow::ensure!(start < i, "truncated PPM header");
        fields.push(std::str::from_utf8(&bytes[start..i])?.to_string());
    }
    anyhow::ensure!(fields[0] == "P6" && fields[3] == "255", "expected an 8-bit binary PPM");
    let (width, height): (u32, u32) = (fields[1].parse()?, fields[2].parse()?);
    let rgb = bytes.get(i + 1..).unwrap_or_default().to_vec();
    anyhow::ensure!(rgb.len() == (width * height * 3) as usize, "PPM pixel data does not match its size");
    Ok(RgbImage { width, height, rgb })
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let precision = match args.precision.as_str() {
        "bf16" => Precision::Bf16,
        "q8_0" => Precision::Q8_0,
        "f32" => Precision::F32,
        other => anyhow::bail!("unknown precision {other}"),
    };
    let t = Instant::now();
    let mut pipeline = Cosmos3::load(&CheckpointFiles::new(&args.model), LoadOptions { precision, ..Default::default() })?;
    println!(
        "loaded on {} in {} ms, {:.2} GiB resident",
        pipeline.device(),
        t.elapsed().as_millis(),
        pipeline.resident_bytes() as f64 / (1u64 << 30) as f64
    );
    let req = VideoRequest {
        prompt: args.prompt.clone(),
        negative_prompt: args.negative.clone(),
        image: args.image.as_deref().map(read_ppm).transpose()?,
        width: args.width,
        height: args.height,
        num_frames: args.frames,
        fps: args.fps,
        steps: args.steps,
        guidance_scale: args.guidance,
        seed: args.seed,
    };
    let mut last = None;
    for run in 0..args.runs {
        let t = Instant::now();
        let video = pipeline.generate(&req)?;
        let tm = video.timings;
        println!(
            "run {run}{}: total {} ms for {} frames (encode {} ms, denoise {} ms over {} evaluations, decode {} ms)",
            if run == 0 { " (warm-up)" } else { "" },
            t.elapsed().as_millis(),
            video.frames,
            tm.encode_ms,
            tm.denoise_ms,
            video.evaluations,
            tm.decode_ms
        );
        last = Some(video);
    }
    if let (Some(path), Some(video)) = (args.out, last) {
        std::fs::write(path, video.mp4(None)?)?;
    }
    Ok(())
}
