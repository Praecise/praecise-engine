//! Generate an image with a FLUX.2 [klein] checkpoint and report per-stage
//! timings. The first run warms the backend; later runs are the steady state.
//!
//! ```text
//! cargo run --release -p flux2 --features cuda -- \
//!     --model <checkpoint dir> --prompt "a lighthouse at dusk" --runs 4 --out out.ppm
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::{CheckpointFiles, Flux2Klein, LoadOptions, Precision, Request};

#[derive(Parser, Debug)]
struct Args {
    /// Checkpoint directory (holding `model_index.json`).
    #[arg(long)]
    model: PathBuf,
    /// Prompt.
    #[arg(long)]
    prompt: String,
    /// Width in pixels.
    #[arg(long, default_value_t = 1024)]
    width: u32,
    /// Height in pixels.
    #[arg(long, default_value_t = 1024)]
    height: u32,
    /// Denoising steps.
    #[arg(long, default_value_t = 4)]
    steps: u32,
    /// Guidance scale (ignored by step-distilled checkpoints).
    #[arg(long, default_value_t = 1.0)]
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
    /// Where to write the last image (binary PPM).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Where to write the starting noise (little-endian f32, tokens by
    /// channels), so another implementation can start from the same latent.
    #[arg(long)]
    noise_out: Option<PathBuf>,
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
    let mut pipeline = Flux2Klein::load(&CheckpointFiles::new(&args.model), LoadOptions { precision, ..Default::default() })?;
    println!(
        "loaded on {} in {} ms, {:.2} GiB resident",
        pipeline.device(),
        t.elapsed().as_millis(),
        pipeline.resident_bytes() as f64 / (1u64 << 30) as f64
    );
    let req = Request {
        prompt: args.prompt.clone(),
        references: Vec::new(),
        width: args.width,
        height: args.height,
        steps: args.steps,
        guidance_scale: args.guidance,
        seed: args.seed,
    };
    if let Some(path) = &args.noise_out {
        let n = (args.width / 16) as usize * (args.height / 16) as usize * 128;
        let noise = praecise_diffusion::schedule::gaussian(args.seed, n);
        std::fs::write(path, noise.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let mut last = None;
    for run in 0..args.runs {
        let t = Instant::now();
        let img = pipeline.generate(&req)?;
        let total = t.elapsed().as_millis();
        let tm = img.timings;
        println!(
            "run {run}{}: total {total} ms (encode {} ms, denoise {} ms over {} evaluations, decode {} ms)",
            if run == 0 { " (warm-up)" } else { "" },
            tm.encode_ms,
            tm.denoise_ms,
            img.evaluations,
            tm.decode_ms
        );
        last = Some(img);
    }
    if let (Some(path), Some(img)) = (args.out, last) {
        let mut f = std::fs::File::create(path)?;
        write!(f, "P6\n{} {}\n255\n", img.width, img.height)?;
        f.write_all(&img.rgb)?;
    }
    Ok(())
}
