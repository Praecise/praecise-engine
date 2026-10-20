//! Generate an image with a FLUX 3 image checkpoint and report per-stage
//! timings and resident memory. The first run warms the backend; later runs
//! are the steady state.
//!
//! ```text
//! cargo run --release -p flux3-image --features cuda -- \
//!     --model <checkpoint dir> --prompt "a lighthouse at dusk" --runs 2 --out out.ppm
//! ```
//!
//! `--variant fp8r` or `--variant nvfp4` picks a quantised transformer from
//! `variants/`.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::{Flux3Image, LoadOptions, Precision, Request};

#[derive(Parser, Debug)]
struct Args {
    /// Checkpoint directory (package or diffusers layout).
    #[arg(long)]
    model: PathBuf,
    /// Transformer variant under `variants/`.
    #[arg(long)]
    variant: Option<String>,
    /// Prompt: a full scene caption.
    #[arg(long)]
    prompt: String,
    /// Width in pixels, a multiple of 16.
    #[arg(long, default_value_t = 1024)]
    width: u32,
    /// Height in pixels, a multiple of 16.
    #[arg(long, default_value_t = 1024)]
    height: u32,
    /// Sampling steps (16 for a flow map, 50 for an undistilled model).
    #[arg(long, default_value_t = 16)]
    steps: u32,
    /// Guidance scale (ignored by a flow map).
    #[arg(long, default_value_t = 1.0)]
    guidance: f32,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Time shift of the sampling grid (default: the checkpoint's).
    #[arg(long)]
    shift: Option<f32>,
    /// Weight format: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// Number of generations; the first is reported as warm-up.
    #[arg(long, default_value_t = 2)]
    runs: u32,
    /// Where to write the last image (binary PPM).
    #[arg(long)]
    out: Option<PathBuf>,
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
    let mut pipeline = Flux3Image::load(&args.model, args.variant.as_deref(), LoadOptions { precision, ..Default::default() })?;
    if let Some(shift) = args.shift {
        pipeline.set_shift(shift)?;
    }
    println!(
        "loaded {:?} on {} in {} ms, flow map {}, {:.2} GiB resident",
        pipeline.transformer_source(),
        pipeline.device(),
        t.elapsed().as_millis(),
        pipeline.is_flow_map(),
        pipeline.resident_bytes() as f64 / f64::from(1u32 << 30)
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
    let mut last = None;
    for run in 0..args.runs {
        let t = Instant::now();
        let img = pipeline.generate(&req)?;
        let tm = img.timings;
        println!(
            "run {run}{}: total {} ms (encode {} ms, denoise {} ms over {} evaluations, decode {} ms)",
            if run == 0 { " (warm-up)" } else { "" },
            t.elapsed().as_millis(),
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
