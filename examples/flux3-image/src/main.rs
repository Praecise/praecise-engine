//! Generate or edit an image with FLUX 3 Image Klein and report per-stage
//! timings and memory. The first run warms the backend; later runs are the
//! steady state.
//!
//! ```text
//! cargo run --release -p flux3-image --features cuda -- \
//!     --model <release dir> --prompt "a lighthouse at dusk" --runs 3 --out out.ppm
//! # editing: one or more binary PPM (P6) references
//! cargo run --release -p flux3-image --features cuda -- \
//!     --model <release dir> --prompt "make it snow" --reference in.ppm --out out.ppm
//! ```
//!
//! The release directory holds `model.safetensors` (or a quantised entry's
//! `variants/<fp8r|nvfp4>/model.safetensors`), `vae/` and `text_encoder/`.
//! The distilled model takes 16 steps without guidance (the defaults); the
//! base model 50 steps at guidance 4.0 (`--steps 50 --guidance 4`).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::flux3::image::{Flux3Image, OutputSize, Recipe};
use praecise_diffusion::{CheckpointFiles, LoadOptions, Precision, Request, RgbImage};

#[derive(Parser, Debug)]
struct Args {
    /// Release directory.
    #[arg(long)]
    model: PathBuf,
    /// Prompt, used as given (no prompt upsampling).
    #[arg(long)]
    prompt: String,
    /// Reference images (binary PPM), at most ten; one or more makes an edit.
    #[arg(long)]
    reference: Vec<PathBuf>,
    /// Width in pixels; with a reference and no size, the reference's.
    #[arg(long)]
    width: Option<u32>,
    /// Height in pixels.
    #[arg(long)]
    height: Option<u32>,
    /// Denoising steps.
    #[arg(long, default_value_t = 16)]
    steps: u32,
    /// Guidance scale (ignored by the distilled model).
    #[arg(long, default_value_t = 1.0)]
    guidance: f32,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Weight format: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// Sampling numbers: `reference` (the release's own code) or `diffusers`.
    #[arg(long, default_value = "reference")]
    recipe: String,
    /// Number of generations; the first is reported as warm-up.
    #[arg(long, default_value_t = 2)]
    runs: u32,
    /// Where to write the last image (binary PPM).
    #[arg(long)]
    out: Option<PathBuf>,
}

/// A binary PPM (P6, maxval 255).
fn read_ppm(path: &Path) -> anyhow::Result<RgbImage> {
    let bytes = std::fs::read(path)?;
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
            at += 1;
        }
        let start = at;
        while bytes.get(at).is_some_and(|b| !b.is_ascii_whitespace()) {
            at += 1;
        }
        anyhow::ensure!(at > start, "{}: not a PPM", path.display());
        fields.push(std::str::from_utf8(&bytes[start..at])?.to_owned());
    }
    anyhow::ensure!(fields[0] == "P6" && fields[3] == "255", "{}: only binary 8-bit PPM", path.display());
    let (width, height): (u32, u32) = (fields[1].parse()?, fields[2].parse()?);
    let rgb = bytes[at + 1..].to_vec();
    anyhow::ensure!(rgb.len() == width as usize * height as usize * 3, "{}: truncated", path.display());
    Ok(RgbImage { width, height, rgb })
}

/// Peak resident memory of this process, from the kernel (Linux).
fn peak_rss_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = status.lines().find_map(|l| l.strip_prefix("VmHWM:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb / (1024.0 * 1024.0))
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
    let recipe = match args.recipe.as_str() {
        "reference" => Recipe::REFERENCE,
        "diffusers" => Recipe::DIFFUSERS,
        other => anyhow::bail!("unknown recipe {other}"),
    };
    let t = Instant::now();
    let pipeline = Flux3Image::load_release(&CheckpointFiles::new(&args.model), LoadOptions { precision, ..Default::default() })?
        .with_recipe(recipe);
    println!(
        "loaded on {} in {} ms, {:.2} GiB resident ({}), peak process memory {:.2} GiB",
        pipeline.device(),
        t.elapsed().as_millis(),
        pipeline.resident_bytes() as f64 / (1u64 << 30) as f64,
        if pipeline.is_distilled() { "distilled flow map" } else { "base, guided" },
        peak_rss_gib().unwrap_or(f64::NAN)
    );
    let references = args.reference.iter().map(|p| read_ppm(p)).collect::<anyhow::Result<Vec<_>>>()?;
    let size = if args.width.is_none() && args.height.is_none() { OutputSize::FromReference } else { OutputSize::AsRequested };
    let req = Request {
        prompt: args.prompt.clone(),
        references,
        width: args.width.unwrap_or(1024),
        height: args.height.unwrap_or(1024),
        steps: args.steps,
        guidance_scale: args.guidance,
        seed: args.seed,
    };
    let mut last = None;
    for run in 0..args.runs {
        let t = Instant::now();
        let img = pipeline.edit(&req, size)?;
        let total = t.elapsed().as_millis();
        let tm = img.timings;
        println!(
            "run {run}{}: {}x{} total {total} ms (encode {} ms, denoise {} ms over {} evaluations, decode {} ms), peak process memory {:.2} GiB",
            if run == 0 { " (warm-up)" } else { "" },
            img.width,
            img.height,
            tm.encode_ms,
            tm.denoise_ms,
            img.evaluations,
            tm.decode_ms,
            peak_rss_gib().unwrap_or(f64::NAN)
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
