//! Generate or edit an image with a released Qwen-Image checkpoint and report
//! per-stage timings and resident weight bytes.
//!
//! ```text
//! cargo run --release -p qwen-image --features cuda -- \
//!     --pipeline qwen-image-2.1 --model <checkpoint dir> --prompt "a lighthouse at dusk" --out out.ppm
//! cargo run --release -p qwen-image --features cuda -- \
//!     --pipeline qwen-image-edit --model <checkpoint dir> --reference in.ppm \
//!     --prompt "make the sky purple" --precision q8_0 --out edit.ppm
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context};
use clap::Parser;
use praecise_diffusion::{CheckpointFiles, Image, LoadOptions, Precision, QwenImage21, QwenImageEdit, Request, RgbImage};

#[derive(Parser, Debug)]
struct Args {
    /// `qwen-image-2.1` (text to image, optional references) or
    /// `qwen-image-edit` (instruction editing, at least one reference).
    #[arg(long)]
    pipeline: String,
    /// Checkpoint directory (holding `model_index.json`).
    #[arg(long)]
    model: PathBuf,
    /// Prompt.
    #[arg(long)]
    prompt: String,
    /// Reference images (binary PPM), in order.
    #[arg(long)]
    reference: Vec<PathBuf>,
    /// Width in pixels; defaults to the pipeline's size for the references.
    #[arg(long)]
    width: Option<u32>,
    /// Height in pixels; defaults to the pipeline's size for the references.
    #[arg(long)]
    height: Option<u32>,
    /// Denoising steps.
    #[arg(long, default_value_t = 40)]
    steps: u32,
    /// Guidance scale.
    #[arg(long, default_value_t = 4.0)]
    guidance: f32,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Weight format: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// Where to write the image (binary PPM).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn read_ppm(path: &Path) -> anyhow::Result<RgbImage> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while i < data.len() && data[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < data.len() && data[i] == b'#' {
            while i < data.len() && data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < data.len() && !data[i].is_ascii_whitespace() {
            i += 1;
        }
        if start == i {
            bail!("{}: truncated header", path.display());
        }
        fields.push(std::str::from_utf8(&data[start..i])?.to_string());
    }
    if fields[0] != "P6" || fields[3] != "255" {
        bail!("{}: only 8-bit binary PPM (P6) is read", path.display());
    }
    let (width, height): (u32, u32) = (fields[1].parse()?, fields[2].parse()?);
    let rgb = data[i + 1..].to_vec();
    if rgb.len() != (width * height * 3) as usize {
        bail!("{}: pixel data does not match {width}x{height}", path.display());
    }
    Ok(RgbImage { width, height, rgb })
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let precision = match args.precision.as_str() {
        "bf16" => Precision::Bf16,
        "q8_0" => Precision::Q8_0,
        "f32" => Precision::F32,
        other => bail!("unknown precision {other}"),
    };
    let references = args.reference.iter().map(|p| read_ppm(p)).collect::<anyhow::Result<Vec<_>>>()?;
    let files = CheckpointFiles::new(&args.model);
    let opts = LoadOptions { precision, ..Default::default() };
    let t = Instant::now();
    let img: Image = match args.pipeline.as_str() {
        "qwen-image-2.1" => {
            let mut p = QwenImage21::load(&files, opts)?;
            let (w, h) = QwenImage21::default_size(references.first());
            println!("loaded on {} in {} ms, resident {} bytes", p.device(), t.elapsed().as_millis(), p.resident_bytes());
            let req = request(&args, references, (w, h));
            p.generate(&req)?
        }
        "qwen-image-edit" => {
            let Some(first) = references.first() else { bail!("qwen-image-edit needs --reference") };
            let (w, h) = QwenImageEdit::default_size(first);
            let mut p = QwenImageEdit::load(&files, opts)?;
            println!("loaded on {} in {} ms, resident {} bytes", p.device(), t.elapsed().as_millis(), p.resident_bytes());
            let req = request(&args, references, (w, h));
            p.generate(&req)?
        }
        other => bail!("unknown pipeline {other}"),
    };
    let tm = img.timings;
    println!(
        "{}x{}: encode {} ms, denoise {} ms over {} evaluations, decode {} ms",
        img.width, img.height, tm.encode_ms, tm.denoise_ms, img.evaluations, tm.decode_ms
    );
    if let Some(path) = &args.out {
        let mut f = std::fs::File::create(path)?;
        write!(f, "P6\n{} {}\n255\n", img.width, img.height)?;
        f.write_all(&img.rgb)?;
    }
    Ok(())
}

fn request(args: &Args, references: Vec<RgbImage>, (w, h): (u32, u32)) -> Request {
    Request {
        prompt: args.prompt.clone(),
        references,
        width: args.width.unwrap_or(w),
        height: args.height.unwrap_or(h),
        steps: args.steps,
        guidance_scale: args.guidance,
        seed: args.seed,
    }
}
