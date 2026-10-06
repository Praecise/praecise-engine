//! Generate one LTX-2 audio-video clip from a prompt.
//!
//! ```text
//! cargo run --release -p ltx2 -- --checkpoint <dir with the checkpoint files> \
//!     --text <dir holding text_encoder/ and tokenizer/> --distilled --upsample \
//!     --prompt "waves rolling onto a beach at sunset" --out clip.mp4
//! ```
//!
//! `--checkpoint` may be given more than once (a GGUF transformer in one
//! directory, its VAEs and upsampler in another). `--frames-out` also writes
//! every frame as one tall binary PPM.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::{LoadOptions, Ltx2Pipeline, Ltx2Request, Precision};

#[derive(Parser, Debug)]
struct Args {
    /// Directory searched for checkpoint files (repeatable).
    #[arg(long, required = true)]
    checkpoint: Vec<PathBuf>,
    /// Directory holding `text_encoder/` and `tokenizer/tokenizer.json`.
    #[arg(long)]
    text: PathBuf,
    /// What to show.
    #[arg(long)]
    prompt: String,
    /// What to avoid (ignored by a distilled checkpoint).
    #[arg(long, default_value = "")]
    negative: String,
    /// Run the distilled schedule without guidance.
    #[arg(long)]
    distilled: bool,
    /// Two stages: half size, latent upsampling, refinement (distilled only).
    #[arg(long)]
    upsample: bool,
    /// Width in pixels.
    #[arg(long, default_value_t = 768)]
    width: usize,
    /// Height in pixels.
    #[arg(long, default_value_t = 512)]
    height: usize,
    /// Frames (one more than a multiple of 8).
    #[arg(long, default_value_t = 121)]
    frames: usize,
    /// Frame rate.
    #[arg(long, default_value_t = 24.0)]
    fps: f32,
    /// Denoising steps (the distilled schedule fixes its own).
    #[arg(long, default_value_t = 30)]
    steps: usize,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Weight precision: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// MP4 output.
    #[arg(long)]
    out: PathBuf,
    /// Every frame as one tall binary PPM.
    #[arg(long)]
    frames_out: Option<PathBuf>,
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
    let p = Ltx2Pipeline::load_dir(&a.checkpoint, &a.text, opts)?;
    println!("loaded on {} in {:.1}s, {:.2} GB resident", p.device(), t.elapsed().as_secs_f64(), p.resident_bytes() as f64 / 1e9);
    let base = if a.distilled { Ltx2Request::distilled(String::new(), a.upsample) } else { Ltx2Request::default() };
    let req = Ltx2Request {
        prompt: a.prompt.clone(),
        negative_prompt: a.negative.clone(),
        width: a.width,
        height: a.height,
        num_frames: a.frames,
        fps: a.fps,
        steps: a.steps,
        seed: a.seed,
        ..base
    };
    let t = Instant::now();
    let out = p.generate(&req)?;
    let v = &out.video;
    let px = v.rgb.iter().map(|&b| f64::from(b)).collect::<Vec<_>>();
    let mean = px.iter().sum::<f64>() / px.len() as f64;
    let sd = (px.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / px.len() as f64).sqrt();
    let tm = v.timings;
    println!(
        "clip: {} frames {}x{} in {:.1}s (encode {} ms, denoise {} ms, decode {} ms), {} evaluations, pixel mean {mean:.1} sd {sd:.1}",
        v.frames,
        v.width,
        v.height,
        t.elapsed().as_secs_f64(),
        tm.encode_ms,
        tm.denoise_ms,
        tm.decode_ms,
        v.evaluations
    );
    let s = &out.audio.samples;
    let rms = (s.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / s.len().max(1) as f64).sqrt();
    let peak = s.iter().fold(0f32, |m, x| m.max(x.abs()));
    println!("sound: {} Hz x {}, {:.2}s, rms {rms:.4}, peak {peak:.3}", out.audio.sample_rate, out.audio.channels, s.len() as f64 / f64::from(out.audio.sample_rate * out.audio.channels.max(1)));
    std::fs::write(&a.out, out.mp4()?)?;
    if let Some(path) = &a.frames_out {
        let mut f = std::fs::File::create(path)?;
        write!(f, "P6\n{} {}\n255\n", v.width, v.height * v.frames)?;
        f.write_all(&v.rgb)?;
    }
    Ok(())
}
