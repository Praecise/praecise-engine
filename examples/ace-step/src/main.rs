//! Generate music with an ACE-Step 1.5 checkpoint and report per-stage
//! timings. The first run warms the backend; later runs are the steady state.
//!
//! ```text
//! cargo run --release -p ace-step --features cuda -- \
//!     --model <checkpoint dir> --prompt "warm lo-fi hip hop, mellow piano" \
//!     --lyrics-file lyrics.txt --duration 30 --runs 2 --out out.wav
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use praecise_diffusion::{AceStep, CheckpointFiles, LoadOptions, MusicRequest, Precision};

#[derive(Parser, Debug)]
struct Args {
    /// Checkpoint directory (holding `model_index.json`).
    #[arg(long)]
    model: PathBuf,
    /// Description of the music.
    #[arg(long)]
    prompt: String,
    /// File holding the lyrics; instrumental when absent.
    #[arg(long)]
    lyrics_file: Option<PathBuf>,
    /// Language code of the lyrics.
    #[arg(long, default_value = "en")]
    language: String,
    /// Length in seconds.
    #[arg(long, default_value_t = 30.0)]
    duration: f32,
    /// Denoising steps; the checkpoint's default when absent.
    #[arg(long)]
    steps: Option<u32>,
    /// Guidance scale (ignored by guidance-distilled checkpoints).
    #[arg(long)]
    guidance: Option<f32>,
    /// Tempo in beats per minute.
    #[arg(long)]
    bpm: Option<u32>,
    /// Key and scale, such as "A minor".
    #[arg(long)]
    keyscale: Option<String>,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Weight format: bf16, q8_0 or f32.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// Number of generations; the first is reported as warm-up.
    #[arg(long, default_value_t = 2)]
    runs: u32,
    /// Where to write the last result (16-bit WAV).
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
    let mut pipeline = AceStep::load(&CheckpointFiles::new(&args.model), LoadOptions { precision, ..Default::default() })?;
    println!(
        "loaded on {} in {} ms, {:.2} GiB resident",
        pipeline.device(),
        t.elapsed().as_millis(),
        pipeline.resident_bytes() as f64 / (1u64 << 30) as f64
    );
    let lyrics = match &args.lyrics_file {
        Some(p) => std::fs::read_to_string(p)?,
        None => String::new(),
    };
    let req = MusicRequest {
        prompt: args.prompt.clone(),
        lyrics,
        language: args.language.clone(),
        duration_secs: args.duration,
        steps: args.steps,
        guidance_scale: args.guidance,
        shift: None,
        seed: args.seed,
        bpm: args.bpm,
        keyscale: args.keyscale.clone(),
        timesignature: None,
    };
    let mut last = None;
    for run in 0..args.runs {
        let t = Instant::now();
        let audio = pipeline.generate(&req)?;
        let total = t.elapsed().as_millis();
        let tm = audio.timings;
        println!(
            "run {run}{}: total {total} ms for {:.1} s of audio (encode {} ms, denoise {} ms over {} evaluations, decode {} ms)",
            if run == 0 { " (warm-up)" } else { "" },
            audio.frames() as f64 / f64::from(audio.sample_rate),
            tm.encode_ms,
            tm.denoise_ms,
            audio.evaluations,
            tm.decode_ms
        );
        last = Some(audio);
    }
    if let (Some(path), Some(audio)) = (args.out, last) {
        std::fs::write(path, audio.wav_pcm16())?;
    }
    Ok(())
}
