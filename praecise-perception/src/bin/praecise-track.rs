//! praecise-track: video in, per-frame object tracks out.
//!
//! ```text
//! praecise-track --model rfdetr-medium.onnx --input-size 576 --video in.mp4 --out out/ [--nvframe]
//! ```
//!
//! Writes `out/tracks.json` (every frame's tracks and the run's summary),
//! `out/summary.json`, and with `--nvframe` `out/frames.nvpb`: one NVIDIA
//! `nv.Frame` per video frame, each prefixed by its length as a varint.
//! Progress goes to stderr as JSON lines `{"progress": f, "message": m}`; the
//! summary is printed to stdout at the end.
//!
//! On a machine with a GPU the detector runs on it or the run is refused;
//! `--device cpu` asks for the CPU explicitly.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use praecise_onnx::detection::{DetrFamily, GenericDetrDetector};
use praecise_onnx::session::{self, Gpus, OnnxConfig, OnnxSettings};
use praecise_perception::pipeline::{self, PipelineConfig};
use serde_json::json;

const USAGE: &str = "usage: praecise-track --model FILE.onnx --video FILE --out DIR [--family rf-detr|d-fine] \
[--input-size N] [--nvframe] [--sensor ID] [--epoch SECONDS] [--ground h11,h12,...,h33] [--max-frames N] \
[--high S] [--low S] [--new-track S] [--match D] [--buffer FRAMES] [--device auto|cpu] [--threads N]";

struct Args {
    model: PathBuf,
    video: PathBuf,
    out: PathBuf,
    family: DetrFamily,
    input_size: u32,
    nvframe: bool,
    sensor: String,
    epoch: Option<f64>,
    cpu: bool,
    threads: Option<usize>,
    config: PipelineConfig,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut model, mut video, mut out) = (None, None, None);
    let mut family = DetrFamily::RfDetr;
    let mut input_size = None;
    let (mut nvframe, mut cpu) = (false, false);
    let mut sensor = "camera".to_string();
    let (mut epoch, mut threads) = (None, None);
    let mut config = PipelineConfig::default();
    let mut classes_set = false;
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{a} needs a value"));
        let num = |v: String| v.parse::<f64>().map_err(|_| format!("{v} is not a number"));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(val()?)),
            "--video" => video = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--family" => {
                family = match val()?.as_str() {
                    "rf-detr" => DetrFamily::RfDetr,
                    "d-fine" => {
                        if !classes_set {
                            config.classes = pipeline::coco80_people_and_vehicles();
                        }
                        DetrFamily::DFine
                    }
                    other => return Err(format!("unknown family {other}")),
                }
            }
            "--classes" => {
                config.classes = serde_json::from_str(&val()?).map_err(|e| format!("--classes: {e}"))?;
                classes_set = true;
            }
            "--input-size" => input_size = Some(num(val()?)? as u32),
            "--nvframe" => nvframe = true,
            "--sensor" => sensor = val()?,
            "--epoch" => epoch = Some(num(val()?)?),
            "--ground" => {
                let v: Vec<f64> = val()?.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| "--ground takes nine numbers".to_string())?;
                let h: [f64; 9] = v.try_into().map_err(|_| "--ground takes nine numbers".to_string())?;
                config.ground = Some(h);
            }
            "--max-frames" => config.max_frames = Some(num(val()?)? as u32),
            "--high" => config.tracker.high_threshold = num(val()?)? as f32,
            "--low" => config.tracker.low_threshold = num(val()?)? as f32,
            "--new-track" => config.tracker.new_track_threshold = num(val()?)? as f32,
            "--match" => config.tracker.match_threshold = num(val()?)? as f32,
            "--buffer" => config.tracker.track_buffer = num(val()?)? as u32,
            "--threads" => threads = Some(num(val()?)? as usize),
            "--device" => {
                cpu = match val()?.as_str() {
                    "auto" => false,
                    "cpu" => true,
                    other => return Err(format!("unknown device {other}")),
                }
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let t = &config.tracker;
    if !(0.0..=1.0).contains(&t.low_threshold) || t.low_threshold > t.high_threshold || t.high_threshold > t.new_track_threshold {
        return Err("thresholds must satisfy 0 <= low <= high <= new-track <= 1".into());
    }
    Ok(Args {
        model: model.ok_or(USAGE)?,
        video: video.ok_or(USAGE)?,
        out: out.ok_or(USAGE)?,
        family,
        input_size: input_size.unwrap_or(match family {
            DetrFamily::RfDetr => 576,
            DetrFamily::DFine => 640,
        }),
        nvframe,
        sensor,
        epoch,
        cpu,
        threads,
        config,
    })
}

/// Peak resident memory of this process in MiB.
fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some((kb / 1024.0 * 10.0).round() / 10.0)
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    let gpus = if a.cpu {
        Gpus::default()
    } else {
        Gpus {
            nvidia: std::path::Path::new("/dev/nvidiactl").exists() || std::path::Path::new("/dev/nvidia0").exists(),
            amd: std::path::Path::new("/dev/kfd").exists(),
        }
    };
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    session::install(OnnxSettings::resolve(&OnnxConfig { threads: a.threads, ..OnnxConfig::default() }, cores), gpus);
    session::verify_accelerators().map_err(|e| e.to_string())?;
    let providers: Vec<&str> = session::accelerators().map_err(|e| e.to_string())?.iter().map(|p| p.name()).collect();
    let num_classes = match a.family {
        DetrFamily::RfDetr => 91,
        DetrFamily::DFine => 80,
    };
    let loaded = std::time::Instant::now();
    let detector = GenericDetrDetector::from_onnx(&a.model, a.family, a.input_size, num_classes).map_err(|e| e.to_string())?;
    let load_s = loaded.elapsed().as_secs_f64();
    let video = std::fs::read(&a.video).map_err(|e| format!("{}: {e}", a.video.display()))?;
    std::fs::create_dir_all(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let epoch = a.epoch.unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()));
    let mut frames = Vec::new();
    let mut nv = Vec::new();
    let summary = pipeline::track_video(&detector, &video, &a.config, |f, total| {
        if a.nvframe {
            f.to_nv_frame(&a.sensor, epoch).write_delimited(&mut nv);
        }
        if f.index % 25 == 0 {
            let p = json!({"progress": f64::from(f.index + 1) / f64::from(total.max(1)), "message": format!("frame {} of {total}", f.index + 1)});
            eprintln!("{p}");
        }
        frames.push(f.clone());
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    let mut s = serde_json::to_value(&summary).map_err(|e| e.to_string())?;
    s["model"] = json!(a.model.file_name().map(|n| n.to_string_lossy().into_owned()));
    s["providers"] = json!(providers);
    s["model_load_s"] = json!((load_s * 100.0).round() / 100.0);
    s["peak_rss_mib"] = json!(peak_rss_mib());
    s["ground_plane"] = json!(a.config.ground.is_some());
    s["classes"] = json!(a.config.classes);
    s["tracker"] = json!(a.config.tracker);
    let doc = json!({"video": a.video.file_name().map(|n| n.to_string_lossy().into_owned()), "sensor": a.sensor, "epoch_s": epoch, "frames": frames, "summary": s});
    std::fs::write(a.out.join("tracks.json"), serde_json::to_vec(&doc).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    std::fs::write(a.out.join("summary.json"), serde_json::to_vec_pretty(&s).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if a.nvframe {
        std::fs::write(a.out.join("frames.nvpb"), &nv).map_err(|e| e.to_string())?;
    }
    Ok(s)
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("praecise-track: {e}");
            ExitCode::FAILURE
        }
    }
}
