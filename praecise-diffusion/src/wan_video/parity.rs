//! Parity against a small random reference checkpoint
//! (tests/parity/make_wan_pipeline_fixtures.py): prompt tokens, final latents
//! and decoded frames of text-to-video and first-frame-conditioned runs from
//! the reference's starting noise. Run with
//! `PRAECISE_WAN_PIPELINE_PARITY=<dir> cargo test wan_pipeline_parity -- --ignored`.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;

use super::*;
use crate::pipeline::RgbImage;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_WAN_PIPELINE_PARITY").expect("PRAECISE_WAN_PIPELINE_PARITY names the fixture dir"))
}

fn bin(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

#[test]
#[ignore = "needs PRAECISE_WAN_PIPELINE_PARITY fixtures"]
fn wan_pipeline_parity() {
    let m: Value = serde_json::from_slice(&std::fs::read(dir().join("meta.json")).unwrap()).unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as u32;
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let opts = LoadOptions { precision: Precision::F32, cpu_threads: threads, device: None };
    let mut wan = Wan22::load(&CheckpointFiles::new(dir().join("checkpoint")), opts).unwrap();
    let want_ids: Vec<i32> = m["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
    assert_eq!(wan.tokens(m["prompt"].as_str().unwrap()).unwrap(), want_ids);
    let ctx = wan.context(&want_ids).unwrap();
    let c_ctx = cos(&ctx, &bin("prompt_states"));
    println!("prompt states cosine {c_ctx:.7}");
    let (w, h) = (u("width"), u("height"));
    let (lt, lh, lw) = (1 + (u("frames") as usize - 1) / 4, h as usize / 16, w as usize / 16);
    let (rows, cols) = (lh / 2, lw / 2);
    let n = lt * rows * cols;
    let mut g = Graph::new(&wan.backend).unwrap();
    let io = wan_dit::build(&mut g, &wan.cfg, &wan.tf, &wan.pe, lt as i64, n as i64, TEXT_TOKENS as i64, 1, true);
    g.finish(&[io.out]).unwrap();
    let (cs, sn) = wan.cfg.rotary_tables(lt, rows, cols);
    let (sigmas, steps) = flow_sigmas_schedule(u("steps") as usize, wan.sched.flow_shift, wan.sched.num_train_timesteps);
    println!("first timestep {} sigma {}", steps[0], sigmas[0]);
    g.set_f32(io.patches, &wan_dit::patchify(&wan.cfg, &bin("t2v_noise"), lt, lh, lw));
    g.set_f32(io.time, &wan_dit::time_features(steps[0] as f32));
    g.set_f32(io.context, &ctx);
    g.set_f32(io.cos, &cs);
    g.set_f32(io.sin, &sn);
    g.compute().unwrap();
    let v = wan_dit::unpatchify(&wan.cfg, &g.read_f32(io.out), lt, lh, lw);
    println!("first velocity cosine {:.7}", cos(&v, &bin("t2v_first_velocity")));
    let first = RgbImage { width: w, height: h, rgb: std::fs::read(dir().join("first.rgb")).unwrap() };
    for (name, image) in [("t2v", None), ("i2v", Some(first))] {
        let req = VideoRequest {
            prompt: m["prompt"].as_str().unwrap().into(),
            negative_prompt: Some(m["negative"].as_str().unwrap().into()),
            image,
            width: w,
            height: h,
            num_frames: u("frames"),
            fps: m["fps"].as_f64().unwrap() as f32,
            steps: u("steps"),
            guidance_scale: m["guidance"].as_f64().unwrap() as f32,
            seed: 0,
        };
        let mut trace = Vec::new();
        let (lat, evals, _, _) = wan.sample_traced(&req, bin(&format!("{name}_noise")), Instant::now(), &mut trace).unwrap();
        // Per step, only text-to-video compares exactly: with a first frame the
        // reference keeps noise in frame 0 of its sampler state and swaps the
        // condition in at the end, where the native sampler holds it fixed.
        for (i, pair) in trace.chunks_exact(2).enumerate() {
            let cv = cos(&pair[0], &bin(&format!("{name}_velocity{i}")));
            let cl = cos(&pair[1], &bin(&format!("{name}_step{i}")));
            println!("{name} step {i}: velocity cosine {cv:.7}, latents cosine {cl:.7}");
        }
        let c_lat = cos(&lat, &bin(&format!("{name}_latents")));
        let px = wan::decode(&wan.backend, &wan.vae_cfg, &wan.vae, &lat, grid(&req)).unwrap();
        let c_px = cos(&px, &bin(&format!("{name}_frames")));
        println!("{name}: latents cosine {c_lat:.7}, frames cosine {c_px:.7}, evaluations {evals}");
        assert!(c_lat > 0.9999 && c_px > 0.9999, "{name}: latents {c_lat} frames {c_px}");
        let video = wan.generate_from(&req, bin(&format!("{name}_noise"))).unwrap();
        assert_eq!(video.frames, req.num_frames);
    }
}

/// A session opened on the first frame, with one chunk filling the rest of
/// the clip, streams exactly the frames a single first-frame-conditioned
/// generation decodes from the same seed; later chunks continue from the
/// bounded memory.
#[test]
#[ignore = "needs PRAECISE_WAN_PIPELINE_PARITY fixtures"]
fn wan_session_matches_generation() {
    use crate::world::{SessionConfig, WorldSession};
    let m: Value = serde_json::from_slice(&std::fs::read(dir().join("meta.json")).unwrap()).unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as u32;
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let opts = LoadOptions { precision: Precision::F32, cpu_threads: threads, device: None };
    let mut wan = Wan22::load(&CheckpointFiles::new(dir().join("checkpoint")), opts).unwrap();
    let (w, h) = (u("width"), u("height"));
    let first = RgbImage { width: w, height: h, rgb: std::fs::read(dir().join("first.rgb")).unwrap() };
    let req = VideoRequest {
        prompt: m["prompt"].as_str().unwrap().into(),
        negative_prompt: Some(m["negative"].as_str().unwrap().into()),
        image: Some(first.clone()),
        width: w,
        height: h,
        num_frames: u("frames"),
        fps: m["fps"].as_f64().unwrap() as f32,
        steps: u("steps"),
        guidance_scale: m["guidance"].as_f64().unwrap() as f32,
        seed: 3,
    };
    let video = wan.generate(&req).unwrap();
    let lt = 1 + (req.num_frames as usize - 1) / 4;
    let cfg = SessionConfig {
        width: w,
        height: h,
        fps: req.fps,
        chunk_latent_frames: lt - 1,
        memory_latent_frames: 2,
        history_latent_frames: 2,
        steps: req.steps,
        guidance_scale: req.guidance_scale,
        seed: 3,
    };
    let mut s = WorldSession::start(&wan, cfg, &req.prompt, req.negative_prompt.as_deref().unwrap(), Some(&first)).unwrap();
    let a = s.step(&[]).unwrap();
    assert_eq!(a.frames, video.frames);
    // The generation's frame 0 goes through the sampler with zero velocity,
    // which holds it up to rounding; the session keeps the encoded frame
    // exactly. The two may differ by one level in a few bytes.
    let diff = a.rgb.iter().zip(&video.rgb).filter(|(x, y)| x != y).count();
    let worst = a.rgb.iter().zip(&video.rgb).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    println!("session chunk 0: {} frames, {diff} of {} bytes differ from the generation, by at most {worst}", a.frames, a.rgb.len());
    assert!(worst <= 1 && diff * 1000 < a.rgb.len(), "{diff} bytes differ, worst {worst}");
    for k in 1..3 {
        let c = s.step(&[]).unwrap();
        assert_eq!(c.frames as usize, 4 * (lt - 1));
        assert!(s.memory().count() <= 2);
        println!("session chunk {k}: first frame {}, {} frames, {} evaluations", c.first_frame, c.frames, c.evaluations);
    }
    assert!((s.simulated_secs() - f64::from(video.frames + 8 * (lt as u32 - 1)) / f64::from(req.fps)).abs() < 1e-9);
}
