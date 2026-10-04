//! Agreement of action-conditioned generation with the reference pipeline,
//! from `tests/parity/make_cosmos3_action_fixtures.py`: a policy run and a
//! forward-dynamics run, each started from the noise the reference drew.
//!
//! Run with `PRAECISE_COSMOS3_ACTION_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored cosmos3_action_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_COSMOS3_ACTION_PARITY").expect("PRAECISE_COSMOS3_ACTION_PARITY names the fixture dir"))
}

fn bin(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn agreement(ours: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(ours.len(), reference.len(), "length");
    let (mut dot, mut na, mut nb, mut maxerr, mut maxref) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (a, b) in ours.iter().zip(reference) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        dot += a * b;
        na += a * a;
        nb += b * b;
        maxerr = maxerr.max((a - b).abs());
        maxref = maxref.max(b.abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr / maxref)
}

fn run(p: &mut Cosmos3, m: &Value, mode: ActionMode, name: &str) -> (f64, f64, Option<(f64, f64)>) {
    let u = |k: &str| m[k].as_u64().unwrap() as u32;
    let (width, height, chunk) = (u("width"), u("height"), u("chunk"));
    let aw = u("action_width") as usize;
    let given: Vec<Vec<f32>> = bin("given_actions").chunks_exact(aw).map(<[f32]>::to_vec).collect();
    let req = ActionRequest {
        mode,
        embodiment: m["embodiment"].as_str().unwrap().into(),
        prompt: m["prompt"].as_str().unwrap().into(),
        negative_prompt: Some(m["negative"].as_str().unwrap().into()),
        view_point: m["view_point"].as_str().unwrap().into(),
        resolution_tier: u("tier"),
        chunk_size: chunk,
        frames: if mode == ActionMode::InverseDynamics {
            let clip = std::fs::read(dir().join("video.bin")).unwrap();
            clip.chunks_exact((width * height * 3) as usize).map(|f| RgbImage { width, height, rgb: f.to_vec() }).collect()
        } else {
            vec![RgbImage { width, height, rgb: std::fs::read(dir().join("image.bin")).unwrap() }]
        },
        actions: (mode == ActionMode::ForwardDynamics).then_some(given),
        fps: m["fps"].as_f64().unwrap() as f32,
        steps: u("steps"),
        guidance_scale: m[format!("{name}_guidance")].as_f64().unwrap() as f32,
        seed: 0,
    };
    let caption = action_caption(&req.prompt, &req.view_point, chunk + 1, req.fps, height, width);
    let ids: Vec<Vec<i64>> = m[format!("{name}_ids")].as_array().unwrap().iter()
        .map(|v| v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect())
        .collect();
    let ours: Vec<i64> = p.tokens(&caption, false).unwrap().iter().map(|&t| i64::from(t)).collect();
    assert_eq!(ours, ids[0], "{name}: caption tokens");
    if ids.len() > 1 {
        let ours: Vec<i64> = p.tokens(req.negative_prompt.as_deref().unwrap(), false).unwrap().iter().map(|&t| i64::from(t)).collect();
        assert_eq!(ours, ids[1], "{name}: negative tokens");
    }
    let (lt, lh, lw) = grid_of(chunk + 1, height, width);
    let text = ids[0].len();
    let mut pos: Vec<[f32; 3]> = (0..text).map(|i| [i as f32; 3]).collect();
    pos.extend(p.video_positions(text, lt, lh / 2, lw / 2, req.fps));
    pos.extend(p.action_positions(text, chunk as usize, req.fps));
    let reference = bin(&format!("{name}_positions"));
    let n = pos.len();
    assert_eq!(reference.len(), 3 * n, "{name}: position count");
    for (i, q) in pos.iter().enumerate() {
        for a in 0..3 {
            assert!((q[a] - reference[a * n + i]).abs() < 1e-3, "{name}: position {i} axis {a}: {} vs {}", q[a], reference[a * n + i]);
        }
    }
    let emb = Embodiment::named(&req.embodiment).unwrap();
    let model_width = p.cfg.action_dim.unwrap() as usize;
    let out = p
        .generate_action_from(&req, emb, model_width, bin(&format!("{name}_vision_noise")), bin(&format!("{name}_action_noise")))
        .unwrap();
    // Reference frames are [F][3][H][W] in [-1, 1]; ours are interleaved bytes.
    let (w, h) = (width as usize, height as usize);
    let reference = bin(&format!("{name}_frames"));
    let frames = reference.len() / (3 * w * h);
    let mut ours = vec![0f32; reference.len()];
    for f in 0..frames {
        for c in 0..3 {
            for i in 0..w * h {
                ours[(f * 3 + c) * w * h + i] = f32::from(out.video.rgb[(f * w * h + i) * 3 + c]) / 127.5 - 1.0;
            }
        }
    }
    let quantised: Vec<f32> = reference.iter().map(|v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() / 127.5 - 1.0).collect();
    let (fc, fr) = agreement(&ours, &quantised);
    let acts = out.actions.map(|a| {
        let flat: Vec<f32> = a.concat();
        agreement(&flat, &bin(&format!("{name}_actions")))
    });
    eprintln!("{name}: frames cosine {fc:.6} max rel {fr:.4}; actions {acts:?}");
    (fc, fr, acts)
}

#[test]
#[ignore = "needs PRAECISE_COSMOS3_ACTION_PARITY fixtures"]
fn cosmos3_action_parity() {
    let m: Value = serde_json::from_slice(&std::fs::read(dir().join("meta.json")).unwrap()).unwrap();
    let precision = match std::env::var("PRAECISE_COSMOS3_PRECISION").as_deref() {
        Ok("bf16") => Precision::Bf16,
        _ => Precision::F32,
    };
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let mut p = Cosmos3::load(&CheckpointFiles::new(dir().join("checkpoint")), LoadOptions { precision, cpu_threads: threads, device: None })
        .unwrap();
    let (min_cos, max_rel) = if matches!(precision, Precision::F32) { (0.9999, 0.05) } else { (0.99, 0.5) };
    let (fc, fr, acts) = run(&mut p, &m, ActionMode::Policy, "policy");
    let (ac, _) = acts.expect("policy predicts actions");
    assert!(fc >= min_cos && fr <= max_rel && ac >= min_cos, "policy");
    let (fc, fr, acts) = run(&mut p, &m, ActionMode::ForwardDynamics, "forward_dynamics");
    assert!(acts.is_none(), "forward dynamics returns no actions");
    assert!(fc >= min_cos && fr <= max_rel, "forward dynamics");
    let (fc, fr, acts) = run(&mut p, &m, ActionMode::InverseDynamics, "inverse_dynamics");
    let (ac, _) = acts.expect("inverse dynamics predicts actions");
    assert!(fc >= min_cos && fr <= max_rel && ac >= min_cos, "inverse dynamics");
}
