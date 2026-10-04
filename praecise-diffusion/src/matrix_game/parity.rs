//! Parity against reference fixtures from `tests/parity/make_mg3_fixtures.py`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::ggml::Backend;
use crate::safetensors::SafeTensors;
use crate::wan_dit::{build, patchify, time_features, unpatchify};

fn root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_MG3_PARITY").expect("PRAECISE_MG3_PARITY names the fixture dir"))
}

fn bin(dir: &Path, name: &str) -> Vec<f32> {
    std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
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

/// `[c][a][plane]` and `[c][b][plane]` joined along time.
fn join(a: &[f32], b: &[f32], c: usize) -> Vec<f32> {
    let (fa, fb) = (a.len() / c, b.len() / c);
    (0..c).flat_map(|i| a[i * fa..(i + 1) * fa].iter().chain(&b[i * fb..(i + 1) * fb]).copied()).collect()
}

fn run(memory: bool, exact: bool) -> f64 {
    let dir = root();
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let cfg = config(&serde_json::to_vec(&m["config"]).unwrap()).unwrap();
    let wd = cfg.world.clone().unwrap();
    let a = &wd.action;
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let (h, w, text) = (u("height"), u("width"), u("text"));
    let t = m["t"].as_f64().unwrap() as f32;
    let backend = Backend::select(std::thread::available_parallelism().map_or(8, usize::from)).unwrap();
    let files = SafeTensors::open(&[dir.join("model.safetensors")]).unwrap().renamed(rename).unwrap();
    let wts = Weights::load(&backend, &files, &cfg.weight_specs(WType::F32)).unwrap();
    let pe = Weights::from_host(&backend, &cfg.patch_weights(&files).unwrap()).unwrap();
    let (rows, cols) = (h / 2, w / 2);
    let hw = rows * cols;
    let z = cfg.in_channels as usize;
    let (frames, latent, positions, act_pos, kb, mo, rays, held) = if memory {
        let fp = u("pred_frames");
        let lat = join(&bin(&dir, "mem_latent"), &bin(&dir, "pred_latent"), z);
        let start = u("pred_start");
        let pos: Vec<usize> = std::iter::once(u("memory_index")).chain(start..start + fp).collect();
        let ap: Vec<usize> = std::iter::once(0).chain(0..fp).collect();
        let kb = windows(a, &bin(&dir, "pred_keyboard"), 2, &bin(&dir, "mem_keyboard")).unwrap();
        let mo = windows(a, &bin(&dir, "pred_mouse"), 2, &bin(&dir, "mem_mouse")).unwrap();
        (1 + fp, lat, pos, ap, kb, mo, bin(&dir, "rays2"), 1)
    } else {
        let f = u("frames");
        let kb = windows(a, &bin(&dir, "keyboard"), 2, &[]).unwrap();
        let mo = windows(a, &bin(&dir, "mouse"), 2, &[]).unwrap();
        (f, bin(&dir, "latent"), (0..f).collect(), (0..f).collect(), kb, mo, bin(&dir, "rays"), 1)
    };
    let n = frames * hw;
    let mut g = Graph::new(&backend).unwrap();
    let io = build(&mut g, &cfg, &wts, &pe, frames as i64, n as i64, text as i64, n as i64, exact);
    g.finish(&[io.out]).unwrap();
    let (cos, sin) = cfg.rotary_tables_at(&positions, rows, cols);
    let time: Vec<f32> = (0..n).flat_map(|i| time_features(if i < held * hw { 0.0 } else { t })).collect();
    g.set_f32(io.patches, &patchify(&cfg, &latent, frames, h, w));
    g.set_f32(io.time, &time);
    g.set_f32(io.context, &bin(&dir, "context"));
    g.set_f32(io.cos, &cos);
    g.set_f32(io.sin, &sin);
    let aio = io.actions.unwrap();
    let (acos, asin) = action_rotary(a, &act_pos);
    g.set_f32(aio.keyboard, &kb);
    g.set_f32(aio.mouse, &mo);
    g.set_f32(aio.cos, &acos);
    g.set_f32(aio.sin, &asin);
    g.set_f32(io.camera.unwrap(), &patchify_rays(&rays, wd.camera_channels as usize, frames, h, w));
    g.compute().unwrap();
    let got = unpatchify(&cfg, &g.read_f32(io.out), frames, h, w);
    let (got, want) = if memory {
        let plane = (frames - 1) * h * w;
        let got: Vec<f32> = (0..z).flat_map(|c| got[c * frames * h * w + h * w..(c + 1) * frames * h * w].iter().copied()).collect();
        assert_eq!(got.len(), z * plane);
        (got, bin(&dir, "out_mem"))
    } else {
        (got, bin(&dir, "out"))
    };
    let c = cosine(&got, &want);
    eprintln!("world dit memory={memory} exact={exact}: cos {c:.7}");
    c
}

#[test]
#[ignore = "needs PRAECISE_MG3_PARITY fixtures"]
fn world_dit_parity() {
    for memory in [false, true] {
        assert!(run(memory, true) > 0.99999);
        assert!(run(memory, false) > 0.999);
    }
}

#[test]
fn camera_rays_match_reference() {
    use super::camera::{clip_rays, extrinsic, memory_rays, poses, select_by_view};
    let dir = root().join("../mg3cam");
    let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let u = |v: &Value| usize::try_from(v.as_u64().unwrap()).unwrap();
    let (frames, lh, lw, s) = (u(&meta["frames"]), u(&meta["lat_h"]), u(&meta["lat_w"]), u(&meta["s"]));
    let (path, _) = poses([0.0; 5], &bin(&dir, "keyboard"), 6, &bin(&dir, "mouse"), frames);
    let want = bin(&dir, "poses");
    for (p, w) in path.iter().flatten().zip(&want) {
        assert!((p - w).abs() <= 1e-4 * w.abs().max(1.0), "pose {p} vs {w}");
    }
    let c2ws: Vec<_> = path.iter().map(extrinsic).collect();
    for (key, name, first) in [("clip1", "rays1", true), ("clip2", "rays2", false)] {
        let c = &meta[key];
        let (start, end, n) = (u(&c[0]), u(&c[1]), u(&c[2]));
        let got = clip_rays(&c2ws, start, end, if first { 0 } else { start + 3 }, n, (lh, lw), s);
        let want = bin(&dir, name);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(cosine(&got, &want) > 0.999_999 && err < 1e-4, "{name}: cos {} max err {err}", cosine(&got, &want));
    }
    let m = &meta["memory"];
    let got = memory_rays(&c2ws, u(&m[0]), u(&m[1]), (lh, lw), s);
    let want = bin(&dir, "mem_rays");
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(err < 1e-4, "memory rays max err {err}");
    let f = &meta["fov"];
    let bases: Vec<usize> = f["bases"].as_array().unwrap().iter().map(u).collect();
    let got = select_by_view(&c2ws, u(&f["start"]), &bases);
    for ((i, r), (wi, wr)) in got.iter().zip(f["selected"].as_array().unwrap().iter().zip(f["confidence"].as_array().unwrap())) {
        #[allow(clippy::cast_possible_truncation)]
        let wr = wr.as_f64().unwrap() as f32;
        assert!(*i == u(wi) && (r - wr).abs() < 2e-3, "view selection {got:?} vs {f}");
    }
}

/// Session parity: the reference generation loop over three clips with a
/// tiny transformer (`tests/parity/make_mg3_session_fixtures.py`), every
/// clip's new latents compared after the full sampler, guidance on.
#[test]
#[ignore = "needs PRAECISE_MG3_SESSION fixtures"]
fn world_session_parity() {
    use super::model::{Session, Transformer};
    use crate::pipeline::Precision;
    use crate::unipc::UniPcConfig;
    use crate::world::{ChunkRequest, LatentFrame, FRAMES_PER_LATENT};

    let dir = PathBuf::from(std::env::var("PRAECISE_MG3_SESSION").expect("PRAECISE_MG3_SESSION names the fixture dir"));
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let cfg = config(&serde_json::to_vec(&m["config"]).unwrap()).unwrap();
    let sched: UniPcConfig = serde_json::from_value(serde_json::json!({
        "num_train_timesteps": 1000, "solver_order": 2, "solver_type": "bh2", "prediction_type": "flow_prediction",
        "predict_x0": true, "lower_order_final": true, "use_karras_sigmas": false, "use_flow_sigmas": true,
        "final_sigmas_type": "zero", "flow_shift": m["shift"], "use_dynamic_shifting": false
    }))
    .unwrap();
    let backend = Backend::select(std::thread::available_parallelism().map_or(8, usize::from)).unwrap();
    let files = SafeTensors::open(&[dir.join("model.safetensors")]).unwrap().renamed(rename).unwrap();
    let dit = Transformer::new(backend, cfg, &files, sched, Precision::F32).unwrap();
    let z = dit.latent_channels();
    let dims = dit.action_dims();
    let (h, w) = (u("height"), u("width"));
    let plane = h * w;
    let rows = bin(&dir, "actions");
    let mut ctx = Session::with_states(vec![bin(&dir, "context"), bin(&dir, "negative")]);
    let mut frames = vec![LatentFrame { index: 0, data: bin(&dir, "image_latent") }];
    let (mut first, mut pixel) = (1, 1);
    let mut worst = 1f64;
    for c in 0..u("clips") {
        let n = if c == 0 { u("first_frames") } else { u("next_frames") };
        let end = FRAMES_PER_LATENT * (first + n - 1) + 1;
        let actions = &rows[pixel * dims..end * dims];
        dit.advance(&mut ctx, actions, pixel).unwrap();
        let want_mem: Vec<usize> = m["memory"][c].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let picked = dit.select_memory(&ctx, first).unwrap();
        assert_eq!(picked, want_mem, "clip {c} memory");
        let memory: Vec<LatentFrame> = picked.iter().map(|&i| LatentFrame { index: i, data: frames[i].data.clone() }).collect();
        let req = ChunkRequest { memory: &memory, new_frames: n, first_index: first, actions, grid: (h, w), steps: u("steps") as u32, guidance_scale: m["guidance"].as_f64().unwrap() as f32, seed: 0 };
        let (got, _) = dit.rollout(&ctx, &req, bin(&dir, &format!("noise{c}"))).unwrap();
        let want = bin(&dir, &format!("out{c}"));
        let cos = cosine(&got, &want);
        eprintln!("world session clip {c}: cos {cos:.7}");
        let per: Vec<String> = (0..n)
            .map(|k| {
                let f = |v: &[f32]| -> Vec<f32> { (0..z).flat_map(|ch| v[(ch * n + k) * plane..(ch * n + k + 1) * plane].iter().copied()).collect() };
                format!("{:.6}", cosine(&f(&got), &f(&want)))
            })
            .collect();
        eprintln!("  per frame {per:?}");
        worst = worst.min(cos);
        // Continue from the reference latents so a clip's error does not feed the next.
        for k in 0..n {
            let data = (0..z).flat_map(|ch| want[(ch * n + k) * plane..(ch * n + k + 1) * plane].iter().copied()).collect();
            frames.push(LatentFrame { index: first + k, data });
        }
        first += n;
        pixel = end;
    }
    assert!(worst > 0.999, "session parity cos {worst}");
}
