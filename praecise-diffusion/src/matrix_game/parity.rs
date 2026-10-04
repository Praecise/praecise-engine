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
