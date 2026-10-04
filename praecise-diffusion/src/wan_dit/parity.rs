//! Parity against reference fixtures from `tests/parity/make_wan_dit_fixtures.py`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::ggml::Backend;

fn root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_WAN_DIT_PARITY").expect("PRAECISE_WAN_DIT_PARITY names the fixture dir"))
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

fn run(per_token: bool, exact: bool) -> f64 {
    let dir = root();
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let cfg: WanDitConfig = serde_json::from_value(m["config"].clone()).unwrap();
    cfg.validate().unwrap();
    let u = |k: &str| m[k].as_u64().unwrap() as usize;
    let (frames, h, w, text) = (u("frames"), u("height"), u("width"), u("text"));
    let t = m["t"].as_f64().unwrap() as f32;
    let backend = Backend::select(std::thread::available_parallelism().map_or(8, usize::from)).unwrap();
    let files = SafeTensors::open(&[dir.join("model.safetensors")]).unwrap();
    let wts = Weights::load(&backend, &files, &cfg.weight_specs(WType::F32)).unwrap();
    let pe = Weights::from_host(&backend, &cfg.patch_weights(&files).unwrap()).unwrap();
    let (rows, cols) = (h / 2, w / 2);
    let n = frames * rows * cols;
    let tt = if per_token { n } else { 1 };
    let mut g = Graph::new(&backend).unwrap();
    let io = build(&mut g, &cfg, &wts, &pe, frames as i64, n as i64, text as i64, tt as i64, exact);
    g.finish(&[io.out]).unwrap();
    let (cos, sin) = cfg.rotary_tables(frames, rows, cols);
    let time: Vec<f32> = if per_token {
        (0..n).flat_map(|i| time_features(if i < rows * cols { 0.0 } else { t })).collect()
    } else {
        time_features(t)
    };
    g.set_f32(io.patches, &patchify(&cfg, &bin(&dir, "latent"), frames, h, w));
    g.set_f32(io.time, &time);
    g.set_f32(io.context, &bin(&dir, "context"));
    g.set_f32(io.cos, &cos);
    g.set_f32(io.sin, &sin);
    g.compute().unwrap();
    let got = unpatchify(&cfg, &g.read_f32(io.out), frames, h, w);
    let want = bin(&dir, if per_token { "out_per_token" } else { "out_shared" });
    let c = cosine(&got, &want);
    eprintln!("wan dit per_token={per_token} exact={exact}: cos {c:.7}");
    c
}

#[test]
#[ignore = "needs PRAECISE_WAN_DIT_PARITY fixtures"]
fn wan_dit_parity() {
    for per_token in [false, true] {
        assert!(run(per_token, true) > 0.99999);
        assert!(run(per_token, false) > 0.999);
    }
}
