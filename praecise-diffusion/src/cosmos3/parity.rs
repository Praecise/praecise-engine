//! Agreement of the joint transformer with the reference implementation on
//! small random checkpoints, one per layout (squared-ReLU with a separate
//! generation key norm; gated SiLU with text query/key norms), each with a
//! video segment and an action segment, from
//! `tests/parity/make_cosmos3_mot_fixtures.py`.
//!
//! Run with `PRAECISE_COSMOS3_MOT_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored cosmos3_mot_parity`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::ggml::{Backend, HostTensor};
use crate::safetensors::SafeTensors;

fn root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_COSMOS3_MOT_PARITY").expect("PRAECISE_COSMOS3_MOT_PARITY names the fixture dir"))
}

fn bin(dir: &Path, name: &str) -> Vec<f32> {
    std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut ab, mut aa, mut bb, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
    }
    (ab / (aa.sqrt() * bb.sqrt()), (dd / bb).sqrt())
}

fn run(name: &str, wtype: WType, exact: bool) -> ((f64, f64), (f64, f64)) {
    let dir = root().join(name);
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let cfg: Cosmos3Config = serde_json::from_value(m["config"].clone()).unwrap();
    cfg.validate().unwrap();
    let u = |k: &str| m[k].as_i64().unwrap();
    let (text, n, cond, na, acond) = (u("text"), u("vision"), u("vision_cond"), u("actions"), u("action_cond"));
    let backend = Backend::select(std::thread::available_parallelism().map_or(8, usize::from)).unwrap();
    let files = SafeTensors::open(&[dir.join("model.safetensors")]).unwrap();
    let w = Weights::load(&backend, &files, &cfg.weight_specs(wtype)).unwrap();

    let ids: Vec<i32> = m["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
    let mut g = Graph::new(&backend).unwrap();
    let io = build_text(&mut g, &cfg, &w, text, exact);
    let outs: Vec<_> = io.keys.iter().chain(&io.values).copied().collect();
    g.finish(&outs).unwrap();
    let pos: Vec<[f32; 3]> = (0..text).map(|i| [i as f32; 3]).collect();
    let (cos, sin) = cfg.rotary_tables(&pos);
    g.set_i32(io.ids, &ids);
    g.set_f32(io.cos, &cos);
    g.set_f32(io.sin, &sin);
    g.set_f16(io.mask, &causal_mask(text as usize));
    g.compute().unwrap();
    let shape = vec![cfg.num_key_value_heads, text as u64, cfg.head_dim];
    let mut host = Vec::new();
    for (i, (k, v)) in io.keys.iter().zip(&io.values).enumerate() {
        host.push(HostTensor { name: format!("k{i}"), shape: shape.clone(), ty: WType::F32, data: g.read_f32(*k) });
        host.push(HostTensor { name: format!("v{i}"), shape: shape.clone(), ty: WType::F32, data: g.read_f32(*v) });
    }
    let cache = Weights::from_host(&backend, &host).unwrap();

    let mut g = Graph::new(&backend).unwrap();
    let io = build_gen(&mut g, &cfg, &w, &cache, n, cond, Some(ActionSpan { tokens: na, cond: acond }), exact);
    let a = io.actions.clone().unwrap();
    g.finish(&[io.out, a.out]).unwrap();
    let gp = bin(&dir, "gen_positions");
    let pos: Vec<[f32; 3]> = gp.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let (cos, sin) = cfg.rotary_tables(&pos);
    g.set_f32(io.patches, &bin(&dir, "patches"));
    g.set_f32(io.time, &cfg.time_features(u("t_vision")));
    g.set_f32(io.cos, &cos);
    g.set_f32(io.sin, &sin);
    g.set_f32(a.values, &bin(&dir, "actions"));
    g.set_f32(a.time, &cfg.time_features(u("t_action")));
    g.set_i32(a.domain, &[u("domain") as i32]);
    g.compute().unwrap();
    let v = cosine(&g.read_f32(io.out), &bin(&dir, "out_vision"));
    let a = cosine(&g.read_f32(a.out), &bin(&dir, "out_action"));
    eprintln!("{name} {wtype:?}: video cos {:.6} rel {:.2e}, action cos {:.6} rel {:.2e}", v.0, v.1, a.0, a.1);
    (v, a)
}

#[test]
#[ignore = "needs PRAECISE_COSMOS3_MOT_PARITY fixtures"]
fn cosmos3_mot_parity() {
    for name in ["relu2", "silu"] {
        let (v, a) = run(name, WType::F32, true);
        assert!(v.0 > 0.99999 && a.0 > 0.99999, "{name} f32");
        let (v, a) = run(name, WType::Bf16, false);
        assert!(v.0 > 0.999 && a.0 > 0.999, "{name} bf16");
    }
}
