//! Parity against reference fixtures from `tests/parity/make_umt5_fixtures.py`.

use serde_json::Value;

use super::*;
use crate::ggml::Backend;
use crate::safetensors::SafeTensors;

#[test]
#[ignore = "needs PRAECISE_UMT5_PARITY fixtures"]
fn umt5_parity() {
    let dir = std::path::PathBuf::from(std::env::var("PRAECISE_UMT5_PARITY").expect("PRAECISE_UMT5_PARITY names the fixture dir"));
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let cfg: Umt5Config = serde_json::from_value(m["config"].clone()).unwrap();
    cfg.validate().unwrap();
    let ids: Vec<i32> = m["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
    let n = ids.len();
    let backend = Backend::select(std::thread::available_parallelism().map_or(8, usize::from)).unwrap();
    let files = SafeTensors::open(&[dir.join("model.safetensors")]).unwrap();
    let w = Weights::load(&backend, &files, &cfg.weight_specs(WType::F32)).unwrap();
    let mut g = Graph::new(&backend).unwrap();
    let io = build(&mut g, &cfg, &w, n as i64);
    g.finish(&[io.out]).unwrap();
    g.set_i32(io.ids, &ids);
    g.set_i32(io.buckets, &cfg.buckets(n));
    g.compute().unwrap();
    let got = g.read_f32(io.out);
    let want: Vec<f32> = std::fs::read(dir.join("out.bin")).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in got.iter().zip(&want) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    let c = ab / (aa.sqrt() * bb.sqrt());
    eprintln!("umt5: cos {c:.7}");
    assert_eq!(got.len(), want.len());
    assert!(c > 0.99999);
}
