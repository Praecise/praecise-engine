//! Agreement with the reference implementation on small random-weight
//! networks, from fixtures written by
//! `tests/parity/make_trellis2_fixtures.py`.
//!
//! Run with `PRAECISE_TRELLIS2_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored trellis2_parity`.

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::ggml::Device;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_TRELLIS2_PARITY").expect("PRAECISE_TRELLIS2_PARITY names the fixture dir"))
}

fn meta() -> Value {
    serde_json::from_slice(&std::fs::read(dir().join("meta.json")).unwrap()).unwrap()
}

fn f32s(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn i32s(name: &str) -> Vec<i32> {
    std::fs::read(dir().join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn coords(name: &str) -> Vec<[i32; 3]> {
    i32s(name).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn backend() -> Backend {
    let device = match std::env::var("PRAECISE_DEVICE").as_deref() {
        Ok(v) => Device::parse(v).unwrap(),
        Err(_) => Device::Cpu,
    };
    Backend::select_device(device, std::thread::available_parallelism().map_or(8, usize::from)).unwrap()
}

/// Cosine similarity and max error relative to the reference's largest
/// magnitude.
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
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30), maxerr / maxref.max(1e-30))
}

fn check(what: &str, ours: &[f32], reference: &[f32], tol: f64) {
    let (cos, rel) = agreement(ours, reference);
    eprintln!("{what}: cosine {cos:.8}, max relative error {rel:.3e}");
    assert!(cos > 1.0 - tol && rel < tol.sqrt(), "{what} disagrees: cosine {cos}, max relative error {rel}");
}

fn files() -> Trellis2Files {
    Trellis2Files { root: dir() }
}

fn tokens_from_volume(v: &[f32], c: usize) -> Vec<f32> {
    let n = v.len() / c;
    let mut x = vec![0f32; v.len()];
    for ch in 0..c {
        for t in 0..n {
            x[t * c + ch] = v[ch * n + t];
        }
    }
    x
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_flow_dense() {
    let be = backend();
    let m = load_flow(&be, &files(), "ss_flow", WType::F32, true).unwrap();
    let grid = Voxels::dense(4);
    let x = tokens_from_volume(&f32s("ss_x"), 4);
    let y = m.forward(&be, &x, &grid.coords, 637.5, &f32s("ss_cond")).unwrap();
    check("dense flow", &y, &tokens_from_volume(&f32s("ss_out"), 4), 1e-6);
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_sampler_dense() {
    let be = backend();
    let m = load_flow(&be, &files(), "ss_flow", WType::F32, true).unwrap();
    let grid = Voxels::dense(4);
    let p: SamplerParams = serde_json::from_value(meta()["sampler"].clone()).unwrap();
    let cond = f32s("ss_cond");
    let neg = vec![0f32; cond.len()];
    let x = tokens_from_volume(&f32s("ss_noise"), 4);
    let out = sample(&p, 1e-5, true, &mut |x, t, c| m.forward(&be, x, &grid.coords, t, c), x, &cond, &neg).unwrap();
    check("dense guided sampling", &out, &tokens_from_volume(&f32s("ss_sample"), 4), 1e-5);
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_flow_sparse_and_sampler() {
    let be = backend();
    let c = coords("slat_coords");
    let m = load_flow(&be, &files(), "slat_flow", WType::F32, true).unwrap();
    let y = m.forward(&be, &f32s("slat_x"), &c, 250.0, &f32s("ss_cond")).unwrap();
    check("sparse flow", &y, &f32s("slat_out"), 1e-6);
    let m4 = load_flow(&be, &files(), "slat_flow4", WType::F32, true).unwrap();
    let p: SamplerParams = serde_json::from_value(meta()["sparse_sampler"].clone()).unwrap();
    let cond = f32s("ss_cond");
    let neg = vec![0f32; cond.len()];
    let out = sample(&p, 1e-5, false, &mut |x, t, cc| m4.forward(&be, x, &c, t, cc), f32s("slat_noise"), &cond, &neg).unwrap();
    check("sparse guided sampling", &out, &f32s("slat_sample"), 1e-5);
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_occupancy() {
    let be = backend();
    let j: Value = serde_json::from_slice(&std::fs::read(dir().join("occupancy.json")).unwrap()).unwrap();
    let cfg: OccupancyConfig = serde_json::from_value(j["args"].clone()).unwrap();
    let st = SafeTensors::open(&[dir().join("occupancy.safetensors")]).unwrap();
    let d = OccupancyDecoder::load(&be, cfg, &st).unwrap();
    let y = d.decode(&be, &f32s("occ_latent"), 4).unwrap();
    // Kernels are held in half precision; the reference runs float32.
    check("occupancy logits", &y, &f32s("occ_logits"), 1e-4);
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_decoders_and_surface() {
    let be = backend();
    let shape = load_decoder(&be, &files(), "shape_dec", "FlexiDualGridVaeDecoder").unwrap();
    let tex = load_decoder(&be, &files(), "tex_dec", "SparseUnetVaeDecoder").unwrap();
    let vox = Voxels::new(coords("dec_coords"), 8);
    let s = shape.decode(&be, &vox, &f32s("dec_shape_latent"), None).unwrap();
    for (i, m) in s.subdivisions.iter().enumerate() {
        let r: Vec<bool> = i32s(&format!("dec_sub{i}")).iter().map(|&v| v != 0).collect();
        let differ = m.iter().zip(&r).filter(|(a, b)| a != b).count();
        eprintln!("subdivision {i}: {differ} of {} flags differ", r.len());
        assert_eq!(differ, 0, "subdivision {i}");
    }
    assert_eq!(s.voxels.coords, coords("dec_out_coords"), "output voxels");
    check("shape decoder", &s.feats, &f32s("dec_shape_out"), 1e-4);
    let t = tex.decode(&be, &vox, &f32s("dec_tex_latent"), Some(&s.subdivisions)).unwrap();
    check("material decoder", &t.feats, &f32s("dec_tex_out"), 1e-4);

    // Surface from the reference's own decoder outputs, so only extraction
    // is compared.
    let f = f32s("dec_shape_out");
    let n = f.len() / 7;
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let off: Vec<[f32; 3]> = (0..n).map(|i| [2.0 * sig(f[i * 7]) - 0.5, 2.0 * sig(f[i * 7 + 1]) - 0.5, 2.0 * sig(f[i * 7 + 2]) - 0.5]).collect();
    let cr: Vec<[bool; 3]> = (0..n).map(|i| [f[i * 7 + 3] > 0.0, f[i * 7 + 4] > 0.0, f[i * 7 + 5] > 0.0]).collect();
    let sp: Vec<f32> = (0..n).map(|i| f[i * 7 + 6].exp().ln_1p()).collect();
    let mesh = mesh3d::dual_grid_mesh(&coords("dec_out_coords"), &off, &cr, &sp, GridFrame { lo: [-0.5; 3], size: 1.0, grid: 32 });
    let rv: Vec<f32> = f32s("mesh_vertices");
    let rf: Vec<u32> = i32s("mesh_faces").iter().map(|&v| v as u32).collect();
    let ov: Vec<f32> = mesh.vertices.iter().flatten().copied().collect();
    check("surface vertices", &ov, &rv, 1e-10);
    let of: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
    assert_eq!(of, rf, "surface faces");
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_image_encoder() {
    let be = backend();
    let cfg: Dinov3Config = serde_json::from_slice(&std::fs::read(dir().join("dinov3/config.json")).unwrap()).unwrap();
    cfg.validate().unwrap();
    let st = SafeTensors::open(&[dir().join("dinov3/model.safetensors")]).unwrap();
    let w = Weights::load(&be, &st, &cfg.weight_specs(WType::F32)).unwrap();
    let enc = Dinov3::new(cfg, w, true);
    let y = enc.encode(&be, &f32s("dino_pixels"), 32, 32).unwrap();
    check("image encoder", &y, &f32s("dino_out"), 1e-6);
}

#[test]
#[ignore = "needs fixtures"]
fn trellis2_parity_image_preparation() {
    let m = meta();
    let (w, h) = (m["image"]["width"].as_u64().unwrap() as u32, m["image"]["height"].as_u64().unwrap() as u32);
    let data = std::fs::read(dir().join("image_rgba.bin")).unwrap();
    let rgb = image::crop_to_object(&image::RgbaImage { width: w, height: h, data }).unwrap();
    let crop: Vec<u8> = i32s("image_crop").iter().map(|&v| v as u8).collect();
    assert_eq!((rgb.width, rgb.height), (m["image"]["crop"][0].as_u64().unwrap() as u32, m["image"]["crop"][1].as_u64().unwrap() as u32));
    assert_eq!(rgb.rgb, crop, "cropped pixels");
    let x = image::encoder_input(&rgb, ENCODER_SIDE);
    check("encoder input", &x, &f32s("image_encoder_input"), 1e-12);
}
