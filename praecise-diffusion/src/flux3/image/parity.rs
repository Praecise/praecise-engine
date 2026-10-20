//! Agreement of the FLUX 3 image pipeline with the reference implementation.
//!
//! Tiny random checkpoints (`tests/parity/make_flux3_image_fixtures.py`):
//! one transformer evaluation with text, the image and two references, from
//! the package layout and from the diffusers layout, and a short sampling run
//! (flow map, and guided), all in float32. Run with
//! `PRAECISE_DEVICE=cpu PRAECISE_FLUX3_IMAGE_PARITY=<fixture dir> cargo test
//! -p praecise-diffusion -- --ignored flux3_image_parity`.
//!
//! The released checkpoints (`tests/parity/compare_flux3_image_reference.py`
//! writes the reference's text context, starting noise, latent after every
//! step and decoded image for fixed prompts and seeds): run with
//! `PRAECISE_FLUX3_IMAGE_REFERENCE=<fixture dir> cargo test --release
//! --features cuda -p praecise-diffusion -- --ignored --nocapture
//! flux3_image_reference`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::flux3::Flux3Transformer;

fn raw(path: &Path) -> Vec<[u8; 4]> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]).collect()
}

fn f32s(dir: &Path, name: &str) -> Vec<f32> {
    raw(&dir.join(format!("{name}.bin"))).into_iter().map(f32::from_le_bytes).collect()
}

fn ids(dir: &Path, name: &str) -> Vec<[i32; 4]> {
    raw(&dir.join(format!("{name}.bin"))).into_iter().map(i32::from_le_bytes).collect::<Vec<_>>().chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
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
    (dot / (na.sqrt() * nb.sqrt()), maxerr / maxref)
}

fn assert_close(what: &str, ours: &[f32], reference: &[f32], min_cos: f64, max_rel: f64) {
    let (cos, rel) = agreement(ours, reference);
    println!("{what}: cosine {cos:.7}, max relative error {rel:.2e}");
    assert!(cos >= min_cos && rel <= max_rel, "{what}: cosine {cos}, max relative error {rel}");
}

fn tiny_root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_FLUX3_IMAGE_PARITY").expect("PRAECISE_FLUX3_IMAGE_PARITY names the fixture dir"))
}

fn exact() -> LoadOptions {
    LoadOptions { precision: Precision::F32, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None }
}

fn context(tokens: Vec<f32>, width: usize) -> Context {
    let n = tokens.len() / width;
    Context { tokens, ids: (0..n as i32).map(|i| [0, 0, 0, i]).collect(), times: vec![0.0; n] }
}

fn tiny_models(dir: &Path) -> Vec<(&'static str, Flux3Transformer)> {
    let package = Flux3Transformer::load(&[dir.join("model.safetensors")], "", exact()).unwrap();
    let source = TransformerSource::Diffusers(dir.join("diffusers").join("transformer"));
    let diffusers = Flux3Transformer::from_safetensors(source.open().unwrap(), WType::F32, exact()).unwrap();
    assert_eq!(package.config(), diffusers.config());
    vec![("package", package.with_position_scale(POSITION_SCALE)), ("diffusers", diffusers.with_position_scale(POSITION_SCALE))]
}

#[test]
#[ignore = "needs PRAECISE_FLUX3_IMAGE_PARITY fixtures"]
fn flux3_image_parity_transformer() {
    let meta: Value = serde_json::from_slice(&std::fs::read(tiny_root().join("meta.json")).unwrap()).unwrap();
    for (tag, m) in meta.as_object().unwrap() {
        let dir = tiny_root().join(tag);
        let ctx = context(f32s(&dir, "ctx"), 24);
        let (refs, ref_ids) = (f32s(&dir, "refs"), ids(&dir, "ref_ids"));
        let noise = f32s(&dir, "noise");
        let grid = (m["grid"][0].as_u64().unwrap() as usize, m["grid"][1].as_u64().unwrap() as usize);
        let img_ids = grid_ids(grid.0, grid.1, 0);
        let t = vec![model_time(m["t"].as_f64().unwrap() as f32); img_ids.len()];
        let target = m["target"].as_f64().map(|v| model_time(v as f32));
        for (layout, tf) in tiny_models(&dir) {
            assert_eq!(tf.config().flow_map, m["flow_map"].as_bool().unwrap());
            let streams = [
                StreamInput { name: IMAGE, tokens: &noise, ids: &img_ids, timesteps: &t },
                StreamInput { name: COND, tokens: &refs, ids: &ref_ids, timesteps: &vec![0.0; ref_ids.len()] },
            ];
            let out = tf.forward_to(ctx.input(), None, &streams, target).unwrap();
            assert_close(&format!("{tag} {layout} velocity"), &out[0], &f32s(&dir, "velocity"), 0.999_999, 1e-4);
            // Without the target (or with one the model does not take) the
            // evaluation is refused.
            let wrong = if target.is_some() { None } else { Some(0.5) };
            assert!(tf.forward_to(ctx.input(), None, &streams, wrong).is_err());
        }
    }
}

#[test]
#[ignore = "needs PRAECISE_FLUX3_IMAGE_PARITY fixtures"]
fn flux3_image_parity_sampling() {
    let meta: Value = serde_json::from_slice(&std::fs::read(tiny_root().join("meta.json")).unwrap()).unwrap();
    for (tag, m) in meta.as_object().unwrap() {
        let dir = tiny_root().join(tag);
        let (ctx, unc) = (context(f32s(&dir, "ctx"), 24), context(f32s(&dir, "uncond"), 24));
        let (refs, ref_ids) = (f32s(&dir, "refs"), ids(&dir, "ref_ids"));
        let grid = (m["grid"][0].as_u64().unwrap() as usize, m["grid"][1].as_u64().unwrap() as usize);
        let steps = m["steps"].as_u64().unwrap() as usize;
        let times = sampling_times(steps, m["shift"].as_f64().unwrap() as f32);
        let want_times: Vec<f32> = serde_json::from_value(m["schedule"].clone()).unwrap();
        assert_close(&format!("{tag} schedule"), &times, &want_times, 0.999_999_9, 1e-6);
        for (layout, tf) in tiny_models(&dir) {
            let mut seen = Vec::new();
            let (_, evaluations) = sample(
                &tf,
                (&ctx, Some(&unc)),
                Conditioning { tokens: &refs, ids: &ref_ids },
                f32s(&dir, "noise"),
                grid,
                &times,
                m["guidance"].as_f64().unwrap() as f32,
                &mut |_, x| seen.push(x.to_vec()),
            )
            .unwrap();
            assert_eq!(evaluations as usize, if tf.config().flow_map { steps } else { 2 * steps });
            for (i, x) in seen.iter().enumerate() {
                assert_close(&format!("{tag} {layout} step {i}"), x, &f32s(&dir, &format!("step{i}")), 0.999_999, 1e-4);
            }
        }
    }
}

fn reference_root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_FLUX3_IMAGE_REFERENCE").expect("PRAECISE_FLUX3_IMAGE_REFERENCE names the fixture dir"))
}

fn rgb_file(dir: &Path, name: &str, width: u32, height: u32) -> RgbImage {
    let rgb = std::fs::read(dir.join(name)).unwrap();
    assert_eq!(rgb.len(), 3 * width as usize * height as usize, "{name}");
    RgbImage { width, height, rgb }
}

/// Peak signal-to-noise ratio of two 8-bit images, in dB.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64.powi(2) / mse.max(1e-12)).log10()
}

/// The released checkpoints against the reference implementation's outputs,
/// case by case (`meta.json`): the prompt's text context, the latent after
/// every step from the same starting noise, and the decoded image. Prints
/// every measure; asserts the bounds each case names.
#[test]
#[ignore = "needs PRAECISE_FLUX3_IMAGE_REFERENCE fixtures and the released weights"]
fn flux3_image_reference() {
    let root = reference_root();
    let meta: Value = serde_json::from_slice(&std::fs::read(root.join("meta.json")).unwrap()).unwrap();
    let mut loaded: Option<(String, Flux3Image)> = None;
    for case in meta["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let dir = root.join(name);
        let weights = case["weights"].as_str().unwrap();
        let variant = case["variant"].as_str();
        let key = format!("{weights}:{variant:?}:{}", case["precision"].as_str().unwrap_or("bf16"));
        if loaded.as_ref().is_none_or(|(k, _)| *k != key) {
            drop(loaded.take());
            let precision = match case["precision"].as_str().unwrap_or("bf16") {
                "q8_0" => Precision::Q8_0,
                "f32" => Precision::F32,
                _ => Precision::Bf16,
            };
            let t = std::time::Instant::now();
            let p = Flux3Image::load(Path::new(weights), variant, LoadOptions { precision, ..LoadOptions::default() }).unwrap();
            println!("{key}: loaded on {} in {} ms, {:.2} GiB resident", p.device(), t.elapsed().as_millis(), p.resident_bytes() as f64 / f64::from(1u32 << 30));
            loaded = Some((key, p));
        }
        let p = &mut loaded.as_mut().unwrap().1;
        let (width, height) = (case["width"].as_u64().unwrap() as u32, case["height"].as_u64().unwrap() as u32);
        let mut references = Vec::new();
        for r in case["references"].as_array().unwrap() {
            let original = rgb_file(&dir, r["original"].as_str().unwrap(), r["original_size"][0].as_u64().unwrap() as u32, r["original_size"][1].as_u64().unwrap() as u32);
            let prepared = rgb_file(&dir, r["prepared"].as_str().unwrap(), r["prepared_size"][0].as_u64().unwrap() as u32, r["prepared_size"][1].as_u64().unwrap() as u32);
            references.push((original, prepared));
        }
        let originals: Vec<RgbImage> = references.iter().map(|r| r.0.clone()).collect();
        let (ours, _) = prepare_references(&originals).unwrap();
        for (i, (o, (_, want))) in ours.iter().zip(&references).enumerate() {
            assert_eq!((o.width, o.height), (want.width, want.height), "{name}: reference {i} size");
            let diff = o.rgb.iter().zip(&want.rgb).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
            println!("{name}: reference {i} prepared {}x{}, max pixel difference {diff}", o.width, o.height);
        }
        let ctx = p.context(case["prompt"].as_str().unwrap()).unwrap();
        assert_close(&format!("{name} text context"), &ctx.tokens, &f32s(&dir, "context"), case["min_context_cosine"].as_f64().unwrap_or(0.999), 1.0);
        let req = Request {
            prompt: case["prompt"].as_str().unwrap().to_string(),
            references: references.iter().map(|r| r.1.clone()).collect(),
            width,
            height,
            steps: case["steps"].as_u64().unwrap() as u32,
            guidance_scale: case["guidance"].as_f64().unwrap_or(1.0) as f32,
            seed: case["seed"].as_u64().unwrap(),
        };
        let mut steps = Vec::new();
        let t = std::time::Instant::now();
        let image = p.run(&req, f32s(&dir, "noise"), &mut |_, x| steps.push(x.to_vec())).unwrap();
        let ms = t.elapsed().as_millis();
        let mut last = 0.0;
        for (i, x) in steps.iter().enumerate() {
            let (cos, rel) = agreement(x, &f32s(&dir, &format!("step{i}")));
            println!("{name}: step {i} latent cosine {cos:.6}, max relative error {rel:.3e}");
            last = cos;
        }
        let want = std::fs::read(dir.join("image.rgb")).unwrap();
        let db = psnr(&image.rgb, &want);
        let mean_abs = image.rgb.iter().zip(&want).map(|(&a, &b)| f64::from(a.abs_diff(b))).sum::<f64>() / want.len() as f64;
        println!(
            "{name}: final latent cosine {last:.6}, image PSNR {db:.2} dB, mean absolute pixel difference {mean_abs:.3}, {} evaluations, {ms} ms",
            image.evaluations
        );
        if let Some(out) = std::env::var_os("PRAECISE_FLUX3_IMAGE_OUT") {
            let path = PathBuf::from(out).join(format!("{name}.ppm"));
            let mut bytes = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
            bytes.extend_from_slice(&image.rgb);
            std::fs::write(path, bytes).unwrap();
        }
        assert!(last >= case["min_latent_cosine"].as_f64().unwrap_or(0.99), "{name}: final latent cosine {last}");
        assert!(db >= case["min_psnr"].as_f64().unwrap_or(25.0), "{name}: image PSNR {db}");
    }
}
