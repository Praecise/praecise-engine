//! Agreement of the image transformer and time grid with the reference
//! implementation, on fixtures produced by
//! `tests/parity/make_flux3_image_fixtures.py` (random diffusers image
//! transformers at tiny widths, in the release's tensor names, with and
//! without a reference image, as a flow map and not).
//!
//! Run with `PRAECISE_FLUX3_IMAGE_PARITY=<fixture dir> cargo test -p
//! praecise-diffusion -- --ignored flux3_image`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "tensor and kernel math: sizes, indices and counts move between usize, i64, u32 and f32 within bounds the tensor shapes fix; a checked conversion at each site would change no value"
)]

use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::flux3::parity::{agreement, f32s, ids};
use crate::pipeline::Precision;

fn root() -> PathBuf {
    PathBuf::from(std::env::var("PRAECISE_FLUX3_IMAGE_PARITY").expect("PRAECISE_FLUX3_IMAGE_PARITY names the fixture dir"))
}

fn run(precision: Precision, min_cos: f64, max_rel: f64) {
    let meta: Value = serde_json::from_slice(&std::fs::read(root().join("meta.json")).unwrap()).unwrap();
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };
    for (tag, m) in meta.as_object().unwrap() {
        let dir = root().join(tag);
        let tf = Flux3Transformer::load(&[dir.join("model.safetensors")], "", opts).unwrap().with_spatial_rope_scale(IMAGE_ROPE_SCALE);
        assert_eq!(tf.config().flow_map, m["flow_map"].as_bool().unwrap(), "{tag}");
        let (img, img_ids, reference, ref_ids) = (f32s(&dir, "img"), ids(&dir, "img_ids"), f32s(&dir, "ref"), ids(&dir, "ref_ids"));
        let (ctx, ctx_ids) = (f32s(&dir, "ctx"), ids(&dir, "ctx_ids"));
        let ctx_t = vec![0f32; ctx_ids.len()];
        let text = TextInput { tokens: &ctx, ids: &ctx_ids, timesteps: &ctx_t };
        let img_t = vec![m["t"].as_f64().unwrap() as f32; img_ids.len()];
        let ref_t = vec![0f32; ref_ids.len()];
        let target = m["target"].as_f64().map(|v| v as f32);
        assert!(tf.forward_to(text, None, &[], if target.is_some() { None } else { Some(0.5) }).is_err(), "{tag}: a target time against the model");
        for case in ["ref", "noref"] {
            let mut streams = vec![StreamInput { name: "image", tokens: &img, ids: &img_ids, timesteps: &img_t }];
            if case == "ref" {
                streams.push(StreamInput { name: "image_cond", tokens: &reference, ids: &ref_ids, timesteps: &ref_t });
            }
            let out = tf.forward_to(text, None, &streams, target).unwrap();
            let (cos, rel) = agreement(&out[0], &f32s(&dir, &format!("out_{case}")));
            eprintln!("{tag} {case}: cosine {cos:.6}, max relative error {rel:.6}");
            assert!(cos >= min_cos && rel <= max_rel, "{tag} {case}: cosine {cos}, max relative error {rel}");
        }
    }
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_image_parity_f32() {
    run(Precision::F32, 0.999_999, 1e-4);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_image_parity_bf16() {
    run(Precision::Bf16, 0.9999, 2e-2);
}

#[test]
#[ignore = "needs the reference fixtures"]
fn flux3_image_time_grid_is_the_reference_schedulers() {
    for steps in [DISTILLED_STEPS, BASE_STEPS] {
        let ours = euler_times(steps as usize, SCHEDULE_SHIFT);
        let reference = f32s(&root(), &format!("sigmas_{steps}"));
        assert_eq!(ours.len(), reference.len(), "{steps} steps");
        let worst = ours.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-6, "{steps} steps: grids differ by {worst}");
    }
}
