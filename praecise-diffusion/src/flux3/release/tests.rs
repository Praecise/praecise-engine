use std::path::{Path, PathBuf};

use super::*;
use crate::flux3::packing::Frame;
use crate::flux3::policy::{encode_conditions, encode_instruction, predict_chunk, Conditioning, Noise, Observation};
use crate::ggml::Device;
use crate::pipeline::Precision;

fn write(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
}

/// A minimal safetensors file of f32 vectors.
fn write_f32s(path: &Path, tensors: &[(&str, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, v) in tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        header.insert(
            (*name).to_owned(),
            serde_json::json!({"dtype": "F32", "shape": [v.len()], "data_offsets": [start, data.len()]}),
        );
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("flux3-release-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("policy")).unwrap();
    std::fs::create_dir_all(d.join("base/text_encoder")).unwrap();
    write(&d.join("base/video_vae.safetensors"), "");
    write(&d.join("policy/model.safetensors"), "");
    d
}

const HISTORY_CONFIG: &str = r#"{"type":"flux3","n_obs_steps":8,"conditioning":"history","packer":null,
"condition_on_past_actions":true,"history_snapshots":2,"chunk_size":42,"n_action_steps":32,"fps":30.0,
"video_position_fps":24.0,"text_fixed_length":320,"camera_layout":"side_by_side","canvas_hw":[256,512],
"camera_keys":["observation.images.scene","observation.images.wrist"],"action_modality":"action","action_scale":2.0,
"gripper_flip_dims":[],
"video_vae_id":"org/base:video_vae.safetensors@abc","text_encoder_id":"org/base:text_encoder@abc","output_features":{"action":{"type":"ACTION","shape":[6]}},"trunk_weights":null,
"sampler":"euler","num_inference_steps":4,"guidance_scale":3.0,"guidance_scale_action":null,"sampler_shift":6.93}"#;

const PREPROCESSOR: &str = r#"{"steps":[{"registry_name":"to_batch_processor","config":{}},
{"registry_name":"flux3_observation_history_normalizer","config":{"action_dim":6,"action_representation":"delta",
"absolute_dims":[-1],"normalization_clip":6.0},"state_file":"stats.safetensors"}]}"#;

#[test]
fn opens_a_history_release() {
    let d = scratch("history");
    write(&d.join("policy/config.json"), HISTORY_CONFIG);
    write(&d.join("policy/policy_preprocessor.json"), PREPROCESSOR);
    let lo = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, 0.0];
    let hi = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 50.0];
    write_f32s(
        &d.join("policy/stats.safetensors"),
        &[("action.q01", lo.clone()), ("action.q99", hi.clone()), ("state.q01", lo.clone()), ("state.q99", hi.clone())],
    );
    let r = PolicyRelease::open(&d.join("policy"), &d.join("base")).unwrap();
    assert_eq!(r.config.conditioning, Conditioning::History);
    assert_eq!(r.camera_keys, ["observation.images.scene", "observation.images.wrist"]);
    assert_eq!(r.dit_files, [d.join("policy/model.safetensors")]);
    assert_eq!(r.vae_files, [d.join("base/video_vae.safetensors")]);
    assert_eq!(r.text_dir, d.join("base/text_encoder"));
    assert_eq!((r.base_repo.as_str(), r.base_revision.as_str()), ("org/base", "abc"));
    let n = r.normalization.unwrap();
    assert_eq!(n.representation, ActionRepresentation::Delta { absolute: vec![-1] });
    assert_eq!(n.action.q01, lo);
    assert_eq!(n.state.q99, hi);
    assert!((n.clip - 6.0).abs() < 1e-6);
    assert_eq!(n.states(&[0.0; 6])[5], -1.0);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn opens_a_frame_release_and_resolves_negative_channels() {
    let d = scratch("frame");
    let cfg = HISTORY_CONFIG
        .replace(r#""conditioning":"history""#, r#""conditioning":"frame""#)
        .replace(r#""gripper_flip_dims":[]"#, r#""gripper_flip_dims":[-1]"#);
    write(&d.join("policy/config.json"), &cfg);
    let r = PolicyRelease::open(&d.join("policy"), &d.join("base")).unwrap();
    assert_eq!(r.config.conditioning, Conditioning::Frame);
    assert_eq!(r.config.gripper_flip_dims, [5]);
    assert!(r.normalization.is_none());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn refuses_incomplete_releases() {
    let d = scratch("refuse");
    // History policies need their command statistics.
    write(&d.join("policy/config.json"), HISTORY_CONFIG);
    assert!(PolicyRelease::open(&d.join("policy"), &d.join("base")).is_err());
    // Out-of-range channel indices.
    write(&d.join("policy/config.json"), &HISTORY_CONFIG.replace(r#""gripper_flip_dims":[]"#, r#""gripper_flip_dims":[6]"#));
    assert!(PolicyRelease::open(&d.join("policy"), &d.join("base")).is_err());
    // Escaping base paths.
    write(&d.join("policy/config.json"), &HISTORY_CONFIG.replace("org/base:text_encoder@abc", "org/base:../x@abc"));
    assert!(PolicyRelease::open(&d.join("policy"), &d.join("base")).is_err());
    // Missing weights.
    write(&d.join("policy/config.json"), &HISTORY_CONFIG.replace(r#""conditioning":"history""#, r#""conditioning":"frame""#));
    std::fs::remove_file(d.join("base/video_vae.safetensors")).unwrap();
    assert!(PolicyRelease::open(&d.join("policy"), &d.join("base")).is_err());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn splits_an_observation_grid() {
    // 2 time steps x 3 cameras of 2x1 pixels; each pixel holds (camera, time).
    let (w, h) = (6, 2);
    let mut rgb = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgb.extend([(x / 2) as u8 * 10, y as u8 * 10, 0]);
        }
    }
    let g = frames_from_grid(w, h, &rgb, 3, 2).unwrap();
    assert_eq!(g.len(), 3);
    assert!(g.iter().all(|c| c.len() == 2 && c[0].w == 2 && c[0].h == 1));
    let probe = Frame::from_u8(1, 2, &[20, 10, 0, 20, 10, 0]).unwrap();
    assert_eq!(g[2][1].data, probe.data);
    assert!(frames_from_grid(7, 2, &vec![0; 42], 3, 2).is_err());
}

/// A smooth synthetic camera frame.
fn frame(h: usize, w: usize, phase: usize) -> Frame {
    let mut px = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                px.push(((x * 255 / w + y * 128 / h + c * 60 + phase * 4) % 256) as u8);
            }
        }
    }
    Frame::from_u8(h, w, &px).unwrap()
}

/// Real released weights, staged so only one component is resident at a
/// time: `PRAECISE_FLUX3_RELEASE=<policy dir>:<base dir>`.
#[test]
#[ignore = "needs released weights"]
fn flux3_release_real() {
    let spec = std::env::var("PRAECISE_FLUX3_RELEASE").expect("PRAECISE_FLUX3_RELEASE=<policy dir>:<base dir>");
    let (policy_dir, base_dir) = spec.split_once(':').unwrap();
    let r = PolicyRelease::open(Path::new(policy_dir), Path::new(base_dir)).unwrap();
    let cfg = r.config.clone();
    let threads = std::env::var("PRAECISE_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(8);
    let opts = LoadOptions { precision: Precision::Bf16, cpu_threads: threads, device: Some(Device::Cpu) };
    let d = cfg.action_dim;
    let frames = match cfg.conditioning {
        Conditioning::Frame => 1,
        Conditioning::History => cfg.n_obs_steps,
    };
    let feats: Vec<(usize, usize)> = r.camera_keys.iter().map(|_| (cfg.content_hw.0, cfg.content_hw.1 / r.camera_keys.len().max(1))).collect();
    let cameras: Vec<Vec<Frame>> = feats.iter().enumerate().map(|(c, &(h, w))| (0..frames).map(|t| frame(h, w, c * 7 + t)).collect()).collect();
    // Commands at the middle of the state range, drifting slowly.
    let (states, past) = match &r.normalization {
        Some(n) => {
            let raw: Vec<f32> = (0..frames * d).map(|i| 0.5 * (n.state.q01[i % d] + n.state.q99[i % d]) + (i / d) as f32 * 0.1).collect();
            let past = cfg.condition_on_past_actions.then(|| n.past_actions(&raw));
            (n.states(&raw), past)
        }
        None => (vec![0.0; frames * d], None),
    };
    let obs = Observation { cameras: &cameras, states: &states, past_actions: past.as_deref(), instruction: "pick up the cube and place it in the bowl" };

    let t0 = std::time::Instant::now();
    let (ctx, uncond) = {
        let text = r.load_text_encoder(opts).unwrap();
        encode_instruction(&cfg, &text, obs.instruction).unwrap()
    };
    eprintln!("text: {} values, guided {}, {:.1}s", ctx.len(), uncond.is_some(), t0.elapsed().as_secs_f32());
    let t1 = std::time::Instant::now();
    let cond = {
        let vae = r.load_autoencoder(opts).unwrap();
        encode_conditions(&cfg, &vae, &obs).unwrap()
    };
    eprintln!("conditions: {} video values, {:.1}s", cond.video.len(), t1.elapsed().as_secs_f32());
    let t2 = std::time::Instant::now();
    let dit = r.load_transformer(opts).unwrap();
    let chunk = predict_chunk(&dit, &cfg, &cond, &ctx, uncond.as_deref(), &Noise::from_seed(&cfg, 42)).unwrap();
    eprintln!("chunk: {:.1}s", t2.elapsed().as_secs_f32());

    assert_eq!(chunk.len(), cfg.chunk_size * d);
    assert!(chunk.iter().all(|x| x.is_finite()), "non-finite actions");
    let max = chunk.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let mean = chunk.iter().map(|x| x.abs()).sum::<f32>() / chunk.len() as f32;
    eprintln!("normalised actions: mean |a| {mean:.4}, max |a| {max:.4}");
    for row in chunk.chunks(d).take(4) {
        eprintln!("  {row:?}");
    }
    if let Some(n) = &r.normalization {
        let anchor: Vec<f32> = (0..d).map(|c| 0.5 * (n.state.q01[c] + n.state.q99[c]) + (frames - 1) as f32 * 0.1).collect();
        let commands = n.commands(&chunk, &anchor);
        eprintln!("first commands: {:?}", &commands[..d]);
        eprintln!("last commands:  {:?}", &commands[commands.len() - d..]);
    }
    // A trained policy keeps its prediction inside the scaled range.
    assert!(max < 4.0, "actions far outside the training range: {max}");
}
