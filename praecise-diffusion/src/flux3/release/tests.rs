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

/// A release of `case` (history conditioning, the fixture's statistics) in
/// a scratch directory.
fn rollout_release(case: &serde_json::Value, d: &Path) -> PolicyRelease {
    let n = |k: &str| case[k].as_u64().unwrap();
    let dim = n("d");
    let keys: Vec<String> = (0..n("cams")).map(|c| format!("observation.images.cam{c}")).collect();
    let mut cfg: serde_json::Value = serde_json::from_str(HISTORY_CONFIG).unwrap();
    cfg["n_obs_steps"] = n("n_obs").into();
    cfg["chunk_size"] = n("chunk").into();
    cfg["n_action_steps"] = n("execute").into();
    cfg["condition_on_past_actions"] = case["past"].clone();
    cfg["camera_keys"] = serde_json::json!(keys);
    cfg["output_features"]["action"]["shape"] = serde_json::json!([dim]);
    write(&d.join("policy/config.json"), &cfg.to_string());
    let pre = serde_json::json!({"steps": [{"registry_name": "flux3_observation_history_normalizer",
        "config": {"action_dim": dim, "action_representation": case["repr"], "absolute_dims": case["absolute"], "normalization_clip": 6.0},
        "state_file": "stats.safetensors"}]});
    write(&d.join("policy/policy_preprocessor.json"), &pre.to_string());
    let q = &case["quantiles"];
    let v = |k: &str| q[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<_>>();
    write_f32s(
        &d.join("policy/stats.safetensors"),
        &[("action.q01", v("action.q01")), ("action.q99", v("action.q99")), ("state.q01", v("state.q01")), ("state.q99", v("state.q99"))],
    );
    PolicyRelease::open(&d.join("policy"), &d.join("base")).unwrap()
}

/// The fixture's stand-in for the chunk sampler: `tanh(W x + b)` over the
/// normalised states, the past actions and the mean of every frame.
fn rollout_stub(case: &serde_json::Value) -> impl Fn(&Observation<'_>) -> Result<Vec<f32>> + 'static {
    let w: Vec<Vec<f64>> = case["w"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
    let b: Vec<f64> = case["b"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    move |o: &Observation<'_>| {
        let mut x: Vec<f64> = o.states.iter().map(|&v| f64::from(v)).collect();
        x.extend(o.past_actions.unwrap_or(&[]).iter().map(|&v| f64::from(v)));
        for cam in o.cameras {
            x.extend(cam.iter().map(|f| f64::from(f.data.iter().sum::<f32>() / f.data.len() as f32)));
        }
        assert_eq!(x.len(), w[0].len(), "stand-in inputs");
        Ok(w.iter().zip(&b).map(|(row, bias)| (row.iter().zip(&x).map(|(a, v)| a * v).sum::<f64>() + bias).tanh() as f32).collect())
    }
}

fn tick_frames(case: &serde_json::Value, tick: &serde_json::Value) -> Vec<Frame> {
    let hw = case["hw"].as_array().unwrap();
    let (h, w) = (hw[0].as_u64().unwrap() as usize, hw[1].as_u64().unwrap() as usize);
    tick["pixels"].as_array().unwrap().iter().map(|px| {
        let px: Vec<u8> = px.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
        Frame::from_u8(h, w, &px).unwrap()
    }).collect()
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// The closed-loop rollout against the reference control loop, tick by tick
/// and chunk by chunk. Fixtures: `make_flux3_rollout_fixtures.py`, at
/// `PRAECISE_FLUX3_ROLLOUT=<dir>`.
#[test]
#[ignore = "needs reference fixtures"]
fn flux3_rollout_parity() {
    let dir = PathBuf::from(std::env::var("PRAECISE_FLUX3_ROLLOUT").expect("PRAECISE_FLUX3_ROLLOUT=<fixture dir>"));
    let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("rollout.json")).unwrap()).unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let d = scratch(&format!("rollout-{name}"));
        let release = rollout_release(case, &d);
        let dim = release.config.action_dim;
        let mut worst = 0.0f32;

        // Tick by tick, with the scripted observations.
        let mut r = Rollout::new(&release, Box::new(rollout_stub(case)), "");
        for (t, tick) in case["scripted"].as_array().unwrap().iter().enumerate() {
            let got = r.tick(&tick_frames(case, tick), &floats(&tick["state"])).unwrap();
            for (a, b) in got.iter().zip(floats(&tick["command"])) {
                worst = worst.max((a - b).abs());
                assert!((a - b).abs() < 1e-4, "{name} tick {t}: {got:?} vs {:?}", tick["command"]);
            }
        }

        // Chunk by chunk from a primed history, every command reached.
        let reached = case["reached"].as_array().unwrap();
        let first = &reached[0];
        let n = release.observation_frames();
        let frames = tick_frames(case, first);
        let cameras: Vec<Vec<Frame>> = frames.iter().map(|f| vec![f.clone(); n]).collect();
        let states: Vec<f32> = floats(&first["state"]).repeat(n);
        let mut r = Rollout::new(&release, Box::new(rollout_stub(case)), "");
        r.prime(&RawObservation { cameras: &cameras, states: &states, commands: None, instruction: "" }).unwrap();
        let exec = release.config.n_action_steps;
        for k in 0..reached.len() / exec {
            let chunk = r.chunk().unwrap();
            assert_eq!(chunk.len(), release.config.chunk_size * dim);
            for (j, tick) in reached[k * exec..(k + 1) * exec].iter().enumerate() {
                for (a, b) in chunk[j * dim..(j + 1) * dim].iter().zip(floats(&tick["command"])) {
                    worst = worst.max((a - b).abs());
                    assert!((a - b).abs() < 1e-4, "{name} chunk {k} step {j}: {a} vs {b}");
                }
            }
        }
        eprintln!("{name}: worst |command difference| {worst:.2e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}

/// The first tick fills the history; a chunk is drawn every execution
/// window, and each draw sees the commands the previous one produced.
#[test]
fn a_rollout_draws_a_chunk_per_execution_window() {
    let case = serde_json::json!({"d": 2, "n_obs": 2, "chunk": 3, "execute": 2, "cams": 1, "past": true,
        "repr": "absolute", "absolute": [],
        "quantiles": {"state.q01": [-1.0, -1.0], "state.q99": [1.0, 1.0], "action.q01": [-1.0, -1.0], "action.q99": [1.0, 1.0]}});
    let d = scratch("rollout-unit");
    let release = rollout_release(&case, &d);
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::<Vec<f32>>::new()));
    let log = seen.clone();
    let predict = move |o: &Observation<'_>| {
        log.borrow_mut().push(o.past_actions.unwrap().to_vec());
        let k = log.borrow().len() as f32;
        Ok(vec![0.1 * k, -0.1 * k, 0.2 * k, -0.2 * k, 0.3 * k, -0.3 * k])
    };
    let mut r = Rollout::new(&release, Box::new(predict), "");
    let cam = [frame(2, 2, 0)];
    let mut sent = Vec::new();
    for _ in 0..5 {
        sent.push(r.tick(&cam, &[0.0, 0.0]).unwrap());
    }
    assert_eq!(seen.borrow().len(), 3, "ticks 0, 2 and 4 draw");
    let near = |a: &[f32], b: [f32; 2]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
    // Ticks 0 and 1 run the first window, tick 2 starts the second draw.
    assert!(near(&sent[0], [0.1, -0.1]) && near(&sent[1], [0.2, -0.2]) && near(&sent[2], [0.2, -0.2]), "{sent:?}");
    // The first draw sees the state standing in for every command; the
    // second sees the command in effect at tick 2, sent at tick 1.
    assert!(seen.borrow()[0].iter().all(|v| v.abs() < 1e-6));
    let second = seen.borrow()[1].clone();
    assert!(near(&second[..2], [0.0, 0.0]) && near(&second[2..], [0.2, -0.2]), "{second:?}");
    std::fs::remove_dir_all(&d).unwrap();
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
    let noise = Noise::from_seed(&cfg, 42);
    let chunk = predict_chunk(&dit, &cfg, &cond, &ctx, uncond.as_deref(), &noise).unwrap();
    eprintln!("chunk: {:.1}s", t2.elapsed().as_secs_f32());
    // The inputs, the noise and the chunk, for a reference run
    // (`compare_flux3_release.py`): `PRAECISE_FLUX3_DUMP=<dir>`.
    if let Ok(dir) = std::env::var("PRAECISE_FLUX3_DUMP") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let put = |name: &str, v: &[f32]| std::fs::write(dir.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        put("cameras.f32", &cameras.iter().flatten().flat_map(|f| f.data.iter().copied()).collect::<Vec<f32>>());
        put("states.f32", obs.states);
        if let Some(p) = obs.past_actions {
            put("past.f32", p);
        }
        put("noise_video.f32", &noise.video);
        put("noise_action.f32", &noise.action);
        put("chunk.f32", &chunk);
        let meta = serde_json::json!({
            "camera_keys": r.camera_keys, "frames": frames, "action_dim": d, "chunk_size": cfg.chunk_size,
            "frame_hw": [feats[0].0, feats[0].1], "instruction": obs.instruction,
            "noise_video_len": noise.video.len(), "noise_action_len": noise.action.len(),
        });
        write(&dir.join("meta.json"), &meta.to_string());
    }

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
