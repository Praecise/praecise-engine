use std::cell::Cell;
use std::path::{Path, PathBuf};

use super::rollout::{ChunkRollout, NoiseSource};
use super::*;
use crate::pipeline::{LoadOptions, Precision};

#[test]
fn time_embedding_is_sines_then_cosines_over_geometric_periods() {
    let e = sinusoidal_time(0.5, 4, 0.004, 4.0);
    let periods = [0.004f64, 4.0];
    for (i, p) in periods.iter().enumerate() {
        let x = 0.5 * 2.0 * std::f64::consts::PI / p;
        assert!((f64::from(e[i]) - x.sin()).abs() < 1e-6);
        assert!((f64::from(e[2 + i]) - x.cos()).abs() < 1e-6);
    }
}

#[test]
fn euler_runs_from_noise_to_data() {
    let (t, dt) = euler_times(10);
    assert_eq!(t.len(), 10);
    assert_eq!(t[0], 1.0);
    assert!((t[9] - 0.1).abs() < 1e-6);
    assert!((dt + 0.1).abs() < 1e-7);
}

#[test]
fn bilinear_upsampling_uses_half_pixel_centres() {
    // One row [0, 1] doubled: the reference gives [0, 0.25, 0.75, 1].
    let f = Frame { h: 1, w: 2, data: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0] };
    let r = resize_bilinear(&f, 1, 4);
    assert_eq!(&r.data[..4], &[0.0, 0.25, 0.75, 1.0]);
}

#[test]
fn padding_goes_before_or_around_the_image() {
    let f = Frame { h: 2, w: 4, data: vec![1.0; 24] };
    let s = resize_with_pad(&f, 4, 4, PadAt::Start, -1.0);
    assert_eq!(&s.data[..4], &[-1.0; 4]);
    assert_eq!(&s.data[8..16], &[1.0; 8]);
    let c = resize_with_pad(&f, 4, 4, PadAt::Centre, -1.0);
    assert_eq!(&c.data[..4], &[-1.0; 4]);
    assert_eq!(&c.data[4..12], &[1.0; 8]);
    assert_eq!(&c.data[12..16], &[-1.0; 4]);
}

#[test]
fn normalisation_round_trips_in_every_mode() {
    let x = [0.5f32, -2.0, 3.0, 0.5, -2.0, 3.0];
    for mode in [NormMode::MeanStd, NormMode::MinMax, NormMode::Quantiles] {
        let n = FeatureNorm { mode, lo: vec![0.1, -3.0, 1.0], hi: vec![2.0, 4.0, 5.0], eps: 1e-8 };
        let back = n.unnormalize(&n.normalize(&x).unwrap()).unwrap();
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5, "{mode:?}");
        }
    }
    assert!(NormMode::parse("LOG").is_err());
}

#[test]
fn expert_mlp_width_rounds_to_256() {
    assert_eq!(smolvla::expert_intermediate(720), 2048);
    assert_eq!(smolvla::expert_intermediate(960), 2560);
}

/// A policy whose chunk rows are the chunk number and the row index.
struct Counting {
    calls: Cell<usize>,
}

impl ChunkPolicy for Counting {
    fn chunk_size(&self) -> usize {
        4
    }
    fn action_dim(&self) -> usize {
        2
    }
    fn n_action_steps(&self) -> usize {
        3
    }
    fn noise_len(&self) -> usize {
        1
    }
    fn predict(&self, obs: &VlaObservation<'_>, noise: &[f32]) -> Result<Vec<f32>> {
        let k = self.calls.get();
        self.calls.set(k + 1);
        Ok((0..4).flat_map(|r| [k as f32 * 10.0 + r as f32 + obs.state[0], noise[0]]).collect())
    }
}

#[test]
fn rollout_predicts_when_the_executed_steps_run_out() {
    let p = Counting { calls: Cell::new(0) };
    let mut r = ChunkRollout::new(&p, NoiseSource::Fixed(vec![7.0])).unwrap();
    let mut got = Vec::new();
    for t in 0..7 {
        let state = [t as f32 * 100.0];
        let obs = VlaObservation { cameras: &[], state: &state, instruction: "" };
        got.push(r.tick(&obs).unwrap());
    }
    // Chunks at ticks 0, 3 and 6 from that tick's state; the fourth row of
    // each chunk is never executed.
    let first: Vec<f32> = got.iter().map(|a| a[0]).collect();
    assert_eq!(first, vec![0.0, 1.0, 2.0, 310.0, 311.0, 312.0, 620.0]);
    assert!(got.iter().all(|a| a[1] == 7.0));
    assert_eq!(r.chunks(), 3);
    assert!(ChunkRollout::new(&p, NoiseSource::Fixed(vec![])).is_err());
}

/// A recorded reference rollout: frames, states, the chunk noise and the
/// actions the reference control loop returned at every tick.
struct Fixture {
    meta: serde_json::Value,
    st: SafeTensors,
}

impl Fixture {
    fn open(dir: &Path) -> Self {
        let meta = read_json(&dir.join("fixture.json")).unwrap();
        let st = SafeTensors::open(&[dir.join("fixture.safetensors")]).unwrap();
        Self { meta, st }
    }
    fn f32s(&self, name: &str) -> Vec<f32> {
        self.st.get(name).unwrap_or_else(|| panic!("fixture has no {name}")).to_f32()
    }
    fn usize(&self, k: &str) -> usize {
        self.meta[k].as_u64().unwrap_or_else(|| panic!("fixture.json has no {k}")) as usize
    }
    fn frame(&self, tick: usize, cam: usize) -> Frame {
        let name = format!("frame.{tick}.{cam}");
        let shape = self.st.get(&name).unwrap().shape.to_vec();
        Frame { h: shape[1] as usize, w: shape[2] as usize, data: self.f32s(&name) }
    }
}

fn parity_options() -> (LoadOptions, f32) {
    let precision = std::env::var("PRAECISE_PARITY_PRECISION").unwrap_or_else(|_| "f32".into());
    let (p, tol) = match precision.as_str() {
        "f32" => (Precision::F32, 1e-3),
        "bf16" => (Precision::Bf16, 0.1),
        other => panic!("unknown precision {other}"),
    };
    let tol = std::env::var("PRAECISE_PARITY_TOL").ok().and_then(|t| t.parse().ok()).unwrap_or(tol);
    let threads = std::env::var("PRAECISE_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(8);
    (LoadOptions { precision: p, cpu_threads: threads, device: None }, tol)
}

/// Run `policy` in closed loop over the fixture's ticks and compare every
/// action with the reference, in units of the action deviation (or range).
fn rollout_matches(policy: &dyn ChunkPolicy, norm: &FeatureNorm, fx: &Fixture, tol: f32) {
    let ticks = fx.usize("ticks");
    let cams = fx.usize("cameras");
    let d = policy.action_dim();
    let states = fx.f32s("states");
    let sd = states.len() / ticks;
    let want = fx.f32s("actions");
    let instruction = fx.meta["instruction"].as_str().unwrap().to_owned();
    let mut r = ChunkRollout::new(policy, NoiseSource::Fixed(fx.f32s("noise"))).unwrap();
    let mut worst = 0f32;
    for t in 0..ticks {
        let frames: Vec<Frame> = (0..cams).map(|c| fx.frame(t, c)).collect();
        let obs = VlaObservation { cameras: &frames, state: &states[t * sd..(t + 1) * sd], instruction: &instruction };
        let got = r.tick(&obs).unwrap();
        let a = norm.normalize(&got).unwrap();
        let b = norm.normalize(&want[t * d..(t + 1) * d]).unwrap();
        let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        eprintln!("tick {t}: max normalised diff {diff:.3e} (got {:?})", &got[..d.min(3)]);
        worst = worst.max(diff);
    }
    eprintln!("rollout of {ticks} ticks, {} chunks: worst normalised diff {worst:.3e} (tolerance {tol:.1e})", r.chunks());
    assert!(worst <= tol, "worst normalised action difference {worst} exceeds {tol}");
}

fn parity_dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a fixture directory")))
}

/// The SmolVLA control loop against the reference policy's `select_action`
/// loop. `PRAECISE_SMOLVLA_PARITY` names a directory with `policy/`,
/// `backbone/`, `fixture.json` and `fixture.safetensors`.
#[test]
#[ignore = "needs a released policy and a reference fixture"]
fn smolvla_rollout_matches_reference() {
    let dir = parity_dir("PRAECISE_SMOLVLA_PARITY");
    let fx = Fixture::open(&dir);
    let (opts, tol) = parity_options();
    let mut p = smolvla::SmolVla::load(&dir.join("policy"), &dir.join("backbone"), opts).unwrap();
    p.set_fp32_gemm(true);
    p.set_n_action_steps(fx.usize("n_action_steps"));
    eprintln!("smolvla on {}: {} MB resident", p.device(), p.resident_bytes() >> 20);
    if let Some(want) = fx.meta["tokens"].as_array() {
        let want: Vec<i32> = want.iter().map(|v| v.as_i64().unwrap() as i32).collect();
        assert_eq!(p.tokens(fx.meta["instruction"].as_str().unwrap()).unwrap(), want, "instruction tokens");
    }
    let pc = read_json(&dir.join("policy/config.json")).unwrap();
    let norm = Normalization::from_release(&dir.join("policy"), &pc).unwrap();
    let start = std::time::Instant::now();
    rollout_matches(&p, &norm.action, &fx, tol);
    eprintln!("wall {:.1}s", start.elapsed().as_secs_f32());
}

fn gemma_dims(v: &serde_json::Value) -> pi05::GemmaDims {
    let u = |k: &str| v[k].as_u64().unwrap_or_else(|| panic!("dims miss {k}"));
    pi05::GemmaDims { width: u("width"), depth: u("depth") as usize, mlp: u("mlp_dim"), heads: u("num_heads"), kv_heads: u("num_kv_heads"), head_dim: u("head_dim") }
}

/// The pi0.5 control loop against the reference policy's `select_action`
/// loop. `PRAECISE_PI05_PARITY` names a directory with `policy/`,
/// `backbone/tokenizer.json`, `fixture.json` and `fixture.safetensors`;
/// network sizes recorded in the fixture (a small random-weight release)
/// override the released ones.
#[test]
#[ignore = "needs a released policy and a reference fixture"]
fn pi05_rollout_matches_reference() {
    let dir = parity_dir("PRAECISE_PI05_PARITY");
    let fx = Fixture::open(&dir);
    let (opts, tol) = parity_options();
    let dims = fx.meta.get("dims").filter(|d| !d.is_null()).map(|d| {
        let v = &d["vision"];
        let u = |k: &str| v[k].as_u64().unwrap();
        pi05::Pi05Dims {
            vlm: gemma_dims(&d["vlm"]),
            expert: gemma_dims(&d["expert"]),
            vision: siglip::SiglipConfig {
                hidden_size: u("hidden_size"),
                num_hidden_layers: u("num_hidden_layers") as usize,
                num_attention_heads: u("num_attention_heads"),
                intermediate_size: u("intermediate_size"),
                patch_size: u("patch_size") as usize,
                image_size: u("image_size") as usize,
                layer_norm_eps: 1e-6,
            },
            vocab: 257_152,
        }
    });
    let mut p = pi05::Pi05::load(&dir.join("policy"), &dir.join("backbone/tokenizer.json"), opts, dims).unwrap();
    p.set_fp32_gemm(true);
    p.set_n_action_steps(fx.usize("n_action_steps"));
    eprintln!("pi05 on {}: {} MB resident", p.device(), p.resident_bytes() >> 20);
    let pc = read_json(&dir.join("policy/config.json")).unwrap();
    let norm = Normalization::from_release(&dir.join("policy"), &pc).unwrap();
    if let Some(want) = fx.meta["tokens"].as_array() {
        let want: Vec<i32> = want.iter().map(|v| v.as_i64().unwrap() as i32).collect();
        let states = fx.f32s("states");
        let sd = states.len() / fx.usize("ticks");
        let s0 = norm.state.normalize(&states[..sd]).unwrap();
        assert_eq!(p.tokens(fx.meta["instruction"].as_str().unwrap(), &s0).unwrap(), want, "prompt tokens");
    }
    let start = std::time::Instant::now();
    rollout_matches(&p, &norm.action, &fx, tol);
    eprintln!("wall {:.1}s", start.elapsed().as_secs_f32());
}

#[test]
fn state_bins_match_the_reference_digitisation() {
    // Bin edges -1 + 2k/256: below -1 is -1, the top edge and above is 255.
    assert_eq!(pi05::discretize_state(&[-1.5, -1.0, 0.0, 0.999, 1.0, 2.0]), vec![-1, 0, 128, 255, 255, 255]);
    assert_eq!(pi05::prompt(" pick_up\nthe cube ", &[3, 200]), "Task: pick up the cube, State: 3 200;\nAction: ");
}

/// The GR00T N1.7 backbone and action head against the reference model.
/// `PRAECISE_GROOT_PARITY` names a directory with `policy/`,
/// `backbone/config.json`, `fixture.json` and `fixture.safetensors`
/// (written by `tests/parity/make_groot_fixture.py`): for every sample the
/// backbone hidden states and the action chunk are compared.
#[test]
#[ignore = "needs a released policy and a reference fixture"]
fn groot_matches_reference() {
    let dir = parity_dir("PRAECISE_GROOT_PARITY");
    let fx = Fixture::open(&dir);
    let (opts, tol) = parity_options();
    let mut p = groot::GrootPolicy::load(&dir.join("policy"), &dir.join("backbone/config.json"), fx.usize("embodiment"), opts).unwrap();
    // The reference computes in FP32; CUDA would otherwise round the
    // products to TF32.
    p.set_fp32_gemm(true);
    let tokens: Vec<u32> = fx.meta["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let start = std::time::Instant::now();
    let (mut worst_f, mut worst_a) = (0f32, 0f32);
    for s in 0..fx.usize("samples") {
        let hw = &fx.meta["sizes"][s];
        let (h, w) = (hw[0].as_u64().unwrap() as usize, hw[1].as_u64().unwrap() as usize);
        let px = fx.f32s(&format!("{s}.pixels"));
        let want_px = fx.f32s(&format!("{s}.pixel_values"));
        let img = p.encoder().image(&px, (h, w)).unwrap();
        let dp = img.patches.iter().zip(&want_px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert_eq!(img.patches.len(), want_px.len(), "patch layout");
        assert!(dp < 1e-5, "sample {s}: patches differ by {dp}");
        let (feats, _) = p.features(&tokens, &[(&px, (h, w))]).unwrap();
        let want_f = fx.f32s(&format!("{s}.features"));
        assert_eq!(feats.len(), want_f.len(), "backbone tokens");
        let scale = want_f.iter().fold(0f32, |m, v| m.max(v.abs()));
        let df = feats.iter().zip(&want_f).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
        let got = p.predict(&tokens, &[(&px, (h, w))], &fx.f32s(&format!("{s}.state")), &fx.f32s(&format!("{s}.noise"))).unwrap();
        let want = fx.f32s(&format!("{s}.actions"));
        let da = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("sample {s}: patches {dp:.1e}, backbone {df:.3e} (relative), actions {da:.3e} (got {:?})", &got[..3]);
        worst_f = worst_f.max(df);
        worst_a = worst_a.max(da);
    }
    eprintln!("worst backbone {worst_f:.3e}, worst action {worst_a:.3e} (tolerance {tol:.1e}); wall {:.1}s", start.elapsed().as_secs_f32());
    assert!(worst_f <= tol && worst_a <= tol, "GR00T differs from the reference");
}

/// The MolmoAct2 vision backbone against the reference.
/// `PRAECISE_MOLMOACT2_VISION_PARITY` names a directory with `policy/`
/// (`config.json`, `model.safetensors`), `fixture.json` and
/// `fixture.safetensors` (written by
/// `tests/parity/make_molmoact2_vision_fixture.py`): for every sample the
/// pooled, projected image-token features are compared.
#[test]
#[ignore = "needs a released policy and a reference fixture"]
fn molmoact2_vision_matches_reference() {
    let dir = parity_dir("PRAECISE_MOLMOACT2_VISION_PARITY");
    let fx = Fixture::open(&dir);
    let (opts, tol) = parity_options();
    let mut v = molmoact2::MolmoAct2Vision::load(&dir.join("policy"), opts).unwrap();
    v.set_fp32_gemm(true);
    let mut worst = 0f32;
    for s in 0..fx.usize("samples") {
        let px = fx.f32s(&format!("{s}.pixels"));
        let pooling: Vec<i32> = fx.f32s(&format!("{s}.pooling")).iter().map(|&i| i as i32).collect();
        let crops = fx.meta["crops"][s].as_u64().unwrap() as usize;
        let pool = fx.usize("pool");
        let got = v.encode(&px, crops, &pooling, pool).unwrap();
        let want = fx.f32s(&format!("{s}.features"));
        assert_eq!(got.len(), want.len(), "sample {s}: image tokens");
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
        eprintln!("sample {s}: {} tokens, features {d:.3e} (relative, scale {scale:.3})", want.len() / v.config().out as usize);
        worst = worst.max(d);
    }
    eprintln!("worst {worst:.3e} (tolerance {tol:.1e})");
    assert!(worst <= tol, "MolmoAct2 vision differs from the reference");
}

#[test]
fn molmoact2_pixels_snap_to_bytes() {
    // Values snap to the nearest of the 256 levels, clamped.
    let x = molmoact2::snap_pixels(&[-1.0, 1.0, -2.0, 1.4 / 255.0 * 2.0 - 1.0, 1.6 / 255.0 * 2.0 - 1.0]);
    assert_eq!(&x[..3], &[-1.0, 1.0, -1.0]);
    assert!((x[3] - (1.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6);
    assert!((x[4] - (2.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6);
}

#[test]
fn molmoact2_vision_config_reads_a_release() {
    let c = serde_json::json!({
        "vit_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16,
            "num_key_value_heads": 16, "head_dim": 72, "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
            "image_default_input_size": [378, 378], "image_patch_size": 14, "image_num_pos": 729},
        "adapter_config": {"vit_layers": [-3, -9], "pooling_attention_mask": true, "hidden_size": 1152, "num_attention_heads": 16,
            "num_key_value_heads": 16, "head_dim": 72, "hidden_act": "silu", "intermediate_size": 9728, "text_hidden_size": 2560}
    });
    let cfg = molmoact2::MolmoAct2VisionConfig::from_json(&c).unwrap();
    assert_eq!(cfg.layers, vec![24, 18]);
    assert_eq!(cfg.blocks(), 25);
    assert_eq!(cfg.patches(), 729);
    assert_eq!(cfg.patch_len(), 588);
    assert!(cfg.pool_mask);
}

/// The MolmoAct2 prompt prefix through the release's preprocessing
/// against the reference policy (`PRAECISE_MOLMOACT2_PREFIX_PARITY`, made by
/// `tests/parity/make_molmoact2_prefix_fixture.py`): the prompt text, ids and
/// pooling indices exactly, the crops to the 8-bit level, then the image
/// features, the language model's final hidden states and every block's
/// keys and values, each relative to its scale.
#[test]
#[ignore = "needs a reference fixture"]
fn molmoact2_prefix_matches_reference() {
    let dir = parity_dir("PRAECISE_MOLMOACT2_PREFIX_PARITY");
    let fx = Fixture::open(&dir);
    let (opts, tol) = parity_options();
    let policy = dir.join("policy");
    let text = |k: &str| fx.meta[k].as_str().unwrap().to_owned();
    let proc = molmoact2_processor::MolmoAct2Processor::load(&policy, &text("setup_type"), &text("control_mode")).unwrap();
    let mut vision = molmoact2::MolmoAct2Vision::load(&policy, opts).unwrap();
    vision.set_fp32_gemm(true);
    let mut lm = molmoact2::MolmoAct2Lm::load(&policy, opts).unwrap();
    lm.set_fp32_gemm(true);
    let rel = |got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale
    };
    let mut worst = 0f32;
    for (s, smp) in fx.meta["samples"].as_array().unwrap().iter().enumerate() {
        let frames: Vec<resample::Frame8> = (0..smp["frames"].as_array().unwrap().len())
            .map(|j| {
                let name = format!("{s}.frame{j}");
                let shape = fx.st.get(&name).unwrap().shape.to_vec();
                let (h, w) = (shape[0] as usize, shape[1] as usize);
                let hwc = fx.f32s(&name);
                let data = (0..3).flat_map(|c| (0..h * w).map(move |i| (c, i))).map(|(c, i)| hwc[i * 3 + c]).collect();
                resample::Frame8::from_unit(&Frame { h, w, data })
            })
            .collect();
        let inputs = proc.prepare(&frames, &fx.f32s(&format!("{s}.state")), smp["task"].as_str().unwrap()).unwrap();
        assert_eq!(inputs.prompt, smp["prompt"].as_str().unwrap(), "sample {s}: prompt");
        let ints = |name: &str| fx.f32s(&format!("{s}.{name}")).iter().map(|&i| i as i32).collect::<Vec<i32>>();
        assert_eq!(inputs.ids, ints("ids"), "sample {s}: prompt ids");
        assert_eq!(inputs.pooling, ints("pooling"), "sample {s}: pooling indices");
        let want_px = fx.f32s(&format!("{s}.pixels"));
        assert_eq!(inputs.pixels.len(), want_px.len(), "sample {s}: crops");
        let off = inputs.pixels.iter().zip(&want_px).filter(|(a, b)| (*a - *b).abs() > 1e-6).count();
        let px = inputs.pixels.iter().zip(&want_px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let feats = vision.encode(&inputs.pixels, inputs.crops, &inputs.pooling, inputs.pool).unwrap();
        let fd = rel(&feats, &fx.f32s(&format!("{s}.features")));
        // The model alone, from the reference's crops.
        let from_ref = vision.encode(&want_px, inputs.crops, &inputs.pooling, inputs.pool).unwrap();
        let fr = rel(&from_ref, &fx.f32s(&format!("{s}.features")));
        let pre = lm.prefill(&inputs.ids, &feats).unwrap();
        let hd = rel(&pre.hidden, &fx.f32s(&format!("{s}.hidden")));
        let (mut kd, mut vd) = (0f32, 0f32);
        for l in 0..lm.config().layers {
            kd = kd.max(rel(&pre.keys[l], &fx.f32s(&format!("{s}.k{l}"))));
            vd = vd.max(rel(&pre.values[l], &fx.f32s(&format!("{s}.v{l}"))));
        }
        eprintln!(
            "sample {s}: {} tokens, {} crops; pixels off {off} (max {px:.3e}); features {fd:.3e} ({fr:.3e} from the reference crops), hidden {hd:.3e}, keys {kd:.3e}, values {vd:.3e} (relative)",
            inputs.ids.len(),
            inputs.crops
        );
        assert!(px <= 2.0 / 255.0 + 1e-6, "sample {s}: crops differ by more than one 8-bit level");
        worst = worst.max(fd).max(hd).max(kd).max(vd);
    }
    eprintln!("worst {worst:.3e} (tolerance {tol:.1e})");
    assert!(worst <= tol, "MolmoAct2 prompt prefix differs from the reference");
}

#[test]
fn molmoact2_language_config_reads_a_release() {
    let c = serde_json::json!({
        "image_patch_id": 154626, "image_col_id": 154627, "image_start_token_id": 154624, "low_res_image_start_token_id": 154628,
        "frame_start_token_id": 154631, "image_end_token_id": 154625, "frame_end_token_id": 154632, "image_low_res_id": 154630,
        "text_config": {"additional_vocab_size": 128, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560, "intermediate_size": 9728,
            "layer_norm_eps": 1e-6, "norm_after": false, "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
            "qk_norm_type": "qwen3", "qkv_bias": false, "rope_parameters": {"rope_theta": 5000000.0, "rope_type": "default"},
            "rope_scaling_layers": null, "use_qk_norm": true, "vocab_size": 154624}
    });
    let cfg = molmoact2::MolmoAct2TextConfig::from_json(&c).unwrap();
    assert_eq!((cfg.layers, cfg.heads, cfg.kv_heads, cfg.vocab, cfg.new_vocab), (36, 32, 8, 154_624, 128));
    assert!((cfg.theta - 5e6).abs() < 1e-6);
    let t = molmoact2::MolmoAct2ImageTokens::from_json(&c).unwrap();
    assert_eq!(t.patch, 154_626);
    assert_eq!(t.all.len(), 8);
    let mut post = c.clone();
    post["text_config"]["norm_after"] = serde_json::json!(true);
    assert!(molmoact2::MolmoAct2TextConfig::from_json(&post).is_err());
    // Rotary tables: position 0 is the identity, both halves share angles.
    let (cos, sin) = cfg.rotary_tables(2);
    assert!(cos[..128].iter().all(|&x| x == 1.0) && sin[..128].iter().all(|&x| x == 0.0));
    assert_eq!(cos[128..192], cos[192..256]);
    assert!((sin[128] - 1f32.sin()).abs() < 1e-7);
}

#[test]
fn molmoact2_prompts_follow_the_reference() {
    use molmoact2_processor::{normalize_task, pooling_indices, robot_prompt, state_bins};
    // Outputs of the reference's `_normalize_question_text`.
    assert_eq!(normalize_task("Task: Pick up the red cube.  Then put it in the bowl!"), "pick up the red cube; then put it in the bowl");
    assert_eq!(normalize_task("  \"The task is to OPEN the drawer...\" "), "open the drawer");
    assert_eq!(normalize_task("goal - wipe (the table)!?"), "wipe (the table");
    assert_eq!(normalize_task("Your task is to fold. the towel. now"), "fold; the towel; now");
    assert_eq!(normalize_task("tasks: x."), "tasks: x");
    // `_build_discrete_state_string` gives 128, 0, 255, 191, 128.
    assert_eq!(state_bins(&[0.0, -1.5, 1.0, 0.5, f32::NAN], 256), vec![128, 0, 255, 191, 128]);
    // `arange_for_pooling` of a 3x3 grid in 2x2 blocks.
    let (th, tw, idx) = pooling_indices(3, 3, 2, 2);
    assert_eq!((th, tw), (2, 2));
    assert_eq!(idx, vec![0, 1, 3, 4, 2, -1, 5, -1, 6, 7, -1, -1, 8, -1, -1, -1]);
    let p = robot_prompt("stack", &[3], "", "joint", (true, true), 2);
    assert_eq!(
        p,
        "Image 1<|image|>Image 2<|image|><|im_start|>user\nThe task is to stack. The setup is . The current state of the robot is <state_start><state_3><state_end>. The expected control mode is <control_start>joint<control_end>. Given these, what action should the robot take to complete the task?<|im_end|>\n<|im_start|>assistant\n<action_output>"
    );
}

#[test]
fn bilinear_resize_matches_torchvision_on_bytes() {
    // torchvision resize(.., BILINEAR, antialias=False) of uint8 planes.
    let up = resample::resize_bilinear_bytes(&frame8(3, 4, &SMALL), 5, 7);
    let want = [12, 79, 187, 118, 41, 71, 90, 109, 113, 119, 97, 74, 77, 80, 255, 164, 18, 64, 123, 87, 64, 106, 107, 108, 146, 173, 95, 45, 7, 69, 168, 200, 207, 100, 33];
    assert_eq!(&up.data[..35], &want);
    let down = resample::resize_bilinear_bytes(&frame8(4, 6, &WIDE), 3, 5);
    assert_eq!(&down.data[..15], &[64, 132, 69, 172, 38, 127, 115, 111, 64, 79, 79, 108, 164, 80, 24]);
    assert_eq!(resample::resize_bilinear_bytes(&frame8(4, 6, &WIDE), 4, 6).data[..24], WIDE);
}

fn frame8(h: usize, w: usize, plane: &[u8]) -> resample::Frame8 {
    resample::Frame8 { h, w, data: plane.repeat(3) }
}

const SMALL: [u8; 12] = [12, 200, 37, 90, 255, 0, 128, 64, 7, 180, 220, 33];
const WIDE: [u8; 24] = [12, 200, 37, 90, 250, 3, 255, 0, 128, 64, 1, 99, 7, 180, 220, 33, 140, 60, 90, 91, 92, 250, 0, 17];

#[test]
fn area_resize_matches_opencv() {
    // cv2.resize(.., INTER_AREA) of the same planes (identical on x86-64
    // and aarch64 builds).
    let up = resample::resize_area(&frame8(3, 4, &SMALL), 5, 7);
    let want_up = [12, 59, 200, 119, 37, 77, 90, 93, 103, 133, 100, 67, 78, 81, 255, 191, 0, 64, 128, 80, 64, 90, 97, 120, 155, 189, 80, 43, 7, 50, 180, 200, 220, 80, 33];
    assert_eq!(&up.data[..35], &want_up);
    let down = resample::resize_area(&frame8(4, 6, &WIDE), 3, 4);
    assert_eq!(&down.data[..12], &[98, 90, 118, 81, 117, 146, 56, 76, 84, 120, 142, 30]);
    let half = resample::resize_area(&frame8(2, 2, &[1, 2, 3, 5]), 1, 1);
    assert_eq!(half.data, vec![3; 3]);
}

#[test]
fn bicubic_resize_matches_torch_on_bytes() {
    // torchvision resize(.., BICUBIC, antialias=True) of uint8 planes.
    let down = resample::resize_bicubic_aa(&frame8(4, 6, &WIDE), 3, 5);
    assert_eq!(&down.data[..15], &[99, 113, 59, 157, 56, 125, 122, 115, 49, 83, 65, 123, 171, 95, 22]);
    let up = resample::resize_bicubic_aa(&frame8(3, 4, &SMALL), 5, 7);
    let want = [0, 68, 211, 133, 27, 66, 97, 101, 115, 123, 93, 64, 75, 84, 255, 173, 8, 52, 129, 88, 59, 101, 103, 105, 160, 188, 91, 31, 0, 49, 183, 236, 219, 91, 15];
    assert_eq!(&up.data[..35], &want);
}

#[test]
fn groot_image_sizes_and_instructions_follow_the_reference() {
    use groot_processor::{format_instruction, smart_resize};
    assert_eq!(smart_resize(256, 455, 32, 65_536, 16_777_216).unwrap(), (256, 448));
    assert_eq!(smart_resize(256, 340, 32, 65_536, 16_777_216).unwrap(), (256, 352));
    assert_eq!(smart_resize(100, 100, 32, 65_536, 16_777_216).unwrap(), (256, 256));
    assert_eq!(format_instruction("Pick up the RED cube, then place it in the bowl!", true), "pick up the red cube then place it in the bowl");
    assert_eq!(format_instruction("", true), "perform the task");
    assert_eq!(format_instruction("Keep it.", false), "Keep it.");
}

#[test]
fn relative_poses_compose_after_the_reference() {
    // A quarter turn about z: a step along the relative x axis moves along
    // the absolute y axis.
    let reference = [1.0, 2.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0];
    let step = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let got = groot_processor::compose_pose(&reference, &step);
    let want = [1.0, 3.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0];
    assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6), "{got:?}");
}

/// The GR00T N1.7 control loop through the release's processor against the
/// reference preprocessor, policy and postprocessor.
/// `PRAECISE_GROOT_ROLLOUT_PARITY` names a directory with `policy/` (weights
/// and processor files), `backbone/`, `fixture.json` and
/// `fixture.safetensors` (written by
/// `tests/parity/make_groot_rollout_fixture.py`). For every chunk the
/// prompt, the image patches, the state and the decoding of the reference
/// prediction are checked; then every executed action (robot units, of
/// order one in the fixture) is compared. Patches must match exactly, except
/// by one level where the fixture comes from another architecture and the
/// release uses OpenCV's area interpolation.
#[test]
#[ignore = "needs a released policy and a reference fixture"]
fn groot_rollout_matches_reference() {
    let dir = parity_dir("PRAECISE_GROOT_ROLLOUT_PARITY");
    let fx = Fixture::open(&dir);
    let (opts, tol) = parity_options();
    let tag = fx.meta["embodiment_tag"].as_str().unwrap();
    let mut p = groot_processor::GrootChunkPolicy::load(&dir.join("policy"), &dir.join("backbone"), tag, opts).unwrap();
    p.set_fp32_gemm(true);
    p.set_n_action_steps(fx.usize("n_action_steps"));
    let (ticks, cams) = (fx.usize("ticks"), fx.usize("cameras"));
    let states = fx.f32s("states");
    let sd = states.len() / ticks;
    let instruction = fx.meta["instruction"].as_str().unwrap().to_owned();
    let width = p.policy().head().config().max_action_dim;
    // OpenCV's area decimation sums with fused multiply-adds on aarch64 and
    // without on x86-64, so a fixture recorded on the other architecture may
    // differ by one level in a few pixels there; otherwise the patches match.
    let arch = |m: &str| match m.to_ascii_lowercase().as_str() {
        "arm64" | "aarch64" => "aarch64".to_owned(),
        "amd64" | "x86_64" => "x86_64".to_owned(),
        other => other.to_owned(),
    };
    let foreign = fx.meta["machine"].as_str().is_some_and(|m| arch(m) != std::env::consts::ARCH);
    let patch_tol = if foreign && p.processor().geometry().area { 1.0 / 127.5 + 1e-6 } else { 1e-6 };
    for (k, c) in fx.meta["chunks"].as_array().unwrap().iter().enumerate() {
        let t = c["tick"].as_u64().unwrap() as usize;
        let frames: Vec<Frame> = (0..cams).map(|cam| fx.frame(t, cam)).collect();
        let state = &states[t * sd..(t + 1) * sd];
        let obs = VlaObservation { cameras: &frames, state, instruction: &instruction };
        let inputs = p.inputs(&obs).unwrap();
        let grids: Vec<(usize, usize)> = inputs.images.iter().map(|i| i.grid).collect();
        let ids = p.policy().encoder().expand_placeholders(&inputs.tokens, &grids);
        let want_ids: Vec<u32> = fx.st.get(&format!("chunk.{k}.input_ids")).unwrap().bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(ids, want_ids, "chunk {k}: prompt");
        let patches: Vec<f32> = inputs.images.iter().flat_map(|i| i.patches.iter().copied()).collect();
        let want_px = fx.f32s(&format!("chunk.{k}.pixel_values"));
        assert_eq!(patches.len(), want_px.len(), "chunk {k}: patch layout");
        let dp = patches.iter().zip(&want_px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let ds = inputs.state.iter().zip(&fx.f32s(&format!("chunk.{k}.state"))).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let want_raw = fx.f32s(&format!("chunk.{k}.raw"));
        let raw = p.policy().predict_images(&inputs.tokens, &inputs.images, &inputs.state, &fx.f32s("noise")).unwrap();
        let dr = raw.iter().zip(&want_raw).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let steps = c["horizon"].as_u64().unwrap() as usize;
        let decoded = p.processor().decode(&want_raw, width, steps, state).unwrap();
        let want_dec = fx.f32s(&format!("chunk.{k}.decoded"));
        assert_eq!(decoded.len(), want_dec.len(), "chunk {k}: decoded layout");
        let dd = decoded.iter().zip(&want_dec).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let np = patches.iter().zip(&want_px).filter(|(a, b)| a != b).count();
        eprintln!("chunk {k} (tick {t}): {} tokens, patches {dp:.1e} ({np} values differ), state {ds:.1e}, prediction {dr:.3e}, decoding {dd:.1e}", ids.len());
        assert!(dp < patch_tol, "chunk {k}: patches differ by {dp}");
        assert!(ds < 1e-6, "chunk {k}: state differs by {ds}");
        assert!(dd < 1e-5, "chunk {k}: decoding differs by {dd}");
        assert!(dr <= tol, "chunk {k}: prediction differs by {dr}");
    }
    let identity = FeatureNorm { mode: NormMode::Identity, lo: Vec::new(), hi: Vec::new(), eps: 0.0 };
    let start = std::time::Instant::now();
    rollout_matches(&p, &identity, &fx, tol);
    eprintln!("wall {:.1}s", start.elapsed().as_secs_f32());
}
