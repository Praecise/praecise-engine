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
