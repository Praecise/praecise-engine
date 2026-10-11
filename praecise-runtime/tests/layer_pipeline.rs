//! Layer-pipeline serving across local processes: one model split into three stages
//! (this process plus two children) gives the logits of the whole model in one process.
//! Needs `PRAECISE_TEST_KV_MODEL` (a small GGUF of a splittable architecture, e.g. a
//! Qwen3 or Llama dense model); ignored by default and fails without it when run
//! with `--ignored`.
#![cfg(feature = "bundled-llama")]

mod common;

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::num::NonZeroU32;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use praecise_runtime::pipeline::driver::{DriverOptions, Pipeline};
use praecise_runtime::pipeline::stage::{self, StageOptions};
use praecise_runtime::pipeline::{split_layers, LayerRange, PipelineError, PipelinePlan, StageAuthenticator, StageSpec};
use praecise_runtime::{BatchPrompt, GenerationConfig, ModelFingerprint};
use sha2::{Digest, Sha256};

const STAGE_ENV: &str = "PRAECISE_PIPELINE_TEST_STAGE";
const PLAN_ENV: &str = "PRAECISE_PIPELINE_TEST_PLAN";
const READY: &str = "PIPELINE-STAGE-READY";
const MICRO_BATCH: usize = 8;
const TEXT_A: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step \
                      and listening to the sea below as the lamp warmed and the harbour lights came on.";
const TEXT_B: &str = "In the valley the river slowed, widened, and turned silver under a low winter sun.";

/// Test-only authenticator: signature = SHA-256(identity || message). It exercises the
/// protocol and proves nothing; a host supplies a real signer.
struct Fixture(Vec<u8>);

impl StageAuthenticator for Fixture {
    fn identity(&self) -> Vec<u8> {
        self.0.clone()
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Sha256::new().chain_update(&self.0).chain_update(message).finalize().to_vec())
    }
    fn verify(&self, identity: &[u8], message: &[u8], signature: &[u8]) -> bool {
        Sha256::new().chain_update(identity).chain_update(message).finalize().as_slice() == signature
    }
}

fn identity(i: usize) -> Vec<u8> {
    vec![0x40 + i as u8; 32]
}

fn opts() -> StageOptions {
    StageOptions { n_ctx: 512, max_rows: MICRO_BATCH as u32, n_seq_max: 2, n_threads: Some(4), connect_timeout: Duration::from_secs(10) }
}

fn encode_plan(p: &PipelinePlan) -> String {
    let stages: Vec<String> = p
        .stages
        .iter()
        .map(|s| format!("{}@{}@{}-{}", hex::encode(&s.identity), s.address, s.layers.begin, s.layers.end))
        .collect();
    format!("{}|{}|{}|{}", hex::encode(p.model_digest), p.n_layer, p.n_embd, stages.join(";"))
}

fn decode_plan(s: &str) -> PipelinePlan {
    let parts: Vec<&str> = s.split('|').collect();
    PipelinePlan {
        model_digest: hex::decode(parts[0]).unwrap().try_into().unwrap(),
        n_layer: parts[1].parse().unwrap(),
        n_embd: parts[2].parse().unwrap(),
        stages: parts[3]
            .split(';')
            .map(|st| {
                let f: Vec<&str> = st.split('@').collect();
                let (b, e) = f[2].split_once('-').unwrap();
                StageSpec {
                    identity: hex::decode(f[0]).unwrap(),
                    address: f[1].to_string(),
                    layers: LayerRange { begin: b.parse().unwrap(), end: e.parse().unwrap() },
                }
            })
            .collect(),
    }
}

fn load_stage(path: &str, r: LayerRange) -> LlamaModel {
    let params: LlamaModelParams = common::params().with_layer_stage(r.begin, r.end);
    LlamaModel::load_from_file(common::backend(), path, &params).expect("load stage")
}

/// Runs only as a child of `three_processes_match_one_process`.
#[test]
#[ignore = "spawned by three_processes_match_one_process"]
fn stage_process() {
    // Run directly rather than spawned as a stage, there is nothing to serve.
    let Ok(index) = std::env::var(STAGE_ENV) else { return };
    let plan = std::env::var(PLAN_ENV).expect("a stage is spawned with its plan");
    let path = common::model_path();
    let index: usize = index.parse().unwrap();
    let plan = decode_plan(&plan);
    let model = load_stage(&path, plan.stages[index].layers);
    let listener = TcpListener::bind(&plan.stages[index].address).expect("bind");
    println!("{READY}");
    let err = stage::serve(&listener, common::backend(), &model, &plan, index, &Fixture(identity(index)), &opts());
    panic!("stage {index} stopped: {err:?}");
}

struct Children(Vec<Child>);

impl Drop for Children {
    fn drop(&mut self) {
        for c in &mut self.0 {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn spawn_stage(index: usize, plan: &PipelinePlan) -> Child {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["stage_process", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env(STAGE_ENV, index.to_string())
        .env(PLAN_ENV, encode_plan(plan))
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn stage");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    loop {
        let line = lines.next().expect("stage exited before it was ready").unwrap();
        if line.contains(READY) {
            break;
        }
    }
    // keep draining so the child never blocks on a full pipe
    std::thread::spawn(move || for _ in lines {});
    child
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// The whole model in this process, fed the same micro-batches.
struct Reference<'m> {
    ctx: llama_cpp_2::context::LlamaContext<'m>,
}

impl<'m> Reference<'m> {
    fn new(model: &'m LlamaModel) -> Self {
        let o = opts();
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(o.n_ctx))
            .with_n_batch(o.max_rows)
            .with_n_ubatch(o.max_rows)
            .with_n_seq_max(o.n_seq_max)
            .with_kv_unified(true)
            .with_n_threads(4)
            .with_n_threads_batch(4);
        Self { ctx: model.new_context(common::backend(), params).expect("reference context") }
    }

    fn run(&mut self, seq: i32, start: usize, tokens: &[LlamaToken]) -> Vec<f32> {
        let mut last = Vec::new();
        for (c, chunk) in tokens.chunks(MICRO_BATCH).enumerate() {
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (i, t) in chunk.iter().enumerate() {
                batch.add(*t, (start + c * MICRO_BATCH + i) as i32, &[seq], i + 1 == chunk.len()).unwrap();
            }
            self.ctx.decode(&mut batch).unwrap();
            last = self.ctx.get_logits_ith(chunk.len() as i32 - 1).to_vec();
        }
        last
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: vocabulary width");
    let diff = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("{what}: max |pipeline - single| = {diff:e}");
    assert!(diff <= 1e-3, "{what}: logits differ by {diff}");
    assert_eq!(argmax(got), argmax(want), "{what}: argmax differs");
}

#[test]
#[ignore = "needs a small dense GGUF: PRAECISE_TEST_KV_MODEL"]
fn three_processes_match_one_process() {
    let path = common::model_path();
    let full = common::load(&path);
    let n_layer = full.n_layer();
    let ranges = split_layers(n_layer, &[1, 1, 1]).unwrap();
    let plan = PipelinePlan {
        model_digest: ModelFingerprint::of_file(&path).unwrap().0,
        n_layer,
        n_embd: full.n_embd() as u32,
        stages: ranges
            .iter()
            .enumerate()
            .map(|(i, r)| StageSpec {
                identity: identity(i),
                address: if i == 0 { "local".into() } else { format!("127.0.0.1:{}", free_port()) },
                layers: *r,
            })
            .collect(),
    };
    plan.validate().unwrap();

    // last stage first: a stage dials the next one when its own upstream arrives
    let mut children = Children(vec![]);
    children.0.push(spawn_stage(2, &plan));
    children.0.push(spawn_stage(1, &plan));

    let head = load_stage(&path, plan.stages[0].layers);
    let dopts = DriverOptions { stage: opts(), max_in_flight: 4, answer_timeout: Duration::from_secs(120) };

    // a driver that is not the plan's stage 0 is refused by stage 1
    let mut forged = plan.clone();
    forged.stages[0].identity = vec![0x7f; 32];
    let err = Pipeline::connect(common::backend(), &head, &forged, &Fixture(vec![0x7f; 32]), &dopts).unwrap_err();
    assert!(matches!(err, PipelineError::Authentication { stage: 1, .. } | PipelineError::StageFailed { stage: 1, .. }), "{err}");

    let mut pipe = Pipeline::connect(common::backend(), &head, &plan, &Fixture(identity(0)), &dopts).expect("connect");
    let mut reference = Reference::new(&full);

    // micro-batched prefill, then greedy decode
    let a: Vec<LlamaToken> = full.str_to_token(TEXT_A, AddBos::Always).unwrap();
    let got = pipe.prefill(0, 0, &a, MICRO_BATCH).expect("prefill");
    let want = reference.run(0, 0, &a);
    assert_close("prefill", &got, &want);
    let mut next = LlamaToken(argmax(&want) as i32);
    for step in 0..6 {
        let pos = (a.len() + step) as i32;
        let got = pipe.step(0, pos, next).expect("step");
        let want = reference.run(0, pos as usize, &[next]);
        assert_close(&format!("decode {step}"), &got, &want);
        next = LlamaToken(argmax(&want) as i32);
    }

    // two sequences in flight at once, interleaved micro-batches
    let b: Vec<LlamaToken> = full.str_to_token(TEXT_B, AddBos::Always).unwrap();
    pipe.truncate(0, -1).unwrap();
    reference.ctx.clear_kv_cache_seq(Some(0), None, None).unwrap();
    let mut tickets = vec![];
    let longest = a.len().max(b.len());
    for start in (0..longest).step_by(MICRO_BATCH) {
        for (seq, toks) in [(0, &a), (1, &b)] {
            if start < toks.len() {
                let chunk = &toks[start..(start + MICRO_BATCH).min(toks.len())];
                let mut outputs = vec![false; chunk.len()];
                let is_last = start + chunk.len() == toks.len();
                *outputs.last_mut().unwrap() = is_last;
                tickets.push((seq, is_last, pipe.submit(seq, start as i32, chunk, &outputs).unwrap()));
            }
        }
    }
    let want_a = reference.run(0, 0, &a);
    let want_b = reference.run(1, 0, &b);
    for (seq, is_last, t) in tickets {
        let rows = pipe.wait(t).unwrap();
        if is_last {
            assert_close(&format!("interleaved seq {seq}"), &rows[0], if seq == 0 { &want_a } else { &want_b });
        } else {
            assert!(rows.is_empty());
        }
    }

    // a whole greedy generation through the pipeline equals one in a single process
    let config = GenerationConfig { temperature: 0.0, repeat_penalty: 1.0, max_tokens: 12, ..Default::default() };
    let result = pipe
        .generate(&BatchPrompt::Raw(TEXT_B.to_string()), &config, false, None, None)
        .expect("generate");
    reference.ctx.clear_kv_cache_seq(Some(0), None, None).unwrap();
    let mut logits = reference.run(0, 0, &b);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut want_text = String::new();
    for i in 0..config.max_tokens as usize {
        let t = LlamaToken(argmax(&logits) as i32);
        if full.is_eog_token(t) {
            break;
        }
        want_text.push_str(&full.token_to_piece(t, &mut decoder, false, None).unwrap());
        logits = reference.run(0, b.len() + i, &[t]);
    }
    eprintln!("generated: {:?}", result.text);
    assert!(result.output_tokens > 0);
    assert_eq!(result.text, want_text);

    // the last stage dies: the next request fails naming it, and so does every later one
    let mut last = children.0.remove(0);
    last.kill().unwrap();
    last.wait().unwrap();
    let err = pipe.step(1, b.len() as i32, next).unwrap_err();
    assert!(matches!(err, PipelineError::StageFailed { stage: 2, .. }), "{err}");
    let err = pipe.step(1, b.len() as i32, next).unwrap_err();
    assert!(matches!(err, PipelineError::StageFailed { stage: 2, .. }), "{err}");
}

/// One model split in two across machines: this process drives the first half,
/// a `stage_process` on another host serves the second, started with the plan
/// this test prints. Needs `PRAECISE_PIPELINE_TEST_REMOTE` (the remote stage's
/// `host:port`) besides the model; `PRAECISE_PIPELINE_TEST_TOKENS` sets the
/// generation length (default 64) and `PRAECISE_PIPELINE_TEST_TOL` the logit
/// tolerance (default 1e-3), since two different accelerators may round apart.
#[test]
#[ignore = "needs a second host: PRAECISE_PIPELINE_TEST_REMOTE"]
fn remote_stage_matches_one_process() {
    let remote =
        std::env::var("PRAECISE_PIPELINE_TEST_REMOTE").expect("PRAECISE_PIPELINE_TEST_REMOTE");
    let tokens: usize = std::env::var("PRAECISE_PIPELINE_TEST_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let tol: f32 = std::env::var("PRAECISE_PIPELINE_TEST_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    let path = common::model_path();
    let full = LlamaModel::load_from_file(common::backend(), &path, &common::params())
        .expect("load model");
    let n_layer = full.n_layer();
    let ranges = split_layers(n_layer, &[1, 1]).unwrap();
    let plan = PipelinePlan {
        model_digest: ModelFingerprint::of_file(&path).unwrap().0,
        n_layer,
        n_embd: full.n_embd() as u32,
        stages: vec![
            StageSpec {
                identity: identity(0),
                address: "local".into(),
                layers: ranges[0],
            },
            StageSpec {
                identity: identity(1),
                address: remote.clone(),
                layers: ranges[1],
            },
        ],
    };
    plan.validate().unwrap();
    println!("PLAN={}", encode_plan(&plan));
    eprintln!("start the remote stage with {STAGE_ENV}=1 and {PLAN_ENV} set to the plan above");

    let head = load_stage(&path, plan.stages[0].layers);
    let dopts = DriverOptions {
        stage: opts(),
        max_in_flight: 4,
        answer_timeout: Duration::from_secs(120),
    };
    let started = std::time::Instant::now();
    let mut pipe = loop {
        match Pipeline::connect(
            common::backend(),
            &head,
            &plan,
            &Fixture(identity(0)),
            &dopts,
        ) {
            Ok(p) => break p,
            Err(e) if started.elapsed() < Duration::from_secs(900) => {
                eprintln!("waiting for the remote stage: {e}");
                std::thread::sleep(Duration::from_secs(3));
            }
            Err(e) => panic!("remote stage never answered: {e}"),
        }
    };
    let mut reference = Reference::new(&full);
    let tol_close = |what: &str, got: &[f32], want: &[f32]| -> f32 {
        assert_eq!(got.len(), want.len(), "{what}: vocabulary width");
        let diff = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff <= tol, "{what}: logits differ by {diff}");
        assert_eq!(argmax(got), argmax(want), "{what}: argmax differs");
        diff
    };

    let a: Vec<LlamaToken> = full.str_to_token(TEXT_A, AddBos::Always).unwrap();
    let want = reference.run(0, 0, &a);
    let mut worst = tol_close(
        "prefill",
        &pipe.prefill(0, 0, &a, MICRO_BATCH).expect("prefill"),
        &want,
    );
    let mut next = LlamaToken(argmax(&want) as i32);
    for step in 0..tokens {
        let pos = (a.len() + step) as i32;
        let got = pipe.step(0, pos, next).expect("step");
        let want = reference.run(0, pos as usize, &[next]);
        worst = worst.max(tol_close(&format!("decode {step}"), &got, &want));
        next = LlamaToken(argmax(&want) as i32);
    }
    eprintln!(
        "prefill + {tokens} decode steps: greedy tokens equal, max |pipeline - single| = {worst:e}"
    );

    // whole generations, timed: through the pipeline, and in one process
    pipe.truncate(0, -1).unwrap();
    let config = GenerationConfig {
        temperature: 0.0,
        repeat_penalty: 1.0,
        max_tokens: tokens as u32,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let result = pipe
        .generate(
            &BatchPrompt::Raw(TEXT_B.to_string()),
            &config,
            false,
            None,
            None,
        )
        .expect("generate");
    let pipe_secs = t.elapsed().as_secs_f64();
    reference
        .ctx
        .clear_kv_cache_seq(Some(0), None, None)
        .unwrap();
    let b: Vec<LlamaToken> = full.str_to_token(TEXT_B, AddBos::Always).unwrap();
    let t = std::time::Instant::now();
    let mut logits = reference.run(0, 0, &b);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut want_text = String::new();
    let mut n = 0;
    for i in 0..tokens {
        let tok = LlamaToken(argmax(&logits) as i32);
        if full.is_eog_token(tok) {
            break;
        }
        want_text.push_str(&full.token_to_piece(tok, &mut decoder, false, None).unwrap());
        logits = reference.run(0, b.len() + i, &[tok]);
        n += 1;
    }
    let single_secs = t.elapsed().as_secs_f64();
    eprintln!(
        "generate {} tokens: pipeline {:.1} tok/s (engine reports {:.1}), single process {:.1} tok/s",
        result.output_tokens,
        f64::from(result.output_tokens) / pipe_secs,
        result.tokens_per_second,
        n as f64 / single_secs
    );
    assert_eq!(result.text, want_text);
}
