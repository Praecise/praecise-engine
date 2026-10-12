//! An omni model's talker against the source checkpoint's own generate().
//!
//! `tests/parity/omni_talker/expected.json` (written by
//! `tests/parity/make_omni_talker_fixtures.py`, greedy everywhere, no
//! repetition penalty) holds, per case, the user positions the talker reads,
//! the thinker's hidden rows at audio positions (`hidden.f32`), the thinker's
//! reply, the voice and the reference waveform (`pcm.f32`). The talker is
//! replayed here with the same conditioning, fed the reply one token at a time
//! as the thinker would produce it, and must give the same waveform.
//!
//! The weights are not committed: the test is ignored by default and, run with
//! `--ignored`, needs `PRAECISE_OMNI_TALKER_GGUF` (the talker backbone converted
//! with `--talker` from the checkpoint revision in `expected.json`) and
//! `PRAECISE_OMNI_TALKER_MMPROJ` (`--talker --mmproj`), and fails without them.
//! `PRAECISE_OMNI_TALKER_FIXTURES` points at another fixture directory of the
//! same layout (a full-size checkpoint's reference, written by
//! `tests/parity/make_omni_talker_real_reference.py`); `PRAECISE_OMNI_TALKER_GPU=1`
//! offloads the talker to the GPU; `PRAECISE_OMNI_TALKER_TOLERANCE` sets the
//! bound (default 1e-3, the CPU float32 bound); `PRAECISE_OMNI_TALKER_DUMP` names a
//! directory the engine's waveform is written to.
#![cfg(feature = "mtmd")]

use std::ops::ControlFlow;
use std::path::Path;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
use llama_cpp_2::token::LlamaToken;
use praecise_runtime::speech::{speak_reply, ReplyRequest, ReplyText};

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn tokens(v: &serde_json::Value) -> Vec<LlamaToken> {
    v.as_array()
        .expect("token list")
        .iter()
        .map(|t| LlamaToken(i32::try_from(t.as_i64().expect("token")).expect("token")))
        .collect()
}

#[test]
#[ignore = "needs the talker weights: PRAECISE_OMNI_TALKER_GGUF and PRAECISE_OMNI_TALKER_MMPROJ"]
fn talker_waveform_matches_the_source_checkpoint() {
    let gguf = std::env::var("PRAECISE_OMNI_TALKER_GGUF").expect("PRAECISE_OMNI_TALKER_GGUF names the talker GGUF");
    let mmproj =
        std::env::var("PRAECISE_OMNI_TALKER_MMPROJ").expect("PRAECISE_OMNI_TALKER_MMPROJ names its projector");
    let default_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/parity/omni_talker"));
    let fixtures = std::env::var("PRAECISE_OMNI_TALKER_FIXTURES").ok();
    let dir = fixtures.as_deref().map_or(default_dir, Path::new);
    let gpu = std::env::var("PRAECISE_OMNI_TALKER_GPU").as_deref() == Ok("1");
    // a GPU run is compared with a CPU float32 reference: its own bound, named when it is set
    let tolerance: f32 = std::env::var("PRAECISE_OMNI_TALKER_TOLERANCE")
        .ok()
        .map_or(1e-3, |t| t.parse().expect("PRAECISE_OMNI_TALKER_TOLERANCE is a number"));
    let expected = dir.join("expected.json");
    let exp: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&expected).unwrap_or_else(|e| panic!("{}: {e}", expected.display())),
    )
    .expect("json");

    let mut backend = LlamaBackend::init().expect("backend");
    // `PRAECISE_TEST_LOGS=1` keeps the backend's log, for a load or decode that fails
    if std::env::var("PRAECISE_TEST_LOGS").as_deref() != Ok("1") {
        backend.void_logs();
    }
    let model_params = if gpu {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };
    let model = LlamaModel::load_from_file(&backend, &gguf, &model_params).expect("load");
    // the reference attends in float32; flash attention on GPU would use half-precision keys
    let params = MtmdContextParams {
        use_gpu: gpu,
        ..MtmdContextParams::default()
    }
    .with_flash_attention(false);
    // SAFETY: the projector is dropped before the model, at the end of this function.
    let mtmd = unsafe { MtmdContext::init_from_file(&mmproj, &model, &params) }.expect("projector");
    // the reference computes in float32; so do the projector's products
    mtmd.set_fp32_gemm(true);

    for case in exp["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let user = tokens(&case["user_tokens"]);
        let reply_tokens = tokens(&case["reply_tokens"]);
        let width = usize::try_from(case["hidden_width"].as_u64().expect("hidden_width")).expect("width");
        let hidden = if case["media_rows"].as_u64().expect("media_rows") > 0 {
            f32s(&dir.join(name).join("hidden.f32"))
        } else {
            Vec::new()
        };
        let reference = f32s(&dir.join(name).join("pcm.f32"));
        let frames = usize::try_from(case["frames"].as_u64().expect("frames")).expect("frames");

        // the waveform decoder alone, on the reference's own codes, when the fixture has them
        let codes_file = dir.join(name).join("codes.i32");
        if codes_file.exists() {
            let codes: Vec<i32> = std::fs::read(&codes_file)
                .expect("codes")
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let per_frame = codes.len() / frames;
            let (_, pcm) = mtmd.decode_codes(&codes, per_frame, 3).expect("decode codes");
            assert_eq!(pcm.len(), reference.len(), "{name}: decoder sample count");
            let worst = pcm.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("{name}: decoder alone on the reference codes, max |diff| {worst:e}");
            assert!(worst < tolerance, "{name}: the waveform decoder differs from the reference by {worst}");
        }

        // the reference keeps attention in float32; so does the backbone here
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048))
            .with_embeddings(true)
            .with_type_k(KvCacheType::F32)
            .with_type_v(KvCacheType::F32)
            .with_flash_attention(false);
        let mut ctx = model.new_context(&backend, ctx_params).expect("context");
        ctx.set_fp32_gemm(true);

        // the thinker's reply arrives one token at a time
        let mut next = 0usize;
        let mut chunks = 0usize;
        let speech = speak_reply(
            &mut ctx,
            &mtmd,
            &ReplyRequest {
                user_tokens: &user,
                hidden: &hidden,
                hidden_width: width,
                voice: case["voice"].as_str().expect("voice"),
                max_frames: frames,
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
                projector_top_p: 1.0,
                repetition_penalty: 1.0,
                seed: 0,
                chunk_frames: 3,
            },
            || {
                let t = reply_tokens.get(next).copied();
                next += 1;
                Ok(ReplyText {
                    tokens: t.into_iter().collect(),
                    done: next >= reply_tokens.len(),
                })
            },
            |_, _| {
                chunks += 1;
                ControlFlow::Continue(())
            },
        )
        .expect("speak");

        if let Ok(out) = std::env::var("PRAECISE_OMNI_TALKER_DUMP") {
            // the engine's waveform, little-endian f32, beside the reference for a closer look
            let bytes: Vec<u8> = speech.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(Path::new(&out).join(format!("{name}.f32")), bytes).expect("dump");
        }
        let agree = speech
            .samples
            .iter()
            .zip(&reference)
            .take_while(|(a, b)| (*a - *b).abs() < tolerance)
            .count();
        eprintln!("{name}: first {agree} samples agree within {tolerance:e}");
        assert!(chunks > 1, "{name}: the waveform must arrive in more than one chunk");
        assert_eq!(speech.samples.len(), reference.len(), "{name}: sample count");
        let max_diff = speech
            .samples
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{name}: {} frames, {} samples, max |diff| {max_diff:e}", speech.frames, speech.samples.len());
        assert!(max_diff < tolerance, "{name}: waveform differs from the reference by {max_diff}");
    }
}
