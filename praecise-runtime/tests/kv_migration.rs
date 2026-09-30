//! KV-cache export and import between two engine instances of one model.
//! Needs `PRAECISE_TEST_KV_MODEL` (a small dense GGUF); skips without it.
#![cfg(feature = "bundled-llama")]

mod common;

use std::sync::OnceLock;

use llama_cpp_2::model::AddBos;
use llama_cpp_2::token::LlamaToken;
use praecise_runtime::{Error, KvBlob, ModelFingerprint, SequenceExport, SequenceImport};

const SEQ: i32 = 0;
const SCRATCH: i32 = 1;
const TEXT: &str = "The lighthouse keeper climbed the spiral stairs each evening, counting every step \
                    and listening to the sea below as the lamp warmed and the harbour lights came on.";

fn fingerprint(path: &str) -> ModelFingerprint {
    static F: OnceLock<ModelFingerprint> = OnceLock::new();
    *F.get_or_init(|| ModelFingerprint::of_file(path).expect("fingerprint"))
}

#[test]
fn export_import_resumes_with_identical_logits_and_deltas_carry_only_new_positions() {
    let Some(path) = common::model_path() else { return };
    let fp = fingerprint(&path);

    // Source engine: prefill a prefix, export it, append more, export the delta.
    let src_model = common::load(&path);
    let tokens: Vec<LlamaToken> = src_model.str_to_token(TEXT, AddBos::Always).expect("tokenize");
    assert!(tokens.len() >= 26, "prompt too short: {}", tokens.len());
    let (first, second, next) = (&tokens[..16], &tokens[16..24], tokens[24]);

    let mut src = common::context(&src_model);
    let mut exporter = SequenceExport::new(fp, SEQ, SCRATCH);
    common::decode(&mut src, SEQ, 0, first);
    let blob1 = exporter.export(&mut src, first).expect("export base");
    common::decode(&mut src, SEQ, 16, second);
    let blob2 = exporter.export(&mut src, &tokens[..24]).expect("export delta");

    let (b1, b2) = (KvBlob::decode(&blob1).unwrap(), KvBlob::decode(&blob2).unwrap());
    assert_eq!((b1.base_pos, b1.tokens.len()), (0, 16));
    assert_eq!((b2.base_pos, b2.tokens.len()), (16, 8), "delta must carry only the appended positions");
    assert_eq!(b2.tokens, second.iter().map(|t| t.0).collect::<Vec<_>>());
    // Cell data scales with positions carried: the 8-position delta is about
    // half the 16-position base, not a re-export of all 24.
    let per_pos = b1.payload.len() as f64 / 16.0;
    assert!(
        (b2.payload.len() as f64) < per_pos * 10.0,
        "delta payload {} vs base {} bytes",
        b2.payload.len(),
        b1.payload.len()
    );
    // Nothing new since the last export: an empty delta.
    let blob3 = exporter.export(&mut src, &tokens[..24]).expect("empty delta");
    assert!(KvBlob::decode(&blob3).unwrap().tokens.is_empty());
    assert_eq!(src.kv_cache_seq_pos_max(SCRATCH), -1, "scratch left clean");

    let src_logits = common::decode(&mut src, SEQ, 24, &[next]);

    // Destination engine: a separate model instance and context, no prefill.
    let dst_model = common::load(&path);
    let mut dst = common::context(&dst_model);
    let mut importer = SequenceImport::new(fp, SEQ, SCRATCH);

    // A delta before its base is refused and leaves the sequence empty.
    assert!(matches!(importer.apply(&mut dst, &blob2), Err(Error::KvSequence(_))));
    assert_eq!(dst.kv_cache_seq_pos_max(SEQ), -1);
    // A corrupted blob is refused.
    let mut bad = blob1.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x40;
    assert!(matches!(importer.apply(&mut dst, &bad), Err(Error::KvBlob(_))));
    assert_eq!(dst.kv_cache_seq_pos_max(SEQ), -1);

    importer.apply(&mut dst, &blob1).expect("import base");
    importer.apply(&mut dst, &blob2).expect("import delta");
    importer.apply(&mut dst, &blob3).expect("import empty delta");
    assert_eq!(importer.next_pos(), 24);
    assert_eq!(dst.kv_cache_seq_pos_max(SEQ), 23);
    assert_eq!(dst.kv_cache_seq_pos_max(SCRATCH), -1);

    let dst_logits = common::decode(&mut dst, SEQ, 24, &[next]);
    assert_eq!(src_logits.len(), dst_logits.len());
    let max_diff = src_logits.iter().zip(&dst_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    eprintln!(
        "next-token logits: max |diff| = {max_diff:e}, bitwise identical = {}, base {} B, delta {} B",
        src_logits == dst_logits,
        blob1.len(),
        blob2.len()
    );
    assert!(max_diff <= 1e-5, "imported sequence diverged: max |diff| {max_diff}");
    assert_eq!(argmax(&src_logits), argmax(&dst_logits));

    // Without the import, the same decode sees no prefix and differs.
    let mut cold = common::context(&dst_model);
    let cold_logits = common::decode(&mut cold, SEQ, 0, &[next]);
    assert!(cold_logits.iter().zip(&src_logits).any(|(a, b)| (a - b).abs() > 1e-3));
}

#[test]
fn a_blob_from_other_weights_is_refused() {
    let Some(path) = common::model_path() else { return };
    let fp = fingerprint(&path);
    let model = common::load(&path);
    let tokens = model.str_to_token(TEXT, AddBos::Always).expect("tokenize");

    let mut src = common::context(&model);
    common::decode(&mut src, SEQ, 0, &tokens[..8]);
    let blob = SequenceExport::new(fp, SEQ, SCRATCH).export(&mut src, &tokens[..8]).expect("export");

    let mut other = fp.0;
    other[0] ^= 0xff;
    let mut dst = common::context(&model);
    let mut importer = SequenceImport::new(ModelFingerprint::from_digest(other), SEQ, SCRATCH);
    assert!(matches!(importer.apply(&mut dst, &blob), Err(Error::ModelMismatch { .. })));
    assert_eq!(dst.kv_cache_seq_pos_max(SEQ), -1, "refused import must not touch the sequence");
}

#[test]
fn a_context_without_a_unified_buffer_is_refused() {
    let Some(path) = common::model_path() else { return };
    let fp = fingerprint(&path);
    let model = common::load(&path);
    let tokens = model.str_to_token(TEXT, AddBos::Always).expect("tokenize");
    let params = llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(256))
        .with_n_seq_max(2)
        .with_kv_unified(false)
        .with_n_threads(4);
    let mut ctx = model.new_context(common::backend(), params).expect("context");
    common::decode(&mut ctx, SEQ, 0, &tokens[..4]);
    let err = SequenceExport::new(fp, SEQ, SCRATCH).export(&mut ctx, &tokens[..4]).unwrap_err();
    assert!(matches!(err, Error::KvSequence(_)), "{err}");
}

/// Models to run the multi-slot test on: the dense one, and a hybrid
/// (attention plus recurrent layers) and a purely recurrent one when given.
fn slot_models() -> Vec<(&'static str, String)> {
    ["PRAECISE_TEST_KV_MODEL", "PRAECISE_TEST_HYBRID_MODEL", "PRAECISE_TEST_RECURRENT_MODEL"]
        .into_iter()
        .filter_map(|var| std::env::var(var).ok().map(|p| (var, p)))
        .collect()
}

/// Two decoding sequences in one unified buffer, plus a scratch id.
fn slots_context(model: &llama_cpp_2::model::LlamaModel) -> llama_cpp_2::context::LlamaContext<'_> {
    let params = llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(512))
        .with_n_batch(512)
        .with_n_seq_max(3)
        .with_kv_unified(true)
        .with_n_threads(4)
        .with_n_threads_batch(4);
    model.new_context(common::backend(), params).expect("context")
}

/// Decode one row per sequence in a single batch; each row's logits index is
/// its position in `rows`.
fn decode_rows(ctx: &mut llama_cpp_2::context::LlamaContext<'_>, rows: &[(i32, usize, &[LlamaToken])]) {
    let n: usize = rows.iter().map(|r| r.2.len()).sum();
    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(n, 2);
    for (seq, start, toks) in rows {
        for (i, t) in toks.iter().enumerate() {
            batch.add(*t, (start + i) as i32, &[*seq], i + 1 == toks.len()).expect("batch add");
        }
    }
    ctx.decode(&mut batch).expect("decode");
}

#[test]
fn a_batch_slot_moves_to_another_slot_of_another_context_with_identical_logits() {
    const SRC_SLOT: i32 = 1;
    const DST_SLOT: i32 = 0;
    const SCRATCH3: i32 = 2;
    let models = slot_models();
    if models.is_empty() {
        eprintln!("no model given; skipping");
        return;
    }
    for (var, path) in models {
        let fp = ModelFingerprint::of_file(&path).expect("fingerprint");
        let model = common::load(&path);
        let tokens: Vec<LlamaToken> = model.str_to_token(TEXT, AddBos::Always).expect("tokenize");
        let other: Vec<LlamaToken> = model.str_to_token("A second request shares the batch.", AddBos::Always).expect("tokenize");
        assert!(tokens.len() >= 26 && other.len() >= 4);

        // Both slots prefill together, then decode together, as a batch
        // engine's step does; slot 1 is exported between steps.
        let mut src = slots_context(&model);
        decode_rows(&mut src, &[(0, 0, &other[..4]), (SRC_SLOT, 0, &tokens[..16])]);
        let mut exporter = SequenceExport::new(fp, SRC_SLOT, SCRATCH3);
        let blob1 = exporter.export(&mut src, &tokens[..16]).expect("first export");
        for k in 16..24 {
            decode_rows(&mut src, &[(0, 4 + k - 16, &other[..1]), (SRC_SLOT, k, &tokens[k..=k])]);
        }
        let blob2 = exporter.export(&mut src, &tokens[..24]).expect("delta export");
        let (b1, b2) = (KvBlob::decode(&blob1).unwrap(), KvBlob::decode(&blob2).unwrap());
        assert_eq!((b1.base_pos, b1.tokens.len()), (0, 16), "{var}");
        assert_eq!((b2.base_pos, b2.tokens.len()), (16, 8), "{var}: the delta carries only the new positions");
        assert_eq!(src.kv_cache_seq_pos_max(SCRATCH3), -1, "{var}: scratch left clean");
        assert_eq!(src.kv_cache_seq_pos_max(0), 11, "{var}: the other slot is untouched");

        let next = tokens[24];
        let src_logits = common::decode(&mut src, SRC_SLOT, 24, &[next]);

        // Destination: another slot id, with its other slot busy.
        let mut dst = slots_context(&model);
        common::decode(&mut dst, 1, 0, &other[..6]);
        let mut importer = SequenceImport::new(fp, DST_SLOT, SCRATCH3);
        importer.apply(&mut dst, &blob1).expect("apply first");
        importer.apply(&mut dst, &blob2).expect("apply delta");
        assert_eq!(dst.kv_cache_seq_pos_max(DST_SLOT), 23, "{var}");
        assert_eq!(dst.kv_cache_seq_pos_max(SCRATCH3), -1, "{var}");
        assert_eq!(dst.kv_cache_seq_pos_max(1), 5, "{var}: the destination's other slot is untouched");
        let dst_logits = common::decode(&mut dst, DST_SLOT, 24, &[next]);

        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let max_diff = src_logits.iter().zip(&dst_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!(
            "{var}: recurrent={} hybrid={} max |diff| = {max_diff:e}, bitwise identical = {}, base {} B, delta {} B",
            model.is_recurrent(),
            model.is_hybrid(),
            src_logits == dst_logits,
            blob1.len(),
            blob2.len()
        );
        assert!(max_diff <= 1e-5, "{var}: moved sequence diverged: max |diff| {max_diff}");
        assert_eq!(argmax(&src_logits), argmax(&dst_logits), "{var}");

        // Without the import the destination knows nothing of the prefix.
        let mut cold = slots_context(&model);
        let cold_logits = common::decode(&mut cold, DST_SLOT, 0, &[next]);
        assert!(cold_logits.iter().zip(&src_logits).any(|(a, b)| (a - b).abs() > 1e-3), "{var}");
    }
}
