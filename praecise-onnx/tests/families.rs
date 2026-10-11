//! One real, small model per ONNX family, run end to end on the CPU.
//!
//! Each test names the model files it needs through environment variables and
//! is ignored by default, so a plain `cargo test` needs no weights. With the
//! files present, run them with `cargo test -p praecise-onnx --test families
//! -- --ignored`.

use std::io::Cursor;
use std::path::PathBuf;

use image::{ImageFormat, Rgb, RgbImage};
use praecise_onnx::audio::{MoonshineTranscriber, TranscribeConfig, Transcriber};
use praecise_onnx::detection::{Detector, DetrFamily, GenericDetrDetector};
use praecise_onnx::segmentation::{
    GenericSamSegmenter, PointPrompt, SamFamily, SegmentPrompt, Segmenter,
};
use praecise_onnx::session::{Gpus, OnnxConfig, OnnxSettings, install};
use praecise_onnx::text_embedding::{
    EmbedRole, EmbeddingTemplate, GenericTextEncoder, TextEmbedConfig, TextEncoder,
    TextEncoderFamily,
};
use praecise_onnx::vision::{
    GenericImageEncoder, ImageEmbedConfig, ImageEncoder, ImageNormalization,
};

fn cpu() {
    install(
        OnnxSettings::resolve(&OnnxConfig::default(), 8),
        Gpus::default(),
    );
}

fn file(var: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} is not set")));
    assert!(path.exists(), "{var}: {} does not exist", path.display());
    path
}

/// A 512x512 PNG: black, with a white square from (128, 128) to (384, 384).
fn square_png() -> Vec<u8> {
    let img = RgbImage::from_fn(512, 512, |x, y| {
        if (128..384).contains(&x) && (128..384).contains(&y) {
            Rgb([255, 255, 255])
        } else {
            Rgb([0, 0, 0])
        }
    });
    png(&img)
}

/// A 512x512 PNG of horizontal colour bands.
fn bands_png() -> Vec<u8> {
    let img = RgbImage::from_fn(512, 512, |_, y| match y / 128 {
        0 => Rgb([200, 30, 30]),
        1 => Rgb([30, 200, 30]),
        2 => Rgb([30, 30, 200]),
        _ => Rgb([220, 220, 40]),
    });
    png(&img)
}

fn png(img: &RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// RF-DETR nano (`PRAECISE_DETR_ONNX`, 384 px, 90 COCO classes): every
/// detection is a normalized box inside the image with a probability score.
#[test]
#[ignore = "needs RF-DETR nano: PRAECISE_DETR_ONNX"]
fn detection_rf_detr_returns_well_formed_boxes() {
    cpu();
    let det =
        GenericDetrDetector::from_onnx(file("PRAECISE_DETR_ONNX"), DetrFamily::RfDetr, 384, 90)
            .unwrap();
    assert_eq!(det.input_size(), 384);
    assert_eq!(det.num_classes(), 90);
    let out = det.detect(&square_png(), 0.0).unwrap();
    assert!(
        !out.detections.is_empty(),
        "a zero threshold keeps every query"
    );
    for d in &out.detections {
        assert!((0.0..=1.0).contains(&d.score), "{d:?}");
        assert!(d.label_id < 90, "{d:?}");
        assert!(d.bbox[0] <= d.bbox[2] && d.bbox[1] <= d.bbox[3], "{d:?}");
        assert!(d.bbox.iter().all(|v| (-1.0..=513.0).contains(v)), "{d:?}");
    }
    let strict = det.detect(&square_png(), 0.99).unwrap();
    assert!(strict.detections.len() <= out.detections.len());
}

/// EdgeSAM (`PRAECISE_SAM_ENCODER`, `PRAECISE_SAM_DECODER`): a point on a
/// white square on black segments the square.
#[test]
#[ignore = "needs EdgeSAM: PRAECISE_SAM_ENCODER, PRAECISE_SAM_DECODER"]
fn segmentation_edgesam_segments_the_prompted_square() {
    cpu();
    let sam = GenericSamSegmenter::from_onnx(
        file("PRAECISE_SAM_ENCODER"),
        file("PRAECISE_SAM_DECODER"),
        SamFamily::Sam1,
        1024,
    )
    .unwrap();
    let prompt = SegmentPrompt::Point(PointPrompt {
        x: 256.0,
        y: 256.0,
        is_foreground: true,
    });
    let out = sam.segment(&square_png(), &[prompt]).unwrap();
    let best = out
        .masks
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("at least one mask");
    assert_eq!((best.width, best.height), (512, 512));
    let (mut inter, mut union) = (0u32, 0u32);
    for y in 0..512u32 {
        for x in 0..512u32 {
            let on = best.mask[(y * 512 + x) as usize] > 0;
            let truth = (128..384).contains(&x) && (128..384).contains(&y);
            inter += u32::from(on && truth);
            union += u32::from(on || truth);
        }
    }
    let iou = inter as f32 / union as f32;
    assert!(iou > 0.8, "mask IoU with the square is {iou}");
}

/// DINOv3 ViT-S/16 (`PRAECISE_IMAGE_ENCODER_ONNX`, 384-dim): the same picture
/// embeds identically, a different one does not, and normalized vectors have
/// unit length.
#[test]
#[ignore = "needs DINOv3 ViT-S/16: PRAECISE_IMAGE_ENCODER_ONNX"]
fn vision_dinov3_embeds_consistently() {
    cpu();
    let enc = GenericImageEncoder::from_onnx(
        file("PRAECISE_IMAGE_ENCODER_ONNX"),
        224,
        384,
        ImageNormalization::IMAGENET,
    )
    .unwrap();
    let cfg = ImageEmbedConfig { normalize: true };
    let a = enc.embed(&square_png(), &cfg).unwrap();
    let a2 = enc.embed(&square_png(), &cfg).unwrap();
    let b = enc.embed(&bands_png(), &cfg).unwrap();
    assert_eq!(a.dim, 384);
    assert_eq!(a.embedding.len(), 384);
    let norm = a.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    assert!(cosine(&a.embedding, &a2.embedding) > 0.9999);
    assert!(cosine(&a.embedding, &b.embedding) < 0.99);
}

/// ModernBERT-embed base (`PRAECISE_TEXT_ENCODER_ONNX`,
/// `PRAECISE_TEXT_TOKENIZER`, 768-dim, Nomic prefixes): a query is closer to
/// the document that answers it than to an unrelated one, and Matryoshka
/// truncation keeps unit length.
#[test]
#[ignore = "needs ModernBERT-embed base: PRAECISE_TEXT_ENCODER_ONNX, PRAECISE_TEXT_TOKENIZER"]
fn text_embedding_modernbert_ranks_the_relevant_document_first() {
    cpu();
    let enc = GenericTextEncoder::from_onnx(
        file("PRAECISE_TEXT_ENCODER_ONNX"),
        file("PRAECISE_TEXT_TOKENIZER"),
        TextEncoderFamily::Mean,
        EmbeddingTemplate::NomicPrefix,
        768,
        8192,
    )
    .unwrap();
    let query = TextEmbedConfig {
        normalize: true,
        role: EmbedRole::Query { task: None },
        ..Default::default()
    };
    let doc = TextEmbedConfig {
        normalize: true,
        ..Default::default()
    };
    let q = enc
        .embed(&["How do plants turn sunlight into energy?".into()], &query)
        .unwrap();
    let d = enc
        .embed(
            &[
                "Photosynthesis converts light energy into chemical energy stored in glucose."
                    .into(),
                "The stock market closed higher on Friday after strong earnings reports.".into(),
            ],
            &doc,
        )
        .unwrap();
    assert_eq!(q.dim, 768);
    let relevant = cosine(&q.embeddings[0], &d.embeddings[0]);
    let unrelated = cosine(&q.embeddings[0], &d.embeddings[1]);
    assert!(relevant > unrelated + 0.05, "{relevant} vs {unrelated}");

    let small = TextEmbedConfig {
        requested_dim: Some(256),
        normalize: true,
        ..Default::default()
    };
    let t = enc.embed(&["truncated".into()], &small).unwrap();
    assert_eq!(t.embeddings[0].len(), 256);
    let norm = t.embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
}

/// Moonshine tiny (`PRAECISE_MOONSHINE_DIR` holding
/// `onnx/encoder_model.onnx`, `onnx/decoder_model_merged.onnx` and
/// `tokenizer.json`): a WAV decodes, runs through encoder and decoder, and
/// reports its duration; silence yields at most a few words.
#[test]
#[ignore = "needs Moonshine tiny: PRAECISE_MOONSHINE_DIR"]
fn audio_moonshine_transcribes_a_wav() {
    cpu();
    let dir = file("PRAECISE_MOONSHINE_DIR");
    let asr = MoonshineTranscriber::from_onnx(
        dir.join("onnx/encoder_model.onnx"),
        dir.join("onnx/decoder_model_merged.onnx"),
        dir.join("tokenizer.json"),
        30,
    )
    .unwrap();
    let mut wav = Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::new(&mut wav, spec).unwrap();
        for _ in 0..32_000 {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
    }
    let out = asr
        .transcribe(&wav.into_inner(), &TranscribeConfig::default())
        .unwrap();
    assert_eq!(out.audio_ms, 2000);
    assert!(out.text.split_whitespace().count() <= 5, "{:?}", out.text);
}
