//! Encode short clips, decode them back, and check every frame: count,
//! size, order and fidelity. Files written by other encoders (made by
//! `fixtures/make.sh`) are decoded against the same frames.

use praecise_codec::{Container, EncodeOptions, Sound, VideoCodec, decode, decode_each, encode, probe};

/// Frame `i` of the test pattern: smooth bands moving at different speeds
/// in each channel, so every frame differs from its neighbours.
fn pattern(w: usize, h: usize, i: usize) -> Vec<u8> {
    let t = i as f32;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f32, y as f32);
            out.push((128.0 + 100.0 * (xf / 9.0 + t * 0.5).sin()).round() as u8);
            out.push((128.0 + 100.0 * (yf / 7.0 - t * 0.3).sin()).round() as u8);
            out.push((128.0 + 100.0 * ((xf + yf) / 11.0 + t * 0.8).sin()).round() as u8);
        }
    }
    out
}

fn mse(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x.abs_diff(y)).powi(2)).sum::<f64>() / a.len() as f64
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    10.0 * (255.0f64 * 255.0 / mse(a, b).max(1e-9)).log10()
}

/// Every decoded frame is closest to the source frame of its own index
/// and within `min_psnr` of it.
fn check_frames(decoded: &[u8], w: usize, h: usize, n: usize, min_psnr: f64) {
    let size = w * h * 3;
    assert_eq!(decoded.len(), n * size, "frame count");
    let source: Vec<Vec<u8>> = (0..n).map(|i| pattern(w, h, i)).collect();
    for (i, f) in decoded.chunks_exact(size).enumerate() {
        let nearest = (0..n).min_by(|&a, &b| mse(f, &source[a]).total_cmp(&mse(f, &source[b]))).unwrap();
        assert_eq!(nearest, i, "frame {i} decodes closest to source frame {nearest}");
        let p = psnr(f, &source[i]);
        assert!(p >= min_psnr, "frame {i}: {p:.1} dB < {min_psnr}");
    }
}

fn clip(w: usize, h: usize, n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| pattern(w, h, i)).collect()
}

fn tone(rate: u32, seconds: f32, channels: u32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..channels).flat_map(|c| (0..n).map(move |i| (2.0 * std::f32::consts::PI * (220.0 * (c + 1) as f32) * i as f32 / rate as f32).sin() * 0.3)).collect()
}

fn round_trip(codec: VideoCodec, container: Container, sound: bool) {
    let (w, h, n, fps) = (96usize, 64usize, 30usize, 24.0f32);
    let rgb = clip(w, h, n);
    let samples = tone(24_000, n as f32 / fps, 2);
    let snd = sound.then_some(Sound { sample_rate: 24_000, channels: 2, samples: &samples });
    let opts = EncodeOptions { codec, container, quantizer: 18 };
    let file = encode(opts, w as u32, h as u32, fps, &rgb, snd).unwrap();
    let info = probe(&file).unwrap();
    assert_eq!((info.container, info.codec, info.width, info.height, info.frames), (container, codec, w as u32, h as u32, n as u32));
    assert!((info.fps - fps).abs() < 1e-3, "fps {}", info.fps);
    let d = decode(&file).unwrap();
    assert_eq!(d.info, info);
    check_frames(&d.rgb, w, h, n, 30.0);
}

#[test]
fn h264_in_mp4() {
    round_trip(VideoCodec::H264, Container::Mp4, false);
}

#[test]
fn h264_in_mp4_with_sound() {
    round_trip(VideoCodec::H264, Container::Mp4, true);
}

#[test]
fn av1_in_mp4_with_sound() {
    round_trip(VideoCodec::Av1, Container::Mp4, true);
}

#[test]
fn av1_in_webm() {
    round_trip(VideoCodec::Av1, Container::WebM, false);
}

#[test]
fn av1_in_webm_with_sound() {
    round_trip(VideoCodec::Av1, Container::WebM, true);
}

#[test]
fn mp4_layout_is_playable_while_downloading() {
    let file = encode(EncodeOptions::default(), 32, 32, 30.0, &clip(32, 32, 3), None).unwrap();
    assert_eq!(&file[4..8], b"ftyp");
    let ftyp = u32::from_be_bytes(file[0..4].try_into().unwrap()) as usize;
    assert_eq!(&file[ftyp + 4..ftyp + 8], b"moov", "the movie box comes before the media data");
}

#[test]
fn decoding_can_stop_early() {
    let file = encode(EncodeOptions::default(), 32, 32, 30.0, &clip(32, 32, 8), None).unwrap();
    let mut seen = Vec::new();
    decode_each(&file, |i, _| {
        seen.push(i);
        i < 2
    })
    .unwrap();
    assert_eq!(seen, vec![0, 1, 2]);
}

#[test]
fn refusals() {
    let rgb = clip(32, 32, 1);
    let bad = |codec, container, w, fps| encode(EncodeOptions { codec, container, quantizer: 20 }, w, 32, fps, &rgb[..(w as usize * 32 * 3)], None).is_err();
    assert!(bad(VideoCodec::H264, Container::WebM, 32, 24.0));
    assert!(bad(VideoCodec::H265, Container::Mp4, 32, 24.0));
    assert!(bad(VideoCodec::H264, Container::Mp4, 31, 24.0));
    assert!(bad(VideoCodec::H264, Container::Mp4, 32, 0.0));
    assert!(encode(EncodeOptions::default(), 32, 32, 24.0, &rgb[..100], None).is_err());
    assert!(probe(b"not a video file at all").is_err());
}

const FIXTURE_W: usize = 64;
const FIXTURE_H: usize = 48;
const FIXTURE_N: usize = 12;

fn fixture(name: &str, codec: VideoCodec, container: Container) {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let d = decode(&bytes).unwrap();
    assert_eq!((d.info.codec, d.info.container), (codec, container));
    assert_eq!((d.info.width, d.info.height, d.info.frames), (FIXTURE_W as u32, FIXTURE_H as u32, FIXTURE_N as u32));
    assert!((d.info.fps - 24.0).abs() < 0.05, "fps {}", d.info.fps);
    check_frames(&d.rgb, FIXTURE_W, FIXTURE_H, FIXTURE_N, 28.0);
}

#[test]
fn h264_high_profile_with_b_frames_in_mp4() {
    fixture("h264-high-bframes.mp4", VideoCodec::H264, Container::Mp4);
}

#[test]
fn h264_in_matroska() {
    fixture("h264.mkv", VideoCodec::H264, Container::WebM);
}

#[test]
fn h265_main_with_b_frames_in_mp4() {
    fixture("h265-main-bframes.mp4", VideoCodec::H265, Container::Mp4);
}

#[test]
fn h265_in_matroska() {
    fixture("h265.mkv", VideoCodec::H265, Container::WebM);
}

#[test]
fn av1_from_another_encoder_in_webm() {
    fixture("av1.webm", VideoCodec::Av1, Container::WebM);
}

#[test]
fn av1_from_another_encoder_in_mp4() {
    fixture("av1.mp4", VideoCodec::Av1, Container::Mp4);
}
