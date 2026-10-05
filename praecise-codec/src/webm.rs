//! WebM: a writer for one AV1 track and an optional Opus track; reading
//! (any Matroska file) goes through `matroska-demuxer`.

use std::io::Cursor;

use matroska_demuxer::{Frame, MatrixCoefficients, MatroskaFile, Range};

use crate::color::Matrix;
use crate::sound::{self, Encoded};
use crate::{Container, Demuxed, Error, Result, Track, VideoCodec, VideoInfo};

/// Block timestamps are in milliseconds.
const SCALE_NS: u64 = 1_000_000;
/// Longest cluster, in milliseconds (block offsets are signed 16-bit).
const CLUSTER_MS: u64 = 30_000;

/// An element: its ID bytes, an eight-byte size and its body.
fn el(id: u32, body: &[u8]) -> Vec<u8> {
    let idb = id.to_be_bytes();
    let skip = idb.iter().take_while(|&&b| b == 0).count();
    let mut o = idb[skip..].to_vec();
    o.push(0x01);
    o.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
    o.extend_from_slice(body);
    o
}

fn uint(id: u32, v: u64) -> Vec<u8> {
    let b = v.to_be_bytes();
    let skip = b.iter().take_while(|&&x| x == 0).count().min(7);
    el(id, &b[skip..])
}

fn float(id: u32, v: f64) -> Vec<u8> {
    el(id, &v.to_be_bytes())
}

fn text(id: u32, v: &str) -> Vec<u8> {
    el(id, v.as_bytes())
}

/// One block to place: track, time in milliseconds, keyframe, payload.
struct Block<'a> {
    track: u8,
    ms: u64,
    key: bool,
    data: &'a [u8],
}

pub(crate) fn write(t: &Track, s: Option<&Encoded>) -> Vec<u8> {
    debug_assert_eq!(t.codec, VideoCodec::Av1);
    let header = el(
        0x1A45_DFA3,
        &[uint(0x4286, 1), uint(0x42F7, 1), uint(0x42F2, 4), uint(0x42F3, 8), text(0x4282, "webm"), uint(0x4287, 4), uint(0x4285, 2)].concat(),
    );
    let frame_ns = |i: u64| i * 1_000_000_000 * u64::from(t.rate.den) / u64::from(t.rate.num);
    let n = t.samples.len() as u64;
    let mut duration_ms = frame_ns(n) as f64 / SCALE_NS as f64;
    let colour = el(0x55B0, &[uint(0x55B1, 1), uint(0x55B9, 1), uint(0x55BA, 1), uint(0x55BB, 1)].concat());
    let video = el(0xE0, &[uint(0xB0, u64::from(t.width)), uint(0xBA, u64::from(t.height)), colour].concat());
    let vtrack = el(
        0xAE,
        &[
            uint(0xD7, 1),
            uint(0x73C5, 1),
            uint(0x83, 1),
            uint(0x9C, 0),
            text(0x86, "V_AV1"),
            el(0x63A2, &t.config),
            uint(0x23_E383, frame_ns(1)),
            video,
        ]
        .concat(),
    );
    let mut tracks = vtrack;
    if let Some(s) = s {
        duration_ms = duration_ms.max(s.samples as f64 * 1000.0 / f64::from(sound::RATE));
        let audio = el(0xE1, &[float(0xB5, f64::from(sound::RATE)), uint(0x9F, u64::from(s.channels))].concat());
        tracks.extend(el(
            0xAE,
            &[
                uint(0xD7, 2),
                uint(0x73C5, 2),
                uint(0x83, 2),
                uint(0x9C, 0),
                text(0x86, "A_OPUS"),
                el(0x63A2, &sound::opus_head(s)),
                uint(0x56AA, u64::from(s.pre_skip) * 1_000_000_000 / u64::from(sound::RATE)),
                uint(0x56BB, 80_000_000),
                audio,
            ]
            .concat(),
        ));
    }
    let info = el(
        0x1549_A966,
        &[uint(0x2A_D7B1, SCALE_NS), float(0x4489, duration_ms), text(0x4D80, "praecise-codec"), text(0x5741, "praecise-codec")].concat(),
    );
    let mut blocks: Vec<Block<'_>> = t
        .samples
        .iter()
        .enumerate()
        .map(|(i, x)| Block { track: 1, ms: (frame_ns(i as u64) + SCALE_NS / 2) / SCALE_NS, key: x.key, data: &x.data })
        .collect();
    if let Some(s) = s {
        blocks.extend(s.packets.iter().enumerate().map(|(i, p)| Block { track: 2, ms: (i * sound::FRAME * 1000 / sound::RATE as usize) as u64, key: true, data: p }));
    }
    // Stable: at equal times video comes first, so clusters open on video.
    blocks.sort_by_key(|b| b.ms);
    let mut clusters = Vec::new();
    let mut cur: Option<(u64, Vec<u8>)> = None;
    for b in &blocks {
        let open_new = match &cur {
            None => true,
            Some((start, _)) => (b.track == 1 && b.key) || b.ms - start >= CLUSTER_MS,
        };
        if open_new {
            if let Some((start, body)) = cur.take() {
                clusters.extend(el(0x1F43_B675, &[uint(0xE7, start), body].concat()));
            }
            cur = Some((b.ms, Vec::new()));
        }
        let (start, body) = cur.as_mut().expect("opened above");
        let mut sb = vec![0x80 | b.track];
        sb.extend_from_slice(&((b.ms - *start) as i16).to_be_bytes());
        sb.push(if b.key { 0x80 } else { 0 });
        sb.extend_from_slice(b.data);
        body.extend(el(0xA3, &sb));
    }
    if let Some((start, body)) = cur {
        clusters.extend(el(0x1F43_B675, &[uint(0xE7, start), body].concat()));
    }
    let segment = el(0x1853_8067, &[info, el(0x1654_AE6B, &tracks), clusters].concat());
    [header, segment].concat()
}

fn container(e: impl std::fmt::Debug) -> Error {
    Error::Container(format!("Matroska: {e:?}"))
}

fn matrix(m: Option<MatrixCoefficients>, r: Option<Range>) -> Option<Matrix> {
    let code = match m? {
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Unknown => 2,
        MatrixCoefficients::Bt470bg => 5,
        MatrixCoefficients::Smpte170 => 6,
        MatrixCoefficients::Bt2020Ncl => 9,
        _ => return None,
    };
    Matrix::from_code(code, matches!(r, Some(Range::Full)))
}

pub(crate) fn read(bytes: &[u8]) -> Result<Demuxed> {
    let mut mkv = MatroskaFile::open(Cursor::new(bytes)).map_err(container)?;
    let track = mkv.tracks().iter().find(|t| t.video().is_some()).ok_or_else(|| container("no video track"))?;
    let codec = match track.codec_id() {
        "V_MPEG4/ISO/AVC" => VideoCodec::H264,
        "V_MPEGH/ISO/HEVC" => VideoCodec::H265,
        "V_AV1" => VideoCodec::Av1,
        other => return Err(Error::Request(format!("Matroska video in {other} is not decoded; H.264, H.265 and AV1 are"))),
    };
    let config = track.codec_private().ok_or_else(|| container("no codec private data"))?.to_vec();
    let number = track.track_number().get();
    let v = track.video().expect("found above");
    let (width, height) = (v.pixel_width().get() as u32, v.pixel_height().get() as u32);
    let m = v.colour().and_then(|c| matrix(c.matrix_coefficients(), c.range()));
    let default_ns = track.default_duration().map(std::num::NonZeroU64::get);
    let scale = mkv.info().timestamp_scale().get();
    let mut samples = Vec::new();
    let (mut first, mut last) = (None, 0u64);
    let mut frame = Frame::default();
    while mkv.next_frame(&mut frame).map_err(container)? {
        if frame.track == number {
            first.get_or_insert(frame.timestamp);
            last = frame.timestamp;
            samples.push(std::mem::take(&mut frame.data));
        }
    }
    if samples.is_empty() {
        return Err(container("the video track has no frames"));
    }
    let frames = samples.len() as u32;
    let fps = match (default_ns, first) {
        (Some(ns), _) if ns > 0 => (1e9 / ns as f64) as f32,
        (_, Some(f)) if last > f => (f64::from(frames - 1) * 1e9 / ((last - f) as f64 * scale as f64)) as f32,
        _ => 0.0,
    };
    let info = VideoInfo { container: Container::WebM, codec, width, height, frames, fps };
    Ok(Demuxed { info, config, samples, matrix: m })
}
