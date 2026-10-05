//! ISO base media (MP4): a writer for one video track and an optional Opus
//! track, with the movie header first so playback can start while the file
//! downloads; reading goes through `re_mp4`.

use re_mp4::{Mp4, StsdBoxContent, TrackKind};

use crate::color::Matrix;
use crate::sound::{self, Encoded};
use crate::{Container, Demuxed, Error, Result, Track, VideoCodec, VideoInfo};

/// Movie timescale (milliseconds).
const MOVIE_SCALE: u32 = 1000;

fn put32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_be_bytes());
}

fn put16(o: &mut Vec<u8>, v: u16) {
    o.extend_from_slice(&v.to_be_bytes());
}

/// A box of type `kind` holding `body`.
fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(body.len() + 8);
    put32(&mut o, (body.len() + 8) as u32);
    o.extend_from_slice(kind);
    o.extend_from_slice(body);
    o
}

/// A full box (version and flags) of type `kind`.
fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(body.len() + 4);
    put32(&mut b, (u32::from(version) << 24) | flags);
    b.extend_from_slice(body);
    bx(kind, &b)
}

fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

/// The unity transformation matrix of `mvhd` and `tkhd`.
fn unity(o: &mut Vec<u8>) {
    for v in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        put32(o, v);
    }
}

/// `value` in `from` units to `to` units, rounded.
fn rescale(value: u64, from: u32, to: u32) -> u64 {
    (value * u64::from(to) + u64::from(from) / 2) / u64::from(from)
}

fn mvhd(duration_ms: u64, next_track: u32) -> Vec<u8> {
    let mut b = Vec::new();
    put32(&mut b, 0);
    put32(&mut b, 0);
    put32(&mut b, MOVIE_SCALE);
    put32(&mut b, duration_ms as u32);
    put32(&mut b, 0x0001_0000);
    put16(&mut b, 0x0100);
    b.extend_from_slice(&[0; 10]);
    unity(&mut b);
    b.extend_from_slice(&[0; 24]);
    put32(&mut b, next_track);
    full(b"mvhd", 0, 0, &b)
}

fn tkhd(id: u32, duration_ms: u64, audio: bool, w: u32, h: u32) -> Vec<u8> {
    let mut b = Vec::new();
    put32(&mut b, 0);
    put32(&mut b, 0);
    put32(&mut b, id);
    put32(&mut b, 0);
    put32(&mut b, duration_ms as u32);
    b.extend_from_slice(&[0; 8]);
    put16(&mut b, 0);
    put16(&mut b, u16::from(audio));
    put16(&mut b, if audio { 0x0100 } else { 0 });
    put16(&mut b, 0);
    unity(&mut b);
    put32(&mut b, w << 16);
    put32(&mut b, h << 16);
    // Enabled, in movie, in preview.
    full(b"tkhd", 0, 3, &b)
}

fn mdhd(scale: u32, duration: u64) -> Vec<u8> {
    let mut b = Vec::new();
    put32(&mut b, 0);
    put32(&mut b, 0);
    put32(&mut b, scale);
    put32(&mut b, duration as u32);
    // Language "und".
    put16(&mut b, 0x55c4);
    put16(&mut b, 0);
    full(b"mdhd", 0, 0, &b)
}

fn hdlr(kind: &[u8; 4], name: &str) -> Vec<u8> {
    let mut b = vec![0; 4];
    b.extend_from_slice(kind);
    b.extend_from_slice(&[0; 12]);
    b.extend_from_slice(name.as_bytes());
    b.push(0);
    full(b"hdlr", 0, 0, &b)
}

fn dinf() -> Vec<u8> {
    let mut d = Vec::new();
    put32(&mut d, 1);
    d.extend_from_slice(&full(b"url ", 0, 1, &[]));
    bx(b"dinf", &full(b"dref", 0, 0, &d))
}

/// Sample tables for samples in one chunk at `offset`.
fn stbl(stsd_entry: &[u8], durations: &[(u32, u32)], sizes: &[u32], keys: Option<&[u32]>, offset: u64) -> Vec<u8> {
    let mut stsd = Vec::new();
    put32(&mut stsd, 1);
    stsd.extend_from_slice(stsd_entry);
    let mut stts = Vec::new();
    put32(&mut stts, durations.len() as u32);
    for &(count, delta) in durations {
        put32(&mut stts, count);
        put32(&mut stts, delta);
    }
    let mut stsc = Vec::new();
    put32(&mut stsc, 1);
    put32(&mut stsc, 1);
    put32(&mut stsc, sizes.len() as u32);
    put32(&mut stsc, 1);
    let mut stsz = Vec::new();
    put32(&mut stsz, 0);
    put32(&mut stsz, sizes.len() as u32);
    for &s in sizes {
        put32(&mut stsz, s);
    }
    let mut parts = vec![full(b"stsd", 0, 0, &stsd), full(b"stts", 0, 0, &stts)];
    if let Some(keys) = keys {
        let mut stss = Vec::new();
        put32(&mut stss, keys.len() as u32);
        for &k in keys {
            put32(&mut stss, k);
        }
        parts.push(full(b"stss", 0, 0, &stss));
    }
    parts.push(full(b"stsc", 0, 0, &stsc));
    parts.push(full(b"stsz", 0, 0, &stsz));
    // A 64-bit offset always: the movie box size then never depends on it.
    let mut co64 = Vec::new();
    put32(&mut co64, 1);
    co64.extend_from_slice(&offset.to_be_bytes());
    parts.push(full(b"co64", 0, 0, &co64));
    bx(b"stbl", &cat(&parts))
}

fn visual_entry(t: &Track) -> Vec<u8> {
    let (fourcc, config_kind) = match t.codec {
        VideoCodec::H264 => (b"avc1", b"avcC"),
        VideoCodec::Av1 => (b"av01", b"av1C"),
        VideoCodec::H265 => unreachable!("not encoded"),
    };
    let mut b = vec![0; 6];
    put16(&mut b, 1);
    b.extend_from_slice(&[0; 16]);
    put16(&mut b, t.width as u16);
    put16(&mut b, t.height as u16);
    put32(&mut b, 0x0048_0000);
    put32(&mut b, 0x0048_0000);
    put32(&mut b, 0);
    put16(&mut b, 1);
    b.extend_from_slice(&[0; 32]);
    put16(&mut b, 0x0018);
    put16(&mut b, 0xffff);
    b.extend_from_slice(&bx(config_kind, &t.config));
    // BT.709 primaries, transfer and matrix, studio range.
    let mut colr = b"nclx".to_vec();
    put16(&mut colr, 1);
    put16(&mut colr, 1);
    put16(&mut colr, 1);
    colr.push(0);
    b.extend_from_slice(&bx(b"colr", &colr));
    let mut pasp = Vec::new();
    put32(&mut pasp, 1);
    put32(&mut pasp, 1);
    b.extend_from_slice(&bx(b"pasp", &pasp));
    bx(fourcc, &b)
}

fn opus_entry(s: &Encoded) -> Vec<u8> {
    let mut b = vec![0; 6];
    put16(&mut b, 1);
    b.extend_from_slice(&[0; 8]);
    put16(&mut b, s.channels as u16);
    put16(&mut b, 16);
    put32(&mut b, 0);
    put32(&mut b, sound::RATE << 16);
    // dOps: the OpusHead fields, big-endian, without magic or version 1.
    let mut d = vec![0, s.channels as u8];
    put16(&mut d, s.pre_skip as u16);
    put32(&mut d, s.input_rate);
    put16(&mut d, 0);
    d.push(0);
    b.extend_from_slice(&bx(b"dOps", &d));
    bx(b"Opus", &b)
}

/// The movie box, given where each track's chunk starts.
fn moov(t: &Track, s: Option<&Encoded>, video_at: u64, sound_at: u64) -> Vec<u8> {
    let n = t.samples.len() as u64;
    let v_scale = t.rate.num;
    let v_dur = n * u64::from(t.rate.den);
    let v_ms = rescale(v_dur, v_scale, MOVIE_SCALE);
    let sizes: Vec<u32> = t.samples.iter().map(|x| x.data.len() as u32).collect();
    let keys: Vec<u32> = t.samples.iter().enumerate().filter(|(_, x)| x.key).map(|(i, _)| i as u32 + 1).collect();
    let keys = (keys.len() != t.samples.len()).then_some(keys.as_slice());
    let vstbl = stbl(&visual_entry(t), &[(n as u32, t.rate.den)], &sizes, keys, video_at);
    let vminf = bx(b"minf", &cat(&[full(b"vmhd", 0, 1, &[0; 8]), dinf(), vstbl]));
    let vmdia = bx(b"mdia", &cat(&[mdhd(v_scale, v_dur), hdlr(b"vide", "Video"), vminf]));
    let vtrak = bx(b"trak", &cat(&[tkhd(1, v_ms, false, t.width, t.height), vmdia]));
    let mut parts = vec![vtrak];
    let mut movie_ms = v_ms;
    if let Some(s) = s {
        let a_ms = rescale(s.samples, sound::RATE, MOVIE_SCALE);
        movie_ms = movie_ms.max(a_ms);
        let a_dur = (s.packets.len() * sound::FRAME) as u64;
        let sizes: Vec<u32> = s.packets.iter().map(|p| p.len() as u32).collect();
        let astbl = stbl(&opus_entry(s), &[(s.packets.len() as u32, sound::FRAME as u32)], &sizes, None, sound_at);
        let aminf = bx(b"minf", &cat(&[full(b"smhd", 0, 0, &[0; 4]), dinf(), astbl]));
        let amdia = bx(b"mdia", &cat(&[mdhd(sound::RATE, a_dur), hdlr(b"soun", "Sound"), aminf]));
        // The edit list skips the encoder's priming samples.
        let mut elst = Vec::new();
        put32(&mut elst, 1);
        put32(&mut elst, a_ms as u32);
        put32(&mut elst, s.pre_skip);
        put32(&mut elst, 0x0001_0000);
        let edts = bx(b"edts", &full(b"elst", 0, 0, &elst));
        parts.push(bx(b"trak", &cat(&[tkhd(2, a_ms, true, 0, 0), edts, amdia])));
    }
    let next = if s.is_some() { 3 } else { 2 };
    parts.insert(0, mvhd(movie_ms, next));
    bx(b"moov", &cat(&parts))
}

pub(crate) fn write(t: &Track, s: Option<&Encoded>) -> Vec<u8> {
    let brand: &[u8; 4] = if t.codec == VideoCodec::Av1 { b"av01" } else { b"avc1" };
    let mut f = b"isom".to_vec();
    put32(&mut f, 0x200);
    for b in [b"isom", b"iso2", brand, b"mp41"] {
        f.extend_from_slice(b);
    }
    let ftyp = bx(b"ftyp", &f);
    let video: usize = t.samples.iter().map(|x| x.data.len()).sum();
    let audio: usize = s.map_or(0, |s| s.packets.iter().map(Vec::len).sum());
    let payload = (video + audio) as u64;
    let large = payload + 8 > u64::from(u32::MAX);
    let header = if large { 16 } else { 8 };
    let moov_len = moov(t, s, 0, 0).len() as u64;
    let video_at = ftyp.len() as u64 + moov_len + header;
    let sound_at = video_at + video as u64;
    let moov = moov(t, s, video_at, sound_at);
    let mut out = Vec::with_capacity(video_at as usize + video + audio);
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    if large {
        put32(&mut out, 1);
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(payload + 16).to_be_bytes());
    } else {
        put32(&mut out, (payload + 8) as u32);
        out.extend_from_slice(b"mdat");
    }
    for x in &t.samples {
        out.extend_from_slice(&x.data);
    }
    if let Some(s) = s {
        for p in &s.packets {
            out.extend_from_slice(p);
        }
    }
    out
}

fn container(e: impl std::fmt::Display) -> Error {
    Error::Container(format!("MP4: {e}"))
}

/// The first `colr` box of type `nclx` in `entry`, as a matrix.
fn nclx(entry: &[u8]) -> Option<Matrix> {
    let at = entry.windows(8).position(|w| w == b"colrnclx")?;
    let b = entry.get(at + 8..at + 15)?;
    Matrix::from_code(u32::from(u16::from_be_bytes([b[4], b[5]])), b[6] & 0x80 != 0)
}

/// The movie box of a file, found by walking the top-level boxes.
fn moov_bytes(bytes: &[u8]) -> Option<&[u8]> {
    let mut at = 0usize;
    while at + 8 <= bytes.len() {
        let mut size = u32::from_be_bytes(bytes[at..at + 4].try_into().ok()?) as usize;
        if size == 1 {
            size = usize::try_from(u64::from_be_bytes(bytes.get(at + 8..at + 16)?.try_into().ok()?)).ok()?;
        } else if size == 0 {
            size = bytes.len() - at;
        }
        if size < 8 {
            return None;
        }
        if &bytes[at + 4..at + 8] == b"moov" {
            return bytes.get(at..at + size);
        }
        at = at.checked_add(size)?;
    }
    None
}

pub(crate) fn read(bytes: &[u8]) -> Result<Demuxed> {
    let mp4 = Mp4::read_bytes(bytes).map_err(container)?;
    let track = mp4
        .tracks()
        .values()
        .find(|t| t.kind == Some(TrackKind::Video))
        .ok_or_else(|| container("no video track"))?;
    let stsd = &track.trak(&mp4).mdia.minf.stbl.stsd;
    let codec = match &stsd.contents {
        StsdBoxContent::Avc1(_) => VideoCodec::H264,
        StsdBoxContent::Hev1(_) | StsdBoxContent::Hvc1(_) => VideoCodec::H265,
        StsdBoxContent::Av01(_) => VideoCodec::Av1,
        _ => {
            let name = track.codec_string(&mp4).unwrap_or_else(|| "unknown".into());
            return Err(Error::Request(format!("MP4 video in {name} is not decoded; H.264, H.265 and AV1 are")));
        }
    };
    let config = track.raw_codec_config(&mp4).ok_or_else(|| container("no decoder configuration record"))?;
    let mut samples = Vec::with_capacity(track.samples.len());
    for s in &track.samples {
        let (a, n) = (usize::try_from(s.offset).map_err(container)?, usize::try_from(s.size).map_err(container)?);
        samples.push(bytes.get(a..a + n).ok_or_else(|| container("sample outside the file"))?.to_vec());
    }
    if samples.is_empty() {
        return Err(container("the video track has no samples"));
    }
    let frames = samples.len() as u32;
    let fps = if track.duration > 0 { (f64::from(frames) * track.timescale as f64 / track.duration as f64) as f32 } else { 0.0 };
    // The colour box sits in the sample entry; look for it inside the
    // movie box rather than re-parsing the entry.
    let matrix = moov_bytes(bytes).and_then(nclx);
    let info = VideoInfo { container: Container::Mp4, codec, width: u32::from(track.width), height: u32::from(track.height), frames, fps };
    Ok(Demuxed { info, config, samples, matrix })
}
