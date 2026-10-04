//! Uncompressed AVI: 24-bit RGB frames interleaved with 16-bit PCM audio,
//! one audio chunk per frame, with a keyframe index.

use super::Video;
use crate::music::Audio;

const KEYFRAME: u32 = 0x10;
const HAS_INDEX: u32 = 0x10;
const INTERLEAVED: u32 = 0x100;

fn put(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Append chunk `id` holding `body`, padded to an even length.
fn chunk(out: &mut Vec<u8>, id: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(id);
    put(out, body.len() as u32);
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
}

/// Wrap `body` in a `LIST` of type `kind`.
fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 12);
    out.extend_from_slice(b"LIST");
    put(&mut out, body.len() as u32 + 4);
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

#[allow(clippy::too_many_arguments)]
fn stream_header(kind: &[u8; 4], scale: u32, rate: u32, length: u32, buffer: u32, sample_size: u32, w: u16, h: u16) -> Vec<u8> {
    let mut s = Vec::with_capacity(56);
    s.extend_from_slice(kind);
    put(&mut s, 0); // handler
    put(&mut s, 0); // flags
    put16(&mut s, 0); // priority
    put16(&mut s, 0); // language
    put(&mut s, 0); // initial frames
    put(&mut s, scale);
    put(&mut s, rate);
    put(&mut s, 0); // start
    put(&mut s, length);
    put(&mut s, buffer);
    put(&mut s, u32::MAX); // quality: default
    put(&mut s, sample_size);
    for v in [0, 0, w, h] {
        put16(&mut s, v);
    }
    s
}

/// The clip `video` with its soundtrack `audio` as one AVI file.
pub(crate) fn encode(video: &Video, audio: &Audio) -> Vec<u8> {
    let (w, h) = (video.width as usize, video.height as usize);
    let n = video.frames as usize;
    let stride = (w * 3 + 3) & !3;
    let frame_bytes = stride * h;
    let (fps_num, fps_den) = super::rational(video.fps);
    let ch = audio.channels.max(1) as usize;
    let block = ch * 2;
    let samples = audio.frames();
    // Audio sample range of frame i, so that each video chunk is followed by
    // the sound that plays during it.
    let span = |i: usize| -> usize {
        let t = i as f64 * f64::from(fps_den) / f64::from(fps_num);
        ((t * f64::from(audio.sample_rate)).round() as usize).min(samples)
    };

    let mut avih = Vec::with_capacity(56);
    put(&mut avih, (1e6 * f64::from(fps_den) / f64::from(fps_num)).round() as u32);
    put(&mut avih, (frame_bytes as f64 * f64::from(fps_num) / f64::from(fps_den)) as u32 + audio.sample_rate * block as u32);
    put(&mut avih, 0);
    put(&mut avih, HAS_INDEX | INTERLEAVED);
    put(&mut avih, n as u32);
    put(&mut avih, 0);
    put(&mut avih, 2);
    put(&mut avih, frame_bytes as u32);
    put(&mut avih, w as u32);
    put(&mut avih, h as u32);
    for _ in 0..4 {
        put(&mut avih, 0);
    }

    let mut vstrl = Vec::new();
    chunk(&mut vstrl, b"strh", &stream_header(b"vids", fps_den, fps_num, n as u32, frame_bytes as u32, 0, w as u16, h as u16));
    let mut bih = Vec::with_capacity(40);
    put(&mut bih, 40);
    put(&mut bih, w as u32);
    put(&mut bih, h as u32); // positive: rows bottom-up
    put16(&mut bih, 1);
    put16(&mut bih, 24);
    put(&mut bih, 0); // uncompressed RGB
    put(&mut bih, frame_bytes as u32);
    for _ in 0..4 {
        put(&mut bih, 0);
    }
    chunk(&mut vstrl, b"strf", &bih);

    let mut astrl = Vec::new();
    let per_frame = span(1).max(1) * block;
    chunk(&mut astrl, b"strh", &stream_header(b"auds", block as u32, audio.sample_rate * block as u32, samples as u32, per_frame as u32, block as u32, 0, 0));
    let mut wfx = Vec::with_capacity(16);
    put16(&mut wfx, 1); // PCM
    put16(&mut wfx, ch as u16);
    put(&mut wfx, audio.sample_rate);
    put(&mut wfx, audio.sample_rate * block as u32);
    put16(&mut wfx, block as u16);
    put16(&mut wfx, 16);
    chunk(&mut astrl, b"strf", &wfx);

    let mut hdrl = Vec::new();
    chunk(&mut hdrl, b"avih", &avih);
    hdrl.extend_from_slice(&list(b"strl", &vstrl));
    hdrl.extend_from_slice(&list(b"strl", &astrl));

    // `movi` body; index offsets count from the `movi` type tag.
    let mut movi = Vec::with_capacity(n * (frame_bytes + per_frame + 16));
    let mut idx = Vec::with_capacity(n * 32);
    let mut entry = |movi: &Vec<u8>, id: &[u8; 4], size: usize, flags: u32| {
        idx.extend_from_slice(id);
        put(&mut idx, flags);
        put(&mut idx, movi.len() as u32 + 4);
        put(&mut idx, size as u32);
    };
    let mut frame = vec![0u8; frame_bytes];
    for (i, rgb) in video.rgb.chunks_exact(w * h * 3).enumerate().take(n) {
        for y in 0..h {
            let row = &rgb[(h - 1 - y) * w * 3..(h - y) * w * 3];
            let dst = &mut frame[y * stride..y * stride + w * 3];
            for (d, s) in dst.chunks_exact_mut(3).zip(row.chunks_exact(3)) {
                d.copy_from_slice(&[s[2], s[1], s[0]]);
            }
        }
        entry(&movi, b"00dc", frame_bytes, KEYFRAME);
        chunk(&mut movi, b"00dc", &frame);
        let (a, b) = (span(i), if i + 1 == n { samples } else { span(i + 1) });
        if b > a {
            let mut pcm = Vec::with_capacity((b - a) * block);
            for s in a..b {
                for c in 0..ch {
                    let v = (audio.samples[c * samples + s].clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    pcm.extend_from_slice(&v.to_le_bytes());
                }
            }
            entry(&movi, b"01wb", pcm.len(), KEYFRAME);
            chunk(&mut movi, b"01wb", &pcm);
        }
    }

    let mut body = Vec::with_capacity(movi.len() + idx.len() + hdrl.len() + 64);
    body.extend_from_slice(b"AVI ");
    body.extend_from_slice(&list(b"hdrl", &hdrl));
    body.extend_from_slice(&list(b"movi", &movi));
    chunk(&mut body, b"idx1", &idx);
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(b"RIFF");
    put(&mut out, body.len() as u32);
    out.extend_from_slice(&body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Timings;

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    #[test]
    fn frames_and_sound_interleave_under_a_consistent_index() {
        let (w, h, n) = (3u32, 2u32, 4u32);
        let rgb: Vec<u8> = (0..w * h * 3 * n).map(|v| v as u8).collect();
        let video = Video { width: w, height: h, frames: n, fps: 8.0, rgb, seed: 0, evaluations: 0, timings: Timings::default() };
        let audio = Audio { sample_rate: 80, channels: 2, samples: vec![0.5; 2 * 40], seed: 0, evaluations: 0, timings: Timings::default() };
        let avi = encode(&video, &audio);
        assert_eq!(&avi[..4], b"RIFF");
        assert_eq!(u32_at(&avi, 4) as usize + 8, avi.len());
        assert_eq!(&avi[8..12], b"AVI ");
        let movi = avi.windows(4).position(|x| x == b"movi").unwrap();
        let idx1 = avi.windows(4).rposition(|x| x == b"idx1").unwrap();
        let entries = u32_at(&avi, idx1 + 4) as usize / 16;
        assert_eq!(entries, 2 * n as usize, "one video and one audio chunk per frame");
        let mut audio_bytes = 0;
        for e in 0..entries {
            let at = idx1 + 8 + e * 16;
            let (id, off, size) = (&avi[at..at + 4], u32_at(&avi, at + 8) as usize, u32_at(&avi, at + 12) as usize);
            assert_eq!(&avi[movi + off..movi + off + 4], id, "entry {e} points at its chunk");
            assert_eq!(u32_at(&avi, movi + off + 4) as usize, size);
            if id == b"01wb" {
                audio_bytes += size;
            } else {
                assert_eq!(size, 12 * 2, "rows padded to four bytes");
            }
        }
        assert_eq!(audio_bytes, 40 * 4, "every sample written once");
        // Top-left pixel lands at the start of the last stored row, as BGR.
        let first = movi + 4 + 8;
        assert_eq!(&avi[first + 12..first + 15], &[2, 1, 0]);
    }
}
