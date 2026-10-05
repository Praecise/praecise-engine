//! Opus sound for the soundtrack track.

use opus::{Application, Bitrate, Channels};

use crate::{Error, Result, Sound};

/// Opus always runs at 48 kHz in containers.
pub(crate) const RATE: u32 = 48_000;
/// 20 ms packets.
pub(crate) const FRAME: usize = 960;

/// An encoded soundtrack.
#[derive(Debug, Clone)]
pub(crate) struct Encoded {
    pub channels: u32,
    /// Decoder priming samples to discard, at 48 kHz.
    pub pre_skip: u32,
    /// The sample rate the sound was given at.
    pub input_rate: u32,
    /// One packet per `FRAME` samples.
    pub packets: Vec<Vec<u8>>,
    /// Samples per channel of sound, at 48 kHz, after priming.
    pub samples: u64,
}

fn codec(e: impl std::fmt::Display) -> Error {
    Error::Codec(format!("Opus: {e}"))
}

/// `sinc(x)`.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) }
}

/// Band-limited resampling of one channel by windowed sinc interpolation.
pub(crate) fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    const HALF: f64 = 24.0;
    let step = f64::from(from) / f64::from(to);
    let cutoff = (f64::from(to) / f64::from(from)).min(1.0) * 0.97;
    let reach = HALF / cutoff;
    let n_out = (x.len() as f64 / step).round() as usize;
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let t = n as f64 * step;
        let lo = (t - reach).ceil().max(0.0) as usize;
        let hi = ((t + reach).floor() as usize).min(x.len().saturating_sub(1));
        let mut acc = 0.0;
        for (k, &s) in x.iter().enumerate().take(hi + 1).skip(lo) {
            let d = t - k as f64;
            let w = 0.5 + 0.5 * (std::f64::consts::PI * d / reach).cos();
            acc += f64::from(s) * cutoff * sinc(cutoff * d) * w;
        }
        out.push(acc as f32);
    }
    out
}

pub(crate) fn encode(s: Sound<'_>) -> Result<Encoded> {
    let ch = s.channels as usize;
    let layout = match s.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        n => return Err(Error::Request(format!("{n} sound channels; 1 or 2 are encoded"))),
    };
    if s.sample_rate == 0 || s.samples.len() % ch != 0 {
        return Err(Error::Request("sound is not a whole number of samples per channel".into()));
    }
    let per = s.samples.len() / ch;
    let planes: Vec<Vec<f32>> = (0..ch).map(|c| resample(&s.samples[c * per..(c + 1) * per], s.sample_rate, RATE)).collect();
    let len = planes[0].len();
    let mut enc = opus::Encoder::new(RATE, layout, Application::Audio).map_err(codec)?;
    enc.set_bitrate(Bitrate::Bits(64_000 * s.channels as i32)).map_err(codec)?;
    let pre_skip = u32::try_from(enc.get_lookahead().map_err(codec)?).map_err(codec)?;
    // Priming plus sound, padded with silence to whole packets.
    let total = (len + pre_skip as usize).div_ceil(FRAME) * FRAME;
    let mut inter = vec![0f32; total * ch];
    for (i, frame) in inter.chunks_exact_mut(ch).enumerate().take(len) {
        for (c, v) in frame.iter_mut().enumerate() {
            *v = planes[c][i].clamp(-1.0, 1.0);
        }
    }
    let mut packets = Vec::with_capacity(total / FRAME);
    let mut buf = vec![0u8; 4000];
    for f in inter.chunks_exact(FRAME * ch) {
        let n = enc.encode_float(f, &mut buf).map_err(codec)?;
        packets.push(buf[..n].to_vec());
    }
    Ok(Encoded { channels: s.channels, pre_skip, input_rate: s.sample_rate, packets, samples: len as u64 })
}

/// The `OpusHead` identification header (RFC 7845, section 5.1).
pub(crate) fn opus_head(e: &Encoded) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.push(1);
    h.push(e.channels as u8);
    h.extend_from_slice(&(e.pre_skip as u16).to_le_bytes());
    h.extend_from_slice(&e.input_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes());
    h.push(0);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_a_tone() {
        let from = 44_100u32;
        let x: Vec<f32> = (0..from).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin() * 0.5).collect();
        let y = resample(&x, from, RATE);
        assert_eq!(y.len(), RATE as usize);
        // Away from the edges, the output is the same tone at the new rate.
        for (i, &v) in y.iter().enumerate().skip(1000).take(RATE as usize - 2000) {
            let want = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE as f32).sin() * 0.5;
            assert!((v - want).abs() < 2e-3, "{i}: {v} vs {want}");
        }
    }

    #[test]
    fn sound_round_trips_through_opus() {
        let n = 24_000usize;
        let tone: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 24_000.0).sin() * 0.4).collect();
        let e = encode(Sound { sample_rate: 24_000, channels: 1, samples: &tone }).unwrap();
        assert_eq!(e.samples, 48_000);
        assert_eq!(e.packets.len(), (48_000 + e.pre_skip as usize).div_ceil(FRAME));
        let mut dec = opus::Decoder::new(RATE, Channels::Mono).unwrap();
        let mut pcm = Vec::new();
        let mut buf = vec![0f32; FRAME];
        for p in &e.packets {
            let k = dec.decode_float(p, &mut buf, false).unwrap();
            pcm.extend_from_slice(&buf[..k]);
        }
        let pcm = &pcm[e.pre_skip as usize..][..48_000];
        let (mut err, mut sig) = (0f64, 0f64);
        for (i, &v) in pcm.iter().enumerate().skip(2000).take(44_000) {
            let want = (2.0 * std::f64::consts::PI * 300.0 * i as f64 / 48_000.0).sin() * 0.4;
            err += (f64::from(v) - want).powi(2);
            sig += want * want;
        }
        let snr = 10.0 * (sig / err).log10();
        assert!(snr > 20.0, "snr {snr:.1} dB");
    }
}
