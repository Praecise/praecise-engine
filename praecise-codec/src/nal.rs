//! NAL unit framing (Annex B start codes and length prefixes) and the H.264
//! and H.265 decoder configuration records.

use crate::{Error, Result};

/// NAL units of an Annex B stream, without their start codes.
pub fn split_annex_b(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push((i, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &(_, body)) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map_or(data.len(), |&(code, _)| code);
        // A four-byte start code leaves its leading zero on the previous unit.
        while end > body && data[end - 1] == 0 && k + 1 < starts.len() {
            end -= 1;
        }
        if end > body {
            out.push(&data[body..end]);
        }
    }
    out
}

/// NAL units of a length-prefixed sample.
pub fn split_prefixed(data: &[u8], len_size: usize) -> Result<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if i + len_size > data.len() {
            return Err(Error::Container("NAL length prefix runs past the sample".into()));
        }
        let n = data[i..i + len_size].iter().fold(0usize, |a, &b| (a << 8) | usize::from(b));
        i += len_size;
        if i + n > data.len() {
            return Err(Error::Container("NAL unit runs past the sample".into()));
        }
        out.push(&data[i..i + n]);
        i += n;
    }
    Ok(out)
}

/// Append `nal` to `out` with a four-byte start code.
pub fn push_annex_b(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
}

/// Append `nal` to `out` with a four-byte length prefix.
pub fn push_prefixed(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
    out.extend_from_slice(nal);
}

/// A parsed decoder configuration record: prefix length and parameter sets
/// in record order.
#[derive(Debug, Clone)]
pub struct ParamSets {
    /// Bytes in each NAL length prefix.
    pub len_size: usize,
    /// Parameter set NAL units.
    pub sets: Vec<Vec<u8>>,
}

/// Strip a box header from a configuration record when one was kept.
fn body<'a>(raw: &'a [u8], fourcc: &[u8; 4]) -> &'a [u8] {
    if raw.len() >= 8 && &raw[4..8] == fourcc { &raw[8..] } else { raw }
}

fn take<'a>(raw: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8]> {
    let s = raw.get(*at..*at + n).ok_or_else(|| Error::Container("truncated decoder configuration record".into()))?;
    *at += n;
    Ok(s)
}

fn u16_at(raw: &[u8], at: &mut usize) -> Result<usize> {
    let s = take(raw, at, 2)?;
    Ok(usize::from(s[0]) << 8 | usize::from(s[1]))
}

/// Parse an `avcC` record.
pub fn parse_avcc(raw: &[u8]) -> Result<ParamSets> {
    let raw = body(raw, b"avcC");
    let mut at = 4;
    let len_size = usize::from(take(raw, &mut at, 1)?[0] & 3) + 1;
    let mut sets = Vec::new();
    let sps = take(raw, &mut at, 1)?[0] & 0x1f;
    for _ in 0..sps {
        let n = u16_at(raw, &mut at)?;
        sets.push(take(raw, &mut at, n)?.to_vec());
    }
    let pps = take(raw, &mut at, 1)?[0];
    for _ in 0..pps {
        let n = u16_at(raw, &mut at)?;
        sets.push(take(raw, &mut at, n)?.to_vec());
    }
    Ok(ParamSets { len_size, sets })
}

/// Parse an `hvcC` record.
pub fn parse_hvcc(raw: &[u8]) -> Result<ParamSets> {
    let raw = body(raw, b"hvcC");
    let mut at = 21;
    let len_size = usize::from(take(raw, &mut at, 1)?[0] & 3) + 1;
    let arrays = take(raw, &mut at, 1)?[0];
    let mut sets = Vec::new();
    for _ in 0..arrays {
        take(raw, &mut at, 1)?;
        let count = u16_at(raw, &mut at)?;
        for _ in 0..count {
            let n = u16_at(raw, &mut at)?;
            sets.push(take(raw, &mut at, n)?.to_vec());
        }
    }
    Ok(ParamSets { len_size, sets })
}

/// Build an `avcC` record (without box header) from one SPS and one PPS.
pub fn build_avcc(sps: &[u8], pps: &[u8]) -> Result<Vec<u8>> {
    if sps.len() < 4 {
        return Err(Error::Codec("the encoder produced a short sequence parameter set".into()));
    }
    let mut out = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
    out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    out.extend_from_slice(sps);
    out.push(1);
    out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    out.extend_from_slice(pps);
    if matches!(sps[1], 100 | 110 | 122 | 144) {
        // 4:2:0, 8-bit, no SPS extensions.
        out.extend_from_slice(&[0xfd, 0xf8, 0xf8, 0]);
    }
    Ok(out)
}

/// The H.264 NAL unit type.
pub fn h264_type(nal: &[u8]) -> u8 {
    nal.first().map_or(0, |b| b & 0x1f)
}

/// The H.265 NAL unit type.
pub fn h265_type(nal: &[u8]) -> u8 {
    nal.first().map_or(0, |b| (b >> 1) & 0x3f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_splits_three_and_four_byte_start_codes() {
        let data = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4, 0];
        let nals = split_annex_b(&data);
        assert_eq!(nals, vec![&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4, 0][..]]);
    }

    #[test]
    fn avcc_round_trips() {
        let sps = [0x67, 66, 0xc0, 30, 0xaa];
        let pps = [0x68, 0xce, 0x38];
        let rec = build_avcc(&sps, &pps).unwrap();
        let p = parse_avcc(&rec).unwrap();
        assert_eq!(p.len_size, 4);
        assert_eq!(p.sets, vec![sps.to_vec(), pps.to_vec()]);
    }

    #[test]
    fn prefixed_round_trips() {
        let mut s = Vec::new();
        push_prefixed(&mut s, &[1, 2, 3]);
        push_prefixed(&mut s, &[4]);
        assert_eq!(split_prefixed(&s, 4).unwrap(), vec![&[1, 2, 3][..], &[4][..]]);
        assert!(split_prefixed(&s[..5], 4).is_err());
    }
}
