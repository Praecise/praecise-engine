//! H.265 decoding (Main profile, 8-bit, 4:2:0).
//!
//! The decoder emits pictures in decode order; they are put in display
//! order here by picture order count, one coded video sequence (from one
//! IDR picture to the next) at a time.

use rust_h265::{DecodeLimits, Decoder, Frame, NalUnitType, PixelData, parse_annex_b};

use crate::color::{Matrix, Planes};
use crate::nal::{parse_hvcc, push_annex_b, split_prefixed};
use crate::{Demuxed, Error, Result, Sink, to_8bit};

fn codec(e: impl std::fmt::Debug) -> Error {
    Error::Codec(format!("H.265: {e:?}"))
}

fn plane(p: &PixelData, depth: u8) -> std::borrow::Cow<'_, [u8]> {
    match p {
        PixelData::U8(v) => std::borrow::Cow::Borrowed(v.as_slice()),
        PixelData::U16(v) => std::borrow::Cow::Owned(to_8bit(v, depth)),
    }
}

/// Chroma format index of 4:2:0, the only layout the sink takes.
const CHROMA_420: u32 = 1;

fn drain(held: &mut Vec<Frame>, sink: &mut Sink<'_>, m: Matrix) -> Result<()> {
    held.sort_by_key(|f| f.pic_order_cnt);
    for f in held.drain(..) {
        let (w, h) = (f.width as usize, f.height as usize);
        let (y, u, v) = (plane(&f.y, f.bit_depth), plane(&f.u, f.bit_depth), plane(&f.v, f.bit_depth));
        sink.picture(Planes { y: &y, u: &u, v: &v, strides: (w, w.div_ceil(2)) }, w, h, m)?;
        if sink.stopped() {
            break;
        }
    }
    Ok(())
}

pub(crate) fn decode(d: &Demuxed, sink: &mut Sink<'_>) -> Result<()> {
    let cfg = parse_hvcc(&d.config)?;
    let m = d.matrix.unwrap_or(Matrix::BT709);
    // No picture size or memory cap, as before the decoder gained default limits.
    let mut dec = Decoder::with_limits(DecodeLimits { max_picture_pixels: u64::MAX, max_picture_memory_bytes: u64::MAX });
    let mut held: Vec<Frame> = Vec::new();
    let mut stream = Vec::new();
    for (i, s) in d.samples.iter().enumerate() {
        stream.clear();
        if i == 0 {
            for set in &cfg.sets {
                push_annex_b(&mut stream, set);
            }
        }
        for unit in split_prefixed(s, cfg.len_size)? {
            push_annex_b(&mut stream, unit);
        }
        for unit in parse_annex_b(&stream) {
            // A picture is returned when the next one starts, so the last
            // picture of the previous sequence is in hand once the IDR
            // picture's first unit has been decoded.
            if let Some(f) = dec.decode_nal(&unit).map_err(codec)? {
                if f.chroma_format_idc != CHROMA_420 {
                    return Err(codec("unsupported: only 4:2:0 streams are decoded"));
                }
                held.push(f);
            }
            if matches!(unit.nal_unit_type, NalUnitType::IdrWRadl | NalUnitType::IdrNLp) && !held.is_empty() {
                drain(&mut held, sink, m)?;
                if sink.stopped() {
                    return Ok(());
                }
            }
        }
    }
    if let Some(f) = dec.flush() {
        if f.chroma_format_idc != CHROMA_420 {
            return Err(codec("unsupported: only 4:2:0 streams are decoded"));
        }
        held.push(f);
    }
    drain(&mut held, sink, m)
}
