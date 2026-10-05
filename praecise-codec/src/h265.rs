//! H.265 decoding (Main profile, 8-bit, 4:2:0).
//!
//! The decoder emits pictures in decode order; they are put in display
//! order here by picture order count, one coded video sequence (from one
//! IDR picture to the next) at a time.

use rust_h265::{Decoder, Frame, PixelData, parse_annex_b};

use crate::color::{Matrix, Planes};
use crate::nal::{h265_type, parse_hvcc, push_annex_b, split_prefixed};
use crate::{Demuxed, Error, Result, Sink, to_8bit};

const IDR_W_RADL: u8 = 19;
const IDR_N_LP: u8 = 20;

fn codec(e: impl std::fmt::Debug) -> Error {
    Error::Codec(format!("H.265: {e:?}"))
}

fn plane(p: &PixelData, depth: u8) -> std::borrow::Cow<'_, [u8]> {
    match p {
        PixelData::U8(v) => std::borrow::Cow::Borrowed(v.as_slice()),
        PixelData::U16(v) => std::borrow::Cow::Owned(to_8bit(v, depth)),
    }
}

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
    let mut dec = Decoder::new();
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
            if matches!(h265_type(unit), IDR_W_RADL | IDR_N_LP) && !held.is_empty() {
                drain(&mut held, sink, m)?;
                if sink.stopped() {
                    return Ok(());
                }
            }
            push_annex_b(&mut stream, unit);
        }
        for unit in parse_annex_b(&stream) {
            if let Some(f) = dec.decode_nal(&unit).map_err(codec)? {
                held.push(f);
            }
        }
    }
    if let Some(f) = dec.flush() {
        held.push(f);
    }
    drain(&mut held, sink, m)
}
