//! AV1: encoding through rav1e, decoding through rav1d.

use std::ptr::NonNull;

use rav1d::Dav1dResult;
use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::DAV1D_PIXEL_LAYOUT_I420;
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data};
use rav1e::prelude::{
    ChromaSampling, ColorDescription, ColorPrimaries, Config, Context, EncoderConfig, EncoderStatus, FrameType, MatrixCoefficients, PixelRange,
    Rational, TransferCharacteristics,
};

use crate::color::{Matrix, Planes, Yuv420};
use crate::{Demuxed, Error, Rate, Result, Sample, Sink, to_8bit};

const OBU_TEMPORAL_DELIMITER: u8 = 2;

fn codec(e: impl std::fmt::Debug) -> Error {
    Error::Codec(format!("AV1: {e:?}"))
}

/// The AV1 quantizer index (0..=255) for an H.264-scale quantizer.
fn quantizer(q: u8) -> usize {
    usize::from(q) * 255 / 51
}

pub(crate) struct Encoder {
    ctx: Context<u8>,
    width: usize,
    samples: Vec<Sample>,
}

impl Encoder {
    pub(crate) fn new(width: u32, height: u32, rate: Rate, q: u8) -> Result<Self> {
        let mut enc = EncoderConfig::with_speed_preset(10);
        enc.width = width as usize;
        enc.height = height as usize;
        enc.time_base = Rational::new(u64::from(rate.den), u64::from(rate.num));
        enc.bit_depth = 8;
        enc.chroma_sampling = ChromaSampling::Cs420;
        enc.pixel_range = PixelRange::Limited;
        enc.color_description = Some(ColorDescription {
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristics: TransferCharacteristics::BT709,
            matrix_coefficients: MatrixCoefficients::BT709,
        });
        // No frame reordering: one packet per frame, in order.
        enc.low_latency = true;
        enc.quantizer = quantizer(q);
        enc.min_quantizer = quantizer(q) as u8;
        enc.bitrate = 0;
        let gop = u64::from(rate.num.div_ceil(rate.den)) * 2;
        enc.max_key_frame_interval = gop.max(1);
        enc.min_key_frame_interval = gop.max(1);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
        let ctx = Config::new().with_encoder_config(enc).with_threads(threads).new_context().map_err(codec)?;
        Ok(Self { ctx, width: width as usize, samples: Vec::new() })
    }

    pub(crate) fn push(&mut self, yuv: &Yuv420) -> Result<()> {
        let cw = self.width.div_ceil(2);
        let mut f = self.ctx.new_frame();
        f.planes[0].copy_from_raw_u8(&yuv.y, self.width, 1);
        f.planes[1].copy_from_raw_u8(&yuv.u, cw, 1);
        f.planes[2].copy_from_raw_u8(&yuv.v, cw, 1);
        self.ctx.send_frame(f).map_err(codec)?;
        self.drain()
    }

    fn drain(&mut self) -> Result<()> {
        loop {
            match self.ctx.receive_packet() {
                Ok(p) => self.samples.push(Sample { data: strip_delimiters(&p.data)?, key: p.frame_type == FrameType::KEY }),
                Err(EncoderStatus::Encoded) => {}
                Err(EncoderStatus::NeedMoreData | EncoderStatus::LimitReached) => return Ok(()),
                Err(e) => return Err(codec(e)),
            }
        }
    }

    pub(crate) fn finish(mut self) -> Result<(Vec<u8>, Vec<Sample>)> {
        self.ctx.flush();
        self.drain()?;
        Ok((self.ctx.container_sequence_header(), self.samples))
    }
}

fn leb128(data: &[u8], at: &mut usize) -> Result<usize> {
    let mut v = 0usize;
    for i in 0..8 {
        let b = *data.get(*at).ok_or_else(|| codec("truncated OBU size"))?;
        *at += 1;
        v |= usize::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(codec("OBU size longer than 8 bytes"))
}

/// The OBUs of a temporal unit without temporal delimiters, which ISO
/// base media and Matroska samples leave out.
pub(crate) fn strip_delimiters(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len());
    let mut at = 0;
    while at < data.len() {
        let start = at;
        let header = data[at];
        let kind = (header >> 3) & 0xf;
        at += 1 + usize::from(header & 4 != 0);
        if header & 2 == 0 {
            return Err(codec("OBU without a size field"));
        }
        let n = leb128(data, &mut at)?;
        let end = at.checked_add(n).filter(|&e| e <= data.len()).ok_or_else(|| codec("OBU runs past the packet"))?;
        if kind != OBU_TEMPORAL_DELIMITER {
            out.extend_from_slice(&data[start..end]);
        }
        at = end;
    }
    Ok(out)
}

fn eagain() -> i32 {
    -libc::EAGAIN
}

/// An open rav1d decoder, closed on drop.
struct Dav1d(Option<Dav1dContext>);

impl Drop for Dav1d {
    fn drop(&mut self) {
        // SAFETY: the context came from `dav1d_open` and is closed once.
        unsafe { dav1d_close(NonNull::new(&raw mut self.0)) };
    }
}

fn check(r: Dav1dResult, what: &str) -> Result<()> {
    if r.0 == 0 { Ok(()) } else { Err(codec(format!("{what} failed ({})", r.0))) }
}

fn emit(pic: &Dav1dPicture, container: Option<Matrix>, sink: &mut Sink<'_>) -> Result<()> {
    if pic.p.layout != DAV1D_PIXEL_LAYOUT_I420 {
        return Err(Error::Request("AV1 video that is not 4:2:0 is not decoded".into()));
    }
    let m = match container {
        Some(m) => m,
        None => {
            let seq = pic.seq_hdr.ok_or_else(|| codec("picture without a sequence header"))?;
            // SAFETY: the header lives as long as the picture reference.
            let seq = unsafe { seq.as_ref() };
            Matrix::from_code(seq.mtrx, seq.color_range != 0).ok_or_else(|| Error::Request(format!("AV1 matrix coefficients {} are not decoded", seq.mtrx)))?
        }
    };
    let (w, h) = (pic.p.w as usize, pic.p.h as usize);
    let (cw, ch) = Yuv420::chroma_dims(w, h);
    let (sy, sc) = (pic.stride[0] as usize, pic.stride[1] as usize);
    let [Some(py), Some(pu), Some(pv)] = pic.data else {
        return Err(codec("picture without planes"));
    };
    if pic.p.bpc == 8 {
        // SAFETY: rav1d hands out planes of `stride * rows` bytes.
        let (y, u, v) = unsafe {
            (
                std::slice::from_raw_parts(py.as_ptr().cast::<u8>(), sy * (h - 1) + w),
                std::slice::from_raw_parts(pu.as_ptr().cast::<u8>(), sc * (ch - 1) + cw),
                std::slice::from_raw_parts(pv.as_ptr().cast::<u8>(), sc * (ch - 1) + cw),
            )
        };
        return sink.picture(Planes { y, u, v, strides: (sy, sc) }, w, h, m);
    }
    let depth = pic.p.bpc as u8;
    let rows = |p: NonNull<std::ffi::c_void>, stride: usize, pw: usize, ph: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(pw * ph);
        for r in 0..ph {
            // SAFETY: as above; strides are in bytes, samples are 16-bit.
            let row = unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>().add(r * stride).cast::<u16>(), pw) };
            out.extend(to_8bit(row, depth));
        }
        out
    };
    let (y, u, v) = (rows(py, sy, w, h), rows(pu, sc, cw, ch), rows(pv, sc, cw, ch));
    sink.picture(Planes { y: &y, u: &u, v: &v, strides: (w, cw) }, w, h, m)
}

/// Pull every picture rav1d has ready.
fn pull(c: Dav1dContext, container: Option<Matrix>, sink: &mut Sink<'_>) -> Result<()> {
    loop {
        let mut pic = Dav1dPicture::default();
        // SAFETY: `c` is open and `pic` is a valid output slot.
        let r = unsafe { dav1d_get_picture(Some(c), NonNull::new(&raw mut pic)) };
        if r.0 == eagain() {
            return Ok(());
        }
        check(r, "dav1d_get_picture")?;
        let out = emit(&pic, container, sink);
        // SAFETY: `pic` holds a reference from `dav1d_get_picture`.
        unsafe { dav1d_picture_unref(NonNull::new(&raw mut pic)) };
        out?;
        if sink.stopped() {
            return Ok(());
        }
    }
}

pub(crate) fn decode(d: &Demuxed, sink: &mut Sink<'_>) -> Result<()> {
    let config_obus = d.config.get(4..).unwrap_or_default();
    let mut s = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
    // SAFETY: `dav1d_default_settings` writes every field.
    let mut s = unsafe {
        dav1d_default_settings(NonNull::new(s.as_mut_ptr()).expect("non-null"));
        s.assume_init()
    };
    s.max_frame_delay = 1;
    let mut dec = Dav1d(None);
    // SAFETY: both pointers are valid for the call.
    check(unsafe { dav1d_open(NonNull::new(&raw mut dec.0), NonNull::new(&raw mut s)) }, "dav1d_open")?;
    let c = dec.0.ok_or_else(|| codec("dav1d_open returned no context"))?;
    for (i, sample) in d.samples.iter().enumerate() {
        let mut tu = Vec::with_capacity(sample.len() + config_obus.len());
        if i == 0 {
            tu.extend_from_slice(config_obus);
        }
        tu.extend_from_slice(sample);
        let mut data = Dav1dData::default();
        // SAFETY: `data_create` allocates `tu.len()` bytes owned by `data`.
        let buf = unsafe { dav1d_data_create(NonNull::new(&raw mut data), tu.len()) };
        if buf.is_null() {
            return Err(codec("dav1d_data_create failed"));
        }
        // SAFETY: `buf` has room for `tu.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(tu.as_ptr(), buf, tu.len()) };
        while data.sz > 0 {
            // SAFETY: `c` is open; `data` is a live reference.
            let r = unsafe { dav1d_send_data(Some(c), NonNull::new(&raw mut data)) };
            if r.0 != 0 && r.0 != eagain() {
                // SAFETY: release the unconsumed data.
                unsafe { dav1d_data_unref(NonNull::new(&raw mut data)) };
                return Err(codec(format!("dav1d_send_data failed ({})", r.0)));
            }
            pull(c, d.matrix, sink)?;
            if sink.stopped() {
                // SAFETY: as above.
                unsafe { dav1d_data_unref(NonNull::new(&raw mut data)) };
                return Ok(());
            }
        }
    }
    pull(c, d.matrix, sink)
}
