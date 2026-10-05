//! H.264 through OpenH264.

use openh264::OpenH264API;
use openh264::decoder::{DecodeOptions, Decoder, Flush};
use openh264::encoder::{BitRate, EncoderConfig, FrameRate, IntraFramePeriod, QpRange, RateControlMode, VuiConfig};
use openh264::formats::{YUVSlices, YUVSource};

use crate::color::{Matrix, Planes, Yuv420};
use crate::nal::{build_avcc, h264_type, parse_avcc, push_annex_b, push_prefixed, split_annex_b, split_prefixed};
use crate::{Demuxed, Error, Result, Sample, Sink};

const SPS: u8 = 7;
const PPS: u8 = 8;
const IDR: u8 = 5;
/// Access unit delimiter, SEI: not carried in samples.
const DROPPED: [u8; 2] = [9, 6];

fn codec(e: impl std::fmt::Display) -> Error {
    Error::Codec(format!("H.264: {e}"))
}

pub(crate) struct Encoder {
    enc: openh264::encoder::Encoder,
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
    samples: Vec<Sample>,
}

impl Encoder {
    pub(crate) fn new(width: u32, height: u32, fps: f32, quantizer: u8) -> Result<Self> {
        // The rate controller only bounds the quantizer here; the target
        // is set far above what any quantizer in range produces.
        let bps = (u64::from(width) * u64::from(height) * 24).min(u64::from(i32::MAX as u32)) as u32;
        let cfg = EncoderConfig::new()
            .bitrate(BitRate::from_bps(bps))
            .max_frame_rate(FrameRate::from_hz(fps))
            .rate_control_mode(RateControlMode::Quality)
            .qp(QpRange::new(quantizer, quantizer))
            .skip_frames(false)
            .intra_frame_period(IntraFramePeriod::from_num_frames((fps * 2.0).round().max(1.0) as u32))
            .vui(VuiConfig::bt709());
        let enc = openh264::encoder::Encoder::with_api_config(OpenH264API::from_source(), cfg).map_err(codec)?;
        Ok(Self { enc, sps: None, pps: None, samples: Vec::new() })
    }

    pub(crate) fn push(&mut self, yuv: &Yuv420) -> Result<()> {
        let cw = yuv.width.div_ceil(2);
        let src = YUVSlices::new((&yuv.y, &yuv.u, &yuv.v), (yuv.width, yuv.height), (yuv.width, cw, cw));
        let bs = self.enc.encode(&src).map_err(codec)?;
        let mut data = Vec::new();
        let mut key = false;
        for l in 0..bs.num_layers() {
            let layer = bs.layer(l).ok_or_else(|| codec("missing layer"))?;
            for n in 0..layer.nal_count() {
                let unit = layer.nal_unit(n).ok_or_else(|| codec("missing NAL unit"))?;
                for nal in split_annex_b(unit) {
                    match h264_type(nal) {
                        SPS => keep_once(&mut self.sps, nal, "sequence")?,
                        PPS => keep_once(&mut self.pps, nal, "picture")?,
                        t if DROPPED.contains(&t) => {}
                        t => {
                            key |= t == IDR;
                            push_prefixed(&mut data, nal);
                        }
                    }
                }
            }
        }
        if data.is_empty() {
            return Err(codec("the encoder skipped a frame"));
        }
        self.samples.push(Sample { data, key });
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<(Vec<u8>, Vec<Sample>)> {
        let (Some(sps), Some(pps)) = (self.sps, self.pps) else {
            return Err(codec("the encoder wrote no parameter sets"));
        };
        Ok((build_avcc(&sps, &pps)?, self.samples))
    }
}

/// Keep the first copy of a parameter set; a different later one would
/// need a second sample description, which is refused.
fn keep_once(slot: &mut Option<Vec<u8>>, nal: &[u8], what: &str) -> Result<()> {
    match slot {
        None => *slot = Some(nal.to_vec()),
        Some(s) if s.as_slice() == nal => {}
        Some(_) => return Err(codec(format!("the encoder changed its {what} parameter set mid-stream"))),
    }
    Ok(())
}

fn emit(sink: &mut Sink<'_>, p: &impl YUVSource, m: Matrix) -> Result<()> {
    let (w, h) = p.dimensions();
    let (sy, su, _) = p.strides();
    sink.picture(Planes { y: p.y(), u: p.u(), v: p.v(), strides: (sy, su) }, w, h, m)
}

pub(crate) fn decode(d: &Demuxed, sink: &mut Sink<'_>) -> Result<()> {
    let cfg = parse_avcc(&d.config)?;
    let m = d.matrix.unwrap_or(Matrix::BT709);
    let mut dec = Decoder::new().map_err(codec)?;
    let mut packet = Vec::new();
    for (i, s) in d.samples.iter().enumerate() {
        packet.clear();
        if i == 0 {
            for set in &cfg.sets {
                push_annex_b(&mut packet, set);
            }
        }
        for unit in split_prefixed(s, cfg.len_size)? {
            push_annex_b(&mut packet, unit);
        }
        if let Some(p) = dec.decode_with_options(&packet, DecodeOptions::new().flush_after_decode(Flush::NoFlush)).map_err(codec)? {
            emit(sink, &p, m)?;
        }
        if sink.stopped() {
            return Ok(());
        }
    }
    for p in dec.flush_remaining().map_err(codec)? {
        emit(sink, &p, m)?;
    }
    Ok(())
}
