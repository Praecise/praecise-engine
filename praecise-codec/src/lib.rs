//! In-process video coding for Praecise Engine.
//!
//! Decoding reads MP4 (ISO base media) and Matroska/WebM files holding
//! H.264, H.265 or AV1 video and returns 8-bit RGB frames in display order.
//! Encoding writes 8-bit RGB frames as H.264 or AV1 in MP4, or AV1 in WebM,
//! with an optional Opus soundtrack. Everything runs in the calling process;
//! no external program or system library is involved.
//!
//! Colour: frames are written as BT.709 studio-range 4:2:0 and signalled so
//! in the bitstream and the container. Decoding uses the matrix and range
//! the container or the AV1 sequence header signal, and BT.709 studio range
//! when neither does.

mod av1;
pub mod color;
mod h264;
mod h265;
mod mp4;
pub mod nal;
mod sound;
mod webm;

use color::Matrix;

/// Errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file's container is malformed or not one this crate reads.
    #[error("container: {0}")]
    Container(String),
    /// The bitstream could not be coded.
    #[error("codec: {0}")]
    Codec(String),
    /// A request this crate refuses (a size, codec or combination).
    #[error("{0}")]
    Request(String),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Video compression formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC (decode only).
    H265,
    /// AV1.
    Av1,
}

/// File formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// ISO base media file format (`.mp4`).
    Mp4,
    /// WebM (`.webm`); decoding also reads other Matroska files.
    WebM,
}

impl Container {
    /// The media type of a file in this container.
    #[must_use]
    pub fn mime(self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::WebM => "video/webm",
        }
    }

    /// The file name extension, without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::WebM => "webm",
        }
    }
}

/// How to encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Video format: H.264 or AV1.
    pub codec: VideoCodec,
    /// File format: MP4 (H.264 or AV1) or WebM (AV1).
    pub container: Container,
    /// Constant quantizer on the H.264 scale, 0 (best) to 51.
    pub quantizer: u8,
}

impl Default for EncodeOptions {
    /// H.264 in MP4 at quantizer 20 (visually lossless for generated video).
    fn default() -> Self {
        Self { codec: VideoCodec::H264, container: Container::Mp4, quantizer: 20 }
    }
}

impl EncodeOptions {
    fn check(&self, width: u32, height: u32, fps: f32) -> Result<()> {
        match (self.codec, self.container) {
            (VideoCodec::H264 | VideoCodec::Av1, Container::Mp4) | (VideoCodec::Av1, Container::WebM) => {}
            (VideoCodec::H265, _) => return Err(Error::Request("H.265 is decoded, not encoded".into())),
            (VideoCodec::H264, Container::WebM) => return Err(Error::Request("WebM carries AV1, not H.264".into())),
        }
        if self.quantizer > 51 {
            return Err(Error::Request(format!("quantizer {} is above 51", self.quantizer)));
        }
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 || width > 8192 || height > 8192 {
            return Err(Error::Request(format!("{width}x{height}: width and height must be even and 2..=8192")));
        }
        if !(fps.is_finite() && (0.1..=480.0).contains(&fps)) {
            return Err(Error::Request(format!("frame rate {fps} is outside 0.1..=480")));
        }
        Ok(())
    }
}

/// A soundtrack to encode beside the frames.
#[derive(Debug, Clone, Copy)]
pub struct Sound<'a> {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channels: 1 or 2.
    pub channels: u32,
    /// Samples in -1..=1, channel-major: all of channel 0, then channel 1.
    pub samples: &'a [f32],
}

/// A frame rate as `num / den` frames per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rate {
    pub num: u32,
    pub den: u32,
}

impl Rate {
    pub(crate) fn from_fps(fps: f32) -> Self {
        let (mut num, mut den) = ((f64::from(fps) * 1000.0).round() as u32, 1000u32);
        let g = gcd(num, den);
        num /= g;
        den /= g;
        Self { num, den }
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// An encoded picture.
#[derive(Debug, Clone)]
pub(crate) struct Sample {
    /// Bitstream, in the container's framing (length-prefixed NAL units for
    /// H.264, OBUs without temporal delimiters for AV1).
    pub data: Vec<u8>,
    /// Decodable on its own.
    pub key: bool,
}

/// An encoded video track.
#[derive(Debug, Clone)]
pub(crate) struct Track {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub rate: Rate,
    /// `avcC` or `av1C` record, without box header.
    pub config: Vec<u8>,
    pub samples: Vec<Sample>,
}

/// An encoder per codec.
enum Coder {
    H264(h264::Encoder),
    Av1(av1::Encoder),
}

/// Encodes frames one at a time and writes the file at the end.
pub struct VideoWriter {
    opts: EncodeOptions,
    width: u32,
    height: u32,
    rate: Rate,
    coder: Coder,
    frames: u32,
}

impl std::fmt::Debug for VideoWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoWriter").field("opts", &self.opts).field("width", &self.width).field("height", &self.height).field("frames", &self.frames).finish_non_exhaustive()
    }
}

impl VideoWriter {
    /// A writer for `width` x `height` frames at `fps`.
    ///
    /// # Errors
    /// An unsupported codec/container pair, an odd or oversized picture, a
    /// frame rate out of range, or an encoder that fails to start.
    pub fn new(opts: EncodeOptions, width: u32, height: u32, fps: f32) -> Result<Self> {
        opts.check(width, height, fps)?;
        let rate = Rate::from_fps(fps);
        let coder = match opts.codec {
            VideoCodec::H264 => Coder::H264(h264::Encoder::new(width, height, fps, opts.quantizer)?),
            VideoCodec::Av1 => Coder::Av1(av1::Encoder::new(width, height, rate, opts.quantizer)?),
            VideoCodec::H265 => unreachable!("refused by check"),
        };
        Ok(Self { opts, width, height, rate, coder, frames: 0 })
    }

    /// Frames pushed so far.
    #[must_use]
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Encode one row-major 8-bit RGB frame.
    ///
    /// # Errors
    /// A frame of the wrong size, or an encoder failure.
    pub fn push(&mut self, rgb: &[u8]) -> Result<()> {
        let (w, h) = (self.width as usize, self.height as usize);
        if rgb.len() != w * h * 3 {
            return Err(Error::Request(format!("frame of {} bytes, expected {}", rgb.len(), w * h * 3)));
        }
        let yuv = color::Yuv420::from_rgb(rgb, w, h, Matrix::BT709);
        match &mut self.coder {
            Coder::H264(e) => e.push(&yuv)?,
            Coder::Av1(e) => e.push(&yuv)?,
        }
        self.frames += 1;
        Ok(())
    }

    /// Finish the stream and write the file, with `sound` as its
    /// soundtrack when given.
    ///
    /// # Errors
    /// No frames, a soundtrack this crate cannot encode, or an encoder
    /// failure.
    pub fn finish(self, sound: Option<Sound<'_>>) -> Result<Vec<u8>> {
        if self.frames == 0 {
            return Err(Error::Request("a video needs at least one frame".into()));
        }
        let (config, samples) = match self.coder {
            Coder::H264(e) => e.finish()?,
            Coder::Av1(e) => e.finish()?,
        };
        if samples.len() != self.frames as usize {
            return Err(Error::Codec(format!("{} frames in, {} pictures out", self.frames, samples.len())));
        }
        let track = Track { codec: self.opts.codec, width: self.width, height: self.height, rate: self.rate, config, samples };
        let sound = sound.map(sound::encode).transpose()?;
        Ok(match self.opts.container {
            Container::Mp4 => mp4::write(&track, sound.as_ref()),
            Container::WebM => webm::write(&track, sound.as_ref()),
        })
    }
}

/// Encode frames (row-major 8-bit RGB, one after another) with an
/// optional soundtrack.
///
/// # Errors
/// As [`VideoWriter::new`], [`VideoWriter::push`] and
/// [`VideoWriter::finish`]; also a buffer that is not a whole number of
/// frames.
pub fn encode(opts: EncodeOptions, width: u32, height: u32, fps: f32, rgb: &[u8], sound: Option<Sound<'_>>) -> Result<Vec<u8>> {
    let frame = width as usize * height as usize * 3;
    if frame == 0 || rgb.len() % frame != 0 {
        return Err(Error::Request(format!("{} bytes is not a whole number of {width}x{height} RGB frames", rgb.len())));
    }
    let mut w = VideoWriter::new(opts, width, height, fps)?;
    for f in rgb.chunks_exact(frame) {
        w.push(f)?;
    }
    w.finish(sound)
}

/// What a video file holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    /// File format.
    pub container: Container,
    /// Video format.
    pub codec: VideoCodec,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Coded pictures in the track.
    pub frames: u32,
    /// Mean frame rate.
    pub fps: f32,
}

/// The video track of a file, demultiplexed.
pub(crate) struct Demuxed {
    pub info: VideoInfo,
    /// `avcC`, `hvcC` or `av1C` record.
    pub config: Vec<u8>,
    /// Samples in decode order.
    pub samples: Vec<Vec<u8>>,
    /// Colour signalled by the container.
    pub matrix: Option<Matrix>,
}

fn demux(bytes: &[u8]) -> Result<Demuxed> {
    if bytes.len() >= 8 && &bytes[4..8] == b"ftyp" {
        mp4::read(bytes)
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        webm::read(bytes)
    } else {
        Err(Error::Container("not an MP4 or Matroska/WebM file".into()))
    }
}

/// The container, codec, size, frame count and rate of a video file,
/// without decoding it.
///
/// # Errors
/// A file that is not MP4 or Matroska/WebM, or has no H.264, H.265 or AV1
/// video track.
pub fn probe(bytes: &[u8]) -> Result<VideoInfo> {
    Ok(demux(bytes)?.info)
}

/// Decode every frame of a video file in display order, calling `frame`
/// with its index and its row-major 8-bit RGB pixels; `frame` returns
/// `false` to stop early. Returns the file's description.
///
/// # Errors
/// As [`probe`]; also a bitstream that fails to decode, or one whose
/// decoded frame count or size differs from what the container declares.
pub fn decode_each(bytes: &[u8], mut frame: impl FnMut(u32, &[u8]) -> bool) -> Result<VideoInfo> {
    let d = demux(bytes)?;
    let info = d.info;
    let mut sink = Sink { info, index: 0, rgb: Vec::new(), stop: false, f: &mut frame };
    match info.codec {
        VideoCodec::H264 => h264::decode(&d, &mut sink)?,
        VideoCodec::H265 => h265::decode(&d, &mut sink)?,
        VideoCodec::Av1 => av1::decode(&d, &mut sink)?,
    }
    if !sink.stop && sink.index != info.frames {
        return Err(Error::Codec(format!("{} frames decoded, the container declares {}", sink.index, info.frames)));
    }
    Ok(info)
}

/// A decoded video.
#[derive(Debug, Clone)]
pub struct DecodedVideo {
    /// What the file holds.
    pub info: VideoInfo,
    /// Frames one after another, each row-major 8-bit RGB.
    pub rgb: Vec<u8>,
}

/// Decode every frame of a video file.
///
/// # Errors
/// As [`decode_each`].
pub fn decode(bytes: &[u8]) -> Result<DecodedVideo> {
    let mut rgb = Vec::new();
    let info = decode_each(bytes, |_, f| {
        rgb.extend_from_slice(f);
        true
    })?;
    Ok(DecodedVideo { info, rgb })
}

/// Receives decoded pictures, converts them to RGB and hands them on.
pub(crate) struct Sink<'a> {
    info: VideoInfo,
    index: u32,
    rgb: Vec<u8>,
    stop: bool,
    f: &'a mut dyn FnMut(u32, &[u8]) -> bool,
}

impl Sink<'_> {
    /// Whether the caller asked to stop.
    pub(crate) fn stopped(&self) -> bool {
        self.stop
    }

    /// Hand on one decoded picture.
    pub(crate) fn picture(&mut self, p: color::Planes<'_>, width: usize, height: usize, m: Matrix) -> Result<()> {
        if self.stop {
            return Ok(());
        }
        if (width, height) != (self.info.width as usize, self.info.height as usize) {
            return Err(Error::Codec(format!("a {width}x{height} picture in a {}x{} track", self.info.width, self.info.height)));
        }
        self.rgb.clear();
        color::write_rgb(p, width, height, m, &mut self.rgb);
        let keep = (self.f)(self.index, &self.rgb);
        self.index += 1;
        self.stop = !keep;
        Ok(())
    }
}

/// Reduce samples above 8 bits to 8 bits, rounding.
pub(crate) fn to_8bit(src: &[u16], depth: u8) -> Vec<u8> {
    let shift = u32::from(depth.saturating_sub(8));
    let half = if shift == 0 { 0 } else { 1u32 << (shift - 1) };
    src.iter().map(|&s| ((u32::from(s) + half) >> shift).min(255) as u8).collect()
}
