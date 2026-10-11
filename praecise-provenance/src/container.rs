//! Where a manifest store lives in each file format, and the hard binding
//! that ties the store to the rest of the file.
//!
//! - PNG: a `caBX` chunk right after `IHDR`; bound by a data hash that
//!   excludes the whole chunk, so the hashed bytes are the original file.
//! - WAV (RIFF): a `C2PA` chunk at the end of the RIFF chunk; bound by a data
//!   hash that excludes the chunk (header and data).
//! - MP4 (ISO BMFF): a top-level `uuid` box right after `ftyp`, before
//!   `moov` and `mdat`; bound by a BMFF hash that excludes `ftyp`, `mfra` and
//!   the store box and hashes each other top-level box after its file offset.
//!   Inserting the box moves the media data, so chunk offsets (`stco`,
//!   `co64`) are moved with it, and moved back when the store is removed.

use std::ops::Range;

use sha2::{Digest, Sha256};

use crate::cbor::Value;
use crate::{Error, Result};

/// The extended type of the BMFF box holding a manifest store.
pub(crate) const BMFF_UUID: [u8; 16] = [
    0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c, 0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81,
];
const BMFF_PURPOSE: &[u8] = b"manifest\0";
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

/// The file formats a manifest can be embedded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    Png,
    Wav,
    Mp4,
}

impl Format {
    pub(crate) fn of(mime: &str) -> Result<Self> {
        match mime {
            "image/png" => Ok(Self::Png),
            "audio/wav" | "audio/wave" | "audio/x-wav" | "audio/vnd.wave" => Ok(Self::Wav),
            "video/mp4" | "audio/mp4" => Ok(Self::Mp4),
            other => Err(Error::Unsupported(format!(
                "no manifest embedding for {other}"
            ))),
        }
    }
}

/// Where a store sits in a file.
#[derive(Debug, Clone)]
pub(crate) struct Found {
    /// The store bytes.
    pub store: Range<usize>,
    /// The chunk or box holding them, as a data hash excludes it.
    pub holder: Range<usize>,
}

/// Places `store` in `asset`, which must not hold one already.
pub(crate) fn place(format: Format, asset: &[u8], store: &[u8]) -> Result<Vec<u8>> {
    if find(format, asset)?.is_some() {
        return Err(Error::AlreadySigned);
    }
    match format {
        Format::Png => {
            let at = png_after_ihdr(asset)?;
            let len = u32::try_from(store.len())
                .ok()
                .filter(|l| *l < 1 << 31)
                .ok_or_else(|| Error::Unsupported("manifest too large for a PNG chunk".into()))?;
            let mut chunk = len.to_be_bytes().to_vec();
            chunk.extend_from_slice(b"caBX");
            chunk.extend_from_slice(store);
            let crc = crc32(&chunk[4..]);
            chunk.extend_from_slice(&crc.to_be_bytes());
            Ok([&asset[..at], &chunk, &asset[at..]].concat())
        }
        Format::Wav => {
            riff_chunks(asset)?;
            let len = u32::try_from(store.len())
                .map_err(|_| Error::Unsupported("manifest too large for a RIFF chunk".into()))?;
            let mut out = asset.to_vec();
            out.extend_from_slice(b"C2PA");
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(store);
            if store.len() % 2 == 1 {
                out.push(0);
            }
            let riff = u32::try_from(out.len() - 8)
                .map_err(|_| Error::Unsupported("RIFF file above 4 GiB".into()))?;
            out[4..8].copy_from_slice(&riff.to_le_bytes());
            Ok(out)
        }
        Format::Mp4 => {
            let boxes = bmff_boxes(asset)?;
            let Some(ftyp) = boxes.first().filter(|b| &b.kind == b"ftyp") else {
                return Err(Error::Unsupported("MP4 does not start with ftyp".into()));
            };
            let at = ftyp.range.end;
            let mut holder = Vec::with_capacity(8 + 16 + 4 + BMFF_PURPOSE.len() + 8 + store.len());
            let size = u32::try_from(holder.capacity())
                .map_err(|_| Error::Unsupported("manifest too large for a BMFF box".into()))?;
            holder.extend_from_slice(&size.to_be_bytes());
            holder.extend_from_slice(b"uuid");
            holder.extend_from_slice(&BMFF_UUID);
            holder.extend_from_slice(&[0; 4]);
            holder.extend_from_slice(BMFF_PURPOSE);
            holder.extend_from_slice(&0u64.to_be_bytes());
            holder.extend_from_slice(store);
            let mut out = [&asset[..at], &holder, &asset[at..]].concat();
            shift_chunk_offsets(&mut out, at as u64, holder.len() as i64)?;
            Ok(out)
        }
    }
}

/// The store in `bytes`, if any.
pub(crate) fn find(format: Format, bytes: &[u8]) -> Result<Option<Found>> {
    match format {
        Format::Png => {
            let mut found = None;
            for (kind, data, whole) in png_chunks(bytes)? {
                if &kind == b"caBX" {
                    if found.is_some() {
                        return Err(Error::Malformed("PNG holds two manifest stores".into()));
                    }
                    found = Some(Found {
                        store: data,
                        holder: whole,
                    });
                }
            }
            Ok(found)
        }
        Format::Wav => {
            let chunks = riff_chunks(bytes)?;
            let n = chunks.len();
            let mut found = None;
            for (i, (kind, data, whole)) in chunks.into_iter().enumerate() {
                if &kind == b"C2PA" {
                    if i + 1 != n {
                        return Err(Error::Malformed(
                            "the C2PA chunk is not at the end of the RIFF chunk".into(),
                        ));
                    }
                    found = Some(Found {
                        store: data,
                        holder: whole,
                    });
                }
            }
            Ok(found)
        }
        Format::Mp4 => {
            let mut found = None;
            for b in bmff_boxes(bytes)? {
                if b.is_store(bytes) {
                    if found.is_some() {
                        return Err(Error::Malformed("MP4 holds two manifest stores".into()));
                    }
                    let head = b.range.start + b.header + 16 + 4 + BMFF_PURPOSE.len() + 8;
                    if head > b.range.end
                        || bytes[b.range.start + b.header + 20
                            ..b.range.start + b.header + 20 + BMFF_PURPOSE.len()]
                            != *BMFF_PURPOSE
                    {
                        return Err(Error::Malformed(
                            "the MP4 manifest box is not a manifest".into(),
                        ));
                    }
                    found = Some(Found {
                        store: head..b.range.end,
                        holder: b.range.clone(),
                    });
                }
            }
            Ok(found)
        }
    }
}

/// `bytes` without its store: the inverse of [`place`].
pub(crate) fn remove(format: Format, bytes: &[u8]) -> Result<Vec<u8>> {
    let found = find(format, bytes)?.ok_or(Error::NotSigned)?;
    let h = found.holder;
    match format {
        Format::Png => Ok([&bytes[..h.start], &bytes[h.end..]].concat()),
        Format::Wav => {
            let mut out = bytes[..h.start].to_vec();
            let riff =
                u32::try_from(out.len() - 8).map_err(|_| Error::Malformed("RIFF size".into()))?;
            out[4..8].copy_from_slice(&riff.to_le_bytes());
            Ok(out)
        }
        Format::Mp4 => {
            let mut out = [&bytes[..h.start], &bytes[h.end..]].concat();
            shift_chunk_offsets(&mut out, h.start as u64, -((h.end - h.start) as i64))?;
            Ok(out)
        }
    }
}

/// The hard-binding assertion's label for a format.
pub(crate) fn binding_label(format: Format) -> &'static str {
    match format {
        Format::Png | Format::Wav => "c2pa.hash.data",
        Format::Mp4 => "c2pa.hash.bmff.v3",
    }
}

/// The hard-binding assertion for a file laid out as `bytes`, whose store
/// is at `found`.
pub(crate) fn binding(format: Format, bytes: &[u8], found: &Found) -> Result<Value> {
    match format {
        Format::Png | Format::Wav => {
            let h = &found.holder;
            Ok(Value::map([
                (
                    "exclusions",
                    Value::Array(vec![Value::map([
                        ("start", Value::Uint(h.start as u64)),
                        ("length", Value::Uint((h.end - h.start) as u64)),
                    ])]),
                ),
                ("name", Value::text("jumbf manifest")),
                ("alg", Value::text("sha256")),
                (
                    "hash",
                    Value::Bytes(data_hash(bytes, std::slice::from_ref(h)).to_vec()),
                ),
                ("pad", Value::Bytes(Vec::new())),
            ]))
        }
        Format::Mp4 => {
            let exclusions = default_bmff_exclusions();
            let hash = bmff_hash(bytes, &exclusions)?;
            Ok(Value::map([
                (
                    "exclusions",
                    Value::Array(exclusions.iter().map(BmffExclusion::to_cbor).collect()),
                ),
                ("name", Value::text("jumbf manifest")),
                ("alg", Value::text("sha256")),
                ("hash", Value::Bytes(hash.to_vec())),
            ]))
        }
    }
}

/// Checks a hard-binding assertion against `bytes`. Every exclusion must be
/// the store's own holder, so no other byte escapes the hash.
pub(crate) fn check_binding(
    format: Format,
    label: &str,
    assertion: &Value,
    bytes: &[u8],
    found: &Found,
) -> Result<()> {
    let mismatch = |m: &str| Error::Mismatch(m.to_string());
    if let Some(alg) = assertion.get("alg").and_then(Value::as_text)
        && alg != "sha256"
    {
        return Err(Error::Unsupported(format!("hash algorithm {alg}")));
    }
    let expected = assertion
        .get("hash")
        .and_then(Value::as_bytes)
        .ok_or_else(|| Error::Malformed("hard binding without a hash".into()))?;
    let actual = match (format, label) {
        (Format::Png | Format::Wav, "c2pa.hash.data") => {
            let list = assertion
                .get("exclusions")
                .and_then(Value::as_array)
                .unwrap_or_default();
            let mut ranges = Vec::new();
            for e in list {
                let start = e
                    .get("start")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| Error::Malformed("exclusion without a start".into()))?;
                let length = e
                    .get("length")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| Error::Malformed("exclusion without a length".into()))?;
                let r = usize::try_from(start)
                    .ok()
                    .zip(usize::try_from(length).ok())
                    .map(|(s, l)| s..s.saturating_add(l))
                    .ok_or_else(|| mismatch("exclusion out of range"))?;
                if r != found.holder {
                    return Err(mismatch(
                        "a data hash excludes bytes other than the manifest store",
                    ));
                }
                ranges.push(r);
            }
            if ranges.len() != 1 {
                return Err(mismatch(
                    "a data hash must exclude exactly the manifest store",
                ));
            }
            data_hash(bytes, &ranges)
        }
        (Format::Mp4, "c2pa.hash.bmff.v2" | "c2pa.hash.bmff.v3") => {
            if assertion.get("merkle").is_some() {
                return Err(Error::Unsupported("BMFF Merkle hashes".into()));
            }
            let list = assertion
                .get("exclusions")
                .and_then(Value::as_array)
                .unwrap_or_default();
            let exclusions = list
                .iter()
                .map(BmffExclusion::from_cbor)
                .collect::<Result<Vec<_>>>()?;
            for b in bmff_boxes(bytes)? {
                if b.is_store(bytes) && !exclusions.iter().any(|e| e.matches(&b, bytes)) {
                    return Err(mismatch(
                        "the BMFF hash does not exclude the manifest store",
                    ));
                }
            }
            bmff_hash(bytes, &exclusions)?
        }
        _ => return Err(Error::Unsupported(format!("{label} for this file format"))),
    };
    if expected == actual {
        Ok(())
    } else {
        Err(mismatch("the file does not match its hard binding"))
    }
}

fn data_hash(bytes: &[u8], exclude: &[Range<usize>]) -> [u8; 32] {
    let mut h = Sha256::new();
    let mut at = 0;
    let mut sorted = exclude.to_vec();
    sorted.sort_by_key(|r| r.start);
    for r in sorted {
        if r.start > at {
            h.update(&bytes[at..r.start.min(bytes.len())]);
        }
        at = at.max(r.end);
    }
    if at < bytes.len() {
        h.update(&bytes[at..]);
    }
    h.finalize().into()
}

// BMFF.

#[derive(Debug)]
struct BmffBox {
    kind: [u8; 4],
    range: Range<usize>,
    header: usize,
}

impl BmffBox {
    fn is_store(&self, bytes: &[u8]) -> bool {
        &self.kind == b"uuid"
            && bytes.get(self.range.start + self.header..self.range.start + self.header + 16)
                == Some(&BMFF_UUID[..])
    }
}

fn bmff_boxes(bytes: &[u8]) -> Result<Vec<BmffBox>> {
    bmff_children(bytes, 0..bytes.len())
}

fn bmff_children(bytes: &[u8], within: Range<usize>) -> Result<Vec<BmffBox>> {
    let bad = || Error::Malformed("malformed BMFF box".into());
    let mut out = Vec::new();
    let mut at = within.start;
    while at < within.end {
        let head = bytes.get(at..at + 8).ok_or_else(bad)?;
        let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        let kind = [head[4], head[5], head[6], head[7]];
        let (size, header) = match size32 {
            0 => (within.end - at, 8),
            1 => {
                let l = bytes.get(at + 8..at + 16).ok_or_else(bad)?;
                (
                    usize::try_from(u64::from_be_bytes(l.try_into().unwrap_or_default()))
                        .map_err(|_| bad())?,
                    16,
                )
            }
            n => (n as usize, 8),
        };
        let end = at
            .checked_add(size)
            .filter(|e| *e <= within.end && size >= header)
            .ok_or_else(bad)?;
        out.push(BmffBox {
            kind,
            range: at..end,
            header,
        });
        at = end;
    }
    Ok(out)
}

/// Adds `delta` to every chunk offset at or after `from`, in every track.
/// Refuses files whose other boxes hold absolute offsets this does not move.
fn shift_chunk_offsets(bytes: &mut [u8], from: u64, delta: i64) -> Result<()> {
    let top = bmff_boxes(bytes)?;
    for b in &top {
        if !matches!(
            &b.kind,
            b"ftyp" | b"moov" | b"mdat" | b"free" | b"skip" | b"uuid"
        ) {
            return Err(Error::Unsupported(format!(
                "MP4 top-level box {} holds offsets this does not move",
                String::from_utf8_lossy(&b.kind)
            )));
        }
    }
    let mut tables = Vec::new();
    for moov in top.iter().filter(|b| &b.kind == b"moov") {
        collect_offset_tables(
            bytes,
            moov.range.start + moov.header..moov.range.end,
            &mut tables,
        )?;
    }
    let bad = || Error::Malformed("malformed chunk offset table".into());
    for (wide, body) in tables {
        let count = u32::from_be_bytes(
            bytes
                .get(body.start + 4..body.start + 8)
                .ok_or_else(bad)?
                .try_into()
                .unwrap_or_default(),
        ) as usize;
        let width = if wide { 8 } else { 4 };
        let entries = body.start + 8;
        if entries + count * width > body.end {
            return Err(bad());
        }
        for i in 0..count {
            let at = entries + i * width;
            let old = if wide {
                u64::from_be_bytes(bytes[at..at + 8].try_into().unwrap_or_default())
            } else {
                u64::from(u32::from_be_bytes(
                    bytes[at..at + 4].try_into().unwrap_or_default(),
                ))
            };
            if old < from {
                continue;
            }
            let new = old.checked_add_signed(delta).ok_or_else(bad)?;
            if wide {
                bytes[at..at + 8].copy_from_slice(&new.to_be_bytes());
            } else {
                let new = u32::try_from(new).map_err(|_| {
                    Error::Unsupported("chunk offset above 4 GiB in a stco table".into())
                })?;
                bytes[at..at + 4].copy_from_slice(&new.to_be_bytes());
            }
        }
    }
    Ok(())
}

fn collect_offset_tables(
    bytes: &[u8],
    within: Range<usize>,
    out: &mut Vec<(bool, Range<usize>)>,
) -> Result<()> {
    for b in bmff_children(bytes, within)? {
        let body = b.range.start + b.header..b.range.end;
        match &b.kind {
            b"trak" | b"mdia" | b"minf" | b"stbl" => collect_offset_tables(bytes, body, out)?,
            b"stco" => out.push((false, body)),
            b"co64" => out.push((true, body)),
            b"iloc" | b"meta" => {
                return Err(Error::Unsupported("MP4 item offsets are not moved".into()));
            }
            _ => {}
        }
    }
    Ok(())
}

/// One BMFF hash exclusion: a top-level box path, optionally only boxes
/// holding given bytes at given offsets.
#[derive(Debug, Clone)]
pub(crate) struct BmffExclusion {
    xpath: String,
    data: Vec<(u64, Vec<u8>)>,
}

impl BmffExclusion {
    fn to_cbor(&self) -> Value {
        let mut m = vec![("xpath", Value::text(self.xpath.clone()))];
        if !self.data.is_empty() {
            m.push((
                "data",
                Value::Array(
                    self.data
                        .iter()
                        .map(|(o, v)| {
                            Value::map([
                                ("offset", Value::Uint(*o)),
                                ("value", Value::Bytes(v.clone())),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        Value::map(m)
    }

    fn from_cbor(v: &Value) -> Result<Self> {
        let bad = |m: &str| Error::Unsupported(format!("BMFF exclusion: {m}"));
        let Value::Map(entries) = v else {
            return Err(bad("not a map"));
        };
        let mut xpath = None;
        let mut data = Vec::new();
        for (k, val) in entries {
            match k.as_text() {
                Some("xpath") => xpath = val.as_text().map(str::to_string),
                Some("data") => {
                    for d in val.as_array().ok_or_else(|| bad("data is not a list"))? {
                        let o = d
                            .get("offset")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| bad("data without offset"))?;
                        let b = d
                            .get("value")
                            .and_then(Value::as_bytes)
                            .ok_or_else(|| bad("data without value"))?;
                        data.push((o, b.to_vec()));
                    }
                }
                _ => return Err(bad("only xpath and data are supported")),
            }
        }
        let xpath = xpath.ok_or_else(|| bad("no xpath"))?;
        if xpath.len() != 5 || !xpath.starts_with('/') {
            return Err(bad("only top-level box paths are supported"));
        }
        Ok(Self { xpath, data })
    }

    fn matches(&self, b: &BmffBox, bytes: &[u8]) -> bool {
        self.xpath.as_bytes()[1..] == b.kind
            && self.data.iter().all(|(o, v)| {
                usize::try_from(*o)
                    .ok()
                    .and_then(|o| bytes.get(b.range.start + o..b.range.start + o + v.len()))
                    == Some(v.as_slice())
            })
    }
}

fn default_bmff_exclusions() -> Vec<BmffExclusion> {
    vec![
        BmffExclusion {
            xpath: "/uuid".into(),
            data: vec![(8, BMFF_UUID.to_vec())],
        },
        BmffExclusion {
            xpath: "/ftyp".into(),
            data: Vec::new(),
        },
        BmffExclusion {
            xpath: "/mfra".into(),
            data: Vec::new(),
        },
    ]
}

/// SHA-256 over every top-level box not excluded, each preceded by its file
/// offset as a big-endian 64-bit number.
fn bmff_hash(bytes: &[u8], exclusions: &[BmffExclusion]) -> Result<[u8; 32]> {
    let mut h = Sha256::new();
    for b in bmff_boxes(bytes)? {
        if exclusions.iter().any(|e| e.matches(&b, bytes)) {
            continue;
        }
        h.update((b.range.start as u64).to_be_bytes());
        h.update(&bytes[b.range.clone()]);
    }
    Ok(h.finalize().into())
}

// PNG.

fn png_chunks(bytes: &[u8]) -> Result<Vec<([u8; 4], Range<usize>, Range<usize>)>> {
    let bad = |m: &str| Error::Malformed(format!("PNG: {m}"));
    if !bytes.starts_with(PNG_SIGNATURE) {
        return Err(bad("no PNG signature"));
    }
    let mut out = Vec::new();
    let mut at = PNG_SIGNATURE.len();
    while at < bytes.len() {
        let head = bytes
            .get(at..at + 8)
            .ok_or_else(|| bad("truncated chunk"))?;
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let kind = [head[4], head[5], head[6], head[7]];
        let end = at
            .checked_add(12 + len)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| bad("chunk runs past the end"))?;
        if &kind == b"caBX" && crc32(&bytes[at + 4..end - 4]).to_be_bytes() != bytes[end - 4..end] {
            return Err(bad("manifest chunk CRC mismatch"));
        }
        out.push((kind, at + 8..end - 4, at..end));
        at = end;
        if &kind == b"IEND" {
            break;
        }
    }
    if out.first().map(|c| &c.0) != Some(b"IHDR") {
        return Err(bad("IHDR is not the first chunk"));
    }
    Ok(out)
}

fn png_after_ihdr(bytes: &[u8]) -> Result<usize> {
    Ok(png_chunks(bytes)?[0].2.end)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for b in bytes {
        c ^= u32::from(*b);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

// RIFF.

fn riff_chunks(bytes: &[u8]) -> Result<Vec<([u8; 4], Range<usize>, Range<usize>)>> {
    let bad = |m: &str| Error::Malformed(format!("RIFF: {m}"));
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" {
        return Err(bad("no RIFF header"));
    }
    let size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if size + 8 != bytes.len() {
        return Err(bad("the RIFF chunk does not span the file"));
    }
    let mut out = Vec::new();
    let mut at = 12;
    while at < bytes.len() {
        let head = bytes
            .get(at..at + 8)
            .ok_or_else(|| bad("truncated chunk"))?;
        let len = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
        let kind = [head[0], head[1], head[2], head[3]];
        let end = at
            .checked_add(8 + len)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| bad("chunk runs past the end"))?;
        out.push((kind, at + 8..end, at..end));
        at = end + len % 2;
    }
    if at != bytes.len() {
        return Err(bad("pad byte missing"));
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Tiny files in each format.

    /// A 1x1 RGB PNG.
    pub fn png() -> Vec<u8> {
        let chunk = |kind: &[u8; 4], data: &[u8]| {
            let mut c = (data.len() as u32).to_be_bytes().to_vec();
            c.extend_from_slice(kind);
            c.extend_from_slice(data);
            let crc = super::crc32(&c[4..]);
            c.extend_from_slice(&crc.to_be_bytes());
            c
        };
        let ihdr = [0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0];
        // zlib stored block of the one scanline: filter 0, then RGB.
        let idat = [
            0x78, 0x01, 0x01, 0x04, 0x00, 0xfb, 0xff, 0x00, 0xff, 0x00, 0x00, 0x03, 0x01, 0x01,
            0x00,
        ];
        [
            super::PNG_SIGNATURE.to_vec(),
            chunk(b"IHDR", &ihdr),
            chunk(b"IDAT", &idat),
            chunk(b"IEND", &[]),
        ]
        .concat()
    }

    /// 16-bit mono PCM WAV of a few samples.
    pub fn wav() -> Vec<u8> {
        let samples: Vec<u8> = [0i16, 1000, -1000, 500, 7]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&16_000u32.to_le_bytes());
        fmt.extend_from_slice(&32_000u32.to_le_bytes());
        fmt.extend_from_slice(&2u16.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        let mut body = b"WAVE".to_vec();
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        body.extend_from_slice(&samples);
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// An MP4 shaped as the engine writes one: `ftyp`, `moov` with two
    /// tracks (one `co64`, one `stco` table), then `mdat`.
    pub fn mp4() -> Vec<u8> {
        let bx = |kind: &[u8; 4], body: &[u8]| {
            [&((body.len() + 8) as u32).to_be_bytes()[..], kind, body].concat()
        };
        let ftyp = bx(b"ftyp", b"isom\0\0\x02\0isomiso2avc1mp41");
        let table = |kind: &[u8; 4], offsets: &[u64]| {
            let mut b = vec![0u8; 4];
            b.extend_from_slice(&(offsets.len() as u32).to_be_bytes());
            for o in offsets {
                if kind == b"co64" {
                    b.extend_from_slice(&o.to_be_bytes());
                } else {
                    b.extend_from_slice(&(*o as u32).to_be_bytes());
                }
            }
            bx(kind, &b)
        };
        let trak = |t: Vec<u8>| bx(b"trak", &bx(b"mdia", &bx(b"minf", &bx(b"stbl", &t))));
        let moov_len = |a: u64| {
            bx(
                b"moov",
                &[
                    trak(table(b"co64", &[a, a + 4])),
                    trak(table(b"stco", &[a + 8])),
                ]
                .concat(),
            )
            .len() as u64
        };
        let data_at = ftyp.len() as u64 + moov_len(0) + 8;
        let moov = bx(
            b"moov",
            &[
                trak(table(b"co64", &[data_at, data_at + 4])),
                trak(table(b"stco", &[data_at + 8])),
            ]
            .concat(),
        );
        [ftyp, moov, bx(b"mdat", b"videovidesound")].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_format_places_finds_and_removes_a_store() {
        for (format, asset) in [
            (Format::Png, fixtures::png()),
            (Format::Wav, fixtures::wav()),
            (Format::Mp4, fixtures::mp4()),
        ] {
            for store in [b"odd".to_vec(), b"even".to_vec()] {
                let placed = place(format, &asset, &store).unwrap();
                let found = find(format, &placed).unwrap().unwrap();
                assert_eq!(&placed[found.store.clone()], &store[..], "{format:?}");
                assert_eq!(remove(format, &placed).unwrap(), asset, "{format:?}");
                assert!(matches!(
                    place(format, &placed, &store),
                    Err(Error::AlreadySigned)
                ));
            }
            assert!(find(format, &asset).unwrap().is_none());
        }
    }

    #[test]
    fn a_png_data_hash_is_the_hash_of_the_original_file() {
        let asset = fixtures::png();
        let placed = place(Format::Png, &asset, b"store").unwrap();
        let found = find(Format::Png, &placed).unwrap().unwrap();
        assert_eq!(
            data_hash(&placed, &[found.holder]),
            <[u8; 32]>::from(Sha256::digest(&asset))
        );
    }

    #[test]
    fn inserting_an_mp4_store_moves_chunk_offsets_with_the_media() {
        let asset = fixtures::mp4();
        let placed = place(Format::Mp4, &asset, b"store").unwrap();
        let mdat = |b: &[u8]| {
            bmff_boxes(b)
                .unwrap()
                .into_iter()
                .find(|x| &x.kind == b"mdat")
                .unwrap()
                .range
                .start as u64
                + 8
        };
        let mut tables = Vec::new();
        let moov = bmff_boxes(&placed)
            .unwrap()
            .into_iter()
            .find(|x| &x.kind == b"moov")
            .unwrap();
        collect_offset_tables(&placed, moov.range.start + 8..moov.range.end, &mut tables).unwrap();
        let (wide, body) = &tables[0];
        assert!(*wide);
        assert_eq!(
            u64::from_be_bytes(placed[body.start + 8..body.start + 16].try_into().unwrap()),
            mdat(&placed)
        );
        assert_eq!(
            &placed[mdat(&placed) as usize..mdat(&placed) as usize + 5],
            b"video"
        );
    }

    #[test]
    fn a_crc_is_the_png_crc() {
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }
}
