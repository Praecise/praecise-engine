//! JUMBF (ISO/IEC 19566-5) boxes as a manifest store uses them: superboxes
//! (`jumb`) opened by a description box (`jumd`: content type UUID,
//! toggles, label) and followed by content boxes or further superboxes.

use crate::{Error, Result};

/// The content type UUID for a four-character code: the code followed by
/// `0011-0010-8000-00AA00389B71`.
#[must_use]
pub(crate) const fn type_uuid(code: &[u8; 4]) -> [u8; 16] {
    [
        code[0], code[1], code[2], code[3], 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00,
        0x38, 0x9B, 0x71,
    ]
}

pub(crate) const STORE: [u8; 16] = type_uuid(b"c2pa");
pub(crate) const MANIFEST: [u8; 16] = type_uuid(b"c2ma");
pub(crate) const ASSERTIONS: [u8; 16] = type_uuid(b"c2as");
pub(crate) const CLAIM: [u8; 16] = type_uuid(b"c2cl");
pub(crate) const SIGNATURE: [u8; 16] = type_uuid(b"c2cs");
pub(crate) const CBOR: [u8; 16] = type_uuid(b"cbor");

/// Requestable, label present.
const TOGGLES: u8 = 0x03;

/// A box of type `kind` holding `payload`.
pub(crate) fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(
        &u32::try_from(8 + payload.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// A superbox of content type `uuid` labelled `label` holding `children`
/// (already boxed).
pub(crate) fn superbox(uuid: [u8; 16], label: &str, children: &[Vec<u8>]) -> Vec<u8> {
    let mut jumd = uuid.to_vec();
    jumd.push(TOGGLES);
    jumd.extend_from_slice(label.as_bytes());
    jumd.push(0);
    let mut payload = boxed(b"jumd", &jumd);
    for c in children {
        payload.extend_from_slice(c);
    }
    boxed(b"jumb", &payload)
}

/// A labelled CBOR superbox: an assertion, the claim or the signature.
pub(crate) fn cbor_superbox(uuid: [u8; 16], label: &str, cbor: &[u8]) -> Vec<u8> {
    superbox(uuid, label, &[boxed(b"cbor", cbor)])
}

/// A parsed superbox.
#[derive(Debug)]
pub(crate) struct Jumb<'a> {
    pub uuid: [u8; 16],
    pub label: String,
    /// Everything after the superbox header: what a hashed URI hashes.
    pub payload: &'a [u8],
    pub children: Vec<Child<'a>>,
}

/// A box inside a superbox, after its description.
#[derive(Debug)]
pub(crate) enum Child<'a> {
    Super(Jumb<'a>),
    Content([u8; 4], &'a [u8]),
}

impl<'a> Jumb<'a> {
    /// The child superbox labelled `label`.
    pub fn child(&self, label: &str) -> Option<&Jumb<'a>> {
        self.children.iter().find_map(|c| match c {
            Child::Super(j) if j.label == label => Some(j),
            _ => None,
        })
    }

    /// The child superboxes, in order.
    pub fn supers(&self) -> impl Iterator<Item = &Jumb<'a>> {
        self.children.iter().filter_map(|c| match c {
            Child::Super(j) => Some(j),
            Child::Content(..) => None,
        })
    }

    /// The payload of the single CBOR content box of this superbox.
    pub fn cbor(&self) -> Result<&'a [u8]> {
        let mut found = self.children.iter().filter_map(|c| match c {
            Child::Content(kind, p) if kind == b"cbor" => Some(*p),
            _ => None,
        });
        match (found.next(), found.next()) {
            (Some(p), None) => Ok(p),
            _ => Err(Error::Malformed(format!(
                "{} does not hold exactly one CBOR box",
                self.label
            ))),
        }
    }
}

/// Splits `bytes` into boxes: `(kind, payload, whole box)`.
pub(crate) fn boxes(mut bytes: &[u8]) -> Result<Vec<([u8; 4], &[u8], &[u8])>> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        if bytes.len() < 8 {
            return Err(Error::Malformed("truncated JUMBF box header".into()));
        }
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if size < 8 || size > bytes.len() {
            return Err(Error::Malformed("JUMBF box size out of range".into()));
        }
        let kind = [bytes[4], bytes[5], bytes[6], bytes[7]];
        out.push((kind, &bytes[8..size], &bytes[..size]));
        bytes = &bytes[size..];
    }
    Ok(out)
}

/// Parses exactly one superbox filling `bytes`.
pub(crate) fn parse(bytes: &[u8]) -> Result<Jumb<'_>> {
    parse_at(bytes, 0)
}

fn parse_at(bytes: &[u8], depth: usize) -> Result<Jumb<'_>> {
    if depth > 8 {
        return Err(Error::Malformed("JUMBF nests too deep".into()));
    }
    let list = boxes(bytes)?;
    let [(kind, payload, _)] = list.as_slice() else {
        return Err(Error::Malformed("expected one JUMBF superbox".into()));
    };
    if kind != b"jumb" {
        return Err(Error::Malformed("expected a JUMBF superbox".into()));
    }
    let inner = boxes(payload)?;
    let Some((b"jumd", desc, _)) = inner.first().map(|(k, p, w)| (k, *p, *w)) else {
        return Err(Error::Malformed(
            "JUMBF superbox without a description box".into(),
        ));
    };
    if desc.len() < 18 || desc[16] & 0x02 == 0 {
        return Err(Error::Malformed(
            "JUMBF description box without a label".into(),
        ));
    }
    let uuid: [u8; 16] = desc[..16].try_into().unwrap_or_default();
    let rest = &desc[17..];
    let end = rest
        .iter()
        .position(|b| *b == 0)
        .ok_or_else(|| Error::Malformed("JUMBF label is not terminated".into()))?;
    let label = String::from_utf8(rest[..end].to_vec())
        .map_err(|_| Error::Malformed("JUMBF label is not UTF-8".into()))?;
    let mut children = Vec::new();
    for (kind, payload, whole) in &inner[1..] {
        if kind == b"jumb" {
            children.push(Child::Super(parse_at(whole, depth + 1)?));
        } else {
            children.push(Child::Content(*kind, payload));
        }
    }
    Ok(Jumb {
        uuid,
        label,
        payload,
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superboxes_round_trip() {
        let inner = cbor_superbox(CBOR, "c2pa.actions.v2", &[0xa0]);
        let outer = superbox(ASSERTIONS, "c2pa.assertions", std::slice::from_ref(&inner));
        let j = parse(&outer).unwrap();
        assert_eq!(j.uuid, ASSERTIONS);
        assert_eq!(j.label, "c2pa.assertions");
        let a = j.child("c2pa.actions.v2").unwrap();
        assert_eq!(a.cbor().unwrap(), &[0xa0]);
        assert_eq!(a.payload, &inner[8..]);
        assert_eq!(&STORE[..4], b"c2pa");
    }

    #[test]
    fn broken_boxes_are_refused() {
        let mut b = cbor_superbox(CBOR, "x", &[0xa0]);
        b[3] += 1;
        assert!(parse(&b).is_err());
        assert!(parse(&boxed(b"jumb", &boxed(b"cbor", &[0xa0]))).is_err());
    }
}
