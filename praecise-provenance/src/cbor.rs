//! The CBOR subset content credentials use (RFC 8949): integers, byte and
//! text strings, arrays, maps, tags, booleans and null, with definite
//! lengths only. Encoding is deterministic (RFC 8949 section 4.2.1): the
//! shortest argument form, and map keys sorted by their encoded bytes.

use crate::{Error, Result};

/// A CBOR data item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// An unsigned integer.
    Uint(u64),
    /// A negative integer `-1 - n`, holding `n`.
    Nint(u64),
    /// A byte string.
    Bytes(Vec<u8>),
    /// A text string.
    Text(String),
    /// An array.
    Array(Vec<Value>),
    /// A map, in the order given; encoding sorts it.
    Map(Vec<(Value, Value)>),
    /// A tagged item.
    Tag(u64, Box<Value>),
    /// `true` or `false`.
    Bool(bool),
    /// `null`.
    Null,
}

impl Value {
    /// A text string.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// A map from text keys.
    pub fn map<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Self {
        Self::Map(
            entries
                .into_iter()
                .map(|(k, v)| (Self::Text(k.into()), v))
                .collect(),
        )
    }

    /// The value under text key `key` of a map.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Map(m) => m
                .iter()
                .find(|(k, _)| matches!(k, Self::Text(t) if t == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }

    /// The text of a text string.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            _ => None,
        }
    }

    /// The bytes of a byte string.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// The value of an unsigned integer.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint(n) => Some(*n),
            _ => None,
        }
    }

    /// The items of an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The deterministic encoding of this item.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::Uint(n) => head(out, 0, *n),
            Self::Nint(n) => head(out, 1, *n),
            Self::Bytes(b) => {
                head(out, 2, b.len() as u64);
                out.extend_from_slice(b);
            }
            Self::Text(t) => {
                head(out, 3, t.len() as u64);
                out.extend_from_slice(t.as_bytes());
            }
            Self::Array(a) => {
                head(out, 4, a.len() as u64);
                for v in a {
                    v.write(out);
                }
            }
            Self::Map(m) => {
                let mut entries: Vec<(Vec<u8>, Vec<u8>)> =
                    m.iter().map(|(k, v)| (k.encode(), v.encode())).collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                head(out, 5, entries.len() as u64);
                for (k, v) in entries {
                    out.extend_from_slice(&k);
                    out.extend_from_slice(&v);
                }
            }
            Self::Tag(t, v) => {
                head(out, 6, *t);
                v.write(out);
            }
            Self::Bool(false) => out.push(0xf4),
            Self::Bool(true) => out.push(0xf5),
            Self::Null => out.push(0xf6),
        }
    }

    /// Decodes exactly one item filling `bytes`.
    ///
    /// # Errors
    /// When `bytes` is not one well-formed item of the supported subset.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader {
            bytes,
            at: 0,
            depth: 0,
        };
        let v = r.item()?;
        if r.at != bytes.len() {
            return Err(Error::Malformed("trailing bytes after a CBOR item".into()));
        }
        Ok(v)
    }
}

fn head(out: &mut Vec<u8>, major: u8, n: u64) {
    let m = major << 5;
    if n < 0x18 {
        out.push(m | n as u8);
    } else if n <= 0xff {
        out.extend_from_slice(&[m | 0x18, n as u8]);
    } else if n <= 0xffff {
        out.push(m | 0x19);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= 0xffff_ffff {
        out.push(m | 0x1a);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(m | 0x1b);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

const MAX_DEPTH: usize = 32;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    depth: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|e| *e <= self.bytes.len())
            .ok_or_else(|| Error::Malformed("CBOR item runs past its end".into()))?;
        let s = &self.bytes[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn head(&mut self) -> Result<(u8, u8, u64)> {
        let b = self.take(1)?[0];
        let (major, info) = (b >> 5, b & 0x1f);
        let n = match info {
            0..=23 => u64::from(info),
            24 => u64::from(self.take(1)?[0]),
            25 => u64::from(u16::from_be_bytes(
                self.take(2)?.try_into().unwrap_or_default(),
            )),
            26 => u64::from(u32::from_be_bytes(
                self.take(4)?.try_into().unwrap_or_default(),
            )),
            27 => u64::from_be_bytes(self.take(8)?.try_into().unwrap_or_default()),
            _ => {
                return Err(Error::Malformed(
                    "indefinite or reserved CBOR length".into(),
                ));
            }
        };
        Ok((major, info, n))
    }

    fn len(&self, n: u64) -> Result<usize> {
        usize::try_from(n)
            .ok()
            .filter(|n| *n <= self.bytes.len() - self.at)
            .ok_or_else(|| Error::Malformed("CBOR length runs past its end".into()))
    }

    fn item(&mut self) -> Result<Value> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Error::Malformed("CBOR nests too deep".into()));
        }
        let (major, info, n) = self.head()?;
        let v = match major {
            0 => Value::Uint(n),
            1 => Value::Nint(n),
            2 => {
                let n = self.len(n)?;
                Value::Bytes(self.take(n)?.to_vec())
            }
            3 => {
                let n = self.len(n)?;
                let s = self.take(n)?.to_vec();
                Value::Text(
                    String::from_utf8(s)
                        .map_err(|_| Error::Malformed("CBOR text is not UTF-8".into()))?,
                )
            }
            4 => {
                let n = self.len(n)?;
                let mut a = Vec::with_capacity(n);
                for _ in 0..n {
                    a.push(self.item()?);
                }
                Value::Array(a)
            }
            5 => {
                let n = self.len(n)?;
                let mut m = Vec::with_capacity(n);
                for _ in 0..n {
                    let k = self.item()?;
                    let v = self.item()?;
                    m.push((k, v));
                }
                Value::Map(m)
            }
            6 => Value::Tag(n, Box::new(self.item()?)),
            _ => match info {
                20 => Value::Bool(false),
                21 => Value::Bool(true),
                22 => Value::Null,
                _ => {
                    return Err(Error::Malformed(
                        "unsupported CBOR simple value or float".into(),
                    ));
                }
            },
        };
        self.depth -= 1;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_is_deterministic_and_round_trips() {
        let a = Value::map([
            ("b", Value::Uint(500)),
            ("a", Value::Bytes(vec![1, 2])),
            ("aa", Value::Nint(0)),
        ]);
        let b = Value::map([
            ("aa", Value::Nint(0)),
            ("a", Value::Bytes(vec![1, 2])),
            ("b", Value::Uint(500)),
        ]);
        assert_eq!(a.encode(), b.encode());
        // Keys sort by their encoded bytes: "a" (0x61 0x61) before "b", before "aa".
        assert_eq!(
            a.encode(),
            [
                0xa3, 0x61, b'a', 0x42, 1, 2, 0x61, b'b', 0x19, 0x01, 0xf4, 0x62, b'a', b'a', 0x20
            ]
        );
        let back = Value::decode(&a.encode()).unwrap();
        assert_eq!(back.encode(), a.encode());
        assert_eq!(back.get("b").and_then(Value::as_u64), Some(500));
    }

    #[test]
    fn malformed_items_are_refused() {
        assert!(Value::decode(&[0x5f]).is_err(), "indefinite length");
        assert!(Value::decode(&[0x43, 1, 2]).is_err(), "short byte string");
        assert!(Value::decode(&[0x01, 0x02]).is_err(), "trailing bytes");
        assert!(
            Value::decode(&[0x9b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]).is_err(),
            "huge array"
        );
    }
}
