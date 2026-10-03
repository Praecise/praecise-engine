//! Canonical binary encoding.
//!
//! Little-endian fixed-width integers, length-prefixed byte strings, floats by
//! their IEEE-754 bit pattern, and fields in call order. The same value always
//! encodes to the same bytes, which is what makes hashes of manifests, recipes
//! and step records reproducible across machines.

use crate::Error;
use crate::hash::Digest;

/// Builds a canonical byte string.
#[derive(Debug, Default, Clone)]
pub struct Encoder {
    buf: Vec<u8>,
}

impl Encoder {
    /// An empty encoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one byte.
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    /// Appends a `u32`.
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Appends a `u64`.
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Appends an `f32` by its bit pattern.
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.u32(v.to_bits())
    }

    /// Appends an `f64` by its bit pattern.
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.u64(v.to_bits())
    }

    /// Appends a boolean as one byte.
    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.u8(u8::from(v))
    }

    /// Appends a length-prefixed byte string.
    pub fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.u64(v.len() as u64);
        self.buf.extend_from_slice(v);
        self
    }

    /// Appends a length-prefixed UTF-8 string.
    pub fn str(&mut self, v: &str) -> &mut Self {
        self.bytes(v.as_bytes())
    }

    /// Appends a digest (fixed 32 bytes).
    pub fn digest(&mut self, v: &Digest) -> &mut Self {
        self.buf.extend_from_slice(&v.0);
        self
    }

    /// Appends an optional digest with a presence byte.
    pub fn opt_digest(&mut self, v: Option<&Digest>) -> &mut Self {
        match v {
            Some(d) => self.u8(1).digest(d),
            None => self.u8(0),
        }
    }

    /// The encoded bytes.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// The encoded bytes so far.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
}

/// Reads a canonical byte string written by [`Encoder`].
#[derive(Debug)]
pub struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// Starts reading `buf`.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes consumed so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| Error::Format("truncated canonical encoding".into()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Reads one byte.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    /// Reads a `u32`.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads a `u64`.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn u64(&mut self) -> Result<u64, Error> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }

    /// Reads an `f32`.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_bits(self.u32()?))
    }

    /// Reads an `f64`.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Reads a boolean.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input or a byte other than 0 or 1.
    pub fn bool(&mut self) -> Result<bool, Error> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(Error::Format(format!("invalid boolean byte {b}"))),
        }
    }

    /// Reads a length-prefixed byte string.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn bytes(&mut self) -> Result<&'a [u8], Error> {
        let n =
            usize::try_from(self.u64()?).map_err(|_| Error::Format("length overflow".into()))?;
        self.take(n)
    }

    /// Reads a length-prefixed UTF-8 string.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input or invalid UTF-8.
    pub fn str(&mut self) -> Result<String, Error> {
        let b = self.bytes()?;
        String::from_utf8(b.to_vec()).map_err(|e| Error::Format(format!("utf-8: {e}")))
    }

    /// Reads a digest.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input.
    pub fn digest(&mut self) -> Result<Digest, Error> {
        let mut a = [0u8; 32];
        a.copy_from_slice(self.take(32)?);
        Ok(Digest(a))
    }

    /// Reads an optional digest.
    ///
    /// # Errors
    /// [`Error::Format`] on truncated input or a bad presence byte.
    pub fn opt_digest(&mut self) -> Result<Option<Digest>, Error> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.digest()?)),
            b => Err(Error::Format(format!("invalid option byte {b}"))),
        }
    }

    /// Fails unless every byte was consumed.
    ///
    /// # Errors
    /// [`Error::Format`] when trailing bytes remain.
    pub fn finish(self) -> Result<(), Error> {
        if self.pos == self.buf.len() {
            Ok(())
        } else {
            Err(Error::Format(format!(
                "{} trailing bytes after canonical encoding",
                self.buf.len() - self.pos
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let d = crate::hash::sha256(b"x");
        let mut e = Encoder::new();
        e.u8(7)
            .u32(9)
            .u64(11)
            .f32(-1.5)
            .f64(0.25)
            .bool(true)
            .str("hé")
            .digest(&d)
            .opt_digest(None);
        let bytes = e.finish();
        let mut r = Decoder::new(&bytes);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 9);
        assert_eq!(r.u64().unwrap(), 11);
        assert_eq!(r.f32().unwrap().to_bits(), (-1.5f32).to_bits());
        assert_eq!(r.f64().unwrap().to_bits(), 0.25f64.to_bits());
        assert!(r.bool().unwrap());
        assert_eq!(r.str().unwrap(), "hé");
        assert_eq!(r.digest().unwrap(), d);
        assert_eq!(r.opt_digest().unwrap(), None);
        r.finish().unwrap();
    }

    #[test]
    fn truncated_and_trailing_are_errors() {
        let mut e = Encoder::new();
        e.u64(5);
        let bytes = e.finish();
        assert!(Decoder::new(&bytes[..4]).u64().is_err());
        let mut r = Decoder::new(&bytes);
        r.u32().unwrap();
        assert!(r.finish().is_err());
    }
}
