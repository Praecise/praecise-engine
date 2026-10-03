//! SHA-256 digests with domain separation.
//!
//! Every content hash in this crate is taken under a domain tag, so the digest
//! of one kind of object (a manifest, a step record, a delta) can never equal
//! the digest of another kind that happens to share its bytes.

use std::fmt;

use sha2::{Digest as _, Sha256};

use crate::Error;

/// A SHA-256 digest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// The all-zero digest, used as the value before the first record of a hash-linked log.
    pub const ZERO: Self = Self([0; 32]);

    /// Lower-case hex encoding.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parses a 64-character hex string.
    ///
    /// # Errors
    /// [`Error::Format`] when the string is not 32 hex-encoded bytes.
    pub fn from_hex(s: &str) -> Result<Self, Error> {
        let bytes = hex::decode(s).map_err(|e| Error::Format(format!("digest hex: {e}")))?;
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| Error::Format("digest must be 32 bytes".into()))?;
        Ok(Self(arr))
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({})", self.to_hex())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Plain SHA-256 of `bytes`.
#[must_use]
pub fn sha256(bytes: &[u8]) -> Digest {
    Digest(Sha256::digest(bytes).into())
}

/// SHA-256 of `parts` under `domain`: the domain is length-prefixed so no
/// choice of parts can imitate a different domain.
#[must_use]
pub fn domain_hash(domain: &str, parts: &[&[u8]]) -> Digest {
    let mut h = Sha256::new();
    h.update((domain.len() as u64).to_le_bytes());
    h.update(domain.as_bytes());
    for p in parts {
        h.update(p);
    }
    Digest(h.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answer() {
        assert_eq!(
            sha256(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hex_round_trip_and_domains_differ() {
        let d = domain_hash("a", &[b"x"]);
        assert_eq!(Digest::from_hex(&d.to_hex()).unwrap(), d);
        assert_ne!(domain_hash("a", &[b"bx"]), domain_hash("ab", &[b"x"]));
        assert!(Digest::from_hex("00").is_err());
    }
}
