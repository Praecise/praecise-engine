//! X.509 certificates for ECDSA P-256 keys (RFC 5280), issued and read in
//! the profile content credentials require: version 3, `ecdsa-with-SHA256`,
//! key usage, extended key usage on a signing certificate, subject and
//! authority key identifiers. The signing key is reached through
//! [`Es256Signer`], so the private key stays wherever it lives.

use sha2::{Digest, Sha256};

use crate::{Error, Es256Signer, Result};

/// `ecdsa-with-SHA256`.
const ECDSA_SHA256: &[u64] = &[1, 2, 840, 10045, 4, 3, 2];
/// `id-ecPublicKey`.
const EC_PUBLIC_KEY: &[u64] = &[1, 2, 840, 10045, 2, 1];
/// `prime256v1`.
const P256: &[u64] = &[1, 2, 840, 10045, 3, 1, 7];
const COMMON_NAME: &[u64] = &[2, 5, 4, 3];
const ORGANIZATION: &[u64] = &[2, 5, 4, 10];
const SUBJECT_KEY_ID: &[u64] = &[2, 5, 29, 14];
const KEY_USAGE: &[u64] = &[2, 5, 29, 15];
const BASIC_CONSTRAINTS: &[u64] = &[2, 5, 29, 19];
const AUTHORITY_KEY_ID: &[u64] = &[2, 5, 29, 35];
const EXT_KEY_USAGE: &[u64] = &[2, 5, 29, 37];
/// `id-kp-emailProtection`, the claim signing purpose validators accept
/// with no trust configuration loaded.
const EMAIL_PROTECTION: &[u64] = &[1, 3, 6, 1, 5, 5, 7, 3, 4];
/// `id-kp-documentSigning` (RFC 9336).
const DOCUMENT_SIGNING: &[u64] = &[1, 3, 6, 1, 5, 5, 7, 3, 36];
/// `c2pa-kp-claimSigning`.
const CLAIM_SIGNING: &[u64] = &[1, 3, 6, 1, 4, 1, 62558, 2, 1];

/// `notAfter` for a certificate with no defined expiry (RFC 5280 4.1.2.5).
pub const NO_EXPIRY: u64 = 253_402_300_799;

/// What a certificate states.
#[derive(Debug, Clone)]
pub struct CertificateRequest {
    /// The subject's common name.
    pub common_name: String,
    /// The subject's organization, when stated.
    pub organization: Option<String>,
    /// The subject's P-256 public key, uncompressed SEC1 (65 bytes).
    pub public_key: [u8; 65],
    /// Validity start and end, Unix seconds.
    pub not_before: u64,
    /// See `not_before`; [`NO_EXPIRY`] for none.
    pub not_after: u64,
    /// A certificate authority (signs certificates) or a claim signer.
    pub authority: bool,
    /// Further non-critical extensions: object identifier arcs and the DER
    /// value.
    pub extensions: Vec<(Vec<u64>, Vec<u8>)>,
}

/// The issuer of a certificate: its certificate (for its name and key
/// identifier) and its key. `None` issues a self-signed certificate.
#[derive(Clone, Copy)]
pub struct Issuer<'a> {
    /// The issuer's certificate (DER).
    pub certificate: &'a [u8],
    /// The issuer's key.
    pub signer: &'a dyn Es256Signer,
}

impl std::fmt::Debug for Issuer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Issuer").finish_non_exhaustive()
    }
}

/// The DER `SubjectPublicKeyInfo` of a P-256 key.
#[must_use]
pub fn subject_public_key_info(public_key: &[u8; 65]) -> Vec<u8> {
    seq(&[
        seq(&[oid(EC_PUBLIC_KEY), oid(P256)]),
        bit_string(public_key),
    ])
}

fn key_id(public_key: &[u8; 65]) -> Vec<u8> {
    Sha256::digest(public_key)[..20].to_vec()
}

/// Issues a certificate for `req`, signed by `issuer`, or self-signed by
/// `self_signer` (the subject's own key) when `issuer` is `None`.
///
/// # Errors
/// When the issuer's certificate does not parse or a key refuses to sign.
pub fn issue(
    req: &CertificateRequest,
    issuer: Option<Issuer<'_>>,
    self_signer: &dyn Es256Signer,
) -> Result<Vec<u8>> {
    let subject = name(&req.common_name, req.organization.as_deref());
    let own_id = key_id(&req.public_key);
    let (issuer_name, authority_id, signer) = match issuer {
        Some(i) => {
            let c = Certificate::parse(i.certificate)?;
            (c.subject.clone(), key_id(&c.public_key), i.signer)
        }
        None => (subject.clone(), own_id.clone(), self_signer),
    };
    let mut serial = Sha256::digest(
        [
            &req.public_key[..],
            &issuer_name,
            &req.not_before.to_be_bytes(),
        ]
        .concat(),
    )[..16]
        .to_vec();
    serial[0] &= 0x7f;
    let mut ext = vec![
        extension(
            BASIC_CONSTRAINTS,
            true,
            &if req.authority {
                seq(&[tlv(0x01, &[0xff]), integer(&[0])])
            } else {
                seq(&[])
            },
        ),
        // digitalSignature for a signer; keyCertSign and cRLSign for an authority.
        extension(
            KEY_USAGE,
            true,
            &if req.authority {
                tlv(0x03, &[0x01, 0x06])
            } else {
                tlv(0x03, &[0x07, 0x80])
            },
        ),
        extension(SUBJECT_KEY_ID, false, &tlv(0x04, &own_id)),
        extension(AUTHORITY_KEY_ID, false, &seq(&[tlv(0x80, &authority_id)])),
    ];
    if !req.authority {
        ext.push(extension(
            EXT_KEY_USAGE,
            false,
            &seq(&[
                oid(EMAIL_PROTECTION),
                oid(DOCUMENT_SIGNING),
                oid(CLAIM_SIGNING),
            ]),
        ));
    }
    for (arcs, value) in &req.extensions {
        ext.push(extension(arcs, false, value));
    }
    let tbs = seq(&[
        tlv(0xa0, &integer(&[2])),
        integer(&serial),
        seq(&[oid(ECDSA_SHA256)]),
        issuer_name,
        seq(&[time(req.not_before), time(req.not_after)]),
        subject,
        subject_public_key_info(&req.public_key),
        tlv(0xa3, &seq(&ext)),
    ]);
    let rs = signer.sign(&tbs).map_err(Error::Signing)?;
    let sig = seq(&[integer(&rs[..32]), integer(&rs[32..])]);
    Ok(seq(&[tbs, seq(&[oid(ECDSA_SHA256)]), bit_string(&sig)]))
}

/// A certificate as read.
#[derive(Debug, Clone)]
pub struct Certificate {
    /// The DER of the signed part.
    pub tbs: Vec<u8>,
    /// The issuer's raw `r || s` signature over `tbs`.
    pub signature: [u8; 64],
    /// The DER `Name` of the issuer.
    pub issuer: Vec<u8>,
    /// The DER `Name` of the subject.
    pub subject: Vec<u8>,
    /// The subject's common name.
    pub common_name: Option<String>,
    /// Validity, Unix seconds.
    pub not_before: u64,
    /// See `not_before`.
    pub not_after: u64,
    /// The subject's P-256 key, uncompressed SEC1.
    pub public_key: [u8; 65],
    /// Basic constraints say this is a certificate authority.
    pub authority: bool,
    /// Key usage: digitalSignature.
    pub digital_signature: bool,
    /// Key usage: keyCertSign.
    pub key_cert_sign: bool,
    /// Extended key usage names a claim signing purpose.
    pub claim_signing: bool,
    /// The subject key identifier was present.
    pub subject_key_id: bool,
    /// The authority key identifier was present.
    pub authority_key_id: bool,
    /// Every extension: identifier arcs, critical, DER value.
    pub extensions: Vec<(Vec<u64>, bool, Vec<u8>)>,
}

impl Certificate {
    /// Reads a DER certificate whose key is P-256 and whose signature is
    /// `ecdsa-with-SHA256`.
    ///
    /// # Errors
    /// When it is not such a certificate, has unique identifiers, or has a
    /// critical extension this profile does not know.
    pub fn parse(der: &[u8]) -> Result<Self> {
        let bad = |m: &str| Error::Certificate(m.to_string());
        let mut outer = Der::new(der).expect(0x30)?;
        let (tbs_raw, tbs) = outer.next_raw(0x30)?;
        let alg = outer.expect(0x30)?.oid()?;
        let sig_bits = outer.bytes(0x03)?;
        if alg != ECDSA_SHA256 || sig_bits.first() != Some(&0) {
            return Err(bad("certificate is not signed with ecdsa-with-SHA256"));
        }
        let mut sig = Der::new(&sig_bits[1..]).expect(0x30)?;
        let signature = rs_from(sig.bytes(0x02)?, sig.bytes(0x02)?)?;

        let mut t = Der::new(tbs);
        if t.peek() == Some(0xa0) {
            let v = t.expect(0xa0)?.bytes(0x02)?;
            if v != [2] {
                return Err(bad("certificate is not version 3"));
            }
        } else {
            return Err(bad("certificate is not version 3"));
        }
        t.bytes(0x02)?;
        if t.expect(0x30)?.oid()? != ECDSA_SHA256 {
            return Err(bad("certificate signature algorithm mismatch"));
        }
        let (issuer, _) = t.next_raw(0x30)?;
        let mut validity = t.expect(0x30)?;
        let not_before = validity.time()?;
        let not_after = validity.time()?;
        let (subject, subject_body) = t.next_raw(0x30)?;
        let mut spki = t.expect(0x30)?;
        let mut alg = spki.expect(0x30)?;
        if alg.oid()? != EC_PUBLIC_KEY || alg.oid()? != P256 {
            return Err(bad("certificate key is not P-256"));
        }
        let key = spki.bytes(0x03)?;
        let public_key: [u8; 65] = key
            .get(1..)
            .and_then(|k| k.try_into().ok())
            .filter(|k: &[u8; 65]| key[0] == 0 && k[0] == 4)
            .ok_or_else(|| bad("certificate key is not an uncompressed P-256 point"))?;
        let mut cert = Self {
            tbs: tbs_raw.to_vec(),
            signature,
            issuer: issuer.to_vec(),
            subject: subject.to_vec(),
            common_name: common_name(subject_body),
            not_before,
            not_after,
            public_key,
            authority: false,
            digital_signature: false,
            key_cert_sign: false,
            claim_signing: false,
            subject_key_id: false,
            authority_key_id: false,
            extensions: Vec::new(),
        };
        while let Some(tag) = t.peek() {
            match tag {
                0x81 | 0x82 => return Err(bad("certificate carries unique identifiers")),
                0xa3 => {
                    let mut list = t.expect(0xa3)?.expect(0x30)?;
                    while list.peek().is_some() {
                        let mut e = list.expect(0x30)?;
                        let id = e.oid()?;
                        let critical = if e.peek() == Some(0x01) {
                            e.bytes(0x01)? == [0xff]
                        } else {
                            false
                        };
                        let value = e.bytes(0x04)?.to_vec();
                        cert.extension(&id, critical, &value)?;
                        cert.extensions.push((id, critical, value));
                    }
                }
                _ => return Err(bad("unexpected field in certificate")),
            }
        }
        Ok(cert)
    }

    fn extension(&mut self, id: &[u64], critical: bool, value: &[u8]) -> Result<()> {
        let bad = |m: &str| Error::Certificate(m.to_string());
        if id == BASIC_CONSTRAINTS {
            let mut s = Der::new(value).expect(0x30)?;
            self.authority = s.peek() == Some(0x01) && s.bytes(0x01)? == [0xff];
        } else if id == KEY_USAGE {
            let bits = Der::new(value).bytes(0x03)?;
            let first = bits.get(1).copied().unwrap_or(0);
            self.digital_signature = first & 0x80 != 0;
            self.key_cert_sign = first & 0x04 != 0;
        } else if id == EXT_KEY_USAGE {
            let mut s = Der::new(value).expect(0x30)?;
            while s.peek().is_some() {
                let o = s.oid()?;
                if o == EMAIL_PROTECTION || o == DOCUMENT_SIGNING || o == CLAIM_SIGNING {
                    self.claim_signing = true;
                }
            }
        } else if id == SUBJECT_KEY_ID {
            self.subject_key_id = true;
        } else if id == AUTHORITY_KEY_ID {
            self.authority_key_id = true;
        } else if critical {
            return Err(bad(
                "certificate has a critical extension this profile does not know",
            ));
        }
        Ok(())
    }

    /// The value of the extension `arcs`, when present.
    #[must_use]
    pub fn extension_value(&self, arcs: &[u64]) -> Option<&[u8]> {
        self.extensions
            .iter()
            .find(|(id, _, _)| id == arcs)
            .map(|(_, _, v)| v.as_slice())
    }
}

fn common_name(name: &[u8]) -> Option<String> {
    let mut rdns = Der::new(name);
    while rdns.peek().is_some() {
        let mut set = rdns.expect(0x31).ok()?;
        let mut atv = set.expect(0x30).ok()?;
        if atv.oid().ok()? == COMMON_NAME {
            return String::from_utf8(atv.any().ok()?.1.to_vec()).ok();
        }
    }
    None
}

fn rs_from(r: &[u8], s: &[u8]) -> Result<[u8; 64]> {
    let mut out = [0u8; 64];
    for (i, v) in [r, s].into_iter().enumerate() {
        if v.first().is_none_or(|b| b & 0x80 != 0) {
            return Err(Error::Certificate(
                "ECDSA signature integer is not positive".into(),
            ));
        }
        let v = match v {
            [0, rest @ ..] => rest,
            v => v,
        };
        if v.len() > 32 {
            return Err(Error::Certificate(
                "ECDSA signature integer out of range".into(),
            ));
        }
        out[i * 32 + 32 - v.len()..(i + 1) * 32].copy_from_slice(v);
    }
    Ok(out)
}

// DER writing.

pub(crate) fn tlv(tag: u8, content: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let n = content.len();
    if n < 0x80 {
        out.push(n as u8);
    } else {
        let len = (n as u64).to_be_bytes();
        let skip = len.iter().take_while(|b| **b == 0).count();
        out.push(0x80 | (8 - skip) as u8);
        out.extend_from_slice(&len[skip..]);
    }
    out.extend_from_slice(content);
    out
}

fn seq(items: &[Vec<u8>]) -> Vec<u8> {
    tlv(0x30, &items.concat())
}

fn integer(be: &[u8]) -> Vec<u8> {
    let mut v: Vec<u8> = be.iter().copied().skip_while(|b| *b == 0).collect();
    if v.is_empty() || v[0] & 0x80 != 0 {
        v.insert(0, 0);
    }
    tlv(0x02, &v)
}

fn bit_string(bytes: &[u8]) -> Vec<u8> {
    tlv(0x03, &[&[0u8][..], bytes].concat())
}

pub(crate) fn oid(arcs: &[u64]) -> Vec<u8> {
    let mut body = Vec::new();
    let first = arcs[0] * 40 + arcs.get(1).copied().unwrap_or(0);
    for a in std::iter::once(first).chain(arcs.iter().skip(2).copied()) {
        let mut chunk = vec![(a & 0x7f) as u8];
        let mut rest = a >> 7;
        while rest > 0 {
            chunk.push(0x80 | (rest & 0x7f) as u8);
            rest >>= 7;
        }
        chunk.reverse();
        body.extend_from_slice(&chunk);
    }
    tlv(0x06, &body)
}

fn name(cn: &str, org: Option<&str>) -> Vec<u8> {
    let rdn = |id: &[u64], v: &str| tlv(0x31, &seq(&[oid(id), tlv(0x0c, v.as_bytes())]));
    let mut parts = Vec::new();
    if let Some(o) = org {
        parts.push(rdn(ORGANIZATION, o));
    }
    parts.push(rdn(COMMON_NAME, cn));
    seq(&parts)
}

fn extension(id: &[u64], critical: bool, value: &[u8]) -> Vec<u8> {
    let mut parts = vec![oid(id)];
    if critical {
        parts.push(tlv(0x01, &[0xff]));
    }
    parts.push(tlv(0x04, value));
    seq(&parts)
}

fn time(unix: u64) -> Vec<u8> {
    let (y, mo, d, h, mi, s) = civil(unix);
    if (1950..2050).contains(&y) {
        tlv(
            0x17,
            format!("{:02}{mo:02}{d:02}{h:02}{mi:02}{s:02}Z", y % 100).as_bytes(),
        )
    } else {
        tlv(
            0x18,
            format!("{y:04}{mo:02}{d:02}{h:02}{mi:02}{s:02}Z").as_bytes(),
        )
    }
}

/// Unix seconds to (year, month, day, hour, minute, second), UTC.
fn civil(unix: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = unix / 86_400;
    let secs = unix % 86_400;
    // Howard Hinnant's civil_from_days, for days since 1970-01-01.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + u64::from(m <= 2);
    (y, m, d, secs / 3600, secs % 3600 / 60, secs % 60)
}

fn days_from_civil(y: u64, m: u64, d: u64) -> Option<u64> {
    let y = if m <= 2 { y.checked_sub(1)? } else { y };
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + doe).checked_sub(719_468)
}

// DER reading.

pub(crate) struct Der<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Der<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    pub(crate) fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    /// The next item: (tag, content, whole item).
    pub(crate) fn any(&mut self) -> Result<(u8, &'a [u8], &'a [u8])> {
        let bad = || Error::Certificate("malformed DER".into());
        let start = self.at;
        let tag = *self.bytes.get(self.at).ok_or_else(bad)?;
        if tag & 0x1f == 0x1f {
            return Err(bad());
        }
        let first = *self.bytes.get(self.at + 1).ok_or_else(bad)?;
        let mut at = self.at + 2;
        let len = if first < 0x80 {
            usize::from(first)
        } else {
            let n = usize::from(first & 0x7f);
            if n == 0 || n > 4 {
                return Err(bad());
            }
            let mut l = 0usize;
            for _ in 0..n {
                l = (l << 8) | usize::from(*self.bytes.get(at).ok_or_else(bad)?);
                at += 1;
            }
            l
        };
        let end = at
            .checked_add(len)
            .filter(|e| *e <= self.bytes.len())
            .ok_or_else(bad)?;
        self.at = end;
        Ok((tag, &self.bytes[at..end], &self.bytes[start..end]))
    }

    pub(crate) fn bytes(&mut self, tag: u8) -> Result<&'a [u8]> {
        let (t, c, _) = self.any()?;
        if t == tag {
            Ok(c)
        } else {
            Err(Error::Certificate(format!(
                "expected DER tag {tag:#x}, found {t:#x}"
            )))
        }
    }

    pub(crate) fn expect(&mut self, tag: u8) -> Result<Der<'a>> {
        Ok(Der::new(self.bytes(tag)?))
    }

    fn next_raw(&mut self, tag: u8) -> Result<(&'a [u8], &'a [u8])> {
        let (t, c, whole) = self.any()?;
        if t == tag {
            Ok((whole, c))
        } else {
            Err(Error::Certificate("unexpected DER item".into()))
        }
    }

    pub(crate) fn oid(&mut self) -> Result<Vec<u64>> {
        let body = self.bytes(0x06)?;
        let mut arcs = Vec::new();
        let mut v: u64 = 0;
        for b in body {
            v = v
                .checked_mul(128)
                .ok_or_else(|| Error::Certificate("object identifier arc too large".into()))?
                | u64::from(b & 0x7f);
            if b & 0x80 == 0 {
                if arcs.is_empty() {
                    let first = (v / 40).min(2);
                    arcs.push(first);
                    arcs.push(v - first * 40);
                } else {
                    arcs.push(v);
                }
                v = 0;
            }
        }
        Ok(arcs)
    }

    fn time(&mut self) -> Result<u64> {
        let bad = || Error::Certificate("malformed certificate time".into());
        let (tag, body, _) = self.any()?;
        let s = std::str::from_utf8(body).map_err(|_| bad())?;
        let s = s.strip_suffix('Z').ok_or_else(bad)?;
        let digits = |r: std::ops::Range<usize>| {
            s.get(r).and_then(|d| d.parse::<u64>().ok()).ok_or_else(bad)
        };
        let (y, rest) = match (tag, s.len()) {
            (0x17, 12) => {
                let yy = digits(0..2)?;
                (if yy < 50 { 2000 + yy } else { 1900 + yy }, 2)
            }
            (0x18, 14) => (digits(0..4)?, 4),
            _ => return Err(bad()),
        };
        let (mo, d, h, mi, sec) = (
            digits(rest..rest + 2)?,
            digits(rest + 2..rest + 4)?,
            digits(rest + 4..rest + 6)?,
            digits(rest + 6..rest + 8)?,
            digits(rest + 8..rest + 10)?,
        );
        if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 59 {
            return Err(bad());
        }
        Ok(days_from_civil(y, mo, d).ok_or_else(bad)? * 86_400 + h * 3600 + mi * 60 + sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_time_round_trips() {
        for t in [0, 951_782_400, 1_760_000_000, 2_524_607_999, NO_EXPIRY] {
            let (y, mo, d, h, mi, s) = civil(t);
            assert_eq!(
                days_from_civil(y, mo, d).unwrap() * 86_400 + h * 3600 + mi * 60 + s,
                t
            );
        }
        assert_eq!(civil(NO_EXPIRY), (9999, 12, 31, 23, 59, 59));
    }

    #[test]
    fn object_identifiers_round_trip() {
        for arcs in [CLAIM_SIGNING, P256, &[2, 25, 1234, 5][..]] {
            assert_eq!(Der::new(&oid(arcs)).oid().unwrap(), arcs);
        }
        assert_eq!(
            oid(P256),
            [0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07]
        );
    }
}
