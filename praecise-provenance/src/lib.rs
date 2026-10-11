//! Content credentials for generated media: signed C2PA manifests embedded
//! in the files the engine produces, and their verification.
//!
//! A manifest states who made a file and how (its assertions), is bound to
//! the file's bytes by a hard-binding hash, and is signed with COSE
//! (`COSE_Sign1`, ES256) by a key named in an X.509 certificate chain
//! carried in the signature. The signing key is never handed to this crate:
//! the caller supplies an [`Es256Signer`] (a TPM, a Secure Enclave, an HSM),
//! and signature checks go through a caller-supplied ES256 verifier.
//!
//! Formats: PNG, WAV and MP4 ([`container`] has the placement rules).
//!
//! Follows the C2PA Technical Specification 2.2 (claims v2 with
//! `created_assertions`, `c2pa.hash.data`, `c2pa.hash.bmff.v3`,
//! `c2pa.actions.v2`, `x5chain` in the protected header) as implemented by
//! the reference SDK, c2pa-rs.

mod cbor;
mod container;
mod jumbf;
pub mod x509;

use sha2::{Digest, Sha256};

pub use cbor::Value;
pub use x509::Certificate;

/// What can go wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A format, algorithm or structure this crate does not handle.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Bytes that do not parse as what they claim to be.
    #[error("malformed: {0}")]
    Malformed(String),
    /// A certificate outside the profile.
    #[error("certificate: {0}")]
    Certificate(String),
    /// The signer refused.
    #[error("signing failed: {0}")]
    Signing(String),
    /// A hash or signature that does not match.
    #[error("does not verify: {0}")]
    Mismatch(String),
    /// The file already carries a manifest store.
    #[error("the file already carries content credentials")]
    AlreadySigned,
    /// The file carries no manifest store.
    #[error("the file carries no content credentials")]
    NotSigned,
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// An ECDSA P-256 key that signs SHA-256 of a message, wherever it lives.
pub trait Es256Signer: Send + Sync {
    /// ECDSA P-256 over SHA-256(`message`), raw `r || s`.
    ///
    /// # Errors
    /// When the key refuses or is unreachable.
    fn sign(&self, message: &[u8]) -> std::result::Result<[u8; 64], String>;
}

/// Checks an ECDSA P-256 signature (raw `r || s`) over SHA-256(`message`)
/// against an uncompressed SEC1 public key.
pub type Es256Verifier<'a> = &'a dyn Fn(&[u8; 65], &[u8], &[u8; 64]) -> bool;

/// One assertion: its label and its CBOR content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assertion {
    /// The assertion label, e.g. `c2pa.actions.v2`.
    pub label: String,
    /// Its content.
    pub data: Value,
}

/// The `c2pa.actions.v2` assertion of a file created by a generative model:
/// one `c2pa.created` action with the IPTC digital source type
/// `trainedAlgorithmicMedia`, naming `agent` as the software that acted.
#[must_use]
pub fn generated_by(agent: &str) -> Assertion {
    Assertion {
        label: "c2pa.actions.v2".into(),
        data: Value::map([(
            "actions",
            Value::Array(vec![Value::map([
                ("action", Value::text("c2pa.created")),
                (
                    "digitalSourceType",
                    Value::text(
                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                    ),
                ),
                ("softwareAgent", Value::map([("name", Value::text(agent))])),
            ])]),
        )]),
    }
}

/// A manifest to sign into a file.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// The manifest's identity; its label is `urn:c2pa:<uuid>`.
    pub instance: [u8; 16],
    /// The claim generator's name.
    pub generator: String,
    /// The claim generator's version.
    pub generator_version: String,
    /// The assertions, besides the hard binding this crate adds.
    pub assertions: Vec<Assertion>,
    /// The signer's certificate, then the certificates that issued it, DER.
    pub certificates: Vec<Vec<u8>>,
}

/// Whether files of type `mime` can carry a manifest.
#[must_use]
pub fn supports(mime: &str) -> bool {
    container::Format::of(mime).is_ok()
}

fn uuid_text(u: &[u8; 16]) -> String {
    let h = |r: std::ops::Range<usize>| u[r].iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        h(0..4),
        h(4..6),
        h(6..8),
        h(8..10),
        h(10..16)
    )
}

fn manifest_label(m: &Manifest) -> String {
    format!("urn:c2pa:{}", uuid_text(&m.instance))
}

/// Signs `manifest` into `asset` (of type `mime`) with `signer`, whose
/// certificate is `manifest.certificates[0]`, and checks the signature
/// with `es256` before answering.
///
/// # Errors
/// When the format is not supported, the asset does not parse, already
/// carries a manifest, an assertion label repeats, or the signer refuses or
/// signs with a key other than its certificate's.
pub fn sign(
    mime: &str,
    asset: &[u8],
    manifest: &Manifest,
    signer: &dyn Es256Signer,
    es256: Es256Verifier<'_>,
) -> Result<Vec<u8>> {
    let format = container::Format::of(mime)?;
    if manifest.certificates.is_empty() {
        return Err(Error::Certificate("no signing certificate".into()));
    }
    let leaf = Certificate::parse(&manifest.certificates[0])?;
    let binding_label = container::binding_label(format);
    let mut labels: Vec<&str> = manifest
        .assertions
        .iter()
        .map(|a| a.label.as_str())
        .collect();
    labels.push(binding_label);
    labels.sort_unstable();
    if labels.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::Unsupported("assertion labels repeat".into()));
    }
    let label = manifest_label(manifest);
    let protected = protected_header(&manifest.certificates);

    // The store's length does not depend on the hash or the signature, both
    // fixed-size; it depends on the exclusion a data hash states, which
    // depends on the length. Settle the length, then lay the file out once.
    let placeholder = [0u8; 64];
    let mut len = 1;
    for _ in 0..4 {
        let laid = container::place(format, asset, &vec![0; len])?;
        let found = container::find(format, &laid)?.ok_or(Error::NotSigned)?;
        let binding = container::binding(format, &laid, &found)?;
        let next = build_store(
            &label,
            manifest,
            binding_label,
            &binding,
            &protected,
            &placeholder,
        )
        .0
        .len();
        if next == len {
            break;
        }
        len = next;
    }
    let laid = container::place(format, asset, &vec![0; len])?;
    let found = container::find(format, &laid)?.ok_or(Error::NotSigned)?;
    let binding = container::binding(format, &laid, &found)?;
    let (_, claim) = build_store(
        &label,
        manifest,
        binding_label,
        &binding,
        &protected,
        &placeholder,
    );
    let to_sign = sig_structure(&protected, &claim);
    let rs = signer.sign(&to_sign).map_err(Error::Signing)?;
    if !es256(&leaf.public_key, &to_sign, &rs) {
        return Err(Error::Signing(
            "the signature does not verify under the signing certificate".into(),
        ));
    }
    let (store, _) = build_store(&label, manifest, binding_label, &binding, &protected, &rs);
    if store.len() != len {
        return Err(Error::Malformed(
            "the manifest length did not settle".into(),
        ));
    }
    container::place(format, asset, &store)
}

fn protected_header(certificates: &[Vec<u8>]) -> Vec<u8> {
    let chain = match certificates {
        [one] => Value::Bytes(one.clone()),
        many => Value::Array(many.iter().cloned().map(Value::Bytes).collect()),
    };
    // alg: ES256 (-7); x5chain (33).
    Value::Map(vec![
        (Value::Uint(1), Value::Nint(6)),
        (Value::Uint(33), chain),
    ])
    .encode()
}

fn sig_structure(protected: &[u8], claim: &[u8]) -> Vec<u8> {
    Value::Array(vec![
        Value::text("Signature1"),
        Value::Bytes(protected.to_vec()),
        Value::Bytes(Vec::new()),
        Value::Bytes(claim.to_vec()),
    ])
    .encode()
}

/// The store bytes and the claim's CBOR.
fn build_store(
    label: &str,
    m: &Manifest,
    binding_label: &str,
    binding: &Value,
    protected: &[u8],
    rs: &[u8; 64],
) -> (Vec<u8>, Vec<u8>) {
    let mut boxes = Vec::new();
    let mut refs = Vec::new();
    let all = m
        .assertions
        .iter()
        .map(|a| (a.label.as_str(), &a.data))
        .chain(std::iter::once((binding_label, binding)));
    for (alabel, data) in all {
        let b = jumbf::cbor_superbox(jumbf::CBOR, alabel, &data.encode());
        refs.push(Value::map([
            (
                "url",
                Value::text(format!("self#jumbf=/c2pa/{label}/c2pa.assertions/{alabel}")),
            ),
            ("hash", Value::Bytes(Sha256::digest(&b[8..]).to_vec())),
        ]));
        boxes.push(b);
    }
    let claim = Value::map([
        (
            "instanceID",
            Value::text(format!("xmp:iid:{}", uuid_text(&m.instance))),
        ),
        (
            "claim_generator_info",
            Value::map([
                ("name", Value::text(m.generator.clone())),
                ("version", Value::text(m.generator_version.clone())),
            ]),
        ),
        (
            "signature",
            Value::text(format!("self#jumbf=/c2pa/{label}/c2pa.signature")),
        ),
        ("created_assertions", Value::Array(refs)),
        ("alg", Value::text("sha256")),
    ])
    .encode();
    let cose = Value::Tag(
        18,
        Box::new(Value::Array(vec![
            Value::Bytes(protected.to_vec()),
            Value::Map(Vec::new()),
            Value::Null,
            Value::Bytes(rs.to_vec()),
        ])),
    )
    .encode();
    let manifest = jumbf::superbox(
        jumbf::MANIFEST,
        label,
        &[
            jumbf::superbox(jumbf::ASSERTIONS, "c2pa.assertions", &boxes),
            jumbf::cbor_superbox(jumbf::CLAIM, "c2pa.claim.v2", &claim),
            jumbf::cbor_superbox(jumbf::SIGNATURE, "c2pa.signature", &cose),
        ],
    );
    (jumbf::superbox(jumbf::STORE, "c2pa", &[manifest]), claim)
}

/// A verified manifest.
#[derive(Debug, Clone)]
pub struct Verified {
    /// The manifest label, `urn:c2pa:<uuid>`.
    pub label: String,
    /// The claim generator's name.
    pub generator: Option<String>,
    /// The assertions, hard binding included, in claim order.
    pub assertions: Vec<Assertion>,
    /// The signer's certificate, then its issuers.
    pub certificates: Vec<Certificate>,
    /// The file without its manifest store: the bytes the generator made.
    pub unsigned: Vec<u8>,
}

impl Verified {
    /// The assertion labelled `label`.
    #[must_use]
    pub fn assertion(&self, label: &str) -> Option<&Value> {
        self.assertions
            .iter()
            .find(|a| a.label == label)
            .map(|a| &a.data)
    }
}

/// Verifies the active manifest of `bytes` (of type `mime`): every
/// assertion against the claim, the hard binding against the file, the
/// claim signature against the signer's certificate, and each certificate
/// against the next one in the chain (the last one against itself when it
/// is self-issued), all valid at `now` (Unix seconds). Which chains to trust
/// is the caller's decision: the chain is returned.
///
/// # Errors
/// When anything does not parse, verify or fit the profile.
pub fn verify(mime: &str, bytes: &[u8], es256: Es256Verifier<'_>, now: u64) -> Result<Verified> {
    let format = container::Format::of(mime)?;
    let found = container::find(format, bytes)?.ok_or(Error::NotSigned)?;
    let store = jumbf::parse(&bytes[found.store.clone()])?;
    if store.uuid != jumbf::STORE || store.label != "c2pa" {
        return Err(Error::Malformed("not a C2PA manifest store".into()));
    }
    let manifest = store
        .supers()
        .filter(|j| j.uuid == jumbf::MANIFEST)
        .last()
        .ok_or_else(|| Error::Malformed("the store holds no manifest".into()))?;
    let label = manifest.label.clone();
    let claim_box = manifest
        .child("c2pa.claim.v2")
        .filter(|j| j.uuid == jumbf::CLAIM)
        .ok_or_else(|| Error::Unsupported("only v2 claims are read".into()))?;
    let claim_bytes = claim_box.cbor()?;
    let claim = Value::decode(claim_bytes)?;
    let sig_box = manifest
        .child("c2pa.signature")
        .filter(|j| j.uuid == jumbf::SIGNATURE)
        .ok_or_else(|| Error::Malformed("no claim signature".into()))?;
    let store_box = manifest
        .child("c2pa.assertions")
        .filter(|j| j.uuid == jumbf::ASSERTIONS)
        .ok_or_else(|| Error::Malformed("no assertion store".into()))?;

    let expected_sig = format!("self#jumbf=/c2pa/{label}/c2pa.signature");
    if claim.get("signature").and_then(Value::as_text) != Some(expected_sig.as_str())
        && claim.get("signature").and_then(Value::as_text) != Some("self#jumbf=c2pa.signature")
    {
        return Err(Error::Malformed(
            "the claim does not point at its signature".into(),
        ));
    }
    if let Some(alg) = claim.get("alg").and_then(Value::as_text)
        && alg != "sha256"
    {
        return Err(Error::Unsupported(format!("claim hash algorithm {alg}")));
    }

    // Assertions: each referenced one hashes as the claim says; none is
    // left unreferenced.
    let refs = claim
        .get("created_assertions")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Malformed("claim without created_assertions".into()))?;
    let mut assertions = Vec::new();
    let absolute = format!("self#jumbf=/c2pa/{label}/c2pa.assertions/");
    for r in refs.iter().chain(
        claim
            .get("gathered_assertions")
            .and_then(Value::as_array)
            .unwrap_or_default(),
    ) {
        let url = r
            .get("url")
            .and_then(Value::as_text)
            .ok_or_else(|| Error::Malformed("hashed URI without url".into()))?;
        let alabel = url
            .strip_prefix(&absolute)
            .or_else(|| url.strip_prefix("self#jumbf=c2pa.assertions/"))
            .ok_or_else(|| Error::Unsupported(format!("assertion reference {url}")))?;
        if let Some(alg) = r.get("alg").and_then(Value::as_text)
            && alg != "sha256"
        {
            return Err(Error::Unsupported(format!(
                "assertion hash algorithm {alg}"
            )));
        }
        let b = store_box
            .child(alabel)
            .ok_or_else(|| Error::Mismatch(format!("assertion {alabel} is missing")))?;
        if r.get("hash").and_then(Value::as_bytes) != Some(&Sha256::digest(b.payload)[..]) {
            return Err(Error::Mismatch(format!(
                "assertion {alabel} does not match its hash"
            )));
        }
        if b.uuid != jumbf::CBOR {
            return Err(Error::Unsupported(format!(
                "assertion {alabel} is not CBOR"
            )));
        }
        assertions.push(Assertion {
            label: alabel.to_string(),
            data: Value::decode(b.cbor()?)?,
        });
    }
    if store_box.supers().count() != assertions.len() {
        return Err(Error::Mismatch(
            "the assertion store holds assertions the claim does not reference".into(),
        ));
    }

    // The hard binding.
    let bindings: Vec<&Assertion> = assertions
        .iter()
        .filter(|a| a.label.starts_with("c2pa.hash."))
        .collect();
    let [binding] = bindings.as_slice() else {
        return Err(Error::Mismatch(
            "a manifest needs exactly one hard binding".into(),
        ));
    };
    container::check_binding(format, &binding.label, &binding.data, bytes, &found)?;

    // The signature.
    let cose = Value::decode(sig_box.cbor()?)?;
    let Value::Tag(18, body) = cose else {
        return Err(Error::Malformed(
            "the claim signature is not a tagged COSE_Sign1".into(),
        ));
    };
    let Some(
        [
            Value::Bytes(protected),
            Value::Map(_),
            Value::Null,
            Value::Bytes(sig),
        ],
    ) = body.as_array()
    else {
        return Err(Error::Malformed(
            "COSE_Sign1 with an attached payload or a malformed field".into(),
        ));
    };
    let header = Value::decode(protected)?;
    let Value::Map(fields) = &header else {
        return Err(Error::Malformed("protected header is not a map".into()));
    };
    let field = |k: u64| {
        fields
            .iter()
            .find(|(key, _)| *key == Value::Uint(k))
            .map(|(_, v)| v)
    };
    if field(1) != Some(&Value::Nint(6)) {
        return Err(Error::Unsupported(
            "claim signature algorithm other than ES256".into(),
        ));
    }
    let chain: Vec<Vec<u8>> = match field(33) {
        Some(Value::Bytes(one)) => vec![one.clone()],
        Some(Value::Array(many)) => many
            .iter()
            .map(|c| {
                c.as_bytes()
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| Error::Malformed("x5chain holds a non-certificate".into()))
            })
            .collect::<Result<_>>()?,
        _ => {
            return Err(Error::Malformed(
                "no x5chain in the protected header".into(),
            ));
        }
    };
    let sig: [u8; 64] = sig
        .as_slice()
        .try_into()
        .map_err(|_| Error::Malformed("ES256 signature is not 64 bytes".into()))?;
    let certificates = chain
        .iter()
        .map(|c| Certificate::parse(c))
        .collect::<Result<Vec<_>>>()?;
    let leaf = &certificates[0];
    if !es256(
        &leaf.public_key,
        &sig_structure(protected, claim_bytes),
        &sig,
    ) {
        return Err(Error::Mismatch(
            "the claim signature does not verify".into(),
        ));
    }
    check_chain(&certificates, es256, now)?;

    Ok(Verified {
        label,
        generator: claim
            .get("claim_generator_info")
            .and_then(|g| g.get("name"))
            .and_then(Value::as_text)
            .map(str::to_string),
        assertions,
        certificates,
        unsigned: container::remove(format, bytes)?,
    })
}

fn check_chain(certs: &[Certificate], es256: Es256Verifier<'_>, now: u64) -> Result<()> {
    let bad = |m: &str| Error::Certificate(m.to_string());
    let leaf = &certs[0];
    if leaf.authority || !leaf.digital_signature || !leaf.claim_signing || !leaf.authority_key_id {
        return Err(bad(
            "the signing certificate is not a claim signing certificate",
        ));
    }
    for (i, c) in certs.iter().enumerate() {
        if now < c.not_before || now > c.not_after {
            return Err(bad("a certificate is outside its validity"));
        }
        if i > 0 && (!c.authority || !c.key_cert_sign || !c.subject_key_id) {
            return Err(bad("an issuing certificate is not a certificate authority"));
        }
        let issuer_key = match certs.get(i + 1) {
            Some(next) => {
                if c.issuer != next.subject {
                    return Err(bad("the chain is out of order"));
                }
                Some(&next.public_key)
            }
            None if c.issuer == c.subject => {
                if i == 0 {
                    return Err(bad("the signing certificate is self-signed"));
                }
                Some(&c.public_key)
            }
            None => None,
        };
        if let Some(k) = issuer_key
            && !es256(k, &c.tbs, &c.signature)
        {
            return Err(bad("a certificate signature does not verify"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
