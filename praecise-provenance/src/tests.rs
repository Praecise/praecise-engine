use p256::ecdsa::signature::{Signer as _, Verifier as _};
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};

use super::*;
use crate::container::fixtures;
use crate::x509::{CertificateRequest, Issuer, NO_EXPIRY};

/// A test-only key standing in for hardware.
struct TestKey(SigningKey);

impl TestKey {
    fn new(seed: u8) -> Self {
        Self(SigningKey::from_slice(&[seed; 32]).unwrap())
    }
    fn public(&self) -> [u8; 65] {
        self.0
            .verifying_key()
            .to_encoded_point(false)
            .as_bytes()
            .try_into()
            .unwrap()
    }
}

impl Es256Signer for TestKey {
    fn sign(&self, message: &[u8]) -> std::result::Result<[u8; 64], String> {
        let s: Signature = self.0.sign(message);
        Ok(s.to_bytes().into())
    }
}

fn es256(key: &[u8; 65], msg: &[u8], sig: &[u8; 64]) -> bool {
    let (Ok(k), Ok(s)) = (
        VerifyingKey::from_sec1_bytes(key),
        Signature::from_slice(sig),
    ) else {
        return false;
    };
    k.verify(msg, &s).is_ok()
}

const NOW: u64 = 1_760_000_000;

fn chain(authority: &TestKey, signer: &TestKey) -> Vec<Vec<u8>> {
    let root = x509::issue(
        &CertificateRequest {
            common_name: "Test issuer".into(),
            organization: Some("Test".into()),
            public_key: authority.public(),
            not_before: NOW - 10,
            not_after: NO_EXPIRY,
            authority: true,
            extensions: vec![],
        },
        None,
        authority,
    )
    .unwrap();
    let leaf = x509::issue(
        &CertificateRequest {
            common_name: "Test signer".into(),
            organization: None,
            public_key: signer.public(),
            not_before: NOW - 10,
            not_after: NO_EXPIRY,
            authority: false,
            extensions: vec![(vec![2, 25, 1], vec![0x05, 0x00])],
        },
        Some(Issuer {
            certificate: &root,
            signer: authority,
        }),
        signer,
    )
    .unwrap();
    vec![leaf, root]
}

fn manifest(certificates: Vec<Vec<u8>>) -> Manifest {
    Manifest {
        instance: [7; 16],
        generator: "test generator".into(),
        generator_version: "1.0".into(),
        assertions: vec![
            generated_by("tiny-model"),
            Assertion {
                label: "org.example.note".into(),
                data: Value::map([("text", Value::text("hello"))]),
            },
        ],
        certificates,
    }
}

fn fixtures() -> [(&'static str, Vec<u8>); 3] {
    [
        ("image/png", fixtures::png()),
        ("audio/wav", fixtures::wav()),
        ("video/mp4", fixtures::mp4()),
    ]
}

#[test]
fn a_signed_file_verifies_and_yields_the_original_bytes() {
    let (ca, key) = (TestKey::new(1), TestKey::new(2));
    let m = manifest(chain(&ca, &key));
    for (mime, asset) in fixtures() {
        let signed = sign(mime, &asset, &m, &key, &es256).unwrap();
        let v = verify(mime, &signed, &es256, NOW).unwrap();
        assert_eq!(v.unsigned, asset, "{mime}");
        assert_eq!(v.label, "urn:c2pa:07070707-0707-0707-0707-070707070707");
        assert_eq!(v.generator.as_deref(), Some("test generator"));
        assert_eq!(v.certificates.len(), 2);
        assert_eq!(v.certificates[0].public_key, key.public());
        assert_eq!(
            v.certificates[0].extension_value(&[2, 25, 1]),
            Some(&[0x05, 0x00][..])
        );
        let actions = v.assertion("c2pa.actions.v2").unwrap();
        let action = &actions.get("actions").unwrap().as_array().unwrap()[0];
        assert_eq!(
            action.get("digitalSourceType").and_then(Value::as_text),
            Some("http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")
        );
        assert_eq!(
            v.assertion("org.example.note")
                .and_then(|n| n.get("text"))
                .and_then(Value::as_text),
            Some("hello")
        );
        let binding = if mime == "video/mp4" {
            "c2pa.hash.bmff.v3"
        } else {
            "c2pa.hash.data"
        };
        assert!(v.assertion(binding).is_some(), "{mime}");
    }
}

#[test]
fn a_changed_media_byte_breaks_the_hard_binding() {
    let (ca, key) = (TestKey::new(1), TestKey::new(2));
    let m = manifest(chain(&ca, &key));
    for (mime, asset) in fixtures() {
        let signed = sign(mime, &asset, &m, &key, &es256).unwrap();
        let mut tampered = signed.clone();
        // The last byte is media in each fixture (PNG: the IEND CRC).
        let n = tampered.len();
        tampered[n - 1] ^= 1;
        assert!(verify(mime, &tampered, &es256, NOW).is_err(), "{mime}");
    }
}

#[test]
fn a_changed_assertion_or_signature_is_refused() {
    let (ca, key) = (TestKey::new(1), TestKey::new(2));
    let m = manifest(chain(&ca, &key));
    let signed = sign("image/png", &fixtures::png(), &m, &key, &es256).unwrap();
    let at = signed.windows(5).position(|w| w == b"hello").unwrap();
    let mut t = signed.clone();
    t[at] = b'j';
    // Keep the PNG chunk CRC right, so only the manifest check can catch it.
    let fix_crc = |b: &mut Vec<u8>| {
        let start = 33;
        let len = u32::from_be_bytes(b[start..start + 4].try_into().unwrap()) as usize;
        let crc = crc(&b[start + 4..start + 8 + len]);
        b[start + 8 + len..start + 12 + len].copy_from_slice(&crc.to_be_bytes());
    };
    fix_crc(&mut t);
    assert!(matches!(
        verify("image/png", &t, &es256, NOW),
        Err(Error::Mismatch(_))
    ));
    let mut s = signed.clone();
    let n = signed.len();
    // The signature's last byte sits right before the chunk CRC, which sits
    // before IDAT; find it from the chunk length.
    let len = u32::from_be_bytes(s[33..37].try_into().unwrap()) as usize;
    s[33 + 8 + len - 1] ^= 1;
    fix_crc(&mut s);
    assert_eq!(s.len(), n);
    assert!(matches!(
        verify("image/png", &s, &es256, NOW),
        Err(Error::Mismatch(_))
    ));
}

fn crc(bytes: &[u8]) -> u32 {
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

#[test]
fn a_signer_whose_key_is_not_its_certificates_is_refused() {
    let (ca, key, other) = (TestKey::new(1), TestKey::new(2), TestKey::new(3));
    let m = manifest(chain(&ca, &key));
    assert!(matches!(
        sign("audio/wav", &fixtures::wav(), &m, &other, &es256),
        Err(Error::Signing(_))
    ));
}

#[test]
fn chains_outside_the_profile_are_refused() {
    let (ca, key) = (TestKey::new(1), TestKey::new(2));
    let good = chain(&ca, &key);
    let asset = fixtures::wav();
    // Before the certificates' validity.
    let signed = sign("audio/wav", &asset, &manifest(good.clone()), &key, &es256).unwrap();
    assert!(matches!(
        verify("audio/wav", &signed, &es256, NOW - 3600),
        Err(Error::Certificate(_))
    ));
    // A leaf issued by a different authority than the chain names.
    let rogue = chain(&TestKey::new(9), &key);
    let mixed = vec![rogue[0].clone(), good[1].clone()];
    let signed = sign("audio/wav", &asset, &manifest(mixed), &key, &es256).unwrap();
    assert!(matches!(
        verify("audio/wav", &signed, &es256, NOW),
        Err(Error::Certificate(_))
    ));
    // A self-signed signing certificate.
    let alone = x509::issue(
        &CertificateRequest {
            common_name: "alone".into(),
            organization: None,
            public_key: key.public(),
            not_before: NOW - 10,
            not_after: NO_EXPIRY,
            authority: false,
            extensions: vec![],
        },
        None,
        &key,
    )
    .unwrap();
    let signed = sign("audio/wav", &asset, &manifest(vec![alone]), &key, &es256).unwrap();
    assert!(matches!(
        verify("audio/wav", &signed, &es256, NOW),
        Err(Error::Certificate(_))
    ));
}

#[test]
fn unsupported_or_unsigned_files_are_refused() {
    let (ca, key) = (TestKey::new(1), TestKey::new(2));
    let m = manifest(chain(&ca, &key));
    assert!(matches!(
        sign("video/webm", b"x", &m, &key, &es256),
        Err(Error::Unsupported(_))
    ));
    assert!(!supports("application/json"));
    assert!(matches!(
        verify("image/png", &fixtures::png(), &es256, NOW),
        Err(Error::NotSigned)
    ));
    let signed = sign("image/png", &fixtures::png(), &m, &key, &es256).unwrap();
    assert!(matches!(
        sign("image/png", &signed, &m, &key, &es256),
        Err(Error::AlreadySigned)
    ));
    let mut dup = m.clone();
    dup.assertions.push(generated_by("again"));
    assert!(matches!(
        sign("image/png", &fixtures::png(), &dup, &key, &es256),
        Err(Error::Unsupported(_))
    ));
}
