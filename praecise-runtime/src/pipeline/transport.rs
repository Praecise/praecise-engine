//! Authenticated, encrypted links between pipeline stages.
//!
//! Handshake, initiator (upstream stage) first:
//!
//! 1. initiator -> responder: hello (protocol, plan digest, ephemeral X25519 key, nonce, identity)
//! 2. responder -> initiator: hello, then its signature over the transcript
//! 3. initiator -> responder: its signature over the transcript
//!
//! The transcript is the hash of both hellos, so each signature binds the plan, both
//! identities and both ephemeral keys; the role label in each signed message keeps one
//! side's signature from being replayed as the other's. Each side checks that the
//! peer's identity is the one the plan puts next to it. Session keys come from the
//! X25519 secret through HKDF with the transcript as salt, one key per direction;
//! frames are ChaCha20-Poly1305 with a per-direction counter as nonce, so a dropped,
//! replayed or reordered frame fails to open.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::Zeroize;

use super::StageAuthenticator;

const PROTOCOL: &[u8; 25] = b"praecise-layer-pipeline/1";
const MAX_HANDSHAKE_FRAME: usize = 64 * 1024;
/// Largest encrypted frame accepted, in bytes.
pub const MAX_FRAME: usize = 1 << 30;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// Which end of a link this side is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The upstream stage, which dials.
    Initiator,
    /// The downstream stage, which accepts.
    Responder,
}

/// Why a handshake failed.
#[derive(Debug, thiserror::Error)]
pub enum HandshakeError {
    /// The peer did not prove the expected identity, or disagreed on the plan.
    #[error("{0}")]
    Rejected(String),
    /// The link failed during the handshake.
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// The receiving half of an established link.
pub struct SecureReader {
    stream: TcpStream,
    cipher: ChaCha20Poly1305,
    counter: u64,
}

/// The sending half of an established link.
pub struct SecureWriter {
    stream: TcpStream,
    cipher: ChaCha20Poly1305,
    counter: u64,
}

impl std::fmt::Debug for SecureReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecureReader").field("counter", &self.counter).finish_non_exhaustive()
    }
}

impl std::fmt::Debug for SecureWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecureWriter").field("counter", &self.counter).finish_non_exhaustive()
    }
}

fn nonce(counter: u64) -> Nonce {
    let mut n = [0u8; 12];
    n[4..].copy_from_slice(&counter.to_le_bytes());
    *Nonce::from_slice(&n)
}

impl SecureWriter {
    /// Encrypt and send one frame.
    ///
    /// # Errors
    ///
    /// Socket errors, or a frame larger than [`MAX_FRAME`].
    pub fn send(&mut self, plaintext: &[u8]) -> std::io::Result<()> {
        let ct = self
            .cipher
            .encrypt(&nonce(self.counter), plaintext)
            .map_err(|_| std::io::Error::other("frame encryption failed"))?;
        if ct.len() > MAX_FRAME {
            return Err(std::io::Error::other("frame too large"));
        }
        self.counter += 1;
        self.stream.write_all(&(ct.len() as u32).to_le_bytes())?;
        self.stream.write_all(&ct)?;
        self.stream.flush()
    }

    /// Close the link in both directions.
    pub fn shutdown(&self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }

    /// A handle that can close this link from another thread.
    ///
    /// # Errors
    ///
    /// Socket errors.
    pub fn closer(&self) -> std::io::Result<LinkCloser> {
        Ok(LinkCloser(self.stream.try_clone()?))
    }
}

/// Closes a link from any thread, which ends any blocked read on it.
#[derive(Debug)]
pub struct LinkCloser(TcpStream);

impl LinkCloser {
    /// Close the link in both directions.
    pub fn close(&self) {
        let _ = self.0.shutdown(std::net::Shutdown::Both);
    }
}

impl SecureReader {
    /// Receive and open one frame. `Ok(None)` when the peer closed the link cleanly
    /// between frames.
    ///
    /// # Errors
    ///
    /// Socket errors, a frame larger than [`MAX_FRAME`], or a frame that fails to open.
    pub fn recv(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        let mut len = [0u8; 4];
        match self.stream.read_exact(&mut len) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let len = u32::from_le_bytes(len) as usize;
        if len > MAX_FRAME {
            return Err(std::io::Error::other("frame too large"));
        }
        let mut ct = vec![0u8; len];
        self.stream.read_exact(&mut ct)?;
        let pt = self
            .cipher
            .decrypt(&nonce(self.counter), ct.as_slice())
            .map_err(|_| std::io::Error::other("frame failed authentication"))?;
        self.counter += 1;
        Ok(Some(pt))
    }
}

/// Dial `address` (`host:port`), trying each resolved address in turn.
///
/// # Errors
///
/// The last connection error, or an error if the name resolves to nothing.
pub fn dial(address: &str, timeout: Duration) -> std::io::Result<TcpStream> {
    use std::net::ToSocketAddrs;
    let mut last = std::io::Error::new(std::io::ErrorKind::NotFound, format!("{address} resolves to no address"));
    for addr in address.to_socket_addrs()? {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn write_plain(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

fn read_plain(stream: &mut TcpStream) -> Result<Vec<u8>, HandshakeError> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_HANDSHAKE_FRAME {
        return Err(HandshakeError::Rejected("oversized handshake message".into()));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

struct Hello {
    plan: [u8; 32],
    ephemeral: [u8; 32],
    identity: Vec<u8>,
}

fn encode_hello(plan: &[u8; 32], ephemeral: &[u8; 32], nonce: &[u8; 32], identity: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(PROTOCOL.len() + 98 + identity.len());
    out.extend_from_slice(PROTOCOL);
    out.extend_from_slice(plan);
    out.extend_from_slice(ephemeral);
    out.extend_from_slice(nonce);
    out.extend_from_slice(&(identity.len() as u16).to_le_bytes());
    out.extend_from_slice(identity);
    out
}

fn decode_hello(bytes: &[u8]) -> Result<Hello, HandshakeError> {
    let bad = || HandshakeError::Rejected("malformed hello".into());
    let rest = bytes.strip_prefix(&PROTOCOL[..]).ok_or_else(|| HandshakeError::Rejected("peer speaks another protocol".into()))?;
    if rest.len() < 98 {
        return Err(bad());
    }
    let plan: [u8; 32] = rest[..32].try_into().map_err(|_| bad())?;
    let ephemeral: [u8; 32] = rest[32..64].try_into().map_err(|_| bad())?;
    let id_len = u16::from_le_bytes([rest[96], rest[97]]) as usize;
    if rest.len() != 98 + id_len {
        return Err(bad());
    }
    Ok(Hello { plan, ephemeral, identity: rest[98..].to_vec() })
}

fn signed_message(transcript: &[u8; 32], role: Role) -> Vec<u8> {
    let mut m = Vec::with_capacity(PROTOCOL.len() + 32 + 10);
    m.extend_from_slice(PROTOCOL);
    m.extend_from_slice(transcript);
    m.extend_from_slice(match role {
        Role::Initiator => b"initiator",
        Role::Responder => b"responder",
    });
    m
}

/// Run the handshake on `stream` and split it into an encrypted reader and writer.
///
/// `expected_peer` is the identity the plan puts at the other end of this link.
///
/// # Errors
///
/// [`HandshakeError::Rejected`] when the peer disagrees on the protocol or plan, shows
/// another identity, or its signature does not verify; [`HandshakeError::Io`] on
/// socket errors or a timeout.
pub fn handshake(
    stream: TcpStream,
    auth: &dyn StageAuthenticator,
    role: Role,
    expected_peer: &[u8],
    plan_digest: &[u8; 32],
) -> Result<(SecureReader, SecureWriter), HandshakeError> {
    let mut stream = stream;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    stream.set_write_timeout(Some(HANDSHAKE_TIMEOUT))?;

    let mut secret_bytes = [0u8; 32];
    let mut nonce_bytes = [0u8; 32];
    getrandom::getrandom(&mut secret_bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    getrandom::getrandom(&mut nonce_bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    let secret = StaticSecret::from(secret_bytes);
    secret_bytes.zeroize();
    let ours = PublicKey::from(&secret);
    let own_identity = auth.identity();
    let our_hello = encode_hello(plan_digest, ours.as_bytes(), &nonce_bytes, &own_identity);

    let (hello_i, hello_r, peer) = match role {
        Role::Initiator => {
            write_plain(&mut stream, &our_hello)?;
            let theirs = read_plain(&mut stream)?;
            let peer = decode_hello(&theirs)?;
            (our_hello, theirs, peer)
        }
        Role::Responder => {
            let theirs = read_plain(&mut stream)?;
            let peer = decode_hello(&theirs)?;
            // check before answering, so a peer the plan does not name learns nothing
            check_peer(&peer, expected_peer, plan_digest)?;
            write_plain(&mut stream, &our_hello)?;
            (theirs, our_hello, peer)
        }
    };
    check_peer(&peer, expected_peer, plan_digest)?;

    let transcript: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(PROTOCOL);
        h.update((hello_i.len() as u64).to_le_bytes());
        h.update(&hello_i);
        h.update((hello_r.len() as u64).to_le_bytes());
        h.update(&hello_r);
        h.finalize().into()
    };
    let peer_role = match role {
        Role::Initiator => Role::Responder,
        Role::Responder => Role::Initiator,
    };
    let own_sig = auth.sign(&signed_message(&transcript, role)).map_err(HandshakeError::Rejected)?;
    let verify_peer = |sig: &[u8]| {
        if auth.verify(expected_peer, &signed_message(&transcript, peer_role), sig) {
            Ok(())
        } else {
            Err(HandshakeError::Rejected("peer signature does not verify under the plan's identity".into()))
        }
    };
    match role {
        Role::Responder => {
            write_plain(&mut stream, &own_sig)?;
            verify_peer(&read_plain(&mut stream)?)?;
        }
        Role::Initiator => {
            verify_peer(&read_plain(&mut stream)?)?;
            write_plain(&mut stream, &own_sig)?;
        }
    }

    let shared = secret.diffie_hellman(&PublicKey::from(peer.ephemeral));
    if !shared.was_contributory() {
        return Err(HandshakeError::Rejected("non-contributory key exchange".into()));
    }
    let hk = Hkdf::<Sha256>::new(Some(&transcript), shared.as_bytes());
    let mut k_i2r = zeroize::Zeroizing::new([0u8; 32]);
    let mut k_r2i = zeroize::Zeroizing::new([0u8; 32]);
    hk.expand(b"initiator to responder", &mut k_i2r[..]).map_err(|_| HandshakeError::Rejected("key derivation".into()))?;
    hk.expand(b"responder to initiator", &mut k_r2i[..]).map_err(|_| HandshakeError::Rejected("key derivation".into()))?;
    let (k_send, k_recv) = match role {
        Role::Initiator => (k_i2r, k_r2i),
        Role::Responder => (k_r2i, k_i2r),
    };

    stream.set_read_timeout(None)?;
    stream.set_write_timeout(None)?;
    let reader = SecureReader {
        stream: stream.try_clone()?,
        cipher: ChaCha20Poly1305::new(Key::from_slice(&k_recv[..])),
        counter: 0,
    };
    let writer = SecureWriter { stream, cipher: ChaCha20Poly1305::new(Key::from_slice(&k_send[..])), counter: 0 };
    Ok((reader, writer))
}

fn check_peer(peer: &Hello, expected: &[u8], plan_digest: &[u8; 32]) -> Result<(), HandshakeError> {
    if &peer.plan != plan_digest {
        return Err(HandshakeError::Rejected("peer runs a different pipeline plan".into()));
    }
    if peer.identity != expected {
        return Err(HandshakeError::Rejected("peer identity is not the one the plan names".into()));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Test authenticator: identity = 32 bytes, signature = SHA-256(identity || message).
    /// Proves nothing cryptographically; it only exercises the protocol.
    pub(crate) struct Fixture(pub Vec<u8>);

    impl StageAuthenticator for Fixture {
        fn identity(&self) -> Vec<u8> {
            self.0.clone()
        }
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(Sha256::new().chain_update(&self.0).chain_update(message).finalize().to_vec())
        }
        fn verify(&self, identity: &[u8], message: &[u8], signature: &[u8]) -> bool {
            Sha256::new().chain_update(identity).chain_update(message).finalize().as_slice() == signature
        }
    }

    fn link(
        a_expects: Vec<u8>,
        b_expects: Vec<u8>,
        plan_a: [u8; 32],
        plan_b: [u8; 32],
    ) -> (Result<(SecureReader, SecureWriter), HandshakeError>, Result<(SecureReader, SecureWriter), HandshakeError>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            handshake(s, &Fixture(vec![2; 32]), Role::Responder, &b_expects, &plan_b)
        });
        let s = TcpStream::connect(addr).unwrap();
        let a = handshake(s, &Fixture(vec![1; 32]), Role::Initiator, &a_expects, &plan_a);
        (a, t.join().unwrap())
    }

    #[test]
    fn frames_flow_both_ways_after_mutual_authentication() {
        let (a, b) = link(vec![2; 32], vec![1; 32], [5; 32], [5; 32]);
        let (mut ar, mut aw) = a.unwrap();
        let (mut br, mut bw) = b.unwrap();
        aw.send(b"down").unwrap();
        aw.send(&[9u8; 100_000]).unwrap();
        assert_eq!(br.recv().unwrap().unwrap(), b"down");
        assert_eq!(br.recv().unwrap().unwrap(), vec![9u8; 100_000]);
        bw.send(b"up").unwrap();
        assert_eq!(ar.recv().unwrap().unwrap(), b"up");
        aw.shutdown();
        assert!(br.recv().unwrap().is_none());
    }

    #[test]
    fn wrong_identity_or_plan_is_refused_by_both_ends() {
        let (a, b) = link(vec![3; 32], vec![1; 32], [5; 32], [5; 32]);
        assert!(matches!(a, Err(HandshakeError::Rejected(_))));
        assert!(b.is_err());
        let (a, b) = link(vec![2; 32], vec![4; 32], [5; 32], [5; 32]);
        assert!(a.is_err());
        assert!(matches!(b, Err(HandshakeError::Rejected(_))));
        let (a, b) = link(vec![2; 32], vec![1; 32], [5; 32], [6; 32]);
        assert!(a.is_err());
        assert!(matches!(b, Err(HandshakeError::Rejected(_))));
    }
}
