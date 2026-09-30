//! Moving a live sequence's KV cache between engine instances.
//!
//! A sequence's attention cache is append-only: once a position is decoded its
//! keys and values never change. That makes it safe to ship the cache in
//! pieces. [`SequenceExport`] captures the positions appended since its last
//! export as a [`KvBlob`]; [`SequenceImport`] applies a chain of blobs, in
//! order, to a sequence in another context of the same model, which can then
//! keep decoding without re-running prefill over the transferred prefix.
//!
//! ## The blob
//!
//! A blob is self-describing and integrity-checked:
//!
//! ```text
//! magic "PRKV" | version u16 | reserved u16
//! model fingerprint [32]      -- SHA-256 of the weights file
//! previous blob digest [32]   -- zero for the first blob of a chain
//! base position u32 | token count u32
//! tokens i32 x count          -- the token ids at base..base+count
//! payload length u64 | payload -- the backend's serialized cache cells
//! digest [32]                 -- SHA-256 of every byte before it
//! ```
//!
//! Integrity, model identity and chain order are all checked before the
//! backend sees a byte of the payload: a corrupt blob, a blob made from other
//! weights, and an out-of-order or skipped delta are each refused with a typed
//! error and leave the destination sequence untouched.
//!
//! ## What the context must provide
//!
//! Incremental export and import stage cells through a *scratch* sequence id
//! that the caller reserves (and never decodes into). The context must be
//! built with `n_seq_max` above the scratch id and with a unified KV buffer
//! (`kv_unified = true`), so that copying a position range between sequences
//! is a metadata update rather than a buffer copy. Both are checked at run
//! time and refused rather than assumed.
//!
//! ## Recurrent and hybrid models
//!
//! A recurrent layer (Mamba, RWKV, Gated Delta Net and the like) keeps one
//! fixed-size state per sequence that every new position overwrites, so it is
//! not the concatenation of per-position deltas. For a model with recurrent
//! layers, every blob carries its attention cells for the new positions as
//! usual, plus the recurrent state as it stands after the blob's last position.
//! Applying a blob replaces the destination's recurrent state with it, which
//! is correct precisely because chain order is enforced: the state in the last
//! applied blob is the state at the position the sequence is rebuilt to. A
//! purely recurrent model has no attention cells, and its blobs carry the state
//! alone.

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const MAGIC: &[u8; 4] = b"PRKV";
/// The blob format version this build writes and reads.
pub const KV_BLOB_VERSION: u16 = 1;
const DIGEST_LEN: usize = 32;
const HEADER_LEN: usize = 4 + 2 + 2 + DIGEST_LEN + DIGEST_LEN + 4 + 4;

/// Identity of a set of model weights: the SHA-256 of the weights file.
///
/// Two engines may exchange KV state only if their fingerprints are equal;
/// a cache computed by one set of weights is meaningless to another even when
/// the tensor shapes happen to line up.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelFingerprint(pub [u8; DIGEST_LEN]);

impl ModelFingerprint {
    /// Hash a weights file, streaming it (no full read into memory).
    ///
    /// # Errors
    /// Any I/O error reading the file.
    pub fn of_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(Self(hasher.finalize().into()))
    }

    /// Hash weights already in memory (for example a staged mapping).
    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Use a digest the host already trusts for these weights (for example the
    /// content digest it fetched them by), avoiding a second full read.
    #[must_use]
    pub const fn from_digest(digest: [u8; DIGEST_LEN]) -> Self {
        Self(digest)
    }

    /// Lower-case hex form.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl std::fmt::Debug for ModelFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelFingerprint({})", self.to_hex())
    }
}

/// One exported piece of a sequence's KV cache: the cells for positions
/// `base_pos .. base_pos + tokens.len()`.
#[derive(Clone, PartialEq, Eq)]
pub struct KvBlob {
    /// Weights the cache was computed with.
    pub model: ModelFingerprint,
    /// Digest of the blob this one extends; all zero for the first blob.
    pub prev_digest: [u8; DIGEST_LEN],
    /// First position this blob carries.
    pub base_pos: u32,
    /// Token ids at `base_pos ..`, one per carried position.
    pub tokens: Vec<i32>,
    /// The backend's serialized cells for exactly those positions.
    pub payload: Vec<u8>,
}

impl std::fmt::Debug for KvBlob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvBlob")
            .field("model", &self.model)
            .field("base_pos", &self.base_pos)
            .field("tokens", &self.tokens.len())
            .field("payload_bytes", &self.payload.len())
            .finish_non_exhaustive()
    }
}

impl KvBlob {
    /// Position one past the last carried cell.
    #[must_use]
    pub fn end_pos(&self) -> u32 {
        self.base_pos + self.tokens.len() as u32
    }

    /// Serialize, appending the integrity digest.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(HEADER_LEN + self.tokens.len() * 4 + 8 + self.payload.len() + DIGEST_LEN);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&KV_BLOB_VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&self.model.0);
        out.extend_from_slice(&self.prev_digest);
        out.extend_from_slice(&self.base_pos.to_le_bytes());
        out.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for t in &self.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.payload);
        let digest = Sha256::digest(&out);
        out.extend_from_slice(&digest);
        out
    }

    /// The digest an encoded blob ends with; the next blob in the chain names
    /// it as its `prev_digest`.
    ///
    /// # Errors
    /// [`Error::KvBlob`] if `encoded` is too short to hold one.
    pub fn digest_of(encoded: &[u8]) -> Result<[u8; DIGEST_LEN]> {
        if encoded.len() < DIGEST_LEN {
            return Err(Error::KvBlob("blob shorter than its digest".into()));
        }
        let mut d = [0u8; DIGEST_LEN];
        d.copy_from_slice(&encoded[encoded.len() - DIGEST_LEN..]);
        Ok(d)
    }

    /// Parse and verify an encoded blob: magic, version, lengths and digest.
    ///
    /// # Errors
    /// [`Error::KvBlob`] for any malformed, truncated, unknown-version or
    /// corrupted input.
    pub fn decode(encoded: &[u8]) -> Result<Self> {
        let bad = |m: &str| Error::KvBlob(m.to_string());
        if encoded.len() < HEADER_LEN + 8 + DIGEST_LEN {
            return Err(bad("blob truncated"));
        }
        let (body, digest) = encoded.split_at(encoded.len() - DIGEST_LEN);
        if Sha256::digest(body).as_slice() != digest {
            return Err(bad("blob digest mismatch (corrupted or truncated)"));
        }
        if &body[0..4] != MAGIC {
            return Err(bad("not a KV blob (bad magic)"));
        }
        let version = u16::from_le_bytes([body[4], body[5]]);
        if version != KV_BLOB_VERSION {
            return Err(Error::KvBlob(format!(
                "unsupported KV blob version {version} (this build reads {KV_BLOB_VERSION})"
            )));
        }
        let mut at = 8;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = body.get(at..at + n).ok_or_else(|| bad("blob truncated"))?;
            at += n;
            Ok(s)
        };
        let mut model = [0u8; DIGEST_LEN];
        model.copy_from_slice(take(DIGEST_LEN)?);
        let mut prev_digest = [0u8; DIGEST_LEN];
        prev_digest.copy_from_slice(take(DIGEST_LEN)?);
        let base_pos = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        let n_tokens = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let token_bytes = take(n_tokens.checked_mul(4).ok_or_else(|| bad("token count overflow"))?)?;
        let tokens = token_bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        let payload_len = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let payload_len = usize::try_from(payload_len).map_err(|_| bad("payload length overflow"))?;
        let payload = take(payload_len)?.to_vec();
        if at != body.len() {
            return Err(bad("trailing bytes after payload"));
        }
        if base_pos.checked_add(n_tokens as u32).is_none() {
            return Err(bad("position range overflow"));
        }
        Ok(Self { model: ModelFingerprint(model), prev_digest, base_pos, tokens, payload })
    }
}

/// Import-side chain state: which weights, and how far the sequence has been
/// rebuilt. Backend-agnostic, so the ordering rules are testable on their own.
#[derive(Debug, Clone)]
pub struct ImportChain {
    model: ModelFingerprint,
    next_pos: u32,
    last_digest: [u8; DIGEST_LEN],
}

impl ImportChain {
    /// A chain that expects its first blob to start at position 0.
    #[must_use]
    pub const fn new(model: ModelFingerprint) -> Self {
        Self { model, next_pos: 0, last_digest: [0; DIGEST_LEN] }
    }

    /// Position the next blob must start at.
    #[must_use]
    pub const fn next_pos(&self) -> u32 {
        self.next_pos
    }

    /// Check that `blob` (already integrity-verified) may be applied next.
    ///
    /// # Errors
    /// [`Error::ModelMismatch`] if it was made with other weights;
    /// [`Error::KvSequence`] if it is not the next link in the chain.
    pub fn check(&self, blob: &KvBlob) -> Result<()> {
        if blob.model != self.model {
            return Err(Error::ModelMismatch { expected: self.model.to_hex(), found: blob.model.to_hex() });
        }
        if blob.base_pos != self.next_pos || blob.prev_digest != self.last_digest {
            return Err(Error::KvSequence(format!(
                "blob starts at position {} but the sequence is rebuilt to {}, or it does not extend the last applied blob",
                blob.base_pos, self.next_pos
            )));
        }
        Ok(())
    }

    /// Record that `blob`, whose encoded form ends with `digest`, was applied.
    /// A backend integration calls this after it installs the cells.
    pub fn advance(&mut self, blob: &KvBlob, digest: [u8; DIGEST_LEN]) {
        self.next_pos = blob.end_pos();
        self.last_digest = digest;
    }
}

#[cfg(feature = "bundled-llama")]
pub use backend::{SequenceExport, SequenceImport};

#[cfg(feature = "bundled-llama")]
mod backend {
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::context::session::LlamaStateSeqFlags;
    use llama_cpp_2::token::LlamaToken;

    use super::{DIGEST_LEN, ImportChain, KvBlob, ModelFingerprint};
    use crate::error::{Error, Result};

    fn seq_err(m: impl Into<String>) -> Error {
        Error::KvSequence(m.into())
    }

    /// Whether the model keeps recurrent state beside (or instead of) an
    /// attention cache.
    fn has_recurrent_state(ctx: &LlamaContext<'_>) -> bool {
        ctx.model.is_recurrent() || ctx.model.is_hybrid()
    }

    /// Position range `(min, max)` a sequence reports once it holds cells for
    /// `p0 .. n`. A recurrent state reports only its last position, and a
    /// hybrid memory reports the intersection of its parts, so with recurrent
    /// state both ends are `n - 1`.
    fn held_range(ctx: &LlamaContext<'_>, p0: u32, n: u32) -> (i32, i32) {
        let last = n as i32 - 1;
        if has_recurrent_state(ctx) { (last, last) } else { (p0 as i32, last) }
    }

    /// Number of positions a sequence holds, requiring them to be `0..n`.
    fn seq_len(ctx: &LlamaContext<'_>, seq: i32) -> Result<u32> {
        let max = ctx.kv_cache_seq_pos_max(seq);
        if max < 0 {
            return Ok(0);
        }
        let min = ctx.kv_cache_seq_pos_min(seq);
        // With recurrent state the reported minimum is the state's position,
        // the last one; the attention part cannot be asked on its own.
        if !has_recurrent_state(ctx) && min != 0 {
            return Err(seq_err(format!(
                "sequence {seq} starts at position {min}, not 0; only a full prefix can be migrated"
            )));
        }
        Ok(max as u32 + 1)
    }

    /// Serialize one sequence's cells.
    fn seq_state(ctx: &LlamaContext<'_>, seq: i32) -> Result<Vec<u8>> {
        let flags = LlamaStateSeqFlags::empty();
        let size = ctx.state_seq_get_size_ext(seq, flags);
        let mut bytes = vec![0u8; size];
        let n = ctx.state_seq_get_data_ext(&mut bytes, seq, flags);
        if n != size {
            return Err(seq_err(format!("backend wrote {n} of {size} state bytes")));
        }
        Ok(bytes)
    }

    /// Refuse contexts the scratch-sequence technique is not safe on.
    fn check_context(ctx: &LlamaContext<'_>, seq: i32, scratch: i32) -> Result<()> {
        if seq == scratch || seq < 0 || scratch < 0 {
            return Err(seq_err("sequence and scratch ids must be distinct and non-negative"));
        }
        let n_seq_max = ctx.n_seq_max();
        if u32::try_from(seq.max(scratch)).map_or(true, |m| m >= n_seq_max) {
            return Err(seq_err(format!(
                "context has n_seq_max {n_seq_max}; sequence {seq} and scratch {scratch} must both fit"
            )));
        }
        if ctx.kv_cache_seq_pos_max(scratch) >= 0 {
            return Err(seq_err(format!("scratch sequence {scratch} is not empty")));
        }
        // A purely recurrent model has no attention cells to copy by range.
        if ctx.model.is_recurrent() {
            return Ok(());
        }
        // An empty sequence's state is a magic and the sequence id, then the
        // attention cache's stream count and an empty cell count per stream. A
        // unified buffer has exactly one stream; per-sequence streams cannot
        // copy a partial range between sequences (the backend aborts).
        let bytes = seq_state(ctx, scratch)?;
        if bytes.len() < 12 || u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) != 1 {
            return Err(seq_err("context KV buffer is not unified (build it with kv_unified = true)"));
        }
        Ok(())
    }

    /// Exports one sequence's KV cache incrementally.
    #[derive(Debug, Clone)]
    pub struct SequenceExport {
        model: ModelFingerprint,
        seq: i32,
        scratch: i32,
        exported_to: u32,
        last_digest: [u8; DIGEST_LEN],
    }

    impl SequenceExport {
        /// Export sequence `seq`, staging through the reserved `scratch` id.
        #[must_use]
        pub const fn new(model: ModelFingerprint, seq: i32, scratch: i32) -> Self {
            Self { model, seq, scratch, exported_to: 0, last_digest: [0; DIGEST_LEN] }
        }

        /// Positions already exported.
        #[must_use]
        pub const fn exported_to(&self) -> u32 {
            self.exported_to
        }

        /// Export the positions appended since the last call (all of them on
        /// the first call) as an encoded [`KvBlob`].
        ///
        /// `tokens` is the sequence's full token history, one per position; its
        /// length must equal the number of positions the context holds.
        ///
        /// # Errors
        /// [`Error::KvSequence`] if the context cannot be exported from, if
        /// `tokens` disagrees with the cache, or if the sequence was rewound
        /// below what was already exported (start a new exporter then).
        pub fn export(&mut self, ctx: &mut LlamaContext<'_>, tokens: &[LlamaToken]) -> Result<Vec<u8>> {
            check_context(ctx, self.seq, self.scratch)?;
            let n = seq_len(ctx, self.seq)?;
            if tokens.len() != n as usize {
                return Err(seq_err(format!(
                    "sequence {} holds {n} positions but {} tokens were given",
                    self.seq,
                    tokens.len()
                )));
            }
            if n < self.exported_to {
                return Err(seq_err(format!(
                    "sequence {} was rewound to {n} below the exported {}",
                    self.seq, self.exported_to
                )));
            }
            let p0 = self.exported_to;
            let payload = if n == p0 {
                Vec::new()
            } else {
                ctx.copy_kv_cache_seq(self.seq, self.scratch, Some(p0), Some(n))
                    .map_err(|e| seq_err(e.to_string()))?;
                let got = (ctx.kv_cache_seq_pos_min(self.scratch), ctx.kv_cache_seq_pos_max(self.scratch));
                let state = seq_state(ctx, self.scratch);
                let _ = ctx.clear_kv_cache_seq(Some(self.scratch as u32), None, None);
                if got != held_range(ctx, p0, n) {
                    return Err(seq_err(format!(
                        "staged range {got:?} does not match positions {p0}..{n}"
                    )));
                }
                state?
            };
            let blob = KvBlob {
                model: self.model,
                prev_digest: self.last_digest,
                base_pos: p0,
                tokens: tokens[p0 as usize..].iter().map(|t| t.0).collect(),
                payload,
            };
            let encoded = blob.encode();
            self.last_digest = KvBlob::digest_of(&encoded)?;
            self.exported_to = n;
            Ok(encoded)
        }
    }

    /// Rebuilds one sequence from a chain of exported blobs.
    #[derive(Debug, Clone)]
    pub struct SequenceImport {
        chain: ImportChain,
        seq: i32,
        scratch: i32,
    }

    impl SequenceImport {
        /// Import into sequence `seq`, staging through the reserved `scratch`
        /// id. `model` is the fingerprint of the weights this context runs.
        #[must_use]
        pub const fn new(model: ModelFingerprint, seq: i32, scratch: i32) -> Self {
            Self { chain: ImportChain::new(model), seq, scratch }
        }

        /// Position the sequence is rebuilt to: the next position to decode.
        #[must_use]
        pub const fn next_pos(&self) -> u32 {
            self.chain.next_pos()
        }

        /// Verify and apply the next encoded blob. Returns the decoded blob
        /// (its tokens extend the caller's token history).
        ///
        /// # Errors
        /// [`Error::KvBlob`] for a corrupt blob, [`Error::ModelMismatch`] for
        /// other weights, [`Error::KvSequence`] for an out-of-order blob or a
        /// destination that is not in the expected state. On any error the
        /// destination sequence is unchanged.
        pub fn apply(&mut self, ctx: &mut LlamaContext<'_>, encoded: &[u8]) -> Result<KvBlob> {
            let blob = KvBlob::decode(encoded)?;
            self.chain.check(&blob)?;
            check_context(ctx, self.seq, self.scratch)?;
            let have = seq_len(ctx, self.seq)?;
            if have != blob.base_pos {
                return Err(seq_err(format!(
                    "destination sequence {} holds {have} positions, blob starts at {}",
                    self.seq, blob.base_pos
                )));
            }
            if !blob.tokens.is_empty() {
                let ok = unsafe {
                    ctx.state_seq_set_data_ext(&blob.payload, self.scratch, LlamaStateSeqFlags::empty())
                };
                let got = (ctx.kv_cache_seq_pos_min(self.scratch), ctx.kv_cache_seq_pos_max(self.scratch));
                let want = held_range(ctx, blob.base_pos, blob.end_pos());
                if !ok || got != want {
                    let _ = ctx.clear_kv_cache_seq(Some(self.scratch as u32), None, None);
                    return Err(seq_err(format!(
                        "backend rejected the payload or restored {got:?} instead of {want:?}"
                    )));
                }
                ctx.copy_kv_cache_seq(self.scratch, self.seq, None, None)
                    .map_err(|e| seq_err(e.to_string()))?;
                let _ = ctx.clear_kv_cache_seq(Some(self.scratch as u32), None, None);
                let now = seq_len(ctx, self.seq)?;
                if now != blob.end_pos() {
                    return Err(seq_err(format!(
                        "sequence {} holds {now} positions after import, expected {}",
                        self.seq,
                        blob.end_pos()
                    )));
                }
            }
            self.chain.advance(&blob, KvBlob::digest_of(encoded)?);
            Ok(blob)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(model: u8, base: u32, n: usize, prev: [u8; 32]) -> KvBlob {
        KvBlob {
            model: ModelFingerprint([model; 32]),
            prev_digest: prev,
            base_pos: base,
            tokens: (0..n as i32).collect(),
            payload: vec![7; n * 3],
        }
    }

    #[test]
    fn round_trips() {
        let b = blob(1, 0, 5, [0; 32]);
        assert_eq!(KvBlob::decode(&b.encode()).unwrap(), b);
    }

    #[test]
    fn corruption_is_refused() {
        let mut enc = blob(1, 0, 5, [0; 32]).encode();
        let mid = enc.len() / 2;
        enc[mid] ^= 1;
        assert!(matches!(KvBlob::decode(&enc), Err(Error::KvBlob(_))));
        let enc = blob(1, 0, 5, [0; 32]).encode();
        assert!(matches!(KvBlob::decode(&enc[..enc.len() - 1]), Err(Error::KvBlob(_))));
    }

    #[test]
    fn model_mismatch_is_refused() {
        let chain = ImportChain::new(ModelFingerprint([2; 32]));
        assert!(matches!(chain.check(&blob(1, 0, 4, [0; 32])), Err(Error::ModelMismatch { .. })));
    }

    #[test]
    fn chain_order_is_enforced() {
        let first = blob(1, 0, 4, [0; 32]);
        let enc = first.encode();
        let mut chain = ImportChain::new(ModelFingerprint([1; 32]));
        // A delta before its base is refused.
        let d = KvBlob::digest_of(&enc).unwrap();
        assert!(matches!(chain.check(&blob(1, 4, 2, d)), Err(Error::KvSequence(_))));
        chain.check(&first).unwrap();
        chain.advance(&first, d);
        // A delta that names a different predecessor is refused.
        assert!(matches!(chain.check(&blob(1, 4, 2, [9; 32])), Err(Error::KvSequence(_))));
        chain.check(&blob(1, 4, 2, d)).unwrap();
    }
}
