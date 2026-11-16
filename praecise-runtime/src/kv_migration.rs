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
//! The caller reserves a *scratch* sequence id (and never decodes into it);
//! the context must be built with `n_seq_max` above it. That is checked at run
//! time and refused rather than assumed.
//!
//! In a unified KV buffer (`kv_unified = true`) a delta carries only its new
//! positions: they are staged through the scratch id, where copying a position
//! range between sequences is a metadata update rather than a buffer copy.
//!
//! A context with a KV stream per sequence (`kv_unified = false`, as the batch
//! engine builds) cannot copy a range between streams. There every blob
//! carries the whole sequence, positions `0 .. end`, read from the sequence's
//! own stream, while its token list still names only the new positions, so a
//! chain is ordered and checked the same way. Applying such a blob replaces the
//! destination sequence. Its payload is written independently of the writer's
//! stream count (see [`WHOLE_MAGIC`]), so it can be applied in a context with
//! any number of streams, unified or not. A blob that carries only a position
//! range cannot be applied in a per-stream context and is refused.
//!
//! One model kind is not exported from a per-stream context: attention plus
//! recurrent layers where the attention uses a sliding window. Its state is not
//! split here into its parts, and is refused.
//!
//! ## What a blob proves
//!
//! The digest proves a blob arrived as written, the fingerprint names the
//! weights it claims, and the chain proves order: each blob names its
//! predecessor, so the token history of a rebuilt sequence is exactly the
//! concatenation of the chain's token lists. Nothing binds the cache *values*
//! to those tokens: they are the writer's computation, taken on trust, in a
//! range blob as in a whole one. A whole-sequence blob carries the cells of
//! every position, so applying it replaces the prefix earlier blobs set with
//! the writer's cells for it; it is refused unless its cells cover exactly the
//! chain's positions `0 .. end`, and since its token list names only its new
//! positions, the prefix tokens it continues are those the chain already
//! applied. A receiver that must not trust the writer recomputes instead.
//!
//! ## Size
//!
//! A whole-sequence blob grows with the sequence: measured, 112 KiB per
//! position on a 0.6B dense model with an f16 cache (3.4 MB for a 30-position
//! prompt, about 0.9 GB at 8192 positions), and 12 KiB per position plus a
//! fixed 20 MB of recurrent state on a 0.8B hybrid model. A chain of exports
//! of one running sequence sends the whole sequence each time, so its total
//! is the sum over its exports. A checkpoint taken from a request of its own
//! is one blob; a long chain costs a multiple of that.
//!
//! ## Across backends
//!
//! A payload does not depend on the backend that wrote it. The backend copies
//! every cache tensor to host memory in ggml's own layout for its type, and
//! writes per layer the key and value types and row sizes, whether values are
//! stored transposed, and the rotation dimensions. A blob written on one
//! backend (CPU, CUDA, Metal, Vulkan) therefore applies on another when the two
//! contexts store the cache alike: same key and value types (an `f16` cache
//! and a `q8_0` cache differ), and both with or without flash attention, which
//! decides whether values are transposed. When they differ, the import is
//! refused with [`Error::KvSequence`]: a whole-sequence blob's layout (see
//! [`CacheLayout`]) is read from its payload and compared with the
//! destination's, probed once on its scratch sequence, before the destination
//! is touched; a range blob is staged in the scratch sequence, where the
//! backend's own refusal leaves the destination as it was. The model
//! fingerprint names the weights only, so two engines learn of a layout
//! mismatch on import. The numbers are the
//! writer's: a destination continues from cells another backend computed,
//! which differ from its own by rounding, as any two backends' outputs do.
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
/// The blob format version this build writes. Version 2 added whole-sequence
/// payloads; this build reads version 1 too.
pub const KV_BLOB_VERSION: u16 = 2;
/// The oldest blob format version this build reads.
const KV_BLOB_MIN_VERSION: u16 = 1;
/// A payload that holds a whole sequence, positions `0 .. end`, in a form that
/// does not depend on how many KV streams the writing context had:
///
/// ```text
/// "PRKW" | part count u32 | per part: kind u8 | length u64 | bytes
/// ```
///
/// A part of kind [`PART_VERBATIM`] is passed to the backend as it is (the
/// state header, a recurrent state). A part of kind [`PART_ONE_STREAM`] is one
/// attention cache's entry for the stream that held the sequence: its cell
/// count, then its cells. The reader writes it back as a cache of its own
/// stream count with every other stream empty.
pub const WHOLE_MAGIC: &[u8; 4] = b"PRKW";
const PART_VERBATIM: u8 = 0;
const PART_ONE_STREAM: u8 = 1;
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
    /// The backend's serialized cells for exactly those positions, or, for a
    /// blob written from a context with a KV stream per sequence, the whole
    /// sequence in the stream-independent form of [`WHOLE_MAGIC`].
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
        if !(KV_BLOB_MIN_VERSION..=KV_BLOB_VERSION).contains(&version) {
            return Err(Error::KvBlob(format!(
                "unsupported KV blob version {version} (this build reads {KV_BLOB_MIN_VERSION} to {KV_BLOB_VERSION})"
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
        // Whole-sequence payloads came with version 2; a version 1 blob that
        // claims one was not written by any build.
        if version < 2 && is_whole_payload(&payload) {
            return Err(bad("a version 1 blob cannot carry a whole-sequence payload"));
        }
        Ok(Self { model: ModelFingerprint(model), prev_digest, base_pos, tokens, payload })
    }
}

/// Whether a payload holds a whole sequence (see [`WHOLE_MAGIC`]).
#[must_use]
pub fn is_whole_payload(payload: &[u8]) -> bool {
    payload.starts_with(WHOLE_MAGIC)
}

fn whole_err(m: impl Into<String>) -> Error {
    Error::KvSequence(m.into())
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4).map(|w| u32::from_le_bytes(w.try_into().expect("4 bytes")))
}

/// Frame the parts of a whole-sequence payload.
fn frame_whole(parts: &[(u8, &[u8])]) -> Vec<u8> {
    let len = 8 + parts.iter().map(|p| 9 + p.1.len()).sum::<usize>();
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(WHOLE_MAGIC);
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    for (kind, bytes) in parts {
        out.push(*kind);
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// The parts of a whole-sequence payload, in order.
fn whole_parts(payload: &[u8]) -> Result<Vec<(u8, &[u8])>> {
    let bad = || whole_err("malformed whole-sequence payload");
    if !is_whole_payload(payload) {
        return Err(bad());
    }
    let count = u32_at(payload, 4).ok_or_else(bad)?;
    let mut at = 8;
    let mut parts = Vec::new();
    for _ in 0..count {
        let kind = *payload.get(at).ok_or_else(bad)?;
        let len = payload.get(at + 1..at + 9).ok_or_else(bad)?;
        let len = usize::try_from(u64::from_le_bytes(len.try_into().expect("8 bytes"))).map_err(|_| bad())?;
        let start = at + 9;
        let bytes = payload.get(start..start.checked_add(len).ok_or_else(bad)?).ok_or_else(bad)?;
        if kind != PART_VERBATIM && kind != PART_ONE_STREAM {
            return Err(whole_err(format!("unknown whole-sequence payload part kind {kind}")));
        }
        parts.push((kind, bytes));
        at = start + len;
    }
    if at != payload.len() {
        return Err(bad());
    }
    Ok(parts)
}

/// One attention cache written for one sequence by a context of `n` streams
/// (`n u32`, then per stream a cell count and, when it is not zero, the cells)
/// holds cells in one stream at most. Return that stream's entry alone.
fn one_stream(cache: &[u8]) -> Result<Vec<u8>> {
    let n = u32_at(cache, 0).filter(|&n| n >= 1).ok_or_else(|| whole_err("attention cache without streams"))? as usize;
    let header = 4 + 4 * n;
    if cache.len() < header {
        return Err(whole_err("attention cache shorter than its stream counts"));
    }
    let Some(k) = (0..n).find(|&s| u32_at(cache, 4 + 4 * s) != Some(0)) else {
        // No stream holds a cell of this sequence.
        return if cache.len() == header {
            Ok(0u32.to_le_bytes().to_vec())
        } else {
            Err(whole_err("empty attention cache with trailing bytes"))
        };
    };
    // The streams after the one that holds the cells are empty: a zero count
    // each, at the very end.
    let end = cache.len().checked_sub(4 * (n - k - 1)).filter(|&e| e > 4 + 4 * k + 4);
    let end = end.ok_or_else(|| whole_err("attention cache too short for its streams"))?;
    if cache[end..].iter().any(|&b| b != 0) {
        return Err(whole_err("the sequence's cells span more than one KV stream"));
    }
    Ok(cache[4 + 4 * k..end].to_vec())
}

/// How one attention cache stores its cells, as its serialized form states it:
/// whether values are transposed, the rotation dimensions, and per layer the
/// key and value types with their row (or element) sizes. Two contexts whose
/// layouts differ cannot read each other's cells; the backend refuses them, but
/// only once it has emptied the destination sequence, so a whole-sequence
/// import compares layouts first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    v_trans: bool,
    n_rot_k: u32,
    n_rot_v: u32,
    /// Key type and row size per layer.
    k: Vec<(i32, u64)>,
    /// Value type, row size (element size when transposed) and, when
    /// transposed, the embedding width, per layer that has values.
    v: Vec<(i32, u64, u32)>,
}

/// The layout of one stream entry of an attention cache (a cell count, then the
/// cells), `None` when it holds no cell. Each cell's metadata may carry a fixed
/// extension (2D positions and the cell's token, 12 bytes), which the entry
/// does not announce: the size that makes the whole entry parse exactly is the
/// one.
fn cache_layout(entry: &[u8]) -> Result<Option<CacheLayout>> {
    let cells = u32_at(entry, 0).ok_or_else(|| whole_err("attention cache entry without a cell count"))?;
    if cells == 0 {
        return Ok(None);
    }
    [0usize, 12, 8, 16]
        .into_iter()
        .find_map(|ext| layout_with(entry, cells as usize, ext))
        .map(Some)
        .ok_or_else(|| whole_err("attention cache entry does not parse as cells of one layout"))
}

fn layout_with(e: &[u8], cells: usize, ext: usize) -> Option<CacheLayout> {
    let i32_at = |at: usize| u32_at(e, at).map(|v| v as i32);
    let u64_at = |at: usize| e.get(at..at + 8).map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")));
    let mut at = 4usize;
    for _ in 0..cells {
        let n_seq = u32_at(e, at + 4)? as usize;
        at = at.checked_add(8 + ext + 4 * n_seq)?;
    }
    let v_trans = u32_at(e, at)?;
    let n_layer = u32_at(e, at + 4)? as usize;
    if v_trans > 1 || n_layer == 0 || n_layer > 4096 {
        return None;
    }
    let (n_rot_k, n_rot_v) = (u32_at(e, at + 8)?, u32_at(e, at + 12)?);
    at += 16;
    let ty_ok = |t: i32| (0..256).contains(&t);
    let mut k = Vec::with_capacity(n_layer);
    for _ in 0..n_layer {
        let (ty, row) = (i32_at(at)?, u64_at(at + 4)?);
        at = at.checked_add(12)?.checked_add(usize::try_from(row).ok()?.checked_mul(cells)?)?;
        if !ty_ok(ty) || row == 0 || at > e.len() {
            return None;
        }
        k.push((ty, row));
    }
    let mut v = Vec::new();
    while at < e.len() && v.len() < n_layer {
        let ty = i32_at(at)?;
        let (size, width, bytes) = if v_trans == 0 {
            let row = u64_at(at + 4)?;
            (row, 0, usize::try_from(row).ok()?.checked_mul(cells)?)
        } else {
            let (el, width) = (u32_at(e, at + 4)?, u32_at(e, at + 8)?);
            (u64::from(el), width, (el as usize).checked_mul(width as usize)?.checked_mul(cells)?)
        };
        at = at.checked_add(12)?.checked_add(bytes)?;
        if !ty_ok(ty) || size == 0 || at > e.len() {
            return None;
        }
        v.push((ty, size, width));
    }
    (at == e.len()).then_some(CacheLayout { v_trans: v_trans == 1, n_rot_k, n_rot_v, k, v })
}

/// How many cells the first attention cache of a whole-sequence payload holds:
/// the full-attention cache, one cell per position. `None` without one.
fn whole_cells(payload: &[u8]) -> Result<Option<u32>> {
    let first = whole_parts(payload)?.into_iter().find(|p| p.0 == PART_ONE_STREAM);
    first.map(|(_, e)| u32_at(e, 0).ok_or_else(|| whole_err("attention cache entry without a cell count"))).transpose()
}

/// The layouts of a whole-sequence payload's attention caches, in order, for
/// those that hold cells.
fn whole_layouts(payload: &[u8]) -> Result<Vec<CacheLayout>> {
    let mut out = Vec::new();
    for (kind, bytes) in whole_parts(payload)? {
        if kind == PART_ONE_STREAM && let Some(l) = cache_layout(bytes)? {
            out.push(l);
        }
    }
    Ok(out)
}

/// Write a whole-sequence payload back in the backend's form for a context of
/// `streams` attention streams (`None` for a model without an attention cache).
fn expand_whole(payload: &[u8], streams: Option<u32>) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(payload.len());
    for (kind, bytes) in whole_parts(payload)? {
        if kind == PART_VERBATIM {
            out.extend_from_slice(bytes);
            continue;
        }
        let n = streams.ok_or_else(|| whole_err("the blob carries an attention cache this model does not have"))?;
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(bytes);
        for _ in 1..n {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    Ok(out)
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

    use super::{
        CacheLayout, DIGEST_LEN, ImportChain, KvBlob, ModelFingerprint, PART_ONE_STREAM, PART_VERBATIM, expand_whole,
        frame_whole, is_whole_payload, one_stream, whole_cells, whole_layouts,
    };
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
        seq_state_with(ctx, seq, LlamaStateSeqFlags::empty())
    }

    fn seq_state_with(ctx: &LlamaContext<'_>, seq: i32, flags: LlamaStateSeqFlags) -> Result<Vec<u8>> {
        let size = ctx.state_seq_get_size_ext(seq, flags);
        let mut bytes = vec![0u8; size];
        let n = ctx.state_seq_get_data_ext(&mut bytes, seq, flags);
        if n != size {
            return Err(seq_err(format!("backend wrote {n} of {size} state bytes")));
        }
        Ok(bytes)
    }

    /// Refuse contexts the scratch-sequence technique is not safe on, and
    /// return how many streams the attention cache has (`None` for a purely
    /// recurrent model, which has no attention cache).
    fn check_context(ctx: &LlamaContext<'_>, seq: i32, scratch: i32) -> Result<Option<u32>> {
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
            return Ok(None);
        }
        // An empty sequence's state is a magic and the sequence id, then the
        // attention cache's stream count and an empty cell count per stream. A
        // unified buffer has exactly one stream; per-sequence streams cannot
        // copy a partial range between sequences (the backend aborts), so
        // there a sequence moves whole.
        let bytes = seq_state(ctx, scratch)?;
        match bytes.get(8..12).map(|w| u32::from_le_bytes(w.try_into().expect("4 bytes"))) {
            Some(n) if n >= 1 => Ok(Some(n)),
            _ => Err(seq_err("the backend's sequence state has no attention stream count")),
        }
    }

    /// The state header the backend writes before a sequence's memory: a magic
    /// and the sequence id.
    const STATE_HEADER: usize = 8;

    /// Sequence `seq`'s whole state, positions `0 ..`, read from its own
    /// stream and framed so that it does not depend on the stream count.
    fn whole_state(ctx: &LlamaContext<'_>, seq: i32) -> Result<Vec<u8>> {
        let full = seq_state(ctx, seq)?;
        if full.len() < STATE_HEADER {
            return Err(seq_err("the backend's sequence state is shorter than its header"));
        }
        let (head, memory) = full.split_at(STATE_HEADER);
        let swa = ctx.model.n_swa() > 0;
        let hybrid = ctx.model.is_hybrid();
        if swa && hybrid {
            return Err(seq_err(
                "a model with recurrent layers and sliding-window attention cannot be exported from a context \
                 with a KV stream per sequence",
            ));
        }
        // A second part follows the first attention cache: the sliding-window
        // cache, or the recurrent state. Only it is written when the backend is
        // asked for the partial state, which measures where it starts.
        let tail_len = if swa || hybrid {
            let partial = seq_state_with(ctx, seq, LlamaStateSeqFlags::PARTIAL_ONLY)?;
            let tail = partial.get(STATE_HEADER..).unwrap_or_default();
            if !memory.ends_with(tail) {
                return Err(seq_err("the partial sequence state is not the end of the whole state"));
            }
            tail.len()
        } else {
            0
        };
        let (first, tail) = memory.split_at(memory.len() - tail_len);
        let first = one_stream(first)?;
        let swa_cache;
        let mut parts: Vec<(u8, &[u8])> = vec![(PART_VERBATIM, head), (PART_ONE_STREAM, &first)];
        if swa {
            swa_cache = one_stream(tail)?;
            parts.push((PART_ONE_STREAM, &swa_cache));
        } else if hybrid {
            parts.push((PART_VERBATIM, tail));
        }
        Ok(frame_whole(&parts))
    }

    /// This context's attention cache layouts, read from a few positions
    /// decoded into the empty scratch sequence and removed again. More than one
    /// cell, so the cell metadata's size is not left to chance.
    fn own_layouts(ctx: &mut LlamaContext<'_>, scratch: i32) -> Result<Vec<CacheLayout>> {
        let bos = ctx.model.token_bos();
        let token = if bos.0 >= 0 { bos } else { LlamaToken(0) };
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(4, 1);
        for pos in 0..4 {
            batch.add(token, pos, &[scratch], false).map_err(|e| seq_err(e.to_string()))?;
        }
        let decoded = ctx.decode(&mut batch).map_err(|e| seq_err(format!("probing the cache layout: {e}")));
        let state = decoded.and_then(|()| whole_state(ctx, scratch));
        let _ = ctx.clear_kv_cache_seq(Some(scratch as u32), None, None);
        whole_layouts(&state?)
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
            let streams = check_context(ctx, self.seq, self.scratch)?;
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
            } else if streams.is_some_and(|s| s > 1) {
                whole_state(ctx, self.seq)?
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
        /// This context's cache layouts, probed on the first whole-sequence
        /// blob.
        layouts: Option<Vec<CacheLayout>>,
    }

    impl SequenceImport {
        /// Import into sequence `seq`, staging through the reserved `scratch`
        /// id. `model` is the fingerprint of the weights this context runs.
        #[must_use]
        pub const fn new(model: ModelFingerprint, seq: i32, scratch: i32) -> Self {
            Self { chain: ImportChain::new(model), seq, scratch, layouts: None }
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
        /// destination sequence is unchanged, except when the backend refuses a
        /// whole-sequence payload it has started to apply: the sequence is then
        /// emptied.
        pub fn apply(&mut self, ctx: &mut LlamaContext<'_>, encoded: &[u8]) -> Result<KvBlob> {
            let blob = KvBlob::decode(encoded)?;
            self.chain.check(&blob)?;
            let streams = check_context(ctx, self.seq, self.scratch)?;
            let have = seq_len(ctx, self.seq)?;
            if have != blob.base_pos {
                return Err(seq_err(format!(
                    "destination sequence {} holds {have} positions, blob starts at {}",
                    self.seq, blob.base_pos
                )));
            }
            if !blob.tokens.is_empty() && is_whole_payload(&blob.payload) {
                // The whole sequence, applied in place: the backend empties
                // the destination sequence first, so a layout it would refuse
                // is refused here, before that.
                let covered = whole_cells(&blob.payload)?;
                if covered.is_some_and(|n| n != blob.end_pos()) {
                    return Err(seq_err(format!(
                        "the whole-sequence payload holds {} cells for positions 0..{}; the destination is unchanged",
                        covered.unwrap_or(0),
                        blob.end_pos()
                    )));
                }
                let theirs = whole_layouts(&blob.payload)?;
                if self.layouts.is_none() {
                    self.layouts = Some(own_layouts(ctx, self.scratch)?);
                }
                let ours = self.layouts.as_ref().expect("probed above");
                if &theirs != ours {
                    return Err(seq_err(format!(
                        "the blob's KV cache is stored as {theirs:?}, this context's as {ours:?} (cache types or \
                         flash attention differ); the destination is unchanged"
                    )));
                }
                let bytes = expand_whole(&blob.payload, streams)?;
                let ok = unsafe { ctx.state_seq_set_data_ext(&bytes, self.seq, LlamaStateSeqFlags::empty()) };
                let got = (ctx.kv_cache_seq_pos_min(self.seq), ctx.kv_cache_seq_pos_max(self.seq));
                let want = held_range(ctx, 0, blob.end_pos());
                if !ok || got != want {
                    let _ = ctx.clear_kv_cache_seq(Some(self.seq as u32), None, None);
                    return Err(seq_err(format!(
                        "backend rejected the whole-sequence payload or restored {got:?} instead of {want:?}; \
                         the destination sequence was emptied"
                    )));
                }
            } else if !blob.tokens.is_empty() && streams.is_some_and(|s| s > 1) {
                return Err(seq_err(
                    "this context keeps a KV stream per sequence and applies only whole-sequence blobs; this blob \
                     carries a position range",
                ));
            } else if !blob.tokens.is_empty() {
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
    /// One sequence's attention cache as a context of `n` streams writes it,
    /// its cells in stream `k`.
    fn cache(n: u32, k: u32, cells: &[u8]) -> Vec<u8> {
        let mut out = n.to_le_bytes().to_vec();
        for s in 0..n {
            if s == k {
                out.extend_from_slice(&3u32.to_le_bytes());
                out.extend_from_slice(cells);
            } else {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn a_whole_sequence_is_written_back_for_any_stream_count() {
        let cells = [5u8, 0, 0, 0, 9, 9, 0, 0];
        let entry = one_stream(&cache(5, 1, &cells)).unwrap();
        assert_eq!(entry, one_stream(&cache(1, 0, &cells)).unwrap(), "the writer's stream count is gone");
        assert_eq!(entry, one_stream(&cache(4, 3, &cells)).unwrap());
        let payload = frame_whole(&[(PART_VERBATIM, b"head"), (PART_ONE_STREAM, &entry), (PART_VERBATIM, b"tail")]);
        assert!(is_whole_payload(&payload));
        for n in [1, 2, 33] {
            let mut want = b"head".to_vec();
            want.extend_from_slice(&cache(n, 0, &cells));
            want.extend_from_slice(b"tail");
            assert_eq!(expand_whole(&payload, Some(n)).unwrap(), want, "{n} streams");
        }
        // A model without an attention cache cannot take one.
        assert!(matches!(expand_whole(&payload, None), Err(Error::KvSequence(_))));
    }

    /// An attention cache entry of `cells` cells over two layers, keys and
    /// values of type `ty` with 8-byte rows, cell metadata extended by `ext`.
    fn entry(cells: u32, ty: i32, ext: usize) -> Vec<u8> {
        let mut e = cells.to_le_bytes().to_vec();
        for p in 0..cells {
            e.extend_from_slice(&p.to_le_bytes());
            e.extend_from_slice(&1u32.to_le_bytes());
            e.extend(std::iter::repeat_n(0u8, ext));
            e.extend_from_slice(&3i32.to_le_bytes());
        }
        for w in [0u32, 2, 64, 64] {
            e.extend_from_slice(&w.to_le_bytes());
        }
        for _ in 0..4 {
            e.extend_from_slice(&ty.to_le_bytes());
            e.extend_from_slice(&8u64.to_le_bytes());
            e.extend(std::iter::repeat_n(5u8, 8 * cells as usize));
        }
        e
    }

    #[test]
    fn a_cache_layout_is_read_from_the_cells_and_compared() {
        let f16 = cache_layout(&entry(3, 1, 0)).unwrap().unwrap();
        assert_eq!(f16, cache_layout(&entry(1, 1, 0)).unwrap().unwrap(), "the layout does not depend on the cell count");
        assert_eq!(f16, cache_layout(&entry(3, 1, 12)).unwrap().unwrap(), "extended cell metadata is found");
        assert_eq!(f16, cache_layout(&entry(1, 1, 12)).unwrap().unwrap(), "one extended cell is not misread");
        assert_ne!(f16, cache_layout(&entry(3, 8, 0)).unwrap().unwrap(), "another cache type is another layout");
        assert_eq!(cache_layout(&0u32.to_le_bytes()).unwrap(), None);
        let mut cut = entry(3, 1, 0);
        cut.pop();
        assert!(cache_layout(&cut).is_err());
        let payload = frame_whole(&[(PART_VERBATIM, b"head"), (PART_ONE_STREAM, &entry(2, 1, 0))]);
        assert_eq!(whole_layouts(&payload).unwrap(), vec![f16]);
    }

    /// Arbitrary bytes and every single-byte change of valid input are refused
    /// or read, never a panic or an allocation sized from untrusted counts.
    #[test]
    fn decoding_never_panics_on_arbitrary_input() {
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let whole = frame_whole(&[(PART_VERBATIM, b"head"), (PART_ONE_STREAM, &entry(2, 1, 0))]);
        let good = KvBlob { payload: whole.clone(), ..blob(1, 0, 2, [0; 32]) }.encode();
        let check = |b: &[u8]| {
            let _ = KvBlob::decode(b);
            let _ = whole_parts(b);
            let _ = one_stream(b);
            let _ = cache_layout(b);
            let _ = whole_layouts(b);
            let _ = whole_cells(b);
            let _ = expand_whole(b, Some(3));
        };
        for _ in 0..20_000 {
            let len = (next() % 160) as usize;
            let mut b: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            if next() % 2 == 0 && b.len() >= 4 {
                b[..4].copy_from_slice(WHOLE_MAGIC);
            }
            check(&b);
        }
        for src in [&good, &whole] {
            for i in 0..src.len() {
                let mut b = src.clone();
                b[i] ^= 1 << (next() % 8);
                check(&b);
                check(&src[..i]);
            }
        }
    }

    #[test]
    fn a_whole_payload_is_refused_in_a_version_1_blob() {
        let whole = frame_whole(&[(PART_ONE_STREAM, &entry(2, 1, 0))]);
        let mut enc = KvBlob { payload: whole, ..blob(1, 0, 2, [0; 32]) }.encode();
        assert!(KvBlob::decode(&enc).is_ok());
        enc[4..6].copy_from_slice(&1u16.to_le_bytes());
        let body = enc.len() - 32;
        let digest = Sha256::digest(&enc[..body]);
        enc[body..].copy_from_slice(&digest);
        assert!(matches!(KvBlob::decode(&enc), Err(Error::KvBlob(_))));
        // A version 1 range blob is still read.
        let mut v1 = blob(1, 0, 2, [0; 32]).encode();
        v1[4..6].copy_from_slice(&1u16.to_le_bytes());
        let n = v1.len();
        let digest = Sha256::digest(&v1[..n - 32]);
        v1[n - 32..].copy_from_slice(&digest);
        assert!(KvBlob::decode(&v1).is_ok());
    }

    #[test]
    fn a_malformed_whole_sequence_is_refused() {
        // Cells in two streams are not one sequence's.
        let mut two = cache(3, 0, &[1, 2, 3, 4]);
        two[8..12].copy_from_slice(&1u32.to_le_bytes());
        two.extend_from_slice(&[7, 7, 7, 7]);
        assert!(one_stream(&two).is_err());
        assert!(one_stream(&[]).is_err());
        assert_eq!(one_stream(&cache(2, 9, &[])).unwrap(), 0u32.to_le_bytes(), "an empty cache stays empty");
        let payload = frame_whole(&[(PART_ONE_STREAM, &[3, 0, 0, 0, 1])]);
        assert!(expand_whole(&payload[..payload.len() - 1], Some(2)).is_err(), "truncated");
        let mut unknown = payload.clone();
        unknown[8] = 7;
        assert!(expand_whole(&unknown, Some(2)).is_err(), "unknown part kind");
        assert!(KvBlob::decode(&blob(1, 0, 2, [0; 32]).encode()).is_ok());
    }
}
