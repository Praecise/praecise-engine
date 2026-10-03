//! Content-hashed checkpoints with lineage.
//!
//! A checkpoint payload is the canonical tensor table followed by every tensor's
//! little-endian bytes, tensors sorted by name. The payload is cut into 4 MiB
//! chunks; the Merkle root over the chunks is the `state_root`. The manifest
//! names its parent, the frozen base, the step range and step-log head, the
//! recipe, the data and the kernel class; its hash is the checkpoint identity,
//! and parent pointers give the lineage back to the base weights.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::Error;
use crate::canonical::{Decoder, Encoder};
use crate::hash::{Digest, domain_hash};
use crate::merkle::{self, CHUNK_SIZE, Chunker};

/// Element type of a stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// bfloat16.
    BF16,
    /// 32-bit signed integer.
    I32,
    /// Raw bytes (opaque payloads such as quantized blocks).
    U8,
}

impl DType {
    /// Bytes per element.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 => 1,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 2,
            Self::I32 => 3,
            Self::U8 => 4,
        }
    }

    fn from_tag(t: u8) -> Result<Self, Error> {
        Ok(match t {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::I32,
            4 => Self::U8,
            _ => return Err(Error::Format(format!("unknown dtype tag {t}"))),
        })
    }
}

/// One named tensor of trainable state.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Element type.
    pub dtype: DType,
    /// Dimensions, innermost first (ggml order).
    pub shape: Vec<u64>,
    /// Little-endian element bytes.
    pub data: Vec<u8>,
}

impl Tensor {
    /// An F32 tensor from values.
    #[must_use]
    pub fn from_f32(shape: Vec<u64>, values: &[f32]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            dtype: DType::F32,
            shape,
            data,
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The values of an F32 tensor.
    ///
    /// # Errors
    /// [`Error::Format`] when the tensor is not F32.
    pub fn to_f32(&self) -> Result<Vec<f32>, Error> {
        if self.dtype != DType::F32 {
            return Err(Error::Format(format!(
                "expected F32 tensor, found {:?}",
                self.dtype
            )));
        }
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn validate(&self, name: &str) -> Result<(), Error> {
        let want = usize::try_from(self.numel())
            .ok()
            .and_then(|n| n.checked_mul(self.dtype.size()))
            .ok_or_else(|| Error::Format(format!("tensor {name}: size overflow")))?;
        if want == self.data.len() {
            Ok(())
        } else {
            Err(Error::Format(format!(
                "tensor {name}: {} bytes for shape {:?} of {:?} ({want} expected)",
                self.data.len(),
                self.shape,
                self.dtype
            )))
        }
    }
}

/// The trainable state of a run: parameters, optimizer moments and any other
/// buffers a step reads and writes. Names are free-form; the convention is
/// `param.<name>`, `opt.m.<name>`, `opt.v.<name>`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainState {
    tensors: BTreeMap<String, Tensor>,
}

impl TrainState {
    /// An empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a tensor.
    ///
    /// # Errors
    /// [`Error::Format`] when the byte length does not match the shape and dtype.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) -> Result<(), Error> {
        let name = name.into();
        tensor.validate(&name)?;
        self.tensors.insert(name, tensor);
        Ok(())
    }

    /// The tensor named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Tensors in canonical (name) order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }

    /// Number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the state holds no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    fn table(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.str("praecise.state.v1").u64(self.tensors.len() as u64);
        for (name, t) in &self.tensors {
            e.str(name).u8(t.dtype.tag()).u64(t.shape.len() as u64);
            for &d in &t.shape {
                e.u64(d);
            }
            e.u64(t.data.len() as u64);
        }
        e.finish()
    }

    /// The full payload: tensor table, then tensor bytes in name order.
    #[must_use]
    pub fn payload(&self) -> Vec<u8> {
        let mut out = self.table();
        for t in self.tensors.values() {
            out.extend_from_slice(&t.data);
        }
        out
    }

    /// Leaf hashes of the payload's chunks.
    #[must_use]
    pub fn chunk_leaves(&self) -> Vec<Digest> {
        let mut c = Chunker::new();
        c.update(&self.table());
        for t in self.tensors.values() {
            c.update(&t.data);
        }
        c.finish()
    }

    /// The Merkle root of the payload chunks.
    #[must_use]
    pub fn state_root(&self) -> Digest {
        merkle::root(&self.chunk_leaves())
    }

    /// Chunk `index` of the payload and its inclusion proof.
    ///
    /// # Errors
    /// [`Error::Format`] when `index` is out of range.
    pub fn chunk_with_proof(&self, index: usize) -> Result<(Vec<u8>, ChunkProof), Error> {
        let payload = self.payload();
        let leaves = self.chunk_leaves();
        if index >= leaves.len() {
            return Err(Error::Format(format!("chunk {index} of {}", leaves.len())));
        }
        let start = index * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(payload.len());
        let proof = ChunkProof {
            index,
            n_chunks: leaves.len(),
            path: merkle::proof(&leaves, index),
        };
        Ok((payload[start..end].to_vec(), proof))
    }

    /// Parses a payload written by [`TrainState::payload`].
    ///
    /// # Errors
    /// [`Error::Format`] when the payload is malformed.
    pub fn from_payload(payload: &[u8]) -> Result<Self, Error> {
        let mut d = Decoder::new(payload);
        if d.str()? != "praecise.state.v1" {
            return Err(Error::Format("not a state payload".into()));
        }
        let n = d.u64()?;
        let mut entries = Vec::new();
        for _ in 0..n {
            let name = d.str()?;
            let dtype = DType::from_tag(d.u8()?)?;
            let rank = d.u64()?;
            let shape = (0..rank).map(|_| d.u64()).collect::<Result<Vec<_>, _>>()?;
            let len =
                usize::try_from(d.u64()?).map_err(|_| Error::Format("length overflow".into()))?;
            entries.push((name, dtype, shape, len));
        }
        let mut pos = d.position();
        let mut state = Self::new();
        for (name, dtype, shape, len) in entries {
            let end = pos
                .checked_add(len)
                .filter(|&e| e <= payload.len())
                .ok_or_else(|| Error::Format(format!("tensor {name} runs past the payload")))?;
            state.insert(
                name,
                Tensor {
                    dtype,
                    shape,
                    data: payload[pos..end].to_vec(),
                },
            )?;
            pos = end;
        }
        if pos != payload.len() {
            return Err(Error::Format("trailing bytes after the last tensor".into()));
        }
        Ok(state)
    }
}

/// Inclusion proof of one payload chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkProof {
    /// Chunk index.
    pub index: usize,
    /// Number of chunks in the payload.
    pub n_chunks: usize,
    /// Sibling hashes, innermost first.
    pub path: Vec<Digest>,
}

impl ChunkProof {
    /// Whether `chunk` is the proven chunk of the payload with root `state_root`.
    #[must_use]
    pub fn verify(&self, state_root: &Digest, chunk: &[u8]) -> bool {
        chunk.len() <= CHUNK_SIZE
            && merkle::verify(state_root, chunk, self.index, self.n_chunks, &self.path)
    }
}

/// Current manifest format.
pub const MANIFEST_FORMAT: u32 = 1;

/// Checkpoint manifest. Its canonical hash is the checkpoint identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// Encoding version.
    pub format_version: u32,
    /// Identity of the checkpoint this one continues from.
    pub parent: Option<Digest>,
    /// Content root of the frozen base weights.
    pub base_root: Digest,
    /// Merkle root of this checkpoint's payload.
    pub state_root: Digest,
    /// Steps covered, `[start, end)`.
    pub step_range: (u64, u64),
    /// Head of the step log after the last covered step.
    pub step_log_head: Digest,
    /// Hash of the canonical recipe.
    pub recipe_hash: Digest,
    /// Content roots of the data shards read.
    pub data_roots: Vec<Digest>,
    /// Policy tags carried by the data.
    pub data_policy_tags: Vec<String>,
    /// Kernel class the steps ran on.
    pub kernel_class: String,
    /// Evaluation metrics, by name.
    pub eval_metrics: BTreeMap<String, f64>,
    /// Hashes of serving exports made from this state, by format name.
    pub exports: BTreeMap<String, Digest>,
}

impl Manifest {
    /// Canonical encoding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.str("praecise.manifest")
            .u32(self.format_version)
            .opt_digest(self.parent.as_ref())
            .digest(&self.base_root)
            .digest(&self.state_root)
            .u64(self.step_range.0)
            .u64(self.step_range.1)
            .digest(&self.step_log_head)
            .digest(&self.recipe_hash);
        e.u64(self.data_roots.len() as u64);
        for r in &self.data_roots {
            e.digest(r);
        }
        let mut tags = self.data_policy_tags.clone();
        tags.sort();
        tags.dedup();
        e.u64(tags.len() as u64);
        for t in &tags {
            e.str(t);
        }
        e.str(&self.kernel_class);
        e.u64(self.eval_metrics.len() as u64);
        for (k, v) in &self.eval_metrics {
            e.str(k).f64(*v);
        }
        e.u64(self.exports.len() as u64);
        for (k, v) in &self.exports {
            e.str(k).digest(v);
        }
        e.finish()
    }

    /// Parses a canonical encoding.
    ///
    /// # Errors
    /// [`Error::Format`] when the bytes are not a manifest.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut d = Decoder::new(bytes);
        if d.str()? != "praecise.manifest" {
            return Err(Error::Format("not a manifest".into()));
        }
        let format_version = d.u32()?;
        if format_version != MANIFEST_FORMAT {
            return Err(Error::Format(format!(
                "manifest format {format_version} is not supported"
            )));
        }
        let parent = d.opt_digest()?;
        let base_root = d.digest()?;
        let state_root = d.digest()?;
        let step_range = (d.u64()?, d.u64()?);
        let step_log_head = d.digest()?;
        let recipe_hash = d.digest()?;
        let data_roots = (0..d.u64()?)
            .map(|_| d.digest())
            .collect::<Result<Vec<_>, _>>()?;
        let data_policy_tags = (0..d.u64()?)
            .map(|_| d.str())
            .collect::<Result<Vec<_>, _>>()?;
        let kernel_class = d.str()?;
        let mut eval_metrics = BTreeMap::new();
        for _ in 0..d.u64()? {
            let k = d.str()?;
            eval_metrics.insert(k, d.f64()?);
        }
        let mut exports = BTreeMap::new();
        for _ in 0..d.u64()? {
            let k = d.str()?;
            exports.insert(k, d.digest()?);
        }
        d.finish()?;
        let m = Self {
            format_version,
            parent,
            base_root,
            state_root,
            step_range,
            step_log_head,
            recipe_hash,
            data_roots,
            data_policy_tags,
            kernel_class,
            eval_metrics,
            exports,
        };
        if m.encode() != bytes {
            return Err(Error::Format("manifest is not in canonical form".into()));
        }
        Ok(m)
    }

    /// Checkpoint identity: the hash of the canonical manifest.
    #[must_use]
    pub fn id(&self) -> Digest {
        domain_hash("praecise.manifest.id", &[&self.encode()])
    }
}

/// A directory of checkpoints, one subdirectory per identity.
#[derive(Debug, Clone)]
pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    /// A store rooted at `root` (created on first save).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn dir(&self, id: &Digest) -> PathBuf {
        self.root.join(id.to_hex())
    }

    /// Writes a checkpoint and returns its identity.
    ///
    /// # Errors
    /// [`Error::Mismatch`] when the manifest's `state_root` is not the state's
    /// root, [`Error::Io`] on write failure.
    pub fn save(&self, manifest: &Manifest, state: &TrainState) -> Result<Digest, Error> {
        let root = state.state_root();
        if root != manifest.state_root {
            return Err(Error::Mismatch(format!(
                "manifest state_root {} but the state hashes to {root}",
                manifest.state_root
            )));
        }
        let id = manifest.id();
        let dir = self.dir(&id);
        fs::create_dir_all(&dir)?;
        write_atomic(&dir.join("state.bin"), &state.payload())?;
        write_atomic(&dir.join("manifest.bin"), &manifest.encode())?;
        Ok(id)
    }

    /// Reads a manifest by identity, verifying it hashes to that identity.
    ///
    /// # Errors
    /// [`Error::Io`] when missing, [`Error::Mismatch`] when the stored bytes do
    /// not hash to `id`.
    pub fn manifest(&self, id: &Digest) -> Result<Manifest, Error> {
        let m = Manifest::decode(&fs::read(self.dir(id).join("manifest.bin"))?)?;
        if m.id() != *id {
            return Err(Error::Mismatch(format!(
                "stored manifest hashes to {}, not {id}",
                m.id()
            )));
        }
        Ok(m)
    }

    /// Reads a checkpoint, verifying the manifest identity and the state root.
    ///
    /// # Errors
    /// [`Error::Io`] when missing, [`Error::Mismatch`] when any hash disagrees.
    pub fn load(&self, id: &Digest) -> Result<(Manifest, TrainState), Error> {
        let m = self.manifest(id)?;
        let state = TrainState::from_payload(&fs::read(self.dir(id).join("state.bin"))?)?;
        let root = state.state_root();
        if root != m.state_root {
            return Err(Error::Mismatch(format!(
                "stored state hashes to {root}, manifest says {}",
                m.state_root
            )));
        }
        Ok((m, state))
    }

    /// The identities from `id` back through its ancestors, nearest first.
    ///
    /// # Errors
    /// [`Error::Io`] when an ancestor is missing, [`Error::Mismatch`] on a
    /// corrupt manifest or a cycle.
    pub fn lineage(&self, id: &Digest) -> Result<Vec<Digest>, Error> {
        let mut out = vec![*id];
        let mut cur = self.manifest(id)?;
        while let Some(parent) = cur.parent {
            if out.contains(&parent) {
                return Err(Error::Mismatch(format!("lineage cycle at {parent}")));
            }
            out.push(parent);
            cur = self.manifest(&parent)?;
        }
        Ok(out)
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn state(seed: f32) -> TrainState {
        let mut s = TrainState::new();
        s.insert("param.b", Tensor::from_f32(vec![3], &[seed, 2.0, 3.0]))
            .unwrap();
        s.insert(
            "param.a",
            Tensor::from_f32(vec![2, 2], &[1.0, -1.0, 0.5, seed]),
        )
        .unwrap();
        s
    }

    fn manifest(state: &TrainState, parent: Option<Digest>) -> Manifest {
        Manifest {
            format_version: MANIFEST_FORMAT,
            parent,
            base_root: crate::hash::sha256(b"base"),
            state_root: state.state_root(),
            step_range: (0, 10),
            step_log_head: crate::hash::sha256(b"log"),
            recipe_hash: crate::hash::sha256(b"recipe"),
            data_roots: vec![crate::hash::sha256(b"data")],
            data_policy_tags: vec!["b".into(), "a".into()],
            kernel_class: "cpu/x86_64/ks1".into(),
            eval_metrics: BTreeMap::from([("loss".into(), 1.25)]),
            exports: BTreeMap::new(),
        }
    }

    #[test]
    fn payload_round_trip_and_root_is_order_independent() {
        let s = state(4.0);
        assert_eq!(TrainState::from_payload(&s.payload()).unwrap(), s);
        let mut t = TrainState::new();
        t.insert("param.a", s.get("param.a").unwrap().clone())
            .unwrap();
        t.insert("param.b", s.get("param.b").unwrap().clone())
            .unwrap();
        assert_eq!(t.state_root(), s.state_root());
        assert_ne!(state(5.0).state_root(), s.state_root());
    }

    #[test]
    fn bad_lengths_are_refused() {
        let mut s = TrainState::new();
        let t = Tensor {
            dtype: DType::F32,
            shape: vec![3],
            data: vec![0; 8],
        };
        assert!(s.insert("x", t).is_err());
    }

    #[test]
    fn chunk_proofs_verify() {
        let mut s = TrainState::new();
        let big: Vec<f32> = (0..(CHUNK_SIZE / 4 + 1000)).map(|i| i as f32).collect();
        s.insert("param.big", Tensor::from_f32(vec![big.len() as u64], &big))
            .unwrap();
        let root = s.state_root();
        let (chunk, proof) = s.chunk_with_proof(1).unwrap();
        assert_eq!(proof.n_chunks, 2);
        assert!(proof.verify(&root, &chunk));
        let mut bad = chunk.clone();
        bad[0] ^= 1;
        assert!(!proof.verify(&root, &bad));
    }

    #[test]
    fn manifest_round_trip_and_canonical_tags() {
        let s = state(1.0);
        let m = manifest(&s, None);
        let d = Manifest::decode(&m.encode()).unwrap();
        assert_eq!(d.data_policy_tags, vec!["a".to_string(), "b".to_string()]);
        let mut m2 = m.clone();
        m2.data_policy_tags = vec!["a".into(), "b".into(), "a".into()];
        assert_eq!(m2.id(), m.id());
    }

    #[test]
    fn store_save_load_lineage() {
        let dir = std::env::temp_dir().join(format!("praecise-train-ckpt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = CheckpointStore::new(&dir);
        let s0 = state(1.0);
        let id0 = store.save(&manifest(&s0, None), &s0).unwrap();
        let s1 = state(2.0);
        let id1 = store.save(&manifest(&s1, Some(id0)), &s1).unwrap();
        let (m1, l1) = store.load(&id1).unwrap();
        assert_eq!(l1, s1);
        assert_eq!(m1.parent, Some(id0));
        assert_eq!(store.lineage(&id1).unwrap(), vec![id1, id0]);
        // a manifest that disagrees with its state is refused
        let mut wrong = manifest(&s1, None);
        wrong.state_root = s0.state_root();
        assert!(matches!(store.save(&wrong, &s1), Err(Error::Mismatch(_))));
        // tampering with the stored state is detected on load
        let p = dir.join(id0.to_hex()).join("state.bin");
        let mut bytes = fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&p, bytes).unwrap();
        assert!(matches!(store.load(&id0), Err(Error::Mismatch(_))));
        let _ = fs::remove_dir_all(&dir);
    }
}
