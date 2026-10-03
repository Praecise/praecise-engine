//! SHA-256 Merkle trees over fixed-size chunks, with inclusion proofs.
//!
//! The tree shape is the one of RFC 6962: leaves are hashed with a `0x00`
//! prefix, interior nodes with `0x01`, and a list of `n > 1` leaves splits at
//! the largest power of two below `n`. A single chunk can be checked against
//! the root with `O(log n)` sibling hashes, without the rest of the payload.

use std::io::Read;

use sha2::{Digest as _, Sha256};

use crate::Error;
use crate::hash::Digest;

/// Payload chunk size: 4 MiB.
pub const CHUNK_SIZE: usize = 4 << 20;

/// Hash of one leaf (a chunk of payload bytes).
#[must_use]
pub fn leaf_hash(data: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update([0u8]);
    h.update(data);
    Digest(h.finalize().into())
}

/// Hash of an interior node.
#[must_use]
pub fn node_hash(left: &Digest, right: &Digest) -> Digest {
    let mut h = Sha256::new();
    h.update([1u8]);
    h.update(left.0);
    h.update(right.0);
    Digest(h.finalize().into())
}

fn split(n: usize) -> usize {
    debug_assert!(n > 1);
    let mut k = 1;
    while k << 1 < n {
        k <<= 1;
    }
    k
}

/// Root over leaf hashes. The empty tree's root is the hash of the empty string.
#[must_use]
pub fn root(leaves: &[Digest]) -> Digest {
    match leaves.len() {
        0 => crate::hash::sha256(&[]),
        1 => leaves[0],
        n => {
            let k = split(n);
            node_hash(&root(&leaves[..k]), &root(&leaves[k..]))
        }
    }
}

/// Inclusion proof (sibling hashes, innermost first) for leaf `index`.
///
/// # Panics
/// When `index` is out of range.
#[must_use]
pub fn proof(leaves: &[Digest], index: usize) -> Vec<Digest> {
    assert!(index < leaves.len(), "leaf index out of range");
    let mut path = Vec::new();
    proof_into(leaves, index, &mut path);
    path
}

fn proof_into(leaves: &[Digest], m: usize, path: &mut Vec<Digest>) {
    let n = leaves.len();
    if n <= 1 {
        return;
    }
    let k = split(n);
    if m < k {
        proof_into(&leaves[..k], m, path);
        path.push(root(&leaves[k..]));
    } else {
        proof_into(&leaves[k..], m - k, path);
        path.push(root(&leaves[..k]));
    }
}

/// Recomputes the root from a leaf hash and its proof; `None` when the proof
/// has the wrong length for a tree of `n` leaves.
#[must_use]
pub fn root_from_proof(leaf: &Digest, index: usize, n: usize, path: &[Digest]) -> Option<Digest> {
    if index >= n {
        return None;
    }
    if n == 1 {
        return path.is_empty().then_some(*leaf);
    }
    let (last, rest) = path.split_last()?;
    let k = split(n);
    if index < k {
        let left = root_from_proof(leaf, index, k, rest)?;
        Some(node_hash(&left, last))
    } else {
        let right = root_from_proof(leaf, index - k, n - k, rest)?;
        Some(node_hash(last, &right))
    }
}

/// Whether `chunk` is leaf `index` of the `n`-leaf tree with root `root`.
#[must_use]
pub fn verify(root: &Digest, chunk: &[u8], index: usize, n: usize, path: &[Digest]) -> bool {
    root_from_proof(&leaf_hash(chunk), index, n, path).is_some_and(|r| r == *root)
}

/// Splits a byte stream into [`CHUNK_SIZE`] chunks and hashes each as a leaf.
#[derive(Debug, Default)]
pub struct Chunker {
    buf: Vec<u8>,
    leaves: Vec<Digest>,
}

impl Chunker {
    /// An empty chunker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds more payload bytes.
    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let take = (CHUNK_SIZE - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == CHUNK_SIZE {
                self.leaves.push(leaf_hash(&self.buf));
                self.buf.clear();
            }
        }
    }

    /// The leaf hashes of every chunk, the last one possibly short.
    #[must_use]
    pub fn finish(mut self) -> Vec<Digest> {
        if !self.buf.is_empty() {
            self.leaves.push(leaf_hash(&self.buf));
        }
        self.leaves
    }
}

/// Leaf hashes of everything `reader` yields.
///
/// # Errors
/// [`Error::Io`] when reading fails.
pub fn leaves_of_reader(mut reader: impl Read) -> Result<Vec<Digest>, Error> {
    let mut chunker = Chunker::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        chunker.update(&buf[..n]);
    }
    Ok(chunker.finish())
}

/// Merkle root of a file's bytes, used as the content root of frozen base weights.
///
/// # Errors
/// [`Error::Io`] when the file cannot be read.
pub fn root_of_file(path: &std::path::Path) -> Result<Digest, Error> {
    let f = std::fs::File::open(path)?;
    Ok(root(&leaves_of_reader(std::io::BufReader::new(f))?))
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Digest> {
        (0..n).map(|i| leaf_hash(&i.to_le_bytes())).collect()
    }

    #[test]
    fn small_tree_shapes() {
        let l = leaves(3);
        assert_eq!(root(&l), node_hash(&node_hash(&l[0], &l[1]), &l[2]));
        assert_eq!(root(&l[..1]), l[0]);
    }

    #[test]
    fn every_proof_verifies_and_tampering_fails() {
        for n in 1..=17 {
            let l = leaves(n);
            let r = root(&l);
            for i in 0..n {
                let p = proof(&l, i);
                assert_eq!(root_from_proof(&l[i], i, n, &p), Some(r), "n={n} i={i}");
                let other = leaf_hash(b"other");
                assert_ne!(root_from_proof(&other, i, n, &p), Some(r));
                if !p.is_empty() {
                    assert_eq!(root_from_proof(&l[i], i, n, &p[1..]), None);
                }
            }
        }
    }

    #[test]
    fn chunker_matches_direct_hashing() {
        let data: Vec<u8> = (0..(CHUNK_SIZE * 2 + 100))
            .map(|i| (i % 251) as u8)
            .collect();
        let mut c = Chunker::new();
        for piece in data.chunks(777_777) {
            c.update(piece);
        }
        let got = c.finish();
        let want: Vec<Digest> = data.chunks(CHUNK_SIZE).map(leaf_hash).collect();
        assert_eq!(got, want);
        assert!(verify(
            &root(&want),
            &data[CHUNK_SIZE * 2..],
            2,
            3,
            &proof(&want, 2)
        ));
        assert_eq!(leaves_of_reader(&data[..]).unwrap(), want);
    }
}
