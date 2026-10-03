//! Stateless batch selection.
//!
//! The records of a step are `indices = PRF(sample_seed, step_index, replica)`:
//! the global sample position `step_index * batch + j` (counted per replica)
//! falls in epoch `p / n` at offset `p % n`, and a keyed permutation of `[0, n)`
//! for that epoch maps the offset to a record. Every record is visited once per
//! epoch, and no data-loader state exists to save or restore.

use crate::philox::Philox;

/// A keyed pseudorandom permutation of `[0, n)`: a balanced Feistel network on
/// the smallest even bit width covering `n`, with cycle walking.
#[derive(Debug, Clone, Copy)]
pub struct Permutation {
    n: u64,
    half_bits: u32,
    rng: Philox,
    epoch: u64,
    replica: u32,
}

impl Permutation {
    /// The permutation of `[0, n)` for one epoch of one replica.
    ///
    /// # Panics
    /// When `n` is zero.
    #[must_use]
    pub fn new(seed: u64, epoch: u64, replica: u32, n: u64) -> Self {
        assert!(n > 0, "cannot permute an empty range");
        let bits = 64 - (n - 1).leading_zeros().min(63);
        let half_bits = bits.div_ceil(2).max(1);
        Self {
            n,
            half_bits,
            rng: Philox::new(seed),
            epoch,
            replica,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn round(&self, r: u64, x: u64) -> u64 {
        let mask = (1u64 << self.half_bits) - 1;
        let w = self.rng.word(
            self.replica ^ ((r as u32) << 24),
            self.epoch as u32,
            (self.epoch >> 32) << 32 | x,
        );
        u64::from(w) & mask
    }

    fn feistel(&self, v: u64) -> u64 {
        let mask = (1u64 << self.half_bits) - 1;
        let (mut l, mut r) = (v >> self.half_bits, v & mask);
        for round in 0..4 {
            let next = l ^ self.round(round, r);
            l = r;
            r = next;
        }
        (l << self.half_bits) | r
    }

    /// The image of `i` (`i < n`).
    #[must_use]
    pub fn apply(&self, i: u64) -> u64 {
        debug_assert!(i < self.n);
        let mut v = self.feistel(i);
        while v >= self.n {
            v = self.feistel(v);
        }
        v
    }
}

/// Record indices of one step: `batch` positions starting at
/// `step_index * batch`, mapped through the epoch permutation.
///
/// # Panics
/// When `n_records` is zero.
#[must_use]
pub fn batch_indices(
    sample_seed: u64,
    step_index: u64,
    replica: u32,
    n_records: u64,
    batch: usize,
) -> Vec<u64> {
    let start = step_index * batch as u64;
    (0..batch as u64)
        .map(|j| {
            let p = start + j;
            Permutation::new(sample_seed, p / n_records, replica, n_records).apply(p % n_records)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutations_are_bijections() {
        for n in [1u64, 2, 3, 7, 64, 100, 1000] {
            let p = Permutation::new(11, 0, 0, n);
            let mut seen = vec![false; usize::try_from(n).unwrap()];
            for i in 0..n {
                let v = usize::try_from(p.apply(i)).unwrap();
                assert!(!seen[v], "n={n} repeats {v}");
                seen[v] = true;
            }
        }
    }

    #[test]
    fn epochs_cover_every_record_once_and_are_reproducible() {
        let n = 50u64;
        let batch = 10usize;
        let mut epoch0: Vec<u64> = (0..5)
            .flat_map(|s| batch_indices(3, s, 0, n, batch))
            .collect();
        assert_eq!(
            epoch0,
            (0..5)
                .flat_map(|s| batch_indices(3, s, 0, n, batch))
                .collect::<Vec<_>>()
        );
        epoch0.sort_unstable();
        assert_eq!(epoch0, (0..n).collect::<Vec<_>>());
        let epoch1: Vec<u64> = (5..10)
            .flat_map(|s| batch_indices(3, s, 0, n, batch))
            .collect();
        let first: Vec<u64> = (0..5)
            .flat_map(|s| batch_indices(3, s, 0, n, batch))
            .collect();
        assert_ne!(epoch1, first);
        assert_ne!(
            batch_indices(3, 0, 1, n, batch),
            batch_indices(3, 0, 0, n, batch)
        );
    }
}
