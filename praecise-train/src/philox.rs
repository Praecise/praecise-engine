//! Philox4x32-10 counter-based random numbers.
//!
//! The value at any position is a pure function of a 64-bit key and a 128-bit
//! counter, so there is no generator state to carry between steps or threads.
//! Sampling, dropout, stochastic rounding and noise draws keyed by
//! `(sample_seed, step_index, tensor_id, element_index)` replay bit for bit.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = u64::from(a) * u64::from(b);
    ((p >> 32) as u32, p as u32)
}

/// One Philox4x32 block with 10 rounds.
#[must_use]
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;
    for round in 0..10 {
        if round > 0 {
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// A keyed Philox stream: `word(stream, index)` is the `index`-th 32-bit word
/// of the stream named by two 32-bit ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Philox {
    key: [u32; 2],
}

impl Philox {
    /// A stream family keyed by `seed`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(seed: u64) -> Self {
        Self {
            key: [seed as u32, (seed >> 32) as u32],
        }
    }

    /// The 32-bit word at `index` of stream `(a, b)`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn word(&self, a: u32, b: u32, index: u64) -> u32 {
        let block = index >> 2;
        let out = philox4x32_10([block as u32, (block >> 32) as u32, a, b], self.key);
        out[(index & 3) as usize]
    }

    /// The word for one element of one tensor at one step: the keying the step
    /// contract uses for stochastic rounding and noise.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn element(&self, step_index: u64, tensor_id: u32, element_index: u64) -> u32 {
        // the step index occupies the last counter word; steps beyond 2^32 wrap into the tensor id space
        // only if a caller passes them, so the step contract caps step indices at u32::MAX
        debug_assert!(u32::try_from(step_index).is_ok());
        self.word(tensor_id, step_index as u32, element_index)
    }
}

/// Maps a word to a uniform `f32` in `[0, 1)` with 24 bits of resolution.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn uniform_f32(word: u32) -> f32 {
    (word >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Maps two words to a uniform `f64` in `[0, 1)` with 53 bits of resolution.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn uniform_f64(hi: u32, lo: u32) -> f64 {
    let bits = (u64::from(hi) << 21) ^ (u64::from(lo) >> 11);
    (bits & ((1u64 << 53) - 1)) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// A standard normal draw from two words (Box-Muller, cosine branch).
#[must_use]
pub fn normal_f64(w0: u32, w1: u32) -> f64 {
    // shift u1 away from zero so the logarithm stays finite
    let u1 = uniform_f64(w0, w0.rotate_left(16)) + f64::EPSILON;
    let u2 = uniform_f64(w1, w1.rotate_left(16));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Rounds `x` to bfloat16 stochastically: the probability of rounding up is the
/// fraction of the gap below `x`, so the expectation equals `x`. Infinities and
/// NaN are truncated unchanged.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn round_bf16_stochastic(x: f32, word: u32) -> u16 {
    let bits = x.to_bits();
    if !x.is_finite() {
        return (bits >> 16) as u16;
    }
    // adding a uniform 16-bit offset to the discarded bits carries into the kept bits
    // with exactly the probability of the discarded fraction
    let sum = bits.wrapping_add(word & 0xFFFF);
    // a carry into the exponent of the largest finite value would produce infinity; clamp it
    let out = (sum >> 16) as u16;
    if (out & 0x7F80) == 0x7F80 {
        (bits >> 16) as u16
    } else {
        out
    }
}

/// Widens a bfloat16 bit pattern to `f32`.
#[must_use]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // known-answer vectors of the reference Philox4x32-10 implementation
    #[test]
    fn known_answers() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn streams_are_pure_and_distinct() {
        let p = Philox::new(42);
        assert_eq!(p.word(1, 2, 77), p.word(1, 2, 77));
        assert_ne!(p.word(1, 2, 77), p.word(1, 3, 77));
        assert_ne!(Philox::new(43).word(1, 2, 77), p.word(1, 2, 77));
        assert_eq!(p.element(5, 9, 1000), p.word(9, 5, 1000));
    }

    #[test]
    fn uniform_ranges() {
        assert!(uniform_f32(u32::MAX) < 1.0);
        assert!((uniform_f32(0) - 0.0).abs() < f32::EPSILON);
        assert!(uniform_f64(u32::MAX, u32::MAX) < 1.0);
    }

    #[test]
    fn stochastic_rounding_is_unbiased() {
        let p = Philox::new(7);
        let x = 1.0f32 + 3.0 * 2f32.powi(-10); // between two bf16 neighbours
        let n = 200_000u64;
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += f64::from(bf16_to_f32(round_bf16_stochastic(x, p.word(0, 0, i))));
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / n as f64;
        assert!((mean - f64::from(x)).abs() < 2e-5, "mean {mean} vs {x}");
        // exactly representable values never move
        assert_eq!(bf16_to_f32(round_bf16_stochastic(1.5, 0xFFFF)), 1.5);
        assert!(bf16_to_f32(round_bf16_stochastic(f32::INFINITY, 0xFFFF)).is_infinite());
    }

    #[test]
    fn normal_moments() {
        let p = Philox::new(3);
        let n = 100_000u64;
        let (mut s, mut s2) = (0.0, 0.0);
        for i in 0..n {
            let z = normal_f64(p.word(1, 0, i), p.word(2, 0, i));
            s += z;
            s2 += z * z;
        }
        #[allow(clippy::cast_precision_loss)]
        let (mean, var) = (s / n as f64, s2 / n as f64);
        assert!(
            mean.abs() < 0.02 && (var - 1.0).abs() < 0.02,
            "mean {mean} var {var}"
        );
    }
}
