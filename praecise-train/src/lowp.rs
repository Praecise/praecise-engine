//! Block-scaled low-precision formats for training: MXFP8 and NVFP4.
//!
//! These are the reference encodings the device kernels are checked against.
//!
//! - MXFP8: blocks of 32 values share a power-of-two scale (E8M0); elements are FP8 E4M3. The
//!   scale is rounded toward +inf, `2^ceil(log2(amax / 448))`, so no element of a block saturates.
//! - NVFP4: blocks of 16 values share an E4M3 scale, under one FP32 scale per tensor; elements are
//!   FP4 E2M1. Block scales are also rounded toward +inf. Blocks run along rows (1x16) for
//!   activations and gradients, or are 16x16 tiles for weights, so a weight and its transpose
//!   quantize to the same values (2D scaling).
//! - Elements round to nearest even, or stochastically with Philox words keyed by step, tensor and
//!   element, so a run replays bit for bit.
//! - A 16-point random Hadamard transform with Philox signs spreads outliers across a block
//!   before quantization; it is orthonormal and its inverse is exact up to f32 rounding.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "bit-level encodings of small floating-point formats"
)]

use crate::Error;
use crate::philox::{Philox, uniform_f32};

/// A sign-magnitude minifloat without infinities.
#[derive(Debug, Clone, Copy)]
struct Minifloat {
    ebits: u32,
    mbits: u32,
    bias: i32,
    max: f32,
}

const E4M3: Minifloat = Minifloat { ebits: 4, mbits: 3, bias: 7, max: 448.0 };
const E2M1: Minifloat = Minifloat { ebits: 2, mbits: 1, bias: 1, max: 6.0 };

/// How a value is rounded onto the grid of a format.
#[derive(Debug, Clone, Copy)]
pub enum Rounding<'a> {
    /// Round to nearest, ties to even.
    Nearest,
    /// Round up or down with probability given by the distance, using the Philox word of each
    /// element `(step, tensor_id, element index)`.
    Stochastic {
        /// Key of the run.
        rng: &'a Philox,
        /// Step index.
        step: u64,
        /// Tensor identifier.
        tensor_id: u32,
    },
}

#[derive(Clone, Copy)]
enum Mode {
    Nearest,
    Up,
    Stochastic(f32),
}

/// Unbiased binary exponent of a positive finite f32 (subnormals report -127).
fn exponent(a: f32) -> i32 {
    ((a.to_bits() >> 23) & 0xff) as i32 - 127
}

fn round_half_even(t: f32) -> f32 {
    let fl = t.floor();
    let d = t - fl;
    if d > 0.5 || (d == 0.5 && fl % 2.0 != 0.0) { fl + 1.0 } else { fl }
}

impl Minifloat {
    fn emin(self) -> i32 {
        1 - self.bias
    }

    /// Rounds a non-negative finite magnitude onto the grid, saturating at the maximum.
    fn round(self, a: f32, mode: Mode) -> f32 {
        if a == 0.0 {
            return 0.0;
        }
        let e = exponent(a).max(self.emin());
        let quantum = 2f32.powi(e - self.mbits as i32);
        let t = a / quantum;
        let r = match mode {
            Mode::Nearest => round_half_even(t),
            Mode::Up => t.ceil(),
            Mode::Stochastic(u) => (t + u).floor(),
        };
        (r * quantum).min(self.max)
    }

    /// Code of a value already on the grid.
    fn encode(self, q: f32) -> u8 {
        let sign = u8::from(q.is_sign_negative() && q != 0.0) << (self.ebits + self.mbits);
        let a = q.abs();
        let (e_field, mant) = if a < 2f32.powi(self.emin()) {
            (0, a / 2f32.powi(self.emin() - self.mbits as i32))
        } else {
            let e = exponent(a);
            (e + self.bias, a / 2f32.powi(e - self.mbits as i32) - (1u32 << self.mbits) as f32)
        };
        sign | ((e_field as u8) << self.mbits) | mant as u8
    }

    fn decode(self, code: u8) -> f32 {
        let mant = u32::from(code) & ((1 << self.mbits) - 1);
        let e_field = (u32::from(code) >> self.mbits) & ((1 << self.ebits) - 1);
        let a = if e_field == 0 {
            mant as f32 * 2f32.powi(self.emin() - self.mbits as i32)
        } else {
            (1.0 + mant as f32 / (1u32 << self.mbits) as f32) * 2f32.powi(e_field as i32 - self.bias)
        };
        if code >> (self.ebits + self.mbits) & 1 == 1 { -a } else { a }
    }

    fn quantize(self, x: f32, mode: Mode) -> u8 {
        let q = self.round(x.abs(), mode);
        self.encode(if x.is_sign_negative() { -q } else { q })
    }
}

fn mode(rounding: Rounding<'_>, index: u64) -> Mode {
    match rounding {
        Rounding::Nearest => Mode::Nearest,
        Rounding::Stochastic { rng, step, tensor_id } => Mode::Stochastic(uniform_f32(rng.element(step, tensor_id, index))),
    }
}

fn check_finite(x: &[f32]) -> Result<(), Error> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(Error::Refused(format!("element {i} is not finite"))),
        None => Ok(()),
    }
}

/// An MXFP8 tensor: one E8M0 scale (exponent biased by 127) per 32 values and one E4M3 code per
/// value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp8 {
    /// Biased scale exponents.
    pub scales: Vec<u8>,
    /// Element codes.
    pub codes: Vec<u8>,
}

/// Values per MXFP8 block.
pub const MX_BLOCK: usize = 32;

/// Quantizes `x` to MXFP8, blocks of 32 consecutive values (the last may be shorter).
///
/// # Errors
/// [`Error::Refused`] on a non-finite value.
pub fn mxfp8_quantize(x: &[f32], rounding: Rounding<'_>) -> Result<Mxfp8, Error> {
    check_finite(x)?;
    let mut scales = Vec::with_capacity(x.len().div_ceil(MX_BLOCK));
    let mut codes = Vec::with_capacity(x.len());
    for (b, block) in x.chunks(MX_BLOCK).enumerate() {
        let amax = f64::from(block.iter().fold(0f32, |m, v| m.max(v.abs())));
        let max = f64::from(E4M3.max);
        // smallest power of two with amax / 2^e <= 448
        let mut e = if amax == 0.0 { -127 } else { (amax / max).log2().ceil() as i32 };
        while e < 127 && amax / 2f64.powi(e) > max {
            e += 1;
        }
        while e > -127 && amax / 2f64.powi(e - 1) <= max {
            e -= 1;
        }
        let e = e.clamp(-127, 127);
        scales.push((e + 127) as u8);
        for (i, &v) in block.iter().enumerate() {
            let y = (f64::from(v) / 2f64.powi(e)) as f32;
            codes.push(E4M3.quantize(y, mode(rounding, (b * MX_BLOCK + i) as u64)));
        }
    }
    Ok(Mxfp8 { scales, codes })
}

/// Values of an MXFP8 tensor.
#[must_use]
pub fn mxfp8_dequantize(q: &Mxfp8) -> Vec<f32> {
    q.codes
        .iter()
        .enumerate()
        .map(|(i, &c)| (f64::from(E4M3.decode(c)) * 2f64.powi(i32::from(q.scales[i / MX_BLOCK]) - 127)) as f32)
        .collect()
}

/// Block layout of an NVFP4 tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Layout {
    /// 16 consecutive values of a row (activations, gradients).
    Rows,
    /// 16x16 tiles (weights): the transpose quantizes to the transposed values.
    Tiles,
}

/// Side of an NVFP4 block.
pub const NV_BLOCK: usize = 16;

/// An NVFP4 tensor of `rows x cols` row-major values.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4 {
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub cols: usize,
    /// Block layout.
    pub layout: Nvfp4Layout,
    /// Per-tensor scale.
    pub tensor_scale: f32,
    /// E4M3 block scale codes, blocks in row-major block order.
    pub block_scales: Vec<u8>,
    /// E2M1 element codes (low nibble), one per value.
    pub codes: Vec<u8>,
}

impl Nvfp4 {
    fn block(&self, r: usize, c: usize) -> usize {
        block_of(self.layout, self.cols, r, c)
    }
}

fn block_of(layout: Nvfp4Layout, cols: usize, r: usize, c: usize) -> usize {
    let row_div = match layout {
        Nvfp4Layout::Rows => 1,
        Nvfp4Layout::Tiles => NV_BLOCK,
    };
    (r / row_div) * cols.div_ceil(NV_BLOCK) + c / NV_BLOCK
}

/// Quantizes a row-major `rows x cols` tensor to NVFP4.
///
/// # Errors
/// [`Error::Refused`] on a non-finite value or a length that is not `rows * cols`.
#[allow(clippy::many_single_char_names)]
pub fn nvfp4_quantize(x: &[f32], rows: usize, cols: usize, layout: Nvfp4Layout, rounding: Rounding<'_>) -> Result<Nvfp4, Error> {
    if x.len() != rows * cols {
        return Err(Error::Refused(format!("{} values for a {rows}x{cols} tensor", x.len())));
    }
    check_finite(x)?;
    let n_blocks = if rows == 0 { 0 } else { block_of(layout, cols, rows - 1, cols.saturating_sub(1)) + 1 };
    let mut amax = vec![0f32; n_blocks];
    for r in 0..rows {
        for c in 0..cols {
            let b = block_of(layout, cols, r, c);
            amax[b] = amax[b].max(x[r * cols + c].abs());
        }
    }
    let tensor_amax = amax.iter().fold(0f32, |m, &v| m.max(v));
    let tensor_scale = if tensor_amax == 0.0 { 1.0 } else { tensor_amax / (E2M1.max * E4M3.max) };
    let block_scales: Vec<u8> = amax.iter().map(|&a| E4M3.encode(E4M3.round(a / E2M1.max / tensor_scale, Mode::Up))).collect();
    let mut q = Nvfp4 { rows, cols, layout, tensor_scale, block_scales, codes: Vec::with_capacity(x.len()) };
    for r in 0..rows {
        for c in 0..cols {
            let s = E4M3.decode(q.block_scales[q.block(r, c)]) * tensor_scale;
            let i = r * cols + c;
            let y = if s == 0.0 { 0.0 } else { x[i] / s };
            q.codes.push(E2M1.quantize(y, mode(rounding, i as u64)));
        }
    }
    Ok(q)
}

/// Values of an NVFP4 tensor, row-major.
#[must_use]
pub fn nvfp4_dequantize(q: &Nvfp4) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.codes.len());
    for r in 0..q.rows {
        for c in 0..q.cols {
            let s = E4M3.decode(q.block_scales[q.block(r, c)]) * q.tensor_scale;
            out.push(E2M1.decode(q.codes[r * q.cols + c]) * s);
        }
    }
    out
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn rht_signs(rng: &Philox, stream: u32) -> [f32; NV_BLOCK] {
    std::array::from_fn(|j| if rng.word(stream, 0, j as u64) & 1 == 1 { -1.0 } else { 1.0 })
}

/// In-place normalized 16-point Walsh-Hadamard transform.
fn hadamard16(v: &mut [f32]) {
    let mut h = 1;
    while h < NV_BLOCK {
        for i in (0..NV_BLOCK).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
    for x in v.iter_mut() {
        *x *= 0.25;
    }
}

/// Applies `H diag(s)` to every 16 consecutive values, signs `s` from Philox stream `stream`.
///
/// # Errors
/// [`Error::Refused`] when the length is not a multiple of 16.
pub fn rht16(x: &mut [f32], rng: &Philox, stream: u32) -> Result<(), Error> {
    if !x.len().is_multiple_of(NV_BLOCK) {
        return Err(Error::Refused(format!("{} values are not whole blocks of {NV_BLOCK}", x.len())));
    }
    let s = rht_signs(rng, stream);
    for block in x.chunks_mut(NV_BLOCK) {
        for (v, sj) in block.iter_mut().zip(s) {
            *v *= sj;
        }
        hadamard16(block);
    }
    Ok(())
}

/// Inverse of [`rht16`] with the same key and stream.
///
/// # Errors
/// [`Error::Refused`] when the length is not a multiple of 16.
pub fn rht16_inverse(x: &mut [f32], rng: &Philox, stream: u32) -> Result<(), Error> {
    if !x.len().is_multiple_of(NV_BLOCK) {
        return Err(Error::Refused(format!("{} values are not whole blocks of {NV_BLOCK}", x.len())));
    }
    let s = rht_signs(rng, stream);
    for block in x.chunks_mut(NV_BLOCK) {
        hadamard16(block);
        for (v, sj) in block.iter_mut().zip(s) {
            *v *= sj;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::philox::normal_f64;

    fn normals(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let rng = Philox::new(seed);
        (0..n).map(|i| normal_f64(rng.word(0, 0, 2 * i as u64), rng.word(0, 0, 2 * i as u64 + 1)) as f32 * scale).collect()
    }

    #[test]
    fn e4m3_and_e2m1_codes_round_trip() {
        for c in 0u8..=255 {
            if c & 0x7f == 0x7f || c == 0x80 {
                continue; // NaN patterns and negative zero
            }
            assert_eq!(E4M3.encode(E4M3.decode(c)), c, "e4m3 {c:#x}");
        }
        assert_eq!(E4M3.decode(0x7e), 448.0);
        assert_eq!(E4M3.decode(0x01), 2f32.powi(-9));
        let e2m1: Vec<f32> = (0u8..8).map(|c| E2M1.decode(c)).collect();
        assert_eq!(e2m1, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        for c in (1u8..16).filter(|&c| c != 0x8) {
            assert_eq!(E2M1.encode(E2M1.decode(c)), c, "e2m1 {c:#x}");
        }
        assert_eq!(E4M3.round(1000.0, Mode::Nearest), 448.0, "saturates");
        assert_eq!(E2M1.round(2.5, Mode::Nearest), 2.0, "ties to even");
        assert_eq!(E2M1.round(5.0, Mode::Nearest), 4.0, "ties to even");
        assert_eq!(E2M1.round(2.1, Mode::Up), 3.0);
    }

    #[test]
    fn mxfp8_scale_is_the_smallest_power_of_two_without_saturation() {
        let mut x = normals(1, 4 * MX_BLOCK + 5, 3.0);
        x[7] = 900.0; // an outlier in the first block
        x[MX_BLOCK..2 * MX_BLOCK].fill(0.0);
        let q = mxfp8_quantize(&x, Rounding::Nearest).unwrap();
        assert_eq!(q.scales.len(), 5);
        assert_eq!(q.scales[1], 0, "an all-zero block takes the smallest scale");
        for (b, block) in x.chunks(MX_BLOCK).enumerate() {
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let e = i32::from(q.scales[b]) - 127;
            assert!(amax / 2f32.powi(e) <= 448.0, "block {b} saturates");
            assert!(amax / 2f32.powi(e - 1) > 448.0, "block {b} scale is not the smallest");
        }
        let y = mxfp8_dequantize(&q);
        for (i, (&a, &b)) in x.iter().zip(&y).enumerate() {
            let e = i32::from(q.scales[i / MX_BLOCK]) - 127;
            // half a unit in the last place of E4M3, relative for normals, absolute at the subnormal step
            let tol = (a.abs() * 2f32.powi(-4)).max(2f32.powi(-10 + e));
            assert!((a - b).abs() <= tol, "element {i}: {a} -> {b}");
        }
        assert!(mxfp8_quantize(&[1.0, f32::NAN], Rounding::Nearest).is_err());
    }

    #[test]
    fn nvfp4_tiles_quantize_a_weight_and_its_transpose_alike() {
        let (rows, cols) = (40, 24);
        let w = normals(2, rows * cols, 0.05);
        let wt: Vec<f32> = (0..cols * rows).map(|i| w[(i % rows) * cols + i / rows]).collect();
        let a = nvfp4_dequantize(&nvfp4_quantize(&w, rows, cols, Nvfp4Layout::Tiles, Rounding::Nearest).unwrap());
        let b = nvfp4_dequantize(&nvfp4_quantize(&wt, cols, rows, Nvfp4Layout::Tiles, Rounding::Nearest).unwrap());
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(a[r * cols + c], b[c * rows + r], "({r},{c})");
            }
        }
        // the row layout does not have that property, and both stay close to the input
        let rq = nvfp4_quantize(&w, rows, cols, Nvfp4Layout::Rows, Rounding::Nearest).unwrap();
        assert_eq!(rq.block_scales.len(), rows * cols.div_ceil(NV_BLOCK));
        let r = nvfp4_dequantize(&rq);
        let err = |y: &[f32]| (w.iter().zip(y).map(|(p, q)| f64::from(p - q).powi(2)).sum::<f64>() / w.iter().map(|p| f64::from(*p).powi(2)).sum::<f64>()).sqrt();
        assert!(err(&a) < 0.15 && err(&r) < 0.15, "relative errors {} {}", err(&a), err(&r));
        // no element exceeds its block's range: every code decodes within 6 block scales
        assert!(rq.codes.iter().all(|&c| E2M1.decode(c).abs() <= 6.0));
    }

    #[test]
    fn stochastic_rounding_is_unbiased_and_replays() {
        // every block's scale is set by its first value; the others sit between two E2M1 values
        let x: Vec<f32> = (0..64 * 1024).map(|i| if i % NV_BLOCK == 0 { 1.0 } else { 0.37 }).collect();
        let rest = |y: &[f32]| -> Vec<f32> { y.iter().enumerate().filter(|(i, _)| i % NV_BLOCK != 0).map(|(_, &v)| v).collect() };
        let rng = Philox::new(9);
        let sr = Rounding::Stochastic { rng: &rng, step: 3, tensor_id: 1 };
        let q = nvfp4_quantize(&x, 64, 1024, Nvfp4Layout::Rows, sr).unwrap();
        let y = rest(&nvfp4_dequantize(&q));
        let mean = y.iter().map(|&v| f64::from(v)).sum::<f64>() / y.len() as f64;
        assert!((mean - 0.37).abs() < 3e-3, "mean {mean}");
        assert!(y.iter().any(|&v| v < 0.37) && y.iter().any(|&v| v > 0.37));
        assert_eq!(q, nvfp4_quantize(&x, 64, 1024, Nvfp4Layout::Rows, sr).unwrap(), "same key, same codes");
        let other = Rounding::Stochastic { rng: &rng, step: 4, tensor_id: 1 };
        assert_ne!(q.codes, nvfp4_quantize(&x, 64, 1024, Nvfp4Layout::Rows, other).unwrap().codes);
        let n = rest(&nvfp4_dequantize(&nvfp4_quantize(&x, 64, 1024, Nvfp4Layout::Rows, Rounding::Nearest).unwrap()));
        assert!(n.iter().all(|&v| v == n[0]), "nearest rounding is deterministic per value");
    }

    #[test]
    fn hadamard_transform_is_orthonormal_and_spreads_outliers() {
        let rng = Philox::new(5);
        let x = {
            let mut x = normals(3, 64, 1.0);
            x[3] = 50.0;
            x
        };
        let mut y = x.clone();
        rht16(&mut y, &rng, 2).unwrap();
        let norm = |v: &[f32]| v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>();
        assert!((norm(&x) - norm(&y)).abs() < 1e-3 * norm(&x));
        let amax = |v: &[f32]| v[..16].iter().fold(0f32, |m, a| m.max(a.abs()));
        assert!(amax(&y) < 0.5 * amax(&x), "outlier spread: {} -> {}", amax(&x), amax(&y));
        rht16_inverse(&mut y, &rng, 2).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5 * a.abs().max(1.0));
        }
        assert!(rht16(&mut [0.0; 15], &rng, 0).is_err());
    }
}
