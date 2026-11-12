//! Expert-FFN compute backend seam.
//!
//! The MoE forward reduces to one primitive: `Y = X · Wᵀ` for a hidden-state
//! batch `X: [n_tokens, in]` against a projection weight `W: [out, in]`
//! (dense `f32` or GGUF block-quantized). Every expert's SwiGLU forward is
//! three such matmuls, so this is the single hot path.
//!
//! [`ExpertCompute`] abstracts that primitive so the runtime can select the
//! fastest available implementation at load time:
//!
//! - [`CpuCompute`] is always present. It is `ndarray` for the dense path,
//!   and for the quantized path it either dequantizes-then-GEMMs or, when the
//!   activation batch is itself Q8_0 and the CPU advertises AVX-512-VNNI at
//!   runtime, runs an integer dot straight over the block payloads. The
//!   feature probe is `is_x86_feature_detected!` — the scalar path is always
//!   reachable, so a build with no SIMD support still executes correctly.
//! - GPU backends are compiled under the `cuda` feature (cuBLAS) and the
//!   `wgpu` feature (WGSL compute on any other GPU).
//!
//! [`ComputeBackend::select`] is the runtime dispatch: the GPUs the host
//! declares decide the backend, and a GPU that no compiled backend can drive
//! is an error rather than a quiet move to the CPU.

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::quant::{QuantKind, QuantMatrix};

/// A projection weight in the form the compute backend consumes it.
///
/// Kept distinct from the private `Projection` enum in [`crate::exec`] so
/// the backend trait has no dependency on the loader's internals — a backend
/// sees only "a dense `f32` matrix" or "a block-quantized matrix", both
/// row-major `[out, in]`.
pub enum Weight<'a> {
    /// Dense row-major `[out, in]` `f32`.
    Dense(ArrayView2<'a, f32>),
    /// GGUF block-quantized `[out, in]`.
    Quant(&'a QuantMatrix),
}

impl Weight<'_> {
    /// `(out, in)` shape.
    pub fn dim(&self) -> (usize, usize) {
        match self {
            Weight::Dense(w) => (w.nrows(), w.ncols()),
            Weight::Quant(q) => (q.nrows(), q.ncols()),
        }
    }
}

/// The one operation every expert FFN forward routes through.
///
/// Implementations must be numerically equivalent to the scalar reference
/// (`Y[i,j] = Σ_k X[i,k] · W[j,k]`) up to floating-point reassociation — a
/// GPU or SIMD path may reorder the reduction but must not change results
/// beyond that. Object-safe so the runtime can hold `Box<dyn ExpertCompute>`.
pub trait ExpertCompute: Send + Sync {
    /// `Y = X · Wᵀ`, `X: [n, in]`, `W: [out, in]` → `Y: [n, out]`.
    fn matmul_xt(&self, x: ArrayView2<'_, f32>, w: &Weight<'_>) -> Array2<f32>;

    /// A short backend tag for status/telemetry (`"cpu-avx512"`, `"cpu"`,
    /// `"cuda"`, `"wgpu"`).
    fn tag(&self) -> &'static str;
}

/// CPU backend — always available.
///
/// Records the SIMD tier detected at construction so per-call dispatch is a
/// field read, not a repeated `is_x86_feature_detected!` (which is cheap but
/// not free). The tier only ever enables a *faster equivalent* path; the
/// scalar/`ndarray` reference is always the floor.
pub struct CpuCompute {
    tier: CpuSimd,
}

/// Detected CPU SIMD tier for the Q8_0 integer-dot fast path.
///
/// Each accelerated variant is gated to the architecture that can construct it,
/// so there is no unreachable variant on any target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CpuSimd {
    /// No SIMD acceleration used — dequantize-then-GEMM / scalar dot.
    Scalar,
    /// AVX-512 Foundation + VNNI (`vpdpbusd`) available.
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    /// AVX-VNNI: the same `vpdpbusd` in its 256-bit VEX encoding, on Intel
    /// client cores that have no AVX-512.
    #[cfg(target_arch = "x86_64")]
    AvxVnni,
    /// NEON advanced SIMD. Architecturally guaranteed on every ARMv8-A part,
    /// so on aarch64 this tier is always selected rather than probed.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// NEON with the Armv8.2 dot-product extension (`SDOT`): every Armv8.4
    /// and Armv9 core, Apple silicon, Neoverse N1 and later.
    #[cfg(target_arch = "aarch64")]
    NeonDotprod,
}

impl CpuCompute {
    /// Probe the running CPU and pick the best Q8_0 dot tier.
    pub fn detect() -> Self {
        Self {
            tier: detect_cpu_simd(),
        }
    }

    /// The dense path is a single `ndarray` GEMM.
    fn matmul_dense(x: ArrayView2<'_, f32>, w: ArrayView2<'_, f32>) -> Array2<f32> {
        x.dot(&w.t())
    }

    /// The quantized path. When the weight is Q8_0 and AVX-512-VNNI is
    /// present, each output column is an integer dot over the block payloads
    /// (activations quantized to Q8_0 per row first). Otherwise dequantize
    /// each weight row into scratch and GEMM against it — the reference path.
    fn matmul_quant(&self, x: ArrayView2<'_, f32>, q: &QuantMatrix) -> Array2<f32> {
        let n = x.nrows();
        let out = q.nrows();
        let cols = q.ncols();
        let mut y = Array2::<f32>::zeros((n, out));

        #[cfg(target_arch = "x86_64")]
        if matches!(self.tier, CpuSimd::Avx512Vnni | CpuSimd::AvxVnni)
            && q.kind() == QuantKind::Q8_0
            && cols.is_multiple_of(crate::quant::QK8_0)
        {
            // Quantize each activation row to Q8_0 once, then integer-dot it
            // against every weight row's blocks. This is the grouped-GEMM
            // inner shape: the whole token batch shares the dequant of W.
            let mut act_blocks: Vec<Vec<u8>> = Vec::with_capacity(n);
            for i in 0..n {
                let row: Vec<f32> = x.row(i).to_vec();
                act_blocks.push(crate::quant::quantize_row_q8_0(&row));
            }
            for j in 0..out {
                let wblocks = q.row_block_bytes(j);
                for (i, ablocks) in act_blocks.iter().enumerate() {
                    // Safety: the tier proves the target features of the
                    // matching function are present on this CPU (checked once
                    // in detect()).
                    y[[i, j]] = if self.tier == CpuSimd::Avx512Vnni {
                        unsafe { dot_q8_0_avx512_vnni(ablocks, wblocks) }
                    } else {
                        unsafe { dot_q8_0_avx_vnni(ablocks, wblocks) }
                    };
                }
            }
            return y;
        }

        #[cfg(target_arch = "aarch64")]
        if matches!(self.tier, CpuSimd::Neon | CpuSimd::NeonDotprod)
            && q.kind() == QuantKind::Q8_0
            && cols.is_multiple_of(crate::quant::QK8_0)
        {
            // Same shape as the AVX-512 path above: quantize each activation
            // row to Q8_0 once, then integer-dot it against every weight row.
            let mut act_blocks: Vec<Vec<u8>> = Vec::with_capacity(n);
            for i in 0..n {
                let row: Vec<f32> = x.row(i).to_vec();
                act_blocks.push(crate::quant::quantize_row_q8_0(&row));
            }
            for j in 0..out {
                let wblocks = q.row_block_bytes(j);
                for (i, ablocks) in act_blocks.iter().enumerate() {
                    y[[i, j]] = if self.tier == CpuSimd::NeonDotprod {
                        // Safety: the tier proves the dot-product extension
                        // is present on this CPU (checked once in detect()).
                        unsafe { dot_q8_0_neon_dotprod(ablocks, wblocks) }
                    } else {
                        dot_q8_0_neon(ablocks, wblocks)
                    };
                }
            }
            return y;
        }

        // Scalar / ndarray reference path (always reachable).
        let mut wrow = vec![0.0f32; cols];
        for j in 0..out {
            q.dequantize_row(j, &mut wrow);
            let wview = ArrayView1::from(&wrow);
            for i in 0..n {
                y[[i, j]] = x.row(i).dot(&wview);
            }
        }
        y
    }
}

impl ExpertCompute for CpuCompute {
    fn matmul_xt(&self, x: ArrayView2<'_, f32>, w: &Weight<'_>) -> Array2<f32> {
        match w {
            Weight::Dense(dw) => Self::matmul_dense(x, *dw),
            Weight::Quant(q) => self.matmul_quant(x, q),
        }
    }

    fn tag(&self) -> &'static str {
        match self.tier {
            CpuSimd::Scalar => "cpu",
            #[cfg(target_arch = "x86_64")]
            CpuSimd::Avx512Vnni => "cpu-avx512-vnni",
            #[cfg(target_arch = "x86_64")]
            CpuSimd::AvxVnni => "cpu-avx-vnni",
            #[cfg(target_arch = "aarch64")]
            CpuSimd::Neon => "cpu-neon",
            #[cfg(target_arch = "aarch64")]
            CpuSimd::NeonDotprod => "cpu-neon-dotprod",
        }
    }
}

/// Probe x86 SIMD once. On non-x86 targets, always `Scalar`.
fn detect_cpu_simd() -> CpuSimd {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            return CpuSimd::Avx512Vnni;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avxvnni") {
            return CpuSimd::AvxVnni;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory in ARMv8-A, so every aarch64 target reaching this
        // code has it; the dot-product extension is probed.
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return CpuSimd::NeonDotprod;
        }
        return CpuSimd::Neon;
    }
    #[allow(unreachable_code)]
    CpuSimd::Scalar
}

/// Dot product of two Q8_0-block rows with the `vpdpbusd` instruction, in the
/// encoding the CPU has: EVEX on AVX-512-VNNI parts (Xeon, Zen 4 and later),
/// VEX on AVX-VNNI parts (Intel client cores without AVX-512).
///
/// Both `a` and `b` are concatenated Q8_0 blocks (`[f16 scale | 32×i8]`,
/// 34 bytes each) of equal length. Per block: widen both quant vectors,
/// accumulate `Σ aᵢ·bᵢ` as i32, then scale by `d_a · d_b`. The two encodings
/// compute the same integers, so the tiers agree bit for bit.
#[cfg(target_arch = "x86_64")]
macro_rules! dot_q8_0_vnni {
    ($name:ident, $features:literal, $dpbusd:ident) => {
        /// # Safety
        /// Caller must ensure the CPU supports the target features this
        /// function is compiled for (guaranteed by the detected tier), and
        /// that `a` and `b` are the same number of whole Q8_0 blocks.
        #[target_feature(enable = $features)]
        unsafe fn $name(a: &[u8], b: &[u8]) -> f32 {
            use half::f16;
            use std::arch::x86_64::*;

            const BB: usize = crate::quant::Q8_0_BLOCK_BYTES; // 34
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len() % BB, 0);

            let mut acc = 0.0f32;
            let nblocks = a.len() / BB;
            for blk in 0..nblocks {
                let ab = &a[blk * BB..(blk + 1) * BB];
                let bb = &b[blk * BB..(blk + 1) * BB];
                let da = f16::from_le_bytes([ab[0], ab[1]]).to_f32();
                let db = f16::from_le_bytes([bb[0], bb[1]]).to_f32();

                let raw = unsafe {
                    // 32 i8 quants per block. Load as __m256i (32 bytes).
                    let qa = _mm256_loadu_si256(ab.as_ptr().add(2) as *const __m256i);
                    let qb = _mm256_loadu_si256(bb.as_ptr().add(2) as *const __m256i);

                    // vpdpbusd wants u8 × i8. Q8_0 quants are signed i8; make the `a`
                    // side unsigned by adding 128 and correct the bias afterward:
                    //   Σ (aᵢ+128)·bᵢ = Σ aᵢ·bᵢ + 128·Σ bᵢ
                    // so subtract 128·Σ bᵢ.
                    let bias = _mm256_set1_epi8(-128i8 as u8 as i8);
                    let qa_u = _mm256_sub_epi8(qa, bias); // aᵢ + 128, as u8 bit-pattern

                    let mut dot = _mm256_setzero_si256();
                    dot = $dpbusd(dot, qa_u, qb);

                    // Σ bᵢ via vpdpbusd of ones(u8) · b(i8).
                    let ones = _mm256_set1_epi8(1);
                    let mut sum_b = _mm256_setzero_si256();
                    sum_b = $dpbusd(sum_b, ones, qb);

                    // Horizontal-sum both i32x8 accumulators.
                    let dot_sum = hsum_i32x8(dot);
                    let b_sum = hsum_i32x8(sum_b);

                    dot_sum - 128 * b_sum
                };
                acc += (raw as f32) * da * db;
            }
            acc
        }
    };
}

#[cfg(target_arch = "x86_64")]
dot_q8_0_vnni!(dot_q8_0_avx512_vnni, "avx512f,avx512bw,avx512vl,avx512vnni", _mm256_dpbusd_epi32);
#[cfg(target_arch = "x86_64")]
dot_q8_0_vnni!(dot_q8_0_avx_vnni, "avx2,avxvnni", _mm256_dpbusd_avx_epi32);

/// Horizontal sum of an `__m256i` interpreted as 8×i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32x8(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
    _mm_cvtsi128_si32(s)
}

/// Dot product of two Q8_0-block rows with the Armv8.2 dot-product
/// extension (`SDOT`).
///
/// Both `a` and `b` are concatenated Q8_0 blocks (`[f16 scale | 32×i8]`,
/// 34 bytes each) of equal length. Per block: two `vdotq_s32` accumulations
/// cover the 32 quants as 2×16 lanes, horizontally sum to i32, then scale by
/// `d_a · d_b`. `SDOT` is signed × signed, so the Q8_0 i8 quants feed it
/// directly and no bias correction is needed. The integers are the ones the
/// widening-multiply tier computes, so the two agree bit for bit.
///
/// # Safety
/// Caller must ensure the CPU has the dot-product extension (guaranteed by
/// `CpuSimd::NeonDotprod`) and that `a` and `b` are the same number of whole
/// Q8_0 blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q8_0_neon_dotprod(a: &[u8], b: &[u8]) -> f32 {
    use half::f16;
    use std::arch::aarch64::*;

    const BB: usize = crate::quant::Q8_0_BLOCK_BYTES; // 34
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % BB, 0);

    let mut acc = 0.0f32;
    let nblocks = a.len() / BB;
    for blk in 0..nblocks {
        let ab = &a[blk * BB..(blk + 1) * BB];
        let bb = &b[blk * BB..(blk + 1) * BB];
        let da = f16::from_le_bytes([ab[0], ab[1]]).to_f32();
        let db = f16::from_le_bytes([bb[0], bb[1]]).to_f32();

        // Safety: both slices are indexed within the 34-byte block bounds
        // asserted above.
        let raw = unsafe {
            let qa0 = vld1q_s8(ab.as_ptr().add(2) as *const i8);
            let qa1 = vld1q_s8(ab.as_ptr().add(2 + 16) as *const i8);
            let qb0 = vld1q_s8(bb.as_ptr().add(2) as *const i8);
            let qb1 = vld1q_s8(bb.as_ptr().add(2 + 16) as *const i8);
            let sum = vdotq_s32(vdotq_s32(vdupq_n_s32(0), qa0, qb0), qa1, qb1);
            vaddvq_s32(sum)
        };
        acc += (raw as f32) * da * db;
    }
    acc
}

/// Dot product of two Q8_0-block rows with baseline NEON, for cores without
/// the dot-product extension.
///
/// Same blocks and result as [`dot_q8_0_neon_dotprod`], computed with
/// `vmull_s8` + `vpaddlq_s16`: the widening-multiply form the vendored
/// llama.cpp falls back to when `__ARM_FEATURE_DOTPROD` is absent
/// (`ggml/src/ggml-cpu/ggml-cpu-impl.h`, `ggml_vdotq_s32`). It needs no
/// runtime probe: NEON is mandatory in ARMv8-A. Per-lane grouping differs
/// from `SDOT`, but the horizontal sum is identical, which is all this
/// function returns.
///
/// # Safety
/// Caller must ensure `a` and `b` are the same number of whole Q8_0 blocks.
#[cfg(target_arch = "aarch64")]
fn dot_q8_0_neon(a: &[u8], b: &[u8]) -> f32 {
    use half::f16;
    use std::arch::aarch64::*;

    const BB: usize = crate::quant::Q8_0_BLOCK_BYTES; // 34
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % BB, 0);

    let mut acc = 0.0f32;
    let nblocks = a.len() / BB;
    for blk in 0..nblocks {
        let ab = &a[blk * BB..(blk + 1) * BB];
        let bb = &b[blk * BB..(blk + 1) * BB];
        let da = f16::from_le_bytes([ab[0], ab[1]]).to_f32();
        let db = f16::from_le_bytes([bb[0], bb[1]]).to_f32();

        // Safety: NEON is architecturally guaranteed on aarch64, and both
        // slices are indexed within the 34-byte block bounds asserted above.
        let raw = unsafe {
            // 32 i8 quants per block, loaded as 2×int8x16_t.
            let qa0 = vld1q_s8(ab.as_ptr().add(2) as *const i8);
            let qa1 = vld1q_s8(ab.as_ptr().add(2 + 16) as *const i8);
            let qb0 = vld1q_s8(bb.as_ptr().add(2) as *const i8);
            let qb1 = vld1q_s8(bb.as_ptr().add(2 + 16) as *const i8);

            // Widening signed multiply into i16x8, then pairwise-accumulate
            // into i32x4. Four halves cover all 32 lanes.
            let p0 = vmull_s8(vget_low_s8(qa0), vget_low_s8(qb0));
            let p1 = vmull_s8(vget_high_s8(qa0), vget_high_s8(qb0));
            let p2 = vmull_s8(vget_low_s8(qa1), vget_low_s8(qb1));
            let p3 = vmull_s8(vget_high_s8(qa1), vget_high_s8(qb1));

            let s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
            let s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
            vaddvq_s32(vaddq_s32(s01, s23))
        };
        acc += (raw as f32) * da * db;
    }
    acc
}

/// Which compute backend the runtime resolved for this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU (scalar / ndarray floor, possibly with AVX-512-VNNI Q8_0 dot).
    Cpu,
    /// CUDA (cuBLAS grouped GEMM) — NVIDIA present, `cuda` compiled.
    Cuda,
    /// wgpu WGSL compute — any other GPU, `wgpu` compiled.
    Wgpu,
}

/// Runtime backend selection.
///
/// The machine's GPU decides the backend: CUDA on an NVIDIA GPU (wgpu when
/// the build has only the wgpu kernels, as the Vulkan build does), wgpu on any
/// other GPU, and the CPU only on a machine with no GPU. A GPU whose backend
/// is not compiled in, or whose device does not initialise, is an error rather
/// than a quiet move to the CPU.
pub struct ComputeBackend;

impl ComputeBackend {
    /// Resolve the backend for a machine with the `gpus` the host detected,
    /// or why there is none.
    pub fn select(
        gpus: Gpus,
    ) -> std::result::Result<(BackendKind, Box<dyn ExpertCompute>), String> {
        if gpus.nvidia && (cfg!(feature = "cuda") || !cfg!(feature = "wgpu")) {
            return cuda_backend().map(|be| (BackendKind::Cuda, be));
        }
        if gpus.any() {
            return wgpu_backend().map(|be| (BackendKind::Wgpu, be));
        }
        Ok((BackendKind::Cpu, Box::new(CpuCompute::detect())))
    }
}

/// The GPUs visible to this process, as the host detected them.
///
/// Build it from the machine: `Gpus::from(&Machine::detect())`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gpus {
    /// An NVIDIA GPU is visible.
    pub nvidia: bool,
    /// A GPU from any other vendor is visible.
    pub other: bool,
}

impl From<&praecise_hardware::Machine> for Gpus {
    /// The GPUs of `machine`. Neural processors and data-centre accelerator
    /// cards are not counted: the expert kernels run on CUDA or through wgpu,
    /// and wgpu drives GPUs only.
    fn from(machine: &praecise_hardware::Machine) -> Self {
        use praecise_hardware::{Kind, Vendor};
        let gpus = || machine.accelerators().iter().filter(|a| a.kind == Kind::Gpu);
        Self {
            nvidia: gpus().any(|a| a.vendor == Vendor::Nvidia),
            other: gpus().any(|a| a.vendor != Vendor::Nvidia),
        }
    }
}

impl Gpus {
    /// Whether any GPU is visible.
    pub fn any(self) -> bool {
        self.nvidia || self.other
    }
}

#[cfg(feature = "cuda")]
fn cuda_backend() -> std::result::Result<Box<dyn ExpertCompute>, String> {
    crate::compute_cuda::CudaCompute::try_new()
        .map(|c| Box::new(c) as Box<dyn ExpertCompute>)
        .ok_or_else(|| {
            "this machine has an NVIDIA GPU but the CUDA device or cuBLAS failed to initialise, \
             so MoE experts cannot run on it"
                .to_string()
        })
}

#[cfg(not(feature = "cuda"))]
fn cuda_backend() -> std::result::Result<Box<dyn ExpertCompute>, String> {
    Err(
        "this machine has an NVIDIA GPU but this binary was built without the `cuda` feature, \
         so MoE experts would run on the CPU; use a build with the `cuda` or `wgpu` feature"
            .to_string(),
    )
}

#[cfg(feature = "wgpu")]
fn wgpu_backend() -> std::result::Result<Box<dyn ExpertCompute>, String> {
    crate::compute_wgpu::WgpuCompute::try_new()
        .map(|c| Box::new(c) as Box<dyn ExpertCompute>)
        .ok_or_else(|| {
            "this machine has a GPU but no wgpu adapter could be acquired for it, so MoE experts \
             cannot run on it"
                .to_string()
        })
}

#[cfg(not(feature = "wgpu"))]
fn wgpu_backend() -> std::result::Result<Box<dyn ExpertCompute>, String> {
    Err(
        "this machine has a GPU but this binary was built without the `wgpu` feature, so \
         MoE experts would run on the CPU; use a build with the `wgpu` feature"
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn ref_matmul_xt(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
        x.dot(&w.t())
    }

    /// The dot-product tier and the baseline tier compute the same integers.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_tiers_agree_bit_for_bit() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let row = |seed: f32| -> Vec<u8> {
            let v: Vec<f32> = (0..96).map(|i| ((i as f32) * seed).sin() * 1.7).collect();
            crate::quant::quantize_row_q8_0(&v)
        };
        let (a, b) = (row(0.37), row(1.13));
        // Safety: the dot-product extension was detected above.
        let fast = unsafe { dot_q8_0_neon_dotprod(&a, &b) };
        assert_eq!(fast.to_bits(), dot_q8_0_neon(&a, &b).to_bits());
    }

    /// The EVEX and VEX encodings of `vpdpbusd` compute the same integers.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vnni_tiers_agree_bit_for_bit() {
        let avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni");
        let avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avxvnni");
        if !(avx512 && avx) {
            return;
        }
        let row = |seed: f32| -> Vec<u8> {
            let v: Vec<f32> = (0..96).map(|i| ((i as f32) * seed).sin() * 1.7).collect();
            crate::quant::quantize_row_q8_0(&v)
        };
        let (a, b) = (row(0.37), row(1.13));
        // Safety: both feature sets were detected above.
        let (evex, vex) = unsafe { (dot_q8_0_avx512_vnni(&a, &b), dot_q8_0_avx_vnni(&a, &b)) };
        assert_eq!(evex.to_bits(), vex.to_bits());
    }

    #[test]
    fn the_machine_names_its_gpus() {
        use praecise_hardware::{Accelerator, Kind, Machine, Vendor};
        let acc = |vendor, kind| Accelerator { vendor, kind };
        let mixed = Machine::with([acc(Vendor::Nvidia, Kind::Gpu), acc(Vendor::Amd, Kind::Gpu)]);
        assert_eq!(Gpus::from(&mixed), Gpus { nvidia: true, other: true });
        let intel = Machine::with([acc(Vendor::Intel, Kind::Gpu), acc(Vendor::Intel, Kind::Npu)]);
        assert_eq!(Gpus::from(&intel), Gpus { nvidia: false, other: true });
        // a neural processor or an accelerator card alone is nothing wgpu can drive
        let npu = Machine::with([acc(Vendor::Huawei, Kind::Npu), acc(Vendor::Intel, Kind::Accelerator)]);
        assert_eq!(Gpus::from(&npu), Gpus::default());
        assert!(!Gpus::from(&Machine::cpu_only()).any());
    }

    #[test]
    fn cpu_dense_matches_reference() {
        let cpu = CpuCompute::detect();
        let x = Array2::from_shape_fn((3, 8), |(i, j)| (i * 8 + j) as f32 * 0.01 - 0.1);
        let w = Array2::from_shape_fn((5, 8), |(i, j)| (i + j) as f32 * 0.03 - 0.2);
        let got = cpu.matmul_xt(x.view(), &Weight::Dense(w.view()));
        let want = ref_matmul_xt(&x, &w);
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn cpu_quant_q8_0_matches_dequantized_reference() {
        // Build a dense weight, quantize it Q8_0, and confirm the backend's
        // quant path (SIMD or scalar, whichever the host has) matches a
        // straight dequantize-then-GEMM within Q8_0 tolerance.
        let cpu = CpuCompute::detect();
        let out = 6usize;
        let cols = 64usize; // multiple of 32
        let n = 4usize;
        let wdense =
            Array2::from_shape_fn((out, cols), |(i, j)| ((i * cols + j) as f32).sin() * 0.5);
        let q =
            QuantMatrix::quantize(QuantKind::Q8_0, out, cols, wdense.as_slice().unwrap()).unwrap();
        let x = Array2::from_shape_fn((n, cols), |(i, j)| {
            ((i + 1) as f32 * (j as f32)).cos() * 0.3
        });

        let got = cpu.matmul_xt(x.view(), &Weight::Quant(&q));

        // Reference: dequantize the same matrix and GEMM.
        let deq = q.dequantize();
        let wq = Array2::from_shape_vec((out, cols), deq).unwrap();
        let want = ref_matmul_xt(&x, &wq);

        for (a, b) in got.iter().zip(want.iter()) {
            // Q8_0 activation quantization adds error on top of weight
            // quantization; bound generously.
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn select_uses_the_cpu_only_without_a_gpu() {
        let (kind, be) = ComputeBackend::select(Gpus::default()).unwrap();
        assert_eq!(kind, BackendKind::Cpu);
        assert!(be.tag().starts_with("cpu"));
    }

    /// A GPU machine never gets the CPU backend: without the GPU backend
    /// compiled in, selection fails with the reason.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn an_nvidia_gpu_without_the_cuda_backend_is_refused() {
        let err = ComputeBackend::select(Gpus {
            nvidia: true,
            other: false,
        })
        .err()
        .expect("an NVIDIA machine must not select the CPU");
        assert!(
            err.contains("NVIDIA") && err.contains("`cuda` feature"),
            "{err}"
        );
    }

    #[cfg(not(feature = "wgpu"))]
    #[test]
    fn a_non_nvidia_gpu_without_the_wgpu_backend_is_refused() {
        let err = ComputeBackend::select(Gpus {
            nvidia: false,
            other: true,
        })
        .err()
        .expect("a GPU machine must not select the CPU");
        assert!(err.contains("`wgpu` feature"), "{err}");
    }
}
