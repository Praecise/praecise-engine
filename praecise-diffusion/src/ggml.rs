//! The ggml backend, weight residency and graph execution.
//!
//! This is the same ggml the language-model family is built on, so a
//! diffusion model and a language model share one device, one allocator and
//! one CUDA build.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use llama_cpp_sys_2 as sys;

use crate::error::{Error, Result};
use crate::safetensors::{Dtype, SafeTensors, TensorView};

/// A ggml compute backend: a GPU when the host has one, the CPU only on a host
/// without GPU hardware.
pub struct Backend {
    raw: sys::ggml_backend_t,
    name: String,
    gpu: bool,
}

// SAFETY: a ggml backend handle is used from one thread at a time; the
// pipeline that owns it serialises every call behind `&mut self`.
unsafe impl Send for Backend {}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend").field("name", &self.name).field("gpu", &self.gpu).finish()
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: `raw` came from ggml_backend_dev_init / cpu_init and is
        // freed exactly once.
        unsafe { sys::ggml_backend_free(self.raw) };
    }
}

/// GPU hardware present on this host, independent of what the build can
/// drive. Used to refuse a CPU run on a machine that has a GPU.
fn gpu_hardware_present() -> Option<&'static str> {
    if Path::new("/proc/driver/nvidia/version").exists() || Path::new("/dev/nvidiactl").exists() {
        return Some("NVIDIA");
    }
    if Path::new("/dev/kfd").exists() {
        return Some("AMD");
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Some("Apple");
    }
    None
}

impl Backend {
    /// Select the backend: the first GPU device ggml registered. With no GPU
    /// device registered, the CPU is used only when the host has no GPU
    /// hardware at all; otherwise loading is refused.
    ///
    /// # Errors
    /// [`Error::GpuRequired`] when GPU hardware is present but no GPU backend
    /// is built in, or the GPU backend fails to initialise.
    pub fn select(cpu_threads: usize) -> Result<Self> {
        // SAFETY: device enumeration has no preconditions; the registry is
        // populated statically at link time.
        let count = unsafe { sys::ggml_backend_dev_count() };
        for i in 0..count {
            let dev = unsafe { sys::ggml_backend_dev_get(i) };
            let ty = unsafe { sys::ggml_backend_dev_type(dev) };
            if ty == sys::GGML_BACKEND_DEVICE_TYPE_GPU || ty == sys::GGML_BACKEND_DEVICE_TYPE_IGPU {
                let name = unsafe { CStr::from_ptr(sys::ggml_backend_dev_name(dev)) }
                    .to_string_lossy()
                    .into_owned();
                let raw = unsafe { sys::ggml_backend_dev_init(dev, ptr::null()) };
                if raw.is_null() {
                    return Err(Error::GpuRequired(format!("GPU device {name} failed to initialise")));
                }
                return Ok(Self { raw, name, gpu: true });
            }
        }
        if let Some(vendor) = gpu_hardware_present() {
            let gpu_backend_built = (0..unsafe { sys::ggml_backend_reg_count() }).any(|i| {
                let name = unsafe { CStr::from_ptr(sys::ggml_backend_reg_name(sys::ggml_backend_reg_get(i))) };
                name.to_bytes() != b"CPU" && name.to_bytes() != b"BLAS" && name.to_bytes() != b"RPC"
            });
            return Err(Error::GpuRequired(if gpu_backend_built {
                format!("{vendor} GPU hardware is present and a GPU backend is built, but it found no usable device (driver or device failure)")
            } else {
                format!("{vendor} GPU hardware is present but this build has no backend for it")
            }));
        }
        let raw = unsafe { sys::ggml_backend_cpu_init() };
        if raw.is_null() {
            return Err(Error::Backend("CPU backend failed to initialise".into()));
        }
        unsafe { sys::ggml_backend_cpu_set_n_threads(raw, cpu_threads.max(1) as i32) };
        Ok(Self { raw, name: "CPU".into(), gpu: false })
    }

    /// Backend device name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this backend runs on a GPU.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.gpu
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    /// Time one attention at a diffusion transformer's shape: 24 heads of
    /// 128 over 4608 tokens. Run explicitly on the machine being measured.
    #[test]
    #[ignore = "a timing, run on the hardware being measured"]
    fn bench_attention() {
        let backend = Backend::select(8).unwrap();
        let (d, h, n) = (128i64, 24i64, 4608i64);
        let mut g = Graph::new(&backend).unwrap();
        let q = g.input(sys::GGML_TYPE_F32, &[d, n, h]);
        let k = g.input(sys::GGML_TYPE_F16, &[d, n, h]);
        let v = g.input(sys::GGML_TYPE_F16, &[d, n, h]);
        let o = g.attention(q, k, v, None, 1.0 / (d as f32).sqrt(), false);
        g.finish(&[o]).unwrap();
        let data: Vec<f32> = (0..d * n * h).map(|i| ((i % 97) as f32 - 48.0) / 97.0).collect();
        g.set_f32(q, &data);
        g.set_f16(k, &data);
        g.set_f16(v, &data);
        g.compute().unwrap();
        let runs = 20;
        let t = std::time::Instant::now();
        for _ in 0..runs {
            g.compute().unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(runs);
        let tflops = 4.0 * (n * n * d * h) as f64 / (ms / 1000.0) / 1e12;
        eprintln!("attention {d}x{h} heads over {n} tokens on {}: {ms:.2} ms ({tflops:.1} TFLOPS)", backend.name());
    }

    /// Flash attention against a direct computation, at key lengths that are
    /// not a multiple of any tile and query counts wide enough for the
    /// pipelined kernels, with and without a mask.
    #[test]
    fn attention_matches_a_direct_computation_off_the_tile_boundary() {
        let backend = Backend::select(8).unwrap();
        let d = 128usize;
        let h = 2usize;
        for (nq, nkv, masked) in [(300usize, 333usize, false), (300, 333, true), (1000, 1037, false), (64, 65, true)] {
            let mut g = Graph::new(&backend).unwrap();
            let q = g.input(sys::GGML_TYPE_F32, &[d as i64, nq as i64, h as i64]);
            let k = g.input(sys::GGML_TYPE_F16, &[d as i64, nkv as i64, h as i64]);
            let v = g.input(sys::GGML_TYPE_F16, &[d as i64, nkv as i64, h as i64]);
            let m = masked.then(|| g.input(sys::GGML_TYPE_F16, &[nkv as i64, nq as i64]));
            let scale = 1.0 / (d as f32).sqrt();
            let o = g.attention(q, k, v, m, scale, true);
            g.finish(&[o]).unwrap();

            let value = |i: usize, salt: usize| (((i * 2_654_435_761 + salt) % 1009) as f32 / 1009.0 - 0.5) * 2.0;
            let qd: Vec<f32> = (0..d * nq * h).map(|i| value(i, 1)).collect();
            // Round K and V through f16 so the reference sees the same values.
            let kd: Vec<f32> = (0..d * nkv * h).map(|i| half::f16::from_f32(value(i, 2)).to_f32()).collect();
            let vd: Vec<f32> = (0..d * nkv * h).map(|i| half::f16::from_f32(value(i, 3)).to_f32()).collect();
            // A band of visible keys per query, different for every row, so
            // masked keys fall inside, before and after the tiles.
            let visible = |iq: usize, ik: usize| !masked || (ik + 7 * iq) % 5 != 0 || ik == iq % nkv;
            let md: Vec<f32> =
                (0..nq * nkv).map(|i| if visible(i / nkv, i % nkv) { 0.0 } else { f32::NEG_INFINITY }).collect();
            g.set_f32(q, &qd);
            g.set_f16(k, &kd);
            g.set_f16(v, &vd);
            if let Some(m) = m {
                g.set_f16(m, &md);
            }
            g.compute().unwrap();
            let got = g.read_f32(o); // [d, h, nq]

            let mut worst = 0f64;
            for ih in 0..h {
                for iq in 0..nq {
                    let qr = &qd[(ih * nq + iq) * d..][..d];
                    let s: Vec<f64> = (0..nkv)
                        .map(|ik| {
                            if !visible(iq, ik) {
                                return f64::NEG_INFINITY;
                            }
                            let kr = &kd[(ih * nkv + ik) * d..][..d];
                            qr.iter().zip(kr).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() * f64::from(scale)
                        })
                        .collect();
                    let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let p: Vec<f64> = s.iter().map(|x| (x - max).exp()).collect();
                    let sum: f64 = p.iter().sum();
                    for c in 0..d {
                        let want: f64 =
                            p.iter().enumerate().map(|(ik, w)| w * f64::from(vd[(ih * nkv + ik) * d + c])).sum::<f64>() / sum;
                        let have = f64::from(got[(iq * h + ih) * d + c]);
                        worst = worst.max((have - want).abs());
                    }
                }
            }
            assert!(worst < 5e-3, "{nq} queries over {nkv} keys (masked {masked}): max error {worst}");
        }
    }

    fn gpu_backend_built_in() -> bool {
        let count = unsafe { sys::ggml_backend_dev_count() };
        (0..count).any(|i| {
            let ty = unsafe { sys::ggml_backend_dev_type(sys::ggml_backend_dev_get(i)) };
            ty == sys::GGML_BACKEND_DEVICE_TYPE_GPU || ty == sys::GGML_BACKEND_DEVICE_TYPE_IGPU
        })
    }

    #[test]
    fn the_cpu_is_chosen_only_on_a_host_without_gpu_hardware() {
        let selected = Backend::select(2);
        match (gpu_hardware_present(), gpu_backend_built_in()) {
            (_, true) => assert!(selected.unwrap().is_gpu()),
            (Some(_), false) => assert!(matches!(selected, Err(Error::GpuRequired(_)))),
            (None, false) => assert!(!selected.unwrap().is_gpu()),
        }
    }
}

/// Storage type for a weight on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WType {
    /// float32.
    F32,
    /// float16 (convolution kernels need it).
    F16,
    /// bfloat16, the checkpoints' native type.
    Bf16,
    /// 8-bit blocks of 32 with an f16 scale, quantised at load.
    Q8_0,
}

impl WType {
    fn ggml(self) -> sys::ggml_type {
        match self {
            Self::F32 => sys::GGML_TYPE_F32,
            Self::F16 => sys::GGML_TYPE_F16,
            Self::Bf16 => sys::GGML_TYPE_BF16,
            Self::Q8_0 => sys::GGML_TYPE_Q8_0,
        }
    }
}

/// One weight to make resident.
#[derive(Debug, Clone)]
pub struct WeightSpec {
    /// Name in the weight files.
    pub name: String,
    /// Shape in file order (outermost first).
    pub shape: Vec<u64>,
    /// Device storage type.
    pub ty: WType,
    /// Order of the outermost dimension on the device: `rows[new] = old`.
    /// `None` keeps the file's order.
    pub rows: Option<Vec<usize>>,
    /// When not empty, the device tensor `name` is these file tensors, of
    /// equal shape, stacked along the outermost dimension (so one matrix
    /// product serves several projections of the same input).
    pub parts: Vec<String>,
}

impl WeightSpec {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, shape: &[u64], ty: WType) -> Self {
        Self { name: name.into(), shape: shape.to_vec(), ty, rows: None, parts: Vec::new() }
    }

    /// A device tensor stacking the file tensors `parts` (each of shape
    /// `part_shape`) along the outermost dimension.
    pub fn stacked(name: impl Into<String>, parts: &[String], part_shape: &[u64], ty: WType) -> Self {
        let mut shape = part_shape.to_vec();
        shape[0] *= parts.len() as u64;
        Self { name: name.into(), shape, ty, rows: None, parts: parts.to_vec() }
    }

    fn part_shape(&self) -> Vec<u64> {
        let mut shape = self.shape.clone();
        shape[0] /= self.parts.len().max(1) as u64;
        shape
    }

    /// Reorder the outermost dimension at load.
    #[must_use]
    pub fn with_rows(mut self, rows: Vec<usize>) -> Self {
        self.rows = Some(rows);
        self
    }
}

/// A weight computed on the host, uploaded by [`Weights::from_host`].
#[derive(Debug, Clone)]
pub struct HostTensor {
    /// Name the graph builders look it up by.
    pub name: String,
    /// Shape, outermost dimension first.
    pub shape: Vec<u64>,
    /// Device storage type.
    pub ty: WType,
    /// Values in row-major order.
    pub data: Vec<f32>,
}

/// Weights resident on a backend.
pub struct Weights {
    ctx: *mut sys::ggml_context,
    buffer: sys::ggml_backend_buffer_t,
    tensors: HashMap<String, *mut sys::ggml_tensor>,
    bytes: usize,
}

// SAFETY: the weights are immutable after upload and only read by graphs run
// under the owning pipeline's `&mut self`.
unsafe impl Send for Weights {}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights").field("tensors", &self.tensors.len()).field("bytes", &self.bytes).finish()
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        // SAFETY: both handles were created in `load` and are freed once.
        unsafe {
            sys::ggml_backend_buffer_free(self.buffer);
            sys::ggml_free(self.ctx);
        }
    }
}

fn ne_of(shape: &[u64]) -> [i64; 4] {
    let mut ne = [1i64; 4];
    for (i, d) in shape.iter().rev().enumerate() {
        ne[i] = *d as i64;
    }
    ne
}

impl Weights {
    /// Upload `specs` from `files` to `backend`, converting each tensor to its
    /// storage type. Quantisation is deterministic: the same file bytes always
    /// give the same device bytes.
    ///
    /// # Errors
    /// Missing or mis-shaped tensors, or an allocation failure.
    pub fn load(backend: &Backend, files: &SafeTensors, specs: &[WeightSpec]) -> Result<Self> {
        for s in specs {
            if s.shape.len() > 4 {
                return Err(Error::Weights(format!("{} has more than four dimensions", s.name)));
            }
            if s.parts.is_empty() {
                files.require(&s.name, &s.shape)?;
            } else {
                let dtypes = s
                    .parts
                    .iter()
                    .map(|p| files.require(p, &s.part_shape()).map(|v| v.dtype))
                    .collect::<Result<Vec<_>>>()?;
                if dtypes.windows(2).any(|w| w[0] != w[1]) {
                    return Err(Error::Weights(format!("{}: stacked tensors differ in type", s.name)));
                }
            }
            if let Some(rows) = &s.rows {
                let n = s.shape.first().copied().unwrap_or(0) as usize;
                let mut seen = vec![false; n];
                if rows.len() != n || rows.iter().any(|&r| r >= n || std::mem::replace(&mut seen[r], true)) {
                    return Err(Error::Weights(format!("{}: row order is not a permutation", s.name)));
                }
            }
            if s.ty == WType::Q8_0 && s.shape.last().copied().unwrap_or(0) % 32 != 0 {
                return Err(Error::Weights(format!("{} rows are not a multiple of 32", s.name)));
            }
        }
        let mut out = Self::alloc(backend, specs)?;
        for s in specs {
            let stacked;
            let view = if s.parts.is_empty() {
                files.require(&s.name, &s.shape)?
            } else {
                let part_shape = s.part_shape();
                let views = s.parts.iter().map(|p| files.require(p, &part_shape)).collect::<Result<Vec<_>>>()?;
                stacked = views.iter().flat_map(|v| v.bytes.iter().copied()).collect::<Vec<u8>>();
                TensorView { dtype: views[0].dtype, shape: &s.shape, bytes: &stacked }
            };
            let t = out.tensors[&s.name];
            let reordered;
            let view = match &s.rows {
                Some(rows) => {
                    let row = view.bytes.len() / rows.len();
                    reordered = rows.iter().flat_map(|&r| &view.bytes[r * row..(r + 1) * row]).copied().collect::<Vec<u8>>();
                    TensorView { dtype: view.dtype, shape: view.shape, bytes: &reordered }
                }
                None => view,
            };
            let data = convert(&view, s)?;
            unsafe { sys::ggml_backend_tensor_set(t, data.as_ptr().cast(), 0, data.len()) };
            out.bytes += data.len();
        }
        Ok(out)
    }

    /// Upload tensors computed on the host (weights derived from the files
    /// at load, such as a fused weight normalisation or a re-laid-out
    /// convolution kernel), each converted to its storage type.
    ///
    /// # Errors
    /// A size that disagrees with the shape, or an allocation failure.
    pub fn from_host(backend: &Backend, tensors: &[HostTensor]) -> Result<Self> {
        let specs: Vec<WeightSpec> = tensors.iter().map(|t| WeightSpec::new(t.name.clone(), &t.shape, t.ty)).collect();
        for (t, s) in tensors.iter().zip(&specs) {
            if t.data.len() as u64 != t.shape.iter().product::<u64>() {
                return Err(Error::Weights(format!("{}: {} values for shape {:?}", t.name, t.data.len(), t.shape)));
            }
            if s.ty == WType::Q8_0 && s.shape.last().copied().unwrap_or(0) % 32 != 0 {
                return Err(Error::Weights(format!("{} rows are not a multiple of 32", s.name)));
            }
        }
        let mut out = Self::alloc(backend, &specs)?;
        for (t, s) in tensors.iter().zip(&specs) {
            let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let view = TensorView { dtype: Dtype::F32, shape: &s.shape, bytes: &bytes };
            let data = convert(&view, s)?;
            unsafe { sys::ggml_backend_tensor_set(out.tensors[&s.name], data.as_ptr().cast(), 0, data.len()) };
            out.bytes += data.len();
        }
        Ok(out)
    }

    /// Resident tensors filled with zeros: state a sequence of graphs carries
    /// from one run to the next (see [`Graph::copy_into`]).
    ///
    /// # Errors
    /// An allocation failure.
    pub fn zeros(backend: &Backend, specs: &[WeightSpec]) -> Result<Self> {
        let mut out = Self::alloc(backend, specs)?;
        unsafe { sys::ggml_backend_buffer_clear(out.buffer, 0) };
        out.bytes = unsafe { sys::ggml_backend_buffer_get_size(out.buffer) };
        Ok(out)
    }

    /// Set every tensor to zero.
    pub fn clear(&self) {
        unsafe { sys::ggml_backend_buffer_clear(self.buffer, 0) };
    }

    fn alloc(backend: &Backend, specs: &[WeightSpec]) -> Result<Self> {
        let params = sys::ggml_init_params {
            mem_size: unsafe { sys::ggml_tensor_overhead() } * (specs.len() + 1),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = unsafe { sys::ggml_init(params) };
        if ctx.is_null() {
            return Err(Error::Backend("ggml_init failed for weights".into()));
        }
        let mut tensors = HashMap::with_capacity(specs.len());
        for s in specs {
            let ne = ne_of(&s.shape);
            let t = unsafe { sys::ggml_new_tensor_4d(ctx, s.ty.ggml(), ne[0], ne[1], ne[2], ne[3]) };
            let cname = CString::new(s.name.as_str()).map_err(|_| Error::Weights("NUL in tensor name".into()))?;
            unsafe { sys::ggml_set_name(t, cname.as_ptr()) };
            tensors.insert(s.name.clone(), t);
        }
        let buffer = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend.raw) };
        if buffer.is_null() {
            unsafe { sys::ggml_free(ctx) };
            return Err(Error::Backend(format!("allocating weights on {} failed", backend.name)));
        }
        Ok(Self { ctx, buffer, tensors, bytes: 0 })
    }

    /// A resident tensor by name.
    ///
    /// # Panics
    /// When the name was not part of the load; graph builders only ask for
    /// names they declared, so this is a programming error.
    #[must_use]
    pub fn get(&self, name: &str) -> Tn {
        Tn(*self.tensors.get(name).unwrap_or_else(|| panic!("weight {name} was not loaded")))
    }

    /// Device bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

fn convert(view: &TensorView<'_>, spec: &WeightSpec) -> Result<Vec<u8>> {
    Ok(match (spec.ty, view.dtype) {
        (WType::Bf16, Dtype::Bf16) | (WType::F16, Dtype::F16) | (WType::F32, Dtype::F32) => view.bytes.to_vec(),
        (WType::F32, _) => view.to_f32().iter().flat_map(|v| v.to_le_bytes()).collect(),
        (WType::F16, _) => view.to_f32().iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
        (WType::Bf16, _) => view.to_f32().iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
        (WType::Q8_0, _) => {
            let f = view.to_f32();
            let n_per_row = *spec.shape.last().expect("checked non-empty") as i64;
            let nrows = f.len() as i64 / n_per_row;
            let row = unsafe { sys::ggml_row_size(sys::GGML_TYPE_Q8_0, n_per_row) };
            let mut out = vec![0u8; row * nrows as usize];
            let written = unsafe {
                sys::ggml_quantize_chunk(
                    sys::GGML_TYPE_Q8_0,
                    f.as_ptr(),
                    out.as_mut_ptr().cast(),
                    0,
                    nrows,
                    n_per_row,
                    ptr::null(),
                )
            };
            if written != out.len() {
                return Err(Error::Weights(format!("quantising {} wrote {written} of {} bytes", spec.name, out.len())));
            }
            out
        }
    })
}

/// A tensor handle inside a graph or a weight store.
#[derive(Debug, Clone, Copy)]
pub struct Tn(pub(crate) *mut sys::ggml_tensor);

impl Tn {
    /// Dimension `i` (ggml order, innermost first).
    #[must_use]
    pub fn ne(self, i: usize) -> i64 {
        unsafe { (*self.0).ne[i] }
    }

    /// Byte stride of dimension `i`.
    #[must_use]
    pub fn nb(self, i: usize) -> usize {
        unsafe { (*self.0).nb[i] }
    }
}

/// Builds and runs one compute graph. Operations are thin wrappers over the
/// ggml calls of the same name.
pub struct Graph {
    ctx: *mut sys::ggml_context,
    gf: *mut sys::ggml_cgraph,
    galloc: sys::ggml_gallocr_t,
    backend: sys::ggml_backend_t,
    backend_name: String,
}

// SAFETY: see `Backend`.
unsafe impl Send for Graph {}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph").field("backend", &self.backend_name).finish()
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if !self.galloc.is_null() {
                sys::ggml_gallocr_free(self.galloc);
            }
            sys::ggml_free(self.ctx);
        }
    }
}

const MAX_NODES: usize = 32768;

/// ggml's multi-axis rotary mode, `GGML_ROPE_TYPE_MROPE` (a preprocessor
/// define, so not in the generated bindings).
const ROPE_MULTI: i32 = 8;

impl Graph {
    /// A new, empty graph for `backend`.
    ///
    /// # Errors
    /// When the metadata context cannot be created.
    pub fn new(backend: &Backend) -> Result<Self> {
        let size = unsafe {
            sys::ggml_tensor_overhead() * MAX_NODES + sys::ggml_graph_overhead_custom(MAX_NODES, false)
        };
        let ctx = unsafe {
            sys::ggml_init(sys::ggml_init_params { mem_size: size, mem_buffer: ptr::null_mut(), no_alloc: true })
        };
        if ctx.is_null() {
            return Err(Error::Backend("ggml_init failed for a graph".into()));
        }
        let gf = unsafe { sys::ggml_new_graph_custom(ctx, MAX_NODES, false) };
        Ok(Self { ctx, gf, galloc: ptr::null_mut(), backend: backend.raw, backend_name: backend.name.clone() })
    }

    /// Declare an input tensor, filled with [`Graph::set`] after
    /// [`Graph::finish`].
    pub fn input(&mut self, ty: sys::ggml_type, ne: &[i64]) -> Tn {
        let mut d = [1i64; 4];
        d[..ne.len()].copy_from_slice(ne);
        let t = unsafe { sys::ggml_new_tensor_4d(self.ctx, ty, d[0], d[1], d[2], d[3]) };
        unsafe { sys::ggml_set_input(t) };
        Tn(t)
    }

    /// Mark `out` as the graph output, allocate every intermediate, and check
    /// that the backend can execute every node.
    ///
    /// # Errors
    /// [`Error::UnsupportedOp`] naming the first node the backend cannot run,
    /// or an allocation failure.
    pub fn finish(&mut self, outputs: &[Tn]) -> Result<()> {
        for o in outputs {
            unsafe {
                sys::ggml_set_output(o.0);
                sys::ggml_build_forward_expand(self.gf, o.0);
            }
        }
        let n = unsafe { sys::ggml_graph_n_nodes(self.gf) };
        for i in 0..n {
            let node = unsafe { sys::ggml_graph_node(self.gf, i) };
            if !unsafe { sys::ggml_backend_supports_op(self.backend, node) } {
                let op = unsafe { CStr::from_ptr(sys::ggml_op_desc(node)) }.to_string_lossy().into_owned();
                let name = unsafe { CStr::from_ptr((*node).name.as_ptr()) }.to_string_lossy().into_owned();
                return Err(Error::UnsupportedOp { backend: self.backend_name.clone(), op, tensor: name });
            }
        }
        let galloc = unsafe { sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(self.backend)) };
        if galloc.is_null() {
            return Err(Error::Backend("graph allocator creation failed".into()));
        }
        self.galloc = galloc;
        if !unsafe { sys::ggml_gallocr_alloc_graph(galloc, self.gf) } {
            return Err(Error::Backend(format!("allocating the compute graph on {} failed", self.backend_name)));
        }
        Ok(())
    }

    /// Place `t` (and what it depends on) in the graph now, so later
    /// operations that use it stay adjacent to each other and can be fused.
    pub fn expand(&mut self, t: Tn) {
        unsafe { sys::ggml_build_forward_expand(self.gf, t.0) };
    }

    /// The graph's operations in execution order, for tests.
    #[cfg(test)]
    pub(crate) fn ops(&self) -> Vec<String> {
        let n = unsafe { sys::ggml_graph_n_nodes(self.gf) };
        (0..n)
            .map(|i| {
                let node = unsafe { sys::ggml_graph_node(self.gf, i) };
                unsafe { CStr::from_ptr(sys::ggml_op_desc(node)) }.to_string_lossy().into_owned()
            })
            .collect()
    }

    /// Bytes of device memory the allocated graph holds.
    #[must_use]
    pub fn compute_bytes(&self) -> usize {
        if self.galloc.is_null() {
            return 0;
        }
        unsafe { sys::ggml_gallocr_get_buffer_size(self.galloc, 0) }
    }

    /// Write f32 data into an input.
    pub fn set_f32(&self, t: Tn, data: &[f32]) {
        let n = unsafe { sys::ggml_nelements(t.0) } as usize;
        assert_eq!(n, data.len(), "input size mismatch");
        unsafe { sys::ggml_backend_tensor_set(t.0, data.as_ptr().cast(), 0, std::mem::size_of_val(data)) };
    }

    /// Write i32 data into an input.
    pub fn set_i32(&self, t: Tn, data: &[i32]) {
        let n = unsafe { sys::ggml_nelements(t.0) } as usize;
        assert_eq!(n, data.len(), "input size mismatch");
        unsafe { sys::ggml_backend_tensor_set(t.0, data.as_ptr().cast(), 0, std::mem::size_of_val(data)) };
    }

    /// Write f16 data (as f32 values) into an input.
    pub fn set_f16(&self, t: Tn, data: &[f32]) {
        let n = unsafe { sys::ggml_nelements(t.0) } as usize;
        assert_eq!(n, data.len(), "input size mismatch");
        let h: Vec<half::f16> = data.iter().map(|v| half::f16::from_f32(*v)).collect();
        unsafe { sys::ggml_backend_tensor_set(t.0, h.as_ptr().cast(), 0, h.len() * 2) };
    }

    /// Run the graph.
    ///
    /// # Errors
    /// When the backend reports a failure.
    pub fn compute(&self) -> Result<()> {
        let status = unsafe { sys::ggml_backend_graph_compute(self.backend, self.gf) };
        if status != sys::GGML_STATUS_SUCCESS {
            return Err(Error::Backend(format!("graph compute on {} failed with status {status}", self.backend_name)));
        }
        Ok(())
    }

    /// Read an f32 tensor back to the host.
    #[must_use]
    pub fn read_f32(&self, t: Tn) -> Vec<f32> {
        let n = unsafe { sys::ggml_nelements(t.0) } as usize;
        let mut out = vec![0f32; n];
        unsafe { sys::ggml_backend_tensor_get(t.0, out.as_mut_ptr().cast(), 0, n * 4) };
        out
    }

    // ---- operations -------------------------------------------------------

    /// `x @ w^T` for a weight stored `[out, in]`: result `[out, tokens]`.
    pub fn linear(&mut self, w: Tn, x: Tn) -> Tn {
        Tn(unsafe { sys::ggml_mul_mat(self.ctx, w.0, x.0) })
    }
    /// Linear with a bias.
    pub fn linear_b(&mut self, w: Tn, b: Tn, x: Tn) -> Tn {
        let y = self.linear(w, x);
        self.add(y, b)
    }
    /// Elementwise add with broadcasting of `b`.
    pub fn add(&mut self, a: Tn, b: Tn) -> Tn {
        Tn(unsafe { sys::ggml_add(self.ctx, a.0, b.0) })
    }
    /// Elementwise subtract with broadcasting of `b`.
    pub fn sub(&mut self, a: Tn, b: Tn) -> Tn {
        Tn(unsafe { sys::ggml_sub(self.ctx, a.0, b.0) })
    }
    /// Elementwise multiply with broadcasting of `b`.
    pub fn mul(&mut self, a: Tn, b: Tn) -> Tn {
        Tn(unsafe { sys::ggml_mul(self.ctx, a.0, b.0) })
    }
    /// `s * a + b`.
    pub fn scale_bias(&mut self, a: Tn, s: f32, b: f32) -> Tn {
        Tn(unsafe { sys::ggml_scale_bias(self.ctx, a.0, s, b) })
    }
    /// SiLU.
    pub fn silu(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_silu(self.ctx, a.0) })
    }
    /// GELU, tanh approximation.
    pub fn gelu_tanh(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_gelu(self.ctx, a.0) })
    }
    /// Logistic sigmoid.
    pub fn sigmoid(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_sigmoid(self.ctx, a.0) })
    }
    /// `silu(a[:n/2]) * a[n/2:]` along dimension 0.
    pub fn swiglu(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_swiglu(self.ctx, a.0) })
    }
    /// `silu(a) * b`.
    pub fn swiglu_split(&mut self, a: Tn, b: Tn) -> Tn {
        Tn(unsafe { sys::ggml_swiglu_split(self.ctx, a.0, b.0) })
    }
    /// Layer norm without affine parameters, over dimension 0.
    pub fn norm(&mut self, a: Tn, eps: f32) -> Tn {
        Tn(unsafe { sys::ggml_norm(self.ctx, a.0, eps) })
    }
    /// RMS norm over dimension 0.
    pub fn rms_norm(&mut self, a: Tn, eps: f32) -> Tn {
        Tn(unsafe { sys::ggml_rms_norm(self.ctx, a.0, eps) })
    }
    /// Group norm over `[W, H, C]` with `groups` groups of channels.
    pub fn group_norm(&mut self, a: Tn, groups: i32, eps: f32) -> Tn {
        Tn(unsafe { sys::ggml_group_norm(self.ctx, a.0, groups, eps) })
    }
    /// Make a tensor contiguous.
    pub fn cont(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_cont(self.ctx, a.0) })
    }
    /// Cast to another type.
    pub fn cast(&mut self, a: Tn, ty: sys::ggml_type) -> Tn {
        Tn(unsafe { sys::ggml_cast(self.ctx, a.0, ty) })
    }
    /// Reshape (the tensor must be contiguous).
    pub fn reshape(&mut self, a: Tn, ne: &[i64]) -> Tn {
        let c = self.ctx;
        Tn(unsafe {
            match ne.len() {
                1 => sys::ggml_reshape_1d(c, a.0, ne[0]),
                2 => sys::ggml_reshape_2d(c, a.0, ne[0], ne[1]),
                3 => sys::ggml_reshape_3d(c, a.0, ne[0], ne[1], ne[2]),
                _ => sys::ggml_reshape_4d(c, a.0, ne[0], ne[1], ne[2], ne[3]),
            }
        })
    }
    /// Permute axes.
    pub fn permute(&mut self, a: Tn, p: [i32; 4]) -> Tn {
        Tn(unsafe { sys::ggml_permute(self.ctx, a.0, p[0], p[1], p[2], p[3]) })
    }
    /// A 1-d view of `n` elements starting at element `offset`.
    pub fn view_1d(&mut self, a: Tn, n: i64, offset_elems: usize) -> Tn {
        let es = unsafe { sys::ggml_element_size(a.0) };
        Tn(unsafe { sys::ggml_view_1d(self.ctx, a.0, n, offset_elems * es) })
    }
    /// Rows `[from, from + n)` of dimension 0 across every column: a
    /// `[n, ne1]` view.
    pub fn view_rows(&mut self, a: Tn, from: i64, n: i64) -> Tn {
        let es = unsafe { sys::ggml_element_size(a.0) };
        Tn(unsafe { sys::ggml_view_2d(self.ctx, a.0, n, a.ne(1), a.nb(1), from as usize * es) })
    }
    /// Rows `[from, from + head_dim * heads)` of a `[rows, tokens]` matrix,
    /// viewed as `[head_dim, heads, tokens]`.
    pub fn view_heads(&mut self, a: Tn, from: i64, head_dim: i64, heads: i64) -> Tn {
        let es = unsafe { sys::ggml_element_size(a.0) };
        let nb1 = head_dim as usize * es;
        Tn(unsafe { sys::ggml_view_3d(self.ctx, a.0, head_dim, heads, a.ne(1), nb1, a.nb(1), from as usize * es) })
    }
    /// Columns `[from, from + n)` of dimension 1: a `[ne0, n]` view.
    pub fn view_cols(&mut self, a: Tn, from: i64, n: i64) -> Tn {
        Tn(unsafe { sys::ggml_view_2d(self.ctx, a.0, a.ne(0), n, a.nb(1), from as usize * a.nb(1)) })
    }
    /// A general 4-d view with explicit byte strides and offset.
    #[allow(clippy::too_many_arguments)]
    pub fn view_4d(&mut self, a: Tn, ne: [i64; 4], nb1: usize, nb2: usize, nb3: usize, offset: usize) -> Tn {
        Tn(unsafe { sys::ggml_view_4d(self.ctx, a.0, ne[0], ne[1], ne[2], ne[3], nb1, nb2, nb3, offset) })
    }
    /// Concatenate along `dim`.
    pub fn concat(&mut self, a: Tn, b: Tn, dim: i32) -> Tn {
        Tn(unsafe { sys::ggml_concat(self.ctx, a.0, b.0, dim) })
    }
    /// Scaled dot-product attention. `q`: `[d, n_q, heads]`, `k`/`v`:
    /// `[d, n_kv, heads_kv]` (f16). Result `[d, heads, n_q]`. `f32_scores`
    /// accumulates the scores in f32; without it they are f16, which is
    /// faster and safe when queries and keys are normalised.
    pub fn attention(&mut self, q: Tn, k: Tn, v: Tn, mask: Option<Tn>, scale: f32, f32_scores: bool) -> Tn {
        let m = mask.map_or(ptr::null_mut(), |m| m.0);
        let t = unsafe { sys::ggml_flash_attn_ext(self.ctx, q.0, k.0, v.0, m, scale, 0.0, 0.0) };
        if f32_scores {
            unsafe { sys::ggml_flash_attn_ext_set_prec(t, sys::GGML_PREC_F32) };
        }
        Tn(t)
    }
    /// Multi-axis rotary embedding over split-half pairs: `pos` holds four
    /// positions per token (see ggml's `ggml_rope_multi`), `sections` the pair
    /// count per axis, `freq_factors` a divisor per pair.
    pub fn rope_multi(&mut self, a: Tn, pos: Tn, freq_factors: Tn, n_dims: i32, sections: [i32; 4], base: f32) -> Tn {
        let mut s = sections;
        Tn(unsafe {
            sys::ggml_rope_multi(
                self.ctx, a.0, pos.0, freq_factors.0, n_dims, s.as_mut_ptr(), ROPE_MULTI, 0, base, 1.0, 0.0, 1.0, 0.0, 0.0,
            )
        })
    }
    /// Attention with the same operands and output layout as
    /// [`Graph::attention`] (`q` `[d, n, heads]`, `k`/`v` `[d, m, kv heads]`,
    /// output `[d, heads, n]`), computed unfused and entirely in float32.
    pub fn attention_exact(&mut self, q: Tn, k: Tn, v: Tn, mask: Option<Tn>, scale: f32) -> Tn {
        let m = mask.map_or(ptr::null_mut(), |m| m.0);
        let kq = self.linear(k, q);
        let kq = Tn(unsafe { sys::ggml_soft_max_ext(self.ctx, kq.0, m, scale, 0.0) });
        let vt = self.permute(v, [1, 0, 2, 3]);
        let vt = self.cont(vt);
        let o = self.linear(vt, kq);
        let o = self.permute(o, [0, 2, 1, 3]);
        self.cont(o)
    }
    /// Softmax over dimension 0 of `scale * a`.
    pub fn soft_max(&mut self, a: Tn, scale: f32) -> Tn {
        Tn(unsafe { sys::ggml_soft_max_ext(self.ctx, a.0, ptr::null_mut(), scale, 0.0) })
    }
    /// Rotary embedding, `mode` as in ggml (NEOX = 2).
    pub fn rope(&mut self, a: Tn, pos: Tn, n_dims: i32, mode: i32, base: f32) -> Tn {
        Tn(unsafe {
            sys::ggml_rope_ext(self.ctx, a.0, pos.0, ptr::null_mut(), n_dims, mode, 0, base, 1.0, 0.0, 1.0, 0.0, 0.0)
        })
    }
    /// 2-d convolution, kernel `[KW, KH, IC, OC]`, input `[W, H, C, N]`,
    /// stride 1, padding `pad`.
    pub fn conv2d(&mut self, kernel: Tn, x: Tn, pad: i32) -> Tn {
        Tn(unsafe { sys::ggml_conv_2d_direct(self.ctx, kernel.0, x.0, 1, 1, pad, pad, 1, 1) })
    }
    /// 2-d convolution with stride 2 and no padding.
    pub fn conv2d_stride2(&mut self, kernel: Tn, x: Tn) -> Tn {
        Tn(unsafe { sys::ggml_conv_2d_direct(self.ctx, kernel.0, x.0, 2, 2, 0, 0, 1, 1) })
    }
    /// Zero-pad the end of dimensions 0 and 1.
    pub fn pad_end(&mut self, a: Tn, p0: i32, p1: i32) -> Tn {
        Tn(unsafe { sys::ggml_pad(self.ctx, a.0, p0, p1, 0, 0) })
    }
    /// Nearest-neighbour upscale of dimensions 0 and 1.
    pub fn upscale_nearest(&mut self, a: Tn, factor: i32) -> Tn {
        Tn(unsafe { sys::ggml_upscale(self.ctx, a.0, factor, sys::GGML_SCALE_MODE_NEAREST) })
    }
    /// Element-wise sine.
    pub fn sin(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_sin(self.ctx, a.0) })
    }
    /// Rotary embedding from tables: `x * cos + rotate_half(x) * sin` for `x`
    /// `[head width, heads, tokens]` and tables `[head width, 1, tokens]`,
    /// where `rotate_half` pairs dimension `i` with `i + head width / 2`.
    pub fn rotate_half_rope(&mut self, x: Tn, cos: Tn, sin: Tn) -> Tn {
        let (hd, heads, n) = (x.ne(0), x.ne(1), x.ne(2));
        let half = hd / 2;
        let es = x.nb(0);
        let lo = self.view_4d(x, [half, heads, n, 1], x.nb(1), x.nb(2), x.nb(3), 0);
        let hi = self.view_4d(x, [half, heads, n, 1], x.nb(1), x.nb(2), x.nb(3), half as usize * es);
        let lo = self.cont(lo);
        let hi = self.cont(hi);
        let neg = self.scale_bias(hi, -1.0, 0.0);
        let rot = self.concat(neg, lo, 0);
        let a = self.mul(x, cos);
        let b = self.mul(rot, sin);
        self.add(a, b)
    }
    /// Element-wise hyperbolic tangent.
    pub fn tanh(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_tanh(self.ctx, a.0) })
    }
    /// `a` repeated to the shape `ne`, each dimension a multiple of `a`'s.
    pub fn repeat_to(&mut self, a: Tn, ne: [i64; 4]) -> Tn {
        Tn(unsafe { sys::ggml_repeat_4d(self.ctx, a.0, ne[0], ne[1], ne[2], ne[3]) })
    }
    /// Element-wise `max(a, 0)`.
    pub fn relu(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_relu(self.ctx, a.0) })
    }
    /// 2-d convolution with stride `s` in both dimensions and no padding.
    pub fn conv2d_strided(&mut self, kernel: Tn, x: Tn, s: i32) -> Tn {
        Tn(unsafe { sys::ggml_conv_2d_direct(self.ctx, kernel.0, x.0, s, s, 0, 0, 1, 1) })
    }
    /// Copy `src` into the resident tensor `dst` (same element count) when
    /// the graph runs. Every reader of `dst` in the graph must be an ancestor
    /// of `src`, so the copy runs after them.
    pub fn copy_into(&mut self, src: Tn, dst: Tn) {
        let c = Tn(unsafe { sys::ggml_cpy(self.ctx, src.0, dst.0) });
        self.expand(c);
    }
    /// Element-wise square.
    pub fn sqr(&mut self, a: Tn) -> Tn {
        Tn(unsafe { sys::ggml_sqr(self.ctx, a.0) })
    }
    /// Rotate along each dimension by the given amounts, wrapping around.
    pub fn roll(&mut self, a: Tn, s0: i32, s1: i32) -> Tn {
        Tn(unsafe { sys::ggml_roll(self.ctx, a.0, s0, s1, 0, 0) })
    }
    /// Row lookup (token embedding).
    pub fn get_rows(&mut self, table: Tn, ids: Tn) -> Tn {
        Tn(unsafe { sys::ggml_get_rows(self.ctx, table.0, ids.0) })
    }
}
