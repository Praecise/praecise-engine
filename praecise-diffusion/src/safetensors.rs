//! Read-only access to safetensors weight files.
//!
//! Files are memory-mapped and never copied whole: a tensor is converted to
//! its device type one at a time and uploaded, so host memory stays at one
//! tensor's worth above the mapping.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde_json::Value;

use crate::error::{Error, Result};
use llama_cpp_sys_2 as sys;

/// Element type of a stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE float32.
    F32,
    /// IEEE float16.
    F16,
    /// bfloat16.
    Bf16,
    /// A non-float type (counters and the like), of the given element size.
    /// Indexed so the file's layout is checked, never loaded as a weight.
    Other(usize),
    /// float8 e4m3 (finite, no infinities), as scaled weights store it; see
    /// [`SafeTensors::dequantize_scaled_f8`].
    F8E4M3,
    /// Block-quantised rows of a GGUF file, by their ggml type. A weight
    /// stored this way keeps its blocks on the device.
    Quant(u32),
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "F64" | "I64" | "U64" => Some(Self::Other(8)),
            "I32" | "U32" => Some(Self::Other(4)),
            "I16" | "U16" => Some(Self::Other(2)),
            "F8_E4M3" => Some(Self::F8E4M3),
            "I8" | "U8" | "BOOL" | "F8_E5M2" => Some(Self::Other(1)),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Other(n) => n,
            Self::F8E4M3 => 1,
            Self::Quant(_) => 0,
        }
    }

    /// Bytes of a tensor of this type and shape (outermost first), or `None`
    /// when quantised rows do not hold whole blocks.
    fn bytes(self, shape: &[u64]) -> Option<u64> {
        let numel: u64 = shape.iter().product();
        let Self::Quant(t) = self else { return Some(numel * self.size() as u64) };
        let row = shape.last().copied().unwrap_or(1);
        // SAFETY: plain lookups in ggml's static type table; `t` was checked
        // to name a type when the file was parsed.
        let (blck, size) = unsafe { (sys::ggml_blck_size(t as sys::ggml_type) as u64, sys::ggml_type_size(t as sys::ggml_type) as u64) };
        (blck > 0 && row % blck == 0).then(|| numel / blck * size)
    }
}

#[derive(Clone)]
struct Entry {
    file: usize,
    dtype: Dtype,
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

/// Tensor bytes: a mapped file, or rows regrouped in memory.
enum Store {
    Map(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Store {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Map(m) => m,
            Self::Owned(v) => v,
        }
    }
}

/// A set of safetensors files addressed as one tensor namespace.
pub struct SafeTensors {
    maps: Vec<Store>,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
}

impl std::fmt::Debug for SafeTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensors")
            .field("files", &self.maps.len())
            .field("tensors", &self.entries.len())
            .finish()
    }
}

/// A borrowed view of one tensor.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// Element type.
    pub dtype: Dtype,
    /// Shape, outermost dimension first (the file's order).
    pub shape: &'a [u64],
    /// Raw little-endian bytes.
    pub bytes: &'a [u8],
}

impl TensorView<'_> {
    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Decode the tensor to f32. Views from [`SafeTensors::require`] are
    /// always float; any other type decodes to nothing.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::Other(_) => Vec::new(),
            Dtype::F8E4M3 => self.bytes.iter().map(|&b| f8e4m3(b)).collect(),
            Dtype::Quant(t) => {
                let mut out = vec![0f32; self.numel()];
                // SAFETY: `t` names a quantised type with a decoder (checked
                // at parse), `bytes` holds exactly `numel` elements of it and
                // `out` has room for them.
                unsafe {
                    let traits = sys::ggml_get_type_traits(t as sys::ggml_type);
                    if let Some(f) = (*traits).to_float {
                        f(self.bytes.as_ptr().cast(), out.as_mut_ptr(), out.len() as i64);
                    }
                }
                out
            }
            Dtype::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::F16 => self
                .bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::Bf16 => self
                .bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
        }
    }
}

impl SafeTensors {
    /// Open one or more safetensors or GGUF files. Tensor names must be
    /// unique across the set; a GGUF file's string metadata joins the
    /// `__metadata__` entries.
    ///
    /// # Errors
    /// Fails when a file cannot be mapped, its header is malformed, a tensor
    /// uses an unsupported dtype, or a name appears twice.
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for (file_idx, path) in paths.iter().enumerate() {
            let map = map_file(path)?;
            if map.len() < 8 {
                return Err(Error::Weights(format!("{} is shorter than a header", path.display())));
            }
            if map[..4] == *b"GGUF" {
                for (name, entry) in parse_gguf(&map, file_idx, &mut metadata).map_err(|e| Error::Weights(format!("{}: {e}", path.display())))? {
                    if entries.insert(name.clone(), entry).is_some() {
                        return Err(Error::Weights(format!("tensor {name} appears in more than one file")));
                    }
                }
                maps.push(Store::Map(map));
                continue;
            }
            let header_len = u64::from_le_bytes(map[..8].try_into().expect("8 bytes")) as usize;
            let data_start = 8usize
                .checked_add(header_len)
                .filter(|&e| e <= map.len())
                .ok_or_else(|| Error::Weights(format!("{} header overruns the file", path.display())))?;
            let header: HashMap<String, Value> = serde_json::from_slice(&map[8..data_start])
                .map_err(|e| Error::Weights(format!("{} header: {e}", path.display())))?;
            for (name, meta) in header {
                if name == "__metadata__" {
                    if let Value::Object(m) = meta {
                        metadata.extend(m.into_iter().filter_map(|(k, v)| v.as_str().map(|v| (k, v.to_string()))));
                    }
                    continue;
                }
                let entry = parse_entry(&name, &meta, file_idx, data_start, map.len())?;
                if entries.insert(name.clone(), entry).is_some() {
                    return Err(Error::Weights(format!("tensor {name} appears in more than one file")));
                }
            }
            maps.push(Store::Map(map));
        }
        Ok(Self { maps, entries, metadata })
    }

    /// A string entry of the files' `__metadata__` header.
    #[must_use]
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// The same tensors under new names: `rename` gives each tensor's new
    /// name, or `None` to leave it out.
    ///
    /// # Errors
    /// When two tensors would share a name.
    pub fn renamed(mut self, rename: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let mut entries = HashMap::with_capacity(self.entries.len());
        for (name, entry) in self.entries.drain() {
            if let Some(new) = rename(&name) {
                if entries.insert(new.clone(), entry).is_some() {
                    return Err(Error::Weights(format!("two tensors are renamed to {new}")));
                }
            }
        }
        self.entries = entries;
        Ok(self)
    }

    /// Tensor `name` viewed under another shape of the same element count
    /// (a file that folds dimensions together, as GGUF does past four).
    ///
    /// # Errors
    /// A missing tensor, a block-quantised one, or a different element count.
    pub fn reshaped(mut self, name: &str, shape: &[u64]) -> Result<Self> {
        let e = self.entries.get_mut(name).ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if matches!(e.dtype, Dtype::Quant(_)) || e.shape.iter().product::<u64>() != shape.iter().product::<u64>() {
            return Err(Error::Weights(format!("{name}: {:?} cannot be viewed as {shape:?}", e.shape)));
        }
        e.shape = shape.to_vec();
        Ok(self)
    }

    /// Replace tensor `name` by tensors made of its rows (outermost
    /// dimension): each part takes the listed half-open row ranges in order.
    /// A part that is one range stays a view of the file; others are copied.
    /// Rows of a block-quantised tensor are whole blocks, so this works for
    /// every stored type.
    ///
    /// # Errors
    /// A missing tensor, or ranges outside its rows.
    pub fn regroup_rows(mut self, name: &str, parts: &[(String, Vec<(u64, u64)>)]) -> Result<Self> {
        let e = self.entries.remove(name).ok_or_else(|| Error::Weights(format!("missing tensor {name}")))?;
        let rows = *e.shape.first().ok_or_else(|| Error::Weights(format!("{name} has no rows")))?;
        let row_bytes = (e.end - e.start) / rows.max(1) as usize;
        for (new, ranges) in parts {
            if ranges.iter().any(|&(a, b)| a >= b || b > rows) {
                return Err(Error::Weights(format!("{name}: row ranges outside its {rows} rows")));
            }
            let n: u64 = ranges.iter().map(|&(a, b)| b - a).sum();
            let mut shape = e.shape.clone();
            shape[0] = n;
            let entry = if let [(a, b)] = ranges[..] {
                Entry { file: e.file, dtype: e.dtype, shape, start: e.start + a as usize * row_bytes, end: e.start + b as usize * row_bytes }
            } else {
                let src = &self.maps[e.file];
                let mut buf = Vec::with_capacity(n as usize * row_bytes);
                for &(a, b) in ranges {
                    buf.extend_from_slice(&src[e.start + a as usize * row_bytes..e.start + b as usize * row_bytes]);
                }
                let len = buf.len();
                self.maps.push(Store::Owned(buf));
                Entry { file: self.maps.len() - 1, dtype: e.dtype, shape, start: 0, end: len }
            };
            if self.entries.insert(new.clone(), entry).is_some() {
                return Err(Error::Weights(format!("tensor {new} already exists")));
            }
        }
        Ok(self)
    }

    /// Float8 (e4m3) weights `<layer>.weight` stored with a float32 scale
    /// `<layer>.weight_scale`, one for the tensor or one per row, become
    /// bfloat16 tensors of their scaled values. The layer's other
    /// quantisation tensors (`<layer>.input_scale`, and non-float ones) are
    /// dropped: the products run at the load precision.
    ///
    /// # Errors
    /// A float8 tensor without a float scale of one element or one per row.
    pub fn dequantize_scaled_f8(mut self) -> Result<Self> {
        let names: Vec<String> = self.entries.iter().filter(|(_, e)| e.dtype == Dtype::F8E4M3).map(|(n, _)| n.clone()).collect();
        for name in names {
            let scale_name = format!("{name}_scale");
            let rows = self.entries[&name].shape.first().copied().unwrap_or(1) as usize;
            let scale = match self.get(&scale_name) {
                Some(v) if (v.numel() == 1 || v.numel() == rows) && !matches!(v.dtype, Dtype::Other(_) | Dtype::Quant(_)) => v.to_f32(),
                _ => return Err(Error::Weights(format!("float8 tensor {name} has no scale {scale_name}"))),
            };
            let e = self.entries.remove(&name).expect("listed");
            let src = &self.maps[e.file][e.start..e.end];
            let row_len = src.len() / rows.max(1);
            let mut buf = Vec::with_capacity(2 * src.len());
            for (i, &b) in src.iter().enumerate() {
                let s = if scale.len() == 1 { scale[0] } else { scale[i / row_len] };
                buf.extend_from_slice(&half::bf16::from_f32(f8e4m3(b) * s).to_bits().to_le_bytes());
            }
            let len = buf.len();
            self.maps.push(Store::Owned(buf));
            self.entries.insert(name.clone(), Entry { file: self.maps.len() - 1, dtype: Dtype::Bf16, shape: e.shape, start: 0, end: len });
            self.entries.remove(&scale_name);
            if let Some(layer) = name.strip_suffix(".weight") {
                let prefix = format!("{layer}.");
                self.entries.retain(|n, e| !(n.starts_with(&prefix) && (n.ends_with(".input_scale") || matches!(e.dtype, Dtype::Other(_)))));
            }
        }
        Ok(self)
    }

    /// NVFP4 weights become bfloat16 tensors of their values. A layer stores
    /// `<layer>.weight` as bytes of two 4-bit e2m1 values (the low nibble
    /// first), `<layer>.weight_scale` as one float8 e4m3 scale per 16 values
    /// of a row, and `<layer>.weight_scale_2` as one float32 scale for the
    /// tensor; a value is `e2m1 * weight_scale * weight_scale_2`. The layer's
    /// scales and `<layer>.input_scale` are dropped.
    ///
    /// # Errors
    /// When a layer's tensors disagree in shape or type.
    pub fn dequantize_nvfp4(mut self) -> Result<Self> {
        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let layers: Vec<String> =
            self.entries.keys().filter_map(|n| n.strip_suffix(".weight_scale_2").map(str::to_owned)).collect();
        for layer in layers {
            let (w, ws, ws2) = (format!("{layer}.weight"), format!("{layer}.weight_scale"), format!("{layer}.weight_scale_2"));
            let bad = |what: &str| Error::Weights(format!("NVFP4 layer {layer}: {what}"));
            let global = match self.get(&ws2) {
                Some(v) if v.numel() == 1 && !matches!(v.dtype, Dtype::Other(_) | Dtype::Quant(_)) => v.to_f32()[0],
                _ => return Err(bad("weight_scale_2 is not one float")),
            };
            let (packed, scales) = match (self.entries.get(&w), self.entries.get(&ws)) {
                (Some(p), Some(s)) if p.dtype == Dtype::Other(1) && s.dtype == Dtype::F8E4M3 => (p.clone(), s.clone()),
                _ => return Err(bad("weight is not packed bytes or weight_scale is not float8")),
            };
            let (rows, half_cols) = match packed.shape[..] {
                [r, c] => (r as usize, c as usize),
                _ => return Err(bad("weight is not a matrix")),
            };
            let cols = 2 * half_cols;
            if cols % 16 != 0 || scales.shape[..] != [rows as u64, (cols / 16) as u64] {
                return Err(bad("weight_scale does not hold one scale per 16 values"));
            }
            let src = &self.maps[packed.file][packed.start..packed.end];
            let sc = &self.maps[scales.file][scales.start..scales.end];
            let mut buf = Vec::with_capacity(2 * rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    let byte = src[r * half_cols + c / 2];
                    let nib = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    let v = E2M1[usize::from(nib & 7)] * if nib & 8 != 0 { -1.0 } else { 1.0 };
                    let v = v * f8e4m3(sc[r * (cols / 16) + c / 16]) * global;
                    buf.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
                }
            }
            let len = buf.len();
            self.maps.push(Store::Owned(buf));
            let shape = vec![rows as u64, cols as u64];
            self.entries.insert(w, Entry { file: self.maps.len() - 1, dtype: Dtype::Bf16, shape, start: 0, end: len });
            for n in [ws, ws2, format!("{layer}.input_scale")] {
                self.entries.remove(&n);
            }
        }
        Ok(self)
    }

    /// One namespace over several opened sets. Tensor names must be unique
    /// across them; a metadata key keeps the value of the first set that has
    /// it.
    ///
    /// # Errors
    /// When a name appears in more than one set.
    pub fn merge(parts: Vec<Self>) -> Result<Self> {
        let mut out = Self { maps: Vec::new(), entries: HashMap::new(), metadata: HashMap::new() };
        for part in parts {
            let base = out.maps.len();
            for (name, mut entry) in part.entries {
                entry.file += base;
                if out.entries.contains_key(&name) {
                    return Err(Error::Weights(format!("tensor {name} appears in more than one file")));
                }
                out.entries.insert(name, entry);
            }
            for (k, v) in part.metadata {
                out.metadata.entry(k).or_insert(v);
            }
            out.maps.extend(part.maps);
        }
        Ok(out)
    }

    /// Look up a tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<TensorView<'_>> {
        self.entries.get(name).map(|e| TensorView {
            dtype: e.dtype,
            shape: &e.shape,
            bytes: &self.maps[e.file][e.start..e.end],
        })
    }

    /// Drop the resident pages of a mapped tensor once its bytes have been
    /// copied out, so a load holds one copy of the weights at a time rather
    /// than the whole file next to the engine's buffers. The bytes stay
    /// readable: a later read faults them in from the file again.
    pub fn release(&self, name: &str) {
        #[cfg(unix)]
        if let Some(e) = self.entries.get(name) {
            if let Some(Store::Map(map)) = self.maps.get(e.file) {
                // A multiple of every page size in use (4, 16 and 64 KiB).
                const PAGE: usize = 1 << 16;
                let start = e.start.div_ceil(PAGE) * PAGE;
                let end = e.end / PAGE * PAGE;
                if end > start {
                    // SAFETY: the mapping is private, read-only and backed by
                    // a file nobody writes while it is mapped, so dropping its
                    // pages only means the next read reloads the same bytes.
                    let _ = unsafe { map.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, start, end - start) };
                }
            }
        }
        #[cfg(not(unix))]
        let _ = name;
    }

    /// Look up a tensor that must exist with the given shape.
    ///
    /// # Errors
    /// [`Error::MissingTensor`] or [`Error::TensorShape`].
    pub fn require(&self, name: &str, shape: &[u64]) -> Result<TensorView<'_>> {
        let view = self.get(name).ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if matches!(view.dtype, Dtype::Other(_)) {
            return Err(Error::Weights(format!("tensor {name} is not a float tensor")));
        }
        if view.shape != shape {
            return Err(Error::TensorShape {
                name: name.to_string(),
                found: view.shape.to_vec(),
                expected: shape.to_vec(),
            });
        }
        Ok(view)
    }

    /// Names of all tensors.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}

/// One float8 e4m3 value: 4 exponent bits (bias 7), 3 mantissa bits,
/// subnormals below exponent 1, `0x7f`/`0xff` NaN, no infinities.
fn f8e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((b >> 3) & 0x0f), f32::from(b & 0x07));
    if e == 0x0f && m == 7.0 {
        return f32::NAN;
    }
    let v = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    sign * v
}

fn map_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path)?;
    // SAFETY: the mapping is read-only and the weight files are not modified
    // while a pipeline holds them; a concurrent writer would be an operator
    // error that the content hash check before load already rules out.
    let map = unsafe { Mmap::map(&file)? };
    Ok(map)
}

fn parse_entry(name: &str, meta: &Value, file: usize, data_start: usize, file_len: usize) -> Result<Entry> {
    let bad = |what: &str| Error::Weights(format!("tensor {name}: {what}"));
    let dtype_str = meta.get("dtype").and_then(Value::as_str).ok_or_else(|| bad("no dtype"))?;
    let dtype = Dtype::parse(dtype_str).ok_or_else(|| bad(&format!("unsupported dtype {dtype_str}")))?;
    let shape: Vec<u64> = meta
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("no shape"))?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| bad("non-integer shape")))
        .collect::<Result<_>>()?;
    let offsets = meta
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 2)
        .ok_or_else(|| bad("no data_offsets"))?;
    let rel_start = offsets[0].as_u64().ok_or_else(|| bad("bad offset"))? as usize;
    let rel_end = offsets[1].as_u64().ok_or_else(|| bad("bad offset"))? as usize;
    let start = data_start + rel_start;
    let end = data_start + rel_end;
    if end < start || end > file_len || Some((end - start) as u64) != dtype.bytes(&shape) {
        return Err(bad("data range does not match its shape"));
    }
    Ok(Entry { file, dtype, shape, start, end })
}

/// Cursor over a GGUF header.
struct Gguf<'a> {
    b: &'a [u8],
    o: usize,
}

impl Gguf<'_> {
    fn take(&mut self, n: usize) -> std::result::Result<&[u8], String> {
        let end = self.o.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("header overruns the file")?;
        let s = &self.b[self.o..end];
        self.o = end;
        Ok(s)
    }

    fn u32(&mut self) -> std::result::Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }

    fn u64(&mut self) -> std::result::Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }

    fn string(&mut self) -> std::result::Result<String, String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "a string is not UTF-8".to_string())
    }

    /// A value of type `ty`, returned when it is a string or an integer.
    fn value(&mut self, ty: u32) -> std::result::Result<Option<String>, String> {
        Ok(match ty {
            0 | 1 | 7 => Some(self.take(1)?[0].to_string()),
            2 | 3 => Some(u16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes")).to_string()),
            4 | 5 => Some(self.u32()?.to_string()),
            6 => {
                self.take(4)?;
                None
            }
            8 => Some(self.string()?),
            9 => {
                let (et, n) = (self.u32()?, self.u64()?);
                for _ in 0..n {
                    self.value(et)?;
                }
                None
            }
            10 | 11 => Some(self.u64()?.to_string()),
            12 => {
                self.take(8)?;
                None
            }
            _ => return Err(format!("unknown metadata type {ty}")),
        })
    }
}

/// The tensors of a GGUF file (version 2 or 3), shapes outermost first.
fn parse_gguf(map: &[u8], file: usize, metadata: &mut HashMap<String, String>) -> std::result::Result<Vec<(String, Entry)>, String> {
    let mut r = Gguf { b: map, o: 4 };
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("GGUF version {version} is not supported"));
    }
    let (n_tensors, n_kv) = (r.u64()?, r.u64()?);
    let mut alignment = 32usize;
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        if let Some(v) = r.value(ty)? {
            if key == "general.alignment" {
                alignment = v.parse().map_err(|_| "bad alignment")?;
            }
            metadata.entry(key).or_insert(v);
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!("alignment {alignment} is not a power of two"));
    }
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims == 0 || n_dims > 4 {
            return Err(format!("tensor {name} has {n_dims} dimensions"));
        }
        let mut shape = (0..n_dims).map(|_| r.u64()).collect::<std::result::Result<Vec<_>, _>>()?;
        shape.reverse();
        let (ty, offset) = (r.u32()?, r.u64()? as usize);
        infos.push((name, shape, ty, offset));
    }
    let data = r.o.div_ceil(alignment) * alignment;
    let mut out = Vec::with_capacity(infos.len());
    for (name, shape, ty, offset) in infos {
        let dtype = match ty {
            0 => Dtype::F32,
            1 => Dtype::F16,
            30 => Dtype::Bf16,
            // SAFETY: a range-checked lookup in ggml's static type table.
            t if t < sys::GGML_TYPE_COUNT as u32 && unsafe { sys::ggml_is_quantized(t as sys::ggml_type) } => Dtype::Quant(t),
            t => return Err(format!("tensor {name}: unsupported type {t}")),
        };
        let len = dtype.bytes(&shape).ok_or_else(|| format!("tensor {name}: rows are not whole blocks"))? as usize;
        let start = data.checked_add(offset).ok_or("bad offset")?;
        let end = start.checked_add(len).filter(|&e| e <= map.len()).ok_or_else(|| format!("tensor {name}: data overruns the file"))?;
        out.push((name, Entry { file, dtype, shape, start, end }));
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn float8_e4m3_decodes() {
        assert_eq!(f8e4m3(0x38), 1.0);
        assert_eq!(f8e4m3(0xb8), -1.0);
        assert_eq!(f8e4m3(0x7e), 448.0);
        assert_eq!(f8e4m3(0x01), 2f32.powi(-9));
        assert!(f8e4m3(0x7f).is_nan());
    }

    /// Write a safetensors file holding f32 tensors, for tests.
    pub(crate) fn write_f32(path: &Path, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape, data) in tensors {
            let bytes = data.len() * 4;
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        for (_, _, data) in tensors {
            for v in data {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    /// Write a safetensors file of raw tensors `(name, dtype, shape, bytes)`.
    fn write_raw(path: &Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, data) in tensors {
            header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.len()]}));
            offset += data.len();
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        for (_, _, _, data) in tensors {
            f.write_all(data).unwrap();
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn float8_weights_take_a_tensor_scale_or_one_per_row() {
        let dir = std::env::temp_dir().join(format!("pd-st-f8-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.safetensors");
        // 1.0 and -1.0 in two rows of two.
        let w = vec![0x38, 0xb8, 0x38, 0xb8];
        write_raw(&path, &[
            ("t.weight", "F8_E4M3", vec![2, 2], w.clone()),
            ("t.weight_scale", "F32", vec![], f32_bytes(&[3.0])),
            ("r.weight", "F8_E4M3", vec![2, 2], w),
            ("r.weight_scale", "F32", vec![2, 1], f32_bytes(&[2.0, 0.5])),
            ("r.input_scale", "F32", vec![], f32_bytes(&[1.0])),
        ]);
        let st = SafeTensors::open(&[path]).unwrap().dequantize_scaled_f8().unwrap();
        assert_eq!(st.require("t.weight", &[2, 2]).unwrap().to_f32(), vec![3.0, -3.0, 3.0, -3.0]);
        assert_eq!(st.require("r.weight", &[2, 2]).unwrap().to_f32(), vec![2.0, -2.0, 0.5, -0.5]);
        assert!(st.get("r.weight_scale").is_none() && st.get("r.input_scale").is_none());
    }

    #[test]
    fn nvfp4_weights_decode_to_their_values() {
        let dir = std::env::temp_dir().join(format!("pd-st-fp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.safetensors");
        // Row 0: the eight positive e2m1 codes, then their negatives; row 1:
        // 6.0 throughout. Low nibble first.
        let mut packed = Vec::new();
        for i in 0..8u8 {
            let (a, b) = if i < 4 { (2 * i, 2 * i + 1) } else { (8 | (2 * (i - 4)), 8 | (2 * (i - 4) + 1)) };
            packed.push(a | (b << 4));
        }
        packed.extend([0x77u8; 8]);
        write_raw(&path, &[
            ("l.weight", "U8", vec![2, 8], packed),
            // Scales 2.0 and 0.5 (e4m3 0x40 and 0x30).
            ("l.weight_scale", "F8_E4M3", vec![2, 1], vec![0x40, 0x30]),
            ("l.weight_scale_2", "F32", vec![], f32_bytes(&[0.25])),
            ("l.input_scale", "F32", vec![], f32_bytes(&[1.0])),
        ]);
        let st = SafeTensors::open(&[path]).unwrap().dequantize_nvfp4().unwrap();
        let v = st.require("l.weight", &[2, 16]).unwrap().to_f32();
        let codes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let row0: Vec<f32> = codes.iter().chain(codes.iter()).enumerate().map(|(i, c)| if i < 8 { c * 0.5 } else { -c * 0.5 }).collect();
        assert_eq!(&v[..16], &row0[..]);
        assert_eq!(&v[16..], &[6.0 * 0.125; 16]);
        assert!(st.get("l.weight_scale").is_none() && st.get("l.weight_scale_2").is_none() && st.get("l.input_scale").is_none());
    }

    #[test]
    fn reads_back_what_was_written_and_rejects_a_wrong_shape() {
        let dir = std::env::temp_dir().join(format!("pd-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.safetensors");
        write_f32(&path, &[("w".into(), vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        let st = SafeTensors::open(&[path]).unwrap();
        let v = st.require("w", &[2, 3]).unwrap();
        assert_eq!(v.to_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matches!(st.require("w", &[3, 2]), Err(Error::TensorShape { .. })));
        assert!(matches!(st.require("x", &[1]), Err(Error::MissingTensor(_))));
    }

    #[test]
    fn a_tensor_is_viewed_under_another_shape_of_its_size() {
        let dir = std::env::temp_dir().join(format!("pd-st-reshape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.safetensors");
        write_f32(&path, &[("w".into(), vec![6, 2], (0..12).map(|v| v as f32).collect())]);
        let st = SafeTensors::open(&[path]).unwrap().reshaped("w", &[2, 3, 2]).unwrap();
        assert_eq!(st.require("w", &[2, 3, 2]).unwrap().to_f32(), (0..12).map(|v| v as f32).collect::<Vec<_>>());
        assert!(st.reshaped("w", &[5]).is_err());
    }

    #[test]
    fn a_header_that_claims_more_data_than_the_file_holds_is_refused() {
        let dir = std::env::temp_dir().join(format!("pd-st-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        write_f32(&path, &[("w".into(), vec![4], vec![1.0, 2.0, 3.0, 4.0])]);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
        assert!(matches!(SafeTensors::open(&[path]), Err(Error::Weights(_))));
    }

    fn gguf_str(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    #[test]
    fn gguf_tensors_keep_their_blocks_and_decode() {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        gguf_str(&mut b, "config");
        b.extend(8u32.to_le_bytes());
        gguf_str(&mut b, "{\"a\":1}");
        gguf_str(&mut b, "general.alignment");
        b.extend(4u32.to_le_bytes());
        b.extend(32u32.to_le_bytes());
        // [2][3] f32, then [2][32] Q8_0 (two blocks).
        gguf_str(&mut b, "a");
        b.extend(2u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        gguf_str(&mut b, "q");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend((sys::GGML_TYPE_Q8_0 as u32).to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            b.extend(v.to_le_bytes());
        }
        b.resize(b.len() + 8, 0);
        for (scale, base) in [(0.5f32, 0i8), (0.25, -16)] {
            b.extend(half::f16::from_f32(scale).to_le_bytes());
            b.extend((0..32).map(|i| (base + i as i8) as u8));
        }
        let path = std::env::temp_dir().join(format!("praecise-gguf-{}.gguf", std::process::id()));
        std::fs::write(&path, &b).unwrap();
        let st = SafeTensors::open(&[path.clone()]).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(st.metadata("config"), Some("{\"a\":1}"));
        let a = st.require("a", &[2, 3]).unwrap();
        assert_eq!(a.to_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let q = st.require("q", &[2, 32]).unwrap();
        assert_eq!(q.dtype, Dtype::Quant(sys::GGML_TYPE_Q8_0 as u32));
        assert_eq!(q.bytes.len(), 2 * 34);
        let f = q.to_f32();
        assert_eq!(f[1], 0.5);
        assert_eq!(f[32], -4.0);
        assert_eq!(f[63], 15.0 * 0.25);
    }

}
