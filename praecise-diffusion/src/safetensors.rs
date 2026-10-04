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
            "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => Some(Self::Other(1)),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Other(n) => n,
        }
    }
}

struct Entry {
    file: usize,
    dtype: Dtype,
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

/// A set of safetensors files addressed as one tensor namespace.
pub struct SafeTensors {
    maps: Vec<Mmap>,
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
    /// Open one or more safetensors files. Tensor names must be unique across
    /// the set.
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
            maps.push(map);
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

    /// Look up a tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<TensorView<'_>> {
        self.entries.get(name).map(|e| TensorView {
            dtype: e.dtype,
            shape: &e.shape,
            bytes: &self.maps[e.file][e.start..e.end],
        })
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
    let numel: u64 = shape.iter().product();
    if end < start || end > file_len || (end - start) as u64 != numel * dtype.size() as u64 {
        return Err(bad("data range does not match its shape"));
    }
    Ok(Entry { file, dtype, shape, start, end })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

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
    fn a_header_that_claims_more_data_than_the_file_holds_is_refused() {
        let dir = std::env::temp_dir().join(format!("pd-st-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        write_f32(&path, &[("w".into(), vec![4], vec![1.0, 2.0, 3.0, 4.0])]);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
        assert!(matches!(SafeTensors::open(&[path]), Err(Error::Weights(_))));
    }
}
