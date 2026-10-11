//! Sparse voxel tensors: integer coordinates on a cubic grid with one
//! feature column per occupied voxel.
//!
//! The device holds features as `[channels, voxels]`; the host holds the
//! coordinates and builds the index lists the graph needs. A 3x3x3
//! submanifold convolution (outputs exactly at the input voxels) is one row
//! gather over the 27 neighbours followed by one matrix product, so it runs
//! wherever a row gather and a matrix product run, CUDA included.

use std::collections::HashMap;

use crate::ggml::{Graph, Tn};

/// Bytes the gathered neighbourhood of one convolution chunk may take; larger
/// voxel sets are convolved a chunk at a time inside the same graph.
const GATHER_BUDGET: usize = 256 << 20;

/// Occupied voxels of a cubic grid, in a fixed order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Voxels {
    /// `(x, y, z)` per voxel.
    pub coords: Vec<[i32; 3]>,
    /// Grid side.
    pub resolution: i32,
}

impl Voxels {
    /// Voxels in the given order.
    #[must_use]
    pub fn new(coords: Vec<[i32; 3]>, resolution: i32) -> Self {
        Self { coords, resolution }
    }

    /// Every voxel of a `res^3` grid, `x` slowest and `z` fastest (the order
    /// of a dense `[C, X, Y, Z]` volume flattened).
    #[must_use]
    pub fn dense(res: i32) -> Self {
        let mut coords = Vec::with_capacity((res * res * res) as usize);
        for x in 0..res {
            for y in 0..res {
                for z in 0..res {
                    coords.push([x, y, z]);
                }
            }
        }
        Self { coords, resolution: res }
    }

    /// Number of voxels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether there are no voxels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    fn index(&self) -> HashMap<[i32; 3], i32> {
        self.coords.iter().enumerate().map(|(i, c)| (*c, i as i32)).collect()
    }

    /// For each voxel and each of the 27 offsets `(dx, dy, dz)` in
    /// `{-1, 0, 1}^3` (`dx` slowest), the index of the voxel at `c + d`, or
    /// `len()` (a zero row) when that voxel is not occupied.
    #[must_use]
    pub fn neighbours(&self) -> Vec<i32> {
        let index = self.index();
        let missing = self.len() as i32;
        let mut out = Vec::with_capacity(self.len() * 27);
        for c in &self.coords {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let n = [c[0] + dx, c[1] + dy, c[2] + dz];
                        out.push(index.get(&n).copied().unwrap_or(missing));
                    }
                }
            }
        }
        out
    }

    /// Split each voxel into the children `subdivide` selects (8 flags per
    /// voxel, child `s` at offset `(s & 1, (s >> 1) & 1, (s >> 2) & 1)`),
    /// returning the children on the doubled grid and, per child, the row of
    /// the `[voxels * 8]` child table it takes its features from.
    ///
    /// # Panics
    /// When `subdivide` does not hold 8 flags per voxel.
    #[must_use]
    pub fn subdivide(&self, subdivide: &[bool]) -> (Voxels, Vec<i32>) {
        assert_eq!(subdivide.len(), self.len() * 8, "8 subdivision flags per voxel");
        let mut coords = Vec::new();
        let mut rows = Vec::new();
        for (i, c) in self.coords.iter().enumerate() {
            for s in 0..8 {
                if subdivide[i * 8 + s] {
                    coords.push([c[0] * 2 + (s & 1) as i32, c[1] * 2 + ((s >> 1) & 1) as i32, c[2] * 2 + ((s >> 2) & 1) as i32]);
                    rows.push((i * 8 + s) as i32);
                }
            }
        }
        (Voxels { coords, resolution: self.resolution * 2 }, rows)
    }
}

/// Index lists for convolving one voxel set, uploaded once per graph.
#[derive(Debug, Clone, Copy)]
pub struct ConvIndex {
    /// The `[27 * voxels]` neighbour indices.
    pub ids: Tn,
    /// Voxel count.
    pub n: i64,
}

impl ConvIndex {
    /// Declare the index input for `voxels`; fill it with
    /// [`ConvIndex::set`] after the graph is finished.
    pub fn declare(g: &mut Graph, voxels: &Voxels) -> Self {
        let n = voxels.len() as i64;
        let ids = g.input(llama_cpp_sys_2::GGML_TYPE_I32, &[n * 27]);
        Self { ids, n }
    }

    /// Upload the neighbour indices.
    pub fn set(&self, g: &Graph, voxels: &Voxels) {
        g.set_i32(self.ids, &voxels.neighbours());
    }
}

/// 3x3x3 submanifold convolution of `x` (`[C, N]`, f32) with a kernel stored
/// `[27 * C, C_out]` (offset-major, channels fastest, as [`Voxels::neighbours`]
/// orders the neighbours) and an optional bias: result `[C_out, N]`.
pub fn conv3(g: &mut Graph, x: Tn, idx: ConvIndex, w: Tn, b: Option<Tn>) -> Tn {
    let c = x.ne(0);
    assert_eq!(w.ne(0), 27 * c, "kernel width is 27 x input channels");
    // Append the zero row missing neighbours point at.
    let table = g.pad_end(x, 0, 1);
    let per = (GATHER_BUDGET / (27 * c as usize * 4)).max(1) as i64;
    let mut out: Option<Tn> = None;
    let mut from = 0i64;
    while from < idx.n {
        let n = per.min(idx.n - from);
        let ids = g.view_1d(idx.ids, n * 27, (from * 27) as usize);
        let cols = g.get_rows(table, ids);
        let cols = g.reshape(cols, &[27 * c, n]);
        let y = g.linear(w, cols);
        out = Some(match out {
            None => y,
            Some(prev) => g.concat(prev, y, 1),
        });
        from += n;
    }
    let y = out.expect("a convolution over no voxels");
    match b {
        Some(b) => g.add(y, b),
        None => y,
    }
}

/// Row-major `[C_out, C_in, 3, 3, 3]` kernel (dense convolution layout)
/// re-laid out as `[C_out, 27, C_in]` for [`conv3`].
#[must_use]
pub fn kernel_from_dense(w: &[f32], c_out: usize, c_in: usize) -> Vec<f32> {
    assert_eq!(w.len(), c_out * c_in * 27, "kernel size");
    let mut out = vec![0f32; w.len()];
    for o in 0..c_out {
        for i in 0..c_in {
            for k in 0..27 {
                out[(o * 27 + k) * c_in + i] = w[(o * c_in + i) * 27 + k];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_point_at_the_zero_row_when_absent() {
        let v = Voxels::new(vec![[0, 0, 0], [0, 0, 1]], 2);
        let n = v.neighbours();
        assert_eq!(n.len(), 54);
        // centre offset is index 13
        assert_eq!(n[13], 0);
        assert_eq!(n[27 + 13], 1);
        // voxel 0, offset (0, 0, +1) is index 14
        assert_eq!(n[14], 1);
        // voxel 1, offset (0, 0, -1) is index 12
        assert_eq!(n[27 + 12], 0);
        assert_eq!(n[0], 2);
    }

    #[test]
    fn subdivision_doubles_coordinates_in_child_order() {
        let v = Voxels::new(vec![[1, 2, 3]], 4);
        let mut flags = vec![false; 8];
        flags[0] = true;
        flags[5] = true;
        let (c, rows) = v.subdivide(&flags);
        assert_eq!(c.coords, vec![[2, 4, 6], [3, 4, 7]]);
        assert_eq!(rows, vec![0, 5]);
        assert_eq!(c.resolution, 8);
    }

    #[test]
    fn dense_kernel_relayout() {
        let (co, ci) = (2, 3);
        let w: Vec<f32> = (0..co * ci * 27).map(|v| v as f32).collect();
        let r = kernel_from_dense(&w, co, ci);
        // out 1, offset 5, in 2
        assert_eq!(r[(27 + 5) * ci + 2], w[(ci + 2) * 27 + 5]);
    }
}
