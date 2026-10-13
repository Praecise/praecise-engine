//! Mesh extraction and texturing for voxel-based 3D generators.
//!
//! * [`dual_grid_mesh`]: one vertex per occupied voxel (placed inside or near
//!   its voxel), one quad per crossed grid edge whose four surrounding voxels
//!   are all occupied, each quad split along the diagonal with the larger
//!   product of split weights.
//! * [`repair`]: drops duplicate faces, splits non-manifold edges and
//!   vertices, drops specks and winds every part consistently outward.
//! * [`fill_small_holes`]: closes boundary loops shorter than a perimeter.
//! * [`simplify`]: quadric-error edge collapse to a face budget.
//! * [`unwrap`]: charts grown over faces within a normal cone, projected to
//!   their mean plane and shelf-packed into one square atlas.
//! * [`bake`]: rasterises every chart into the atlas and samples a sparse
//!   attribute volume trilinearly at each texel's surface point, then grows
//!   the texels into the gutters.
//!
//! Every step is deterministic: ties break on indices, never on addresses
//! or hash order.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f32 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> Option<V3> {
    let n = norm(a);
    (n > 1e-20 && n.is_finite()).then(|| scale(a, 1.0 / n))
}

/// A triangle mesh.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<V3>,
    /// Triangles (counter-clockwise seen from outside when consistent).
    pub faces: Vec<[u32; 3]>,
}

/// Summary statistics of a mesh, for agreement checks between
/// implementations and for receipts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStats {
    /// Vertex count.
    pub vertices: usize,
    /// Face count.
    pub faces: usize,
    /// Total surface area.
    pub area: f64,
    /// Axis-aligned bounds.
    pub min: [f64; 3],
    /// Axis-aligned bounds.
    pub max: [f64; 3],
    /// Area-weighted centroid of the surface.
    pub centroid: [f64; 3],
}

impl Mesh {
    /// Area, bounds and surface centroid.
    #[must_use]
    pub fn stats(&self) -> MeshStats {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for v in &self.vertices {
            for a in 0..3 {
                min[a] = min[a].min(f64::from(v[a]));
                max[a] = max[a].max(f64::from(v[a]));
            }
        }
        let mut area = 0f64;
        let mut c = [0f64; 3];
        for f in &self.faces {
            let (a, b, d) = (self.vertices[f[0] as usize], self.vertices[f[1] as usize], self.vertices[f[2] as usize]);
            let ar = 0.5 * f64::from(norm(cross(sub(b, a), sub(d, a))));
            area += ar;
            for k in 0..3 {
                c[k] += ar * f64::from(a[k] + b[k] + d[k]) / 3.0;
            }
        }
        if area > 0.0 {
            c.iter_mut().for_each(|x| *x /= area);
        }
        MeshStats { vertices: self.vertices.len(), faces: self.faces.len(), area, min, max, centroid: c }
    }

    /// Area-weighted unit vertex normals; an isolated or degenerate vertex
    /// takes the normal of the largest face touching it, or +Z when it
    /// touches none.
    #[must_use]
    pub fn vertex_normals(&self) -> Vec<V3> {
        let mut acc = vec![[0f32; 3]; self.vertices.len()];
        for f in &self.faces {
            let (a, b, c) = (self.vertices[f[0] as usize], self.vertices[f[1] as usize], self.vertices[f[2] as usize]);
            let n = cross(sub(b, a), sub(c, a));
            for &i in f {
                acc[i as usize] = add(acc[i as usize], n);
            }
        }
        acc.into_iter().map(|n| unit(n).unwrap_or([0.0, 0.0, 1.0])).collect()
    }

    /// Drop faces with a repeated vertex or zero area, and vertices no face
    /// uses.
    pub fn clean(&mut self) {
        let vs = &self.vertices;
        self.faces.retain(|f| {
            f[0] != f[1] && f[1] != f[2] && f[0] != f[2] && {
                let (a, b, c) = (vs[f[0] as usize], vs[f[1] as usize], vs[f[2] as usize]);
                norm(cross(sub(b, a), sub(c, a))) > 0.0
            }
        });
        let mut map = vec![u32::MAX; self.vertices.len()];
        let mut verts = Vec::new();
        for f in &mut self.faces {
            for i in f.iter_mut() {
                if map[*i as usize] == u32::MAX {
                    map[*i as usize] = verts.len() as u32;
                    verts.push(self.vertices[*i as usize]);
                }
                *i = map[*i as usize];
            }
        }
        self.vertices = verts;
    }
}

/// Grid placement of a dual-grid mesh: the cube `[lo, lo + size]^3` split
/// into `grid` voxels per side.
#[derive(Debug, Clone, Copy)]
pub struct GridFrame {
    /// Lower corner.
    pub lo: V3,
    /// Side length of the whole cube.
    pub size: f32,
    /// Voxels per side.
    pub grid: i32,
}

/// Extract the dual-grid surface of occupied voxels `coords` with per-voxel
/// vertex offsets `offsets` (in voxel units from the voxel's lower corner),
/// edge-crossing flags `crossed` (`[x, y, z]` edges leaving the voxel's lower
/// corner) and quad split weights `split`.
///
/// # Panics
/// When the per-voxel inputs disagree in length.
#[must_use]
pub fn dual_grid_mesh(coords: &[[i32; 3]], offsets: &[V3], crossed: &[[bool; 3]], split: &[f32], frame: GridFrame) -> Mesh {
    let n = coords.len();
    assert!(offsets.len() == n && crossed.len() == n && split.len() == n, "per-voxel inputs disagree");
    const EDGE: [[[i32; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let index: HashMap<[i32; 3], u32> = coords.iter().enumerate().map(|(i, c)| (*c, i as u32)).collect();
    let vs = frame.size / frame.grid as f32;
    let vertices: Vec<V3> = coords
        .iter()
        .zip(offsets)
        .map(|(c, o)| {
            [
                (c[0] as f32 + o[0]) * vs + frame.lo[0],
                (c[1] as f32 + o[1]) * vs + frame.lo[1],
                (c[2] as f32 + o[2]) * vs + frame.lo[2],
            ]
        })
        .collect();
    let mut faces = Vec::new();
    for (i, c) in coords.iter().enumerate() {
        for axis in 0..3 {
            if !crossed[i][axis] {
                continue;
            }
            let mut q = [0u32; 4];
            let mut ok = true;
            for (k, off) in EDGE[axis].iter().enumerate() {
                match index.get(&[c[0] + off[0], c[1] + off[1], c[2] + off[2]]) {
                    Some(&v) => q[k] = v,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let w02 = split[q[0] as usize] * split[q[2] as usize];
            let w13 = split[q[1] as usize] * split[q[3] as usize];
            if w02 > w13 {
                faces.push([q[0], q[1], q[2]]);
                faces.push([q[0], q[2], q[3]]);
            } else {
                faces.push([q[0], q[1], q[3]]);
                faces.push([q[3], q[1], q[2]]);
            }
        }
    }
    Mesh { vertices, faces }
}

/// Close every boundary loop whose perimeter is below `max_perimeter` with a
/// fan around a new centroid vertex. Returns the number of loops closed.
pub fn fill_small_holes(mesh: &mut Mesh, max_perimeter: f32) -> usize {
    // Directed boundary half-edges: those whose twin no face has.
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for f in &mesh.faces {
        for k in 0..3 {
            *count.entry((f[k], f[(k + 1) % 3])).or_default() += 1;
        }
    }
    let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut boundary: Vec<(u32, u32)> = count.keys().copied().filter(|&(a, b)| !count.contains_key(&(b, a))).collect();
    boundary.sort_unstable();
    for &(a, b) in &boundary {
        next.entry(a).or_default().push(b);
    }
    let mut used: HashMap<(u32, u32), bool> = HashMap::new();
    let mut closed = 0;
    for &(a0, b0) in &boundary {
        if used.contains_key(&(a0, b0)) {
            continue;
        }
        // Trace the loop; give up on branching (non-manifold) boundaries.
        let mut lp = vec![a0];
        let mut cur = b0;
        let mut ok = true;
        used.insert((a0, b0), true);
        while cur != a0 {
            let outs = next.get(&cur).map_or(&[][..], Vec::as_slice);
            if outs.len() != 1 || lp.len() > 4096 {
                ok = false;
                break;
            }
            lp.push(cur);
            let nx = outs[0];
            if used.insert((cur, nx), true).is_some() {
                ok = false;
                break;
            }
            cur = nx;
        }
        if !ok || lp.len() < 3 {
            continue;
        }
        let per: f32 = (0..lp.len()).map(|i| norm(sub(mesh.vertices[lp[(i + 1) % lp.len()] as usize], mesh.vertices[lp[i] as usize]))).sum();
        if per >= max_perimeter {
            continue;
        }
        if lp.len() == 3 {
            mesh.faces.push([lp[2], lp[1], lp[0]]);
        } else {
            let c = lp.iter().fold([0f32; 3], |s, &v| add(s, mesh.vertices[v as usize]));
            let ci = mesh.vertices.len() as u32;
            mesh.vertices.push(scale(c, 1.0 / lp.len() as f32));
            for i in 0..lp.len() {
                // The hole's own faces run opposite to the boundary edges.
                mesh.faces.push([lp[(i + 1) % lp.len()], lp[i], ci]);
            }
        }
        closed += 1;
    }
    closed
}

/// Drop faces over the same three vertices as an earlier face. Returns the
/// number dropped.
pub fn remove_duplicate_faces(mesh: &mut Mesh) -> usize {
    let before = mesh.faces.len();
    let mut seen: HashMap<[u32; 3], ()> = HashMap::with_capacity(before);
    mesh.faces.retain(|f| {
        let mut k = *f;
        k.sort_unstable();
        seen.insert(k, ()).is_none()
    });
    before - mesh.faces.len()
}

/// Faces on each undirected edge, in face order.
fn edge_faces(faces: &[[u32; 3]]) -> HashMap<(u32, u32), Vec<u32>> {
    let mut e: HashMap<(u32, u32), Vec<u32>> = HashMap::with_capacity(faces.len() * 3 / 2);
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            e.entry((a.min(b), a.max(b))).or_default().push(fi as u32);
        }
    }
    e
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Make every edge border at most two faces and every vertex a single fan:
/// faces meet only across edges they share with exactly one other face, and
/// a vertex whose faces fall into several such fans is copied once per extra
/// fan. Edges that still border more than two faces keep their first two.
/// Returns the number of vertex copies plus faces dropped.
pub fn split_non_manifold(mesh: &mut Mesh) -> usize {
    let orig = mesh.faces.clone();
    let e = edge_faces(&orig);
    let nv = mesh.vertices.len();
    let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (fi, f) in mesh.faces.iter().enumerate() {
        for &v in f {
            vfaces[v as usize].push(fi as u32);
        }
    }
    let mut changes = 0;
    let mut parent: Vec<u32> = Vec::new();
    let mut root_vertex: Vec<(u32, u32)> = Vec::new();
    for v in 0..nv as u32 {
        let fs = std::mem::take(&mut vfaces[v as usize]);
        if fs.len() < 2 {
            continue;
        }
        parent.clear();
        parent.extend(0..fs.len() as u32);
        for (i, &f) in fs.iter().enumerate() {
            for &w in &orig[f as usize] {
                if w == v {
                    continue;
                }
                let on = &e[&(v.min(w), v.max(w))];
                if on.len() == 2 {
                    let g = if on[0] == f { on[1] } else { on[0] };
                    if let Ok(j) = fs.binary_search(&g) {
                        let (ri, rj) = (find(&mut parent, i as u32), find(&mut parent, j as u32));
                        if ri != rj {
                            parent[ri.max(rj) as usize] = ri.min(rj);
                        }
                    }
                }
            }
        }
        // The fan holding the lowest face keeps `v`; each other fan gets a copy.
        root_vertex.clear();
        for i in 0..fs.len() {
            let r = find(&mut parent, i as u32);
            let target = match root_vertex.iter().find(|(rr, _)| *rr == r) {
                Some(&(_, t)) => t,
                None => {
                    let t = if root_vertex.is_empty() {
                        v
                    } else {
                        mesh.vertices.push(mesh.vertices[v as usize]);
                        changes += 1;
                        (mesh.vertices.len() - 1) as u32
                    };
                    root_vertex.push((r, t));
                    t
                }
            };
            if target != v {
                for x in &mut mesh.faces[fs[i] as usize] {
                    if *x == v {
                        *x = target;
                    }
                }
            }
        }
    }
    let mut keep = vec![true; mesh.faces.len()];
    let mut over: Vec<_> = edge_faces(&mesh.faces).into_values().filter(|fs| fs.len() > 2).collect();
    over.sort_unstable();
    for fs in over {
        for &f in &fs[2..] {
            if std::mem::replace(&mut keep[f as usize], false) {
                changes += 1;
            }
        }
    }
    let mut k = keep.into_iter();
    mesh.faces.retain(|_| k.next().unwrap_or(true));
    changes
}

/// Wind every face consistently with its neighbours across two-face edges,
/// spreading from the lowest face of each connected part, then turn each
/// part so its signed volume is positive (outward-facing when closed).
/// Returns the number of faces flipped.
pub fn orient_faces(mesh: &mut Mesh) -> usize {
    let e = edge_faces(&mesh.faces);
    let nf = mesh.faces.len();
    let mut seen = vec![false; nf];
    let mut flips = 0;
    let mut part = Vec::new();
    let mut stack = Vec::new();
    for start in 0..nf {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        part.clear();
        stack.push(start as u32);
        while let Some(f) = stack.pop() {
            part.push(f);
            let tri = mesh.faces[f as usize];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let on = &e[&(a.min(b), a.max(b))];
                if on.len() != 2 {
                    continue;
                }
                let g = if on[0] == f { on[1] } else { on[0] } as usize;
                if seen[g] {
                    continue;
                }
                seen[g] = true;
                let t = &mut mesh.faces[g];
                // A consistent neighbour runs the shared edge as b -> a.
                if (0..3).any(|j| t[j] == a && t[(j + 1) % 3] == b) {
                    t.swap(1, 2);
                    flips += 1;
                }
                stack.push(g as u32);
            }
        }
        let vol: f64 = part
            .iter()
            .map(|&f| {
                let t = mesh.faces[f as usize];
                let (a, b, c) = (mesh.vertices[t[0] as usize], mesh.vertices[t[1] as usize], mesh.vertices[t[2] as usize]);
                f64::from(dot(a, cross(b, c)))
            })
            .sum();
        if vol < 0.0 {
            for &f in &part {
                mesh.faces[f as usize].swap(1, 2);
            }
            flips += part.len();
        }
    }
    flips
}

/// Drop edge-connected parts whose total area is below `min_area`. Returns
/// the number of faces dropped.
pub fn remove_small_components(mesh: &mut Mesh, min_area: f64) -> usize {
    let nf = mesh.faces.len();
    let mut parent: Vec<u32> = (0..nf as u32).collect();
    let mut on: Vec<_> = edge_faces(&mesh.faces).into_values().collect();
    on.sort_unstable();
    for fs in on {
        for &g in &fs[1..] {
            let (r0, r1) = (find(&mut parent, fs[0]), find(&mut parent, g));
            if r0 != r1 {
                parent[r0.max(r1) as usize] = r0.min(r1);
            }
        }
    }
    let mut area = vec![0f64; nf];
    for (fi, t) in mesh.faces.iter().enumerate() {
        let (a, b, c) = (mesh.vertices[t[0] as usize], mesh.vertices[t[1] as usize], mesh.vertices[t[2] as usize]);
        let r = find(&mut parent, fi as u32);
        area[r as usize] += 0.5 * f64::from(norm(cross(sub(b, a), sub(c, a))));
    }
    let before = nf;
    let mut fi = 0u32;
    mesh.faces.retain(|_| {
        let r = find(&mut parent, fi);
        fi += 1;
        area[r as usize] >= min_area
    });
    before - mesh.faces.len()
}

/// The repairs before and after simplification: duplicate faces, then
/// non-manifold edges and vertices, then specks, then winding; finally
/// unused vertices.
pub fn repair(mesh: &mut Mesh, min_area: f64) {
    mesh.clean();
    remove_duplicate_faces(mesh);
    split_non_manifold(mesh);
    remove_small_components(mesh, min_area);
    orient_faces(mesh);
    mesh.clean();
}

#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: [f64; 3], d: f64, w: f64) -> Self {
        let (a, b, c) = (n[0], n[1], n[2]);
        Self([a * a * w, a * b * w, a * c * w, a * d * w, b * b * w, b * c * w, b * d * w, c * c * w, c * d * w, d * d * w])
    }
    fn add(&mut self, o: &Self) {
        for i in 0..10 {
            self.0[i] += o.0[i];
        }
    }
    fn eval(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let (x, y, z) = (p[0], p[1], p[2]);
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y + q[7] * z * z + 2.0 * q[8] * z + q[9]
    }
    fn optimum(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        let m = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let scale = q[0].abs() + q[4].abs() + q[7].abs();
        if det.abs() <= 1e-12 * scale * scale * scale || !det.is_finite() {
            return None;
        }
        let r = [-q[3], -q[6], -q[8]];
        let solve = |col: usize| {
            let mut mm = m;
            for i in 0..3 {
                mm[i][col] = r[i];
            }
            (mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1]) - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0]) + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0])) / det
        };
        Some([solve(0), solve(1), solve(2)])
    }
}

#[derive(PartialEq)]
struct Cand {
    cost: f64,
    a: u32,
    b: u32,
    stamp: u64,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.total_cmp(&self.cost).then_with(|| o.a.cmp(&self.a)).then_with(|| o.b.cmp(&self.b))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Collapse edges in order of least quadric error until at most
/// `max_faces` faces remain, never flipping a face or pinching the surface
/// into a non-manifold edge. Returns the number of collapses.
pub fn simplify(mesh: &mut Mesh, max_faces: usize) -> usize {
    if mesh.faces.len() <= max_faces {
        return 0;
    }
    let nv = mesh.vertices.len();
    let mut pos: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect();
    let mut faces: Vec<[u32; 3]> = mesh.faces.clone();
    let mut alive = vec![true; faces.len()];
    let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (fi, f) in faces.iter().enumerate() {
        for &v in f {
            vfaces[v as usize].push(fi as u32);
        }
    }
    let mut quad = vec![Quadric::default(); nv];
    let face_plane = |p: &[[f64; 3]], f: &[u32; 3]| {
        let (a, b, c) = (p[f[0] as usize], p[f[1] as usize], p[f[2] as usize]);
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        (n, l)
    };
    for f in &faces {
        let (n, l) = face_plane(&pos, f);
        if l > 0.0 {
            let u = [n[0] / l, n[1] / l, n[2] / l];
            let a = pos[f[0] as usize];
            let d = -(u[0] * a[0] + u[1] * a[1] + u[2] * a[2]);
            let q = Quadric::plane(u, d, l * 0.5);
            for &v in f {
                quad[v as usize].add(&q);
            }
        }
    }
    // Boundary edges get a perpendicular penalty plane so borders keep shape.
    let mut ecount: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            let e = ecount.entry(key).or_insert((0, fi as u32));
            e.0 += 1;
        }
    }
    let mut keys: Vec<_> = ecount.iter().filter(|(_, c)| c.0 == 1).map(|(k, c)| (*k, c.1)).collect();
    keys.sort_unstable();
    for ((a, b), fi) in keys {
        let (n, l) = face_plane(&pos, &faces[fi as usize]);
        if l <= 0.0 {
            continue;
        }
        let pa = pos[a as usize];
        let pb = pos[b as usize];
        let e = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let p = [e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]];
        let pl = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if pl > 0.0 {
            let u = [p[0] / pl, p[1] / pl, p[2] / pl];
            let d = -(u[0] * pa[0] + u[1] * pa[1] + u[2] * pa[2]);
            let q = Quadric::plane(u, d, 1000.0 * (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));
            quad[a as usize].add(&q);
            quad[b as usize].add(&q);
        }
    }
    let mut stamp = vec![0u64; nv];
    let mut epoch = 0u64;
    let target = |q: &Quadric, pa: [f64; 3], pb: [f64; 3]| -> ([f64; 3], f64) {
        let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5];
        let mut best = (mid, q.eval(mid));
        if let Some(o) = q.optimum() {
            // Keep the optimum near the edge to avoid spikes.
            let d2 = |x: [f64; 3], y: [f64; 3]| (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2);
            if d2(o, mid) <= d2(pa, pb) {
                let c = q.eval(o);
                if c < best.1 {
                    best = (o, c);
                }
            }
        }
        for p in [pa, pb] {
            let c = q.eval(p);
            if c < best.1 {
                best = (p, c);
            }
        }
        best
    };
    let mut heap = BinaryHeap::new();
    let mut edges: Vec<(u32, u32)> = ecount.keys().copied().collect();
    edges.sort_unstable();
    for (a, b) in edges {
        let mut q = quad[a as usize];
        q.add(&quad[b as usize]);
        let (_, c) = target(&q, pos[a as usize], pos[b as usize]);
        heap.push(Cand { cost: c, a, b, stamp: 0 });
    }
    let mut live = faces.len();
    let mut collapses = 0;
    let mut vmap: Vec<u32> = (0..nv as u32).collect();
    while live > max_faces {
        let Some(c) = heap.pop() else { break };
        let (a, b) = (c.a, c.b);
        if vmap[a as usize] != a || vmap[b as usize] != b || c.stamp != stamp[a as usize].max(stamp[b as usize]) {
            continue;
        }
        let mut q = quad[a as usize];
        q.add(&quad[b as usize]);
        let (p, _) = target(&q, pos[a as usize], pos[b as usize]);
        // Faces around the edge and the link condition.
        let fa: Vec<u32> = vfaces[a as usize].iter().copied().filter(|&f| alive[f as usize]).collect();
        let fb: Vec<u32> = vfaces[b as usize].iter().copied().filter(|&f| alive[f as usize]).collect();
        let shared: Vec<u32> = fa.iter().copied().filter(|f| fb.contains(f)).collect();
        if shared.is_empty() || shared.len() > 2 {
            continue;
        }
        let neigh = |faces: &[[u32; 3]], fs: &[u32], skip: u32| -> Vec<u32> {
            let mut v: Vec<u32> = fs.iter().flat_map(|&f| faces[f as usize]).filter(|&x| x != skip).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let na = neigh(&faces, &fa, a);
        let nb = neigh(&faces, &fb, b);
        let common = na.iter().filter(|x| nb.contains(x) && **x != a && **x != b).count();
        if common != shared.len() {
            continue;
        }
        // Reject collapses that flip a surviving face.
        let mut flips = false;
        for &f in fa.iter().chain(fb.iter()) {
            if shared.contains(&f) {
                continue;
            }
            let tri = faces[f as usize];
            let (before, l0) = face_plane(&pos, &tri);
            let moved: Vec<[f64; 3]> = tri.iter().map(|&v| if v == a || v == b { p } else { pos[v as usize] }).collect();
            let (after, l1) = face_plane(&moved, &[0, 1, 2]);
            if l1 <= 1e-30 || (l0 > 0.0 && before[0] * after[0] + before[1] * after[1] + before[2] * after[2] <= 0.0) {
                flips = true;
                break;
            }
        }
        if flips {
            continue;
        }
        // Collapse b into a.
        pos[a as usize] = p;
        quad[a as usize] = q;
        vmap[b as usize] = a;
        for &f in &shared {
            alive[f as usize] = false;
            live -= 1;
        }
        for &f in &fb {
            if alive[f as usize] {
                for v in &mut faces[f as usize] {
                    if *v == b {
                        *v = a;
                    }
                }
                vfaces[a as usize].push(f);
            }
        }
        vfaces[b as usize].clear();
        vfaces[a as usize].retain(|&f| alive[f as usize]);
        collapses += 1;
        // The collapse changes the one-ring of every neighbour of `a`, so a
        // collapse rejected earlier around them may now be legal: restamp
        // the neighbourhood and requeue every edge touching it.
        epoch += 1;
        let around = neigh(&faces, &vfaces[a as usize], a);
        stamp[a as usize] = epoch;
        for &x in &around {
            stamp[x as usize] = epoch;
        }
        for &x in std::iter::once(&a).chain(around.iter()) {
            vfaces[x as usize].retain(|&f| alive[f as usize]);
            for y in neigh(&faces, &vfaces[x as usize], x) {
                let (lo, hi) = (x.min(y), x.max(y));
                if x != lo && (y == a || around.binary_search(&y).is_ok()) {
                    // Pushed already from the other end.
                    continue;
                }
                let mut qq = quad[lo as usize];
                qq.add(&quad[hi as usize]);
                let (_, cost) = target(&qq, pos[lo as usize], pos[hi as usize]);
                heap.push(Cand { cost, a: lo, b: hi, stamp: stamp[lo as usize].max(stamp[hi as usize]) });
            }
        }
    }
    mesh.vertices = pos.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    mesh.faces = faces.into_iter().zip(alive).filter_map(|(f, a)| a.then_some(f)).collect();
    mesh.clean();
    collapses
}

/// A mesh cut into charts with one texture coordinate per corner.
#[derive(Debug, Clone)]
pub struct Unwrapped {
    /// Vertex positions (seam vertices duplicated).
    pub vertices: Vec<V3>,
    /// Source vertex of each vertex.
    pub source: Vec<u32>,
    /// Texture coordinates in `[0, 1]`, origin at the atlas's top-left.
    pub uvs: Vec<[f32; 2]>,
    /// Triangles.
    pub faces: Vec<[u32; 3]>,
    /// Chart count.
    pub charts: usize,
}

/// Cut `mesh` into charts of faces whose normals lie within `cone_deg` of
/// their chart's seed face, project each chart onto the plane of its mean
/// normal, and pack the charts into a square atlas of `size` texels with
/// `pad` texels between charts.
#[must_use]
pub fn unwrap(mesh: &Mesh, cone_deg: f32, size: u32, pad: u32) -> Unwrapped {
    let nf = mesh.faces.len();
    let fnorm: Vec<V3> = mesh
        .faces
        .iter()
        .map(|f| {
            let (a, b, c) = (mesh.vertices[f[0] as usize], mesh.vertices[f[1] as usize], mesh.vertices[f[2] as usize]);
            unit(cross(sub(b, a), sub(c, a))).unwrap_or([0.0, 0.0, 1.0])
        })
        .collect();
    // Face adjacency over shared edges.
    let mut edge_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (fi, f) in mesh.faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            edge_faces.entry((a.min(b), a.max(b))).or_default().push(fi as u32);
        }
    }
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); nf];
    let mut ekeys: Vec<_> = edge_faces.keys().copied().collect();
    ekeys.sort_unstable();
    for k in ekeys {
        let fs = &edge_faces[&k];
        if fs.len() == 2 {
            adj[fs[0] as usize].push(fs[1]);
            adj[fs[1] as usize].push(fs[0]);
        }
    }
    let cos_lim = cone_deg.to_radians().cos();
    let mut chart = vec![u32::MAX; nf];
    let mut charts: Vec<Vec<u32>> = Vec::new();
    for seed in 0..nf {
        if chart[seed] != u32::MAX {
            continue;
        }
        let id = charts.len() as u32;
        let axis = fnorm[seed];
        let mut members = vec![seed as u32];
        chart[seed] = id;
        let mut i = 0;
        while i < members.len() {
            let f = members[i] as usize;
            i += 1;
            for &g in &adj[f] {
                if chart[g as usize] == u32::MAX && dot(fnorm[g as usize], axis) >= cos_lim {
                    chart[g as usize] = id;
                    members.push(g);
                }
            }
        }
        charts.push(members);
    }
    // Project charts to 2D.
    struct Chart2 {
        faces: Vec<u32>,
        verts: Vec<u32>,
        uv: Vec<[f32; 2]>,
        w: f32,
        h: f32,
    }
    let mut c2 = Vec::with_capacity(charts.len());
    for members in &charts {
        let mut n = [0f32; 3];
        for &f in members {
            let ff = mesh.faces[f as usize];
            let (a, b, c) = (mesh.vertices[ff[0] as usize], mesh.vertices[ff[1] as usize], mesh.vertices[ff[2] as usize]);
            n = add(n, cross(sub(b, a), sub(c, a)));
        }
        let n = unit(n).unwrap_or(fnorm[members[0] as usize]);
        let t = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = unit(cross(n, t)).unwrap_or([0.0, 1.0, 0.0]);
        let v = cross(n, u);
        let mut verts: Vec<u32> = members.iter().flat_map(|&f| mesh.faces[f as usize]).collect();
        verts.sort_unstable();
        verts.dedup();
        let mut uv: Vec<[f32; 2]> = verts.iter().map(|&i| [dot(mesh.vertices[i as usize], u), dot(mesh.vertices[i as usize], v)]).collect();
        // Align the principal axis with x.
        let m = uv.iter().fold([0f32; 2], |s, p| [s[0] + p[0], s[1] + p[1]]);
        let m = [m[0] / uv.len() as f32, m[1] / uv.len() as f32];
        let (mut sxx, mut sxy, mut syy) = (0f32, 0f32, 0f32);
        for p in &uv {
            let (x, y) = (p[0] - m[0], p[1] - m[1]);
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
        }
        let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let (cs, sn) = (th.cos(), th.sin());
        for p in &mut uv {
            let (x, y) = (p[0] - m[0], p[1] - m[1]);
            *p = [x * cs + y * sn, -x * sn + y * cs];
        }
        let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for p in &uv {
            for a in 0..2 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        for p in &mut uv {
            *p = [p[0] - lo[0], p[1] - lo[1]];
        }
        c2.push(Chart2 { faces: members.clone(), verts, uv, w: hi[0] - lo[0], h: hi[1] - lo[1] });
    }
    // Shelf packing: tallest first; shrink until everything fits.
    let area: f32 = c2.iter().map(|c| c.w.max(1e-9) * c.h.max(1e-9)).sum::<f32>().max(1e-12);
    let s_f = size as f32;
    let p_f = pad as f32;
    let mut order: Vec<usize> = (0..c2.len()).collect();
    order.sort_by(|&i, &j| c2[j].h.total_cmp(&c2[i].h).then(i.cmp(&j)));
    let mut texel = (0.7 * s_f * s_f / area).sqrt();
    let mut place = vec![(0f32, 0f32); c2.len()];
    loop {
        let (mut x, mut y, mut row) = (p_f, p_f, 0f32);
        let mut fits = true;
        for &i in &order {
            let (w, h) = (c2[i].w * texel, c2[i].h * texel);
            if x + w + p_f > s_f {
                x = p_f;
                y += row + p_f;
                row = 0.0;
            }
            if x + w + p_f > s_f || y + h + p_f > s_f {
                fits = false;
                break;
            }
            place[i] = (x, y);
            x += w.ceil() + p_f;
            row = row.max(h.ceil());
        }
        if fits {
            break;
        }
        texel *= 0.9;
    }
    let mut out = Unwrapped { vertices: Vec::new(), source: Vec::new(), uvs: Vec::new(), faces: vec![[0; 3]; nf], charts: c2.len() };
    for (ci, c) in c2.iter().enumerate() {
        let base = out.vertices.len() as u32;
        let (ox, oy) = place[ci];
        let local: HashMap<u32, u32> = c.verts.iter().enumerate().map(|(k, &v)| (v, base + k as u32)).collect();
        for (k, &v) in c.verts.iter().enumerate() {
            out.vertices.push(mesh.vertices[v as usize]);
            out.source.push(v);
            out.uvs.push([(ox + c.uv[k][0] * texel) / s_f, (oy + c.uv[k][1] * texel) / s_f]);
        }
        for &f in &c.faces {
            let ff = mesh.faces[f as usize];
            out.faces[f as usize] = [local[&ff[0]], local[&ff[1]], local[&ff[2]]];
        }
    }
    out
}

/// A sparse voxel attribute volume: `channels` values per occupied voxel of
/// the frame's grid, voxel `c` covering `[c, c + 1)` in grid units.
#[derive(Debug, Clone)]
pub struct AttrVolume<'a> {
    /// Occupied voxels.
    pub coords: &'a [[i32; 3]],
    /// `channels` values per voxel.
    pub values: &'a [f32],
    /// Values per voxel.
    pub channels: usize,
    /// Placement.
    pub frame: GridFrame,
}

impl AttrVolume<'_> {
    fn index(&self) -> HashMap<[i32; 3], usize> {
        self.coords.iter().enumerate().map(|(i, c)| (*c, i)).collect()
    }

    /// Trilinear sample at `p` between voxel centres, over occupied voxels
    /// only (weights renormalised); `None` when no voxel of the 2x2x2
    /// neighbourhood is occupied.
    fn sample(&self, index: &HashMap<[i32; 3], usize>, p: V3, out: &mut [f32]) -> bool {
        let vs = self.frame.size / self.frame.grid as f32;
        let g = [(p[0] - self.frame.lo[0]) / vs - 0.5, (p[1] - self.frame.lo[1]) / vs - 0.5, (p[2] - self.frame.lo[2]) / vs - 0.5];
        let i0 = [g[0].floor() as i32, g[1].floor() as i32, g[2].floor() as i32];
        let f = [g[0] - i0[0] as f32, g[1] - i0[1] as f32, g[2] - i0[2] as f32];
        out.iter_mut().for_each(|x| *x = 0.0);
        let mut wsum = 0f32;
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let w = (if dx == 1 { f[0] } else { 1.0 - f[0] }) * (if dy == 1 { f[1] } else { 1.0 - f[1] }) * (if dz == 1 { f[2] } else { 1.0 - f[2] });
                    if let Some(&vi) = index.get(&[i0[0] + dx, i0[1] + dy, i0[2] + dz]) {
                        for c in 0..self.channels {
                            out[c] += w * self.values[vi * self.channels + c];
                        }
                        wsum += w;
                    }
                }
            }
        }
        if wsum <= 1e-8 {
            return false;
        }
        out.iter_mut().for_each(|x| *x /= wsum);
        true
    }
}

/// Atlas texels: `channels` values per texel plus a coverage mask.
#[derive(Debug, Clone)]
pub struct Atlas {
    /// Side in texels.
    pub size: u32,
    /// Values per texel.
    pub channels: usize,
    /// `size * size * channels` values, top row first.
    pub values: Vec<f32>,
    /// Texels covered by a triangle and sampled.
    pub covered: Vec<bool>,
}

/// Rasterise each triangle of `u` into a `size^2` atlas and sample `vol` at
/// the surface point under every texel centre; uncovered texels are then
/// filled by repeatedly averaging their filled neighbours (`dilate` rings),
/// and anything left takes the mean of the covered texels.
#[must_use]
pub fn bake(u: &Unwrapped, vol: &AttrVolume<'_>, size: u32, dilate: u32) -> Atlas {
    let s = size as usize;
    let ch = vol.channels;
    let index = vol.index();
    let mut values = vec![0f32; s * s * ch];
    let mut covered = vec![false; s * s];
    let mut tmp = vec![0f32; ch];
    let mut misses = Vec::new();
    for f in &u.faces {
        let t: Vec<[f32; 2]> = f.iter().map(|&i| [u.uvs[i as usize][0] * size as f32, u.uvs[i as usize][1] * size as f32]).collect();
        let p: Vec<V3> = f.iter().map(|&i| u.vertices[i as usize]).collect();
        let x0 = t.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let x1 = (t.iter().map(|q| q[0]).fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(s);
        let y0 = t.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let y1 = (t.iter().map(|q| q[1]).fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(s);
        let den = (t[1][1] - t[2][1]) * (t[0][0] - t[2][0]) + (t[2][0] - t[1][0]) * (t[0][1] - t[2][1]);
        if den.abs() < 1e-12 {
            continue;
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let l0 = ((t[1][1] - t[2][1]) * (px - t[2][0]) + (t[2][0] - t[1][0]) * (py - t[2][1])) / den;
                let l1 = ((t[2][1] - t[0][1]) * (px - t[2][0]) + (t[0][0] - t[2][0]) * (py - t[2][1])) / den;
                let l2 = 1.0 - l0 - l1;
                if l0 < -1e-4 || l1 < -1e-4 || l2 < -1e-4 {
                    continue;
                }
                let q = add(add(scale(p[0], l0), scale(p[1], l1)), scale(p[2], l2));
                let k = y * s + x;
                if vol.sample(&index, q, &mut tmp) {
                    values[k * ch..(k + 1) * ch].copy_from_slice(&tmp);
                    covered[k] = true;
                } else {
                    misses.push((k, q));
                }
            }
        }
    }
    // Surface points between voxels with no occupied neighbour take the
    // nearest occupied voxel within two voxels.
    let vs = vol.frame.size / vol.frame.grid as f32;
    for (k, q) in misses {
        if covered[k] {
            continue;
        }
        let g = [((q[0] - vol.frame.lo[0]) / vs).floor() as i32, ((q[1] - vol.frame.lo[1]) / vs).floor() as i32, ((q[2] - vol.frame.lo[2]) / vs).floor() as i32];
        let mut best: Option<(i32, usize)> = None;
        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    if let Some(&vi) = index.get(&[g[0] + dx, g[1] + dy, g[2] + dz]) {
                        let d = dx * dx + dy * dy + dz * dz;
                        if best.is_none_or(|(bd, bi)| d < bd || (d == bd && vi < bi)) {
                            best = Some((d, vi));
                        }
                    }
                }
            }
        }
        if let Some((_, vi)) = best {
            values[k * ch..(k + 1) * ch].copy_from_slice(&vol.values[vi * ch..(vi + 1) * ch]);
            covered[k] = true;
        }
    }
    let mut filled = covered.clone();
    for _ in 0..dilate {
        let prev = filled.clone();
        let snapshot = values.clone();
        let mut any = false;
        for y in 0..s {
            for x in 0..s {
                let k = y * s + x;
                if prev[k] {
                    continue;
                }
                let mut n = 0;
                tmp.iter_mut().for_each(|v| *v = 0.0);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < 0 || yy < 0 || xx >= s as i32 || yy >= s as i32 {
                        continue;
                    }
                    let kk = yy as usize * s + xx as usize;
                    if prev[kk] {
                        for c in 0..ch {
                            tmp[c] += snapshot[kk * ch + c];
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    for c in 0..ch {
                        values[k * ch + c] = tmp[c] / n as f32;
                    }
                    filled[k] = true;
                    any = true;
                }
            }
        }
        if !any {
            break;
        }
    }
    let cov = covered.iter().filter(|&&c| c).count().max(1);
    let mut mean = vec![0f32; ch];
    for k in 0..s * s {
        if covered[k] {
            for c in 0..ch {
                mean[c] += values[k * ch + c];
            }
        }
    }
    mean.iter_mut().for_each(|m| *m /= cov as f32);
    for k in 0..s * s {
        if !filled[k] {
            values[k * ch..(k + 1) * ch].copy_from_slice(&mean);
        }
    }
    Atlas { size, channels: ch, values, covered }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed surface of a solid block: grid points with every
    /// coordinate in `1..=n` are inside, and each voxel's edge (the one its
    /// quad's four voxels surround) is crossed where its two end points
    /// disagree.
    fn cube(n: i32) -> Mesh {
        let mut coords = Vec::new();
        for x in -1..=n {
            for y in -1..=n {
                for z in -1..=n {
                    coords.push([x, y, z]);
                }
            }
        }
        let inside = |p: [i32; 3]| p.iter().all(|v| (1..=n).contains(v));
        let crossed: Vec<[bool; 3]> = coords
            .iter()
            .map(|&c| {
                let mut f = [false; 3];
                for a in 0..3 {
                    let (b, d) = ((a + 1) % 3, (a + 2) % 3);
                    let mut p0 = c;
                    p0[b] += 1;
                    p0[d] += 1;
                    let mut p1 = p0;
                    p1[a] += 1;
                    f[a] = inside(p0) != inside(p1);
                }
                f
            })
            .collect();
        let n_ = coords.len();
        dual_grid_mesh(&coords, &vec![[0.5; 3]; n_], &crossed, &vec![1.0; n_], GridFrame { lo: [-1.0; 3], size: 2.0, grid: 2 * n })
    }

    #[test]
    fn dual_grid_quads_split_into_triangles() {
        let m = cube(3);
        assert!(!m.faces.is_empty());
        assert_eq!(m.faces.len() % 2, 0);
        let s = m.stats();
        assert!(s.area > 0.0);
    }

    #[test]
    fn small_holes_close_and_large_ones_stay() {
        // A square of two triangles has one boundary loop of four edges.
        let mut m = Mesh { vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], faces: vec![[0, 1, 2], [0, 2, 3]] };
        assert_eq!(fill_small_holes(&mut m.clone(), 1.0), 0);
        assert_eq!(fill_small_holes(&mut m, 10.0), 1);
        assert_eq!(m.faces.len(), 6);
    }

    #[test]
    fn simplification_meets_the_budget_and_is_deterministic() {
        let mut a = cube(6);
        a.clean();
        let before = a.faces.len();
        assert_eq!(before, 6 * 6 * 6 * 2);
        assert!(closed_manifold(&a));
        let box_before = bounds(&a);
        let mut b = a.clone();
        simplify(&mut a, before / 4);
        simplify(&mut b, before / 4);
        assert!(a.faces.len() <= before / 4 + 2, "{} faces", a.faces.len());
        assert_eq!(a, b);
        assert!(closed_manifold(&a));
        let box_after = bounds(&a);
        for k in 0..2 {
            for i in 0..3 {
                assert!((box_before[k][i] - box_after[k][i]).abs() < 1e-5, "{box_before:?} vs {box_after:?}");
            }
        }
    }

    #[test]
    fn repair_restores_a_closed_outward_surface() {
        let mut m = cube(3);
        m.clean();
        let faces = m.faces.len();
        let (a, b) = (m.faces[0][0], m.faces[0][1]);
        m.faces[5].swap(1, 2);
        m.faces.push(m.faces[7]);
        // A sliver fin on an existing edge makes it non-manifold; a speck
        // floats apart.
        let mid = scale(add(m.vertices[a as usize], m.vertices[b as usize]), 0.5);
        m.vertices.push(add(mid, [1e-4, 1e-4, 1e-4]));
        m.faces.push([a, b, (m.vertices.len() - 1) as u32]);
        let s = m.vertices.len() as u32;
        m.vertices.extend([[5.0, 5.0, 5.0], [5.001, 5.0, 5.0], [5.0, 5.001, 5.0]]);
        m.faces.push([s, s + 1, s + 2]);
        repair(&mut m, 1e-3);
        assert_eq!(m.faces.len(), faces);
        assert!(closed_manifold(&m));
        let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &m.faces {
            for k in 0..3 {
                *directed.entry((f[k], f[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(directed.iter().all(|(&(x, y), &c)| c == 1 && directed.get(&(y, x)) == Some(&1)), "winding is inconsistent");
        let vol: f32 = m.faces.iter().map(|t| dot(m.vertices[t[0] as usize], cross(m.vertices[t[1] as usize], m.vertices[t[2] as usize]))).sum();
        assert!(vol > 0.0);
    }

    /// Every edge is shared by exactly two faces.
    fn closed_manifold(m: &Mesh) -> bool {
        let mut count: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &m.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        count.values().all(|&c| c == 2)
    }

    fn bounds(m: &Mesh) -> [V3; 2] {
        let mut b = [[f32::MAX; 3], [f32::MIN; 3]];
        for v in &m.vertices {
            for i in 0..3 {
                b[0][i] = b[0][i].min(v[i]);
                b[1][i] = b[1][i].max(v[i]);
            }
        }
        b
    }

    #[test]
    fn unwrap_keeps_uvs_in_the_atlas_and_bake_covers_texels() {
        let mut m = cube(3);
        m.clean();
        let u = unwrap(&m, 60.0, 256, 2);
        assert!(u.uvs.iter().all(|t| (0.0..=1.0).contains(&t[0]) && (0.0..=1.0).contains(&t[1])));
        assert_eq!(u.faces.len(), m.faces.len());
        let coords: Vec<[i32; 3]> = (0..6).flat_map(|x| (0..6).flat_map(move |y| (0..6).map(move |z| [x, y, z]))).collect();
        let values: Vec<f32> = coords.iter().flat_map(|c| [c[0] as f32 / 6.0, 0.5, 0.25]).collect();
        let vol = AttrVolume { coords: &coords, values: &values, channels: 3, frame: GridFrame { lo: [-1.0; 3], size: 2.0, grid: 6 } };
        let at = bake(&u, &vol, 256, 4);
        assert!(at.covered.iter().any(|&c| c));
        assert!(at.values.iter().all(|v| v.is_finite()));
    }
}
