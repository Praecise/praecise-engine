//! Camera conditioning for the action world model: the camera path is
//! integrated from the keyboard and mouse actions, turned into
//! camera-to-world matrices, resampled at the latent frames and encoded as
//! per-pixel Plucker rays folded onto the latent grid.

/// Units moved per frame along one axis.
pub const MOVE_STEP: f64 = 12.35;
/// Units moved per frame along each axis when moving diagonally.
pub const DIAGONAL_STEP: f64 = 8.73;
/// Degrees of pitch per unit of the first mouse value.
pub const PITCH_PER_UNIT: f64 = 15.0;
/// Degrees of yaw per unit of the second mouse value.
pub const YAW_PER_UNIT: f64 = 15.0;
/// Mouse values below this magnitude are ignored.
pub const MOUSE_DEADZONE: f32 = 0.02;
/// Horizontal and vertical field of view in degrees.
pub const FOV_DEG: f64 = 90.0;

/// Camera pose `[x, y, z, pitch, yaw]` (degrees), one per pixel frame.
pub type Pose = [f32; 5];

type Mat4 = [[f64; 4]; 4];
type Mat3 = [[f64; 3]; 3];

/// The pose after one frame under `keyboard` (forward, back, left, right
/// first) and `mouse` (pitch, yaw first). Movement uses the mean of the old
/// and new yaw so paths are symmetric.
#[must_use]
pub fn next_pose(p: &Pose, keyboard: &[f32], mouse: &[f32]) -> Pose {
    let (w, s, a, d) = (keyboard[0], keyboard[1], keyboard[2], keyboard[3]);
    let (mx, my) = (mouse[0], mouse[1]);
    let dp = if mx.abs() >= MOUSE_DEADZONE { f64::from(mx * PITCH_PER_UNIT as f32) } else { 0.0 };
    let dy = if my.abs() >= MOUSE_DEADZONE { f64::from(my * YAW_PER_UNIT as f32) } else { 0.0 };
    let (yaw, pitch) = (f64::from(p[4]), f64::from(p[3]));
    let new_pitch = pitch + dp;
    let mut new_yaw = yaw + dy;
    while new_yaw > 180.0 {
        new_yaw -= 360.0;
    }
    while new_yaw < -180.0 {
        new_yaw += 360.0;
    }
    let mut fwd = if w > 0.5 && s < 0.5 {
        MOVE_STEP
    } else if s > 0.5 && w < 0.5 {
        -MOVE_STEP
    } else {
        0.0
    };
    let mut right = if d > 0.5 && a < 0.5 {
        MOVE_STEP
    } else if a > 0.5 && d < 0.5 {
        -MOVE_STEP
    } else {
        0.0
    };
    if fwd.abs() > 0.1 && right.abs() > 0.1 {
        fwd = fwd.signum() * DIAGONAL_STEP;
        right = right.signum() * DIAGONAL_STEP;
    }
    let r = ((yaw + new_yaw) / 2.0).to_radians();
    let (c, sn) = (r.cos(), r.sin());
    #[allow(clippy::cast_possible_truncation)]
    [
        (f64::from(p[0]) + c * fwd - sn * right) as f32,
        (f64::from(p[1]) + sn * fwd + c * right) as f32,
        p[2],
        new_pitch as f32,
        new_yaw as f32,
    ]
}

/// Poses for `frames` pixel frames: `first` for frame 0, then each frame's
/// pose follows from the previous frame's actions. `keyboard` rows are
/// `kdim` wide, `mouse` rows 2 wide. Also returns the pose after the last
/// frame's actions, the next chunk's first pose.
#[must_use]
pub fn poses(first: Pose, keyboard: &[f32], kdim: usize, mouse: &[f32], frames: usize) -> (Vec<Pose>, Pose) {
    let mut out = Vec::with_capacity(frames);
    let mut p = first;
    for i in 0..frames {
        out.push(p);
        p = next_pose(&p, &keyboard[i * kdim..(i + 1) * kdim], &mouse[i * 2..i * 2 + 2]);
    }
    (out, p)
}

/// Camera-to-world matrix of a pose: yaw about z, pitch about y (no roll),
/// the camera axes mapped onto the world axes, translation scaled by 0.01.
#[must_use]
pub fn extrinsic(p: &Pose) -> Mat4 {
    let (pitch, yaw) = (f64::from(p[3]).to_radians(), f64::from(p[4]).to_radians());
    let rz = [[yaw.cos(), -yaw.sin(), 0.0], [yaw.sin(), yaw.cos(), 0.0], [0.0, 0.0, 1.0]];
    let ry = [[pitch.cos(), 0.0, pitch.sin()], [0.0, 1.0, 0.0], [-pitch.sin(), 0.0, pitch.cos()]];
    let init = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
    let r = mul3(&mul3(&rz, &ry), &init);
    let mut m = [[0.0; 4]; 4];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&r[i]);
        m[i][3] = f64::from(p[i]) * 0.01;
    }
    m[3][3] = 1.0;
    m
}

fn mul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn mul4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut o = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn rot(m: &Mat4) -> Mat3 {
    [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]
}

fn det3(r: &Mat3) -> f64 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

/// Unit quaternion `[x, y, z, w]` of a proper rotation.
fn quat(r: &Mat3) -> [f64; 4] {
    let t = r[0][0] + r[1][1] + r[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [(r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [(r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s, (r[1][0] - r[0][1]) / s]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / n)
}

fn quat_mat(q: &[f64; 4]) -> Mat3 {
    let [x, y, z, w] = *q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Shortest-path spherical interpolation.
fn slerp(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let mut d: f64 = (0..4).map(|i| a[i] * b[i]).sum();
    let mut b = *b;
    if d < 0.0 {
        d = -d;
        b = b.map(|v| -v);
    }
    let th = d.min(1.0).acos();
    let (wa, wb) = if th < 1e-12 { (1.0 - t, t) } else { (((1.0 - t) * th).sin() / th.sin(), (t * th).sin() / th.sin()) };
    let q = [0, 1, 2, 3].map(|i| wa * a[i] + wb * b[i]);
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / n)
}

/// Resample camera-to-world matrices given at `src` (increasing) onto
/// `tgt`: translation linearly (extrapolated past the ends), rotation by
/// spherical interpolation. Reflections (negative determinant, as the axis
/// mapping here gives) are interpolated in the mirrored proper frame.
#[must_use]
pub fn interpolate(c2ws: &[Mat4], src: &[f64], tgt: &[f64]) -> Vec<Mat4> {
    let mut dets: Vec<f64> = c2ws.iter().map(|m| det3(&rot(m))).collect();
    dets.sort_by(f64::total_cmp);
    let n = dets.len();
    let median = if n % 2 == 1 { dets[n / 2] } else { (dets[n / 2 - 1] + dets[n / 2]) / 2.0 };
    let flip = n > 0 && median < 0.0;
    let fl = |r: Mat3| if flip { r.map(|row| [row[0], row[1], -row[2]]) } else { r };
    let quats: Vec<[f64; 4]> = c2ws.iter().map(|m| quat(&fl(rot(m)))).collect();
    tgt.iter()
        .map(|&t| {
            let seg = if n < 2 { 0 } else { src.windows(2).position(|w| t <= w[1]).unwrap_or(n - 2) };
            let (i0, i1) = if n < 2 { (0, 0) } else { (seg, seg + 1) };
            let span = src[i1] - src[i0];
            let u = if span == 0.0 { 0.0 } else { (t - src[i0]) / span };
            let r = fl(quat_mat(&slerp(&quats[i0], &quats[i1], u.clamp(0.0, 1.0))));
            let mut m = [[0.0; 4]; 4];
            for i in 0..3 {
                m[i][..3].copy_from_slice(&r[i]);
                m[i][3] = c2ws[i0][i][3] + u * (c2ws[i1][i][3] - c2ws[i0][i][3]);
            }
            m[3][3] = 1.0;
            m
        })
        .collect()
}

fn inverse(m: &Mat4) -> Mat4 {
    let mut o = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = m[j][i];
        }
        o[i][3] = -(0..3).map(|k| m[k][i] * m[k][3]).sum::<f64>();
    }
    o[3][3] = 1.0;
    o
}

/// Poses relative to the first (which becomes the identity); with
/// `framewise`, each pose after the first relative to the one before.
/// Translations are scaled so the longest is 1.
#[must_use]
pub fn relative(c2ws: &[Mat4], framewise: bool) -> Vec<Mat4> {
    let w2c = inverse(&c2ws[0]);
    let mut rel: Vec<Mat4> = c2ws.iter().map(|m| mul4(&w2c, m)).collect();
    let mut eye = [[0.0; 4]; 4];
    for (i, row) in eye.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    rel[0] = eye;
    if framewise {
        let prev = rel.clone();
        for i in 1..rel.len() {
            rel[i] = mul4(&inverse(&prev[i - 1]), &prev[i]);
        }
    }
    let max = rel.iter().map(|m| (m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3]).sqrt()).fold(0.0, f64::max);
    if max > 0.0 {
        for m in &mut rel {
            for row in m.iter_mut().take(3) {
                row[3] /= max;
            }
        }
    }
    rel
}

/// Plucker rays `[6 * s * s][frames][lat_h][lat_w]` for `c2ws` at pixel
/// size `(lat_h * s, lat_w * s)`: each latent cell carries the origin and
/// unit direction of the rays through its `s x s` pixels.
#[must_use]
pub fn plucker(c2ws: &[Mat4], lat_h: usize, lat_w: usize, s: usize) -> Vec<f32> {
    let (h, w) = (lat_h * s, lat_w * s);
    let f = c2ws.len();
    let half = (FOV_DEG.to_radians() / 2.0).tan();
    #[allow(clippy::cast_precision_loss)]
    let (fx, fy, cx, cy) = (w as f64 / (2.0 * half), h as f64 / (2.0 * half), w as f64 / 2.0, h as f64 / 2.0);
    let c = 6 * s * s;
    let mut out = vec![0f32; c * f * lat_h * lat_w];
    for (fi, m) in c2ws.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                #[allow(clippy::cast_precision_loss)]
                let (xs, ys) = ((x as f64 + 0.5 - cx) / fx, (y as f64 + 0.5 - cy) / fy);
                let n = (xs * xs + ys * ys + 1.0).sqrt();
                let dir = [xs / n, ys / n, 1.0 / n];
                let mut ray = [m[0][3], m[1][3], m[2][3], 0.0, 0.0, 0.0];
                for k in 0..3 {
                    ray[3 + k] = (0..3).map(|j| dir[j] * m[k][j]).sum();
                }
                let (lh, i1, lw, i2) = (y / s, y % s, x / s, x % s);
                for (ch, v) in ray.iter().enumerate() {
                    let cc = (ch * s + i1) * s + i2;
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[((cc * f + fi) * lat_h + lh) * lat_w + lw] = *v as f32;
                    }
                }
            }
        }
    }
    out
}

/// Evenly spaced points from `a` to `b` inclusive.
#[must_use]
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    (0..n).map(|i| if n == 1 { a } else { a + (b - a) * i as f64 / (n - 1) as f64 }).collect()
}

/// Rays for the latent frames of a clip spanning pixel frames
/// `start..end` of `path`: the clip's poses resampled at `latents` evenly
/// spaced points from `tgt_first` to `end - 1`, made relative frame to
/// frame.
#[must_use]
pub fn clip_rays(path: &[Mat4], start: usize, end: usize, tgt_first: usize, latents: usize, lat: (usize, usize), s: usize) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    let src = linspace(start as f64, (end - 1) as f64, end - start);
    #[allow(clippy::cast_precision_loss)]
    let tgt = linspace(tgt_first as f64, (end - 1) as f64, latents);
    let c2ws = relative(&interpolate(&path[start..end], &src, &tgt), true);
    plucker(&c2ws, lat.0, lat.1, s)
}

/// Rays for one memory latent frame: the pose at the end of its four-frame
/// block, relative to the pose of pixel frame `reference`.
#[must_use]
pub fn memory_rays(path: &[Mat4], memory_px: usize, reference: usize, lat: (usize, usize), s: usize) -> Vec<f32> {
    let a = if memory_px > 0 { (memory_px - 1) / 4 * 4 + 1 } else { 1 };
    #[allow(clippy::cast_precision_loss)]
    let src = linspace(a as f64, (a + 3) as f64, 4);
    #[allow(clippy::cast_precision_loss)]
    let pose = interpolate(&path[a..a + 4], &src, &[(a + 3) as f64]);
    let rel = relative(&[path[reference], pose[0]], false);
    plucker(&rel[1..], lat.0, lat.1, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_moves_along_yaw_and_mouse_turns() {
        let p = next_pose(&[0.0; 5], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0, 0.0]);
        assert!((p[0] - 12.35).abs() < 1e-5 && p[1].abs() < 1e-6);
        let q = next_pose(&[0.0; 5], &[0.0; 6], &[0.0, 0.1]);
        assert!((q[4] - 1.5).abs() < 1e-5 && q[0] == 0.0);
        let dead = next_pose(&[0.0; 5], &[0.0; 6], &[0.01, -0.01]);
        assert_eq!(dead, [0.0; 5]);
        let diag = next_pose(&[0.0; 5], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[0.0, 0.0]);
        assert!((diag[0] - 8.73).abs() < 1e-5 && (diag[1] - 8.73).abs() < 1e-5);
    }

    #[test]
    fn interpolation_hits_its_sources() {
        let path: Vec<Mat4> = (0..4).map(|i| extrinsic(&[i as f32, 0.0, 0.0, 0.0, 10.0 * i as f32])).collect();
        let src = linspace(0.0, 3.0, 4);
        let back = interpolate(&path, &src, &src);
        for (a, b) in path.iter().zip(&back) {
            for i in 0..4 {
                for j in 0..4 {
                    assert!((a[i][j] - b[i][j]).abs() < 1e-9);
                }
            }
        }
    }
}
