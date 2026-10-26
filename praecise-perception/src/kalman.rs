//! The constant-velocity Kalman filter ByteTrack predicts boxes with.
//!
//! State: box centre `x, y`, aspect ratio `a = w / h`, height `h`, and their
//! velocities per frame. Measurements are `x, y, a, h`. Noise scales with the
//! box height, so a near, large box is allowed to move more pixels than a far
//! one.

/// State vector.
pub type Mean = [f64; 8];
/// State covariance.
pub type Cov = [[f64; 8]; 8];

const STD_POSITION: f64 = 1.0 / 20.0;
const STD_VELOCITY: f64 = 1.0 / 160.0;

fn diag8(v: [f64; 8]) -> Cov {
    let mut m = [[0.0; 8]; 8];
    for i in 0..8 {
        m[i][i] = v[i] * v[i];
    }
    m
}

/// Start a track at measurement `z = [x, y, a, h]`.
#[must_use]
pub fn initiate(z: [f64; 4]) -> (Mean, Cov) {
    let h = z[3];
    let mean = [z[0], z[1], z[2], z[3], 0.0, 0.0, 0.0, 0.0];
    let std = [
        2.0 * STD_POSITION * h,
        2.0 * STD_POSITION * h,
        1e-2,
        2.0 * STD_POSITION * h,
        10.0 * STD_VELOCITY * h,
        10.0 * STD_VELOCITY * h,
        1e-5,
        10.0 * STD_VELOCITY * h,
    ];
    (mean, diag8(std))
}

/// Advance one frame.
pub fn predict(mean: &mut Mean, cov: &mut Cov) {
    let h = mean[3];
    let q = diag8([
        STD_POSITION * h,
        STD_POSITION * h,
        1e-2,
        STD_POSITION * h,
        STD_VELOCITY * h,
        STD_VELOCITY * h,
        1e-5,
        STD_VELOCITY * h,
    ]);
    for i in 0..4 {
        mean[i] += mean[i + 4];
    }
    // F P F^T with F = [[I, I], [0, I]]
    let mut fp = *cov;
    for i in 0..4 {
        for j in 0..8 {
            fp[i][j] += cov[i + 4][j];
        }
    }
    let mut out = fp;
    for i in 0..8 {
        for j in 0..4 {
            out[i][j] += fp[i][j + 4];
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] += q[i][j];
        }
    }
    *cov = out;
}

fn invert4(m: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut a = m;
    let mut inv = [[0.0; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let p = a[col][col];
        for j in 0..4 {
            a[col][j] /= p;
            inv[col][j] /= p;
        }
        for r in 0..4 {
            if r != col {
                let f = a[r][col];
                for j in 0..4 {
                    a[r][j] -= f * a[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

/// Correct the state with measurement `z = [x, y, a, h]`.
pub fn update(mean: &mut Mean, cov: &mut Cov, z: [f64; 4]) {
    let h = mean[3];
    let r = [STD_POSITION * h, STD_POSITION * h, 1e-1, STD_POSITION * h];
    // S = H P H^T + R, the top-left 4x4 block plus R
    let mut s = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            s[i][j] = cov[i][j];
        }
        s[i][i] += r[i] * r[i];
    }
    let Some(s_inv) = invert4(s) else { return };
    // K = P H^T S^-1 (8x4); P H^T is the first four columns of P
    let mut k = [[0.0; 4]; 8];
    for i in 0..8 {
        for j in 0..4 {
            k[i][j] = (0..4).map(|l| cov[i][l] * s_inv[l][j]).sum();
        }
    }
    let innovation = [z[0] - mean[0], z[1] - mean[1], z[2] - mean[2], z[3] - mean[3]];
    for i in 0..8 {
        mean[i] += (0..4).map(|j| k[i][j] * innovation[j]).sum::<f64>();
    }
    // P - K S K^T, which equals P - K (H P)
    let mut out = *cov;
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] -= (0..4).map(|l| k[i][l] * cov[l][j]).sum::<f64>();
        }
    }
    *cov = out;
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn a_box_moving_steadily_is_predicted_ahead() {
        let (mut m, mut p) = initiate([100.0, 100.0, 0.5, 80.0]);
        for t in 1..=30 {
            predict(&mut m, &mut p);
            update(&mut m, &mut p, [100.0 + 4.0 * f64::from(t), 100.0, 0.5, 80.0]);
        }
        assert!((m[4] - 4.0).abs() < 0.2, "velocity {}", m[4]);
        predict(&mut m, &mut p);
        assert!((m[0] - 224.0).abs() < 1.0, "predicted x {}", m[0]);
        for i in 0..8 {
            assert!(p[i][i] > 0.0);
        }
    }

    #[test]
    fn inversion_round_trips() {
        let m = [[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.5, 0.0], [0.0, 0.5, 2.0, 0.1], [0.0, 0.0, 0.1, 1.0]];
        let inv = invert4(m).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let v: f64 = (0..4).map(|l| m[i][l] * inv[l][j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
        assert!(invert4([[0.0; 4]; 4]).is_none());
    }
}
