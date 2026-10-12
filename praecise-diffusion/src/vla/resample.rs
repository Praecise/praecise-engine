//! 8-bit image resampling as the policies' reference image pipelines
//! compute it, to the bit: OpenCV's `INTER_AREA` and torch's antialiased
//! bicubic resize of `uint8` images (the torchvision path on CPU).
//!
//! Both work in integers or `float32` exactly as the libraries do, so the
//! pixels a policy sees match the reference. OpenCV's area decimation sums
//! in `float32` with fused multiply-adds on aarch64 builds and separate
//! ones elsewhere; the build target picks the same here.

use crate::flux3::packing::Frame;

/// An 8-bit RGB frame, channel-major `[3][h][w]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame8 {
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Pixels `[3][h][w]`.
    pub data: Vec<u8>,
}

impl Frame8 {
    /// A frame in `[0, 1]` scaled to 8 bits by truncation, as
    /// `(x.clamp(0, 1) * 255).to(uint8)` does.
    #[must_use]
    pub fn from_unit(f: &Frame) -> Self {
        Self { h: f.h, w: f.w, data: f.data.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect() }
    }

    /// The `h x w` window at `(top, left)`.
    #[must_use]
    pub fn crop(&self, top: usize, left: usize, h: usize, w: usize) -> Self {
        let mut data = Vec::with_capacity(3 * h * w);
        for c in 0..3 {
            for y in top..top + h {
                let row = (c * self.h + y) * self.w;
                data.extend_from_slice(&self.data[row + left..row + left + w]);
            }
        }
        Self { h, w, data }
    }

    /// Padded with zeros to `h x w`, the frame at `(top, left)`.
    #[must_use]
    pub fn pad(&self, top: usize, left: usize, h: usize, w: usize) -> Self {
        let mut data = vec![0u8; 3 * h * w];
        for c in 0..3 {
            for y in 0..self.h {
                let dst = (c * h + top + y) * w + left;
                let src = (c * self.h + y) * self.w;
                data[dst..dst + self.w].copy_from_slice(&self.data[src..src + self.w]);
            }
        }
        Self { h, w, data }
    }

    fn plane(&self, c: usize) -> &[u8] {
        &self.data[c * self.h * self.w..(c + 1) * self.h * self.w]
    }
}

#[cfg(target_arch = "aarch64")]
fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(target_arch = "aarch64"))]
fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

/// Source index and weight of every contribution to each destination
/// index of a box-filter decimation (`computeResizeAreaTab`).
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<(usize, usize, f32)> {
    let mut tab = Vec::with_capacity(2 * ssize);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let sx2 = (fsx2.floor() as i64).min(ssize as i64 - 1);
        let sx1 = (fsx1.ceil() as i64).min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push((dx, (sx1 - 1) as usize, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for sx in sx1..sx2 {
            tab.push((dx, sx as usize, (1.0 / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push((dx, sx2 as usize, ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32));
        }
    }
    tab
}

fn saturate_round(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// Decimation by box filter (both scales at least one, not both integers).
fn area_down(f: &Frame8, h: usize, w: usize, scale_x: f64, scale_y: f64) -> Frame8 {
    let xtab = area_tab(f.w, w, scale_x);
    let ytab = area_tab(f.h, h, scale_y);
    let mut data = vec![0u8; 3 * h * w];
    for c in 0..3 {
        let src = f.plane(c);
        let rows: Vec<Vec<f32>> = (0..f.h)
            .map(|sy| {
                let s = &src[sy * f.w..(sy + 1) * f.w];
                let mut r = vec![0f32; w];
                for &(dx, sx, a) in &xtab {
                    r[dx] = mul_add(f32::from(s[sx]), a, r[dx]);
                }
                r
            })
            .collect();
        let mut sum = vec![0f32; w];
        let mut prev = None;
        let out = &mut data[c * h * w..(c + 1) * h * w];
        for &(dy, sy, b) in &ytab {
            if prev == Some(dy) {
                for (s, r) in sum.iter_mut().zip(&rows[sy]) {
                    *s = mul_add(b, *r, *s);
                }
            } else {
                if let Some(p) = prev {
                    for (o, s) in out[p * w..(p + 1) * w].iter_mut().zip(&sum) {
                        *o = saturate_round(*s);
                    }
                }
                for (s, r) in sum.iter_mut().zip(&rows[sy]) {
                    *s = b * r;
                }
                prev = Some(dy);
            }
        }
        if let Some(p) = prev {
            for (o, s) in out[p * w..(p + 1) * w].iter_mut().zip(&sum) {
                *o = saturate_round(*s);
            }
        }
    }
    Frame8 { h, w, data }
}

/// Decimation by whole factors: the mean of each block.
fn area_fast(f: &Frame8, sx: usize, sy: usize) -> Frame8 {
    let (h, w) = (f.h / sy, f.w / sx);
    let scale = 1.0f32 / (sx * sy) as f32;
    let mut data = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        let src = f.plane(c);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0i32;
                for dy in 0..sy {
                    for dx in 0..sx {
                        sum += i32::from(src[(y * sy + dy) * f.w + x * sx + dx]);
                    }
                }
                data.push(if sx == 2 && sy == 2 { ((sum + 2) >> 2) as u8 } else { saturate_round(sum as f32 * scale) });
            }
        }
    }
    Frame8 { h, w, data }
}

/// Fixed-point linear taps of each destination index when area
/// interpolation enlarges: two source indices and weights out of 2048.
fn area_linear_taps(ssize: usize, dsize: usize) -> Vec<(usize, usize, i32, i32)> {
    let inv = dsize as f64 / ssize as f64;
    let scale = 1.0 / inv;
    (0..dsize)
        .map(|d| {
            let mut s = (d as f64 * scale).floor() as i64;
            let mut fx = ((d + 1) as f64 - (s + 1) as f64 * inv) as f32;
            fx = if fx <= 0.0 { 0.0 } else { fx - fx.floor() };
            if s < 0 {
                fx = 0.0;
                s = 0;
            }
            if s >= ssize as i64 - 1 {
                fx = 0.0;
                s = ssize as i64 - 1;
            }
            let s = s as usize;
            let a0 = ((1.0 - fx) * 2048.0).round_ties_even() as i32;
            let a1 = (fx * 2048.0).round_ties_even() as i32;
            (s, (s + 1).min(ssize - 1), a0, a1)
        })
        .collect()
}

/// Area interpolation that enlarges an axis: linear interpolation with area
/// weights, horizontal pass in integers, vertical pass as OpenCV's vector
/// kernel computes it (shifted to 16 bits, high halves of the products,
/// rounded down by four).
fn area_up(f: &Frame8, h: usize, w: usize) -> Frame8 {
    let xt = area_linear_taps(f.w, w);
    let yt = area_linear_taps(f.h, h);
    let mut data = vec![0u8; 3 * h * w];
    for c in 0..3 {
        let src = f.plane(c);
        let rows: Vec<Vec<i32>> = (0..f.h)
            .map(|sy| {
                let s = &src[sy * f.w..(sy + 1) * f.w];
                xt.iter().map(|&(s0, s1, a0, a1)| i32::from(s[s0]) * a0 + i32::from(s[s1]) * a1).collect()
            })
            .collect();
        for (dy, &(s0, s1, b0, b1)) in yt.iter().enumerate() {
            let (r0, r1) = (&rows[s0], &rows[s1]);
            let out = &mut data[(c * h + dy) * w..(c * h + dy + 1) * w];
            for x in 0..w {
                let x0 = (r0[x] >> 4).clamp(-32768, 32767);
                let x1 = (r1[x] >> 4).clamp(-32768, 32767);
                let v = ((x0 * b0) >> 16) + ((x1 * b1) >> 16);
                out[x] = ((v + 2) >> 2).clamp(0, 255) as u8;
            }
        }
    }
    Frame8 { h, w, data }
}

/// Resize to `h x w` as `cv2.resize(.., interpolation=cv2.INTER_AREA)`
/// does for 8-bit images.
#[must_use]
pub fn resize_area(f: &Frame8, h: usize, w: usize) -> Frame8 {
    if f.h == h && f.w == w {
        return f.clone();
    }
    let (scale_x, scale_y) = (1.0 / (w as f64 / f.w as f64), 1.0 / (h as f64 / f.h as f64));
    if scale_x >= 1.0 && scale_y >= 1.0 {
        let (ix, iy) = (scale_x.round(), scale_y.round());
        if (scale_x - ix).abs() < f64::EPSILON && (scale_y - iy).abs() < f64::EPSILON {
            return area_fast(f, ix as usize, iy as usize);
        }
        return area_down(f, h, w, scale_x, scale_y);
    }
    area_up(f, h, w)
}

/// The antialiased bicubic kernel (`a = -0.5`).
fn cubic_aa(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// First source index and 16-bit weights of every destination index of an
/// antialiased bicubic pass, with the weights' fixed-point precision.
fn aa_taps(insize: usize, outsize: usize) -> (Vec<(usize, Vec<i32>)>, u32) {
    let scale = insize as f64 / outsize as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_taps = support.ceil() as usize * 2 + 1;
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let rows: Vec<(usize, Vec<f64>)> = (0..outsize)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(insize as i64) as usize;
            let n = xmax.saturating_sub(xmin).min(max_taps);
            let mut w: Vec<f64> = (0..n).map(|j| cubic_aa(((j + xmin) as f64 - center + 0.5) * inv)).collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (xmin, w)
        })
        .collect();
    let wmax = rows.iter().flat_map(|(_, w)| w.iter().copied()).fold(0.0f64, f64::max);
    let mut precision = 0u32;
    while precision < 22 && ((0.5 + wmax * f64::from(1u32 << (precision + 1))) as i64) < (1 << 15) {
        precision += 1;
    }
    let one = f64::from(1u32 << precision);
    let taps = rows
        .into_iter()
        .map(|(xmin, w)| (xmin, w.iter().map(|&v| if v < 0.0 { (-0.5 + v * one) as i32 } else { (0.5 + v * one) as i32 }).collect()))
        .collect();
    (taps, precision)
}

/// One antialiased bicubic pass along the rows (`horizontal`) or columns.
fn aa_pass(f: &Frame8, out: usize, horizontal: bool) -> Frame8 {
    let (taps, precision) = aa_taps(if horizontal { f.w } else { f.h }, out);
    let (h, w) = if horizontal { (f.h, out) } else { (out, f.w) };
    let mut data = vec![0u8; 3 * h * w];
    for c in 0..3 {
        let src = f.plane(c);
        let dst = &mut data[c * h * w..(c + 1) * h * w];
        for y in 0..h {
            for x in 0..w {
                let (i, stride, base) = if horizontal { (x, 1, y * f.w) } else { (y, f.w, x) };
                let (xmin, wt) = &taps[i];
                let mut acc = 1i32 << (precision - 1);
                for (j, &k) in wt.iter().enumerate() {
                    acc += i32::from(src[base + (xmin + j) * stride]) * k;
                }
                dst[y * w + x] = (acc >> precision).clamp(0, 255) as u8;
            }
        }
    }
    Frame8 { h, w, data }
}

/// Resize to `h x w` with the antialiased bicubic filter as torch computes
/// it on 8-bit images (`resize(.., BICUBIC, antialias=True)` on CPU): the
/// width first, then the height, each rounded to 8 bits.
#[must_use]
pub fn resize_bicubic_aa(f: &Frame8, h: usize, w: usize) -> Frame8 {
    let mut r = f.clone();
    if r.w != w {
        r = aa_pass(&r, w, true);
    }
    if r.h != h {
        r = aa_pass(&r, h, false);
    }
    r
}
