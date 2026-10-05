//! RGB and 4:2:0 YCbCr conversion.

/// The YCbCr matrix and range of a picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix {
    /// Red luma weight, in ten-thousandths.
    kr: u32,
    /// Blue luma weight, in ten-thousandths.
    kb: u32,
    /// Full-range samples (0..=255) rather than studio range (16..=235).
    pub full: bool,
}

impl Matrix {
    /// BT.709, studio range: what this crate writes.
    pub const BT709: Self = Self { kr: 2126, kb: 722, full: false };
    /// BT.601, studio range.
    pub const BT601: Self = Self { kr: 2990, kb: 1140, full: false };

    /// The matrix of an ISO/IEC 23091-2 `matrix_coefficients` code and a
    /// full-range flag; `None` for codes this crate does not convert
    /// (identity, YCgCo, constant-luminance).
    #[must_use]
    pub fn from_code(code: u32, full: bool) -> Option<Self> {
        let m = match code {
            // Unspecified: the BT.709 default every current player uses.
            1 | 2 => Self::BT709,
            5 | 6 => Self::BT601,
            9 => Self { kr: 2627, kb: 593, full: false },
            _ => return None,
        };
        Some(Self { full, ..m })
    }

    fn weights(self) -> (f32, f32, f32) {
        let kr = self.kr as f32 / 10000.0;
        let kb = self.kb as f32 / 10000.0;
        (kr, 1.0 - kr - kb, kb)
    }

    /// `(offset, luma scale, chroma scale)` from 0..=255 component units.
    fn range(self) -> (f32, f32, f32) {
        if self.full { (0.0, 1.0, 1.0) } else { (16.0, 219.0 / 255.0, 224.0 / 255.0) }
    }
}

fn clamp8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Planar 8-bit 4:2:0 pictures: luma `w * h`, then both chroma planes of
/// `ceil(w/2) * ceil(h/2)`.
#[derive(Debug, Clone)]
pub struct Yuv420 {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Luma.
    pub y: Vec<u8>,
    /// Blue difference.
    pub u: Vec<u8>,
    /// Red difference.
    pub v: Vec<u8>,
}

impl Yuv420 {
    /// Chroma plane size.
    #[must_use]
    pub fn chroma_dims(width: usize, height: usize) -> (usize, usize) {
        (width.div_ceil(2), height.div_ceil(2))
    }

    /// Convert one row-major 8-bit RGB picture; each chroma sample is the
    /// mean of its (up to) four pixels.
    #[must_use]
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize, m: Matrix) -> Self {
        let (kr, kg, kb) = m.weights();
        let (off, ys, cs) = m.range();
        let (cw, ch) = Self::chroma_dims(width, height);
        let mut y = vec![0u8; width * height];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        let mut cb_acc = vec![0f32; cw * ch];
        let mut cr_acc = vec![0f32; cw * ch];
        let mut n = vec![0u8; cw * ch];
        for row in 0..height {
            for col in 0..width {
                let p = &rgb[(row * width + col) * 3..][..3];
                let (r, g, b) = (f32::from(p[0]), f32::from(p[1]), f32::from(p[2]));
                let luma = kr * r + kg * g + kb * b;
                y[row * width + col] = clamp8(off + ys * luma);
                let c = (row / 2) * cw + col / 2;
                cb_acc[c] += (b - luma) / (2.0 * (1.0 - kb));
                cr_acc[c] += (r - luma) / (2.0 * (1.0 - kr));
                n[c] += 1;
            }
        }
        for c in 0..cw * ch {
            let k = f32::from(n[c].max(1));
            u[c] = clamp8(128.0 + cs * cb_acc[c] / k);
            v[c] = clamp8(128.0 + cs * cr_acc[c] / k);
        }
        Self { width, height, y, u, v }
    }

    /// Append the picture as row-major 8-bit RGB to `out`; each chroma
    /// sample covers its 2x2 block.
    pub fn write_rgb(&self, m: Matrix, out: &mut Vec<u8>) {
        write_rgb(Planes { y: &self.y, u: &self.u, v: &self.v, strides: (self.width, self.width.div_ceil(2)) }, self.width, self.height, m, out);
    }
}

/// Borrowed 8-bit 4:2:0 planes with `(luma, chroma)` row strides.
#[derive(Debug, Clone, Copy)]
pub struct Planes<'a> {
    /// Luma.
    pub y: &'a [u8],
    /// Blue difference.
    pub u: &'a [u8],
    /// Red difference.
    pub v: &'a [u8],
    /// Row strides of luma and chroma.
    pub strides: (usize, usize),
}

/// Append a 4:2:0 picture as row-major 8-bit RGB to `out`.
pub fn write_rgb(p: Planes<'_>, width: usize, height: usize, m: Matrix, out: &mut Vec<u8>) {
    let (kr, kg, kb) = m.weights();
    let (off, ys, cs) = m.range();
    let (ry, rc) = (1.0 / ys, 1.0 / cs);
    let (cr_r, cb_b) = (2.0 * (1.0 - kr), 2.0 * (1.0 - kb));
    let (cb_g, cr_g) = (cb_b * kb / kg, cr_r * kr / kg);
    out.reserve(width * height * 3);
    for row in 0..height {
        for col in 0..width {
            let luma = (f32::from(p.y[row * p.strides.0 + col]) - off) * ry;
            let c = (row / 2) * p.strides.1 + col / 2;
            let cb = (f32::from(p.u[c]) - 128.0) * rc;
            let cr = (f32::from(p.v[c]) - 128.0) * rc;
            out.push(clamp8(luma + cr_r * cr));
            out.push(clamp8(luma - cb_g * cb - cr_g * cr));
            out.push(clamp8(luma + cb_b * cb));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_colours_survive_a_round_trip() {
        for px in [[0u8, 0, 0], [255, 255, 255], [200, 30, 90], [12, 180, 240]] {
            let rgb: Vec<u8> = px.iter().copied().cycle().take(4 * 4 * 3).collect();
            for m in [Matrix::BT709, Matrix::BT601, Matrix { full: true, ..Matrix::BT709 }] {
                let yuv = Yuv420::from_rgb(&rgb, 4, 4, m);
                let mut back = Vec::new();
                yuv.write_rgb(m, &mut back);
                for (a, b) in back.iter().zip(&rgb) {
                    assert!(a.abs_diff(*b) <= 2, "{px:?} {m:?}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn odd_sizes_have_rounded_up_chroma() {
        let yuv = Yuv420::from_rgb(&[9u8; 3 * 5 * 3], 5, 3, Matrix::BT709);
        assert_eq!((yuv.y.len(), yuv.u.len(), yuv.v.len()), (15, 6, 6));
    }
}
