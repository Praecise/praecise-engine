//! Image preparation for the image-to-3D pipeline: crop to the alpha mask's
//! square bounding box, premultiply, and resample with the same 8-bit
//! fixed-point Lanczos filter the reference's image library uses, so the
//! encoder sees the same pixels.

use crate::error::{Error, Result};
use crate::pipeline::RgbImage;

/// An 8-bit RGBA image, row-major, top row first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// `width * height * 4` bytes.
    pub data: Vec<u8>,
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) { sinc(x) * sinc(x / 3.0) } else { 0.0 }
}

/// Fixed-point Lanczos taps for resampling `inp` samples to `out`:
/// per output sample its first input index and integer weights.
fn taps(inp: usize, out: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = inp as f64 / out as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let ss = 1.0 / filterscale;
    (0..out)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).min(inp as i64) as usize).saturating_sub(xmin);
            let mut k: Vec<f64> = (0..xmax).map(|x| lanczos((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            let kk = k
                .iter()
                .map(|&v| {
                    let f = v * f64::from(1u32 << PRECISION_BITS);
                    if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 }
                })
                .collect();
            (xmin, kk)
        })
        .collect()
}

fn clip8(v: i64) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resize interleaved 8-bit pixels (`ch` channels) with the Lanczos filter,
/// horizontal pass first, rounding to 8 bits after each pass.
#[must_use]
pub fn resize_lanczos(src: &[u8], w: usize, h: usize, ch: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut cur = src.to_vec();
    let (mut cw, ch_h) = (w, h);
    if nw != w {
        let t = taps(w, nw);
        let mut out = vec![0u8; nw * h * ch];
        for y in 0..h {
            for (xx, (xmin, k)) in t.iter().enumerate() {
                for c in 0..ch {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (x, &kv) in k.iter().enumerate() {
                        ss += i64::from(cur[(y * w + xmin + x) * ch + c]) * i64::from(kv);
                    }
                    out[(y * nw + xx) * ch + c] = clip8(ss);
                }
            }
        }
        cur = out;
        cw = nw;
    }
    if nh != ch_h {
        let t = taps(h, nh);
        let mut out = vec![0u8; cw * nh * ch];
        for (yy, (ymin, k)) in t.iter().enumerate() {
            for x in 0..cw {
                for c in 0..ch {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (y, &kv) in k.iter().enumerate() {
                        ss += i64::from(cur[((ymin + y) * cw + x) * ch + c]) * i64::from(kv);
                    }
                    out[(yy * cw + x) * ch + c] = clip8(ss);
                }
            }
        }
        cur = out;
    }
    cur
}

/// Python's `round`: halves go to the even neighbour.
fn round_half_even(v: f64) -> i64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 {
        let f = v.floor();
        if (f as i64) % 2 == 0 { f as i64 } else { f as i64 + 1 }
    } else {
        r as i64
    }
}

/// Crop `img` to the square around its alpha mask (alpha above 80%),
/// premultiply colour by alpha, and return the RGB result. Images wider or
/// taller than 1024 are first scaled down to fit.
///
/// # Errors
/// [`Error::Request`] when the image has no transparency to locate the
/// object by, or the mask is empty.
pub fn crop_to_object(img: &RgbaImage) -> Result<RgbImage> {
    let (mut w, mut h) = (img.width as usize, img.height as usize);
    if img.data.len() != w * h * 4 || w == 0 || h == 0 {
        return Err(Error::Request("image size does not match its pixels".into()));
    }
    if img.data.chunks_exact(4).all(|p| p[3] == 255) {
        return Err(Error::Request(
            "the image is fully opaque: image-to-3D needs the object isolated by an alpha mask (background removal is not part of this engine)".into(),
        ));
    }
    let mut data = img.data.clone();
    let max = w.max(h);
    if max > 1024 {
        let s = 1024.0 / max as f64;
        let (nw, nh) = ((w as f64 * s) as usize, (h as f64 * s) as usize);
        data = resize_lanczos(&data, w, h, 4, nw, nh);
        (w, h) = (nw, nh);
    }
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if f64::from(data[(y * w + x) * 4 + 3]) > 0.8 * 255.0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 == usize::MAX {
        return Err(Error::Request("the alpha mask holds no opaque object".into()));
    }
    let cx = (x0 + x1) as f64 / 2.0;
    let cy = (y0 + y1) as f64 / 2.0;
    let size = (x1 - x0).max(y1 - y0) as i64;
    let half = (size / 2) as f64;
    let bx0 = round_half_even(cx - half);
    let by0 = round_half_even(cy - half);
    let bx1 = round_half_even(cx + half);
    let by1 = round_half_even(cy + half);
    let (cw, chh) = ((bx1 - bx0).max(1) as usize, (by1 - by0).max(1) as usize);
    let mut rgb = vec![0u8; cw * chh * 3];
    for y in 0..chh {
        for x in 0..cw {
            let (sx, sy) = (bx0 + x as i64, by0 + y as i64);
            if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                continue;
            }
            let p = &data[(sy as usize * w + sx as usize) * 4..][..4];
            let a = f32::from(p[3]) / 255.0;
            for c in 0..3 {
                rgb[(y * cw + x) * 3 + c] = ((f32::from(p[c]) / 255.0 * a) * 255.0) as u8;
            }
        }
    }
    Ok(RgbImage { width: cw as u32, height: chh as u32, data: rgb })
}

/// Resize to `side x side` and normalise with the ImageNet mean and
/// deviation: `[3, side, side]` row-major floats.
#[must_use]
pub fn encoder_input(img: &RgbImage, side: usize) -> Vec<f32> {
    let px = resize_lanczos(&img.data, img.width as usize, img.height as usize, 3, side, side);
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut out = vec![0f32; 3 * side * side];
    for i in 0..side * side {
        for c in 0..3 {
            out[c * side * side + i] = (f32::from(px[i * 3 + c]) / 255.0 - mean[c]) / std[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_size_is_a_copy_and_constant_images_stay_constant() {
        let src: Vec<u8> = (0..4 * 4 * 3).map(|v| v as u8).collect();
        assert_eq!(resize_lanczos(&src, 4, 4, 3, 4, 4), src);
        let flat = vec![77u8; 10 * 6 * 3];
        assert!(resize_lanczos(&flat, 10, 6, 3, 7, 3).iter().all(|&v| v == 77));
    }

    #[test]
    fn python_rounding() {
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
        assert_eq!(round_half_even(-0.5), 0);
        assert_eq!(round_half_even(1.4), 1);
    }

    #[test]
    fn opaque_images_are_refused() {
        let img = RgbaImage { width: 2, height: 2, data: vec![255; 16] };
        assert!(crop_to_object(&img).is_err());
    }
}
