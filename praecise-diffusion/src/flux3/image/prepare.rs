//! Reference-image preparation of the FLUX 3 Image Klein release, as its
//! reference pipeline does it (`prepare_reference`,
//! `prepare_multi_reference`, `default_size` in `flux3/pipeline.py`).
//!
//! One reference is kept whole: it is fitted so its 16-aligned canvas stays
//! under 4 MiB pixels (sides 64 to 4096), resampled with Lanczos, and
//! reflect-padded to the canvas, centred; the generated image is cropped back
//! to the fitted box. Several references are each fitted under 1 MiB pixels
//! and centre-cropped to the 16-pixel grid.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "pixel geometry: sizes are bounded by the 4096-pixel side limit and the pixel budgets"
)]

use super::MAX_REFERENCE_PIXELS;
use crate::error::{Error, Result};
use crate::pipeline::RgbImage;

/// The token cell: 8x VAE and 2x2 patchify.
pub const CELL: u32 = 16;
/// Smallest canvas side.
pub const MIN_SIDE: u32 = 64;
/// Largest canvas side.
pub const MAX_SIDE: u32 = 4096;
/// Pixel budget of each reference of a multi-reference edit.
pub const MULTI_REFERENCE_MAX_PIXELS: u64 = 1024 * 1024;
/// Output size when no reference gives one.
pub const DEFAULT_SIZE: (u32, u32) = (1024, 1024);

/// `(left, top, right, bottom)` of the fitted image inside its canvas.
pub type FillBox = [u32; 4];

const fn ceil_cell(v: u32) -> u32 {
    v.div_ceil(CELL) * CELL
}

/// The fitted size and the canvas of a single reference of `width x height`.
///
/// # Errors
/// A zero side.
pub fn reference_canvas(width: u32, height: u32) -> Result<((u32, u32), (u32, u32))> {
    if width == 0 || height == 0 {
        return Err(Error::Request("reference with a zero side".into()));
    }
    let (w, h) = (f64::from(width), f64::from(height));
    let mut scale = 1f64.min((f64::from(MAX_REFERENCE_PIXELS) / (w * h)).sqrt()).min(f64::from(MAX_SIDE) / w.max(h));
    loop {
        // Python's round(): halves to even.
        let fit = ((w * scale).round_ties_even().max(1.0) as u32, (h * scale).round_ties_even().max(1.0) as u32);
        let canvas = (ceil_cell(fit.0).clamp(MIN_SIDE, MAX_SIDE), ceil_cell(fit.1).clamp(MIN_SIDE, MAX_SIDE));
        if u64::from(canvas.0) * u64::from(canvas.1) <= u64::from(MAX_REFERENCE_PIXELS) {
            return Ok((fit, canvas));
        }
        scale *= 0.995;
    }
}

/// Index into a periodic extension of `0..n` (numpy `pad`'s `reflect`, or
/// `symmetric` when `edge`: the edge pixel repeats).
fn extend(i: i64, n: i64, edge: bool) -> usize {
    let period = if edge { 2 * n } else { 2 * (n - 1) };
    let m = i.rem_euclid(period);
    let folded = if m < n {
        m
    } else if edge {
        period - 1 - m
    } else {
        period - m
    };
    folded as usize
}

/// A single reference, fitted to its canvas (see the module docs). Returns
/// the canvas image and, when it was padded, the box of the fitted image
/// inside it.
///
/// # Errors
/// A malformed image.
pub fn prepare_reference(image: &RgbImage) -> Result<(RgbImage, Option<FillBox>)> {
    check(image)?;
    let (fit, (cw, ch)) = reference_canvas(image.width, image.height)?;
    let fitted = if fit == (image.width, image.height) { image.clone() } else { image.lanczos(fit.0, fit.1) };
    if (cw, ch) == fit {
        return Ok((fitted, None));
    }
    let (left, top) = ((cw - fit.0) / 2, (ch - fit.1) / 2);
    // reflect needs a pad shorter than the side; small images mirror with the edge pixel repeated.
    let edge = fit.0.min(fit.1) <= CELL;
    let mut rgb = vec![0u8; cw as usize * ch as usize * 3];
    for y in 0..ch as usize {
        let sy = extend(y as i64 - i64::from(top), i64::from(fit.1), edge);
        for x in 0..cw as usize {
            let sx = extend(x as i64 - i64::from(left), i64::from(fit.0), edge);
            let (s, d) = ((sy * fit.0 as usize + sx) * 3, (y * cw as usize + x) * 3);
            rgb[d..d + 3].copy_from_slice(&fitted.rgb[s..s + 3]);
        }
    }
    Ok((RgbImage { width: cw, height: ch, rgb }, Some([left, top, left + fit.0, top + fit.1])))
}

/// One reference of a multi-reference edit: fitted under 1 MiB pixels
/// (sides truncated), centre-cropped to the 16-pixel grid.
///
/// # Errors
/// A malformed image, or one smaller than a token cell.
pub fn prepare_multi_reference(image: &RgbImage) -> Result<RgbImage> {
    check(image)?;
    let (mut width, mut height) = (image.width, image.height);
    let fitted;
    let mut src = image;
    if u64::from(width) * u64::from(height) > MULTI_REFERENCE_MAX_PIXELS {
        let scale = (MULTI_REFERENCE_MAX_PIXELS as f64 / (f64::from(width) * f64::from(height))).sqrt();
        width = (f64::from(width) * scale) as u32;
        height = (f64::from(height) * scale) as u32;
        if width == 0 || height == 0 {
            return Err(Error::Request("reference too thin to fit the pixel budget".into()));
        }
        fitted = image.lanczos(width, height);
        src = &fitted;
    }
    let (w, h) = (width / CELL * CELL, height / CELL * CELL);
    if w == 0 || h == 0 {
        return Err(Error::Request(format!("reference {width}x{height} is smaller than a {CELL}-pixel cell")));
    }
    Ok(crop(src, (width - w) / 2, (height - h) / 2, w, h))
}

/// The `w x h` window of `image` at `(left, top)`.
#[must_use]
pub fn crop(image: &RgbImage, left: u32, top: u32, w: u32, h: u32) -> RgbImage {
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    for y in top..top + h {
        let start = (y as usize * image.width as usize + left as usize) * 3;
        rgb.extend_from_slice(&image.rgb[start..start + w as usize * 3]);
    }
    RgbImage { width: w, height: h, rgb }
}

/// The output size when the request leaves it open: the first prepared
/// reference's own size, else 1024 x 1024.
#[must_use]
pub fn default_size(prepared: &[RgbImage]) -> (u32, u32) {
    prepared.first().map_or(DEFAULT_SIZE, |r| (r.width, r.height))
}

fn check(image: &RgbImage) -> Result<()> {
    if image.width == 0 || image.height == 0 || image.rgb.len() as u64 != u64::from(image.width) * u64::from(image.height) * 3 {
        return Err(Error::Request("reference pixel buffer does not match its size".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> RgbImage {
        let rgb = (0..w * h).flat_map(|i| [(i % w) as u8, (i / w) as u8, (i % 251) as u8]).collect();
        RgbImage { width: w, height: h, rgb }
    }

    #[test]
    fn a_small_odd_image_pads_to_the_cell_grid_without_resizing() {
        let (fit, canvas) = reference_canvas(1000, 700).unwrap();
        assert_eq!(fit, (1000, 700));
        assert_eq!(canvas, (1008, 704));
    }

    #[test]
    fn a_huge_image_fits_four_mebipixels_on_the_grid() {
        let (fit, canvas) = reference_canvas(8000, 6000).unwrap();
        assert!(u64::from(canvas.0) * u64::from(canvas.1) <= 4 * 1024 * 1024);
        assert!(canvas.0.is_multiple_of(CELL) && canvas.1.is_multiple_of(CELL));
        assert!(fit.0 <= canvas.0 && fit.1 <= canvas.1);
        assert!((f64::from(fit.0) / f64::from(fit.1) - 4.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn tiny_images_get_the_minimum_canvas() {
        let (fit, canvas) = reference_canvas(20, 30).unwrap();
        assert_eq!(fit, (20, 30));
        assert_eq!(canvas, (64, 64));
    }

    #[test]
    fn padding_reflects_without_repeating_the_edge_and_the_fitted_box_is_centred() {
        let img = gradient(1000, 700);
        let (out, fill) = prepare_reference(&img).unwrap();
        assert_eq!((out.width, out.height), (1008, 704));
        let b = fill.unwrap();
        assert_eq!(b, [4, 2, 1004, 702]);
        assert_eq!(crop(&out, b[0], b[1], 1000, 700), img);
        // Left pad pixel x=3 mirrors source x=1 (reflect), x=0 mirrors source x=4.
        let px = |x: u32, y: u32| out.rgb[((y * out.width + x) * 3) as usize];
        assert_eq!(px(3, 100), 1);
        assert_eq!(px(0, 100), 4);
    }

    #[test]
    fn a_grid_aligned_image_is_returned_whole() {
        let img = gradient(256, 128);
        let (out, fill) = prepare_reference(&img).unwrap();
        assert!(fill.is_none());
        assert_eq!(out, img);
    }

    #[test]
    fn small_images_mirror_with_the_edge_pixel_repeated() {
        let img = gradient(10, 10);
        let (out, fill) = prepare_reference(&img).unwrap();
        assert_eq!((out.width, out.height), (64, 64));
        assert_eq!(fill.unwrap(), [27, 27, 37, 37]);
        let px = |x: u32, y: u32| out.rgb[((y * out.width + x) * 3) as usize];
        assert_eq!(px(26, 30), px(27, 30));
    }

    #[test]
    fn multi_reference_floors_to_the_grid_with_a_centred_crop() {
        let out = prepare_multi_reference(&gradient(100, 50)).unwrap();
        assert_eq!((out.width, out.height), (96, 48));
        assert_eq!(out.rgb[0], 2);
        let big = prepare_multi_reference(&gradient(2000, 1500)).unwrap();
        assert!(u64::from(big.width) * u64::from(big.height) <= MULTI_REFERENCE_MAX_PIXELS);
        assert!(big.width.is_multiple_of(CELL) && big.height.is_multiple_of(CELL));
        assert!(prepare_multi_reference(&gradient(10, 400)).is_err());
    }

    #[test]
    fn the_default_size_follows_the_first_reference() {
        assert_eq!(default_size(&[]), (1024, 1024));
        assert_eq!(default_size(&[gradient(96, 48)]), (96, 48));
    }

    #[test]
    fn extension_matches_numpy_pad() {
        // np.pad([0,1,2,3], 3, 'reflect') = 3 2 1 0 1 2 3 2 1 0; 'symmetric' = 2 1 0 0 1 2 3 3 2 1
        let reflect: Vec<usize> = (-3..7).map(|i| extend(i, 4, false)).collect();
        assert_eq!(reflect, [3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
        let sym: Vec<usize> = (-3..7).map(|i| extend(i, 4, true)).collect();
        assert_eq!(sym, [2, 1, 0, 0, 1, 2, 3, 3, 2, 1]);
    }
}
