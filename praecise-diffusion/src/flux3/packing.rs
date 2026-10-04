//! Inputs of a FLUX 3 action policy: camera canvas composition, latent and
//! action token layout with their `(t, h, w, l)` positions on a 10 ms clock,
//! quantile normalisation of states and actions, and command deltas.

use crate::error::{Error, Result};

/// Video latent channels.
pub const LATENT_CHANNELS: usize = 96;
/// Spatial downsampling of the video autoencoder.
pub const SPATIAL_DOWNSAMPLE: usize = 32;
/// Temporal downsampling of the video autoencoder.
pub const TEMPORAL_DOWNSAMPLE: usize = 4;
/// Width of the global conditioning vector.
pub const VEC_DIM: usize = 768;

/// One RGB frame, channel-major `[3][h][w]`, values in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Pixels `[3][h][w]`.
    pub data: Vec<f32>,
}

impl Frame {
    /// A frame from 8-bit channel-major pixels.
    ///
    /// # Errors
    /// When the pixel count is not `3 * h * w`.
    pub fn from_u8(h: usize, w: usize, pixels: &[u8]) -> Result<Self> {
        if pixels.len() != 3 * h * w {
            return Err(Error::Request(format!("frame needs {} bytes, got {}", 3 * h * w, pixels.len())));
        }
        Ok(Self { h, w, data: pixels.iter().map(|&p| f32::from(p) / 255.0).collect() })
    }
}

/// How several cameras share one canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraLayout {
    /// One camera resized to the canvas.
    Single,
    /// Two cameras, each resized to half the canvas width.
    SideBySide,
    /// Wrist camera full size on top, the two exterior cameras at half size
    /// side by side below, reflect-padded to the canvas.
    WristOverPair,
    /// Any number of cameras in a near-square, row-major grid; empty cells black.
    Grid,
}

impl CameraLayout {
    /// The layout named in a policy configuration.
    ///
    /// # Errors
    /// On an unknown name.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "single" => Ok(Self::Single),
            "side_by_side" => Ok(Self::SideBySide),
            "droid" => Ok(Self::WristOverPair),
            "grid" => Ok(Self::Grid),
            other => Err(Error::Config(format!("unknown camera layout {other}"))),
        }
    }
}

/// Latent frames produced from `frames` video frames.
#[must_use]
pub fn latent_frames(frames: usize) -> usize {
    1 + (frames - 1) / TEMPORAL_DOWNSAMPLE
}

/// Latent grid kept for a canvas: `ceil(h / 32) x ceil(w / 32)`.
#[must_use]
pub fn latent_hw(h: usize, w: usize) -> (usize, usize) {
    (h.div_ceil(SPATIAL_DOWNSAMPLE), w.div_ceil(SPATIAL_DOWNSAMPLE))
}

/// Floor division of single-precision floats, rounding like the reference.
fn floor_div(a: f32, b: f32) -> f32 {
    let m = a % b;
    let mut d = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        d -= 1.0;
    }
    let f = d.floor();
    if d - f > 0.5 { f + 1.0 } else { f }
}

/// Position id of a time in seconds: tens of milliseconds, floored.
#[must_use]
pub fn time_id(seconds: f32) -> i32 {
    floor_div(seconds * 1000.0, 10.0) as i32
}

/// Seconds of latent frame `i` at `fps` (`i * 4 / fps`).
#[must_use]
pub fn latent_time(i: usize, fps: f32) -> f32 {
    i as f32 * TEMPORAL_DOWNSAMPLE as f32 / fps
}

/// Positions of a latent clip `[t][h][w]` laid out time-major, at the given
/// per-frame time ids and layer `l`.
#[must_use]
pub fn video_ids(time_ids: &[i32], h: usize, w: usize, l: i32) -> Vec<[i32; 4]> {
    let mut ids = Vec::with_capacity(time_ids.len() * h * w);
    for &t in time_ids {
        for y in 0..h {
            for x in 0..w {
                ids.push([t, y as i32, x as i32, l]);
            }
        }
    }
    ids
}

/// Tokens `[(t h w)][c]` from latents `[c][t][h][w]`.
#[must_use]
pub fn video_tokens(latents: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let n = t * h * w;
    let mut out = vec![0.0; n * c];
    for ch in 0..c {
        for i in 0..n {
            out[i * c + ch] = latents[ch * n + i];
        }
    }
    out
}

/// Latents `[c][t][h][w]` from tokens `[(t h w)][c]`.
#[must_use]
pub fn video_latents(tokens: &[f32], c: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * c];
    for i in 0..n {
        for ch in 0..c {
            out[ch * n + i] = tokens[i * c + ch];
        }
    }
    out
}

/// Positions of a sequence of action-like tokens: `(time id, 0, 0, l)`.
#[must_use]
pub fn sequence_ids(times: &[f32], l: i32) -> Vec<[i32; 4]> {
    times.iter().map(|&s| [time_id(s), 0, 0, l]).collect()
}

/// Positions of `n` text tokens: `(0, 0, 0, i)`.
#[must_use]
pub fn text_ids(n: usize) -> Vec<[i32; 4]> {
    (0..n).map(|i| [0, 0, 0, i as i32]).collect()
}

/// Index ranges and weights of one resized axis.
struct Taps {
    start: Vec<usize>,
    weights: Vec<Vec<f32>>,
}

/// Separable triangle-filter taps, widened by the downscale factor.
fn antialias_taps(input: usize, output: usize) -> Taps {
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut start = Vec::with_capacity(output);
    let mut weights = Vec::with_capacity(output);
    for i in 0..output {
        let center = scale * (i as f32 + 0.5);
        let lo = ((center - support + 0.5) as i64).max(0) as usize;
        let hi = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let mut w: Vec<f32> = (lo..hi)
            .map(|j| {
                let x = ((j as f32 - center + 0.5) * inv).abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            })
            .collect();
        let total: f32 = w.iter().sum();
        if total != 0.0 {
            for v in &mut w {
                *v /= total;
            }
        }
        start.push(lo);
        weights.push(w);
    }
    Taps { start, weights }
}

/// Two-tap bilinear taps with half-pixel centres, clamped at the edges.
fn bilinear_taps(input: usize, output: usize) -> Taps {
    let scale = input as f32 / output as f32;
    let mut start = Vec::with_capacity(output);
    let mut weights = Vec::with_capacity(output);
    for i in 0..output {
        let src = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
        let i0 = (src as usize).min(input - 1);
        let lambda = src - i0 as f32;
        if i0 + 1 < input {
            start.push(i0);
            weights.push(vec![1.0 - lambda, lambda]);
        } else {
            start.push(i0);
            weights.push(vec![1.0]);
        }
    }
    Taps { start, weights }
}

/// Resize one `[3][h][w]` frame, width first.
fn resize(f: &Frame, h: usize, w: usize, antialias: bool) -> Frame {
    let (tx, ty) = if antialias {
        (antialias_taps(f.w, w), antialias_taps(f.h, h))
    } else {
        (bilinear_taps(f.w, w), bilinear_taps(f.h, h))
    };
    let mut mid = vec![0.0f32; 3 * f.h * w];
    for c in 0..3 {
        for y in 0..f.h {
            let row = &f.data[(c * f.h + y) * f.w..][..f.w];
            for x in 0..w {
                mid[(c * f.h + y) * w + x] = tx.weights[x].iter().enumerate().map(|(k, wt)| wt * row[tx.start[x] + k]).sum();
            }
        }
    }
    let mut out = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                out[(c * h + y) * w + x] =
                    ty.weights[y].iter().enumerate().map(|(k, wt)| wt * mid[(c * f.h + ty.start[y] + k) * w + x]).sum();
            }
        }
    }
    Frame { h, w, data: out }
}

/// Copy `src` into `dst` (`[3][dh][dw]`) at row `y0`, column `x0`.
fn blit(dst: &mut [f32], dh: usize, dw: usize, src: &Frame, y0: usize, x0: usize) {
    for c in 0..3 {
        for y in 0..src.h {
            let s = &src.data[(c * src.h + y) * src.w..][..src.w];
            dst[(c * dh + y0 + y) * dw + x0..][..src.w].copy_from_slice(s);
        }
    }
}

/// Reflect index `i` into `0..n`.
fn reflect(i: usize, n: usize) -> usize {
    if i < n { i } else { 2 * (n - 1) - i }
}

/// Near-square `(rows, cols)`, wider than tall.
#[must_use]
pub fn grid_shape(n: usize) -> (usize, usize) {
    let cols = (n as f64).sqrt().ceil() as usize;
    (n.div_ceil(cols), cols)
}

/// Compose one time step of every camera onto a `[3][ch][cw]` canvas in `[0, 1]`.
fn compose_step(cams: &[&Frame], layout: CameraLayout, ch: usize, cw: usize) -> Result<Vec<f32>> {
    let mut canvas = vec![0.0f32; 3 * ch * cw];
    match layout {
        CameraLayout::Single => {
            if cams.len() != 1 {
                return Err(Error::Request(format!("single layout needs one camera, got {}", cams.len())));
            }
            blit(&mut canvas, ch, cw, &resize(cams[0], ch, cw, true), 0, 0);
        }
        CameraLayout::SideBySide => {
            if cams.len() != 2 || cw % 2 != 0 {
                return Err(Error::Request("side-by-side layout needs two cameras and an even canvas width".into()));
            }
            for (i, cam) in cams.iter().enumerate() {
                blit(&mut canvas, ch, cw, &resize(cam, ch, cw / 2, true), 0, i * (cw / 2));
            }
        }
        CameraLayout::WristOverPair => {
            if cams.len() != 3 {
                return Err(Error::Request(format!("wrist-over-pair layout needs three cameras, got {}", cams.len())));
            }
            let (h, w) = (cams[0].h, cams[0].w);
            if cams.iter().any(|c| c.h != h || c.w != w) {
                return Err(Error::Request("wrist-over-pair cameras must share one size".into()));
            }
            let (hh, hw) = (h / 2, w / 2);
            let (ah, aw) = (h + hh, 2 * hw);
            if ah > ch || aw > cw || (ch > ah && ch - ah >= ah) || (cw > aw && cw - aw >= aw) {
                return Err(Error::Request(format!("canvas {ch}x{cw} does not fit the {ah}x{aw} composite")));
            }
            let mut comp = vec![0.0f32; 3 * ah * aw];
            blit(&mut comp, ah, aw, cams[0], 0, 0);
            blit(&mut comp, ah, aw, &resize(cams[1], hh, hw, false), h, 0);
            blit(&mut comp, ah, aw, &resize(cams[2], hh, hw, false), h, hw);
            for c in 0..3 {
                for y in 0..ch {
                    let sy = reflect(y, ah);
                    for x in 0..cw {
                        canvas[(c * ch + y) * cw + x] = comp[(c * ah + sy) * aw + reflect(x, aw)];
                    }
                }
            }
        }
        CameraLayout::Grid => {
            let (rows, cols) = grid_shape(cams.len());
            let (gh, gw) = (ch / rows, cw / cols);
            for (i, cam) in cams.iter().enumerate() {
                let (r, k) = (i / cols, i % cols);
                blit(&mut canvas, ch, cw, &resize(cam, gh, gw, true), r * gh, k * gw);
            }
        }
    }
    Ok(canvas)
}

/// Compose camera clips (`cams[camera][time]`) onto a canvas video
/// `[3][t][ch][cw]` in `[-1, 1]`.
///
/// # Errors
/// On an empty or ragged input or a layout/camera-count mismatch.
pub fn compose_canvas(cams: &[Vec<Frame>], layout: CameraLayout, ch: usize, cw: usize) -> Result<Vec<f32>> {
    let t = cams.first().map_or(0, Vec::len);
    if t == 0 || cams.iter().any(|c| c.len() != t) {
        return Err(Error::Request("every camera needs the same, non-zero number of frames".into()));
    }
    let plane = ch * cw;
    let mut out = vec![0.0f32; 3 * t * plane];
    for step in 0..t {
        let frames: Vec<&Frame> = cams.iter().map(|c| &c[step]).collect();
        let canvas = compose_step(&frames, layout, ch, cw)?;
        for c in 0..3 {
            for (o, v) in out[(c * t + step) * plane..][..plane].iter_mut().zip(&canvas[c * plane..][..plane]) {
                *o = 2.0 * v - 1.0;
            }
        }
    }
    Ok(out)
}

/// Per-channel 1st and 99th percentiles mapping a stream to `[-1, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantiles {
    /// Lower quantile per channel.
    pub q01: Vec<f32>,
    /// Upper quantile per channel.
    pub q99: Vec<f32>,
}

impl Quantiles {
    fn span(&self, i: usize) -> f32 {
        let s = self.q99[i] - self.q01[i];
        if s > 1e-6 { s } else { 1.0 }
    }

    /// `2 (x - q01) / span - 1`, clipped to `±clip`, over rows of `q01.len()` channels.
    #[must_use]
    pub fn normalize(&self, x: &[f32], clip: f32) -> Vec<f32> {
        let d = self.q01.len();
        x.iter().enumerate().map(|(i, v)| (2.0 * (v - self.q01[i % d]) / self.span(i % d) - 1.0).clamp(-clip, clip)).collect()
    }

    /// Inverse of [`Quantiles::normalize`] (without the clip).
    #[must_use]
    pub fn denormalize(&self, x: &[f32]) -> Vec<f32> {
        let d = self.q01.len();
        x.iter().enumerate().map(|(i, v)| (v + 1.0) * self.span(i % d) / 2.0 + self.q01[i % d]).collect()
    }
}

/// How the action stream encodes commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionRepresentation {
    /// Absolute commands.
    Absolute,
    /// Consecutive command differences, except the listed channels which stay absolute.
    Delta {
        /// Channels left absolute (negative indices count from the end).
        absolute: Vec<i64>,
    },
}

impl ActionRepresentation {
    fn absolute_mask(&self, d: usize) -> Vec<bool> {
        let mut m = vec![matches!(self, Self::Absolute); d];
        if let Self::Delta { absolute } = self {
            for &a in absolute {
                m[a.rem_euclid(d as i64) as usize] = true;
            }
        }
        m
    }
}

/// Past-action conditioning from an absolute command history `[n][d]`:
/// a zero row followed by the normalised steps between consecutive commands.
#[must_use]
pub fn past_actions(commands: &[f32], d: usize, repr: &ActionRepresentation, q: &Quantiles, clip: f32) -> Vec<f32> {
    let n = commands.len() / d;
    let abs = repr.absolute_mask(d);
    let mut values = Vec::with_capacity((n.saturating_sub(1)) * d);
    for i in 1..n {
        for c in 0..d {
            let cur = commands[i * d + c];
            values.push(if abs[c] { cur } else { cur - commands[(i - 1) * d + c] });
        }
    }
    let mut out = vec![0.0; d];
    out.extend(q.normalize(&values, clip));
    out
}

/// Absolute commands `[k][d]` from a predicted, normalised chunk: undo the
/// normalisation and, for deltas, integrate from the last command `anchor`.
#[must_use]
pub fn integrate_actions(chunk: &[f32], d: usize, repr: &ActionRepresentation, q: &Quantiles, anchor: &[f32]) -> Vec<f32> {
    let mut values = q.denormalize(chunk);
    if matches!(repr, ActionRepresentation::Delta { .. }) {
        let abs = repr.absolute_mask(d);
        for c in (0..d).filter(|&c| !abs[c]) {
            let mut acc = 0.0f32;
            for row in values.chunks_mut(d) {
                acc += row[c];
                row[c] = acc + anchor[c];
            }
        }
    }
    values
}

/// `x -> 1 - x` on the listed channels of rows of `d` channels.
pub fn flip_channels(x: &mut [f32], d: usize, dims: &[usize]) {
    for row in x.chunks_mut(d) {
        for &c in dims {
            row[c] = 1.0 - row[c];
        }
    }
}

/// Indices of the history frames encoded as conditioning snapshots.
#[must_use]
pub fn snapshot_indices(history: usize, snapshots: usize) -> Vec<usize> {
    if snapshots == 1 {
        return vec![history - 1];
    }
    (0..snapshots).map(|j| ((j * (history - 1)) as f64 / (snapshots - 1) as f64).round_ties_even() as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_ids_are_tens_of_milliseconds() {
        assert_eq!(time_id(0.0), 0);
        assert_eq!(time_id(1.0 / 15.0), 6);
        assert_eq!(time_id(2.0 / 15.0), 13);
        assert_eq!(time_id(-1.0 / 30.0), -4);
    }

    #[test]
    fn snapshots_spread_over_the_history() {
        assert_eq!(snapshot_indices(8, 2), vec![0, 7]);
        assert_eq!(snapshot_indices(8, 1), vec![7]);
        assert_eq!(snapshot_indices(5, 3), vec![0, 2, 4]);
    }

    #[test]
    fn deltas_round_trip_through_integration() {
        let q = Quantiles { q01: vec![-1.0, 0.0], q99: vec![1.0, 1.0] };
        let repr = ActionRepresentation::Delta { absolute: vec![-1] };
        let cmds = [0.0f32, 0.2, 0.1, 0.4, 0.3, 0.9];
        let past = past_actions(&cmds, 2, &repr, &q, 6.0);
        assert_eq!(&past[..2], &[0.0, 0.0]);
        let back = integrate_actions(&past[2..], 2, &repr, &q, &cmds[..2]);
        for (b, c) in back.iter().zip(&cmds[2..]) {
            assert!((b - c).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "needs the reference fixtures"]
    fn packing_parity() {
        let dir = std::path::PathBuf::from(std::env::var("PRAECISE_FLUX3_PACKING").expect("PRAECISE_FLUX3_PACKING"));
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        for (t, id) in meta["times"].as_array().unwrap().iter().zip(meta["ids"].as_array().unwrap()) {
            assert_eq!(i64::from(time_id(t.as_f64().unwrap() as f32)), id.as_i64().unwrap(), "time {t}");
        }
        for case in meta["cases"].as_array().unwrap() {
            let layout = case["layout"].as_str().unwrap();
            let n = case["cams"].as_u64().unwrap() as usize;
            let t = case["frames"].as_u64().unwrap() as usize;
            let (h, w) = (case["hw"][0].as_u64().unwrap() as usize, case["hw"][1].as_u64().unwrap() as usize);
            let (ch, cw) = (case["canvas"][0].as_u64().unwrap() as usize, case["canvas"][1].as_u64().unwrap() as usize);
            let raw = std::fs::read(dir.join(format!("{layout}_cams.u8"))).unwrap();
            let cams: Vec<Vec<Frame>> = (0..n)
                .map(|c| (0..t).map(|k| Frame::from_u8(h, w, &raw[(c * t + k) * 3 * h * w..][..3 * h * w]).unwrap()).collect())
                .collect();
            let got = compose_canvas(&cams, CameraLayout::parse(layout).unwrap(), ch, cw).unwrap();
            let want: Vec<f32> = std::fs::read(dir.join(format!("{layout}_canvas.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(got.len(), want.len(), "{layout}");
            let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("{layout}: max abs error {max:.2e}");
            assert!(max < 1e-5, "{layout}: max abs error {max}");
        }
    }

    #[test]
    fn a_same_size_resize_is_the_identity() {
        let f = Frame { h: 2, w: 3, data: (0..18).map(|i| i as f32 / 18.0).collect() };
        assert_eq!(resize(&f, 2, 3, true).data, f.data);
    }
}
