//! ByteTrack: multi-object tracking by associating every detection box.
//!
//! Each frame, tracks are predicted forward with a Kalman filter and matched
//! to detections by box overlap in two passes: first the confident detections
//! against every track (including those lost a short while), then the
//! low-confidence detections against the tracks still unmatched, which keeps
//! a track through occlusion and motion blur where its detector score drops.
//! A new track needs a confident detection and a second frame to confirm it;
//! a lost track is kept for a buffer of frames so it can resume under its id.
//! Boxes of different classes never match.
//!
//! The algorithm is ByteTrack (Zhang et al., ECCV 2022, MIT licence); this is
//! an independent implementation.

use serde::{Deserialize, Serialize};

use crate::assignment::assign;
use crate::kalman::{self, Cov, Mean};

/// One detection handed to the tracker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Box `(x0, y0, x1, y1)` in pixels.
    pub bbox: [f32; 4],
    /// Detector confidence in `[0, 1]`.
    pub score: f32,
    /// Class index; boxes of different classes are never associated.
    pub class: u32,
}

/// Tracker settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackerConfig {
    /// Detections at or above this score take part in the first association.
    pub high_threshold: f32,
    /// Detections above this score (and below `high_threshold`) take part in
    /// the second association; lower ones are ignored.
    pub low_threshold: f32,
    /// A new track starts only from a detection at or above this score.
    pub new_track_threshold: f32,
    /// Largest `1 - IoU` (score-weighted) accepted in the first association.
    pub match_threshold: f32,
    /// Frames a lost track is kept before it is dropped.
    pub track_buffer: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self { high_threshold: 0.5, low_threshold: 0.1, new_track_threshold: 0.6, match_threshold: 0.8, track_buffer: 30 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Tracked,
    Lost,
}

#[derive(Debug, Clone)]
struct Track {
    /// Public id, given when the track is confirmed.
    id: Option<u64>,
    class: u32,
    score: f32,
    mean: Mean,
    cov: Cov,
    state: State,
    start_frame: u64,
    last_frame: u64,
    hits: u32,
}

/// A confirmed track as of the latest frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackBox {
    /// Track id, stable for the life of the track.
    pub id: u64,
    /// Class index of the detections it follows.
    pub class: u32,
    /// Box `(x0, y0, x1, y1)` in pixels, from the filtered state.
    pub bbox: [f32; 4],
    /// Score of the detection matched this frame.
    pub score: f32,
    /// Centre velocity in pixels per frame.
    pub velocity: [f32; 2],
    /// Frames since the track started.
    pub age: u64,
}

fn xyah(b: [f32; 4]) -> [f64; 4] {
    let w = f64::from(b[2] - b[0]).max(1e-3);
    let h = f64::from(b[3] - b[1]).max(1e-3);
    [f64::from(b[0]) + w / 2.0, f64::from(b[1]) + h / 2.0, w / h, h]
}

impl Track {
    fn new(d: &Detection, frame: u64) -> Self {
        let (mean, cov) = kalman::initiate(xyah(d.bbox));
        Self { id: None, class: d.class, score: d.score, mean, cov, state: State::Tracked, start_frame: frame, last_frame: frame, hits: 1 }
    }

    fn bbox(&self) -> [f32; 4] {
        let h = self.mean[3];
        let w = self.mean[2] * h;
        [
            (self.mean[0] - w / 2.0) as f32,
            (self.mean[1] - h / 2.0) as f32,
            (self.mean[0] + w / 2.0) as f32,
            (self.mean[1] + h / 2.0) as f32,
        ]
    }

    fn predict(&mut self) {
        if self.state != State::Tracked {
            self.mean[7] = 0.0;
        }
        kalman::predict(&mut self.mean, &mut self.cov);
    }

    fn update(&mut self, d: &Detection, frame: u64) {
        kalman::update(&mut self.mean, &mut self.cov, xyah(d.bbox));
        self.score = d.score;
        self.state = State::Tracked;
        self.last_frame = frame;
        self.hits += 1;
    }
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0) + (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// `1 - IoU` between tracks and detections; with `fuse`, the IoU is weighted
/// by the detection score. Different classes cost 1.
fn distances(tracks: &[&Track], dets: &[&Detection], fuse: bool) -> Vec<Vec<f64>> {
    tracks
        .iter()
        .map(|t| {
            let tb = t.bbox();
            dets.iter()
                .map(|d| {
                    if d.class != t.class {
                        return 1.0;
                    }
                    let s = iou(tb, d.bbox);
                    1.0 - f64::from(if fuse { s * d.score } else { s })
                })
                .collect()
        })
        .collect()
}

/// The ByteTrack tracker. Feed it every frame's detections in order.
#[derive(Debug, Clone)]
pub struct ByteTracker {
    config: TrackerConfig,
    frame: u64,
    next_id: u64,
    tracks: Vec<Track>,
}

impl ByteTracker {
    /// A tracker with `config`.
    #[must_use]
    pub fn new(config: TrackerConfig) -> Self {
        Self { config, frame: 0, next_id: 1, tracks: Vec::new() }
    }

    /// Tracks confirmed so far (the largest id handed out).
    #[must_use]
    pub fn confirmed(&self) -> u64 {
        self.next_id - 1
    }

    /// Advance one frame with its detections; returns the confirmed tracks
    /// matched in this frame.
    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackBox> {
        self.frame += 1;
        let frame = self.frame;
        let cfg = self.config;
        let high: Vec<usize> = (0..detections.len()).filter(|&i| detections[i].score >= cfg.high_threshold).collect();
        let low: Vec<usize> = (0..detections.len())
            .filter(|&i| detections[i].score > cfg.low_threshold && detections[i].score < cfg.high_threshold)
            .collect();

        // confirmed tracks (tracked or lost) are predicted and matched first; unconfirmed ones after
        let (pool, unconfirmed): (Vec<usize>, Vec<usize>) = (0..self.tracks.len()).partition(|&i| self.tracks[i].id.is_some());
        for &i in &pool {
            self.tracks[i].predict();
        }
        let mut matched_track = vec![false; self.tracks.len()];

        // first association: confident detections against every confirmed track
        let pt: Vec<&Track> = pool.iter().map(|&i| &self.tracks[i]).collect();
        let hd: Vec<&Detection> = high.iter().map(|&i| &detections[i]).collect();
        let (m1, u_pool, u_high) = assign(&distances(&pt, &hd, true), hd.len(), f64::from(cfg.match_threshold));
        for &(t, d) in &m1 {
            self.tracks[pool[t]].update(&detections[high[d]], frame);
            matched_track[pool[t]] = true;
        }

        // second association: low-score detections against tracks still tracked
        let remaining: Vec<usize> = u_pool.iter().map(|&t| pool[t]).filter(|&i| self.tracks[i].state == State::Tracked).collect();
        let rt: Vec<&Track> = remaining.iter().map(|&i| &self.tracks[i]).collect();
        let ld: Vec<&Detection> = low.iter().map(|&i| &detections[i]).collect();
        let (m2, u_rem, _) = assign(&distances(&rt, &ld, false), ld.len(), 0.5);
        for &(t, d) in &m2 {
            self.tracks[remaining[t]].update(&detections[low[d]], frame);
            matched_track[remaining[t]] = true;
        }
        for &t in &u_rem {
            self.tracks[remaining[t]].state = State::Lost;
        }

        // unconfirmed tracks against the confident detections left; confirmed on a match, dropped otherwise
        let left_high: Vec<usize> = u_high.iter().map(|&d| high[d]).collect();
        let ut: Vec<&Track> = unconfirmed.iter().map(|&i| &self.tracks[i]).collect();
        let lh: Vec<&Detection> = left_high.iter().map(|&i| &detections[i]).collect();
        let (m3, _, u_left) = assign(&distances(&ut, &lh, true), lh.len(), 0.7);
        let mut drop = vec![false; self.tracks.len()];
        for &i in &unconfirmed {
            drop[i] = true;
        }
        for &(t, d) in &m3 {
            let i = unconfirmed[t];
            self.tracks[i].update(&detections[left_high[d]], frame);
            self.tracks[i].id = Some(self.next_id);
            self.next_id += 1;
            matched_track[i] = true;
            drop[i] = false;
        }

        // lost tracks past the buffer are dropped
        for (i, t) in self.tracks.iter().enumerate() {
            if t.state == State::Lost && frame - t.last_frame > u64::from(cfg.track_buffer) {
                drop[i] = true;
            }
        }

        // a lost track that overlaps a tracked one is a duplicate; the younger of the two goes
        let tracked: Vec<usize> = (0..self.tracks.len()).filter(|&i| !drop[i] && self.tracks[i].state == State::Tracked && self.tracks[i].id.is_some()).collect();
        let lost: Vec<usize> = (0..self.tracks.len()).filter(|&i| !drop[i] && self.tracks[i].state == State::Lost).collect();
        for &a in &tracked {
            for &b in &lost {
                let (ta, tb) = (&self.tracks[a], &self.tracks[b]);
                if ta.class == tb.class && 1.0 - iou(ta.bbox(), tb.bbox()) < 0.15 {
                    if frame - ta.start_frame >= frame - tb.start_frame {
                        drop[b] = true;
                    } else {
                        drop[a] = true;
                    }
                }
            }
        }

        let mut out = Vec::new();
        for (i, t) in self.tracks.iter().enumerate() {
            if !drop[i]
                && matched_track[i]
                && t.state == State::Tracked
                && let Some(id) = t.id
            {
                out.push(TrackBox {
                    id,
                    class: t.class,
                    bbox: t.bbox(),
                    score: t.score,
                    velocity: [t.mean[4] as f32, t.mean[5] as f32],
                    age: frame - t.start_frame,
                });
            }
        }

        let mut kept = Vec::with_capacity(self.tracks.len());
        for (i, t) in std::mem::take(&mut self.tracks).into_iter().enumerate() {
            if !drop[i] {
                kept.push(t);
            }
        }
        // new tracks from confident detections nothing claimed; on the first frame they are confirmed at once
        for &d in &u_left {
            let det = &detections[left_high[d]];
            if det.score >= cfg.new_track_threshold {
                let mut t = Track::new(det, frame);
                if frame == 1 {
                    t.id = Some(self.next_id);
                    self.next_id += 1;
                    out.push(TrackBox { id: t.id.unwrap_or(0), class: t.class, bbox: det.bbox, score: det.score, velocity: [0.0, 0.0], age: 0 });
                }
                kept.push(t);
            }
        }
        self.tracks = kept;
        out.sort_by_key(|t| t.id);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, score: f32, class: u32) -> Detection {
        Detection { bbox: [x, y, x + 40.0, y + 100.0], score, class }
    }

    #[test]
    fn two_walkers_keep_their_ids_through_a_weak_frame() {
        let mut t = ByteTracker::new(TrackerConfig::default());
        let mut ids = Vec::new();
        for f in 0..40 {
            let x = f as f32 * 3.0;
            // frames 20 to 24: the first walker is half hidden and scored low
            let s0 = if (20..25).contains(&f) { 0.3 } else { 0.9 };
            let out = t.update(&[det(100.0 + x, 50.0, s0, 0), det(400.0 - x, 60.0, 0.85, 0)]);
            assert_eq!(out.len(), 2, "frame {f}: {out:?}");
            ids.push((out[0].id, out[1].id));
        }
        assert!(ids.iter().all(|&p| p == ids[0]), "ids changed: {ids:?}");
        assert_eq!(t.confirmed(), 2);
    }

    #[test]
    fn a_track_resumes_after_a_gap_within_the_buffer() {
        let mut t = ByteTracker::new(TrackerConfig::default());
        for f in 0..10 {
            t.update(&[det(100.0 + f as f32 * 2.0, 50.0, 0.9, 2)]);
        }
        for _ in 0..5 {
            assert!(t.update(&[]).is_empty());
        }
        let out = t.update(&[det(130.0, 50.0, 0.9, 2)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 1);
    }

    #[test]
    fn classes_never_swap_and_a_one_frame_blip_is_not_confirmed() {
        let mut t = ByteTracker::new(TrackerConfig::default());
        t.update(&[det(100.0, 50.0, 0.9, 0)]);
        // a car appears where the person stands: a new track, not the person's
        let out = t.update(&[det(100.0, 50.0, 0.9, 0), det(102.0, 52.0, 0.9, 2)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class, 0);
        let out = t.update(&[det(100.0, 50.0, 0.9, 0), det(103.0, 52.0, 0.9, 2)]);
        assert_eq!(out.len(), 2);
        assert_eq!((out[1].id, out[1].class), (2, 2));
        // a confident box seen once is not reported
        t.update(&[det(100.0, 50.0, 0.9, 0), det(104.0, 52.0, 0.9, 2), det(600.0, 300.0, 0.95, 0)]);
        let out = t.update(&[det(100.0, 50.0, 0.9, 0), det(105.0, 52.0, 0.9, 2)]);
        assert_eq!(out.len(), 2);
        assert_eq!(t.confirmed(), 2);
    }

    #[test]
    fn velocity_follows_the_motion() {
        let mut t = ByteTracker::new(TrackerConfig::default());
        let mut last = Vec::new();
        for f in 0..30 {
            last = t.update(&[det(f as f32 * 5.0, 20.0, 0.9, 0)]);
        }
        assert!((last[0].velocity[0] - 5.0).abs() < 0.5, "{:?}", last[0].velocity);
        assert!(last[0].velocity[1].abs() < 0.5);
    }
}
