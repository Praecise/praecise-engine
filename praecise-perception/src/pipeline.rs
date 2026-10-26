//! Video in, per-frame tracks out: decode, detect, track.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::nvframe;
use crate::tracker::{ByteTracker, Detection, TrackBox, TrackerConfig};
use crate::{Error, Result};

/// Detects objects in raw RGB frames.
pub trait FrameDetector {
    /// Boxes `(x0, y0, x1, y1)` in frame pixels with their class index and
    /// score, for every detection scoring above `threshold`.
    ///
    /// # Errors
    /// When inference fails.
    fn detect(&self, rgb: &[u8], width: u32, height: u32, threshold: f32) -> Result<Vec<Detection>>;
}

impl FrameDetector for praecise_onnx::detection::GenericDetrDetector {
    fn detect(&self, rgb: &[u8], width: u32, height: u32, threshold: f32) -> Result<Vec<Detection>> {
        let r = self.detect_rgb(rgb, width, height, threshold).map_err(|e| Error::Detector(e.to_string()))?;
        Ok(r.detections.into_iter().map(|d| Detection { bbox: d.bbox, score: d.score, class: d.label_id }).collect())
    }
}

/// A class the pipeline tracks: the detector's class index, its name and the
/// object type it is reported as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackedClass {
    /// The detector's class index.
    pub index: u32,
    /// Class name, for instance `car`.
    pub name: String,
    /// Object type, for instance `Vehicle` (the `nv.Object.type`).
    pub kind: String,
}

/// People and road vehicles in COCO's 91-slot indexing (RF-DETR's).
#[must_use]
pub fn coco91_people_and_vehicles() -> Vec<TrackedClass> {
    [(1, "person", "Person"), (2, "bicycle", "Bicycle"), (3, "car", "Vehicle"), (4, "motorcycle", "Vehicle"), (6, "bus", "Vehicle"), (8, "truck", "Vehicle")]
        .into_iter()
        .map(|(index, name, kind)| TrackedClass { index, name: name.into(), kind: kind.into() })
        .collect()
}

/// People and road vehicles in COCO's 80-class indexing (D-FINE's).
#[must_use]
pub fn coco80_people_and_vehicles() -> Vec<TrackedClass> {
    [(0, "person", "Person"), (1, "bicycle", "Bicycle"), (2, "car", "Vehicle"), (3, "motorcycle", "Vehicle"), (5, "bus", "Vehicle"), (7, "truck", "Vehicle")]
        .into_iter()
        .map(|(index, name, kind)| TrackedClass { index, name: name.into(), kind: kind.into() })
        .collect()
}

/// Pipeline settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// The classes tracked; detections of other classes are dropped.
    pub classes: Vec<TrackedClass>,
    /// Tracker settings. Detections below `tracker.low_threshold` are
    /// dropped before tracking.
    pub tracker: TrackerConfig,
    /// Image-to-ground homography (row-major 3x3) taking a pixel `(u, v, 1)`
    /// to ground-plane metres; each track's ground point is the bottom centre
    /// of its box. Without it positions and speeds stay in pixels.
    pub ground: Option<[f64; 9]>,
    /// Stop after this many frames.
    pub max_frames: Option<u32>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { classes: coco91_people_and_vehicles(), tracker: TrackerConfig::default(), ground: None, max_frames: None }
    }
}

/// One tracked object in one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tracked {
    /// Track id, stable for the life of the track.
    pub id: u64,
    /// Class name.
    pub class: String,
    /// Object type.
    pub kind: String,
    /// Box `(x0, y0, x1, y1)` in pixels.
    pub bbox: [f32; 4],
    /// Score of this frame's detection.
    pub score: f32,
    /// Box-centre velocity in pixels per second.
    pub velocity_px_s: [f32; 2],
    /// Ground position in metres, with a ground homography.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_m: Option<[f64; 2]>,
    /// Ground velocity in metres per second, with a ground homography.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_velocity_m_s: Option<[f64; 2]>,
}

/// The tracks of one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameTracks {
    /// Frame index in display order.
    pub index: u32,
    /// Presentation time in seconds from the first frame.
    pub time_s: f64,
    /// Tracked objects.
    pub tracks: Vec<Tracked>,
}

impl FrameTracks {
    /// The frame as an `nv.Frame`, stamped `epoch_s` + its time.
    #[must_use]
    pub fn to_nv_frame(&self, sensor_id: &str, epoch_s: f64) -> nvframe::Frame {
        let t = epoch_s + self.time_s;
        let secs = t.floor();
        let objects = self
            .tracks
            .iter()
            .map(|tr| {
                let (speed, dir) = match tr.ground_velocity_m_s {
                    Some(v) => (v[0].hypot(v[1]) as f32, unit(v[0] as f32, v[1] as f32)),
                    None => (tr.velocity_px_s[0].hypot(tr.velocity_px_s[1]), unit(tr.velocity_px_s[0], tr.velocity_px_s[1])),
                };
                nvframe::Object {
                    id: tr.id.to_string(),
                    bbox: tr.bbox,
                    kind: tr.kind.clone(),
                    confidence: tr.score,
                    info: vec![("class".into(), tr.class.clone())],
                    speed,
                    dir,
                    coordinate: tr.ground_m.map(|g| [g[0], g[1], 0.0]),
                }
            })
            .collect();
        nvframe::Frame {
            version: "4.0".into(),
            id: self.index.to_string(),
            timestamp: (secs as i64, ((t - secs) * 1e9) as i32),
            sensor_id: sensor_id.into(),
            objects,
            info: Vec::new(),
        }
    }
}

fn unit(x: f32, y: f32) -> Option<[f32; 2]> {
    let n = x.hypot(y);
    if n > 1e-6 { Some([x / n, y / n]) } else { None }
}

fn to_ground(h: &[f64; 9], u: f64, v: f64) -> Option<[f64; 2]> {
    let w = h[6] * u + h[7] * v + h[8];
    if w.abs() < 1e-12 {
        return None;
    }
    Some([(h[0] * u + h[1] * v + h[2]) / w, (h[3] * u + h[4] * v + h[5]) / w])
}

/// What a run measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// Frames processed.
    pub frames: u32,
    /// Frame size.
    pub width: u32,
    /// Frame size.
    pub height: u32,
    /// The video's frame rate.
    pub video_fps: f32,
    /// Tracks confirmed over the run.
    pub tracks: u64,
    /// Track boxes reported, summed over frames.
    pub track_boxes: u64,
    /// Frames processed per second of wall time, decode included.
    pub fps: f64,
    /// Mean milliseconds per frame spent decoding.
    pub decode_ms: f64,
    /// Mean milliseconds per frame in the detector.
    pub detect_ms: f64,
    /// Mean milliseconds per frame in the tracker.
    pub track_ms: f64,
    /// Seconds of wall time.
    pub seconds: f64,
}

/// Run the pipeline over a video file's bytes (MP4 or WebM; H.264, H.265 or
/// AV1), calling `on_frame` with each frame's tracks in order. `on_frame`
/// returning an error stops the run with it.
///
/// # Errors
/// A file the decoder cannot read, a detector failure, or `on_frame`'s error.
pub fn track_video(
    detector: &dyn FrameDetector,
    video: &[u8],
    config: &PipelineConfig,
    mut on_frame: impl FnMut(&FrameTracks, u32) -> Result<()>,
) -> Result<Summary> {
    let info = praecise_codec::probe(video).map_err(|e| Error::Video(e.to_string()))?;
    let fps = if info.fps > 0.0 { f64::from(info.fps) } else { 30.0 };
    let total = config.max_frames.map_or(info.frames, |m| m.min(info.frames));
    let mut tracker = ByteTracker::new(config.tracker);
    let mut failed: Option<Error> = None;
    let (mut detect_s, mut track_s, mut boxes) = (0.0, 0.0, 0u64);
    let mut frames = 0u32;
    let started = Instant::now();
    let mut handed = Instant::now();
    let mut decode_s = 0.0;
    let decoded = praecise_codec::decode_each(video, |index, rgb| {
        decode_s += handed.elapsed().as_secs_f64();
        if index >= total {
            return false;
        }
        let t0 = Instant::now();
        let dets = match detector.detect(rgb, info.width, info.height, config.tracker.low_threshold) {
            Ok(d) => d,
            Err(e) => {
                failed = Some(e);
                return false;
            }
        };
        let kept: Vec<Detection> = dets.into_iter().filter(|d| config.classes.iter().any(|c| c.index == d.class)).collect();
        let t1 = Instant::now();
        let out = tracker.update(&kept);
        let t2 = Instant::now();
        detect_s += (t1 - t0).as_secs_f64();
        track_s += (t2 - t1).as_secs_f64();
        let tracks = out.iter().map(|b| describe(b, &config.classes, config.ground.as_ref(), fps)).collect::<Vec<_>>();
        boxes += tracks.len() as u64;
        let ft = FrameTracks { index, time_s: f64::from(index) / fps, tracks };
        frames += 1;
        if let Err(e) = on_frame(&ft, total) {
            failed = Some(e);
            return false;
        }
        handed = Instant::now();
        true
    });
    if let Some(e) = failed {
        return Err(e);
    }
    // the decoder refuses a stream that stops short of the container's count, unless we stopped it
    if let Err(e) = decoded
        && frames < total
    {
        return Err(Error::Video(e.to_string()));
    }
    let seconds = started.elapsed().as_secs_f64();
    let per = |s: f64| if frames == 0 { 0.0 } else { s * 1000.0 / f64::from(frames) };
    Ok(Summary {
        frames,
        width: info.width,
        height: info.height,
        video_fps: info.fps,
        tracks: tracker.confirmed(),
        track_boxes: boxes,
        fps: if seconds > 0.0 { f64::from(frames) / seconds } else { 0.0 },
        decode_ms: per(decode_s),
        detect_ms: per(detect_s),
        track_ms: per(track_s),
        seconds,
    })
}

fn describe(b: &TrackBox, classes: &[TrackedClass], ground: Option<&[f64; 9]>, fps: f64) -> Tracked {
    let class = classes.iter().find(|c| c.index == b.class);
    let velocity_px_s = [(f64::from(b.velocity[0]) * fps) as f32, (f64::from(b.velocity[1]) * fps) as f32];
    let (mut ground_m, mut ground_velocity_m_s) = (None, None);
    if let Some(h) = ground {
        let u = f64::from(b.bbox[0] + b.bbox[2]) / 2.0;
        let v = f64::from(b.bbox[3]);
        ground_m = to_ground(h, u, v);
        // the foot point one frame earlier, from the box-centre velocity
        let before = to_ground(h, u - f64::from(b.velocity[0]), v - f64::from(b.velocity[1]));
        if let (Some(now), Some(then)) = (ground_m, before) {
            ground_velocity_m_s = Some([(now[0] - then[0]) * fps, (now[1] - then[1]) * fps]);
        }
    }
    Tracked {
        id: b.id,
        class: class.map_or_else(|| b.class.to_string(), |c| c.name.clone()),
        kind: class.map_or_else(|| "Object".to_string(), |c| c.kind.clone()),
        bbox: b.bbox,
        score: b.score,
        velocity_px_s,
        ground_m,
        ground_velocity_m_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finds the bright regions of a frame: one box per connected region.
    struct Squares;

    impl FrameDetector for Squares {
        fn detect(&self, rgb: &[u8], width: u32, height: u32, _threshold: f32) -> Result<Vec<Detection>> {
            let (w, h) = (width as usize, height as usize);
            let bright = |i: usize| rgb[i * 3] > 128;
            let mut seen = vec![false; w * h];
            let mut out = Vec::new();
            for start in 0..w * h {
                if seen[start] || !bright(start) {
                    continue;
                }
                let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
                let mut stack = vec![start];
                seen[start] = true;
                while let Some(i) = stack.pop() {
                    let (x, y) = (i % w, i / w);
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                    let mut push = |j: usize| {
                        if !seen[j] && bright(j) {
                            seen[j] = true;
                            stack.push(j);
                        }
                    };
                    if x > 0 { push(i - 1); }
                    if x + 1 < w { push(i + 1); }
                    if y > 0 { push(i - w); }
                    if y + 1 < h { push(i + w); }
                }
                if (x1 - x0) * (y1 - y0) >= 64 {
                    out.push(Detection { bbox: [x0 as f32, y0 as f32, x1 as f32, y1 as f32], score: 0.9, class: 1 });
                }
            }
            Ok(out)
        }
    }

    fn clip(frames: u32) -> Vec<u8> {
        let (w, h) = (256u32, 128u32);
        let mut rgb = Vec::new();
        for f in 0..frames {
            let mut img = vec![0u8; (w * h * 3) as usize];
            // one square moving right by 16 px every 4 frames, one standing still
            for &(sx, sy) in &[(16 + 16 * (f / 4), 32u32), (192, 64)] {
                for y in sy..sy + 32 {
                    for x in sx..sx + 32 {
                        let i = ((y * w + x) * 3) as usize;
                        img[i..i + 3].copy_from_slice(&[255, 255, 255]);
                    }
                }
            }
            rgb.extend_from_slice(&img);
        }
        praecise_codec::encode(praecise_codec::EncodeOptions::default(), w, h, 10.0, &rgb, None).unwrap()
    }

    #[test]
    fn a_clip_is_decoded_detected_and_tracked() {
        let video = clip(24);
        let mut seen = Vec::new();
        let s = track_video(&Squares, &video, &PipelineConfig::default(), |f, total| {
            assert_eq!(total, 24);
            seen.push(f.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(s.frames, 24);
        assert_eq!(seen.len(), 24);
        assert_eq!(s.tracks, 2, "{:?}", seen.last());
        let last = seen.last().unwrap();
        assert_eq!(last.tracks.len(), 2);
        assert!(last.tracks.iter().all(|t| t.class == "person" && t.kind == "Person"));
        let mover = last.tracks.iter().find(|t| t.bbox[0] < 150.0).unwrap();
        // 16 px every 4 frames at 10 fps is 40 px/s
        assert!((mover.velocity_px_s[0] - 40.0).abs() < 12.0, "{:?}", mover.velocity_px_s);
        let nv = last.to_nv_frame("cam-1", 1_700_000_000.0);
        assert_eq!(nv.objects.len(), 2);
        assert_eq!(nv.timestamp.0, 1_700_000_002);
        assert!(!nv.encode().is_empty());
    }

    #[test]
    fn a_ground_homography_gives_metres() {
        // 100 px to the metre, ground y down the image
        let h = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0];
        let b = TrackBox { id: 1, class: 1, bbox: [100.0, 100.0, 140.0, 300.0], score: 0.9, velocity: [2.0, 0.0], age: 5 };
        let t = describe(&b, &coco91_people_and_vehicles(), Some(&h), 25.0);
        assert_eq!(t.ground_m, Some([1.2, 3.0]));
        let v = t.ground_velocity_m_s.unwrap();
        assert!((v[0] - 0.5).abs() < 1e-9 && v[1].abs() < 1e-9);
    }

    #[test]
    fn frames_past_the_limit_are_not_processed() {
        let video = clip(12);
        let cfg = PipelineConfig { max_frames: Some(5), ..PipelineConfig::default() };
        let s = track_video(&Squares, &video, &cfg, |_, _| Ok(())).unwrap();
        assert_eq!(s.frames, 5);
    }
}
