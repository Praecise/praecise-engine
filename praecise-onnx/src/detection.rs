//! Object detection on ONNX Runtime.
//!
//! Supports two open-license, NMS-free DETR families:
//!
//! - **RF-DETR** (Roboflow, Apache 2.0, ICLR 2026 — the first real-time
//!   detector to break >60 AP on COCO). Single input `"input"` of shape
//!   `[1, 3, H, W]` with ImageNet normalization. Two outputs: `"dets"` —
//!   cxcywh boxes in `[0, 1]` normalized image coordinates of shape
//!   `[1, num_queries, 4]`, and `"labels"` — raw class logits of shape
//!   `[1, num_queries, num_classes]` that the client must sigmoid.
//!   RF-DETR was trained with **90-class COCO indexing** (the original
//!   Detectron2 layout that includes 11 deprecated "background" slots), so
//!   such a model has `num_classes: 90`.
//!
//! - **D-FINE** (Peterande/D-FINE, Apache 2.0). Two inputs `"images"` of
//!   shape `[1, 3, 640, 640]` and `"orig_target_sizes"` of dtype int64,
//!   shape `[1, 2]` carrying the unresized `(H, W)` of the source image.
//!   Pixel values are scaled to `[0, 1]` *only* — no ImageNet normalization.
//!   Three outputs already in fully postprocessed form: `"labels"` (int64),
//!   `"boxes"` (xyxy pixel coordinates relative to the original image),
//!   and `"scores"` (post-sigmoid, sorted descending). D-FINE uses standard
//!   80-class COCO indexing.
//!
//! # Pre/post-processing
//!
//! - **Decode** PNG/JPEG/WebP via the `image` crate.
//! - **Resize** to `(input_size, input_size)` with Lanczos3.
//! - **NHWC → NCHW** float layout.
//! - **RF-DETR:** normalize with ImageNet mean/std. Postprocess by
//!   sigmoiding the class logits, taking the per-query top-1 class, filtering
//!   by `score_threshold`, converting cxcywh → xyxy, and scaling to the
//!   *original* image pixel coordinates.
//! - **D-FINE:** scale to `[0, 1]` only and pass `orig_target_sizes`
//!   directly; the graph emits final xyxy pixel boxes already in the
//!   original image's coordinate frame.
//!
//! # Threading
//!
//! Models are `Send + Sync`. ORT sessions are not safe to call
//! concurrently, so each session lives behind a `parking_lot::Mutex`.
//! Inference is synchronous; async callers run it on a blocking thread.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A single detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// Bounding box in input-image pixel coordinates: (x0, y0, x1, y1).
    pub bbox: [f32; 4],
    /// Class index (model-dependent — RF-DETR uses 0..90, D-FINE uses 0..80).
    pub label_id: u32,
    /// Optional class label string when the detector was loaded with
    /// a labels file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Confidence in [0, 1].
    pub score: f32,
}

/// Result of a detection call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectResult {
    pub detections: Vec<Detection>,
    pub generation_time_ms: u64,
}

/// Trait for object detectors.
pub trait Detector: Send + Sync {
    fn detect(&self, image_bytes: &[u8], score_threshold: f32) -> Result<DetectResult>;
    fn input_size(&self) -> u32;
    fn num_classes(&self) -> u32;
}

/// Which DETR family the loaded ONNX graph implements. Drives both
/// preprocessing (normalization vs. scale-only) and postprocessing
/// (raw logits vs. baked-in NMS-free output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetrFamily {
    /// RF-DETR — single `"input"` tensor, ImageNet norm, raw logit outputs.
    RfDetr,
    /// D-FINE — `"images"` + `"orig_target_sizes"` int64 input, no norm,
    /// postprocessed output (labels int64, xyxy pixel boxes, sigmoid scores).
    DFine,
}

mod onnx_backend {
    use super::*;
    use image::imageops::FilterType;
    use ndarray::{Array2, Array4};
    use ort::session::Session;
    use ort::value::Tensor;
    use std::time::Instant;

    /// ImageNet normalization constants — RF-DETR was trained with these.
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    /// Generic ONNX detector covering both RF-DETR and D-FINE.
    ///
    /// `Session::run` requires `&mut self` in ort 2.x, so the session is
    /// wrapped in a `Mutex` to expose a `&self` API. ONNX Runtime sessions
    /// are not safe to call concurrently from multiple threads regardless,
    /// so the mutex matches the underlying contract.
    pub struct GenericDetrDetector {
        session: parking_lot::Mutex<Session>,
        family: DetrFamily,
        input_size: u32,
        num_classes: u32,
        /// Tensor name for the image input (RF-DETR: "input", D-FINE: "images").
        image_input_name: String,
        /// Tensor name for the original-size int64 input. Only used by D-FINE.
        orig_size_input_name: Option<String>,
    }

    impl GenericDetrDetector {
        /// Load an ONNX file and inspect its inputs to pick names. Falls back
        /// to family-canonical names ("input" / "images" / "orig_target_sizes")
        /// when introspection is ambiguous.
        pub fn from_onnx(
            path: impl AsRef<Path>,
            family: DetrFamily,
            input_size: u32,
            num_classes: u32,
        ) -> Result<Self> {
            let session = crate::session::build_onnx_session(path.as_ref(), "model")?;

            let mut image_input_name = None;
            let mut orig_size_input_name = None;

            for inp in session.inputs() {
                let n = inp.name();
                if n == "orig_target_sizes" {
                    orig_size_input_name = Some(inp.name().to_string());
                } else if image_input_name.is_none() {
                    image_input_name = Some(inp.name().to_string());
                }
            }

            let image_input_name = image_input_name.unwrap_or_else(|| match family {
                DetrFamily::RfDetr => "input".to_string(),
                DetrFamily::DFine => "images".to_string(),
            });

            if family == DetrFamily::DFine && orig_size_input_name.is_none() {
                orig_size_input_name = Some("orig_target_sizes".to_string());
            }

            Ok(Self {
                session: parking_lot::Mutex::new(session),
                family,
                input_size,
                num_classes,
                image_input_name,
                orig_size_input_name,
            })
        }

        /// Decode → resize → NCHW float. Returns the `[1, 3, H, W]` tensor and
        /// the original `(width, height)` pair for downstream coordinate
        /// scaling.
        fn preprocess(&self, image_bytes: &[u8]) -> Result<(Array4<f32>, u32, u32)> {
            let img = image::load_from_memory(image_bytes)
                .map_err(|e| Error::InvalidModel(format!("image decode: {}", e)))?
                .to_rgb8();
            let orig_w = img.width();
            let orig_h = img.height();

            let size = self.input_size;
            let resized = image::imageops::resize(&img, size, size, FilterType::Lanczos3);

            let h = size as usize;
            let w = size as usize;
            let mut arr = Array4::<f32>::zeros((1, 3, h, w));

            let (apply_norm, mean, std) = match self.family {
                DetrFamily::RfDetr => (true, IMAGENET_MEAN, IMAGENET_STD),
                // D-FINE: scale to [0, 1] only; no per-channel norm.
                DetrFamily::DFine => (false, [0.0; 3], [1.0; 3]),
            };

            for y in 0..h {
                for x in 0..w {
                    let px = resized.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let scaled = px[c] as f32 / 255.0;
                        let v = if apply_norm {
                            (scaled - mean[c]) / std[c]
                        } else {
                            scaled
                        };
                        arr[[0, c, y, x]] = v;
                    }
                }
            }

            Ok((arr, orig_w, orig_h))
        }
    }

    impl GenericDetrDetector {
        /// Detect objects in one decoded frame: `rgb` holds `width * height`
        /// row-major 8-bit RGB pixels, as a video decoder returns them. The
        /// frame is resampled bilinearly to the model's input size; boxes come
        /// back in the frame's pixel coordinates.
        ///
        /// # Errors
        /// When `rgb` does not hold `width * height` pixels, or inference fails.
        pub fn detect_rgb(&self, rgb: &[u8], width: u32, height: u32, score_threshold: f32) -> Result<DetectResult> {
            let start = Instant::now();
            let arr = preprocess_rgb(self.family, self.input_size, rgb, width, height)?;
            let image_tensor = Tensor::from_array(arr)
                .map_err(|e| Error::InferenceError(format!("ORT tensor: {}", e)))?;
            let detections = match self.family {
                DetrFamily::RfDetr => self.run_rfdetr(image_tensor, width, height, score_threshold)?,
                DetrFamily::DFine => self.run_dfine(image_tensor, width, height, score_threshold)?,
            };
            Ok(DetectResult {
                detections,
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }
    }

    /// Bilinear resample of a raw RGB frame straight into a `[1, 3, S, S]`
    /// model input, normalised for the family.
    pub(super) fn preprocess_rgb(
        family: DetrFamily,
        input_size: u32,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Array4<f32>> {
        let (w, h) = (width as usize, height as usize);
        if w == 0 || h == 0 || rgb.len() != w * h * 3 {
            return Err(Error::InvalidModel(format!(
                "a {width}x{height} RGB frame needs {} bytes, got {}",
                w * h * 3,
                rgb.len()
            )));
        }
        let size = input_size as usize;
        let (mean, std) = match family {
            DetrFamily::RfDetr => (IMAGENET_MEAN, IMAGENET_STD),
            DetrFamily::DFine => ([0.0; 3], [1.0; 3]),
        };
        let scale = [1.0 / (255.0 * std[0]), 1.0 / (255.0 * std[1]), 1.0 / (255.0 * std[2])];
        let offset = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]];
        // source pixels and weight for each output column or row, pixel centres aligned
        let taps = |n_out: usize, n_in: usize| -> Vec<(usize, usize, f32)> {
            (0..n_out)
                .map(|o| {
                    let x = ((o as f32 + 0.5) * n_in as f32 / n_out as f32 - 0.5).max(0.0);
                    let x0 = (x.floor() as usize).min(n_in - 1);
                    let x1 = (x0 + 1).min(n_in - 1);
                    (x0, x1, x - x0 as f32)
                })
                .collect()
        };
        let cols = taps(size, w);
        let rows = taps(size, h);
        let mut arr = Array4::<f32>::zeros((1, 3, size, size));
        let plane = size * size;
        let out = arr.as_slice_mut().expect("standard layout");
        for (oy, &(y0, y1, fy)) in rows.iter().enumerate() {
            let r0 = &rgb[y0 * w * 3..(y0 + 1) * w * 3];
            let r1 = &rgb[y1 * w * 3..(y1 + 1) * w * 3];
            for (ox, &(x0, x1, fx)) in cols.iter().enumerate() {
                for c in 0..3 {
                    let a = f32::from(r0[x0 * 3 + c]) * (1.0 - fx) + f32::from(r0[x1 * 3 + c]) * fx;
                    let b = f32::from(r1[x0 * 3 + c]) * (1.0 - fx) + f32::from(r1[x1 * 3 + c]) * fx;
                    let v = a * (1.0 - fy) + b * fy;
                    out[c * plane + oy * size + ox] = v * scale[c] - offset[c];
                }
            }
        }
        Ok(arr)
    }

    /// Numerically stable sigmoid.
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// Convert a single cxcywh-normalized box to xyxy pixel coordinates.
    #[inline]
    fn cxcywh_norm_to_xyxy_pixels(
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        img_w: f32,
        img_h: f32,
    ) -> [f32; 4] {
        let x0 = (cx - w * 0.5) * img_w;
        let y0 = (cy - h * 0.5) * img_h;
        let x1 = (cx + w * 0.5) * img_w;
        let y1 = (cy + h * 0.5) * img_h;
        [
            x0.clamp(0.0, img_w),
            y0.clamp(0.0, img_h),
            x1.clamp(0.0, img_w),
            y1.clamp(0.0, img_h),
        ]
    }

    impl Detector for GenericDetrDetector {
        fn detect(&self, image_bytes: &[u8], score_threshold: f32) -> Result<DetectResult> {
            if image_bytes.is_empty() {
                return Err(Error::InvalidModel("image bytes are empty".to_string()));
            }

            let start = Instant::now();
            let (arr, orig_w, orig_h) = self.preprocess(image_bytes)?;

            let image_tensor = Tensor::from_array(arr)
                .map_err(|e| Error::InferenceError(format!("ORT tensor: {}", e)))?;

            let detections = match self.family {
                DetrFamily::RfDetr => {
                    self.run_rfdetr(image_tensor, orig_w, orig_h, score_threshold)?
                }
                DetrFamily::DFine => {
                    self.run_dfine(image_tensor, orig_w, orig_h, score_threshold)?
                }
            };

            Ok(DetectResult {
                detections,
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }

        fn input_size(&self) -> u32 {
            self.input_size
        }

        fn num_classes(&self) -> u32 {
            self.num_classes
        }
    }

    impl GenericDetrDetector {
        fn run_rfdetr(
            &self,
            image_tensor: Tensor<f32>,
            orig_w: u32,
            orig_h: u32,
            score_threshold: f32,
        ) -> Result<Vec<Detection>> {
            let (dets_shape, dets, labels_shape, labels) = {
                let mut session = self.session.lock();
                let outputs = session
                    .run(ort::inputs![self.image_input_name.as_str() => image_tensor])
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;

                let dets_value = outputs.get("dets").ok_or_else(|| {
                    Error::InferenceError("RF-DETR missing 'dets' output".to_string())
                })?;
                let (ds, dd) = dets_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract dets: {}", e)))?;
                let dets_shape: Vec<i64> = ds.iter().copied().collect();
                let dets: Vec<f32> = dd.to_vec();

                let labels_value = outputs.get("labels").ok_or_else(|| {
                    Error::InferenceError("RF-DETR missing 'labels' output".to_string())
                })?;
                let (ls, ld) = labels_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract labels: {}", e)))?;
                let labels_shape: Vec<i64> = ls.iter().copied().collect();
                let labels: Vec<f32> = ld.to_vec();

                (dets_shape, dets, labels_shape, labels)
            };

            // dets: [1, Q, 4], labels: [1, Q, C]
            let (q_d, four) = match dets_shape.as_slice() {
                [1, q, 4] => (*q as usize, 4_usize),
                other => {
                    return Err(Error::InferenceError(format!(
                        "RF-DETR unexpected dets shape {:?}, expected [1, Q, 4]",
                        other
                    )));
                }
            };
            let (q_l, c) = match labels_shape.as_slice() {
                [1, q, c] => (*q as usize, *c as usize),
                other => {
                    return Err(Error::InferenceError(format!(
                        "RF-DETR unexpected labels shape {:?}, expected [1, Q, C]",
                        other
                    )));
                }
            };
            if q_d != q_l {
                return Err(Error::InferenceError(format!(
                    "RF-DETR query count mismatch: dets={} labels={}",
                    q_d, q_l
                )));
            }
            let _ = four;

            let img_w = orig_w as f32;
            let img_h = orig_h as f32;

            let mut out = Vec::new();
            for q in 0..q_d {
                // Per-query top-1 class via sigmoid (DETR's NMS-free scoring).
                let logit_base = q * c;
                let mut best_score = 0.0_f32;
                let mut best_class = 0_usize;
                for k in 0..c {
                    let s = sigmoid(labels[logit_base + k]);
                    if s > best_score {
                        best_score = s;
                        best_class = k;
                    }
                }
                if best_score < score_threshold {
                    continue;
                }

                let box_base = q * 4;
                let cx = dets[box_base];
                let cy = dets[box_base + 1];
                let bw = dets[box_base + 2];
                let bh = dets[box_base + 3];
                let bbox = cxcywh_norm_to_xyxy_pixels(cx, cy, bw, bh, img_w, img_h);

                out.push(Detection {
                    bbox,
                    label_id: best_class as u32,
                    label: None,
                    score: best_score,
                });
            }

            // Sort descending by score so callers can take top-K cheaply.
            out.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(out)
        }

        fn run_dfine(
            &self,
            image_tensor: Tensor<f32>,
            orig_w: u32,
            orig_h: u32,
            score_threshold: f32,
        ) -> Result<Vec<Detection>> {
            // D-FINE expects orig_target_sizes as int64 [1, 2] in (H, W) order
            // — matches Peterande/D-FINE export.
            let orig_arr =
                Array2::<i64>::from_shape_vec((1, 2), vec![orig_h as i64, orig_w as i64])
                    .map_err(|e| Error::InferenceError(format!("orig_sizes shape: {}", e)))?;
            let orig_tensor = Tensor::from_array(orig_arr)
                .map_err(|e| Error::InferenceError(format!("ORT orig_sizes tensor: {}", e)))?;

            let orig_name = self
                .orig_size_input_name
                .clone()
                .unwrap_or_else(|| "orig_target_sizes".to_string());

            // Build the feed manually because the two inputs have different
            // element types (f32 image, i64 orig sizes).
            let feed: Vec<(String, ort::session::SessionInputValue<'_>)> = vec![
                (self.image_input_name.clone(), image_tensor.into()),
                (orig_name, orig_tensor.into()),
            ];

            let (labels_shape, labels, boxes_shape, boxes, scores_shape, scores) = {
                let mut session = self.session.lock();
                let outputs = session
                    .run(feed)
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;

                let labels_v = outputs.get("labels").ok_or_else(|| {
                    Error::InferenceError("D-FINE missing 'labels' output".to_string())
                })?;
                let (ls, ld) = labels_v
                    .try_extract_tensor::<i64>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract labels: {}", e)))?;
                let labels_shape: Vec<i64> = ls.iter().copied().collect();
                let labels: Vec<i64> = ld.to_vec();

                let boxes_v = outputs.get("boxes").ok_or_else(|| {
                    Error::InferenceError("D-FINE missing 'boxes' output".to_string())
                })?;
                let (bs, bd) = boxes_v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract boxes: {}", e)))?;
                let boxes_shape: Vec<i64> = bs.iter().copied().collect();
                let boxes: Vec<f32> = bd.to_vec();

                let scores_v = outputs.get("scores").ok_or_else(|| {
                    Error::InferenceError("D-FINE missing 'scores' output".to_string())
                })?;
                let (ss, sd) = scores_v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract scores: {}", e)))?;
                let scores_shape: Vec<i64> = ss.iter().copied().collect();
                let scores: Vec<f32> = sd.to_vec();

                (
                    labels_shape,
                    labels,
                    boxes_shape,
                    boxes,
                    scores_shape,
                    scores,
                )
            };

            let n_labels = match labels_shape.as_slice() {
                [1, n] => *n as usize,
                [n] => *n as usize,
                other => {
                    return Err(Error::InferenceError(format!(
                        "D-FINE unexpected labels shape {:?}",
                        other
                    )));
                }
            };
            let n_scores = match scores_shape.as_slice() {
                [1, n] => *n as usize,
                [n] => *n as usize,
                other => {
                    return Err(Error::InferenceError(format!(
                        "D-FINE unexpected scores shape {:?}",
                        other
                    )));
                }
            };
            let n_boxes = match boxes_shape.as_slice() {
                [1, n, 4] => *n as usize,
                [n, 4] => *n as usize,
                other => {
                    return Err(Error::InferenceError(format!(
                        "D-FINE unexpected boxes shape {:?}, expected [.., N, 4]",
                        other
                    )));
                }
            };
            if n_labels != n_scores || n_labels != n_boxes {
                return Err(Error::InferenceError(format!(
                    "D-FINE shape mismatch: labels={} scores={} boxes={}",
                    n_labels, n_scores, n_boxes
                )));
            }

            let mut out = Vec::with_capacity(n_labels);
            for i in 0..n_labels {
                let score = scores[i];
                // D-FINE emits scores sorted desc — early-exit once we drop below threshold.
                if score < score_threshold {
                    break;
                }
                let bb_base = i * 4;
                let bbox = [
                    boxes[bb_base],
                    boxes[bb_base + 1],
                    boxes[bb_base + 2],
                    boxes[bb_base + 3],
                ];
                let label_id = labels[i].max(0) as u32;
                out.push(Detection {
                    bbox,
                    label_id,
                    label: None,
                    score,
                });
            }

            Ok(out)
        }
    }
}

pub use onnx_backend::GenericDetrDetector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_raw_frame_is_resampled_and_normalised() {
        // a 4x2 frame: left half black, right half white
        let mut rgb = Vec::new();
        for _y in 0..2 {
            for x in 0..4 {
                let v = if x < 2 { 0u8 } else { 255u8 };
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        let arr = onnx_backend::preprocess_rgb(DetrFamily::DFine, 4, &rgb, 4, 2).unwrap();
        assert_eq!(arr.shape(), &[1, 3, 4, 4]);
        for c in 0..3 {
            for y in 0..4 {
                assert!((arr[[0, c, y, 0]] - 0.0).abs() < 1e-6);
                assert!((arr[[0, c, y, 3]] - 1.0).abs() < 1e-6);
            }
        }
        let norm = onnx_backend::preprocess_rgb(DetrFamily::RfDetr, 4, &rgb, 4, 2).unwrap();
        assert!((norm[[0, 0, 0, 0]] - (0.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((norm[[0, 2, 0, 3]] - (1.0 - 0.406) / 0.225).abs() < 1e-5);
        assert!(onnx_backend::preprocess_rgb(DetrFamily::RfDetr, 4, &rgb[1..], 4, 2).is_err());
    }

    #[test]
    fn detection_serializes_round_trip() {
        let d = Detection {
            bbox: [1.0, 2.0, 3.0, 4.0],
            label_id: 0,
            label: Some("person".into()),
            score: 0.95,
        };
        let s = serde_json::to_string(&d).unwrap();
        let back: Detection = serde_json::from_str(&s).unwrap();
        assert_eq!(back.bbox, d.bbox);
        assert_eq!(back.label_id, d.label_id);
    }
}
