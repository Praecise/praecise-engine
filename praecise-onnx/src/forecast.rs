//! Time-series forecasting on ONNX Runtime.
//!
//! # Scope
//!
//! This module provides a common `ForecastModel` trait plus a
//! `GenericForecast` implementation for ONNX foundation forecasters that
//! fit a single-tensor input contract:
//!
//!   input  : `[batch, context_len]`             (f32 univariate history)
//!   output : `[batch, horizon]` or
//!            `[batch, horizon, n_quantiles]`    (point or quantile forecast)
//!
//! TimesFM 2.5 200M fits this shape.
//! Multi-input encoder-decoder forecasters (T5-based families with
//! `context_mask`, `group_ids`, `future_covariates`, …) need their own
//! adapter — `GenericForecast` will reject them at load time.
//!
//! # Threading
//!
//! Models are `Send + Sync`. ORT sessions are not safe to call
//! concurrently, so each session lives behind a `parking_lot::Mutex`.
//! Inference is synchronous; async callers run it on a blocking thread.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Configuration for a forecasting request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    /// Number of future timesteps to predict.
    pub horizon: usize,
    /// Optional quantile levels in (0, 1) to return alongside the point
    /// forecast. Empty = point forecast only.
    #[serde(default)]
    pub quantiles: Vec<f32>,
    /// Frequency hint in seconds per step (e.g. 3600 = hourly). Some
    /// models (TimesFM) use this; others ignore it. `None` = unspecified.
    #[serde(default)]
    pub frequency_seconds: Option<u64>,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            horizon: 64,
            quantiles: Vec::new(),
            frequency_seconds: None,
        }
    }
}

/// Result of a forecasting call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Point forecast: `[horizon]` f32 values.
    pub point: Vec<f32>,
    /// Quantile forecasts: outer = quantile level, inner = horizon.
    /// Empty when no quantiles were requested or the model is point-only.
    #[serde(default)]
    pub quantiles: Vec<Vec<f32>>,
    /// Quantile levels actually emitted (mirrors `ForecastConfig.quantiles`
    /// when supported, may be empty otherwise).
    #[serde(default)]
    pub quantile_levels: Vec<f32>,
    /// Total inference wall time in milliseconds.
    pub generation_time_ms: u64,
}

/// Trait for timeseries forecast models. Implementations adapt their
/// model-specific tensor layout to the common `forecast` signature.
pub trait ForecastModel: Send + Sync {
    /// Run a forecast.
    ///
    /// `history` is the observed univariate series, oldest first. Models
    /// silently truncate or right-pad to fit their native context window.
    fn forecast(&self, history: &[f32], config: &ForecastConfig) -> Result<ForecastResult>;

    /// Native context window length. Histories longer than this are
    /// truncated to the most recent `context_length` points.
    fn context_length(&self) -> usize;

    /// Maximum supported horizon. Requests for longer horizons are
    /// rejected before inference.
    fn max_horizon(&self) -> usize;

    /// Whether the model reads covariates beside the target series.
    fn takes_covariates(&self) -> bool {
        false
    }

    /// Forecast `history` with past covariates read beside it (each oldest
    /// first, aligned at the newest sample). Refused by a model that does not
    /// read them; see [`takes_covariates`](Self::takes_covariates).
    fn forecast_with_covariates(
        &self,
        _history: &[f32],
        _covariates: &[Vec<f32>],
        _config: &ForecastConfig,
    ) -> Result<ForecastResult> {
        Err(Error::InvalidModel(
            "this forecaster does not read covariates".to_string(),
        ))
    }
}

mod onnx_backend {
    use super::*;
    use ndarray::Array2;
    use ort::session::Session;
    use ort::value::Tensor;
    use std::time::Instant;

    /// Generic ONNX forecast model — fits TimesFM 2.5 and similar
    /// single-input architectures with `[1, context_len] -> [1, horizon]` shape.
    ///
    /// `Session::run` requires `&mut self` in ort 2.x, so the session is
    /// wrapped in a `Mutex` to expose a `&self` API. ONNX Runtime sessions
    /// are not safe to call concurrently from multiple threads regardless,
    /// so the mutex matches the underlying contract rather than restricting it.
    pub struct GenericForecast {
        session: parking_lot::Mutex<Session>,
        context_length: usize,
        max_horizon: usize,
        input_name: String,
        output_name: String,
        batch_size: usize,
    }

    impl GenericForecast {
        /// Load an ONNX file and inspect its input/output names.
        ///
        /// `output_name` is the prediction output to read. If `None`, the
        /// first output is used — fine for single-output forecasters.
        /// Multi-output exports (e.g. TimesFM transformers ONNX which returns
        /// `last_hidden_state`, `mean_predictions`, `full_predictions`) must
        /// pass the explicit prediction tensor name (typically
        /// `"full_predictions"` for the quantile head or `"mean_predictions"`
        /// for the point forecast).
        ///
        /// `batch_size` is the leading input dimension. For most foundation
        /// forecasters this is `None` (treated as 1). TimesFM 2.5
        /// transformers ONNX requires `Some(2)` because its decoder applies
        /// flip-invariance averaging across the batch axis (config flag
        /// `force_flip_invariance: true`). When `batch_size > 1` the history
        /// is tiled across the batch dim and the row `[0, ..]` is read from
        /// the output.
        pub fn from_onnx(
            path: impl AsRef<Path>,
            context_length: usize,
            max_horizon: usize,
            output_name: Option<String>,
            batch_size: Option<usize>,
        ) -> Result<Self> {
            let session = crate::session::build_onnx_session(path.as_ref(), "model")?;

            let input_name = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .ok_or_else(|| Error::InvalidModel("ONNX model has no inputs".to_string()))?;
            let resolved_output_name = match output_name {
                Some(name) => {
                    if !session.outputs().iter().any(|o| o.name() == name) {
                        let available: Vec<&str> =
                            session.outputs().iter().map(|o| o.name()).collect();
                        return Err(Error::InvalidModel(format!(
                            "ONNX output '{}' not found; available outputs: {:?}",
                            name, available
                        )));
                    }
                    name
                }
                None => session
                    .outputs()
                    .first()
                    .map(|o| o.name().to_string())
                    .ok_or_else(|| Error::InvalidModel("ONNX model has no outputs".to_string()))?,
            };

            let batch_size = batch_size.unwrap_or(1).max(1);

            Ok(Self {
                session: parking_lot::Mutex::new(session),
                context_length,
                max_horizon,
                input_name,
                output_name: resolved_output_name,
                batch_size,
            })
        }

        /// Truncate or right-pad a history to exactly `context_length`.
        /// Right-padding uses the last observed value (carry-forward) so
        /// the model sees a flat tail rather than zeros, which most
        /// foundation models handle better.
        fn fit_history(&self, history: &[f32]) -> Vec<f32> {
            if history.len() >= self.context_length {
                history[history.len() - self.context_length..].to_vec()
            } else {
                let mut out = Vec::with_capacity(self.context_length);
                let pad_value = *history.first().unwrap_or(&0.0);
                let pad_len = self.context_length - history.len();
                out.extend(std::iter::repeat_n(pad_value, pad_len));
                out.extend_from_slice(history);
                out
            }
        }
    }

    /// Chronos-2 — a five-input encoder whose output is quantile-major.
    ///
    /// [`GenericForecast`] handles a graph that takes one tensor and returns
    /// `[B, T]` or `[B, T, Q]`. Chronos-2 matches neither: it takes five
    /// tensors and returns `[B, Q, T]`, quantiles before time. Feeding it
    /// through the generic path would silently read a quantile index as a
    /// timestep — producing a plausible-looking series of the wrong thing —
    /// so it gets its own adapter rather than a widened generic one.
    ///
    /// Every fact below is read off the published graph
    /// (`OpenSTEF/chronos-2-small-onnx`) and its metadata sidecar, and
    /// confirmed by running the model:
    ///
    /// | Input | Type | Shape |
    /// |---|---|---|
    /// | `context` | f32 | `[batch, context_length]` |
    /// | `group_ids` | i64 | `[batch]` |
    /// | `attention_mask` | **f32** | `[batch, context_length]` |
    /// | `future_covariates` | f32 | `[batch, 672]` |
    /// | `future_covariates_mask` | f32 | `[batch, 672]` |
    ///
    /// Output `quantile_preds` is `[batch, 13, 672]`.
    ///
    /// Two of those are easy to get wrong. `attention_mask` is **float**, not
    /// the int64 every other transformer export uses. And the covariate
    /// tensors cannot be zero-length even when there are no covariates: the
    /// graph reshapes them to `[batch, 42, 16]`, so a `[batch, 0]` tensor
    /// fails inside the model rather than at the boundary. They are passed
    /// full-width with an all-zero mask, which is how "no covariates" is
    /// expressed.
    pub struct Chronos2Forecast {
        session: parking_lot::Mutex<Session>,
        context_length: usize,
    }

    /// Timesteps the graph always emits: `horizon_patches (42) ×
    /// output_patch_size (16)`. Fixed in the exported graph, not a request
    /// parameter — a shorter horizon is a truncation of this.
    const CHRONOS2_HORIZON: usize = 672;

    /// The graph's `context` input is patch-tokenized in groups of 16
    /// (`/model/patch_1/Reshape` folds it to `[1, -1, 16]`), so a context
    /// length that is not a multiple of 16 fails ONNX Runtime's reshape
    /// with a shape-mismatch error rather than anything forecast-shaped.
    const CHRONOS2_INPUT_PATCH: usize = 16;

    /// Quantile levels the head emits, in order.
    const CHRONOS2_QUANTILES: [f32; 13] = [
        0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
    ];

    /// Index of the median in [`CHRONOS2_QUANTILES`].
    ///
    /// Named rather than computed as `len / 2`: that happens to be 6 here, and
    /// would keep being 6 for a head whose levels were not centred on 0.5,
    /// which is how a point forecast quietly becomes a different quantile.
    const CHRONOS2_MEDIAN_IDX: usize = 6;

    impl Chronos2Forecast {
        /// Load the graph and check it is the shape this adapter expects.
        pub fn from_onnx(path: impl AsRef<Path>, context_length: usize) -> Result<Self> {
            let session = crate::session::build_onnx_session(path.as_ref(), "model")?;
            // Fail at load rather than at the first forecast. An operator who
            // pointed this at the wrong artifact should learn when they serve
            // it, not when a caller asks for a prediction.
            for required in [
                "context",
                "group_ids",
                "attention_mask",
                "future_covariates",
                "future_covariates_mask",
            ] {
                if !session.inputs().iter().any(|i| i.name() == required) {
                    let found: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
                    return Err(Error::InvalidModel(format!(
                        "not a Chronos-2 graph: missing input '{required}'; found {found:?}"
                    )));
                }
            }
            if !session.outputs().iter().any(|o| o.name() == "quantile_preds") {
                let found: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
                return Err(Error::InvalidModel(format!(
                    "not a Chronos-2 graph: missing output 'quantile_preds'; found {found:?}"
                )));
            }
            Ok(Self {
                session: parking_lot::Mutex::new(session),
                context_length,
            })
        }

        /// Truncate to the most recent `context_length` points, or return the
        /// history as-is when it is shorter.
        fn fit_history<'a>(&self, history: &'a [f32]) -> &'a [f32] {
            if history.len() > self.context_length {
                &history[history.len() - self.context_length..]
            } else {
                history
            }
        }

        /// Left-pad `fitted` up to the next multiple of
        /// [`CHRONOS2_INPUT_PATCH`], carry-forwarding the earliest observed
        /// value the way [`GenericForecast`]'s own right-pad does. The graph's
        /// context dimension is NOT actually dynamic below patch granularity —
        /// see [`CHRONOS2_INPUT_PATCH`] — so a short or oddly-sized series has
        /// to be padded, not passed at its own length. Padding goes on the
        /// left (oldest end) so the real observations stay anchored to "now"
        /// at the end of the sequence, matching how the model reads the final
        /// timestep as the forecast origin. Returns the padded series and how
        /// many of its leading positions are padding (for the attention mask).
        fn pad_to_patch(fitted: &[f32]) -> (Vec<f32>, usize) {
            let ctx = fitted.len();
            let remainder = ctx % CHRONOS2_INPUT_PATCH;
            let pad_len = if remainder == 0 {
                0
            } else {
                CHRONOS2_INPUT_PATCH - remainder
            };
            if pad_len == 0 {
                return (fitted.to_vec(), 0);
            }
            let pad_value = *fitted.first().unwrap_or(&0.0);
            let mut padded = Vec::with_capacity(ctx + pad_len);
            padded.extend(std::iter::repeat_n(pad_value, pad_len));
            padded.extend_from_slice(fitted);
            (padded, pad_len)
        }
    }

    impl Chronos2Forecast {
        /// Forecast the first of `rows` with the rest as its covariates: every
        /// row goes into one group, so the model attends across them. Rows are
        /// aligned at their newest sample and cut to the shortest.
        fn run_group(&self, rows: &[&[f32]], config: &ForecastConfig) -> Result<ForecastResult> {
            if rows.is_empty() || rows.iter().any(|r| r.is_empty()) {
                return Err(Error::InvalidModel("history is empty".to_string()));
            }
            if config.horizon == 0 {
                return Err(Error::InvalidModel("horizon must be > 0".to_string()));
            }
            if config.horizon > CHRONOS2_HORIZON {
                return Err(Error::InvalidModel(format!(
                    "horizon {} exceeds model max {}",
                    config.horizon, CHRONOS2_HORIZON
                )));
            }

            let start = Instant::now();
            let shortest = rows.iter().map(|r| r.len()).min().unwrap_or(0);
            let fitted: Vec<&[f32]> = rows
                .iter()
                .map(|r| self.fit_history(&r[r.len() - shortest..]))
                .collect();
            let padded: Vec<(Vec<f32>, usize)> =
                fitted.iter().map(|f| Self::pad_to_patch(f)).collect();
            let batch = padded.len();
            let ctx = padded[0].0.len();

            let flat: Vec<f32> = padded.iter().flat_map(|(p, _)| p.iter().copied()).collect();
            let context = Array2::<f32>::from_shape_vec((batch, ctx), flat)
                .map_err(|e| Error::InferenceError(format!("context shape: {}", e)))?;
            // Real positions are 1, padded (leading) positions are 0 so the
            // graph does not treat carried-forward padding as observed data.
            // Float, not int64.
            let mut attention = Array2::<f32>::from_elem((batch, ctx), 1.0f32);
            for (row, (_, pad_len)) in padded.iter().enumerate() {
                for i in 0..*pad_len {
                    attention[[row, i]] = 0.0;
                }
            }
            // One group: the target and its covariates are read together.
            let groups = Array2::<i64>::zeros((1, batch))
                .into_shape_with_order(vec![batch])
                .map_err(|e| Error::InferenceError(format!("group_ids shape: {}", e)))?;
            // Full-width and all-zero: the graph reshapes these to [B, 42, 16],
            // so they cannot be empty, and a zero mask says no future values
            // are known for any row.
            let cov = Array2::<f32>::zeros((batch, CHRONOS2_HORIZON));
            let cov_mask = Array2::<f32>::zeros((batch, CHRONOS2_HORIZON));

            let t_context = Tensor::from_array(context)
                .map_err(|e| Error::InferenceError(format!("ORT tensor context: {}", e)))?;
            let t_groups = Tensor::from_array(groups)
                .map_err(|e| Error::InferenceError(format!("ORT tensor group_ids: {}", e)))?;
            let t_attention = Tensor::from_array(attention)
                .map_err(|e| Error::InferenceError(format!("ORT tensor attention_mask: {}", e)))?;
            let t_cov = Tensor::from_array(cov).map_err(|e| {
                Error::InferenceError(format!("ORT tensor future_covariates: {}", e))
            })?;
            let t_cov_mask = Tensor::from_array(cov_mask).map_err(|e| {
                Error::InferenceError(format!("ORT tensor future_covariates_mask: {}", e))
            })?;

            let (dims, raw): (Vec<i64>, Vec<f32>) = {
                let mut session = self.session.lock();
                let outputs = session
                    .run(ort::inputs![
                        "context" => t_context,
                        "group_ids" => t_groups,
                        "attention_mask" => t_attention,
                        "future_covariates" => t_cov,
                        "future_covariates_mask" => t_cov_mask,
                    ])
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;
                let out = outputs.get("quantile_preds").ok_or_else(|| {
                    Error::InferenceError("missing output 'quantile_preds'".to_string())
                })?;
                let (shape, data) = out
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract: {}", e)))?;
                (shape.iter().copied().collect(), data.to_vec())
            };

            // `[batch, quantile, time]` — quantile-major, which is the whole
            // reason this adapter exists.
            let [b, q, t] = dims.as_slice() else {
                return Err(Error::InferenceError(format!(
                    "unexpected output shape {dims:?}, expected [B, Q, T]"
                )));
            };
            if *b as usize != batch {
                return Err(Error::InferenceError(format!(
                    "output batch dim is {b}, expected {batch}"
                )));
            }
            let (q, t) = (*q as usize, *t as usize);
            if q <= CHRONOS2_MEDIAN_IDX {
                return Err(Error::InferenceError(format!(
                    "output has {q} quantiles, too few to contain the median at index {CHRONOS2_MEDIAN_IDX}"
                )));
            }
            let h = t.min(config.horizon);

            // Row 0 of the batch (the target) spans `0 .. q * t`; quantile qi
            // occupies `qi * t .. (qi + 1) * t` within it.
            let mut quantiles: Vec<Vec<f32>> = Vec::with_capacity(q);
            for qi in 0..q {
                quantiles.push(raw[qi * t..qi * t + h].to_vec());
            }
            let point = quantiles[CHRONOS2_MEDIAN_IDX].clone();

            Ok(ForecastResult {
                point,
                quantiles,
                quantile_levels: CHRONOS2_QUANTILES[..q.min(CHRONOS2_QUANTILES.len())].to_vec(),
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }
    }

    impl ForecastModel for Chronos2Forecast {
        fn forecast(&self, history: &[f32], config: &ForecastConfig) -> Result<ForecastResult> {
            self.run_group(&[history], config)
        }

        fn takes_covariates(&self) -> bool {
            true
        }

        fn forecast_with_covariates(
            &self,
            history: &[f32],
            covariates: &[Vec<f32>],
            config: &ForecastConfig,
        ) -> Result<ForecastResult> {
            let mut rows: Vec<&[f32]> = vec![history];
            rows.extend(covariates.iter().map(Vec::as_slice));
            self.run_group(&rows, config)
        }

        fn context_length(&self) -> usize {
            self.context_length
        }

        fn max_horizon(&self) -> usize {
            CHRONOS2_HORIZON
        }
    }

    /// Grouped patch-decoder forecaster with 21 native quantiles and
    /// known-future covariate inputs.
    ///
    /// The graph takes raw observations (its scaler is inside the graph), a
    /// dynamic context and a compute horizon that must be a multiple of
    /// [`T0_HORIZON_MULTIPLE`]; the horizon is selected by the width of the
    /// `future_covariate_future` input, which stays present with zero rows
    /// when no known-future covariates are supplied. The output is
    /// time-major: `[rows, compute_horizon, quantiles]`.
    pub struct T0Forecast {
        session: parking_lot::Mutex<Session>,
        context_length: usize,
        max_horizon: usize,
    }

    /// The compute horizon granularity of the grouped graph.
    pub const T0_HORIZON_MULTIPLE: usize = 32;

    /// Quantile levels in output order.
    pub const T0_QUANTILES: [f32; 21] = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 0.99,
    ];

    /// Index of the 0.5 level in [`T0_QUANTILES`].
    pub const T0_MEDIAN_IDX: usize = 10;

    const T0_INPUTS: [&str; 5] = [
        "target_context",
        "target_group_ids",
        "future_covariate_context",
        "future_covariate_future",
        "future_covariate_group_ids",
    ];

    /// Rounds a requested horizon up to the graph's compute horizon.
    pub fn t0_compute_horizon(horizon: usize) -> usize {
        horizon.div_ceil(T0_HORIZON_MULTIPLE) * T0_HORIZON_MULTIPLE
    }

    impl T0Forecast {
        pub fn from_onnx(
            path: impl AsRef<Path>,
            context_length: usize,
            max_horizon: usize,
        ) -> Result<Self> {
            let session = crate::session::build_onnx_session(path.as_ref(), "model")?;
            for required in T0_INPUTS {
                if !session.inputs().iter().any(|i| i.name() == required) {
                    let found: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
                    return Err(Error::InvalidModel(format!(
                        "not a grouped t0 graph: missing input '{required}'; found {found:?}"
                    )));
                }
            }
            if !session.outputs().iter().any(|o| o.name() == "quantiles") {
                let found: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
                return Err(Error::InvalidModel(format!(
                    "not a grouped t0 graph: missing output 'quantiles'; found {found:?}"
                )));
            }
            if max_horizon == 0 || max_horizon % T0_HORIZON_MULTIPLE != 0 {
                return Err(Error::InvalidModel(format!(
                    "max_horizon {max_horizon} is not a positive multiple of {T0_HORIZON_MULTIPLE}"
                )));
            }
            Ok(Self {
                session: parking_lot::Mutex::new(session),
                context_length,
                max_horizon,
            })
        }

        /// Forecast the first of `rows`; the remaining rows are past-only
        /// covariates placed in the same group, so the graph attends across
        /// them. Rows are aligned at their newest sample and cut to the
        /// shortest, then to the context length.
        fn run_group(&self, rows: &[&[f32]], config: &ForecastConfig) -> Result<ForecastResult> {
            if rows.is_empty() || rows.iter().any(|r| r.is_empty()) {
                return Err(Error::InvalidModel("history is empty".to_string()));
            }
            if config.horizon == 0 {
                return Err(Error::InvalidModel("horizon must be > 0".to_string()));
            }
            if config.horizon > self.max_horizon {
                return Err(Error::InvalidModel(format!(
                    "horizon {} exceeds model max {}",
                    config.horizon, self.max_horizon
                )));
            }
            let start = Instant::now();
            let ctx = rows
                .iter()
                .map(|r| r.len())
                .min()
                .unwrap_or(0)
                .min(self.context_length);
            let batch = rows.len();
            let compute_h = t0_compute_horizon(config.horizon);

            let flat: Vec<f32> = rows
                .iter()
                .flat_map(|r| r[r.len() - ctx..].iter().copied())
                .collect();
            let context = Array2::<f32>::from_shape_vec((batch, ctx), flat)
                .map_err(|e| Error::InferenceError(format!("context shape: {}", e)))?;
            let groups = ndarray::Array1::<i32>::zeros(batch);
            let cov_context = Array2::<f32>::zeros((0, ctx));
            let cov_future = Array2::<f32>::zeros((0, compute_h));
            let cov_groups = ndarray::Array1::<i32>::zeros(0);

            let tensor_err = |name: &str, e: ort::Error| {
                Error::InferenceError(format!("ORT tensor {name}: {e}"))
            };
            let t_context =
                Tensor::from_array(context).map_err(|e| tensor_err("target_context", e))?;
            let t_groups =
                Tensor::from_array(groups).map_err(|e| tensor_err("target_group_ids", e))?;
            let t_cov_context = Tensor::from_array(cov_context)
                .map_err(|e| tensor_err("future_covariate_context", e))?;
            let t_cov_future = Tensor::from_array(cov_future)
                .map_err(|e| tensor_err("future_covariate_future", e))?;
            let t_cov_groups = Tensor::from_array(cov_groups)
                .map_err(|e| tensor_err("future_covariate_group_ids", e))?;

            let (dims, raw): (Vec<i64>, Vec<f32>) = {
                let mut session = self.session.lock();
                let outputs = session
                    .run(ort::inputs![
                        "target_context" => t_context,
                        "target_group_ids" => t_groups,
                        "future_covariate_context" => t_cov_context,
                        "future_covariate_future" => t_cov_future,
                        "future_covariate_group_ids" => t_cov_groups,
                    ])
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;
                let out = outputs.get("quantiles").ok_or_else(|| {
                    Error::InferenceError("missing output 'quantiles'".to_string())
                })?;
                let (shape, data) = out
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract: {}", e)))?;
                (shape.iter().copied().collect(), data.to_vec())
            };
            let quantiles = t0_target_quantiles(&dims, &raw, batch, config.horizon)?;
            let point = quantiles[T0_MEDIAN_IDX].clone();
            Ok(ForecastResult {
                point,
                quantiles,
                quantile_levels: T0_QUANTILES.to_vec(),
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }
    }

    /// Reads row 0 of a time-major `[rows, T, Q]` output into one series per
    /// quantile level, each cut to `horizon`.
    pub(crate) fn t0_target_quantiles(
        dims: &[i64],
        raw: &[f32],
        batch: usize,
        horizon: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let [b, t, q] = dims else {
            return Err(Error::InferenceError(format!(
                "unexpected output shape {dims:?}, expected [rows, T, Q]"
            )));
        };
        let (b, t, q) = (*b as usize, *t as usize, *q as usize);
        if b != batch {
            return Err(Error::InferenceError(format!(
                "output batch dim is {b}, expected {batch}"
            )));
        }
        if q != T0_QUANTILES.len() {
            return Err(Error::InferenceError(format!(
                "output has {q} quantiles, expected {}",
                T0_QUANTILES.len()
            )));
        }
        if t < horizon || raw.len() < t * q {
            return Err(Error::InferenceError(format!(
                "output horizon {t} is shorter than the requested {horizon}"
            )));
        }
        Ok((0..q)
            .map(|qi| (0..horizon).map(|ti| raw[ti * q + qi]).collect())
            .collect())
    }

    impl ForecastModel for T0Forecast {
        fn forecast(&self, history: &[f32], config: &ForecastConfig) -> Result<ForecastResult> {
            self.run_group(&[history], config)
        }

        fn takes_covariates(&self) -> bool {
            true
        }

        fn forecast_with_covariates(
            &self,
            history: &[f32],
            covariates: &[Vec<f32>],
            config: &ForecastConfig,
        ) -> Result<ForecastResult> {
            let mut rows: Vec<&[f32]> = vec![history];
            rows.extend(covariates.iter().map(Vec::as_slice));
            self.run_group(&rows, config)
        }

        fn context_length(&self) -> usize {
            self.context_length
        }

        fn max_horizon(&self) -> usize {
            self.max_horizon
        }
    }

    impl ForecastModel for GenericForecast {
        fn forecast(&self, history: &[f32], config: &ForecastConfig) -> Result<ForecastResult> {
            if history.is_empty() {
                return Err(Error::InvalidModel("history is empty".to_string()));
            }
            if config.horizon == 0 {
                return Err(Error::InvalidModel("horizon must be > 0".to_string()));
            }
            if config.horizon > self.max_horizon {
                return Err(Error::InvalidModel(format!(
                    "horizon {} exceeds model max {}",
                    config.horizon, self.max_horizon
                )));
            }

            let start = Instant::now();
            let fitted = self.fit_history(history);
            // Tile the fitted history across the batch dimension. For
            // single-batch models this is a single copy; for flip-invariant
            // exports (TimesFM 2.5 requires batch=2) the same row is
            // repeated and the output is read from row 0.
            let mut tiled = Vec::with_capacity(self.batch_size * self.context_length);
            for _ in 0..self.batch_size {
                tiled.extend_from_slice(&fitted);
            }
            let arr = Array2::<f32>::from_shape_vec((self.batch_size, self.context_length), tiled)
                .map_err(|e| Error::InferenceError(format!("input shape: {}", e)))?;

            let input_tensor = Tensor::from_array(arr)
                .map_err(|e| Error::InferenceError(format!("ORT tensor: {}", e)))?;

            // Lock the session for the duration of run() — ORT sessions are
            // not safe to call concurrently. Extract the result into owned
            // memory before releasing the lock.
            let (dims, raw): (Vec<i64>, Vec<f32>) = {
                let mut session = self.session.lock();
                let outputs = session
                    .run(ort::inputs![self.input_name.as_str() => input_tensor])
                    .map_err(|e| Error::InferenceError(format!("ORT run: {}", e)))?;

                let out_value = outputs.get(self.output_name.as_str()).ok_or_else(|| {
                    Error::InferenceError(format!("missing output '{}'", self.output_name))
                })?;

                let (shape, data) = out_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::InferenceError(format!("ORT extract: {}", e)))?;
                // `Shape` derefs to `&[i64]`; copy out so we can drop the lock.
                (shape.iter().copied().collect(), data.to_vec())
            };

            // Expect [B, horizon] or [B, horizon, n_quantiles] where B is
            // `self.batch_size`. Always read row 0 — when B > 1 the input
            // was tiled across the batch dim so row 0 is the answer for
            // this caller's history. Slice off exactly `config.horizon`
            // rows from the time dimension.
            let (point, quantiles) = match dims.as_slice() {
                [b, t] => {
                    let b = *b as usize;
                    if b == 0 {
                        return Err(Error::InferenceError("output batch dim is 0".to_string()));
                    }
                    let t = *t as usize;
                    let h = t.min(config.horizon);
                    // Row 0 starts at index 0 and runs for `t` elements.
                    (raw[..h].to_vec(), Vec::new())
                }
                [b, t, q] => {
                    let b = *b as usize;
                    if b == 0 {
                        return Err(Error::InferenceError("output batch dim is 0".to_string()));
                    }
                    let t = *t as usize;
                    let q = *q as usize;
                    let h = t.min(config.horizon);
                    // Layout [batch, time, quantile], row-major. Row 0
                    // occupies indices `0 .. t * q`.
                    let median_idx = q / 2;
                    let mut point = Vec::with_capacity(h);
                    let mut q_series: Vec<Vec<f32>> = vec![Vec::with_capacity(h); q];
                    for ti in 0..h {
                        for qi in 0..q {
                            let v = raw[ti * q + qi];
                            q_series[qi].push(v);
                            if qi == median_idx {
                                point.push(v);
                            }
                        }
                    }
                    (point, q_series)
                }
                other => {
                    return Err(Error::InferenceError(format!(
                        "unexpected output shape {:?}, expected [B, T] or [B, T, Q]",
                        other
                    )));
                }
            };

            Ok(ForecastResult {
                point,
                quantiles,
                quantile_levels: config.quantiles.clone(),
                generation_time_ms: start.elapsed().as_millis() as u64,
            })
        }

        fn context_length(&self) -> usize {
            self.context_length
        }

        fn max_horizon(&self) -> usize {
            self.max_horizon
        }
    }

    #[cfg(test)]
    mod pad_tests {
        use super::*;

        // The bug: `chronos2_reads_a_quantile_major_output` (in the parent
        // module's own test suite) used a 256-point history — already a
        // multiple of 16 — so it never exercised any length that needed
        // padding at all. These don't need the ONNX artifact: `pad_to_patch`
        // is pure data-shaping, exercised directly.

        #[test]
        fn already_aligned_length_is_untouched() {
            let fitted: Vec<f32> = (0..32).map(|i| i as f32).collect();
            let (padded, pad_len) = Chronos2Forecast::pad_to_patch(&fitted);
            assert_eq!(pad_len, 0);
            assert_eq!(padded, fitted);
        }

        #[test]
        fn short_of_a_patch_boundary_is_left_padded_to_it() {
            // 20 points -- the exact length that crashed ONNX Runtime's
            // reshape before this fix, per the live error observed serving
            // a real forecast request.
            let fitted: Vec<f32> = (0..20).map(|i| i as f32).collect();
            let (padded, pad_len) = Chronos2Forecast::pad_to_patch(&fitted);
            // 20 is 4 past a patch boundary, so the next multiple of 16 is 32,
            // not 24. An earlier revision of this test asserted 4/24 — an
            // 8-wide rounding — while its own message said "multiple of 16",
            // which 24 is not. The graph settles it: `/model/patch_1/Reshape`
            // concatenates a literal 16 into its target shape, folding context
            // to [batch, -1, 16]. A 24-long context would fail that reshape
            // exactly as 20 did.
            assert_eq!(pad_len, 12);
            assert_eq!(padded.len(), 32, "rounds up to the next multiple of 16");
            assert_eq!(padded.len() % CHRONOS2_INPUT_PATCH, 0);
            // Padding on the LEFT: the real, most-recent observations stay
            // anchored at the end of the sequence.
            assert_eq!(&padded[12..], fitted.as_slice());
        }

        #[test]
        fn padding_carries_forward_the_earliest_observed_value() {
            let fitted = vec![10.0f32, 11.0, 12.0];
            let (padded, pad_len) = Chronos2Forecast::pad_to_patch(&fitted);
            assert_eq!(pad_len, 13);
            assert!(padded[..pad_len].iter().all(|&v| v == 10.0));
        }

        #[test]
        fn a_single_point_is_padded_to_one_full_patch() {
            let fitted = vec![42.0f32];
            let (padded, pad_len) = Chronos2Forecast::pad_to_patch(&fitted);
            assert_eq!(pad_len, 15);
            assert_eq!(padded.len(), CHRONOS2_INPUT_PATCH);
        }
    }
}

pub use onnx_backend::{
    Chronos2Forecast, GenericForecast, T0_HORIZON_MULTIPLE, T0_MEDIAN_IDX, T0_QUANTILES,
    T0Forecast, t0_compute_horizon,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t0_compute_horizon_rounds_up_to_the_patch() {
        assert_eq!(t0_compute_horizon(1), 32);
        assert_eq!(t0_compute_horizon(32), 32);
        assert_eq!(t0_compute_horizon(50), 64);
        assert_eq!(t0_compute_horizon(1024), 1024);
    }

    #[test]
    fn t0_reads_row_zero_of_a_time_major_output() {
        // Two rows, T = 4, Q = 21; value encodes (row, t, q).
        let (b, t, q) = (2usize, 4usize, 21usize);
        let raw: Vec<f32> = (0..b * t * q)
            .map(|i| {
                let (r, rem) = (i / (t * q), i % (t * q));
                (r * 1000 + (rem / q) * 100 + rem % q) as f32
            })
            .collect();
        let dims = [b as i64, t as i64, q as i64];
        let out = onnx_backend::t0_target_quantiles(&dims, &raw, b, 3).unwrap();
        assert_eq!(out.len(), 21);
        assert_eq!(out[T0_MEDIAN_IDX], vec![10.0, 110.0, 210.0]);
        assert_eq!(out[20], vec![20.0, 120.0, 220.0]);
        assert!(onnx_backend::t0_target_quantiles(&dims, &raw, b, 5).is_err());
        assert!(onnx_backend::t0_target_quantiles(&[2, 4, 13], &raw, b, 3).is_err());
    }

    /// Runs the published grouped graph and compares every quantile with the
    /// values the reference runtime produced for the same input
    /// (`PRAECISE_T0_EXPECTED`: JSON `{"history": [...], "horizon": n,
    /// "quantiles": [[...] x 21]}`).
    #[test]
    #[ignore = "needs the T0 graph and its reference output: PRAECISE_T0_ONNX, PRAECISE_T0_EXPECTED"]
    fn t0_matches_the_reference_runtime() {
        let path = std::env::var("PRAECISE_T0_ONNX").expect("PRAECISE_T0_ONNX");
        let expected = std::env::var("PRAECISE_T0_EXPECTED").expect("PRAECISE_T0_EXPECTED");
        crate::session::install(
            crate::session::OnnxSettings::resolve(&crate::session::OnnxConfig::default(), 8),
            crate::session::Gpus::default(),
        );
        let exp: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(expected).unwrap()).unwrap();
        let history: Vec<f32> = serde_json::from_value(exp["history"].clone()).unwrap();
        let horizon = exp["horizon"].as_u64().unwrap() as usize;
        let want: Vec<Vec<f32>> = serde_json::from_value(exp["quantiles"].clone()).unwrap();
        let model = onnx_backend::T0Forecast::from_onnx(&path, 8192, 1024).unwrap();
        let cfg = ForecastConfig {
            horizon,
            ..Default::default()
        };
        let out = model.forecast(&history, &cfg).unwrap();
        assert_eq!(out.quantiles.len(), want.len());
        let mut worst = 0f32;
        for (got, want) in out.quantiles.iter().zip(&want) {
            assert_eq!(got.len(), horizon);
            for (g, w) in got.iter().zip(want) {
                worst = worst.max((g - w).abs() / w.abs().max(1.0));
            }
        }
        assert!(worst < 1e-4, "max relative error {worst}");
    }

    /// Exercises the real Chronos-2 adapter when the artifact is present.
    ///
    /// Set `PRAECISE_CHRONOS2_ONNX` to `chronos-2-small.onnx`. The assertions
    /// are the ones that caught the three things this adapter exists to get right: the
    /// quantile-major output layout, covariate tensors that cannot be empty,
    /// and a context length that isn't a multiple of the graph's 16-wide
    /// input patch (`onnx_backend::pad_tests` covers the padding function in
    /// isolation; this covers it end to end against the real graph).
    #[test]
    #[ignore = "needs the Chronos-2 graph: PRAECISE_CHRONOS2_ONNX"]
    fn chronos2_reads_a_quantile_major_output() {
        let path = std::path::PathBuf::from(
            std::env::var("PRAECISE_CHRONOS2_ONNX").expect("PRAECISE_CHRONOS2_ONNX"),
        );
        crate::session::install(
            crate::session::OnnxSettings::resolve(&crate::session::OnnxConfig::default(), 8),
            crate::session::Gpus::default(),
        );
        let model = onnx_backend::Chronos2Forecast::from_onnx(&path, 5760)
            .expect("the artifact is a Chronos-2 graph");
        assert_eq!(model.max_horizon(), 672);

        // A clean sine, deliberately NOT a multiple of 16 (250, not 256) — the
        // exact shape of history a real caller sends, since nothing about a
        // trader's or economist's actual series happens to align to the
        // graph's patch size. Before the padding fix this failed ONNX
        // Runtime's reshape outright rather than returning a forecast.
        let history: Vec<f32> = (0..250).map(|i| (i as f32 / 10.0).sin()).collect();
        let cfg = ForecastConfig {
            horizon: 24,
            ..Default::default()
        };
        let out = model.forecast(&history, &cfg).expect("forecast runs");

        assert_eq!(out.point.len(), 24, "horizon is honoured");
        assert_eq!(out.quantiles.len(), 13, "the head emits 13 quantiles");
        for q in &out.quantiles {
            assert_eq!(q.len(), 24);
        }
        assert_eq!(out.quantile_levels.len(), 13);
        assert!(
            (out.quantile_levels[6] - 0.5).abs() < 1e-6,
            "median is index 6"
        );

        // The point forecast must be the median series, not some other row.
        assert_eq!(out.point, out.quantiles[6]);

        // Quantiles are non-decreasing at each timestep. This is the assertion
        // that fails loudly if the [B, Q, T] output were read as [B, T, Q]:
        // transposed, the "quantiles" become slices across time and stop being
        // ordered.
        for t in 0..24 {
            for qi in 1..13 {
                assert!(
                    out.quantiles[qi][t] >= out.quantiles[qi - 1][t] - 1e-3,
                    "quantiles out of order at t={t}, q={qi}: {} < {}",
                    out.quantiles[qi][t],
                    out.quantiles[qi - 1][t]
                );
            }
        }

        // A horizon past the single-pass width is refused rather than
        // silently truncated.
        let too_far = ForecastConfig {
            horizon: 1000,
            ..Default::default()
        };
        assert!(model.forecast(&history, &too_far).is_err());
    }

    #[test]
    fn forecast_config_default() {
        let c = ForecastConfig::default();
        assert_eq!(c.horizon, 64);
        assert!(c.quantiles.is_empty());
        assert!(c.frequency_seconds.is_none());
    }

    #[test]
    fn forecast_result_serializes() {
        let r = ForecastResult {
            point: vec![1.0, 2.0, 3.0],
            quantiles: Vec::new(),
            quantile_levels: Vec::new(),
            generation_time_ms: 42,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"point\""));
        assert!(json.contains("\"generation_time_ms\":42"));
        let back: ForecastResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.point, vec![1.0, 2.0, 3.0]);
    }
}
