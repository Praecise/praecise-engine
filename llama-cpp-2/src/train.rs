//! Gradient-only training on the serving graph.
//!
//! A [`LlamaContext`] computes parameter gradients of a cross-entropy objective on the same
//! graph, kernels and devices that serve the model; the caller owns the optimizer and writes the
//! updated parameters back with [`TrainTensor::write_f32`].
//!
//! This module also writes `LoRA` adapters as GGUF ([`write_lora_gguf`]) and builds a model from
//! GGUF metadata with caller-provided weights ([`LlamaModel::from_metadata`]).

use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::token::LlamaToken;

/// A tensor of a model or of a `LoRA` adapter, or a gradient accumulator.
///
/// The handle borrows nothing: it stays valid while the model, adapter or context that owns the
/// tensor is alive, which the caller guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrainTensor {
    ptr: NonNull<llama_cpp_sys_2::ggml_tensor>,
}

impl TrainTensor {
    fn from_ptr(ptr: *mut llama_cpp_sys_2::ggml_tensor) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    /// Tensor name.
    #[must_use]
    pub fn name(&self) -> String {
        let name = unsafe { CStr::from_ptr(llama_cpp_sys_2::ggml_get_name(self.ptr.as_ptr())) };
        name.to_string_lossy().into_owned()
    }

    /// Shape in ggml order (innermost dimension first), trailing ones included.
    #[must_use]
    pub fn shape(&self) -> [i64; 4] {
        unsafe { (*self.ptr.as_ptr()).ne }
    }

    /// Number of elements.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        usize::try_from(unsafe { llama_cpp_sys_2::ggml_nelements(self.ptr.as_ptr()) }).unwrap_or(0)
    }

    /// Whether the tensor holds F32 values.
    #[must_use]
    pub fn is_f32(&self) -> bool {
        unsafe { (*self.ptr.as_ptr()).type_ == llama_cpp_sys_2::GGML_TYPE_F32 }
    }

    /// Whether the tensor is a trainable parameter of a gradient context.
    #[must_use]
    pub fn is_param(&self) -> bool {
        unsafe { (*self.ptr.as_ptr()).flags & llama_cpp_sys_2::GGML_TENSOR_FLAG_PARAM.cast_signed() != 0 }
    }

    /// Copies the values out of the tensor's device buffer.
    ///
    /// # Errors
    ///
    /// [`TrainError::NotF32`] for a tensor that is not F32.
    pub fn read_f32(&self) -> Result<Vec<f32>, TrainError> {
        if !self.is_f32() {
            return Err(TrainError::NotF32(self.name()));
        }
        let mut out = vec![0.0f32; self.n_elements()];
        unsafe {
            llama_cpp_sys_2::ggml_backend_tensor_get(
                self.ptr.as_ptr(),
                out.as_mut_ptr().cast::<c_void>(),
                0,
                out.len() * size_of::<f32>(),
            );
        }
        Ok(out)
    }

    /// Overwrites the tensor's values in its device buffer.
    ///
    /// # Errors
    ///
    /// [`TrainError::NotF32`] for a tensor that is not F32, [`TrainError::Length`] when `values`
    /// does not have one entry per element.
    pub fn write_f32(&self, values: &[f32]) -> Result<(), TrainError> {
        if !self.is_f32() {
            return Err(TrainError::NotF32(self.name()));
        }
        if values.len() != self.n_elements() {
            return Err(TrainError::Length { expected: self.n_elements(), got: values.len() });
        }
        unsafe {
            llama_cpp_sys_2::ggml_backend_tensor_set(
                self.ptr.as_ptr(),
                values.as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(values),
            );
        }
        Ok(())
    }
}

/// Errors of gradient-only training.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TrainError {
    /// The context already computes gradients.
    #[error("the context is already initialized for training")]
    AlreadyTraining,
    /// The context's memory has no differentiable path.
    #[error("the context memory has no differentiable path")]
    NoDifferentiableMemory,
    /// The KV cache is not F32.
    #[error("training needs an F32 KV cache")]
    KvCacheNotF32,
    /// The selection matched no F32 parameter.
    #[error("no parameter selected")]
    NoParameters,
    /// A weight is in a CPU extra buffer type (a repacked layout) that only runs the forward pass.
    #[error("a weight is in a repacked buffer; load the model with use_extra_bufts disabled")]
    ExtraBufferWeights,
    /// A pass before initialization.
    #[error("the context is not initialized for training")]
    NotTraining,
    /// The sequence is empty or longer than the context's ubatch.
    #[error("the sequence is empty or longer than the ubatch")]
    SequenceLength,
    /// The graph needs a gradient through an op without a backward; the op is in the log.
    #[error("the graph needs a gradient through an op without a backward")]
    NoBackward,
    /// A tensor that must be F32 is not.
    #[error("tensor {0} is not F32")]
    NotF32(String),
    /// A buffer of the wrong length.
    #[error("expected {expected} values, got {got}")]
    Length {
        /// Required length.
        expected: usize,
        /// Supplied length.
        got: usize,
    },
    /// A failure reported by the engine.
    #[error("engine: {0}")]
    Engine(String),
}

unsafe extern "C" fn select_trampoline(tensor: *const llama_cpp_sys_2::ggml_tensor, ud: *mut c_void) -> bool {
    let select = unsafe { &mut *ud.cast::<&mut dyn FnMut(&str) -> bool>() };
    let name = unsafe { CStr::from_ptr((*tensor).name.as_ptr()) };
    select(&name.to_string_lossy())
}

/// What the targets of a gradient pass mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradLoss {
    /// Distributions over the vocabulary; the pass differentiates
    /// `-sum_rows sum(targets * log_softmax(logits)) / n_tokens`.
    CrossEntropy,
    /// The gradient of the caller's objective with respect to the logits; the pass
    /// differentiates `sum(targets * logits)`.
    WeightedSum,
}

impl LlamaContext<'_> {
    /// Makes the context compute gradients of `loss`. `select` receives each F32 tensor name of
    /// the model and of the `LoRA` adapters set on the context, and returns whether it is
    /// trainable.
    ///
    /// # Errors
    ///
    /// The refusal reasons of [`TrainError`].
    pub fn grad_init(&mut self, loss: GradLoss, mut select: impl FnMut(&str) -> bool) -> Result<(), TrainError> {
        let mut dyn_select: &mut dyn FnMut(&str) -> bool = &mut select;
        let loss_type = match loss {
            GradLoss::CrossEntropy => llama_cpp_sys_2::GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
            GradLoss::WeightedSum => llama_cpp_sys_2::GGML_OPT_LOSS_TYPE_WEIGHTED_SUM,
        };
        let rc = unsafe {
            llama_cpp_sys_2::llama_opt_grad_init(
                self.context.as_ptr(),
                self.model.model.as_ptr(),
                loss_type,
                Some(select_trampoline),
                (&raw mut dyn_select).cast::<c_void>(),
            )
        };
        match rc {
            0 => Ok(()),
            -1 => Err(TrainError::AlreadyTraining),
            -2 => Err(TrainError::NoDifferentiableMemory),
            -3 => Err(TrainError::KvCacheNotF32),
            -4 => Err(TrainError::NoParameters),
            -6 => Err(TrainError::ExtraBufferWeights),
            other => Err(TrainError::Engine(format!("llama_opt_grad_init returned {other}"))),
        }
    }

    /// Number of floats in the outputs of a pass over `n_tokens`: the logits
    /// (`n_tokens x n_vocab`) of a generative context; for an embedding context the per-token
    /// embeddings (no pooling), the rank scores, or the pooled embedding.
    #[must_use]
    pub fn grad_output_size(&self, n_tokens: usize) -> usize {
        let n = i32::try_from(n_tokens).unwrap_or(i32::MAX);
        usize::try_from(unsafe { llama_cpp_sys_2::llama_opt_grad_output_size(self.context.as_ptr(), n) }).unwrap_or(0)
    }

    /// Runs one sequence at positions `0..tokens.len()` from an empty memory. `outputs` receives
    /// the forward outputs ([`Self::grad_output_size`] floats). With `targets` (as many floats) the
    /// pass also adds the gradient of the context's [`GradLoss`] to the parameter gradients.
    ///
    /// # Errors
    ///
    /// The refusal reasons of [`TrainError`], and [`TrainError::Length`] for a target or output
    /// buffer of the wrong size.
    pub fn grad_sequence(
        &mut self,
        tokens: &[LlamaToken],
        targets: Option<&[f32]>,
        outputs: Option<&mut [f32]>,
    ) -> Result<(), TrainError> {
        let want = self.grad_output_size(tokens.len());
        if let Some(t) = targets {
            if t.len() != want {
                return Err(TrainError::Length { expected: want, got: t.len() });
            }
        }
        if let Some(l) = outputs.as_deref() {
            if l.len() != want {
                return Err(TrainError::Length { expected: want, got: l.len() });
            }
        }
        let n_tokens = i32::try_from(tokens.len()).map_err(|_| TrainError::SequenceLength)?;
        let ids: Vec<llama_cpp_sys_2::llama_token> = tokens.iter().map(|t| t.0).collect();
        let rc = unsafe {
            llama_cpp_sys_2::llama_opt_grad_sequence(
                self.context.as_ptr(),
                ids.as_ptr(),
                n_tokens,
                targets.map_or(std::ptr::null(), <[f32]>::as_ptr),
                outputs.map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr),
            )
        };
        match rc {
            0 => Ok(()),
            -1 => Err(TrainError::NotTraining),
            -2 => Err(TrainError::SequenceLength),
            -3 => Err(TrainError::NoBackward),
            other => Err(TrainError::Engine(format!("llama_opt_grad_sequence returned {other}"))),
        }
    }

    /// The gradient accumulator of a parameter, `None` if it is not a parameter or before the
    /// first pass with targets.
    #[must_use]
    pub fn grad(&self, param: &TrainTensor) -> Option<TrainTensor> {
        TrainTensor::from_ptr(unsafe { llama_cpp_sys_2::llama_opt_grad(self.context.as_ptr(), param.ptr.as_ptr()) })
    }

    /// Sets every parameter gradient to zero.
    pub fn grad_reset(&mut self) {
        unsafe { llama_cpp_sys_2::llama_opt_grad_reset(self.context.as_ptr()) }
    }

    /// Distinct op names of the graph the last [`Self::grad_sequence`] evaluated, forward and
    /// backward, in ascending order.
    #[must_use]
    pub fn grad_graph_ops(&self) -> Vec<String> {
        let ctx = self.context.as_ptr();
        let n = unsafe { llama_cpp_sys_2::llama_opt_grad_n_ops(ctx) };
        (0..n)
            .filter_map(|i| {
                let p = unsafe { llama_cpp_sys_2::llama_opt_grad_op(ctx, i) };
                (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
            })
            .collect()
    }
}

impl LlamaModel {
    /// A weight of the model by its GGUF name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<TrainTensor> {
        let name = CString::new(name).ok()?;
        TrainTensor::from_ptr(unsafe { llama_cpp_sys_2::llama_model_get_tensor(self.model.as_ptr(), name.as_ptr()) })
    }

    /// Builds a model from GGUF metadata, with every weight created as F32 and filled by `fill`
    /// (tensor name, element count) -> values.
    ///
    /// # Errors
    ///
    /// [`TrainError::Engine`] when the metadata does not describe a model the engine can build.
    pub fn from_metadata(
        _: &LlamaBackend,
        metadata: &GgufMetadata,
        mut fill: impl FnMut(&str, usize) -> Vec<f32>,
        params: &LlamaModelParams,
    ) -> Result<Self, TrainError> {
        unsafe extern "C" fn set_data(tensor: *mut llama_cpp_sys_2::ggml_tensor, ud: *mut c_void) {
            let fill = unsafe { &mut *ud.cast::<&mut dyn FnMut(&str, usize) -> Vec<f32>>() };
            let t = TrainTensor { ptr: unsafe { NonNull::new_unchecked(tensor) } };
            let values = fill(&t.name(), t.n_elements());
            t.write_f32(&values).expect("weights created from metadata are F32 and filled in full");
        }
        let mut dyn_fill: &mut dyn FnMut(&str, usize) -> Vec<f32> = &mut fill;
        let model = unsafe {
            llama_cpp_sys_2::llama_model_init_from_user(
                metadata.ctx.as_ptr(),
                Some(set_data),
                (&raw mut dyn_fill).cast::<c_void>(),
                params.params,
            )
        };
        NonNull::new(model)
            .map(|model| LlamaModel { model, served_lora: None })
            .ok_or_else(|| TrainError::Engine("the metadata does not describe a buildable model".into()))
    }
}

impl LlamaLoraAdapter {
    /// The `lora_a` and `lora_b` tensors of every adapted weight, in a stable order.
    #[must_use]
    pub fn tensors(&self) -> Vec<TrainTensor> {
        let adapter = self.lora_adapter.as_ptr();
        let n = unsafe { llama_cpp_sys_2::llama_adapter_lora_n_tensors(adapter) };
        (0..n)
            .filter_map(|i| TrainTensor::from_ptr(unsafe { llama_cpp_sys_2::llama_adapter_lora_get_tensor(adapter, i) }))
            .collect()
    }
}

/// GGUF key-value metadata under construction.
#[derive(Debug)]
pub struct GgufMetadata {
    ctx: NonNull<llama_cpp_sys_2::gguf_context>,
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufMetadata {
    /// Empty metadata.
    ///
    /// # Panics
    ///
    /// When the engine cannot allocate a context.
    #[must_use]
    pub fn new() -> Self {
        let ctx = NonNull::new(unsafe { llama_cpp_sys_2::gguf_init_empty() }).expect("gguf_init_empty");
        Self { ctx }
    }

    fn key(key: &str) -> CString {
        CString::new(key).expect("GGUF keys contain no NUL byte")
    }

    /// Sets a string value.
    ///
    /// # Panics
    ///
    /// When the key or the value contains a NUL byte.
    pub fn set_str(&mut self, key: &str, value: &str) -> &mut Self {
        let (k, v) = (Self::key(key), CString::new(value).expect("GGUF values contain no NUL byte"));
        unsafe { llama_cpp_sys_2::gguf_set_val_str(self.ctx.as_ptr(), k.as_ptr(), v.as_ptr()) };
        self
    }

    /// Sets a `u32` value.
    pub fn set_u32(&mut self, key: &str, value: u32) -> &mut Self {
        let k = Self::key(key);
        unsafe { llama_cpp_sys_2::gguf_set_val_u32(self.ctx.as_ptr(), k.as_ptr(), value) };
        self
    }

    /// Sets an `f32` value.
    pub fn set_f32(&mut self, key: &str, value: f32) -> &mut Self {
        let k = Self::key(key);
        unsafe { llama_cpp_sys_2::gguf_set_val_f32(self.ctx.as_ptr(), k.as_ptr(), value) };
        self
    }
}

impl Drop for GgufMetadata {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::gguf_free(self.ctx.as_ptr()) }
    }
}

/// One adapted weight of a `LoRA` adapter, in the layout the engine loads.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraWeight {
    /// Name of the base weight, for example `blk.0.attn_q.weight`.
    pub target: String,
    /// Input width of the base weight (its ggml `ne[0]`).
    pub n_in: usize,
    /// Output width of the base weight (its ggml `ne[1]`).
    pub n_out: usize,
    /// Rank.
    pub rank: usize,
    /// `lora_a`, ggml shape `[n_in, rank]`, row-major in ggml order.
    pub a: Vec<f32>,
    /// `lora_b`, ggml shape `[rank, n_out]`, row-major in ggml order.
    pub b: Vec<f32>,
}

/// Writes an F32 `LoRA` adapter for a base model of architecture `arch` as GGUF. The update applied
/// to a weight is `(alpha / rank) * B A x`.
///
/// # Errors
///
/// [`TrainError::Length`] for a tensor of the wrong size, [`TrainError::Engine`] when the file
/// cannot be written.
///
/// # Panics
///
/// When a target name contains a NUL byte.
pub fn write_lora_gguf(path: &Path, arch: &str, alpha: f32, weights: &[LoraWeight]) -> Result<(), TrainError> {
    for w in weights {
        if w.a.len() != w.n_in * w.rank {
            return Err(TrainError::Length { expected: w.n_in * w.rank, got: w.a.len() });
        }
        if w.b.len() != w.rank * w.n_out {
            return Err(TrainError::Length { expected: w.rank * w.n_out, got: w.b.len() });
        }
    }
    let data_bytes: usize = weights.iter().map(|w| (w.a.len() + w.b.len()) * size_of::<f32>()).sum();
    let overhead = unsafe { llama_cpp_sys_2::ggml_tensor_overhead() };
    let params = llama_cpp_sys_2::ggml_init_params {
        mem_size: data_bytes + (2 * weights.len() + 1) * (overhead + 64),
        mem_buffer: std::ptr::null_mut(),
        no_alloc: false,
    };
    let ctx = NonNull::new(unsafe { llama_cpp_sys_2::ggml_init(params) })
        .ok_or_else(|| TrainError::Engine("ggml_init failed".into()))?;

    let mut meta = GgufMetadata::new();
    meta.set_str("general.type", "adapter")
        .set_str("general.architecture", arch)
        .set_str("adapter.type", "lora")
        .set_f32("adapter.lora.alpha", alpha);

    let add = |name: String, ne0: usize, ne1: usize, values: &[f32]| {
        let t = unsafe {
            llama_cpp_sys_2::ggml_new_tensor_2d(
                ctx.as_ptr(),
                llama_cpp_sys_2::GGML_TYPE_F32,
                i64::try_from(ne0).expect("tensor width fits i64"),
                i64::try_from(ne1).expect("tensor height fits i64"),
            )
        };
        let cname = CString::new(name).expect("tensor names contain no NUL byte");
        unsafe {
            llama_cpp_sys_2::ggml_set_name(t, cname.as_ptr());
            std::ptr::copy_nonoverlapping(values.as_ptr(), (*t).data.cast::<f32>(), values.len());
            llama_cpp_sys_2::gguf_add_tensor(meta.ctx.as_ptr(), t);
        }
    };
    for w in weights {
        add(format!("{}.lora_a", w.target), w.n_in, w.rank, &w.a);
        add(format!("{}.lora_b", w.target), w.rank, w.n_out, &w.b);
    }

    let cpath = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| TrainError::Engine(e.to_string()))?;
    let ok = unsafe { llama_cpp_sys_2::gguf_write_to_file(meta.ctx.as_ptr(), cpath.as_ptr(), false) };
    unsafe { llama_cpp_sys_2::ggml_free(ctx.as_ptr()) };
    if ok { Ok(()) } else { Err(TrainError::Engine(format!("cannot write {}", path.display()))) }
}
