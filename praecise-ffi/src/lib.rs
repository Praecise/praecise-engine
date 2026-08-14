//! Praecise Engine — C ABI surface.
//!
//! A stable `extern "C"` boundary over praecise-runtime's backend-agnostic
//! types so consumers in other languages (C, C++, Python via cffi, Go via cgo,
//! …) can speak to the engine without a Rust toolchain. This crate is
//! **header-generatable** with `cbindgen`.
//!
//! ## Scope
//!
//! This module marshals the *stable* value types — [`PraeciseConfig`],
//! [`PraeciseInferenceResult`], [`PraeciseStopReason`], the version, and the
//! thread-local error channel. It builds and links **without** a backend
//! (no llama.cpp compile), because these types carry no backend state.
//!
//! The generation entry point (`praecise_generate`) binds once the synchronous
//! runtime orchestrator moves into `praecise-runtime`: today that orchestrator
//! (context creation, prefill, the decode loop, tool/multimodal/speculative
//! paths) still lives in the first consumer's runtime. The leaf
//! pieces it drives (`BatchEngine`, `LoadedModel`, speculative decode) are
//! already in praecise-runtime; the C ABI grows a `generate` call over them
//! without any change to the marshalling below.
//!
//! ## Memory ownership
//!
//! Every pointer this library returns that the caller must release has a
//! matching `*_free` function. Strings returned by the result accessors are
//! owned by the [`PraeciseInferenceResult`] and stay valid until it is freed;
//! the caller must not free them individually.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::sync::OnceLock;

use praecise_runtime::{GenerationConfig, InferenceResult, StopReason};

/// Version of this C ABI contract. Bump on any breaking change to a struct
/// layout or function signature; a consumer can compare it against the value it
/// was compiled for.
pub const PRAECISE_ABI_VERSION: u32 = 1;

thread_local! {
    /// Last error message on the calling thread, if any. Set by fallible entry
    /// points, read via [`praecise_last_error`].
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Record `msg` as the calling thread's last error (see [`praecise_last_error`]).
// Wired by the generation entry point once the runtime orchestrator moves into
// praecise-runtime; the error channel it reports through is part of the ABI now
// so consumers can code against it from day one.
#[allow(dead_code)]
fn set_last_error(msg: &str) {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("error (contained NUL)").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

/// Return the calling thread's last error message as a NUL-terminated C string,
/// or `NULL` if there has been none. The pointer is valid until the next
/// fallible call on the same thread; copy it if you need to keep it.
#[must_use]
pub extern "C" fn praecise_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr())
    })
}

/// Return this ABI's [`PRAECISE_ABI_VERSION`].
#[must_use]
pub extern "C" fn praecise_abi_version() -> u32 {
    PRAECISE_ABI_VERSION
}

/// Return the crate's semantic version as a static NUL-terminated C string
/// (e.g. `"0.1.0"`). The pointer is valid for the lifetime of the process and
/// must not be freed.
#[must_use]
pub extern "C" fn praecise_version() -> *const c_char {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).expect("version has no NUL"))
        .as_ptr()
}

/// Termination cause for a generation, mirroring [`StopReason`]. The explicit
/// discriminants are part of the ABI.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PraeciseStopReason {
    /// The model emitted an end-of-generation token.
    Eos = 0,
    /// The `max_tokens` budget was exhausted.
    Length = 1,
    /// Decoded text ended with a configured stop sequence.
    StopSequence = 2,
}

impl From<StopReason> for PraeciseStopReason {
    fn from(r: StopReason) -> Self {
        match r {
            StopReason::Eos => Self::Eos,
            StopReason::Length => Self::Length,
            StopReason::StopSequence => Self::StopSequence,
        }
    }
}

/// C-ABI mirror of [`GenerationConfig`]'s scalar sampling parameters.
///
/// `Option` fields use sentinels so the struct stays plain-old-data: `top_k`
/// and `draft_n` treat `0` as "unset", and `min_p` treats any negative value
/// as "unset". Stop sequences are variable-length and are passed separately
/// when the generation entry point lands, so they are not part of this struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PraeciseConfig {
    /// Softmax temperature.
    pub temperature: f64,
    /// Nucleus-sampling probability mass.
    pub top_p: f64,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Repetition penalty applied over the last `repeat_last_n` tokens.
    pub repeat_penalty: f32,
    /// Window size for the repetition penalty.
    pub repeat_last_n: usize,
    /// RNG seed.
    pub seed: u64,
    /// Top-k cutoff; `0` means unset.
    pub top_k: u32,
    /// Min-p cutoff; a negative value means unset.
    pub min_p: f64,
    /// OpenAI-style frequency penalty.
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty.
    pub presence_penalty: f32,
    /// Speculative-decode draft length; `0` means unset (no MTP drafter).
    pub draft_n: u8,
}

impl From<GenerationConfig> for PraeciseConfig {
    fn from(c: GenerationConfig) -> Self {
        Self {
            temperature: c.temperature,
            top_p: c.top_p,
            max_tokens: c.max_tokens,
            repeat_penalty: c.repeat_penalty,
            repeat_last_n: c.repeat_last_n,
            seed: c.seed,
            top_k: c.top_k.unwrap_or(0),
            min_p: c.min_p.unwrap_or(-1.0),
            frequency_penalty: c.frequency_penalty,
            presence_penalty: c.presence_penalty,
            draft_n: c.draft_n.unwrap_or(0),
        }
    }
}

impl From<PraeciseConfig> for GenerationConfig {
    fn from(c: PraeciseConfig) -> Self {
        let base = GenerationConfig::default();
        Self {
            temperature: c.temperature,
            top_p: c.top_p,
            max_tokens: c.max_tokens,
            repeat_penalty: c.repeat_penalty,
            repeat_last_n: c.repeat_last_n,
            seed: c.seed,
            top_k: (c.top_k != 0).then_some(c.top_k),
            min_p: (c.min_p >= 0.0).then_some(c.min_p),
            frequency_penalty: c.frequency_penalty,
            presence_penalty: c.presence_penalty,
            draft_n: (c.draft_n != 0).then_some(c.draft_n),
            // Stop sequences are marshalled separately; keep the default (empty).
            ..base
        }
    }
}

/// Return a [`PraeciseConfig`] populated from [`GenerationConfig::default`], so
/// a C caller starts from the same defaults a Rust caller would.
#[must_use]
pub extern "C" fn praecise_config_default() -> PraeciseConfig {
    GenerationConfig::default().into()
}

/// Opaque, heap-owned inference result. Construct it (in the bundled build, via
/// the generation entry point), read fields through the accessors, then release
/// it with [`praecise_result_free`].
///
/// The C strings the accessors return are cached inside this struct, so they
/// live exactly as long as it does.
#[derive(Debug)]
pub struct PraeciseInferenceResult {
    inner: InferenceResult,
    text_c: CString,
    thinking_c: Option<CString>,
}

impl PraeciseInferenceResult {
    /// Wrap a runtime [`InferenceResult`], pre-encoding its strings for the C
    /// accessors. Interior NULs (which cannot occur in model text) are dropped
    /// defensively so an accessor never returns a truncated-at-NUL surprise.
    #[must_use]
    pub fn new(inner: InferenceResult) -> Self {
        let text_c = CString::new(inner.text.replace('\0', "")).unwrap_or_default();
        let thinking_c = inner
            .thinking
            .as_ref()
            .map(|t| CString::new(t.replace('\0', "")).unwrap_or_default());
        Self {
            inner,
            text_c,
            thinking_c,
        }
    }

    /// Move the result onto the heap and hand the caller an owning pointer.
    #[must_use]
    pub fn into_raw(self) -> *mut PraeciseInferenceResult {
        Box::into_raw(Box::new(self))
    }
}

/// Free a [`PraeciseInferenceResult`] returned by this library.
///
/// # Safety
/// `res` must be a pointer previously returned by this library and not already
/// freed. Passing `NULL` is allowed and is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn praecise_result_free(res: *mut PraeciseInferenceResult) {
    if !res.is_null() {
        drop(unsafe { Box::from_raw(res) });
    }
}

/// Generated text as a NUL-terminated C string, valid until `res` is freed.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`, which yields
/// `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_text(
    res: *const PraeciseInferenceResult,
) -> *const c_char {
    match unsafe { res.as_ref() } {
        Some(r) => r.text_c.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Reasoning/thinking text as a NUL-terminated C string, or `NULL` if the model
/// produced none. Valid until `res` is freed.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_thinking(
    res: *const PraeciseInferenceResult,
) -> *const c_char {
    match unsafe { res.as_ref() } {
        Some(r) => r
            .thinking_c
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr()),
        None => std::ptr::null(),
    }
}

/// Number of prompt (input) tokens. Returns `0` for a `NULL` result.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_input_tokens(
    res: *const PraeciseInferenceResult,
) -> u32 {
    unsafe { res.as_ref() }.map_or(0, |r| r.inner.input_tokens)
}

/// Number of generated (output) tokens. Returns `0` for a `NULL` result.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_output_tokens(
    res: *const PraeciseInferenceResult,
) -> u32 {
    unsafe { res.as_ref() }.map_or(0, |r| r.inner.output_tokens)
}

/// Wall-clock generation time in milliseconds. Returns `0` for a `NULL` result.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_generation_time_ms(
    res: *const PraeciseInferenceResult,
) -> u64 {
    unsafe { res.as_ref() }.map_or(0, |r| r.inner.generation_time_ms)
}

/// Decode throughput in tokens per second. Returns `0.0` for a `NULL` result.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_tokens_per_second(
    res: *const PraeciseInferenceResult,
) -> f64 {
    unsafe { res.as_ref() }.map_or(0.0, |r| r.inner.tokens_per_second)
}

/// Termination cause. Returns [`PraeciseStopReason::Eos`] for a `NULL` result.
///
/// # Safety
/// `res` must be a valid pointer from this library (or `NULL`).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn praecise_result_stop_reason(
    res: *const PraeciseInferenceResult,
) -> PraeciseStopReason {
    unsafe { res.as_ref() }.map_or(PraeciseStopReason::Eos, |r| r.inner.stop_reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_the_c_struct() {
        let rust = GenerationConfig::default();
        let c: PraeciseConfig = rust.clone().into();
        let back: GenerationConfig = c.into();
        assert_eq!(rust.temperature, back.temperature);
        assert_eq!(rust.max_tokens, back.max_tokens);
        assert_eq!(rust.top_k, back.top_k);
        assert_eq!(rust.min_p, back.min_p);
        assert_eq!(rust.draft_n, back.draft_n);
    }

    #[test]
    fn result_accessors_expose_the_wrapped_values() {
        let inner = InferenceResult {
            text: "hello".to_string(),
            thinking: Some("because".to_string()),
            input_tokens: 3,
            output_tokens: 5,
            generation_time_ms: 100,
            tokens_per_second: 50.0,
            stop_reason: StopReason::Eos,
        };
        let raw = PraeciseInferenceResult::new(inner).into_raw();
        unsafe {
            let text = std::ffi::CStr::from_ptr(praecise_result_text(raw));
            assert_eq!(text.to_str().unwrap(), "hello");
            assert_eq!(praecise_result_output_tokens(raw), 5);
            assert_eq!(praecise_result_stop_reason(raw), PraeciseStopReason::Eos);
            praecise_result_free(raw);
        }
    }

    #[test]
    fn version_and_abi_are_reported() {
        assert_eq!(praecise_abi_version(), PRAECISE_ABI_VERSION);
        let v = unsafe { std::ffi::CStr::from_ptr(praecise_version()) };
        assert!(!v.to_str().unwrap().is_empty());
    }
}
