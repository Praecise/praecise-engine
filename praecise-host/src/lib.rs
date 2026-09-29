//! Runtime hosts: engines run as confined child processes.
//!
//! An engine an application does not run in its own process (a Python
//! pipeline, a third-party inference server) runs in a runtime host: a
//! process the application starts itself, one per loaded model, confined by
//! [`sandbox`] (no network, a filesystem limited to the engine's own files,
//! the model's files and one scratch directory, a syscall filter). The
//! application speaks to it over the process's standard input and output
//! with the line-delimited JSON protocol described in [`host`], which streams
//! events (tokens, reasoning, tool calls, progress) ahead of each reply.
//!
//! The application calls [`sandbox::enter_if_requested`] first thing in
//! `main`: hosts are started by re-running the application's own executable
//! as the confinement launcher.
//!
//! [`Engines`] is the one place an application's models are assigned to
//! engines: each model is claimed by exactly one engine, in the process or in
//! a supervised runtime host started from a pinned [`EngineEnv`], within one
//! GPU memory budget.

pub mod engine;
pub mod engines;
pub mod generation;
pub mod host;
pub mod sandbox;

pub use engine::EngineEnv;
pub use engines::{Budget, Claim, EngineStatus, Engines, FixedBudget, Health, HostedModel, Usage};
pub use host::{HostSpec, RuntimeHost};

/// Errors from starting or talking to a runtime host.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The host could not be confined or started.
    #[error("runtime host could not start: {0}")]
    Start(String),
    /// The host refused a request.
    #[error("{0}")]
    Refused(String),
    /// The host exited or broke the protocol.
    #[error("{0}")]
    Failed(String),
    /// The model is already served, by this engine or another one.
    #[error("model {model_id} is served by {held_by}; {requested} cannot load it")]
    Conflict {
        /// The model.
        model_id: String,
        /// The engine that serves it.
        held_by: String,
        /// The engine that asked.
        requested: String,
    },
    /// The model does not fit the GPU memory left in the budget.
    #[error("model {model_id} needs {needed} bytes of GPU memory; {free} of the budget are free")]
    OverBudget {
        /// The model.
        model_id: String,
        /// Bytes it needs.
        needed: u64,
        /// Bytes of the budget not yet claimed.
        free: u64,
    },
    /// Filesystem error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A request could not be encoded.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for runtime hosts.
pub type Result<T> = std::result::Result<T, Error>;
