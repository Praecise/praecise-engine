//! Praecise Engine hardware layer.
//!
//! One place that answers four questions for every silicon vendor the engine
//! can run on, so no crate above it carries a private idea of what "a GPU" is:
//!
//! - [`detect`] — which accelerators this machine has, by vendor and kind,
//!   read from the operating system (PCI ids, kernel driver bindings, device
//!   nodes) rather than from one vendor's management library.
//! - [`route`] — which ggml backend, torch device and hosted-runtime platform
//!   drives each accelerator, in order of preference.
//! - [`confine`] — the device nodes, runtime directories and selection
//!   variables a confined engine needs to reach them.
//! - [`silicon`] — the memory and peak bandwidth of the systems worth planning
//!   for, and the decode ceiling that follows from them.
//! - [`cpu`] — the vector and matrix extensions of the host CPU.
//!
//! The crate has no dependencies and performs no vendor calls. Detection is a
//! pure function of what a [`detect::Probe`] reports, so every rule is tested
//! against a described machine on any build host.
//!
//! Nothing here claims a device works. It says what is present and what could
//! drive it; a backend that is not built in, or a device that fails to
//! initialise, is still refused where it is loaded.

pub mod confine;
pub mod cpu;
pub mod detect;
pub mod route;
pub mod silicon;
pub mod vendor;

pub use detect::{Accelerator, Kind, Machine};
pub use vendor::Vendor;
