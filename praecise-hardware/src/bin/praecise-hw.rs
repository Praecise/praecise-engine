//! `praecise-hw`: what this machine has and which build drives it.
//!
//! Prints the accelerators detected (or declared), the device nodes and
//! runtime directories a confined engine would be given, the CPU extensions,
//! the cargo features a build for this machine needs, and the route each
//! accelerator is reached by. `praecise-hw catalogue [bytes-per-token]` prints
//! the silicon catalogue with the decode ceiling of each system, and
//! `praecise-hw measure` measures the memory bandwidth this machine's CPU
//! threads reach.

use std::process::ExitCode;

use praecise_hardware::confine;
use praecise_hardware::cpu::CpuFeatures;
use praecise_hardware::detect::{HostProbe, Machine};
use praecise_hardware::route;
use praecise_hardware::silicon::{Availability, SYSTEMS};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("machine") => machine(),
        Some("catalogue") => catalogue(args.get(1).map(String::as_str)),
        Some("measure") => measure(),
        Some(other) => {
            eprintln!(
                "praecise-hw: unknown command {other:?}; use `machine`, `measure` or `catalogue [bytes-per-token]`"
            );
            ExitCode::from(2)
        }
    }
}

fn machine() -> ExitCode {
    let machine = match Machine::from_env_or_detect() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("praecise-hw: {e}");
            return ExitCode::from(2);
        }
    };
    println!("accelerators: {machine}");
    for a in machine.accelerators() {
        let linked: Vec<&str> = route::ggml_backends(*a)
            .iter()
            .map(|b| b.feature())
            .collect();
        let linked = if linked.is_empty() {
            "none".to_string()
        } else {
            linked.join(", ")
        };
        let torch = route::torch_device(*a).unwrap_or("none");
        let vllm = route::vllm_platform(*a).map_or("none".to_string(), |p| {
            if p.builtin {
                p.package.to_string()
            } else {
                format!("{} (plugin)", p.package)
            }
        });
        println!("  {a}: linked backends [{linked}]; torch device {torch}; vLLM {vllm}");
    }
    let features = route::features_for(&machine);
    println!(
        "build features: {}",
        if features.is_empty() {
            "none (CPU)".to_string()
        } else {
            features.join(",")
        }
    );
    let cpu = CpuFeatures::detect();
    println!("cpu: {} [{}]", cpu.arch, cpu.features.join(" "));
    println!("cpu build options: {}", cpu.ggml_options().join(" "));
    let nodes: Vec<String> = confine::device_nodes(&HostProbe)
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    println!(
        "device nodes: {}",
        if nodes.is_empty() {
            "none".to_string()
        } else {
            nodes.join(" ")
        }
    );
    let roots: Vec<String> = confine::runtime_roots(&HostProbe)
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    println!(
        "runtime directories: {}",
        if roots.is_empty() {
            "none".to_string()
        } else {
            roots.join(" ")
        }
    );
    for (k, v) in confine::selection_env() {
        println!("selection: {k}={v}");
    }
    ExitCode::SUCCESS
}

/// Measures what the CPU threads of this machine read from memory.
fn measure() -> ExitCode {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    // One gibibyte across the threads, at least 32 MiB each: far past any
    // last-level cache, and small enough not to push a machine into paging,
    // which would measure the disk instead.
    let per_thread = ((1usize << 30) / threads).max(32 << 20);
    let one = praecise_hardware::silicon::measure_host_bandwidth(per_thread, 1, 8);
    let all = praecise_hardware::silicon::measure_host_bandwidth(per_thread, threads, 8);
    println!(
        "host memory read bandwidth: {all:.1} GB/s on {threads} threads, {one:.1} GB/s on one"
    );
    println!(
        "a model reading 3.82 GB per token decodes at most {:.1} tok/s on this CPU",
        all / 3.82
    );
    ExitCode::SUCCESS
}

fn catalogue(bytes_per_token: Option<&str>) -> ExitCode {
    // Llama-2-7B Q4_0, the model of the bandwidth note, unless one is given.
    let bytes = match bytes_per_token.map(str::parse::<u64>) {
        None => 3_820_000_000,
        Some(Ok(b)) if b > 0 => b,
        Some(_) => {
            eprintln!("praecise-hw: bytes-per-token must be a positive integer");
            return ExitCode::from(2);
        }
    };
    println!("decode ceiling at {bytes} bytes per token (arithmetic bound at peak bandwidth)");
    println!(
        "{:<42} {:>9} {:>10} {:>10}  status",
        "system", "memory", "GB/s", "tok/s"
    );
    for s in SYSTEMS {
        let memory = if s.memory_gb == 0 {
            "per build".to_string()
        } else {
            format!("{} GB", s.memory_gb)
        };
        let ceiling = if s.fits(bytes) {
            format!("{:.1}", s.decode_ceiling(bytes, 1.0))
        } else {
            "no fit".to_string()
        };
        let status = match s.availability {
            Availability::Shipping => "shipping",
            Availability::Announced => "announced",
        };
        println!(
            "{:<42} {:>9} {:>10.0} {:>10}  {}",
            s.name, memory, s.bandwidth_gbs, ceiling, status
        );
    }
    ExitCode::SUCCESS
}
