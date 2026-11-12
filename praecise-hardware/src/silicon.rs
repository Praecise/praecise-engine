//! The systems worth planning for, and the ceiling their memory sets.
//!
//! Decode reads every active weight once per token, so throughput is bounded
//! by `bandwidth ÷ bytes per token` (see `docs/gb10-bandwidth-ceiling.md`).
//! That makes memory, not compute, the figure that ranks hardware for
//! generation, and it holds for every vendor: the same arithmetic that
//! places a GB10 places an Instinct card, an M-series Mac or a TPU.
//!
//! The figures here are published peaks, each tagged with where it comes
//! from. A peak is a ceiling, not a prediction: measured decode reaches 44%
//! to 80% of it on the machines in the bandwidth note. Measure the achieved
//! figure on the machine before planning against it; these exist to rank
//! options and to refuse a plan that cannot fit, not to forecast a run.

use crate::vendor::Vendor;

/// Where a figure comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// The vendor's own specification or announcement.
    Vendor,
    /// Trade press or an analyst, not confirmed against the vendor.
    Press,
}

/// Whether a system can be bought and run today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// Shipping.
    Shipping,
    /// Announced or sampling; plan for it, do not depend on it.
    Announced,
}

/// One system or accelerator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct System {
    /// Its name.
    pub name: &'static str,
    /// Whose silicon it is.
    pub vendor: Vendor,
    /// Memory a model can occupy, in GB; `0` where it is configured per
    /// machine (a CPU socket).
    pub memory_gb: u32,
    /// Peak memory bandwidth in GB/s.
    pub bandwidth_gbs: f64,
    /// Whether the accelerator's memory is the host's (unified memory), so
    /// weights, KV cache and host checkpoints draw on one pool.
    pub unified: bool,
    /// Whether it can be bought today.
    pub availability: Availability,
    /// Where the figures come from.
    pub basis: Basis,
}

impl System {
    /// The decode ceiling in tokens per second for a model that reads
    /// `bytes_per_token` bytes of weights per token, at `efficiency` of peak
    /// bandwidth (`1.0` for the arithmetic bound; `0.44` to `0.80` is what
    /// measured machines reach).
    #[must_use]
    pub fn decode_ceiling(&self, bytes_per_token: u64, efficiency: f64) -> f64 {
        decode_ceiling(self.bandwidth_gbs * efficiency, bytes_per_token)
    }

    /// Whether weights of `bytes` fit in its memory at all. Always true for
    /// a system whose memory is configured per machine.
    #[must_use]
    pub fn fits(&self, bytes: u64) -> bool {
        self.memory_gb == 0 || bytes <= u64::from(self.memory_gb) * 1_000_000_000
    }
}

/// Tokens per second at `bandwidth_gbs` GB/s for a model reading
/// `bytes_per_token` bytes per token. Zero for a model that reads nothing.
#[must_use]
pub fn decode_ceiling(bandwidth_gbs: f64, bytes_per_token: u64) -> f64 {
    if bytes_per_token == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let bytes = bytes_per_token as f64;
    bandwidth_gbs * 1e9 / bytes
}

const fn system(
    name: &'static str,
    vendor: Vendor,
    memory_gb: u32,
    bandwidth_gbs: f64,
    unified: bool,
    availability: Availability,
    basis: Basis,
) -> System {
    System {
        name,
        vendor,
        memory_gb,
        bandwidth_gbs,
        unified,
        availability,
        basis,
    }
}

use Availability::{Announced, Shipping};

/// The catalogue, as of November 2026.
pub const SYSTEMS: &[System] = &[
    // NVIDIA
    system(
        "NVIDIA GB10 (DGX Spark)",
        Vendor::Nvidia,
        128,
        273.0,
        true,
        Shipping,
        Basis::Vendor,
    ),
    system(
        "NVIDIA Rubin",
        Vendor::Nvidia,
        288,
        22_000.0,
        false,
        Shipping,
        Basis::Press,
    ),
    // AMD
    system(
        "AMD Ryzen AI Max (Strix Halo)",
        Vendor::Amd,
        128,
        256.0,
        true,
        Shipping,
        Basis::Press,
    ),
    system(
        "AMD Radeon AI PRO R9700",
        Vendor::Amd,
        32,
        640.0,
        false,
        Shipping,
        Basis::Press,
    ),
    system(
        "AMD Instinct MI350P",
        Vendor::Amd,
        144,
        4_000.0,
        false,
        Shipping,
        Basis::Press,
    ),
    system(
        "AMD Instinct MI355X",
        Vendor::Amd,
        288,
        8_000.0,
        false,
        Shipping,
        Basis::Press,
    ),
    system(
        "AMD Instinct MI455X",
        Vendor::Amd,
        432,
        19_600.0,
        false,
        Shipping,
        Basis::Press,
    ),
    // Apple
    system(
        "Apple M4 Max",
        Vendor::Apple,
        128,
        546.0,
        true,
        Shipping,
        Basis::Vendor,
    ),
    system(
        "Apple M3 Ultra",
        Vendor::Apple,
        512,
        800.0,
        true,
        Shipping,
        Basis::Vendor,
    ),
    system(
        "Apple M5 Max",
        Vendor::Apple,
        128,
        614.0,
        true,
        Shipping,
        Basis::Vendor,
    ),
    system(
        "Apple M5 Ultra",
        Vendor::Apple,
        512,
        1_200.0,
        true,
        Shipping,
        Basis::Vendor,
    ),
    system(
        "Apple M6",
        Vendor::Apple,
        32,
        170.0,
        true,
        Shipping,
        Basis::Vendor,
    ),
    // Intel
    system(
        "Intel Xeon 6 (12-channel MRDIMM-8800)",
        Vendor::Intel,
        0,
        844.8,
        true,
        Shipping,
        Basis::Press,
    ),
    system(
        "Intel Xeon 6 (12-channel DDR5-6400)",
        Vendor::Intel,
        0,
        614.0,
        true,
        Shipping,
        Basis::Press,
    ),
    // Qualcomm
    system(
        "Qualcomm Snapdragon X2 Elite",
        Vendor::Qualcomm,
        128,
        152.0,
        true,
        Announced,
        Basis::Press,
    ),
    system(
        "Qualcomm Snapdragon X2 Elite Extreme",
        Vendor::Qualcomm,
        128,
        228.0,
        true,
        Announced,
        Basis::Press,
    ),
    // Google
    system(
        "Google TPU 8i",
        Vendor::Google,
        288,
        8_601.0,
        false,
        Announced,
        Basis::Vendor,
    ),
];

/// The systems of one vendor.
pub fn systems_of(vendor: Vendor) -> impl Iterator<Item = &'static System> {
    SYSTEMS.iter().filter(move |s| s.vendor == vendor)
}

/// A tier of memory a model's bytes can live in. Weights, the live KV cache
/// and cold KV state are placed separately, and each tier bounds what is
/// read from it per token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryTier {
    /// Its name.
    pub name: &'static str,
    /// Sustained bandwidth of one unit in GB/s.
    pub bandwidth_gbs: f64,
    /// What one unit is.
    pub unit: &'static str,
    /// Whether it can be bought today.
    pub availability: Availability,
}

/// Memory tiers by the standard that defines them, fastest first.
pub const MEMORY_TIERS: &[MemoryTier] = &[
    MemoryTier {
        name: "HBM4",
        bandwidth_gbs: 2_000.0,
        unit: "stack (2048 I/O at 8 Gb/s)",
        availability: Shipping,
    },
    MemoryTier {
        name: "LPDDR5X SOCAMM2",
        bandwidth_gbs: 1_200.0,
        unit: "server CPU",
        availability: Shipping,
    },
    MemoryTier {
        name: "DDR5 MRDIMM-12800",
        bandwidth_gbs: 102.0,
        unit: "channel",
        availability: Announced,
    },
    MemoryTier {
        name: "DDR5 MRDIMM-8800",
        bandwidth_gbs: 70.4,
        unit: "channel",
        availability: Shipping,
    },
    MemoryTier {
        name: "DDR5-6400",
        bandwidth_gbs: 51.2,
        unit: "channel",
        availability: Shipping,
    },
];

impl MemoryTier {
    /// Peak bandwidth of `units` of this tier in GB/s: a socket's channels,
    /// an accelerator's stacks.
    #[must_use]
    pub fn bandwidth(&self, units: u32) -> f64 {
        self.bandwidth_gbs * f64::from(units)
    }
}

/// Measure the memory bandwidth this machine's CPU threads reach, in GB/s.
///
/// Each of `threads` threads reads its own buffer of `bytes_per_thread`
/// bytes `passes` times, all at once; the result is the bytes read over the
/// time the slowest thread took. Buffers larger than the last-level cache
/// make it a measurement of memory, not of cache.
///
/// A published peak is a ceiling and machines reach 44% to 80% of it, so
/// this is the figure to plan a CPU or unified-memory decode against. It
/// says nothing about a discrete accelerator's own memory, which only that
/// accelerator can read.
#[must_use]
pub fn measure_host_bandwidth(bytes_per_thread: usize, threads: usize, passes: u32) -> f64 {
    use std::sync::{Arc, Barrier};
    use std::time::Instant;

    let threads = threads.max(1);
    let words = (bytes_per_thread / 8).max(1);
    let start = Arc::new(Barrier::new(threads));
    let workers: Vec<_> = (0..threads)
        .map(|t| {
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                // written once, so every page is resident before the clock starts
                let buffer: Vec<u64> = (0..words as u64).map(|i| i ^ t as u64).collect();
                start.wait();
                let began = Instant::now();
                let mut sum = 0u64;
                for _ in 0..passes.max(1) {
                    for word in &buffer {
                        sum = sum.wrapping_add(*word);
                    }
                    std::hint::black_box(sum);
                }
                began.elapsed()
            })
        })
        .collect();
    let slowest = workers
        .into_iter()
        .filter_map(|w| w.join().ok())
        .max()
        .unwrap_or_default()
        .as_secs_f64();
    if slowest <= 0.0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let read = (words * 8 * threads) as f64 * f64::from(passes.max(1));
    read / slowest / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(name: &str) -> &'static System {
        SYSTEMS.iter().find(|s| s.name.contains(name)).unwrap()
    }

    #[test]
    fn the_ceiling_reproduces_the_measured_gb10_figure() {
        // docs/gb10-bandwidth-ceiling.md: 3.82 GB/token at 57.21 tok/s is
        // 218.7 GB/s achieved, 80.1% of the 273 GB/s peak.
        let gb10 = find("GB10");
        let at_peak = gb10.decode_ceiling(3_820_000_000, 1.0);
        assert!((at_peak - 71.47).abs() < 0.01, "{at_peak}");
        let measured = gb10.decode_ceiling(3_820_000_000, 0.801);
        assert!((measured - 57.21).abs() < 0.1, "{measured}");
    }

    #[test]
    fn the_same_arithmetic_ranks_every_vendor() {
        let bytes = 3_820_000_000;
        let order = [
            "M6",
            "Strix Halo",
            "GB10",
            "M5 Max",
            "R9700",
            "M5 Ultra",
            "MI355X",
            "TPU 8i",
            "MI455X",
        ];
        let ceilings: Vec<f64> = order
            .iter()
            .map(|n| find(n).decode_ceiling(bytes, 1.0))
            .collect();
        assert!(ceilings.windows(2).all(|w| w[0] < w[1]), "{ceilings:?}");
    }

    #[test]
    fn a_model_larger_than_the_memory_does_not_fit() {
        assert!(find("R9700").fits(30_000_000_000));
        assert!(!find("R9700").fits(40_000_000_000));
        // a CPU socket's memory is whatever the machine was built with
        assert!(find("MRDIMM-8800").fits(900_000_000_000));
    }

    #[test]
    fn a_model_that_reads_nothing_has_no_ceiling() {
        assert_eq!(decode_ceiling(273.0, 0), 0.0);
    }

    #[test]
    fn every_major_vendor_is_in_the_catalogue() {
        for v in [
            Vendor::Nvidia,
            Vendor::Amd,
            Vendor::Intel,
            Vendor::Apple,
            Vendor::Qualcomm,
            Vendor::Google,
        ] {
            assert!(systems_of(v).next().is_some(), "{v}");
        }
        for s in SYSTEMS {
            assert!(s.bandwidth_gbs > 0.0, "{}", s.name);
        }
    }

    #[test]
    fn the_host_bandwidth_is_measured() {
        // small buffers: this checks the measurement runs and is finite,
        // not what the machine reaches
        let gbs = measure_host_bandwidth(1 << 20, 2, 4);
        assert!(gbs.is_finite() && gbs > 0.0, "{gbs}");
        assert!(measure_host_bandwidth(0, 0, 0) >= 0.0);
    }

    #[test]
    fn a_socket_is_its_channels() {
        let tier = |n: &str| MEMORY_TIERS.iter().find(|t| t.name == n).unwrap();
        // twelve MRDIMM-8800 channels are the Xeon 6 figure in the catalogue
        assert!(
            (tier("DDR5 MRDIMM-8800").bandwidth(12) - find("MRDIMM-8800").bandwidth_gbs).abs()
                < 0.01
        );
        // eight HBM4 stacks at the base pin speed; shipping parts run faster
        assert!(tier("HBM4").bandwidth(8) <= find("Rubin").bandwidth_gbs);
    }
}
