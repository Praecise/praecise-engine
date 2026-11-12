//! Vector and matrix extensions of the host CPU.
//!
//! The CPU is an inference device on every machine and the only one on many.
//! What it can do is decided by extensions that differ between two parts of
//! the same generation: AVX-512 and AMX on x86, the dot-product, int8-matmul,
//! SVE and SME families on Arm, the vector extension on RISC-V, LSX and LASX
//! on `LoongArch`, the vector facilities on IBM Z. This reports the ones the
//! engine's kernels and the ggml CPU backend select on.

use crate::detect::Probe;

/// The extensions detected on a CPU, by the name the architecture gives them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuFeatures {
    /// Target architecture (`x86_64`, `aarch64`, `riscv64`, …).
    pub arch: &'static str,
    /// Extensions present, in the order the architecture introduced them.
    pub features: Vec<&'static str>,
}

impl CpuFeatures {
    /// The extensions of the CPU this process runs on.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            arch: std::env::consts::ARCH,
            features: detect_features(),
        }
    }

    /// Whether `feature` is present.
    #[must_use]
    pub fn has(&self, feature: &str) -> bool {
        self.features.contains(&feature)
    }

    /// The ggml CPU build options this CPU can use beyond the baseline, as
    /// `llama-cpp-sys-2` sets them from the matching Rust target features.
    #[must_use]
    pub fn ggml_options(&self) -> Vec<&'static str> {
        GGML_OPTIONS
            .iter()
            .filter(|(f, _)| self.has(f))
            .map(|(_, o)| *o)
            .collect()
    }
}

/// CPU extension to the ggml `CMake` option it enables.
const GGML_OPTIONS: &[(&str, &str)] = &[
    ("avx2", "GGML_AVX2"),
    ("avxvnni", "GGML_AVX_VNNI"),
    ("avx512f", "GGML_AVX512"),
    ("avx512vbmi", "GGML_AVX512_VBMI"),
    ("avx512vnni", "GGML_AVX512_VNNI"),
    ("avx512bf16", "GGML_AVX512_BF16"),
    ("amx-tile", "GGML_AMX_TILE"),
    ("amx-int8", "GGML_AMX_INT8"),
    ("amx-bf16", "GGML_AMX_BF16"),
    ("v", "GGML_RVV"),
    ("zvfh", "GGML_RV_ZVFH"),
    ("lsx", "GGML_LSX"),
    ("lasx", "GGML_LASX"),
    ("vxe", "GGML_VXE"),
];

#[cfg(target_arch = "x86_64")]
fn detect_features() -> Vec<&'static str> {
    macro_rules! probe {
        ($($name:tt),* $(,)?) => {{
            let mut out = Vec::new();
            $(if std::arch::is_x86_feature_detected!($name) { out.push($name); })*
            out
        }};
    }
    probe!(
        "sse4.2",
        "avx",
        "f16c",
        "fma",
        "avx2",
        "bmi2",
        "avxvnni",
        "avx512f",
        "avx512bw",
        "avx512vbmi",
        "avx512vnni",
        "avx512bf16",
        "avx512fp16",
    )
    .into_iter()
    .chain(amx_features())
    .collect()
}

/// AMX, read from CPUID: leaf 7 names the tile, int8 and bf16 units, and the
/// operating system has to have enabled the tile state (XCR0 bits 17 and 18)
/// for any of them to be usable.
#[cfg(target_arch = "x86_64")]
fn amx_features() -> Vec<&'static str> {
    #[target_feature(enable = "xsave")]
    fn xcr0() -> u64 {
        // SAFETY: the caller checked that XSAVE is present.
        #[allow(unused_unsafe)]
        unsafe {
            std::arch::x86_64::_xgetbv(0)
        }
    }
    if !std::arch::is_x86_feature_detected!("xsave") {
        return Vec::new();
    }
    // SAFETY: XSAVE was detected above.
    let tile_state = unsafe { xcr0() } & (0b11 << 17) == (0b11 << 17);
    // SAFETY: CPUID is available on every x86-64 processor.
    #[allow(unused_unsafe)]
    let leaf7 = unsafe { std::arch::x86_64::__cpuid_count(7, 0) };
    if !tile_state {
        return Vec::new();
    }
    [(24, "amx-tile"), (25, "amx-int8"), (22, "amx-bf16")]
        .into_iter()
        .filter(|(bit, _)| leaf7.edx & (1 << bit) != 0)
        .map(|(_, name)| name)
        .collect()
}

#[cfg(target_arch = "aarch64")]
fn detect_features() -> Vec<&'static str> {
    macro_rules! probe {
        ($($name:tt),* $(,)?) => {{
            let mut out = Vec::new();
            $(if std::arch::is_aarch64_feature_detected!($name) { out.push($name); })*
            out
        }};
    }
    let mut out = probe!("neon", "fp16", "dotprod", "i8mm", "bf16", "sve", "sve2");
    out.extend(sme_features());
    out
}

/// The scalable matrix extension, which the standard library does not yet
/// detect on a stable toolchain: read from the kernel's feature list on
/// Linux and Android, and from the hardware feature flags on macOS.
#[cfg(target_arch = "aarch64")]
fn sme_features() -> Vec<&'static str> {
    if cfg!(target_os = "macos") {
        let flag = |name: &str| {
            std::process::Command::new("sysctl")
                .args(["-n", name])
                .output()
                .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "1")
        };
        return [
            ("hw.optional.arm.FEAT_SME", "sme"),
            ("hw.optional.arm.FEAT_SME2", "sme2"),
        ]
        .into_iter()
        .filter(|(key, _)| flag(key))
        .map(|(_, name)| name)
        .collect();
    }
    arm_sme_from_cpuinfo(&crate::detect::HostProbe)
}

/// `sme` and `sme2` as the `Features` line of an `AArch64` `/proc/cpuinfo`
/// lists them.
#[must_use]
pub fn arm_sme_from_cpuinfo(probe: &dyn Probe) -> Vec<&'static str> {
    let Some(cpuinfo) = probe.read("/proc/cpuinfo") else {
        return Vec::new();
    };
    let Some(features) = cpuinfo.lines().find(|l| l.starts_with("Features")) else {
        return Vec::new();
    };
    ["sme", "sme2"]
        .into_iter()
        .filter(|k| features.split_whitespace().any(|f| f == *k))
        .collect()
}

/// Architectures whose extensions the kernel reports in `/proc/cpuinfo`.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_features() -> Vec<&'static str> {
    from_cpuinfo(std::env::consts::ARCH, &crate::detect::HostProbe)
}

/// The extensions `/proc/cpuinfo` reports for `arch`: the single-letter and
/// `z` extensions of a RISC-V `isa` line, the `features` line of `LoongArch`
/// and IBM Z.
#[must_use]
pub fn from_cpuinfo(arch: &str, probe: &dyn Probe) -> Vec<&'static str> {
    let Some(cpuinfo) = probe.read("/proc/cpuinfo") else {
        return Vec::new();
    };
    let line = |key: &str| {
        cpuinfo
            .lines()
            .find(|l| l.trim_start().to_ascii_lowercase().starts_with(key))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_ascii_lowercase())
    };
    match arch {
        "riscv64" => {
            let Some(isa) = line("isa") else {
                return Vec::new();
            };
            let mut parts = isa.split('_');
            let base = parts.next().unwrap_or("");
            let mut out = Vec::new();
            // `rv64imafdcv`: the letters after the width are extensions
            if base
                .strip_prefix("rv64")
                .is_some_and(|ext| ext.contains('v'))
            {
                out.push("v");
            }
            for ext in parts {
                for known in [
                    "zfh",
                    "zvfh",
                    "zvfbfwma",
                    "zicbop",
                    "zihintpause",
                    "zba",
                    "xtheadvector",
                ] {
                    if ext == known {
                        out.push(known);
                    }
                }
            }
            out
        }
        "loongarch64" | "s390x" => {
            let Some(features) = line("features") else {
                return Vec::new();
            };
            let known: &[&'static str] = if arch == "s390x" {
                &["vx", "vxe", "vxe2", "nnpa"]
            } else {
                &["lsx", "lasx"]
            };
            known
                .iter()
                .copied()
                .filter(|k| features.split_whitespace().any(|f| f == *k))
                .collect()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::testing::Described;

    #[test]
    fn detection_reports_this_cpu() {
        let cpu = CpuFeatures::detect();
        assert_eq!(cpu.arch, std::env::consts::ARCH);
        // every x86-64 and every AArch64 part has its baseline vector unit
        if cfg!(target_arch = "aarch64") {
            assert!(cpu.has("neon"));
        }
        for option in cpu.ggml_options() {
            assert!(option.starts_with("GGML_"));
        }
    }

    #[test]
    fn the_build_options_follow_the_extensions() {
        let xeon = CpuFeatures {
            arch: "x86_64",
            features: vec!["avx2", "avx512f", "avx512vnni", "amx-tile", "amx-int8"],
        };
        assert_eq!(
            xeon.ggml_options(),
            [
                "GGML_AVX2",
                "GGML_AVX512",
                "GGML_AVX512_VNNI",
                "GGML_AMX_TILE",
                "GGML_AMX_INT8"
            ]
        );
        let client = CpuFeatures {
            arch: "x86_64",
            features: vec!["avx2", "avxvnni"],
        };
        assert_eq!(client.ggml_options(), ["GGML_AVX2", "GGML_AVX_VNNI"]);
        // Arm extensions are selected by the architecture string, not options
        let arm = CpuFeatures {
            arch: "aarch64",
            features: vec!["neon", "dotprod", "i8mm", "sve2", "sme2"],
        };
        assert_eq!(arm.ggml_options(), [] as [&str; 0]);
    }

    #[test]
    fn risc_v_extensions_are_read_from_the_isa_line() {
        let k3 = Described::default().file(
            "/proc/cpuinfo",
            "processor\t: 0\nisa\t\t: rv64imafdcv_zicbom_zicbop_zihintpause_zba_zfh_zvfh\nmmu\t\t: sv39\n",
        );
        assert_eq!(
            from_cpuinfo("riscv64", &k3),
            ["v", "zicbop", "zihintpause", "zba", "zfh", "zvfh"]
        );
        let plain = Described::default().file("/proc/cpuinfo", "isa\t\t: rv64imafdc_zicsr\n");
        assert_eq!(from_cpuinfo("riscv64", &plain), [] as [&str; 0]);
    }

    #[test]
    fn arm_matrix_extensions_are_read_from_the_features_line() {
        let c1 = Described::default().file(
            "/proc/cpuinfo",
            "processor	: 0
Features	: fp asimd asimddp i8mm bf16 sve sve2 sme sme2 smei8i32
",
        );
        assert_eq!(arm_sme_from_cpuinfo(&c1), ["sme", "sme2"]);
        // a Neoverse V3-class server core has SVE2 and no SME
        let v3 = Described::default().file(
            "/proc/cpuinfo",
            "Features	: fp asimd asimddp i8mm bf16 sve sve2
",
        );
        assert_eq!(arm_sme_from_cpuinfo(&v3), [] as [&str; 0]);
    }

    #[test]
    fn loongarch_and_ibm_z_are_read_from_the_features_line() {
        let loong = Described::default().file(
            "/proc/cpuinfo",
            "Features\t\t: cpucfg lam ual fpu lsx lasx crc32\n",
        );
        assert_eq!(from_cpuinfo("loongarch64", &loong), ["lsx", "lasx"]);
        let z = Described::default().file(
            "/proc/cpuinfo",
            "features\t: esan3 zarch vx vxe vxe2 nnpa\n",
        );
        assert_eq!(from_cpuinfo("s390x", &z), ["vx", "vxe", "vxe2", "nnpa"]);
        assert_eq!(
            from_cpuinfo("s390x", &Described::default()),
            [] as [&str; 0]
        );
    }
}
