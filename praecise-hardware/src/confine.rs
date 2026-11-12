//! What a confined engine needs to reach an accelerator.
//!
//! A runtime host is confined to an explicit list of paths. An accelerator a
//! host may not open does not exist for it, so the list has to name every
//! vendor's device nodes, the directories vendor runtimes install into, and
//! the variables an operator selects devices with. Naming one vendor's alone
//! confines every other accelerator out of reach.

use std::path::PathBuf;

use crate::detect::Probe;

/// How an entry of `/dev` is recognised.
enum Match {
    /// The name is exactly this.
    Exact(&'static str),
    /// The name starts with this.
    Prefix(&'static str),
}

/// Entries of `/dev` that are, or hold, accelerator device nodes.
///
/// `dri` is the kernel's DRM render and card nodes (every GPU with an
/// upstream driver: AMD, Intel, Moore Threads, Mali, Adreno, Asahi, and
/// NVIDIA beside its own nodes). `accel` is the kernel's compute-accelerator
/// class (Intel and AMD NPUs, Gaudi, Qualcomm Cloud AI).
const NODES: &[(Match, &str)] = &[
    (Match::Prefix("nvidia"), "NVIDIA"),
    (Match::Prefix("nvhost-"), "NVIDIA Jetson"),
    (Match::Exact("nvmap"), "NVIDIA Jetson"),
    (Match::Exact("kfd"), "AMD ROCm"),
    (Match::Exact("dri"), "DRM GPUs"),
    (Match::Exact("accel"), "compute accelerators"),
    (Match::Prefix("davinci"), "Huawei Ascend"),
    (Match::Exact("devmm_svm"), "Huawei Ascend"),
    (Match::Exact("hisi_hdc"), "Huawei Ascend"),
    (Match::Exact("ascend_manager"), "Huawei Ascend"),
    (Match::Exact("user_config"), "Huawei Ascend"),
    (Match::Exact("ummu"), "Huawei Ascend"),
    (Match::Exact("uburma"), "Huawei Ascend"),
    (Match::Prefix("neuron"), "AWS Neuron"),
    (Match::Exact("tenstorrent"), "Tenstorrent"),
    (Match::Prefix("kgsl-3d"), "Qualcomm Adreno"),
    (Match::Prefix("fastrpc-"), "Qualcomm Hexagon"),
    (Match::Prefix("adsprpc-"), "Qualcomm Hexagon"),
    (Match::Prefix("mali"), "Arm Mali"),
    (Match::Prefix("cambricon_"), "Cambricon"),
];

/// The accelerator device nodes present on this host, sorted: every entry of
/// `/dev` a vendor's driver or the kernel's GPU and accelerator classes
/// created.
#[must_use]
pub fn device_nodes(probe: &dyn Probe) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = probe
        .entries("/dev")
        .into_iter()
        .filter(|name| {
            NODES.iter().any(|(m, _)| match m {
                Match::Exact(want) => name == want,
                Match::Prefix(prefix) => name.starts_with(prefix),
            })
        })
        .map(|name| PathBuf::from(format!("/dev/{name}")))
        .collect();
    out.sort();
    out
}

/// Directories vendor runtimes install into outside the system library
/// paths. A host that loads a vendor runtime needs them readable.
pub const RUNTIME_ROOTS: &[&str] = &[
    "/opt/rocm",
    "/opt/intel",
    "/opt/habanalabs",
    "/opt/aws/neuron",
    "/opt/qti-aic",
    "/opt/et",
    "/usr/local/Ascend",
    "/usr/local/musa",
    "/usr/local/neuware",
];

/// The [`RUNTIME_ROOTS`] that exist on this host.
#[must_use]
pub fn runtime_roots(probe: &dyn Probe) -> Vec<PathBuf> {
    RUNTIME_ROOTS
        .iter()
        .filter(|p| probe.exists(p))
        .map(PathBuf::from)
        .collect()
}

/// Variables an operator selects or restricts devices with. A host's
/// environment is built from nothing, so each one the operator set has to be
/// carried into it or the choice is lost.
pub const SELECTION_ENV: &[&str] = &[
    // NVIDIA
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    // AMD ROCm
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    // Intel Level Zero, oneAPI and Gaudi
    "ZE_AFFINITY_MASK",
    "ONEAPI_DEVICE_SELECTOR",
    "HABANA_VISIBLE_DEVICES",
    // Huawei Ascend
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    // Moore Threads
    "MUSA_VISIBLE_DEVICES",
    // AWS Neuron
    "NEURON_RT_VISIBLE_CORES",
    "NEURON_RT_NUM_CORES",
    // ggml backends
    "GGML_VK_VISIBLE_DEVICES",
    "GGML_OPENVINO_DEVICE",
    "GGML_HEXAGON_DEVICES",
    // this engine
    crate::detect::ACCELERATORS_ENV,
];

/// The [`SELECTION_ENV`] variables set in this process, with their values.
#[must_use]
pub fn selection_env() -> Vec<(String, String)> {
    SELECTION_ENV
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| ((*k).to_string(), v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::testing::Described;

    fn names(nodes: &[PathBuf]) -> Vec<String> {
        nodes.iter().map(|p| p.display().to_string()).collect()
    }

    #[test]
    fn a_host_without_accelerators_has_no_nodes() {
        let probe = Described::default()
            .node("/dev/null")
            .node("/dev/sda")
            .node("/dev/tty0");
        assert_eq!(device_nodes(&probe), [] as [std::path::PathBuf; 0]);
    }

    #[test]
    fn nvidia_and_amd_nodes_are_found_as_before() {
        let probe = Described::default()
            .node("/dev/nvidia0")
            .node("/dev/nvidiactl")
            .node("/dev/nvidia-uvm")
            .node("/dev/kfd")
            .node("/dev/dri/renderD128")
            .node("/dev/null");
        assert_eq!(
            names(&device_nodes(&probe)),
            [
                "/dev/dri",
                "/dev/kfd",
                "/dev/nvidia-uvm",
                "/dev/nvidia0",
                "/dev/nvidiactl"
            ]
        );
    }

    #[test]
    fn every_other_vendor_is_reachable_too() {
        let ascend = Described::default()
            .node("/dev/davinci0")
            .node("/dev/davinci_manager")
            .node("/dev/devmm_svm")
            .node("/dev/hisi_hdc");
        assert_eq!(device_nodes(&ascend).len(), 4);
        let npu = Described::default()
            .node("/dev/accel/accel0")
            .node("/dev/dri/card0");
        assert_eq!(names(&device_nodes(&npu)), ["/dev/accel", "/dev/dri"]);
        let phone = Described::default()
            .node("/dev/kgsl-3d0")
            .node("/dev/fastrpc-cdsp")
            .node("/dev/fastrpc-cdsp-secure");
        assert_eq!(device_nodes(&phone).len(), 3);
        let trainium = Described::default()
            .node("/dev/neuron0")
            .node("/dev/neuron1");
        assert_eq!(device_nodes(&trainium).len(), 2);
    }

    #[test]
    fn only_the_runtime_roots_that_exist_are_listed() {
        let probe = Described::default()
            .node("/opt/rocm/lib/libamdhip64.so")
            .node("/usr/local/Ascend/cann/set_env.sh");
        assert_eq!(
            names(&runtime_roots(&probe)),
            ["/opt/rocm", "/usr/local/Ascend"]
        );
        assert_eq!(
            runtime_roots(&Described::default()),
            [] as [std::path::PathBuf; 0]
        );
    }

    #[test]
    fn selection_variables_cover_each_vendor_once() {
        let mut sorted: Vec<&str> = SELECTION_ENV.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), SELECTION_ENV.len());
        for must in [
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ZE_AFFINITY_MASK",
            "ASCEND_RT_VISIBLE_DEVICES",
        ] {
            assert!(SELECTION_ENV.contains(&must), "{must}");
        }
    }
}
