//! Which accelerators a machine has.
//!
//! Detection reads what the operating system already knows and no vendor
//! library: PCI vendor and class codes, the kernel driver bound to each
//! compute-accelerator and DRM device, and the device nodes vendor drivers
//! create. That keeps it working in an environment where a vendor's own tools
//! are absent, and makes every rule a pure function of a [`Probe`].
//!
//! An operator can declare the machine instead of having it detected, with
//! [`ACCELERATORS_ENV`]. That is the only way to name an accelerator the
//! system exposes no stable sign of (a cloud TPU, for instance), and a way to
//! state that a machine must be treated as having none.

use std::fmt;

use crate::vendor::Vendor;

/// Environment variable that declares the machine's accelerators instead of
/// detecting them: a comma-separated list of `vendor[:kind]`, or `none`.
/// For example `amd:gpu,amd:npu` or `google:accelerator`.
pub const ACCELERATORS_ENV: &str = "PRAECISE_ACCELERATORS";

/// What kind of device an accelerator is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// A graphics processor, discrete or integrated.
    Gpu,
    /// A neural processor beside a CPU: fixed-function, usually quantized and
    /// static-shape, reached through the vendor's own runtime.
    Npu,
    /// A compute accelerator that is neither: a data-centre inference or
    /// training card with its own software stack.
    Accelerator,
}

impl Kind {
    /// Stable lowercase identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Kind::Gpu => "gpu",
            Kind::Npu => "npu",
            Kind::Accelerator => "accelerator",
        }
    }

    /// Parse a kind identifier.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gpu" => Some(Kind::Gpu),
            "npu" => Some(Kind::Npu),
            "accelerator" | "accel" => Some(Kind::Accelerator),
            _ => None,
        }
    }
}

/// One class of accelerator present on a machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Accelerator {
    /// Whose silicon it is.
    pub vendor: Vendor,
    /// What kind of device it is.
    pub kind: Kind,
}

impl fmt::Display for Accelerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.vendor, self.kind.as_str())
    }
}

/// The accelerators of one machine: each `(vendor, kind)` at most once, in a
/// stable order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Machine {
    accelerators: Vec<Accelerator>,
}

impl Machine {
    /// A machine with exactly these accelerators.
    #[must_use]
    pub fn with(accelerators: impl IntoIterator<Item = Accelerator>) -> Self {
        let mut accelerators: Vec<Accelerator> = accelerators.into_iter().collect();
        accelerators.sort();
        accelerators.dedup();
        Self { accelerators }
    }

    /// A machine with no accelerator.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self::default()
    }

    /// The machine the operator declared in [`ACCELERATORS_ENV`], otherwise
    /// the one detected.
    ///
    /// # Errors
    /// When the declaration names an unknown vendor or kind. A declaration
    /// that cannot be read is refused rather than replaced by detection: the
    /// operator set it to override what detection would say.
    pub fn from_env_or_detect() -> Result<Self, String> {
        match std::env::var(ACCELERATORS_ENV) {
            Ok(declared) => Self::declared(&declared),
            Err(std::env::VarError::NotPresent) => Ok(Self::detect()),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{ACCELERATORS_ENV} is not valid UTF-8"))
            }
        }
    }

    /// Parse a declaration: `none`, or a comma-separated list of
    /// `vendor[:kind]` where the kind is `gpu` (the default), `npu` or
    /// `accelerator`.
    ///
    /// # Errors
    /// When an entry names an unknown vendor or kind, or the list is empty.
    pub fn declared(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("none") {
            return Ok(Self::cpu_only());
        }
        let mut found = Vec::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let (vendor, kind) = entry.split_once(':').unwrap_or((entry, "gpu"));
            let vendor = Vendor::parse(vendor).ok_or_else(|| {
                let known: Vec<&str> = Vendor::ALL.iter().map(|v| v.as_str()).collect();
                format!(
                    "{ACCELERATORS_ENV}: unknown vendor {vendor:?} (known: {})",
                    known.join(", ")
                )
            })?;
            let kind = Kind::parse(kind).ok_or_else(|| {
                format!("{ACCELERATORS_ENV}: unknown kind {kind:?} (gpu, npu or accelerator)")
            })?;
            found.push(Accelerator { vendor, kind });
        }
        if found.is_empty() {
            return Err(format!(
                "{ACCELERATORS_ENV} is empty; write `none` for a machine without accelerators"
            ));
        }
        Ok(Self::with(found))
    }

    /// Detect the accelerators of the machine this process runs on.
    #[must_use]
    pub fn detect() -> Self {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Self::apple_silicon();
        }
        if cfg!(target_os = "windows") {
            return Self::from_windows(
                &windows_adapter_ids(DISPLAY_CLASS),
                &windows_adapter_ids(COMPUTE_CLASS),
            );
        }
        if cfg!(any(target_os = "linux", target_os = "android")) {
            return Self::from_linux(&HostProbe);
        }
        Self::cpu_only()
    }

    /// An Apple silicon machine: its GPU and its Neural Engine.
    #[must_use]
    pub fn apple_silicon() -> Self {
        Self::with([
            Accelerator {
                vendor: Vendor::Apple,
                kind: Kind::Gpu,
            },
            Accelerator {
                vendor: Vendor::Apple,
                kind: Kind::Npu,
            },
        ])
    }

    /// The accelerators of a Linux machine, read through `probe`.
    #[must_use]
    pub fn from_linux(probe: &dyn Probe) -> Self {
        let mut found = Vec::new();
        let mut add = |vendor, kind| found.push(Accelerator { vendor, kind });

        // PCI: display controllers, processing accelerators and co-processors
        // of a vendor the engine knows.
        for dev in probe.entries("/sys/bus/pci/devices") {
            let base = format!("/sys/bus/pci/devices/{dev}");
            let (Some(class), Some(id)) = (
                hex(probe.read(&format!("{base}/class"))),
                hex(probe.read(&format!("{base}/vendor"))),
            ) else {
                continue;
            };
            let Ok(id) = u16::try_from(id) else { continue };
            let Some(vendor) = Vendor::from_pci(id) else {
                continue;
            };
            match class >> 16 {
                0x03 => add(vendor, Kind::Gpu),
                0x12 => add(vendor, pci_accelerator_kind(vendor, id)),
                0x0b if (class >> 8) & 0xff == 0x40 => add(vendor, Kind::Accelerator),
                _ => {}
            }
        }

        // The kernel's compute-accelerator class, by the driver bound.
        for dev in probe.entries("/sys/class/accel") {
            match probe
                .link_target(&format!("/sys/class/accel/{dev}/device/driver"))
                .as_deref()
            {
                Some("intel_vpu") => add(Vendor::Intel, Kind::Npu),
                Some("amdxdna") => add(Vendor::Amd, Kind::Npu),
                Some("habanalabs") => add(Vendor::Intel, Kind::Accelerator),
                Some("qaic") => add(Vendor::Qualcomm, Kind::Accelerator),
                Some("rocket") => add(Vendor::Rockchip, Kind::Npu),
                Some("ethosu") => add(Vendor::Arm, Kind::Npu),
                _ => {}
            }
        }

        // GPUs that are platform devices of a system-on-chip, not PCI devices.
        for card in probe.entries("/sys/class/drm") {
            if !card.starts_with("card") || card.contains('-') {
                continue;
            }
            match probe
                .link_target(&format!("/sys/class/drm/{card}/device/driver"))
                .as_deref()
            {
                Some("panfrost" | "panthor" | "mali") => add(Vendor::Arm, Kind::Gpu),
                Some("asahi") => add(Vendor::Apple, Kind::Gpu),
                // the card hangs off `msm-mdss`, `msm_dpu`, `msm_mdp` or `adreno`
                Some(d) if d == "adreno" || d.starts_with("msm") => {
                    add(Vendor::Qualcomm, Kind::Gpu);
                }
                _ => {}
            }
        }

        // Device nodes of vendor drivers. These also cover a container, where
        // the nodes are passed in and the PCI tree may not be.
        for (node, vendor, kind) in NODE_SIGNS {
            if probe.exists(node) {
                add(*vendor, *kind);
            }
        }

        // IBM Z: the on-chip accelerator is a CPU facility, not a device.
        if let Some(cpuinfo) = probe.read("/proc/cpuinfo") {
            let nnpa = cpuinfo
                .lines()
                .filter(|l| l.starts_with("features"))
                .any(|l| l.split_whitespace().any(|f| f == "nnpa"));
            if nnpa {
                add(Vendor::Ibm, Kind::Accelerator);
            }
        }

        Self::with(found)
    }

    /// The accelerators of a Windows machine, from the hardware ids of its
    /// display adapters and of its compute accelerators (neural processors),
    /// as the device classes record them (`PCI\VEN_10DE&DEV_…`,
    /// `ACPI\VEN_QCOM&DEV_…`).
    #[must_use]
    pub fn from_windows(display_ids: &[String], compute_ids: &[String]) -> Self {
        let vendor_of = |id: &String| {
            let id = id.to_ascii_lowercase();
            let ven = id.split("ven_").nth(1)?;
            let ven: String = ven
                .chars()
                .take_while(char::is_ascii_alphanumeric)
                .collect();
            if ven == "qcom" {
                return Some(Vendor::Qualcomm);
            }
            Vendor::from_pci(u16::from_str_radix(&ven, 16).ok()?)
        };
        let gpus = display_ids
            .iter()
            .filter_map(vendor_of)
            .map(|vendor| Accelerator {
                vendor,
                kind: Kind::Gpu,
            });
        let npus = compute_ids
            .iter()
            .filter_map(vendor_of)
            .map(|vendor| Accelerator {
                vendor,
                kind: Kind::Npu,
            });
        Self::with(gpus.chain(npus))
    }

    /// Every accelerator, in a stable order.
    #[must_use]
    pub fn accelerators(&self) -> &[Accelerator] {
        &self.accelerators
    }

    /// Whether the machine has no accelerator at all.
    #[must_use]
    pub fn is_cpu_only(&self) -> bool {
        self.accelerators.is_empty()
    }

    /// Whether any accelerator of `vendor` is present.
    #[must_use]
    pub fn has(&self, vendor: Vendor) -> bool {
        self.accelerators.iter().any(|a| a.vendor == vendor)
    }

    /// Whether an accelerator of `vendor` and `kind` is present.
    #[must_use]
    pub fn has_kind(&self, vendor: Vendor, kind: Kind) -> bool {
        self.accelerators.contains(&Accelerator { vendor, kind })
    }

    /// Whether any GPU is present.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.accelerators.iter().any(|a| a.kind == Kind::Gpu)
    }

    /// The vendors present, each once.
    #[must_use]
    pub fn vendors(&self) -> Vec<Vendor> {
        let mut v: Vec<Vendor> = self.accelerators.iter().map(|a| a.vendor).collect();
        v.dedup();
        v
    }
}

impl fmt::Display for Machine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.accelerators.is_empty() {
            return f.write_str("none");
        }
        let parts: Vec<String> = self.accelerators.iter().map(ToString::to_string).collect();
        f.write_str(&parts.join(","))
    }
}

/// Device nodes whose presence shows an accelerator.
const NODE_SIGNS: &[(&str, Vendor, Kind)] = &[
    ("/dev/nvidiactl", Vendor::Nvidia, Kind::Gpu),
    ("/dev/nvhost-gpu", Vendor::Nvidia, Kind::Gpu),
    ("/dev/kfd", Vendor::Amd, Kind::Gpu),
    ("/dev/davinci_manager", Vendor::Huawei, Kind::Npu),
    ("/dev/neuron0", Vendor::Aws, Kind::Accelerator),
    ("/dev/tenstorrent", Vendor::Tenstorrent, Kind::Accelerator),
    ("/dev/kgsl-3d0", Vendor::Qualcomm, Kind::Gpu),
    ("/dev/fastrpc-cdsp", Vendor::Qualcomm, Kind::Npu),
    ("/dev/adsprpc-smd", Vendor::Qualcomm, Kind::Npu),
    ("/dev/mali0", Vendor::Arm, Kind::Gpu),
    ("/dev/cambricon_ctl", Vendor::Cambricon, Kind::Accelerator),
];

/// The kind of a device in the PCI "processing accelerators" class.
fn pci_accelerator_kind(vendor: Vendor, id: u16) -> Kind {
    match vendor {
        // Ascend and the Rockchip NPUs are neural processors.
        Vendor::Huawei | Vendor::Rockchip => Kind::Npu,
        // AMD's processor devices (1022): the XDNA NPU.
        Vendor::Amd if id == 0x1022 => Kind::Npu,
        // Intel's own id is the Core Ultra NPU; Habana's is Gaudi.
        Vendor::Intel if id == 0x8086 => Kind::Npu,
        _ => Kind::Accelerator,
    }
}

fn hex(s: Option<String>) -> Option<u32> {
    let s = s?;
    let s = s.trim();
    u32::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

/// What detection reads from the system.
pub trait Probe {
    /// Names of the entries of a directory; empty when it cannot be read.
    fn entries(&self, dir: &str) -> Vec<String>;
    /// The contents of a file, when it can be read.
    fn read(&self, path: &str) -> Option<String>;
    /// The last component of the target of a symbolic link.
    fn link_target(&self, path: &str) -> Option<String>;
    /// Whether a path exists.
    fn exists(&self, path: &str) -> bool;
}

/// The system this process runs on.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostProbe;

impl Probe for HostProbe {
    fn entries(&self, dir: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir).map_or_default(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        });
        names.sort();
        names
    }

    fn read(&self, path: &str) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    fn link_target(&self, path: &str) -> Option<String> {
        let target = std::fs::read_link(path).ok()?;
        Some(target.file_name()?.to_string_lossy().into_owned())
    }

    fn exists(&self, path: &str) -> bool {
        std::path::Path::new(path).exists()
    }
}

/// Windows device setup class of display adapters.
const DISPLAY_CLASS: &str = "{4d36e968-e325-11ce-bfc1-08002be10318}";
/// Windows device setup class of compute accelerators (neural processors).
const COMPUTE_CLASS: &str = "{f01a9d53-3ff6-48d2-9f97-c8a7004be10c}";

/// The hardware ids of the devices installed in a Windows device class, read
/// from the registry with the system's own `reg` tool. Empty when the class
/// has no device or the query fails.
fn windows_adapter_ids(class: &str) -> Vec<String> {
    let key = format!(r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{class}");
    let Ok(out) = std::process::Command::new("reg")
        .args(["query", &key, "/s", "/v", "MatchingDeviceId"])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("MatchingDeviceId"))
        .filter_map(|rest| rest.split_whitespace().last().map(str::to_string))
        .collect()
}

#[cfg(test)]
pub(crate) mod testing {
    use std::collections::BTreeMap;

    use super::Probe;

    /// A described machine: files, directory listings and links.
    #[derive(Default)]
    pub(crate) struct Described {
        files: BTreeMap<String, String>,
        links: BTreeMap<String, String>,
        nodes: Vec<String>,
    }

    impl Described {
        pub(crate) fn pci(mut self, slot: &str, vendor: &str, class: &str) -> Self {
            self.files.insert(
                format!("/sys/bus/pci/devices/{slot}/vendor"),
                format!("{vendor}\n"),
            );
            self.files.insert(
                format!("/sys/bus/pci/devices/{slot}/class"),
                format!("{class}\n"),
            );
            self
        }
        pub(crate) fn accel(mut self, name: &str, driver: &str) -> Self {
            self.links.insert(
                format!("/sys/class/accel/{name}/device/driver"),
                driver.to_string(),
            );
            self
        }
        pub(crate) fn drm(mut self, card: &str, driver: &str) -> Self {
            self.links.insert(
                format!("/sys/class/drm/{card}/device/driver"),
                driver.to_string(),
            );
            self
        }
        pub(crate) fn node(mut self, path: &str) -> Self {
            self.nodes.push(path.to_string());
            self
        }
        pub(crate) fn file(mut self, path: &str, contents: &str) -> Self {
            self.files.insert(path.to_string(), contents.to_string());
            self
        }
    }

    impl Probe for Described {
        fn entries(&self, dir: &str) -> Vec<String> {
            let prefix = format!("{}/", dir.trim_end_matches('/'));
            let mut names: Vec<String> = self
                .files
                .keys()
                .chain(self.links.keys())
                .chain(self.nodes.iter())
                .filter_map(|p| p.strip_prefix(&prefix))
                .map(|rest| rest.split('/').next().unwrap_or(rest).to_string())
                .collect();
            names.sort();
            names.dedup();
            names
        }
        fn read(&self, path: &str) -> Option<String> {
            self.files.get(path).cloned()
        }
        fn link_target(&self, path: &str) -> Option<String> {
            self.links.get(path).cloned()
        }
        fn exists(&self, path: &str) -> bool {
            let under = format!("{}/", path.trim_end_matches('/'));
            self.nodes
                .iter()
                .any(|n| n == path || n.starts_with(&under))
                || self.files.contains_key(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Described;
    use super::*;

    fn acc(vendor: Vendor, kind: Kind) -> Accelerator {
        Accelerator { vendor, kind }
    }

    #[test]
    fn a_machine_without_devices_is_cpu_only() {
        let m = Machine::from_linux(&Described::default());
        assert!(m.is_cpu_only());
        assert_eq!(m.to_string(), "none");
    }

    #[test]
    fn pci_gpus_are_found_by_vendor_and_class() {
        // a server: baseboard VGA, two NVIDIA GPUs, an Intel network card
        let m = Machine::from_linux(
            &Described::default()
                .pci("0000:03:00.0", "0x1a03", "0x030000")
                .pci("0000:17:00.0", "0x10de", "0x030200")
                .pci("0000:65:00.0", "0x10de", "0x030200")
                .pci("0000:01:00.0", "0x8086", "0x020000"),
        );
        assert_eq!(m.accelerators(), &[acc(Vendor::Nvidia, Kind::Gpu)]);
    }

    #[test]
    fn a_laptop_reports_each_vendor_and_its_npu() {
        // Ryzen AI: Radeon integrated GPU, XDNA NPU bound to its driver
        let amd = Machine::from_linux(
            &Described::default()
                .pci("0000:c5:00.0", "0x1002", "0x030000")
                .accel("accel0", "amdxdna"),
        );
        assert_eq!(
            amd.accelerators(),
            &[acc(Vendor::Amd, Kind::Gpu), acc(Vendor::Amd, Kind::Npu)]
        );
        // Core Ultra: Arc integrated GPU and the NPU as a PCI processing accelerator
        let intel = Machine::from_linux(
            &Described::default()
                .pci("0000:00:02.0", "0x8086", "0x030000")
                .pci("0000:00:0b.0", "0x8086", "0x120000"),
        );
        assert_eq!(
            intel.accelerators(),
            &[acc(Vendor::Intel, Kind::Gpu), acc(Vendor::Intel, Kind::Npu)]
        );
    }

    #[test]
    fn data_centre_accelerators_are_not_called_gpus() {
        let gaudi =
            Machine::from_linux(&Described::default().pci("0000:19:00.0", "0x1da3", "0x120000"));
        assert_eq!(
            gaudi.accelerators(),
            &[acc(Vendor::Intel, Kind::Accelerator)]
        );
        let ascend =
            Machine::from_linux(&Described::default().pci("0000:81:00.0", "0x19e5", "0x120000"));
        assert_eq!(ascend.accelerators(), &[acc(Vendor::Huawei, Kind::Npu)]);
        let mlu =
            Machine::from_linux(&Described::default().pci("0000:3b:00.0", "0xcabc", "0x120000"));
        assert_eq!(
            mlu.accelerators(),
            &[acc(Vendor::Cambricon, Kind::Accelerator)]
        );
    }

    #[test]
    fn device_nodes_show_an_accelerator_inside_a_container() {
        let ascend = Machine::from_linux(
            &Described::default()
                .node("/dev/davinci_manager")
                .node("/dev/davinci0"),
        );
        assert_eq!(ascend.accelerators(), &[acc(Vendor::Huawei, Kind::Npu)]);
        let nvidia = Machine::from_linux(&Described::default().node("/dev/nvidiactl"));
        assert_eq!(nvidia.accelerators(), &[acc(Vendor::Nvidia, Kind::Gpu)]);
        let tt = Machine::from_linux(&Described::default().node("/dev/tenstorrent/0"));
        assert_eq!(
            tt.accelerators(),
            &[acc(Vendor::Tenstorrent, Kind::Accelerator)]
        );
    }

    #[test]
    fn system_on_chip_gpus_are_found_by_their_driver() {
        let phone = Machine::from_linux(
            &Described::default()
                .drm("card0", "msm_dpu")
                .drm("card0-DSI-1", "ignored")
                .node("/dev/fastrpc-cdsp"),
        );
        assert_eq!(
            phone.accelerators(),
            &[
                acc(Vendor::Qualcomm, Kind::Gpu),
                acc(Vendor::Qualcomm, Kind::Npu)
            ]
        );
        let mali = Machine::from_linux(
            &Described::default()
                .drm("card1", "panthor")
                .accel("accel0", "ethosu"),
        );
        assert_eq!(
            mali.accelerators(),
            &[acc(Vendor::Arm, Kind::Gpu), acc(Vendor::Arm, Kind::Npu)]
        );
    }

    #[test]
    fn the_same_device_seen_twice_is_listed_once() {
        let m = Machine::from_linux(
            &Described::default()
                .pci("0000:01:00.0", "0x10de", "0x030000")
                .node("/dev/nvidiactl")
                .node("/dev/nvidia0"),
        );
        assert_eq!(m.accelerators().len(), 1);
    }

    #[test]
    fn ibm_z_reports_its_on_chip_accelerator() {
        let z = Machine::from_linux(&Described::default().file(
            "/proc/cpuinfo",
            "vendor_id       : IBM/S390\nfeatures\t: esan3 zarch stfle vx vxe nnpa sie \n",
        ));
        assert_eq!(z.accelerators(), &[acc(Vendor::Ibm, Kind::Accelerator)]);
        let x86 = Machine::from_linux(
            &Described::default().file("/proc/cpuinfo", "flags : fpu vme avx2\n"),
        );
        assert!(x86.is_cpu_only());
    }

    #[test]
    fn windows_ids_name_the_vendor() {
        let display = vec![
            r"pci\ven_10de&dev_1f95&subsys_14471025".to_string(),
            r"PCI\VEN_1002&DEV_1636&REV_C6".to_string(),
        ];
        let m = Machine::from_windows(&display, &[]);
        assert_eq!(
            m.accelerators(),
            &[acc(Vendor::Nvidia, Kind::Gpu), acc(Vendor::Amd, Kind::Gpu)]
        );
        let arm = Machine::from_windows(
            &[r"ACPI\VEN_QCOM&DEV_0C36".to_string()],
            &[r"ACPI\VEN_QCOM&DEV_0D0A".to_string()],
        );
        assert_eq!(
            arm.accelerators(),
            &[
                acc(Vendor::Qualcomm, Kind::Gpu),
                acc(Vendor::Qualcomm, Kind::Npu)
            ]
        );
        let npu = Machine::from_windows(&[], &[r"PCI\VEN_8086&DEV_7D1D".to_string()]);
        assert_eq!(npu.accelerators(), &[acc(Vendor::Intel, Kind::Npu)]);
    }

    #[test]
    fn a_declaration_replaces_detection() {
        assert!(Machine::declared("none").unwrap().is_cpu_only());
        let m = Machine::declared("AMD, amd:npu ,google:accelerator").unwrap();
        assert_eq!(
            m.accelerators(),
            &[
                acc(Vendor::Amd, Kind::Gpu),
                acc(Vendor::Amd, Kind::Npu),
                acc(Vendor::Google, Kind::Accelerator)
            ]
        );
        assert_eq!(Machine::declared(&m.to_string()).unwrap(), m);
        assert!(
            Machine::declared("voodoo")
                .unwrap_err()
                .contains("unknown vendor")
        );
        assert!(
            Machine::declared("amd:fpga")
                .unwrap_err()
                .contains("unknown kind")
        );
        assert!(Machine::declared(" ").unwrap_err().contains("empty"));
    }

    #[test]
    fn detection_runs_on_this_machine() {
        // whatever it finds, it is well-formed and stable
        let m = Machine::detect();
        assert_eq!(Machine::with(m.accelerators().iter().copied()), m);
    }
}
