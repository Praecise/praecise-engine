//! Silicon vendors and how the operating system names them.

use std::fmt;

/// A silicon vendor whose accelerators the engine can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Vendor {
    /// NVIDIA: `GeForce`, RTX, the data-centre GPUs, Jetson and GB10.
    Nvidia,
    /// AMD: Radeon, Instinct, the Ryzen integrated GPUs and the XDNA NPUs.
    Amd,
    /// Intel: Arc, the integrated Xe GPUs, the Core Ultra NPUs and Gaudi.
    Intel,
    /// Apple silicon: the M-series and A-series GPU and Neural Engine.
    Apple,
    /// Qualcomm: Adreno GPUs, the Hexagon NPU and the Cloud AI cards.
    Qualcomm,
    /// Arm: Mali and Immortalis GPUs.
    Arm,
    /// Huawei: the Ascend NPUs.
    Huawei,
    /// Google: the TPUs.
    Google,
    /// Amazon: Trainium and Inferentia.
    Aws,
    /// Moore Threads: the MTT GPUs.
    MooreThreads,
    /// Cambricon: the MLU accelerators.
    Cambricon,
    /// Hygon: the DCU accelerators.
    Hygon,
    /// `MetaX`: the C-series GPUs.
    MetaX,
    /// Iluvatar `CoreX`.
    Iluvatar,
    /// Enflame: the GCU accelerators.
    Enflame,
    /// Tenstorrent: Wormhole and Blackhole.
    Tenstorrent,
    /// `FuriosaAI`: RNGD.
    Furiosa,
    /// Rebellions.
    Rebellions,
    /// IBM: the Telum on-chip accelerator and Spyre.
    Ibm,
    /// Rockchip: the `RK35xx` NPUs.
    Rockchip,
    /// Esperanto ET-SoC.
    Esperanto,
}

impl Vendor {
    /// Every vendor, in a stable order.
    pub const ALL: &'static [Vendor] = &[
        Vendor::Nvidia,
        Vendor::Amd,
        Vendor::Intel,
        Vendor::Apple,
        Vendor::Qualcomm,
        Vendor::Arm,
        Vendor::Huawei,
        Vendor::Google,
        Vendor::Aws,
        Vendor::MooreThreads,
        Vendor::Cambricon,
        Vendor::Hygon,
        Vendor::MetaX,
        Vendor::Iluvatar,
        Vendor::Enflame,
        Vendor::Tenstorrent,
        Vendor::Furiosa,
        Vendor::Rebellions,
        Vendor::Ibm,
        Vendor::Rockchip,
        Vendor::Esperanto,
    ];

    /// Stable lowercase identifier, used in configuration and logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
            Vendor::Intel => "intel",
            Vendor::Apple => "apple",
            Vendor::Qualcomm => "qualcomm",
            Vendor::Arm => "arm",
            Vendor::Huawei => "huawei",
            Vendor::Google => "google",
            Vendor::Aws => "aws",
            Vendor::MooreThreads => "moore-threads",
            Vendor::Cambricon => "cambricon",
            Vendor::Hygon => "hygon",
            Vendor::MetaX => "metax",
            Vendor::Iluvatar => "iluvatar",
            Vendor::Enflame => "enflame",
            Vendor::Tenstorrent => "tenstorrent",
            Vendor::Furiosa => "furiosa",
            Vendor::Rebellions => "rebellions",
            Vendor::Ibm => "ibm",
            Vendor::Rockchip => "rockchip",
            Vendor::Esperanto => "esperanto",
        }
    }

    /// Parse a vendor identifier, ignoring case and the `-`/`_` spelling.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let want = s.trim().to_ascii_lowercase().replace('_', "-");
        Vendor::ALL
            .iter()
            .copied()
            .find(|v| v.as_str() == want || v.as_str().replace('-', "") == want)
    }

    /// The vendor a PCI vendor id belongs to (ids from the PCI ID database).
    ///
    /// AMD has two: `1002` for its GPUs and `1022` for the processor devices,
    /// the NPU among them. Habana Labs (`1da3`) is Intel's Gaudi.
    #[must_use]
    pub const fn from_pci(vendor_id: u16) -> Option<Self> {
        Some(match vendor_id {
            0x10de => Vendor::Nvidia,
            0x1002 | 0x1022 => Vendor::Amd,
            0x8086 | 0x1da3 => Vendor::Intel,
            0x106b => Vendor::Apple,
            0x17cb | 0x5143 => Vendor::Qualcomm,
            0x13b5 => Vendor::Arm,
            0x19e5 => Vendor::Huawei,
            0x1d0f => Vendor::Aws,
            0x1ed5 => Vendor::MooreThreads,
            0xcabc => Vendor::Cambricon,
            0x1d94 => Vendor::Hygon,
            0x9999 => Vendor::MetaX,
            0x1e3e => Vendor::Iluvatar,
            0x1e36 => Vendor::Enflame,
            0x1e52 => Vendor::Tenstorrent,
            0x1ed2 => Vendor::Furiosa,
            0x1eff => Vendor::Rebellions,
            0x1d87 => Vendor::Rockchip,
            _ => return None,
        })
    }
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for v in Vendor::ALL {
            assert_eq!(Vendor::parse(v.as_str()), Some(*v));
        }
        assert_eq!(Vendor::parse("Moore_Threads"), Some(Vendor::MooreThreads));
        assert_eq!(Vendor::parse("moorethreads"), Some(Vendor::MooreThreads));
        assert_eq!(Vendor::parse("3dfx"), None);
    }

    #[test]
    fn pci_ids_map_to_their_vendor() {
        assert_eq!(Vendor::from_pci(0x10de), Some(Vendor::Nvidia));
        assert_eq!(Vendor::from_pci(0x1002), Some(Vendor::Amd));
        assert_eq!(Vendor::from_pci(0x1022), Some(Vendor::Amd));
        assert_eq!(Vendor::from_pci(0x1da3), Some(Vendor::Intel));
        assert_eq!(Vendor::from_pci(0x19e5), Some(Vendor::Huawei));
        assert_eq!(Vendor::from_pci(0xcabc), Some(Vendor::Cambricon));
        // a baseboard controller's display device is nobody's accelerator
        assert_eq!(Vendor::from_pci(0x1a03), None);
    }
}
