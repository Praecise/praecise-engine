# Foundations: what sits under the silicon

Chip vendors are the visible layer. Under them, a small number of foundries,
packaging lines, memory makers and interconnect standards decide what silicon
can exist, how much memory it carries and how fast that memory is read. An
engine that plans only against today's vendor lineup is planning against the
output of those decisions and not the decisions.

This note records the state of that layer in November 2026 and, for each
part, the consequence for this engine. The premise it is read against is the
one this engine is built on: **decode is memory-bandwidth-bound**
([gb10-bandwidth-ceiling.md](gb10-bandwidth-ceiling.md)), so memory and the
links between memories are the first-order facts.

⚠ Most of this comes from trade press and analyst estimates gathered on
2026-11-12, not from primary documents. Roadmap dates slip. Treat every
figure as a ranking aid and every date as provisional; where a standard body
or vendor published the fact, the source is named.

## Process

- **TSMC** makes nearly every leading accelerator. N3 is the binding wafer
  constraint for 2026 AI silicon; N2 has been in volume since late 2025; A16
  (backside power) slipped to 2027; A14 is 2028.
- **Samsung** ships SF2 in Exynos 2600 and has Tesla's AI6 on SF2P for late
  2027. **Intel** ships 18A in Panther Lake; 14A is 2028 to 2029.
- **SMIC** reaches a 5 nm-class node (N+3, below TSMC N5 density) by DUV
  multipatterning, without EUV. Huawei's Ascend line depends on it.

**Consequence.** Process sets compute per watt; it does not move the decode
ceiling, which memory sets. Two fleets follow from it: one on leading nodes
with HBM4, one in China on 7 nm and 5 nm-class nodes with scarcer, slower
memory. The second is not a slower copy of the first. It needs the INT8 and
FP8 paths, aggressive weight compression and CPU fallbacks kept as well
tested as the FP4 paths.

## Packaging

- **CoWoS** capacity, not wafers, limits how many accelerators are built, and
  the interposer limits how much memory each carries: about 12 HBM stacks at
  the current size, with larger interposers targeted for 2027. Warpage
  reportedly forced one 2027 design back from 16 stacks.
- Panel-level packaging and glass substrates are 2027 to 2029.

**Consequence.** Memory per accelerator plateaus (288 GB, 432 GB, then about
768 GB to 1 TB) while memory per rack grows. A model that does not fit one
device does not wait for a bigger device; it spans devices. Sharding, and
emitting more tokens per pass over the weights, stay the main levers.

## Memory

This is the part that bounds the engine directly.

| Tier | Capacity | Bandwidth | State |
|---|---|---|---|
| HBM4 (JEDEC JESD270-4) | 36 GB per 12-high stack; to 64 GB at 16-high | 2 TB/s per stack at the base pin speed | shipping |
| LPDDR5X SOCAMM2 | 192 GB modules; 1.5 to 2 TB per server CPU | about 1.2 TB/s per CPU | shipping |
| DDR5 MRDIMM | per channel | about 70 GB/s (8800), about 102 GB/s (12800) | 8800 shipping; 12800 near standard |
| LPDDR6 (JEDEC JESD209-6) | to 512 GB per module in the next revision | 10,667 to 14,400 MT/s | announced |
| CXL 4.0 | expansion, not pooling, in real deployments | bounded by the link | expansion shipping |
| High-bandwidth flash (OCP) | to 512 GB per stack | 0.4 to 3 TB/s by grade, read-mostly | announced |

- All three HBM makers are qualified for the current generation; HBM4E with
  customer-specific base dies is 2027.
- DRAM and NAND are short through 2027, with prices rising each quarter.
  Client devices are being built with less memory than 2025 roadmaps assumed.
- Vendors are building a tier for KV cache between memory and storage.

**Consequences.**

1. **Model memory as tiers, not as "VRAM and RAM".** Weights, live KV cache
   and cold KV state have different read rates and belong in different tiers.
   `praecise_hardware::silicon::MEMORY_TIERS` is the start of that table.
2. **Measure bandwidth; do not infer it from a part's name.** Pin speed
   varies by vendor and bin, MRDIMM against RDIMM changes a socket by half
   again, and CXL memory sits behind a slower link. `praecise-hw measure`
   measures what a machine's CPU threads reach.
3. **KV-cache placement is a policy of its own.** Prefix checkpoints already
   live in host memory ([prefix-reuse.md](prefix-reuse.md)); the same
   interface should take a slower, larger tier when one exists, with the cost
   of restoring set against the cost of recomputing.
4. **Plan for less memory on client devices.** Sub-4-bit weights, partial
   residency of mapped weights and small default contexts matter more, not
   less.

## Interconnect

- **Inside a rack**: NVLink 6 at 3.6 TB/s per GPU; UALink 2.0 published
  April 2026, with 1.0 silicon in products in 2027; AMD's Helios ships now on
  UALink over Ethernet. Ultra Ethernet 1.0 is published and shipping in NICs.
- **Between cards**: PCIe 6 is the shipping generation; PCIe 7 is specified;
  PCIe 8 is a draft for 2028.
- **Between desktops**: RDMA over Thunderbolt 5 on macOS 26.2 and later, in a
  full mesh, at about 50 to 60 Gb/s. Apple reports tensor-parallel decode of
  an 816B model across four machines on it.

**Consequence.** The transport is not one thing and will not converge before
2027. A multi-device layout should be given a topology (hops, bandwidth,
latency) and not assume a fabric. The engine's layer pipeline
(`praecise_runtime::pipeline`) splits whole blocks across machines on a LAN;
which split pays on which link is a measurement to take per topology, not a
rule to derive here. A rule drafted for this note, that a slow link forces a
whole-layer split, was dropped because published Thunderbolt results
contradict it.

## Datatypes and instruction sets

- **OCP Microscaling (MX)**: MXFP8, MXFP6, MXFP4 and MXINT8, blocks of 32
  with a shared 8-bit scale. **NVFP4** uses blocks of 16 with an E4M3 scale.
  The two 4-bit formats are not interchangeable. AMD CDNA 4 and NVIDIA
  Blackwell implement block-scaled formats natively.
- **IEEE P3109** (arithmetic formats for machine learning) is an unapproved
  draft with no silicon.
- **x86**: ACE (matrix multiply into AMX-style tiles, block-scaled FP4) and
  AVX10.2 are specified; no CPU with either has shipped. First parts are
  expected from late 2026 (AVX10.2, APX) to 2028 or later (ACE).
- **Arm**: SME2 is on Apple M4 and M5, Exynos 2600 and Arm's C1-Ultra; server
  cores of the Neoverse V3 class have SVE2 without it.
- **RISC-V**: RVA23 silicon ships; no matrix extension is ratified.

**Consequences.**

5. **Block-scaled 4-bit is a family, not a type.** A tensor has to carry its
   block size and scale encoding so that one artefact can be converted at
   load for the device it lands on.
6. **FP8 E4M3 is the interchange type** between vendors; the fork gains it
   with the next upstream sync (charter item D).
7. **CPU detection covers what ships.** `praecise_hardware::cpu` reports
   AVX-512, AVX-VNNI and AMX, the Arm dot-product, int8-matmul, SVE and SME
   families, RISC-V vector and LoongArch LSX/LASX. AVX10.2, APX and ACE are
   added when silicon exists to test against.

## Power

Racks are power-capped (about 120 to 140 kW for current flagship racks, with
600 kW designs announced), and tokens per watt is becoming the figure buyers
compare. MLPerf Power validates wall power for a whole system; there is no
standard tokens-per-joule metric.

**Consequence.** Throughput reported without energy is half a measurement on
hardware that is bought by the megawatt. Energy per token belongs beside
tokens per second where a machine exposes power, labelled with where the
reading came from.

## Compute that sidesteps the premise

Wafer-scale and SRAM-centric parts (Cerebras, Groq, d-Matrix) hold weights in
on-chip memory, so their bound is capacity and compute, not DRAM bandwidth.
They are reached over an API today, which this engine does only through its
opt-in remote backend ([silicon-charter.md](silicon-charter.md)), where the
provider's hardware cannot be measured or steered. If one becomes locally
addressable it is a different class of device: small, latency-critical
models, such as a drafter for speculation, are what it is for.

## Sources

Standards: JEDEC (JESD270-4 HBM4, JESD209-6 LPDDR6), the CXL Consortium (CXL
4.0), the UALink Consortium (UALink 2.0), the Ultra Ethernet Consortium, the
Open Compute Project (Microscaling Formats; the high-bandwidth flash
workstream), PCI-SIG. Vendor announcements: TSMC, Samsung, Intel, AMD,
Apple, SK hynix, Micron, Sandisk. Analysis and reporting: TrendForce,
SemiAnalysis, Tom's Hardware, The Next Platform, ServeTheHome, Phoronix.
