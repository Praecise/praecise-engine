# Silicon coverage

Praecise Engine is described as backend-agnostic. Until November 2026 that
was true of its backends and not of its hardware: every route that touched a
device assumed NVIDIA, with AMD and Apple as partial exceptions. This page
is the account of that difference: what the engine had, every gap found,
what was done about each, and what remains that cannot be done without the
hardware.

Surveyed 2026-11-12 against vendor documentation, release notes and source.
Version-specific facts are dated, because they move.

## Principles

1. **No vendor is the default.** A machine's accelerators are read from the
   operating system, by vendor and kind. Nothing above that layer carries a
   private idea of what "a GPU" is.
2. **An accelerator is used or the load is refused.** The rule that already
   held for NVIDIA holds for every vendor: a machine with an accelerator never
   runs a model on the CPU because the build lacks a backend. The refusal names
   the build that would work.
3. **A claim says how far it goes.** `verified`, `built`, `compiled` and
   `wired` are different words ([hardware.md](hardware.md)). A route that was
   never run on its silicon is not called supported without that qualifier.
4. **Published figures rank; measurements decide.** The catalogue's bandwidth
   figures are ceilings. Decode on a measured machine reaches 44% to 80% of
   them ([gb10-bandwidth-ceiling.md](gb10-bandwidth-ceiling.md)).
5. **Decode is memory-bound on every vendor.** The arithmetic that places a
   GB10 places an Instinct card, an M-series Mac and a TPU. Hardware is
   ranked by memory and its bandwidth first, compute second.

## What the engine had

| Area | State before |
|---|---|
| ggml features on `llama-cpp-sys-2` | cuda, metal, vulkan, opencl, rocm, sycl, openvino, musa, cann, webgpu, blas, mkl, zdnn, rpc |
| ggml backends in the fork with no feature | Hexagon, ZenDNN, ET, VirtGPU, KleidiAI |
| Features reachable from `praecise-runtime` | cuda, rocm, metal, vulkan |
| Features on `praecise-diffusion` | cuda, metal |
| ONNX providers | CUDA and TensorRT, ROCm and MIGraphX, CoreML; `Gpus { nvidia, amd }` |
| MoE expert kernels | CPU (AVX-512-VNNI, NEON), cuBLAS, wgpu |
| Hosted backends | vLLM, SGLang, TensorRT-LLM, transformers; the adapter required `torch.cuda` |
| Host confinement | device nodes `nvidia*`, `kfd`, `/dev/dri`; `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` carried through |
| Hardware detection | none in the engine; two crates each checked `/dev/nvidiactl` and `/dev/kfd` themselves |
| Training kernel classes | CPU, CUDA, Metal, Vulkan |
| Documentation | NVIDIA architectures; everything else "planned" |
| Verification | GB10 (sm_121) and RTX 5070 Ti |

## Gaps, and what was done

### Closed

| Gap | What exists now |
|---|---|
| No single account of a machine's accelerators | `praecise-hardware`: detection for 21 vendors from PCI ids, kernel driver bindings and device nodes; Linux, Windows and macOS; `PRAECISE_ACCELERATORS` to declare a machine; tested against described machines |
| No account of which software drives which silicon | `praecise_hardware::route`: ggml backend, torch device and vLLM platform per accelerator; `praecise-hw` prints the build a machine needs |
| A backend could be assigned to silicon it cannot reach | `Backend::reaches`, `reachable` and `ensure_reaches`; a refusal names the backends that do reach the machine |
| Intel silicon had no hosted route of its own | `Backend::OpenVino`: OpenVINO GenAI on Intel CPUs, GPUs and NPUs, as a hosted engine |
| Hosted engines ran only on CUDA | The adapter takes the accelerator the environment drives: torch `cuda` (NVIDIA, ROCm), `xpu`, `mps`, `hpu`, `npu`, `musa`; vLLM's own platform, including every vendor plugin; SGLang's device argument. TensorRT-LLM tells ROCm from CUDA and refuses it |
| The memory cap was read from `torch.cuda` | Read from the device that is present; an engine whose platform reports no size takes `memory_fraction` from the operator or refuses |
| A confined host could open only NVIDIA and AMD devices | Device nodes of every vendor; vendor runtime directories; every vendor's device-selection variables |
| ONNX models ran on the CPU on Intel, Qualcomm and Huawei machines | OpenVINO, QNN, CANN, Vitis AI, DirectML and WebGPU providers, with the same refuse-or-accelerate planning; NPUs on request |
| The ROCm execution provider no longer exists in ONNX Runtime (removed in 1.23) | The `rocm` feature builds MIGraphX; the dead provider is gone |
| Five ggml backends in the fork had no feature | `hexagon`, `zendnn`, `et`, `virtgpu`, `kleidiai` |
| Ten ggml backends could not be selected from `praecise-runtime` | Every backend is a feature of `praecise-runtime` and `praecise-diffusion` |
| AVX-512 foundation and AMX were never passed to ggml | Mapped from the Rust target features; AVX-512 only when all five subsets are present |
| A published `llama-cpp-sys-2` crate dropped the sources of ten backends | The package include list names every backend directory |
| MoE kernels ignored AVX-VNNI and the Arm dot-product extension | AVX-VNNI tier for Intel client cores; `SDOT` tier for Armv8.2 and later; tiers proved equal bit for bit where the CPU has both |
| Diffusion refused the CPU only on NVIDIA, AMD and Apple machines | The refusal covers every accelerator a ggml backend can drive |
| Training recognised four device families | Every backend's devices are recognised and a run is held to its device; deterministic mode refuses them until audited |
| `praecise-track` detected NVIDIA and AMD itself | Reads the machine from the hardware layer |
| No test ran on ARM, macOS or Windows for the hardware paths | `silicon-coverage` workflow: four host platforms, every ONNX provider, and ZenDNN, KleidiAI and Metal linked into a binary. The workflow has not run yet (see the open gaps below) |
| No figure for non-NVIDIA systems | `praecise_hardware::silicon`: memory and bandwidth of 17 systems across six vendors, each tagged with its source and whether it ships; `praecise-hw catalogue` |
| Bandwidth was inferred from a part's name | `praecise-hw measure` measures what the CPU threads of the machine reach |
| Silicon sold only as a service had no route at all | `praecise-remote`: an opt-in client for OpenAI-compatible inference APIs (Groq, Cerebras, SambaNova or any other), answering in the runtime's types. It exists only in a build that depends on it and sends only to hosts the operator names in `PRAECISE_REMOTE_ALLOW` |
| bindgen did not compile from the workspace lockfile, so nothing that links llama.cpp built | bindgen is built without its `prettyplease` feature (bindgen 0.73 parses with syn 2; prettyplease 0.3 moved to syn 3) |

### Open: needs the hardware

Each of these is a real gap that cannot be closed from a build host. None is
hidden behind a passing check.

| Gap | What closing it takes |
|---|---|
| Every non-NVIDIA row is unverified | A run on the silicon, recorded in [hardware.md](hardware.md). In rough order of reach: AMD Instinct and Radeon (HIP, Vulkan), Apple M-series (Metal), Intel Arc (SYCL, OpenVINO), Ryzen AI and Core Ultra NPUs, Ascend, Snapdragon |
| Per-hardware, per-quantization dispatch tables exist only for NVIDIA | Pinned-clock sweeps on each architecture (`gfx942`, `gfx950`, `gfx1151`, Apple, Arc), as [acceleration-research.md](acceleration-research.md) describes for NVIDIA |
| No deterministic kernel set off CPU and CUDA | An audit of reduction order per backend on its own device. HIP and MUSA compile the CUDA kernel sources but use hipBLAS and muBLAS and split by a different device, so the CUDA audit does not carry over |
| The fork's sync with llama.cpp 1.0.0 has not been compiled | The fork is synced in source with upstream 1.0.0 (ggml 1.0.0), which removes the deprecated API, moves the multimodal code out of `clip` and changes the CUDA attention and quantized-product kernels the fork patches. The merge and the binding changes were made by reading: nothing has been compiled against the synced tree, on the CPU or on any GPU backend, and no test has run. The last tree proved, on the CPU only, is the sync with 0.9.0. Closing this takes a CPU build with the architecture and gradient tests, the crates type-checked and the runtime tests run, then a CUDA build, the backend comparison tests on each backend, and the fork's patches re-validated on hardware |
| No hosted engines on macOS | Confinement for macOS written and tested on a Mac. Until then Apple is reached through linked Metal and CoreML, and MLX and vllm-metal are not hosted |
| Static links for vendor-SDK backends are unproven | A build of `sycl`, `openvino`, `cann`, `musa`, `zdnn`, `webgpu`, `hexagon` and `et` on a machine with each SDK. Until then a static build leaves the vendor runtime to the final crate and a `dynamic-link` build carries it |
| The Windows compute-accelerator device class is unchecked on an NPU | A run of `praecise-hw` on a Windows machine with an NPU |
| Cambricon's torch device is unconfirmed from a primary source | Its `torch_mlu` package; until then Cambricon is reached through its vLLM plugin only |
| The `silicon-coverage` workflow has not run | Its first green run. The CPU build of llama.cpp and every crate above it were built and type-checked once, on Windows x86-64 with MinGW; nothing was built on Linux, AArch64 or macOS. Until the workflow runs, `zendnn`, `kleidiai`, `metal` and the AVX-512 options are `wired`, not `built`, and the ONNX providers are compiled on that one build host only |
| The remote backend has not met a real provider | A run against Groq, Cerebras and SambaNova with a key. It is tested against a local server speaking the same protocol; whether each provider returns usage in a stream, and which length field each takes, is read from their client libraries and not from a response |
| Quantum jobs simulate on NVIDIA GPUs or the CPU only, and no quantum processor is reachable | A simulation program on a simulator with an AMD path (Qiskit Aer on ROCm, `lightning.kokkos` or QuEST with HIP), run on that hardware; and, for a real device, an opted-in network route for jobs. See "Quantum" below |

### Out of scope, by design

- **Silicon with no public interface.** Microsoft Maia, Meta MTIA and
  OpenAI's own chips are not reachable by a third party at all.
- **A network inside a runtime host.** Hosts and jobs stay without one. The
  remote backend is not a host: it is a separate crate, off unless built in,
  and it sends only where the operator permits.
- **Job engines.** Warp, Sionna, Cosmos, PhysicsNeMo and CUDA-Q are NVIDIA's
  own systems. The ones built on torch run on ROCm, which presents as `cuda`;
  the rest are CUDA programs. An equivalent on another vendor's stack is a
  different program, not a port.

## The state of silicon, November 2026

What the list above was built against. ⚠ marks a figure from press coverage
that was not confirmed against the vendor.

**AMD.** Instinct MI350X/MI355X (CDNA 4, 288 GB HBM3E, ⚠ 8 TB/s) ship with
native MXFP8/MXFP6/MXFP4; the MI400 series launched 2026-07-23 with MI455X
(⚠ 432 GB HBM4, 19.6 TB/s) in the 72-GPU Helios rack. ROCm is 10.1.0
(2026-10-05), built by TheRock; it pairs with vLLM 0.29 and SGLang 0.5.18.
Ryzen AI Max (Strix Halo) carries 128 GB unified at ⚠ 256 GB/s. ONNX Runtime
dropped its ROCm provider in 1.23; MIGraphX is now a plugin provider.

**Intel.** Arc Pro B-series (to 32 GB) are the current GPUs; Crescent Island
(Xe3P, 160 GB) is enabled in oneAPI 2026.2 and not shipping. Gaudi 3 is
supported through a vLLM hardware plugin; the vLLM fork is end of life. IPEX
and IPEX-LLM are archived. OpenVINO is 2026.4.1 with a GGUF backend in
llama.cpp (`GGML_OPENVINO`, device chosen by `GGML_OPENVINO_DEVICE`). vLLM
has a built-in XPU platform. Xeon 6 reaches ⚠ 845 GB/s on twelve channels of
MRDIMM-8800; AVX10.2 and APX arrive with Nova Lake and Diamond Rapids.

**Apple.** M5 Max (128 GB, 614 GB/s) and M5 Ultra (512 GB, 1.2 TB/s) ship;
M6 (32 GB, 170 GB/s) is in the Mac mini and 14-inch MacBook Pro. MLX is
0.32.4 with a CUDA backend on Linux and RDMA over Thunderbolt 5 for
clustering. `vllm-metal` runs vLLM's scheduler on MLX. macOS 27 adds Core AI;
it has no ONNX provider yet.

**Google.** TPU 8t and 8i were announced 2026-04-22 (8i: 288 GB HBM,
8.6 TB/s); general availability is unconfirmed. Inference runs through vLLM's
TPU platform (`tpu-inference`) and, in preview, TorchTPU.

**Arm.** KleidiAI 1.33 carries SME2, SVE2 and SVE kernels and is in
llama.cpp (`GGML_CPU_KLEIDIAI`), ONNX Runtime 1.31 and XNNPACK. Neoverse
V3-class servers (Graviton5, Cobalt 200) have SVE2 and no SME2; Apple cores
and Arm's C1/C2 mobile cores have SME2. The kernel now has a compute
accelerator driver for Ethos-U.

**Qualcomm.** llama.cpp has an OpenCL backend for Adreno and a Hexagon NPU
backend (`GGML_HEXAGON`, experimental, v73 to v81, about 3.5 GB per NPU
session). The QNN provider takes quantized, static-shape models. AI200 racks
are sampling; AI250 is 2027.

**Huawei.** Ascend 910B/910C are the installed base; 950PR/950DT are entering
use; 960 is 2027. CANN is 9.2.1 and open source. `vllm-ascend` 0.30.0 tracks
vLLM 0.30.0 with `torch_npu` 2.10; llama.cpp's CANN backend documents 910B
and 310P. The ONNX Runtime CANN provider is community-maintained and covers
convolutional models, not language models.

**Others with a route.** Moore Threads (MUSA backend in llama.cpp, vLLM-MUSA,
`torch_musa`), Cambricon, MetaX, Iluvatar and Enflame (vLLM plugins, each
pinned to a different vLLM), AWS Trainium (`vllm-neuron`, Neuron SDK 2.33),
Tenstorrent and FuriosaAI (their own vLLM-compatible servers), IBM Z (zDNN;
Spyre through `vllm-spyre`).

**The frameworks.** llama.cpp 1.0.0 (ggml 1.0.0), released 2026-11-13; ONNX Runtime 1.31.1, with
DirectML in sustained engineering and execution providers moving to
separately released plugins; the `ort` crate 2.0.0-rc.15; vLLM 0.33 with
CUDA, ROCm, XPU, TPU and CPU platforms built in and a platform-plugin entry
point for the rest; SGLang 0.5.23 with a device argument for `cuda`, `xpu`,
`hpu`, `npu` and `musa`; PyTorch 2.15 with `torch.accelerator` as the
vendor-neutral device API (memory information from 2.10).

## Quantum

Surveyed 2026-11-12. ⚠ marks what came from secondary coverage.

- **No quantum processor accelerates language-model inference.** The one
  demonstration on record (Multiverse Computing and IBM, arXiv:2605.05914,
  May 2026) put small quantum-circuit adapters into Llama 3.1 8B and ran
  inference end to end on a 156-qubit IBM Heron r2. It improved perplexity by
  1.4% and its authors claim no quantum advantage; ⚠ one report puts a
  129-token reply at about 90 minutes. A quantum processor is not an
  inference device, and this engine does not treat it as one.
- **The traffic runs the other way.** Classical accelerators serve quantum
  hardware: CUDA-Q QEC 0.6 (April 2026) decodes error correction on GPUs in
  real time over NVQLink, and circuits are simulated on GPUs. That is work
  this engine's job engines are the right shape for.
- **What the engine has.** Two job programs on NVIDIA CUDA-Q (0.16.0,
  2026-09-12): `cudaq.ghz` samples a GHZ state and `cudaq.maxcut` runs QAOA on
  a graph. Each takes a simulator target. `qpp-cpu` is a CPU state-vector
  simulator and runs on any machine; `nvidia` and `tensornet` need an NVIDIA
  GPU, and CUDA-Q's GPU simulation is NVIDIA-only in practice.
- **Vendor-neutral simulation exists, outside CUDA-Q.** Qiskit Aer has a
  ROCm build for AMD GPUs (⚠ `qiskit-aer-gpu-rocm`), PennyLane's
  `lightning.kokkos` builds with HIP for AMD, qsim has a HIP backend for
  Instinct, and QuEST builds with CUDA or HIP. None is confirmed on Intel or
  Apple GPUs. A simulation program on one of these is the way to a quantum
  job that does not need NVIDIA; none is written yet.
- **Real devices.** Quantinuum Helios, IonQ Tempo and IBM's systems are
  reached through a provider's cloud, with credentials; some are also sold
  for installation on site (IQM reports 23 systems), and Google's Willow is
  open to selected researchers only. Whether any can be driven without a
  network by third-party software is not established. A job is confined
  without a network, so reaching a device is the same decision as the remote
  backend, for jobs: a separate, opted-in route that names its hosts. It is
  not built.

## Staying ahead

What to watch, and the change each would call for. None is built on
speculation; each is a note of where the next gap will open.

| Signal | Why it matters here | Response when it lands |
|---|---|---|
| ONNX Runtime plugin execution providers | Providers stop being compile-time features of one runtime binary | Register provider libraries at run time; keep the planning rules |
| Windows ML execution-provider catalogue | The system, not the application, supplies providers on Windows | Enumerate the catalogue instead of DirectML |
| Core AI on macOS 27 | The Neural Engine's new front door; no ONNX path yet | A provider, when ONNX Runtime has one |
| OCP FP8 E4M3 in ggml | The interchange type between vendors; the fork has it since the sync with 0.9.0 | Converters and loaders that carry it, once a model ships in it |
| Block-scaled 4-bit formats diverging | MXFP4 (blocks of 32, E8M0 scale) and NVFP4 (blocks of 16, E4M3 scale) are not interchangeable; x86 ACE defines its own | Carry block size and scale format with a tensor, convert at load |
| x86 AVX10.2, APX, ACE | Matrix and block-scaled arithmetic on CPUs from 2026 to 2029 | Detection in `praecise_hardware::cpu` once the standard library or CPUID leaf is settled |
| A ratified RISC-V matrix extension | None is ratified; three proposals overlap | No path until one is |
| Vendor KV tiers (CXL, high-bandwidth flash) | KV-cache placement becomes a tier choice, not a device choice | [foundations.md](foundations.md), consequence 3 |
| A vLLM plugin converging on upstream | Fewer per-vendor engine environments | Collapse the environments that can share a vLLM |

## Changelog

- **2026-11-13.** The llama.cpp fork is synced in source with upstream 1.0.0
  (ggml 1.0.0). Upstream removed its deprecated API, so the bindings call
  the replacements it names (`llama_model_*`, `llama_vocab_*`,
  `llama_state_*`, `llama_init_from_model`, `ggml_prec_set_acc`), and
  `defrag_thold` is gone from the context parameters; the MoE cache size and
  its CPU share are now settable from Rust. Upstream dissolved `clip` into
  `mtmd`, and the fork's Inkling towers, Kimi-K3 projector and Qwen3-Omni
  talker generator moved with it. Not compiled: the merged tree, the crates
  against it, any GPU backend. Not run: any test. The CUDA attention and
  quantized-product kernels, which both upstream and the fork changed, were
  merged by reading. No speed figure is claimed for this release.
- **2026-11-13.** The llama.cpp fork is synced with upstream 0.9.0 (ggml
  0.29). Upstream's row-list attention and the fork's banded attention read
  the same tensor slot, and each reader now tells them apart by operation.
  Inkling and DiffusionGemma moved to upstream's attention-mask type. Built
  on the CPU (Windows x86-64, MinGW): the architecture test passes 152 of
  152 and the gradient test passes. The crates that link llama.cpp
  type-check against it and the runtime's 272 library tests pass. Not
  compiled: any GPU backend, including the CUDA attention kernels the merge
  touched. Not run: the backend comparison tests, which need a second
  backend. Upstream publishes no speed figures for 0.8.0 or 0.9.0, and none
  was measured here.
- **2026-11-13.** Versions refreshed: the lockfile and the four crates that
  had a newer release (`cc`, `encoding_rs`, `openh264`, `wgpu`); the cloud
  build image moved from Debian 12 to Debian 13; the CUDA test image moved
  to CUDA 13.5.0 on Ubuntu 26.04. Neither image has been built since the
  change. Every other declared crate was already at its newest stable
  release; `ort`, `dashmap`, `tokenizers` and `libc` have newer
  pre-releases only, and stay where they are. A November survey of decode
  speedups was added to [acceleration-research.md](acceleration-research.md):
  no new batch-1 kernel was found.
- **2026-11-12.** Remote backend (`praecise-remote`) added as an opt-in
  route to API-only silicon. Quantum section added, with two open gaps: the
  remote backend against a real provider, and quantum simulation off NVIDIA. bindgen built without `prettyplease` so the workspace builds.
- **2026-11-12.** Page written. Hardware layer, backend reach, OpenVINO
  hosted backend, vendor-neutral host adapter and confinement, six ONNX
  providers, five ggml features, CPU kernel tiers, the silicon catalogue, the
  bandwidth measurement and the `silicon-coverage` workflow added. Nine open
  gaps recorded.
