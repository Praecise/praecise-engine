# Hardware support

Praecise Engine runs on any vendor's silicon, not one. This page is the
record of what is supported on each, by which route, and how far each claim
has been proved. The plan behind it is [silicon-charter.md](silicon-charter.md);
the reasoning about memory, packaging and interconnect that shapes it is
[foundations.md](foundations.md).

## How far a claim goes

Every row carries one of four words. They are not interchangeable.

| Word | Meaning |
|---|---|
| **verified** | Inference has run on that silicon, including speculative decode. |
| **built** | Compiled and linked into a binary. Nothing has run on the device. |
| **compiled** | Type-checked against the vendor's API. It is not linked against the vendor's runtime. |
| **wired** | The feature exists and sets the upstream switch. It has not been built here, usually because it needs the vendor's SDK. |

Only NVIDIA rows are verified today. Every other row is a route that exists
and is tested as far as it can be without the hardware; it becomes verified
when someone runs it on the device and records the run here.

The `silicon-coverage` workflow builds the routes that need no vendor SDK
(ZenDNN, KleidiAI, Metal, the AVX-512 options) and compiles every ONNX
provider on each change. A row moves from `wired` to `built` when that
workflow has built it; until its first run the rows it covers stay `wired`
([silicon-charter.md](silicon-charter.md), open item I).

## What a machine has

`praecise-hardware` reads the accelerators of a machine from the operating
system: PCI vendor and class codes, the kernel driver bound to each DRM and
compute-accelerator device, and the device nodes vendor drivers create. It
calls no vendor library, so it works where a vendor's tools are absent.

```
praecise-hw                 # accelerators, routes, build features, CPU extensions, device nodes
praecise-hw catalogue       # decode ceiling of each system in the catalogue
praecise-hw measure         # memory bandwidth this machine's CPU threads reach
```

`PRAECISE_ACCELERATORS` declares the machine instead of detecting it
(`amd:gpu,amd:npu`, `google:accelerator`, `none`). It is the only way to
name an accelerator the system shows no stable sign of, a cloud TPU for
instance.

Detection is tested against described machines on every build host: a
multi-GPU server with a baseboard VGA device, Ryzen AI and Core Ultra
laptops with their NPUs, Gaudi, Ascend and Cambricon cards, a container with
only device nodes, Snapdragon and Mali system-on-chips, IBM Z. On Windows
the display-adapter and compute-accelerator device classes are read from the
registry; the display class is checked on a real machine, the
compute-accelerator class on none.

## Three routes onto an accelerator

1. **Linked**: a ggml backend of the bundled llama.cpp. The engine drives
   the decode loop, so speculation and structured output are available.
2. **ONNX**: an ONNX Runtime execution provider, for the ONNX model
   families (speech, detection, segmentation, embedding, forecasting).
3. **Hosted**: vLLM, SGLang, transformers, TensorRT-LLM or OpenVINO GenAI in
   a confined runtime host. The runtime owns its decode loop.

`praecise_runtime::Backend::reaches` answers, per accelerator, whether a
backend has a route onto it, and `ensure_reaches` refuses a backend with
none, naming the ones that have.

## Coverage by vendor

Feature names are the same on `praecise-runtime`, `praecise-diffusion`,
`llama-cpp-2` and `llama-cpp-sys-2`. ONNX feature names are those of
`praecise-onnx`.

| Vendor | Silicon | Linked (ggml feature) | ONNX provider (feature) | Hosted |
|---|---|---|---|---|
| NVIDIA | GeForce, RTX, data-centre GPUs, GB10, Jetson | `cuda` verified; `vulkan` wired | TensorRT, CUDA (`cuda`) compiled | vLLM, SGLang, TensorRT-LLM, transformers |
| AMD | Radeon, Instinct, Ryzen integrated GPUs | `rocm` wired; `vulkan` wired | MIGraphX (`rocm`) compiled | vLLM (ROCm), SGLang, transformers |
| AMD | XDNA NPU | none | Vitis AI (`vitis`) compiled | none |
| AMD | Zen CPUs | `zendnn` wired | CPU provider | every hosted engine on its CPU path |
| Intel | Arc, integrated Xe, data-centre GPUs | `sycl` wired; `openvino` wired; `vulkan` wired | OpenVINO (`openvino`) compiled | vLLM (XPU), SGLang, transformers, OpenVINO GenAI |
| Intel | Core Ultra NPU | `openvino` wired | OpenVINO on the NPU (`openvino`) compiled | OpenVINO GenAI |
| Intel | Gaudi | none | none | vLLM (Gaudi plugin), SGLang, transformers |
| Intel | Xeon | AVX-512 and AMX options wired | CPU provider | every hosted engine on its CPU path |
| Apple | M-series GPU | `metal` wired | CoreML (`coreml`) compiled | none: runtime hosts need Linux |
| Apple | Neural Engine | none | CoreML | none |
| Qualcomm | Adreno GPU | `opencl` wired; `vulkan` wired | QNN on the GPU (`qnn`) compiled | none |
| Qualcomm | Hexagon NPU | `hexagon` wired | QNN on HTP (`qnn`) compiled | none |
| Arm | Mali, Immortalis | `vulkan` wired | WebGPU (`webgpu`) compiled | none |
| Arm | AArch64 CPUs | `kleidiai` wired; NEON, dot-product, int8-matmul, SVE, SME selected at run time | CPU provider | every hosted engine on its CPU path |
| Huawei | Ascend NPU | `cann` wired | CANN (`cann`) compiled | vLLM (vllm-ascend), SGLang, transformers (torch_npu) |
| Google | TPU | none | none | vLLM (tpu-inference) |
| Amazon | Trainium, Inferentia | none | none | vLLM (vllm-neuron) |
| Moore Threads | MTT GPUs | `musa` wired; `vulkan` wired | none | vLLM (vllm-musa), SGLang, transformers (torch_musa) |
| Cambricon | MLU | none | none | vLLM (vllm-mlu) |
| MetaX, Iluvatar, Enflame | GPUs and GCUs | `vulkan` wired where the driver has it | none | vLLM (vendor plugin) |
| Tenstorrent, FuriosaAI | Blackhole, RNGD | none | none | the vendor's vLLM-compatible server |
| IBM | Z on-chip accelerator | `zdnn` wired | none | vLLM (vllm-spyre, for Spyre) |
| Esperanto | ET-SoC | `et` wired | none | none |
| RISC-V | RVV, SpacemiT | CPU backend; RVV options default on | CPU provider | none |
| LoongArch | LSX, LASX | CPU backend; LSX and LASX default on | CPU provider | none |
| any GPU | DirectX 12 or WebGPU | `webgpu` wired | DirectML (`directml`), WebGPU (`webgpu`) compiled | none |

Notes that qualify the table:

- **ROCm.** ONNX Runtime removed its ROCm execution provider in 1.23. The
  `rocm` feature of `praecise-onnx` builds MIGraphX, the provider that runs on
  ROCm. On Ryzen integrated GPUs Vulkan is often level with or ahead of HIP
  on decode; benchmark both before choosing.
- **Neural processors.** An NPU takes quantized, static-shape models, so an
  ONNX session goes to one only when the operator asks for it
  (`OnnxConfig.npu`). It is registered ahead of the GPU's provider.
- **Portable providers.** DirectML (Windows) and WebGPU serve a GPU whose own
  feature is not built. A build with neither, and without the vendor's
  feature, refuses ONNX models on a machine with a GPU rather than running
  them on the CPU.
- **Vendor SDK backends.** `sycl`, `openvino`, `cann`, `musa`, `zdnn`,
  `webgpu`, `hexagon`, `et` and `opencl` link a vendor runtime this build
  script does not name. A static build leaves that link to the final crate, as
  `opencl` always has; a `dynamic-link` build carries it in the shared
  libraries. `cuda`, `rocm`, `mkl` and `zendnn` are linked by the build
  script; KleidiAI's archive is installed beside the ggml ones and linked
  with them.
- **Hosted engines need Linux.** Confinement is namespaces, Landlock and
  seccomp. On macOS and Windows only the linked and ONNX routes exist.
- **API-only silicon.** Cerebras, Groq, SambaNova and the hyperscalers'
  internal chips are reached over a network API. A runtime host has no
  network by design, so they are outside this engine.

## NVIDIA CUDA (llama.cpp backend)

The CUDA build is a **multi-architecture** binary: one build runs across the
NVIDIA GPU generations below. Architecture-specific fast paths are compiled in
and selected at runtime, so a single artifact serves a mixed fleet.

| Arch (SM) | Generation | Support | Notes |
|---|---|---|---|
| 75 | Turing (RTX 20xx, T4) | supported | int8 tensor cores |
| 80 | Ampere (A100) | supported | async loads, faster TC |
| 86 | Ampere (RTX 30xx) | supported | |
| 89 | Ada (RTX 40xx, L4/L40) | supported | |
| 90 | Hopper (H100/H200) | supported | |
| 120a | Blackwell (RTX 50xx, B-series) | supported | FP4 tensor cores (CUDA ≥ 12.8) |
| 121a | Blackwell (Grace-Blackwell) | supported + **verified** | arch-gated MMQ/MMA tuning; speculative decode validated here |

- **Toolkit:** CUDA ≥ 12.8 for the Blackwell (120/121) paths; earlier toolkits
  build the ≤ 90 set. Older virtual archs (50/61/70) are included on CUDA < 13.
- **Tuning specifics** (per-arch MMQ config, MMA/flash-attention paths) are guarded
  by capability macros and only activate on the architecture they target. Adding a
  specialization never regresses the others.
- **Verified** means we have run inference (including speculative decode) on that
  architecture. Other rows are supported by the backend's standard architecture
  coverage; we mark them verified as we test them.

## CPUs

The CPU is an inference device on every machine and the only one on many.

| Architecture | Selected at build | Selected at run time |
|---|---|---|
| x86-64 | AVX, AVX2, FMA, F16C, AVX-VNNI, AVX-512 (F, CD, VL, DQ, BW together), AVX-512 VBMI/VNNI/BF16, AMX tile/int8/bf16, from the Rust target features | MoE expert kernels: AVX-512-VNNI, then AVX-VNNI, then scalar |
| AArch64 | `GGML_CPU_ARM_ARCH`; KleidiAI (`kleidiai`) | KleidiAI picks SVE, SVE2, SME, SME2 kernels; MoE expert kernels: NEON dot-product, then NEON |
| RISC-V | RVV, Zfh, Zvfh, Zicbop, Zihintpause on by default | |
| LoongArch | LSX, LASX on by default | |
| s390x | VXE with `GGML_NATIVE`; `zdnn` | |

`praecise-hw` prints the extensions of the CPU it runs on and the ggml
options they enable.

## Confined engines reach every vendor's device

A runtime host is given exactly the device nodes that exist: NVIDIA's, AMD's
`kfd`, the kernel's DRM and compute-accelerator classes (`/dev/dri`,
`/dev/accel`: Intel and AMD NPUs, Gaudi, Qualcomm Cloud AI), Ascend's
`davinci*`, Neuron's, Tenstorrent's, Adreno's and Hexagon's, Mali's and
Cambricon's. Vendor runtime directories outside the system library paths
(`/opt/rocm`, `/opt/intel`, `/opt/habanalabs`, `/opt/aws/neuron`, …) are
readable, and the device-selection variables the operator set
(`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ZE_AFFINITY_MASK`,
`ASCEND_RT_VISIBLE_DEVICES`, …) are carried into the host.

Inside the host, the adapter takes whichever accelerator the environment's
packages drive: torch's `cuda` (NVIDIA and ROCm), `xpu`, `mps`, `hpu`, `npu`,
`musa`, or the platform vLLM reports, which includes every vendor plugin. A
machine with an accelerator the engine cannot use refuses the load. Each
vendor's vLLM plugin pins the upstream release it was built against, so each
needs an engine environment of its own.

## Adding an architecture or backend

1. Confirm it builds and runs a correct forward pass on real hardware.
2. Gate any specialization behind a capability check so other targets are unaffected.
3. Add its vendor, detection sign and route to `praecise-hardware`, with a
   test against a described machine.
4. Record it here, marking `verified` only after a real run.
