# Praecise Engine

**A backend-agnostic inference-acceleration layer for large language models.**

Praecise Engine sits between an application and an inference backend and speeds
up token generation — through speculative decoding, hardware-aware tuning, and a
uniform serving surface — without the application needing to know which backend
is underneath. **llama.cpp is the first supported backend.**

It is as agnostic about silicon as it is about backends. NVIDIA, AMD, Intel,
Apple, Qualcomm, Arm, Huawei and the rest are each reached by a route that is
named, built and tested as far as it can be without the device, and none is
the default. [docs/hardware.md](docs/hardware.md) says what is supported on
each and how far each claim has been proved;
[docs/silicon-coverage.md](docs/silicon-coverage.md) is the plan behind it.

## Design

```
        application / serving layer
                   │
        ┌──────────▼───────────┐
        │ Praecise Engine      │   speculation policy, drafter selection,
        │   uniform interface  │   request shaping, sampling, KV strategy
        └──────────┬───────────┘
                   │  pluggable backends
     ┌─────────────┴───────────────┐
     ▼                             ▼
  llama.cpp                vLLM · SGLang · transformers        an inference API
  linked (FFI)             TensorRT-LLM · OpenVINO GenAI       remote (opt-in)
                           hosted (confined runtime host)
     │                             │
     └──────────────┬──────────────┘
                    ▼
   praecise-hardware: which accelerators the machine has,
   which backend drives each, what a confined engine needs
   to reach them, and the decode ceiling their memory sets
```

Praecise Engine is the acceleration layer — not a model server on its own, and
not a fork of any single engine. Each backend is an adapter. Optimizations that
can be expressed generically live in the engine layer; backend-specific work
lives in that backend's adapter.

## What this layer is for, measured

Decode is memory-bandwidth-bound. On a DGX Spark (GB10) it already runs at
**~80% of the 273 GB/s peak** — the highest fraction of peak of any comparable
machine (M4 Max 58%, M3 Ultra 44%). See
[docs/gb10-bandwidth-ceiling.md](docs/gb10-bandwidth-ceiling.md).

That constrains what a layer above a backend can honestly claim: everything
host-side shares the ~20% that is not already saturated. So this engine
concentrates on decisions the backend cannot make for itself, and on avoiding
losses rather than manufacturing wins.

Measured on an RTX 5070 Ti with `qwen3.5-0.8b` and n-gram self-speculation:

| prompt | tok/s | drafted | vs baseline |
|---|---:|---:|---:|
| SQL | 226.3 | 171 | **3.5x** |
| code | 94.5 | 73 | **1.5x** |
| number list | 63.8 | 0 | — |
| JSON | 65.4 | 0 | — |
| prose | 62–65 | 0 | — |

Speculation is worth 1.5–3.5x on structured text and **nothing** on prose. The
decision of *whether* to speculate is where the value is; see
[docs/measured-acceptance.md](docs/measured-acceptance.md).

## Selecting a backend

`praecise_runtime::Backend` names the runtimes the engine knows about and
reports, per backend, what is actually possible:

```rust
use praecise_runtime::Backend;

let backend = Backend::parse("vllm")?;   // unknown names error, never default
backend.ensure_available()?;             // refuses rather than silently substituting
let caps = backend.supports();
```

The distinction that matters is **how** a backend is reached, because it decides
what acceleration is available at all:

| Backend | Integration | Engine-side speculation | Structured output |
|---|---|---|---|
| llama.cpp | linked (FFI) | yes — the engine drives the decode loop | yes |
| vLLM | hosted | no — the runtime owns its decode loop | no |
| SGLang | hosted | no | no |
| TensorRT-LLM | hosted | no | no |
| transformers | hosted (CPU or GPU) | no | no |
| OpenVINO GenAI | hosted (Intel CPU, GPU or NPU) | no | no |
| an inference API | remote (opt-in) | no — the provider owns the decode loop | no |

A *linked* backend is compiled in, so the engine can propose a draft block,
verify it, and read per-position logits. A *hosted* backend runs in a runtime
host: the engine chooses the model and the sampling parameters, but there is no
seam at which to interpose on token-by-token decoding. Speculation against a
hosted backend is therefore not merely unimplemented — it is impossible by
construction, and those runtimes do their own instead.

A *remote* backend runs nowhere on this machine: `praecise-remote` asks an
OpenAI-compatible inference API (Groq, Cerebras, SambaNova or any other) and
answers in the same types. It is the one route to silicon sold as a service,
and the one backend whose prompts leave the machine, so it is opt-in twice:
the application has to depend on the crate, and the operator has to name each
host in `PRAECISE_REMOTE_ALLOW`. Runtime hosts still have no network.

```rust
use praecise_remote::{Allow, Provider, Remote};

let groq = Remote::open(Provider::Groq.endpoint(model), &Allow::from_env())?;
let reply = groq.generate(&messages, &config, |text| { print!("{text}"); true })?;
```

An unknown backend name is an error listing the valid ones, never a quiet
fallback: a typo that silently ran on a different runtime would invalidate any
measurement taken against it, with nothing in the output to show it.

## Any vendor's silicon

A backend is not a vendor. llama.cpp reaches every accelerator a ggml backend
is built for; vLLM, SGLang and transformers reach whichever one the packages
of their environment drive; TensorRT-LLM is NVIDIA's and OpenVINO is Intel's.
The engine asks which is which before it assigns a model:

```rust
use praecise_runtime::hardware::Machine;
use praecise_runtime::Backend;

let machine = Machine::from_env_or_detect()?;   // every vendor, read from the OS
Backend::Vllm.ensure_reaches(&machine)?;        // refuses, naming the backends that do
```

| Silicon | Linked (ggml feature) | ONNX provider | Hosted |
|---|---|---|---|
| NVIDIA GPUs | `cuda`, `vulkan` | TensorRT, CUDA | vLLM, SGLang, TensorRT-LLM, transformers |
| AMD Radeon, Instinct, Ryzen | `rocm`, `vulkan`; `zendnn` on Zen CPUs | MIGraphX; Vitis AI on the NPU | vLLM, SGLang, transformers |
| Intel Arc, Xe, Core Ultra NPU, Gaudi | `sycl`, `openvino`, `vulkan` | OpenVINO | vLLM, SGLang, transformers, OpenVINO GenAI |
| Apple M-series | `metal` | CoreML | none (hosts need Linux) |
| Qualcomm Adreno, Hexagon | `opencl`, `hexagon` | QNN | none |
| Arm CPUs, Mali | `kleidiai`, `vulkan` | WebGPU | none |
| Huawei Ascend | `cann` | CANN | vLLM, SGLang, transformers |
| Google TPU, AWS Trainium, Moore Threads, Cambricon and others | `musa` for Moore Threads | none | vLLM through the vendor's plugin |
| Groq, Cerebras, SambaNova | none | none | none; remote, opt-in (`praecise-remote`) |

`praecise-hw` prints the accelerators of the machine it runs on, the build
features that machine needs, and the route to each device. Only the NVIDIA
rows have been run on their silicon; the rest are compiled or wired and wait
for a run on the device. The full table, with what each word means, is
in [docs/hardware.md](docs/hardware.md).

## Running several engines: `praecise-host`

An application can serve several models at once, each on the engine that suits
it: llama.cpp in its own process, and any hosted backend in a **runtime host**.
`praecise_host::Engines` is the one place models are assigned to engines:

- **One engine per model.** Every load claims the model id first; a claim held
  by another engine, or a second host for the same model, is refused with
  `Error::Conflict`. A model is never in memory twice.
- **One memory budget.** Each claim reserves the model's memory from a `Budget`
  shared by every engine: the application's own ledger, or a `FixedBudget`.
- **Confined, not trusted.** A runtime host is a child process the application
  starts by re-running its own executable as the confinement launcher (call
  `praecise_host::sandbox::enter_if_requested()` first thing in `main`). It
  gets new user, network, mount, IPC and PID namespaces, a Landlock ruleset
  (read-only engine environment and model files, one scratch directory, the GPU
  device nodes), a seccomp filter and resource limits. It has no network: no
  interface but an optional loopback with no route out, and no internet socket
  unless loopback is granted. It speaks to the application only over its
  standard streams, one JSON object per line, with request ids so an engine that
  batches sees every outstanding request.
- **Pinned environments.** An `EngineEnv` is a directory holding the engine's
  program, packages and protocol adapter, pinned by the SHA-256 of its whole
  tree (`praecise_host::engine::digest`). The digest is checked before every
  start; an environment that changed is refused, never run.
- **Supervised.** A host that exits fails the requests it was serving and is
  restarted in the background from the same verified environment, while every
  other engine keeps serving. After three failures in a row it is left stopped
  and reported, until the model is released and served again.
- **Accelerator or refusal.** On a machine with an accelerator of any
  vendor, an engine that cannot use it refuses to load rather than falling
  back to the CPU. A host is given the device nodes of every vendor (NVIDIA,
  AMD, the kernel's DRM and compute-accelerator classes, Ascend, Neuron and
  the rest), the vendor runtime directories, and the device-selection
  variables the operator set.

The hosted backends are driven through their offline Python engine APIs (vLLM
`AsyncLLM`, SGLang `Engine`, TensorRT-LLM `LLM`, transformers, OpenVINO GenAI
`LLMPipeline`) by one adapter,
`praecise-host/adapters/praecise_adapter.py`, which `engine::install_adapter`
writes into a provisioned environment before it is pinned. No engine opens an
HTTP server. `praecise_host::generation` maps `GenerationConfig` and
`InferenceResult` onto the protocol, so a hosted engine answers with the same
types, token counts and stop reasons as the linked one; a reply without token
counts is refused rather than metered as free.

A kernel that forbids unprivileged user namespaces (for example
`kernel.apparmor_restrict_unprivileged_userns=1`) refuses the confinement, and
the error names the setting; grant the application's executable `userns` in
its AppArmor profile rather than lowering the setting for the whole machine.
On Ubuntu that is one profile per executable, for example
`/etc/apparmor.d/praecise-job`:

```
abi <abi/4.0>,
include <tunables/global>
profile praecise-job /opt/praecise/bin/praecise-job flags=(unconfined) {
  userns,
}
```

followed by `apparmor_parser -r /etc/apparmor.d/praecise-job`. Where that is
not possible an operator may choose `--confinement landlock-only` explicitly:
the same Landlock ruleset and syscall filter (no internet sockets, no file
outside the rules), without the namespaces. It is never chosen on its own.

### Job engines

Beside the model engines, `praecise_host::job` runs job engines: a fixed
program with typed parameters, input files in, output files and progress out,
run once to completion. A job engine is a pinned environment (the same
SHA-256 tree digest, checked before every run) and one Python adapter,
`praecise-host/adapters/praecise_job.py`, compiled into the binary and placed
read-only beside the run's scratch directory. Every run is confined like a
model host: no network, the environment, model caches and inputs read-only,
one scratch directory writable, the GPU device nodes; it is supervised (time
limit, the whole process tree stopped at the end, the last lines of its error
output on failure); on a machine with a GPU a program refuses rather than run
on the CPU. Parameters are checked against each program's declaration (type
and range) before anything starts, and again by the program.

The programs drive NVIDIA's open simulation systems. They are NVIDIA's own
products and run on CUDA; the ones built on torch also run on AMD ROCm, which
presents as `cuda`:

| Program | System | Out |
|---|---|---|
| `sionna.radio_map` | Sionna RT | radio_map.npz, radio_map.png |
| `warp.example` | Warp | the example's USD stage |
| `newton.example` | Newton | sim.usd |
| `cosmos.generate` | Cosmos 3 (guardrail forced on) | video.mp4 |
| `physicsnemo.darcy_fno` | PhysicsNeMo | fno_darcy.pt and its error on unseen fields |
| `cudaq.ghz`, `cudaq.maxcut` | CUDA-Q | sampled states, the best cut against the optimum |
| `boltz.predict` | Boltz-2 | structures, confidence, affinity |

Each writes result.json. The `praecise-job` binary runs one:

```
praecise-job programs                      # the catalogue, with every parameter's type and range
praecise-job digest /opt/envs/cudaq        # the digest to pin an environment by
praecise-job run --engines engines.json --program cudaq.ghz --param qubits=24 --out out/
```

`engines.json` names each engine's environment and pin:
`{"engines": {"cudaq": {"root": "/opt/envs/cudaq", "digest": "<sha256>",
"read_only": [...], "env": {...}}}}`. Progress is printed as JSON lines and
the last line is the result.

## Perception: `praecise-perception`

Video in, per-frame object tracks out, as an open replacement for a
Metropolis/DeepStream perception pipeline. `praecise-codec` decodes the video
in process, an ONNX detector from `praecise-onnx` (RF-DETR Nano to Large,
Apache-2.0, or D-FINE) runs on every frame on the GPU's execution provider,
and ByteTrack (an independent Rust implementation: Kalman prediction,
two-stage association with low-score detections, minimum-cost assignment)
gives every person and vehicle a stable id, box, score and velocity. With an
image-to-ground homography each track also has a ground position and speed
in metres. Frames can be written as NVIDIA Metropolis `nv.Frame` protobuf
messages, the input Halos Safety Core reads.

```
praecise-track --model rfdetr-medium.onnx --input-size 576 --video in.mp4 --out out/ --nvframe
```

writes `tracks.json`, `summary.json` (frames per second, time per stage,
peak memory) and `frames.nvpb` (one `nv.Frame` per frame, each prefixed by its
length as a varint).


## What the layer provides

### Speculation policy

`praecise_runtime::spec_policy` decides *whether* to speculate given model shape
and load, and returns a `reason` with every plan — a policy that cannot explain
itself gets switched off. It refuses where speculation does not pay: MoE under
concurrency, a loaded engine, a request too short to amortise a draft.

### Self-speculation with no draft model

`praecise_runtime::ngram` drafts from token history alone — no second model, no
GPU memory. It is deliberately **not tied to a request**, so an agent loop or a
code-editing session reuses patterns across turns, which no single-request
runtime is positioned to do.

This is the technique behind the 3.5x above.

### Drafter catalogue with licence enforcement

`praecise_runtime::drafters` records published drafter checkpoints **with their
licences**, which are frequently not the licence of the model they draft for.
`Qwen3.8-27B-DFlash2` is Apache-2.0; `GLM-5.3-Flash-DFlash2` is CC-BY-NC-ND. Both
sit on the same hub and a checkpoint path is just a string.

Only permissive checkpoints are offered. The rest are kept in the catalogue so
that naming one explains *why* it was refused, and a checkpoint whose card states
no licence at all is refused too — absent terms grant nothing.

It also preflights four documented ways a drafter deployment silently
underperforms: a block size the checkpoint was not trained at, a model runner
that quietly downgrades a newer drafter to an older one, grammar conflicts, and
partial-vocabulary sampling. Each produces a smaller speedup rather than an
error, which is why they are worth checking for.

### Request shaping

`praecise_runtime::shaping` makes a batch cheap before it reaches a backend.
The sharpest item: `all_greedy` and `no_penalties` are **batch-wide booleans**, so
one request using a penalty drags the entire batch onto the expensive sampling
path — including every request that asked for none. Penalties cost 6–47 ms/step
at batch 4; the sampler itself costs ~43 µs.

See [docs/request-shaping.md](docs/request-shaping.md).

## Why speculative decoding

Single-stream (batch-1) decode is memory-bandwidth-bound: each token streams the
model's active weights from memory once, so throughput is bounded by
`bandwidth ÷ bytes-per-token`. The software levers are to move fewer bytes per
token (quantization, sparsity) or to emit more tokens per weight-read
(speculative decoding). Praecise Engine focuses on the latter — techniques that
are largely backend-independent, which is why they belong in a layer rather than
a fork.

For what moves the ceiling itself rather than working within it, see
[docs/what-actually-helps.md](docs/what-actually-helps.md). The short version:
KV-cache quantization is worth **+48%** and is off by default.

## What was tried and does not work

`praecise_runtime::prax` sized the draft block per position. **Measured on
hardware, block size does not change throughput** — a 16x range moves it ~5%
non-monotonically, and acceptance is identical to one decimal place at every
setting. The module is retired and inert, kept only so the negative result stops
the idea being rebuilt. See [docs/praecise-method.md](docs/praecise-method.md).

The simulation that predicted otherwise had an oracle bound, seed averaging and
179 passing tests. None of that could detect a mistaken cost model. A simulation
is only as right as its assumptions, and the only test of those is measurement.

## Crates

- `llama-cpp-sys-2` — low-level FFI; pulls the llama.cpp backend (the Praecise
  fork, synced with upstream llama.cpp 0.9.0 of 2026-11-04; built and tested
  on the CPU, not yet on a GPU).
- `llama-cpp-2` — safe Rust bindings, including the speculative-decode primitives.
- `praecise-hardware` — the hardware layer: which accelerators a machine has
  (every vendor, read from the operating system), which backend drives each,
  what a confined engine needs to reach them, the memory and bandwidth of the
  systems worth planning for, and the `praecise-hw` binary. No dependencies.
- `praecise-runtime` — the backend-agnostic acceleration runtime and inference API.
- `praecise-host` — runtime hosts and the engine set: confined, supervised
  engines, one per model, within one memory budget; job engines and the
  `praecise-job` binary.
- `praecise-remote` — generation on an inference API outside the machine
  (Groq, Cerebras, SambaNova or any OpenAI-compatible endpoint), in the
  runtime's types. Opt-in: nothing opens a connection unless the application
  depends on this crate and the operator permits the host.
- `praecise-ffi` — the stable C ABI over the runtime's backend-agnostic types
  (config, result, stop reason, error), for callers in other languages.
- `praecise-diffusion` — flow-matching image, video, audio and 3D generation on
  the same ggml backend, device and memory budget as the language-model family.
- `praecise-provenance` — signed C2PA content credentials embedded in generated
  PNG, WAV and MP4 files, and their verification.
- `praecise-train` — the training substrate: the deterministic replayable step,
  the hash-linked step log, content-hashed checkpoints with lineage, objectives,
  optimizers and parameter-delta primitives.
- `praecise-codec` — in-process video coding: H.264, H.265 and AV1 decoding
  from MP4 and Matroska/WebM, H.264 and AV1 encoding to MP4 (AV1 also to
  WebM) with an Opus soundtrack. Generated videos are written as MP4.
- `praecise-perception` — video in, object tracks out: detection on every
  decoded frame, ByteTrack, `nv.Frame` output; the `praecise-track` binary.
- `praecise-onnx` — the ONNX model families: speech-to-text, object
  detection, promptable and text-prompted segmentation, image encoders,
  time-series forecasting and text embedding. Sessions register the execution
  provider the machine's GPU needs (CUDA and TensorRT, MIGraphX, OpenVINO,
  QNN, CANN, CoreML, or DirectML and WebGPU on any vendor's GPU) and refuse a
  model they cannot accelerate rather than running it on the CPU. A neural
  processor is used when the operator asks for it.
- `praecise-moe` — expert-parallel mixture-of-experts execution: expert FFNs
  and gating from safetensors blobs, GGUF block-quantized expert weights, a
  memory and disk residency cache, and the combine step, on the CPU
  (AVX-512-VNNI, AVX-VNNI, the Arm dot-product extension), CUDA or any other
  GPU through wgpu. The `cuda` feature builds for the CUDA toolkit
  that nvcc reports (12 or 13); `cuda-12` and `cuda-13` pin the version on a
  build host without a toolkit.

## Consuming Praecise Engine

Depend on the crate you need (today, `llama-cpp-2` / `praecise-runtime`). The
backend is pulled as a dependency; a host that already builds the same backend
shares it rather than building it twice.

`praecise-runtime` defaults to **no backend at all** (`default = []`), so the
policy, catalogue and shaping surfaces compile and are testable without pulling
llama.cpp or a GPU.

## Testing

```
gcloud builds submit --config cloudbuild-test.yaml --project <project> .
```

Runs fmt, clippy and the test suite on the hardware layer and the
backend-agnostic runtime, plus a compile check of the backend-gated code. It
validates *correctness* on x86 — not an aarch64 artifact, and nothing about
CUDA kernels or GB10-specific tuning, which need real hardware.

The `silicon-coverage` workflow covers what can be proved about hardware
without the hardware, on x86-64 and AArch64 Linux, macOS and Windows: the
hardware layer and the decisions made from it, every ONNX execution provider
compiled against ONNX Runtime, and the ggml builds that need no vendor SDK
(ZenDNN, KleidiAI, Metal) linked into a binary. It proves that a route
builds, not that a device runs.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
SPDX for the distribution as a whole: `Apache-2.0 AND MIT`.

Apache-2.0 rather than MIT because this engine is meant to sit in front of
several vendors' backends, and Apache-2.0 carries an explicit patent grant where
MIT is silent. That matters more to whoever integrates it than it does to us.

The vendored backend (llama.cpp, © the ggml authors) is MIT and stays MIT; MIT
permits redistribution under other terms provided its notice travels with the
code, so those portions keep their own licence. [NOTICE](NOTICE) records what is
derived from where, and [LICENSE-MIT](LICENSE-MIT) reproduces the MIT terms in
full.
