# Hardware support

Praecise Engine inherits its device coverage from the active backend and adds
architecture-gated tuning on top. This page tracks what is supported and what has
been verified.

## NVIDIA CUDA (llama.cpp backend)

The CUDA build is a **multi-architecture** binary — one build runs across the
NVIDIA GPU generations below. Architecture-specific fast paths are compiled in and
selected at runtime, so a single artifact serves a mixed fleet.

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
  by capability macros and only activate on the architecture they target — adding a
  specialization never regresses the others.
- **Verified** means we have run inference (including speculative decode) on that
  architecture. Other rows are supported by the backend's standard architecture
  coverage; we mark them verified as we test them.

## Other backends / accelerators

Planned as backend adapters are added: ROCm (AMD), Metal (Apple), Vulkan, SYCL
(Intel). Each will get its own row set here once building and running are confirmed.

## Adding an architecture or backend

1. Confirm it builds and runs a correct forward pass on real hardware.
2. Gate any specialization behind a capability check so other targets are unaffected.
3. Record it here, marking `verified` only after a real run.
