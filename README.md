# Praecise Inference

A high-throughput LLM inference engine: a hardened fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) plus Rust bindings, carrying a
set of speculative-decoding and hardware optimizations that target fast,
single-stream token generation on modern accelerators.

Praecise is developed as a standalone, open-source engine. It is consumed by
larger systems through a thin, stable adapter so the engine can evolve — or be
renamed — without churning its consumers.

## What's inside

- **`praecise-llama/`** — the low-level engine: a vendored, patched llama.cpp
  fork and its Rust FFI (`llama-cpp-sys-2`) + safe bindings (`llama-cpp-2`).

### Optimizations carried in this fork

- **DFlash block-diffusion speculative decoding** — self-speculative decode wired
  end to end through the FFI (`spec_type` → `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`),
  including the `ctx_other` draft/target context linkage and the draft-context
  restore required for correct block verification. Measured ~2.5–2.7× single-stream
  decode speedup on a 30B dense model at NVIDIA GB10, across temperatures.
- **MTP (multi-token-prediction) speculative path** — the same machinery for
  MTP-head models.
- **Muse-Glimmer architecture support** — the Meta Muse-Glimmer arch, its ATEM
  chat format, DFlash drafter, and multimodal projector.
- **NVIDIA Blackwell / GB10 tuning** — per-arch MMQ config, MMA/flash-attention
  paths, and a dedicated DGX-Spark compute-capability constant.

## Why speculative decoding

Single-stream (batch=1) decode is memory-bandwidth-bound: every token streams the
model's entire active weight set from memory once, so `tok/s ≈ bandwidth ÷
bytes-per-token`. Short of putting weights in on-chip SRAM (the wafer-scale
approach), the only software levers are to move fewer bytes per token
(quantization, MoE) or to emit more tokens per weight-read — which is exactly what
speculative decoding does. Praecise focuses on the latter.

## License

MIT. The bundled llama.cpp fork is MIT (© the ggml authors); Praecise's additions
are MIT as well. See [LICENSE](LICENSE).
