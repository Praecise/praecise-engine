# Praecise Engine

**A backend-agnostic inference-acceleration engine for large language models.**

Praecise Engine sits between an application and an inference backend and speeds up
token generation — through speculative decoding, hardware-aware tuning, and a
uniform serving surface — without the application needing to know which backend is
underneath. It is designed to support multiple backends behind one interface;
**llama.cpp is the first supported backend.**

## Design

```
        application / serving layer
                   │
        ┌──────────▼───────────┐
        │   Praecise (accel)   │   speculative decode, hardware tuning,
        │   uniform interface  │   sampling, KV strategy
        └──────────┬───────────┘
                   │  pluggable backends
     ┌─────────────┼───────────────────────────┐
     ▼             ▼                             ▼
  llama.cpp     (planned) vLLM /            (planned) MLX /
  supported     TensorRT-LLM / SGLang       other runtimes
```

Praecise is the acceleration layer — not a model server on its own, and not a fork
of any single engine. Each backend is an adapter. Optimizations that can be
expressed generically live in the Praecise layer; backend-specific work lives in
that backend's adapter. Praecise does not vendor a backend: it **pulls** the one
it needs (or uses the one a host already provides).

## Backends

### llama.cpp

- **Speculative decoding** — self-speculative block (DFlash) and multi-token
  prediction (MTP) paths, wired through the FFI with draft/target context sharing.
- **Architecture coverage** — including the muse-glimmer family and its projector.
- **Hardware tuning** — per-architecture matmul configuration and attention paths.
  The CUDA build is a single multi-architecture binary spanning Turing → Blackwell;
  architecture-specific fast paths are gated and selected at runtime. See
  [docs/hardware.md](docs/hardware.md) for the support matrix.

### Planned

vLLM, TensorRT-LLM, SGLang, MLX and others — added as adapters behind the same
interface.

## Why speculative decoding

Single-stream (batch-1) decode is memory-bandwidth-bound: each token streams the
model's active weights from memory once, so throughput is bounded by
`bandwidth ÷ bytes-per-token`. The software levers are to move fewer bytes per
token (quantization, sparsity) or to emit more tokens per weight-read (speculative
decoding). Praecise focuses on the latter and on tuning each backend for the target
hardware — techniques that are largely backend-independent, which is why they belong
in a layer rather than a fork.

## Crates

- `llama-cpp-sys-2` — low-level FFI; pulls the llama.cpp backend.
- `llama-cpp-2` — safe Rust bindings, including the speculative-decode primitives.
- `praecise-runtime` — the backend-agnostic acceleration runtime and inference API.

## Consuming Praecise

Depend on the crate you need (today, `llama-cpp-2` / `praecise-runtime`). The
backend is pulled as a dependency; a host that already builds the same backend
shares it rather than building it twice.

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
