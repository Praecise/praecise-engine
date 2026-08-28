# Praecise Engine

**A backend-agnostic inference-acceleration layer for large language models.**

Praecise Engine sits between an application and an inference backend and speeds
up token generation — through speculative decoding, hardware-aware tuning, and a
uniform serving surface — without the application needing to know which backend
is underneath. **llama.cpp is the first supported backend.**

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
  llama.cpp                vLLM · SGLang
  linked (FFI)             TensorRT-LLM · MLX
                           served (HTTP)
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
| vLLM | served (HTTP) | no — the runtime owns its decode loop | `structured_outputs` |
| SGLang | served (HTTP) | no | `response_format` |
| TensorRT-LLM | served (HTTP) | no | `response_format` |
| MLX | served (HTTP) | no | **none** |

A *linked* backend is compiled in, so the engine can propose a draft block,
verify it, and read per-position logits. A *served* backend is a separate
process behind an OpenAI-compatible API: the engine can choose a model, sampling
parameters and a schema, but there is no seam at which to interpose on
token-by-token decoding. Speculation against a served backend is therefore not
merely unimplemented — it is impossible by construction, and those runtimes do
their own instead.

An unknown backend name is an error listing the valid ones, never a quiet
fallback: a typo that silently ran on a different runtime would invalidate any
measurement taken against it, with nothing in the output to show it.

`praecise_runtime::served` builds and parses the HTTP requests; the caller
supplies the client. That keeps the crate free of a networking dependency and
lets a host reuse the connection pool, timeouts and tracing it already has.

The four served runtimes are **not interchangeable**, and the differences are
silent rather than loud — an unknown JSON key is dropped, not rejected:

- vLLM takes `top_k`/`min_p`/`repetition_penalty` **flat**; SGLang wants them
  nested under `extra_body`. Sent the wrong way, generation proceeds with
  different sampling than was asked for.
- vLLM's `structured_outputs` accepts **exactly one** of json/regex/choice/
  grammar — two is a hard validation error, not a preference. `guided_json` was
  removed in v0.12.0.
- **MLX never reads `response_format` at all**, so a schema-constrained request
  returns free text with no error. `Backend::MlxLm.supports().structured_output`
  is `false` so a caller can find out before sending rather than after.
- MLX reads `max_completion_tokens` before `max_tokens`; ports differ
  (8000/30000/8000/8080); SGLang's `/metrics` needs `--enable-metrics`, and MLX
  has no metrics endpoint at all.

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

- `llama-cpp-sys-2` — low-level FFI; pulls the llama.cpp backend.
- `llama-cpp-2` — safe Rust bindings, including the speculative-decode primitives.
- `praecise-runtime` — the backend-agnostic acceleration runtime and inference API.

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

Runs fmt, clippy and the test suite on the backend-agnostic runtime, plus a
compile check of the backend-gated code. It validates *correctness* on x86 — not
an aarch64 artifact, and nothing about CUDA kernels or GB10-specific tuning,
which need real hardware.

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
