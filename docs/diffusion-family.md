# Adding a diffusion family to Praecise Engine

**Status:** design. Written after serving LTX-2.5 (22B video DiT) alongside
Praecise Engine-accelerated LLMs on a GB10.

> **Naming.** *Praecise* is the family. **Praecise Engine** is the acceleration
> layer — it makes a model faster. **praecise-harness** is the agentic
> framework — it makes an agent work. They sit at different levels and are used
> independently. This document is about Praecise Engine.

## The problem

Praecise Engine is backend-agnostic acceleration on top of an inference backend,
llama.cpp first, with vLLM / TensorRT-LLM / MLX planned. That is agnosticism
across **backends**. Every backend on that roadmap is an **autoregressive token
generator**.

A diffusion model is not a different backend for the same task. It is a
different task, and the public surface says so:

```rust
pub struct GenerationConfig {
    pub temperature: f64,      pub top_p: f64,
    pub max_tokens: u32,       pub repeat_penalty: f32,
    pub repeat_last_n: usize,  pub seed: u64,
    pub top_k: Option<u32>,    pub min_p: Option<f64>,
    pub frequency_penalty: f32, pub presence_penalty: f32,
    pub stop: Vec<String>,
    pub draft_n: Option<u8>,   pub commitment_k: Option<u8>,
}

pub struct InferenceResult {
    pub text: String,          pub thinking: Option<String>,
    pub input_tokens: u32,     pub output_tokens: u32,
    pub tokens_per_second: f64, pub stop_reason: StopReason,
    pub commitment: Option<InferenceCommitment>,
}
```

One field — `seed` — means anything to a denoiser. There is no token to sample,
no distribution to shape, no stop string, no KV cache, no draft model to verify
against. `StopReason` has no referent: a diffusion run ends because the schedule
ended.

The question is whether Praecise Engine takes a **second task family**. That is
a larger commitment than a second backend.

## What is shared

1. **Residency and admission.** The binding constraint is not decode speed, it
   is which weights may occupy memory. On a 121.7 GiB unified-memory GB10 a 22B
   video DiT needs a 66.5 GiB serving set against an LLM holding 41.6 GiB under
   an 88 GiB ceiling. That arithmetic is identical in kind for both families,
   and today neither Praecise Engine nor its consumer owns it — it leaks into
   node config.
2. **The hardware-tuning harness.** The machinery for measuring and selecting
   per device. Not the tunings themselves.
3. **Load-time numeric casts.** `fp8-cast`-style downcasting is the same
   plumbing for a decode loop or a denoiser.
4. **Determinism.** Seeded, reproducible runs matter more for media than text.
5. **Verifiable inference** (`toploc`), if it stays — see open questions.

## What is not shared

Sampling, speculative decode, KV cache, sequence batching, stop-string
streaming. `batching.rs` batches *sequences*; diffusion batches *latents* across
a fixed step count — a different scheduling problem with a different cost model.

## Acceleration conclusions invert between the families

This is the finding that shapes the design.

| technique | autoregressive decode | video diffusion |
|---|---|---|
| weight-only NVFP4 | wins — bandwidth-bound | loses — ~33% smaller, ~6% *slower* |
| speculative decode (DFlash/MTP) | 1.85–2.13x measured | n/a — nothing to draft |
| step caching (TeaCache) | n/a | wins |
| attention sparsification (STA) | limited | wins — ~3x reported |

LLM decode is memory-bandwidth-bound, so shrinking weights buys time. Video
diffusion is compute-bound, so shrinking weights buys an unpack penalty.
NVIDIA's published "20% faster, 40% memory savings" for LTX-2.5 is NVFP4 *plus*
FastVideo kernels, STA and TeaCache. The kernels are the speedup, not the format.

**The shared layer is therefore the harness — measure, select, record — never a
table of techniques presumed portable.** A `praecise-core` exporting "use NVFP4
on Blackwell" is wrong half the time.

## Shape

```
praecise-core        error, seed/determinism, device + memory budget,
                     tuning harness, telemetry, (toploc?)
praecise-runtime     autoregressive LM family — today's crate, API unchanged
praecise-diffusion   iterative-denoiser family — new
```

Additive. `praecise-runtime` keeps its public API verbatim. The families share
`praecise-core` and nothing else, each owning its config and result types:

```rust
pub struct DenoiseConfig {
    pub steps: u32,
    pub guidance_scale: f32,
    pub schedule: SigmaSchedule,   // distilled checkpoints ship fixed ones
    pub seed: u64,
    pub cast: Option<NumericCast>, // fp8-cast et al, load-time
}

pub struct DenoiseResult {
    pub steps_run: u32,
    pub time_ms: u64,
    pub steps_per_second: f64,
}
```

No `StopReason`, no token counts, no sampler. Unifying these with
`GenerationConfig` behind one trait produces a struct where most fields are
`None` for any given caller.

## What this has to earn

A diffusion path already exists outside Praecise Engine: a Python/diffusers
worker with a `FamilyAdapter` seam, a job marketplace, and per-family conversion
fixes. A second implementation is justified by one of:

- **In-process serving** — no Python worker, no IPC, for a node already linking
  Praecise Engine.
- **One memory budget across families**, so a DiT and an LLM contend through the
  same admission control rather than two schedulers that cannot see each other.
  This is the strongest case, and the exact failure observed: 66.5 + 41.6 GiB
  against an 88 GiB ceiling, neither side aware of the other.
- **Verifiable media inference**, if commitments extend to latents.

## Staging

1. Extract `praecise-core` from `praecise-runtime` with no behaviour change —
   error, seed, device/memory budget, tuning harness. `praecise-runtime` depends
   on it and keeps its API.
2. Land the memory/admission model in `praecise-core`; the LM family uses it.
   This carries its own value independent of a second family.
3. `praecise-diffusion` behind a feature flag, one model (LTX-2.5 dev bf16), no
   kernel work — correctness first, `fp8-cast` and CPU offload as the levers.
4. Kernels (NVFP4-aware linears, STA, TeaCache) last, against measured
   baselines, given the inversion above.

Step 3 is the commitment point, and it is reversible.

## Open questions

- **`toploc` placement.** `InferenceResult.commitment` and `toploc.rs` are the
  largest non-batching module, and verifiable-inference is slated to leave
  Praecise Engine as a consumer concern. Settle this before a second family
  inherits the ambiguity.
- **Does `batching` generalise?** Latent batching across a fixed step count may
  reuse the slot accounting.
- **ARM.** FastVideo ships no prebuilt aarch64 wheel; kernel work on GB10 builds
  from source.
