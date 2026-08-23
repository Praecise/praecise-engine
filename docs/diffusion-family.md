# Adding a diffusion family to Praecise

**Status:** design, not implemented. Written after an attempt to serve
LTX-2.5 (22B video DiT) alongside Praecise-accelerated LLMs on a GB10.

## The problem

Praecise describes itself as *"backend-agnostic acceleration on top of an
inference backend (llama.cpp first)"*, with the stated plan to generalise to
vLLM / TensorRT-LLM / MLX. That is agnosticism across **backends**. Every
backend named is still an **autoregressive token generator**.

A diffusion model is not a different backend for the same task. It is a
different task. The current public surface says so plainly:

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

Of those, exactly one field — `seed` — means anything to a denoiser. There is
no token to sample, no distribution to shape, no stop string, no KV cache to
reuse, and no draft model to verify against. `StopReason` has no referent: a
diffusion run ends because the schedule ended.

So the question is not "can Praecise load a DiT". It is whether Praecise wants
a **second task family**, which is a larger commitment than a second backend
and should be made deliberately.

## What is actually shared

The case for one engine rather than two projects rests on these, and they are
real:

1. **Residency and admission.** The hard problem in practice is not decode
   speed, it is which weights are allowed to occupy memory. On a 121.7 GiB
   unified-memory GB10 we measured a 22B video DiT needing 66.5 GiB of serving
   set against an LLM already holding 41.6 GiB, under an 88 GiB ceiling. That
   arithmetic is identical in kind for both families, and today neither
   Praecise nor its consumer owns it — it leaks into node config.

2. **The hardware-tuning harness.** Not the tunings — see below — but the
   machinery for measuring and selecting them per device.

3. **Load-time numeric casts.** `fp8-cast`-style downcasting is the same
   plumbing whether the weights feed a decode loop or a denoiser.

4. **Determinism.** Seeded, reproducible runs matter more for media than for
   text, and the discipline is shared.

5. **Verifiable inference** (`toploc`), if it stays — see open questions.

## What is not shared

Sampling, speculative decode, KV cache, sequence batching, stop-string
streaming. None of it transfers. `batching.rs` (896 lines) batches *sequences*;
diffusion batches *latents* across a fixed step count, which is a different
scheduling problem with a different cost model.

## The finding that should shape the design

**Acceleration conclusions invert between the families.** This is not a detail;
it is the reason a shared "accelerator" abstraction would be actively harmful
if it assumed the LLM answer generalises.

On the same GB10 class of machine:

| technique | autoregressive decode | video diffusion |
|---|---|---|
| weight-only NVFP4 | wins — decode is bandwidth-bound | **loses** — ~33% smaller, ~6% *slower* |
| speculative decode (DFlash/MTP) | 1.85–2.13x measured | **not applicable** — nothing to draft |
| step/timestep caching (TeaCache) | not applicable | wins |
| attention sparsification (Sliding Tile) | limited | wins — ~3x reported |

LLM decode is memory-bandwidth-bound, so shrinking weights buys time. Video
diffusion is compute-bound, so shrinking weights buys only an unpack penalty.
NVIDIA's published "20% faster, 40% memory savings" for LTX-2.5 is NVFP4
*plus* FastVideo kernels, STA and TeaCache — the kernels are the speedup, not
the format.

**Design consequence:** the shared layer must be the *harness* (measure, select,
record), never a table of techniques presumed portable. A `praecise-core` that
exports "use NVFP4 on Blackwell" would be wrong half the time.

## Proposed shape

```
praecise-core        error, seed/determinism, device + memory budget,
                     tuning harness, telemetry, (toploc?)
praecise-runtime     autoregressive LM family — today's crate, API unchanged
praecise-diffusion   iterative-denoiser family — new
```

`praecise-runtime` keeps its current public API verbatim; this is additive.
The families share `praecise-core` and nothing else. Each owns its own config
and result types rather than contorting one pair to cover both:

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

Note what is absent: no `StopReason`, no token counts, no sampler. Trying to
unify these with `GenerationConfig` behind one trait produces a struct where
most fields are `None` for any given caller, which is how a shared abstraction
becomes worse than two honest ones.

## The scoping question, stated honestly

**A working diffusion path already exists outside Praecise.** Tenzro serves
LTX through a Python/diffusers worker with a `FamilyAdapter` seam
(`Ltx2Adapter`, `Cosmos3Adapter`), a job marketplace, and per-family
conversion fixes already carried. Reimplementing that in Rust is a second
implementation of a solved problem.

So a Praecise diffusion family should only be built if it earns one of these:

- **In-process serving** — no Python worker, no IPC, for a node that already
  links Praecise for LLMs.
- **One memory budget** across both families, so a DiT and an LLM contend
  through the same admission control instead of two schedulers that cannot see
  each other. This is the strongest argument, and it is the exact failure we
  hit: 66.5 + 41.6 GiB against an 88 GiB ceiling, with neither side aware of
  the other.
- **Verifiable media inference**, if commitments extend to latents.

If none of those is wanted, the honest recommendation is: **do not build it**,
and keep diffusion in the Python worker where it already works.

## Staging

1. Extract `praecise-core` from `praecise-runtime` with no behaviour change —
   error, seed, device/memory budget, tuning harness. `praecise-runtime`
   depends on it and keeps its API.
2. Land the memory/admission model in `praecise-core` and have the LLM family
   use it. This is worth doing on its own merits, family or no family.
3. Only then, `praecise-diffusion` behind a feature flag, with one model
   (LTX-2.5 dev bf16) and no kernel work — correctness first, `fp8-cast` and
   CPU offload as the only levers.
4. Kernels (NVFP4-aware linears, STA, TeaCache) last, and only against measured
   baselines, given the inversion above.

Steps 1 and 2 have standalone value. Step 3 is the reversible commitment point.

## Open questions

- **`toploc` placement.** `InferenceResult.commitment` and `toploc.rs` (374
  lines) are the largest non-batching module, and the extraction notes say
  verifiable-inference is a tenzro-ism to strip out of Praecise. It is still
  here. Whether it moves to `praecise-core`, stays in the LM family, or leaves
  entirely should be settled before a second family inherits the ambiguity.
- **Does `batching` generalise?** Probably not, but latent batching across a
  fixed step count may reuse the slot accounting.
- **ARM.** FastVideo ships no prebuilt aarch64 wheel; any kernel work has a
  from-source build on GB10.
