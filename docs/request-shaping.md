# Request shaping: what a layer above the backend can still control

Decode on GB10 is bandwidth-bound at ~80% of peak (see
[gb10-bandwidth-ceiling.md](gb10-bandwidth-ceiling.md)), so neither speculative
allocation nor host-side scheduling has much room. What remains reachable from
*outside* a backend is how requests are **shaped and grouped** before they reach
it — and a few of those levers are surprisingly sharp.

Every item here is arithmetic about batching, not a theory about text. That is
why they are worth more than the allocation work: nothing has to be true about
the model for them to hold.

## 1. One request can make a whole batch expensive

`all_greedy` and `no_penalties` are **batch-wide booleans** in vLLM. A single
request using frequency/presence/repetition penalties drags the *entire batch*
onto the expensive sampling path, including every request that asked for none.

This is the highest-leverage item here and it is invisible from inside a single
request. A layer that groups penalty-using requests together — or that drops
penalties a caller set to a no-op value — keeps the cheap path cheap for
everyone else.

⚠ Verified in vLLM source, and cited in no secondary write-up found. Do not
expect a backend to do it for you.

## 2. Penalties dominate host cost; the sampler does not

Measured at batch 4: **6-47 ms/step for penalties**, growing with context length,
against **~43 us for the sampler itself** — three orders of magnitude apart.

The intuition that a 150k-250k vocabulary makes sampling expensive is wrong. It
is the penalty bookkeeping over the generated sequence that costs, and it grows
as the sequence does.

## 3. Prefer `min_p` to `top_p`

`min_p` is softmax + amax + compare, with **no vocabulary sort**. `top_p`
requires the sort. Where a caller's intent is "drop implausible tokens" rather
than "keep a fixed probability mass", `min_p` expresses it and costs less.

## 4. Round chunk sizes to a multiple of 256

Tile quantisation: a chunk size of **257 costs 32% more than 256**. Trivially
enforceable from outside, and a caller will almost never notice the difference
between the two values they asked for.

## 5. Prefix reuse is worth structuring prompts for

Production hit rates of **52.4% and 74.1%**, worth **1.7x on TTFT** — but capped
near ~50% on long-context traffic, so treat 50% as the planning figure rather
than the headline.

Mechanics that decide whether it works:
- Block granularity is **16 tokens**, so a shared prefix must align to that
  boundary to count.
- **Sampling parameters do not break the cache hash. LoRA does.**
- `cache_salt` is a per-request lever for deliberately *avoiding* reuse.

Structure prompts with the stable part first — system prompt, tools, few-shot
exemplars — and the variable part last. That is a prompt-construction decision,
which is squarely a layer-above concern.

⚠ A "91% production hit rate" for Mooncake circulates and is **refuted by the
paper's own Table 1**. Use ~50%.

## What does NOT apply here, and why

**Most host-side optimisation is measured on host-bound H100 serving.** GB10
decode at 22.7-38.7 tok/s gives a **~44 ms step budget**, against which
detokenisation is under 0.01%. At batch 1 on this machine, host work is not the
constraint — the memory bus is.

Before adopting anything from the host-overhead literature, measure whether the
workload is actually CPU-launch-bound. It very likely is not, here.

⚠ Two cautions on citing this area at all:

- The widely-repeated "GPU is only 38% of a decode step; scheduling 29%, API
  server 33%" breakdown **has no locatable source**. The ~50% CPU-overhead claim
  is real and traces to the SGLang v0.4 blog; that specific triple does not.
  It was cited several times during this work before being checked.
- One vLLM host-side optimisation **decayed from a real 5.1% win to 0.20% over
  ~1,640 commits** and was rejected. Re-measure any published delta against
  current main rather than trusting the number in its PR.
- Async scheduling **cannot be combined with structured outputs**.

## An untested cheap experiment

llama.cpp has **no big.LITTLE awareness on ARM**. The GB10 has 10 Cortex-X925
performance cores and 10 Cortex-A725 efficiency cores; threads are placed
without regard to which. Pinning to the X925 cores only is a cheap experiment,
and **no published ARM/Grace threading or core-pinning study for LLM decode
exists** — so it is genuinely unmeasured rather than known-useless.
