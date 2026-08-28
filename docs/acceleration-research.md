# Acceleration research — what Praecise Engine should implement

Surveyed August 2026 against llama.cpp, vLLM, SGLang and unsloth. Every claim
below was checked against source, release notes or a paper at the time of
writing; version-specific numbers are labelled as such, because several of them
have already moved once and will move again.

Praecise Engine is a layer, not a runtime. unsloth is the same shape — it sits
above the backends and makes them faster — and is the standard of craft this
engine aims at: bit-exact optimizations rather than approximations, gradients
and reductions hand-derived so results can be written into buffers that already
exist, and honest measurement (they once published a correction because their
tokens/sec had been counting startup time). llama.cpp, vLLM and SGLang are
backends to integrate and optimize for.

That framing decides what belongs here. An optimization belongs in Praecise
Engine when it is expressible above a backend; it belongs in a backend adapter
when it needs that backend's kernels or memory layout.

## What we already have

- Speculative decoding: self-speculative block (DFlash) and MTP paths.
- Per-architecture matmul tuning; multi-arch CUDA binary, Turing -> Blackwell.
- Architecture coverage including muse-glimmer and qwen4exp.
- A build-time guard refusing a CUDA build the environment would silently make
  CPU-only.

## Tier 1 — portable, and worth building here

These need no backend's internals, so they are the engine layer's natural work.

### 1. Suffix / n-gram self-speculation

The cheapest real speedup in the field. A suffix tree over the prompt plus prior
generations, frequency-ranked, proposing continuations. No draft model, no extra
weights, no GPU memory — pure CPU token-ID matching.

llama.cpp's `ngram-mod` (shared hash pool, constant memory, variable draft
length) documents **0.703 acceptance**, against 0.576 for the simpler variant.
vLLM ships the same idea as suffix decoding, defaulting to a *maximum* of 32
speculative tokens chosen adaptively per request per step.

Two properties make this the highest value-per-effort item for a layer:
it needs nothing from the backend but token history, and — uniquely — the index
can span requests and sessions. Agentic loops, code editing and RL rollouts
repeat themselves heavily, and a cross-session suffix index is something no
single runtime is positioned to build.

### 2. Speculation as a portfolio, with honest economics

Speculative decoding is not one feature to switch on. Choose per model, per
hardware, per load:

- Published speedups are almost always quoted at concurrency 1 and decay hard
  under load. vLLM's P-EAGLE measures 1.55x at c=1 and **1.05x at c=64** on
  MT-Bench. Speculation trades FLOPs for latency, and a loaded server has none
  spare.
- **MoE inverts the economics.** Parallel verification activates far more experts
  than single-token decode. llama.cpp measures EAGLE3 at 3.28x on dense
  Llama-3.1-8B but **1.08x on GPT-OSS-120B**.
- MTP is a latency win that costs throughput: it forces `n_parallel=1` and drops
  prefill to roughly half, because hidden states move device->host.
- The field has moved past tree verification. DFlash verifies a linear block,
  DSpark schedules a linear window, and SGLang's CPU/GPU overlap scheduler
  supports only `topk=1`. Designing for trees aims at a receding target.

The portable part of the frontier is a *scheduling policy*: DSpark sizes each
verify window from the draft's own calibrated confidence against a pre-profiled
steps-per-second curve. That is implementable above any backend that exposes
per-position draft confidence. Its paper also flags a trap worth respecting — a
scheduler searching without causal early stopping biases the output distribution
toward tokens that happen to trigger longer verifications. Causal stopping is
load-bearing for correctness, not just speed.

### 3. Per-hardware x per-quant-type dispatch

The single global `MMVQ_MAX_BATCH_SIZE = 8` left up to **49%** on the table.
llama.cpp replaced it with an empirically swept table: RTX 5090 crosses at batch
5 for Q2_K..Q5_K and 7 for Q6_K; RTX 4090 at 4/6/7; DGX Spark only Q2_K at 6;
legacy quants stay at 8.

Two things generalize. The methodology: sweep with **clocks pinned**
(`llama-bench -p 1..8 -n 0 -embd 1 -r 50`), because thermal drift is larger than
the effect being measured. And the classification, which matters more than any
single number: **MMVQ is bandwidth-bound and wants occupancy; MMQ is
register-pressure-bound and wants per-block resources.** The same occupancy
change measured +3-6% on one and **-30%** on the other. Never apply one policy
globally.

### 4. Evaluate quantization by trajectory divergence

Perplexity and single-token top-1 both under-detect damage that compounds over a
generation. The stronger metric now in use is KL divergence over a multi-token
greedy decode against the unquantized reference — unsloth scores 300 held-out
examples over 32 tokens. Any quantization decision this engine makes should be
justified that way rather than by perplexity.

Per-tensor bit allocation itself is a *packaging* decision, not a runtime one:
`llama-quantize --tensor-type` takes regexes, so a scheme is expressed as data
and costs nothing at inference time.

### 5. Structured output: pick the backend before dispatch, not after

The mechanism worth adopting is llguidance's lexer/parser split. GBNF has no
lexer; llguidance runs a regex lexer in front of the CFG parser, and because
there are ~10x fewer lexemes than bytes and tokens tend to align with lexemes,
**the CFG parser is engaged on under 0.5% of tokens.** Measured mask cost is
~50us mean, 0.5ms p99 on a 128k vocab.

Backend selection should be a *schema inspection* up front, not an exception
caught downstream. vLLM discovers xgrammar cannot handle a schema by catching a
validation error; the predicate is a pure function and can be run first. This
matters beyond tidiness: xgrammar **silently drops `minLength`/`maxLength`** when
combined with `pattern`/`format`, so output can violate the schema with no error
raised.

Grammar failures must degrade, never abort. A grammar that gives up should be
reported once, not once per token.

## Tier 2 — backend-adapter work

Real wins, but they need a backend's internals, so they belong in adapters.

- **Paged KV.** llama.cpp's unified cache preallocates `n_ctx * n_seq_max`, so
  concurrency is bounded by *maximum* context rather than average. A paged
  prototype reaches 247 concurrent sequences against 25 for unified (2.5x
  aggregate throughput) at equal low-concurrency performance. Its user-facing
  defrag knob is already deprecated in favour of context checkpoints.
- **KV cache quantization.** FP8 KV measures +14.9% throughput and -14.8% median
  ITL on Llama-3.1-8B/H100 — but the break-even context length is
  *version-specific* (24,889 tokens on one release, 7,010 on a later one; 741,565
  vs 22,109 on another model). Two gates: head_dim=256 makes TTFT ~1.6x slower,
  and on hybrid models skipping sliding-window layers beats quantizing everything.
- **Flash attention** is now tri-state and defaults to `auto`. The trap is that
  quantized K/V combinations are only compiled with `GGML_CUDA_FA_ALL_QUANTS=ON`;
  without it an unsupported combination silently falls off the FA path.
- **Blackwell FP4.** The generic NVFP4 MMQ kernel gives +118% to +191% on prefill
  but is **flat (~0.99x) on token generation** — a prefill win, not a decode one.

## Correctness traps to test for explicitly

Cheap to check, and each one produces plausible-looking bad output rather than an
error:

- **BOS handling.** Adding or omitting the start-of-sequence token is the single
  most reported cause of gibberish, endless generation and repetition across
  engines. Assert it rather than trusting the template.
- **Untrained tokens.** Reserved/special tokens ship with garbage embeddings in
  some base models; constrained decoding can *force* sampling into exactly those.
- **Per-model numerical quirks** — logit scaling, logit softcapping — silently
  corrupt output when a runtime omits them.
- **Tokenizer bugs live in the runtime, not the artifact.** At least one Gemma 4
  tokenizer fix was C++-only and needed no GGUF regeneration, so "re-download the
  model" is often the wrong remedy.
- **A denominator correct per-microbatch is wrong across microbatches of unequal
  token count.** The same class of bug appears in any per-sequence-averaged metric
  under ragged batching.

## What this implies for the roadmap

1. Suffix/n-gram self-speculation, cross-session — highest value per unit effort,
   **implemented** as `praecise_runtime::ngram::NgramCache` (backend-agnostic;
   still to wire into the decode loop and measure against the 0.703 reference),
   and structurally ours rather than any backend's.
2. A speculation policy layer: pick method and verify-window per model, hardware
   and load; refuse to speculate where it does not pay (MoE under concurrency).
3. Per-hardware x per-quant dispatch tables, swept with pinned clocks.
4. A trajectory-divergence harness, so quantization and speculation claims are
   measured the way they actually degrade.
5. Schema-inspecting structured-output routing with a shared compile cache.

## Where the frontier converged (verified August 2026)

The models this engine has to serve are no longer a zoo of unrelated designs.
Four labs independently shipped the same primitives, which is what makes an
acceleration layer tractable at all.

**The common design.** DeepSeek V4, GLM-5.3-Flash, Kimi K3 and qwen4exp all run
MoE with aux-loss-free (`noaux_tc`) balancing plus shared experts, compressed
latent attention, a learned indexer selecting a sparse top-k, and Sinkhorn-
projected hyper-connections. The hyper-connection hyperparameters are
*byte-identical* across DeepSeek V4 and GLM-5.3-Flash (`hc_mult: 4`,
`hc_sinkhorn_iters: 20`, `hc_eps: 1e-06`), and llama.cpp reuses DeepSeek V4's
`HC_PRE` kernel for Kimi K3. One kernel serves four architectures.

**A 3:1 hybrid of linear and full attention is now the mainstream layout** --
qwen4exp, GLM-5.3-Flash and Kimi K3 all use it. Gemma 4 (5:1) and muse-glimmer
(3:1) alternate sliding and full instead. So the engine needs exactly three
attention regimes, not one per model: hybrid linear+full, sliding/full
alternation, and indexer-based block-sparse.

This is worth stating plainly about our own work: **qwen4exp's hybrid memory,
block-sparse QSA indexer and gated hyper-connection residual are the convergent
frontier design, not an outlier.** That is a reason to invest in the shared
primitives rather than treat the architecture as bespoke.

**Speculation has four shapes** to support: MTP heads in the model, dedicated
draft models, DSpark (confidence-scheduled), and DFlash (block diffusion). Note
Kimi K3 ships `num_nextn_predict_layers: 0` -- no MTP head at all -- and relies
on DSpark. An engine that assumes MTP is available will simply have nothing to
speculate with.

### Architecture traps to test against

Each of these degrades silently rather than erroring:

- **NoPE is spreading.** GLM-5.3-Flash and Kimi K3 have *no RoPE in the text
  tower* (`mla_use_nope: true`, `qk_rope_head_dim: 0`); position comes from the
  linear-attention recurrence. muse-glimmer drops RoPE on global layers only.
  Code that assumes RoPE everywhere mis-serves two frontier families.
- **Keep selection machinery out of the quantized path.** Every one of these
  models quantizes routed experts to FP4/MXFP4 while holding indexers, gates and
  hyper-connection mixers at higher precision. Quantizing an indexer corrupts
  top-k selection, which degrades quality without any error -- and that applies
  directly to qwen4exp's QSA indexer.
- **Frontier models increasingly ship no working Jinja chat template.**
  DeepSeek V4 and Kimi K3 have no `chat_template` at all and need custom
  encoders; DeepSeeks format uses full-width characters in its role markers.
  This is the same class of failure as the BOS trap below, and it is why
  `force_pure_content` exists as an escape hatch.
- **Some models are not bit-exact between cached and fresh runs** (an MXFP4
  kernel constraint on DeepSeek V4). Equality assertions in tests will flake;
  compare scores, not tokens.

## Implementation language: what the evidence actually says

Worth recording, because the intuition here is wrong in both directions.

**Host overhead is real and large.** Profiling vLLM on an H100 with Llama-3-8B
found GPU execution was only **38%** of step time -- scheduling 29%, API server
33%. An independent study measured scheduling at up to **~50%** of end-to-end
latency. "The kernels dominate, so the host language does not matter" is false.

**But no one solved it by changing language.** vLLM stayed in Python and moved
input preparation onto the GPU (Model Runner V2, **+56%** throughput); SGLang's
overlap scheduler runs one batch ahead, worth **10-20%**; CUDA graphs remove the
host from the launch loop entirely. The fix is always *taking work off the
critical path*, not making that work faster.

Two natural experiments already ran, and both cut against a compiled
orchestration layer. NVIDIA **deleted** TensorRT-LLM's C++ backend ("PyTorch is
now the sole execution backend", v1.2) for developer efficiency. And TGI -- the
Rust-router precedent -- was **archived** in March 2026, pointing users at
vLLM/SGLang/llama.cpp. Rust did not save it; failing to keep pace with model
architectures did.

That is the real cost to weigh: EAGLE-3, P-EAGLE, DFlash and DSpark all arrive
as Python/Triton PRs against vLLM or SGLang. There is **no non-Python adoption
path**. A compiled layer reimplements each by hand and permanently trails.

**Where a compiled layer genuinely wins is embedding.** Praecise Engine is a
library that loads inside someone else's process, and there Python's single
interpreter, GIL and dependency weight are disqualifying -- vLLM's own fix for
GIL contention was to move its API server into a *separate process*. Rust gives
a `cdylib` with no runtime to initialise.

So the split this engine already has is the defensible one: Rust for the
embeddable layer, CUDA C++ for kernels. The conclusion to carry forward is not
about language at all -- it is that **host work must come off the critical
path**, by preparing inputs on the GPU and overlapping scheduling with compute.
Note also that no controlled benchmark of a Rust orchestration layer against a
Python one over identical kernels appears to exist publicly; anyone claiming a
decisive language win in either direction is extrapolating.

## Kernel strategy is set by batch shape, not by hardware

The assumption worth discarding first: "batch-1 decode leaves tensor cores idle,
so the vector path always wins." That is true of some engines and false of
others, and the difference is not the GPU.

Verified by reading the kernels at HEAD (2026-08-27):

| Codebase | Batch-1 path |
|---|---|
| llama.cpp | dp4a SIMT GEMV — `mmvq.cu` contains no MMA at all |
| FlashInfer | scalar FMA + warp shuffle — `decode.cuh` has none, `prefill.cuh` has 40 |
| vLLM | **stays on tensor cores** — no GEMV kernel, no `dp4a` anywhere in `csrc/` |

vLLM's Marlin specializes tiles for small M but keeps the arithmetic on
`mma.sync.aligned.m16n8k16`, adapting through split-K to fill the SMs rather
than leaving the tensor cores. Its AllSpark W8A16 path inverts the naive
expectation outright: small M runs a tensor-core kernel, while large M (>1024)
falls back to cuBLAS.

**The dividing line is whether a deployment ever presents M=1 to the GEMM.**
llama.cpp's steady state is single-user local decode, which is M=1 exactly.
vLLM's is a continuous-batching server holding dozens of sequences, so it has
little reason to keep a true GEMV path at all.

That is a decision this engine has to make explicitly rather than inherit. A
batching server should follow vLLM's precedent — tensor cores plus split-K —
even at low nominal batch, because the effective M is the number of concurrent
sequences, not one.

Two more measured points worth keeping:

- `MMQ_DP4A_MAX_BATCH_SIZE = 64`: even on hardware with FP16 tensor cores,
  llama.cpp prefers integer SIMT below batch 64.
- Tensor-core availability *shrinks* the vector regime rather than widening it
  (`mmvf.cu`): Ampere and later use the vector path only at `ne11 == 1`, Ada up
  to 4, and hardware without tensor cores up to 8. The per-quant tables carry
  comments naming the GPU they were tuned on — these are measurements, not
  derivations, which is the standard any tuning added here should meet.

### Two corrections to received wisdom

**H100 at batch 1 is launch-bound, not bandwidth-bound** — roughly 26% of peak.
CUDA graphs are worth ~1.26x there and nothing on an L4. Buying HBM bandwidth
does not buy batch-1 latency, so "bandwidth ÷ bytes-per-token" is a ceiling, not
a prediction; on a fast GPU at batch 1 the host and launch path bind first. This
is the same conclusion the host-overhead numbers reach from the other direction.

**Halve NVIDIA's headline tensor figures unless they say dense.** The GB200 page
states the convention in plain text: the large number is sparse, and dense is
half of it.
