# PRAX -- retired

**Measured on hardware to have no effect on throughput.** This document is kept
because the negative result is worth more than its absence: it stops the same
idea being rebuilt.

## The measurement that settled it

RTX 5070 Ti, qwen3.5-0.8b, llama.cpp b10667, n-gram self-speculation, restricted
to prompts where the drafter actually fires, best of three passes:

```
block    sql tok/s   code tok/s   accept%
    1        308.7        256.7     88.5
    4        305.8        208.9     88.5
    8        297.9        209.1     88.5
   16        304.9        206.7     88.5
```

A **16x range in block size moves throughput ~5%, non-monotonically**, and
acceptance is identical to one decimal place at every setting. The parameter
PRAX exists to tune does not change what the engine does.

## Why the simulation said otherwise

The harness charged `drafted x slot_cost` -- every drafted position paying
verification whether accepted or not. Under that model an oversized block is
expensive and adapting beats a constant by +152%.

llama.cpp verifies a block in **one batched pass**, so a large mostly-rejected
block costs barely more than a small one. The cost the simulation charged for is
largely not charged in reality, and the knob is flat because the cost is flat.

The simulation was rigorous -- oracle bound, seed averaging, a memoryless
control, 179 passing tests, and it caught several real bugs including two of its
own metrics. **None of that could detect a mistaken premise.** A simulation is
only as right as its cost model, and the only test of a cost model is
measurement.

## What the same run showed DOES work

```
prompt          tok/s   drafted
sql            226.29       171     3.5x
code            94.54        73     1.5x
number list     63.82         0
json            65.40         0
prose (sky)     64.57         0
prose (hist)    62.11         0
```

Speculation is worth **1.5-3.5x on structured text and nothing on prose**, and
the drafter makes that call correctly on its own. The surviving idea is
[`spec_policy`](../praecise-runtime/src/spec_policy.rs) -- deciding *whether* to
speculate given model shape and load. Sizing the block is not.

See [measured-acceptance.md](measured-acceptance.md) for the acceptance curves
and [what-actually-helps.md](what-actually-helps.md) for what to do instead.

---

# Original design note, retained for context

Every published drafter — EAGLE-3, DFlash, DFlash2, DSpark, MTP — answers the
same question: *how do we propose better tokens?* This note proposes a different
one: *which tokens are worth proposing at all?*

The distinction matters because the theory says the first question is nearly
exhausted and the second is barely asked.

## What the theory actually permits

Three results bound what any speculative method can achieve.

**The classical relation** (Leviathan et al., arXiv:2211.17192). Acceptance is
`α = E[min(p, q)] = 1 − E[D_TV(p, q)]`, expected tokens per step is
`(1 − α^(γ+1))/(1 − α)`, and — the part that explains why speculation works at
all on memory-bound hardware — target weights and KV are read **once per
iteration** regardless of block size. Speculation is a *bandwidth amplifier*,
not a compute trick.

**The hard ceiling** (Pankratov & Alistarh, arXiv:2512.11718). For parallel
verify capacity `P` and target entropy `μ` nats:

```
E[tokens per step] ≈ log(P) / μ
```

Two consequences. Widening the draft tree buys only `log(P)` — "exponentially
increasing the computational budget P will only yield a linear improvement in
speedup". And speedup is inversely proportional to **entropy**, which varies by
almost 4× across tasks on the same model (measured: 0.279 nats on HumanEval,
1.088 on MT-Bench).

**Verification is solved.** Optimal-transport analysis (arXiv:2502.18779) puts
existing verifiers within **0.1–3.3%** of the true optimum, while the same work
measures EAGLE-3 at roughly **2× off** the achievable bound. The gap is in
*drafting and allocation*, not in the accept/reject rule.

So: ~3% left in verification, ~2× left in drafting, and a `log(P)` wall on
width. Anyone building a wider tree is optimizing the exhausted dimension.

## The opening

Two facts, both measured, that no shipped system puts together.

**Half of all tokens are nearly free.** On Qwen3-8B over 10⁶ tokens,
**50.64% have entropy below 10⁻² nats** (arXiv:2506.01939). These are closing
brackets, formatting, the second half of a known identifier. The high-entropy
tokens are logical connectives — "wait", "however", "thus" — the genuine branch
points.

**The bound assumes something false.** Both Leviathan's `α` and the
speed-of-light ceiling assume acceptance is **i.i.d.** across positions. It is
not. Entropy is bursty and autocorrelated: near-deterministic runs are followed
by branch points, in structured patterns. A bound derived under an assumption
the data violates is not a barrier — it is a description of the *average* case,
and the average is not what a decoder faces.

That is the cleanest theoretical opening available: **model acceptance as a
regime-switching process rather than an i.i.d. one, and allocate the speculation
budget accordingly.**

## The method

Call it **entropy-routed speculation**. Three parts, in increasing ambition.

### 1. Route on predicted entropy, not on a fixed block size

Every deployed method picks `draft_n` per *request*. The theory says the payoff
varies per *token*, by orders of magnitude. So:

- Estimate the entropy of the next position before drafting.
- Below a floor (~10⁻² nats), the token is near-deterministic: draft it with the
  cheapest available method — an n-gram hit costs nothing — or skip verification
  overhead entirely.
- Above a ceiling, the position is a genuine branch point. Drafting past it is
  spending verify slots on a coin flip; stop the block there.
- In between, size the block to the local entropy: `γ ≈ log(P)/μ_local` is what
  the bound itself prescribes, applied locally rather than globally.

The estimator is the design question. Candidates, cheapest first: entropy of the
*previous* position (autocorrelation makes this a real signal), the drafter's own
output entropy where one exists, and a small calibrated head. Existing work
(AdaEDL, SpecDec++) computes something similar but uses it only to *stop* a
block, never to allocate a budget across a whole request.

### 2. Exploit the burstiness the bound assumes away

If entropy runs in regimes, then a two-state model — "low-entropy run" versus
"branch region" — predicts not just the current position but the *expected
length of the current run*. That is exactly the quantity a block-size decision
needs, and it is unavailable to any i.i.d. formulation.

Concretely: in a low-entropy regime, extend the block aggressively, because the
run is likely to continue. On entering a branch region, collapse to `γ = 1` or
none. The transition matrix is cheap to estimate online, per request, and is
itself reusable across a session.

**Why this can legitimately exceed the published bound:** the bound is over
i.i.d. sequences. A method that conditions on regime is not violating it — it is
operating outside its hypothesis.

### 3. Fix the reveal order

An identity from information theory (arXiv:2608.25505, cs.IT) gives the *exact*
cost of parallel token generation:

```
D_KL(p ‖ q_π) = E_x[ Σ_t TC(X_{S_t} | x_{C_t}) ]
```

Conditional total correlation **is** the information cost of revealing a set of
positions in parallel. Not a bound — an identity.

Its practical consequence is startling and inverted from practice: measured over
542 texts, **confidence-based top-k selection costs 1.650 bits/token against
0.151 for uniform-random** — worse on *every single text*. Hierarchical and
bisection reveal orders are exponentially better than left-to-right.

A draft tree **is** a reveal-order policy. Nobody has carried this result from
diffusion LMs across to draft-tree construction. That is a free, theoretically
grounded improvement sitting unclaimed.

### Free wins worth taking regardless

- **Sample without replacement.** Measured **3.0–4.0 points** of acceptance over
  with-replacement, provably, and largely unimplemented.
- **Cross-request structure.** Suffix decoding reports **5.3×** on agentic SQL
  where EAGLE-2/3 fail outright, and production data shows **55% cross-turn** KV
  hit rates. This is the largest empirical gap in the field, and it is the one
  our [`ngram`](../praecise-runtime/src/ngram.rs) cache is already positioned
  for — it is deliberately not tied to a request.

## Where this must not be built naively

**The verify path must compile to a batched GEMM.** On quantized Metal, verify
cost was measured growing *linearly* in block size (51→218 ms for 2→7) because
the path emitted GEMV-per-position instead of a batched GEMM — and three of five
configurations **decelerated**, to 0.33–0.52×. If verification does not become a
single matmul, speculation's entire arithmetic-intensity argument is notional.
This belongs in the backend contract, asserted, not assumed.

**Quantization and speculation spend the same budget.** Both raise arithmetic
intensity from ≈1 FLOP/byte toward the hardware ridge. On a low-ridge machine
they overshoot together: on Apple Silicon, 1-bit quantization is already
*arithmetic*-bound, so IQ1_M (21.28 ms/tok) is **slower** than Q2_K (15.59).
Budget them jointly.

**Past the ridge, speculation goes negative.** Measured crossovers: compute-bound
above batch 8 on A100, batch 32 on H200, where speculation reaches **0.82×**.
This is the same conclusion [`spec_policy`](../praecise-runtime/src/spec_policy.rs)
already enforces on utilization, arriving from roofline rather than from
benchmarks.

**MoE inverts twice.** Batch-1 MoE reads only active experts — good on unified
memory (80B at 52 tok/s on 273 GB/s). But draft trees fan expert activation out
**3.5× at K=7**, cancelling it. And standard MBU overstates utilization by
`n_expert/k_active` — a **262% error** on Mixtral. Measure with S-MBU or the
numbers are fiction.

## What is genuinely unknown

Worth stating plainly, because it is where our hardware gives us an advantage
nobody else has published:

- **No roofline analysis exists for GB10 or GH200 LLM decode.** None. The
  literature covers A100/H100/B200 and Apple Silicon.
- **No speculative crossover data for Grace/Blackwell/Jetson.**
- No convergence bound for Jacobi/Gauss-Seidel autoregressive decoding.
- No information-theoretic account of *why* MTP works.
- EAGLE's central claim — that features are more compressible than tokens — is
  **never actually stated or proven** in its papers.

Measuring the first two on the GB10 would be a contribution in itself, and it is
a measurement, not a research programme.

## Order of work

1. **Entropy estimator + routing** — the whole method rests on it, and it is
   testable offline against logged entropy distributions with no GPU.
2. **Regime model** — cheap to add once routing exists, and it is the part with
   a real claim to novelty.
3. **Reveal-order (tree construction)** — highest theoretical leverage, most
   engineering.
4. **Without-replacement sampling** — small, provable, do it early.
5. **GB10 roofline + crossover measurement** — fills a genuine hole in the
   literature and calibrates everything above for our own hardware.

Everything in 1–4 is orchestration: it lives in this layer, above any backend,
which is the whole argument for the layer existing.

## Sources

Leviathan et al. arXiv:2211.17192 · Pankratov & Alistarh arXiv:2512.11718 ·
optimal transport arXiv:2502.18779 · SpecTr arXiv:2310.15141 · entropy
distribution arXiv:2506.01939 · total-correlation identity arXiv:2608.25505 ·
parallel sampling arXiv:2511.07869 · SuffixDecoding arXiv:2411.04975 · DReSD
arXiv:2502.15572 · Quest arXiv:2406.10774 · TEAL arXiv:2408.14690 · SpecDec++
arXiv:2405.19715.

⚠ Papers with 26xx identifiers are unreviewed preprints with self-reported
numbers. The load-bearing claims here — Leviathan's relations, the
speed-of-light bound and its EAGLE-3 gap, the optimal-transport gaps, and the
total-correlation identity with its 1.650-vs-0.151 inversion — were extracted
from full paper text rather than abstracts.
