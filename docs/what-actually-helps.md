# What actually speeds up decode on a GB10, ranked

Decode on this hardware runs at ~80% of its 273 GB/s memory-bandwidth ceiling
(see [gb10-bandwidth-ceiling.md](gb10-bandwidth-ceiling.md)), so the only things
that help are the ones that **move fewer bytes per token**. This is that list,
ordered by measured effect, with the things that look promising and are not.

Every figure here is cited to a primary source — an engine PR, vendor
documentation, or a paper. Where no primary source exists for a claim, that is
stated rather than filled in.

## The structural correction that reorders everything

For an MoE model the weight term is **active** parameters only. On a
Qwen3-30B-A3B-class model at 32k context, weights are only ~35% of bytes moved
once quantised to 4 bits — so the KV cache, not the weights, dominates.

At short context the reverse holds. This is why the ranking below is
context-dependent and why "quantise the weights harder" stops paying so quickly.

## 1. Pick the architecture first

The largest single difference is decided before any flag is set.
Gemma-3-27B (5:1 local/global, sliding window 1024) holds **2.91 GB of KV at
32k**; Qwen3-32B holds **8.00 GB**. Nothing downstream recovers a 2.7x
difference in the dominant term.

## 2. Quantise the KV cache — the biggest lever, and it is off by default

```
-ctk q8_0 -ctv q4_0 -fa on
```

**+48% throughput for about 0.002 PPL** (llama.cpp PR #7412). Flash attention is
required for the quantised-KV path to be taken at all.

This is the highest-value line in this document: a measured, large win, available
now, that a default configuration does not give you.

## 3. Four-bit weights: already solved — do not re-engineer

Use **Q4_K_M** or **MXFP4** and move on. Switching between 4-bit formats is worth
about 5%, and one popular choice is worth *less than nothing*:

| Qwen3.5-0.8B, same hardware | pp512 | **tg128 (decode)** |
|---|---:|---:|
| NVFP4 | 25,596 | **388.12** |
| Q4_K_M | 38,339 | **521.69** |

**Q4_K_M beats NVFP4 at decode by 34%**, despite NVFP4 having native Blackwell
support and comparable bits per weight (4.50 vs 4.25 — NVFP4 is the *heavier*
format).

The reason, from three independent sources: every engine deliberately bypasses
the tensor cores below batch 8. NVIDIA's own TensorRT-LLM routes
`qinput.shape[0] <= 8` to `cuda_scaled_mm`, whose NVFP4 small-batch kernel is
`TILE_M=1` scalar `fma()` with **zero MMA instructions**; llama.cpp caps
`MMVQ_MAX_BATCH_SIZE = 8` and decodes NVFP4 through a 16-entry LUT into `dp4a`,
an **sm_61 intrinsic**. The one native-FP4 merge reports **prefill only** — there
is no decode row in it anywhere, which is itself the tell.

⚠ Also note `Q4_K` the *format* is 4.50 bpw; the widely-quoted ~4.85 is the
`Q4_K_M` **recipe**, which promotes `attn_v` and `ffn_down` to Q6_K. They are not
comparable numbers.

⚠ **sm_120 FP4 is narrower than "Blackwell has FP4".** Per the PTX ISA,
`.kind::mxf4nvf4` and `.kind::mxf4` are supported on **sm_120a and sm_121a
only** — excluded from the family-forward `sm_120f` path. And `tcgen05.mma` FP4
kinds are sm_100a/103a/110a only, never sm_120.

## 4. Speculative decoding — the most underused batch-1 lever

3-4x, lossless, and it composes multiplicatively with everything above. It is a
bandwidth *amplifier*: the weights are read once and several tokens come out.

Note this is about *using* speculation at all, not about tuning how many tokens
to draft — see [praecise-method.md](praecise-method.md) for why the tuning
question turned out to be second-order.

## Things that look promising and are not

**ParoQuant is ~10% slower than AWQ** (2.1x vs 2.3x), not "near-AWQ speed" as
its model card states. It buys accuracy at equal bytes, not speed.

**No KV eviction method ships.** H2O, SnapKV, Quest and PyramidKV are in neither
vLLM nor SGLang. What actually shipped is *trained-in* sparsity — NSA, DeepSeek
DSA — which is a property of the model, not a runtime option. Of the post-hoc
methods only Quest works on off-the-shelf dense models (2.23x attention, 7.03x
end-to-end), and it is a build-it-yourself item.

**Activation sparsity is the wrong lever on fast hardware.** TEAL measures
1.53x/1.80x on an A6000 but only 1.25x/1.40x on an A100 — the speedup *shrinks*
as the machine gets faster, because it exploits bandwidth slack that a saturated
machine does not have.

**Host-side optimisation has near-zero return here.** Every published figure is
from host-bound H100 serving. GB10 decode at 22.7-38.7 tok/s gives a **~44 ms
step budget**, against which detokenisation is under 0.01%. Measure whether the
workload is CPU-launch-bound before adopting any of it — at batch 1 on this
machine it is not.

## The honest gap

**No primary source anywhere gives a PTQ accuracy table comparing NVFP4, MXFP4,
AWQ, GPTQ, Q4_K_M and IQ4_XS on a shared model and benchmark.** NVIDIA's only
MXFP4 comparison is *pretraining* (8B, 1T tokens, 36% more tokens needed).

Anyone ranking these formats for post-training quantisation quality is
extrapolating — including sources that present such rankings confidently. The
throughput numbers above are measured; a quality ranking is not available.
