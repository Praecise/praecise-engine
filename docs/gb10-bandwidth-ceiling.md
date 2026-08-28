# The GB10 decode ceiling

**Decode on a DGX Spark is bandwidth-bound and already running at ~80% of peak.
There is no software win waiting.** This is the governing constraint on anything
Praecise Engine does on this hardware: it bounds every host-side optimisation to
the ~20% that is not already saturated, and should be established before any
such work is planned.

## The derivation

Llama-2-7B Q4_0 weights are 3.56 GiB = 3.82 GB. Decode reads every weight once
per token, so tokens/sec × bytes/token is achieved bandwidth:

```
3.82 GB/token × 57.21 tok/s = 218.7 GB/s
218.7 / 273 GB/s peak       = 80.1%
```

Same benchmark, same model, same quantisation, other hardware:

| Platform | Peak GB/s | tg t/s | Achieved | Efficiency |
|---|---|---|---|---|
| **GB10** | 273 | 57.21 | 218.7 | **80.1%** |
| M4 Max | 546 | 83.06 | 317.5 | 58.1% |
| M3 Ultra | 800 | 92.14 | 352.2 | 44.0% |
| M2 Ultra | 800 | 94.27 | 360.3 | 45.0% |

GB10 extracts the **highest** fraction of peak in the group. A machine at 44% of
peak has software headroom; one at 80% does not.

**Independent cross-check.** Applying the same figure to a different model:
218.7 ÷ 60.57 tok/s = 3.61 GB/token on gpt-oss-120b. Over 5.1B active
parameters that is **5.66 bits/parameter** — exactly what MXFP4 experts plus
higher-precision attention should cost. The model predicts a measurement it was
not fitted to, which is the reason to trust it.

⚠ No published STREAM/BabelStream microbenchmark for GB10 exists, so the 80% is
inferred from decode rather than measured directly. For this purpose that is
arguably the better metric — it is the bandwidth actually reachable by the
workload — but it is an inference, not a datasheet number.

## What follows

**Ceilings at 219 GB/s achieved:** a 30B Q4 model tops out around 13 tok/s, a
70B Q4 around 5.5 tok/s. Those are arithmetic, not engineering targets.

**Arithmetic intensity.** Decode sits at ≈1 FLOP/byte. The FP4 ridge point is
1832 FLOP/byte and the BF16 ridge is 458, so **less than 1% of the advertised
FP4 PFLOP is reachable at batch 1**, and batch ~229 would be needed to reach even
the BF16 ridge. The compute is not the constraint and mostly cannot be used.

**Flash attention buys prefill, not decode**: measured +20% on pp512 and
**zero** on tg128, which is what a bandwidth ceiling predicts.

**Relative position:** GB10 is 2.08× an M3 Ultra on prefill and 0.62× on decode
— it is a prefill machine. Against discrete parts on gpt-oss-20b: RTX 5090
205 t/s, RTX Pro 6000 215 t/s, Spark ~50 t/s.

## Levers that remain, in order

1. **Move fewer bytes per token.** The only lever that moves the ceiling itself.
   4-bit is the practical floor: 465+ pretraining runs put the compute-optimal
   point at 7-8 bits and warn that below 4 bits model size must grow more than
   4× to compensate. 4-bit PTQ is effectively solved (QuaRot W4A4KV4 on
   LLaMA2-70B: ≤0.47 PPL loss).
2. **Remove the KV cache.** Linear and hybrid attention is the strongest decode
   lever available, because it deletes bytes rather than exploiting slack — so
   the win *grows* with context and *survives* batching. Kimi Linear reports
   "up to 6 times decoding throughput for a 1M context" and "reducing KV cache
   usage by up to 75%"; Qwen3-Next-80B-A3B claims "10 times inference throughput
   for context over 32K".
3. **Raise batch size.** The only way to reach the compute at all.

**Activation sparsity is the wrong lever here**, and its own data says so: TEAL
measures 1.53×/1.80× on an A6000 but only 1.25×/1.40× on an A100 — the speedup
*shrinks on faster hardware*, because it exploits bandwidth slack that a
saturated machine does not have. Its authors concede batch scaling "is a
limitation of most activation sparsity work" (per-layer sparsity falls 60% → 38%
at batch 4). The 4-5× claims in this area are all CPU/mobile/offload settings.

**CPU+GPU cooperation is a capacity mechanism here, not a speed one.** Every
offload system that wins arbitrages fast-small VRAM against slow-large DRAM
across a narrow PCIe link; GB10 has none of those three (`integrated=1`,
`pageableMemoryAccess=1` — genuinely coherent, verified from device properties).
Fiddler's scheduling rule compares `cpu_lat` against `gpu_lat + transfer_lat`;
set transfer to zero at equal bandwidth and the GPU wins essentially always.
Measured on the closest analogue, Strix Halo: moving only `tok_embd` to CPU cost
**41.33 → 47.24 t/s, a 14% loss**, because CPU and GPU share the same DDR.

## What this means for this engine

**Allocation policy cannot beat a bandwidth wall.** Speculative decoding is a
bandwidth *amplifier* — it reads the weights once and emits several tokens — so
it remains the right family of technique. But tuning *how many* tokens to draft
is a second-order effect on a machine already at 80% of peak, and that is the
honest frame for PRAX's measured gains.

The first-order work on this hardware is quantisation and KV-cache removal, both
of which are properties of the **artefact and the architecture**, not of the
serving layer.

⚠ **sm_121 has a real software gap** worth knowing before blaming hardware: six
or more open vLLM bugs where frameworks gate on Blackwell family 100 or 120 and
sm_121 falls between — MXFP8 and NVFP4 MoE silently falling back to Marlin,
`support_deep_gemm()` returning true then hard-faulting, launch failures after
1.5-3 days with **zero Xid** (software, not hardware). Also no prebuilt wheels
for `mamba` or `causal-conv1d` on sm_121 — which is awkward, since the hybrid
models are the best decode lever and the hardest to build here.

## Attention on sm_121: most of the modern stack is absent

Worth knowing before attributing slow decode to the hardware. On SM120/121 vLLM
dispatches **FlashInfer -> XQA**; the newer paths are simply unavailable:

- **FA3** is SM90-only.
- **TRTLLM-Gen** is gated on `major == 10`, so SM100 only.
- **CUTLASS FMHA** excluded; **FlashMLA** unusable on sm_121.
- **FA4** is released but targets sm_90/100/103 — its SM120 path is a "graceful
  SplitKV fallback", not a real kernel.

Same pattern as the MXFP8/NVFP4 MoE fallbacks: frameworks gate on Blackwell
family 100 or 120, and sm_121 falls between.

⚠ **Operational trap.** A FlashInfer cubin version mismatch crash-loops
vLLM >= 0.27 on ARM64. Known-good pairing is **vLLM 0.25.1 + flashinfer 0.6.13**.

### Why this does not change the ranking

Attention is a small share of decode bytes in the regime that matters here. The
crossover context — where KV traffic overtakes weight traffic — is far above
typical use at low batch:

| Model | B=1 | B=4 | B=8 |
|---|---|---|---|
| 8B BF16 | 122k | 30.5k | 15.3k |
| 8B Q4 | 34.2k | 8.5k | 4.3k |
| 70B Q4 | 120k | 29.9k | 15.0k |

Below 8k context at batch 1-8, attention is **1.7-6.3% of decode bytes**. Note
the Q4 row though: quantising weights *pulls the crossover in* to 4-8k at batch
4-8, so attention starts to matter once the quantisation win has been taken —
the levers interact.

**Sparse attention splits cleanly on whether it needs retraining.** NSA, DeepSeek
DSA and MiniMax MSA all require training from scratch. Only **Quest** works on
off-the-shelf dense models (2.23x attention, 7.03x end-to-end) — but it is
integrated into neither vLLM nor SGLang, so it is a build-it-yourself item.

⚠ The crossover table is arithmetic from published specs, not a measured GB10
benchmark; no published GB10 decode breakdown exists. The direction is
corroborated independently (NSA's arithmetic-intensity argument, and vAttention
finding paged and non-paged decode latency similar), but validate before planning
against the exact numbers.

## The cheap experiment that would settle the last open question

Sweep `--n-cpu-moe` from 0 on gpt-oss-120b, and separately measure whether the
GPU alone saturates 273 GB/s. If it saturates, cooperative CPU+GPU execution
cannot win by construction. If there is headroom, that is a real and unpublished
result. **No published measurement of CPU+GPU cooperative inference on coherent
memory exists in either direction** — the absence is itself a finding.
