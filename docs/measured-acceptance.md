# Measured draft acceptance, and what it refutes

**The first non-simulated measurement in this work.** Run 28 Aug 2026 on an
RTX 5070 Ti (16 GB, discrete), Qwen3-0.6B Q8_0, llama.cpp b10667, n-gram
self-speculation (`--spec-type ngram-mod`), greedy decoding, 8 prompts spanning
structured text through open prose.

## The numbers

```
draft tokens:    598
accepted:        556       overall acceptance 93.0%
draft events:     12
```

Per-position acceptance, out of 12 draft events:

| position | accepted | rate |
|---:|---:|---:|
| 0-4 | 12 | **100%** |
| 5-20 | 11 | **92%** |
| 21-28 | 10 | **83%** |
| 29 | 9 | **75%** |

## What it refutes

`prax::affordable_block` prices a draft block as `a^k` -- geometric decay, on
the reasoning that position `k` is reached only if every earlier position
matched. That reasoning is sound and the model is still wrong, because
acceptance at each position is **not independent**. Once a drafter has locked
onto a repetitive or structured passage it keeps being right.

| position | measured | `a^k` predicts | error |
|---:|---:|---:|---:|
| 0 | 1.000 | 0.930 | +0.070 |
| 5 | 0.917 | 0.646 | +0.271 |
| 10 | 0.917 | 0.449 | +0.468 |
| 20 | 0.917 | 0.217 | +0.700 |
| 29 | 0.750 | 0.113 | **+0.637** |

**At position 29 the geometric model is off by 7x.** It predicts 11% where the
drafter actually delivers 75%.

The practical consequence runs one way: the closed form makes PRAX draft
**far shorter blocks than it should**, so it leaves throughput on the table
rather than wasting verification. Every simulated result in
[praecise-method.md](praecise-method.md) rests on this model, and is therefore
pessimistic about long blocks.

Two incidental findings:

- llama.cpp drafted **past `--spec-draft-n-max 8`** -- the histogram runs to
  position 29 -- so that flag is not the hard cap it reads as.
- **12 draft events produced 556 accepted tokens.** On structured text an
  n-gram drafter is extraordinarily effective, which is the case for
  `crate::ngram` needing no model at all.

## Second run: Qwen3.5-0.8B, the model actually in use

Repeated on `qwen3.5-0.8b` -- already present on the host, and the model chosen
over 0.6B -- with the same prompts, drafter and settings.

```
draft tokens:    397
accepted:        397       overall acceptance 100.0%
draft events:      8
```

**Every drafted token was accepted.** And yet the per-position curve still
decays:

| position | accepted of 8 | rate |
|---:|---:|---:|
| 0-9 | 8 | **100%** |
| 10-18 | 7 | **87.5%** |
| 42 | 6 | **75%** |
| 43-63 | 5 | **62.5%** |

Note it runs to **position 63** -- llama.cpp drafted eight times the nominal
`--spec-draft-n-max 8`.

### This kills the geometric model from both directions

| position | measured | `a^k` at a=1.00 | `a^k` at a=0.93 |
|---:|---:|---:|---:|
| 9 | 1.000 | 1.000 | 0.484 |
| 18 | 0.875 | 1.000 | 0.252 |
| 43 | 0.625 | 1.000 | 0.041 |
| 63 | 0.625 | 1.000 | 0.010 |

Fit `a` to the *overall* acceptance of 100% and the model predicts no decay
ever -- it would draft without limit. Fit it to the observed tail and it
collapses immediately, drafting far too short. **No single `a` reproduces the
shape**, because the shape is not geometric.

### What the shape actually is

A **step function**: long flat runs punctuated by sudden drops. That is exactly
what an n-gram drafter does -- it holds a pattern until the pattern ends, then
loses several positions at once. Independence is the wrong model not because the
decay rate is wrong but because the decay is not smooth.

This is the strongest argument for
[`AcceptanceProfile`](../praecise-runtime/src/prax.rs): a measured per-position
curve represents a step function natively, where no closed form with one
parameter can. `DECAY_DAMPING` remains only as the prior used before enough
positions have been observed.

## What this does NOT establish

- **One model, one drafter, one machine.** Qwen3-0.6B with n-gram
  self-speculation on a 5070 Ti. A larger target, a trained drafter (EAGLE,
  DFlash) or a different task mix could all look different.
- **12 draft events is a small sample.** The shape is clear; the exact rates
  are not tightly bounded.
- **Prompts skewed structured.** Half were code, JSON, SQL or literal
  repetition, chosen deliberately because that is where an n-gram drafter earns
  its place -- but it flatters the acceptance figure relative to open prose.
- **No tokens/sec comparison.** This measures acceptance, not speedup. Whether
  a better block-size model produces a faster server is still unmeasured.

## Reproducing it

llama.cpp b10667 (CUDA 12.4 build) serving the model with `--metrics` and
`--spec-type ngram-mod`, greedy decoding, `n_predict 200`.

Bind the server to a port nothing else is using, and put the working directory
on a volume with room -- both sound obvious and both cost a retry here.

Read the curve from `/metrics`:
`llamacpp:spec_decode_num_accepted_tokens_per_pos_total`.

⚠ Fetch it with `System.Net.WebClient`, not `Invoke-WebRequest` -- the latter
throws `NullReferenceException` trying to parse Prometheus text as a DOM.

⚠ GitHub's `/releases/latest` for llama.cpp resolves to a **nightly tag holding
only a text file**. Walk `/releases` and match `^llama-.*bin-win-cuda`; note
`cudart-*` is the CUDA runtime, a different archive, and both are needed.
