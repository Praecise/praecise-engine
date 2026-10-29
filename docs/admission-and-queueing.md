# Admission and queueing: why a 30-second deadline broke a 30 tok/s decoder

**A serving layer that admits on predicted end-to-end time, against a fixed
deadline, from a cost model that never saw the output length, will refuse
almost everything on a slow decoder — and it will do so more, not less, the
longer it runs.** Observed on a DGX Spark serving Qwen3.8-Flash-Next
(UD-Q2_K_XL, 31.6 tok/s decode, 730 tok/s prefill) on 2026-09-02. Fixed by
`praecise_runtime::admission`.

## What was observed

A 200-token request, prompt "Reply with exactly: OK", was refused:

```
refused up front: predicted 48779 ms exceeds the 29999 ms deadline,
so admitting it would miss and cost the requests already running
reason: deadline-unattainable
```

Other clients on the same node saw the same, routed around it to a hosted
model, and reported the local model as "burning its budget on hidden
reasoning". It was not reasoning. It was not admitted.

## The three defects

The host's admission was a reasonable design — admit on predicted deadline,
order least-deadline-first, refuse what will miss rather than let it degrade
everything else — fed by the wrong inputs.

**1. Output tokens were not in the cost.** The estimate was `ms/kilotoken ×
prompt_tokens`, with prompt tokens approximated as bytes/4 and, on the
OpenAI-compatible path, as a flat 1,000. On a decoder, cost is almost entirely
output: a 20-token prompt generating 1,000 tokens is 32 s of work. Charged as
20 tokens, that one request taught the model "128 s per kilotoken".

**2. The deadline was fixed while the work was not.** Thirty seconds
end-to-end at 31 tok/s is ~900 output tokens. Every generation longer than
that — every real coding turn — was predicted to miss and refused. A caller
that asks for 4,096 tokens has asked for a two-minute answer; refusing because
thirty is a round number protects no one.

**3. Refusal where there should have been a queue.** The batching engine
underneath holds a channel and admits into slots as they free. Turning callers
away above it with a `retry_after_ms` they cannot act on discarded the queue
that already existed. The only exemption was "admit if idle" — so the node
served one request at a time and refused the rest.

The comment in that code named the flaw itself: *"elapsed time measures
sojourn while the prediction uses it as service time; until those are
measured separately…"*. They are now.

## The model that replaces it

```
service_ms = prompt_tokens × prefill_rate + max_output_tokens × decode_rate × concurrency
```

- **Two rates, learned separately.** Prefill from time-to-first-token when the
  host reports it; decode from the remainder per generated token, normalised
  to a solo-equivalent by the concurrency it ran under (at c=2 on a
  bandwidth-bound decoder each request sees roughly half the solo rate, and
  the 52.5 vs 31.6 tok/s aggregate measured on this box bears that out).
- **The SLO is on the wait, not the answer.** Interactive: 30 s in the queue
  before the first token. Batch: 600 s. Generation runs for as long as
  `max_tokens` allows. This is the convention of every production serving
  stack (TTFT and queue timeout; never a cap on total generation) and the
  only one under which long completions exist on a slow decoder.
- **ETA before refusal.** With every slot busy, the answer is how long until
  one frees, computed from the remaining work of what is running (a request
  that overran its estimate counts as freeing now — the estimate was wrong,
  not the request). The host holds the caller and asks again as things
  finish. Only an ETA past the class budget, or a queue at its depth limit,
  is refused.

A free slot always admits, whatever the estimate says. The caller chose the
length; a free slot is capacity; there is nothing to protect.

## What it means for the numbers

On the same hardware, same model, the 30-second interactive budget now buys a
queue wait rather than a generation cap. What decode speed still governs is
*how long the answer takes* — and that is where the rest of this repository's
work (prefix reuse across agent turns, n-gram self-speculation) applies: at
~94 tok/s warm under concurrency 8 (sxuff's llama-server measurement for this
model on one Spark) the same answer that took 32 s takes ~11 s.

## Tests

`admission::tests` uses the measured rates above. The case that matters is
`a_long_generation_does_not_poison_the_next_short_one`: after observing one
1,000-token answer, a 200-token request must still be predicted under 8 s.

## Choosing concurrency: a threshold, a validity gate, a pinned result

**The transferable part of Nebius's Cosmos 3 Super study is its method, not
its numbers.** Count only output that passes a validity gate, report it per
hour of wall clock, choose concurrency by a marginal-gain threshold fixed in
advance, and pin the exact configuration beside every figure. The topology
result (eight single-GPU replicas beat one eight-GPU replica) is about
splitting one diffusion forward pass across GPUs and has no counterpart on a
single GB10. The method is implemented in
`praecise_runtime::concurrency_sweep`, and is wired as an opt-in into the
existing `praecise-runtime/tests/batch_throughput.rs` rather than a second tool.

Source: Nebius, "Serving NVIDIA Cosmos 3 Super: Request Latency and Node
Throughput", 2026-10-26,
<https://nebius.com/blog/posts/serving-world-models-latency-node-throughput>.
Every figure quoted from it below was read from the post on 2026-10-29. None
was reproduced here: no GPU benchmark was run for this note.

### What the study measured

Cosmos 3 Super text-to-video (`nvidia/Cosmos3-Super`, one pinned revision) in
`vllm/vllm-omni:cosmos3` (vLLM 0.25.0, pinned by digest), BF16, 1280x720,
189 frames at 24 fps, 35 denoising steps, guardrails off, one fixed prompt, three
fixed seeds, 24 measured requests per topology, one warmup request per
replica, on a single 8-GPU HGX B200 node and a single HGX H200 node.

| Topology | B200 mean latency (s) | B200 valid video-s/node-h | H200 mean latency (s) | H200 valid video-s/node-h |
|---|---|---|---|---|
| 1 replica x 8 GPUs | 68.5 | 413.6 | 123.3 | 229.9 |
| 2 x 4 | 121.5 | 466.3 | 230.0 | 245.6 |
| 4 x 2 | 212.4 | 529.0 | 416.5 | 271.4 |
| 8 x 1 | 378.1 | 589.4 | 779.0 | 289.1 |

One 8-GPU replica to eight single-GPU replicas: +42.5% throughput on B200 and
+25.7% on H200, at 5.5x and 6.3x the mean latency. A second request in flight
per replica added 0.4 to 3.2% throughput (below the authors' 5% bar, set in
advance) and 32.9 to 48.8% latency. All 408 recorded attempts passed the gate,
so the gate changed no reported number. The post states its limits: one prompt,
one shape, one software stack; the gate checks file integrity and basic motion,
not prompt adherence or quality; B200 and H200 nodes differ in more than the
GPU.

### What transfers

**1. Rate-versus-latency is a curve with a knee, and the knee is found by
measuring.** Both a video replica and a decode slot trade per-request latency
for aggregate output. The shape of the trade differs (next section), but the
procedure is the same: sweep, then stop where the next step stops paying.
`concurrency_sweep::run_sweep` runs the levels in ascending order and records
mean, p50 and p95 latency beside the rate at each.

**2. A marginal-gain threshold, fixed before the run.** The study's rule was
"under 5% throughput gain is not worth the latency". `choose` applies it as a
stopping rule: take the highest level reached by consecutive steps that each
gain at least the threshold (default 5%, `DEFAULT_GAIN_THRESHOLD`), and stop at
the first step that does not, even if a noisier later level looks better. An
optional latency budget caps the choice, which is the study's advice for
intermediate latency budgets. The threshold is a parameter, not a finding: 5%
is the authors' choice for their fleet and is the default here only for
comparability.

**3. A validity gate before output counts.** The study counted a clip only if
it decoded fully, had the expected shape and was neither blank nor frozen;
rejected attempts added no output but stayed in the wall clock. For an LLM
the analogues are a request that completed, produced at least a minimum number
of tokens, did not collapse to one repeated token, and did not decode to
replacement characters. `LlmGate` implements those four; any other `Gate` can
be passed. This is an integrity gate, not a quality gate: a fluent wrong
answer passes, exactly as the study's gate does not judge realism. Where the
engine already has a correctness check for a path (speculative decode
acceptance, parity tests in `praecise-diffusion/tests/parity`), that check is
the better gate for that path.

**4. Valid output per hour, with the clock including failures.** `units /
wall_s * 3600`, with `wall_s` from first dispatch to last finish. For a decode
workload `units` is generated tokens. A concurrency level that wins on
tokens per second but loses requests (timeouts, out-of-memory, a slot reset)
is penalised automatically.

**5. Pin the configuration with the number.** The study pinned the model
revision, the container by digest, the engine version, the precision, the
shape, the step count and the seeds. `RunPin` requires `model_digest`,
`build`, `shape` and `hardware`, refuses to produce a report without them, and
hashes the whole pin (`pin_digest`) so two reports can be compared by
digest. For this engine the pin should carry at least: the GGUF SHA-256, the
praecise-engine commit and the llama.cpp submodule commit, the CUDA build and
the compiled architectures, the speculative method and block length, context
size and slot count (`BATCH_MAX_SLOTS`), prompt and output lengths, seed or
temperature, and the GPU memory other tenants held during the run.

**6. Benchmark your own workload before committing.** The authors' last
advice, and the repository's own history agrees: the 30-second deadline in
[admission-and-queueing.md](admission-and-queueing.md) came from a cost model
that never saw the real output length.

### What does not transfer

**Replica topology.** The 1x8 versus 8x1 axis is tensor, CFG and Ulysses
sequence parallelism against independent replicas. A GB10 is one device with
one memory pool. There is no replica count to choose; the nearest axis is the
number of engine instances sharing the pool, which the shared `Budget` in
`praecise-host` already forbids duplicating for one model.

**The shape of the concurrency curve.** In the study one request in flight per
replica was the best operating point, because a diffusion replica runs one
generation at a time and a second request only queues (the post says vLLM-Omni
gave no real parallelism at two in flight). Slots in a batching decoder are
not queued requests: they share one pass over the weights. Decode on GB10 is
bandwidth-bound at about 80% of the 273 GB/s peak
([gb10-bandwidth-ceiling.md](gb10-bandwidth-ceiling.md)), so extra sequences in
the same step add little read traffic and aggregate throughput does rise with
slots. The repository already records two points of that curve for
Qwen3.8-Flash-Next UD-Q2_K_XL on one Spark: 31.6 tok/s solo against 52.5
tok/s aggregate at two in flight
([admission-and-queueing.md](admission-and-queueing.md)), and about 94 tok/s
warm at concurrency 8 (a llama-server measurement recorded in that note, not
reproduced here). So the study's "one in flight is enough" does not carry over; what
carries over is that the gain eventually falls under the threshold and the
latency price keeps rising. Where it falls on GB10 for a given model is
unmeasured here, and it depends on KV size per slot, context length and
speculation (a drafted block per slot already raises the work per weight
read).

**Latency metrics.** The study uses time to a complete file because a
synchronous video endpoint returns nothing before it. An LLM has time to first
token and inter-token latency, and `Attempt.latency_s` is time to completion
only. A sweep that feeds an interactive service should also record time to
first token; that is a gap in this first version (see below).

**Unified memory and other tenants.** The study had a whole node reserved. A
GB10 shares one LPDDR pool, and bandwidth, with whatever else is running, so a
sweep result is valid only for the load present while it ran. The pin must say
what else held the device, and results measured next to a loaded neighbour
should not be compared with results from an idle box. On the shared Spark any
real run must go through the host's rules (capped build wrapper, the free
memory floor, the exclusive GPU lock); `SweepBackend::admit` is the hook where
an adapter enforces them, and a refusal stops the sweep at that level.

**The 8-GPU parallelism vocabulary** (CFG parallelism, Ulysses-SP, HSDP) has
no equivalent in the llama.cpp backend on one device.

### What exists in the repository already

Inventory taken before building anything (2026-10-29, `main` at 9e72e77):

- `praecise-runtime/tests/batch_throughput.rs`: an ignored test that runs the
  batch engine at several concurrencies and prints aggregate tok/s. It has no
  threshold, no validity gate, no latency, and asserts only that tokens were
  produced. It now has the opt-in sweep report described below; that path compiles
  but has not been run, because it needs a model and the shared device.
- `praecise_runtime::admission`: a two-term cost model and queue-wait SLO. It
  decides admission at run time; it does not choose a slot count.
- `BATCH_MAX_SLOTS` (default 32, `batching.rs`): the slot ceiling, reduced when
  whole contexts do not fit. Nothing chose the default by sweep.
- `praecise_host::Engines` and `Budget`: one engine per model, one memory
  ledger.
- `praecise-diffusion`: see the next section.

Added, opt-in, without changing any default: `praecise-runtime/src/concurrency_sweep.rs`
(13 CPU unit tests with a fake backend whose rate saturates at a chosen slot
count) and a `PRAECISE_TEST_THROUGHPUT_SWEEP=1` mode in `batch_throughput.rs`
that scores its existing levels through the gate and prints the report.

### Known gaps in this first version

- No time-to-first-token or inter-token latency; completion latency only.
- The `batch_throughput.rs` sweep mode is compiled but unrun (no GPU benchmark
  was run for this change); its latency is the engine's generation time, which
  excludes queue wait. On the shared Spark it must follow the host rules above
  and stay small.
- Repeats and variance: one run per level, no confidence interval. The study
  reports repeat runs agreeing within 2.65% on its own harness; this version
  gives no such bound, so a gain close to the threshold should be repeated
  before it decides anything.
- The gate is an integrity gate. Quality needs a separate evaluation.

### Tests

`cargo test -p praecise-runtime --lib concurrency_sweep` (13 tests, no model,
no GPU): the knee is found at the saturation slot count; a linear curve takes
every step; a gain of 4.9% is refused and 5.1% taken; the walk does not
resume after a failed step; the latency budget caps the choice; rejected
output adds no rate but keeps its wall time; the rate formula; each gate
check; an incomplete pin is refused; the pin digest is order independent and
value sensitive; bad plans are refused before any run; a host refusal stops
the sweep; the report round-trips through JSON. The curve in the tests is
synthetic and says nothing about GB10.
