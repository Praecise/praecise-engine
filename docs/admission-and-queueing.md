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

`admission::tests` uses the measured spark rates. The case that matters is
`a_long_generation_does_not_poison_the_next_short_one`: after observing one
1,000-token answer, a 200-token request must still be predicted under 8 s.
