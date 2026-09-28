# Prefix reuse in the batch engine

`BatchEngine` serves every running request from one context and a fixed pool of
sequence slots. When a request finishes, its slot keeps what it computed, and
the next request that shares a prefix with it starts where the two diverge
instead of at position zero. A multi-turn chat or an agent loop repeats the
whole conversation on every request, so this is the difference between
prefilling a few hundred tokens and prefilling tens of thousands.

The decisions live in `praecise_runtime::prefix_cache`, which is backend-free
and tested without a model. `batching.rs` acts on them.

## What can be rewound is read from the model

Whether a slot's cache can be cut back to a divergence is a property of the
model's memory, read from llama.cpp at load (`llama_model_is_recurrent`,
`llama_model_is_hybrid`, `llama_model_n_swa`, `llama_model_has_encoder`,
`llama_model_has_decoder`, `llama_model_is_diffusion`, and the GGUF's
`<arch>.attention.causal` and `<arch>.pooling_type`), never from the model's
name. The result is a `MemoryTraits`, and `MemoryTraits::reuse()` a `Reuse`:

| Memory | Examples | How a slot goes back to a shorter prefix |
|---|---|---|
| Attention (`Reuse::Trim`) | dense models; MoE models (experts do not change the cache); sliding-window models whose cache keeps every position | Trim every position past the divergence. |
| Sliding window holding only the window (`Reuse::Window`, `BATCH_SWA_FULL=0`) | Gemma 3 | Trim while the window before the divergence is still held (checked against `llama_memory_seq_pos_min`); otherwise restore a checkpoint of the window. |
| Recurrent state (`Reuse::Recurrent`) | Mamba, RWKV, gated delta nets, and every hybrid of recurrent layers with attention | Restore the deepest checkpoint at or below the divergence, then trim the attention part from there. |
| Encoder-decoder (`Reuse::EncoderOutput`) | T5-style models | The context holds one encoder output; a request whose encoder input and cache namespace match it skips the encoder pass. Such a model decodes one request at a time. |
| Bidirectional or pooled attention, and diffusion decoding (`Reuse::None`) | embedding and reranking GGUFs; masked-diffusion LLMs | Refused by `BatchEngine::spawn` with the reason. The engine generates text a token at a time from a causal cache, which these do not, and a prefix's states depend on what follows it, so nothing could carry over anyway. |

`plan_rewind` turns (kind, cached length, shared prefix, checkpoints held,
lowest live position) into `Keep`, `Trim`, `Restore` or `Reset`, and the engine
verifies each rewind against the cache before it relies on it.

## Where checkpoints go

A checkpoint is a copy of the part of a sequence's memory that cannot be
trimmed: `llama_state_seq_get_data_ext` with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`,
which is the recurrent state or the sliding window. Checkpoints cost memory, so
they go where a following request actually diverges: at the start of a chat
turn.

At load the engine renders a short probe conversation through the model's own
chat template and reads the control token that closes each turn (plus every
end-of-generation token); a turn starts right after one (`TurnEnds`). While
prefilling, a slot's prefill stops at each chosen turn start, so the snapshot
is taken exactly there. `checkpoint_cuts` always keeps the last turn start
(where the reply being generated begins, and where the next request re-renders
that reply, with its reasoning dropped or a stop sequence trimmed) and one
token before the prompt's end (an identical re-send then prefills a single
token), then earlier turn starts, latest first, at least 256 tokens apart.

## How many checkpoints memory affords

Checkpoints are host memory. At load the engine measures one checkpoint and
one token of KV on the live context, reads the host's free memory from the
backend device list, and gives checkpoints half of what is free past a reserve
(a tenth of total, at least 2 GiB), split across slots (`checkpoint_budget`),
at most 32 per slot or `BATCH_PROMPT_CHECKPOINTS` (0 turns them off).

Where the KV cache and checkpoints draw on the same memory (CPU serving,
integrated GPUs, Apple silicon, and superchips whose GPU reports the host's
memory as its own), a context too large to leave every slot its two most useful
checkpoints is recreated smaller until it does, never below a quarter of what
was asked or 8K tokens (`context_to_release`). On a discrete GPU the KV cache
lives in device memory and the two do not compete.

Both readings come from the backend's device list at load, so nothing is keyed
to a particular machine. An operator can override them:
`BATCH_CHECKPOINT_MEMORY=<bytes>` fixes the memory checkpoints may use across
every slot, and `BATCH_UNIFIED_MEMORY=1` (or `0`) says whether that memory is
shared with the KV cache.

## Reasoning and stop sequences

A slot's identity is the prompt as it was prefilled plus the tokens it
generated. The next request usually re-renders that reply differently: the
reasoning dropped by the template, a stop sequence trimmed, or the text
retokenized. The divergence then falls inside the last turn, and the engine
rewinds to it: attention models trim there, recurrent models restore the
last-turn-start checkpoint. `ChatMessage::reasoning_content` lets a template
that renders reasoning into history reproduce the previous turn exactly, so it
stays an exact prefix.

## Media

An image or audio attachment is part of the prompt's identity by content: the
SHA-256 of its bytes is its bitmap id, and each of its positions gets an
identity element derived from that hash (`media_ids`). A later request reuses
through an attachment only when it carries the same bytes at the same place,
and a reuse never ends inside one (`snap_out_of_media`). Multi-axis RoPE gives a
chunk a different number of KV positions than identity positions; `kv_pos`
translates between the two. Media chunks are evaluated one at a time
(`MtmdInputChunks::eval_chunk`), so chunks a slot already holds are never
re-encoded, and the text around them is prefilled in the ordinary batch.

## Cache namespaces

`GenerationConfig::cache_salt` is an opaque isolation key chosen by the caller.
A slot's cache is tagged with the namespace it was filled under (`Namespace`),
and a request only matches slots of its own namespace. When nothing is
reachable, it takes an empty slot before any other namespace's cache, then the
least recently used one, so whether a prompt under another key is resident
changes neither what is reused nor where the request lands, and cannot be read
from time to first token. Unset, requests share one namespace.

## What a result reports

`InferenceResult::cached_tokens` is the number of prompt tokens the request took
from the cache instead of prefilling. A request with `commitment_k` set gets its
top-k logit commitment from the batch engine, one record per generated token,
speculative rows included. The minimum `k` is 4 (`toploc::MIN_COMMITMENT_K`); a smaller request is raised to it, and a verifier refuses a commitment whose `k` is outside 4 to 64, whose steps carry other than `k` logits, or whose emitted token is not in the recomputed top-k at its step. A prompt that ends in a media chunk takes its first
logits from a row the engine cannot read, so such a request returns no
commitment rather than a partial one.

## Streaming never waits on a reader

Each sequence streams through `StopStream::nonblocking()`. A chunk the
receiver's channel cannot take yet is queued for that sequence alone and
retried every step. A receiver more than `STREAM_BACKLOG_LIMIT` (1 MiB) behind
ends its own request with an error, and a finished request's result is
delivered once its stream has been fully taken, or after 60 s. One slow reader
never holds back the scheduler thread.

A request may also set `GenerationConfig::reasoning_tx`: the model's reasoning
is then delivered there as it is produced, kept apart from the answer on
`token_tx`, instead of only in `InferenceResult::thinking` at the end. It is
held and retried the same way, and a reasoning receiver that goes away ends
only the reasoning stream. The speculative and layer-pipeline decode paths honour
the same field.

## A bounded request queue

`BatchEngine::submit` never blocks and never queues without bound. Behind the
running sequences at most one slot's worth of requests may wait (each holds its
whole prompt and any media bytes); past that, `submit` returns
`Error::QueueFull` and the caller's admission decides whether to hold or shed.

## What is billed

`InferenceResult::output_tokens` counts the tokens the model generated. When a
reasoning budget runs out the engine forces the block's close marker into the
output; those tokens appear in the text and in the commitment's steps, but are
not counted as generated.

## Measured

One small GGUF per memory kind, served on CPU with 4 slots and a 4096-token
context, greedy decoding, 12 output tokens. Every request was also sent to a
second engine that had never seen it, and the two answers were compared token
by token through their top-8 logit commitments; a difference is accepted only
as a near tie (the other engine's token within 1.0 logit of the top), which is
the spread two runs of an identical prompt show with nothing reused.
`cached` is the prompt tokens not prefilled, out of the prompt's length.

| Memory kind (model) | identical re-send | exact extension | divergence at a turn boundary | same prompt, other namespace | prompt tokens saved |
|---|---|---|---|---|---|
| dense (Qwen3 0.6B) | 171 / 172 | 183 / 188 | 168 / 192 | 0 / 172 | 690 of 1208 |
| sliding window (Gemma 3 270M) | 164 / 165 | 176 / 182 | 165 / 185 | 0 / 165 | 670 of 1161 |
| hybrid gated delta net (Qwen3.5 0.8B) | 170 / 171 | 182 / 187 | 163 / 191 | 0 / 171 | 678 of 1201 |
| Mamba (130M) | 163 / 164 | 175 / 182 | 163 / 181 | 0 / 164 | 664 of 1153 |
| RWKV-7 (0.1B) | 163 / 164 | 175 / 180 | 163 / 182 | 0 / 164 | 664 of 1150 |
| MoE (Granite 3.1 1B-A400M) | 168 / 169 | 180 / 187 | 169 / 189 | 0 / 169 | 686 of 1190 |
| vision (SmolVLM 256M) | 303 / 304 | 305 / 311 | 304 / 323 | 0 / 304 | 1222 of 2280 |
| hybrid vision (Qwen3.5 0.8B + projector) | 183 / 184 | 185 / 190 | 176 / 204 | 0 / 184 | 720 of 1443 |
| encoder-decoder (Flan-T5 small) | 185 / 185 | 0 / 191 | 0 / 200 | 0 / 185 | 185 of 1251 |
| embedding (EmbeddingGemma 300M) | refused at spawn | | | | n/a |

On the encoder-decoder model what carries over is the encoder output itself:
an identical input skips the encoder pass entirely (so it reuses the whole
prompt, not all but one token), and any other input, an extension included,
reuses nothing, because a bidirectional encoder's output for a prefix depends
on what follows it. Another namespace sending the same input re-encodes.

The same image replaced by a different one reused exactly the text before it
on the attention model (155 of 304) and nothing on the recurrent one, which
keeps no checkpoint below the image. Two conversations served together
answered as each did alone on every model. A build whose trim reports success
without removing the stale positions fails the same test on its first
divergent request.
