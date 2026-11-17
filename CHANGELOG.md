# Changelog

## Unreleased (tenzro-0.2.2)

### Changed, read before upgrading

- `StopReason` is `#[non_exhaustive]` and gains `Preempted`. A match on it
  outside this crate needs a wildcard arm.
- `BatchLimits` is `#[non_exhaustive]`: build it from `BatchLimits::default()`
  or `BatchLimits::from_env()` and the `with_*` methods.
- C ABI: `PraeciseStopReason::Preempted = 3`. Values 0 to 2 are unchanged; a
  stop reason added later reads as `Eos` (0) until the ABI names it.
- KV blobs are format version 2: a blob written from a context with a KV
  stream per sequence (every batch engine) carries the whole sequence. Version
  1 blobs are still read; a version 1 reader refuses version 2.
- `BatchEngine::submit_tracked` works on any batch engine; exporting a ticket
  still needs one spawned with migration.
- The minimum versions of `cc` (1.10.0) and `encoding_rs` (0.8.42) are those
  of the previous release again; `Cargo.lock` still selects newer releases.

### Added

- Session checkpoints from the batch engine: a sequence exports from a KV
  stream of its own to a context of any stream count; a pending export is
  answered when the first sampled token ends the request, and an export of a
  queued ticket waits. A whole-sequence import compares cache layouts first,
  and a refused import leaves the destination as it was.
- `BatchEngine::spawn_with_limits` and `BatchLimits`: slot count, physical
  batch and checkpoint budget per engine.
- `BatchEngine::score_continuation`: the top-k rows of a fixed continuation in
  one prefill; `toploc::Tolerance` with `compare_step_with` and
  `verify_commitment_with`.
- `BatchEngine::yield_sequence`: the request ends at the next token boundary
  and returns what it has with `StopReason::Preempted`, its commitment covering
  exactly those tokens. A request yielded while still waiting for a slot or
  inside its prompt returns an empty result: no text, no tokens.
- Encoder-decoder models run on the batch engine with one slot.
- A chat template a model names (`chatml`, `llama3`, ...) renders through
  llama.cpp's built-in path.
