# qwen4exp: provenance, and how to get it properly

**qwen4exp is upstream llama.cpp's work, not ours.** This note exists because a
modified copy of it sat uncommitted in the vendored fork, and committing that to
`hilarl/llama.cpp` would have published someone else's architecture
implementation under our name with no attribution.

## Who wrote it

```
commit  6c84c7d5d8833c6e0df69628f75a0f599797934e
author  Daniel Han
date    2026-08-27T19:32:31Z
title   model: add Qwen3.8-Flash-Next (qwen4exp) (#27742)
repo    ggml-org/llama.cpp
```

It is the only commit touching `src/models/qwen4exp.cpp`, so the whole
implementation -- the architecture, the hybrid memory index, the conversion
script -- arrived in that one PR. Merged upstream the day before this work
started.

## How to tell, in two minutes

The check that settles it, and that should have been run first:

```sh
curl -o /dev/null -w '%{http_code}\n' \
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen4exp.cpp
```

`200` means upstream has it. A local copy of a file upstream already ships is
**vendoring**, not authorship, however much local editing it has accumulated.

The local copy was 1,199 lines against upstream's 1,199 -- identical length,
differing by ~440 lines of reordering and local edits. Same file, modified.

## The right way to get it

**Do not commit the local copy to the fork.** Move the fork's base forward to an
upstream commit that already contains qwen4exp, and let upstream's authorship
travel with it.

qwen4exp lands between build tags **b10658** (absent) and **b10660** (present):

| tag | has `src/models/qwen4exp.cpp` |
|---|---|
| b10650, b10652, b10654, b10656, b10658 | no |
| **b10660**, b10666, b10667 | **yes** |

So: rebase the fork onto **b10660 or later**, keep the fork's own commits on
top, and bump the submodule pointer in this repo. The engine gets qwen4exp, and
the commit history says who wrote what.

## What the fork legitimately owns

One commit, and it is real work:

```
dddf1d9  Halve the MMVQ warps per block on Blackwell
```

A measured decode-kernel tuning -- two warps per block instead of four, so more
blocks stay resident on an SM and more memory requests are in flight to hide
each other's latency. Decode is bandwidth-bound, so latency hiding is what
helps. That belongs to us and should survive the rebase.

⚠ It is authored `Tenzro Eng <eng@tenzro.com>`, which is identity drift -- the
convention is `Hilal Agil <hilaal@gmail.com>` for both author and committer.
Worth fixing while rebasing, since the history is being rewritten anyway.

## The general rule

Before committing vendored code to a fork under our name, check whether upstream
already ships it. A fork exists to carry *our* changes on top of theirs, not to
re-author theirs. When the answer is "upstream has this", the correct action is
always to move the base forward rather than to commit the copy.
