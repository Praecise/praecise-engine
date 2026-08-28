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

**Do not commit the local copy to the fork.** Take it from a newer upstream
instead, so upstream's authorship travels with the code.

qwen4exp lands between build tags **b10658** (absent) and **b10660** (present):

| tag | has `src/models/qwen4exp.cpp` |
|---|---|
| b10650, b10652, b10654, b10656, b10658 | no |
| **b10660**, b10666, b10667 | **yes** |

⚠ **A rebase does not work here, despite being the obvious answer.**
`hilarl/llama.cpp` shares **no history with upstream** — `git merge-base`
against b10660 returns nothing. The repo has two commits, and the older one is a
**squashed import with no parent**: 3,396 files and 1,435,194 insertions in a
single root commit. There is nothing to rebase onto.

So the options are:

**A. Re-import from b10660+.** Take upstream at that tag, re-apply the local
changes listed in `MODIFICATIONS.md`, commit under the correct identity. Same
shape as today, current, and qwen4exp arrives with it.

**B. Re-fork properly.** Clone upstream with its real history and put the local
commits on top, so `git log` shows upstream's authorship rather than only the
tree doing so. More work, and it moves the submodule pin.

## On the licence, which is the thing that actually matters

**The current arrangement is not a violation.** The import preserves upstream's
`LICENSE`, `AUTHORS` and `README.md`, and `MODIFICATIONS.md` states it directly:

> "This is **not** stock upstream llama.cpp... Upstream:
> https://github.com/ggml-org/llama.cpp — Copyright (c) 2023-2026 The ggml
> authors, MIT License. Those terms continue to govern this directory... The
> Apache-2.0 licence applied to Praecise Engine's own crates does not relicense
> this code."

MIT asks that the copyright notice travel with the code, and it does. The commit
*metadata* is a poor record of provenance — a squashed import authored locally
reads as though the code were written here — but the attribution a reader needs
is present in the tree.

## What the fork legitimately owns

One of its two commits, and it is real work:

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
