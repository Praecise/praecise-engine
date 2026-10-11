"""Record the reference decoder's answers for the tiny diffusion fixtures.

Runs `llama-diffusion-cli` (the backend's own example decoder) on each
fixture with the plan the engine uses for that architecture, greedy, and
writes `expected.json` next to the fixtures for `tests/denoise_engine.rs`.
The fixtures' tokenizer is one token per byte, so a prompt's length is its
byte count and the reference's canvas (`-ub`) is the prompt plus the answer.

    python denoise_reference.py <llama-diffusion-cli> <fixture dir>
"""

import json
import re
import subprocess
import sys
from pathlib import Path

CLI = sys.argv[1]
DIR = Path(sys.argv[2])

# (prompt, canvas) pairs. Block plans need the canvas to be a whole number
# of blocks; the answer is whatever the prompt leaves of it.
CASES = [
    ("The lighthouse keeper", 64),
    ("ab", 96),
    ("A longer prompt that fills most of the first block", 128),
]


def plan(arch: str, canvas: int, answer: int) -> list[str]:
    if arch in ("dream", "rnd1"):
        # Whole-answer canvas, lowest entropy first, one step per position.
        return ["--diffusion-eps", "0.001", "--diffusion-algorithm", "1", "--diffusion-steps", str(answer)]
    # Blocks of 32 from the end of the prompt, one position per step, surest first.
    return ["--diffusion-block-length", "32", "--diffusion-algorithm", "4", "--diffusion-steps", str(32 * (canvas // 32))]


out = []
for arch in ("dream", "llada", "llada-moe", "rnd1"):
    for prompt, canvas in CASES:
        answer = canvas - len(prompt.encode())
        cmd = [
            CLI, "-m", str(DIR / f"{arch}-tiny.gguf"), "-p", prompt,
            "-ub", str(canvas), "-b", str(canvas), "-c", str(canvas), "-t", "4",
            "--top-k", "1", "--temp", "0", "--seed", "1", "-ngl", "0",
        ] + plan(arch, canvas, answer)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"{arch} {prompt!r}: {r.stderr[-2000:]}")
        log = r.stderr + r.stdout
        # The answer is the log entry after the timing line: a timestamped
        # entry whose message is a newline, the text, and a newline.
        tail = log.split("time per step:", 1)[1].split("\n", 1)[1]
        text = re.sub(r"^[0-9.]+ I \n", "", tail)
        text = text[:-1] if text.endswith("\n") else text
        out.append({"arch": arch, "prompt": prompt, "max_tokens": answer, "text": text})
        print(f"{arch} {prompt!r}: {text!r}")

(DIR / "expected.json").write_text(json.dumps(out, indent=1))
