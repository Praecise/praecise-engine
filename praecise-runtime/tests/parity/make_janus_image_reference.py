"""Reference for the Janus-Pro image path, from the checkpoint's own transformers model.

Writes into OUT (default tests/parity/janus_image):
  expected.json   prompt and unconditioned token ids, guidance scale, the image tokens
                  chosen greedily under guidance, the checkpoint revision and the steps
                  whose guided logits are kept
  logits.f32      guided logits of the kept steps, row-major [steps, codebook]
  pixels.f32      the VQ decoder's output for those tokens, planar [3, side, side] in [-1, 1]

The loop below is the model's own image generate() written out step by step so the
guided logits can be kept; the tokens it chooses are checked against generate() itself.

Progress goes to stdout with elapsed time and resident memory. A watchdog dumps every thread's
stack when no progress is reported for --stall-dump seconds, and ends the run when resident
memory passes --max-rss-gb, so a stalled run shows where it stopped instead of going quiet.

usage: python make_janus_image_reference.py MODEL_DIR [OUT] [--prompt TEXT] [--revision SHA]
"""

import argparse
import faulthandler
import json
import os
import sys
import threading
import time

import numpy as np
import torch
from transformers import JanusForConditionalGeneration, JanusProcessor, LogitsProcessor, LogitsProcessorList

KEPT_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 64, 128, 255, 383, 511, 575]

T0 = time.monotonic()
_last = [T0]


def rss_gb():
    try:
        import psutil
    except ImportError:
        return float("nan")
    return psutil.Process().memory_info().rss / 2**30


def progress(msg):
    _last[0] = time.monotonic()
    print(f"[{_last[0] - T0:8.1f}s rss {rss_gb():5.1f} GB] {msg}", flush=True)


def watchdog(stall_dump, max_rss_gb):
    dumped_at = 0.0
    while True:
        time.sleep(15)
        now = time.monotonic()
        if rss_gb() > max_rss_gb:
            print(f"resident memory {rss_gb():.1f} GB is over {max_rss_gb} GB, stopping", flush=True)
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
            sys.stdout.flush()
            os._exit(3)
        if now - _last[0] > stall_dump and now - dumped_at > stall_dump:
            print(f"no progress for {now - _last[0]:.0f}s, stacks:", flush=True)
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
            sys.stdout.flush()
            dumped_at = now


class StepProgress(LogitsProcessor):
    """Passes scores through unchanged; reports every 32nd generate() step."""

    def __init__(self, n_tok):
        self.n_tok = n_tok
        self.step = 0

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.step % 32 == 1 or self.step == self.n_tok:
            progress(f"generate() step {self.step}/{self.n_tok}")
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out", nargs="?", default=os.path.join(os.path.dirname(__file__), "janus_image"))
    ap.add_argument("--prompt", default="A red fox sitting in fresh snow, watercolor.")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--revision", default="")
    ap.add_argument("--skip-generate-check", action="store_true")
    ap.add_argument("--stall-dump", type=float, default=600.0)
    ap.add_argument("--max-rss-gb", type=float, default=20.0)
    a = ap.parse_args()
    threading.Thread(target=watchdog, args=(a.stall_dump, a.max_rss_gb), daemon=True).start()
    progress(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    processor = JanusProcessor.from_pretrained(a.model)
    model = JanusForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).eval()
    progress("model loaded")
    gc = model.generation_config
    messages = [{"role": "user", "content": [{"type": "text", "text": a.prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # A bare string is taken as a batch of characters; the prompt goes in as a one-item batch.
    inputs = processor(text=[prompt], generation_mode="image", return_tensors="pt")
    cond = inputs["input_ids"][0]

    bos = gc.bos_token_id
    boi = gc.generation_kwargs["boi_token_id"]
    pad = gc.pad_token_id
    uncond = cond.clone()
    uncond[(uncond != bos) & (uncond != boi)] = pad
    ids = torch.stack([cond, uncond])

    n_tok = model.config.vq_config.num_patches ** 2
    lm = model.model.language_model
    embeds = model.get_input_embeddings()(ids)
    out = lm(inputs_embeds=embeds, use_cache=True)
    past = out.past_key_values
    hidden = out.last_hidden_state[:, -1, :]
    progress(f"prefill done, {ids.shape[1]} prompt tokens, {n_tok} image tokens to go")
    tokens = []
    kept = []
    for step in range(n_tok):
        logits = model.model.generation_head(hidden)
        c, u = logits[0], logits[1]
        mixed = u + a.guidance * (c - u)
        if step in KEPT_STEPS:
            kept.append(mixed.float().numpy())
        tok = int(torch.argmax(mixed))
        tokens.append(tok)
        if step % 32 == 0 or step + 1 == n_tok:
            progress(f"step {step + 1}/{n_tok}")
        if step + 1 == n_tok:
            break
        t = torch.tensor([[tok], [tok]])
        emb = model.prepare_embeddings_for_image_generation(t)
        out = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

    if not a.skip_generate_check:
        progress("generate() cross-check")
        gen = model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([StepProgress(n_tok)]),
            # generate() sizes its static KV cache from max_length; left at the default it is
            # allocated for the full context and runs the host out of memory.
            max_length=ids.shape[1] + n_tok,
            generation_mode="image",
            do_sample=False,
            guidance_scale=a.guidance,
            num_return_sequences=1,
        )
        ref = gen[0].tolist()[-n_tok:]
        same = sum(int(x == y) for x, y in zip(ref, tokens))
        progress(f"generate() agrees on {same}/{n_tok} tokens")
        if ref != tokens:
            raise SystemExit("the step loop differs from generate()")

    grid = torch.tensor(tokens).unsqueeze(0)
    pixels = model.model.vqmodel.decode(grid)[0].float().numpy()
    progress("VQ decode done")

    os.makedirs(a.out, exist_ok=True)
    np.stack(kept).astype("<f4").tofile(os.path.join(a.out, "logits.f32"))
    pixels.astype("<f4").tofile(os.path.join(a.out, "pixels.f32"))
    with open(os.path.join(a.out, "expected.json"), "w") as f:
        json.dump(
            {
                "revision": a.revision,
                "prompt": a.prompt,
                "guidance_scale": a.guidance,
                "cond": cond.tolist(),
                "uncond": uncond.tolist(),
                "tokens": tokens,
                "kept_steps": KEPT_STEPS,
                "codebook": int(model.config.vq_config.num_embeddings),
                "side": int(pixels.shape[-1]),
            },
            f,
        )
    print("tokens", tokens[:16], "pixels", pixels.shape, float(pixels.min()), float(pixels.max()))


if __name__ == "__main__":
    main()
