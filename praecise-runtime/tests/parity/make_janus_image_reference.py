"""Reference for the Janus-Pro image path, from the checkpoint's own transformers model.

Writes into OUT (default tests/parity/janus_image):
  expected.json   prompt and unconditioned token ids, guidance scale, the image tokens
                  chosen greedily under guidance, the checkpoint revision and the steps
                  whose guided logits are kept
  logits.f32      guided logits of the kept steps, row-major [steps, codebook]
  pixels.f32      the VQ decoder's output for those tokens, planar [3, side, side] in [-1, 1]

The loop below is the model's own image generate() written out step by step so the
guided logits can be kept; the tokens it chooses are checked against generate() itself.

usage: python make_janus_image_reference.py MODEL_DIR [OUT] [--prompt TEXT] [--revision SHA]
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import JanusForConditionalGeneration, JanusProcessor

KEPT_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 64, 128, 255, 383, 511, 575]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out", nargs="?", default=os.path.join(os.path.dirname(__file__), "janus_image"))
    ap.add_argument("--prompt", default="A red fox sitting in fresh snow, watercolor.")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--revision", default="")
    ap.add_argument("--skip-generate-check", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    processor = JanusProcessor.from_pretrained(a.model)
    model = JanusForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).eval()
    gc = model.generation_config
    messages = [{"role": "user", "content": [{"type": "text", "text": a.prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, generation_mode="image", return_tensors="pt")
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
        if step + 1 == n_tok:
            break
        t = torch.tensor([[tok], [tok]])
        emb = model.prepare_embeddings_for_image_generation(t)
        out = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]

    if not a.skip_generate_check:
        gen = model.generate(
            **inputs,
            generation_mode="image",
            do_sample=False,
            guidance_scale=a.guidance,
            num_return_sequences=1,
        )
        ref = gen[0].tolist()[-n_tok:]
        same = sum(int(x == y) for x, y in zip(ref, tokens))
        print(f"generate() agrees on {same}/{n_tok} tokens")
        if ref != tokens:
            raise SystemExit("the step loop differs from generate()")

    grid = torch.tensor(tokens).unsqueeze(0)
    pixels = model.model.vqmodel.decode(grid)[0].float().numpy()

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
