"""A full-size FLUX 3 Image Klein transformer with random weights, for timing and memory only.

The release weights are gated. Compute and memory of a 5.6B transformer do not depend on the weight values, so this
writes `model.safetensors` in the release's tensor names and shapes (BFL `flux3/dit.py`: hidden 3072, 24 heads, 5 blocks
per mode, 28 single-stream blocks, 128-channel latent tokens, 20480-wide text context) filled with small random
values. Images from it are noise; it says nothing about quality. Pair it with the public Qwen3-VL-4B-Instruct as
`text_encoder/` and the FLUX.2 [klein] 4B `vae/` (the same autoencoder) to run `examples/flux3-image`.

Usage: python make_flux3_image_random_release.py <out dir> [--base]
  --base  no `ref_time_in` (the undistilled, guided model); default is the distilled flow map.
Needs numpy only; streams one tensor at a time (about 11 GB written).
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

HIDDEN, HEADS, DEPTH, DEPTH_SINGLE, MLP_RATIO = 3072, 24, 5, 28, 3
IN = {"image": 128, "image_cond": 128, "text": 20480}
HEAD_DIM = HIDDEN // HEADS
MLP = HIDDEN * MLP_RATIO


def block(prefix):
    return {
        f"{prefix}.q_proj.weight": (HIDDEN, HIDDEN),
        f"{prefix}.k_proj.weight": (HIDDEN, HIDDEN),
        f"{prefix}.v_proj.weight": (HIDDEN, HIDDEN),
        f"{prefix}.attn_out.weight": (HIDDEN, HIDDEN),
        f"{prefix}.mlp_in.weight": (2 * MLP, HIDDEN),
        f"{prefix}.mlp_out.weight": (HIDDEN, MLP),
        f"{prefix}.norm.query_norm.scale": (HEAD_DIM,),
        f"{prefix}.norm.key_norm.scale": (HEAD_DIM,),
    }


def shapes(flow_map):
    s = {
        "emb_in.image.weight": (HIDDEN, IN["image"]),
        "emb_in.image_cond.weight": (HIDDEN, IN["image_cond"]),
        "txt_in.weight": (HIDDEN, IN["text"]),
        "time_in.in_layer.weight": (HIDDEN, 256),
        "time_in.out_layer.weight": (HIDDEN, HIDDEN),
        "final_layer.image.linear.weight": (IN["image"], HIDDEN),
        "final_layer.image.adaLN_modulation.1.weight": (2 * HIDDEN, HIDDEN),
    }
    if flow_map:
        s["ref_time_in.in_layer.weight"] = (HIDDEN, 256)
        s["ref_time_in.out_layer.weight"] = (HIDDEN, HIDDEN)
    for kind in ("early", "single"):
        for m in ("image", "image_cond", "txt"):
            s[f"{kind}_stream_modulations.{m}.lin.weight"] = (3 * HIDDEN, HIDDEN)
    for m in ("image", "image_cond"):
        for i in range(DEPTH):
            s.update(block(f"content_mode_blocks.{m}.{i}"))
    for i in range(DEPTH):
        s.update(block(f"txt_mode_blocks.{i}"))
    for i in range(DEPTH_SINGLE):
        s.update(block(f"single_blocks.{i}"))
    return s


def bf16(a):
    return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype("<u2")


def main():
    out = Path(sys.argv[1])
    flow_map = "--base" not in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    s = shapes(flow_map)
    header, offset = {}, 0
    for name, shape in s.items():
        n = int(np.prod(shape)) * 2
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    rng = np.random.default_rng(1)
    with open(out / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name, shape in s.items():
            if name.endswith("_norm.scale") or name.endswith("norm.scale"):
                a = np.ones(shape, dtype=np.float32)
            else:
                a = rng.standard_normal(shape, dtype=np.float32) * (0.5 / np.sqrt(shape[-1]))
            f.write(bf16(a).tobytes())
    total = sum(int(np.prod(v)) for v in s.values())
    print(f"{out / 'model.safetensors'}: {total / 1e9:.2f}B parameters, {offset / 2**30:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
