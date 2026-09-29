"""Render with the reference implementation from the starting noise the
`flux2` example wrote, and report PSNR against the example's image.

Usage: python compare_reference.py <checkpoint dir> <noise.bin> <ours.ppm> <prompt> [width height steps]
"""

import sys

import numpy as np
import torch
from diffusers import Flux2KleinPipeline


def read_ppm(path):
    parts = open(path, "rb").read().split(b"\n", 3)
    w, h = (int(v) for v in parts[1].split())
    return np.frombuffer(parts[3], dtype=np.uint8).reshape(h, w, 3)


def main():
    model, noise_path, ours_path, prompt = sys.argv[1:5]
    width, height, steps = (int(v) for v in (sys.argv[5:8] + ["1024", "1024", "4"][len(sys.argv[5:8]):]))
    gh, gw = height // 16, width // 16
    noise = np.fromfile(noise_path, "<f4").reshape(gh * gw, 128).T.reshape(1, 128, gh, gw)
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    ref = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        latents=torch.from_numpy(noise).to("cuda", torch.bfloat16),
        output_type="np",
    ).images[0]
    ours = read_ppm(ours_path).astype(np.float32) / 255.0
    mse = float(np.mean((ours - ref) ** 2))
    print(f"PSNR vs reference: {10 * np.log10(1 / max(mse, 1e-12)):.2f} dB")
    reference_u8 = (np.clip(ref, 0, 1) * 255).round().astype(np.uint8)
    open(ours_path + ".reference.ppm", "wb").write(f"P6\n{width} {height}\n255\n".encode() + reference_u8.tobytes())


if __name__ == "__main__":
    main()
