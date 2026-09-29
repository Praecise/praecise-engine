"""Time the reference implementation on the same checkpoint and settings as
the `flux2` example, for a like-for-like comparison.

Usage: python bench_reference.py <checkpoint dir> [width height steps runs]
"""

import sys
import time

import torch
from diffusers import Flux2KleinPipeline


def main():
    model = sys.argv[1]
    width, height, steps, runs = (int(v) for v in (sys.argv[2:6] + ["1024", "1024", "4", "4"][len(sys.argv[2:6]):]))
    t = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    print(f"loaded in {1000 * (time.time() - t):.0f} ms")
    prompt = "a lighthouse on a rocky coast at dusk, photograph"
    for run in range(runs):
        torch.cuda.synchronize()
        t = time.time()
        pipe(prompt=prompt, width=width, height=height, num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(0))
        torch.cuda.synchronize()
        tag = " (warm-up)" if run == 0 else ""
        print(f"run {run}{tag}: total {1000 * (time.time() - t):.0f} ms")
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
