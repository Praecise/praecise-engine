"""Reference outputs of action-conditioned Cosmos3 runs.

A policy run (video and actions from a first frame) and a forward-dynamics run
(video from a first frame and given actions), each saving the starting noise
the reference drew so the native pipeline can start from the same values.

Usage: python make_cosmos3_action_fixtures.py <out_dir> <released_checkpoint_dir>
       python make_cosmos3_action_fixtures.py --checkpoint <checkpoint_dir> <out_dir>

The first form builds a small random checkpoint with action projections (see
make_cosmos3_fixtures.py); the second runs a released action checkpoint in
float32 and links `<out_dir>/checkpoint` to it.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

from make_cosmos3_fixtures import build, first_frame, frames_of, save

PROMPT = "Pick up the red cup and place it in the bowl."
NEGATIVE = "blurry, distorted, low quality"
SPEC = {"embodiment": "droid_lerobot", "action_width": 10, "tier": 256, "width": 256, "height": 256,
        "chunk": 4, "fps": 15.0, "view_point": "ego_view"}
SMALL = {"steps": 2, "guidance": 3.0}
REAL = {"steps": 4, "guidance": 3.0}


class NoiseLog:
    """Records every tensor the reference draws, in order."""

    def __init__(self):
        self.draws = []
        self.inner = pipeline_cosmos3_omni.randn_tensor

    def __call__(self, *args, **kwargs):
        t = self.inner(*args, **kwargs)
        self.draws.append(t.detach().clone())
        return t


class CallLog:
    """Records the token ids and positions of every transformer evaluation."""

    def __init__(self):
        self.calls = []

    def __call__(self, module, args, kwargs):
        self.calls.append({k: kwargs[k].detach().clone() for k in ("input_ids", "position_ids")})


def given_actions(n, width):
    i = torch.arange(n, dtype=torch.float32).view(-1, 1)
    c = torch.arange(width, dtype=torch.float32).view(1, -1)
    return 0.5 * torch.sin(0.7 * i + 0.3 * c)


def reference(pipe, out, run_spec):
    spec = {**SPEC, **run_spec}
    w, h, chunk = spec["width"], spec["height"], spec["chunk"]
    image = first_frame(w, h)
    image.tofile(os.path.join(out, "image.bin"))
    acts = given_actions(chunk, spec["action_width"])
    meta = {"prompt": PROMPT, "negative": NEGATIVE, **spec}
    meta["given_actions"] = save(out, "given_actions", acts)
    log = NoiseLog()
    calls = CallLog()
    hook = pipe.transformer.register_forward_pre_hook(calls, with_kwargs=True)
    pipeline_cosmos3_omni.randn_tensor = log
    try:
        for mode, guidance in (("policy", spec["guidance"]), ("forward_dynamics", 1.0)):
            log.draws.clear()
            calls.calls.clear()
            cond = CosmosActionCondition(
                mode=mode, chunk_size=chunk, domain_name=spec["embodiment"], resolution_tier=spec["tier"],
                raw_actions=acts if mode == "forward_dynamics" else None, image=Image.fromarray(image),
                view_point=spec["view_point"],
            )
            with torch.no_grad():
                res = pipe(PROMPT, NEGATIVE, action=cond, fps=spec["fps"], num_inference_steps=spec["steps"],
                           guidance_scale=guidance, generator=torch.Generator().manual_seed(7),
                           output_type="latent", enable_safety_check=False)
                vision, action = log.draws
                meta[f"{mode}_vision_noise"] = save(out, f"{mode}_vision_noise", vision[0])
                meta[f"{mode}_action_noise"] = save(out, f"{mode}_action_noise", action)
                meta[f"{mode}_latents"] = save(out, f"{mode}_latents", res.video[0])
                meta[f"{mode}_frames"] = save(out, f"{mode}_frames", frames_of(pipe, res.video))
                if res.action is not None:
                    meta[f"{mode}_actions"] = save(out, f"{mode}_actions", res.action[0])
                meta[f"{mode}_guidance"] = guidance
                # Cond pass first, then the uncond pass when guided.
                passes = calls.calls[: 2 if guidance != 1.0 else 1]
                meta[f"{mode}_ids"] = [c["input_ids"].flatten().tolist() for c in passes]
                meta[f"{mode}_positions"] = save(out, f"{mode}_positions", passes[0]["position_ids"].squeeze())
    finally:
        pipeline_cosmos3_omni.randn_tensor = log.inner
        hook.remove()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    if sys.argv[1] == "--checkpoint":
        ckpt, out = sys.argv[2], sys.argv[3]
        os.makedirs(out, exist_ok=True)
        link = os.path.join(out, "checkpoint")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(ckpt), link)
        pipe = Cosmos3OmniPipeline.from_pretrained(ckpt, torch_dtype=torch.float32, enable_safety_checker=False)
        reference(pipe, out, REAL)
    else:
        out, released = sys.argv[1], sys.argv[2]
        os.makedirs(out, exist_ok=True)
        pipe = build(out, released, action_gen=True, action_dim=12, num_embodiment_domains=9)
        reference(pipe, out, SMALL)


if __name__ == "__main__":
    main()
