"""Reference outputs of the released FLUX 3 image checkpoints, for the
`flux3_image_reference` test.

usage: compare_flux3_image_reference.py <flux-3-image-klein checkout>/src <cases.json> <out dir>

cases.json lists the cases: {"name", "weights" (the checkpoint directory the
engine loads), "dit" (the reference's transformer: a package directory or
file), "text_encoder", "autoencoder" (package files), optional "variant",
"precision", "prompt", "width", "height", "steps", "guidance" (absent for a
flow map), "shift", "seed", "references" (image files) and the bounds the
test asserts}. For each case this writes the prompt's text context, the
starting noise, the latent after every step, the decoded image and the
references before and after preparation, as the reference pipeline computes
them on the GPU in bfloat16, and the meta.json the test reads.
"""
import json
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, sys.argv[1])
from flux3.autoencoder import AutoEncoder, latent_ids, latent_shape  # noqa: E402
from flux3.checkpoint import is_flow_map, load_checkpoint  # noqa: E402
from flux3.dit import DiT, IN_CHANNELS  # noqa: E402
from flux3.pipeline import pack, prepare_multi_reference, prepare_reference, scaled_ids, to_pil, to_tensor  # noqa: E402
from flux3.sampling import timesteps  # noqa: E402
from flux3.text_encoder import TextEncoder  # noqa: E402
from flux3 import text_encoder as ref_text_encoder  # noqa: E402


class TextOnlyProcessor:
    """The released processor without its image and video halves (text prompts only)."""

    @classmethod
    def from_pretrained(cls, path, **_):
        from transformers import AutoTokenizer
        p = cls()
        p.tokenizer = AutoTokenizer.from_pretrained(path)
        p.tokenizer.chat_template = json.loads((pathlib.Path(path) / "chat_template.json").read_text())["chat_template"]
        return p

    def apply_chat_template(self, messages, **kw):
        return self.tokenizer.apply_chat_template(messages, **kw)


ref_text_encoder.AutoProcessor = TextOnlyProcessor

if os.environ.get("FLUX3_REFERENCE_ATTENTION") == "efficient":
    # A control run: the same pipeline with another attention kernel, to
    # measure how far the reference drifts from itself.
    from flux3 import dit as ref_dit
    from torch.nn.attention import SDPBackend

    class _Backends:
        CUDNN_ATTENTION = SDPBackend.EFFICIENT_ATTENTION
        MATH = SDPBackend.MATH

    ref_dit.SDPBackend = _Backends

cases = json.loads(pathlib.Path(sys.argv[2]).read_text())
out = pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
device = "cuda"


def f32(path, t):
    path.write_bytes(t.detach().float().contiguous().cpu().numpy().tobytes())


loaded = {}


def _dit(path):
    with torch.device("meta"):
        return DiT(is_flow_map(path))


def component(kind, path, make):
    key = (kind, path)
    if key not in loaded:
        for k in [k for k in loaded if k[0] == kind]:
            del loaded[k]
        torch.cuda.empty_cache()
        loaded[key] = make()
    return loaded[key]


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def run(dit, packed, schedule, guidance, flow_map):
    steps = []
    for step in range(len(schedule) - 1):
        t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        if flow_map:
            target = schedule[step + 1].reshape(1).to(device)
            packed.data["image"] = packed.data["image"] + dt * dit(packed, timesteps_ref=target)
            steps.append(packed.data["image"].float().clone())
        else:
            unconditional, conditional = dit(packed).chunk(2)
            x = packed.data["image"].chunk(2)[0] + dt * (unconditional + guidance * (conditional - unconditional))
            packed.data["image"] = torch.cat((x, x))
            steps.append(x.float().clone())
    return steps


meta = {"cases": []}
for case in cases:
    d = out / case["name"]
    d.mkdir(exist_ok=True)
    dit = component("dit", case["dit"], lambda: load_checkpoint(_dit(case["dit"]), case["dit"], device, torch.bfloat16))
    te = component("te", case["text_encoder"], lambda: TextEncoder(case["text_encoder"], device))
    ae = component("ae", case["autoencoder"], lambda: load_checkpoint(AutoEncoder(), case["autoencoder"], device, torch.float32))
    flow_map = dit.flow_map

    refs_meta, images = [], [Image.open(p) for p in case.get("references", [])]
    images = [i.convert("RGB").resize(tuple(case["reference_size"]), Image.Resampling.LANCZOS) if case.get("reference_size") else i.convert("RGB") for i in images]
    prepared = [prepare_multi_reference(i) for i in images] if len(images) > 1 else [prepare_reference(i)[0] for i in images]
    for k, (o, p) in enumerate(zip(images, prepared)):
        (d / f"ref{k}.rgb").write_bytes(np.asarray(o).tobytes())
        (d / f"ref{k}_prepared.rgb").write_bytes(np.asarray(p).tobytes())
        refs_meta.append({"original": f"ref{k}.rgb", "original_size": list(o.size), "prepared": f"ref{k}_prepared.rgb", "prepared_size": list(p.size)})

    with torch.inference_mode():
        tokens, text_ids = te.encode(case["prompt"])
        f32(d / "context.bin", tokens)
        cond = None
        if prepared:
            parts, ids = [], []
            for k, r in enumerate(prepared):
                t, i = ae.encode(to_tensor(r).to(device))
                i = scaled_ids(i)
                i[:, 3] = -(1 + k)
                parts.append(t.to(torch.bfloat16))
                ids.append(i)
            cond = (torch.cat(parts), torch.cat(ids))
        h, w = latent_shape(case["height"], case["width"])
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(case["seed"]))
        noise = noise.to(torch.bfloat16).float().to(device)
        f32(d / "noise.bin", noise)
        ids = latent_ids(h, w, device)
        image = (noise, scaled_ids(ids))
        if flow_map:
            packed = pack([[(tokens, text_ids), cond, image]])
        else:
            packed = pack([[te.encode(""), cond, image], [(tokens, text_ids), cond, image]])
    schedule = timesteps(case["steps"], case.get("shift", 6.93))
    steps = run(dit, packed, schedule, case.get("guidance"), flow_map)
    for i, x in enumerate(steps):
        f32(d / f"step{i}.bin", x)
    with torch.inference_mode():
        pil = to_pil(ae.decode(steps[-1], ids))
    (d / "image.rgb").write_bytes(np.asarray(pil).tobytes())
    pil.save(d / "reference.png")
    entry = {k: v for k, v in case.items() if k not in ("dit", "text_encoder", "autoencoder", "references", "reference_size")}
    entry["references"] = refs_meta
    meta["cases"].append(entry)
    print(case["name"], "done", flush=True)

(out / "meta.json").write_text(json.dumps(meta, indent=1))

