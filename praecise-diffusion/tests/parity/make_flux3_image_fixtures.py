"""Reference fixtures for the FLUX 3 image parity tests (tiny random checkpoints).

usage: make_flux3_image_fixtures.py <flux-3-image-klein checkout>/src <diffusers checkout>/src <out dir>

Two random image transformers with the released layout at tiny widths: a flow
map ("flow_map") and an undistilled model ("guided"). For each: the package
checkpoint, the same weights converted by diffusers to its layout, one
transformer evaluation with text, the image and two references, and a short
sampling run (the reference's flow-map Euler, or Euler with classifier-free
guidance) with the latent after every step.

Everything runs in float32 on the CPU. The released models see their times as
bfloat16 (the GPU reference keeps its timesteps in bfloat16 and scales them by
1000 in bfloat16); the time embedding here rounds the same way, so the
fixtures pin that rounding too.
"""
import json
import pathlib
import sys

import torch
from safetensors.torch import save_file

ref_src, diffusers_src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sys.path.insert(0, str(ref_src))
sys.path.insert(0, str(diffusers_src))

from flux3 import dit as ref_dit  # noqa: E402
from flux3 import sampling as ref_sampling  # noqa: E402
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers  # noqa: E402

# Tiny widths, released layout.
ref_dit.HIDDEN, ref_dit.HEADS, ref_dit.DEPTH, ref_dit.DEPTH_SINGLE = 64, 2, 2, 2
ref_dit.AXES_DIM, ref_dit.MLP_RATIO = (8, 8, 8, 8), 3.0
ref_dit.IN_CHANNELS = {"image": 8, "image_cond": 8, "text": 24}
SHIFT, STEPS, GUIDANCE = 6.93, 4, 4.0
ROPE_SCALE = 0.5
torch.set_grad_enabled(False)

_embedding = ref_dit.timestep_embedding


def bf16_timestep_embedding(t, dim=256, max_period=10000):
    """The GPU reference's arithmetic: t in bfloat16, scaled by 1000 in bfloat16."""
    t = (t.to(torch.bfloat16) * 1000.0).to(torch.bfloat16).float() / 1000.0
    return _embedding(t.float(), dim, max_period).float()


ref_dit.timestep_embedding = bf16_timestep_embedding


def grid_ids(h, w, l):
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ids = torch.stack([torch.zeros_like(rows), rows, cols, torch.full_like(rows, l)], dim=-1).reshape(h * w, 4)
    return ids


def scaled(ids):
    ids = ids.float()
    ids[:, 1:3] *= ROPE_SCALE
    return ids


def text_ids(n):
    pos = torch.arange(n)
    return torch.stack([torch.zeros_like(pos)] * 3 + [pos], dim=-1)


def pack(samples):
    """The reference pack, in float32: samples[s] = [text, image_cond or None, image] (tokens, ids)."""
    modes = ref_dit.MODES
    data, ids, idx = {m: [] for m in modes}, {m: [] for m in modes}, {m: [] for m in modes}
    counts = dict.fromkeys(modes, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            idx[mode].append((start, counts[mode]))
    cat = lambda parts, width: torch.cat(parts) if parts else torch.zeros((0, width))
    data = {m: cat(data[m], ref_dit.IN_CHANNELS[m]).float() for m in modes}
    ids = {m: cat(ids[m], 4).float() for m in modes}
    return ref_dit.Packed(data=data, ids=ids, timesteps={m: torch.zeros(data[m].shape[0]) for m in modes}, mode_indices=idx)


def save(d, name, t):
    t = t.contiguous()
    (d / f"{name}.bin").write_bytes((t.int() if name.endswith("ids") else t.float()).numpy().tobytes())


meta = {}
for tag, flow_map in (("flow_map", True), ("guided", False)):
    d = out / tag
    (d / "diffusers" / "transformer").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1 if flow_map else 2)
    model = ref_dit.DiT(flow_map).eval()
    for n, p in model.named_parameters():
        if n.endswith(".scale"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * (0.5 / p.shape[-1] ** 0.5))
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(d / "model.safetensors"))
    save_file({k: v.contiguous() for k, v in convert_flux3_transformer_checkpoint_to_diffusers(dict(state)).items()},
              str(d / "diffusers" / "transformer" / "diffusion_pytorch_model.safetensors"))

    gh, gw = 3, 4
    ctx, unc = torch.randn(7, 24), torch.randn(5, 24)
    refs = [(torch.randn(6, 8), grid_ids(2, 3, -1)), (torch.randn(6, 8), grid_ids(3, 2, -2))]
    ref_tokens = torch.cat([r[0] for r in refs])
    ref_ids = torch.cat([r[1] for r in refs])
    noise = torch.randn(gh * gw, 8)
    img_ids = grid_ids(gh, gw, 0)
    image = (noise.clone(), scaled(img_ids))
    cond = (ref_tokens, scaled(ref_ids))
    for name, t in (("ctx", ctx), ("uncond", unc), ("noise", noise), ("refs", ref_tokens), ("ref_ids", ref_ids)):
        save(d, name, t)

    # One evaluation at t = 0.7 (to 0.55 for the flow map).
    packed = pack([[(ctx, text_ids(7)), cond, image]])
    packed.timesteps["image"].fill_(0.7)
    velocity = model(packed, timesteps_ref=torch.tensor([0.55]) if flow_map else None)
    save(d, "velocity", velocity)

    schedule = ref_sampling.timesteps(STEPS, SHIFT)
    steps = []
    if flow_map:
        packed = pack([[(ctx, text_ids(7)), cond, image]])
        for step in range(STEPS):
            t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
            packed.timesteps["image"].fill_(t)
            packed.data["image"] = packed.data["image"] + dt * model(packed, timesteps_ref=schedule[step + 1].reshape(1))
            steps.append(packed.data["image"].clone())
    else:
        packed = pack([[(unc, text_ids(5)), cond, image], [(ctx, text_ids(7)), cond, image]])
        for step in range(STEPS):
            t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
            packed.timesteps["image"].fill_(t)
            unconditional, conditional = model(packed).chunk(2)
            x = packed.data["image"].chunk(2)[0] + dt * (unconditional + GUIDANCE * (conditional - unconditional))
            packed.data["image"] = torch.cat((x, x))
            steps.append(x.clone())
    for i, x in enumerate(steps):
        save(d, f"step{i}", x)
    meta[tag] = {"flow_map": flow_map, "grid": [gh, gw], "steps": STEPS, "shift": SHIFT, "guidance": GUIDANCE,
                 "t": 0.7, "target": 0.55 if flow_map else None, "schedule": schedule.tolist()}

(out / "meta.json").write_text(json.dumps(meta, indent=1))
print("wrote", out)
