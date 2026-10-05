"""Compare one released-weights FLUX 3 action chunk against the reference policy.

usage: compare_flux3_release.py <lerobot src dir> <policy dir> <base dir> <dump dir>

The dump is written by the `flux3_release_real` test with
`PRAECISE_FLUX3_DUMP=<dump dir>`: the camera frames, the normalised states
and past actions, the noise and the chunk the engine predicted. This script
loads the same released policy with the reference implementation, runs its
own `predict_action_chunk` on those inputs with that noise, and prints the
difference.
"""
import json
import os
import pathlib
import sys
import time
import types

import numpy as np
import torch

src, policy_dir, base_dir, dump = (pathlib.Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(src))
pkg = types.ModuleType("lerobot.policies")
pkg.__path__ = [str(src / "lerobot" / "policies")]
sys.modules["lerobot.policies"] = pkg

from lerobot.policies.flux3.modeling_flux3 import Flux3Policy  # noqa: E402
from lerobot.policies.flux3.processor_flux3 import PAST_ACTIONS  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

meta = json.loads((dump / "meta.json").read_text())


def load(name, shape=None):
    v = np.fromfile(dump / name, dtype="<f4")
    return torch.from_numpy(v.reshape(shape) if shape else v)


# The released policy with its base files resolved to the local copies.
local = dump / "policy"
local.mkdir(exist_ok=True)
cfg = json.loads((policy_dir / "config.json").read_text())
cfg.update(video_vae_id=str(base_dir / "video_vae.safetensors"), text_encoder_id=str(base_dir / "text_encoder"), device="cpu")
(local / "config.json").write_text(json.dumps(cfg))
for f in policy_dir.iterdir():
    if f.is_file() and f.name != "config.json" and not (local / f.name).exists():
        os.symlink(f, local / f.name)

t0 = time.time()
policy = Flux3Policy.from_pretrained(str(local))
policy.eval()
print(f"loaded in {time.time() - t0:.0f}s, dit dtype {next(policy.dit.parameters()).dtype}", flush=True)

frames, d, (h, w) = meta["frames"], meta["action_dim"], meta["frame_hw"]
keys = meta["camera_keys"]
cams = load("cameras.f32", (len(keys), frames, 3, h, w))
batch = {k: cams[i][None] for i, k in enumerate(keys)}
batch[OBS_STATE] = load("states.f32", (1, frames, d))
if (dump / "past.f32").exists():
    batch[PAST_ACTIONS] = load("past.f32", (1, frames, d))
batch["task"] = [meta["instruction"]]

# The engine's noise in place of the reference generator's draws, in order:
# the video latents, then the actions.
noise = [load("noise_video.f32"), load("noise_action.f32")]
randn = torch.randn


def seeded(*shape, generator=None, **kw):
    if generator is not None and noise:
        n = noise.pop(0)
        size = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else shape
        assert n.numel() == int(np.prod(size)), f"noise of {n.numel()} values for shape {tuple(size)}"
        return n.reshape(size).clone()
    return randn(*shape, generator=generator, **kw)


torch.randn = seeded
t1 = time.time()
with torch.no_grad():
    ref = policy.predict_action_chunk(batch)[0].float()
torch.randn = randn
assert not noise, "the reference drew less noise than recorded"
print(f"reference chunk in {time.time() - t1:.0f}s", flush=True)

ours = load("chunk.f32", (meta["chunk_size"], d))
diff = (ours - ref).abs()
print(f"max |diff| {diff.max().item():.5f}, mean |diff| {diff.mean().item():.5f}, max |ref| {ref.abs().max().item():.4f}")
print("ref first rows ", ref[:2].tolist())
print("ours first rows", ours[:2].tolist())
