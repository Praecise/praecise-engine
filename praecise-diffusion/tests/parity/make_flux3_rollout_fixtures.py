"""Reference closed-loop rollouts for the FLUX 3 policy rollout test.

usage: make_flux3_rollout_fixtures.py <lerobot src dir> <out dir>

Runs the reference policy's control loop (its history preprocessor, its
action queue and its command postprocessor, all from its own source) tick by
tick, with a small fixed function standing in for the chunk sampler, so the
fixture pins the history bookkeeping, the normalisation and the command
integration across chunks. Two scenarios per case: a scripted loop whose
measured state follows the commands loosely and whose cameras change every
tick, and a loop where every command is reached exactly and the cameras hold
the first frame.
"""
import json
import pathlib
import sys
import types
from collections import deque

import torch

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(src))
pkg = types.ModuleType("lerobot.policies")
pkg.__path__ = [str(src / "lerobot" / "policies")]
sys.modules["lerobot.policies"] = pkg

from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.policies.flux3 import processor_flux3 as proc  # noqa: E402
from lerobot.policies.flux3.modeling_flux3 import Flux3Policy  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

CASES = [
    dict(name="delta", d=3, n_obs=3, chunk=5, execute=2, cams=2, hw=(2, 3), repr="delta", absolute=[-1], past=True),
    dict(name="absolute", d=2, n_obs=2, chunk=4, execute=4, cams=1, hw=(2, 2), repr="absolute", absolute=[], past=False),
]
TICKS = 9


def run(case, q, w, b, states_of, pixels_of):
    d, n, cams = case["d"], case["n_obs"], case["cams"]
    keys = [f"observation.images.cam{c}" for c in range(cams)]
    pre = proc.ObservationHistoryNormalizerProcessorStep(
        d, n_obs_steps=n, chunk_size=case["chunk"], camera_keys=keys,
        action_representation=case["repr"], absolute_dims=case["absolute"], condition_on_past_actions=case["past"],
    )
    pre.load_state_dict({k: torch.tensor(v) for k, v in q.items()})
    post = proc.ActionHistoryUnnormalizerProcessorStep(d, action_representation=case["repr"], absolute_dims=case["absolute"])
    post.load_state_dict({k: torch.tensor(v) for k, v in q.items() if k.startswith("action.")})
    post.history = pre

    def predict(batch):
        x = [batch[OBS_STATE][0].reshape(-1)]
        if case["past"]:
            x.append(batch[proc.PAST_ACTIONS][0].reshape(-1))
        x.append(torch.stack([batch[k][0, t].mean() for k in keys for t in range(n)]))
        x = torch.cat(x).double()
        y = torch.tanh(torch.tensor(w, dtype=torch.float64) @ x + torch.tensor(b, dtype=torch.float64))
        return y.float().reshape(1, case["chunk"], d)

    cfg = types.SimpleNamespace(use_relative_actions=False, n_action_steps=case["execute"])
    policy = types.SimpleNamespace(config=cfg, eval=lambda: None, predict_action_chunk=lambda batch, **_: predict(batch))
    policy._action_queue = deque([], maxlen=case["execute"])

    ticks, last = [], None
    for t in range(TICKS):
        state = states_of(t, last)
        pixels = pixels_of(t)
        obs = {OBS_STATE: torch.tensor([state])}
        for k, px in zip(keys, pixels):
            obs[k] = (torch.tensor(px, dtype=torch.uint8).float() / 255.0).reshape(1, 3, *case["hw"])
        tr = pre(create_transition(observation=obs))
        action = Flux3Policy.select_action(policy, tr[TransitionKey.OBSERVATION])
        command = post(create_transition(action=action))[TransitionKey.ACTION][0]
        last = command.tolist()
        ticks.append({"state": [float(s) for s in state], "pixels": pixels, "command": last})
    return ticks


def main():
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    cases = []
    for case in CASES:
        d, n, cams, (h, wd) = case["d"], case["n_obs"], case["cams"], case["hw"]
        lo = (torch.rand(d, generator=g) * -2.0 - 0.5).tolist()
        hi = (torch.rand(d, generator=g) * 2.0 + 0.5).tolist()
        dlo = (torch.rand(d, generator=g) * -0.3 - 0.05).tolist()
        dhi = (torch.rand(d, generator=g) * 0.3 + 0.05).tolist()
        q = {"state.q01": lo, "state.q99": hi, "action.q01": dlo if case["repr"] == "delta" else lo, "action.q99": dhi if case["repr"] == "delta" else hi}
        if case["repr"] == "delta":
            for c in case["absolute"]:
                q["action.q01"][c], q["action.q99"][c] = 0.0, 1.0
        inputs = n * d * (2 if case["past"] else 1) + cams * n
        w = (torch.randn(case["chunk"] * d, inputs, generator=g) * 0.4).tolist()
        b = (torch.randn(case["chunk"] * d, generator=g) * 0.3).tolist()
        start = [0.5 * (l + u) for l, u in zip(lo, hi)]
        npx = 3 * h * wd

        def scripted_state(t, last, start=start, d=d):
            if last is None:
                return start
            return [float(torch.tensor(0.8 * last[c] + 0.1 * ((t * 7 + c * 3) % 5 - 2), dtype=torch.float32)) for c in range(d)]

        def scripted_pixels(t, cams=cams, npx=npx):
            return [[(t * 37 + c * 91 + i * 11) % 256 for i in range(npx)] for c in range(cams)]

        def reached_state(t, last, start=start):
            return start if last is None else last

        def held_pixels(t, cams=cams, npx=npx):
            return scripted_pixels(0, cams, npx)

        cases.append({
            **case,
            "quantiles": q,
            "w": w,
            "b": b,
            "scripted": run(case, q, w, b, scripted_state, scripted_pixels),
            "reached": run(case, q, w, b, reached_state, held_pixels),
        })
    (out / "rollout.json").write_text(json.dumps({"cases": cases}))
    print("wrote", out / "rollout.json")


main()
