"""Build small random FLUX 3 multi-stream transformers and reference outputs.

The released transformer (about 7B parameters) does not fit a parity host, so
each fixture is a random checkpoint with the released layout at tiny widths:
video and video-conditioning streams plus an action stream and its
conditioning stream, per-token timesteps, a global vector and a text context.
Query/key norm scales are randomized so the rotary head reordering is checked.

Usage: python make_flux3_fixtures.py <reference transformer.py> <out_dir>
"""

import importlib.util
import json
import os
import sys

import torch
from safetensors.torch import save_file

VIDEO, ACTION, ACTION_COND = 12, 6, 12
CTX, VEC = 48, 16
CONFIGS = {
    "main": dict(depth=2, depth_single_blocks=2, depth_late_blocks=0),
    "late": dict(depth=1, depth_single_blocks=2, depth_late_blocks=1),
}


def load_reference(path):
    spec = importlib.util.spec_from_file_location("flux3_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def video_ids(frames, h, w):
    return torch.tensor([[t, y, x, 0] for t in range(frames) for y in range(h) for x in range(w)], dtype=torch.int64)[None]


def main():
    ref = load_reference(sys.argv[1])
    root = sys.argv[2]
    torch.manual_seed(0)
    meta = {}
    for tag, depths in CONFIGS.items():
        out = os.path.join(root, tag)
        os.makedirs(out, exist_ok=True)
        params = ref.JointSingleSeqParams(
            in_channels={"video": VIDEO, "video_cond": VIDEO, "action": ACTION, "action_cond": ACTION_COND},
            sequence={"x_video": "video", "x_video_cond": "video_cond", "x_action": "action", "x_action_cond": "action_cond"},
            vec_in_dim=VEC,
            context_in_dim=CTX,
            hidden_size=64,
            num_heads=2,
            axes_dim=[8, 8, 8, 8],
            mlp_ratio=3.0,
            attn_mode="torch",
            **depths,
        )
        model = ref.JointSingleSeq(params).float().eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("_norm.scale"):
                    p.copy_(1.0 + 0.3 * torch.randn_like(p))
        save_file({"dit." + k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(out, "model.safetensors"))

        n_txt = 5
        ctx = torch.randn(1, n_txt, CTX)
        ctx_ids = torch.tensor([[0, 0, 0, l] for l in range(n_txt)], dtype=torch.int64)[None]
        vector = torch.randn(1, VEC)
        streams = {
            "x_video": (torch.randn(1, 2 * 2 * 3, VIDEO), video_ids(2, 2, 3) + torch.tensor([1, 0, 0, 0]), torch.full((1, 12), 0.7)),
            "x_video_cond": (torch.randn(1, 2 * 3, VIDEO), video_ids(1, 2, 3), torch.zeros(1, 6)),
            "x_action": (torch.randn(1, 7, ACTION), torch.tensor([[t + 1, 0, 0, 1] for t in range(7)])[None], torch.full((1, 7), 0.4)),
            "x_action_cond": (torch.randn(1, 1, ACTION_COND), torch.tensor([[0, 0, 0, 1]])[None], torch.tensor([[0.05]])),
        }
        cases = {"all": list(streams), "video_action": ["x_video", "x_action"]}
        save(out, "ctx", ctx[0])
        save(out, "vector", vector[0])
        ctx_ids[0].numpy().astype("<i4").tofile(os.path.join(out, "ctx_ids.bin"))
        for k, (x, ids, ts) in streams.items():
            save(out, k, x[0])
            save(out, k + "_t", ts[0])
            ids[0].numpy().astype("<i4").tofile(os.path.join(out, k + "_ids.bin"))
        ctx_t = torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0]])
        save(out, "ctx_t", ctx_t[0])
        for case, keys in cases.items():
            kw = {}
            for k in keys:
                x, ids, ts = streams[k]
                kw[k], kw[k + "_ids"], kw[k + "_timesteps"] = x, ids, ts
            with torch.no_grad():
                res = model(ctx=ctx, ctx_ids=ctx_ids, vector=vector, timesteps_ctx=ctx_t, **kw)
            for k in keys:
                save(out, f"out_{case}_{k}", res[k][0])
        meta[tag] = {"cases": cases, "lens": {k: v[0].shape[1] for k, v in streams.items()}, "n_txt": n_txt}
    json.dump(meta, open(os.path.join(root, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
