"""Small random joint transformers with action projections, and reference
outputs of one evaluation over a text prefix, a video segment and an action
segment.

Two layouts are written: `relu2` (squared-ReLU feed-forward, Nemotron norms,
a separate key norm for the keys the generation stream reads) and `silu`
(gated SiLU feed-forward, query/key norms on the text stream). Weights are
rounded to bfloat16 before anything runs.

Usage: python make_cosmos3_mot_fixtures.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import Cosmos3OmniTransformer
from safetensors.torch import save_file

BASE = {
    "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128,
    "rope_scaling": {"mrope_section": [24, 20, 20]}, "rope_axes_dim": [24, 20, 20],
    "latent_channel": 4, "latent_patch_size": 2, "patch_latent_dim": 16,
    "timestep_scale": 0.001, "vocab_size": 40, "base_fps": 24,
    "enable_fps_modulation": True, "unified_3d_mrope_temporal_modality_margin": 15000,
    "unified_3d_mrope_reset_spatial_ids": True, "attention_bias": False,
    "action_gen": True, "action_dim": 8, "num_embodiment_domains": 4,
}
LAYOUTS = {
    "relu2": {"hidden_act": "relu2", "qk_norm_for_text": False, "use_und_k_norm_for_gen": True,
              "rms_norm_eps": 1e-5, "rope_theta": 1e8},
    "silu": {"hidden_act": "silu", "qk_norm_for_text": True, "use_und_k_norm_for_gen": False,
             "rms_norm_eps": 1e-6, "rope_theta": 5e6},
}
TEXT, LT, LH, LW, COND_FRAMES = 7, 3, 4, 6, 1
ACTIONS, ACTION_COND, DOMAIN = 5, 0, 2
T_VISION, T_ACTION = 700, 420


def patchify(lat, p):
    c, t, h, w = lat.shape
    x = lat.reshape(c, t, h // p, p, w // p, p)
    return torch.einsum("cthpwq->thwpqc", x).reshape(-1, p * p * c)


def write(out, name, layout, seed):
    torch.manual_seed(seed)
    cfg = dict(BASE, **layout)
    model = Cosmos3OmniTransformer.from_config(cfg).float().eval()
    with torch.no_grad():
        for prm in model.parameters():
            prm.copy_((torch.randn_like(prm) * 0.2).to(torch.bfloat16).float())
        for n, prm in model.named_parameters():
            if "norm" in n and n.endswith("weight"):
                prm.copy_((1.0 + 0.1 * torch.randn_like(prm)).to(torch.bfloat16).float())
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(d, "model.safetensors"))

    p = cfg["latent_patch_size"]
    gh, gw = LH // p, LW // p
    nv = LT * gh * gw
    cond = COND_FRAMES * gh * gw
    ids = torch.randint(0, cfg["vocab_size"], (TEXT,))
    lat = torch.randn(cfg["latent_channel"], LT, LH, LW)
    act = torch.randn(ACTIONS, cfg["action_dim"])
    # Positions: text on the diagonal; video (t, h, w) after a margin with
    # fractional time; actions on their own time axis at (0, 0).
    vpos = []
    for t in range(LT):
        for y in range(gh):
            for x in range(gw):
                vpos.append([TEXT + 3 + t * 1.5, y, x])
    apos = [[TEXT + 3 + 0.25 + a * 0.375, 0, 0] for a in range(ACTIONS)]
    gpos = torch.tensor(vpos + apos, dtype=torch.float32)
    tpos = torch.arange(TEXT, dtype=torch.float32)[:, None].expand(TEXT, 3)
    pos = torch.cat([tpos, gpos], 0).T.contiguous()
    total = TEXT + nv + ACTIONS
    vis_idx = torch.arange(TEXT, TEXT + nv)
    act_idx = torch.arange(TEXT + nv, total)
    vnoisy = torch.arange(COND_FRAMES, LT)
    anoisy = torch.arange(ACTION_COND, ACTIONS)
    with torch.no_grad():
        o = model(
            input_ids=ids, text_indexes=torch.arange(TEXT), position_ids=pos, und_len=TEXT,
            sequence_length=total, vision_tokens=[lat[None]], vision_token_shapes=[(LT, gh, gw)],
            vision_sequence_indexes=vis_idx, vision_mse_loss_indexes=vis_idx[cond:],
            vision_timesteps=torch.full((nv - cond,), float(T_VISION)),
            vision_noisy_frame_indexes=[vnoisy],
            action_tokens=[act], action_token_shapes=[(ACTIONS, 1, 1)], action_sequence_indexes=act_idx,
            action_mse_loss_indexes=act_idx[ACTION_COND:],
            action_timesteps=torch.full((ACTIONS - ACTION_COND,), float(T_ACTION)),
            action_noisy_frame_indexes=[anoisy], action_domain_ids=[torch.tensor(DOMAIN)],
            return_dict=True,
        )
    out_v = patchify(o.sample[0].reshape(cfg["latent_channel"], LT, LH, LW)[:, COND_FRAMES:], p)
    out_a = o.action[0][ACTION_COND:]

    def b(n, t):
        np.asarray(t, dtype=np.float32).tofile(os.path.join(d, n + ".bin"))

    b("patches", patchify(lat, p))
    b("actions", act)
    b("gen_positions", gpos)
    b("out_vision", out_v)
    b("out_action", out_a)
    cfg["rope_axes_dim"] = cfg.pop("rope_axes_dim")
    meta = {"config": cfg, "ids": ids.tolist(), "text": TEXT, "vision": nv, "vision_cond": cond,
            "actions": ACTIONS, "action_cond": ACTION_COND, "domain": DOMAIN,
            "t_vision": T_VISION, "t_action": T_ACTION}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(name, "video", tuple(out_v.shape), "action", tuple(out_a.shape))


if __name__ == "__main__":
    for i, (n, l) in enumerate(LAYOUTS.items()):
        write(sys.argv[1], n, l, i)
