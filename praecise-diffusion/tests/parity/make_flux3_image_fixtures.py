"""Build small random FLUX 3 image transformers and reference outputs.

Each fixture is a random diffusers `Flux3Transformer2DModel` at tiny widths,
written in the release's own tensor names (the inverse of diffusers'
`convert_flux3_transformer_checkpoint_to_diffusers`, checked by converting
back), with an image stream, a reference stream and a text context placed as
the reference pipeline places them: the latent grid at half steps on the
rotary h and w axes, the reference at l = -1, the text along l. One fixture
is a flow map (given the time each step jumps to), one is not. Also writes
the time grid of the reference scheduler for 16 and 50 steps.

Usage: python make_flux3_image_fixtures.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers
from diffusers.modular_pipelines.flux3.before_denoise import _prepare_image_ids
from safetensors.torch import save_file

IN, CTX, HEAD, HEADS = 8, 24, 16, 2
IMG, REF, TXT = (2, 3), (2, 2), 5
SHIFT, ROPE_SCALE = 6.93, 0.5

RENAME = {
    "timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "target_timestep_embedder.linear_1.weight": "ref_time_in.in_layer.weight",
    "target_timestep_embedder.linear_2.weight": "ref_time_in.out_layer.weight",
    "proj_out.weight": "final_layer.image.linear.weight",
    "norm_out.linear.weight": "final_layer.image.adaLN_modulation.1.weight",
}
PREFIXES = [
    ("x_embedders.", "emb_in."),
    ("context_embedder.", "txt_in."),
    ("mode_blocks.", "content_mode_blocks."),
    ("single_transformer_blocks.", "single_blocks."),
]
SUFFIXES = {
    "attn.to_q.weight": "q_proj.weight",
    "attn.to_k.weight": "k_proj.weight",
    "attn.to_v.weight": "v_proj.weight",
    "attn.to_out.0.weight": "attn_out.weight",
    "attn.norm_q.weight": "norm.query_norm.scale",
    "attn.norm_k.weight": "norm.key_norm.scale",
    "ff.linear_in.weight": "mlp_in.weight",
    "ff.linear_out.weight": "mlp_out.weight",
}


def native(sd):
    out = {}
    for k, v in sd.items():
        if k in RENAME:
            if k == "norm_out.linear.weight":
                scale, shift = v.chunk(2, dim=0)
                v = torch.cat([shift, scale])
            out[RENAME[k]] = v.clone().contiguous()
            continue
        n = k
        for a, b in PREFIXES:
            if n.startswith(a):
                n = b + n[len(a) :]
                break
        if "stream_modulations." in n:
            n = n.replace(".linear.weight", ".lin.weight")
        for a, b in SUFFIXES.items():
            if n.endswith(a):
                n = n[: -len(a)] + b
                break
        out[n] = v.clone().contiguous()
    back = convert_flux3_transformer_checkpoint_to_diffusers(dict(out))
    assert back.keys() == sd.keys(), sorted(set(back) ^ set(sd))
    assert all(torch.equal(back[k], sd[k]) for k in sd)
    return out


def text_ids(n):
    return torch.tensor([[0, 0, 0, i] for i in range(n)], dtype=torch.float32)


def grid(gh, gw, l):
    return [[0, y, x, l] for y in range(gh) for x in range(gw)]


def save(d, name, t, dtype=np.float32):
    np.asarray(t, dtype=dtype).tofile(os.path.join(d, f"{name}.bin"))


def main(out):
    meta = {}
    for tag, flow_map, t, target in [("flow", True, 0.7, 0.4), ("base", False, 0.55, None)]:
        d = os.path.join(out, tag)
        os.makedirs(d, exist_ok=True)
        torch.manual_seed(7 if flow_map else 11)
        model = Flux3Transformer2DModel(
            in_channels=IN,
            cond_in_channels=IN,
            num_layers=1,
            num_single_layers=2,
            attention_head_dim=HEAD,
            num_attention_heads=HEADS,
            joint_attention_dim=CTX,
            axes_dims_rope=(4, 4, 4, 4),
            flow_map=flow_map,
        ).eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(1 + 0.2 * torch.randn_like(p) if "norm" in name else 0.2 * torch.randn_like(p))
        save_file(native(model.state_dict()), os.path.join(d, "model.safetensors"))

        img = torch.randn(1, IMG[0] * IMG[1], IN)
        ref = torch.randn(1, REF[0] * REF[1], IN)
        ctx = torch.randn(1, TXT, CTX)
        for name, v in [("img", img), ("ref", ref), ("ctx", ctx)]:
            save(d, name, v.flatten())
        save(d, "img_ids", grid(*IMG, 0), np.int32)
        save(d, "ref_ids", grid(*REF, -1), np.int32)
        save(d, "ctx_ids", text_ids(TXT), np.int32)
        kwargs = dict(
            hidden_states=img,
            encoder_hidden_states=ctx,
            timestep=torch.tensor([t]),
            img_ids=_prepare_image_ids(*IMG, ROPE_SCALE),
            txt_ids=text_ids(TXT),
            target_timestep=None if target is None else torch.tensor([target]),
            return_dict=False,
        )
        with torch.no_grad():
            with_ref = model(**kwargs, image_cond_hidden_states=ref, image_cond_ids=_prepare_image_ids(*REF, ROPE_SCALE, l_coord=-1))[0]
            no_ref = model(**kwargs)[0]
        save(d, "out_ref", with_ref.flatten())
        save(d, "out_noref", no_ref.flatten())
        meta[tag] = {"flow_map": flow_map, "t": t, "target": target}

    for steps in (16, 50):
        sched = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
        sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps))
        save(out, f"sigmas_{steps}", sched.sigmas.numpy())
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
