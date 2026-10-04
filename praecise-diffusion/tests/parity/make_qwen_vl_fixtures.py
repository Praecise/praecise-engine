"""Build a small random Qwen2.5-VL prompt encoder and reference outputs.

A random checkpoint with the released layout at tiny widths: windowed vision
blocks with one full-attention block, a merged-patch grid whose right column
of windows is partial, two images of different sizes spliced into one prompt,
and three-axis rotary sections in the language model. Weights are rounded to
bfloat16 first, so the native loader sees the values the reference used.

The token-type ids are passed explicitly: without them the reference falls
back to one-axis positions for image tokens, which is not how the released
encoder places them.

Outputs: the normalised patches of each image, the merged tokens of each
image, and the final normalised hidden state of the whole prompt.

Usage: python make_qwen_vl_fixtures.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

IMAGE, START, END = 250, 251, 252
TEXT = dict(
    hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
    rms_norm_eps=1e-6, vocab_size=300, max_position_embeddings=4096,
)
SECTIONS = [2, 3, 3]
THETA = 1e6
VISION = dict(
    depth=3, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64, patch_size=14,
    spatial_merge_size=2, temporal_patch_size=2, window_size=112, fullatt_block_indexes=[1], in_channels=3,
    hidden_act="silu",
)
# Patch grids (rows, cols): 8x12 patches = 4x6 merged, windows of 4x4 merged
# leave a partial right column; the second image fits one window.
GRIDS = [(8, 12), (4, 6)]


MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def patches(px):
    """The reference image processor's normalisation and patch layout."""
    c, h, w = px.shape
    p, m, tp = 14, 2, 2
    x = (px - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    x = x[None].repeat(tp, 1, 1, 1)
    gh, gw = h // p, w // p
    x = x.reshape(1, tp, c, gh // m, m, p, gw // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gh * gw, c * tp * p * p)


def save(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "checkpoint", "text_encoder"), exist_ok=True)
    torch.manual_seed(0)
    rope = dict(rope_type="default", rope_theta=THETA, mrope_section=SECTIONS)
    cfg = Qwen2_5_VLConfig(
        text_config=dict(TEXT, rope_parameters=rope),
        vision_config=dict(VISION),
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END,
    )
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    sd = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.normal_(0, 0.02 if p.dim() > 1 else 0.1)
            if name.endswith("norm.weight") or "norm1" in name or "norm2" in name or "ln_q" in name:
                p.add_(1.0)
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        for name, p in model.state_dict().items():
            if name == "lm_head.weight":
                continue
            n = name.replace("model.language_model.", "model.").replace("model.visual.", "visual.")
            sd[n] = p.to(torch.bfloat16).contiguous()
    save_file(sd, os.path.join(out, "checkpoint", "text_encoder", "model.safetensors"))
    released = dict(
        TEXT, model_type="qwen2_5_vl", rope_theta=THETA,
        rope_scaling=dict(mrope_section=SECTIONS, rope_type="default", type="default"),
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END, vision_config=VISION,
    )
    with open(os.path.join(out, "checkpoint", "text_encoder", "config.json"), "w") as f:
        json.dump(released, f, indent=1)

    g = torch.Generator().manual_seed(1)
    pv, thw = [], []
    for i, (gh, gw) in enumerate(GRIDS):
        px = torch.rand(3, gh * 14, gw * 14, generator=g)
        save(out, f"pixels_{i}", px)
        r = patches(px)
        save(out, f"patches_{i}", r)
        pv.append(r)
        thw.append(torch.tensor([[1, gh, gw]]))
    pv = torch.cat(pv).to(torch.float32)
    thw = torch.cat(thw)

    with torch.no_grad():
        feats = model.model.get_image_features(pixel_values=pv, image_grid_thw=thw)
        feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
        for i, f in enumerate(feats if isinstance(feats, (list, tuple)) else torch.split(feats, [a * b // 4 for a, b in GRIDS])):
            save(out, f"tokens_{i}", f)
        ids = list(range(10, 15)) + [START] + [IMAGE] * (GRIDS[0][0] * GRIDS[0][1] // 4) + [END]
        ids += list(range(40, 44)) + [START] + [IMAGE] * (GRIDS[1][0] * GRIDS[1][1] // 4) + [END] + list(range(70, 77))
        ids_t = torch.tensor([ids])
        o = model.model(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), pixel_values=pv,
                        image_grid_thw=thw, output_hidden_states=True, mm_token_type_ids=(ids_t == IMAGE).int())
        last = o.last_hidden_state[0]
        assert torch.equal(last, o.hidden_states[-1][0]), "hidden_states[-1] is not the normalised state"
        save(out, "hidden", last)
        # Text only.
        tids = torch.tensor([list(range(20, 33))])
        t = model.model(input_ids=tids, attention_mask=torch.ones_like(tids)).last_hidden_state[0]
        save(out, "hidden_text", t)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(ids=ids, text_ids=tids[0].tolist(), grids=GRIDS), f, indent=1)
    print("rows", last.shape, "features", [a * b // 4 for a, b in GRIDS])


if __name__ == "__main__":
    main()
