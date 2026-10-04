"""Build a small random Qwen3-VL prompt encoder and reference outputs.

A random checkpoint with the released layout at tiny widths: a vision tower
with a learned position grid smaller than the patch grids (so it is
interpolated), three deepstack blocks feeding the first language-model
layers, per-head query and key norms, and interleaved three-axis rotary
sections. Two images of different sizes are spliced into one prompt.
Weights are rounded to bfloat16 first. The patches come from the reference
image processor, and the token-type ids are passed as the processor returns
them.

Outputs: the normalised patches, merged tokens and deepstack features of
each image, and the last layer's hidden state before the final norm (what
the image generator reads) for the prompt with images and a text-only one.

Usage: python make_qwen3_vl_fixtures.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen2VLImageProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

IMAGE, START, END = 250, 251, 252
SECTIONS = [4, 2, 2]
THETA = 5e6
TEXT = dict(
    hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_hidden_layers=4, rms_norm_eps=1e-6, vocab_size=300, max_position_embeddings=4096,
)
VISION = dict(
    depth=4, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64, patch_size=4,
    spatial_merge_size=2, temporal_patch_size=2, in_channels=3, num_position_embeddings=16,
    deepstack_visual_indexes=[0, 2, 3], hidden_act="gelu_pytorch_tanh",
)
# Patch grids (rows, cols).
GRIDS = [(8, 12), (4, 6)]


# An intermediate hidden state (as MiniMax-H3 reads hidden_states[50]); this
# one sits inside the deepstack layers.
LAYER = 2


def save(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "checkpoint", "text_encoder"), exist_ok=True)
    torch.manual_seed(0)
    rope = dict(rope_type="default", rope_theta=THETA, mrope_section=SECTIONS, mrope_interleaved=True)
    cfg = Qwen3VLConfig(
        text_config=dict(TEXT, rope_parameters=rope),
        vision_config=dict(VISION),
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END,
    )
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    sd = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.normal_(0, 0.02 if p.dim() > 1 else 0.1)
            if name.endswith("norm.weight") or "norm1" in name or "norm2" in name or "layernorm" in name:
                p.add_(1.0)
            if name.endswith("pos_embed.weight"):
                p.normal_(0, 0.5)
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        for name, p in model.state_dict().items():
            if name == "lm_head.weight":
                continue
            sd[name] = p.to(torch.bfloat16).contiguous()
    save_file(sd, os.path.join(out, "checkpoint", "text_encoder", "model.safetensors"))
    released = dict(
        architectures=["Qwen3VLForConditionalGeneration"], model_type="qwen3_vl",
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END, tie_word_embeddings=False,
        text_config=dict(TEXT, model_type="qwen3_vl_text", rope_theta=THETA,
                         rope_scaling=dict(mrope_interleaved=True, mrope_section=SECTIONS, rope_type="default")),
        vision_config=dict(VISION, model_type="qwen3_vl"),
    )
    with open(os.path.join(out, "checkpoint", "text_encoder", "config.json"), "w") as f:
        json.dump(released, f, indent=1)
    proc = Qwen2VLImageProcessor(
        patch_size=4, merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
        size={"shortest_edge": 64, "longest_edge": 1 << 24},
    )
    g = torch.Generator().manual_seed(1)
    pv, thw = [], []
    for i, (gh, gw) in enumerate(GRIDS):
        px = torch.randint(0, 256, (gh * 4, gw * 4, 3), generator=g, dtype=torch.uint8)
        save(out, f"pixels_{i}", px.permute(2, 0, 1).float() / 255)
        b = proc(images=[px.numpy()], return_tensors="pt")
        assert b["image_grid_thw"].tolist() == [[1, gh, gw]], b["image_grid_thw"]
        save(out, f"patches_{i}", b["pixel_values"])
        pv.append(b["pixel_values"])
        thw.append(b["image_grid_thw"])
    pv = torch.cat(pv).to(torch.float32)
    thw = torch.cat(thw)
    lm = model.model.language_model
    handle = lm.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            feats = model.model.get_image_features(pixel_values=pv, image_grid_thw=thw, return_dict=True)
            sizes = [a * b // 4 for a, b in GRIDS]
            pooled = feats.pooler_output
            pooled = torch.split(torch.cat(list(pooled)) if isinstance(pooled, (list, tuple)) else pooled, sizes)
            for i, f in enumerate(pooled):
                save(out, f"tokens_{i}", f)
            for k, ds in enumerate(feats.deepstack_features):
                for i, f in enumerate(ds):
                    save(out, f"deepstack_{i}_{k}", f)
            ids = list(range(10, 15)) + [START] + [IMAGE] * sizes[0] + [END]
            ids += list(range(40, 44)) + [START] + [IMAGE] * sizes[1] + [END] + list(range(70, 77))
            ids_t = torch.tensor([ids])
            o = model.model(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), pixel_values=pv,
                            image_grid_thw=thw, mm_token_type_ids=(ids_t == IMAGE).int())
            save(out, "hidden", o.last_hidden_state[0])
            o = model.model(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), pixel_values=pv,
                            image_grid_thw=thw, mm_token_type_ids=(ids_t == IMAGE).int(), output_hidden_states=True)
            save(out, "hidden_at", o.hidden_states[LAYER][0])
            tids = torch.tensor([list(range(20, 33))])
            t = model.model(input_ids=tids, attention_mask=torch.ones_like(tids),
                            mm_token_type_ids=torch.zeros_like(tids), output_hidden_states=True)
            save(out, "hidden_text", t.last_hidden_state[0])
            save(out, "hidden_text_at", t.hidden_states[LAYER][0])
    finally:
        handle.remove()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(ids=ids, text_ids=tids[0].tolist(), grids=GRIDS, layer=LAYER), f, indent=1)
    print("rows", len(ids), "features", sizes)


if __name__ == "__main__":
    main()
