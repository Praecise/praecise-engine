"""Record reference GR00T N1.7 action chunks for random-weight small models.

A small Qwen3-VL backbone config (written next to the release) and a
GR00T N1.7 policy with random weights are built from their configs. For
each sample, a random image and instruction go through the Qwen3-VL image
processor (no resizing: the sides are multiples of the merged patch) and
the chat prompt, a random normalised state goes to the state encoder, and
the action head integrates a fixed noise chunk. The fixture keeps every
model input, the backbone hidden states the head reads, and the chunk.

Usage:
  make_groot_fixture.py <lerobot/src> <qwen3-vl dir> <out> [--samples N]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

TEXT = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16}
MROPE = [2, 3, 3]
VISION = {"depth": 3, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
          "deepstack_visual_indexes": [0, 1]}
HEAD = {
    "backbone_embedding_dim": 64, "select_layer": 3, "hidden_size": 48, "input_embedding_dim": 64,
    "max_state_dim": 16, "max_action_dim": 8, "action_horizon": 8, "max_seq_len": 64, "max_num_embodiments": 4,
    "use_flash_attention": False, "load_bf16": False, "model_dtype": "float32", "dtype": "float32",
    "diffusion_model_cfg": {"positional_embeddings": None, "num_layers": 4, "num_attention_heads": 2,
                            "attention_head_dim": 32, "norm_type": "ada_norm", "dropout": 0.0,
                            "final_dropout": False, "output_dim": 48, "interleave_self_attention": True},
    "vl_self_attention_cfg": {"positional_embeddings": None, "num_layers": 2, "num_attention_heads": 2,
                              "attention_head_dim": 32, "dropout": 0.0, "final_dropout": False},
}
SIZES = [(64, 96), (96, 64)]


def randomize(model, torch):
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                fan_in = p.shape[-2] if "W" == name.split(".")[-1] else p.shape[-1]
                p.copy_(torch.randn(p.shape, generator=g) / fan_in**0.5)
            elif name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))


def patchify(torch, px, pc):
    """The Qwen2-VL image processor without resizing: rescale, normalise,
    repeat over the temporal patch and lay out merged 2x2 blocks."""
    p, m, tp = pc["patch_size"], pc["merge_size"], pc["temporal_patch_size"]
    x = px.permute(2, 0, 1).to(torch.float32) / 255.0
    mean = torch.tensor(pc["image_mean"])[:, None, None]
    std = torch.tensor(pc["image_std"])[:, None, None]
    x = (x - mean) / std
    c, h, w = x.shape
    gh, gw = h // p, w // p
    x = x[None].repeat(tp, 1, 1, 1)
    x = x.reshape(1, tp, c, gh // m, m, p, gw // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = x.reshape(gh * gw, c * tp * p * p)
    return {"pixel_values": flat, "image_grid_thw": torch.tensor([[1, gh, gw]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot")
    ap.add_argument("qwen3vl")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--embodiment", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--instruction", default="pick up the red cube and place it in the bowl")
    a = ap.parse_args()
    sys.path.insert(0, a.lerobot)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_vla_rollout_fixtures import light_policies_package

    light_policies_package(a.lerobot)
    import importlib.util
    import types

    # The package's training-time image pipeline is not used here.
    for name in ("cv2", "albumentations"):
        if importlib.util.find_spec(name) is None:
            sys.modules[name] = types.ModuleType(name)

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    out = Path(a.out)
    backbone = out / "backbone"
    shutil.copytree(a.qwen3vl, backbone, dirs_exist_ok=True)
    bc = json.loads((backbone / "config.json").read_text())
    bc["text_config"].update(TEXT)
    bc["text_config"]["rope_scaling"]["mrope_section"] = MROPE
    bc["vision_config"].update(VISION)
    bc["dtype"] = "float32"
    (backbone / "config.json").write_text(json.dumps(bc, indent=1))

    torch.manual_seed(a.seed)
    cfg = GR00TN17Config(model_name=str(backbone), **HEAD)
    model = GR00TN17(cfg, transformers_loading_kwargs={}, load_backbone_weights=False).to(torch.float32).eval()
    randomize(model, torch)
    model.save_pretrained(out / "policy", safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(str(backbone))
    pc = json.loads((backbone / "preprocessor_config.json").read_text())
    prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{a.instruction}<|im_end|>\n<|im_start|>assistant\n"
    tokens = tok(prompt, add_special_tokens=False)["input_ids"]
    image_token = bc["image_token_id"]
    merge = bc["vision_config"]["spatial_merge_size"]

    g = torch.Generator().manual_seed(a.seed)
    head = model.action_head
    tensors, meta = {}, {"samples": a.samples, "embodiment": a.embodiment, "tokens": tokens, "sizes": [],
                         "torch": torch.__version__}
    captured = {}
    hook = model.backbone.language_model.layers[-1].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("f", o[0] if isinstance(o, (tuple, list)) else o))
    for s in range(a.samples):
        h, w = SIZES[s % len(SIZES)]
        px = torch.randint(0, 256, (h, w, 3), generator=g, dtype=torch.uint8)
        ip = patchify(torch, px, pc)
        grid = ip["image_grid_thw"]
        n_img = int(grid[0, 1] * grid[0, 2]) // (merge * merge)
        ids = []
        for t in tokens:
            ids.extend([t] * n_img if t == image_token else [t])
        input_ids = torch.tensor([ids])
        state = torch.randn(1, 1, cfg.max_state_dim, generator=g) * 0.5
        noise = torch.randn(1, cfg.action_horizon, cfg.max_action_dim, generator=g)
        vl = BatchFeature(data={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                                "pixel_values": ip["pixel_values"].to(torch.float32), "image_grid_thw": grid})
        act_in = BatchFeature(data={"state": state, "embodiment_id": torch.tensor([a.embodiment])})
        randn = torch.randn
        torch.randn = lambda *x, **k: noise.clone()
        try:
            with torch.no_grad():
                bo = model.backbone(vl)
                pred = head.get_action(bo, act_in)["action_pred"]
        finally:
            torch.randn = randn
        tensors[f"{s}.pixels"] = (px.permute(2, 0, 1).to(torch.float32) / 255.0).contiguous()
        tensors[f"{s}.pixel_values"] = ip["pixel_values"].to(torch.float32).contiguous()
        tensors[f"{s}.input_ids"] = input_ids[0].to(torch.int32).contiguous()
        tensors[f"{s}.state"] = state.reshape(-1).contiguous()
        tensors[f"{s}.noise"] = noise.reshape(-1).contiguous()
        tensors[f"{s}.features"] = captured["f"][0].to(torch.float32).contiguous()
        tensors[f"{s}.actions"] = pred.reshape(-1).to(torch.float32).contiguous()
        meta["sizes"].append([h, w])
        print(s, (h, w), tuple(grid[0].tolist()), len(ids), pred.reshape(-1)[:4])
    hook.remove()
    save_file(tensors, str(out / "fixture.safetensors"))
    (out / "fixture.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
