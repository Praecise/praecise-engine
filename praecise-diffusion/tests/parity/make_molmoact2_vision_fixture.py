"""Record reference MolmoAct2 image-token features for a random-weight small backbone.

The MolmoAct2 vision backbone (ViT, multi-layer feature concatenation,
pooling attention and projector) is built from a small config with random
weights, in float32. For each sample, random processor-range crops and
pooling indices (some slots padded, one token entirely padding) go through
its forward. The fixture keeps the policy dir (config.json in the release
layout, model.safetensors under the release's key prefix), every input and
the pooled features of the tokens the reference keeps.

The modeling module is loaded straight from the lerobot tree, without
importing the lerobot package.

Usage:
  make_molmoact2_vision_fixture.py <lerobot/src> <out> [--samples N] [--no-mask]
"""

import argparse
import json
import sys
import types
from pathlib import Path

VIT = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 6, "num_attention_heads": 4,
       "num_key_value_heads": 4, "head_dim": 16, "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
       "image_default_input_size": [56, 56], "image_patch_size": 14, "image_num_pos": 16,
       "attn_implementation": "sdpa", "float32_attention": True}
ADAPTER = {"vit_layers": [-2, -4], "pooling_attention_mask": True, "hidden_size": 48, "num_attention_heads": 3,
           "num_key_value_heads": 3, "head_dim": 16, "hidden_act": "silu", "intermediate_size": 96,
           "text_hidden_size": 80, "attn_implementation": "sdpa", "float32_attention": True}
POOL = 4
TOKENS = 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot_src")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args()

    d = Path(a.lerobot_src) / "lerobot/policies/molmoact2/molmoact2_hf_model"
    pkg = types.ModuleType("m2hf")
    pkg.__path__ = [str(d)]
    sys.modules["m2hf"] = pkg
    import torch
    from safetensors.torch import save_file

    import m2hf.configuration_molmoact2 as c
    import m2hf.modeling_molmoact2 as m

    torch.manual_seed(0)
    adapter = dict(ADAPTER, pooling_attention_mask=not a.no_mask)
    vit_cfg = c.MolmoAct2VitConfig(**VIT)
    ad_cfg = c.MolmoAct2AdapterConfig(**adapter)
    vit_cfg._attn_implementation = "sdpa"
    ad_cfg._attn_implementation = "sdpa"
    bb = m.MolmoAct2VisionBackbone(vit_cfg, ad_cfg).float().eval()
    with torch.no_grad():
        for p in bb.parameters():
            p.copy_(torch.randn_like(p) * 0.08)
        for name, p in bb.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.1)

    out = Path(a.out)
    (out / "policy").mkdir(parents=True, exist_ok=True)
    config = {"model_type": "molmoact2", "vit_config": VIT, "adapter_config": adapter}
    (out / "policy/config.json").write_text(json.dumps(config, indent=1))
    save_file({f"model.vision_backbone.{k}": v.contiguous() for k, v in bb.state_dict().items()},
              str(out / "policy/model.safetensors"))

    n = vit_cfg.image_num_patch[0] * vit_cfg.image_num_patch[1]
    pl = vit_cfg.image_patch_size ** 2 * 3
    tensors, crops = {}, []
    for s in range(a.samples):
        k = 2 + s % 2
        px = torch.rand(1, k, n, pl) * 2 - 1
        idx = torch.randint(0, k * n, (1, TOKENS, POOL))
        idx[0, 1, 2:] = -1
        idx[0, 4, 0] = -1
        idx[0, 6, :] = -1
        with torch.no_grad():
            feats = bb(px, idx)
        tensors[f"{s}.pixels"] = px[0].contiguous()
        tensors[f"{s}.pooling"] = idx[0].float().contiguous()
        tensors[f"{s}.features"] = feats.float().contiguous()
        crops.append(k)
        print(f"sample {s}: crops {k}, features {tuple(feats.shape)}, |max| {feats.abs().max():.3f}")
    save_file(tensors, str(out / "fixture.safetensors"))
    meta = {"samples": a.samples, "crops": crops, "pool": POOL, "tokens": TOKENS, "pooling_mask": not a.no_mask,
            "torch": torch.__version__}
    (out / "fixture.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
