"""Tiny video diffusion transformer fixtures: random weights, one forward
with a shared timestep and one with a per-token timestep (first frame clean)."""
import json
import os
import sys

import torch
from diffusers import WanTransformer3DModel
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
cfg = dict(patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=12, in_channels=8,
           out_channels=8, text_dim=32, freq_dim=256, ffn_dim=64, num_layers=2, cross_attn_norm=True,
           qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None,
           rope_max_seq_len=1024)
m = WanTransformer3DModel(**cfg).float().eval()
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.05)
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f"{out}/model.safetensors")
F, H, W, L = 3, 4, 6, 7
x = torch.randn(1, 8, F, H, W)
ctx = torch.randn(1, L, 32)
t = 637.5
seq = F * (H // 2) * (W // 2)
per = torch.full((1, seq), t)
per[0, : (H // 2) * (W // 2)] = 0.0
with torch.no_grad():
    o1 = m(x, torch.tensor([t]), ctx, return_dict=False)[0]
    o2 = m(x, per, ctx, return_dict=False)[0]
for name, v in [("latent", x), ("context", ctx), ("out_shared", o1), ("out_per_token", o2)]:
    v.float().contiguous().numpy().tofile(f"{out}/{name}.bin")
json.dump(dict(config=cfg, frames=F, height=H, width=W, text=L, t=t), open(f"{out}/meta.json", "w"))
print("ok", o1.abs().mean().item(), o2.abs().mean().item())
