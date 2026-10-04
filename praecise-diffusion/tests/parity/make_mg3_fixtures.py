"""Tiny action-conditioned world-model fixtures: random weights in the
original layout, one forward over a fresh clip and one continuing from a
memory frame. Run with the reference sources on PYTHONPATH (the directory
holding the `wan` package, its package `__init__` files emptied and its
hard `torch.bfloat16` casts in `modules/model.py` turned into `torch.float32`
so the reference runs in float32)."""
import json
import os
import sys

import torch

torch.cuda.current_device = lambda: "cpu"
from wan.modules import model as ref  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def sdpa(q, k, v, *args, **kwargs):
    o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
    return o.transpose(1, 2).contiguous()


ref.flash_attention = sdpa
from wan.modules import action_module  # noqa: E402

action_module.flash_attn_ops = None

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
action = dict(blocks=[0], enable_keyboard=True, enable_mouse=True, heads_num=2, hidden_size=4, img_hidden_size=24,
              keyboard_dim_in=2, keyboard_hidden_dim=12, mouse_dim_in=2, mouse_hidden_dim=12, mouse_qk_dim_list=[2, 2, 2],
              patch_size=[1, 2, 2], qk_norm=True, qkv_bias=False, rope_dim_list=[2, 2, 2], rope_theta=256,
              vae_time_compression_ratio=4, windows_size=3)
L = 7
cfg = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=L, in_dim=8, dim=24, ffn_dim=48, freq_dim=256, text_dim=16,
           out_dim=8, num_heads=2, num_layers=2, eps=1e-6, action_config=action, use_memory=True, sigma_theta=0.8)
m = ref.WanModel(**cfg).float().eval()
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.05)
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f"{out}/model.safetensors")

F, H, W = 3, 4, 6
hw = (H // 2) * (W // 2)
N = 4 * (F - 1) + 1
x = torch.randn(1, 8, F, H, W)
ctx = torch.randn(1, L, 16)
kb = torch.randn(1, N, 2)
mouse = torch.randn(1, N, 2)
rays = torch.randn(1, 6 * 256, F, H, W) * 0.1
t = 637.5
per = torch.full((1, F * hw), t)
per[0, :hw] = 0.0
with torch.no_grad():
    o1 = m(x, per, ctx, F * hw, mouse_cond=mouse, keyboard_cond=kb, plucker_emb=rays)[0]
# Continuation: one memory frame at latent index 5, a 2-frame chunk at 6..8.
Fp = 2
xm = torch.randn(1, 8, 1, H, W)
xp = torch.randn(1, 8, Fp, H, W)
kbp = torch.randn(1, 4 * Fp, 2)
mop = torch.randn(1, 4 * Fp, 2)
kbm = torch.randn(1, 1, 2)
mom = torch.randn(1, 1, 2)
rays2 = torch.randn(1, 6 * 256, 1 + Fp, H, W) * 0.1
with torch.no_grad():
    o2 = m(xp, torch.full((1, Fp * hw), t), ctx, (1 + Fp) * hw, mouse_cond=mop, keyboard_cond=kbp, x_memory=xm,
           timestep_memory=torch.zeros(1, hw), mouse_cond_memory=mom, keyboard_cond_memory=kbm, plucker_emb=rays2,
           memory_latent_idx=[5], predict_latent_idx=(6, 6 + Fp))[0]
for name, v in [("latent", x), ("context", ctx), ("keyboard", kb), ("mouse", mouse), ("rays", rays), ("out", o1),
                ("mem_latent", xm), ("pred_latent", xp), ("pred_keyboard", kbp), ("pred_mouse", mop),
                ("mem_keyboard", kbm), ("mem_mouse", mom), ("rays2", rays2), ("out_mem", o2)]:
    v.float().contiguous().numpy().tofile(f"{out}/{name}.bin")
c = dict(cfg)
c.pop("patch_size")
json.dump(dict(config=c, frames=F, pred_frames=Fp, height=H, width=W, text=L, t=t, memory_index=5, pred_start=6),
          open(f"{out}/meta.json", "w"))
print("ok", o1.abs().mean().item(), o2.abs().mean().item())
