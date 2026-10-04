"""Tiny action world-model session: the reference generation loop (memory
chosen by field of view, camera rays, actions, guidance and the flow UniPC
sampler) over three clips with random weights. Saves the starting noise and
the new latents of every clip. Run with the reference sources on PYTHONPATH
as for make_mg3_fixtures.py (the directory holding `wan` and `utils`)."""
import json
import os
import sys
import types

import numpy as np
import torch

torch.cuda.current_device = lambda: "cpu"
import diffusers.quantizers  # noqa: F401  (imported before the stubs below; it probes for them)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
for name in ("trimesh", "pandas", "torchvision", "torchvision.transforms"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["torchvision.transforms"].Lambda = None
sys.modules.setdefault("utils.conditions", types.ModuleType("utils.conditions")).Bench_actions_universal = None
sys.modules.setdefault("utils.transform", types.ModuleType("utils.transform")).get_video_transform = None
from wan.modules import model as ref  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def sdpa(q, k, v, *args, **kwargs):
    o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
    return o.transpose(1, 2).contiguous()


ref.flash_attention = sdpa
from wan.modules import action_module  # noqa: E402

action_module.flash_attn_ops = None
from utils.utils import compute_all_poses_from_actions, build_plucker_from_c2ws, build_plucker_from_pose  # noqa: E402
from utils.cam_utils import get_extrinsics, get_intrinsics, _interpolate_camera_poses_handedness, compute_relative_poses, select_memory_idx_fov  # noqa: E402
for name in ("trimesh", "pandas", "torchvision", "torchvision.transforms"):
    if getattr(sys.modules.get(name), "__spec__", 0) is None:
        del sys.modules[name]

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
KB, MO, Z, TEXT = 6, 2, 8, 512
action = dict(blocks=[0], enable_keyboard=True, enable_mouse=True, heads_num=2, hidden_size=4, img_hidden_size=24,
              keyboard_dim_in=KB, keyboard_hidden_dim=12, mouse_dim_in=MO, mouse_hidden_dim=12, mouse_qk_dim_list=[2, 2, 2],
              patch_size=[1, 2, 2], qk_norm=True, qkv_bias=False, rope_dim_list=[2, 2, 2], rope_theta=256,
              vae_time_compression_ratio=4, windows_size=3)
cfg = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=TEXT, in_dim=Z, dim=24, ffn_dim=48, freq_dim=256, text_dim=16,
           out_dim=Z, num_heads=2, num_layers=2, eps=1e-6, action_config=action, use_memory=True, sigma_theta=0.8)
m = ref.WanModel(**cfg).float().eval()
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.05)
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f"{out}/model.safetensors")

# Clips: a first clip of N1 new latent frames, then CLIPS - 1 clips of N2,
# each continuing the last 4 latent frames (the reference's 57 / 56 / 16
# pixel-frame schedule scaled down).
N1, N2, CLIPS, STEPS, SHIFT = 6, 2, 3, 4, 5.0
GUIDE = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
H, W, S = 4, 6, 16
first_clip_frame = 4 * N1 + 1
clip_frame = 4 * (N2 + 4)
past_frame = 16
frames = first_clip_frame + (CLIPS - 1) * (clip_frame - past_frame)
target_h, target_w = H * S, W * S
base_K = get_intrinsics(target_h, target_w)

keyboard = (torch.rand(frames, KB) > 0.6).float()
mouse = (torch.rand(frames, MO) - 0.5) * 0.4
keyboard[0] = 0.0
mouse[0] = 0.0
poses = compute_all_poses_from_actions(keyboard, mouse, first_pose=np.zeros(5))
rot = np.concatenate([np.zeros((frames, 1)), poses[:, 3:5]], axis=1).tolist()
extrinsics_all = get_extrinsics(rot, poses[:, :3].tolist())
keyboard_condition_all = keyboard.unsqueeze(0)
mouse_condition_all = mouse.unsqueeze(0)
cond = torch.randn(1, TEXT, 16)
neg_cond = torch.randn(1, TEXT, 16)
img_cond = torch.randn(1, Z, 1, H, W)
image_latent = img_cond.clone()
generator = torch.Generator().manual_seed(42)
hw = H * W // 4


def dump(name, t):
    t.detach().float().contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def run(x, t, conditions):
    seq = (x.shape[2] + (0 if conditions["x_memory"] is None else conditions["x_memory"].shape[2])) * hw
    return m(x=x, t=t, seq_len=seq, **conditions)[0].unsqueeze(0)


def align_frame_to_block(frame_idx):
    return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1


def get_latent_idx(frame_idx):
    return (frame_idx - 1) // 4 + 1


all_latents_list = []
memory_used = []
with torch.no_grad():
    for clip_idx in range(CLIPS):
        first_clip = clip_idx == 0
        current_end_frame_idx = first_clip_frame if first_clip else first_clip_frame + clip_idx * (clip_frame - past_frame)
        current_start_frame_idx = 0 if first_clip else current_end_frame_idx - clip_frame
        c2ws_chunk = extrinsics_all[current_start_frame_idx:current_end_frame_idx]
        src_indices = np.linspace(current_start_frame_idx, current_end_frame_idx - 1, first_clip_frame if first_clip else clip_frame)
        tgt_len = (first_clip_frame - 1) // 4 + 1 if first_clip else (clip_frame // 4)
        tgt_indices = np.linspace(0 if first_clip else current_start_frame_idx + 3, current_end_frame_idx - 1, tgt_len)
        plucker = build_plucker_from_c2ws(c2ws_chunk, src_indices, tgt_indices, framewise=True, base_K=base_K,
                                          target_h=target_h, target_w=target_w, lat_h=H, lat_w=W)
        plucker_no_mem = plucker
        if first_clip:
            x_memory = memory_mouse_condition = memory_keyboard_condition = latent_idx = timestep_memory = None
            memory_used.append([0])
        else:
            selected_index_base = [current_end_frame_idx - o for o in range(1, 34, 8)]
            selected_index = select_memory_idx_fov(extrinsics_all, current_start_frame_idx, selected_index_base, use_gpu=True)
            selected_index[-1] = 4
            memory_pluckers = []
            latent_idx = []
            for mem_idx, reference_idx in zip(selected_index, selected_index_base):
                latent_idx.append(get_latent_idx(mem_idx))
                a = align_frame_to_block(mem_idx)
                mem_block = extrinsics_all[a:a + 4]
                mem_pose = _interpolate_camera_poses_handedness(
                    src_indices=np.linspace(a, a + 3, mem_block.shape[0]), src_rot_mat=mem_block[:, :3, :3].cpu().numpy(),
                    src_trans_vec=mem_block[:, :3, 3].cpu().numpy(), tgt_indices=np.array([a + 3], dtype=np.float32))
                rel_pair = torch.cat([extrinsics_all[reference_idx:reference_idx + 1], mem_pose], dim=0)
                rel_pose = compute_relative_poses(rel_pair, framewise=False)[1:2]
                memory_pluckers.append(build_plucker_from_pose(rel_pose, base_K=base_K, target_h=target_h, target_w=target_w, lat_h=H, lat_w=W))
            plucker = torch.cat(memory_pluckers + [plucker], dim=2)
            src = torch.cat(all_latents_list, dim=2)
            x_memory = src[:, :, latent_idx]
            memory_mouse_condition = torch.ones((1, len(selected_index), MO))
            memory_keyboard_condition = -torch.ones((1, len(selected_index), KB))
            timestep_memory = x_memory.new_zeros((1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4))
            latent_start = get_latent_idx(current_start_frame_idx)
            memory_used.append([int(i) for i in latent_idx] + list(range(latent_start, latent_start + 4)))

        keyboard_condition = keyboard_condition_all[:, current_start_frame_idx:current_end_frame_idx]
        mouse_condition = mouse_condition_all[:, current_start_frame_idx:current_end_frame_idx]
        plucker = plucker.float()
        plucker_no_mem = plucker_no_mem.float()
        sched = FlowUniPCMultistepScheduler()
        sched.set_timesteps(STEPS, device="cpu", shift=SHIFT)
        latent_start_idx = get_latent_idx(current_start_frame_idx)
        latent_end_idx = get_latent_idx(current_end_frame_idx)
        latents = torch.randn((1, Z, latent_end_idx - latent_start_idx, H, W), generator=generator)
        dump(f"noise{clip_idx}", latents[0])
        latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2]:]], dim=2)
        full = dict(mouse_cond=mouse_condition, keyboard_cond=keyboard_condition, context=cond, plucker_emb=plucker, x_memory=x_memory,
                    timestep_memory=timestep_memory, keyboard_cond_memory=memory_keyboard_condition, mouse_cond_memory=memory_mouse_condition,
                    memory_latent_idx=latent_idx, predict_latent_idx=(latent_start_idx, latent_end_idx))
        null = dict(mouse_cond=torch.ones_like(mouse_condition), keyboard_cond=-torch.ones_like(keyboard_condition), context=neg_cond,
                    plucker_emb=plucker_no_mem, x_memory=None, timestep_memory=None, keyboard_cond_memory=None, mouse_cond_memory=None,
                    memory_latent_idx=None, predict_latent_idx=(latent_start_idx, latent_end_idx))
        for t in sched.timesteps:
            timestep = latents.new_full((latents.shape[2], latents.shape[3] * latents.shape[4] // 4), float(t))
            timestep[:img_cond.shape[2]].zero_()
            timestep = timestep.flatten().unsqueeze(0)
            noise_pred_full = run(latents, timestep, full)
            noise_pred_null = run(latents, timestep, null)
            noise_pred = noise_pred_null + GUIDE * (noise_pred_full - noise_pred_null)
            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]
            latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2]:]], dim=2)
        new = latents[:, :, img_cond.shape[2]:]
        dump(f"out{clip_idx}", new[0])
        img_cond = latents[:, :, -4:]
        all_latents_list.append(latents if first_clip else new)

dump("context", cond)
dump("negative", neg_cond)
dump("image_latent", image_latent[0])
dump("actions", torch.cat([keyboard, mouse], dim=1))
c = dict(cfg)
c.pop("patch_size")
json.dump(dict(config=c, height=H, width=W, clips=CLIPS, first_frames=N1, next_frames=N2, steps=STEPS, guidance=GUIDE, shift=SHIFT,
               memory=memory_used), open(f"{out}/meta.json", "w"))
print("ok", [round(float(x.abs().mean()), 4) for x in all_latents_list])
