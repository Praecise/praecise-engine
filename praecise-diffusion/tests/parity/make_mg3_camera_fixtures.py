"""Camera-ray fixtures for the action world model: a random keyboard and
mouse path integrated into poses, the rays of a first clip, of a continuing
clip and of one memory frame. Run with the reference sources on PYTHONPATH
(the directory holding `utils`)."""
import json
import os
import sys
import types

import numpy as np
import torch

for name in ("trimesh", "pandas"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules.setdefault("utils.conditions", types.ModuleType("utils.conditions")).Bench_actions_universal = None
sys.modules.setdefault("utils.transform", types.ModuleType("utils.transform")).get_video_transform = None
from utils.utils import compute_all_poses_from_actions, build_plucker_from_c2ws, build_plucker_from_pose  # noqa: E402
from utils.cam_utils import get_extrinsics, get_intrinsics, _interpolate_camera_poses_handedness, compute_relative_poses  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
frames, lat_h, lat_w, s = 29, 2, 3, 2
keyboard = (torch.rand(frames, 6) > 0.6).float()
mouse = (torch.rand(frames, 2) - 0.5) * 0.4
poses = compute_all_poses_from_actions(keyboard, mouse, first_pose=np.zeros(5))
rot = np.concatenate([np.zeros((frames, 1)), poses[:, 3:5]], axis=1).tolist()
ext = get_extrinsics(rot, poses[:, :3].tolist())
K = get_intrinsics(lat_h * s, lat_w * s)


def dump(name, t):
    t.detach().float().contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def clip(start, end, first):
    n = end - start
    src = np.linspace(start, end - 1, n)
    tgt_len = (n - 1) // 4 + 1 if first else n // 4
    tgt = np.linspace(0 if first else start + 3, end - 1, tgt_len)
    return build_plucker_from_c2ws(ext[start:end], src, tgt, True, K, lat_h * s, lat_w * s, lat_h, lat_w)[0], tgt_len


rays1, n1 = clip(0, 13, True)
rays2, n2 = clip(9, 21, False)
mem, ref_idx = 5, 20
a = (mem - 1) // 4 * 4 + 1
block = ext[a:a + 4]
pose = _interpolate_camera_poses_handedness(np.linspace(a, a + 3, 4), block[:, :3, :3].numpy(), block[:, :3, 3].numpy(), np.array([a + 3], dtype=np.float32))
rel = compute_relative_poses(torch.cat([ext[ref_idx:ref_idx + 1], pose.double()], 0), framewise=False)[1:2]
mrays = build_plucker_from_pose(rel, K, lat_h * s, lat_w * s, lat_h, lat_w)[0]
dump("keyboard", keyboard)
dump("mouse", mouse)
dump("poses", torch.from_numpy(poses))
dump("rays1", rays1)
dump("rays2", rays2)
dump("mem_rays", mrays)
json.dump(dict(frames=frames, lat_h=lat_h, lat_w=lat_w, s=s, clip1=[0, 13, n1], clip2=[9, 21, n2], memory=[mem, ref_idx]), open(os.path.join(out, "meta.json"), "w"))
