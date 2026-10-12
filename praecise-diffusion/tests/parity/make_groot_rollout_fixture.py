"""Record a reference GR00T N1.7 control loop through the policy's own
processors, for a random-weight small model.

The small backbone and policy are those of make_groot_fixture.py (with a
wider state and action), saved as a raw N1.7 release with the processor
files a release carries: processor_config.json (image geometry, percentile
normalisation, relative actions and the embodiment's modality layout),
statistics.json and embodiment_id.json. Each control tick sends camera
frames in [0, 1], the measured state and the instruction through the
reference preprocessor (camera packing, image crop and resize, language
formatting, the chat prompt, the Qwen3-VL image processor, state
normalisation), predicts a chunk with fixed noise, decodes the whole chunk
with the reference postprocessor (relative end-effector and joint actions
against the state of that tick) and queues its first actions, as the
reference runtime does for releases with relative actions. The fixture keeps
the frames, states and executed actions, and for every chunk the model
inputs, the raw prediction and the decoded chunk.

Usage:
  make_groot_rollout_fixture.py <lerobot/src> <qwen3-vl dir> <out>
      [--ticks N] [--n-action-steps K] [--size HxW] [--torch-images]

The reference packs the cameras into one array, so they share a size.
"""

import argparse
import json
import platform
import shutil
import sys
import types
from pathlib import Path

TAG = "small_arm_relative_eef_relative_joint"
EMBODIMENT = 1
STATE_KEYS = [("eef_9d", 9), ("gripper_position", 1), ("joint_position", 6)]
ACTION_CONFIGS = [
    {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "eef_9d"},
    {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "gripper_position"},
    {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "joint_position"},
]
ACTION_STEPS = 6
CAMERAS = ["exterior_image_1_left", "wrist_image_left"]
INSTRUCTION = "Pick up the RED cube, then place it in the bowl!"


def stats_entry(torch, g, dim, rows=None):
    shape = (rows, dim) if rows else (dim,)
    lo = torch.randn(shape, generator=g) * 0.5 - 1.0
    hi = lo + 0.5 + torch.rand(shape, generator=g) * 2.0
    q01 = lo + 0.1 * (hi - lo)
    q99 = hi - 0.1 * (hi - lo)
    mean = (lo + hi) / 2
    std = (hi - lo) / 4
    e = {"min": lo, "max": hi, "q01": q01, "q99": q99, "mean": mean, "std": std}
    return {k: v.tolist() for k, v in e.items()}


def rotation6d(torch, g):
    q = torch.randn(4, generator=g, dtype=torch.float64)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    m = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]]
    return [v for row in m for v in row]


def frame(torch, g, h, w, t):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    base = torch.stack([0.5 + 0.4 * torch.sin(xx / (9 + c) + yy / 13 + 0.3 * t + c) for c in range(3)])
    px = (base + 0.15 * torch.rand((3, h, w), generator=g)).clamp(0, 1)
    return (px * 255).round().to(torch.uint8).to(torch.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot")
    ap.add_argument("qwen3vl")
    ap.add_argument("out")
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=4)
    ap.add_argument("--size", default="180x320")
    ap.add_argument("--torch-images", action="store_true", help="the default (non-albumentations) image transform")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    sys.path.insert(0, a.lerobot)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_vla_rollout_fixtures import light_policies_package

    light_policies_package(a.lerobot)
    import torch
    from safetensors.torch import save_file
    from transformers.feature_extraction_utils import BatchFeature

    import make_groot_fixture as small
    from lerobot.policies.groot import processor_groot
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    out = Path(a.out)
    backbone = out / "backbone"
    shutil.copytree(a.qwen3vl, backbone, dirs_exist_ok=True)
    bc = json.loads((backbone / "config.json").read_text())
    bc["text_config"].update(small.TEXT)
    bc["text_config"]["rope_scaling"]["mrope_section"] = small.MROPE
    bc["vision_config"].update(small.VISION)
    bc["dtype"] = "float32"
    (backbone / "config.json").write_text(json.dumps(bc, indent=1))

    head = dict(small.HEAD, max_state_dim=20, max_action_dim=20, max_seq_len=1024)
    torch.manual_seed(a.seed)
    cfg = GR00TN17Config(model_name=str(backbone), **head)
    model = GR00TN17(cfg, transformers_loading_kwargs={}, load_backbone_weights=False).to(torch.float32).eval()
    small.randomize(model, torch)
    policy = out / "policy"
    model.save_pretrained(policy, safe_serialization=True)

    g = torch.Generator().manual_seed(a.seed)
    stats = {"state": {}, "action": {}, "relative_action": {}}
    for k, d in STATE_KEYS:
        stats["state"][k] = stats_entry(torch, g, d)
        stats["action"][k] = stats_entry(torch, g, d)
    # Per-step statistics for the end effector, one row for the joints.
    stats["relative_action"]["eef_9d"] = stats_entry(torch, g, 9, rows=ACTION_STEPS)
    stats["relative_action"]["joint_position"] = stats_entry(torch, g, 6)
    keys = [k for k, _ in STATE_KEYS]
    modality = {
        "video": {"delta_indices": [-15, 0], "modality_keys": CAMERAS},
        "state": {"delta_indices": [0], "modality_keys": keys},
        "action": {"delta_indices": list(range(ACTION_STEPS)), "modality_keys": keys, "action_configs": ACTION_CONFIGS},
        "language": {"delta_indices": [0], "modality_keys": ["annotation.language.language_instruction"]},
    }
    kwargs = {
        "use_percentiles": True, "use_mean_std": False, "image_crop_size": [230, 230], "image_target_size": [256, 256],
        "formalize_language": True, "max_state_dim": 20, "max_action_dim": 20, "letter_box_transform": False,
        "state_dropout_prob": 0.2, "use_albumentations": not a.torch_images, "shortest_image_edge": 256,
        "crop_fraction": 0.95, "max_action_horizon": head["action_horizon"], "use_relative_action": True,
        "modality_configs": {TAG: modality},
    }
    (policy / "processor_config.json").write_text(json.dumps({"processor_kwargs": kwargs}, indent=1))
    (policy / "statistics.json").write_text(json.dumps({TAG: stats}, indent=1))
    (policy / "embodiment_id.json").write_text(json.dumps({TAG: EMBODIMENT, "new_embodiment": 3}, indent=1))

    # The processors read the backbone's tokenizer and image processor from
    # the release's backbone name; point it at the small backbone.
    processor_groot.GROOT_N1_7_BACKBONE_MODEL = str(backbone)
    env_dim = sum(d for _, d in STATE_KEYS)
    config = types.SimpleNamespace(
        base_model_path=str(policy), embodiment_tag=TAG, use_relative_actions=False, chunk_size=head["action_horizon"],
        max_state_dim=20, max_action_dim=20, output_features={"action": types.SimpleNamespace(shape=(env_dim,))},
        device="cpu", action_decode_transform=None, relative_exclude_joints=[], _runtime_dataset_meta=None)
    pre, post = processor_groot.make_groot_pre_post_processors(config)
    assets = processor_groot._load_n1_7_checkpoint_processor_assets(config)
    queue_steps = min(a.n_action_steps, ACTION_STEPS)
    h, w = (int(v) for v in a.size.split("x"))

    noise = torch.randn(1, head["action_horizon"], head["max_action_dim"], generator=g)
    eef = torch.tensor([0.3, -0.1, 0.4] + rotation6d(torch, g))
    joints = torch.randn(6, generator=g) * 0.5
    tensors, states, executed, chunks = {}, [], [], []
    queue = []
    inputs_keys = {"state", "action_mask", "embodiment_id", "input_ids", "attention_mask", "pixel_values",
                   "image_grid_thw", "mm_token_type_ids"}
    for t in range(a.ticks):
        drift = torch.randn(16, generator=g) * 0.02
        eef = torch.cat([eef[:3] + drift[:3], torch.tensor(rotation6d(torch, g)) if t % 3 == 0 else eef[3:]])
        grip = torch.tensor([0.5 + 0.4 * (t % 2)])
        joints = joints + drift[10:16]
        state = torch.cat([eef, grip, joints]).to(torch.float32)
        states.append(state)
        batch = {"observation.state": state[None].clone(), "task": INSTRUCTION}
        for c, name in enumerate(CAMERAS):
            px = frame(torch, g, h, w, t)
            tensors[f"frame.{t}.{c}"] = px.contiguous()
            batch[f"observation.images.{name}"] = px[None].clone()
        if not queue:
            k = len(chunks)
            enc = pre(batch)
            vl = {key: v for key, v in enc.items() if key in inputs_keys}
            randn = torch.randn
            torch.randn = lambda *x, **kw: noise.clone()
            try:
                with torch.no_grad():
                    raw = model.get_action(BatchFeature(data=vl))["action_pred"]
            finally:
                torch.randn = randn
            horizon = min(raw.shape[1], ACTION_STEPS, head["action_horizon"], a.n_action_steps)
            cut = raw[:, :horizon, :env_dim]
            decoded = post(cut)
            if isinstance(decoded, dict):
                decoded = decoded["action"]
            tensors[f"chunk.{k}.input_ids"] = enc["input_ids"][0].to(torch.int32).contiguous()
            tensors[f"chunk.{k}.pixel_values"] = enc["pixel_values"].to(torch.float32).contiguous()
            tensors[f"chunk.{k}.grid"] = enc["image_grid_thw"].to(torch.int32).contiguous()
            tensors[f"chunk.{k}.state"] = enc["state"].reshape(-1).to(torch.float32).contiguous()
            tensors[f"chunk.{k}.raw"] = raw.reshape(-1).to(torch.float32).contiguous()
            tensors[f"chunk.{k}.decoded"] = decoded.reshape(-1).to(torch.float32).contiguous()
            chunks.append({"tick": t, "horizon": horizon, "tokens": len(enc["input_ids"][0]),
                           "grids": enc["image_grid_thw"].tolist()})
            queue = list(decoded[0, :queue_steps])
            print("chunk", k, "tick", t, "tokens", len(enc["input_ids"][0]), "grids", enc["image_grid_thw"].tolist(),
                  "decoded", decoded[0, 0, :4].tolist())
        executed.append(queue.pop(0).to(torch.float32))
    tensors["states"] = torch.stack(states).reshape(-1).contiguous()
    tensors["actions"] = torch.stack(executed).reshape(-1).contiguous()
    tensors["noise"] = noise.reshape(-1).contiguous()
    meta = {
        "ticks": a.ticks, "cameras": len(CAMERAS), "n_action_steps": a.n_action_steps, "instruction": INSTRUCTION,
        "embodiment_tag": TAG, "action_dim": env_dim, "chunks": chunks, "torch_images": a.torch_images,
        "geometry": {"crop_fraction": assets.crop_fraction, "shortest_image_edge": assets.shortest_image_edge,
                     "use_albumentations": assets.use_albumentations},
        "torch": torch.__version__,
        # OpenCV's area decimation rounds differently on aarch64 and x86-64.
        "machine": platform.machine(),
    }
    save_file(tensors, str(out / "fixture.safetensors"))
    (out / "fixture.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
