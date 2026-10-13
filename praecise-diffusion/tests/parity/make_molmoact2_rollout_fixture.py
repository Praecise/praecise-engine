"""Record a reference MolmoAct2 closed-loop rollout through the lerobot policy.

A random-weight small model (the text, vision and expert configs of
make_molmoact2_action_fixture.py) is written as a release: config.json,
model.safetensors with every key the reference loads strictly, the release
tokenizer, processor config and chat template, and a norm_stats.json with
one embodiment of random q01/q99 statistics. The lerobot MolmoAct2 policy
then loads it by norm tag, in float32, with its own pre- and post-processor
pipelines (state frame transform, masked quantile normalisation and clamp,
prompt and image packing; action clamp, masked unnormalisation, inverse frame
transform), and runs its `select_action` loop: one action per tick, a new
chunk when the executed steps run out, the same seeded noise every chunk.

The default embodiment has two cameras, a gripper value passed through
unnormalised (its mask entry is false) and a chunk of 12 of which 4 are
executed; one state value has equal quantiles. `--alt` uses one camera, the
gripper normalised (`normalize_gripper`), the release's full horizon with 7
executed, a joint sign/offset transform, the expert's other options and the
continuous action mode.

Per chunk the fixture keeps the prompt ids, the normalised state, the crops
and the chunk the policy predicted (normalised, before the post-processor);
per tick the frames, the state and the executed action in robot units.

Usage:
  make_molmoact2_rollout_fixture.py <lerobot/src> <release dir> <out> [--alt] [--ticks N]
"""

import argparse
import json
import shutil
import sys
import types
from pathlib import Path

TEXT = {"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 3, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "hidden_act": "silu", "layer_norm_eps": 1e-6,
        "vocab_size": 154624, "additional_vocab_size": 128, "max_position_embeddings": 4096,
        "qkv_bias": False, "use_qk_norm": True, "qk_norm_type": "qwen3", "norm_after": False,
        "rope_theta": 5000000.0, "rope_parameters": {"rope_theta": 5000000.0, "rope_type": "default"},
        "rope_scaling_layers": None, "attn_implementation": "sdpa"}
VIT = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 2,
       "num_key_value_heads": 2, "head_dim": 16, "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
       "image_default_input_size": [378, 378], "image_patch_size": 14, "image_num_pos": 729,
       "attn_implementation": "sdpa", "float32_attention": True}
ADAPTER = {"vit_layers": [-1, -2], "pooling_attention_mask": True, "hidden_size": 32, "num_attention_heads": 2,
           "num_key_value_heads": 2, "head_dim": 16, "hidden_act": "silu", "intermediate_size": 48,
           "text_hidden_size": 64, "attn_implementation": "sdpa", "float32_attention": True}
EXPERT = {"hidden_size": 48, "num_layers": 3, "num_heads": 3, "mlp_ratio": 2.5, "ffn_multiple_of": 64,
          "timestep_embed_dim": 32, "dropout": 0.0, "attn_dropout": 0.0, "context_layer_norm": True,
          "qk_norm": True, "qk_norm_eps": 1e-6, "rope": True, "causal_attn": False}
ALT_EXPERT = {"context_layer_norm": False, "qk_norm": False, "rope": False}
EMBODIMENTS = {
    "default": {"tag": "tiny_arm", "frames": [(96, 128), (240, 320)],
                "names": [f"joint_{i}" for i in range(7)] + ["gripper"], "mask": True, "normalize_gripper": False,
                "horizon": 12, "n_action_steps": 4, "setup": "single arm on a tabletop",
                "control": "absolute joint pose", "joint_signs": None, "joint_offsets": None, "seed": 21,
                "instruction": "Task: Pick up the red cube.  Then put it in the bowl!"},
    "alt": {"tag": "tiny_so", "frames": [(480, 640)],
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
            "mask": False, "normalize_gripper": True, "horizon": None, "n_action_steps": 7,
            "setup": "single so100/so101 robotic arm", "control": "absolute joint pose",
            "joint_signs": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0], "joint_offsets": [0.0, 1.5708, -1.5708, 0.0, 0.0, 0.0],
            "seed": 22, "instruction": "stack the blocks"},
}


def light_policies_package(src):
    """Register `lerobot.policies` without running its `__init__`, which
    imports every policy and the simulation environments."""
    import lerobot

    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(Path(src) / "lerobot" / "policies")]
    sys.modules["lerobot.policies"] = pkg
    lerobot.policies = pkg


def stats(gen, names, mask, flat=None):
    """Random q01/q99 (and min/max/mean/std) statistics; a gripper outside
    the mask keeps its range inside [-1, 1] as the reference requires."""
    import torch

    d = len(names)
    centre = torch.randn(d, generator=gen) * 0.8
    half = 0.2 + torch.rand(d, generator=gen) * 1.3
    q01, q99 = centre - half, centre + half
    if flat is not None:
        q99[flat] = q01[flat]
    out = {"q01": q01, "q99": q99, "min": q01 - 0.1, "max": q99 + 0.1, "mean": centre, "std": half / 2}
    if mask:
        for k in out:
            out[k][-1] = {"q01": 0.0, "q99": 1.0, "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.4}[k]
    out = {k: [float(x) for x in v] for k, v in out.items()}
    out["count"] = [1000.0]
    out["names"] = list(names)
    if mask:
        out["mask"] = ["gripper" not in n for n in names]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot_src")
    ap.add_argument("release")
    ap.add_argument("out")
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--ticks", type=int, default=10)
    a = ap.parse_args()

    light_policies_package(a.lerobot_src)
    import torch
    from safetensors.torch import save_file

    import lerobot.policies.molmoact2.molmoact2_hf_model.configuration_molmoact2 as c
    import lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 as m
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors

    e = EMBODIMENTS["alt" if a.alt else "default"]
    rel = Path(a.release)
    release = json.loads((rel / "config.json").read_text())
    expert = dict(EXPERT, **(ALT_EXPERT if a.alt else {}))
    mode = "continuous" if a.alt else release["action_mode"]
    top = {k: v for k, v in release.items() if not isinstance(v, dict) and k not in ("architectures", "auto_map")}
    top.update(add_action_expert=True, action_mode=mode)
    cfg = c.MolmoAct2Config(text_config=TEXT, vit_config=VIT, adapter_config=ADAPTER,
                            action_expert_config=dict(expert), **top)
    for sub in (cfg, cfg.text_config, cfg.vit_config, cfg.adapter_config):
        sub._attn_implementation = "sdpa"

    torch.manual_seed(0)
    model = m.MolmoAct2ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_f.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.1)
            elif name.startswith("model.action_expert."):
                p.copy_(torch.randn_like(p) * 0.15)
            else:
                p.copy_(torch.randn_like(p) * 0.08)

    out = Path(a.out)
    pol = out / "policy"
    pol.mkdir(parents=True, exist_ok=True)
    config = dict(release, text_config=TEXT, vit_config=VIT, adapter_config=ADAPTER, action_expert_config=expert,
                  add_action_expert=True, action_mode=mode)
    (pol / "config.json").write_text(json.dumps(config, indent=1))
    for f in ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (rel / f).exists():
            shutil.copy(rel / f, pol / f)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(pol / "model.safetensors"))

    gen = torch.Generator().manual_seed(e["seed"])
    names = e["names"]
    cams = [f"observation.images.cam{i}" for i in range(len(e["frames"]))]
    horizon = e["horizon"] or cfg.max_action_horizon
    meta = {"action_key": "action", "state_key": "observation.state", "camera_keys": cams,
            "normalize_gripper": e["normalize_gripper"], "action_horizon": horizon,
            "n_action_steps": e["n_action_steps"], "setup_type": e["setup"], "control_mode": e["control"],
            "action_stats": stats(gen, names, e["mask"]),
            "state_stats": stats(gen, names, e["mask"], flat=None if a.alt else 2)}
    (pol / "norm_stats.json").write_text(json.dumps(
        {"format": "molmoact2_norm_stats.v1", "norm_mode": "q01_q99", "metadata_by_tag": {e["tag"]: meta}}, indent=1))

    d = len(names)
    features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for k, (h, w) in zip(cams, e["frames"])}
    features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(d,))
    pcfg = MolmoAct2Config(
        checkpoint_path=str(pol), norm_tag=e["tag"], device="cpu", dtype=torch.float32, train_mode_vlm="fft",
        compile_model=False, enable_inference_cuda_graph=False, normalize_gripper=e["normalize_gripper"],
        joint_signs=e["joint_signs"], joint_offsets=e["joint_offsets"], input_features=features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(d,))})
    policy = MolmoAct2Policy(pcfg).to(torch.float32).eval()
    pre, post = make_molmoact2_pre_post_processors(pcfg)
    assert (pcfg.chunk_size, pcfg.n_action_steps) == (horizon, e["n_action_steps"])

    seed = 1000 + e["seed"]
    noise = torch.randn((1, horizon, cfg.max_action_dim), generator=torch.Generator().manual_seed(seed))
    chunks = []
    predict = policy.predict_action_chunk

    def recording(batch, **kw):
        actions = predict(batch, **kw)
        valid = batch["attention_mask"][0].bool()
        chunks.append({"ids": batch["input_ids"][0][valid].float(), "state": batch["observation.state"][0].float(),
                       "pixels": batch["pixel_values"].float(), "raw": actions[0].float()})
        return actions

    policy.predict_action_chunk = recording
    policy.reset()
    tensors = {"noise": noise.reshape(-1).contiguous()}
    states, actions, chunk_ticks = [], [], []
    q01 = torch.tensor(meta["state_stats"]["q01"])
    q99 = torch.tensor(meta["state_stats"]["q99"])
    for t in range(a.ticks):
        obs = {"task": e["instruction"]}
        for i, (k, (h, w)) in enumerate(zip(cams, e["frames"])):
            img = torch.rand(3, h, w, generator=gen)
            tensors[f"frame.{t}.{i}"] = img.contiguous()
            obs[k] = img
        # Around the quantile range, some values beyond it (clamped).
        u = torch.rand(d, generator=gen) * 1.4 - 0.2
        state = q01 + u * (q99 - q01)
        if e["mask"]:
            state[-1] = float(torch.rand(1, generator=gen))
        if e["joint_signs"] is not None:
            # Robot frame: the inverse of the model-frame transform.
            sg, off = torch.tensor(e["joint_signs"]), torch.tensor(e["joint_offsets"])
            state = sg * (state - off)
        states.append(state)
        obs["observation.state"] = state
        before = len(chunks)
        batch = pre(obs)
        with torch.no_grad():
            act = policy.select_action(batch, generator=torch.Generator().manual_seed(seed))
        if len(chunks) > before:
            chunk_ticks.append(t)
        act = post(act)
        actions.append(act.reshape(-1).to(torch.float32))
    tensors["states"] = torch.stack(states).contiguous()
    tensors["actions"] = torch.stack(actions).contiguous()
    for k, ch in enumerate(chunks):
        for name, v in ch.items():
            tensors[f"chunk.{k}.{name}"] = v.contiguous()
    save_file(tensors, str(out / "fixture.safetensors"))
    fx = {"tag": e["tag"], "ticks": a.ticks, "cameras": len(cams), "camera_keys": cams, "frames": e["frames"],
          "instruction": e["instruction"], "horizon": horizon, "n_action_steps": e["n_action_steps"],
          "normalize_gripper": e["normalize_gripper"], "joint_signs": e["joint_signs"],
          "joint_offsets": e["joint_offsets"], "action_mode": mode, "expert": expert,
          "chunks": [{"tick": t, "tokens": int(ch["ids"].numel())} for t, ch in zip(chunk_ticks, chunks)],
          "torch": torch.__version__}
    import platform

    fx["machine"] = platform.machine()
    (out / "fixture.json").write_text(json.dumps(fx, indent=1))
    for k, (t, ch) in enumerate(zip(chunk_ticks, chunks)):
        print(f"chunk {k} at tick {t}: {ch['ids'].numel()} tokens, |raw| {ch['raw'].abs().max():.3f}, "
              f"clamped {(ch['raw'].abs() > 1).float().mean():.2f}")
    print("actions", tensors["actions"][:, :3])
    print("wrote", out)


if __name__ == "__main__":
    main()
