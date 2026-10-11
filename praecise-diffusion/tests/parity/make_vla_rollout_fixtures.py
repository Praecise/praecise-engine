"""Record a reference closed-loop rollout of a vision-language-action policy.

The policy runs through its released pre- and post-processors and its own
`select_action` loop (one action per tick, a new chunk when the queue of
executed steps runs out), with the same chunk noise every chunk. Frames and
states change every tick. The engine's rollout test replays the same
observations and compares every action.

usage: make_vla_rollout_fixtures.py <lerobot-src> <policy-dir> <backbone-dir> <out-dir>
       [--ticks N] [--n-action-steps K] [--seed S] [--instruction TEXT] [--hw H W]

The policy is built from its config with float32 parameters and loaded from
the release's weights; the backbone directory supplies the backbone config
and tokenizer, so nothing is downloaded.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot")
    ap.add_argument("policy")
    ap.add_argument("backbone")
    ap.add_argument("out")
    ap.add_argument("--ticks", type=int, default=12)
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--instruction", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--hw", type=int, nargs=2, default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.lerobot)

    import torch
    from safetensors.torch import save_file

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    torch.manual_seed(a.seed)
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.device = "cpu"
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = False
    if hasattr(cfg, "vlm_model_name"):
        cfg.vlm_model_name = a.backbone
    cfg.n_action_steps = a.n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg)
    policy = policy.to(torch.float32).eval()
    overrides = {"device_processor": {"device": "cpu"}, "tokenizer_processor": {"tokenizer_name": a.backbone}}
    pre = PolicyProcessorPipeline.from_pretrained(a.policy, config_filename="policy_preprocessor.json", overrides=overrides)
    post = PolicyProcessorPipeline.from_pretrained(a.policy, config_filename="policy_postprocessor.json", overrides={"device_processor": {"device": "cpu"}})

    cams = [k for k, f in cfg.input_features.items() if f.type.name == "VISUAL"]
    state_dim = cfg.input_features["observation.state"].shape[0]
    hw = a.hw or list(cfg.input_features[cams[0]].shape[1:])
    g = torch.Generator().manual_seed(a.seed)
    chunk = cfg.chunk_size
    max_action = getattr(cfg, "max_action_dim", cfg.output_features["action"].shape[0])
    noise = torch.randn(1, chunk, max_action, generator=g)

    tensors = {"noise": noise.reshape(-1).contiguous()}
    states, actions = [], []
    policy.reset()
    tokens = None
    for t in range(a.ticks):
        obs = {"task": a.instruction}
        for c, k in enumerate(cams):
            px = torch.randint(0, 256, (3, hw[0], hw[1]), generator=g, dtype=torch.uint8)
            img = px.to(torch.float32) / 255.0
            tensors[f"frame.{t}.{c}"] = img.contiguous()
            obs[k] = img
        state = torch.randn(state_dim, generator=g) * 0.5
        states.append(state)
        obs["observation.state"] = state
        batch = pre(obs)
        if tokens is None:
            ids = batch["observation.language.tokens"][0]
            mask = batch["observation.language.attention_mask"][0].bool()
            tokens = ids[mask].tolist()
        with torch.no_grad():
            act = policy.select_action(batch, noise=noise.clone())
        act = post(act)
        actions.append(act.reshape(-1).to(torch.float32))
    tensors["states"] = torch.stack(states).contiguous()
    tensors["actions"] = torch.stack(actions).contiguous()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "fixture.safetensors"))
    meta = {
        "policy": cfg.type,
        "ticks": a.ticks,
        "cameras": len(cams),
        "camera_keys": cams,
        "n_action_steps": a.n_action_steps,
        "instruction": a.instruction,
        "tokens": tokens,
        "torch": torch.__version__,
    }
    (out / "fixture.json").write_text(json.dumps(meta, indent=1))
    print("actions", tensors["actions"][:, :3])
    print("wrote", out)


if __name__ == "__main__":
    main()
