"""Record a reference closed-loop rollout of a vision-language-action policy.

The policy runs through its released pre- and post-processors and its own
`select_action` loop (one action per tick, a new chunk when the queue of
executed steps runs out), with the same chunk noise every chunk. Frames and
states change every tick. The engine's rollout test replays the same
observations and compares every action.

usage: make_vla_rollout_fixtures.py <lerobot-src> <policy-dir> <backbone-dir> <out-dir>
       [--ticks N] [--n-action-steps K] [--seed S] [--instruction TEXT] [--hw H W]
       [--tiny pi05]

With --tiny, <policy-dir> is created: a policy of the named type with the
same structure at small sizes and random weights, random normalisation
statistics and the type's processors, written as a release. Its sizes are
recorded in fixture.json for the engine test.

The policy is built from its config with float32 parameters and loaded from
the release's weights; the backbone directory supplies the backbone config
and tokenizer, so nothing is downloaded.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

# Small sizes with the released structure, per policy type.
TINY = {
    "smolvla": {
        "text": {"hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "num_hidden_layers": 4},
        "vision": {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64, "patch_size": 16, "image_size": 64},
        "scale_factor": 2,
    },
    "pi05": {
        "vlm": {"width": 64, "depth": 2, "mlp_dim": 128, "num_heads": 2, "num_kv_heads": 1, "head_dim": 32},
        "expert": {"width": 48, "depth": 2, "mlp_dim": 96, "num_heads": 2, "num_kv_heads": 1, "head_dim": 32},
        "vision": {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64, "patch_size": 14, "image_size": 224},
    },
}


def light_policies_package(src):
    """Register `lerobot.policies` without running its `__init__`, which
    imports every policy and the simulation environments; only the policy
    under test and its processor are imported."""
    import importlib.util
    import types

    import lerobot

    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(Path(src) / "lerobot" / "policies")]
    sys.modules["lerobot.policies"] = pkg
    lerobot.policies = pkg
    assert importlib.util.find_spec("lerobot.policies.pretrained") is not None


def adapt_transformers():
    """Let the reference run on a transformers release whose causal-mask
    helper takes no cache positions (the mask the policy passes is complete)."""
    import inspect as _inspect

    import lerobot.policies.pi_gemma as pg

    if "cache_position" not in _inspect.signature(pg.create_causal_mask).parameters:
        base = pg.create_causal_mask

        def create_causal_mask(**kw):
            kw.pop("cache_position", None)
            return base(**kw)

        pg.create_causal_mask = create_causal_mask


def policy_class(kind):
    if kind == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    if kind == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    raise SystemExit(f"no reference loader for {kind}")


def shrink_pi05(dims):
    """Make the pi05 modules build at the tiny sizes."""
    import lerobot.policies.pi05.modeling_pi05 as m

    m.get_gemma_config = lambda v: m.GemmaConfig(**dims["vlm" if v == "gemma_2b" else "expert"])
    pins = {
        "hidden_size": dims["vision"]["hidden_size"],
        "num_hidden_layers": dims["vision"]["num_hidden_layers"],
        "num_attention_heads": dims["vision"]["num_attention_heads"],
        "intermediate_size": dims["vision"]["intermediate_size"],
        "patch_size": dims["vision"]["patch_size"],
        "projection_dim": dims["vlm"]["width"],
    }
    from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma as Pali

    build = Pali.__init__

    def init(self, config):
        for k, v in pins.items():
            setattr(config.vision_config, k, v)
        build(self, config)

    Pali.__init__ = init


def make_tiny_smolvla(a, dims):
    """Write a small backbone config next to the release and a random-weight
    SmolVLA release to a.policy; returns the backbone directory."""
    import shutil

    import torch

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    out = Path(a.policy)
    backbone = out.parent / "backbone"
    shutil.copytree(a.backbone, backbone, dirs_exist_ok=True)
    bc = json.loads((backbone / "config.json").read_text())
    bc["text_config"].update(dims["text"])
    bc["vision_config"].update(dims["vision"])
    bc["scale_factor"] = dims["scale_factor"]
    (backbone / "config.json").write_text(json.dumps(bc, indent=1))

    torch.manual_seed(a.seed + 1)
    cfg = SmolVLAConfig(
        input_features={
            "observation.images.scene": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=16,
        n_action_steps=16,
        resize_imgs_with_padding=(64, 64),
        vlm_model_name=str(backbone),
        load_vlm_weights=False,
        num_vlm_layers=dims["text"]["num_hidden_layers"],
        device="cpu",
    )
    policy = SmolVLAPolicy(cfg).to(torch.float32)
    randomize(policy)
    stats = random_stats(torch)
    pre, post = make_smolvla_pre_post_processors(cfg, stats)
    save_release(policy, cfg, pre, post, out)
    return str(backbone)


def randomize(policy):
    import torch

    with torch.no_grad():
        for _, p in policy.named_parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 1.0 / p[0].numel() ** 0.5)
            else:
                p.normal_(0.0, 0.2)


def random_stats(torch):
    stats = {}
    for k in ("observation.state", "action"):
        lo = torch.randn(6) * 0.5 - 1.0
        hi = lo + 1.0 + torch.rand(6)
        stats[k] = {"mean": (lo + hi) / 2, "std": (hi - lo) / 4, "min": lo - 0.2, "max": hi + 0.2, "q01": lo, "q99": hi,
                    "q10": lo + 0.1, "q90": hi - 0.1}
    return stats


def save_release(policy, cfg, pre, post, out):
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    # The weights as the policy's own saver writes them (one tensor per name).
    save_file({k: v.detach().clone().contiguous() for k, v in policy.state_dict().items()}, str(out / "model.safetensors"))
    cfg.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)


def make_tiny_release(a, dims):
    """Write a random-weight release of the tiny policy to a.policy."""
    import torch

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    torch.manual_seed(a.seed + 1)
    cfg = PI05Config(
        input_features={
            "observation.images.scene": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=16,
        n_action_steps=16,
        num_inference_steps=10,
        tokenizer_max_length=64,
        text_tokenizer_name=a.backbone,
        device="cpu",
    )
    policy = PI05Policy(cfg).to(torch.float32)
    randomize(policy)
    pre, post = make_pi05_pre_post_processors(cfg, random_stats(torch))
    save_release(policy, cfg, pre, post, Path(a.policy))


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
    ap.add_argument("--tiny", choices=["pi05", "smolvla"], default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.lerobot)
    light_policies_package(a.lerobot)
    adapt_transformers()
    dims = TINY[a.tiny] if a.tiny else None
    if a.tiny == "pi05":
        shrink_pi05(dims)
        make_tiny_release(a, dims)
    if a.tiny == "smolvla":
        a.backbone = make_tiny_smolvla(a, dims)
        dims = None

    import torch
    from safetensors.torch import save_file

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    torch.manual_seed(a.seed)
    # Importing the policy class registers its config type.
    policy_class(json.loads((Path(a.policy) / "config.json").read_text())["type"])
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.device = "cpu"
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = False
    if hasattr(cfg, "vlm_model_name"):
        cfg.vlm_model_name = a.backbone
    cfg.n_action_steps = a.n_action_steps
    policy = policy_class(cfg.type).from_pretrained(a.policy, config=cfg)
    policy = policy.to(torch.float32).eval()
    overrides = {"device_processor": {"device": "cpu"}, "tokenizer_processor": {"tokenizer_name": a.backbone}}
    pre = PolicyProcessorPipeline.from_pretrained(a.policy, config_filename="policy_preprocessor.json", overrides=overrides)
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    post = PolicyProcessorPipeline.from_pretrained(
        a.policy,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    cams = [k for k, f in cfg.input_features.items() if f.type.name == "VISUAL"]
    state_dim = cfg.input_features["observation.state"].shape[0]
    hw = a.hw or list(cfg.input_features[cams[0]].shape[1:])
    g = torch.Generator().manual_seed(a.seed)
    chunk = cfg.chunk_size
    max_action = getattr(cfg, "max_action_dim", cfg.output_features["action"].shape[0])
    noise = torch.randn(1, chunk, max_action, generator=g)

    if "noise" not in inspect.signature(policy.select_action).parameters:
        # The loop draws its own noise: hand it the fixed noise instead.
        policy.model.sample_noise = lambda shape, device: noise.clone().to(device)

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
            if "noise" in inspect.signature(policy.select_action).parameters:
                act = policy.select_action(batch, noise=noise.clone())
            else:
                act = policy.select_action(batch)
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
        "dims": dims,
    }
    (out / "fixture.json").write_text(json.dumps(meta, indent=1))
    print("actions", tensors["actions"][:, :3])
    print("wrote", out)


if __name__ == "__main__":
    main()
