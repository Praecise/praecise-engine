"""Record reference MolmoAct2 action chunks for a random-weight small model.

The small text and vision configs of make_molmoact2_prefix_fixture.py get a
small action expert (one block per language-model block). Each sample's
prompt goes through the release processor and tokenizer as in the prefix
fixture; the model's own prefill then gives every layer's keys and values,
and `generate_actions_from_inputs` runs the flow-matching loop from noise
drawn by a seeded generator (the same draws are recorded).

Samples cover a full and a short horizon, padded action dimensions, one and
two cameras. `--alt` flips the expert's options (no context norm, no
query/key norms, no rotary positions) and the action mode (continuous only,
so the context mask is the plain one). Causal self-attention is not covered:
the reference's inference path cannot run it (it passes an explicit mask
together with is_causal).

The fixture keeps the policy dir (config.json in the release layout with the
small sub-configs, model.safetensors under the release's key prefixes, the
release's tokenizer and processor config), per sample the prompt ids, the
processor's crops and pooling indices, the context mask, every layer's keys
and values at the valid positions, the noise and the actions; and one
synthetic id list with discrete action spans and its context mask.

Usage:
  make_molmoact2_action_fixture.py <lerobot/src> <release dir> <out> [--alt]
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
# MLP width: int(48 * 2.5) = 120, rounded up to a multiple of 64 -> 128.
EXPERT = {"hidden_size": 48, "num_layers": 3, "num_heads": 3, "mlp_ratio": 2.5, "ffn_multiple_of": 64,
          "timestep_embed_dim": 32, "dropout": 0.0, "attn_dropout": 0.0, "context_layer_norm": True,
          "qk_norm": True, "qk_norm_eps": 1e-6, "rope": True, "causal_attn": False}
ALT = {"context_layer_norm": False, "qk_norm": False, "rope": False}
SAMPLES = [
    {"task": "Task: Pick up the red cube.  Then put it in the bowl!", "frames": [(240, 320)],
     "state": [0.0, -0.5, 0.25, 1.5, -1.2, 0.999, 0.3], "horizon": None, "action_dim": None, "seed": 11},
    {"task": "stack the blocks", "frames": [(96, 128), (480, 640)],
     "state": [0.1, -0.9, 0.6, 0.0, -0.33, 0.75, -1.0, 1.0], "horizon": 12, "action_dim": 7, "seed": 12},
]
SETUP = "single arm on a tabletop"
CONTROL = "joint position"


def to_bytes(frame):
    import numpy as np
    arr = frame
    if float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot_src")
    ap.add_argument("release")
    ap.add_argument("out")
    ap.add_argument("--alt", action="store_true")
    a = ap.parse_args()

    d = Path(a.lerobot_src) / "lerobot/policies/molmoact2/molmoact2_hf_model"
    pkg = types.ModuleType("m2hf")
    pkg.__path__ = [str(d)]
    sys.modules["m2hf"] = pkg
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2Tokenizer

    import m2hf.configuration_molmoact2 as c
    import m2hf.modeling_molmoact2 as m
    from m2hf.image_processing_molmoact2 import MolmoAct2ImageProcessor
    from m2hf.processing_molmoact2 import MolmoAct2Processor
    from m2hf.video_processing_molmoact2 import MolmoAct2VideoProcessor

    rel = Path(a.release)
    release = json.loads((rel / "config.json").read_text())
    expert = dict(EXPERT, **(ALT if a.alt else {}))
    mode = "continuous" if a.alt else release["action_mode"]
    top = {k: v for k, v in release.items() if not isinstance(v, dict) and k not in ("architectures", "auto_map")}
    top.update(add_action_expert=True, action_mode=mode)
    cfg = c.MolmoAct2Config(text_config=TEXT, vit_config=VIT, adapter_config=ADAPTER,
                            action_expert_config=dict(expert), **top)
    for sub in (cfg, cfg.text_config, cfg.vit_config, cfg.adapter_config):
        sub._attn_implementation = "sdpa"

    torch.manual_seed(0)
    model = m.MolmoAct2Model(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_f.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.1)
            elif name.startswith("action_expert."):
                p.copy_(torch.randn_like(p) * 0.15)
            else:
                p.copy_(torch.randn_like(p) * 0.08)

    pc = json.loads((rel / "processor_config.json").read_text())
    strip = lambda x, k: {kk: vv for kk, vv in x.items() if kk not in {"auto_map", "processor_class", k}}
    tok = Qwen2Tokenizer.from_pretrained(str(rel))
    tok.padding_side = "left"
    proc = MolmoAct2Processor(
        image_processor=MolmoAct2ImageProcessor(**strip(pc["image_processor"], "image_processor_type")),
        video_processor=MolmoAct2VideoProcessor(**strip(pc["video_processor"], "video_processor_type")),
        tokenizer=tok,
        image_use_col_tokens=pc.get("image_use_col_tokens", True),
        use_single_crop_col_tokens=pc.get("use_single_crop_col_tokens"),
        use_single_crop_start_token=pc.get("use_single_crop_start_token", True),
        video_use_col_tokens=pc.get("video_use_col_tokens", False),
        use_frame_special_tokens=pc.get("use_frame_special_tokens", True),
    )

    out = Path(a.out)
    (out / "policy").mkdir(parents=True, exist_ok=True)
    config = dict(release, text_config=TEXT, vit_config=VIT, adapter_config=ADAPTER, action_expert_config=expert,
                  add_action_expert=True, action_mode=mode)
    (out / "policy/config.json").write_text(json.dumps(config, indent=1))
    for f in ("processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(rel / f, out / "policy" / f)
    save_file({f"model.{k}": v.contiguous() for k, v in model.state_dict().items()},
              str(out / "policy/model.safetensors"))

    gen = torch.Generator().manual_seed(1)
    tensors, meta = {}, []
    for s, smp in enumerate(SAMPLES):
        frames = [torch.rand(h, w, 3, generator=gen).numpy() for h, w in smp["frames"]]
        images = [to_bytes(f) for f in frames]
        state = np.asarray(smp["state"], dtype=np.float32)
        task = m._normalize_question_text(smp["task"])
        prompt = m._build_robot_text(
            task=task, style="robot_action",
            discrete_state_string=m._build_discrete_state_string(state, cfg.num_state_tokens),
            setup_type=SETUP, control_mode=CONTROL, add_setup_tokens=cfg.add_setup_tokens,
            add_control_tokens=cfg.add_control_tokens, num_images=len(images))
        inputs = proc(text=[prompt], images=images, return_tensors="pt", padding=True, pad_to_multiple_of=8)
        inputs["position_ids"] = torch.clamp(torch.cumsum(inputs["attention_mask"].long(), dim=-1) - 1, min=0)
        valid = inputs["attention_mask"][0].bool()
        model_inputs = dict(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"].float(),
            image_token_pooling=inputs["image_token_pooling"], image_grids=inputs["image_grids"],
            image_num_crops=inputs["image_num_crops"], attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"], token_type_ids=inputs["token_type_ids"])
        _, pooling = model.build_batched_images(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"].float(),
            image_token_pooling=inputs["image_token_pooling"], image_grids=inputs["image_grids"],
            image_num_crops=inputs["image_num_crops"])
        horizon = smp["horizon"] or cfg.max_action_horizon
        pad = None
        if smp["action_dim"] is not None:
            pad = (torch.arange(cfg.max_action_dim) >= smp["action_dim"]).unsqueeze(0)
        with torch.no_grad():
            o = model(**model_inputs, use_cache=True)
            kv = model._extract_kv_states(o.past_key_values)
            mask = model._get_encoder_attention_mask(inputs["input_ids"], inputs["attention_mask"])
            noise = torch.randn((1, horizon, cfg.max_action_dim), generator=torch.Generator().manual_seed(smp["seed"]))
            actions = model.generate_actions_from_inputs(
                **model_inputs, action_dim_is_pad=pad, action_horizon=horizon,
                generator=torch.Generator().manual_seed(smp["seed"]))
        tensors[f"{s}.ids"] = inputs["input_ids"][0][valid].float().contiguous()
        tensors[f"{s}.pixels"] = inputs["pixel_values"].float().contiguous()
        tensors[f"{s}.pooling"] = pooling[0].float().contiguous()
        tensors[f"{s}.mask"] = mask[0][valid].float().contiguous()
        for layer, (key, value) in enumerate(kv):
            tensors[f"{s}.k{layer}"] = key[0][valid].float().contiguous()
            tensors[f"{s}.v{layer}"] = value[0][valid].float().contiguous()
        tensors[f"{s}.noise"] = noise[0].contiguous()
        tensors[f"{s}.actions"] = actions[0].float().contiguous()
        if pad is not None:
            tensors[f"{s}.padding"] = pad[0].float().contiguous()
        n = int(valid.sum())
        start = noise if pad is None else noise.masked_fill(pad.unsqueeze(1), 0.0)
        moved = float((actions - start).abs().max())
        meta.append({"task": smp["task"], "prompt": prompt, "frames": smp["frames"],
                     "crops": int(inputs["pixel_values"].shape[0]), "tokens": n, "horizon": horizon,
                     "action_dim": smp["action_dim"], "masked": int((~mask[0][valid]).sum()), "layers": len(kv)})
        print(f"sample {s}: {n} tokens ({meta[-1]['masked']} hidden from the expert), crops {meta[-1]['crops']}, "
              f"horizon {horizon}, |actions| {actions.abs().max():.3f}, |actions - noise| {moved:.3f}")
    # Discrete action spans: one closed, one unclosed.
    st, en, eos = cfg.action_start_token_id, cfg.action_end_token_id, cfg.eos_token_id
    span_ids = torch.tensor([[5, st, 6, 7, en, 8, eos, 9, en, st, 10, en, 11, st, 12, 13]])
    span_mask = model._get_encoder_attention_mask(span_ids, torch.ones_like(span_ids))
    tensors["span.ids"] = span_ids[0].float().contiguous()
    tensors["span.mask"] = span_mask[0].float().contiguous()
    print(f"span mask: {span_mask[0].int().tolist()}")
    save_file(tensors, str(out / "fixture.safetensors"))
    (out / "fixture.json").write_text(json.dumps(
        {"samples": meta, "setup_type": SETUP, "control_mode": CONTROL, "action_mode": mode, "expert": expert,
         "pool": int(pooling.shape[-1]), "torch": torch.__version__}, indent=1))


if __name__ == "__main__":
    main()
