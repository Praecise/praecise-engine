"""Record a reference MolmoAct2 prompt prefix for a random-weight small model.

A MolmoAct2 model without action expert is built from a small text config
and a small vision config (at the release's crop size, so the release's
image processor applies unchanged), with random weights, in float32. Each
sample goes through the policy's inference preprocessing: frames in [0, 1]
to 8 bits, the robot prompt (normalised task, discrete state tokens, setup
and control tokens, one placeholder per camera), the release processor and
tokenizer (left padding to a multiple of 8, BOS, image token types) and
positions over the valid tokens. The model's forward then gives the image
features, the final hidden states and every layer's keys and values.

The fixture keeps the policy dir (config.json in the release layout with the
small sub-configs, model.safetensors under the release's key prefixes, the
release's tokenizer and processor config), every input, the prompt text and
ids, the processor's crops and pooling indices (offset per image as the
model batches them) and the outputs at the valid positions.

The modeling module is loaded straight from the lerobot tree, without
importing the lerobot package.

Usage:
  make_molmoact2_prefix_fixture.py <lerobot/src> <release dir> <out>
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
SAMPLES = [
    {"task": "Task: Pick up the red cube.  Then put it in the bowl!", "frames": [(240, 320)],
     "state": [0.0, -0.5, 0.25, 1.5, -1.2, 0.999, 0.3]},
    {"task": "stack the blocks", "frames": [(96, 128), (480, 640)],
     "state": [0.1, -0.9, 0.6, 0.0, -0.33, 0.75, -1.0, 1.0]},
]
SETUP = "single arm on a tabletop"
CONTROL = "joint position"


def to_bytes(frame):
    # The policy processor's image normalisation for float frames.
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
    top = {k: v for k, v in release.items() if not isinstance(v, dict) and k not in ("architectures", "auto_map")}
    top["add_action_expert"] = False
    cfg = c.MolmoAct2Config(text_config=TEXT, vit_config=VIT, adapter_config=ADAPTER, **top)
    for sub in (cfg, cfg.text_config, cfg.vit_config, cfg.adapter_config):
        sub._attn_implementation = "sdpa"

    torch.manual_seed(0)
    model = m.MolmoAct2Model(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_f.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.1)
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
    config = dict(release, text_config=TEXT, vit_config=VIT, adapter_config=ADAPTER, add_action_expert=False)
    config.pop("action_expert_config", None)
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
        _, pooling = model.build_batched_images(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"].float(),
            image_token_pooling=inputs["image_token_pooling"], image_grids=inputs["image_grids"],
            image_num_crops=inputs["image_num_crops"])
        with torch.no_grad():
            o = model(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"].float(),
                image_token_pooling=inputs["image_token_pooling"], image_grids=inputs["image_grids"],
                image_num_crops=inputs["image_num_crops"], attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"], token_type_ids=inputs["token_type_ids"], use_cache=True)
        for j, f in enumerate(frames):
            tensors[f"{s}.frame{j}"] = torch.from_numpy(f).contiguous()
        tensors[f"{s}.state"] = torch.from_numpy(state)
        tensors[f"{s}.ids"] = inputs["input_ids"][0][valid].float().contiguous()
        tensors[f"{s}.pixels"] = inputs["pixel_values"].float().contiguous()
        tensors[f"{s}.pooling"] = pooling[0].float().contiguous()
        tensors[f"{s}.features"] = o.image_hidden_states.float().contiguous()
        tensors[f"{s}.hidden"] = o.last_hidden_state[0][valid].float().contiguous()
        layers = 0
        for key, value in m._iter_cache_key_values(o.past_key_values):
            tensors[f"{s}.k{layers}"] = model._cache_to_sequence(key)[0][valid].float().contiguous()
            tensors[f"{s}.v{layers}"] = model._cache_to_sequence(value)[0][valid].float().contiguous()
            layers += 1
        n = int(valid.sum())
        meta.append({"task": smp["task"], "prompt": prompt, "frames": smp["frames"],
                     "crops": int(inputs["pixel_values"].shape[0]), "tokens": n,
                     "padded": int(inputs["input_ids"].shape[1]), "image_tokens": int(o.image_hidden_states.shape[0]),
                     "layers": layers})
        print(f"sample {s}: {n} valid of {inputs['input_ids'].shape[1]} tokens, crops {meta[-1]['crops']}, "
              f"image tokens {meta[-1]['image_tokens']}, |hidden| {o.last_hidden_state.abs().max():.3f}")
        print(f"  prompt: {prompt}")
    save_file(tensors, str(out / "fixture.safetensors"))
    (out / "fixture.json").write_text(json.dumps(
        {"samples": meta, "setup_type": SETUP, "control_mode": CONTROL, "torch": torch.__version__}, indent=1))


if __name__ == "__main__":
    main()
