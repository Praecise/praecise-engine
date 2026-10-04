"""Build a small random Qwen-Image 2.1 checkpoint and reference pipeline runs.

Every component has the released layout at tiny widths: the transformer,
the autoencoder (16x, so the reference pipeline's fixed scale applies) and
the Qwen3-VL prompt encoder (with the released vocabulary, so the released
tokenizer and prompt template apply). Condition images are 64x64 and the
pipeline's resolution is set to 64, so every resize is the identity and the
native pipeline sees the same pixels as the reference; the processor's
minimum pixel count is lowered for the same reason. Cases: text-to-image,
and editing with two condition images; three steps with guidance against
" ", the reference using its key/value cache.

Needs a diffusers build with QwenImage21Pipeline, and torchvision.

Usage: QWEN21_PROCESSOR=<processor dir> python make_qwen_image21_pipeline_fixtures.py <out_dir>
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)

sys.path.insert(0, os.path.dirname(__file__))
from make_qwen_image21_fixtures import randomise  # noqa: E402

SIDE = 64
STEPS = 3
CFG = 4.0
PROMPT = "a red kite over a green hill"
Z = 8
TF = dict(
    patch_size=1, in_channels=Z, out_channels=Z, num_layers=2, attention_head_dim=16, num_attention_heads=2,
    context_in_dim=64, mlp_ratio=3, axes_dims_rope=(4, 6, 6), causal_condition=True,
)
VAE = dict(
    base_dim=4, decoder_base_dim=6, z_dim=Z, dim_mult=[1, 2, 2, 2, 2], num_res_blocks=1,
    temperal_downsample=[False, True, True, True], in_channels=4, out_channels=4, scale_factor_spatial=16,
)
TEXT = dict(
    hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_hidden_layers=4, rms_norm_eps=1e-6, vocab_size=151936, max_position_embeddings=4096,
)
SECTIONS = [4, 2, 2]
THETA = 5e6
VISION = dict(
    depth=3, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64, patch_size=16,
    spatial_merge_size=2, temporal_patch_size=2, in_channels=3, num_position_embeddings=16,
    deepstack_visual_indexes=[0, 1, 2], hidden_act="gelu_pytorch_tanh",
)
SCHED = dict(
    base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=8192, max_shift=0.9,
    num_train_timesteps=1000, shift=1.0, shift_terminal=0.02, stochastic_sampling=False,
    time_shift_type="exponential", use_beta_sigmas=False, use_dynamic_shifting=True,
    use_exponential_sigmas=False, use_karras_sigmas=False,
)
IMAGE, START, END = 151655, 151652, 151653
CASES = {"t2i": 0, "edit": 2}


def save(out, name, t):
    t = torch.as_tensor(t)
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    ck = os.path.join(out, "checkpoint")
    torch.manual_seed(0)
    tf = QwenImage21Transformer2DModel(**TF).eval()
    randomise(tf, 1)
    tf.save_pretrained(os.path.join(ck, "transformer"), safe_serialization=True)

    gv = torch.Generator().manual_seed(3)
    mean = torch.randn(Z, generator=gv).mul(0.5).bfloat16().float()
    std = torch.rand(Z, generator=gv).add(0.5).bfloat16().float()
    vae = AutoencoderKLQwenImage21(**VAE, latents_mean=mean.tolist(), latents_std=std.tolist()).eval()
    randomise(vae, 4)
    vae.save_pretrained(os.path.join(ck, "vae"), safe_serialization=True)

    rope = dict(rope_type="default", rope_theta=THETA, mrope_section=SECTIONS, mrope_interleaved=True)
    cfg = Qwen3VLConfig(
        text_config=dict(TEXT, rope_parameters=rope), vision_config=dict(VISION),
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END,
    )
    te = Qwen3VLForConditionalGeneration(cfg).eval()
    sd = {}
    with torch.no_grad():
        for name, p in te.named_parameters():
            p.normal_(0, 0.02 if p.dim() > 1 else 0.1)
            if name.endswith("norm.weight") or "norm1" in name or "norm2" in name or "layernorm" in name:
                p.add_(1.0)
            if name.endswith("pos_embed.weight"):
                p.normal_(0, 0.5)
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        for name, p in te.state_dict().items():
            if name != "lm_head.weight":
                sd[name] = p.to(torch.bfloat16).contiguous()
    os.makedirs(os.path.join(ck, "text_encoder"), exist_ok=True)
    save_file(sd, os.path.join(ck, "text_encoder", "model.safetensors"))
    released = dict(
        architectures=["Qwen3VLForConditionalGeneration"], model_type="qwen3_vl",
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END, tie_word_embeddings=False,
        text_config=dict(TEXT, model_type="qwen3_vl_text", rope_theta=THETA,
                         rope_scaling=dict(mrope_interleaved=True, mrope_section=SECTIONS, rope_type="default")),
        vision_config=dict(VISION, model_type="qwen3_vl"),
    )
    with open(os.path.join(ck, "text_encoder", "config.json"), "w") as f:
        json.dump(released, f, indent=1)

    proc_dir = os.path.join(ck, "processor")
    shutil.copytree(os.environ["QWEN21_PROCESSOR"], proc_dir, dirs_exist_ok=True)
    pc = os.path.join(proc_dir, "preprocessor_config.json")
    with open(pc) as f:
        pre = json.load(f)
    pre["size"] = {"longest_edge": 1 << 24, "shortest_edge": 1024}
    with open(pc, "w") as f:
        json.dump(pre, f, indent=1)
    processor = Qwen3VLProcessor.from_pretrained(proc_dir)

    sched = FlowMatchEulerDiscreteScheduler(**SCHED)
    os.makedirs(os.path.join(ck, "scheduler"), exist_ok=True)
    with open(os.path.join(ck, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(dict(SCHED, _class_name="FlowMatchEulerDiscreteScheduler"), f, indent=1)

    pipe = QwenImage21Pipeline(scheduler=sched, vae=vae, text_encoder=te, processor=processor, transformer=tf)
    g = torch.Generator().manual_seed(5)
    images = []
    for i in range(max(CASES.values())):
        rgb = torch.randint(0, 256, (SIDE, SIDE, 3), generator=g, dtype=torch.uint8)
        save(out, f"image_{i}", rgb)
        images.append(Image.fromarray(rgb.numpy()))
    lh = lw = SIDE // 16
    cases = []
    for tag, n in CASES.items():
        imgs = images[:n] or None
        with torch.no_grad():
            pe, _, mask = pipe._get_qwen_prompt_embeds(PROMPT, imgs)
        save(out, f"{tag}_prompt_embeds", pe[0])
        save(out, f"{tag}_image_mask", mask[0].float())
        noise = torch.randn(1, Z, lh, lw, generator=g)
        save(out, f"{tag}_noise", noise[0])
        packed = noise.view(1, Z, lh * lw).transpose(1, 2)
        common = dict(prompt=PROMPT, image=imgs, negative_prompt=" ", true_cfg_scale=CFG, height=SIDE, width=SIDE,
                      num_inference_steps=STEPS, output_resolution=SIDE)
        with torch.no_grad():
            lat = pipe(**common, latents=packed.clone(), output_type="latent").images
            img = pipe(**common, latents=packed.clone(), output_type="np").images
        save(out, f"{tag}_latents", lat[0])
        save(out, f"{tag}_decoded", torch.from_numpy(np.asarray(img[0][..., :3]) * 255).round())
        cases.append(dict(tag=tag, images=n))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(side=SIDE, steps=STEPS, cfg=CFG, prompt=PROMPT, cases=cases), f, indent=1)


if __name__ == "__main__":
    main()
