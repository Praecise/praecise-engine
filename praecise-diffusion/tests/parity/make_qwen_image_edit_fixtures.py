"""Build a small random Qwen-Image editing checkpoint and reference outputs.

Every component has the released layout at tiny widths: the editing
transformer, the autoencoder, and the Qwen2.5-VL prompt encoder (with the
released vocabulary, so the released tokenizer and prompt template apply).
One 224x224 reference image is used: at that size the encoder's and the
autoencoder's resizes are both the identity (the reference areas are patched
to 224x224), so the native pipeline sees the same pixels as the reference.

The reference pipeline is given prompt states computed the way it computes
them, except that the token-type ids are passed so image tokens get their
three-axis positions (see make_qwen_vl_fixtures.py). Three steps with
norm-rescaled guidance against " ".

Usage: QWEN_TOKENIZER=<processor/tokenizer.json> python make_qwen_image_edit_fixtures.py <out_dir>
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as edit
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel

sys.path.insert(0, os.path.dirname(__file__))
from make_qwen_image_fixtures import TINY, VAE, randomise  # noqa: E402
from make_qwen_vl_fixtures import patches  # noqa: E402

SIDE = 224
STEPS = 3
CFG = 4.0
PROMPT = "make the sky green"
IMAGE, START, END = 151655, 151652, 151653
TEXT = dict(
    hidden_size=TINY["joint_attention_dim"], intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
    num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=152064, max_position_embeddings=4096,
)
SECTIONS = [2, 2, 2]
VISION = dict(
    depth=2, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=TINY["joint_attention_dim"],
    patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=112, fullatt_block_indexes=[1],
    in_channels=3, hidden_act="silu",
)
SCHED = dict(
    base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=8192, max_shift=0.9,
    num_train_timesteps=1000, shift=1.0, shift_terminal=0.02, stochastic_sampling=False,
    time_shift_type="exponential", use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False,
    use_karras_sigmas=False,
)


def save(out, name, t):
    np.asarray(t, dtype="<f4").tofile(os.path.join(out, name + ".bin"))


def encoder(ck):
    rope = dict(rope_type="default", rope_theta=1e6, mrope_section=SECTIONS)
    cfg = Qwen2_5_VLConfig(text_config=dict(TEXT, rope_parameters=rope), vision_config=dict(VISION),
                           image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END)
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    randomise(model, 7)
    sd = {}
    for name, p in model.state_dict().items():
        if name != "lm_head.weight":
            sd[name.replace("model.language_model.", "model.").replace("model.visual.", "visual.")] = p.bfloat16().contiguous()
    os.makedirs(os.path.join(ck, "text_encoder"), exist_ok=True)
    save_file(sd, os.path.join(ck, "text_encoder", "model.safetensors"))
    released = dict(TEXT, model_type="qwen2_5_vl", rope_theta=1e6,
                    rope_scaling=dict(mrope_section=SECTIONS, rope_type="default", type="default"),
                    image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END, vision_config=VISION)
    with open(os.path.join(ck, "text_encoder", "config.json"), "w") as f:
        json.dump(released, f, indent=1)
    return model


def embeds(model, tok, prompt, px):
    template = TEMPLATE.format("Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + prompt)
    ids = tok.encode(template, add_special_tokens=False).ids
    n = (SIDE // 28) ** 2
    out = []
    for t in ids:
        out.extend([t] * n if t == IMAGE else [t])
    ids = torch.tensor([out])
    pv = patches(px)
    with torch.no_grad():
        h = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
                        image_grid_thw=torch.tensor([[1, SIDE // 14, SIDE // 14]]),
                        mm_token_type_ids=(ids == IMAGE).int()).last_hidden_state
    return h[:, 64:]


TEMPLATE = None


def main():
    global TEMPLATE
    out = sys.argv[1]
    ck = os.path.join(out, "checkpoint")
    os.makedirs(os.path.join(ck, "processor"), exist_ok=True)
    os.makedirs(os.path.join(ck, "scheduler"), exist_ok=True)
    shutil.copy(os.environ["QWEN_TOKENIZER"], os.path.join(ck, "processor", "tokenizer.json"))
    with open(os.path.join(ck, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(dict(SCHED, _class_name="FlowMatchEulerDiscreteScheduler"), f, indent=1)
    torch.manual_seed(0)
    tf = QwenImageTransformer2DModel(**TINY).eval()
    randomise(tf, 1)
    tf.save_pretrained(os.path.join(ck, "transformer"), safe_serialization=True)
    gv = torch.Generator().manual_seed(3)
    mean = torch.randn(VAE["z_dim"], generator=gv).mul(0.5).bfloat16().float()
    std = torch.rand(VAE["z_dim"], generator=gv).add(0.5).bfloat16().float()
    vae = AutoencoderKLQwenImage(**VAE, latents_mean=mean.tolist(), latents_std=std.tolist()).eval()
    randomise(vae, 4)
    vae.save_pretrained(os.path.join(ck, "vae"), safe_serialization=True)
    model = encoder(ck)
    sched = FlowMatchEulerDiscreteScheduler(**SCHED)
    pipe = edit.QwenImageEditPlusPipeline(scheduler=sched, vae=vae, text_encoder=model, tokenizer=None,
                                          processor=None, transformer=tf)
    TEMPLATE = pipe.prompt_template_encode
    assert pipe.prompt_template_encode_start_idx == 64
    edit.CONDITION_IMAGE_SIZE = edit.VAE_IMAGE_SIZE = SIDE * SIDE

    g = torch.Generator().manual_seed(5)
    rgb = (torch.rand(SIDE, SIDE, 3, generator=g) * 255).round().to(torch.uint8)
    save(out, "image", rgb.float().numpy())
    pil = Image.fromarray(rgb.numpy())
    px = rgb.permute(2, 0, 1).float() / 255.0
    tok = Tokenizer.from_file(os.environ["QWEN_TOKENIZER"])
    pe = embeds(model, tok, PROMPT, px)
    ne = embeds(model, tok, " ", px)
    save(out, "prompt_embeds", pe[0].numpy())
    z = VAE["z_dim"]
    lh = lw = SIDE // pipe.vae_scale_factor
    noise = torch.randn(1, z, lh, lw, generator=g)
    save(out, "noise", noise[0].numpy())
    packed = noise.view(1, z, lh // 2, 2, lw // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, (lh // 2) * (lw // 2), z * 4)
    common = dict(image=[pil], prompt_embeds=pe, prompt_embeds_mask=torch.ones(pe.shape[:2], dtype=torch.long),
                  negative_prompt_embeds=ne, negative_prompt_embeds_mask=torch.ones(ne.shape[:2], dtype=torch.long),
                  true_cfg_scale=CFG, height=SIDE, width=SIDE, num_inference_steps=STEPS)
    with torch.no_grad():
        lat = pipe(**common, latents=packed.clone(), output_type="latent").images
        save(out, "latents", lat[0].numpy())
        img = pipe(**common, latents=packed.clone(), output_type="np").images
        save(out, "decoded", (img[0] * 255.0))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(prompt=PROMPT, side=SIDE, steps=STEPS, cfg=CFG), f, indent=1)
    print("prompt tokens", pe.shape, "latents", lat.shape)


if __name__ == "__main__":
    main()
