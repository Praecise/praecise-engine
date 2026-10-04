"""Build a small random Z-Image checkpoint and reference outputs.

The checkpoint has the real layout, the real tokenizer and the real scheduler
but tiny widths, so the native pipeline can be checked against the reference
implementation on a CPU in seconds. Weights are rounded to bfloat16 before
anything runs, so the native loader sees exactly the values the reference
computed with.

Usage: python make_zimage_fixtures.py <out_dir> <released_checkpoint_dir>
       python make_zimage_fixtures.py --checkpoint <checkpoint_dir> <out_dir> [text|image]

Only the tokenizer and the scheduler configuration are taken from the
released checkpoint in the first form. With --checkpoint, the reference
outputs are computed from that checkpoint in float32 at a low resolution, and
`<out_dir>/checkpoint` links to it; `text` computes the caption features alone
and `image` the rest from them, so the two halves of a large checkpoint need
not be on disk at once.
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

PROMPT = "A lighthouse on a rocky coast at dusk, warm light in the window."
SMALL = {"width": 64, "height": 48, "steps": 4, "guidance": 2.0}
REAL = {"width": 256, "height": 192, "steps": 8, "guidance": 3.0}


def randomise(module, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if p.ndim == 1 and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def save(out, name, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    a.astype("<f4").tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


def build(out, released):
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(released, "tokenizer"))
    te = Qwen3Model(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            tie_word_embeddings=True,
        )
    )
    randomise(te, 1)
    transformer = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=256,
        n_layers=2,
        n_refiner_layers=2,
        n_heads=2,
        n_kv_heads=2,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=64,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1536, 512, 512],
    )
    randomise(transformer, 2)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[32, 32, 64, 64],
        layers_per_block=2,
        latent_channels=16,
        norm_num_groups=32,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
        mid_block_add_attention=True,
    )
    randomise(vae, 3)
    pipe = ZImagePipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(released, "scheduler")),
        vae=vae,
        text_encoder=te,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    pipe.save_pretrained(ckpt, safe_serialization=True)
    from safetensors.torch import load_file, save_file

    for sub in ("transformer", "vae", "text_encoder"):
        d = os.path.join(ckpt, sub)
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                p = os.path.join(d, f)
                sd = load_file(p)
                # The released text encoder keeps its causal-LM key prefix.
                prefix = "model." if sub == "text_encoder" else ""
                save_file({prefix + k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}, p)
    return pipe


def reference(pipe, out, spec, text=True):
    w, h, steps = spec["width"], spec["height"], spec["steps"]
    meta = {"prompt": PROMPT, **spec, "has_text_encoder": text}
    with torch.no_grad():
        if text:
            for key, prompt in (("cond_ids", PROMPT), ("uncond_ids", "")):
                templated = pipe.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                enc = pipe.tokenizer(templated, truncation=True, max_length=512)
                meta[key] = enc["input_ids"]
            cap = pipe._encode_prompt(PROMPT, device="cpu")[0]
            cap_uncond = pipe._encode_prompt("", device="cpu")[0]
            meta["cap"] = save(out, "cap", cap)
            meta["cap_uncond"] = save(out, "cap_uncond", cap_uncond)
        else:
            width = pipe.transformer.config.cap_feat_dim
            cap = torch.from_numpy(np.fromfile(os.path.join(out, "cap.bin"), dtype="<f4")).view(-1, width)
            cap_uncond = torch.from_numpy(np.fromfile(os.path.join(out, "cap_uncond.bin"), dtype="<f4")).view(-1, width)
            with open(os.path.join(out, "meta.json")) as f:
                meta = {**json.load(f), **meta}

        noise = torch.randn((1, 16, h // 8, w // 8), generator=torch.Generator().manual_seed(5))
        meta["noise"] = save(out, "noise", noise[0])

        def latents(n, gs):
            kw = {"negative_prompt_embeds": [cap_uncond]} if gs > 0 else {}
            return pipe(
                prompt_embeds=[cap],
                height=h,
                width=w,
                num_inference_steps=n,
                guidance_scale=gs,
                latents=noise.clone(),
                output_type="latent",
                **kw,
            ).images

        def pixels(lat):
            z = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            return pipe.vae.decode(z).sample[0]

        meta["one_step"] = save(out, "one_step", latents(1, 0.0)[0])
        meta["decoded"] = save(out, "decoded", pixels(noise))
        meta["runs"] = []
        for name, gs in (("unguided", 0.0), ("guided", spec["guidance"])):
            meta["runs"].append([name, gs])
            lat = latents(steps, gs)
            meta[name + "_latents"] = save(out, name + "_latents", lat[0])
            meta[name + "_pixels"] = save(out, name + "_pixels", pixels(lat))

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    if sys.argv[1] == "--checkpoint":
        ckpt, out = sys.argv[2], sys.argv[3]
        os.makedirs(out, exist_ok=True)
        link = os.path.join(out, "checkpoint")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(ckpt), link)
        stage = sys.argv[4] if len(sys.argv) > 4 else "all"
        if stage == "text":
            # Text features only, from the text encoder and tokenizer.
            pipe = ZImagePipeline.from_pretrained(ckpt, torch_dtype=torch.float32, transformer=None, vae=None)
            with torch.no_grad():
                meta = {"prompt": PROMPT, **REAL}
                for key, prompt in (("cond_ids", PROMPT), ("uncond_ids", "")):
                    templated = pipe.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    meta[key] = pipe.tokenizer(templated, truncation=True, max_length=512)["input_ids"]
                meta["cap"] = save(out, "cap", pipe._encode_prompt(PROMPT, device="cpu")[0])
                meta["cap_uncond"] = save(out, "cap_uncond", pipe._encode_prompt("", device="cpu")[0])
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
        else:
            pipe = ZImagePipeline.from_pretrained(ckpt, torch_dtype=torch.float32, text_encoder=None, tokenizer=None)
            reference(pipe, out, REAL, text=False)
    else:
        out, released = sys.argv[1], sys.argv[2]
        os.makedirs(out, exist_ok=True)
        reference(build(out, released), out, SMALL)


if __name__ == "__main__":
    main()
