"""Build a small random FLUX.2 [klein] checkpoint and reference outputs.

The checkpoint has the real layout and the real tokenizer but tiny widths, so
the native pipeline can be checked against the reference implementation on a
CPU in seconds. Weights are rounded to bfloat16 before anything runs, so the
native loader sees exactly the values the reference computed with.

Usage: python make_fixtures.py <out_dir> <tokenizer_dir>

<tokenizer_dir> holds the checkpoint's tokenizer files (tokenizer.json and
friends); only the tokenizer is taken from a real checkpoint.
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

PROMPT = "a red fox in the snow, photograph"
WIDTH, HEIGHT, STEPS = 32, 48, 3
REF_W, REF_H = 80, 64


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


def main():
    out, tok_dir = sys.argv[1], sys.argv[2]
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    te = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=151936,
            hidden_size=32,
            intermediate_size=64,
            # Deeper than the last conditioning layer (27), as in the real
            # encoder: the final hidden state is normalised, the others are not.
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            # 64 is the narrowest head the GPU attention kernels serve.
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            tie_word_embeddings=True,
            max_position_embeddings=40960,
        )
    ).eval()
    randomise(te, 1)
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=1,
        num_single_layers=2,
        attention_head_dim=64,
        num_attention_heads=2,
        joint_attention_dim=96,
        timestep_guidance_channels=256,
        mlp_ratio=3.0,
        axes_dims_rope=(16, 16, 16, 16),
        rope_theta=2000,
        eps=1e-6,
        guidance_embeds=False,
    ).eval()
    randomise(transformer, 2)
    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        patch_size=(2, 2),
        batch_norm_eps=1e-4,
        use_quant_conv=True,
        use_post_quant_conv=True,
        mid_block_add_attention=True,
    ).eval()
    randomise(vae, 3)
    with torch.no_grad():
        g = torch.Generator().manual_seed(4)
        vae.bn.running_mean.copy_((0.2 * torch.randn(16, generator=g)).to(torch.bfloat16).float())
        vae.bn.running_var.copy_((0.5 + torch.rand(16, generator=g)).to(torch.bfloat16).float())
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        max_shift=1.15,
        num_train_timesteps=1000,
        shift=3.0,
        time_shift_type="exponential",
        use_dynamic_shifting=True,
    )
    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=te,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )
    pipe.save_pretrained(ckpt, safe_serialization=True)
    # Store the checkpoint in bfloat16 like the real one.
    for sub in ("transformer", "vae", "text_encoder"):
        d = os.path.join(ckpt, sub)
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                from safetensors.torch import load_file, save_file

                p = os.path.join(d, f)
                sd = load_file(p)
                save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}, p)

    meta = {"prompt": PROMPT, "width": WIDTH, "height": HEIGHT, "steps": STEPS}
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(text, padding="max_length", truncation=True, max_length=512)["input_ids"]
        meta["tokens"] = ids
        emb = pipe._get_qwen3_prompt_embeds(te, tokenizer, PROMPT, dtype=torch.float32, device="cpu")
        meta["prompt_embeds"] = save(out, "prompt_embeds", emb[0])

        cell = pipe.vae_scale_factor * 2
        gh, gw = HEIGHT // cell, WIDTH // cell
        noise = torch.randn((1, 16, gh, gw), generator=torch.Generator().manual_seed(5))
        # Tokens in row-major order, channels innermost.
        packed = noise.reshape(1, 16, -1).permute(0, 2, 1)
        meta["noise"] = save(out, "noise", packed[0])

        # One transformer evaluation at the first step's noise level.
        latent_ids = pipe._prepare_latent_ids(noise)
        text_ids = pipe._prepare_text_ids(emb)
        sigma = 0.8
        v = transformer(
            hidden_states=packed,
            encoder_hidden_states=emb,
            timestep=torch.tensor([sigma]),
            img_ids=latent_ids,
            txt_ids=text_ids,
            return_dict=False,
        )[0]
        meta["dit_sigma"] = sigma
        meta["dit_out"] = save(out, "dit_out", v[0])

        # Decode of the noise as if it were final latents.
        lat = pipe._unpack_latents_with_ids(packed, latent_ids, gh, gw)
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        lat = pipe._unpatchify_latents(lat * std + mean)
        img = vae.decode(lat, return_dict=False)[0]
        meta["decoded"] = save(out, "decoded", img[0])

        # The whole pipeline from the same starting noise.
        full = pipe(
            prompt=PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            latents=noise,
            output_type="np",
        ).images[0]
        np.asarray(full, dtype="<f4").tofile(os.path.join(out, "pipeline.bin"))
        meta["pipeline"] = list(full.shape)

        # Conditioned on a reference image already at a servable size, so the
        # reference preprocessing is only the normalisation.
        from PIL import Image

        rng = np.random.default_rng(6)
        ref = rng.integers(0, 256, size=(REF_H, REF_W, 3), dtype=np.uint8)
        ref.tofile(os.path.join(out, "reference.bin"))
        meta["reference"] = [REF_H, REF_W]
        edited = pipe(
            prompt=PROMPT,
            image=[Image.fromarray(ref)],
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            latents=noise,
            output_type="np",
        ).images[0]
        np.asarray(edited, dtype="<f4").tofile(os.path.join(out, "pipeline_ref.bin"))

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
