"""Build a small random Qwen-Image transformer and reference outputs.

The released editing transformer has 20B parameters, so this suite checks
the native transformer against the reference one on a random checkpoint with
the released layout (dual-stream blocks, per-head RMS norms, centred
three-axis rotary positions, timestep-zero modulation of reference tokens)
at tiny widths. Cases: the target alone, and the target followed by two
reference images of other sizes. Weights are rounded to bfloat16 before
anything runs, so the native loader sees exactly the values the reference
computed with.

A tiny autoencoder of the released layout (plain residual stages, middle
attention, causal convolutions) is checked on single images both ways:
decoding a normalised latent and encoding pixels to the normalised mean.

Usage: python make_qwen_image_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel

TINY = dict(
    patch_size=2,
    in_channels=16,
    out_channels=4,
    num_layers=2,
    attention_head_dim=32,
    num_attention_heads=2,
    joint_attention_dim=48,
    axes_dims_rope=(8, 12, 12),
    zero_cond_t=True,
)

VAE = dict(base_dim=16, z_dim=4, dim_mult=[1, 2, 2], num_res_blocks=1, temperal_downsample=[False, True])
LATENT = (6, 5)

TEXT_TOKENS = 7
T = 0.73
CASES = {
    "target": [(1, 4, 6)],
    "refs": [(1, 4, 6), (1, 2, 4), (1, 6, 3)],
}


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
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(**TINY).eval()
    randomise(model, 1)
    model.save_pretrained(os.path.join(out, "checkpoint", "transformer"), safe_serialization=True)
    g = torch.Generator().manual_seed(2)
    text = torch.randn(1, TEXT_TOKENS, TINY["joint_attention_dim"], generator=g)
    save(out, "text", text)
    cases = []
    for tag, shapes in CASES.items():
        n = sum(f * h * w for f, h, w in shapes)
        img = torch.randn(1, n, TINY["in_channels"], generator=g)
        save(out, f"img_{tag}", img)
        with torch.no_grad():
            y = model(
                hidden_states=img,
                encoder_hidden_states=text,
                encoder_hidden_states_mask=torch.ones(1, TEXT_TOKENS),
                timestep=torch.tensor([T]),
                img_shapes=[shapes],
                return_dict=False,
            )[0]
        save(out, f"out_{tag}", y)
        cases.append(dict(tag=tag, images=[[h, w] for _, h, w in shapes]))
    gv = torch.Generator().manual_seed(3)
    mean = torch.randn(VAE["z_dim"], generator=gv).mul(0.5).bfloat16().float()
    std = torch.rand(VAE["z_dim"], generator=gv).add(0.5).bfloat16().float()
    vae = AutoencoderKLQwenImage(**VAE, latents_mean=mean.tolist(), latents_std=std.tolist()).eval()
    randomise(vae, 4)
    vae.save_pretrained(os.path.join(out, "checkpoint", "vae"), safe_serialization=True)
    lh, lw = LATENT
    z = torch.randn(1, VAE["z_dim"], lh, lw, generator=gv)
    save(out, "vae_latent", z)
    s = 2 ** (len(VAE["dim_mult"]) - 1)
    px = torch.rand(1, 3, lh * s, lw * s, generator=gv) * 2 - 1
    save(out, "vae_pixels", px)
    with torch.no_grad():
        raw = z * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        dec = vae.decode(raw[:, :, None]).sample[:, :, 0]
        mu = vae.encode(px[:, :, None]).latent_dist.mode()[:, :, 0]
        enc = (mu - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    save(out, "vae_decoded", dec)
    save(out, "vae_encoded", enc)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(t=T, cases=cases, latent=list(LATENT), scale=s), f, indent=1)


if __name__ == "__main__":
    main()
