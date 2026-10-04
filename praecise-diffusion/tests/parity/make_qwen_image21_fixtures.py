"""Build a small random Qwen-Image 2.1 transformer and reference outputs.

The released transformer has 7B parameters, so this suite checks the native
transformer against the reference one on a random checkpoint with the
released layout (single-stream blocks, shared scale-and-gate modulation,
zero-centred text norm, block-causal joint attention with timestep-zero
condition tokens, three-axis rotary positions) at tiny widths. The reference
runs without its key/value cache, so the native prefill-then-target split is
checked against the full block-causal pass. Cases: a prompt with two
condition images between text runs, and a text-only prompt; each at two
timesteps against the same prefix. Weights are rounded to bfloat16 first.

A tiny autoencoder of the released layout (residual stages with averaging
and duplicating shortcuts, one stage that halves time, RGBA pixels, a wider
decoder) is checked on single images both ways: decoding a normalised latent
and encoding pixels to the normalised mean.

Needs a diffusers build that has QwenImage21Transformer2DModel.

Usage: python make_qwen_image21_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel

TINY = dict(
    patch_size=1,
    in_channels=8,
    out_channels=8,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    context_in_dim=24,
    mlp_ratio=3,
    axes_dims_rope=(4, 6, 6),
    causal_condition=True,
)

VAE = dict(
    base_dim=8,
    decoder_base_dim=12,
    z_dim=4,
    dim_mult=[1, 2, 2],
    num_res_blocks=1,
    temperal_downsample=[False, True],
    in_channels=4,
    out_channels=4,
    scale_factor_spatial=4,
)
LATENT = (5, 6)

SLOT = 4
TIMES = [0.73, 0.21]
# ("text", n) or ("image", rows, cols); the last entry is the target.
CASES = {
    "edit": [("text", 5), ("image", 2, 4), ("text", 3), ("image", 4, 2), ("text", 2), ("image", 4, 4)],
    "t2i": [("text", 6), ("image", 4, 6)],
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
    model = QwenImage21Transformer2DModel(**TINY).eval()
    randomise(model, 1)
    model.save_pretrained(os.path.join(out, "checkpoint", "transformer"), safe_serialization=True)
    g = torch.Generator().manual_seed(2)
    cases = []
    for tag, layout in CASES.items():
        *prefix, (_, th, tw) = layout
        # The prompt sequence: text tokens, and one slot per SLOT latent tokens
        # of every condition image.
        slots = []
        for s in prefix:
            slots += [False] * s[1] if s[0] == "text" else [True] * (s[1] * s[2] // SLOT)
        n_vlm = len(slots)
        states = torch.randn(1, n_vlm, TINY["context_in_dim"], generator=g)
        img_mask = torch.tensor([slots + [True] * (th * tw // SLOT)])
        text = states[0][~torch.tensor(slots, dtype=torch.bool)] if any(slots) else states[0]
        save(out, f"{tag}_text", text)
        n_cond = sum(s[1] * s[2] for s in prefix if s[0] == "image")
        cond = torch.randn(1, n_cond, TINY["in_channels"], generator=g)
        save(out, f"{tag}_cond", cond)
        shapes = [(1, s[1], s[2]) for s in layout if s[0] == "image"]
        for k, t in enumerate(TIMES):
            target = torch.randn(1, th * tw, TINY["in_channels"], generator=g)
            save(out, f"{tag}_target{k}", target)
            with torch.no_grad():
                y = model(
                    hidden_states=torch.cat([cond, target], dim=1),
                    encoder_hidden_states=states,
                    timestep=torch.tensor([t]),
                    img_shapes=[shapes],
                    img_mask=img_mask,
                    return_dict=False,
                )[0]
            save(out, f"{tag}_out{k}", y[:, -th * tw :])
        cases.append(dict(tag=tag, layout=[list(s[1:]) if s[0] == "image" else s[1] for s in layout]))
    gv = torch.Generator().manual_seed(3)
    mean = torch.randn(VAE["z_dim"], generator=gv).mul(0.5).bfloat16().float()
    std = torch.rand(VAE["z_dim"], generator=gv).add(0.5).bfloat16().float()
    vae = AutoencoderKLQwenImage21(**VAE, latents_mean=mean.tolist(), latents_std=std.tolist()).eval()
    randomise(vae, 4)
    vae.save_pretrained(os.path.join(out, "checkpoint", "vae"), safe_serialization=True)
    lh, lw = LATENT
    z = torch.randn(1, VAE["z_dim"], lh, lw, generator=gv)
    save(out, "vae_latent", z)
    s = VAE["scale_factor_spatial"]
    px = torch.rand(1, VAE["in_channels"], lh * s, lw * s, generator=gv) * 2 - 1
    save(out, "vae_pixels", px)
    with torch.no_grad():
        raw = z * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        dec = vae.decode(raw[:, :, None]).sample[:, :, 0]
        mu = vae.encode(px[:, :, None]).latent_dist.mode()[:, :, 0]
        enc = (mu - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    save(out, "vae_decoded", dec)
    save(out, "vae_encoded", enc)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(times=TIMES, cases=cases, latent=list(LATENT), scale=s), f, indent=1)


if __name__ == "__main__":
    main()
