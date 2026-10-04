"""Build a small random MiniMax-H3 video autoencoder and reference outputs.

The released autoencoder layout at tiny widths: a causal 3D CNN encoder with
two spatial and two temporal halvings (the released temporal geometry: 17
frame clips, 3 latent frames dropped), and a ViT decoder with registers and a
partial three-axis rotary embedding. Spatial tiling is on with 32 pixel tiles
so both the tiled encode and the tiled decode blend overlaps. Latents are
saved normalised with the checkpoint's latents_mean / latents_std; pixels are
in the autoencoder's own (ImageNet normalised) space.

Usage: python make_minimax_h3_vae_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLMiniMaxH3

sys.path.insert(0, os.path.dirname(__file__))
from make_qwen_image21_fixtures import randomise, save  # noqa: E402

L = 6
TINY = dict(
    latent_channels=L, block_out_channels=(16, 32, 32), layers_per_block=1, spatial_downsample_factors=(2, 2, 1),
    temporal_downsample_factors=(2, 2, 1), norm_num_groups=8, decoder_num_layers=2, decoder_num_attention_heads=2,
    decoder_attention_head_dim=24, decoder_num_register_tokens=4, decoder_ffn_mult=2, clip_length=17, token_drop=3,
)
TILE = dict(tile_sample_min_height=32, tile_sample_min_width=32, tile_sample_min_overlap_height=8,
            tile_sample_min_overlap_width=8)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    g = torch.Generator().manual_seed(2)
    mean = (0.5 * torch.randn(L, generator=g)).tolist()
    std = (0.5 + torch.rand(L, generator=g)).tolist()
    model = AutoencoderKLMiniMaxH3(**TINY, latents_mean=mean, latents_std=std).eval()
    randomise(model, 1)
    model.save_pretrained(os.path.join(out, "checkpoint", "vae"), safe_serialization=True)
    m = torch.tensor(mean).view(1, L, 1, 1, 1)
    s = torch.tensor(std).view(1, L, 1, 1, 1)
    cases = {}
    model.enable_tiling(**TILE)
    with torch.no_grad():
        for name, shape in [("enc_video", (22, 40, 60)), ("enc_image", (1, 40, 60))]:
            x = torch.randn(1, 3, *shape, generator=g)
            z = (model.encode(x).latent_dist.mode() - m) / s
            save(out, name + "_in", x)
            cases[name] = dict(input=list(shape), output=save(out, name + "_out", z)[2:])
        for name, shape, tiled in [("dec_a", (7, 10, 15), True), ("dec_b", (9, 10, 15), True),
                                   ("dec_c", (3, 10, 15), True), ("dec_plain", (4, 3, 4), False)]:
            if tiled:
                model.enable_tiling(**TILE)
            else:
                model.disable_tiling()
            z = torch.randn(1, L, *shape, generator=g)
            x = model.decode(z * s + m).sample
            save(out, name + "_in", z)
            cases[name] = dict(input=list(shape), output=save(out, name + "_out", x)[2:], tiled=tiled)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(cases=cases, tile=32, overlap=8), f, indent=1)


if __name__ == "__main__":
    main()
