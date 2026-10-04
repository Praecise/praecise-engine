"""Build a small random MiniMax-H3 audio autoencoder and reference outputs.

The released layout at tiny widths: a strided snake-activated convolutional
encoder (weight-normed, strides 2 and 5), the causal attention projection to
the latent width (heads averaged, then adaptively pooled), the mean head, and
an anti-aliased AMP decoder (weight-normed, transposed-convolution
upsampling by 5 and 2, clamped output). Latents are saved normalised with the
checkpoint's latents_mean / latents_std.

Usage: python make_minimax_h3_audio_vae_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLMiniMaxH3Audio

sys.path.insert(0, os.path.dirname(__file__))
from make_qwen_image21_fixtures import randomise, save  # noqa: E402

L = 6
TINY = dict(
    encoder_dim=4, encoder_rates=(2, 5), latent_dim=24, latent_channels=L, num_attention_heads=2, decoder_dim=16,
    decoder_rates=(5, 2), decoder_kernel_sizes=(9, 4), resblock_kernel_sizes=(3, 5),
    resblock_dilation_sizes=((1, 3), (1, 2)), sampling_rate=32000,
)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    g = torch.Generator().manual_seed(2)
    mean = (0.5 * torch.randn(L, generator=g)).tolist()
    std = (0.5 + torch.rand(L, generator=g)).tolist()
    model = AutoencoderKLMiniMaxH3Audio(**TINY, latents_mean=mean, latents_std=std).eval()
    randomise(model, 1)
    with torch.no_grad():
        # Snake frequencies near zero blow up 1 / alpha; keep them near one.
        for name, p in model.named_parameters():
            if name.startswith("encoder") and name.endswith("alpha"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    model.save_pretrained(os.path.join(out, "checkpoint", "audio_vae"), safe_serialization=True)
    m = torch.tensor(mean).view(1, L, 1)
    s = torch.tensor(std).view(1, L, 1)
    cases = {}
    with torch.no_grad():
        wave = torch.rand(1, 1, 75, generator=g) - 0.5
        z = (model.encode(wave).latent_dist.mode() - m) / s
        save(out, "enc_in", wave)
        cases["enc"] = dict(input=75, output=save(out, "enc_out", z)[2])
        z = torch.randn(1, L, 7, generator=g)
        x = model.decode(z * s + m).sample
        save(out, "dec_in", z)
        cases["dec"] = dict(input=7, output=save(out, "dec_out", x)[2])
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(cases=cases), f, indent=1)


if __name__ == "__main__":
    main()
