"""Build a small random LTX-2.3 video decoder and reference outputs.

The released decoder runs at widths up to 1024 over full-resolution video, so
this suite checks the native decoder against the reference one on a random
checkpoint with the released LTX-2.3 stage list (residual groups between a
2x2x2, a second 2x2x2, a temporal and a spatial depth-to-space upsampling,
non-causal frame padding, a 4x4 pixel patch) at tiny widths. The weights are
written as a single file under the release's original names with the
configuration in the header metadata; the reference model is built from the
equivalent configuration. Weights are rounded to bfloat16 first.

Usage: python make_ltx2_vae_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLLTX2Video
from safetensors.torch import save_file

BASE = 8
LATENT = 16
# Released order (pixel side first); the decoder runs it reversed.
BLOCKS = [
    ["res_x", {"num_layers": 2}],
    ["compress_space", {"multiplier": 2}],
    ["res_x", {"num_layers": 1}],
    ["compress_time", {"multiplier": 2}],
    ["res_x", {"num_layers": 2}],
    ["compress_all", {"multiplier": 1}],
    ["res_x", {"num_layers": 1}],
    ["compress_all", {"multiplier": 2}],
    ["res_x", {"num_layers": 1}],
]
HEADER = {
    "_class_name": "CausalVideoAutoencoder", "dims": 3, "in_channels": 3, "out_channels": 3,
    "latent_channels": LATENT, "decoder_blocks": BLOCKS, "scaling_factor": 1.0, "norm_layer": "pixel_norm",
    "patch_size": 4, "causal_decoder": False, "timestep_conditioning": False,
    "decoder_base_channels": BASE, "spatial_padding_mode": "zeros",
}
KIND = {"compress_space": "spatial", "compress_time": "temporal", "compress_all": "spatiotemporal"}


def reference_config():
    """The same decoder in the reference's terms (listed pixel side first)."""
    res = [b[1]["num_layers"] for b in BLOCKS if b[0] == "res_x"]
    ups = [b for b in BLOCKS if b[0] != "res_x"]
    width, outs = BASE, []
    for name, p in ups:
        outs.append(width * p["multiplier"])
        width *= p["multiplier"]
    return dict(
        latent_channels=LATENT, patch_size=4,
        block_out_channels=(8, 16, 16, 16), layers_per_block=(1, 1, 1, 1, 1),
        decoder_block_out_channels=tuple(outs), decoder_layers_per_block=tuple(res),
        decoder_spatio_temporal_scaling=(True,) * len(ups), decoder_inject_noise=(False,) * len(res),
        # The reference reverses every per-block list except this one, which it
        # reads latent side first.
        upsample_type=tuple(KIND[n] for n, _ in reversed(ups)), upsample_residual=(False,) * len(ups),
        upsample_factor=tuple(p["multiplier"] for _, p in ups),
        decoder_causal=False, decoder_spatial_padding_mode="zeros", timestep_conditioning=False,
    )


def original_name(k):
    """Reference decoder name to the release's original name."""
    k = k.replace("resnets", "res_blocks")
    if k.startswith("decoder.mid_block."):
        return "vae.decoder.up_blocks.0." + k[len("decoder.mid_block."):]
    if k.startswith("decoder.up_blocks."):
        i, rest = k[len("decoder.up_blocks."):].split(".", 1)
        i = int(i)
        if rest.startswith("upsamplers.0."):
            return f"vae.decoder.up_blocks.{2 * i + 1}." + rest[len("upsamplers.0."):]
        return f"vae.decoder.up_blocks.{2 * i + 2}." + rest
    return "vae." + k


def dump(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(**reference_config()).eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_((torch.randn_like(p) * 0.2).to(torch.bfloat16).to(torch.float32))
        vae.latents_mean.copy_(torch.randn(LATENT).to(torch.bfloat16).float() * 0.3)
        vae.latents_std.copy_((torch.rand(LATENT) + 0.5).to(torch.bfloat16).float())
    sd = {}
    for k, v in vae.state_dict().items():
        if k.startswith("decoder."):
            sd[original_name(k)] = v.contiguous()
    sd["vae.per_channel_statistics.mean-of-means"] = vae.latents_mean.clone()
    sd["vae.per_channel_statistics.std-of-means"] = vae.latents_std.clone()
    save_file(sd, os.path.join(out, "single.safetensors"), metadata={"config": json.dumps({"vae": HEADER})})
    frames, height, width = 3, 3, 5
    latent = torch.randn(1, LATENT, frames, height, width)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1)
    std = vae.latents_std.view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video = vae.decode(latent * std + mean, return_dict=False)[0]
    dump(out, "latent", latent[0])
    dump(out, "video", video[0])
    meta = dict(frames=frames, height=height, width=width, out_shape=list(video.shape[1:]))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(meta)


if __name__ == "__main__":
    main()
