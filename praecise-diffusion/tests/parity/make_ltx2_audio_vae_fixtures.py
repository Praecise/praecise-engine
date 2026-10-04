"""Build a small random LTX-2.3 audio decoder and reference outputs.

The released audio decoder has the layout below at widths up to 512; this
suite checks the native decoder against the reference one on a random
checkpoint with the same layout (pixel norm, time-causal convolutions, three
levels, two upsamplings, stereo output) at tiny widths. The weights are
written as a single file under the release's original names with the
configuration in the header metadata. Latents go through the reference
pipeline's own denormalisation and unpacking. Weights are rounded to
bfloat16 first.

Usage: python make_ltx2_audio_vae_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
from safetensors.torch import save_file

# The latent statistics have one entry per packed slot, which the reference
# sizes by the base width: keep base width = latent channels x latent mel bins.
DD = {
    "double_z": True, "mel_bins": 16, "z_channels": 4, "resolution": 64, "downsample_time": False,
    "in_channels": 2, "out_ch": 2, "ch": 16, "ch_mult": [1, 2, 4], "num_res_blocks": 1,
    "attn_resolutions": [], "dropout": 0.0, "mid_block_add_attention": False,
    "norm_type": "pixel", "causality_axis": "height",
}


def dump(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Audio(
        base_channels=DD["ch"], output_channels=DD["out_ch"], ch_mult=tuple(DD["ch_mult"]),
        num_res_blocks=DD["num_res_blocks"], attn_resolutions=None, in_channels=DD["in_channels"],
        resolution=DD["resolution"], latent_channels=DD["z_channels"], norm_type="pixel",
        causality_axis="height", mel_bins=DD["mel_bins"],
    ).eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_((torch.randn_like(p) * 0.2).to(torch.bfloat16).to(torch.float32))
        vae.latents_mean.copy_(torch.randn_like(vae.latents_mean).to(torch.bfloat16).float() * 0.3)
        vae.latents_std.copy_((torch.rand_like(vae.latents_std) + 0.5).to(torch.bfloat16).float())
    sd = {}
    for k, v in vae.state_dict().items():
        if k.startswith("decoder."):
            sd["audio_vae." + k] = v.contiguous()
    sd["audio_vae.per_channel_statistics.mean-of-means"] = vae.latents_mean.clone()
    sd["audio_vae.per_channel_statistics.std-of-means"] = vae.latents_std.clone()
    header = {"audio_vae": {"model": {"params": {"ddconfig": DD, "sampling_rate": 16000}}}}
    save_file(sd, os.path.join(out, "single.safetensors"), metadata={"config": json.dumps(header)})
    frames = 5
    latent_mel = DD["mel_bins"] // 4
    packed = torch.randn(1, frames, DD["z_channels"] * latent_mel)
    with torch.no_grad():
        z = LTX2Pipeline._denormalize_audio_latents(packed, vae.latents_mean, vae.latents_std)
        z = LTX2Pipeline._unpack_audio_latents(z, frames, num_mel_bins=latent_mel)
        mel = vae.decode(z, return_dict=False)[0]
    dump(out, "packed", packed[0])
    dump(out, "mel", mel[0])
    meta = dict(frames=frames, out_shape=list(mel.shape[1:]))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(meta)


if __name__ == "__main__":
    main()
