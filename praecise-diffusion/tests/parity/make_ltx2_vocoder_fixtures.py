"""Build a small random LTX-2.3 vocoder and reference outputs.

The released vocoder (a mel-to-16 kHz generator, then a 48 kHz bandwidth
extension driven by the causal log-mel spectrogram of its output) is checked
here on a random checkpoint with the same layout at tiny widths: anti-aliased
periodic activations, averaged residual blocks with dilations, transposed
convolutions whose kernel is not twice the stride, a causal STFT whose window
is not a whole number of hops, and the windowed-sinc 3x resampler. The
weights are written as a single file under the release's original names with
the configuration in the header metadata. Weights are rounded to bfloat16.

Usage: python make_ltx2_vocoder_fixtures.py <out_dir>
"""

import json
import math
import os
import sys

import torch
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from safetensors.torch import save_file

MELS = 8
HOP = 10
NFFT = 32
VOC = dict(upsample_initial_channel=32, upsample_rates=[5, 2], upsample_kernel_sizes=[11, 4],
           resblock_kernel_sizes=[3, 7], resblock_dilation_sizes=[[1, 3], [1, 3]])
BWE = dict(upsample_initial_channel=32, upsample_rates=[5, 3, 2], upsample_kernel_sizes=[11, 5, 4],
           resblock_kernel_sizes=[3, 5], resblock_dilation_sizes=[[1, 2], [1, 2]])
COMMON = dict(resblock="AMP1", stereo=True, use_tanh_at_final=False, activation="snakebeta", use_bias_at_final=False)
HEADER = {
    "vocoder": {**VOC, **COMMON},
    "bwe": {**BWE, **COMMON, "apply_final_activation": False, "input_sampling_rate": 16000,
            "output_sampling_rate": 48000, "hop_length": HOP, "n_fft": NFFT, "win_size": NFFT, "num_mels": MELS},
}


def original_name(k):
    for a, b in [("conv_in", "conv_pre"), ("upsamplers", "ups"), ("resnets", "resblocks"), ("act_out", "act_post"),
                 ("conv_out", "conv_post"), ("downsample.filter", "downsample.lowpass.filter")]:
        k = k.replace(a, b)
    return "vocoder." + k


def dump(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    voc = LTX2VocoderWithBWE(
        in_channels=2 * MELS, hidden_channels=VOC["upsample_initial_channel"], out_channels=2,
        upsample_kernel_sizes=VOC["upsample_kernel_sizes"], upsample_factors=VOC["upsample_rates"],
        resnet_kernel_sizes=VOC["resblock_kernel_sizes"], resnet_dilations=VOC["resblock_dilation_sizes"],
        bwe_in_channels=2 * MELS, bwe_hidden_channels=BWE["upsample_initial_channel"], bwe_out_channels=2,
        bwe_upsample_kernel_sizes=BWE["upsample_kernel_sizes"], bwe_upsample_factors=BWE["upsample_rates"],
        bwe_resnet_kernel_sizes=BWE["resblock_kernel_sizes"], bwe_resnet_dilations=BWE["resblock_dilation_sizes"],
        filter_length=NFFT, hop_length=HOP, window_length=NFFT, num_mel_channels=MELS,
    ).eval()
    with torch.no_grad():
        for name, p in voc.named_parameters():
            scale = 0.3 if name.endswith(("alpha", "beta")) else 0.15
            p.copy_((torch.randn_like(p) * scale).to(torch.bfloat16).float())
        n = torch.arange(NFFT, dtype=torch.float64)
        k = torch.arange(NFFT // 2 + 1, dtype=torch.float64)[:, None]
        win = torch.hann_window(NFFT, periodic=True, dtype=torch.float64)
        ang = 2 * math.pi * k * n / NFFT
        basis = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win]).float()[:, None, :]
        voc.mel_stft.stft_fn.forward_basis.copy_(basis.to(torch.bfloat16).float())
        voc.mel_stft.stft_fn.inverse_basis.copy_(basis.to(torch.bfloat16).float())
        voc.mel_stft.mel_basis.copy_((torch.rand(MELS, NFFT // 2 + 1) * 0.3).to(torch.bfloat16).float())
    sd = {original_name(k): v.contiguous() for k, v in voc.state_dict().items()}
    save_file(sd, os.path.join(out, "single.safetensors"), metadata={"config": json.dumps({"vocoder": HEADER})})
    frames = 6
    mel = torch.randn(1, 2, frames, MELS) - 1.0
    with torch.no_grad():
        wave = voc(mel)
    dump(out, "mel", mel[0])
    dump(out, "wave", wave[0])
    meta = dict(frames=frames, out_shape=list(wave.shape[1:]))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(meta)


if __name__ == "__main__":
    main()
