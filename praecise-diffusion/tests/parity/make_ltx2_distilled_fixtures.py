"""Reference fixtures for the LTX-2 distilled two-stage recipe over a split,
partly quantised checkpoint.

The tiny checkpoint of make_ltx2_pipeline_fixtures.py is split the way
quantised releases ship it:

- ckpt/transformer.gguf: the diffusion model (transformer and prompt
  connectors) without its leading `model.diffusion_model.`, block linear
  weights quantised (Q5_K and Q6_K where rows are 256 wide, Q8_0 where they
  are 32-aligned), the header `config` as a string entry;
- ckpt/connectors.safetensors: the text feature projection;
- ckpt/video_vae.safetensors: the video autoencoder without `vae.`;
- ckpt/audio_vae.safetensors: the audio autoencoder and the vocoder;
- ckpt/upsampler.safetensors: a tiny latent upsampler.

K-quant blocks are random bytes with scales chosen to match the original
weights' spread (quantising K blocks is not available here); the reference
model then runs on the dequantised values, so only the backend's handling of
quantised weights differs.

The reference runs the distilled schedule with no guidance on either stream
at half size, upsamples the video latents, re-noises both streams with
recorded noise and refines at full size.

Usage: python make_ltx2_distilled_fixtures.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from gguf import GGMLQuantizationType as Q
from gguf import GGUFWriter
from gguf.quants import dequantize, quantize
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_ltx2_pipeline_fixtures as pf  # noqa: E402
import make_ltx2_vae_fixtures as vvae  # noqa: E402

FRAMES, HEIGHT, WIDTH, FPS = 9, 128, 192, 24.0
MID, BLOCKS = 64, 1
PREFIX = "model.diffusion_model."
BLOCK_BYTES = {Q.Q5_K: 176, Q.Q6_K: 210}


def k_quant(w, qtype, rng):
    """Random K-quant blocks whose dequantised values have `w`'s spread."""
    rows, cols = w.shape
    nb = rows * cols // 256
    raw = rng.integers(0, 256, size=(nb, BLOCK_BYTES[qtype]), dtype=np.uint8)

    def deq(d, dmin=None):
        b = raw.copy()
        if qtype == Q.Q5_K:
            b[:, 0:2] = np.frombuffer(np.float16(d).tobytes(), np.uint8)
            b[:, 2:4] = np.frombuffer(np.float16(dmin).tobytes(), np.uint8)
        else:
            b[:, 208:210] = np.frombuffer(np.float16(d).tobytes(), np.uint8)
        return b, dequantize(b.reshape(rows, -1), qtype).reshape(rows, cols)

    target = float(w.std())
    if qtype == Q.Q5_K:
        _, scaled = deq(1.0, 0.0)
        _, offset = deq(0.0, 1.0)
        d = target / float(scaled.std())
        dmin = d * float(scaled.mean()) / float(-offset.mean())
        raw, values = deq(d, dmin)
    else:
        _, scaled = deq(1.0)
        raw, values = deq(target / float(scaled.std()))
    return raw.reshape(rows, -1), values


def main():
    out = sys.argv[1]
    ck = os.path.join(out, "ckpt")
    os.makedirs(ck, exist_ok=True)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    r = pf.build(out, scheduler)
    os.remove(os.path.join(out, "single.safetensors"))
    pipe, transformer, sd = r.pipe, r.transformer, r.sd

    # Quantise the transformer blocks' linear weights; the reference model
    # takes the dequantised values.
    rng = np.random.default_rng(7)
    original = {pf.tf.original_name(k): k for k in transformer.state_dict()}
    quantised, k_turn = {}, 0
    with torch.no_grad():
        params = dict(transformer.named_parameters())
        for name in sorted(sd):
            t = sd[name]
            if not (name.startswith(PREFIX + "transformer_blocks.") and name.endswith(".weight") and t.ndim == 2):
                continue
            w = t.float().numpy()
            if w.shape[1] % 256 == 0:
                qtype = (Q.Q5_K, Q.Q6_K)[k_turn % 2]
                k_turn += 1
                raw, values = k_quant(w, qtype, rng)
            elif w.shape[1] % 32 == 0:
                qtype = Q.Q8_0
                raw = quantize(w, qtype)
                values = dequantize(raw, qtype).reshape(w.shape)
            else:
                continue
            quantised[name] = (raw, qtype)
            values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
            params[original[name]].copy_(values)
            sd[name] = values
    print("quantised:", {q.name: sum(1 for v in quantised.values() if v[1] == q) for q in (Q.Q5_K, Q.Q6_K, Q.Q8_0)})

    gw = GGUFWriter(os.path.join(ck, "transformer.gguf"), "ltxv")
    gw.add_string("config", json.dumps(pf.HEADER))
    for name in sorted(k for k in sd if k.startswith(PREFIX)):
        short = name[len(PREFIX):]
        if name in quantised:
            raw, qtype = quantised[name]
            gw.add_tensor(short, raw, raw_dtype=qtype)
        else:
            gw.add_tensor(short, sd[name].float().numpy())
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    save_file({k: v for k, v in sd.items() if k.startswith("text_embedding_projection.")}, os.path.join(ck, "connectors.safetensors"))
    save_file({k[len("vae."):]: v for k, v in sd.items() if k.startswith("vae.")}, os.path.join(ck, "video_vae.safetensors"))
    save_file({k: v for k, v in sd.items() if k.startswith(("audio_vae.", "vocoder."))}, os.path.join(ck, "audio_vae.safetensors"))

    up = LTX2LatentUpsamplerModel(
        in_channels=vvae.LATENT, mid_channels=MID, num_blocks_per_stage=BLOCKS, dims=3, spatial_upsample=True,
        temporal_upsample=False, rational_spatial_scale=2.0, use_rational_resampler=False,
    ).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(11)
        for name, p in up.named_parameters():
            if "norm" in name and name.endswith("weight"):
                v = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
            elif name.endswith("bias"):
                v = 0.05 * torch.randn(p.shape, generator=g)
            else:
                fan_in = p[0].numel()
                v = torch.randn(p.shape, generator=g) / fan_in ** 0.5
            p.copy_(v.to(torch.bfloat16).float())
    up_cfg = {
        "_class_name": "LatentUpsampler", "in_channels": vvae.LATENT, "mid_channels": MID, "num_blocks_per_stage": BLOCKS,
        "dims": 3, "spatial_upsample": True, "temporal_upsample": False, "spatial_scale": 2.0, "rational_resampler": False,
    }
    save_file({k: v.contiguous() for k, v in up.state_dict().items()}, os.path.join(ck, "upsampler.safetensors"),
              metadata={"config": json.dumps(up_cfg)})

    lf = (FRAMES - 1) // 8 + 1
    h1, w1 = HEIGHT // 2, WIDTH // 2
    duration = FRAMES / FPS
    audio_frames = round(duration * 16000 / 160 / 4)
    g = torch.Generator().manual_seed(3)
    video_noise = torch.randn(1, vvae.LATENT, lf, h1 // 32, w1 // 32, generator=g)
    audio_noise = torch.randn(1, pf.avae.DD["z_channels"], audio_frames, pf.MELS // 4, generator=g)
    pos, pos_mask = pf.prompt_embeds(r.gemma, pf.POSITIVE)

    start, final = {}, {}
    forward = transformer.forward

    def first(*a, **k):
        start.setdefault("video", k["hidden_states"][0].clone())
        start.setdefault("audio", k["audio_hidden_states"][0].clone())
        return forward(*a, **k)

    transformer.forward = first

    def grab(p, i, t, kw):
        final["video"], final["audio"] = kw["latents"].clone(), kw["audio_latents"].clone()
        return {}

    off = dict(guidance_scale=1.0, stg_scale=0.0, modality_scale=1.0, guidance_rescale=0.0, audio_guidance_scale=1.0,
               audio_stg_scale=0.0, audio_modality_scale=1.0, audio_guidance_rescale=0.0)
    common = dict(prompt_embeds=pos, prompt_attention_mask=pos_mask, num_frames=FRAMES, frame_rate=FPS, return_dict=False,
                  max_sequence_length=pf.SEQ, callback_on_step_end=grab,
                  callback_on_step_end_tensor_inputs=["latents", "audio_latents"], **off)
    with torch.no_grad():
        v1, a1 = pipe(height=h1, width=w1, sigmas=DISTILLED_SIGMA_VALUES, latents=video_noise, audio_latents=audio_noise,
                      output_type="latent", **common)
        sigmas1 = pipe.scheduler.sigmas.tolist()
        dump = pf.dump
        dump(out, "video_start", start["video"])
        dump(out, "audio_start", start["audio"])
        dump(out, "stage1_video", final["video"][0])
        dump(out, "stage1_audio", final["audio"][0])
        upsampled = up(v1)
        dump(out, "up_in", v1[0])
        dump(out, "up_out", upsampled[0])

        noises = []

        def noised(latents, noise_scale, generator=None):
            noise = torch.randn(latents.shape, generator=g)
            noises.append(noise[0].clone())
            return noise_scale * noise + (1 - noise_scale) * latents

        pipe._create_noised_state = noised
        start.clear()
        video, audio = pipe(height=HEIGHT, width=WIDTH, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, latents=upsampled,
                            audio_latents=a1, noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], output_type="pt", **common)
        sigmas2 = pipe.scheduler.sigmas.tolist()
    assert len(noises) == 2, len(noises)
    dump(out, "refine_video_noise", noises[0])
    dump(out, "refine_audio_noise", noises[1])
    dump(out, "refine_video_start", start["video"])
    dump(out, "refine_audio_start", start["audio"])
    dump(out, "final_video", final["video"][0])
    dump(out, "final_audio", final["audio"][0])
    dump(out, "frames", video[0])
    dump(out, "wave", audio[0])
    json.dump({
        "positive": pf.POSITIVE, "seq_len": pf.SEQ, "frames": FRAMES, "height": HEIGHT, "width": WIDTH, "fps": FPS,
        "audio_frames": audio_frames, "latent1": [lf, h1 // 32, w1 // 32], "latent2": [lf, HEIGHT // 32, WIDTH // 32],
        "sigmas1": sigmas1, "sigmas2": sigmas2, "frames_shape": list(video[0].shape), "wave_shape": list(audio[0].shape),
    }, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
