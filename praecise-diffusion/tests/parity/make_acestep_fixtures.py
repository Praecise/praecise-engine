"""Build a small random ACE-Step 1.5 checkpoint and reference outputs.

The checkpoint has the real layout and the real tokenizer but tiny widths, so
the native pipeline can be checked against the reference implementation on a
CPU in seconds. Weights are rounded to bfloat16 before anything runs, so the
native loader sees exactly the values the reference computed with.

Usage: python make_acestep_fixtures.py <out_dir> <tokenizer_dir>
       python make_acestep_fixtures.py --checkpoint <checkpoint_dir> <out_dir>

<tokenizer_dir> holds the checkpoint's tokenizer files (tokenizer.json and
friends); only the tokenizer is taken from a real checkpoint. With
--checkpoint, the reference outputs are computed from a released checkpoint
in float32 instead, over a few seconds of audio, and `<out_dir>/checkpoint`
links to it.
"""

import json
import math
import os
import sys

import numpy as np
import torch
from diffusers import (
    AceStepConditionEncoder,
    AceStepPipeline,
    AceStepTransformer1DModel,
    AutoencoderOobleck,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

PROMPT = "warm lo-fi hip hop, mellow piano, vinyl crackle"
LYRICS = "[verse]\nsoft rain on the window\nslow light in the hall"
LANGUAGE = "en"
BPM = 84
KEYSCALE = "A minor"
TIMESIG = "4"
# The test sample rate and strides give 20 latent frames a second; 1.55 s is
# 31 frames, an odd count, so the transformer's patch padding is exercised.
# The strides are even, as in the released autoencoder.
SAMPLE_RATE = 160
DURATION = 1.55
STEPS = 4
# Long enough that the 128-position attention window bites: 12 s is 300
# latent frames, 150 transformer positions.
REAL_DURATION = 12.0
SHIFT = 3.0
GUIDANCE = 4.0
# Narrower than the real 128 so the sliding-window masks bite on short inputs.
WINDOW = 4
# 64 is the narrowest head the GPU attention kernels serve.
HEAD = 64


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


def build(out, tok_dir):
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    te = Qwen3Model(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=HEAD,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            max_position_embeddings=32768,
        )
    ).eval()
    randomise(te, 1)

    acoustic = 8
    cond = AceStepConditionEncoder(
        hidden_size=96,
        intermediate_size=160,
        text_hidden_dim=48,
        timbre_hidden_dim=acoustic,
        num_lyric_encoder_hidden_layers=3,
        num_timbre_encoder_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        sliding_window=WINDOW,
    ).eval()
    randomise(cond, 2)
    with torch.no_grad():
        g = torch.Generator().manual_seed(6)
        sil = 0.5 * torch.randn(cond.silence_latent.shape, generator=g)
        cond.silence_latent.copy_(sil.to(torch.bfloat16).float())

    transformer = AceStepTransformer1DModel(
        hidden_size=128,
        intermediate_size=224,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD,
        in_channels=3 * acoustic,
        audio_acoustic_hidden_dim=acoustic,
        patch_size=2,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        sliding_window=WINDOW,
        encoder_hidden_size=96,
        is_turbo=False,
        model_version="base",
    ).eval()
    randomise(transformer, 3)

    vae = AutoencoderOobleck(
        encoder_hidden_size=8,
        downsampling_ratios=[2, 4],
        channel_multiples=[1, 2],
        decoder_channels=8,
        decoder_input_channels=acoustic,
        audio_channels=2,
        sampling_rate=SAMPLE_RATE,
    ).eval()
    randomise(vae, 4)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, use_dynamic_shifting=False)
    pipe = AceStepPipeline(
        vae=vae,
        text_encoder=te,
        tokenizer=tokenizer,
        transformer=transformer,
        condition_encoder=cond,
        scheduler=scheduler,
    )
    pipe.save_pretrained(ckpt, safe_serialization=True)
    # Store the checkpoint in bfloat16 like the real one.
    from safetensors.torch import load_file, save_file

    for sub in ("transformer", "vae", "text_encoder", "condition_encoder"):
        d = os.path.join(ckpt, sub)
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                p = os.path.join(d, f)
                sd = load_file(p)
                save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}, p)
    return pipe


def reference(pipe, out, duration, steps, guidances):
    tokenizer, cond, transformer, vae = pipe.tokenizer, pipe.condition_encoder, pipe.transformer, pipe.vae
    acoustic = transformer.config.audio_acoustic_hidden_dim
    meta = {
        "prompt": PROMPT,
        "lyrics": LYRICS,
        "language": LANGUAGE,
        "bpm": BPM,
        "keyscale": KEYSCALE,
        "timesignature": TIMESIG,
        "duration": duration,
        "steps": steps,
        "shift": SHIFT,
    }
    with torch.no_grad():
        text_str, lyric_str = pipe._format_prompt(
            prompt=PROMPT,
            lyrics=LYRICS,
            vocal_language=LANGUAGE,
            audio_duration=duration,
            instruction=None,
            bpm=BPM,
            keyscale=KEYSCALE,
            timesignature=TIMESIG,
        )
        meta["text_tokens"] = tokenizer(text_str, truncation=True, max_length=256)["input_ids"]
        meta["lyric_tokens"] = tokenizer(lyric_str, truncation=True, max_length=2048)["input_ids"]
        th, tm, lh, lm = pipe.encode_prompt(
            prompt=PROMPT,
            lyrics=LYRICS,
            device="cpu",
            vocal_language=LANGUAGE,
            audio_duration=duration,
            bpm=BPM,
            keyscale=KEYSCALE,
            timesignature=TIMESIG,
        )
        meta["text_hidden"] = save(out, "text_hidden", th[0])
        meta["lyric_embeds"] = save(out, "lyric_embeds", lh[0])

        frames = math.ceil(duration * pipe.latents_per_second)
        timbre = math.ceil(30 * pipe.latents_per_second)
        refer = cond.silence_latent[:, :timbre, :]
        enc, _ = cond(
            text_hidden_states=th,
            text_attention_mask=tm,
            lyric_hidden_states=lh,
            lyric_attention_mask=lm,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=torch.arange(1, dtype=torch.long),
        )
        meta["encoder_hidden"] = save(out, "encoder_hidden", enc[0])

        noise = torch.randn((1, frames, acoustic), generator=torch.Generator().manual_seed(5))
        meta["noise"] = save(out, "noise", noise[0])

        # One transformer evaluation at a mid noise level, on the text-to-music
        # context (silence source latents, all-ones chunk mask).
        src = cond.silence_latent[:, :frames, :]
        context = torch.cat([src, torch.ones_like(src)], dim=-1)
        t = torch.tensor([0.7])
        v = transformer(
            hidden_states=noise,
            timestep=t,
            timestep_r=t,
            encoder_hidden_states=enc,
            context_latents=context,
            return_dict=False,
        )[0]
        meta["dit_t"] = 0.7
        meta["dit_out"] = save(out, "dit_out", v[0])

        # Decode of the noise as if it were final latents.
        audio = vae.decode(noise.transpose(1, 2)).sample
        meta["decoded"] = save(out, "decoded", audio[0])

        # The whole pipeline from the same starting noise, without and with
        # guidance.
        meta["runs"] = []
        for name, gs in zip(("pipeline", "pipeline_cfg"), guidances):
            meta["runs"].append([name, gs])
            res = pipe(
                prompt=PROMPT,
                lyrics=LYRICS,
                audio_duration=duration,
                vocal_language=LANGUAGE,
                num_inference_steps=steps,
                guidance_scale=gs,
                shift=SHIFT,
                latents=noise.clone(),
                bpm=BPM,
                keyscale=KEYSCALE,
                timesignature=TIMESIG,
                output_type="latent",
            ).audios
            meta[name + "_latents"] = save(out, name + "_latents", res[0])
            res = pipe(
                prompt=PROMPT,
                lyrics=LYRICS,
                audio_duration=duration,
                vocal_language=LANGUAGE,
                num_inference_steps=steps,
                guidance_scale=gs,
                shift=SHIFT,
                latents=noise.clone(),
                bpm=BPM,
                keyscale=KEYSCALE,
                timesignature=TIMESIG,
            ).audios
            meta[name] = save(out, name, res[0])

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    if sys.argv[1] == "--checkpoint":
        ckpt, out = sys.argv[2], sys.argv[3]
        os.makedirs(out, exist_ok=True)
        link = os.path.join(out, "checkpoint")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(ckpt), link)
        pipe = AceStepPipeline.from_pretrained(ckpt, torch_dtype=torch.float32)
        turbo = pipe.is_turbo
        reference(pipe, out, REAL_DURATION, 8 if turbo else 30, (1.0,) if turbo else (1.0, 7.0))
    else:
        out, tok_dir = sys.argv[1], sys.argv[2]
        pipe = build(out, tok_dir)
        reference(pipe, out, DURATION, STEPS, (1.0, GUIDANCE))


if __name__ == "__main__":
    main()
