"""Run the MiniMax-H3 text-to-video-and-audio blocks end to end on tiny parts.

The tiny transformer, video and audio autoencoders of the component fixtures
(latent widths matched to each other) go through the reference modular
blocks from the packed layout to the decoded outputs: layout and rotary
grid, noise packing, the two shifted schedules, the denoising loop, the
unpacking and both decoders. Prompt states are given directly (the prompt
encoder has its own fixture); the noise is fixed and saved. The shortest
clip the pipeline accepts (124 frames) at 16x24.

Usage: python make_minimax_h3_pipeline_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3CoreDenoiseStep,
    MiniMaxH3DecodeStep,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

sys.path.insert(0, os.path.dirname(__file__))
import make_minimax_h3_audio_vae_fixtures as audio_fx  # noqa: E402
import make_minimax_h3_fixtures as tf_fx  # noqa: E402
import make_minimax_h3_vae_fixtures as vae_fx  # noqa: E402
from make_qwen_image21_fixtures import randomise, save  # noqa: E402

FRAMES, HEIGHT, WIDTH, STEPS, NT = 124, 16, 24, 3, 5


def stats(n, g):
    return (0.5 * torch.randn(n, generator=g)).tolist(), (0.5 + torch.rand(n, generator=g)).tolist()


def main():
    out = sys.argv[1]
    ck = os.path.join(out, "checkpoint")
    g = torch.Generator().manual_seed(3)
    lv, la = tf_fx.TINY["in_channels"], tf_fx.TINY["audio_in_channels"]
    tf = MiniMaxH3Transformer3DModel(**tf_fx.TINY).eval()
    randomise(tf, 1)
    mean, std = stats(lv, g)
    vae = AutoencoderKLMiniMaxH3(**{**vae_fx.TINY, "latent_channels": lv}, latents_mean=mean, latents_std=std).eval()
    randomise(vae, 2)
    mean, std = stats(la, g)
    avae = AutoencoderKLMiniMaxH3Audio(**{**audio_fx.TINY, "latent_channels": la}, latents_mean=mean, latents_std=std).eval()
    randomise(avae, 3)
    with torch.no_grad():
        for name, p in avae.named_parameters():
            if name.startswith("encoder") and name.endswith("alpha"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    tf.save_pretrained(os.path.join(ck, "transformer"), safe_serialization=True)
    vae.save_pretrained(os.path.join(ck, "vae"), safe_serialization=True)
    avae.save_pretrained(os.path.join(ck, "audio_vae"), safe_serialization=True)
    sched, asched = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sched.save_pretrained(os.path.join(ck, "scheduler"))
    asched.save_pretrained(os.path.join(ck, "audio_scheduler"))

    blocks = SequentialPipelineBlocks.from_blocks_dict({"denoise": MiniMaxH3CoreDenoiseStep(),
                                                        "decode": MiniMaxH3DecodeStep()})
    pipe = blocks.init_pipeline()
    pipe.update_components(transformer=tf, vae=vae, audio_vae=avae, scheduler=sched, audio_scheduler=asched)
    nlat = (FRAMES - 5) // 17 * 5 + 2
    naud = round(FRAMES / 24 * 40)
    text = torch.randn(1, NT, tf_fx.TINY["text_dim"], generator=g)
    video = torch.randn(1, lv, nlat, HEIGHT // 4, WIDTH // 4, generator=g)
    audio = torch.randn(2, la, naud, generator=g)
    with torch.no_grad():
        state = pipe(prompt_embeds=text, text_token_tags=torch.full((NT,), 1, dtype=torch.long), num_frames=FRAMES,
                     height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, latents=video.clone(),
                     audio_latents=audio.clone(), output_type="pt")
    vids = state.get("videos")
    wav = state.get("audio")
    for name, t in [("text", text), ("video_noise", video), ("audio_noise", audio), ("frames", vids[0]),
                    ("waveform", wav[0]), ("latents", state.get("latents")[0]),
                    ("audio_latents", state.get("audio_latents"))]:
        print(name, save(out, name, t))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(frames=FRAMES, height=HEIGHT, width=WIDTH, steps=STEPS, latent_frames=nlat,
                       audio_latents=naud, frames_shape=list(vids[0].shape), wave_shape=list(wav.shape)), f, indent=1)


if __name__ == "__main__":
    main()
