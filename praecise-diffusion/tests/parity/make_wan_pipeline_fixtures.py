"""Reference outputs of a small random Wan2.2 text/image-to-video pipeline.

Builds a tiny checkpoint in the diffusers layout (UMT5 text encoder, 48-channel
video transformer, residual video autoencoder) around the released tokenizer
and scheduler, then runs text-to-video and first-frame-conditioned generation
from saved starting noise.

Usage: python make_wan_pipeline_fixtures.py <out_dir> <released_dir>
  <released_dir> holds the released tokenizer/ and scheduler/ folders.
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

Z = 48
W, H, F = 96, 64, 9
STEPS, GUIDANCE, FPS = 3, 5.0, 24.0
PROMPT = "A  small red boat drifts across a calm lake at dawn."
NEGATIVE = "blurry, low quality"


def randomise(module, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("gamma") or (p.ndim == 1 and name.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))


def save(out, name, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    a.astype("<f4").tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


def build(out, released):
    tok = AutoTokenizer.from_pretrained(os.path.join(released, "tokenizer"))
    te = UMT5EncoderModel(UMT5Config(vocab_size=len(tok), d_model=32, d_kv=8, d_ff=48, num_heads=4, num_layers=2,
                                     relative_attention_num_buckets=32, relative_attention_max_distance=128,
                                     layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu", dropout_rate=0.0))
    randomise(te, 4)
    tf = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=Z,
                               out_channels=Z, text_dim=32, freq_dim=256, ffn_dim=64, num_layers=2,
                               cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None,
                               added_kv_proj_dim=None, rope_max_seq_len=1024)
    randomise(tf, 1)
    g = torch.Generator().manual_seed(3)
    vae = AutoencoderKLWan(base_dim=8, decoder_base_dim=16, z_dim=Z, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                           attn_scales=[], temperal_downsample=[False, True, True], is_residual=True,
                           in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
                           scale_factor_temporal=4,
                           latents_mean=(0.2 * torch.randn(Z, generator=g)).tolist(),
                           latents_std=(0.5 + torch.rand(Z, generator=g)).tolist())
    randomise(vae, 2)
    sched = UniPCMultistepScheduler.from_pretrained(os.path.join(released, "scheduler"))
    pipe = WanPipeline(tokenizer=tok, text_encoder=te.eval(), transformer=tf.eval(), vae=vae.eval(), scheduler=sched,
                       expand_timesteps=True)
    ckpt = os.path.join(out, "checkpoint")
    pipe.save_pretrained(ckpt, safe_serialization=True)
    return pipe


def first_frame():
    y, x = np.mgrid[0:H, 0:W]
    rgb = np.stack([(x * 255 // (W - 1)), (y * 255 // (H - 1)), ((x + y) * 7) % 256], -1).astype(np.uint8)
    return Image.fromarray(rgb)


def frames_of(pipe, latents):
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, Z, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, Z, 1, 1, 1)
    return pipe.vae.decode(latents * std + mean).sample[0].permute(1, 0, 2, 3)


def run(pipe, out, name, image):
    lt = 1 + (F - 1) // 4
    noise = torch.randn(1, Z, lt, H // 16, W // 16, generator=torch.Generator().manual_seed(11))
    save(out, f"{name}_noise", noise[0])
    kw = dict(prompt=PROMPT, negative_prompt=NEGATIVE, height=H, width=W, num_frames=F, num_inference_steps=STEPS,
              guidance_scale=GUIDANCE, latents=noise.clone(), output_type="latent")

    def per_step(_pipe, i, _t, kw2):
        save(out, f"{name}_step{i}", kw2["latents"][0])
        return kw2

    kw["callback_on_step_end"] = per_step
    sched_step = pipe.scheduler.step
    calls = []

    def logged_step(model_output, timestep, sample, *a, **k):
        save(out, f"{name}_velocity{len(calls)}", model_output[0])
        calls.append(int(timestep))
        return sched_step(model_output, timestep, sample, *a, **k)

    pipe.scheduler.step = logged_step
    with torch.no_grad():
        if image is None:
            lat = pipe(**kw).frames
        else:
            i2v = WanImageToVideoPipeline(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                                          transformer=pipe.transformer, vae=pipe.vae, scheduler=pipe.scheduler,
                                          image_processor=None, image_encoder=None, expand_timesteps=True)
            lat = i2v(image=image, **kw).frames
        save(out, f"{name}_latents", lat[0])
        save(out, f"{name}_frames", frames_of(pipe, lat))
    del pipe.scheduler.step


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    out, released = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    pipe = build(out, released)
    for f in os.listdir(os.path.join(released, "tokenizer")):
        shutil.copy(os.path.join(released, "tokenizer", f), os.path.join(out, "checkpoint", "tokenizer", f))
    image = first_frame()
    image.save(os.path.join(out, "first.png"))
    np.asarray(image).astype(np.uint8).tofile(os.path.join(out, "first.rgb"))
    ids = pipe.tokenizer(PROMPT, add_special_tokens=True).input_ids
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, max_sequence_length=512, device="cpu")
        save(out, "prompt_states", pe[0])
        pipe.scheduler.set_timesteps(STEPS)
        t0 = pipe.scheduler.timesteps[0]
        noise = torch.randn(1, Z, 1 + (F - 1) // 4, H // 16, W // 16, generator=torch.Generator().manual_seed(11))
        v = pipe.transformer(hidden_states=noise, timestep=t0.expand(1), encoder_hidden_states=pe).sample
        save(out, "t2v_first_velocity", v[0])
    run(pipe, out, "t2v", None)
    run(pipe, out, "i2v", image)
    json.dump(dict(prompt=PROMPT, negative=NEGATIVE, width=W, height=H, frames=F, steps=STEPS, guidance=GUIDANCE,
                   fps=FPS, seed=11, prompt_ids=ids), open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
