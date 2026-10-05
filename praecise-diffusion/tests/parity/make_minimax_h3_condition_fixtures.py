"""Run the MiniMax-H3 keyframe (fl2va) and reference (ref2va) blocks on tiny parts.

Three parts, each in its own sub-directory of <out_dir>:

  checkpoint/  tiny transformer, reference transformer, video and audio
               autoencoders and both schedulers (latent widths matched).
  fl2va/       a first and a last keyframe on a 16x24 canvas: the reference
               keyframe encoder, the fl2va core denoise blocks and the
               decoders, from given prompt states and tags (some tagged
               video, as a vision block is).
  ref2va/      two reference images of other sizes through the reference
               encoder, the ref2va core denoise blocks and the decoders.
  present/     the prompt presentations of both text encoders (token ids
               and tags) with the released tokenizer and image processor,
               the encoder call itself replaced by a recorder.
  lanczos/     PIL Lanczos resampling of one 8-bit image to several sizes.

The request generator, the posterior noise (a fresh generator seeded 42 per
condition) and the conditioning noise drawn from the request generator are
saved; the script checks its replay of the conditioning noise against the
rows the blocks packed.

Usage: QWEN_PROCESSOR=<dir with tokenizer + preprocessor config> \
       python make_minimax_h3_condition_fixtures.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import encoders as h3_encoders
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3DecodeStep,
    MiniMaxH3FL2VACoreDenoiseStep,
    MiniMaxH3Ref2VACoreDenoiseStep,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3FL2VATextEncoderStep,
    MiniMaxH3KeyframeVaeEncoderStep,
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

sys.path.insert(0, os.path.dirname(__file__))
import make_minimax_h3_audio_vae_fixtures as audio_fx  # noqa: E402
import make_minimax_h3_fixtures as tf_fx  # noqa: E402
import make_minimax_h3_vae_fixtures as vae_fx  # noqa: E402
from make_qwen_image21_fixtures import randomise, save  # noqa: E402

FRAMES, HEIGHT, WIDTH, STEPS = 124, 16, 24, 3
TAGS = [1, 1, 0, 0, 0, 1, 1]


def stats(n, g):
    return (0.5 * torch.randn(n, generator=g)).tolist(), (0.5 + torch.rand(n, generator=g)).tolist()


def image(g, h, w):
    return Image.fromarray(torch.randint(0, 256, (h, w, 3), generator=g, dtype=torch.uint8).numpy())


def save_image(out, name, im):
    a = np.asarray(im, dtype=np.uint8)
    a.tofile(os.path.join(out, name + ".rgb"))
    return [a.shape[1], a.shape[0]]


def run(out, blocks, comps, conds_from, extra, g):
    nlat = (FRAMES - 5) // 17 * 5 + 2
    naud = round(FRAMES / 24 * 40)
    lv, la = tf_fx.TINY["in_channels"], tf_fx.TINY["audio_in_channels"]
    text = torch.randn(1, len(TAGS), tf_fx.TINY["text_dim"], generator=g)
    video = torch.randn(1, lv, nlat, HEIGHT // 4, WIDTH // 4, generator=g)
    audio = torch.randn(2, la, naud, generator=g)
    pipe = SequentialPipelineBlocks.from_blocks_dict(blocks).init_pipeline()
    pipe.update_components(**comps)
    with torch.no_grad():
        state = pipe(prompt_embeds=text, text_token_tags=torch.tensor(TAGS), num_frames=FRAMES, height=HEIGHT,
                     width=WIDTH, num_inference_steps=STEPS, latents=video.clone(), audio_latents=audio.clone(),
                     generator=torch.Generator().manual_seed(7), output_type="pt", **extra)
    conds = state.get("condition_latents")
    # Replay: one draw per condition from the request generator; the
    # posterior noise is a fresh seed-42 generator per condition.
    rg = torch.Generator().manual_seed(7)
    noises = [torch.randn(c.shape, generator=rg, dtype=torch.float32) for c in conds]
    meta = dict(conds=[])
    for i, (c, n) in enumerate(zip(conds, noises)):
        meta["conds"].append(list(c.shape[2:]))
        print("cond_noise", i, save(out, f"cond_noise_{i}", n))
        print("eps", i, save(out, f"eps_{i}", torch.randn(c.shape, generator=torch.Generator().manual_seed(42))))
    packed = state.get("condition_rows")
    replay = torch.cat([conds_from(0.999 * c + (1.0 - torch.tensor(0.999)) * n) for c, n in zip(conds, noises)])
    assert torch.allclose(packed, replay, atol=1e-6), (packed - replay).abs().max()
    vids, wav = state.get("videos"), state.get("audio")
    for name, t in [("text", text), ("video_noise", video), ("audio_noise", audio), ("frames", vids[0]),
                    ("waveform", wav[0]), ("conditions", torch.cat([c.flatten() for c in conds]))]:
        print(name, save(out, name, t))
    meta.update(frames=FRAMES, height=HEIGHT, width=WIDTH, steps=STEPS, tags=TAGS, frames_shape=list(vids[0].shape),
                wave_shape=list(wav.shape))
    return meta


def presentation(out):
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    src = os.environ["QWEN_PROCESSOR"]
    tok = Qwen2TokenizerFast.from_pretrained(src)
    proc = Qwen3VLProcessor.from_pretrained(src)
    te = Qwen3VLForConditionalGeneration.from_pretrained(os.environ["QWEN3_VL_TINY"])
    tok.save_pretrained(out)
    seen = []

    def record(text_encoder, processor, token_ids, vision_inputs, **kw):
        seen.append((list(token_ids), vision_inputs))
        return torch.zeros(1, len(token_ids), 4)

    h3_encoders.get_qwen3vl_prompt_embeds = record
    g = torch.Generator().manual_seed(11)
    ims = [image(g, 256, 256), image(g, 256, 320)]
    prompt = "A red kite rises over the dunes at dusk."
    meta = {}
    for name, step, kw in [("fl2va", MiniMaxH3FL2VATextEncoderStep(), dict(keyframes=ims)),
                           ("ref2va", MiniMaxH3Ref2VATextEncoderStep(),
                            dict(normalized_references=[MiniMaxH3ImageReference(image=i) for i in ims]))]:
        pipe = SequentialPipelineBlocks.from_blocks_dict({"text": step}).init_pipeline()
        pipe.update_components(text_encoder=te, tokenizer=tok, processor=proc)
        state = pipe(prompt=prompt, **kw)
        ids, vis = seen[-1]
        meta[name] = dict(ids=ids, tags=state.get("text_token_tags").tolist(),
                          grids=vis["image_grid_thw"].tolist())
    meta["prompt"] = prompt
    meta["sizes"] = [list(i.size) for i in ims]
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)


def lanczos(out):
    g = torch.Generator().manual_seed(13)
    src = image(g, 17, 23)
    meta = dict(src=save_image(out, "src", src), outs=[])
    for i, (w, h) in enumerate([(40, 30), (9, 7), (23, 30), (40, 17), (11, 40)]):
        meta["outs"].append(save_image(out, f"out_{i}", src.resize((w, h), Image.Resampling.LANCZOS)))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)


def main():
    out = sys.argv[1]
    for d in ("fl2va", "ref2va", "present", "lanczos"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    ck = os.path.join(out, "checkpoint")
    g = torch.Generator().manual_seed(3)
    lv, la = tf_fx.TINY["in_channels"], tf_fx.TINY["audio_in_channels"]
    tf = MiniMaxH3Transformer3DModel(**tf_fx.TINY).eval()
    randomise(tf, 1)
    tf_ref = MiniMaxH3Transformer3DModel(**tf_fx.TINY).eval()
    randomise(tf_ref, 4)
    mean, std = stats(lv, g)
    vae = AutoencoderKLMiniMaxH3(**{**vae_fx.TINY, "latent_channels": lv}, latents_mean=mean, latents_std=std).eval()
    randomise(vae, 2)
    with torch.no_grad():
        # Keep the posterior spread small, as a trained encoder's is.
        q = vae.quant_conv
        q.bias[lv:].fill_(-6.0)
    mean, std = stats(la, g)
    avae = AutoencoderKLMiniMaxH3Audio(**{**audio_fx.TINY, "latent_channels": la}, latents_mean=mean, latents_std=std).eval()
    randomise(avae, 3)
    with torch.no_grad():
        for name, p in avae.named_parameters():
            if name.startswith("encoder") and name.endswith("alpha"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    tf.save_pretrained(os.path.join(ck, "transformer"), safe_serialization=True)
    tf_ref.save_pretrained(os.path.join(ck, "transformer_ref"), safe_serialization=True)
    vae.save_pretrained(os.path.join(ck, "vae"), safe_serialization=True)
    avae.save_pretrained(os.path.join(ck, "audio_vae"), safe_serialization=True)
    sched, asched = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sched.save_pretrained(os.path.join(ck, "scheduler"))
    asched.save_pretrained(os.path.join(ck, "audio_scheduler"))
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    def rows(z):
        return patchify_video_latents(z, tuple(tf_fx.TINY["patch_size"]))

    common = dict(vae=vae, audio_vae=avae, scheduler=sched, audio_scheduler=asched)
    gi = torch.Generator().manual_seed(9)
    first, last = image(gi, HEIGHT, WIDTH), image(gi, HEIGHT, WIDTH)
    o = os.path.join(out, "fl2va")
    meta = run(o, {"vae_encode": MiniMaxH3KeyframeVaeEncoderStep(), "denoise": MiniMaxH3FL2VACoreDenoiseStep(),
                   "decode": MiniMaxH3DecodeStep()}, dict(transformer=tf, **common), rows,
               dict(keyframes=[first, last], keyframe_anchors=("first", "last")), torch.Generator().manual_seed(5))
    meta["images"] = [save_image(o, "image_0", first), save_image(o, "image_1", last)]
    meta["anchors"] = ["first", "last"]
    with open(os.path.join(o, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    refs = [image(gi, 16, 16), image(gi, 24, 32)]
    o = os.path.join(out, "ref2va")
    meta = run(o, {"ref_encode": MiniMaxH3Ref2VAReferenceEncoderStep(), "denoise": MiniMaxH3Ref2VACoreDenoiseStep(),
                   "decode": MiniMaxH3DecodeStep()}, dict(transformer_ref=tf_ref, **common), rows,
               dict(normalized_references=[MiniMaxH3ImageReference(image=r) for r in refs]),
               torch.Generator().manual_seed(6))
    meta["images"] = [save_image(o, f"image_{i}", r) for i, r in enumerate(refs)]
    meta["anchors"] = ["reference", "reference"]
    with open(os.path.join(o, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    lanczos(os.path.join(out, "lanczos"))
    presentation(os.path.join(out, "present"))


if __name__ == "__main__":
    main()
