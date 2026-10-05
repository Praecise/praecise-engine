"""Build a small random Cosmos3 checkpoint and reference outputs.

The checkpoint has the real layout, the real tokenizer and the real
scheduler, but tiny widths, so the native pipeline can be checked against the
reference implementation on a CPU in seconds. Weights are rounded to bfloat16
before anything runs, so the native loader sees exactly the values the
reference computed with.

Usage: python make_cosmos3_fixtures.py [--layout nano|super] <out_dir> <released_checkpoint_dir>
       python make_cosmos3_fixtures.py --checkpoint <checkpoint_dir> <out_dir>

Only the tokenizer and the scheduler configuration are taken from the
released checkpoint in the first form. `--layout nano` or `--layout super`
builds the joint transformer of those checkpoints (gated SiLU feed-forward,
query/key norms on the text stream, sound projections; Super's query width
differs from its model width) and writes the checkpoint in their older
repository layout (pipeline class, transformer configuration keys, extra
components); the released directory is then the Nano or Super one, for its
Qwen tokenizer. The reference pipeline runs with that layout's defaults
(system prompt on, the scheduler's own sigmas). With --checkpoint, the reference
outputs are computed from the released checkpoint itself in float32, over a
short low-resolution clip, and `<out_dir>/checkpoint` links to it.
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import (
    AutoencoderKLWan,
    Cosmos3OmniPipeline,
    Cosmos3OmniTransformer,
    UniPCMultistepScheduler,
)
from transformers import AutoTokenizer

PROMPT = "A red kite drifts over a sandy beach as waves roll in."
NEGATIVE = "blurry, distorted, low quality"
# 16 fps against the 24 fps base makes the temporal positions fractional.
SMALL = {"width": 96, "height": 64, "frames": 9, "fps": 16.0, "steps": 4, "guidance": 3.0}
REAL = {"width": 256, "height": 160, "frames": 17, "fps": 24.0, "steps": 6, "guidance": 5.0}
Z = 16


def randomise(module, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("gamma") or (p.ndim == 1 and name.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def save(out, name, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    a.astype("<f4").tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


LAYOUTS = {
    "edge": dict(hidden_size=64, intermediate_size=96, num_attention_heads=2, num_key_value_heads=1,
                 rms_norm_eps=1e-5, rope_theta=100000000.0, vocab_size=131072, hidden_act="relu2",
                 qk_norm_for_text=False, use_und_k_norm_for_gen=True),
    "nano": dict(hidden_size=256, intermediate_size=192, num_attention_heads=2, num_key_value_heads=1,
                 rms_norm_eps=1e-6, rope_theta=5000000.0, vocab_size=151936, hidden_act="silu",
                 qk_norm_for_text=True, use_und_k_norm_for_gen=False, sound_gen=True, sound_dim=64),
    "super": dict(hidden_size=160, intermediate_size=320, num_attention_heads=4, num_key_value_heads=1,
                  rms_norm_eps=1e-6, rope_theta=5000000.0, vocab_size=151936, hidden_act="silu",
                  qk_norm_for_text=True, use_und_k_norm_for_gen=False, sound_gen=True, sound_dim=64),
}


def older_layout(ckpt):
    """Rewrite a saved checkpoint into the repository layout of the released
    Nano and Super checkpoints."""
    path = os.path.join(ckpt, "transformer", "config.json")
    with open(path) as f:
        cfg = json.load(f)
    section = cfg.pop("rope_axes_dim", None) or cfg["rope_scaling"]["mrope_section"]
    cfg.pop("use_und_k_norm_for_gen", None)
    cfg.update(
        _diffusers_version="0.37.1", dtype="bfloat16", freeze_und=False, initializer_range=0.02,
        joint_attn_implementation="two_way", max_action_dim=cfg.get("action_dim") or 64,
        max_position_embeddings=262144, model_type="qwen3_vl_text",
        position_embedding_type="unified_3d_mrope", qk_norm=False, qk_norm_for_diffusion=True,
        rope_scaling={"mrope_interleaved": True, "mrope_section": section, "rope_type": "default"},
        temporal_compression_factor_sound=1, use_cache=True, use_moe=True, video_temporal_causal=False,
    )
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    index = {
        "_class_name": "Cosmos3OmniDiffusersPipeline",
        "_diffusers_version": "0.37.1",
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
        "text_tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "transformer": ["diffusers", "Cosmos3OmniTransformer"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "vision_encoder": ["transformers", "Qwen3VLVisionModel"],
        "sound_tokenizer": ["diffusers", "Cosmos3AVAEAudioTokenizer"],
    }
    with open(os.path.join(ckpt, "model_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    for sub in ("scheduler", "vae"):
        p = os.path.join(ckpt, sub, "config.json" if sub == "vae" else "scheduler_config.json")
        with open(p) as f:
            c = json.load(f)
        c["_diffusers_version"] = "0.37.1"
        with open(p, "w") as f:
            json.dump(c, f, indent=2, sort_keys=True)


def build(out, released, layout="edge", **extra):
    """The small checkpoint; `extra` adds transformer settings."""
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    torch.manual_seed(0)
    transformer = Cosmos3OmniTransformer(
        attention_bias=False,
        head_dim=128,
        base_fps=24,
        enable_fps_modulation=True,
        latent_channel=Z,
        latent_patch_size=2,
        num_hidden_layers=2,
        patch_latent_dim=4 * Z,
        rope_scaling={"mrope_section": [24, 20, 20]},
        timestep_scale=0.001,
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        **LAYOUTS[layout],
        **extra,
    )
    randomise(transformer, 1)
    g = torch.Generator().manual_seed(3)
    vae = AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=16,
        z_dim=Z,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        is_residual=True,
        in_channels=12,
        out_channels=12,
        patch_size=2,
        scale_factor_spatial=16,
        scale_factor_temporal=4,
        latents_mean=(0.2 * torch.randn(Z, generator=g)).tolist(),
        latents_std=(0.5 + torch.rand(Z, generator=g)).tolist(),
    )
    randomise(vae, 2)
    pipe = Cosmos3OmniPipeline(
        transformer=transformer,
        text_tokenizer=AutoTokenizer.from_pretrained(os.path.join(released, "text_tokenizer")),
        vae=vae,
        scheduler=UniPCMultistepScheduler.from_pretrained(os.path.join(released, "scheduler")),
        enable_safety_checker=False,
        default_use_system_prompt=layout != "edge",
        use_native_flow_schedule=layout == "edge",
    )
    pipe.save_pretrained(ckpt, safe_serialization=True)
    # Store the checkpoint in bfloat16 like the real one.
    from safetensors.torch import load_file, save_file

    for sub in ("transformer", "vae"):
        d = os.path.join(ckpt, sub)
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                p = os.path.join(d, f)
                sd = load_file(p)
                save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}, p)
    if layout != "edge":
        older_layout(ckpt)
    return pipe


def first_frame(w, h):
    """A deterministic test image: gradients and a bright disc."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 255 * x / max(w - 1, 1)
    gch = 255 * y / max(h - 1, 1)
    disc = ((x - 0.6 * w) ** 2 + (y - 0.4 * h) ** 2) < (0.2 * min(w, h)) ** 2
    b = np.where(disc, 230.0, 40.0 + 60.0 * np.sin(x / 5.0) ** 2)
    return np.stack([r, gch, b], axis=-1).round().clip(0, 255).astype(np.uint8)


def frames_of(pipe, latents):
    """Decoded frames `[F][3][H][W]` in [-1, 1] from normalised latents."""
    inv = pipe._vae_latents_inv_std.view(1, -1, 1, 1, 1)
    mean = pipe._vae_latents_mean.view(1, -1, 1, 1, 1)
    return pipe.vae.decode(latents / inv + mean).sample[0].permute(1, 0, 2, 3)


def reference(pipe, out, spec):
    w, h, nf, fps = spec["width"], spec["height"], spec["frames"], spec["fps"]
    steps, guidance = spec["steps"], spec["guidance"]
    meta = {"prompt": PROMPT, "negative": NEGATIVE, **spec}
    common = {"num_frames": nf, "height": h, "width": w, "fps": fps}
    with torch.no_grad():
        cond, uncond = pipe.tokenize_prompt(PROMPT, NEGATIVE, **common)
        meta["cond_ids"], meta["uncond_ids"] = list(cond), list(uncond)
        image = first_frame(w, h)
        image.tofile(os.path.join(out, "image.bin"))

        z = pipe.transformer.config.latent_channel
        shape = (1, z, (nf - 1) // 4 + 1, h // 16, w // 16)
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(5))
        meta["noise"] = save(out, "noise", noise[0])
        prep = pipe.prepare_latents(
            image=image, generator=torch.Generator().manual_seed(5), device="cpu", dtype=torch.float32, **common
        )
        latents_i2v = prep[0]
        assert torch.equal(latents_i2v[:, :, 1:], noise[:, :, 1:])
        meta["x0_first"] = save(out, "x0_first", latents_i2v[0, :, 0])

        def run(name, **kw):
            res = pipe(PROMPT, NEGATIVE, output_type="latent", enable_safety_check=False, **common, **kw).video
            meta[name] = save(out, name, res[0])
            return res

        # One unguided step from the highest noise level: the clean prediction
        # x - sigma v, so the transformer is measured on its own.
        run("one_step", image=image, num_inference_steps=1, guidance_scale=1.0, latents=latents_i2v.clone())
        meta["decoded"] = save(out, "decoded", frames_of(pipe, noise))

        lat = run("i2v_latents", image=image, num_inference_steps=steps, guidance_scale=guidance, latents=latents_i2v.clone())
        meta["i2v_frames"] = save(out, "i2v_frames", frames_of(pipe, lat))
        lat = run("t2v_latents", num_inference_steps=steps, guidance_scale=1.0, latents=noise.clone())
        meta["t2v_frames"] = save(out, "t2v_frames", frames_of(pipe, lat))

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    if sys.argv[1] == "--checkpoint":
        ckpt, out = sys.argv[2], sys.argv[3]
        os.makedirs(out, exist_ok=True)
        link = os.path.join(out, "checkpoint")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(ckpt), link)
        pipe = Cosmos3OmniPipeline.from_pretrained(ckpt, torch_dtype=torch.float32, enable_safety_checker=False)
        reference(pipe, out, REAL)
    else:
        args, layout = sys.argv[1:], "edge"
        if args[0] == "--layout":
            layout, args = args[1], args[2:]
        out, released = args[0], args[1]
        os.makedirs(out, exist_ok=True)
        reference(build(out, released, layout), out, SMALL)


if __name__ == "__main__":
    main()
