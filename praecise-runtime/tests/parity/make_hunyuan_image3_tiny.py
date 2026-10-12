"""Reference for the HunyuanImage 3.0 image path on a tiny random checkpoint.

The full checkpoint (an 80B mixture-of-experts backbone) does not fit a single desktop GPU, so
this builds the checkpoint's own model classes (the `hunyuan.py` family published with the
weights) at a tiny size with seeded random weights, saves it as a checkpoint the converter
reads, and records what the reference computes for one denoising forward and one latent decode.

Writes into OUT:
  checkpoint/config.json, checkpoint/model.safetensors
                  the tiny checkpoint (no tokenizer: the test feeds token ids)
  expected.json   prompt and unconditioned token ids, latent grid, timestep, the flow
                  schedule of the reference scheduler, the reference code revision and the
                  tolerances' scale (largest reference magnitudes)
  latent.f32      the noisy latent, planar [C, h, w]
  timestep.f32    the timestep token's backbone input, [hidden]
  tokens.f32      the image tokens' backbone input, row-major [h * w, hidden]
  hidden.f32      the backbone's last hidden states at the image positions (before the
                  final norm), [2, h * w, hidden]: conditioned, then unconditioned
  velocity.f32    the final layer's output for both sequences, planar [2, C, h, w]
  vae_latent.f32  a latent for the decoder, planar [C, vh, vw], as the sampler leaves it
  pixels.f32      the decoder's last frame for it, planar [3, vh * f, vw * f]

The forward is the model's own first denoising step (`mode="gen_image"`, `first_step=True`):
text tokens attend causally, the timestep token sits right after the prompt, the image tokens
attend to each other in full, and positions are the model's two-axis rotary positions.

usage: python make_hunyuan_image3_tiny.py REFERENCE_CODE_DIR OUT [--revision SHA] [--seed N]
"""

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

PROMPT_LEN = 10
GRID = (3, 4)
VAE_GRID = (2, 3)
TIMESTEP = 700.0
STEPS = 8
SHIFT = 3.0


def load_reference(code_dir):
    """Import the reference code directory as a package (it uses relative imports)."""
    init = os.path.join(code_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location("hy3ref", init, submodule_search_locations=[code_dir])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["hy3ref"] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module("hy3ref.hunyuan"), importlib.import_module("hy3ref.configuration_hunyuan")


def tiny_config(code_dir):
    with open(os.path.join(code_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    layers = 2
    vocab = 1024
    cfg.update(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=32,
        attention_head_dim=16,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=layers,
        num_experts=4,
        moe_topk=[2] * layers,
        moe_intermediate_size=[32] * layers,
        num_shared_expert=[1] * layers,
        max_position_embeddings=1024,
        patch_embed_hidden_dim=32,
        patch_size=1,
        img_proj_type="unet",
        torch_dtype="float32",
        moe_impl="eager",
        _attn_implementation="sdpa",
    )
    # special token ids inside the tiny vocabulary
    for k, i in zip(sorted(k for k in cfg if k.endswith("_id") and isinstance(cfg[k], int)), range(64)):
        cfg[k] = vocab - 1 - i
    cfg["pad_token_id"] = cfg["pad_id"]
    vae = dict(cfg["vae"])
    vae.update(block_out_channels=[32, 32, 64], latent_channels=16, layers_per_block=1, ffactor_spatial=4,
               ffactor_temporal=4, sample_size=64, sample_tsize=16)
    cfg["vae"] = vae
    vit = dict(cfg["vit"])
    vit.update(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=1)
    cfg["vit"] = vit
    cfg["vit_aligner"] = dict(cfg["vit_aligner"], input_dim=32, n_embed=64)
    cfg["vae_downsample_factor"] = [4, 4]
    return cfg


def randomize(model, seed):
    """Seeded weights that keep activations of order one (the reference zero-initializes the
    residual branches and the final layer, which would make the comparison trivial)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if p.dim() >= 2:
                fan_in = p[0].numel()
                std = 1.0 if name.endswith("wte.weight") else fan_in ** -0.5
                p.copy_(torch.randn(p.shape, generator=g) * std)
            elif name.endswith("bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            else:
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)


def write(out, name, a):
    np.ascontiguousarray(a, dtype="<f4").tofile(os.path.join(out, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("out")
    ap.add_argument("--revision", default="")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()

    # the reference pins its MoE layer to a CUDA device; this run is CPU-only
    torch.cuda.set_device = lambda *_a, **_k: None
    # it also wraps blocks in NVTX profiling ranges, which a CPU build of torch refuses
    nvtx = torch.cuda.nvtx
    nvtx.range = lambda *_a, **_k: contextlib.nullcontext()
    nvtx.range_push = nvtx.range_pop = lambda *_a, **_k: None
    torch.manual_seed(a.seed)
    torch.set_grad_enabled(False)
    hy, hycfg = load_reference(a.code)

    cfg_dict = tiny_config(a.code)
    config = hycfg.HunyuanImage3Config(**{k: v for k, v in cfg_dict.items() if k not in ("auto_map",)})
    config._attn_implementation = "sdpa"
    model = hy.HunyuanImage3ForCausalMM(config).float().eval()
    randomize(model, a.seed)

    ckpt = os.path.join(a.out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(ckpt, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(ckpt, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=1)

    # one denoising forward over a conditioned and an unconditioned prompt
    g = torch.Generator().manual_seed(a.seed + 1)
    vocab = cfg_dict["vocab_size"]
    cond = torch.randint(16, vocab - 64, (PROMPT_LEN,), generator=g)
    uncond = cond.clone()
    uncond[1:] = cfg_dict["mask_init_id"]
    h, w = GRID
    n = h * w
    seq = PROMPT_LEN + 1 + n
    lc = cfg_dict["vae"]["latent_channels"]
    latent = torch.randn((lc, h, w), generator=g)

    filler = 0
    rows = []
    for prompt in (cond, uncond):
        rows.append(torch.cat([prompt, torch.full((1 + n,), filler, dtype=torch.long)]))
    input_ids = torch.stack(rows)
    image_mask = torch.zeros((2, seq), dtype=torch.long)
    image_mask[:, PROMPT_LEN + 1:] = 1
    ts_index = torch.full((2, 1), PROMPT_LEN, dtype=torch.long)
    timestep = torch.full((2,), TIMESTEP)
    images = latent.unsqueeze(0).repeat(2, 1, 1, 1)
    position_ids = torch.arange(seq).unsqueeze(0).repeat(2, 1)
    image_slice = slice(PROMPT_LEN + 1, seq)
    cos, sin = hy.build_batch_2d_rope(seq, cfg_dict["attention_head_dim"],
                                      image_infos=[[(image_slice, (h, w))]] * 2,
                                      base=cfg_dict["rope_theta"])
    mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    mask[image_slice, image_slice] = True
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)

    caught = {}
    hooks = [
        model.timestep_emb.register_forward_hook(lambda _m, _i, o: caught.__setitem__("timestep", o)),
        model.patch_embed.register_forward_hook(lambda _m, _i, o: caught.__setitem__("tokens", o[0])),
        model.model.register_forward_hook(lambda _m, _i, o: caught.__setitem__("hidden", o[0])),
    ]
    out = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
        custom_pos_emb=(cos, sin),
        mode="gen_image",
        first_step=True,
        images=images,
        image_mask=image_mask,
        timestep=timestep,
        gen_timestep_scatter_index=ts_index,
        use_cache=False,
        return_dict=True,
    )
    for hk in hooks:
        hk.remove()
    velocity = out.diffusion_prediction.float()
    assert velocity.shape == (2, lc, h, w), velocity.shape
    assert torch.equal(caught["tokens"][0], caught["tokens"][1])
    hidden = caught["hidden"][:, PROMPT_LEN + 1:]

    # the latent decoder on its own latent
    vh, vw = VAE_GRID
    vae_latent = torch.randn((lc, vh, vw), generator=g)
    z = vae_latent / cfg_dict["vae"]["scaling_factor"]
    if cfg_dict["vae"].get("shift_factor"):
        z = z + cfg_dict["vae"]["shift_factor"]
    pixels = model.vae.decode(z.unsqueeze(0).unsqueeze(2), return_dict=False)[0]
    pixels = pixels.float()[0, :, 0]
    f = cfg_dict["vae"]["ffactor_spatial"]
    assert pixels.shape == (3, vh * f, vw * f), pixels.shape

    # the reference scheduler's noise levels for the engine's schedule
    sched = hy.FlowMatchDiscreteScheduler(shift=SHIFT, reverse=True, solver="euler")
    sched.set_timesteps(STEPS)
    sigmas = sched.sigmas.tolist()

    write(a.out, "latent.f32", latent.numpy())
    write(a.out, "timestep.f32", caught["timestep"][0].numpy())
    write(a.out, "tokens.f32", caught["tokens"][0].numpy())
    write(a.out, "hidden.f32", hidden.numpy())
    write(a.out, "velocity.f32", velocity.numpy())
    write(a.out, "vae_latent.f32", vae_latent.numpy())
    write(a.out, "pixels.f32", pixels.numpy())
    with open(os.path.join(a.out, "expected.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "reference": "tencent/HunyuanImage-3.0",
                "revision": a.revision,
                "seed": a.seed,
                "cond": cond.tolist(),
                "uncond": uncond.tolist(),
                "grid": [h, w],
                "timestep": TIMESTEP,
                "hidden_size": cfg_dict["hidden_size"],
                "latent_channels": lc,
                "vae_grid": [vh, vw],
                "scale_factor": f,
                "steps": STEPS,
                "shift": SHIFT,
                "sigmas": sigmas,
                "peak_hidden": float(hidden.abs().max()),
                "peak_velocity": float(velocity.abs().max()),
                "peak_pixels": float(pixels.abs().max()),
            },
            fh,
            indent=1,
        )
    print(f"velocity peak {velocity.abs().max():.4f}, pixels peak {pixels.abs().max():.4f}, "
          f"cond/uncond velocity gap {(velocity[0] - velocity[1]).abs().max():.4f}")


if __name__ == "__main__":
    main()
