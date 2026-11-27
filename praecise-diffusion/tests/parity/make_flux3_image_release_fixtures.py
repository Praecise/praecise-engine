"""Fixtures for the FLUX 3 Image Klein release parity test, from BFL's reference code.

Needs the gated release weights and BFL's `flux3` package (github.com/black-forest-labs/flux-3-image-klein,
`pip install -e .` there). Run on a GPU host:

    WEIGHTS_DIR=<dir holding flux-3-image-klein and flux-3-image-klein-base> \
    python make_flux3_image_release_fixtures.py <out dir> [--model distilled|base] [--reference IMAGE ...]

Each case writes, under <out>/<tag>/: meta.json (the final prompt, size, steps, guidance, seed, reference
paths as given, [orig w, h, prepared w, h] per reference), refN_orig.rgb / refN_prepared.rgb (raw 8-bit RGB
before and after the reference's own preparation), out.rgb (the decoded image), noise.bin (the starting latent tokens, rounded to bfloat16, row-major (h*w, 128)),
ctx.bin (the text context of the prompt, (L, 20480)), out.bin (the clean latent tokens), out.png.
The engine test (`flux3::image::parity::flux3_image_release_parity`) feeds the same noise and compares.
The reference's torch CPU randn is not reproduced by the engine, so the noise is always handed over.
No token is read or written here: the weights are on disk already.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from flux3.autoencoder import latent_ids, latent_shape
from flux3.dit import IN_CHANNELS
from flux3.pipeline import (MODELS, Pipeline, default_size, dit_path, pack, prepare_multi_reference,
                            prepare_reference, scaled_ids, to_pil, WEIGHTS)
from flux3.sampling import denoise, denoise_flow_map, timesteps

CASES = {
    "t2i": {"prompt": "A red fox sitting in fresh snow at dawn, soft light, shallow depth of field", "refs": [], "seed": 7},
    "edit": {"prompt": "Turn the scene into a watercolour painting", "refs": "ONE", "seed": 21},
}


def dump(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().float().cpu().numpy().astype("<f4").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", choices=MODELS, default="distilled")
    parser.add_argument("--reference", nargs="*", default=[], help="images for the edit and multi-reference cases")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    shared = WEIGHTS / MODELS[args.model]["dit"]
    pipe = Pipeline({args.model: str(dit_path(args.model))}, str(shared / "text_encoder"), str(shared / "ae.safetensors"))
    steps = args.steps or MODELS[args.model]["steps"]
    guidance = MODELS[args.model]["guidance"]
    out = Path(args.out)

    cases = {k: v for k, v in CASES.items() if v["refs"] != "ONE" or args.reference}
    if len(args.reference) > 1:
        cases["multi"] = {"prompt": "Combine the subjects of the images into one scene", "refs": args.reference, "seed": 3}
    for tag, case in cases.items():
        refs = [args.reference[0]] if case["refs"] == "ONE" else case["refs"]
        images = [Image.open(r) for r in refs]
        if len(images) > 1:
            prepared = [prepare_multi_reference(i) for i in images]
        else:
            prepared = [prepare_reference(i)[0] for i in images]
        width, height = default_size(prepared)
        dit = pipe.dit(args.model)
        cond = pipe.encode_references(prepared)
        h, w = latent_shape(height, width)
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(case["seed"]))
        noise = noise.to(torch.bfloat16).float().to(pipe.device)
        ids = latent_ids(h, w, pipe.device)
        image = (noise, scaled_ids(ids))
        schedule = timesteps(steps, 16.0)
        text, ctx = None, None
        if dit.flow_map:
            text = pipe.text_encoder.encode(case["prompt"])
            ctx = text[0]
            tokens = denoise_flow_map(dit, pack([[text, cond, image]]), schedule)
        else:
            text = [pipe.text_encoder.encode(""), pipe.text_encoder.encode(case["prompt"])]
            ctx = text[1][0]
            tokens = denoise(dit, pack([[text[0], cond, image], [text[1], cond, image]]), schedule, guidance)
        directory = out / tag
        directory.mkdir(parents=True, exist_ok=True)
        dump(directory / "noise.bin", noise)
        dump(directory / "ctx.bin", ctx)
        dump(directory / "out.bin", tokens)
        picture = to_pil(pipe.autoencoder.decode(tokens, ids))
        picture.save(directory / "out.png")
        (directory / "out.rgb").write_bytes(np.asarray(picture).tobytes())
        sizes = []
        for i, (original, prep) in enumerate(zip(images, prepared)):
            original = ImageOps.exif_transpose(original).convert("RGB")
            (directory / f"ref{i}_orig.rgb").write_bytes(np.asarray(original).tobytes())
            (directory / f"ref{i}_prepared.rgb").write_bytes(np.asarray(prep).tobytes())
            sizes.append([original.width, original.height, prep.width, prep.height])
        meta = {"model": args.model, "prompt": case["prompt"], "width": width, "height": height, "steps": steps,
                "guidance": guidance, "seed": case["seed"], "references": refs, "ref_sizes": sizes, "alpha": 16.0}
        (directory / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{tag}: {directory}", file=sys.stderr)


if __name__ == "__main__":
    main()
