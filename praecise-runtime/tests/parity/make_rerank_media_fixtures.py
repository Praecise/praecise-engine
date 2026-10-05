"""Reference scores for tests/rerank_media_parity.rs.

Scores every pair with the Qwen3-VL Reranker 2B checkpoint's own pipeline
(its scripts/qwen3_vl_reranker.py: chat template, processor, model and
yes/no score head) in float32 on the CPU, and writes `expected.json` plus the
raw RGB pictures and video frames it names into rerank_media/.

The media are small synthetic pictures, already sized the way the checkpoint's
processor sizes them (sides multiples of 32, area within its pixel budget), so
the processor runs with do_resize=False on exactly the bytes the engine reads.
Each side's text is the processor's expanded prompt with every picture and
every frame written as the media marker; a video is written as one timestamp
per pair of frames followed by the two frame markers, as the checkpoint
expands it.

    python make_rerank_media_fixtures.py <checkpoint dir>

Pinned with transformers==4.57.6, torch (CPU), qwen-vl-utils, pillow, numpy,
scipy and torchvision.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw
from transformers.video_utils import VideoMetadata

REPO = "Qwen/Qwen3-VL-Reranker-2B"
REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
MARKER = "<__media__>"
OUT = Path(__file__).resolve().parent / "rerank_media"

# The GGUF rerank template: the checkpoint's chat template rendered with the
# default instruction, the query and document slots left open.
TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n<Instruct>: Given a search query, retrieve relevant candidates that answer the "
    "query.<Query>:{query}\n<Document>:{document}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def picture(w, h, draw):
    img = Image.new("RGB", (w, h), (245, 245, 240))
    draw(ImageDraw.Draw(img), w, h)
    return img


def red_circle(d, w, h):
    d.ellipse([w * 0.2, h * 0.15, w * 0.8, h * 0.85], fill=(210, 30, 30))


def blue_square(d, w, h):
    d.rectangle([w * 0.25, h * 0.2, w * 0.75, h * 0.8], fill=(30, 60, 200))


def green_stripes(d, w, h):
    for x in range(0, w, 16):
        d.rectangle([x, 0, x + 7, h], fill=(30, 160, 60))


def yellow_triangle(d, w, h):
    d.polygon([(w * 0.5, h * 0.1), (w * 0.9, h * 0.9), (w * 0.1, h * 0.9)], fill=(230, 200, 20))


def moving_ball(n, w, h, color):
    frames = []
    for i in range(n):
        img = Image.new("RGB", (w, h), (20, 20, 30))
        x = 4 + (w - 28) * i / max(n - 1, 1)
        ImageDraw.Draw(img).ellipse([x, h / 2 - 12, x + 24, h / 2 + 12], fill=color)
        frames.append(img)
    return frames


def side(text=None, image=None, video=None):
    return {"text": text, "image": image, "video": video}


def cases():
    pic_red = picture(160, 128, red_circle)
    pic_blue = picture(160, 128, blue_square)
    pic_green = picture(160, 128, green_stripes)
    tiny = picture(64, 64, yellow_triangle)
    ball = moving_ball(16, 64, 64, (220, 40, 40))
    ball_blue = moving_ball(8, 96, 64, (40, 80, 220))
    return [
        ("pictures", side("a red circle"), [side(image=pic_blue), side(image=pic_red), side(image=pic_green)]),
        ("tiny_picture", side("a yellow triangle"), [side(image=tiny), side("a yellow triangle drawn on paper")]),
        (
            "picture_text",
            side("which shape is this?", image=pic_red),
            [side("a red circle"), side("a blue square", image=pic_blue), side("green vertical stripes")],
        ),
        (
            "video_query",
            side("where does the ball go?", video=ball),
            [side("a red ball rolls from left to right"), side("a blue square stays still")],
        ),
        (
            "video_document",
            side("a blue ball moving to the right"),
            [side(video=ball_blue), side(image=pic_green), side("a blue ball moving to the right", video=ball_blue)],
        ),
        (
            "text",
            side("What is the capital of France?"),
            [side("Paris is the capital and largest city of France."), side("Berlin is the capital of Germany.")],
        ),
    ]


def load_reranker(ckpt):
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", ckpt / "scripts" / "qwen3_vl_reranker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Qwen3VLReranker(str(ckpt), torch_dtype=torch.float32)


def score(rr, query, document):
    """One pair through the checkpoint's template, processor, model and head."""
    pair = rr.format_mm_instruction(
        query["text"], query["image"], query["video"] and "video",
        document["text"], document["image"], document["video"] and "video",
        instruction=rr.default_instruction,
    )
    text = rr.processor.apply_chat_template([pair], tokenize=False, add_generation_prompt=True)
    images = [m for s in (query, document) for m in ([s["image"]] if s["image"] else [])]
    clips = [s["video"] for s in (query, document) if s["video"]]
    videos = [torch.from_numpy(np.stack([np.asarray(f) for f in v])).permute(0, 3, 1, 2).contiguous() for v in clips]
    metadata = [
        VideoMetadata(
            total_num_frames=len(v), fps=2.0, width=v[0].width, height=v[0].height,
            duration=len(v) / 2.0, frames_indices=list(range(len(v))),
        )
        for v in clips
    ]
    inputs = rr.processor(
        text=text,
        images=images or None,
        videos=videos or None,
        video_metadata=metadata or None,
        do_resize=False,
        do_sample_frames=False,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    ids = inputs["input_ids"][0].tolist()
    p_yes = rr.compute_scores(inputs)[0]
    expanded = rr.processor.tokenizer.decode(ids, skip_special_tokens=False)
    return p_yes, len(ids), expanded


def split_sides(expanded):
    """The processor's prompt with every picture and frame as the marker."""
    # A picture is one marker; a video's pair of frames is two, merged over
    # time by the projector.
    text = re.sub(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>", MARKER, expanded)
    text = re.sub(r"<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>", MARKER * 2, text)
    head, rest = TEMPLATE.split("{query}")
    mid, tail = rest.split("{document}")
    assert text.startswith(head) and text.endswith(tail), text[:400]
    body = text[len(head) : len(text) - len(tail)]
    assert body.count(mid) == 1, body
    q, d = body.split(mid)
    assert TEMPLATE.replace("{query}", q).replace("{document}", d) == text
    return q, d


def write_side(s, prefix, text, files):
    pictures = []
    n = 0
    for kind, media in ([("video", s["video"])] if s["video"] else []) + ([("image", s["image"])] if s["image"] else []):
        frames = media if kind == "video" else [media]
        for f in frames:
            name = f"{prefix}_{n}.rgb"
            n += 1
            if name not in files:
                files[name] = f.tobytes()
            pictures.append({"file": name, "width": f.width, "height": f.height, "video_frame": kind == "video"})
    assert text.count(MARKER) == len(pictures), (text, len(pictures))
    return {"text": text, "pictures": pictures}


def main():
    ckpt = Path(sys.argv[1])
    rr = load_reranker(ckpt)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    out_cases = []
    for name, query, documents in cases():
        p, n, q_side, d_sides = [], [], None, []
        for j, document in enumerate(documents):
            p_yes, n_tokens, expanded = score(rr, query, document)
            q, d = split_sides(expanded)
            if q_side is None:
                q_side = write_side(query, f"{name}_q", q, files)
            assert q == q_side["text"], (name, j)
            d_sides.append(write_side(document, f"{name}_d{j}", d, files))
            p.append(p_yes)
            n.append(n_tokens)
            print(f"{name}[{j}]: P(yes) {p_yes:.6f}, {n_tokens} tokens", flush=True)
        out_cases.append({"name": name, "query": q_side, "documents": d_sides, "p_yes": p, "n_tokens": n})
    for f in OUT.glob("*.rgb"):
        f.unlink()
    for name, data in files.items():
        (OUT / name).write_bytes(data)
    meta = {
        "source": {"repo": REPO, "revision": REVISION},
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "cases": out_cases,
    }
    (OUT / "expected.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
