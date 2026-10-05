"""Reference vectors for tests/embed_media_parity.rs.

Runs a vision-language embedder checkpoint through its own pipeline
(scripts/qwen3_vl_embedding.py in the checkpoint, float32 on the CPU) and
writes, for each case, the vector it produces, the user turn as the engine
receives it (media markers where the pipeline put its vision blocks) and the
raw RGB pictures and frames, sized so the pipeline keeps them as they are.

    python embed_media.py <checkpoint dir> <output dir>
"""
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

MARKER = "<__media__>"
CHECKPOINT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(CHECKPOINT, "scripts"))
from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
emb = Qwen3VLEmbedder(CHECKPOINT, torch_dtype=torch.float32)
tok = emb.processor.tokenizer


def picture(w, h, phase):
    """A smooth colour pattern; `phase` moves it."""
    x = np.arange(w)[None, :]
    y = np.arange(h)[:, None]
    r = 128 + 100 * np.sin(x / 23 + phase)
    g = 128 + 100 * np.sin(y / 17 - phase * 0.7)
    b = 128 + 100 * np.sin((x + y) / 31 + phase * 1.3)
    return Image.fromarray(np.stack([r + 0 * y, g + 0 * x, b], -1).round().clip(0, 255).astype(np.uint8), "RGB")


def save(name, img, video_frame):
    f = f"{name}.rgb"
    with open(os.path.join(OUT, f), "wb") as fh:
        fh.write(img.tobytes())
    return {"file": f, "width": img.width, "height": img.height, "video_frame": video_frame}


IMAGE = picture(256, 192, 0.4)
FRAMES = [picture(416, 320, i * 0.25) for i in range(64)]

CASES = [
    ("text", {"text": "A red kite over a beach."}),
    ("text_instruction", {"text": "kites at sunset", "instruction": "Retrieve images that match the caption"}),
    ("empty", {}),
    ("image", {"image": IMAGE}),
    ("image_text", {"image": IMAGE, "text": "A colourful pattern."}),
    ("video", {"video": FRAMES}),
    ("video_image_text", {"video": FRAMES, "image": IMAGE, "text": "Moving colour bands, then a still."}),
]

BLOCK = re.compile(r"<\|vision_start\|>(<\|image_pad\|>|<\|video_pad\|>)+<\|vision_end\|>")
TURN = re.compile(r"<\|im_start\|>user\n(.*)<\|im_end\|>\n<\|im_start\|>assistant\n$", re.S)

expected = {"cases": []}
for name, inp in CASES:
    conv = emb.format_model_input(text=inp.get("text"), image=inp.get("image"), video=inp.get("video"), instruction=inp.get("instruction"))
    proc = emb._preprocess_inputs([conv])
    ids = proc["input_ids"][0][proc["attention_mask"][0].bool()]
    prompt = tok.decode(ids, skip_special_tokens=False)
    # The pipeline must not have resized anything: the engine is handed the
    # same pixels.
    if "image_grid_thw" in proc:
        _, gh, gw = proc["image_grid_thw"][0].tolist()
        assert (gh * 16, gw * 16) == (IMAGE.height, IMAGE.width), (name, gh, gw)
    pictures = []
    slices = 0
    if "video_grid_thw" in proc:
        gt, gh, gw = proc["video_grid_thw"][0].tolist()
        assert (gt * 2, gh * 16, gw * 16) == (len(FRAMES), FRAMES[0].height, FRAMES[0].width), (name, gt, gh, gw)
        slices = gt
        pictures += [save(f"frame{i}", f, True) for i, f in enumerate(FRAMES)]
    if inp.get("image") is not None:
        pictures.append(save("image", IMAGE, False))

    # Video blocks hold one slice of two frames: two markers each.
    def marker(m):
        return MARKER * (2 if "video_pad" in m.group(0) else 1)

    turn = TURN.search(prompt)
    assert turn, prompt[-200:]
    user = BLOCK.sub(marker, turn.group(1))
    assert user.count(MARKER) == len(pictures), (name, user.count(MARKER), len(pictures))
    system = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.S).group(1)
    with torch.no_grad():
        out = emb.forward({k: v for k, v in proc.items()})
        vec = emb._pooling_last(out["last_hidden_state"], out["attention_mask"])
        vec = torch.nn.functional.normalize(vec, p=2, dim=-1)[0]
    expected["cases"].append(
        {
            "name": name,
            "instruction": system,
            "text": user,
            "pictures": pictures,
            "tokens": int(ids.shape[0]),
            "slices": slices,
            "embedding": [float(v) for v in vec],
        }
    )
    print(name, int(ids.shape[0]), "tokens", flush=True)

with open(os.path.join(OUT, "expected.json"), "w") as fh:
    json.dump(expected, fh)
