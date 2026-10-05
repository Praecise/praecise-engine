"""Build a small random MiniMax-H3 transformer and reference outputs.

The released transformer is far too large to run here, so this checks the
native one against the reference on a random checkpoint with the released
layout at tiny widths: a prompt refiner, per-token modulation chosen by
(timestep, modality), a partial three-axis rotary embedding, and a joint
sequence whose text, video and audio tokens are interleaved out of order.
Weights are rounded to bfloat16 first.

Needs a diffusers build with MiniMaxH3Transformer3DModel.

Usage: python make_minimax_h3_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import MiniMaxH3Transformer3DModel

sys.path.insert(0, os.path.dirname(__file__))
from make_qwen_image21_fixtures import randomise, save  # noqa: E402

TINY = dict(
    num_attention_heads=3, attention_head_dim=32, hidden_size=64, num_layers=2, num_refiner_layers=1, ffn_dim=64,
    in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=20, freq_dim=256, time_embed_hidden_dim=40,
    time_embed_dim=24, rope_freq_dim=2, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)
NT, NV, NA = 5, 12, 4
TIMESTEPS = [730.0, 0.0]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    randomise(model, 1)
    model.save_pretrained(os.path.join(out, "checkpoint", "transformer"), safe_serialization=True)
    g = torch.Generator().manual_seed(2)
    n = NT + NV + NA
    order = torch.randperm(n, generator=g)
    text_idx, video_idx, audio_idx = order[:NT].sort().values, order[NT:NT + NV], order[NT + NV:]
    tags = torch.zeros(n, dtype=torch.long)
    tags[video_idx] = 1
    tags[audio_idx] = 2
    steps = torch.randint(0, len(TIMESTEPS), (n,), generator=g)
    pos = torch.randint(-3, 9, (n, 3), generator=g)
    video = torch.randn(1, NV, TINY["in_channels"] * 4, generator=g)
    audio = torch.randn(1, NA, TINY["audio_in_channels"], generator=g)
    text = torch.randn(1, NT, TINY["text_dim"], generator=g)
    with torch.no_grad():
        v, a = model(
            hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=text,
            timestep=torch.tensor(TIMESTEPS), timestep_indices=steps, token_tags=tags, position_ids=pos,
            video_indices=video_idx, audio_indices=audio_idx, text_indices=text_idx, return_dict=False,
        )
    for name, t in [("video", video), ("audio", audio), ("text", text), ("out_video", v), ("out_audio", a),
                    ("positions", pos.float()), ("timesteps", torch.tensor(TIMESTEPS))]:
        save(out, name, t)
    meta = dict(tags=tags.tolist(), timestep_indices=steps.tolist(), text_indices=text_idx.tolist(),
                video_indices=video_idx.tolist(), audio_indices=audio_idx.tolist())
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
