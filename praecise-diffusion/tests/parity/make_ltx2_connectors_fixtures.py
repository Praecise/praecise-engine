"""Build small random LTX-2.3 prompt connectors and reference outputs.

The released connectors take the stacked hidden states of a 12B text encoder,
so this suite checks the native connectors against the reference ones on a
random checkpoint with the released LTX-2.3 layout (per-stream projections,
learned registers, gated attention, split rotary embeddings) at tiny widths.
The same weights are also written as a single file under the release's
original names, with the configuration in the header metadata, so the name
map is checked end to end. Weights are rounded to bfloat16 first.

Usage: python make_ltx2_connectors_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from safetensors.torch import save_file

TINY = dict(
    caption_channels=16,
    text_proj_in_factor=3,
    video_connector_num_attention_heads=2,
    video_connector_attention_head_dim=16,
    video_connector_num_layers=2,
    video_connector_num_learnable_registers=4,
    video_gated_attn=True,
    audio_connector_num_attention_heads=2,
    audio_connector_attention_head_dim=8,
    audio_connector_num_layers=2,
    audio_connector_num_learnable_registers=4,
    audio_gated_attn=True,
    connector_rope_base_seq_len=4096,
    rope_theta=10000.0,
    rope_type="split",
    per_modality_projections=True,
    video_hidden_dim=32,
    audio_hidden_dim=16,
    proj_bias=True,
)
SEQ = 8
VALID = [5, 8]

HEADER = dict(
    caption_channels=16,
    connector_num_attention_heads=2,
    connector_attention_head_dim=16,
    audio_connector_num_attention_heads=2,
    audio_connector_attention_head_dim=8,
    connector_num_layers=2,
    connector_num_learnable_registers=4,
    connector_positional_embedding_max_pos=[4096],
    connector_apply_gated_attention=True,
    positional_embedding_theta=10000.0,
    rope_type="split",
    text_encoder_norm_type="per_token_rms",
    caption_proj_before_connector=True,
)


def original_name(k):
    for s in ("video", "audio"):
        if k.startswith(f"{s}_text_proj_in."):
            return k.replace(f"{s}_text_proj_in.", f"text_embedding_projection.{s}_aggregate_embed.")
        if k.startswith(f"{s}_connector."):
            k = k.replace(f"{s}_connector.", f"model.diffusion_model.{s}_embeddings_connector.")
            return k.replace("transformer_blocks", "transformer_1d_blocks").replace("norm_q", "q_norm").replace("norm_k", "k_norm")
    raise KeyError(k)


def save(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = LTX2TextConnectors(**TINY).eval()
    g = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        with torch.no_grad():
            if p.ndim == 1 and "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    model.save_pretrained(os.path.join(out, "checkpoint", "connectors"), safe_serialization=True)
    sd = {original_name(k): v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out, "single.safetensors"), metadata={"config": json.dumps({"transformer": HEADER})})

    g = torch.Generator().manual_seed(2)
    hidden = 3.0 * torch.randn(1, SEQ, TINY["caption_channels"], TINY["text_proj_in_factor"], generator=g)
    cases = []
    for n in VALID:
        mask = torch.zeros(1, SEQ, dtype=torch.int64)
        mask[:, SEQ - n:] = 1
        with torch.no_grad():
            v, a, m = model(hidden, mask, padding_side="left")
        assert bool(m.all()), "registers leave every slot valid"
        save(out, f"hidden_{n}", hidden[0, SEQ - n:])
        save(out, f"out_video_{n}", v[0])
        save(out, f"out_audio_{n}", a[0])
        cases.append(n)
    json.dump({"seq_len": SEQ, "valid": cases}, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
