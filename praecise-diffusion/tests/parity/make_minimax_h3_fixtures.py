"""Build a small random MiniMax-H3 transformer and reference outputs.

The released transformer is far too large to run here, so this checks the
native one against the reference on a random checkpoint with the released
layout at tiny widths: a prompt refiner, per-token modulation chosen by
(timestep, modality), a partial three-axis rotary embedding, and a joint
sequence whose text, video and audio tokens are interleaved out of order.
Weights are rounded to bfloat16 first.

A second checkpoint in <out_dir>/single is the same kind of model in the
single-file layout as a GGUF (fused qkv, gate-first feed-forward, renamed
projections, 8-bit linears) with a timestep-embedding table in place of the
time embedder; its reference is the diffusers model with the table
interpolation substituted for the time embedder and no SiLU before the
modulation projections, run on the dequantised weights.

Needs a diffusers build with MiniMaxH3Transformer3DModel and gguf.

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
    inp = dict(video=video, audio=audio, text=text, steps=steps, tags=tags, pos=pos, idx=(text_idx, video_idx, audio_idx))
    single_file(os.path.join(out, "single"), meta, inp, "gguf")
    single_file(os.path.join(out, "single_f8"), meta, inp, "f8")


GRID, WIDTH = 1025, 8
CURVE_TIMESTEPS = [0.73, 0.0]


def single_file(out, meta, inp, fmt):
    import numpy as np
    import torch.nn as nn
    from gguf import GGMLQuantizationType as Q
    from gguf import GGUFWriter
    from gguf.quants import dequantize, quantize
    import diffusers.models.transformers.transformer_minimax_h3 as h3

    os.makedirs(os.path.join(out, "checkpoint", "transformer"), exist_ok=True)
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    randomise(model, 5)
    g = torch.Generator().manual_seed(6)
    d = TINY["hidden_size"]
    table = torch.randn(GRID, WIDTH, generator=g)
    with torch.no_grad():
        for b in model.transformer_blocks:
            b.adaln_proj.linear = nn.Linear(WIDTH, 6 * d * 3)
            b.adaln_proj.linear.weight.copy_(0.2 * torch.randn(6 * d * 3, WIDTH, generator=g))
            b.adaln_proj.linear.bias.copy_(0.1 * torch.randn(6 * d * 3, generator=g))
        model.norm_out.linear = nn.Linear(WIDTH, 2 * d)
        model.norm_out.linear.weight.copy_(0.2 * torch.randn(2 * d, WIDTH, generator=g))
        model.norm_out.linear.bias.copy_(0.1 * torch.randn(2 * d, generator=g))
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    inner, ff = TINY["num_attention_heads"] * TINY["attention_head_dim"], TINY["ffn_dim"]
    # Single-file names and fused rows.
    out_sd = {"adaln_t_table": table}
    ren = {"proj_in.": "video_patch_proj.", "audio_proj_in.": "audio_patch_proj.", "context_embedder.": "condition_proj.",
           "proj_out.": "final_layer.video_out.", "audio_proj_out.": "final_layer.audio_out.",
           "norm_out.norm.": "final_layer.norm.", "norm_out.linear.": "final_layer.adaln_proj.linear.",
           "token_refiner.final_norm.": "token_refiner.final_norm."}
    blocks = [(f"transformer_blocks.{i}", f"blocks.{i}") for i in range(TINY["num_layers"])]
    blocks += [(f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}") for i in range(TINY["num_refiner_layers"])]
    for k, v in sd.items():
        if k.startswith("time_embedder."):
            continue
        hit = [p for p in ren if k.startswith(p)]
        if hit:
            out_sd[ren[hit[0]] + k[len(hit[0]):]] = v
            continue
        src = [b for b in blocks if k.startswith(b[0] + ".")][0]
        rest = k[len(src[0]):]
        if rest.startswith(".attn.to_q."):
            out_sd[src[1] + ".attn.qkv_proj.weight"] = torch.cat(
                [sd[src[0] + f".attn.{n}.weight"] for n in ("to_q", "to_k", "to_v")])
            continue
        if rest.startswith((".attn.to_k.", ".attn.to_v.")):
            continue
        if rest == ".ff.net.0.proj.weight":
            out_sd[src[1] + ".mlp.fc1.weight"] = torch.cat([v[ff:], v[:ff]])
            continue
        rest = (rest.replace(".attn.to_out.0.", ".attn.out_proj.").replace(".attn.norm_q.", ".attn.q_norm.")
                .replace(".attn.norm_k.", ".attn.k_norm.").replace(".ff.net.2.", ".mlp.fc2."))
        out_sd[src[1] + rest] = v
    out_sd["rope.inv_freq"] = torch.arange(TINY["rope_freq_dim"], dtype=torch.float32)
    deq = {}
    if fmt == "f8":
        # Scaled float8 linears: e4m3 values, a float32 scale per tensor, an
        # input scale and a non-float per-layer record the loader drops.
        from safetensors.torch import save_file
        st = {}
        for name in sorted(out_sd):
            w = out_sd[name].float()
            if w.ndim == 2 and ("attn." in name or "mlp." in name):
                sc = w.abs().max() / 448.0
                w8 = (w / sc).to(torch.float8_e4m3fn)
                layer = name[: -len(".weight")]
                st[name], st[name + "_scale"] = w8, sc.reshape(()).float()
                st[layer + ".input_scale"] = torch.tensor(1.0)
                st[layer + ".layer_format"] = torch.zeros(27, dtype=torch.uint8)
                deq[name] = w8.float() * sc
            else:
                st[name] = w.contiguous()
                deq[name] = w
        save_file(st, os.path.join(out, "checkpoint", "transformer", "model.safetensors"))
    gw = GGUFWriter(os.path.join(out, "checkpoint", "transformer", "model.gguf"), "minimax_h3") if fmt == "gguf" else None
    for name in sorted(out_sd) if gw else []:
        w = out_sd[name].float().numpy()
        if w.ndim == 2 and w.shape[1] % 32 == 0 and ("attn." in name or "mlp." in name):
            raw = quantize(w, Q.Q8_0)
            gw.add_tensor(name, raw, raw_dtype=Q.Q8_0)
            deq[name] = torch.from_numpy(np.ascontiguousarray(dequantize(raw, Q.Q8_0).reshape(w.shape), dtype=np.float32))
        elif name.endswith("adaln_proj.linear.weight"):
            h = w.astype(np.float16)
            gw.add_tensor(name, h)
            deq[name] = torch.from_numpy(h.astype(np.float32))
        else:
            gw.add_tensor(name, w)
            deq[name] = torch.from_numpy(w)
    if gw:
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
    with open(os.path.join(out, "checkpoint", "transformer", "config.json"), "w") as f:
        json.dump({"_class_name": "MiniMaxH3Transformer3DModel", **TINY}, f)
    # Reference: the dequantised weights back in the diffusers model.
    with torch.no_grad():
        for src, dst in blocks:
            qkv = deq[dst + ".attn.qkv_proj.weight"]
            for j, n in enumerate(("to_q", "to_k", "to_v")):
                sd[src + f".attn.{n}.weight"] = qkv[j * inner:(j + 1) * inner]
            fc1 = deq[dst + ".mlp.fc1.weight"]
            sd[src + ".ff.net.0.proj.weight"] = torch.cat([fc1[ff:], fc1[:ff]])
            sd[src + ".attn.to_out.0.weight"] = deq[dst + ".attn.out_proj.weight"]
            sd[src + ".ff.net.2.weight"] = deq[dst + ".mlp.fc2.weight"]
            if src.startswith("transformer_blocks"):
                sd[src + ".adaln_proj.linear.weight"] = deq[dst + ".adaln_proj.linear.weight"].float()
        sd["norm_out.linear.weight"] = deq["final_layer.adaln_proj.linear.weight"]
        model.load_state_dict(sd)

    class Curve(nn.Module):
        def forward(self, t):
            pos = t.float().clamp(0.0, 1.0) * (GRID - 1)
            i0 = pos.floor().long().clamp(max=GRID - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    model.time_proj = Curve()
    # An exact identity that still has a parameter dtype.
    ident = nn.Linear(WIDTH, WIDTH)
    with torch.no_grad():
        ident.weight.copy_(torch.eye(WIDTH))
        ident.bias.zero_()
    model.time_embedder = ident

    def ada(self, temb):
        temb = self.linear(temb).view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)

    def ada_out(self, hidden_states, temb, timestep_indices):
        shift, scale = self.linear(temb).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)

    type(model.transformer_blocks[0].adaln_proj).forward = ada
    type(model.norm_out).forward = ada_out
    text_idx, video_idx, audio_idx = inp["idx"]
    with torch.no_grad():
        v, a = model(
            hidden_states=inp["video"], audio_hidden_states=inp["audio"], encoder_hidden_states=inp["text"],
            timestep=torch.tensor(CURVE_TIMESTEPS), timestep_indices=inp["steps"], token_tags=inp["tags"],
            position_ids=inp["pos"], video_indices=video_idx, audio_indices=audio_idx, text_indices=text_idx,
            return_dict=False,
        )
    for name, t in [("video", inp["video"]), ("audio", inp["audio"]), ("text", inp["text"]), ("out_video", v),
                    ("out_audio", a), ("positions", inp["pos"].float()), ("timesteps", torch.tensor(CURVE_TIMESTEPS))]:
        print(fmt, name, save(out, name, t))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
