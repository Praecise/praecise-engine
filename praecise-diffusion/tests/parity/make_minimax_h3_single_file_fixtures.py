"""Rewrite tiny MiniMax-H3 component checkpoints into their released layouts.

No reference model runs here: each converted checkpoint holds the same
weights as the fixture it is made from, so the loader must reproduce that
fixture's reference outputs exactly.

- <video_vae_dir>/single/vae: the single-file video autoencoder
  (encoder.down.N.block.M, downsample, nin_shortcut, decoder.x_embedder,
  to_qkv with each head's query, key and value rows together, ff.w1 with the
  gate half first, ff.w2, latent statistics and a mask token stored beside).
- <audio_vae_dir>/single/audio_vae: the single-file audio autoencoder (weight
  norms folded into plain weights, latent statistics stored beside).
- <qwen3_vl_dir>/flat/text_encoder: the flat-layout GGUF prompt encoder
  (model.layers.N, model.embed_tokens, top-level visual.*, the patch
  embedding folded to four dimensions), holding only the first meta["layer"]
  decoder layers and no final norm.

Usage: python make_minimax_h3_single_file_fixtures.py <video_vae_dir> <audio_vae_dir> <qwen3_vl_dir>
"""

import glob
import json
import os
import shutil
import sys

import numpy as np
from safetensors.numpy import save_file


def load_dir(d):
    from safetensors.torch import load_file as load_torch

    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        out.update({k: v.float().numpy() for k, v in load_torch(f).items()})
    return out


def video_vae(fx):
    src, dst = os.path.join(fx, "checkpoint", "vae"), os.path.join(fx, "single", "vae")
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    heads, hd = cfg["decoder_num_attention_heads"], cfg["decoder_attention_head_dim"]
    sd = load_dir(src)
    out = {}
    for k, v in sd.items():
        if ".attn.to_k." in k or ".attn.to_v." in k:
            continue
        if ".attn.to_q." in k:
            t = k.rsplit(".", 1)[1]
            p = k[: k.index(".attn.to_q.")]
            q, kk, vv = (sd[f"{p}.attn.{n}.{t}"] for n in ("to_q", "to_k", "to_v"))
            rows = []
            for h in range(heads):
                s = slice(h * hd, (h + 1) * hd)
                rows += [q[s], kk[s], vv[s]]
            out[f"{p}.attn.to_qkv.{t}"] = np.concatenate(rows)
            continue
        if ".ff.net.0.proj." in k:
            ff = v.shape[0] // 2
            out[k.replace(".ff.net.0.proj.", ".ff.w1.")] = np.concatenate([v[ff:], v[:ff]])
            continue
        n = (k.replace("decoder.proj_in.", "decoder.x_embedder.").replace(".attn.to_out.0.", ".attn.to_out.")
             .replace(".ff.net.2.", ".ff.w2."))
        if n.startswith("encoder.down_blocks."):
            n = (n.replace("encoder.down_blocks.", "encoder.down.").replace(".resnets.", ".block.")
                 .replace(".downsamplers.0.conv.", ".downsample.conv.").replace(".conv_shortcut.", ".nin_shortcut."))
        out[n] = v
    out["latents_mean"] = np.asarray(cfg["latents_mean"], np.float32)
    out["latents_std"] = np.asarray(cfg["latents_std"], np.float32)
    out["decoder.mask_token"] = np.ones((1, 1, heads * hd), np.float32)
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, os.path.join(dst, "model.safetensors"))
    shutil.copy(os.path.join(src, "config.json"), dst)


def audio_vae(fx):
    src, dst = os.path.join(fx, "checkpoint", "audio_vae"), os.path.join(fx, "single", "audio_vae")
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    sd = load_dir(src)
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            continue
        if k.endswith(".weight_v"):
            stem = k[: -len(".weight_v")]
            g = sd[stem + ".weight_g"]
            norm = np.sqrt((v.astype(np.float64) ** 2).reshape(v.shape[0], -1).sum(1)).reshape(g.shape)
            out[stem + ".weight"] = (g * v / norm).astype(np.float32)
            continue
        out[k] = v
    out["latents_mean"] = np.asarray(cfg["latents_mean"], np.float32)
    out["latents_std"] = np.asarray(cfg["latents_std"], np.float32)
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, os.path.join(dst, "model.safetensors"))
    shutil.copy(os.path.join(src, "config.json"), dst)


def qwen3_vl(fx):
    from gguf import GGUFWriter

    src, dst = os.path.join(fx, "checkpoint", "text_encoder"), os.path.join(fx, "flat", "text_encoder")
    os.makedirs(dst, exist_ok=True)
    keep = json.load(open(os.path.join(fx, "meta.json")))["layer"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    sd = load_dir(src)
    w = GGUFWriter(os.path.join(dst, "model.gguf"), "qwen3vl")
    for k in sorted(sd):
        v = sd[k]
        if k.startswith("model.language_model.layers."):
            if int(k.split(".")[3]) >= keep:
                continue
            n = "model.layers." + k[len("model.language_model.layers."):]
        elif k.startswith("model.language_model.embed_tokens."):
            n = "model.embed_tokens." + k[len("model.language_model.embed_tokens."):]
        elif k.startswith("model.visual."):
            n = "visual." + k[len("model.visual."):]
        else:
            continue
        if n == "visual.patch_embed.proj.weight":
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
        w.add_tensor(n, np.ascontiguousarray(v))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)


def main():
    video_vae(sys.argv[1])
    audio_vae(sys.argv[2])
    qwen3_vl(sys.argv[3])


if __name__ == "__main__":
    main()
