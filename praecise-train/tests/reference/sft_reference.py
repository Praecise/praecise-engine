#!/usr/bin/env python3
"""Reference LoRA SFT loss curve for `tests/real_model.rs`, computed by an independent stack.

The base weights are the GGUF's own quantized blocks, dequantized to f32 by gguf-py at every use,
so the reference and the trainer see the same weights; the model graph is the transformers
Qwen3 implementation in f32 on the CPU. The adapter starts from the `init.gguf` the test wrote, the
batch is the test's token ids, and the optimizer, clipping and loss reduction follow the recipe of
the test (AdamW with bias correction, global-norm clip, mean over all target tokens).

Inputs come from a run of the test with `PRAECISE_TRAIN_REFERENCE_DUMP=<dir>` (tokens.txt,
init.gguf, recipe.txt). Output is the fixture the test reads (`model_sha256`, then one line per
step: step, loss and gradient norm before clipping).

usage: sft_reference.py <model.gguf> <dump dir> <out fixture> [--steps N]
Needs torch, transformers and gguf-py (llama.cpp/gguf-py on PYTHONPATH). CPU only.
"""

import argparse
import hashlib
import math
import os

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

torch.set_num_threads(int(os.environ.get("REFERENCE_THREADS", "8")))


def field(reader, key):
    return reader.fields[key].contents()


def tensor_f32(t):
    shape = tuple(int(x) for x in reversed(t.shape))
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type)).reshape(shape).astype(np.float32))


class QuantWeight(torch.autograd.Function):
    """y = x W^T with W dequantized at each use; W carries no gradient."""

    @staticmethod
    def forward(ctx, x, holder):
        ctx.holder = holder
        return x @ holder.weight().T

    @staticmethod
    def backward(ctx, gy):
        return gy @ ctx.holder.weight(), None


class QuantLinear(torch.nn.Module):
    def __init__(self, t, cache):
        super().__init__()
        self.t = t
        self.cached = tensor_f32(t) if cache else None
        self.lora = None

    def weight(self):
        return self.cached if self.cached is not None else tensor_f32(self.t)

    def forward(self, x):
        y = QuantWeight.apply(x, self)
        if self.lora is not None:
            a, b, scale = self.lora
            y = y + scale * ((x @ a.T) @ b.T)
        return y


def build(reader, cache):
    tensors = {t.name: t for t in reader.tensors}
    arch = field(reader, "general.architecture")
    assert arch == "qwen3", f"reference covers qwen3, got {arch}"
    n_embd = int(field(reader, "qwen3.embedding_length"))
    vocab = int(tensors["token_embd.weight"].shape[1])
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=n_embd,
        intermediate_size=int(field(reader, "qwen3.feed_forward_length")),
        num_hidden_layers=int(field(reader, "qwen3.block_count")),
        num_attention_heads=int(field(reader, "qwen3.attention.head_count")),
        num_key_value_heads=int(field(reader, "qwen3.attention.head_count_kv")),
        head_dim=int(field(reader, "qwen3.attention.key_length")),
        rms_norm_eps=float(field(reader, "qwen3.attention.layer_norm_rms_epsilon")),
        rope_parameters={"rope_type": "default", "rope_theta": float(field(reader, "qwen3.rope.freq_base"))},
        max_position_embeddings=4096,
        tie_word_embeddings="output.weight" not in tensors,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    m = model.model

    def plain(name):
        return torch.nn.Parameter(tensor_f32(tensors[name]), requires_grad=False)

    m.embed_tokens.weight = plain("token_embd.weight")
    m.norm.weight = plain("output_norm.weight")
    for i, layer in enumerate(m.layers):
        p = f"blk.{i}."
        layer.input_layernorm.weight = plain(p + "attn_norm.weight")
        layer.post_attention_layernorm.weight = plain(p + "ffn_norm.weight")
        at = layer.self_attn
        at.q_norm.weight = plain(p + "attn_q_norm.weight")
        at.k_norm.weight = plain(p + "attn_k_norm.weight")
        at.q_proj = QuantLinear(tensors[p + "attn_q.weight"], cache)
        at.k_proj = QuantLinear(tensors[p + "attn_k.weight"], cache)
        at.v_proj = QuantLinear(tensors[p + "attn_v.weight"], cache)
        at.o_proj = QuantLinear(tensors[p + "attn_output.weight"], cache)
        mlp = layer.mlp
        mlp.gate_proj = QuantLinear(tensors[p + "ffn_gate.weight"], cache)
        mlp.up_proj = QuantLinear(tensors[p + "ffn_up.weight"], cache)
        mlp.down_proj = QuantLinear(tensors[p + "ffn_down.weight"], cache)
    if cfg.tie_word_embeddings:
        emb = m.embed_tokens.weight
        model.lm_head = torch.nn.Linear(n_embd, vocab, bias=False, device="cpu")
        model.lm_head.weight = emb
    else:
        model.lm_head = QuantLinear(tensors["output.weight"], cache)
    with torch.device("cpu"):
        m.rotary_emb = Qwen3RotaryEmbedding(cfg)
    for name, buf in list(model.named_buffers()):
        assert buf.device.type == "cpu", f"buffer {name} left on meta"
    for name, prm in model.named_parameters():
        assert prm.device.type == "cpu", f"parameter {name} left on meta"
    return model


TARGET_MODULE = {"attn_q.weight": "q_proj", "attn_k.weight": "k_proj", "attn_v.weight": "v_proj", "attn_output.weight": "o_proj"}


def attach_lora(model, init_path):
    reader = GGUFReader(init_path)
    alpha = float(field(reader, "adapter.lora.alpha"))
    tensors = {t.name: t for t in reader.tensors}
    params = []
    for name in sorted(n for n in tensors if n.endswith(".lora_a")):
        base = name[: -len(".lora_a")]
        _, il, suffix = base.split(".", 2)
        a = tensor_f32(tensors[name]).clone().requires_grad_(True)
        b = tensor_f32(tensors[base + ".lora_b"]).clone().requires_grad_(True)
        rank = a.shape[0]
        mod = getattr(model.model.layers[int(il)].self_attn, TARGET_MODULE[suffix])
        assert a.shape[1] == mod.weight().shape[1] and b.shape[0] == mod.weight().shape[0], base
        mod.lora = (a, b, alpha / rank)
        params += [a, b]
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cache", action="store_true", help="keep dequantized weights in memory")
    args = ap.parse_args()

    recipe = dict(line.split(maxsplit=1) for line in open(os.path.join(args.dump, "recipe.txt")).read().splitlines())
    lr, beta1, beta2, eps, wd, clip = (float(recipe[k]) for k in ("lr", "beta1", "beta2", "eps", "weight_decay", "grad_clip"))
    tokens = torch.tensor([[int(x) for x in line.split()] for line in open(os.path.join(args.dump, "tokens.txt")).read().splitlines()])

    model = build(GGUFReader(args.model), args.cache)
    model.eval()
    params = attach_lora(model, os.path.join(args.dump, "init.gguf"))
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]

    losses = []
    norms = []
    for step in range(1, args.steps + 1):
        logits = model(input_ids=tokens).logits
        logp = torch.log_softmax(logits[:, :-1].double(), dim=-1)
        nll = -logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)
        loss = nll.sum() / nll.numel()
        for p in params:
            p.grad = None
        loss.backward()
        norm = math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in params))
        scale = clip / norm if clip > 0 and norm > clip else 1.0
        bc1 = 1.0 / (1.0 - beta1**step)
        bc2 = 1.0 / (1.0 - beta2**step)
        with torch.no_grad():
            for p, mi, vi in zip(params, m, v):
                g = p.grad * scale
                mi.mul_(beta1).add_((1 - beta1) * g)
                vi.mul_(beta2).add_((1 - beta2) * g * g)
                p.mul_(1 - lr * wd).sub_(lr * (mi * bc1) / ((vi * bc2).sqrt() + eps))
        losses.append(float(loss.detach()))
        norms.append(norm)
        print(f"step {step} loss {losses[-1]:.6f} grad_norm {norm:.6f}", flush=True)

    h = hashlib.sha256()
    with open(args.model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    with open(args.out, "w") as f:
        f.write(f"model_sha256 {h.hexdigest()}\n")
        f.write(f"model {os.path.basename(args.model)}\n")
        for i, (x, n) in enumerate(zip(losses, norms)):
            f.write(f"{i} {x:.6f} {n:.6f}\n")


if __name__ == "__main__":
    main()
