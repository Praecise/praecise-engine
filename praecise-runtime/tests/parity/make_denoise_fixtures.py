"""Write tiny random-weight diffusion language models as GGUF files.

One file per architecture the denoising engine serves, small enough to decode
on a CPU in milliseconds: two layers, a byte-level vocabulary with an end
token and a mask token. The weights are random, so the text is noise; what
the fixtures pin is the decode itself, compared token for token with the
reference command-line decoder that ships with the backend.

    python make_denoise_fixtures.py <llama.cpp checkout> <out dir> [--block-causal]
"""

import sys
from pathlib import Path

import numpy as np

LLAMA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(LLAMA / "gguf-py"))
import gguf  # noqa: E402

N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_LAYER = 64, 4, 2, 128, 2
N_EXPERT, N_EXPERT_USED, N_FF_EXP = 4, 2, 32
HEAD = N_EMBD // N_HEAD
SCALE = 0.5


def byte_vocab():
    """GPT-2 byte-level tokens for the printable ASCII bytes, then the special
    tokens: whatever the model writes decodes to plain text."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    by_byte = dict(zip(bs, (chr(c) for c in cs)))
    tokens = [by_byte[b] for b in range(0x20, 0x7F)]
    types = [gguf.TokenType.NORMAL] * len(tokens)
    first_special = len(tokens)
    specials = ["<|endoftext|>", "<|mask|>", "<|pad|>"]
    tokens += specials
    types += [gguf.TokenType.CONTROL] * len(specials)
    # One merge no ASCII text can form: a prompt is one token per byte.
    merges = [f"{by_byte[0xFE]} {by_byte[0xFF]}"]
    return tokens, types, merges, first_special, first_special + 1, first_special + 2


def write(arch: str, moe: bool, qk_norm: bool, qkv_bias: bool, shift: bool | None, seed: int, bailing: bool = False):
    rng = np.random.default_rng(seed)
    tokens, types, merges, eos, mask, pad = byte_vocab()
    n_vocab = len(tokens)
    w = gguf.GGUFWriter(OUT / f"{arch}-tiny.gguf", arch)
    w.add_name(f"{arch} tiny random")
    w.add_context_length(4096)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    if moe:
        w.add_expert_count(N_EXPERT)
        w.add_expert_used_count(N_EXPERT_USED)
        w.add_expert_feed_forward_length(N_FF_EXP)
    if bailing:
        w.add_expert_shared_count(1)
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        w.add_expert_weights_norm(True)
        w.add_expert_weights_scale(1.0)
    if shift is not None:
        w.add_bool("diffusion.shift_logits", shift)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(eos)
    w.add_bos_token_id(eos)
    w.add_pad_token_id(pad)
    w.add_mask_token_id(mask)
    w.add_add_bos_token(False)
    # The bare content, so a chat-formatted prompt is the text itself.
    w.add_chat_template("{% for m in messages %}{{ m['content'] }}{% endfor %}")

    def t(name, *shape):
        # GGUF lists dimensions innermost first; numpy outermost first.
        w.add_tensor(name, (rng.standard_normal(shape[::-1]) * SCALE / np.sqrt(shape[0])).astype(np.float32))

    def ones(name, n):
        w.add_tensor(name, np.ones(n, dtype=np.float32))

    t("token_embd.weight", N_EMBD, n_vocab)
    ones("output_norm.weight", N_EMBD)
    t("output.weight", N_EMBD, n_vocab)
    kv = HEAD * N_HEAD_KV
    for i in range(N_LAYER):
        p = f"blk.{i}."
        ones(p + "attn_norm.weight", N_EMBD)
        if bailing:
            t(p + "attn_qkv.weight", N_EMBD, N_EMBD + 2 * kv)
        else:
            t(p + "attn_q.weight", N_EMBD, N_EMBD)
            t(p + "attn_k.weight", N_EMBD, kv)
            t(p + "attn_v.weight", N_EMBD, kv)
        if qkv_bias:
            t(p + "attn_q.bias", N_EMBD)
            t(p + "attn_k.bias", kv)
            t(p + "attn_v.bias", kv)
        if qk_norm:
            ones(p + "attn_q_norm.weight", HEAD)
            ones(p + "attn_k_norm.weight", HEAD)
        t(p + "attn_output.weight", N_EMBD, N_EMBD)
        ones(p + "ffn_norm.weight", N_EMBD)
        if moe:
            t(p + "ffn_gate_inp.weight", N_EMBD, N_EXPERT)
            t(p + "ffn_gate_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT)
            t(p + "ffn_up_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT)
            t(p + "ffn_down_exps.weight", N_FF_EXP, N_EMBD, N_EXPERT)
            if bailing:
                t(p + "exp_probs_b.bias", N_EXPERT)
                t(p + "ffn_gate_shexp.weight", N_EMBD, N_FF_EXP)
                t(p + "ffn_up_shexp.weight", N_EMBD, N_FF_EXP)
                t(p + "ffn_down_shexp.weight", N_FF_EXP, N_EMBD)
        else:
            t(p + "ffn_gate.weight", N_EMBD, N_FF)
            t(p + "ffn_up.weight", N_EMBD, N_FF)
            t(p + "ffn_down.weight", N_FF, N_EMBD)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


OUT.mkdir(parents=True, exist_ok=True)
write("dream", moe=False, qk_norm=False, qkv_bias=True, shift=True, seed=1)
write("llada", moe=False, qk_norm=False, qkv_bias=False, shift=False, seed=2)
write("llada-moe", moe=True, qk_norm=True, qkv_bias=False, shift=False, seed=3)
write("rnd1", moe=True, qk_norm=True, qkv_bias=False, shift=None, seed=4)
if len(sys.argv) > 3 and sys.argv[3] == "--block-causal":
    # Needs a backend with the llada2 architecture.
    write("llada2", moe=True, qk_norm=True, qkv_bias=False, shift=False, seed=5, bailing=True)
