"""Build a small random Gemma 3 text encoder and its reference hidden states.

The released prompt encoder is a 12B Gemma 3, so this suite checks the
native encoder against the reference one on a random checkpoint with the
released layout at tiny widths: sliding and full attention layers (the
window shorter than the prompt), linearly stretched full-layer positions,
grouped keys and a query scale that differs from the head width. The
checkpoint is written under the released multimodal names with the released
configuration keys. Prompts are left-padded as the pipeline pads them, and
only the prompt positions are kept. Weights are rounded to bfloat16 first.

Usage: python make_gemma3_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

TEXT = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=7,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    query_pre_attn_scalar=24,
    rms_norm_eps=1e-6,
    sliding_window=4,
    sliding_window_pattern=6,
    rope_theta=1000000.0,
    rope_local_base_freq=10000.0,
    rope_scaling={"rope_type": "linear", "factor": 8.0},
    hidden_activation="gelu_pytorch_tanh",
    attn_logit_softcapping=None,
    final_logit_softcapping=None,
    use_bidirectional_attention=False,
)
SEQ = 12
CASES = [("padded", 9), ("full", 12)]


def main():
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "checkpoint", "text_encoder"), exist_ok=True)
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(**TEXT, pad_token_id=0)
    cfg._attn_implementation = "eager"
    model = Gemma3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        with torch.no_grad():
            if "norm" in name:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    sd = {k.replace("model.", "language_model.model.", 1): v.contiguous() for k, v in model.state_dict().items() if k.startswith("model.")}
    save_file(sd, os.path.join(out, "checkpoint", "text_encoder", "model.safetensors"))
    text = dict(TEXT, model_type="gemma3_text", layer_types=list(cfg.layer_types))
    assert "full_attention" in text["layer_types"] and "sliding_attention" in text["layer_types"]
    json.dump({"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3", "text_config": text},
              open(os.path.join(out, "checkpoint", "text_encoder", "config.json"), "w"), indent=1)

    g = torch.Generator().manual_seed(2)
    cases = []
    for tag, n in CASES:
        ids = torch.zeros(1, SEQ, dtype=torch.int64)
        ids[0, SEQ - n:] = torch.randint(1, TEXT["vocab_size"], (n,), generator=g)
        mask = torch.zeros(1, SEQ, dtype=torch.int64)
        mask[0, SEQ - n:] = 1
        with torch.no_grad():
            hs = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        assert len(hs) == TEXT["num_hidden_layers"] + 1
        st = torch.stack(hs, dim=-1)[0, SEQ - n:]
        st.to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, f"out_{tag}.bin"))
        cases.append({"tag": tag, "start": SEQ - n, "tokens": ids[0, SEQ - n:].tolist()})
    json.dump({"cases": cases}, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
