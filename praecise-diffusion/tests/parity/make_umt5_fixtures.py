"""Tiny multilingual T5 encoder fixtures: random weights, one prompt encoded
with padding and an attention mask (the real-token states are compared)."""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import UMT5Config, UMT5EncoderModel

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
cfg = dict(vocab_size=97, d_model=32, d_kv=8, d_ff=48, num_heads=4, num_layers=3,
           relative_attention_num_buckets=32, relative_attention_max_distance=128,
           layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu")
m = UMT5EncoderModel(UMT5Config(**cfg, dropout_rate=0.0)).float().eval()
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.1)
sd = {k: v.contiguous() for k, v in m.state_dict().items() if k != "encoder.embed_tokens.weight"}
save_file(sd, f"{out}/model.safetensors")
n, pad = 150, 170
ids = torch.randint(1, 97, (1, n))
full = torch.cat([ids, torch.zeros(1, pad - n, dtype=torch.long)], 1)
mask = torch.cat([torch.ones(1, n), torch.zeros(1, pad - n)], 1).long()
with torch.no_grad():
    h = m(full, attention_mask=mask).last_hidden_state[:, :n]
h.float().contiguous().numpy().tofile(f"{out}/out.bin")
json.dump(dict(config=cfg, ids=ids[0].tolist()), open(f"{out}/meta.json", "w"))
print("ok", h.abs().mean().item())
