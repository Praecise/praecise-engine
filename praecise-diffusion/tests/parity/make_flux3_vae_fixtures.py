"""Reference encodes and decodes for the FLUX 3 video autoencoder parity test.

usage: make_flux3_vae_fixtures.py <path/to/flux3/f3/video_vae.py> <out dir>

Needs NATTEN importable (its pure-PyTorch flex backend runs on CPU).
"""
import json
import pathlib
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = src.read_text()
text = text[: text.index("class ViTNormInference(nn.Module)")]
keep = [l for l in text.splitlines() if not l.startswith(("from lerobot", "from ..", "from ."))]
ns = {"__name__": "ref_vae", "_natten_available": lambda: True, "require_package": lambda *a, **k: None}
exec(compile("\n".join(keep), str(src), "exec"), ns)
# Backends differ in speed only; let NATTEN pick one that runs on this host.
ns["_natten_attention_kwargs"] = lambda q, k, v: {}

CONFIGS = {
    "a": dict(z_dim=4, embed_dim=16, patch_size=[1, 4, 4], window_size=[3, 3, 3], alternate_window_size=None,
              enc_depths=[1, 2, 1, 1], dec_depths=[1, 1, 2, 1], num_heads=[2, 4, 8, 16],
              temporal=[False, False, True, True], enc_causal=True, dec_causal=False, qk_norm=True, patch_norm=False),
    "b": dict(z_dim=4, embed_dim=16, patch_size=[1, 4, 4], window_size=[3, 3, 3], alternate_window_size=[5, 3, 3],
              enc_depths=[2, 1, 2, 1], dec_depths=[1, 2, 1, 2], num_heads=[2, 2, 4, 8],
              temporal=[False, False, True, True], enc_causal=True, dec_causal=False, qk_norm=True, patch_norm=True),
}
CASES = {"a": [(9, 96, 128), (1, 96, 96)], "b": [(17, 96, 96), (21, 96, 128)]}

torch.manual_seed(0)
for name, cfg in CONFIGS.items():
    model = ns["ViTNorm"](smooth=True, **cfg).eval()
    with torch.no_grad():
        for p_name, p in model.named_parameters():
            if p_name.endswith("norm.weight") or ".norm1.weight" in p_name or ".norm2.weight" in p_name:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif p.dim() == 1:
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p[0].numel() ** 0.5)
        model.z_normalizer.running_mean.copy_(0.3 * torch.randn(cfg["z_dim"]))
        model.z_normalizer.running_var.copy_(0.5 + torch.rand(cfg["z_dim"]))
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    sd = {"model." + k: v.contiguous() for k, v in model.state_dict().items() if k != "z_normalizer.initialized"}
    save_file(sd, str(d / "vae.safetensors"))
    (d / "config.json").write_text(json.dumps(cfg))
    for i, (t, h, w) in enumerate(CASES[name]):
        x = torch.rand(1, 3, t, h, w) * 2 - 1
        with torch.no_grad():
            z = model.encode(x)
            y = model.decode(z)
        (d / f"video{i}.f32").write_bytes(x.numpy().tobytes())
        (d / f"latent{i}.f32").write_bytes(z.numpy().tobytes())
        (d / f"decoded{i}.f32").write_bytes(y.numpy().tobytes())
        (d / f"shape{i}.json").write_text(json.dumps({"video": list(x.shape), "latent": list(z.shape), "decoded": list(y.shape)}))
        print(name, i, list(x.shape), "->", list(z.shape), "->", list(y.shape))
