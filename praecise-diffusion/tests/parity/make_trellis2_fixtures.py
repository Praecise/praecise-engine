"""Fixtures for the TRELLIS.2 parity tests (praecise-diffusion src/trellis2/parity.rs).

Builds small random-weight instances of every TRELLIS.2 network with the
reference implementation's own modules, runs them on CPU in float32 and
writes inputs, weights and outputs:

    python make_trellis2_fixtures.py <TRELLIS.2 source checkout> <out dir>

The reference's CUDA-only kernels are replaced by plain PyTorch equivalents:
submanifold sparse convolution by an explicit neighbour gather (checked here
against a dense 3D convolution), sparse attention by scaled dot-product
attention over the single batch, and the dual-grid hash map by a dictionary.
"""

import ast
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

SRC = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
os.environ["ATTN_BACKEND"] = "sdpa"
sys.path.insert(0, SRC)
torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float32)

# ---- stand-ins for the CUDA extensions -------------------------------------

def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

_stub("cumesh")
_stub("flex_gemm")
_stub("flex_gemm.ops")
_stub("flex_gemm.ops.grid_sample", grid_sample_3d=None)
_stub("flex_gemm.ops.spconv", sparse_submanifold_conv3d=None)

_HASH = {}

def _insert(keys, vals, coords, *grid):
    _HASH[id(keys)] = {tuple(c[1:].tolist()): i for i, c in enumerate(coords)}

def _lookup(keys, vals, q, *grid):
    table = _HASH[id(keys)]
    return torch.tensor([table.get(tuple(c[1:].tolist()), 0xFFFFFFFF) for c in q], dtype=torch.int64)

ov = _stub("o_voxel")
ov.__path__ = [os.path.join(SRC, "o-voxel", "o_voxel")]
ov._C = _stub("o_voxel._C", hashmap_insert_3d_idx_as_val_cuda=_insert, hashmap_lookup_3d_cuda=_lookup)
conv_pkg = _stub("o_voxel.convert")
spec = importlib.util.spec_from_file_location("o_voxel.convert.flexible_dual_grid", os.path.join(SRC, "o-voxel", "o_voxel", "convert", "flexible_dual_grid.py"))
fdg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fdg
spec.loader.exec_module(fdg)
conv_pkg.flexible_dual_grid_to_mesh = fdg.flexible_dual_grid_to_mesh

# The mesh class only needs to hold vertices and faces here.
class _Mesh:
    def __init__(self, vertices, faces):
        self.vertices, self.faces = vertices, faces

rep = _stub("trellis2.representations", Mesh=_Mesh)
rep.__path__ = []

import trellis2.modules.sparse as sp  # noqa: E402
from trellis2.modules.sparse import config as spconfig  # noqa: E402

def _conv_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and padding is None
    self.weight = torch.nn.Parameter(torch.randn(out_channels, 3, 3, 3, in_channels) * (1.0 / np.sqrt(27 * in_channels)))
    self.bias = torch.nn.Parameter(torch.randn(out_channels) * 0.1) if bias else None
    self.dilation = (1, 1, 1)

def neighbour_gather_conv(feats, coords, weight, bias):
    """out[i] = sum over offsets d of W[d] @ feats[voxel at coords[i] + d - 1]."""
    index = {tuple(c): i for i, c in enumerate(coords.tolist())}
    n, ci = feats.shape
    padded = torch.cat([feats, torch.zeros(1, ci)], 0)
    ids = torch.empty(n, 27, dtype=torch.long)
    for i, c in enumerate(coords.tolist()):
        k = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    ids[i, k] = index.get((c[0] + dx, c[1] + dy, c[2] + dz), n)
                    k += 1
    cols = padded[ids].reshape(n, 27 * ci)
    out = cols @ weight.reshape(weight.shape[0], 27 * ci).t()
    return out + bias if bias is not None else out

def _conv_forward(self, x):
    return x.replace(neighbour_gather_conv(x.feats, x.coords[:, 1:], self.weight, self.bias))

_stub("trellis2.modules.sparse.conv.conv_cpuref", sparse_conv3d_init=_conv_init, sparse_conv3d_forward=_conv_forward)
spconfig.CONV = "cpuref"

def _sparse_sdpa(*args):
    if len(args) == 1:
        qkv = args[0]
        qf, kf, vf = qkv.feats.unbind(dim=1)
        o = F.scaled_dot_product_attention(qf.permute(1, 0, 2)[None], kf.permute(1, 0, 2)[None], vf.permute(1, 0, 2)[None])[0].permute(1, 0, 2)
        return qkv.replace(o)
    if len(args) == 3:
        q, k, v = args
    else:
        q, kv = args
        k, v = (kv.unbind(dim=2) if not isinstance(kv, sp.VarLenTensor) else kv.feats.unbind(dim=1))
    qf = q.feats if isinstance(q, sp.VarLenTensor) else q[0]
    kf = k.feats if isinstance(k, sp.VarLenTensor) else (k[0] if k.dim() == 4 else k)
    vf = v.feats if isinstance(v, sp.VarLenTensor) else (v[0] if v.dim() == 4 else v)
    o = F.scaled_dot_product_attention(qf.permute(1, 0, 2)[None], kf.permute(1, 0, 2)[None], vf.permute(1, 0, 2)[None])[0].permute(1, 0, 2)
    return q.replace(o) if isinstance(q, sp.VarLenTensor) else o[None]

import trellis2.modules.sparse.attention.modules as spattn  # noqa: E402
spattn.sparse_scaled_dot_product_attention = _sparse_sdpa

from trellis2.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis2.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis2.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

meta = {}

def put(name, t):
    a = t.detach().cpu().numpy()
    a.astype(np.int32 if a.dtype.kind in "iub" else np.float32).tofile(os.path.join(OUT, name + ".bin"))

def randomize(model, scale=0.05):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("gamma") or ("norm" in n and n.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif "lambda1" in n:
                p.copy_(0.5 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(scale * torch.randn_like(p))

def save_model(model, name, cls, args):
    sd = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    if cls in ("SparseUnetVaeDecoder", "FlexiDualGridVaeDecoder"):
        # The released decoders store their kernels in half precision.
        sd = {k: (v.half() if k.endswith("weight") and v.dim() == 5 else v) for k, v in sd.items()}
    save_file(sd, os.path.join(OUT, name + ".safetensors"))
    json.dump({"name": cls, "args": args}, open(os.path.join(OUT, name + ".json"), "w"))

# ---- sparse convolution reference check ------------------------------------

w = torch.randn(5, 3, 3, 3, 4)
b = torch.randn(5)
x = torch.randn(1, 4, 6, 6, 6)
dense = F.conv3d(x, w.permute(0, 4, 1, 2, 3), b, padding=1)
grid = torch.stack(torch.meshgrid(*[torch.arange(6)] * 3, indexing="ij"), -1).reshape(-1, 3)
gathered = neighbour_gather_conv(x[0].reshape(4, -1).t(), grid, w, b)
assert torch.allclose(gathered, dense[0].reshape(5, -1).t(), atol=1e-4), "gather convolution disagrees with conv3d"

# ---- flow transformer, dense tokens and sampler ------------------------------

flow_args = dict(resolution=4, in_channels=4, out_channels=4, model_channels=128, cond_channels=32, num_blocks=2,
                 num_heads=2, mlp_ratio=2.5, pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)
ss = SparseStructureFlowModel(**flow_args).eval()
randomize(ss)
save_model(ss, "ss_flow", "SparseStructureFlowModel", flow_args)
xs = torch.randn(1, 4, 4, 4, 4)
cond = torch.randn(1, 7, 32)
with torch.no_grad():
    ys = ss(xs, torch.tensor([637.5]), cond)
put("ss_x", xs)
put("ss_cond", cond)
put("ss_out", ys)

sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
sparams = dict(steps=4, guidance_strength=3.0, guidance_rescale=0.7, guidance_interval=[0.3, 1.0], rescale_t=3.0)
noise = torch.randn(1, 4, 4, 4, 4)
with torch.no_grad():
    smp = sampler.sample(ss, noise, cond=cond, neg_cond=torch.zeros_like(cond), verbose=False, **sparams).samples
put("ss_noise", noise)
put("ss_sample", smp)
meta["sampler"] = sparams

# ---- flow transformer, sparse tokens ------------------------------------------

slat_args = dict(resolution=8, in_channels=6, out_channels=4, model_channels=128, cond_channels=32, num_blocks=2,
                 num_heads=2, mlp_ratio=2.5, pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)
sl = SLatFlowModel(**slat_args).eval()
randomize(sl)
save_model(sl, "slat_flow", "SLatFlowModel", slat_args)
cells = torch.randperm(512)[:40].sort().values
coords = torch.stack([cells // 64, (cells // 8) % 8, cells % 8], -1).int()
xf = torch.randn(40, 6)
with torch.no_grad():
    st = sp.SparseTensor(feats=xf, coords=torch.cat([torch.zeros(40, 1, dtype=torch.int32), coords], 1))
    yl = sl(st, torch.tensor([250.0]), cond)
put("slat_coords", coords)
put("slat_x", xf)
put("slat_out", yl.feats)
noise_s = torch.randn(40, 4)
sp_params = dict(steps=3, guidance_strength=7.5, guidance_rescale=0.5, guidance_interval=[0.6, 1.0], rescale_t=3.0)
sl4 = SLatFlowModel(**{**slat_args, "in_channels": 4}).eval()
randomize(sl4)
save_model(sl4, "slat_flow4", "SLatFlowModel", {**slat_args, "in_channels": 4})
with torch.no_grad():
    st = sp.SparseTensor(feats=noise_s, coords=torch.cat([torch.zeros(40, 1, dtype=torch.int32), coords], 1))
    smp_s = sampler.sample(sl4, st, cond=cond, neg_cond=torch.zeros_like(cond), verbose=False, **sp_params).samples
put("slat_noise", noise_s)
put("slat_sample", smp_s.feats)
meta["sparse_sampler"] = sp_params

# ---- occupancy decoder ---------------------------------------------------------

occ_args = dict(out_channels=1, latent_channels=4, num_res_blocks=1, channels=[16, 8], num_res_blocks_middle=1)
occ = SparseStructureDecoder(**occ_args).eval()
randomize(occ, 0.2)
save_model(occ, "occupancy", "SparseStructureDecoder", occ_args)
zl = torch.randn(1, 4, 4, 4, 4)
with torch.no_grad():
    lo = occ(zl)
put("occ_latent", zl)
put("occ_logits", lo)

# ---- structured-latent decoders and the dual-grid surface ------------------------

dec_common = dict(model_channels=[32, 16, 8], latent_channels=8, num_blocks=[1, 2, 0],
                  block_type=["SparseConvNeXtBlock3d"] * 3, up_block_type=["SparseResBlockC2S3d"] * 2, block_args=[{}, {}, {}])
shape_args = dict(resolution=32, **dec_common)
sdec = FlexiDualGridVaeDecoder(**shape_args).eval()
randomize(sdec, 0.2)
save_model(sdec, "shape_dec", "FlexiDualGridVaeDecoder", shape_args)
tex_args = dict(out_channels=6, pred_subdiv=False, **dec_common)
tdec = SparseUnetVaeDecoder(**tex_args).eval()
randomize(tdec, 0.2)
save_model(tdec, "tex_dec", "SparseUnetVaeDecoder", tex_args)
dcells = torch.randperm(512)[:30].sort().values
dcoords = torch.stack([dcells // 64, (dcells // 8) % 8, dcells % 8], -1).int()
zs = torch.randn(30, 8)
zt = torch.randn(30, 8)
with torch.no_grad():
    s_in = sp.SparseTensor(feats=zs, coords=torch.cat([torch.zeros(30, 1, dtype=torch.int32), dcoords], 1))
    h, subs = SparseUnetVaeDecoder.forward(sdec, s_in, return_subs=True)
    meshes, subs2 = sdec(sp.SparseTensor(feats=zs, coords=s_in.coords), return_subs=True)
    t_in = sp.SparseTensor(feats=zt, coords=s_in.coords)
    tex = tdec(t_in, guide_subs=subs)
put("dec_coords", dcoords)
put("dec_shape_latent", zs)
put("dec_tex_latent", zt)
put("dec_out_coords", h.coords[:, 1:])
put("dec_shape_out", h.feats)
put("dec_tex_out", tex.feats)
for i, s in enumerate(subs):
    put(f"dec_sub{i}", (s.feats > 0).int())
put("mesh_vertices", meshes[0].vertices)
put("mesh_faces", meshes[0].faces.int())
meta["mesh"] = {"vertices": int(meshes[0].vertices.shape[0]), "faces": int(meshes[0].faces.shape[0])}

# ---- image encoder -------------------------------------------------------------

from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402
dcfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, patch_size=4,
                       num_register_tokens=4, image_size=32, rope_theta=100.0)
dino = DINOv3ViTModel(dcfg).eval()
randomize(dino, 0.1)
os.makedirs(os.path.join(OUT, "dinov3"), exist_ok=True)
save_file({k: v.contiguous() for k, v in dino.state_dict().items()}, os.path.join(OUT, "dinov3", "model.safetensors"))
json.dump(dcfg.to_dict(), open(os.path.join(OUT, "dinov3", "config.json"), "w"))
img = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    hs = dino.embeddings(img, bool_masked_pos=None)
    pe = dino.rope_embeddings(img)
    for layer in dino.layer:
        hs = layer(hs, position_embeddings=pe)
    feats = F.layer_norm(hs, hs.shape[-1:])
put("dino_pixels", img)
put("dino_out", feats)

# ---- image preparation -----------------------------------------------------------

from PIL import Image  # noqa: E402
src = open(os.path.join(SRC, "trellis2", "pipelines", "trellis2_image_to_3d.py")).read()
fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "preprocess_image")
ns = {"np": np, "Image": Image}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "preprocess", "exec"), ns)
rgba = np.zeros((200, 300, 4), dtype=np.uint8)
rgba[..., :3] = np.random.randint(0, 256, (200, 300, 3))
yy, xx = np.mgrid[:200, :300]
rgba[..., 3] = np.clip(255 * (1.5 - np.hypot((xx - 170) / 70.0, (yy - 90) / 50.0)), 0, 255).astype(np.uint8)
pre = ns["preprocess_image"](types.SimpleNamespace(low_vram=False), Image.fromarray(rgba, "RGBA"))
pre512 = np.array(pre.resize((512, 512), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
enc_in = ((pre512 - mean) / std).transpose(2, 0, 1)
rgba.tofile(os.path.join(OUT, "image_rgba.bin"))
put("image_crop", torch.from_numpy(np.array(pre).astype(np.int32)))
put("image_encoder_input", torch.from_numpy(np.ascontiguousarray(enc_in)))
meta["image"] = {"width": 300, "height": 200, "crop": list(pre.size)}

json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("fixtures written to", OUT)
