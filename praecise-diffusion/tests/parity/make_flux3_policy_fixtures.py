"""Reference action chunks for the FLUX 3 policy parity test.

usage: make_flux3_policy_fixtures.py <path/to/policies/flux3> <out dir>

Runs the reference policy's chunk sampler (its own source, with a tiny random
transformer) for a frame-packer and a history-packer configuration, from
recorded noise, conditioning tokens and text contexts.
"""
import importlib.util
import json
import pathlib
import sys
import textwrap
import types

import torch
from safetensors.torch import save_file

policy, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
pkg = types.ModuleType("ref")
pkg.__path__ = [str(policy)]
sys.modules["ref"] = pkg
sub = types.ModuleType("ref.f3")
sub.__path__ = [str(policy / "f3")]
sys.modules["ref.f3"] = sub
cfg_mod = types.ModuleType("ref.configuration_flux3")
cfg_mod.Flux3Config = object
sys.modules["ref.configuration_flux3"] = cfg_mod
try:
    import torchvision.transforms.v2.functional  # noqa: F401  (augmentation only)
except ImportError:
    for name in ("torchvision", "torchvision.transforms", "torchvision.transforms.v2", "torchvision.transforms.v2.functional"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["torchvision.transforms.v2"].functional = sys.modules["torchvision.transforms.v2.functional"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


load("ref.utils", policy / "utils.py")
positional = load("ref.f3.positional", policy / "f3" / "positional.py")
packing = load("ref.f3.packing", policy / "f3" / "packing.py")
sampling = load("ref.f3.sampling", policy / "f3" / "sampling.py")
transformer = load("ref.f3.transformer", policy / "f3" / "transformer.py")

src = (policy / "modeling_flux3.py").read_text()
start = src.index("    def _sample(")
end = src.index("    @torch.no_grad()\n    def predict_action_chunk")
ns = {
    "torch": torch, "Tensor": torch.Tensor, "packing": packing, "sampling": sampling, "VEC_DIM": 768,
    "batched_prc_vid": positional.batched_prc_vid, "batched_prc_audio": positional.batched_prc_audio,
    "times_to_ids": positional.times_to_ids,
}
exec(textwrap.dedent(src[start:end]), ns)
reference_sample = ns["_sample"]

LAT = packing.LATENT_CHANNELS
CTX = 32
CONFIGS = {
    "frame": dict(packer="frame", modality="action", d=3, chunk=6, n_obs=1, snapshots=1, past=False, fps=30.0, vfps=24.0,
                  canvas=(64, 96), lat_full=(3, 4), sampler="euler", steps=3, shift=6.93, g=3.0, ga=None, scale=2.0, flip=[2]),
    "history": dict(packer="history", modality="action_prediction_droid", d=4, chunk=8, n_obs=5, snapshots=2, past=True,
                    fps=15.0, vfps=None, canvas=(32, 64), lat_full=(1, 2), sampler="cosmos_unipc", steps=4, shift=5.0,
                    g=4.0, ga=1.0, scale=1.5, flip=[]),
}


def save(path, t):
    path.write_bytes(t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes())


def save_ids(path, t):
    path.write_bytes(t.contiguous().numpy().astype("<i4").tobytes())


torch.manual_seed(0)
for name, c in CONFIGS.items():
    d_out = out / name
    d_out.mkdir(parents=True, exist_ok=True)
    m, d = c["modality"], c["d"]
    cond_ch = 2 * d if c["past"] else d
    params = transformer.JointSingleSeqParams(
        in_channels={"video": LAT, "video_cond": LAT, m: d, f"{m}_cond": cond_ch},
        sequence={"x_video": "video", "x_video_cond": "video_cond", f"x_{m}": m, f"x_{m}_cond": f"{m}_cond"},
        vec_in_dim=768, context_in_dim=CTX, hidden_size=64, num_heads=2, axes_dim=[8, 8, 8, 8], mlp_ratio=3.0,
        attn_mode="torch", depth=1, depth_single_blocks=1, depth_late_blocks=0,
    )
    model = transformer.JointSingleSeq(params).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("_norm.scale"):
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
    save_file({"dit." + k: v.contiguous() for k, v in model.state_dict().items()}, str(d_out / "model.safetensors"))

    lh, lw = -(-c["canvas"][0] // 32), -(-c["canvas"][1] // 32)
    cfg = types.SimpleNamespace(
        conditioning=c["packer"], chunk_size=c["chunk"], n_obs_steps=c["n_obs"], fps=c["fps"],
        video_position_fps=c["vfps"], latent_hw=(lh, lw), guidance_scale=c["g"], guidance_scale_action=c["ga"],
        sampler=c["sampler"], num_inference_steps=c["steps"], sampler_shift=c["shift"], action_scale=c["scale"],
        action_modality=m, history_snapshots=c["snapshots"], action_dim=d,
    )
    cfg.window_frames = c["chunk"] + (c["n_obs"] if c["packer"] == "history" else 1)
    pk = c["packer"]
    packer = types.SimpleNamespace(
        predicted_latent_frames=getattr(packing, f"{pk}_predicted_latent_frames"),
        predicted_video_times=getattr(packing, f"{pk}_predicted_video_times"),
        action_times=getattr(packing, f"{pk}_action_times"),
    )
    vfps = c["vfps"] or c["fps"]

    # Video conditioning from oversized latents, cropped like the packers.
    picks = [0] if pk == "frame" else [round(j * (c["n_obs"] - 1) / (c["snapshots"] - 1)) for j in range(c["snapshots"])]
    lats = [torch.randn(1, LAT, 1, *c["lat_full"]) for _ in picks]
    toks, ids = [], []
    for i, lat in zip(picks, lats):
        secs = 0.0 if pk == "frame" else i / vfps
        tok, pos = positional.batched_prc_vid(lat[..., :lh, :lw], positional.times_to_ids(torch.full((1, 1), secs)))
        toks.append(tok)
        ids.append(pos)
    for k, lat in enumerate(lats):
        save(d_out / f"lat{k}.bin", lat[0])
    cond = {"x_video_cond": torch.cat(toks, 1), "x_video_cond_ids": torch.cat(ids, 1)}

    states = torch.rand(1, c["n_obs"], d)
    past = torch.randn(1, c["n_obs"], d) if c["past"] else None
    if pk == "frame":
        flipped = states[:, 0].clone()
        flipped[..., c["flip"]] = 1.0 - flipped[..., c["flip"]]
        act = packing.pack_actions(flipped[:, None], None, None, m, scale=c["scale"], targets=False)
    else:
        act = packing.pack_history_actions(states, past, None, m, c["fps"], c["scale"], targets=False)
    cond.update(act)
    save(d_out / "states.bin", states[0])
    if past is not None:
        save(d_out / "past.bin", past[0])
    save(d_out / "action_cond.bin", act[f"x_{m}_cond"][0])
    save_ids(d_out / "action_cond_ids.bin", act[f"x_{m}_cond_ids"][0])
    save(d_out / "video_cond.bin", cond["x_video_cond"][0])
    save_ids(d_out / "video_cond_ids.bin", cond["x_video_cond_ids"][0])

    ctx = {"go": torch.randn(1, 11, CTX), "": torch.randn(1, 7, CTX)}
    save(d_out / "ctx_c.bin", ctx["go"][0])
    save(d_out / "ctx_uc.bin", ctx[""][0])
    self = types.SimpleNamespace(
        config=cfg, modality=m, dtype_=torch.float32, packer=packer,
        _context=lambda caption, device: (ctx[caption], torch.tensor([[[0, 0, 0, i] for i in range(ctx[caption].shape[1])]])),
        _inference_dit=lambda: model,
    )
    seed = 3
    rng = torch.Generator().manual_seed(seed)
    n_pred = packer.predicted_latent_frames(cfg)
    save(d_out / "noise_video.bin", torch.randn(1, LAT, n_pred, lh, lw, generator=rng)[0])
    save(d_out / "noise_action.bin", torch.randn(1, d, c["chunk"], generator=rng)[0])
    with torch.no_grad():
        chunk = reference_sample(self, cond, "go", seed)
    save(d_out / "chunk.bin", chunk)
    save_ids(d_out / "video_ids.bin", packer.predicted_video_times(cfg, 1)[0])
    save_ids(d_out / "action_ids.bin", positional.times_to_ids(packer.action_times(cfg, 1))[0])

    policy_json = {
        "packer": pk, "condition_on_past_actions": c["past"], "n_obs_steps": c["n_obs"],
        "history_snapshots": c["snapshots"], "chunk_size": c["chunk"], "fps": c["fps"],
        "video_position_fps": c["vfps"], "canvas_hw": list(c["canvas"]), "camera_layout": "single",
        "action_modality": m, "action_scale": c["scale"], "gripper_flip_dims": c["flip"], "sampler": c["sampler"],
        "num_inference_steps": c["steps"], "guidance_scale": c["g"], "guidance_scale_action": c["ga"],
        "sampler_shift": c["shift"], "text_fixed_length": None,
        "output_features": {"action": {"type": "ACTION", "shape": [d]}},
    }
    (d_out / "config.json").write_text(json.dumps(policy_json))
    (d_out / "meta.json").write_text(json.dumps({"lat_full": c["lat_full"], "picks": picks}))
    print(name, tuple(chunk.shape), float(chunk.abs().max()))
