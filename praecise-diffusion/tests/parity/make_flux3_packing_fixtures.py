"""Reference canvases and time ids for the FLUX 3 packing parity test.

usage: make_flux3_packing_fixtures.py <path/to/flux3/f3/packing.py> <out dir>
"""
import importlib.util
import json
import pathlib
import sys
import types

import torch

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
policy = src.parent.parent
pkg = types.ModuleType("ref")
pkg.__path__ = [str(policy)]
sys.modules["ref"] = pkg
sub = types.ModuleType("ref.f3")
sub.__path__ = [str(policy / "f3")]
sys.modules["ref.f3"] = sub
cfg = types.ModuleType("ref.configuration_flux3")
cfg.Flux3Config = object
sys.modules["ref.configuration_flux3"] = cfg
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
packing = load("ref.f3.packing", src)

out.mkdir(parents=True, exist_ok=True)
g = torch.Generator().manual_seed(7)
cases = [
    ("single", 1, 2, (20, 30), (40, 50)),
    ("side_by_side", 2, 2, (48, 40), (32, 64)),
    ("droid", 3, 1, (36, 64), (56, 72)),
    ("grid", 5, 1, (30, 30), (40, 60)),
]
meta = []
for layout, n, t, hw, canvas in cases:
    cams = torch.randint(0, 256, (n, t, 3, *hw), generator=g, dtype=torch.uint8)
    video = packing.materialize_video(cams, None, "cpu", layout=layout, canvas_hw=canvas)
    cams.numpy().tofile(out / f"{layout}_cams.u8")
    video.float().contiguous().numpy().tofile(out / f"{layout}_canvas.f32")
    meta.append({"layout": layout, "cams": n, "frames": t, "hw": list(hw), "canvas": list(canvas)})
times = [i / 15.0 for i in range(40)] + [(i - 7) / 30.0 for i in range(8)] + [i * 4 / 30.0 for i in range(12)]
ids = positional.times_to_ids(torch.tensor(times, dtype=torch.float32)).tolist()
(out / "meta.json").write_text(json.dumps({"cases": meta, "times": times, "ids": ids}))
print("wrote", len(meta), "cases to", out)
