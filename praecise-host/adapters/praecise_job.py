"""Job adapter: runs one fixed program in a confined job host and speaks the
runtime host protocol on stdin and stdout.

Usage: python3 -I praecise_job.py, under the environment's own interpreter.

One request arrives on stdin:
  {"id": 0, "op": "run", "program": "<name>", "params": {...}, "inputs": {name: path}, "out": dir, "work": dir}
While the program runs the adapter writes {"id": 0, "event": {"progress": f, "message": m}} lines, then one reply
{"id": 0, "ok": true, "result": {...}} or {"id": 0, "ok": false, "error": "..."}. Output files go into `out`;
`work` is scratch. Every other line a library prints goes to stderr, so stdout carries only the protocol.

The parameters were checked against the program's declaration before the host started; each program checks its
ranges again. When PRAECISE_GPU_REQUIRED is 1 the machine has a GPU and a program that would run on the CPU refuses
instead. The host has no network: model weights are read from caches passed to it read-only.

Programs:
  sionna.radio_map       Sionna RT radio map of a built-in or uploaded Mitsuba scene
  warp.example           one of NVIDIA Warp's examples, headless, with its USD stage
  newton.example         one of Newton's examples, headless, as a USD stage
  cosmos.generate        NVIDIA Cosmos 3 video generation, guardrail always on
  physicsnemo.darcy_fno  PhysicsNeMo Fourier neural operator trained on Darcy flow
  cudaq.ghz              CUDA-Q GHZ state sampled on a simulator
  cudaq.maxcut           CUDA-Q QAOA MaxCut on an input graph
  boltz.predict          Boltz-2 structure and affinity prediction
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import deque
from pathlib import Path

PROTOCOL = sys.stdout
sys.stdout = sys.stderr
REQUEST_ID = 0
PROGRAMS = {}
SEED = (0, 2**31 - 1)


def send(obj):
    PROTOCOL.write(json.dumps(obj) + "\n")
    PROTOCOL.flush()


def progress(fraction=None, message=None):
    send({"id": REQUEST_ID, "event": {"progress": fraction, "message": message}})


def gpu_required():
    return os.environ.get("PRAECISE_GPU_REQUIRED") == "1"


def program(name):
    def register(fn):
        PROGRAMS[name] = fn
        return fn
    return register


def limit(params, key, default, lo, hi):
    v = int(params.get(key, default))
    if not lo <= v <= hi:
        raise ValueError(f"{key} is between {lo} and {hi}")
    return v


def flimit(params, key, default, lo, hi):
    v = float(params.get(key, default))
    if not lo <= v <= hi:
        raise ValueError(f"{key} is between {lo} and {hi}")
    return v


def write_result(out, result):
    Path(out, "result.json").write_text(json.dumps(result, indent=1))
    return result


def torch_cuda(what):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{what} needs a CUDA GPU and none is usable here")
    return torch


def warp_cuda(what):
    import warp as wp

    wp.init()
    if gpu_required() and not wp.is_cuda_available():
        raise RuntimeError(f"{what} needs a CUDA GPU and Warp sees none")
    return wp


def run_logged(argv, cwd, steps=None, timeout=None, env=None):
    """Runs argv, passing progress on when a line matches `steps` (a regex with groups done and total), and returns
    the last lines of its output. A failure raises with those lines."""
    tail = deque(maxlen=60)
    p = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        for line in p.stdout:
            line = line.rstrip()
            tail.append(line)
            print(line, file=sys.stderr)
            if steps and (m := steps.search(line)):
                done, total = int(m.group(1)), int(m.group(2))
                if total:
                    progress(min(0.98, done / total), f"{done} of {total}")
        code = p.wait(timeout=timeout)
    finally:
        if p.poll() is None:
            p.kill()
    if code:
        raise RuntimeError(f"exited with {code}:\n" + "\n".join(tail))
    return list(tail)


# ---------------------------------------------------------------- Sionna RT

@program("sionna.radio_map")
def radio_map(params, inputs, out, work):
    import numpy as np
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene, scene as builtin
    import mitsuba as mi

    if gpu_required() and not mi.variant().startswith("cuda"):
        raise RuntimeError(f"Sionna RT runs on Mitsuba's {mi.variant()} variant here, not on the GPU")
    if "scene" in inputs:
        dest = Path(work, "scene")
        with zipfile.ZipFile(inputs["scene"]) as z:
            for n in z.namelist():
                if n.startswith("/") or ".." in Path(n).parts:
                    raise ValueError(f"unsafe path in the scene zip: {n}")
            z.extractall(dest)
        found = sorted(dest.rglob("scene.xml"))
        if not found:
            raise ValueError("the scene zip has no scene.xml")
        path, name = str(found[0]), "input"
    else:
        name = params.get("scene", "etoile")
        path = getattr(builtin, name, None) if not name.startswith("_") else None
        if not isinstance(path, str):
            raise ValueError(f"unknown scene {name!r}; Sionna's scenes: "
                             + ", ".join(n for n in dir(builtin) if not n.startswith("_")
                                         and isinstance(getattr(builtin, n), str)))
    progress(0.05, f"loading {name}")
    scene = load_scene(path)
    scene.frequency = flimit(params, "frequency_ghz", 3.5, 0.1, 100.0) * 1e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")

    lo, hi = (np.array(v, dtype=float) for v in (scene.mi_scene.bbox().min, scene.mi_scene.bbox().max))
    txs = params.get("transmitters") or [{"position": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2),
                                                       float(hi[2] + 40.0)], "power_dbm": 44.0}]
    if not isinstance(txs, list) or not 1 <= len(txs) <= 16:
        raise ValueError("transmitters is a list of 1 to 16 {position, power_dbm}")
    for i, t in enumerate(txs):
        pos = t.get("position") if isinstance(t, dict) else None
        if not (isinstance(pos, list) and len(pos) == 3 and all(isinstance(c, (int, float)) for c in pos)):
            raise ValueError(f"transmitter {i}: position is [x, y, z] in metres")
        scene.add(Transmitter(name=f"tx{i}", position=[float(c) for c in pos],
                              power_dbm=float(t.get("power_dbm", 44.0))))

    height = flimit(params, "height_m", 1.5, 0.0, 500.0)
    cell = flimit(params, "cell_m", 2.0, 0.25, 50.0)
    size = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    if size[0] / cell * size[1] / cell > 4e6:
        raise ValueError("the grid is over 4 million cells: raise cell_m")
    progress(0.15, f"tracing {len(txs)} transmitter(s)")
    rm = RadioMapSolver()(scene, center=[float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(lo[2] + height)],
                          orientation=[0.0, 0.0, 0.0], size=size, cell_size=[cell, cell],
                          max_depth=limit(params, "max_depth", 5, 0, 10),
                          samples_per_tx=limit(params, "samples", 10**7, 10**4, 10**8),
                          seed=limit(params, "seed", 1, *SEED))
    progress(0.85, "writing the map")
    gain = np.array(rm.path_gain)
    rss = np.array(rm.rss)
    sinr = np.array(rm.sinr)
    centers = np.array(rm.cell_centers)
    np.savez_compressed(Path(out, "radio_map.npz"), path_gain=gain, rss_w=rss, sinr=sinr, cell_centers=centers)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = gain.max(axis=0)
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(best)
    covered = np.isfinite(db)
    fig, ax = plt.subplots(figsize=(8, 8 * size[1] / max(size[0], 1e-6)))
    im = ax.imshow(np.where(covered, db, np.nan), origin="lower", cmap="viridis",
                   extent=[lo[0], hi[0], lo[1], hi[1]], vmin=np.nanpercentile(db[covered], 2) if covered.any() else None)
    for t in txs:
        ax.plot(t["position"][0], t["position"][1], "r^", ms=8)
    fig.colorbar(im, ax=ax, label="path gain (dB)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.savefig(Path(out, "radio_map.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    freq = scene.frequency
    report = {"scene": name, "frequency_ghz": freq.numpy().item() / 1e9 if hasattr(freq, "numpy") else float(freq) / 1e9,
              "transmitters": len(txs), "cells": int(best.size), "covered": round(float(covered.mean()), 4),
              "cell_m": cell, "height_m": height,
              "median_gain_db": round(float(np.median(db[covered])), 2) if covered.any() else None,
              "notice": "Sionna by NVIDIA, Apache-2.0"}
    return write_result(out, report)


# ---------------------------------------------------------------- Warp and Newton

def warp_examples():
    import warp
    root = Path(warp.__file__).parent / "examples"
    return {f"{f.parent.name}.{f.stem[len('example_'):]}": f for f in sorted(root.glob("*/example_*.py"))}


@program("warp.example")
def warp_example(params, inputs, out, work):
    warp_cuda("warp.example")
    found = warp_examples()
    name = params["example"]
    if name not in found:
        raise ValueError(f"no Warp example {name}; examples are named folder.name, for instance core.sph or fem.navier_stokes")
    src = found[name].read_text(encoding="utf-8", errors="replace")
    module = f"warp.examples.{found[name].parent.name}.{found[name].stem}"
    argv = [sys.executable, "-m", module]
    stage = Path(out, "sim.usd")
    if '"--stage-path"' in src or "'--stage-path'" in src:
        argv += ["--stage-path", str(stage)]
    if '"--num-frames"' in src:
        argv += ["--num-frames", str(limit(params, "frames", 120, 1, 2000))]
    if '"--headless"' in src:
        argv += ["--headless"]
    progress(0.02, f"running {name}")
    tail = run_logged(argv, work, timeout=3600)
    return {"example": name, "stage": stage.name if stage.exists() else None, "log": tail[-15:]}


@program("newton.example")
def newton_example(params, inputs, out, work):
    warp_cuda("newton.example")
    name = params["example"]
    stage = Path(out, "sim.usd")
    argv = [sys.executable, "-m", "newton.examples", name, "--viewer", "usd", "--output-path", str(stage)]
    if "frames" in params:
        argv.append(f"--num-frames={limit(params, 'frames', 100, 1, 2000)}")
    progress(0.02, f"running {name}")
    tail = run_logged(argv, work, timeout=3600)
    if not stage.exists():
        raise RuntimeError(f"the example {name} wrote no stage:\n" + "\n".join(tail[-15:]))
    return {"example": name, "stage": stage.name, "log": tail[-15:]}


# ---------------------------------------------------------------- Cosmos

COSMOS_MODELS = {"edge": "nvidia/Cosmos3-Edge", "nano": "nvidia/Cosmos3-Nano"}


@program("cosmos.generate")
def cosmos_generate(params, inputs, out, work):
    torch = torch_cuda("cosmos.generate")
    from diffusers import Cosmos3OmniPipeline
    from diffusers.utils import export_to_video, load_image

    prompt = params["prompt"].strip()
    if not prompt:
        raise ValueError("prompt is empty")
    repo = COSMOS_MODELS[params.get("model", "edge")]
    fps = limit(params, "fps", 24, 1, 60)
    progress(0.02, "loading the guardrail")
    try:
        from cosmos_guardrail import CosmosSafetyChecker

        guard = CosmosSafetyChecker()
    except Exception as ex:
        raise RuntimeError("this machine has no Cosmos guardrail (nvidia/Cosmos-1.0-Guardrail in its model cache), "
                           "and Cosmos does not run without it") from ex
    progress(0.05, f"loading {repo}")
    # the guardrail is on whatever the checkpoint's own configuration says
    pipe = Cosmos3OmniPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda",
                                               safety_checker=guard, enable_safety_checker=True)
    steps = limit(params, "steps", 35, 1, 100)
    kw = {"prompt": prompt, "num_frames": limit(params, "frames", 121, 1, 241), "num_inference_steps": steps,
          "generator": torch.Generator("cuda").manual_seed(limit(params, "seed", 0, *SEED)), "enable_safety_check": True}
    if "image" in inputs:
        kw["image"] = load_image(inputs["image"])

    def step(_pipe, i, _t, cb):
        progress(0.1 + 0.85 * (i + 1) / steps, f"step {i + 1} of {steps}")
        return cb

    progress(0.1, "generating")
    try:
        result = pipe(**kw, callback_on_step_end=step)
    except TypeError:
        result = pipe(**kw)
    video = result.video if hasattr(result, "video") else result.frames[0]
    if video is None:
        raise RuntimeError("the guardrail stopped this prompt or its video")
    export_to_video(video, str(Path(out, "video.mp4")), fps=fps)
    info = {"model": repo, "frames": len(video), "fps": fps, "prompt": prompt, "image": "image" in inputs,
            "notice": "Built on NVIDIA Cosmos"}
    write_result(out, info)
    del pipe
    torch.cuda.empty_cache()
    return info


# ---------------------------------------------------------------- PhysicsNeMo

@program("physicsnemo.darcy_fno")
def darcy_fno(params, inputs, out, work):
    torch = torch_cuda("physicsnemo.darcy_fno")
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
    from physicsnemo.models.fno import FNO

    steps = limit(params, "steps", 2000, 1, 20000)
    res = limit(params, "resolution", 64, 32, 512)
    batch = limit(params, "batch", 32, 1, 256)
    torch.manual_seed(limit(params, "seed", 0, *SEED))
    dev = torch.device("cuda")
    norm = {"permeability": (1.25, 0.75), "darcy": (4.52e-2, 2.79e-2)}
    train = Darcy2D(resolution=res, batch_size=batch, normaliser=norm)
    model = FNO(in_channels=1, out_channels=1, decoder_layers=1, decoder_layer_size=32, dimension=2, latent_channels=32,
                num_fno_layers=4, num_fno_modes=12, padding=9).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, it, loss = time.time(), 0, None
    while it < steps:
        for b in train:
            pred = model(b["permeability"])
            loss = torch.nn.functional.mse_loss(pred, b["darcy"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % 50 == 0:
                progress(it / steps, f"step {it} of {steps}, loss {loss.item():.3e}")
            if it >= steps:
                break
    # error on fields drawn after training, never seen by the model
    test = Darcy2D(resolution=res, batch_size=batch, normaliser=norm)
    model.eval()
    errs = []
    with torch.no_grad():
        for i, b in enumerate(test):
            pred = model(b["permeability"])
            errs.append((torch.linalg.norm(pred - b["darcy"]) / torch.linalg.norm(b["darcy"])).item())
            if i >= 7:
                break
    torch.save(model.state_dict(), Path(out, "fno_darcy.pt"))
    res_out = {"steps": steps, "resolution": res, "batch": batch, "train_loss_last": float(loss.item()),
               "relative_l2_error_unseen": sum(errs) / len(errs), "unseen_batches": len(errs),
               "seconds": round(time.time() - t0, 1), "parameters": sum(p.numel() for p in model.parameters())}
    return write_result(out, res_out)


# ---------------------------------------------------------------- CUDA-Q

def cudaq_target(params):
    import cudaq
    name = params.get("target", "nvidia")
    cudaq.set_target(name)
    return cudaq, name


@program("cudaq.ghz")
def ghz(params, inputs, out, work):
    cudaq, target = cudaq_target(params)
    n, shots = limit(params, "qubits", 10, 2, 34), limit(params, "shots", 1000, 1, 100000)
    if target == "qpp-cpu" and n > 24:
        raise ValueError("the CPU simulator takes at most 24 qubits")

    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        h(q[0])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
        mz(q)

    progress(0.05, f"{n} qubits on {target}")
    t0 = time.time()
    counts = cudaq.sample(kernel, n, shots_count=shots)
    secs = time.time() - t0
    top = sorted(((k, v) for k, v in counts.items()), key=lambda kv: -kv[1])[:8]
    ok = sum(v for k, v in counts.items() if set(k) in ({"0"}, {"1"}))
    res = {"qubits": n, "shots": shots, "target": target, "seconds": round(secs, 3), "ghz_fraction": ok / shots,
           "top": dict(top), "cudaq": cudaq.__version__}
    return write_result(out, res)


@program("cudaq.maxcut")
def maxcut(params, inputs, out, work):
    import itertools

    import numpy as np

    if "graph" not in inputs:
        raise ValueError('send the input graph: JSON {"nodes": n, "edges": [[i, j], ...]}')
    g = json.loads(Path(inputs["graph"]).read_text())
    n = int(g["nodes"])
    edges = [(int(a), int(b)) for a, b in g["edges"]]
    if not 2 <= n <= 30 or any(not (0 <= a < n and 0 <= b < n and a != b) for a, b in edges) or not edges:
        raise ValueError("a graph has 2 to 30 nodes and edges between distinct nodes")
    cudaq, target = cudaq_target(params)
    layers, shots = limit(params, "layers", 2, 1, 12), limit(params, "shots", 2000, 1, 100000)
    us, vs = [a for a, _ in edges], [b for _, b in edges]

    @cudaq.kernel
    def qaoa(n: int, us: list[int], vs: list[int], layers: int, theta: list[float]):
        q = cudaq.qvector(n)
        h(q)
        for l in range(layers):
            for e in range(len(us)):
                x.ctrl(q[us[e]], q[vs[e]])
                rz(2.0 * theta[l], q[vs[e]])
                x.ctrl(q[us[e]], q[vs[e]])
            for i in range(n):
                rx(2.0 * theta[layers + l], q[i])

    ham = 0
    for a, b in edges:
        ham += 0.5 * cudaq.spin.z(a) * cudaq.spin.z(b)
    rng = np.random.default_rng(limit(params, "seed", 0, *SEED))
    opt = cudaq.optimizers.COBYLA()
    opt.initial_parameters = list(rng.uniform(-np.pi / 4, np.pi / 4, 2 * layers))
    opt.max_iterations = 200
    calls = [0]

    def cost(theta):
        calls[0] += 1
        if calls[0] % 10 == 0:
            progress(min(0.9, calls[0] / 200), f"{calls[0]} evaluations")
        return cudaq.observe(qaoa, ham, n, us, vs, layers, theta).expectation()

    best_e, theta = opt.optimize(2 * layers, cost)
    counts = cudaq.sample(qaoa, n, us, vs, layers, theta, shots_count=shots)

    def cut(bits):
        return sum(1 for a, b in edges if bits[a] != bits[b])

    found = max(counts.items(), key=lambda kv: (cut(kv[0]), kv[1]))[0]
    res = {"nodes": n, "edges": len(edges), "layers": layers, "target": target, "evaluations": calls[0],
           "best_bitstring": found, "best_cut": cut(found), "expectation": best_e}
    if n <= 22:
        res["optimum"] = max(cut("".join(b)) for b in itertools.product("01", repeat=n))
        res["ratio"] = res["best_cut"] / res["optimum"] if res["optimum"] else 1.0
    return write_result(out, res)


# ---------------------------------------------------------------- Boltz-2

def boltz_query(inputs, work):
    import yaml

    text = Path(inputs["query"]).read_text()
    msas = {Path(p).name: p for k, p in inputs.items() if k.startswith("msa_")}
    if text.lstrip().startswith(">"):
        chains, cur = [], None
        for line in text.splitlines():
            if line.startswith(">"):
                cur = {"id": chr(ord("A") + len(chains)), "seq": ""}
                chains.append(cur)
            elif cur is not None:
                cur["seq"] += line.strip()
        doc = {"version": 1, "sequences": [{"protein": {"id": c["id"], "sequence": c["seq"]}} for c in chains]}
    else:
        doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or not doc.get("sequences"):
        raise ValueError("the query has no sequences")
    for entry in doc["sequences"]:
        for kind, body in entry.items():
            if kind not in ("protein", "dna", "rna", "ligand"):
                raise ValueError(f"unknown entity {kind!r}")
            if kind == "protein":
                m = body.get("msa")
                if m in (None, ""):
                    body["msa"] = "empty"
                elif m != "empty":
                    name = Path(str(m)).name
                    if name not in msas:
                        raise ValueError(f"alignment {m!r} is not among the inputs")
                    body["msa"] = str(Path(msas[name]).resolve())
    q = Path(work, "query.yaml")
    q.write_text(yaml.safe_dump(doc, sort_keys=False))
    return q


@program("boltz.predict")
def boltz_predict(params, inputs, out, work):
    if gpu_required():
        torch_cuda("boltz.predict")
    if "query" not in inputs:
        raise ValueError("input `query` (a Boltz YAML or a FASTA) is required")
    query = boltz_query(inputs, work)
    cache = os.environ.get("BOLTZ_CACHE")
    if not cache:
        raise RuntimeError("the engine sets no BOLTZ_CACHE: Boltz-2's weights are read from that folder")
    fmt = params.get("format", "mmcif")
    cmd = [str(Path(sys.executable).parent / "boltz"), "predict", str(query), "--out_dir", str(Path(work, "pred")),
           "--cache", cache, "--model", "boltz2", "--accelerator", "gpu", "--devices", "1",
           "--recycling_steps", str(limit(params, "recycling_steps", 3, 1, 10)),
           "--sampling_steps", str(limit(params, "sampling_steps", 200, 20, 400)),
           "--diffusion_samples", str(limit(params, "samples", 1, 1, 10)),
           "--seed", str(limit(params, "seed", 0, *SEED)), "--output_format", fmt, "--override",
           # one process: forked data-loader workers stall under confinement
           "--num_workers", "0"]
    progress(0.05, "predicting")
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    Path(out, "boltz.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    preds = sorted(Path(work, "pred").rglob("predictions/*"))
    if r.returncode != 0 or not preds:
        tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
        raise RuntimeError("Boltz failed: " + " | ".join(tail))
    report = {"model": "boltz2", "structures": [], "notice": "Boltz-2, MIT licence"}
    for d in preds:
        for f in sorted(d.iterdir()):
            shutil.copy2(f, Path(out, f.name))
            if f.name.startswith("confidence_") and f.suffix == ".json":
                c = json.loads(f.read_text())
                report["structures"].append({k: c.get(k) for k in ("confidence_score", "ptm", "iptm",
                                                                       "complex_plddt")} | {"file": f.name})
            if f.name.startswith("affinity_") and f.suffix == ".json":
                report["affinity"] = json.loads(f.read_text())
    return write_result(out, report)


# ---------------------------------------------------------------- protocol

def main():
    global REQUEST_ID
    line = sys.stdin.readline()
    try:
        req = json.loads(line)
    except ValueError:
        send({"id": 0, "ok": False, "error": "the request is not JSON"})
        return 1
    REQUEST_ID = req.get("id", 0)
    fn = PROGRAMS.get(req.get("program")) if req.get("op") == "run" else None
    if fn is None:
        send({"id": REQUEST_ID, "ok": False, "error": f"no job program {req.get('program')!r}"})
        return 1
    try:
        os.chdir(req["work"])
        result = fn(req.get("params") or {}, req.get("inputs") or {}, req["out"], req["work"])
    except Exception as ex:
        traceback.print_exc(file=sys.stderr)
        send({"id": REQUEST_ID, "ok": False, "error": f"{type(ex).__name__}: {ex}"})
        return 1
    send({"id": REQUEST_ID, "ok": True, "result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
