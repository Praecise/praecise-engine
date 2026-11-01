"""Run under the engine's own interpreter on a GPU machine, outside the confined host (a development test of the program
itself): physicsnemo env: python -I test_surrogate.py (SURROGATE_CONTRACT=path to ipnops_worker/surrogate_contract.py checks the card against it)
"""
import importlib.util, json, sys, tempfile, hashlib
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("pj", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "praecise_job.py")); pj = importlib.util.module_from_spec(spec); spec.loader.exec_module(pj)
d = Path(tempfile.mkdtemp(prefix="surrogate-test-")); work = d / "work"; work.mkdir()
rng = np.random.default_rng(0)
# synthetic: y = smooth function of x (a 2D field passed through a fixed linear blur + nonlinearity)
N, R = 400, 32
x = rng.uniform(0.1, 1.0, (N, 1, R, R)).astype("float32")
from scipy.ndimage import uniform_filter
y = np.stack([uniform_filter(x[i, 0], size=5, mode="wrap") ** 2 for i in range(N)])[:, None].astype("float32")
np.savez(d / "ds.npz", x=x, y=y, meta=json.dumps({"id": "blur-test", "x_names": ["field"], "y_names": ["blurred"], "licence": "Apache-2.0"}))
out = d / "fit"; out.mkdir()
r = pj.PROGRAMS["physicsnemo.surrogate_fit"]({"model": "fno", "steps": 600, "batch": 16, "error_limit": 0.2, "seed": 1}, {"dataset": str(d / "ds.npz")}, str(out), str(work))
print("fit:", {k: r[k] for k in ("relative_l2_held_out", "admitted", "seconds", "parameters")})
card = json.loads((out / "model_card.json").read_text())
assert card["checkpoint_sha256"] == hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
# the contract of the ipnops-compute repo
import os
if os.environ.get("SURROGATE_CONTRACT"):
    spec2 = importlib.util.spec_from_file_location("sc", os.environ["SURROGATE_CONTRACT"]); sc = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sc)
    sc.check_card(card); print("card passes surrogate_contract.check_card:", card["id"])
# predict inside the band
xin = d / "xin.npz"; np.savez(xin, x=x[:8])
o = d / "pred"; o.mkdir()
p = pj.PROGRAMS["physicsnemo.surrogate_predict"]({}, {"model": str(out / "model.pt"), "card": str(out / "model_card.json"), "x": str(xin)}, str(o), str(work))
yy = np.load(o / "y.npz")["y"]; print("predict:", yy.shape, "rel err vs truth", float(np.linalg.norm(yy - y[:8]) / np.linalg.norm(y[:8])), p["outputs_flagged"])
# outside the band: refused
xout = d / "xout.npz"; np.savez(xout, x=x[:4] * 5.0)
try:
    pj.PROGRAMS["physicsnemo.surrogate_predict"]({}, {"model": str(out / "model.pt"), "card": str(out / "model_card.json"), "x": str(xout)}, str(o), str(work)); print("BAD: out-of-band accepted")
except ValueError as e: print("refused out-of-band:", str(e)[:80])
# a wrong checkpoint is refused
(d / "other.pt").write_bytes(b"not the model")
try:
    pj.PROGRAMS["physicsnemo.surrogate_predict"]({}, {"model": str(d / "other.pt"), "card": str(out / "model_card.json"), "x": str(xin)}, str(o), str(work)); print("BAD: wrong checkpoint")
except ValueError as e: print("refused checkpoint mismatch:", str(e)[:60])
# not admitted when over the limit
o2 = d / "fit2"; o2.mkdir()
try:
    pj.PROGRAMS["physicsnemo.surrogate_fit"]({"model": "fno", "steps": 5, "error_limit": 0.0001}, {"dataset": str(d / "ds.npz")}, str(o2), str(work)); print("BAD: admitted")
except RuntimeError as e: print("not admitted:", str(e)[:70], "| card written:", (o2 / "model_card.json").exists())
