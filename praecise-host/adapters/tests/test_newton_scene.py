"""Run under the engine's own interpreter on a GPU machine, outside the confined host (a development test of the program
itself): newton env: python -I test_newton_scene.py
"""
import importlib.util, json, os, sys, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location("pj", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "praecise_job.py")); pj = importlib.util.module_from_spec(spec); spec.loader.exec_module(pj)
pj.REQUEST_ID = 0
d = Path(tempfile.mkdtemp(prefix="newton-scene-test-")); out = d / "out"; out.mkdir(); work = d / "work"; work.mkdir()
mjcf = d / "pendulum.xml"
mjcf.write_text("""<mujoco model="pend"><compiler angle="radian"/><worldbody><body name="arm" pos="0 0 2"><joint name="j" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" mass="1"/></body></worldbody></mujoco>""")
r = pj.PROGRAMS["newton.scene"]({"frames": 60, "substeps": 4, "solver": "mujoco", "ground": False}, {"scene": str(mjcf)}, str(out), str(work))
print({k: r[k] for k in ("bodies", "joints", "finite", "min_body_height", "seconds", "stage")})
import numpy as np
q = np.load(out / "trajectory.npz")["body_q"]; print("trajectory", q.shape)
# the pendulum swings: z of the tip moves
print("moved:", float(abs(q[-1] - q[0]).max()) > 1e-3)
r2 = pj.PROGRAMS["newton.scene"]({"frames": 30, "substeps": 4, "solver": "xpbd", "seed": 3}, {"scene": str(mjcf)}, str(d / "out2") if (d / "out2").mkdir() is None else "", str(work))
a = np.load(out / "trajectory.npz")["body_q"]
try:
    bad = d / "scene.txt"; bad.write_text("x")
    pj.PROGRAMS["newton.scene"]({}, {"scene": str(bad)}, str(out), str(work)); print("BAD: wrong suffix accepted")
except ValueError as e: print("refused wrong suffix:", str(e)[:60])
print("files:", sorted(p.name for p in out.iterdir()))
