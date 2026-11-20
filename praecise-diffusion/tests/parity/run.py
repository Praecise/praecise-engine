"""Run the ignored parity tests of one family: build its fixtures with the
family's script, then run its tests with the fixture variable set.

    python3 run.py <family> [<family> ...]     # run from the repository root
    python3 run.py --list

Each family names the fixture script, the variable its tests read, the test
path filter, and the Python packages the script imports; the packages are
installed into the virtual environment in PRAECISE_PARITY_VENV (default
`.parity-venv`) on first use. The fixtures are tiny random models, so a run
needs no released weights. The named gate job for these tests is this script.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# family: (script, variable, test filter, packages, extra pip arguments)
FAMILIES = {
    "umt5": ("make_umt5_fixtures.py", "PRAECISE_UMT5_PARITY", "umt5::parity", ["numpy", "safetensors", "transformers"]),
}


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def venv_python():
    venv = os.environ.get("PRAECISE_PARITY_VENV", ".parity-venv")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        run([sys.executable, "-m", "venv", venv])
        run([py, "-m", "pip", "install", "-q", "--index-url", "https://download.pytorch.org/whl/cpu", "torch"])
    return py


def main(argv):
    if argv == ["--list"] or not argv:
        for name, (script, var, test, _) in sorted(FAMILIES.items()):
            print(f"{name}: {script} -> {var}, tests {test}")
        return 0
    py = venv_python()
    for name in argv:
        script, var, test, packages = FAMILIES[name]
        run([py, "-m", "pip", "install", "-q", *packages])
        with tempfile.TemporaryDirectory() as out:
            run([py, os.path.join(HERE, script), out])
            env = dict(os.environ, **{var: out})
            run(
                ["cargo", "test", "--locked", "-p", "praecise-diffusion", "--lib", "--release", test, "--", "--ignored", "--nocapture"],
                env=env,
            )
    return 0


sys.exit(main(sys.argv[1:]))
