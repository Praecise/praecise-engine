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

# family: (script, variable, test filter, packages; diffusers from its main branch, which the newer models need)
FAMILIES = {
    "pipeline": ("make_fixtures.py", "PRAECISE_DIFFUSION_PARITY", "pipeline::parity", ["git+https://github.com/huggingface/diffusers", "numpy", "transformers"]),
    "cosmos3_mot": ("make_cosmos3_mot_fixtures.py", "PRAECISE_COSMOS3_MOT_PARITY", "cosmos3::parity", ["git+https://github.com/huggingface/diffusers", "numpy", "safetensors"]),
    "flux3": ("make_flux3_fixtures.py", "PRAECISE_FLUX3_PARITY", "flux3::parity", ["safetensors"]),
    "flux3_image": ("make_flux3_image_fixtures.py", "PRAECISE_FLUX3_IMAGE_PARITY", "flux3::image::parity", ["git+https://github.com/huggingface/diffusers", "numpy", "safetensors"]),
    "flux3_policy": ("make_flux3_policy_fixtures.py", "PRAECISE_FLUX3_POLICY_PARITY", "flux3::policy::parity", ["safetensors"]),
    "flux3_text": ("make_flux3_text_fixtures.py", "PRAECISE_FLUX3_TEXT_PARITY", "flux3::text_encoder", ["transformers"]),
    "flux3_vae": ("make_flux3_vae_fixtures.py", "PRAECISE_FLUX3_VAE_PARITY", "flux3::video_vae", ["safetensors"]),
    "gemma3": ("make_gemma3_fixtures.py", "PRAECISE_GEMMA3_PARITY", "gemma3", ["safetensors", "transformers"]),
    "qwen_image": ("make_qwen_image_fixtures.py", "PRAECISE_QWEN_IMAGE_PARITY", "qwen_image::parity", ["git+https://github.com/huggingface/diffusers"]),
    "qwen_image21": ("make_qwen_image21_fixtures.py", "PRAECISE_QWEN_IMAGE21_PARITY", "qwen_image21::parity", ["git+https://github.com/huggingface/diffusers"]),
    "umt5": ("make_umt5_fixtures.py", "PRAECISE_UMT5_PARITY", "umt5::parity", ["safetensors", "transformers"]),
    "wan_dit": ("make_wan_dit_fixtures.py", "PRAECISE_WAN_DIT_PARITY", "wan_dit::parity", ["git+https://github.com/huggingface/diffusers", "safetensors"]),
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
