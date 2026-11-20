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

# family: (test filter, [(fixture script, variable), ...], packages); a module whose tests read several
# variables gets every fixture before its tests run once; diffusers comes from its main branch, which the newer models need
FAMILIES = {
    "cosmos3_mot": ("cosmos3::parity", [("make_cosmos3_mot_fixtures.py", "PRAECISE_COSMOS3_MOT_PARITY")], ["git+https://github.com/huggingface/diffusers", "numpy", "safetensors"]),
    "flux3": ("flux3::parity", [("make_flux3_fixtures.py", "PRAECISE_FLUX3_PARITY")], ["safetensors"]),
    "flux3_image": ("flux3::image::parity", [("make_flux3_image_fixtures.py", "PRAECISE_FLUX3_IMAGE_PARITY")], ["git+https://github.com/huggingface/diffusers", "numpy", "safetensors"]),
    "flux3_policy": ("flux3::policy::parity", [("make_flux3_policy_fixtures.py", "PRAECISE_FLUX3_POLICY_PARITY")], ["safetensors"]),
    "flux3_text": ("flux3::text_encoder", [("make_flux3_text_fixtures.py", "PRAECISE_FLUX3_TEXT_PARITY")], ["transformers"]),
    "flux3_vae": ("flux3::video_vae", [("make_flux3_vae_fixtures.py", "PRAECISE_FLUX3_VAE_PARITY")], ["safetensors"]),
    "gemma3": ("gemma3", [("make_gemma3_fixtures.py", "PRAECISE_GEMMA3_PARITY")], ["safetensors", "transformers"]),
    "ltx2": ("ltx2::parity", [("make_ltx2_fixtures.py", "PRAECISE_LTX2_PARITY"), ("make_ltx2_audio_vae_fixtures.py", "PRAECISE_LTX2_AUDIO_VAE_PARITY"), ("make_ltx2_connectors_fixtures.py", "PRAECISE_LTX2_CONNECTORS_PARITY"), ("make_ltx2_distilled_fixtures.py", "PRAECISE_LTX2_DISTILLED_PARITY"), ("make_ltx2_pipeline_fixtures.py", "PRAECISE_LTX2_PIPELINE_PARITY"), ("make_ltx2_vae_fixtures.py", "PRAECISE_LTX2_VAE_PARITY"), ("make_ltx2_vocoder_fixtures.py", "PRAECISE_LTX2_VOCODER_PARITY")], ["git+https://github.com/huggingface/diffusers", "numpy", "safetensors", "transformers"]),
    "matrix_game": ("matrix_game::parity", [("make_mg3_fixtures.py", "PRAECISE_MG3_PARITY"), ("make_mg3_session_fixtures.py", "PRAECISE_MG3_SESSION")], ["git+https://github.com/huggingface/diffusers", "numpy", "safetensors"]),
    "pipeline": ("pipeline::parity", [("make_fixtures.py", "PRAECISE_DIFFUSION_PARITY")], ["git+https://github.com/huggingface/diffusers", "numpy", "transformers"]),
    "qwen_image": ("qwen_image::parity", [("make_qwen_image_fixtures.py", "PRAECISE_QWEN_IMAGE_PARITY")], ["git+https://github.com/huggingface/diffusers"]),
    "qwen_image21": ("qwen_image21::parity", [("make_qwen_image21_fixtures.py", "PRAECISE_QWEN_IMAGE21_PARITY")], ["git+https://github.com/huggingface/diffusers"]),
    "umt5": ("umt5::parity", [("make_umt5_fixtures.py", "PRAECISE_UMT5_PARITY")], ["safetensors", "transformers"]),
    "wan_dit": ("wan_dit::parity", [("make_wan_dit_fixtures.py", "PRAECISE_WAN_DIT_PARITY")], ["git+https://github.com/huggingface/diffusers", "safetensors"]),
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
        for name, (test, scripts, _) in sorted(FAMILIES.items()):
            print(f"{name}: tests {test}: " + ", ".join(f"{sc} -> {var}" for sc, var in scripts))
        return 0
    py = venv_python()
    for name in argv:
        test, scripts, packages = FAMILIES[name]
        run([py, "-m", "pip", "install", "-q", *packages])
        with tempfile.TemporaryDirectory() as root:
            env = dict(os.environ)
            for script, var in scripts:
                out = os.path.join(root, var)
                run([py, os.path.join(HERE, script), out])
                env[var] = out
            run(
                ["cargo", "test", "--locked", "-p", "praecise-diffusion", "--lib", "--release", test, "--", "--ignored", "--nocapture"],
                env=env,
            )
    return 0


sys.exit(main(sys.argv[1:]))
