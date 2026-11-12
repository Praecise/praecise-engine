"""Accelerator selection in the runtime host adapter, on every vendor.

Runs the adapter's own code against stand-ins for torch and the engine
packages, each describing one machine: what torch reports there, and what
the engine was then asked to do. No engine, model or accelerator is needed:

    python3 praecise-host/tests/adapter_accelerators.py
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

ADAPTER = Path(__file__).resolve().parent.parent / "adapters" / "praecise_adapter.py"
GIB = 1 << 30


def load_adapter():
    """A fresh copy of the adapter module, with stdout left alone."""
    stdout = sys.stdout
    spec = importlib.util.spec_from_file_location("praecise_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout = stdout
    return module


def device_module(available, total=None):
    """A torch device namespace (torch.cuda, torch.xpu, ...)."""
    module = types.SimpleNamespace(is_available=lambda: available)
    if total is not None:
        module.mem_get_info = lambda index=None: (total // 2, total)
        module.get_device_properties = lambda index: types.SimpleNamespace(total_memory=total)
    return module


def fake_torch(kind=None, total=64 * GIB, accelerator_api=True, hip=None, memory_api=True):
    """A torch whose accelerator is of device type `kind` (None: CPU only)."""
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=hip)
    torch.cuda = device_module(kind == "cuda", total)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: kind == "mps"))
    torch.mps = types.SimpleNamespace(recommended_max_memory=lambda: total)
    for name in ("xpu", "hpu", "npu", "musa"):
        if kind == name:
            setattr(torch, name, device_module(True, total))
    if accelerator_api:
        api = types.SimpleNamespace(
            current_accelerator=lambda check_available=False: (
                types.SimpleNamespace(type=kind) if kind in ("cuda", "xpu", "mps", "hpu") else None
            )
        )
        if memory_api:

            def get_memory_info(index=None):
                if kind == "mps":
                    raise RuntimeError("not implemented for this device")
                return (total // 2, total)

            api.get_memory_info = get_memory_info
        torch.accelerator = api
    return torch


class Machine:
    """Installs stand-in modules for one test and removes them after."""

    def __init__(self, test, gpu_required=True, **modules):
        self.saved = {name: sys.modules.get(name) for name in modules}
        self.names = list(modules)
        sys.modules.update(modules)
        self.env = os.environ.get("PRAECISE_GPU_REQUIRED")
        os.environ["PRAECISE_GPU_REQUIRED"] = "1" if gpu_required else "0"
        test.addCleanup(self.restore)

    def restore(self):
        for name in self.names:
            if self.saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self.saved[name]
        if self.env is None:
            os.environ.pop("PRAECISE_GPU_REQUIRED", None)
        else:
            os.environ["PRAECISE_GPU_REQUIRED"] = self.env


def entry(gpu_bytes=16 * GIB, **options):
    return {"model_dir": "/models/m", "gpu_bytes": gpu_bytes, "options": options}


class TorchAccelerator(unittest.TestCase):
    def test_each_vendor_device_type_is_found(self):
        for kind in ("cuda", "xpu", "mps", "hpu", "npu", "musa"):
            with self.subTest(kind=kind):
                Machine(self, torch=fake_torch(kind))
                self.assertEqual(load_adapter().torch_accelerator(), kind)

    def test_a_torch_without_the_accelerator_api_is_still_read(self):
        # torch before 2.6, and vendor builds that predate torch.accelerator
        for kind in ("cuda", "xpu", "npu"):
            with self.subTest(kind=kind):
                Machine(self, torch=fake_torch(kind, accelerator_api=False))
                self.assertEqual(load_adapter().torch_accelerator(), kind)

    def test_a_cpu_only_torch_has_no_accelerator(self):
        Machine(self, torch=fake_torch(None))
        self.assertIsNone(load_adapter().torch_accelerator())

    def test_a_machine_with_an_accelerator_refuses_a_cpu_only_torch(self):
        Machine(self, gpu_required=True, torch=fake_torch(None))
        with self.assertRaisesRegex(RuntimeError, "cannot use it"):
            load_adapter().require_accelerator("transformers")
        Machine(self, gpu_required=False, torch=fake_torch(None))
        self.assertIsNone(load_adapter().require_accelerator("transformers"))

    def test_memory_is_read_from_the_device_that_is_present(self):
        for kind in ("cuda", "xpu", "mps", "npu", "musa"):
            for memory_api in (True, False):
                with self.subTest(kind=kind, memory_api=memory_api):
                    Machine(self, torch=fake_torch(kind, total=48 * GIB, memory_api=memory_api))
                    self.assertEqual(load_adapter().torch_memory(kind), 48 * GIB)


class MemoryFraction(unittest.TestCase):
    def test_the_cap_is_the_reservation_over_the_device(self):
        adapter = load_adapter()
        options = {}
        self.assertAlmostEqual(adapter.memory_fraction(entry(16 * GIB), options, lambda: 64 * GIB), 0.25)

    def test_an_operator_fraction_wins_and_leaves_the_options(self):
        adapter = load_adapter()
        options = {"memory_fraction": 0.5, "dtype": "auto"}

        def unreported():
            raise NotImplementedError("no memory size")

        self.assertEqual(adapter.memory_fraction(entry(), options, unreported), 0.5)
        self.assertEqual(options, {"dtype": "auto"})

    def test_an_unreported_memory_size_without_a_fraction_is_refused(self):
        adapter = load_adapter()

        def unreported():
            raise NotImplementedError("no memory size")

        with self.assertRaisesRegex(RuntimeError, "memory_fraction"):
            adapter.memory_fraction(entry(), {}, unreported)


def fake_vllm(platform):
    """vllm with `platform` as its current platform; records the engine args."""
    seen = {}
    vllm = types.ModuleType("vllm")

    def engine_args(**kwargs):
        seen.update(kwargs)
        return kwargs

    vllm.AsyncEngineArgs = engine_args
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = platform
    async_llm = types.ModuleType("vllm.v1.engine.async_llm")
    async_llm.AsyncLLM = types.SimpleNamespace(from_engine_args=lambda args: object())
    modules = {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": async_llm,
    }
    return modules, seen


def platform(cpu=False, total=64 * GIB):
    def memory(index=0):
        if total is None:
            raise NotImplementedError("this platform reports no memory size")
        return total

    return types.SimpleNamespace(is_cpu=lambda: cpu, get_device_total_memory=memory)


class Vllm(unittest.TestCase):
    def test_any_accelerator_platform_is_capped_by_its_own_memory(self):
        # ROCm, XPU and a vendor plugin all answer through the platform
        modules, seen = fake_vllm(platform(total=32 * GIB))
        Machine(self, **modules)
        load_adapter().Vllm(entry(8 * GIB))
        self.assertAlmostEqual(seen["gpu_memory_utilization"], 0.25)
        self.assertEqual(seen["model"], "/models/m")

    def test_the_cpu_platform_is_refused_on_a_machine_with_an_accelerator(self):
        modules, _ = fake_vllm(platform(cpu=True))
        Machine(self, gpu_required=True, **modules)
        with self.assertRaisesRegex(RuntimeError, "CPU platform"):
            load_adapter().Vllm(entry())

    def test_the_cpu_platform_serves_a_machine_without_one(self):
        modules, seen = fake_vllm(platform(cpu=True))
        Machine(self, gpu_required=False, **modules)
        load_adapter().Vllm(entry(memory_fraction=0.5))
        self.assertNotIn("gpu_memory_utilization", seen)
        self.assertNotIn("memory_fraction", seen)

    def test_a_platform_that_reports_no_memory_needs_the_operator_fraction(self):
        modules, seen = fake_vllm(platform(total=None))
        Machine(self, **modules)
        with self.assertRaisesRegex(RuntimeError, "memory_fraction"):
            load_adapter().Vllm(entry())
        load_adapter().Vllm(entry(memory_fraction=0.4))
        self.assertEqual(seen["gpu_memory_utilization"], 0.4)


def fake_sglang(seen):
    sglang = types.ModuleType("sglang")

    def engine(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(tokenizer_manager=types.SimpleNamespace(tokenizer=None))

    sglang.Engine = engine
    return sglang


class Sglang(unittest.TestCase):
    def test_the_device_and_cap_follow_the_accelerator(self):
        for kind in ("cuda", "xpu", "hpu", "npu", "musa"):
            with self.subTest(kind=kind):
                seen = {}
                Machine(self, torch=fake_torch(kind, total=64 * GIB), sglang=fake_sglang(seen))
                load_adapter().Sglang(entry(16 * GIB))
                self.assertEqual(seen["device"], kind)
                self.assertAlmostEqual(seen["mem_fraction_static"], 0.25)

    def test_a_device_sglang_has_no_backend_for_is_refused(self):
        Machine(self, torch=fake_torch("mps"), sglang=fake_sglang({}))
        with self.assertRaisesRegex(RuntimeError, "no backend for torch's 'mps'"):
            load_adapter().Sglang(entry())

    def test_a_machine_without_an_accelerator_runs_on_the_cpu(self):
        seen = {}
        Machine(self, gpu_required=False, torch=fake_torch(None), sglang=fake_sglang(seen))
        load_adapter().Sglang(entry())
        self.assertEqual(seen["device"], "cpu")
        self.assertNotIn("mem_fraction_static", seen)


class TensorrtLlm(unittest.TestCase):
    def test_rocm_presenting_as_cuda_is_refused(self):
        Machine(self, torch=fake_torch("cuda", hip="7.1.0"))
        with self.assertRaisesRegex(RuntimeError, "NVIDIA GPUs only"):
            load_adapter().TensorrtLlm(entry())

    def test_other_vendors_are_refused(self):
        for kind in ("xpu", "npu", None):
            with self.subTest(kind=kind):
                Machine(self, torch=fake_torch(kind))
                with self.assertRaisesRegex(RuntimeError, "NVIDIA GPUs only"):
                    load_adapter().TensorrtLlm(entry())


def fake_openvino(devices, gpu_total=16 * GIB, seen=None):
    seen = {} if seen is None else seen
    openvino = types.ModuleType("openvino")
    core = types.SimpleNamespace(available_devices=devices, get_property=lambda device, name: gpu_total)
    openvino.Core = lambda: core
    genai = types.ModuleType("openvino_genai")

    class Config:
        apply_chat_template = True

    class Tokenizer:
        def apply_chat_template(self, history, add_generation_prompt):
            return "<rendered>" + history[-1]["content"]

    class Pipeline:
        def __init__(self, model_dir, device, **properties):
            seen.update(model_dir=model_dir, device=device, properties=properties)

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, prompt, config, streamer):
            seen.update(prompt=prompt, config=config)
            for piece in ("Hel", "lo"):
                streamer(piece)
            metrics = types.SimpleNamespace(get_num_input_tokens=lambda: 7, get_num_generated_tokens=lambda: 2)
            return types.SimpleNamespace(finish_reasons=[genai.GenerationFinishReason.LENGTH], perf_metrics=metrics)

    genai.GenerationConfig = Config
    genai.LLMPipeline = Pipeline
    genai.StreamingStatus = types.SimpleNamespace(RUNNING=0)
    genai.GenerationFinishReason = types.SimpleNamespace(STOP=1, LENGTH=2)
    return {"openvino": openvino, "openvino_genai": genai}, seen


class OpenVino(unittest.TestCase):
    def test_the_gpu_is_preferred_then_the_npu_then_the_cpu(self):
        for devices, want in (
            (["CPU", "GPU.0", "GPU.1", "NPU"], "GPU.0"),
            (["CPU", "GPU"], "GPU"),
            (["CPU", "NPU"], "NPU"),
        ):
            with self.subTest(devices=devices):
                modules, seen = fake_openvino(devices)
                Machine(self, **modules)
                load_adapter().OpenVino(entry(4 * GIB))
                self.assertEqual(seen["device"], want)

    def test_the_cpu_is_refused_on_a_machine_with_an_accelerator(self):
        modules, _ = fake_openvino(["CPU"])
        Machine(self, gpu_required=True, **modules)
        with self.assertRaisesRegex(RuntimeError, "would run on the CPU"):
            load_adapter().OpenVino(entry())
        modules, seen = fake_openvino(["CPU"])
        Machine(self, gpu_required=False, **modules)
        load_adapter().OpenVino(entry())
        self.assertEqual(seen["device"], "CPU")

    def test_an_absent_device_and_a_model_too_large_are_refused(self):
        modules, _ = fake_openvino(["CPU", "GPU"])
        Machine(self, **modules)
        with self.assertRaisesRegex(RuntimeError, "no NPU device"):
            load_adapter().OpenVino(entry(device="NPU"))
        with self.assertRaisesRegex(RuntimeError, "reserves"):
            load_adapter().OpenVino(entry(32 * GIB))

    def test_a_generation_streams_and_counts_tokens(self):
        modules, seen = fake_openvino(["CPU", "GPU"])
        Machine(self, **modules)
        engine = load_adapter().OpenVino(entry(4 * GIB, device="GPU", CACHE_DIR="/scratch"))
        self.assertEqual(seen["properties"], {"CACHE_DIR": "/scratch"})
        pieces = []
        request = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0, "stop": ["\n"]}
        text, finish, prompt_tokens, completion_tokens = engine.run(request, pieces.append)
        self.assertEqual((text, finish, prompt_tokens, completion_tokens), ("Hello", "length", 7, 2))
        self.assertEqual(pieces, ["Hel", "lo"])
        self.assertEqual(seen["prompt"], "<rendered>hi")
        config = seen["config"]
        self.assertFalse(config.do_sample)
        self.assertFalse(config.apply_chat_template)
        self.assertEqual(config.stop_strings, {"\n"})
        self.assertEqual(config.max_new_tokens, 2)


if __name__ == "__main__":
    unittest.main()
