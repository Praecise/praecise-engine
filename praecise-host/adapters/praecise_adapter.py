"""Runtime host adapter: drives one inference engine through its offline
Python API and speaks the runtime host protocol on stdin and stdout.

Usage: python3 praecise_adapter.py <kind>, with <kind> one of vllm, sglang,
tensorrt-llm, transformers or openvino.

The accelerator is whichever one this environment's packages drive, of any
vendor: NVIDIA and AMD ROCm (torch "cuda"), Intel GPUs ("xpu"), Intel Gaudi
("hpu"), Apple ("mps"), Huawei Ascend ("npu", with torch_npu), Moore Threads
("musa", with torch_musa), and for vLLM any platform its plugins provide
(Ascend, Gaudi, TPU, Neuron, Spyre and the rest). TensorRT-LLM is NVIDIA's and
runs on NVIDIA GPUs only; OpenVINO runs on Intel CPUs, GPUs and NPUs.

Requests, one JSON object per line on stdin, each with the "id" the
application chose:
  {"id": 0, "op": "load", "entry": {"model_dir": ..., "gpu_bytes": ..., "options": {...}}}
  {"id": n, "op": "generate", "messages": [...] | "prompt": "...", "max_tokens": n,
   "temperature": t, "top_p": p, "stop": [...], "seed": s}
While generating, the adapter writes {"id": n, "event": {"text": "<delta>"}}
lines, then one reply {"id": n, "ok": true, "text": ..., "finish_reason": ...,
"usage": {"prompt_tokens": n, "completion_tokens": m}}. Any failure is replied
as {"id": n, "ok": false, "error": "..."}. Generation requests run
concurrently, so engines that batch see every outstanding request.

The engine's own output goes to stderr so stdout carries only the protocol.
When PRAECISE_GPU_REQUIRED is 1 the machine has an accelerator and the engine
must run on it: a load that would fall back to the CPU is refused.

The engine's memory is capped at the entry's "gpu_bytes" as a fraction of the
accelerator's memory. An engine whose platform cannot report that memory takes
the fraction from options["memory_fraction"]; without either the load is
refused, since an uncapped engine would take memory other models reserved.
"""

import asyncio
import importlib
import json
import os
import sys
import threading

PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def send(obj):
    PROTOCOL.write(json.dumps(obj) + "\n")
    PROTOCOL.flush()


def gpu_required():
    return os.environ.get("PRAECISE_GPU_REQUIRED") == "1"


# Vendor packages that register their device type with torch when imported.
TORCH_PLUGINS = ("torch_npu", "torch_musa", "habana_frameworks.torch")

# Device types looked for when torch.accelerator does not name one.
TORCH_DEVICES = ("cuda", "xpu", "mps", "hpu", "npu", "musa")


def torch_accelerator():
    """The torch device type of this environment's accelerator, or None when
    torch here can only use the CPU."""
    import torch

    for plugin in TORCH_PLUGINS:
        try:
            importlib.import_module(plugin)
        except ImportError:
            pass
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        device = accelerator.current_accelerator(check_available=True)
        if device is not None:
            return device.type
    for kind in TORCH_DEVICES:
        module = torch.backends.mps if kind == "mps" else getattr(torch, kind, None)
        available = getattr(module, "is_available", None)
        if available is not None and available():
            return kind
    return None


def torch_memory(kind):
    """Total memory of the first device of torch device type `kind`, in bytes."""
    import torch

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "get_memory_info"):
        try:
            return accelerator.get_memory_info(0)[1]
        except (RuntimeError, NotImplementedError):
            pass
    if kind == "cuda":
        return torch.cuda.get_device_properties(0).total_memory
    if kind == "mps":
        return torch.mps.recommended_max_memory()
    module = getattr(torch, kind, None)
    if hasattr(module, "mem_get_info"):
        return module.mem_get_info(0)[1]
    raise RuntimeError(f"torch reports no memory size for its {kind!r} device")


def fraction(entry, total):
    """The memory cap as a fraction of `total` bytes of device memory."""
    return max(0.01, min(0.99, entry["gpu_bytes"] / total))


def memory_fraction(entry, options, total):
    """The engine's memory cap: options["memory_fraction"] when the operator
    set one, otherwise gpu_bytes over what `total()` reports. `options` loses
    the key."""
    explicit = options.pop("memory_fraction", None)
    if explicit is not None:
        return max(0.01, min(0.99, float(explicit)))
    try:
        return fraction(entry, total())
    except (RuntimeError, NotImplementedError, AttributeError) as e:
        raise RuntimeError(
            f"the accelerator's memory size is not reported here ({e}); set options.memory_fraction "
            "to the share of it this model may use"
        ) from e


def require_accelerator(engine):
    """The torch device type the engine runs on; None only on a machine
    without an accelerator, where the engine may use the CPU."""
    kind = torch_accelerator()
    if kind is None and gpu_required():
        raise RuntimeError(f"the machine has an accelerator but torch in this {engine} environment cannot use it")
    return kind


def delta(previous, current):
    """The new text, whether the engine reports cumulative text or pieces."""
    if current.startswith(previous):
        return current[len(previous):], current
    return current, previous + current


def chat_prompt(tokenizer, req):
    if "messages" in req:
        return tokenizer.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
    return req["prompt"]


class Transformers:
    """Hugging Face transformers; one generation at a time."""

    def __init__(self, entry):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        options = entry.get("options", {})
        self.device = require_accelerator("transformers") or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(entry["model_dir"], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            entry["model_dir"], local_files_only=True, dtype=options.get("dtype", "auto")
        ).to(self.device)
        self.model.eval()
        self.lock = threading.Lock()

    def run(self, req, emit):
        import torch
        from transformers import TextIteratorStreamer

        with self.lock:
            inputs = self.tokenizer(chat_prompt(self.tokenizer, req), return_tensors="pt").to(self.device)
            if "seed" in req:
                torch.manual_seed(req["seed"])
            temperature = req.get("temperature", 1.0)
            kwargs = dict(
                **inputs,
                max_new_tokens=req.get("max_tokens", 256),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=req.get("top_p", 1.0))
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            result = {}

            def work():
                result["out"] = self.model.generate(**kwargs, streamer=streamer)

            worker = threading.Thread(target=work)
            worker.start()
            text = ""
            for piece in streamer:
                if piece:
                    text += piece
                    emit(piece)
            worker.join()
            prompt_tokens = int(inputs["input_ids"].shape[1])
            completion_tokens = int(result["out"].shape[1]) - prompt_tokens
            finish = "length" if completion_tokens >= kwargs["max_new_tokens"] else "stop"
            return text, finish, prompt_tokens, completion_tokens

    async def generate(self, req, emit):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, req, lambda p: loop.call_soon_threadsafe(emit, p))


class Vllm:
    def __init__(self, entry):
        from vllm import AsyncEngineArgs
        from vllm.platforms import current_platform
        from vllm.v1.engine.async_llm import AsyncLLM

        # vLLM names its own platform: CUDA, ROCm, XPU, TPU, or the one an
        # installed vendor plugin provides. Only its CPU platform is not an
        # accelerator.
        options = dict(entry.get("options", {}))
        if current_platform.is_cpu():
            if gpu_required():
                raise RuntimeError("the machine has an accelerator but vLLM in this environment runs on its CPU platform")
            options.pop("memory_fraction", None)
        else:
            options["gpu_memory_utilization"] = memory_fraction(
                entry, options, lambda: current_platform.get_device_total_memory(0)
            )
        args = AsyncEngineArgs(model=entry["model_dir"], **options)
        self.engine = AsyncLLM.from_engine_args(args)
        self.tokenizer = None

    async def generate(self, req, emit):
        from vllm import SamplingParams

        if self.tokenizer is None:
            self.tokenizer = await self.engine.get_tokenizer()
        params = SamplingParams(
            max_tokens=req.get("max_tokens", 256),
            temperature=req.get("temperature", 1.0),
            top_p=req.get("top_p", 1.0),
            stop=req.get("stop"),
            seed=req.get("seed"),
        )
        text, last = "", None
        async for out in self.engine.generate(chat_prompt(self.tokenizer, req), params, str(req["id"])):
            piece, text = delta(text, out.outputs[0].text)
            if piece:
                emit(piece)
            last = out
        completion = last.outputs[0]
        return text, completion.finish_reason or "stop", len(last.prompt_token_ids), len(completion.token_ids)


# Device types SGLang's server accepts.
SGLANG_DEVICES = ("cuda", "xpu", "hpu", "npu", "musa")


class Sglang:
    def __init__(self, entry):
        import sglang

        options = dict(entry.get("options", {}))
        kind = require_accelerator("sglang")
        if kind is None:
            options.pop("memory_fraction", None)
            options.setdefault("device", "cpu")
        else:
            if kind not in SGLANG_DEVICES:
                raise RuntimeError(f"SGLang has no backend for torch's {kind!r} device ({', '.join(SGLANG_DEVICES)})")
            options.setdefault("device", kind)
            options["mem_fraction_static"] = memory_fraction(entry, options, lambda: torch_memory(kind))
        self.engine = sglang.Engine(model_path=entry["model_dir"], **options)
        self.tokenizer = self.engine.tokenizer_manager.tokenizer

    async def generate(self, req, emit):
        params = {
            "max_new_tokens": req.get("max_tokens", 256),
            "temperature": req.get("temperature", 1.0),
            "top_p": req.get("top_p", 1.0),
        }
        if req.get("stop"):
            params["stop"] = req["stop"]
        if "seed" in req:
            params["sampling_seed"] = req["seed"]
        text, meta = "", {}
        stream = await self.engine.async_generate(chat_prompt(self.tokenizer, req), params, stream=True)
        async for chunk in stream:
            piece, text = delta(text, chunk["text"])
            if piece:
                emit(piece)
            meta = chunk["meta_info"]
        finish = meta.get("finish_reason") or {}
        return text, finish.get("type", "stop"), meta["prompt_tokens"], meta["completion_tokens"]


class TensorrtLlm:
    def __init__(self, entry):
        import torch

        # TensorRT-LLM is NVIDIA's own: ROCm also presents as torch "cuda",
        # so the HIP build is told apart and refused by name.
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            raise RuntimeError("tensorrt-llm runs on NVIDIA GPUs only and none is usable in this environment")
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        options = dict(entry.get("options", {}))
        share = memory_fraction(entry, options, lambda: torch_memory("cuda"))
        self.llm = LLM(
            model=entry["model_dir"],
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=share),
            **options,
        )
        self.tokenizer = self.llm.tokenizer

    async def generate(self, req, emit):
        from tensorrt_llm import SamplingParams

        params = SamplingParams(
            max_tokens=req.get("max_tokens", 256),
            temperature=req.get("temperature", 1.0),
            top_p=req.get("top_p", 1.0),
            stop=req.get("stop"),
            seed=req.get("seed"),
        )
        text, last = "", None
        async for out in self.llm.generate_async(chat_prompt(self.tokenizer, req), params, streaming=True):
            piece, text = delta(text, out.outputs[0].text)
            if piece:
                emit(piece)
            last = out
        completion = last.outputs[0]
        return text, completion.finish_reason or "stop", len(last.prompt_token_ids), len(completion.token_ids)


class OpenVino:
    """OpenVINO GenAI on an Intel CPU, GPU or NPU; one generation at a time.

    options["device"] names the OpenVINO device ("GPU", "GPU.1", "NPU",
    "CPU"); unset, the first GPU is used, then the NPU, then the CPU. The
    remaining options are passed to the pipeline as its properties.
    """

    def __init__(self, entry):
        import openvino
        import openvino_genai

        self.genai = openvino_genai
        options = dict(entry.get("options", {}))
        options.pop("memory_fraction", None)
        core = openvino.Core()
        present = list(core.available_devices)
        device = options.pop("device", None)
        if device is None:
            device = next((d for d in present if d.startswith("GPU")), "NPU" if "NPU" in present else "CPU")
        family = device.split(".")[0]
        if not any(d.split(".")[0] == family for d in present):
            raise RuntimeError(f"OpenVINO has no {device} device here (present: {', '.join(present)})")
        if gpu_required() and family == "CPU":
            raise RuntimeError("the machine has an accelerator but OpenVINO in this environment would run on the CPU")
        if family == "GPU":
            # OpenVINO has no memory cap to set, so a model that cannot fit
            # the device is refused before it loads.
            total = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")
            if entry["gpu_bytes"] > total:
                raise RuntimeError(f"the model reserves {entry['gpu_bytes']} bytes and {device} has {total}")
        self.pipe = openvino_genai.LLMPipeline(entry["model_dir"], device, **options)
        self.tokenizer = self.pipe.get_tokenizer()
        self.lock = threading.Lock()

    def run(self, req, emit):
        config = self.genai.GenerationConfig()
        config.max_new_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 1.0)
        config.do_sample = temperature > 0
        if temperature > 0:
            config.temperature = temperature
            config.top_p = req.get("top_p", 1.0)
        if req.get("stop"):
            config.stop_strings = set(req["stop"])
        if "seed" in req:
            config.rng_seed = req["seed"]
        # the prompt below is already rendered; the pipeline must not render it again
        if hasattr(config, "apply_chat_template"):
            config.apply_chat_template = False
        if "messages" in req:
            prompt = self.tokenizer.apply_chat_template(req["messages"], True)
        else:
            prompt = req["prompt"]
        pieces = []

        def streamer(piece):
            pieces.append(piece)
            emit(piece)
            return self.genai.StreamingStatus.RUNNING

        with self.lock:
            result = self.pipe.generate(prompt, config, streamer)
        reasons = list(getattr(result, "finish_reasons", None) or [])
        length = bool(reasons) and reasons[0] == self.genai.GenerationFinishReason.LENGTH
        metrics = result.perf_metrics
        return (
            "".join(pieces),
            "length" if length else "stop",
            metrics.get_num_input_tokens(),
            metrics.get_num_generated_tokens(),
        )

    async def generate(self, req, emit):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, req, lambda p: loop.call_soon_threadsafe(emit, p))


ENGINES = {
    "transformers": Transformers,
    "vllm": Vllm,
    "sglang": Sglang,
    "tensorrt-llm": TensorrtLlm,
    "openvino": OpenVino,
}


def refusal(rid, e):
    return {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}


async def serve(kind):
    loop = asyncio.get_running_loop()
    lines = asyncio.Queue()

    def read():
        for line in sys.stdin:
            loop.call_soon_threadsafe(lines.put_nowait, line)
        loop.call_soon_threadsafe(lines.put_nowait, None)

    threading.Thread(target=read, daemon=True).start()
    engine = None
    tasks = set()

    async def generate(req):
        rid = req["id"]
        try:
            text, finish, prompt_tokens, completion_tokens = await engine.generate(
                req, lambda piece: send({"id": rid, "event": {"text": piece}})
            )
            send(
                {
                    "id": rid,
                    "ok": True,
                    "text": text,
                    "finish_reason": finish,
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                }
            )
        except Exception as e:  # a failed request is refused; the others go on
            send(refusal(rid, e))

    while (line := await lines.get()) is not None:
        req = json.loads(line)
        rid = req.get("id")
        try:
            op = req.get("op")
            if op == "load":
                if engine is not None:
                    raise RuntimeError("a model is already loaded in this host")
                engine = ENGINES[kind](req["entry"])
                send({"id": rid, "ok": True})
            elif op == "generate":
                if engine is None:
                    raise RuntimeError("no model is loaded")
                task = asyncio.create_task(generate(req))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            else:
                raise RuntimeError(f"unknown op {op!r}")
        except Exception as e:  # every failure becomes a refusal on the protocol
            send(refusal(rid, e))
    if tasks:
        await asyncio.gather(*tasks)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ENGINES:
        sys.stderr.write(f"usage: praecise_adapter.py <{'|'.join(ENGINES)}>\n")
        return 2
    asyncio.run(serve(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
