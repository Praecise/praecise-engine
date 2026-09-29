"""Runtime host adapter: drives one inference engine through its offline
Python API and speaks the runtime host protocol on stdin and stdout.

Usage: python3 praecise_adapter.py <kind>, with <kind> one of vllm, sglang,
tensorrt-llm or transformers.

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
When PRAECISE_GPU_REQUIRED is 1 the machine has a GPU and the engine must run
on it: a load that would fall back to the CPU is refused.
"""

import asyncio
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


def gpu_fraction(entry):
    """The GPU memory cap as a fraction of the first device's memory."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    return max(0.01, min(0.99, entry["gpu_bytes"] / total))


def require_cuda(kind):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{kind} needs a CUDA GPU and none is usable in this environment")


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

        cuda = torch.cuda.is_available()
        if gpu_required() and not cuda:
            raise RuntimeError("the machine has a GPU but torch in this environment cannot use it")
        options = entry.get("options", {})
        self.device = "cuda" if cuda else "cpu"
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
        require_cuda("vllm")
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        options = entry.get("options", {})
        args = AsyncEngineArgs(model=entry["model_dir"], gpu_memory_utilization=gpu_fraction(entry), **options)
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


class Sglang:
    def __init__(self, entry):
        require_cuda("sglang")
        import sglang

        options = entry.get("options", {})
        self.engine = sglang.Engine(model_path=entry["model_dir"], mem_fraction_static=gpu_fraction(entry), **options)
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
        require_cuda("tensorrt-llm")
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        options = entry.get("options", {})
        self.llm = LLM(
            model=entry["model_dir"],
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=gpu_fraction(entry)),
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


ENGINES = {"transformers": Transformers, "vllm": Vllm, "sglang": Sglang, "tensorrt-llm": TensorrtLlm}


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
