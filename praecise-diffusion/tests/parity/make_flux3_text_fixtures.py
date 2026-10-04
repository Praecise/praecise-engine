"""Reference text contexts for the FLUX 3 text-encoder parity test.

usage: make_flux3_text_fixtures.py <path/to/flux3/f3/text_encoder.py> <tokenizer dir> <out dir>

The tokenizer dir holds the released text encoder's tokenizer, chat template
and processor files; the model is a tiny random Qwen3-VL with the same vocab.
"""
import json
import pathlib
import shutil
import sys

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

src, tok, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
keep = [l for l in src.read_text().splitlines() if not l.startswith("from lerobot")]
ns = {"__name__": "ref_te", "_transformers_available": True, "require_package": lambda *a, **k: None}
exec(compile("\n".join(keep), str(src), "exec"), ns)


class TextOnlyProcessor:
    """The released processor without its image/video halves (text prompts only)."""

    @classmethod
    def from_pretrained(cls, path, **_):
        from transformers import AutoTokenizer
        p = cls()
        p.tokenizer = AutoTokenizer.from_pretrained(path)
        p.tokenizer.chat_template = json.loads((pathlib.Path(path) / "chat_template.json").read_text())["chat_template"]
        return p

    def apply_chat_template(self, messages, **kw):
        return self.tokenizer.apply_chat_template(messages, **kw)


ns["AutoProcessor"] = TextOnlyProcessor

model_dir = out / "model"
model_dir.mkdir(parents=True, exist_ok=True)
for f in tok.iterdir():
    if f.name != "config.json":
        shutil.copy(f, model_dir / f.name)
released = json.loads((tok / "config.json").read_text())
text = dict(released["text_config"], hidden_size=64, intermediate_size=96, num_hidden_layers=6,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rope_scaling={"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"})
vision = dict(released["vision_config"], depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
              out_hidden_size=64, deepstack_visual_indexes=[0])
cfg = Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=released.get("image_token_id"),
                    video_token_id=released.get("video_token_id"), tie_word_embeddings=True)
torch.manual_seed(0)
model = Qwen3VLForConditionalGeneration(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("norm.weight"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
model.save_pretrained(model_dir, safe_serialization=True)

# Interior layers only, like the released policies (36 layers, last read 32):
# the reference returns the final hidden state after the model norm.
layers = [1, 3, 5]
enc = ns["Qwen3VLEmbedder"](str(model_dir), ns["Qwen3VLEmbedderParams"](output_layer=layers, torch_dtype="float32"))
cases = []
for i, (prompt, fixed) in enumerate([
    ("pick up the red cube and put it in the bowl", 80),
    ("Fold the towel in half, then slide it to the left edge of the table.", None),
    ("stack " * 60 + "the blocks", 40),
]):
    ctx = enc.forward_bucketed(prompt, fixed_length=fixed)[0]
    formatted = enc.processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    real = enc.processor.tokenizer(formatted, truncation=True, max_length=fixed or 8192)["input_ids"]
    n = ctx.shape[0]
    ids = list(real) + [enc.processor.tokenizer.pad_token_id] * (n - len(real))
    (out / f"ctx{i}.f32").write_bytes(ctx.float().numpy().tobytes())
    cases.append({"prompt": prompt, "fixed_length": fixed, "ids": ids, "real": len(real)})
    print(i, n, len(real), repr(formatted[:60]))
(out / "cases.json").write_text(json.dumps({"layers": layers, "cases": cases}))
