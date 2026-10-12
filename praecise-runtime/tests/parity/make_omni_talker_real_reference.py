"""Reference speech of a full-size omni talker, without loading its thinker.

The talker reads the thinker's token embeddings (through its text projection) for
text, so a text-only turn needs the thinker's embedding table and nothing else of
it. This builds the talker's inputs exactly as the checkpoint's generate() does
(its own `_get_talker_assistant_parts`), runs the checkpoint's talker and code2wav
greedy with no repetition penalty, and writes the same fixture layout as
make_omni_talker_fixtures.py, for tests/omni_speech_parity.rs.

The checkpoint directory needs config.json, the tokenizer files and safetensors
holding `talker.*`, `code2wav.*` and `thinker.model.embed_tokens.weight` (other
tensors may be absent).

usage: python make_omni_talker_real_reference.py <checkpoint dir> <out dir> [--frames N] [--device cuda]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def load_part(ckpt: Path, prefix: str):
    out = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = s.get_tensor(k)
    return out


def fuse_experts(state: dict, n_layers: int, n_experts: int) -> dict:
    """The checkpoint stores one tensor per expert; the module holds them stacked
    (gate and up concatenated), as from_pretrained would convert them."""
    for i in range(n_layers):
        p = f"model.layers.{i}.mlp.experts."
        if p + "0.gate_proj.weight" not in state:
            continue
        gate = [state.pop(f"{p}{e}.gate_proj.weight") for e in range(n_experts)]
        up = [state.pop(f"{p}{e}.up_proj.weight") for e in range(n_experts)]
        down = [state.pop(f"{p}{e}.down_proj.weight") for e in range(n_experts)]
        state[p + "gate_up_proj"] = torch.stack([torch.cat((g, u), dim=0) for g, u in zip(gate, up)])
        state[p + "down_proj"] = torch.stack(down)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--voice", default="ethan")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ckpt, out_dir = Path(args.checkpoint), Path(args.out)
    dev = torch.device(args.device)

    config = AutoConfig.from_pretrained(ckpt)
    tok = AutoTokenizer.from_pretrained(ckpt)
    talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(config.talker_config, dtype=torch.float32)
    text_cfg = config.talker_config.text_config
    state = fuse_experts(load_part(ckpt, "talker."), text_cfg.num_hidden_layers, text_cfg.num_experts)
    missing = talker.load_state_dict(state, strict=False)
    assert not missing.missing_keys, missing.missing_keys[:5]
    assert not missing.unexpected_keys, missing.unexpected_keys[:5]
    code2wav = Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config, dtype=torch.float32)
    missing = code2wav.load_state_dict(load_part(ckpt, "code2wav."), strict=False)
    assert not missing.missing_keys, missing.missing_keys[:5]
    embed = load_part(ckpt, "thinker.model.embed_tokens.")["weight"].float()
    talker, code2wav, embed = talker.to(dev).eval(), code2wav.to(dev).eval(), embed.to(dev)

    user = "<|im_start|>user\nPlease say hello and tell me the weather is lovely today.<|im_end|>\n"
    reply = "Hello there! The weather is lovely today, isn't it?<|im_end|>"
    user_ids = tok(user, add_special_tokens=False)["input_ids"]
    head_ids = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    reply_ids = tok(reply, add_special_tokens=False)["input_ids"]
    assert len(head_ids) == 3

    fake = SimpleNamespace(config=config, talker=talker)
    with torch.no_grad():
        thinker_embed = embed[torch.tensor([user_ids + head_ids + reply_ids], device=dev)]
        user_part = talker.text_projection(thinker_embed[:, : len(user_ids)])
        special = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], device=dev)
        tts_bos, tts_eos, tts_pad = talker.text_projection(embed[special]).chunk(3, dim=1)
        speaker_id = config.talker_config.speaker_id[args.voice]
        assistant, assistant_ids, trailing = Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts(
            fake, len(user_ids), thinker_embed.shape[1], speaker_id, thinker_embed, tts_pad, tts_bos, tts_eos
        )
        inputs = torch.cat((user_part, assistant), dim=1)
        input_ids = torch.cat((torch.tensor([user_ids], device=dev), assistant_ids), dim=1)
        vocab = config.talker_config.text_config.vocab_size
        eos = config.talker_config.codec_eos_token_id
        result = talker.generate(
            inputs_embeds=inputs,
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            talker_input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape, dtype=torch.long, device=dev),
            max_new_tokens=args.frames + 1,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=eos,
            suppress_tokens=[i for i in range(vocab - 1024, vocab) if i != eos],
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        codes = torch.stack([h[-1] for h in result.hidden_states if h[-1] is not None], dim=1).transpose(1, 2)
        wav = code2wav.chunked_decode(codes, chunk_size=300, left_context_size=25).reshape(-1).float().cpu()

    case_dir = out_dir / "text"
    case_dir.mkdir(parents=True, exist_ok=True)
    wav.numpy().astype("<f4").tofile(case_dir / "pcm.f32")
    # the codes, frame-major, for a check of the waveform decoder on its own
    codes[0].transpose(0, 1).contiguous().cpu().numpy().astype("<i4").tofile(case_dir / "codes.i32")
    revision = (ckpt / ".revision").read_text().strip() if (ckpt / ".revision").exists() else ""
    case = {
        "name": "text",
        "user_tokens": user_ids,
        "media_rows": 0,
        "hidden_width": 0,
        "reply_tokens": reply_ids,
        "voice": args.voice,
        "frames": int(codes.shape[-1]),
        "samples": int(wav.shape[0]),
        "talker_prompt_rows": int(inputs.shape[1]),
    }
    (out_dir / "expected.json").write_text(
        json.dumps({"checkpoint_revision": revision, "sample_rate": 24000, "cases": [case]}, indent=1) + "\n"
    )
    print(json.dumps({"frames": case["frames"], "samples": case["samples"], "rows": case["talker_prompt_rows"]}))


if __name__ == "__main__":
    main()
