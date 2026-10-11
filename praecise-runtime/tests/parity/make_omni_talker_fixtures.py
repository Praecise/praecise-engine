"""Reference speech of an omni model's talker, for tests/omni_speech_parity.rs.

Runs the checkpoint's own generate() (thinker reply, then talker and code2wav) with greedy
sampling everywhere and no repetition penalty, and records what the engine needs to repeat
the talker part: the user positions the talker reads (thinker token ids, and the thinker's
inner hidden state for audio positions), the reply tokens, the voice, the number of frames
and the waveform.

usage: python make_omni_talker_fixtures.py <checkpoint dir> <out dir> [--frames N] [--reply N]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration, WhisperFeatureExtractor
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths


def user_positions(model, input_ids):
    cfg = model.config
    ids = input_ids[0].tolist()
    out = []
    role = None
    for i, t in enumerate(ids):
        if t == cfg.im_start_token_id and i + 1 < len(ids):
            role = ids[i + 1]
        if role == cfg.user_token_id:
            out.append(i)
    return out


def prepare(model, tokenizer, features, conversation, audio):
    """What the checkpoint's processor makes of a text-and-audio conversation."""
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    extra = {}
    if audio is not None:
        feats = features(audio, sampling_rate=features.sampling_rate, padding=True, truncation=False,
                         return_attention_mask=True, return_tensors="pt")
        n_window = model.config.thinker_config.audio_config.n_window
        n = int(_get_feat_extract_output_lengths(feats["attention_mask"].sum(-1), n_window)[0])
        assert text.count("<|audio_pad|>") == 1
        text = text.replace("<|audio_pad|>", "<|audio_pad|>" * n)
        extra = {"input_features": feats["input_features"], "feature_attention_mask": feats["attention_mask"]}
    inputs = dict(tokenizer(text, return_tensors="pt"))
    inputs.update(extra)
    return inputs


def run_case(model, processor, name, conversation, audio, frames, reply_len, voice, out_dir):
    tokenizer, features = processor
    inputs = prepare(model, tokenizer, features, conversation, audio)

    captured = {}
    talker_generate = model.talker.generate

    def spy(*args, **kw):
        captured["inputs_embeds"] = kw["inputs_embeds"].detach().clone()
        return talker_generate(*args, **kw)

    model.talker.generate = spy

    thinker_hidden = {}
    hook_layer = model.config.talker_config.accept_hidden_layer

    with torch.no_grad():
        # the thinker's inner hidden state over the prompt, as generate() hands it to the talker
        out = model.thinker(**inputs, output_hidden_states=True)
        thinker_hidden["prompt"] = out.hidden_states[hook_layer][0]

        seqs, wav = model.generate(
            **inputs,
            speaker=voice,
            thinker_max_new_tokens=reply_len,
            thinker_do_sample=False,
            talker_do_sample=False,
            talker_repetition_penalty=1.0,
            talker_max_new_tokens=frames + 1,
        )
    model.talker.generate = talker_generate

    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    pos = user_positions(model, input_ids)
    user_tokens = [int(input_ids[0, i]) for i in pos]
    media = [i for i in pos if int(input_ids[0, i]) == model.config.thinker_config.audio_token_id]
    hidden = thinker_hidden["prompt"][media].float().numpy() if media else np.zeros((0, 1), dtype=np.float32)
    reply = [int(t) for t in seqs[0, prompt_len:].tolist()]

    wav = wav.reshape(-1).float().numpy()
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    wav.astype("<f4").tofile(case_dir / "pcm.f32")
    if len(media):
        hidden.astype("<f4").tofile(case_dir / "hidden.f32")
    return {
        "name": name,
        "user_tokens": user_tokens,
        "media_rows": len(media),
        "hidden_width": int(hidden.shape[1]) if len(media) else 0,
        "reply_tokens": reply,
        "voice": voice,
        "frames": frames,
        "samples": int(wav.shape[0]),
        "talker_prompt_rows": int(captured["inputs_embeds"].shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--reply", type=int, default=12)
    ap.add_argument("--voice", default="ethan")
    args = ap.parse_args()

    torch.manual_seed(0)
    ckpt = Path(args.checkpoint)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(ckpt, dtype=torch.float32).eval()
    # text and audio only: no picture or video processor is needed (or loaded)
    processor = (AutoTokenizer.from_pretrained(ckpt), WhisperFeatureExtractor.from_pretrained(ckpt))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = processor[1].sampling_rate
    t = np.arange(int(sr * 1.5)) / sr
    tone = (0.3 * np.sin(2 * math.pi * 220.0 * t) * np.exp(-t)).astype(np.float32)

    cases = [
        run_case(model, processor, "text",
                 [{"role": "user", "content": [{"type": "text", "text": "Say something short."}]}],
                 None, args.frames, args.reply, args.voice, out_dir),
        run_case(model, processor, "audio",
                 [{"role": "user", "content": [{"type": "audio", "audio": "tone"},
                                               {"type": "text", "text": "What was that?"}]}],
                 tone, args.frames, args.reply, args.voice, out_dir),
    ]
    revision = (ckpt / ".revision").read_text().strip() if (ckpt / ".revision").exists() else ""
    expected = {"checkpoint_revision": revision, "sample_rate": 24000, "cases": cases}
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(json.dumps({c["name"]: (c["samples"], c["talker_prompt_rows"], len(c["reply_tokens"])) for c in cases}))


if __name__ == "__main__":
    main()
