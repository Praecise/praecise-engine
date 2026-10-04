"""Build a small random LTX-2.3 checkpoint and reference end-to-end outputs.

Every component (text encoder, prompt connectors, transformer, video and audio
decoders, vocoder) is the released layout at tiny, mutually consistent widths.
The reference pipeline runs a few guided steps (classifier-free,
spatio-temporal and modality-isolation guidance, with rescaling) from saved
starting noise and fixed prompt tokens; the final latents of both streams, the
decoded frames and the waveform are written next to the checkpoint.

The checkpoint is written the way the release ships it: one single file under
the original names with the configuration in its header, plus the text
encoder in `text/text_encoder/`.

Usage: python make_ltx2_pipeline_fixtures.py <out dir>
"""

import json
from types import SimpleNamespace
import os
import sys

import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from safetensors.torch import save_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_gemma3_fixtures as gem  # noqa: E402
import make_ltx2_audio_vae_fixtures as avae  # noqa: E402
import make_ltx2_connectors_fixtures as conn  # noqa: E402
import make_ltx2_fixtures as tf  # noqa: E402
import make_ltx2_vae_fixtures as vvae  # noqa: E402
import make_ltx2_vocoder_fixtures as voc  # noqa: E402

SEQ = 8
POSITIVE = [2, 17, 33, 9, 50]
NEGATIVE = [2]
FRAMES, HEIGHT, WIDTH, FPS, STEPS = 9, 64, 96, 24.0, 3
STG_BLOCKS = [1]

STATES = gem.TEXT["num_hidden_layers"] + 1
WIDTH_TEXT = gem.TEXT["hidden_size"]
AUDIO_CH = avae.DD["z_channels"] * (avae.DD["mel_bins"] // 4)
MELS = avae.DD["mel_bins"]

TRANSFORMER = dict(tf.TINY, caption_channels=64, audio_in_channels=AUDIO_CH, audio_out_channels=AUDIO_CH)
CONNECTORS = dict(
    conn.TINY, caption_channels=WIDTH_TEXT, text_proj_in_factor=STATES,
    video_connector_attention_head_dim=32, audio_connector_attention_head_dim=16,
    video_hidden_dim=64, audio_hidden_dim=32,
)
HEADER = {
    "transformer": dict(
        tf.HEADER["transformer"], audio_out_channels=AUDIO_CH,
        **dict(conn.HEADER, caption_channels=WIDTH_TEXT, connector_attention_head_dim=32, audio_connector_attention_head_dim=16),
    ),
    "vae": vvae.HEADER,
    "audio_vae": {
        "model": {"params": {"ddconfig": avae.DD, "sampling_rate": 16000}},
        "preprocessing": {"audio": {"sampling_rate": 16000}, "stft": {"hop_length": 160}},
    },
    "vocoder": dict(voc.HEADER, bwe=dict(voc.HEADER["bwe"], num_mels=MELS)),
}


def randomise(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 1 and ("norm" in name or name.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def dump(out, name, t):
    t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))


def text_encoder(out):
    cfg = Gemma3TextConfig(**gem.TEXT, pad_token_id=0)
    cfg._attn_implementation = "eager"
    model = Gemma3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_((0.1 if "norm" in name else 0.08) * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    d = os.path.join(out, "text", "text_encoder")
    os.makedirs(d, exist_ok=True)
    sd = {k.replace("model.", "language_model.model.", 1): v.contiguous() for k, v in model.state_dict().items() if k.startswith("model.")}
    save_file(sd, os.path.join(d, "model.safetensors"))
    text = dict(gem.TEXT, model_type="gemma3_text", layer_types=list(cfg.layer_types))
    json.dump({"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3", "text_config": text},
              open(os.path.join(d, "config.json"), "w"), indent=1)
    return model


def prompt_embeds(model, tokens):
    """As the reference pipeline encodes a prompt: left padding, every hidden state."""
    ids = torch.zeros(1, SEQ, dtype=torch.int64)
    ids[0, SEQ - len(tokens):] = torch.tensor(tokens)
    mask = torch.zeros(1, SEQ, dtype=torch.int64)
    mask[0, SEQ - len(tokens):] = 1
    with torch.no_grad():
        hs = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    return torch.stack(hs, dim=-1).flatten(2, 3), mask


def build(out, scheduler):
    """The tiny components, the single checkpoint file under `out`, and the
    reference pipeline over them."""
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    gemma = text_encoder(out)

    transformer = LTX2VideoTransformer3DModel(**TRANSFORMER).eval()
    tf.randomise(transformer, 1)
    connectors = LTX2TextConnectors(**CONNECTORS).eval()
    randomise(connectors, 2)

    vae = AutoencoderKLLTX2Video(**vvae.reference_config()).eval()
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=avae.DD["ch"], output_channels=avae.DD["out_ch"], ch_mult=tuple(avae.DD["ch_mult"]),
        num_res_blocks=avae.DD["num_res_blocks"], attn_resolutions=None, in_channels=avae.DD["in_channels"],
        resolution=avae.DD["resolution"], latent_channels=avae.DD["z_channels"], norm_type="pixel",
        causality_axis="height", mel_bins=MELS,
    ).eval()
    with torch.no_grad():
        for m in (vae, audio_vae):
            for p in m.parameters():
                p.copy_((torch.randn_like(p) * 0.2).to(torch.bfloat16).to(torch.float32))
            m.latents_mean.copy_(torch.randn_like(m.latents_mean).to(torch.bfloat16).float() * 0.3)
            m.latents_std.copy_((torch.rand_like(m.latents_std) + 0.5).to(torch.bfloat16).float())

    V, B = voc.VOC, voc.BWE
    vocoder = LTX2VocoderWithBWE(
        in_channels=2 * MELS, hidden_channels=V["upsample_initial_channel"], out_channels=2,
        upsample_kernel_sizes=V["upsample_kernel_sizes"], upsample_factors=V["upsample_rates"],
        resnet_kernel_sizes=V["resblock_kernel_sizes"], resnet_dilations=V["resblock_dilation_sizes"],
        bwe_in_channels=2 * MELS, bwe_hidden_channels=B["upsample_initial_channel"], bwe_out_channels=2,
        bwe_upsample_kernel_sizes=B["upsample_kernel_sizes"], bwe_upsample_factors=B["upsample_rates"],
        bwe_resnet_kernel_sizes=B["resblock_kernel_sizes"], bwe_resnet_dilations=B["resblock_dilation_sizes"],
        filter_length=voc.NFFT, hop_length=voc.HOP, window_length=voc.NFFT, num_mel_channels=MELS,
    ).eval()
    with torch.no_grad():
        for name, p in vocoder.named_parameters():
            scale = 0.3 if name.endswith(("alpha", "beta")) else 0.15
            p.copy_((torch.randn_like(p) * scale).to(torch.bfloat16).float())
        st = vocoder.mel_stft
        st.mel_basis.copy_((torch.rand(st.mel_basis.shape) * 0.3).to(torch.bfloat16).float())

    sd = {tf.original_name(k): v.contiguous() for k, v in transformer.state_dict().items()}
    sd.update({conn.original_name(k): v.contiguous() for k, v in connectors.state_dict().items()})
    sd.update({vvae.original_name(k): v.contiguous() for k, v in vae.state_dict().items() if k.startswith("decoder.")})
    sd["vae.per_channel_statistics.mean-of-means"] = vae.latents_mean.clone()
    sd["vae.per_channel_statistics.std-of-means"] = vae.latents_std.clone()
    sd.update({"audio_vae." + k: v.contiguous() for k, v in audio_vae.state_dict().items() if k.startswith("decoder.")})
    sd["audio_vae.per_channel_statistics.mean-of-means"] = audio_vae.latents_mean.clone()
    sd["audio_vae.per_channel_statistics.std-of-means"] = audio_vae.latents_std.clone()
    sd.update({voc.original_name(k): v.contiguous() for k, v in vocoder.state_dict().items()})
    save_file(sd, os.path.join(out, "single.safetensors"), metadata={"config": json.dumps(HEADER)})

    pipe = LTX2Pipeline(scheduler=scheduler, vae=vae, audio_vae=audio_vae, text_encoder=None, tokenizer=None,
                        connectors=connectors, transformer=transformer, vocoder=vocoder)
    pipe._callback_tensor_inputs = pipe._callback_tensor_inputs + ["audio_latents"]
    return SimpleNamespace(**locals())


def main():
    out = sys.argv[1]
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=1024, base_shift=0.95, max_image_seq_len=4096, max_shift=2.05, num_train_timesteps=1000,
        shift=1.0, shift_terminal=0.1, time_shift_type="exponential", use_dynamic_shifting=True,
    )
    r = build(out, scheduler)
    pipe, gemma, transformer = r.pipe, r.gemma, r.transformer

    lf, lh, lw = (FRAMES - 1) // 8 + 1, HEIGHT // 32, WIDTH // 32
    duration = FRAMES / FPS
    audio_frames = round(duration * 16000 / 160 / 4)
    g = torch.Generator().manual_seed(3)
    video_noise = torch.randn(1, vvae.LATENT, lf, lh, lw, generator=g)
    audio_noise = torch.randn(1, avae.DD["z_channels"], audio_frames, MELS // 4, generator=g)
    pos, pos_mask = prompt_embeds(gemma, POSITIVE)
    neg, neg_mask = prompt_embeds(gemma, NEGATIVE)

    final, start = {}, {}
    forward = transformer.forward

    def first(*a, **k):
        # Given starting latents are taken as decoder-space values and
        # normalised by the reference, so record what the loop starts from.
        start.setdefault("video", k["hidden_states"][0].clone())
        start.setdefault("audio", k["audio_hidden_states"][0].clone())
        return forward(*a, **k)

    transformer.forward = first

    def grab(p, i, t, kw):
        final["video"], final["audio"] = kw["latents"].clone(), kw["audio_latents"].clone()
        return {}

    with torch.no_grad():
        video, audio = pipe(
            prompt_embeds=pos, prompt_attention_mask=pos_mask, negative_prompt_embeds=neg,
            negative_prompt_attention_mask=neg_mask, height=HEIGHT, width=WIDTH, num_frames=FRAMES, frame_rate=FPS,
            num_inference_steps=STEPS, spatio_temporal_guidance_blocks=STG_BLOCKS, latents=video_noise,
            audio_latents=audio_noise, output_type="pt", return_dict=False, max_sequence_length=SEQ,
            callback_on_step_end=grab, callback_on_step_end_tensor_inputs=["latents", "audio_latents"],
        )
    dump(out, "video_start", start["video"])
    dump(out, "audio_start", start["audio"])
    dump(out, "final_video", final["video"][0])
    dump(out, "final_audio", final["audio"][0])
    dump(out, "frames", video[0])
    dump(out, "wave", audio[0])
    json.dump({
        "positive": POSITIVE, "negative": NEGATIVE, "seq_len": SEQ, "frames": FRAMES, "height": HEIGHT,
        "width": WIDTH, "fps": FPS, "steps": STEPS, "stg_blocks": STG_BLOCKS, "audio_frames": audio_frames,
        "latent": [lf, lh, lw], "frames_shape": list(video[0].shape), "wave_shape": list(audio[0].shape),
        "sigmas": pipe.scheduler.sigmas.tolist(),
    }, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
