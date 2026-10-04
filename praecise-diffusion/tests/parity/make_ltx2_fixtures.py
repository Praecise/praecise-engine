"""Build a small random LTX-2.3 audio-video transformer and reference outputs.

The released LTX-2.3 transformer has 22B parameters (about 46 GB in
bfloat16), far more than a parity run can hold, so this suite checks the
native transformer against the reference one on a random checkpoint with the
released LTX-2.3 layout (gated attention, prompt-side modulation, split
rotary embeddings, no caption projection) at tiny widths. Weights are rounded
to bfloat16 before anything runs, so the native loader sees exactly the
values the reference computed with.

The same weights are also written as a single file under the release's
original names, with the configuration in the header metadata in the
release's own form, so the single-file loader is checked on the same outputs.

Usage: python make_ltx2_fixtures.py <out_dir>
"""

import json
import os
import sys

import torch
from diffusers import LTX2VideoTransformer3DModel
from safetensors.torch import save_file

TINY = dict(
    in_channels=16,
    out_channels=16,
    num_attention_heads=2,
    attention_head_dim=32,
    cross_attention_dim=64,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_num_attention_heads=2,
    audio_attention_head_dim=16,
    audio_cross_attention_dim=32,
    num_layers=2,
    caption_channels=64,
    gated_attn=True,
    audio_gated_attn=True,
    cross_attn_mod=True,
    audio_cross_attn_mod=True,
    use_prompt_embeddings=False,
    use_prompt_adaln_single=True,
    rope_type="split",
)
SHAPE = {"frames": 3, "height": 4, "width": 5, "audio_frames": 7, "text_tokens": 6, "fps": 24.0}
HEADER = {
    "transformer": dict(
        activation_fn="gelu-approximate", apply_gated_attention=True, attention_bias=True, attention_head_dim=32,
        attention_type="default", audio_attention_head_dim=16, audio_cross_attention_dim=32, audio_num_attention_heads=2,
        audio_out_channels=8, audio_positional_embedding_max_pos=[20], av_ca_timestep_scale_multiplier=1000.0,
        av_cross_ada_norm=True, caption_proj_before_connector=True, causal_temporal_positioning=True,
        cross_attention_adaln=True, cross_attention_dim=64, double_self_attention=False, frequencies_precision="float64",
        in_channels=16, norm_elementwise_affine=False, norm_eps=1e-06, num_attention_heads=2, num_layers=2,
        only_cross_attention=False, out_channels=16, positional_embedding_max_pos=[20, 2048, 2048],
        positional_embedding_theta=10000.0, positional_embedding_type="rope", qk_norm="rms_norm", rope_type="split",
        share_ff=False, standardization_norm="rms_norm", timestep_scale_multiplier=1000, upcast_attention=False,
        use_audio_video_cross_attention=True, use_embeddings_connector=True, use_middle_indices_grid=True,
    ),
    "vae": dict(
        dims=3, latent_channels=128, patch_size=4, decoder_base_channels=128, causal_decoder=False,
        decoder_blocks=[["res_x", {"num_layers": 4}], ["compress_space", {"multiplier": 2}], ["res_x", {"num_layers": 6}],
                        ["compress_time", {"multiplier": 2}], ["res_x", {"num_layers": 4}], ["compress_all", {"multiplier": 1}],
                        ["res_x", {"num_layers": 2}], ["compress_all", {"multiplier": 2}], ["res_x", {"num_layers": 2}]],
    ),
    "audio_vae": {
        "model": {"params": {"ddconfig": dict(
            double_z=True, mel_bins=64, z_channels=8, resolution=256, downsample_time=False, in_channels=2, out_ch=2,
            ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0, mid_block_add_attention=False,
            norm_type="pixel", causality_axis="height")}},
        "preprocessing": {"audio": {"sampling_rate": 16000}, "stft": {"hop_length": 160}},
    },
}
WHOLE = [
    ("time_embed.", "adaln_single."),
    ("audio_time_embed.", "audio_adaln_single."),
    ("prompt_adaln.", "prompt_adaln_single."),
    ("audio_prompt_adaln.", "audio_prompt_adaln_single."),
]
RENAMES = [
    ("proj_in", "patchify_proj"),
    ("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single"),
    ("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single"),
    ("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single"),
    ("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single"),
    ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
    ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
    ("norm_q", "q_norm"),
    ("norm_k", "k_norm"),
]


def original_name(k):
    for ours, theirs in WHOLE:
        if k.startswith(ours):
            return "model.diffusion_model." + theirs + k[len(ours):]
    for ours, theirs in RENAMES:
        k = k.replace(ours, theirs)
    return "model.diffusion_model." + k


TIMES = [("high", 812.5, 812.5), ("split", 640.0, 275.0)]


def randomise(module, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if p.ndim == 1 and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def save(out, name, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    a.astype("<f4").tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(**TINY).eval()
    randomise(model, 1)
    model.save_pretrained(os.path.join(out, "checkpoint", "transformer"), safe_serialization=True)
    sd = {original_name(k): v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out, "single.safetensors"), metadata={"config": json.dumps(HEADER)})

    s = SHAPE
    g = torch.Generator().manual_seed(2)
    n_video = s["frames"] * s["height"] * s["width"]
    video = torch.randn(1, n_video, TINY["in_channels"], generator=g)
    audio = torch.randn(1, s["audio_frames"], TINY["audio_in_channels"], generator=g)
    inner = TINY["num_attention_heads"] * TINY["attention_head_dim"]
    audio_inner = TINY["audio_num_attention_heads"] * TINY["audio_attention_head_dim"]
    text = torch.randn(1, s["text_tokens"], inner, generator=g)
    audio_text = torch.randn(1, s["text_tokens"], audio_inner, generator=g)
    shapes = {}
    for name, t in [("video", video), ("audio", audio), ("text", text), ("audio_text", audio_text)]:
        shapes[name] = save(out, name, t[0])

    cases = []
    with torch.no_grad():
        for tag, tv, ta in TIMES:
            v = torch.tensor([tv])
            a = torch.tensor([ta])
            o = model(
                hidden_states=video,
                audio_hidden_states=audio,
                encoder_hidden_states=text,
                audio_encoder_hidden_states=audio_text,
                timestep=v,
                audio_timestep=a,
                sigma=v,
                audio_sigma=a,
                num_frames=s["frames"],
                height=s["height"],
                width=s["width"],
                fps=s["fps"],
                audio_num_frames=s["audio_frames"],
                use_cross_timestep=True,
                return_dict=False,
            )
            save(out, f"out_video_{tag}", o[0][0])
            save(out, f"out_audio_{tag}", o[1][0])
            cases.append({"tag": tag, "video_t": tv, "audio_t": ta})
    meta = dict(SHAPE, cases=cases, shapes=shapes)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
