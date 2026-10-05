"""Reference stage outputs of a released Wan2.2 checkpoint for one step.

Phases run separately so each fits a small memory budget: `te` writes the
prompt states, `dit` the velocity at the first timestep after k blocks, `vae`
the autoencoder round trip of a photo. Consumed by the `wan_checkpoint_stages`
test.

Usage: python make_wan_checkpoint_stages.py <te|dit|vae> <checkpoint> <out_dir> [photo]
"""
import json, os, sys
import numpy as np, torch
phase, ck, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(6)
PROMPT = "a red car driving along a coastal road at sunset, cinematic"
W = H = 256
def save(name, t):
    t.detach().float().contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))
if phase == "te":
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(os.path.join(ck, "tokenizer"))
    te = UMT5EncoderModel.from_pretrained(os.path.join(ck, "text_encoder"), torch_dtype=torch.bfloat16).eval()
    t = tok([PROMPT], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        h = te(t.input_ids, t.attention_mask).last_hidden_state[0, :n]
    st = torch.zeros(512, h.shape[1]); st[:n] = h.float()
    save("prompt_states", st)
    json.dump(dict(prompt=PROMPT, prompt_ids=t.input_ids[0, :n].tolist(), width=W, height=H), open(os.path.join(out, "meta.json"), "w"))
    print("te done", n, float(st[:n].abs().max()))
elif phase == "dit":
    from diffusers import WanTransformer3DModel, UniPCMultistepScheduler
    dt = torch.float32 if os.environ.get("REF_F32") else torch.bfloat16
    tf = WanTransformer3DModel.from_pretrained(os.path.join(ck, "transformer"), torch_dtype=dt).eval()
    sch = UniPCMultistepScheduler.from_pretrained(os.path.join(ck, "scheduler"))
    sch.set_timesteps(30)
    t0 = sch.timesteps[0]
    ctx = torch.from_numpy(np.fromfile(os.path.join(out, "prompt_states.bin"), "<f4").reshape(1, 512, -1)).to(dt)
    noise = torch.randn(1, 48, 1, H // 16, W // 16, generator=torch.Generator().manual_seed(11))
    save("noise", noise[0])
    blocks = tf.blocks
    res = {}
    for k in [1, 2, 4, 8, 15, 30]:
        tf.blocks = blocks[:k]
        with torch.no_grad():
            v = tf(hidden_states=noise.to(dt), timestep=t0.expand(1), encoder_hidden_states=ctx).sample
        save(f"velocity_k{k}", v[0]); res[k] = float(v.float().std())
    json.dump(dict(t0=float(t0), sigmas=sch.sigmas.tolist()[:3], std=res), open(os.path.join(out, "dit.json"), "w"))
    print("dit done", float(t0), res)
elif phase == "vae":
    from diffusers import AutoencoderKLWan
    from PIL import Image
    vae = AutoencoderKLWan.from_pretrained(os.path.join(ck, "vae"), torch_dtype=torch.float32).eval()
    img = Image.open(sys.argv[4]).convert("RGB").resize((W, H))
    np.asarray(img).astype(np.uint8).tofile(os.path.join(out, "image.rgb"))
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None]
    mean = torch.tensor(vae.config.latents_mean).view(1, 48, 1, 1, 1); std = torch.tensor(vae.config.latents_std).view(1, 48, 1, 1, 1)
    with torch.no_grad():
        mu = vae.encode(x).latent_dist.mode()
        z = (mu - mean) / std
        y = vae.decode(z * std + mean).sample
    save("vae_latents", z[0]); save("vae_decoded", y[0])
    print("vae done", float((y - x).abs().mean()), float(z.std()))
