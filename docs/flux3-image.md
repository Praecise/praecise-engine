# FLUX 3 Image Klein: a native image pipeline

Built as `praecise_diffusion::flux3::image::Flux3Image`. The weights are
gated, so it has been checked against the public reference pipeline
(diffusers 0.42, `Flux3Transformer2DModel` and the FLUX 3 modular blocks)
on tiny random transformers, not yet on the release itself. The design as
first written is kept below; "Settled" records what the reference answered.

## Settled

- Layout: the release ships the native transformer (`model.safetensors`,
  the tensor names `flux3.rs` already reads, and `variants/fp8r`,
  `variants/nvfp4`), the FLUX.2 autoencoder under `vae/`, and Qwen3-VL with
  its tokenizer under `text_encoder/`. The transformer has two streams,
  `image` and `image_cond`, and no global vector.
- Flow map: the distilled model adds `ref_time_in`, an embedding of the
  time the step jumps to, to every token's time embedding. Its update is
  the ordinary Euler step; no new solver.
- Schedule: Euler on `linspace(1, 1/N, N)` shifted by 6.93 (then 0); 16
  steps distilled, 50 at guidance 3.5 against the empty prompt for the base.
- Positions: the latent grid at half steps on the rotary `h` and `w` axes;
  reference `k` at `l = -(1 + k)`; text along `l`. The text encoder gives
  padding tokens position 1.
- Autoencoder: the FLUX.2 one, shared with `Flux2Klein`.
- Reference resizing and prompt upsampling are the caller's.

Not checked: the release's own outputs (needs the gated weights), and the
`fp8r` scale layout, read as one scale per tensor or per row.

## What is wanted

FLUX 3 Image Klein (5.6B, distilled, 16 flow-map steps without guidance) and
FLUX 3 Image Klein Base (5.6B, undistilled, 50 steps at guidance 3.5), each in
bfloat16 and the distilled one also as float8 and NVFP4 transformer
variants. Text-to-image and editing from up to ten reference images, at up
to 4 MP. The text encoder is Qwen3-VL-4B-Instruct. Bounding boxes for
placed elements are written into the prompt on a 0-1000 grid, so they are
prompt text, not a separate input.

The pipeline takes the interface of the FLUX.2 [klein] one
(`praecise_diffusion::pipeline`): `CheckpointFiles` for the diffusers
layout, `LoadOptions`, a `Request` of prompt, references, size, steps,
guidance scale and seed, and an `Image` with per-stage `Timings`. A host
that serves FLUX.2 [klein] serves FLUX 3 Image Klein by choosing the
pipeline type and nothing else.

## What exists to build on

- `flux3.rs`: the FLUX 3 multi-stream flow transformer, built for the
  action models. Per-token timesteps, per-stream parallel blocks, shared
  single-stream blocks with joint attention over the text and every
  active stream, per-stream output heads, configuration read from the
  checkpoint. An image model is the case of one target stream (the image
  latents) and, for editing, one conditioning stream (the reference
  latents, clean, at timestep zero). To confirm: that the image
  checkpoint uses this block layout and its tensor names, and what its
  stream set is.
- `flux3/text_encoder.rs`: FLUX 3 text context from a Qwen3-VL language
  model (chat-template wrap, fixed padding, hidden states of every fourth
  layer stacked per token). To confirm: that the image model reads the
  same layers and padding as the action models.
- `flux3/sampling.rs`: rectified-flow Euler and bh2 UniPC on a shifted
  linear schedule, and per-stream guidance. The base model's 50 guided
  steps fit these. The distilled model's flow-map steps do not: a flow
  map predicts a jump between two times rather than a velocity at one, so
  its update is new code (one function of the current latents, the model
  output and the two times). To confirm from the vendor's
  `Flux3DistilledModularPipeline`: the time grid, the update rule, and
  whether the model takes both times as conditioning.
- `flux3/video_vae.rs`: the FLUX 3 video autoencoder (hierarchical 3D
  neighbourhood-attention transformer). To confirm: whether the image
  model uses it on single frames or ships its own image autoencoder. On
  the first, an image is a one-frame video and the encoder path for
  references is the same code; on the second, a new autoencoder module.
- `pipeline.rs` (FLUX.2 [klein]): every weight resident once on one
  backend, refusing a CPU run on a host whose GPU the build cannot drive,
  reference-image resizing and the 16-pixel grid, and parity tests
  against fixtures from the reference implementation.

## Shape

A `Flux3Image` type beside `Flux2Klein`, in a new `flux3/image.rs`:

1. Load: the transformer (bf16, or the float8 and NVFP4 variants through
   the existing quantised linear kinds), the Qwen3-VL-4B text decoder, the
   autoencoder. `is_distilled()` from the checkpoint, as FLUX.2 does.
2. Encode the prompt once; encode each reference to latents once.
3. Pack: image latents as the target stream with positions on the image
   grid; references as conditioning tokens at time zero, each offset on the
   axis that keeps them apart (FLUX.2 uses a per-reference time index; to
   confirm for FLUX 3).
4. Denoise: the flow-map sampler for the distilled model, Euler/UniPC with
   guidance for the base model. Unconditional prediction only under
   guidance, as `Image::evaluations` already counts.
5. Decode the target stream to RGB.

## Staging and what each stage must show

1. Transformer parity on fixtures from the reference implementation at
   tiny widths, in the existing `flux3/parity.rs` style (one image stream,
   one conditioning stream).
2. Autoencoder round trip and parity on a fixed image.
3. Sampler parity: the flow-map update step for step against the vendor
   pipeline on recorded tensors.
4. End to end on the bf16 checkpoint at 1024x1024 against the vendor
   pipeline's output for fixed seeds: a perceptual distance bound, the
   same way the FLUX.2 port was accepted.
5. The float8 and NVFP4 variants against bf16.

Memory: the vendor reports 31 GB peak at 1024x1024. The FLUX.2 port's
resident-once layout should come in under that; measure it, do not quote
it.

## Open questions

- Whether the distilled model's flow-map time grid is fixed or a function
  of resolution (the action models shift their schedule by token count).
- Whether references and the target share the autoencoder's latent scale
  and shift.
- How the prompt upsampler interacts: the card recommends full scene
  captions written by a separate model, which callers run themselves; the
  pipeline takes the prompt as given.

## Release reference code against diffusers

BFL's own inference code (github.com/black-forest-labs/flux-3-image-klein,
`flux3/pipeline.py`, `sampling.py`, `dit.py`) was read against this
pipeline. The DiT, the text context (Qwen3-VL-4B, layers 4 to 32 every
fourth, padded to a multiple of 80, padding at rotary position 1), the
half-step image positions, the reference `l` coordinates and the
autoencoder agree. Differences, and what was done:

- Time shift: the release code uses `alpha = 16.0` for both models; the
  diffusers modular pipeline defaults to 6.93. Base guidance: 4.0 in the
  release code and its model card, 3.5 in diffusers. `Recipe` holds both
  sets; `Recipe::REFERENCE` (the release code) is the default and
  `Recipe::DIFFUSERS` reproduces the earlier numbers. Only the release
  weights can settle which the model was trained with:
  `flux3_image_release_parity` and the example prompts and seeds in BFL's
  `examples/` are the check.
- bfloat16 inputs: the release code starts from noise rounded to bfloat16
  and hands the transformer bfloat16 timesteps, so its sinusoid sees
  `bf16(1000 * bf16(t))`. `Recipe::bf16_inputs` does the same.
- Prompts longer than 4096 tokens are truncated (the encoder's own limit is
  8192).
- Reference preparation is in the release's pipeline, not the model:
  `flux3::image::prepare` ports it (single reference kept whole on a padded
  canvas under 4 MiP, Lanczos, reflect padding, cropped back; several
  references under 1 MiP, centre-cropped to the 16-pixel grid), and
  `Flux3Image::edit` applies it.
- torch's CPU `randn` is not reproduced: a seed gives different noise than
  the reference's. Parity runs hand the reference's noise over.
- Not here: the prompt upsampler (a separate Qwen3.5 model the release asks
  callers to run), the safety classifier, and fp8 and NVFP4 compute (the
  quantised transformers load and are widened to the load precision).
- Layout: the transformer is the root `model.safetensors`, read by
  `Flux3Image::load_release`, which takes a quantised entry's
  `variants/<fp8r|nvfp4>/model.safetensors` when the root file is absent.
  The diffusers `transformer/` shards are not read.
