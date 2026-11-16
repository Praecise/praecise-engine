# FLUX 3 Image Klein: a native image pipeline

Design note, before any code. Nothing here has been built or run; every
claim about the checkpoint is marked as one to confirm against its
`model_index.json` and component configs before the port starts.

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
