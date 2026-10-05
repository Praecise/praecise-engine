# TODO: image and video input for reranking and embedding

Status: reranking and pooled embedding done. `rerank::rerank_media` scores
pairs whose sides carry pictures and video frames through the projector
(items 1, 2, 4 and 7 below for rerankers); the caller sizes pictures and
samples frames by the checkpoint's own rule (items 5 and 6) and writes each
side with media markers. Against the Qwen3-VL Reranker 2B pipeline, with the
text model in F16 and the projector from the same converter commit, every
pair is within 0.00036 of the checkpoint's P(yes) (15 pairs,
`tests/rerank_media_parity.rs`); in Q8_0 the text model alone moves scores
by up to 0.055.

`embed::embed_media` (item 3) pools one vector for an input carrying text,
pictures and video frames together: the checkpoint's chat template with the
instruction as the system turn, the end-of-text token the tokenizer appends,
LAST pooling, L2 normalisation; `embed::embed` is the same for text alone.
Against the Qwen3-VL Embedding 2B pipeline (transformers 5.18, float32;
`tests/reference/embed_media.py`), F16 text model and projector, on the CPU,
the token counts are identical and every case is above 0.999998
(`tests/embed_media_parity.rs`, which requires 0.99999):

| input | tokens | cosine |
| --- | ---: | ---: |
| text | 27 | 0.9999998 |
| text with an instruction | 25 | 0.9999993 |
| empty input | 21 | 0.9999999 |
| 256x192 picture | 70 | 0.9999982 |
| picture and text | 74 | 0.9999993 |
| 32-frame video | 2396 | 0.9999995 |
| video, picture and text | 2454 | 0.9999994 |
| 1280x960 picture | 1222 | 0.9999990 |
| 64-frame video | 4780 | 0.9999993 |

Long visual inputs used to drift (0.99983 for the 1280x960 picture, 0.99853
for the 64-frame video). The cause was the CPU flash-attention kernel: a
chunk of a few queries over many keys (the text decoded after a picture or a
video, which carries the pooled position) went through its per-row path,
which summed the weighted values in half precision and so dropped the small
terms of a long row; the decoder fed the reference's own vision features
showed the same drift, and none without flash attention. That path now sums
in FP32, as the tiled path for long query chunks already did. Two smaller
differences were removed with it: the Qwen3-VL patch mergers use the exact
(erf) GELU, as the checkpoint does, and image tokens carry the temporal
position in the fourth rotary section, which interleaved M-RoPE reads for
its last pairs. The patch convolution now runs in F32: `ggml_conv_2d`
unfolded its input in half precision whatever the kernel type, so the pixels
reached the patch embedding rounded to F16, and the vision tower amplifies
that on some pictures (a 288x192 picture went from 0.99987 to 0.999998).

## Where things stand

- **Conversion.** In the converter (the llama.cpp fork the engine links,
  `convert_hf_to_gguf.py`), `conversion/qwen.py` recognises a vision-language reranker
  by its README heading or directory name and writes RANK pooling, the yes/no
  classification rows and the rerank prompt template.
  `conversion/qwen3vl.py` (`Qwen3VLTextModel`) writes the text decoder only;
  the vision tower belongs in a separate projector file (`mmproj`), which is
  not produced or tested for reranker or embedder checkpoints.
- **Reranking.** `praecise-runtime/src/rerank.rs` tokenizes each
  query/document pair as text (`pair_tokens`), decodes it in a fresh
  sequence and reads the classification output. There is no way to pass an
  image or a video as the query or the document.
- **Embedding.** There is no pooled-embedding entry point for GGUF models:
  `memory_traits` recognises a pooled model, but nothing reads a pooled
  vector back out. Vision-language embedders are therefore served from a
  text-only export outside the engine.
- **Multimodal plumbing.** `batching.rs` already evaluates `mtmd` chunks
  (`MtmdInputChunks::eval_chunk`) for generation, behind the `mtmd`
  feature. That is the path to reuse; it is not wired into rerank or embed.

## Work items

1. **Projector export.** Produce the `mmproj` file for reranker and embedder
   checkpoints from the same converter commit as the text GGUF, and confirm
   it is byte-reproducible (same checkpoint, same converter, same toolchain:
   identical bytes across runs).
2. **Multimodal rerank.** Accept a query and documents that each carry text,
   images or video. Build each pair from the checkpoint's rerank template
   with the media markers in place, tokenize through `mtmd`, evaluate the
   chunks into a fresh sequence, and read the RANK output as today. Keep the
   per-pair isolation: a score must not depend on the other documents in
   the request.
3. **Pooled embedding.** Add an embed entry point that evaluates text or
   `mtmd` chunks, reads the pooled vector (LAST pooling for decoder
   embedders), and L2-normalises it. Instructions are applied through the
   checkpoint's chat template (a system instruction, the input as the user
   turn, the generation prompt appended), exactly as the checkpoint's own
   pipeline formats them.
4. **Positions.** These decoders use multi-axis rotary positions. Text-only
   input gives identical positions on every axis; image and video chunks do
   not. Position assignment must come from the `mtmd` chunk evaluation, never
   from a flat counter.
5. **Image preprocessing.** Resizing (to multiples of patch size times merge
   size, within the checkpoint's minimum and maximum pixel budget) and
   normalisation must match the checkpoint's processor configuration, or
   embeddings drift without any error. Read the limits from the checkpoint,
   not from constants.
6. **Video.** Frame sampling (rate, maximum frame count, how the final frame
   is kept) and per-frame timestamps follow the checkpoint's video processor
   configuration. Sampling must be deterministic so two runs on the same
   file give the same embedding and the same score.
7. **Parity.** For each modality (text, image, image plus text, video):
   reference scores and embeddings from the checkpoint's own pipeline at a
   pinned revision; pass when the reranker's P(yes) matches within 1e-2 and
   the embedding cosine exceeds 0.999. Text-only parity must stay unchanged.

## Text format of vision-language rerankers (done)

The converter now writes the vision-language reranker's own `rerank`
template: `<Instruct>: {instruction}<Query>:{query}\n<Document>:{document}`
in the user turn, the checkpoint's default instruction ("Given a search
query, retrieve relevant candidates that answer the query."), and an
assistant turn without a think block. It is byte-equal to the checkpoint's
chat template rendered for one text pair. The text-only reranker template is
unchanged.

## Out of scope

Audio input, and training or fine-tuning of the vision tower.
