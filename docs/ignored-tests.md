# Ignored tests

Every `#[ignore]` test in the engine crates, by file. Each needs something the default
`cargo test` run does not have (a model, fixtures, or the hardware being measured), so it runs
under a named gate job instead. Regenerate with the script in this file's git history; keep it
in step when a test is added or removed.

161 tests in 57 files.

## Named jobs

- **models** (x86 build host, `PRAECISE_TEST_KV_MODEL`, `PRAECISE_TEST_HYBRID_MODEL`, a small encoder-decoder GGUF):
  `cargo test --locked -p praecise-runtime --features bundled-llama --release --test <name> -- --ignored`
  for `kv_migration`, `batch_migration`, `batch_scoring`, `batch_yield`, `batch_limits`,
  `batch_encoder_decoder`, `batch_prefix_reuse`. Run on every change to `batching.rs`, `toploc.rs` or the migration code.
- **throughput** (`batch_throughput`, a timing): the same command with `PRAECISE_TEST_THROUGHPUT_MODEL`
  and `_CONCURRENCY`; `_DRAFT=mtp:<n>` loads the MTP head.
- **flux3-image** (`PRAECISE_FLUX3_IMAGE_PARITY`): fixtures from `praecise-diffusion/tests/parity/make_flux3_image_fixtures.py`;
  `cargo test -p praecise-diffusion --lib flux3::image::parity -- --ignored`.
- **Not yet under a job** (a fix row with an owner, not a waiver): every other parity test below, which
  needs a reference fixture set or released weights; each is run by generating its fixtures with its own
  `tests/parity` script and setting the variable named in its `ignore` reason.

## `praecise-diffusion/src/cosmos3/parity.rs` (1)
- `cosmos3_mot_parity`: needs PRAECISE_COSMOS3_MOT_PARITY fixtures

## `praecise-diffusion/src/flux3/image/parity.rs` (3)
- `flux3_image_parity_f32`: needs the reference fixtures
- `flux3_image_parity_bf16`: needs the reference fixtures
- `flux3_image_time_grid_is_the_reference_schedulers`: needs the reference fixtures

## `praecise-diffusion/src/flux3/packing.rs` (1)
- `packing_parity`: needs the reference fixtures

## `praecise-diffusion/src/flux3/parity.rs` (2)
- `flux3_parity_f32`: needs the reference fixtures
- `flux3_parity_bf16`: needs the reference fixtures

## `praecise-diffusion/src/flux3/policy/parity.rs` (2)
- `flux3_policy_parity_f32`: needs PRAECISE_FLUX3_POLICY_PARITY fixtures
- `flux3_policy_parity_bf16`: needs PRAECISE_FLUX3_POLICY_PARITY fixtures

## `praecise-diffusion/src/flux3/release/tests.rs` (2)
- `flux3_rollout_parity`: needs reference fixtures
- `flux3_release_real`: needs released weights

## `praecise-diffusion/src/flux3/text_encoder.rs` (2)
- `flux3_text_parity_f32`: needs PRAECISE_FLUX3_TEXT_PARITY fixtures
- `flux3_text_parity_bf16`: needs PRAECISE_FLUX3_TEXT_PARITY fixtures

## `praecise-diffusion/src/flux3/video_vae.rs` (2)
- `flux3_vae_parity_f32`: needs PRAECISE_FLUX3_VAE_PARITY fixtures
- `flux3_vae_parity_bf16`: needs PRAECISE_FLUX3_VAE_PARITY fixtures

## `praecise-diffusion/src/gemma3.rs` (2)
- `gemma3_parity_f32`: needs the reference fixtures
- `gemma3_parity_bf16`: needs the reference fixtures

## `praecise-diffusion/src/ggml.rs` (2)
- `bench_attention`: a timing, run on the hardware being measured
- `bench_linear`: a timing, run on the hardware being measured

## `praecise-diffusion/src/ltx2/parity.rs` (18)
- `ltx2_parity_transformer_f32`: needs the reference fixtures
- `ltx2_parity_transformer_bf16`: needs the reference fixtures
- `ltx2_parity_transformer_single_file_f32`: needs the reference fixtures
- `ltx2_parity_connectors_f32`: needs the reference fixtures
- `ltx2_parity_connectors_single_file_f32`: needs the reference fixtures
- `ltx2_parity_connectors_bf16`: needs the reference fixtures
- `ltx2_parity_video_decoder_f32`: needs the reference fixtures
- `ltx2_parity_video_decoder_tiled_f32`: needs the reference fixtures
- `ltx2_parity_video_decoder_bf16`: needs the reference fixtures
- `ltx2_parity_audio_decoder_f32`: needs the reference fixtures
- `ltx2_parity_audio_decoder_bf16`: needs the reference fixtures
- `ltx2_parity_vocoder_f32`: needs the reference fixtures
- `ltx2_parity_vocoder_bf16`: needs the reference fixtures
- `ltx2_parity_pipeline_f32`: needs the reference fixtures
- `ltx2_parity_pipeline_bf16`: needs the reference fixtures
- `ltx2_parity_latent_upsampler_f32`: needs the reference fixtures
- `ltx2_parity_distilled_two_stage_f32`: needs the reference fixtures
- `ltx2_parity_distilled_two_stage_bf16`: needs the reference fixtures

## `praecise-diffusion/src/matrix_game/parity.rs` (3)
- `world_dit_parity`: needs PRAECISE_MG3_PARITY fixtures
- `camera_rays_match_reference`: needs PRAECISE_MG3_PARITY fixtures
- `world_session_parity`: needs PRAECISE_MG3_SESSION fixtures

## `praecise-diffusion/src/minimax_h3/audio_vae.rs` (2)
- `minimax_h3_parity_audio_vae`: needs the reference fixtures
- `minimax_h3_parity_audio_vae_single_file`: needs the reference fixtures

## `praecise-diffusion/src/minimax_h3/pipeline.rs` (9)
- `minimax_h3_parity_fl2va_f32`: needs the reference fixtures
- `minimax_h3_parity_fl2va_bf16`: needs the reference fixtures
- `minimax_h3_parity_ref2va_f32`: needs the reference fixtures
- `minimax_h3_parity_ref2va_bf16`: needs the reference fixtures
- `minimax_h3_parity_presentation`: needs the reference fixtures
- `minimax_h3_parity_lanczos`: needs the reference fixtures
- `minimax_h3_parity_pipeline_f32`: needs the reference fixtures
- `minimax_h3_parity_pipeline_bf16`: needs the reference fixtures
- `minimax_h3_parity_pipeline_q8_0`: needs the reference fixtures

## `praecise-diffusion/src/minimax_h3.rs` (4)
- `minimax_h3_parity_single_file_gguf`: needs the reference fixtures
- `minimax_h3_parity_single_file_f8`: needs the reference fixtures
- `minimax_h3_parity_transformer_f32`: needs the reference fixtures
- `minimax_h3_parity_transformer_bf16`: needs the reference fixtures

## `praecise-diffusion/src/minimax_h3/vae.rs` (3)
- `minimax_h3_parity_video_vae_f32`: needs the reference fixtures
- `minimax_h3_parity_video_vae_bf16`: needs the reference fixtures
- `minimax_h3_parity_video_vae_single_file`: needs the reference fixtures

## `praecise-diffusion/src/music/parity.rs` (7)
- `acestep_parity_tokens_match_the_reference_tokenizer`: needs the reference fixtures
- `acestep_parity_text_and_condition_encoders`: needs the reference fixtures
- `acestep_parity_one_transformer_step`: needs the reference fixtures
- `acestep_parity_decoder`: needs the reference fixtures
- `acestep_parity_whole_pipeline`: needs the reference fixtures
- `acestep_parity_the_clip_sounds_as_long_as_the_reference`: needs the reference fixtures
- `acestep_parity_tiled_decode_matches_one_pass`: needs the reference fixtures

## `praecise-diffusion/src/pipeline/parity.rs` (8)
- `parity_prompt_tokens_match_the_reference_tokenizer`: needs fixtures from tests/parity/make_fixtures.py
- `parity_prompt_embedding_matches_the_reference_encoder`: needs fixtures from tests/parity/make_fixtures.py
- `parity_one_transformer_evaluation_matches_the_reference`: needs fixtures from tests/parity/make_fixtures.py
- `parity_decode_matches_the_reference_autoencoder`: needs fixtures from tests/parity/make_fixtures.py
- `parity_whole_pipeline_matches_the_reference_image`: needs fixtures from tests/parity/make_fixtures.py
- `parity_reference_conditioned_generation_matches_the_reference_image`: needs fixtures from tests/parity/make_fixtures.py
- `every_layer_norm_is_followed_by_its_scale_and_shift`: needs fixtures from tests/parity/make_fixtures.py
- `every_group_norm_is_followed_by_its_scale_and_shift`: needs fixtures from tests/parity/make_fixtures.py

## `praecise-diffusion/src/qwen3_vl.rs` (3)
- `qwen3_vl_parity_flat_gguf`: needs the reference fixtures
- `qwen3_vl_parity_f32`: needs the reference fixtures
- `qwen3_vl_parity_bf16`: needs the reference fixtures

## `praecise-diffusion/src/qwen_image21/parity.rs` (4)
- `qwen_image21_parity_transformer_f32`: needs the reference fixtures
- `qwen_image21_parity_transformer_bf16`: needs the reference fixtures
- `qwen_image21_parity_vae_f32`: needs the reference fixtures
- `qwen_image21_parity_vae_f16`: needs the reference fixtures

## `praecise-diffusion/src/qwen_image21/pipeline.rs` (2)
- `qwen_image21_pipeline_parity_f32`: needs the reference fixtures
- `qwen_image21_pipeline_parity_bf16`: needs the reference fixtures

## `praecise-diffusion/src/qwen_image/parity.rs` (4)
- `qwen_image_parity_transformer_f32`: needs the reference fixtures
- `qwen_image_parity_transformer_bf16`: needs the reference fixtures
- `qwen_image_parity_vae_f32`: needs the reference fixtures
- `qwen_image_parity_vae_f16`: needs the reference fixtures

## `praecise-diffusion/src/qwen_image/pipeline.rs` (2)
- `qwen_image_edit_parity_f32`: needs the reference fixtures
- `qwen_image_edit_parity_bf16`: needs the reference fixtures

## `praecise-diffusion/src/qwen_vl.rs` (2)
- `qwen_vl_parity_f32`: needs the reference fixtures
- `qwen_vl_parity_bf16`: needs the reference fixtures

## `praecise-diffusion/src/trellis2/parity.rs` (7)
- `trellis2_parity_flow_dense`: needs fixtures
- `trellis2_parity_sampler_dense`: needs fixtures
- `trellis2_parity_flow_sparse_and_sampler`: needs fixtures
- `trellis2_parity_occupancy`: needs fixtures
- `trellis2_parity_decoders_and_surface`: needs fixtures
- `trellis2_parity_image_encoder`: needs fixtures
- `trellis2_parity_image_preparation`: needs fixtures

## `praecise-diffusion/src/umt5/parity.rs` (1)
- `umt5_parity`: needs PRAECISE_UMT5_PARITY fixtures

## `praecise-diffusion/src/video/action_parity.rs` (2)
- `cosmos3_action_parity`: needs PRAECISE_COSMOS3_ACTION_PARITY fixtures
- `cosmos3_older_layout_action_parity`: needs PRAECISE_COSMOS3_OLDER_ACTION_PARITY fixtures

## `praecise-diffusion/src/video/parity.rs` (6)
- `cosmos3_parity_tokens_match_the_reference_tokenizer`: needs the reference fixtures
- `cosmos3_parity_encoder`: needs the reference fixtures
- `cosmos3_parity_one_transformer_step`: needs the reference fixtures
- `cosmos3_parity_decoder`: needs the reference fixtures
- `cosmos3_parity_whole_pipeline`: needs the reference fixtures
- `cosmos3_older_layout_parity`: needs PRAECISE_COSMOS3_OLDER_PARITY fixtures

## `praecise-diffusion/src/vla/tests.rs` (8)
- `smolvla_rollout_matches_reference`: needs a released policy and a reference fixture
- `pi05_rollout_matches_reference`: needs a released policy and a reference fixture
- `groot_matches_reference`: needs a released policy and a reference fixture
- `molmoact2_vision_matches_reference`: needs a released policy and a reference fixture
- `molmoact2_prefix_matches_reference`: needs a reference fixture
- `molmoact2_actions_match_reference`: needs a reference fixture
- `molmoact2_rollout_matches_reference`: needs a reference fixture
- `groot_rollout_matches_reference`: needs a released policy and a reference fixture

## `praecise-diffusion/src/wan_dit/parity.rs` (1)
- `wan_dit_parity`: needs PRAECISE_WAN_DIT_PARITY fixtures

## `praecise-diffusion/src/wan_dit.rs` (1)
- `bench_wan_dit_step`: a timing, run on the hardware being measured

## `praecise-diffusion/src/wan_video/parity.rs` (3)
- `wan_pipeline_parity`: needs PRAECISE_WAN_PIPELINE_PARITY fixtures
- `wan_session_matches_generation`: needs PRAECISE_WAN_PIPELINE_PARITY fixtures
- `wan_checkpoint_stages`: needs a released checkpoint and its reference stage outputs

## `praecise-diffusion/src/zimage/parity.rs` (4)
- `zimage_parity_tokens_and_caption_features`: needs the reference fixtures
- `zimage_parity_one_transformer_step`: needs the reference fixtures
- `zimage_parity_decoder`: needs the reference fixtures
- `zimage_parity_whole_pipeline`: needs the reference fixtures

## `praecise-onnx/src/forecast.rs` (2)
- `t0_matches_the_reference_runtime`: needs the T0 graph and its reference output: PRAECISE_T0_ONNX, PRAECISE_T0_EXPECTED
- `chronos2_reads_a_quantile_major_output`: needs the Chronos-2 graph: PRAECISE_CHRONOS2_ONNX

## `praecise-onnx/src/text_embedding.rs` (1)
- `an_onnx_encoder_matches_its_source_checkpoint`: needs an encoder and its reference vectors: PRAECISE_ENCODER_*

## `praecise-onnx/tests/families.rs` (5)
- `detection_rf_detr_returns_well_formed_boxes`: needs RF-DETR nano: PRAECISE_DETR_ONNX
- `segmentation_edgesam_segments_the_prompted_square`: needs EdgeSAM: PRAECISE_SAM_ENCODER, PRAECISE_SAM_DECODER
- `vision_dinov3_embeds_consistently`: needs DINOv3 ViT-S/16: PRAECISE_IMAGE_ENCODER_ONNX
- `text_embedding_modernbert_ranks_the_relevant_document_first`: needs ModernBERT-embed base: PRAECISE_TEXT_ENCODER_ONNX, PRAECISE_TEXT_TOKENIZER
- `audio_moonshine_transcribes_a_wav`: needs Moonshine tiny: PRAECISE_MOONSHINE_DIR

## `praecise-runtime/src/decision.rs` (1)
- `the_real_laya_model_decides`: needs LAYA_GGUF=<path to Laya-Q8_0.gguf>

## `praecise-runtime/tests/batch_encoder_decoder.rs` (1)
- `an_encoder_decoder_model_is_served_from_one_slot`: needs a GGUF: PRAECISE_TEST_ENCODER_DECODER_MODEL

## `praecise-runtime/tests/batch_limits.rs` (1)
- `two_engines_of_one_process_run_with_their_own_slot_counts`: needs a GGUF: PRAECISE_TEST_KV_MODEL

## `praecise-runtime/tests/batch_migration.rs` (2)
- `a_prompt_checkpoint_is_exported_when_the_first_token_ends_the_request_or_it_is_still_queued`: needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL
- `a_running_sequence_moves_to_another_engine_and_continues_where_it_stopped`: needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL

## `praecise-runtime/tests/batch_scoring.rs` (1)
- `a_batch_commitment_verifies_against_a_continuation_scored_on_the_batch_engine`: needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL

## `praecise-runtime/tests/batch_throughput.rs` (1)
- `decode_throughput_by_concurrency`: needs a model: PRAECISE_TEST_THROUGHPUT_MODEL

## `praecise-runtime/tests/batch_yield.rs` (1)
- `a_yielded_request_returns_its_partial_output_with_a_commitment_that_verifies`: needs a GGUF: PRAECISE_TEST_KV_MODEL or PRAECISE_TEST_HYBRID_MODEL

## `praecise-runtime/tests/chat_model_turns.rs` (3)
- `same_tokens_beside_other_requests`: needs chat models: PRAECISE_TEST_CHAT_MODELS
- `same_tokens_however_delivered`: needs chat models: PRAECISE_TEST_CHAT_MODELS
- `tool_calls_and_reasoning_from_real_models`: needs chat models: PRAECISE_TEST_CHAT_MODELS

## `praecise-runtime/tests/denoise_engine.rs` (2)
- `tiny_models_decode_as_the_reference_decoder_does`: needs PRAECISE_DENOISE_FIXTURES: tiny diffusion GGUFs and the reference decoder's answers
- `a_served_model_answers`: needs PRAECISE_DENOISE_MODEL: a real diffusion language model

## `praecise-runtime/tests/embed_media_parity.rs` (1)
- `pooled_embeddings_match_the_source_checkpoint`: needs the embedder weights and reference: PRAECISE_EMBED_GGUF, PRAECISE_EMBED_MMPROJ and PRAECISE_EMBED_MEDIA

## `praecise-runtime/tests/hunyuan_image3_parity.rs` (1)
- `hunyuan_image3_matches_the_reference_model`: needs the converted tiny checkpoint: PRAECISE_HY3_GGUF and PRAECISE_HY3_FIXTURES

## `praecise-runtime/tests/janus_image_parity.rs` (1)
- `janus_image_matches_the_source_checkpoint`: needs the Janus-Pro weights: PRAECISE_JANUS_GGUF and PRAECISE_JANUS_DIR

## `praecise-runtime/tests/kv_migration.rs` (4)
- `export_import_resumes_with_identical_logits_and_deltas_carry_only_new_positions`: needs a small dense GGUF: PRAECISE_TEST_KV_MODEL
- `a_blob_from_other_weights_is_refused`: needs a small dense GGUF: PRAECISE_TEST_KV_MODEL
- `a_sequence_moves_whole_between_contexts_of_any_stream_count`: needs a GGUF: PRAECISE_TEST_KV_MODEL, PRAECISE_TEST_HYBRID_MODEL or PRAECISE_TEST_RECURRENT_MODEL
- `a_batch_slot_moves_to_another_slot_of_another_context_with_identical_logits`: needs a GGUF: PRAECISE_TEST_KV_MODEL, PRAECISE_TEST_HYBRID_MODEL or PRAECISE_TEST_RECURRENT_MODEL

## `praecise-runtime/tests/layer_pipeline.rs` (3)
- `stage_process`: spawned by three_processes_match_one_process
- `three_processes_match_one_process`: needs a small dense GGUF: PRAECISE_TEST_KV_MODEL
- `remote_stage_matches_one_process`: needs a second host: PRAECISE_PIPELINE_TEST_REMOTE

## `praecise-runtime/tests/long_context.rs` (1)
- `needle_is_recalled_at_depth`: needs a long-context model: PRAECISE_TEST_LONG_MODEL, PRAECISE_TEST_LONG_CTX

## `praecise-runtime/tests/omni_speech_parity.rs` (1)
- `talker_waveform_matches_the_source_checkpoint`: needs the talker weights: PRAECISE_OMNI_TALKER_GGUF and PRAECISE_OMNI_TALKER_MMPROJ

## `praecise-runtime/tests/rerank_media_parity.rs` (1)
- `multimodal_reranker_scores_match_the_source_checkpoint`: needs the reranker weights: PRAECISE_RERANK_GGUF and PRAECISE_RERANK_MMPROJ

## `praecise-runtime/tests/rerank_parity.rs` (1)
- `reranker_scores_match_the_source_checkpoint`: needs a reranker GGUF and its reference scores: PRAECISE_RERANK_GGUF and PRAECISE_RERANK_EXPECTED

## `praecise-runtime/tests/staging_activation.rs` (1)
- `staged_activation_is_faster_than_a_cold_load`: needs a small dense GGUF: PRAECISE_TEST_KV_MODEL

## `praecise-train/tests/real_model.rs` (1)
- `lora_sft_on_a_real_model_and_served_logprobs_match`: needs PRAECISE_TRAIN_MODEL

## `praecise-train/tests/round.rs` (1)
- `rounds_replay_bitwise_and_resume_from_a_saved_state`: needs PRAECISE_TRAIN_MODEL
