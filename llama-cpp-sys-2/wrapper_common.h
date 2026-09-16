#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_rs_spec_batch;
struct llama_vocab;

struct llama_rs_grammar_trigger {
    int type;
    char * value;
    llama_token token;
};

struct llama_rs_chat_template_result {
    char * prompt;
    char * grammar;
    char * parser;
    char * generation_prompt;
    int chat_format;
    bool grammar_lazy;
    struct llama_rs_grammar_trigger * grammar_triggers;
    size_t grammar_triggers_count;
    char ** preserved_tokens;
    size_t preserved_tokens_count;
    char ** additional_stops;
    size_t additional_stops_count;
};

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

// spec_type selects the speculative draft algorithm: 0 = draft-mtp (jointly
// trained MTP head, the default), 1 = draft-dflash (block-diffusion DFlash
// drafter, e.g. Muse-Glimmer's dflash-*.gguf). Both run through the same
// common_speculative machinery.
struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    int32_t spec_type);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted);

// Speculative decoding across many sequences of one target context, as
// llama-server runs it for parallel slots: each sequence requests a draft, one
// draft call fills them all, the caller verifies every block in a single target
// decode and reports how many drafts each sequence accepted. The draft context
// must be created with n_seq_max == n_seq.
struct llama_rs_spec_batch * llama_rs_spec_batch_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    int32_t spec_type,
    uint32_t n_seq);

void llama_rs_spec_batch_free(struct llama_rs_spec_batch * spec);

llama_rs_status llama_rs_spec_batch_begin(
    struct llama_rs_spec_batch * spec,
    llama_seq_id seq_id,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_spec_batch_request(
    struct llama_rs_spec_batch * spec,
    llama_seq_id seq_id,
    int32_t n_max,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_spec_batch_draft(struct llama_rs_spec_batch * spec);

llama_rs_status llama_rs_spec_batch_result(
    struct llama_rs_spec_batch * spec,
    llama_seq_id seq_id,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_spec_batch_process(
    struct llama_rs_spec_batch * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_spec_batch_accept(
    struct llama_rs_spec_batch * spec,
    llama_seq_id seq_id,
    uint16_t n_accepted);

void llama_rs_string_free(char * ptr);

void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result);

#ifdef __cplusplus
}
#endif
