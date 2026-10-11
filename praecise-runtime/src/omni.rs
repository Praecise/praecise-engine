//! A spoken reply from an omni model: the reading model (thinker) reads the
//! conversation, text and audio, and writes its reply; the talker speaks the
//! reply while it is being written. Both run in process on the same backend.
//!
//! The talker is conditioned on the user turns the thinker read: their
//! tokens, and the thinker's inner hidden state at each audio position (the
//! layer the talker was trained on, captured while the prompt is read). It
//! asks for reply text only when it has spoken everything it was given, so
//! the first audio follows the first reply token and the talker's own prompt,
//! not the whole reply.

use std::ops::ControlFlow;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdInputChunkType, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::error::{Error, Result};
use crate::speech::{speak_reply, ReplyRequest, ReplyText, Speech};

fn inference(what: impl std::fmt::Display) -> Error {
    Error::Inference(what.to_string())
}

/// How the thinker samples its reply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplySampling {
    /// Temperature; 0 picks the most likely token.
    pub temperature: f32,
    /// Top-k.
    pub top_k: i32,
    /// Top-p.
    pub top_p: f32,
    /// Seed.
    pub seed: u32,
}

/// How the talker speaks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceSampling<'a> {
    /// Voice name.
    pub voice: &'a str,
    /// Upper bound on frames.
    pub max_frames: usize,
    /// Temperature for the backbone's codes.
    pub temperature: f32,
    /// Top-k for the backbone's codes and the projector.
    pub top_k: i32,
    /// Top-p for the backbone's codes.
    pub top_p: f32,
    /// Top-p for the projector's codes.
    pub projector_top_p: f32,
    /// Repetition penalty over the backbone's codes.
    pub repetition_penalty: f32,
    /// Seed.
    pub seed: u32,
    /// Frames between two audio chunks handed to the caller.
    pub chunk_frames: usize,
}

/// One spoken turn.
#[derive(Debug)]
pub struct TurnRequest<'a> {
    /// The conversation in the thinker's chat format, ending with the opening
    /// of its reply; one media marker per audio clip.
    pub prompt: &'a str,
    /// The audio clips, in marker order.
    pub audio: &'a [&'a MtmdBitmap],
    /// Upper bound on reply tokens.
    pub max_reply_tokens: usize,
    /// The thinker's sampling.
    pub reply: ReplySampling,
    /// The talker's sampling.
    pub voice: VoiceSampling<'a>,
}

/// What one turn produced.
#[derive(Debug)]
pub struct Turn {
    /// The reply as written, in thinker tokens.
    pub reply_tokens: Vec<LlamaToken>,
    /// The reply as spoken.
    pub speech: Speech,
    /// Tokens the thinker read.
    pub prompt_tokens: usize,
}

/// Token ids and settings the talker carries about its thinker.
#[derive(Debug, Clone, Copy)]
struct Pairing {
    accept_layer: u32,
    im_start: LlamaToken,
    user: LlamaToken,
    audio: LlamaToken,
    image: LlamaToken,
    video: LlamaToken,
    thinker_width: usize,
}

impl Pairing {
    fn read(talker: &LlamaContext<'_>) -> Result<Self> {
        let get = |key: &str| -> Result<i64> {
            let full = format!("qwen3omnitalker.{key}");
            talker
                .model
                .meta_val_str(&full)
                .map_err(|_| inference(format!("the talker does not record {key}")))?
                .parse::<i64>()
                .map_err(|_| inference(format!("the talker's {key} is not a number")))
        };
        let tok = |key: &str| -> Result<LlamaToken> { Ok(LlamaToken(i32::try_from(get(key)?).map_err(inference)?)) };
        Ok(Self {
            accept_layer: u32::try_from(get("accept_hidden_layer")?).map_err(inference)?,
            im_start: tok("im_start_token_id")?,
            user: tok("user_token_id")?,
            audio: tok("audio_token_id")?,
            image: tok("image_token_id")?,
            video: tok("video_token_id")?,
            thinker_width: usize::try_from(get("thinker_embedding_length")?).map_err(inference)?,
        })
    }
}

/// The positions of `tokens` that belong to user turns: a position's role is
/// the token after the last turn opening at or before it.
fn user_positions(tokens: &[LlamaToken], im_start: LlamaToken, user: LlamaToken) -> Vec<usize> {
    let mut out = Vec::new();
    let mut role = None;
    for (i, t) in tokens.iter().enumerate() {
        if *t == im_start {
            role = tokens.get(i + 1).copied();
        }
        if role == Some(user) {
            out.push(i);
        }
    }
    out
}

/// Read `request` with the thinker, write the reply and speak it. `thinker`
/// needs a context whose batch holds the longest audio clip in one decode;
/// `talker` needs embeddings output on. `on_text` sees each reply token as it
/// is written; `on_audio` gets the speech as it is decoded and stops the turn
/// by returning [`ControlFlow::Break`] (a listener talking over it).
///
/// # Errors
///
/// When the projectors are not an omni pair, an audio clip does not fit one
/// decode, the talker refuses the input (picture or video turns, unknown
/// voice), or the backend fails.
#[allow(clippy::too_many_lines)]
pub fn speak_turn(
    thinker: &mut LlamaContext<'_>,
    thinker_mtmd: &MtmdContext,
    talker: &mut LlamaContext<'_>,
    talker_mtmd: &MtmdContext,
    request: &TurnRequest<'_>,
    mut on_text: impl FnMut(LlamaToken),
    on_audio: impl FnMut(u32, &[f32]) -> ControlFlow<()>,
) -> Result<Turn> {
    let pairing = Pairing::read(talker)?;
    let thinker_width = usize::try_from(thinker.model.n_embd()).map_err(inference)?;
    if thinker_width != pairing.thinker_width {
        return Err(inference(format!(
            "the talker reads a {}-wide thinker, this thinker is {thinker_width} wide",
            pairing.thinker_width
        )));
    }

    let chunks = thinker_mtmd
        .tokenize(
            MtmdInputText {
                text: request.prompt.to_owned(),
                add_special: false,
                parse_special: true,
            },
            request.audio,
        )
        .map_err(inference)?;

    // read the prompt chunk by chunk, keeping the hidden rows of every audio position
    thinker.clear_kv_cache();
    let n_batch = i32::try_from(thinker.n_batch()).unwrap_or(i32::MAX);
    thinker.set_layer_input_capture(pairing.accept_layer, true);
    let mut flat: Vec<LlamaToken> = Vec::new();
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::new(); // (first position, rows) per audio chunk
    let mut n_past = 0;
    let read = (|| -> Result<()> {
        for i in 0..chunks.len() {
            let chunk = chunks.get(i).ok_or_else(|| inference("prompt chunk out of range"))?;
            let last = i + 1 == chunks.len();
            match chunk.chunk_type() {
                MtmdInputChunkType::Text => {
                    flat.extend_from_slice(chunk.text_tokens().unwrap_or(&[]));
                }
                MtmdInputChunkType::Audio => {
                    let n = chunk.n_tokens();
                    if n > usize::try_from(n_batch).unwrap_or(usize::MAX) {
                        return Err(inference(format!(
                            "an audio clip of {n} positions does not fit one decode of {n_batch}; use a larger batch"
                        )));
                    }
                    rows.push((flat.len(), Vec::new()));
                    flat.extend(std::iter::repeat_n(pairing.audio, n));
                }
                MtmdInputChunkType::Image => {
                    flat.extend(std::iter::repeat_n(pairing.image, chunk.n_tokens()));
                }
            }
            n_past = chunks
                .eval_chunk(i, thinker_mtmd, thinker, n_past, 0, n_batch, last)
                .map_err(inference)?;
            if chunk.chunk_type() == MtmdInputChunkType::Audio {
                let got = thinker
                    .layer_input(pairing.accept_layer, chunk.n_tokens())
                    .map_err(inference)?
                    .to_vec();
                if let Some(entry) = rows.last_mut() {
                    entry.1 = got;
                }
            }
        }
        Ok(())
    })();
    thinker.set_layer_input_capture(pairing.accept_layer, false);
    read?;

    let positions = user_positions(&flat, pairing.im_start, pairing.user);
    let user_tokens: Vec<LlamaToken> = positions.iter().map(|&p| flat[p]).collect();
    let mut hidden: Vec<f32> = Vec::new();
    for &p in &positions {
        if flat[p] != pairing.audio {
            continue;
        }
        let (start, chunk_rows) = rows
            .iter()
            .rev()
            .find(|(start, r)| *start <= p && p < start + r.len() / thinker_width)
            .ok_or_else(|| inference("an audio position has no hidden row"))?;
        let k = p - start;
        hidden.extend_from_slice(&chunk_rows[k * thinker_width..(k + 1) * thinker_width]);
    }
    if user_tokens.contains(&pairing.video) {
        return Err(inference("speech after video input is not supported"));
    }

    // the thinker writes one token whenever the talker asks for more
    let mut sampler = if request.reply.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(request.reply.top_k),
            LlamaSampler::top_p(request.reply.top_p, 1),
            LlamaSampler::temp(request.reply.temperature),
            LlamaSampler::dist(request.reply.seed),
        ])
    };
    let mut reply_tokens: Vec<LlamaToken> = Vec::new();
    let mut batch = LlamaBatch::new(1, 1);
    let mut finished = false;
    let mut write = || -> Result<ReplyText> {
        if finished {
            return Ok(ReplyText { tokens: Vec::new(), done: true });
        }
        let token = sampler.sample(thinker, -1);
        reply_tokens.push(token);
        on_text(token);
        let end = thinker.model.is_eog_token(token) || reply_tokens.len() >= request.max_reply_tokens;
        if end {
            finished = true;
        } else {
            batch.clear();
            batch.add(token, n_past, &[0], true).map_err(inference)?;
            thinker.decode(&mut batch).map_err(inference)?;
            n_past += 1;
        }
        Ok(ReplyText { tokens: vec![token], done: end })
    };

    let speech = speak_reply(
        talker,
        talker_mtmd,
        &ReplyRequest {
            user_tokens: &user_tokens,
            hidden: &hidden,
            hidden_width: thinker_width,
            voice: request.voice.voice,
            max_frames: request.voice.max_frames,
            temperature: request.voice.temperature,
            top_k: request.voice.top_k,
            top_p: request.voice.top_p,
            projector_top_p: request.voice.projector_top_p,
            repetition_penalty: request.voice.repetition_penalty,
            seed: request.voice.seed,
            chunk_frames: request.voice.chunk_frames,
        },
        &mut write,
        on_audio,
    )?;
    drop(write);
    Ok(Turn {
        reply_tokens,
        speech,
        prompt_tokens: flat.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_positions_cover_every_user_turn_and_nothing_else() {
        let (s, u, a, sys) = (LlamaToken(1), LlamaToken(2), LlamaToken(3), LlamaToken(4));
        let x = LlamaToken(9);
        let t = [s, sys, x, x, s, u, x, x, s, a, x, s, u, x, s, a];
        assert_eq!(user_positions(&t, s, u), vec![4, 5, 6, 7, 11, 12, 13]);
        assert!(user_positions(&[x, x], s, u).is_empty());
    }
}
