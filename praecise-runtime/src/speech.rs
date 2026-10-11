//! Speech synthesis on a language-model backbone and its audio projector.
//!
//! The backbone reads its prompt, then each step samples the backbone's next
//! code, hands its hidden state to the projector for one frame of audio
//! features, and feeds the projector's state back. Frames are decoded to a
//! mono waveform in chunks; each new run of samples goes to the caller as it
//! appears, so a reply can start playing before the utterance is finished,
//! and the caller can stop it there (a listener talking over it).
//!
//! Two kinds of prompt: [`synthesize`] speaks raw text; [`speak_reply`]
//! speaks the reply of a reading model it is paired with (an omni model's
//! talker), conditioned on the user turns that model read and fed the reply
//! token by token while the reading model is still producing it.
//!
//! With a fixed seed the same input, voice and weights give the same samples.

use std::ops::ControlFlow;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdGenAudio, MtmdGenAudioCond, MtmdGenAudioInput, MtmdGenAudioType,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use llama_cpp_2::token::LlamaToken;

use crate::error::{Error, Result};

/// What to speak, and how.
#[derive(Debug)]
pub struct SpeechRequest<'a> {
    /// The text.
    pub text: &'a str,
    /// Reference audio of the voice, where the model takes one.
    pub speaker_reference: Option<&'a MtmdBitmap>,
    /// Language, where the model takes one.
    pub language: Option<&'a str>,
    /// Upper bound on frames; the model usually stops earlier.
    pub max_frames: usize,
    /// Sampling temperature for the backbone's codes.
    pub temperature: f32,
    /// Top-k for the backbone's codes and the projector.
    pub top_k: i32,
    /// Top-p for the backbone's codes and the projector.
    pub top_p: f32,
    /// Seed for every sampler.
    pub seed: u32,
    /// Frames between two chunks handed to the caller.
    pub chunk_frames: usize,
}

/// A finished utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Speech {
    /// Samples per second.
    pub sample_rate: u32,
    /// Mono samples in [-1, 1].
    pub samples: Vec<f32>,
    /// Frames generated.
    pub frames: usize,
    /// The caller stopped the utterance before the model finished it.
    pub interrupted: bool,
}

impl Speech {
    /// Length in seconds.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// The utterance as a 16-bit mono WAV file.
    #[must_use]
    pub fn wav(&self) -> Vec<u8> {
        wav16(self.sample_rate, &self.samples)
    }
}

/// `samples` as a 16-bit PCM mono WAV file.
#[must_use]
pub fn wav16(sample_rate: u32, samples: &[f32]) -> Vec<u8> {
    let data_len = u32::try_from(samples.len() * 2).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        #[allow(clippy::cast_possible_truncation)]
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn inference(what: impl std::fmt::Display) -> Error {
    Error::Inference(what.to_string())
}

/// The backbone's sampler: the model's suppressed codes are never drawn, an
/// optional repetition penalty over the codes drawn so far, then top-k, top-p,
/// temperature and a seeded draw.
fn backbone_sampler(ctx: &LlamaContext<'_>, top_k: i32, top_p: f32, temperature: f32, repetition_penalty: f32, seed: u32) -> LlamaSampler {
    let n_vocab = ctx.model.n_vocab();
    let banned: Vec<LlamaLogitBias> = ctx
        .model
        .suppress_tokens()
        .into_iter()
        .map(|t| LlamaLogitBias::new(t, f32::NEG_INFINITY))
        .collect();
    let mut chain = Vec::with_capacity(6);
    if !banned.is_empty() {
        chain.push(LlamaSampler::logit_bias(n_vocab, &banned));
    }
    if (repetition_penalty - 1.0).abs() > f32::EPSILON {
        chain.push(LlamaSampler::penalties(n_vocab, -1, repetition_penalty, 0.0, 0.0));
    }
    chain.push(LlamaSampler::top_k(top_k));
    chain.push(LlamaSampler::top_p(top_p, 1));
    chain.push(LlamaSampler::temp(temperature));
    chain.push(LlamaSampler::dist(seed));
    LlamaSampler::chain_simple(chain)
}

/// Decodes and hands out the samples made since the last call.
fn flush(
    generator: &mut MtmdGenAudio,
    sent: &mut usize,
    on_chunk: &mut impl FnMut(u32, &[f32]) -> ControlFlow<()>,
) -> Result<ControlFlow<()>> {
    let (rate, pcm) = generator.pcm().map_err(inference)?;
    if pcm.len() > *sent {
        let flow = on_chunk(rate, &pcm[*sent..]);
        *sent = pcm.len();
        return Ok(flow);
    }
    Ok(ControlFlow::Continue(()))
}

/// Speak `request` with the backbone context `ctx` (embeddings output on)
/// and its audio projector `mtmd`. `on_chunk` receives the sample rate and
/// each new run of samples as it is decoded, and stops the utterance by
/// returning [`ControlFlow::Break`]; what was made up to then is returned.
///
/// # Errors
///
/// When the projector does not speak raw text or the backend fails.
pub fn synthesize(
    ctx: &mut LlamaContext<'_>,
    mtmd: &MtmdContext,
    request: &SpeechRequest<'_>,
    mut on_chunk: impl FnMut(u32, &[f32]) -> ControlFlow<()>,
) -> Result<Speech> {
    match mtmd.gen_audio_type().map_err(inference)? {
        MtmdGenAudioType::None => return Err(inference("the projector does not generate audio")),
        MtmdGenAudioType::Qwen3Omni => {
            return Err(inference("this projector speaks a reading model's reply, not raw text"));
        }
        MtmdGenAudioType::Qwen3Tts | MtmdGenAudioType::PocketTts => {}
    }
    ctx.clear_kv_cache();
    let n_batch = i32::try_from(ctx.n_batch()).unwrap_or(i32::MAX);
    let mut generator = MtmdGenAudio::new(ctx, mtmd).map_err(inference)?;
    generator.set_input(&MtmdGenAudioInput {
        text: request.text,
        speaker_reference: request.speaker_reference,
        language: request.language,
        top_k: request.top_k,
        top_p: request.top_p,
        seed: request.seed,
    })
    .map_err(inference)?;
    while generator.step_prompt(n_batch).map_err(inference)? > 0 {}

    let mut sampler = backbone_sampler(ctx, request.top_k, request.top_p, request.temperature, 1.0, request.seed);
    let mut sampled = sampler.sample(ctx, -1);
    let mut h_state = ctx.embeddings_ith(-1).map_err(inference)?.as_ptr();
    let mut frames = 0usize;
    let mut sent = 0usize;
    let mut interrupted = false;
    let chunk = request.chunk_frames.max(1);
    while frames < request.max_frames {
        // SAFETY: `h_state` is the context's last output row or the state the
        // previous step returned; both live until the next step.
        let step = unsafe { generator.step_gen(Some(sampled), h_state) }.map_err(inference)?;
        let Some(next) = step.h_state else { break };
        frames += 1;
        h_state = next;
        if step.stop {
            break;
        }
        if frames.is_multiple_of(chunk) && flush(&mut generator, &mut sent, &mut on_chunk)?.is_break() {
            interrupted = true;
            break;
        }
        sampled = sampler.sample(ctx, -1);
    }
    finish(&mut generator, sent, frames, interrupted, &mut on_chunk)
}

fn finish(
    generator: &mut MtmdGenAudio,
    mut sent: usize,
    frames: usize,
    mut interrupted: bool,
    on_chunk: &mut impl FnMut(u32, &[f32]) -> ControlFlow<()>,
) -> Result<Speech> {
    if !interrupted && flush(generator, &mut sent, on_chunk)?.is_break() {
        interrupted = true;
    }
    let (sample_rate, pcm) = generator.pcm().map_err(inference)?;
    // an interrupted utterance ends where the listener stopped it
    let samples = if interrupted { pcm[..sent.min(pcm.len())].to_vec() } else { pcm.to_vec() };
    Ok(Speech {
        sample_rate,
        samples,
        frames,
        interrupted,
    })
}

/// What a paired reading model's reply is spoken about, and how.
#[derive(Debug)]
pub struct ReplyRequest<'a> {
    /// The user turns the reading model read, one token per position, media
    /// positions holding its placeholder token.
    pub user_tokens: &'a [LlamaToken],
    /// The reading model's inner hidden state at each media position, in
    /// order, `hidden_width` floats each.
    pub hidden: &'a [f32],
    /// Width of a hidden row.
    pub hidden_width: usize,
    /// Voice name.
    pub voice: &'a str,
    /// Upper bound on frames; the model usually stops earlier.
    pub max_frames: usize,
    /// Sampling temperature for the backbone's codes.
    pub temperature: f32,
    /// Top-k for the backbone's codes and the projector.
    pub top_k: i32,
    /// Top-p for the backbone's codes.
    pub top_p: f32,
    /// Top-p for the projector's codes.
    pub projector_top_p: f32,
    /// Repetition penalty over the backbone's codes (1.0 for none).
    pub repetition_penalty: f32,
    /// Seed for every sampler.
    pub seed: u32,
    /// Frames between two chunks handed to the caller.
    pub chunk_frames: usize,
}

/// A run of the reading model's reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyText {
    /// New reply tokens, its end-of-turn token included when it ends here.
    pub tokens: Vec<LlamaToken>,
    /// The reply is complete.
    pub done: bool,
}

/// Speak the reply of the reading model paired with this talker. `ctx` is the
/// talker backbone (embeddings output on), `mtmd` its projector. `reply` is
/// called whenever the talker has spoken all reply text it was given: it
/// advances the reading model and returns what it produced (at least one
/// token unless the reply is complete). `on_chunk` receives the audio as it
/// is decoded and stops the utterance by returning [`ControlFlow::Break`].
///
/// # Errors
///
/// When the projector does not speak a reading model's reply, the request is
/// refused (unknown voice, rows not matching the media positions, picture or
/// video input), `reply` fails, or the backend fails.
pub fn speak_reply(
    ctx: &mut LlamaContext<'_>,
    mtmd: &MtmdContext,
    request: &ReplyRequest<'_>,
    mut reply: impl FnMut() -> Result<ReplyText>,
    mut on_chunk: impl FnMut(u32, &[f32]) -> ControlFlow<()>,
) -> Result<Speech> {
    if mtmd.gen_audio_type().map_err(inference)? != MtmdGenAudioType::Qwen3Omni {
        return Err(inference("the projector does not speak a reading model's reply"));
    }
    ctx.clear_kv_cache();
    let n_batch = i32::try_from(ctx.n_batch()).unwrap_or(i32::MAX);
    let mut generator = MtmdGenAudio::new(ctx, mtmd).map_err(inference)?;
    generator
        .set_cond(&MtmdGenAudioCond {
            user_tokens: request.user_tokens,
            hidden: request.hidden,
            hidden_width: request.hidden_width,
            voice: request.voice,
            top_k: request.top_k,
            top_p: request.projector_top_p,
            seed: request.seed,
        })
        .map_err(inference)?;

    // the prompt ends on the reply's first token
    let mut done: bool;
    loop {
        let text = reply()?;
        done = text.done;
        let got = !text.tokens.is_empty();
        generator.push_text(&text.tokens, done).map_err(inference)?;
        if got {
            break;
        }
        if done {
            return Err(inference("the reading model gave an empty reply"));
        }
    }
    while generator.step_prompt(n_batch).map_err(inference)? > 0 {}

    let mut sampler = backbone_sampler(
        ctx,
        request.top_k,
        request.top_p,
        request.temperature,
        request.repetition_penalty,
        request.seed,
    );
    let mut sampled = sampler.sample(ctx, -1);
    let mut h_state = ctx.embeddings_ith(-1).map_err(inference)?.as_ptr();
    let mut frames = 0usize;
    let mut sent = 0usize;
    let mut interrupted = false;
    let chunk = request.chunk_frames.max(1);
    while frames < request.max_frames {
        // SAFETY: as in `synthesize`.
        let step = unsafe { generator.step_gen(Some(sampled), h_state) }.map_err(inference)?;
        if step.needs_text {
            if done {
                return Err(inference("the talker asked for text after the reply was complete"));
            }
            let text = reply()?;
            if text.tokens.is_empty() && !text.done {
                return Err(inference("the reading model produced nothing"));
            }
            done = text.done;
            generator.push_text(&text.tokens, done).map_err(inference)?;
            continue;
        }
        let Some(next) = step.h_state else { break };
        frames += 1;
        h_state = next;
        if step.stop {
            break;
        }
        if frames.is_multiple_of(chunk) && flush(&mut generator, &mut sent, &mut on_chunk)?.is_break() {
            interrupted = true;
            break;
        }
        sampled = sampler.sample(ctx, -1);
    }
    finish(&mut generator, sent, frames, interrupted, &mut on_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav16_writes_a_mono_pcm_header_and_clamped_samples() {
        let wav = wav16(24_000, &[0.0, 1.0, -1.0, 2.0]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        let samples: Vec<i16> = wav[44..].chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(samples, vec![0, 32767, -32767, 32767]);
        let speech = Speech { sample_rate: 24_000, samples: vec![0.0; 12_000], frames: 6, interrupted: false };
        assert!((speech.seconds() - 0.5).abs() < 1e-9);
    }
}
