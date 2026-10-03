//! Speech synthesis on a language-model backbone and its audio projector.
//!
//! The backbone reads the text, then each step samples the backbone's next
//! code, hands its hidden state to the projector for one frame of audio
//! features, and feeds the projector's state back. Frames are decoded to a
//! mono waveform in chunks; [`synthesize`] hands each new run of samples to
//! the caller as it appears, so a reply can start playing before the
//! utterance is finished.
//!
//! With a fixed seed the same text, voice and weights give the same samples.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdGenAudio, MtmdGenAudioInput, MtmdGenAudioType};
use llama_cpp_2::sampling::LlamaSampler;

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

/// Speak `request` with the backbone context `ctx` (embeddings output on)
/// and its audio projector `mtmd`. `on_chunk` receives the sample rate and
/// each new run of samples as it is decoded; the full utterance is returned.
///
/// # Errors
///
/// When the projector does not generate audio or the backend fails.
pub fn synthesize(
    ctx: &mut LlamaContext<'_>,
    mtmd: &MtmdContext,
    request: &SpeechRequest<'_>,
    mut on_chunk: impl FnMut(u32, &[f32]),
) -> Result<Speech> {
    if mtmd.gen_audio_type().map_err(inference)? == MtmdGenAudioType::None {
        return Err(inference("the projector does not generate audio"));
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

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(request.top_k),
        LlamaSampler::top_p(request.top_p, 1),
        LlamaSampler::temp(request.temperature),
        LlamaSampler::dist(request.seed),
    ]);
    let mut sampled = sampler.sample(ctx, -1);
    let mut h_state = ctx.embeddings_ith(-1).map_err(inference)?.as_ptr();
    let mut frames = 0usize;
    let mut sent = 0usize;
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
        if frames.is_multiple_of(chunk) {
            let (rate, pcm) = generator.pcm().map_err(inference)?;
            if pcm.len() > sent {
                on_chunk(rate, &pcm[sent..]);
                sent = pcm.len();
            }
        }
        sampled = sampler.sample(ctx, -1);
    }
    let (sample_rate, pcm) = generator.pcm().map_err(inference)?;
    if pcm.len() > sent {
        on_chunk(sample_rate, &pcm[sent..]);
    }
    Ok(Speech {
        sample_rate,
        samples: pcm.to_vec(),
        frames,
    })
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
        let speech = Speech { sample_rate: 24_000, samples: vec![0.0; 12_000], frames: 6 };
        assert!((speech.seconds() - 0.5).abs() < 1e-9);
    }
}
