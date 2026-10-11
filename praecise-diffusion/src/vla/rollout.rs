//! Closed-loop execution of a chunk policy, one control tick at a time.
//!
//! Each tick records the observation and returns one action. When the queue
//! of actions is empty the policy predicts a new chunk from the current
//! observation and its first `n_action_steps` actions are queued; the rest
//! of the chunk is discarded. This is the control loop the policies'
//! reference runtimes execute.

use std::collections::VecDeque;

use super::{ChunkPolicy, VlaObservation};
use crate::error::{Error, Result};
use crate::schedule;

/// Where each chunk's initial noise comes from.
#[derive(Debug, Clone)]
pub enum NoiseSource {
    /// The same noise for every chunk.
    Fixed(Vec<f32>),
    /// A fresh draw per chunk from a seeded stream (chunk `k` uses seed
    /// `seed + k`).
    Seeded(u64),
}

/// A chunk policy run in closed loop.
pub struct ChunkRollout<'p> {
    policy: &'p dyn ChunkPolicy,
    noise: NoiseSource,
    chunks: u64,
    queue: VecDeque<Vec<f32>>,
}

impl std::fmt::Debug for ChunkRollout<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkRollout").field("chunks", &self.chunks).field("queued", &self.queue.len()).finish_non_exhaustive()
    }
}

impl<'p> ChunkRollout<'p> {
    /// A rollout of `policy` drawing chunk noise from `noise`.
    ///
    /// # Errors
    /// When fixed noise has the wrong length.
    pub fn new(policy: &'p dyn ChunkPolicy, noise: NoiseSource) -> Result<Self> {
        if let NoiseSource::Fixed(n) = &noise {
            if n.len() != policy.noise_len() {
                return Err(Error::Request(format!("the policy reads {} noise values, got {}", policy.noise_len(), n.len())));
            }
        }
        Ok(Self { policy, noise, chunks: 0, queue: VecDeque::new() })
    }

    /// Actions executed from each chunk.
    #[must_use]
    pub fn execute_steps(&self) -> usize {
        self.policy.n_action_steps().clamp(1, self.policy.chunk_size())
    }

    /// Chunks predicted so far.
    #[must_use]
    pub fn chunks(&self) -> u64 {
        self.chunks
    }

    /// Drop the queued actions, so the next tick predicts a new chunk.
    pub fn reset(&mut self) {
        self.queue.clear();
    }

    /// One control tick: the action to send for this observation.
    ///
    /// # Errors
    /// On a malformed observation or a backend failure.
    pub fn tick(&mut self, obs: &VlaObservation<'_>) -> Result<Vec<f32>> {
        if self.queue.is_empty() {
            let noise = match &self.noise {
                NoiseSource::Fixed(n) => n.clone(),
                NoiseSource::Seeded(seed) => schedule::gaussian(seed.wrapping_add(self.chunks), self.policy.noise_len()),
            };
            let chunk = self.policy.predict(obs, &noise)?;
            let d = self.policy.action_dim();
            if chunk.len() != d * self.policy.chunk_size() {
                return Err(Error::Backend(format!("the policy returned {} values for a chunk of {}x{d}", chunk.len(), self.policy.chunk_size())));
            }
            self.chunks += 1;
            self.queue.extend(chunk.chunks(d).take(self.execute_steps()).map(<[f32]>::to_vec));
        }
        Ok(self.queue.pop_front().expect("a chunk queues at least one action"))
    }
}
