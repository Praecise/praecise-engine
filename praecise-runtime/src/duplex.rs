//! Full-duplex speech sessions: the model listens while it speaks.
//!
//! A [`DuplexSession`] takes the listener's audio continuously and the
//! model's audio as it is generated. It finds where the listener's turns
//! start and end (voice activity over a noise floor taken as the quietest
//! frame of the last few seconds, with hangover),
//! hands each finished turn to the caller, and reports a barge-in when the
//! listener talks over the model, so the caller stops generating at once and
//! drops the audio not yet played. It also measures the delay from the end of
//! the listener's turn to the model's first audio.
//!
//! The session is plain state over sample counts, so the same input gives the
//! same events. Input audio is expected echo-cancelled at the client: the
//! model's own playback picked up by the microphone would otherwise read as
//! the listener talking.

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

/// How a session decides who is talking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuplexConfig {
    /// Sample rate of the listener's audio.
    pub sample_rate: u32,
    /// Analysis frame, in milliseconds.
    pub frame_ms: u32,
    /// A frame is voiced when its level is this many dB over the noise floor...
    pub margin_db: f32,
    /// ...and over this absolute level (dBFS).
    pub min_level_db: f32,
    /// Voiced time before a turn counts as started.
    pub start_ms: u32,
    /// Silence after voice that ends the turn.
    pub end_ms: u32,
    /// Voiced time over the model's speech that counts as a barge-in.
    pub barge_in_ms: u32,
    /// Longest turn kept; older audio of a longer turn is dropped.
    pub max_turn_ms: u32,
    /// The noise floor is the quietest frame of this much recent audio.
    pub floor_window_ms: u32,
}

impl Default for DuplexConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_ms: 20,
            margin_db: 12.0,
            min_level_db: -50.0,
            start_ms: 100,
            end_ms: 600,
            barge_in_ms: 200,
            max_turn_ms: 30_000,
            floor_window_ms: 3_000,
        }
    }
}

/// What the session saw in the listener's audio.
#[derive(Debug, Clone, PartialEq)]
pub enum DuplexEvent {
    /// The listener started a turn.
    TurnStarted,
    /// The listener finished a turn: its audio, with the onset included.
    TurnEnded {
        /// The turn's samples.
        audio: Vec<f32>,
    },
    /// The listener talked over the model: stop generating and drop the
    /// model's audio not yet played.
    BargeIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Listen {
    Quiet,
    Onset,
    Turn,
}

/// One conversation's turn-taking state.
#[derive(Debug)]
pub struct DuplexSession {
    cfg: DuplexConfig,
    frame: usize,
    pending: Vec<f32>,
    levels: VecDeque<f32>,
    listen: Listen,
    voiced_run: usize,
    quiet_run: usize,
    turn: Vec<f32>,
    speaking: bool,
    barged: bool,
    over_run: usize,
    turn_ended_at: Option<Instant>,
    first_audio: Option<Duration>,
}

fn level_db(frame: &[f32]) -> f32 {
    let energy = frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32;
    10.0 * energy.max(1e-12).log10()
}

impl DuplexSession {
    /// A session with nobody talking yet.
    ///
    /// # Panics
    ///
    /// When the analysis frame holds no samples.
    #[must_use]
    pub fn new(cfg: DuplexConfig) -> Self {
        let frame = (cfg.sample_rate as usize * cfg.frame_ms as usize) / 1000;
        assert!(frame > 0, "the analysis frame holds no samples");
        Self {
            cfg,
            frame,
            pending: Vec::new(),
            levels: VecDeque::new(),
            listen: Listen::Quiet,
            voiced_run: 0,
            quiet_run: 0,
            turn: Vec::new(),
            speaking: false,
            barged: false,
            over_run: 0,
            turn_ended_at: None,
            first_audio: None,
        }
    }

    fn frames(&self, ms: u32) -> usize {
        (ms as usize).div_ceil(self.cfg.frame_ms as usize).max(1)
    }

    /// Listener audio, in chunks of any size; returns what it caused.
    pub fn push_input(&mut self, pcm: &[f32]) -> Vec<DuplexEvent> {
        let mut events = Vec::new();
        self.pending.extend_from_slice(pcm);
        let mut at = 0;
        while self.pending.len() - at >= self.frame {
            let frame: Vec<f32> = self.pending[at..at + self.frame].to_vec();
            at += self.frame;
            self.frame_in(&frame, &mut events);
        }
        self.pending.drain(..at);
        events
    }

    fn frame_in(&mut self, frame: &[f32], events: &mut Vec<DuplexEvent>) {
        let db = level_db(frame);
        self.levels.push_back(db);
        if self.levels.len() > self.frames(self.cfg.floor_window_ms) {
            self.levels.pop_front();
        }
        // a steady noise fills the window and becomes the floor; speech pauses keep it low
        let floor = self.levels.iter().copied().fold(f32::INFINITY, f32::min);
        let voiced = db > floor + self.cfg.margin_db && db > self.cfg.min_level_db;

        if self.speaking && !self.barged {
            self.over_run = if voiced { self.over_run + 1 } else { 0 };
            if self.over_run >= self.frames(self.cfg.barge_in_ms) {
                self.barged = true;
                events.push(DuplexEvent::BargeIn);
            }
        }

        match self.listen {
            Listen::Quiet => {
                if voiced {
                    self.listen = Listen::Onset;
                    self.voiced_run = 1;
                    self.turn.clear();
                    self.turn.extend_from_slice(frame);
                }
            }
            Listen::Onset => {
                self.turn.extend_from_slice(frame);
                if voiced {
                    self.voiced_run += 1;
                    if self.voiced_run >= self.frames(self.cfg.start_ms) {
                        self.listen = Listen::Turn;
                        self.quiet_run = 0;
                        events.push(DuplexEvent::TurnStarted);
                    }
                } else {
                    self.listen = Listen::Quiet;
                    self.turn.clear();
                }
            }
            Listen::Turn => {
                self.turn.extend_from_slice(frame);
                let max = (self.cfg.sample_rate as usize * self.cfg.max_turn_ms as usize) / 1000;
                if self.turn.len() > max {
                    let excess = self.turn.len() - max;
                    self.turn.drain(..excess);
                }
                self.quiet_run = if voiced { 0 } else { self.quiet_run + 1 };
                if self.quiet_run >= self.frames(self.cfg.end_ms) {
                    self.listen = Listen::Quiet;
                    let trailing = self.quiet_run * self.frame;
                    let keep = self.turn.len().saturating_sub(trailing);
                    self.turn.truncate(keep);
                    self.turn_ended_at = Some(Instant::now());
                    self.first_audio = None;
                    events.push(DuplexEvent::TurnEnded {
                        audio: std::mem::take(&mut self.turn),
                    });
                }
            }
        }
    }

    /// The model starts answering.
    pub fn start_response(&mut self) {
        self.speaking = true;
        self.barged = false;
        self.over_run = 0;
    }

    /// The model's audio as it is generated. Returns [`ControlFlow::Break`]
    /// once the listener has barged in: the caller stops generating.
    pub fn push_output(&mut self, pcm: &[f32]) -> ControlFlow<()> {
        if self.barged {
            return ControlFlow::Break(());
        }
        if !pcm.is_empty() && self.first_audio.is_none() {
            if let Some(t) = self.turn_ended_at {
                self.first_audio = Some(t.elapsed());
            }
        }
        ControlFlow::Continue(())
    }

    /// The model finished or was stopped.
    pub fn end_response(&mut self) {
        self.speaking = false;
        self.over_run = 0;
    }

    /// The listener barged in during the current or last response.
    #[must_use]
    pub fn barged_in(&self) -> bool {
        self.barged
    }

    /// Delay from the end of the listener's last turn to the model's first
    /// audio for it.
    #[must_use]
    pub fn first_audio_latency(&self) -> Option<Duration> {
        self.first_audio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn tone(ms: u32, amp: f32) -> Vec<f32> {
        let n = (RATE * ms / 1000) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / RATE as f32).sin())
            .collect()
    }

    fn hiss(ms: u32, amp: f32) -> Vec<f32> {
        // deterministic low-level noise
        let n = (RATE * ms / 1000) as usize;
        let mut x = 12345u32;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                amp * ((x >> 16) as f32 / 32768.0 - 1.0)
            })
            .collect()
    }

    #[test]
    fn a_turn_is_found_between_quiet_stretches_and_handed_over_whole() {
        let mut s = DuplexSession::new(DuplexConfig::default());
        let mut events = s.push_input(&hiss(500, 0.001));
        events.extend(s.push_input(&tone(800, 0.3)));
        events.extend(s.push_input(&hiss(800, 0.001)));
        assert_eq!(events.first(), Some(&DuplexEvent::TurnStarted));
        let ended: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                DuplexEvent::TurnEnded { audio } => Some(audio.len()),
                _ => None,
            })
            .collect();
        assert_eq!(ended.len(), 1);
        // the whole tone, give or take a frame either side
        let tone_len = (RATE * 800 / 1000) as usize;
        assert!(ended[0].abs_diff(tone_len) <= 2 * 320, "turn of {} samples for a {tone_len}-sample tone", ended[0]);
    }

    #[test]
    fn a_click_is_not_a_turn() {
        let mut s = DuplexSession::new(DuplexConfig::default());
        let mut events = s.push_input(&hiss(400, 0.001));
        events.extend(s.push_input(&tone(40, 0.5)));
        events.extend(s.push_input(&hiss(1000, 0.001)));
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn talking_over_the_model_stops_it_and_a_short_noise_does_not() {
        let mut s = DuplexSession::new(DuplexConfig::default());
        s.push_input(&hiss(400, 0.001));
        s.start_response();
        assert!(s.push_output(&[0.1; 480]).is_continue());
        // a 60 ms knock is below the barge-in time
        let events = s.push_input(&tone(60, 0.4));
        assert!(!events.contains(&DuplexEvent::BargeIn));
        s.push_input(&hiss(200, 0.001));
        assert!(s.push_output(&[0.1; 480]).is_continue());
        // speech over the model
        let events = s.push_input(&tone(300, 0.3));
        assert!(events.contains(&DuplexEvent::BargeIn), "{events:?}");
        assert!(s.barged_in());
        assert!(s.push_output(&[0.1; 480]).is_break());
        s.end_response();
    }

    #[test]
    fn first_audio_latency_counts_from_the_end_of_the_turn() {
        let mut s = DuplexSession::new(DuplexConfig::default());
        s.push_input(&hiss(300, 0.001));
        s.push_input(&tone(500, 0.3));
        let events = s.push_input(&hiss(700, 0.001));
        assert!(events.iter().any(|e| matches!(e, DuplexEvent::TurnEnded { .. })));
        assert!(s.first_audio_latency().is_none());
        s.start_response();
        std::thread::sleep(Duration::from_millis(5));
        assert!(s.push_output(&[]).is_continue());
        assert!(s.first_audio_latency().is_none(), "an empty run is not audio");
        assert!(s.push_output(&[0.2; 160]).is_continue());
        let latency = s.first_audio_latency().expect("latency");
        assert!(latency >= Duration::from_millis(5));
    }

    #[test]
    fn a_steady_noise_becomes_the_floor() {
        let mut s = DuplexSession::new(DuplexConfig::default());
        // a fan well over the absolute threshold, switched on in a quiet room
        s.push_input(&hiss(500, 0.001));
        let events = s.push_input(&hiss(6000, 0.02));
        let turns = events.iter().filter(|e| matches!(e, DuplexEvent::TurnEnded { .. })).count();
        assert!(turns <= 1, "{turns} turns in steady noise");
        let events = s.push_input(&hiss(3000, 0.02));
        assert!(events.is_empty(), "steady noise keeps making events: {events:?}");
        // speech over the fan is still found
        let events = s.push_input(&tone(600, 0.3));
        assert!(events.contains(&DuplexEvent::TurnStarted), "{events:?}");
    }
}
