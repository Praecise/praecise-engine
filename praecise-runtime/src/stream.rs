//! Streaming stop-sequence and reasoning-span handling.
//!
//! Backend-free: this is pure text bookkeeping over decoded pieces, so it
//! compiles and runs without a bundled backend. It accumulates decoded token
//! pieces, splits reasoning spans (`<think>` … `</think>`) from visible text,
//! detects stop sequences (which may span several tokens), and — when the
//! caller is streaming — releases bytes only once they can no longer turn out
//! to be the leading part of a stop sequence or a reasoning marker.

/// Byte length of the longest stop sequence that `text` ends with, or `None`
/// when no stop sequence matched. The caller truncates by that many bytes so
/// the delimiter never reaches the client.
pub fn matched_stop_len(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter(|s| !s.is_empty() && text.ends_with(s.as_str()))
        .map(|s| s.len())
        .max()
}

/// Where a generation's reasoning span is, as its chat template frames it.
///
/// Two things differ by template, and a splitter that guesses either one leaks
/// reasoning into the answer:
///
/// * **The markers.** Qwen-family templates wrap reasoning in `<think>` …
///   `</think>`. Gemma 4 uses a thought channel, `<|channel>thought` …
///   `<channel|>`, and never emits `<think>` at all.
/// * **Whether the prompt already opened the span.** Qwen3.8's generation
///   prompt ends `<|im_start|>assistant\n<think>\n` whenever thinking is on,
///   so the model's output begins mid-reasoning and only the close marker ever
///   appears. Gemma 4 opens its channel in the prompt only for the turn after a
///   tool response. A splitter that starts outside the span streams the
///   reasoning as answer text, and by the time the close marker arrives those
///   bytes have already been sent. That was measured on qwen3.8-27b: correct
///   when not streaming (nothing had been released, so the text was
///   reclaimable), and the whole reasoning in `content` when streaming.
///
/// Both are read off the rendered prompt, the one place that knows, which is
/// what the reference parsers do: vLLM's and SGLang's `qwen3` parsers start in
/// reasoning when thinking is enabled, and llama.cpp re-parses the output
/// prefixed with the generation prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningFrame {
    /// Marker that opens a reasoning span.
    pub open: &'static str,
    /// Marker that closes it.
    pub close: &'static str,
    /// Whether the prompt ended inside an open span.
    pub open_at_start: bool,
}

impl ReasoningFrame {
    /// `<think>` … `</think>`, not already open.
    pub const THINK: Self = Self {
        open: "<think>",
        close: "</think>",
        open_at_start: false,
    };

    /// Gemma 4's thought channel, not already open.
    pub const GEMMA4_THOUGHT: Self = Self {
        open: "<|channel>thought",
        close: "<channel|>",
        open_at_start: false,
    };

    /// The frame a rendered prompt implies.
    pub fn for_prompt(prompt: &str) -> Self {
        // `<|turn>` is Gemma 4's turn marker; no other supported template has it.
        let base = if prompt.contains("<|turn>") {
            Self::GEMMA4_THOUGHT
        } else {
            Self::THINK
        };
        // Trailing whitespace is the template's own newline after the marker.
        // A prompt that opened and closed an empty span (thinking switched off)
        // ends with the close marker, so it is correctly not open.
        Self {
            open_at_start: prompt.trim_end().ends_with(base.open),
            ..base
        }
    }
}

impl Default for ReasoningFrame {
    fn default() -> Self {
        Self::THINK
    }
}

/// Length of the longest suffix of `s` that is a strict prefix of `marker`.
///
/// Those bytes cannot be classified yet: `<thi` is either the start of a marker
/// or four literal characters, and only the next piece decides. Holding them is
/// the same trick the stop-sequence path uses.
fn dangling_prefix(s: &str, marker: &str) -> usize {
    let max = s.len().min(marker.len() - 1);
    (1..=max)
        .rev()
        .find(|&k| {
            s.is_char_boundary(s.len() - k) && s.as_bytes()[s.len() - k..] == marker.as_bytes()[..k]
        })
        .unwrap_or(0)
}

/// Accumulates decoded token pieces, detects stop sequences, and — when the
/// caller is streaming — releases bytes only once they can no longer turn out
/// to be the leading part of a stop sequence.
///
/// A stop sequence may span several tokens, so the last `hold` bytes are kept
/// back until either more text disambiguates them or generation ends. With no
/// stop sequences configured `hold` is zero and every piece is released the
/// moment it is decoded.
pub struct StopStream {
    /// Text the caller is meant to see: reasoning spans removed.
    text: String,
    /// The model's reasoning, accumulated separately.
    reasoning: String,
    /// Bytes decoded but not yet classified, because they could still turn out
    /// to be the leading part of a `<think>` / `</think>` marker.
    pending: String,
    emitted: usize,
    hold: usize,
    stop: Vec<String>,
    hit: bool,
    in_think: bool,
    frame: ReasoningFrame,
    /// Chunks a non-blocking stream could not hand over yet, because the
    /// receiver's channel was full. `None` for a blocking stream.
    outbox: Option<Outbox>,
    /// Where reasoning is streamed as it is produced, when the caller asked
    /// for it, with the reasoning not handed over yet.
    reasoning_sink: Option<ReasoningSink>,
}

/// A receiver for reasoning deltas and what it has not taken yet.
struct ReasoningSink {
    tx: tokio::sync::mpsc::Sender<String>,
    unsent: String,
}

/// Most bytes a non-blocking stream holds for a receiver that is not keeping
/// up. Past this the receiver is treated as stalled and the stream fails,
/// rather than growing without bound.
pub const STREAM_BACKLOG_LIMIT: usize = 1 << 20;

#[derive(Default)]
struct Outbox {
    chunks: std::collections::VecDeque<String>,
    bytes: usize,
    closed: bool,
}

impl Outbox {
    /// Hand over as many queued chunks as the channel takes. False once the
    /// receiver is gone.
    fn flush(&mut self, tx: &tokio::sync::mpsc::Sender<String>) -> bool {
        while let Some(chunk) = self.chunks.pop_front() {
            let len = chunk.len();
            match tx.try_send(chunk) {
                Ok(()) => self.bytes -= len,
                Err(tokio::sync::mpsc::error::TrySendError::Full(chunk)) => {
                    self.chunks.push_front(chunk);
                    return true;
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    self.chunks.clear();
                    self.bytes = 0;
                    self.closed = true;
                    return false;
                }
            }
        }
        !self.closed
    }
}

impl StopStream {
    /// Build a stream that trims the given stop sequences.
    pub fn new(stop: Vec<String>) -> Self {
        let hold = stop
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        Self {
            text: String::new(),
            reasoning: String::new(),
            pending: String::new(),
            emitted: 0,
            hold,
            stop,
            hit: false,
            in_think: false,
            frame: ReasoningFrame::THINK,
            outbox: None,
            reasoning_sink: None,
        }
    }

    /// Stream the reasoning to `tx` as it is produced, separately from the
    /// visible text. Without this, reasoning is only returned at the end.
    pub fn with_reasoning(mut self, tx: Option<tokio::sync::mpsc::Sender<String>>) -> Self {
        self.reasoning_sink = tx.map(|tx| ReasoningSink { tx, unsent: String::new() });
        self
    }

    /// Record reasoning text, and queue it for the reasoning receiver.
    fn think(&mut self, s: &str) {
        self.reasoning.push_str(s);
        if let Some(sink) = self.reasoning_sink.as_mut() {
            sink.unsent.push_str(s);
        }
    }

    /// Hand queued reasoning to its receiver: waiting for room on a blocking
    /// stream, without waiting on a non-blocking one. A receiver that is gone
    /// only stops the reasoning stream, not the generation.
    fn send_reasoning(&mut self) {
        let blocking = self.outbox.is_none();
        let Some(sink) = self.reasoning_sink.as_mut() else { return };
        if sink.unsent.is_empty() {
            return;
        }
        let chunk = std::mem::take(&mut sink.unsent);
        let sent = if blocking {
            sink.tx.blocking_send(chunk).map_err(|_| None)
        } else {
            sink.tx.try_send(chunk).map_err(|e| match e {
                tokio::sync::mpsc::error::TrySendError::Full(chunk) => Some(chunk),
                tokio::sync::mpsc::error::TrySendError::Closed(_) => None,
            })
        };
        match sent {
            Ok(()) => {}
            Err(Some(chunk)) => sink.unsent = chunk,
            Err(None) => self.reasoning_sink = None,
        }
    }

    /// Never wait on the receiver. A chunk the channel cannot take right now
    /// is queued and handed over by a later [`Self::flush`], so a slow reader
    /// delays only its own text. The batch engine serves every sequence from
    /// one thread, where waiting on one receiver would stall all of them.
    pub fn nonblocking(mut self) -> Self {
        self.outbox = Some(Outbox::default());
        self
    }

    /// Hand queued chunks to the receiver without waiting. False once the
    /// receiver is gone.
    pub fn flush(&mut self, tx: Option<&tokio::sync::mpsc::Sender<String>>) -> bool {
        self.send_reasoning();
        match (self.outbox.as_mut(), tx) {
            (Some(outbox), Some(tx)) => outbox.flush(tx),
            _ => true,
        }
    }

    /// Whether every released chunk has reached the receiver.
    pub fn delivered(&self) -> bool {
        self.outbox.as_ref().is_none_or(|o| o.chunks.is_empty())
            && self.reasoning_sink.as_ref().is_none_or(|r| r.unsent.is_empty())
    }

    /// Whether the receiver has fallen [`STREAM_BACKLOG_LIMIT`] bytes behind.
    pub fn stalled(&self) -> bool {
        self.outbox.as_ref().is_some_and(|o| o.bytes > STREAM_BACKLOG_LIMIT)
            || self.reasoning_sink.as_ref().is_some_and(|r| r.unsent.len() > STREAM_BACKLOG_LIMIT)
    }

    /// Split reasoning with this template's markers, starting inside a span
    /// when the prompt left one open.
    /// Whether the text arriving now is reasoning.
    pub fn in_reasoning(&self) -> bool {
        self.in_think
    }

    /// The reasoning markers this stream splits on.
    pub fn frame(&self) -> ReasoningFrame {
        self.frame
    }

    pub fn framed(mut self, frame: ReasoningFrame) -> Self {
        self.in_think = frame.open_at_start;
        self.frame = frame;
        self
    }

    /// Absorb one decoded piece. Returns `false` when the stream receiver has
    /// been dropped, which the generation loops treat as "stop generating".
    pub fn push(&mut self, piece: &str, tx: Option<&tokio::sync::mpsc::Sender<String>>) -> bool {
        self.pending.push_str(piece);
        self.classify();
        if let Some(n) = matched_stop_len(&self.text, &self.stop) {
            self.text.truncate(self.text.len() - n);
            self.emitted = self.emitted.min(self.text.len());
            self.hit = true;
        }
        self.release(tx)
    }

    /// Move settled bytes out of `pending` into either the visible text or the
    /// reasoning buffer, leaving behind only what a marker could still claim.
    fn classify(&mut self) {
        loop {
            if self.in_think {
                if let Some(i) = self.pending.find(self.frame.close) {
                    let span = self.pending[..i].to_string();
                    self.think(&span);
                    self.pending.drain(..i + self.frame.close.len());
                    self.in_think = false;
                    continue;
                }
                let keep = dangling_prefix(&self.pending, self.frame.close);
                let take = self.pending.len() - keep;
                let span = self.pending[..take].to_string();
                self.think(&span);
                self.pending.drain(..take);
                return;
            }

            if let Some(i) = self.pending.find(self.frame.open) {
                self.text.push_str(&self.pending[..i]);
                self.pending.drain(..i + self.frame.open.len());
                self.in_think = true;
                continue;
            }

            // A close with no open: the chat template opened the block in the
            // prompt, so the model's output starts mid-thought. Everything so
            // far was reasoning — reclaimable only while nothing has been
            // streamed yet, since bytes already sent cannot be recalled.
            if let Some(i) = self.pending.find(self.frame.close) {
                self.text.push_str(&self.pending[..i]);
                self.pending.drain(..i + self.frame.close.len());
                if self.emitted == 0 {
                    let span = std::mem::take(&mut self.text);
                    self.think(&span);
                }
                continue;
            }

            let keep = dangling_prefix(&self.pending, self.frame.open)
                .max(dangling_prefix(&self.pending, self.frame.close));
            let take = self.pending.len() - keep;
            self.text.push_str(&self.pending[..take]);
            self.pending.drain(..take);
            return;
        }
    }

    fn release(&mut self, tx: Option<&tokio::sync::mpsc::Sender<String>>) -> bool {
        self.send_reasoning();
        let Some(tx) = tx else { return true };
        let mut boundary = if self.hit {
            self.text.len()
        } else {
            self.text.len().saturating_sub(self.hold)
        };
        while boundary > self.emitted && !self.text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        if boundary <= self.emitted {
            return true;
        }
        let chunk = self.text[self.emitted..boundary].to_string();
        self.emitted = boundary;
        match self.outbox.as_mut() {
            None => tx.blocking_send(chunk).is_ok(),
            Some(outbox) => {
                outbox.bytes += chunk.len();
                outbox.chunks.push_back(chunk);
                outbox.flush(tx) && outbox.bytes <= STREAM_BACKLOG_LIMIT
            }
        }
    }

    /// Whether a stop sequence has been matched.
    pub fn hit_stop(&self) -> bool {
        self.hit
    }

    /// Release anything still held back, then hand over the visible text and
    /// the reasoning span the model produced, if any.
    pub fn finish_parts(
        mut self,
        tx: Option<&tokio::sync::mpsc::Sender<String>>,
    ) -> (String, Option<String>) {
        self.close(tx)
    }

    /// [`Self::finish_parts`] without consuming the stream, so a non-blocking
    /// stream keeps what the receiver has not taken yet for later
    /// [`Self::flush`]es.
    pub fn close(&mut self, tx: Option<&tokio::sync::mpsc::Sender<String>>) -> (String, Option<String>) {
        let leftover = std::mem::take(&mut self.pending);
        if self.in_think {
            self.think(&leftover);
        } else if !self.frame.open.starts_with(&leftover) && !self.frame.close.starts_with(&leftover) {
            self.text.push_str(&leftover);
        }
        self.hit = true;
        self.release(tx);
        let reasoning = self.reasoning.trim().to_string();
        self.reasoning.clear();
        self.emitted = 0;
        (std::mem::take(&mut self.text), (!reasoning.is_empty()).then_some(reasoning))
    }
}

#[cfg(test)]
mod reasoning_frame_tests {
    use super::{ReasoningFrame, StopStream};

    fn stream_all(frame: ReasoningFrame, pieces: &[&str]) -> (String, String, Option<String>) {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1024);
        let mut s = StopStream::new(vec![]).framed(frame);
        for p in pieces {
            assert!(s.push(p, Some(&tx)));
        }
        let (text, reasoning) = s.finish_parts(Some(&tx));
        drop(tx);
        let mut streamed = String::new();
        while let Ok(chunk) = rx.try_recv() {
            streamed.push_str(&chunk);
        }
        (streamed, text, reasoning)
    }

    #[test]
    fn qwen_prompt_that_opened_think_streams_no_reasoning() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        let frame = ReasoningFrame::for_prompt(prompt);
        assert!(frame.open_at_start);
        let (streamed, text, reasoning) = stream_all(
            frame,
            &["The user", " wants a greeting.", "\n</thi", "nk>\n\n", "Hello", "!"],
        );
        assert_eq!(streamed.trim(), "Hello!", "reasoning must never reach the stream");
        assert_eq!(text.trim(), "Hello!");
        assert_eq!(reasoning.as_deref(), Some("The user wants a greeting."));
    }

    #[test]
    fn reasoning_reaches_its_own_receiver_while_it_is_produced() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        let frame = ReasoningFrame::for_prompt(prompt);
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let (rtx, mut rrx) = tokio::sync::mpsc::channel::<String>(64);
        let mut s = StopStream::new(vec![]).framed(frame).nonblocking().with_reasoning(Some(rtx));
        s.push("The user", Some(&tx));
        s.push(" wants a greeting.", Some(&tx));
        // Before the block closes, and before any answer text, the reasoning
        // so far is already with its receiver.
        let mut reasoning = String::new();
        while let Ok(c) = rrx.try_recv() {
            reasoning.push_str(&c);
        }
        assert_eq!(reasoning, "The user wants a greeting.");
        assert!(rx.try_recv().is_err(), "no answer text yet");
        for p in ["\n</thi", "nk>\n\n", "Hello", "!"] {
            s.push(p, Some(&tx));
        }
        let (text, thinking) = s.close(Some(&tx));
        while let Ok(c) = rrx.try_recv() {
            reasoning.push_str(&c);
        }
        let mut streamed = String::new();
        while let Ok(c) = rx.try_recv() {
            streamed.push_str(&c);
        }
        assert_eq!(reasoning.trim(), "The user wants a greeting.", "reasoning never repeats or leaks the marker");
        assert_eq!(streamed.trim(), "Hello!", "reasoning never reaches the answer stream");
        assert_eq!(text.trim(), "Hello!");
        assert_eq!(thinking.as_deref(), Some("The user wants a greeting."));
        assert!(s.delivered());
    }

    #[test]
    fn a_full_or_gone_reasoning_receiver_holds_back_nothing_else() {
        let frame = ReasoningFrame { open_at_start: true, ..ReasoningFrame::THINK };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        // Capacity 1 and never read: the second delta waits in the sink.
        let (rtx, mut rrx) = tokio::sync::mpsc::channel::<String>(1);
        let mut s = StopStream::new(vec![]).framed(frame).nonblocking().with_reasoning(Some(rtx));
        assert!(s.push("one ", Some(&tx)));
        assert!(s.push("two ", Some(&tx)));
        assert!(!s.delivered(), "the second delta is still queued");
        assert!(s.push("</think>answer", Some(&tx)), "the answer stream stays open");
        assert_eq!(rx.try_recv().as_deref(), Ok("answer"));
        assert_eq!(rrx.try_recv().as_deref(), Ok("one "));
        assert!(s.flush(Some(&tx)));
        assert_eq!(rrx.try_recv().as_deref(), Ok("two "));
        assert!(s.delivered());
        // A reasoning receiver that goes away ends only the reasoning stream.
        drop(rrx);
        let mut s2 = StopStream::new(vec![]).framed(frame).nonblocking();
        let (rtx2, rrx2) = tokio::sync::mpsc::channel::<String>(4);
        drop(rrx2);
        s2 = s2.with_reasoning(Some(rtx2));
        assert!(s2.push("thinking</think>ok", Some(&tx)));
        assert!(s2.delivered());
        assert_eq!(rx.try_recv().as_deref(), Ok("ok"));
    }

    #[test]
    fn qwen_prompt_with_thinking_off_is_not_open() {
        let prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let frame = ReasoningFrame::for_prompt(prompt);
        assert!(!frame.open_at_start);
        let (streamed, _, reasoning) = stream_all(frame, &["Hello", "!"]);
        assert_eq!(streamed, "Hello!");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn unclosed_span_at_the_token_limit_is_all_reasoning() {
        let frame = ReasoningFrame::for_prompt("<|im_start|>assistant\n<think>\n");
        let (streamed, text, reasoning) = stream_all(frame, &["still ", "thinking"]);
        assert_eq!(streamed, "");
        assert_eq!(text, "");
        assert_eq!(reasoning.as_deref(), Some("still thinking"));
    }

    #[test]
    fn gemma4_thought_channel_is_split() {
        let prompt = "<|turn>user\nhi<turn|>\n<|turn>model\n";
        let frame = ReasoningFrame::for_prompt(prompt);
        assert_eq!(frame.open, "<|channel>thought");
        assert!(!frame.open_at_start);
        let (streamed, text, reasoning) = stream_all(
            frame,
            &["<|channel>", "thought\nweigh it", "<channel|>", "Answer."],
        );
        assert_eq!(streamed, "Answer.");
        assert_eq!(text, "Answer.");
        assert_eq!(reasoning.as_deref(), Some("weigh it"));
    }

    #[test]
    fn gemma4_prompt_after_a_tool_response_starts_in_the_channel() {
        let prompt = "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>response:f{}<tool_response|><|channel>thought\n";
        let frame = ReasoningFrame::for_prompt(prompt);
        assert!(frame.open_at_start);
        let (streamed, _, reasoning) = stream_all(frame, &["the tool said yes", "<channel|>", "Yes."]);
        assert_eq!(streamed, "Yes.");
        assert_eq!(reasoning.as_deref(), Some("the tool said yes"));
    }

    #[test]
    fn gemma4_prompt_with_thinking_off_closes_the_empty_channel() {
        let frame = ReasoningFrame::for_prompt("<|turn>model\n<|channel>thought\n<channel|>");
        assert!(!frame.open_at_start);
    }

    #[test]
    fn a_full_channel_never_blocks_a_nonblocking_stream() {
        // Capacity 1 and nobody reading: a blocking stream would park here
        // forever on the second chunk.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);
        let mut s = StopStream::new(vec![]).nonblocking();
        for piece in ["one ", "two ", "three"] {
            assert!(s.push(piece, Some(&tx)), "an open receiver keeps the stream open");
        }
        assert!(!s.delivered(), "two chunks are still queued");
        let (text, _) = s.close(Some(&tx));
        assert_eq!(text, "one two three", "the result carries the whole text regardless");
        let mut got = String::new();
        while !s.delivered() {
            while let Ok(c) = rx.try_recv() {
                got.push_str(&c);
            }
            assert!(s.flush(Some(&tx)));
        }
        while let Ok(c) = rx.try_recv() {
            got.push_str(&c);
        }
        assert_eq!(got, "one two three", "every chunk arrives, in order");
    }

    #[test]
    fn a_receiver_that_stops_reading_is_stalled_not_waited_on() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<String>(1);
        let mut s = StopStream::new(vec![]).nonblocking();
        let piece = "x".repeat(64 * 1024);
        let mut open = true;
        for _ in 0..(super::STREAM_BACKLOG_LIMIT / piece.len() + 2) {
            open = s.push(&piece, Some(&tx));
            if !open {
                break;
            }
        }
        assert!(!open, "past the backlog limit the stream gives up on the receiver");
        assert!(s.stalled());
    }

    #[test]
    fn a_dropped_receiver_closes_a_nonblocking_stream() {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(4);
        drop(rx);
        let mut s = StopStream::new(vec![]).nonblocking();
        assert!(!s.push("hello", Some(&tx)));
        assert!(!s.stalled(), "gone is not the same as slow");
    }

    #[test]
    fn a_template_with_no_reasoning_passes_text_through() {
        let frame = ReasoningFrame::for_prompt("<|start|>assistant");
        let (streamed, _, reasoning) = stream_all(frame, &["plain ", "answer"]);
        assert_eq!(streamed, "plain answer");
        assert_eq!(reasoning, None);
    }
}
