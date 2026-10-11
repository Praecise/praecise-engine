//! Streaming stop sequences and the reading of a reply in its chat format.
//!
//! Backend-free: this is text bookkeeping over decoded pieces, so it compiles
//! and runs without a bundled backend. It accumulates decoded token pieces,
//! detects stop sequences (which may span several tokens), and, when the
//! generation has a chat format, hands the text to that format's
//! [`ReplyReader`], which splits it into answer, reasoning and tool calls.
//! When the caller is streaming, bytes are released only once they can no
//! longer turn out to be the leading part of a stop sequence.

use std::sync::Arc;

use crate::result::{ParsedToolCall, ReplyDelta};

/// Byte length of the longest stop sequence that `text` ends with, or `None`
/// when no stop sequence matched. The caller truncates by that many bytes so
/// the delimiter never reaches the client.
pub fn matched_stop_len(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter(|s| !s.is_empty() && text.ends_with(s.as_str()))
        .map(|s| s.len())
        .max()
}

/// Reads one generated reply in its chat format, as it arrives.
pub trait ReplyReader: Send {
    /// Read the next generated text.
    ///
    /// # Errors
    /// Text the format cannot read.
    fn read(&mut self, text: &str) -> crate::Result<Vec<ReplyDelta>>;

    /// The reply is complete.
    ///
    /// # Errors
    /// As [`read`](Self::read).
    fn finish(&mut self) -> crate::Result<Vec<ReplyDelta>>;
}

/// The chat format a generation is read in.
pub trait ReplyFormat: Send + Sync + std::fmt::Debug {
    /// A reader for one reply.
    ///
    /// # Errors
    /// A format whose reader cannot be built.
    fn reader(&self) -> crate::Result<Box<dyn ReplyReader>>;

    /// The text that ends a reasoning block, for an engine that has to end
    /// one itself. `None` for a format without reasoning blocks.
    fn reasoning_end(&self) -> Option<&str>;
}

/// A finished reply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reply {
    /// The answer.
    pub text: String,
    /// The reasoning, when there was any.
    pub thinking: Option<String>,
    /// The tool calls, in the order written.
    pub tool_calls: Vec<ParsedToolCall>,
}

/// A tool call being assembled from streamed steps.
#[derive(Default)]
struct CallParts {
    id: String,
    name: String,
    arguments: String,
}

/// Accumulates decoded token pieces, detects stop sequences, reads the reply
/// in its chat format, and — when the caller is streaming — releases answer
/// text only once it can no longer turn out to be the leading part of a stop
/// sequence.
///
/// A stop sequence may span several tokens, so the last `hold` bytes are kept
/// back until either more text disambiguates them or generation ends. With no
/// stop sequences configured `hold` is zero and every piece is read the moment
/// it is decoded.
pub struct StopStream {
    /// Everything decoded, less a matched stop sequence.
    raw: String,
    /// Bytes of `raw` handed to the reader.
    read: usize,
    /// The answer.
    text: String,
    /// The reasoning.
    reasoning: String,
    calls: Vec<CallParts>,
    /// Whether the text read most recently was reasoning.
    in_reasoning: bool,
    reader: Option<Box<dyn ReplyReader>>,
    /// The reader's refusal, reported when the stream closes.
    failed: Option<crate::Error>,
    emitted: usize,
    hold: usize,
    stop: Vec<String>,
    hit: bool,
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
    /// Build a stream that trims the given stop sequences and passes the
    /// text through as the answer.
    pub fn new(stop: Vec<String>) -> Self {
        let hold = stop
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        Self {
            raw: String::new(),
            read: 0,
            text: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            in_reasoning: false,
            reader: None,
            failed: None,
            emitted: 0,
            hold,
            stop,
            hit: false,
            outbox: None,
            reasoning_sink: None,
        }
    }

    /// Read the text in `format`, when the generation has one: the answer,
    /// reasoning and tool calls are split by the format rather than passed
    /// through as text. A format whose reader cannot be built fails the
    /// stream: the first [`push`](Self::push) stops the generation and
    /// [`close`](Self::close) reports why.
    pub fn reading(mut self, format: Option<&Arc<dyn ReplyFormat>>) -> Self {
        match format.map(|f| f.reader()).transpose() {
            Ok(reader) => self.reader = reader,
            Err(e) => self.failed = Some(e),
        }
        self
    }

    /// Continue a reply another stream began: `generated` is everything it
    /// decoded. The reader is brought to where that text left it, and none of
    /// it is reported again.
    pub fn continuing(mut self, generated: &str) -> Self {
        self.raw = generated.to_string();
        self.read = self.raw.len();
        if let Some(reader) = self.reader.as_mut() {
            match reader.read(generated) {
                Ok(deltas) => {
                    if let Some(last) = deltas.last() {
                        self.in_reasoning = matches!(last, ReplyDelta::Reasoning(_));
                    }
                }
                Err(e) => self.failed = Some(e),
            }
        }
        self
    }

    /// Everything decoded so far, less a matched stop sequence.
    pub fn generated(&self) -> &str {
        &self.raw
    }

    /// Stream the reasoning to `tx` as it is produced, separately from the
    /// visible text. Without this, reasoning is only returned at the end.
    pub fn with_reasoning(mut self, tx: Option<tokio::sync::mpsc::Sender<String>>) -> Self {
        self.reasoning_sink = tx.map(|tx| ReasoningSink {
            tx,
            unsent: String::new(),
        });
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
        let Some(sink) = self.reasoning_sink.as_mut() else {
            return;
        };
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
            && self
                .reasoning_sink
                .as_ref()
                .is_none_or(|r| r.unsent.is_empty())
    }

    /// Whether the receiver has fallen [`STREAM_BACKLOG_LIMIT`] bytes behind.
    pub fn stalled(&self) -> bool {
        self.outbox
            .as_ref()
            .is_some_and(|o| o.bytes > STREAM_BACKLOG_LIMIT)
            || self
                .reasoning_sink
                .as_ref()
                .is_some_and(|r| r.unsent.len() > STREAM_BACKLOG_LIMIT)
    }

    /// Whether the text arriving now is reasoning.
    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    /// Absorb one decoded piece. Returns `false` when the stream receiver has
    /// been dropped, or the reader refused the text, which the generation
    /// loops treat as "stop generating".
    pub fn push(&mut self, piece: &str, tx: Option<&tokio::sync::mpsc::Sender<String>>) -> bool {
        self.raw.push_str(piece);
        if let Some(n) = matched_stop_len(&self.raw, &self.stop) {
            self.raw.truncate(self.raw.len() - n);
            self.hit = true;
        }
        let mut boundary = if self.hit {
            self.raw.len()
        } else {
            self.raw.len().saturating_sub(self.hold)
        };
        while boundary > self.read && !self.raw.is_char_boundary(boundary) {
            boundary -= 1;
        }
        if boundary > self.read {
            let settled = self.raw[self.read..boundary].to_string();
            self.read = boundary;
            self.absorb(&settled, false);
        }
        self.release(tx) && self.failed.is_none()
    }

    /// [`push`](Self::push) for a caller that takes the answer itself rather
    /// than through a channel: the answer text this piece settled, or `None`
    /// once the reader has refused the text.
    pub fn push_text(&mut self, piece: &str) -> Option<String> {
        self.push(piece, None);
        if self.failed.is_some() {
            return None;
        }
        let settled = self.text[self.emitted..].to_string();
        self.emitted = self.text.len();
        Some(settled)
    }

    /// Pass settled text to the reader, or to the answer when there is none.
    fn absorb(&mut self, settled: &str, done: bool) {
        if self.failed.is_some() {
            return;
        }
        let Some(reader) = self.reader.as_mut() else {
            self.text.push_str(settled);
            return;
        };
        let mut deltas = if settled.is_empty() {
            Ok(Vec::new())
        } else {
            reader.read(settled)
        };
        if done && let Ok(d) = deltas.as_mut() {
            match reader.finish() {
                Ok(rest) => d.extend(rest),
                Err(e) => deltas = Err(e),
            }
        }
        match deltas {
            Ok(deltas) => deltas.into_iter().for_each(|d| self.apply(d)),
            Err(e) => self.failed = Some(e),
        }
    }

    fn apply(&mut self, delta: ReplyDelta) {
        match delta {
            ReplyDelta::Content(c) => {
                self.in_reasoning = false;
                self.text.push_str(&c);
            }
            ReplyDelta::Reasoning(r) => {
                self.in_reasoning = true;
                self.think(&r);
            }
            ReplyDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => {
                self.in_reasoning = false;
                if self.calls.len() <= index {
                    self.calls.resize_with(index + 1, CallParts::default);
                }
                let call = &mut self.calls[index];
                if let Some(id) = id {
                    call.id = id;
                }
                if let Some(name) = name {
                    call.name.push_str(&name);
                }
                call.arguments.push_str(&arguments);
            }
        }
    }

    fn release(&mut self, tx: Option<&tokio::sync::mpsc::Sender<String>>) -> bool {
        self.send_reasoning();
        let Some(tx) = tx else { return true };
        if self.text.len() <= self.emitted {
            return true;
        }
        let chunk = self.text[self.emitted..].to_string();
        self.emitted = self.text.len();
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

    /// Release anything still held back, then hand over the reply.
    ///
    /// # Errors
    /// Text the reply's format could not read.
    pub fn finish_parts(
        mut self,
        tx: Option<&tokio::sync::mpsc::Sender<String>>,
    ) -> crate::Result<Reply> {
        self.close(tx)
    }

    /// [`Self::finish_parts`] without consuming the stream, so a non-blocking
    /// stream keeps what the receiver has not taken yet for later
    /// [`Self::flush`]es.
    ///
    /// # Errors
    /// As [`Self::finish_parts`].
    pub fn close(
        &mut self,
        tx: Option<&tokio::sync::mpsc::Sender<String>>,
    ) -> crate::Result<Reply> {
        let rest = self.raw[self.read..].to_string();
        self.read = self.raw.len();
        self.absorb(&rest, true);
        self.hit = true;
        self.release(tx);
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        let reasoning = self.reasoning.trim().to_string();
        self.reasoning.clear();
        self.emitted = 0;
        let tool_calls = std::mem::take(&mut self.calls)
            .into_iter()
            .filter(|c| !c.name.is_empty())
            .map(|c| ParsedToolCall {
                id: if c.id.is_empty() {
                    crate::result::new_call_id()
                } else {
                    c.id
                },
                name: c.name,
                arguments: crate::result::call_arguments(&c.arguments),
            })
            .collect();
        Ok(Reply {
            text: std::mem::take(&mut self.text),
            thinking: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ReplyFormat, ReplyReader, StopStream};
    use crate::result::ReplyDelta;

    /// Reads `<r>`…`</r>` as reasoning and `@name(args)` as a call; enough to
    /// check the routing without a backend.
    #[derive(Debug)]
    struct Tagged;

    struct TaggedReader {
        buf: String,
        in_r: bool,
        calls: usize,
    }

    impl ReplyFormat for Tagged {
        fn reader(&self) -> crate::Result<Box<dyn ReplyReader>> {
            Ok(Box::new(TaggedReader {
                buf: String::new(),
                in_r: false,
                calls: 0,
            }))
        }
        fn reasoning_end(&self) -> Option<&str> {
            Some("</r>")
        }
    }

    impl TaggedReader {
        fn drain(&mut self, done: bool) -> Vec<ReplyDelta> {
            let mut out = Vec::new();
            loop {
                if self.in_r {
                    if let Some(i) = self.buf.find("</r>") {
                        out.push(ReplyDelta::Reasoning(self.buf[..i].to_string()));
                        self.buf.drain(..i + 4);
                        self.in_r = false;
                        continue;
                    }
                } else if let Some(i) = self.buf.find("<r>") {
                    out.push(ReplyDelta::Content(self.buf[..i].to_string()));
                    self.buf.drain(..i + 3);
                    self.in_r = true;
                    continue;
                } else if let (Some(a), Some(b)) = (self.buf.find('@'), self.buf.find(')')) {
                    let open = self.buf.find('(').unwrap();
                    out.push(ReplyDelta::Content(self.buf[..a].to_string()));
                    out.push(ReplyDelta::ToolCall {
                        index: self.calls,
                        id: None,
                        name: Some(self.buf[a + 1..open].to_string()),
                        arguments: self.buf[open + 1..b].to_string(),
                    });
                    self.calls += 1;
                    self.buf.drain(..=b);
                    continue;
                }
                if done {
                    let rest = std::mem::take(&mut self.buf);
                    out.push(if self.in_r {
                        ReplyDelta::Reasoning(rest)
                    } else {
                        ReplyDelta::Content(rest)
                    });
                }
                return out;
            }
        }
    }

    impl ReplyReader for TaggedReader {
        fn read(&mut self, text: &str) -> crate::Result<Vec<ReplyDelta>> {
            self.buf.push_str(text);
            Ok(self.drain(false))
        }
        fn finish(&mut self) -> crate::Result<Vec<ReplyDelta>> {
            Ok(self.drain(true))
        }
    }

    fn tagged() -> Arc<dyn ReplyFormat> {
        Arc::new(Tagged)
    }

    #[test]
    fn without_a_format_the_text_is_the_answer() {
        let mut s = StopStream::new(vec!["END".into()]);
        for p in ["<r>a", "</r>b", "EN", "D"] {
            s.push(p, None);
        }
        let r = s.finish_parts(None).unwrap();
        assert_eq!(r.text, "<r>a</r>b");
        assert_eq!(r.thinking, None);
    }

    #[test]
    fn a_format_splits_reasoning_answer_and_calls() {
        let mut s = StopStream::new(vec![]).reading(Some(&tagged()));
        for p in ["<r>wei", "gh</r>ok ", "@get(", "{\"a\":1})"] {
            s.push(p, None);
        }
        let r = s.finish_parts(None).unwrap();
        assert_eq!(r.thinking.as_deref(), Some("weigh"));
        assert_eq!(r.text, "ok ");
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "get");
        assert_eq!(r.tool_calls[0].arguments, serde_json::json!({"a": 1}));
        assert!(!r.tool_calls[0].id.is_empty());
    }

    #[test]
    fn a_stop_sequence_is_matched_before_the_format_reads() {
        let mut s = StopStream::new(vec!["<|end|>".into()]).reading(Some(&tagged()));
        for p in ["<r>x</r>an", "swer<|en", "d|>"] {
            s.push(p, None);
        }
        assert!(s.hit_stop());
        let r = s.finish_parts(None).unwrap();
        assert_eq!(r.text, "answer");
        assert_eq!(r.thinking.as_deref(), Some("x"));
    }

    #[test]
    fn streamed_answer_and_reasoning_go_to_their_own_channels() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let (rtx, mut rrx) = tokio::sync::mpsc::channel(64);
        let mut s = StopStream::new(vec![])
            .reading(Some(&tagged()))
            .with_reasoning(Some(rtx))
            .nonblocking();
        for p in ["<r>th", "ink</r>", "hel", "lo"] {
            assert!(s.push(p, Some(&tx)));
        }
        let r = s.close(Some(&tx)).unwrap();
        let mut answer = String::new();
        while let Ok(c) = rx.try_recv() {
            answer.push_str(&c);
        }
        let mut reasoning = String::new();
        while let Ok(c) = rrx.try_recv() {
            reasoning.push_str(&c);
        }
        assert_eq!(answer, "hello");
        assert_eq!(reasoning, "think");
        assert_eq!(r.text, "hello");
    }
}
