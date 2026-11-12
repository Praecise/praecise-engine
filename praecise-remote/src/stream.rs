//! Reading a chat-completions event stream.
//!
//! The stream is server-sent events: `data: {json}` lines, a blank line
//! between events, `data: [DONE]` at the end. Each event carries a delta of
//! the answer, and one of them carries the token usage. Reading it is a pure
//! function of the lines, tested without a connection.

use serde_json::Value;

/// Why the model stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// It finished, or met a stop sequence.
    Stop,
    /// It reached the length limit.
    Length,
    /// Any other reason the provider gives (a tool call, a content filter).
    Other,
}

/// Token counts of one reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u64,
    /// Tokens in the reply.
    pub completion_tokens: u64,
}

/// A reply as it accumulates from the stream.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reply {
    /// The answer so far.
    pub text: String,
    /// The reasoning so far, on a model that streams it apart.
    pub thinking: String,
    /// Why the model stopped, once it has.
    pub finish: Option<Finish>,
    /// The token usage, once the stream has carried it.
    pub usage: Option<Usage>,
    done: bool,
}

impl Reply {
    /// Whether the stream has ended.
    #[must_use]
    pub fn done(&self) -> bool {
        self.done
    }

    /// Take one line of the stream. Returns the piece of the answer it
    /// carried, if any.
    ///
    /// # Errors
    /// When a `data:` line is not JSON, or the provider reports an error in
    /// the stream.
    pub fn push_line(&mut self, line: &str) -> Result<Option<String>, String> {
        let line = line.trim_end_matches('\r');
        // comments, event names and blank separators carry nothing
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(None);
        };
        let data = data.trim();
        if data == "[DONE]" {
            self.done = true;
            return Ok(None);
        }
        let event: Value =
            serde_json::from_str(data).map_err(|e| format!("an event that is not JSON ({e})"))?;
        if let Some(error) = event.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| error.to_string(), str::to_string);
            return Err(format!("the provider reported an error: {message}"));
        }
        // Usage arrives on a final event of its own, on the event that ends
        // the choice, or under a provider's own key.
        for usage in [event.get("usage"), event.pointer("/x_groq/usage")]
            .into_iter()
            .flatten()
        {
            if let (Some(prompt), Some(completion)) = (
                usage.get("prompt_tokens").and_then(Value::as_u64),
                usage.get("completion_tokens").and_then(Value::as_u64),
            ) {
                self.usage = Some(Usage {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                });
            }
        }
        let Some(choice) = event.pointer("/choices/0") else {
            return Ok(None);
        };
        match choice.get("finish_reason").and_then(Value::as_str) {
            Some("stop") => self.finish = Some(Finish::Stop),
            Some("length") => self.finish = Some(Finish::Length),
            Some(_) => self.finish = Some(Finish::Other),
            None => {}
        }
        let delta = choice.get("delta");
        for key in ["reasoning", "reasoning_content"] {
            if let Some(piece) = delta.and_then(|d| d.get(key)).and_then(Value::as_str) {
                self.thinking.push_str(piece);
            }
        }
        let piece = delta
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.text.push_str(piece);
        Ok(Some(piece.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(lines: &[&str]) -> Reply {
        let mut reply = Reply::default();
        for line in lines {
            reply.push_line(line).unwrap();
        }
        reply
    }

    #[test]
    fn deltas_accumulate_and_the_stream_ends() {
        let reply = read(&[
            ": keep-alive",
            "event: message",
            r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#,
            "",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"length\"}]}\r",
            r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
            "data: [DONE]",
        ]);
        assert_eq!(reply.text, "Hello");
        assert_eq!(reply.finish, Some(Finish::Length));
        assert_eq!(
            reply.usage,
            Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 2
            })
        );
        assert!(reply.done());
    }

    #[test]
    fn usage_is_found_wherever_the_provider_puts_it() {
        let on_last_choice = read(&[
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#,
        ]);
        assert_eq!(
            on_last_choice.usage,
            Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 1
            })
        );
        let vendor_key = read(&[
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":4,"completion_tokens":6}}}"#,
        ]);
        assert_eq!(
            vendor_key.usage,
            Some(Usage {
                prompt_tokens: 4,
                completion_tokens: 6
            })
        );
        // a null usage on the early events is not a count
        let null = read(&[r#"data: {"choices":[{"delta":{"content":"x"}}],"usage":null}"#]);
        assert_eq!(null.usage, None);
    }

    #[test]
    fn reasoning_is_kept_apart_from_the_answer() {
        let reply = read(&[
            r#"data: {"choices":[{"delta":{"reasoning":"a"}}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"b"}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"c"}}]}"#,
        ]);
        assert_eq!((reply.thinking.as_str(), reply.text.as_str()), ("ab", "c"));
    }

    #[test]
    fn a_broken_event_and_a_provider_error_are_errors() {
        let mut reply = Reply::default();
        assert!(
            reply
                .push_line("data: {not json")
                .unwrap_err()
                .contains("not JSON")
        );
        let e = reply
            .push_line(r#"data: {"error":{"message":"model overloaded"}}"#)
            .unwrap_err();
        assert!(e.contains("model overloaded"), "{e}");
        assert_eq!(reply.push_line("data: [DONE]"), Ok(None));
    }
}
