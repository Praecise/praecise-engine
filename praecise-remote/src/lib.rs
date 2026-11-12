//! Generation on an inference API outside the machine.
//!
//! Some inference silicon is not sold as a device. Groq's LPU, Cerebras's
//! wafer-scale engine and SambaNova's RDU are reached over a network API and
//! no other way. This crate is the one route the engine has to them: a
//! client for an OpenAI-compatible chat-completions endpoint that answers in
//! the runtime's own types, so a remote model returns the same
//! [`InferenceResult`], token counts and stop reasons as a local one.
//!
//! ## Why it is a crate of its own
//!
//! A runtime host has no network, and that rule is not relaxed here. A
//! remote backend is not a host: nothing runs on this machine, and the
//! prompt leaves it. That is a different promise from every other backend,
//! so it is kept apart and has to be asked for twice:
//!
//! 1. **At build time.** Nothing in the engine opens a connection unless the
//!    application depends on this crate. `Backend::Remote` reports itself
//!    unavailable in a build without it.
//! 2. **At run time.** [`Remote::open`] refuses an endpoint whose host the
//!    operator has not named in an [`Allow`] list (`PRAECISE_REMOTE_ALLOW`).
//!    There is no default list, and a preset in [`Provider`] names an address,
//!    not a permission.
//!
//! The connection is HTTPS, except to the machine itself. The key is read
//! from the environment variable the endpoint names and is never logged. A
//! reply without token counts is refused, as a hosted engine's is, rather
//! than metered as free.
//!
//! ## What a remote backend cannot do
//!
//! The provider owns the decode loop, the weights and the hardware. There is
//! no engine-side speculation, no logit access, no inference commitment and
//! no measurement of the silicon behind the endpoint: what comes back is
//! text, counts and the time the request took, network included.

mod stream;

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use praecise_runtime::{ChatMessage, GenerationConfig, InferenceResult, StopReason};
use serde_json::{Value, json};

pub use stream::{Finish, Reply, Usage};

/// Environment variable naming the hosts prompts may be sent to: a
/// comma-separated list such as `api.groq.com,api.cerebras.ai`.
pub const ALLOW_ENV: &str = "PRAECISE_REMOTE_ALLOW";

/// Why a remote generation did not happen or did not finish.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The request is one this backend does not make.
    #[error("refused: {0}")]
    Refused(String),
    /// The endpoint could not be reached or the connection broke.
    #[error("{endpoint}: {cause}")]
    Transport {
        /// The endpoint's name.
        endpoint: String,
        /// What went wrong.
        cause: String,
    },
    /// The endpoint answered with an error status.
    #[error("{endpoint}: HTTP {status}: {body}")]
    Status {
        /// The endpoint's name.
        endpoint: String,
        /// The HTTP status code.
        status: u16,
        /// The start of the response body.
        body: String,
    },
    /// The endpoint's reply could not be read.
    #[error("{endpoint}: {cause}")]
    Reply {
        /// The endpoint's name.
        endpoint: String,
        /// What was wrong with it.
        cause: String,
    },
}

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// An inference provider whose silicon is reached only through its API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    /// Groq: the LPU.
    Groq,
    /// Cerebras: the wafer-scale engine.
    Cerebras,
    /// SambaNova: the reconfigurable dataflow unit.
    SambaNova,
}

impl Provider {
    /// Every preset.
    pub const ALL: &'static [Provider] = &[Provider::Groq, Provider::Cerebras, Provider::SambaNova];

    /// Stable lowercase identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Provider::Groq => "groq",
            Provider::Cerebras => "cerebras",
            Provider::SambaNova => "sambanova",
        }
    }

    /// The base URL of the provider's OpenAI-compatible API.
    #[must_use]
    pub const fn base_url(self) -> &'static str {
        match self {
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::Cerebras => "https://api.cerebras.ai/v1",
            Provider::SambaNova => "https://api.sambanova.ai/v1",
        }
    }

    /// The environment variable the provider's own tools read the key from.
    #[must_use]
    pub const fn key_env(self) -> &'static str {
        match self {
            Provider::Groq => "GROQ_API_KEY",
            Provider::Cerebras => "CEREBRAS_API_KEY",
            Provider::SambaNova => "SAMBANOVA_API_KEY",
        }
    }

    /// An endpoint for `model` on this provider.
    #[must_use]
    pub fn endpoint(self, model: impl Into<String>) -> Endpoint {
        Endpoint {
            name: self.as_str().to_string(),
            base_url: self.base_url().to_string(),
            key_env: self.key_env().to_string(),
            model: model.into(),
            max_tokens_field: "max_tokens".to_string(),
        }
    }
}

/// One model on one OpenAI-compatible API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// A name for logs and errors.
    pub name: String,
    /// The API's base URL, up to and excluding `/chat/completions`.
    pub base_url: String,
    /// The environment variable holding the API key. Empty for an endpoint
    /// that takes none.
    pub key_env: String,
    /// The model to ask for.
    pub model: String,
    /// The request field that caps the reply's length: `max_tokens`, or
    /// `max_completion_tokens` on an API that has moved to it.
    pub max_tokens_field: String,
}

impl Endpoint {
    /// The host the endpoint's requests go to, lowercased, without a port.
    ///
    /// # Errors
    /// When the base URL is not an `http` or `https` URL with a host.
    pub fn host(&self) -> Result<String> {
        let (_, host) = self.scheme_and_host()?;
        Ok(host)
    }

    fn scheme_and_host(&self) -> Result<(&str, String)> {
        let bad = || Error::Refused(format!("{}: {:?} is not an http(s) URL with a host", self.name, self.base_url));
        let (scheme, rest) = self.base_url.split_once("://").ok_or_else(bad)?;
        if scheme != "https" && scheme != "http" {
            return Err(bad());
        }
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        // credentials in a URL would end up in logs; they are not accepted
        if authority.contains('@') {
            return Err(Error::Refused(format!("{}: the base URL carries credentials", self.name)));
        }
        let host = if let Some(bracketed) = authority.strip_prefix('[') {
            bracketed.split(']').next().unwrap_or("")
        } else {
            authority.rsplit_once(':').map_or(authority, |(h, _)| h)
        };
        if host.is_empty() {
            return Err(bad());
        }
        Ok((scheme, host.to_ascii_lowercase()))
    }
}

/// The hosts the operator permits prompts to be sent to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Allow {
    hosts: Vec<String>,
}

impl Allow {
    /// Permit exactly these hosts.
    #[must_use]
    pub fn hosts<S: AsRef<str>>(hosts: impl IntoIterator<Item = S>) -> Self {
        let mut hosts: Vec<String> =
            hosts.into_iter().map(|h| h.as_ref().trim().to_ascii_lowercase()).filter(|h| !h.is_empty()).collect();
        hosts.sort();
        hosts.dedup();
        Self { hosts }
    }

    /// The hosts named in [`ALLOW_ENV`]; none when it is unset.
    #[must_use]
    pub fn from_env() -> Self {
        std::env::var(ALLOW_ENV).map(|v| Self::hosts(v.split(','))).unwrap_or_default()
    }

    /// Whether `host` is permitted. A host is matched whole: permitting
    /// `groq.com` does not permit `api.groq.com`.
    #[must_use]
    pub fn permits(&self, host: &str) -> bool {
        self.hosts.iter().any(|h| h == &host.to_ascii_lowercase())
    }
}

/// Whether `host` is this machine.
fn is_loopback(host: &str) -> bool {
    host == "localhost" || host == "::1" || host.starts_with("127.")
}

/// A model on a remote endpoint the operator has permitted.
#[derive(Debug)]
pub struct Remote {
    endpoint: Endpoint,
    key: Option<String>,
    agent: ureq::Agent,
}

impl Remote {
    /// Open `endpoint`, which sends nothing yet.
    ///
    /// # Errors
    /// [`Error::Refused`] when the endpoint's host is not in `allow`, the
    /// URL is not HTTPS (plain HTTP is accepted only to this machine), or the
    /// key variable the endpoint names is unset.
    pub fn open(endpoint: Endpoint, allow: &Allow) -> Result<Self> {
        let (scheme, host) = endpoint.scheme_and_host()?;
        if !allow.permits(&host) {
            return Err(Error::Refused(format!(
                "{}: prompts would leave this machine for {host}, which is not permitted; name it in {ALLOW_ENV} to allow that",
                endpoint.name
            )));
        }
        if scheme != "https" && !is_loopback(&host) {
            return Err(Error::Refused(format!("{}: {host} is reached without TLS", endpoint.name)));
        }
        let key = if endpoint.key_env.is_empty() {
            None
        } else {
            Some(std::env::var(&endpoint.key_env).map_err(|_| {
                Error::Refused(format!("{}: the API key variable {} is not set", endpoint.name, endpoint.key_env))
            })?)
        };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(20)))
            .build()
            .into();
        Ok(Self { endpoint, key, agent })
    }

    /// The endpoint this model is on.
    #[must_use]
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Generate a reply to `messages`, passing each streamed piece of the
    /// answer to `on_text`. `on_text` returns false when its caller has
    /// gone, which abandons the request. Blocking.
    ///
    /// # Errors
    /// A request for an inference commitment, which a remote model cannot
    /// produce; a transport failure; an error status; an abandoned request;
    /// a reply that cannot be read or carries no token usage.
    pub fn generate(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<InferenceResult> {
        let started = Instant::now();
        let body = request(&self.endpoint, messages, config)?;
        let transport = |cause: String| Error::Transport { endpoint: self.endpoint.name.clone(), cause };
        let url = format!("{}/chat/completions", self.endpoint.base_url.trim_end_matches('/'));
        let mut call = self.agent.post(&url).header("Content-Type", "application/json").header("Accept", "text/event-stream");
        if let Some(key) = &self.key {
            call = call.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = call.send(body.to_string()).map_err(|e| transport(e.to_string()))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.body_mut().read_to_string().unwrap_or_default();
            return Err(Error::Status {
                endpoint: self.endpoint.name.clone(),
                status,
                body: body.chars().take(500).collect(),
            });
        }
        let mut reply = Reply::default();
        for line in BufReader::new(response.body_mut().as_reader()).lines() {
            let line = line.map_err(|e| transport(e.to_string()))?;
            let piece = reply
                .push_line(&line)
                .map_err(|cause| Error::Reply { endpoint: self.endpoint.name.clone(), cause })?;
            if let Some(piece) = piece {
                if !piece.is_empty() && !on_text(&piece) {
                    return Err(transport("the caller went away".to_string()));
                }
            }
            if reply.done() {
                break;
            }
        }
        result(&self.endpoint, reply, config, started.elapsed())
    }
}

/// The chat-completions request for `messages` under `config`.
fn request(endpoint: &Endpoint, messages: &[ChatMessage], config: &GenerationConfig) -> Result<Value> {
    if config.commitment_k.is_some() {
        return Err(Error::Refused("a remote model does not produce inference commitments".into()));
    }
    if messages.is_empty() {
        return Err(Error::Refused("a remote model is asked with chat turns, and none were given".into()));
    }
    let messages = serde_json::to_value(messages).map_err(|e| Error::Refused(format!("the chat turns: {e}")))?;
    let mut req = json!({
        "model": endpoint.model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
    });
    req[endpoint.max_tokens_field.as_str()] = config.max_tokens.into();
    if !config.stop.is_empty() {
        req["stop"] = config.stop.clone().into();
    }
    if let Some(format) = &config.response_format {
        req["response_format"] = format.clone();
    }
    Ok(req)
}

/// The finished stream read back as an [`InferenceResult`].
fn result(endpoint: &Endpoint, reply: Reply, config: &GenerationConfig, elapsed: Duration) -> Result<InferenceResult> {
    let usage = reply.usage.ok_or_else(|| Error::Reply {
        endpoint: endpoint.name.clone(),
        cause: "the reply carried no token usage; it is refused rather than metered as free".to_string(),
    })?;
    let output_tokens = u32::try_from(usage.completion_tokens).unwrap_or(u32::MAX);
    let stop_reason = match reply.finish {
        Some(Finish::Length) => StopReason::Length,
        _ if output_tokens >= config.max_tokens => StopReason::Length,
        _ => StopReason::Eos,
    };
    Ok(InferenceResult {
        text: reply.text,
        thinking: if reply.thinking.is_empty() { None } else { Some(reply.thinking) },
        input_tokens: u32::try_from(usage.prompt_tokens).unwrap_or(u32::MAX),
        output_tokens,
        generation_time_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        tokens_per_second: if elapsed.is_zero() { 0.0 } else { f64::from(output_tokens) / elapsed.as_secs_f64() },
        stop_reason,
        commitment: None,
        cached_tokens: 0,
        forward_positions: None,
        denoise_steps: None,
        tool_calls: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    fn local(model: &str, port: u16) -> Endpoint {
        Endpoint {
            name: "local".into(),
            base_url: format!("http://127.0.0.1:{port}/v1"),
            key_env: String::new(),
            model: model.into(),
            max_tokens_field: "max_tokens".into(),
        }
    }

    #[test]
    fn nothing_is_permitted_until_the_operator_names_it() {
        let groq = Provider::Groq.endpoint("llama");
        let e = Remote::open(groq.clone(), &Allow::default()).unwrap_err().to_string();
        assert!(e.contains("api.groq.com") && e.contains(ALLOW_ENV), "{e}");
        // a preset is an address, not a permission; neither is a parent domain
        assert!(Remote::open(groq.clone(), &Allow::hosts(["groq.com", "api.cerebras.ai"])).is_err());
        assert!(Allow::hosts([" API.Groq.com "]).permits("api.groq.com"));
    }

    #[test]
    fn a_permitted_endpoint_still_needs_tls_and_its_key() {
        let plain = Endpoint { base_url: "http://api.example.com/v1".into(), ..local("m", 1) };
        let e = Remote::open(plain, &Allow::hosts(["api.example.com"])).unwrap_err().to_string();
        assert!(e.contains("without TLS"), "{e}");
        let keyed = Endpoint { key_env: "PRAECISE_REMOTE_TEST_KEY_THAT_IS_NOT_SET".into(), ..Provider::Cerebras.endpoint("m") };
        let e = Remote::open(keyed, &Allow::hosts(["api.cerebras.ai"])).unwrap_err().to_string();
        assert!(e.contains("is not set"), "{e}");
        let creds = Endpoint { base_url: "https://user:pw@api.example.com/v1".into(), ..local("m", 1) };
        assert!(Remote::open(creds, &Allow::hosts(["api.example.com"])).unwrap_err().to_string().contains("credentials"));
    }

    #[test]
    fn hosts_are_read_from_the_url() {
        for (url, want) in [
            ("https://api.groq.com/openai/v1", "api.groq.com"),
            ("https://API.Cerebras.ai:443/v1", "api.cerebras.ai"),
            ("http://127.0.0.1:8080/v1", "127.0.0.1"),
            ("http://[::1]:8080/v1", "::1"),
        ] {
            assert_eq!(Endpoint { base_url: url.into(), ..local("m", 1) }.host().unwrap(), want, "{url}");
        }
        for url in ["api.groq.com/v1", "ftp://api.groq.com", "https:///v1"] {
            assert!(Endpoint { base_url: url.into(), ..local("m", 1) }.host().is_err(), "{url}");
        }
        for p in Provider::ALL {
            assert!(p.base_url().starts_with("https://"));
            assert!(p.endpoint("m").host().is_ok());
        }
    }

    #[test]
    fn the_request_carries_the_sampling_the_caller_asked_for() {
        let config = GenerationConfig {
            max_tokens: 16,
            temperature: 0.2,
            top_p: 0.9,
            seed: 7,
            stop: vec!["END".into()],
            ..Default::default()
        };
        let req = request(&Provider::Groq.endpoint("llama"), &[ChatMessage::new("user", "hi")], &config).unwrap();
        assert_eq!(req["model"], "llama");
        assert_eq!(req["messages"][0]["content"], "hi");
        assert_eq!((req["max_tokens"].as_u64(), req["seed"].as_u64()), (Some(16), Some(7)));
        assert_eq!(req["stream"], true);
        assert_eq!(req["stream_options"]["include_usage"], true);
        assert_eq!(req["stop"][0], "END");
        let newer = Endpoint { max_tokens_field: "max_completion_tokens".into(), ..Provider::Groq.endpoint("m") };
        let req = request(&newer, &[ChatMessage::new("user", "hi")], &config).unwrap();
        assert_eq!(req["max_completion_tokens"], 16);
        assert!(req.get("max_tokens").is_none());
    }

    #[test]
    fn a_commitment_and_an_empty_chat_are_refused() {
        let endpoint = Provider::Groq.endpoint("m");
        let config = GenerationConfig { commitment_k: Some(4), ..Default::default() };
        assert!(request(&endpoint, &[ChatMessage::new("user", "x")], &config).is_err());
        assert!(request(&endpoint, &[], &GenerationConfig::default()).is_err());
    }

    /// Serve one HTTP response on a loopback port and return that port and
    /// the request the client sent.
    fn serve_once(status: &str, body: &'static str) -> (u16, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let head = format!("HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
        let handle = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut seen = Vec::new();
            let mut buf = [0u8; 4096];
            // read the head, then as much body as its Content-Length says
            loop {
                let n = socket.read(&mut buf).unwrap();
                seen.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&seen).to_string();
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let want = head
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                        .unwrap_or(0);
                    if body.len() >= want {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            socket.write_all(head.as_bytes()).unwrap();
            socket.write_all(body.as_bytes()).unwrap();
            String::from_utf8_lossy(&seen).to_string()
        });
        (port, handle)
    }

    const STREAM: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n\
        data: {\"choices\":[{\"delta\":{\"reasoning\":\"thinking\"}}]}\n\n\
        data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
        data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
        data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\n\
        data: [DONE]\n\n";

    #[test]
    fn a_reply_streams_and_reads_back_as_an_inference_result() {
        let (port, server) = serve_once("200 OK", STREAM);
        let remote = Remote::open(local("test-model", port), &Allow::hosts(["127.0.0.1"])).unwrap();
        let mut pieces = Vec::new();
        let config = GenerationConfig { max_tokens: 32, ..Default::default() };
        let r = remote
            .generate(&[ChatMessage::new("user", "hi")], &config, |p| {
                pieces.push(p.to_string());
                true
            })
            .unwrap();
        assert_eq!(pieces, ["Hel", "lo"]);
        assert_eq!((r.text.as_str(), r.input_tokens, r.output_tokens), ("Hello", 9, 2));
        assert_eq!(r.thinking.as_deref(), Some("thinking"));
        assert_eq!(r.stop_reason, StopReason::Eos);
        let sent = server.join().unwrap();
        assert!(sent.starts_with("POST /v1/chat/completions "), "{sent}");
        assert!(sent.contains("\"model\":\"test-model\""), "{sent}");
        assert!(!sent.to_ascii_lowercase().contains("authorization"), "an endpoint without a key sends none");
    }

    #[test]
    fn an_error_status_and_a_reply_without_usage_are_refused() {
        let (port, server) = serve_once("429 Too Many Requests", "{\"error\":{\"message\":\"rate limited\"}}");
        let remote = Remote::open(local("m", port), &Allow::hosts(["127.0.0.1"])).unwrap();
        let e = remote.generate(&[ChatMessage::new("user", "hi")], &GenerationConfig::default(), |_| true).unwrap_err();
        assert!(matches!(e, Error::Status { status: 429, .. }), "{e}");
        assert!(e.to_string().contains("rate limited"));
        server.join().unwrap();

        let (port, server) = serve_once("200 OK", "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n");
        let remote = Remote::open(local("m", port), &Allow::hosts(["127.0.0.1"])).unwrap();
        let e = remote.generate(&[ChatMessage::new("user", "hi")], &GenerationConfig::default(), |_| true).unwrap_err();
        assert!(e.to_string().contains("no token usage"), "{e}");
        server.join().unwrap();
    }

    #[test]
    fn a_caller_that_goes_away_abandons_the_request() {
        let (port, server) = serve_once("200 OK", STREAM);
        let remote = Remote::open(local("m", port), &Allow::hosts(["127.0.0.1"])).unwrap();
        let e = remote.generate(&[ChatMessage::new("user", "hi")], &GenerationConfig::default(), |_| false).unwrap_err();
        assert!(e.to_string().contains("went away"), "{e}");
        server.join().unwrap();
    }
}
