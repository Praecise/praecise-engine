//! Replies of each model family read back through the parser built from that
//! family's own chat template. The templates are the ones the backend ships in
//! `models/templates`; each reply is text in the form the family's models write
//! it, so no model is loaded.
#![cfg(feature = "bundled-llama")]

use praecise_runtime::{
    ChatFormat, ChatMessage, ChatTemplateSource, ChatTurn, ParsedReply, ReplyDelta,
};
use serde_json::{Value, json};

fn template(file: &str) -> ChatTemplateSource {
    let path = format!(
        "{}/../llama-cpp-sys-2/llama.cpp/models/templates/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    ChatTemplateSource::new(text, "", "", false)
}

fn tool(name: &str, param: &str, kind: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": format!("The {name} tool."),
            "parameters": {
                "type": "object",
                "properties": { param: { "type": kind } },
                "required": [param],
            },
        },
    })
}

fn get_time() -> Value {
    tool("get_time", "city", "string")
}

fn special_function() -> Value {
    tool("special_function", "arg1", "integer")
}

fn user() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: "What time is it in Tokyo?".into(),
        ..Default::default()
    }]
}

/// Render a turn offering `tools` and read `reply` back.
fn read(file: &str, tools: &[Value], thinking: bool, reply: &str) -> ParsedReply {
    let messages = user();
    let tools = Value::Array(tools.to_vec());
    let turn = ChatTurn {
        tools: Some(&tools),
        enable_thinking: thinking,
        ..ChatTurn::new(&messages)
    };
    let format = ChatFormat::render(&template(file), &turn)
        .unwrap_or_else(|e| panic!("{file}: render: {e}"));
    assert!(!format.prompt().is_empty(), "{file}: empty prompt");
    format
        .parse(reply)
        .unwrap_or_else(|e| panic!("{file}: parse: {e}"))
}

fn assert_call(file: &str, reply: &ParsedReply, name: &str, arguments: Value) {
    assert_eq!(reply.tool_calls.len(), 1, "{file}: {reply:?}");
    assert_eq!(reply.tool_calls[0].name, name, "{file}: {reply:?}");
    assert_eq!(
        reply.tool_calls[0].arguments, arguments,
        "{file}: {reply:?}"
    );
    assert!(
        !reply.tool_calls[0].id.is_empty(),
        "{file}: a call without an id"
    );
}

#[test]
fn qwen3_xml_call() {
    let file = "Qwen3-Coder.jinja";
    let r = read(
        file,
        &[special_function()],
        true,
        "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
    assert_eq!(r.content.trim(), "");
}

#[test]
fn qwen3_5_reasoning_then_call() {
    let file = "Qwen3.5-4B.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "The user wants the time in Tokyo.\n</think>\n\n<tool_call>\n<function=get_time>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
    assert_eq!(
        r.reasoning.as_deref().map(str::trim),
        Some("The user wants the time in Tokyo.")
    );
    assert_eq!(r.content.trim(), "");
}

#[test]
fn deepseek_v3_1_call() {
    let file = "deepseek-ai-DeepSeek-V3.1.jinja";
    let r = read(
        file,
        &[get_time()],
        false,
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
}

#[test]
fn deepseek_v3_2_reasoning_then_call() {
    let file = "deepseek-ai-DeepSeek-V3.2.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "Let me check the time</think>\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_time\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
    assert_eq!(r.reasoning.as_deref(), Some("Let me check the time"));
}

#[test]
fn deepseek_v4_reasoning_then_call() {
    let file = "deepseek-ai-DeepSeek-V4.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "Let me check the time</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_time\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
    assert_eq!(r.reasoning.as_deref(), Some("Let me check the time"));
}

#[test]
fn deepseek_v4_answer_without_tools() {
    let file = "deepseek-ai-DeepSeek-V4.jinja";
    let r = read(
        file,
        &[],
        true,
        "I'm\nthinking</think>Hello, world!\nWhat's up?",
    );
    assert!(r.tool_calls.is_empty());
    assert_eq!(r.reasoning.as_deref(), Some("I'm\nthinking"));
    assert_eq!(r.content, "Hello, world!\nWhat's up?");
}

#[test]
fn glm_call_with_arg_pairs() {
    let file = "GLM-4.7-Flash.jinja";
    let r = read(
        file,
        &[special_function()],
        false,
        "<tool_call>special_function<arg_key>arg1</arg_key><arg_value>1</arg_value></tool_call>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
}

#[test]
fn glm_4_6_call_with_newlines() {
    let file = "GLM-4.6.jinja";
    let r = read(
        file,
        &[special_function()],
        false,
        "<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
}

#[test]
fn kimi_k2_call_section() {
    let file = "moonshotai-Kimi-K2.jinja";
    let r = read(
        file,
        &[special_function()],
        true,
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
    assert_eq!(r.tool_calls[0].id, "functions.special_function:0");
}

#[test]
fn kimi_k3_reasoning_and_call() {
    let file = "Kimi-K3.jinja";
    let r = read(
        file,
        &[special_function()],
        true,
        "I should call it<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"special_function\" index=\"1\"<|sep|><|open|>argument key=\"arg1\" type=\"number\"<|sep|>1<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
    assert_eq!(r.reasoning.as_deref(), Some("I should call it"));
}

#[test]
fn minimax_m2_reasoning_then_invoke() {
    let file = "MiniMax-M2.jinja";
    let r = read(
        file,
        &[special_function()],
        true,
        "Let's call a tool:\n</think>\n\n<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
    assert_eq!(
        r.reasoning.as_deref().map(str::trim),
        Some("Let's call a tool:")
    );
}

#[test]
fn minimax_m3_reasoning_then_invoke() {
    let file = "MiniMax-M3.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "<mm:think>Let me check the time</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_time\">]<]minimax[>[<city>Tokyo]<]minimax[>[</city>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
    assert_eq!(r.reasoning.as_deref(), Some("Let me check the time"));
}

#[test]
fn gemma4_call() {
    let file = "google-gemma-4-31B-it.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "<|tool_call>call:get_time{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
}

#[test]
fn gemma4_reasoning_channel() {
    let file = "google-gemma-4-31B-it.jinja";
    let r = read(
        file,
        &[],
        true,
        "<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?",
    );
    assert_eq!(r.reasoning.as_deref(), Some("I'm\nthinking"));
    assert_eq!(r.content, "Hello, world!\nWhat's up?");
}

#[test]
fn gpt_oss_analysis_then_commentary_call() {
    let file = "openai-gpt-oss-120b.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "<|channel|>analysis<|message|>Need the time.<|end|><|start|>assistant<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{\"city\": \"Tokyo\"}",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
    assert_eq!(r.reasoning.as_deref(), Some("Need the time."));
}

#[test]
fn harmony_variant_answer_then_call() {
    let file = "muse-glimmer.jinja";
    let r = read(
        file,
        &[special_function()],
        true,
        " to=user<|message|>Checking now.<|eom|><|start|>assistant to=special_function<|message|><atem:function_calls>\n<atem:invoke name=\"special_function\">\n<atem:parameter name=\"arg1\">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
    assert_eq!(r.content, "Checking now.");
}

#[test]
fn mistral_call_with_id() {
    let file = "Mistral-Small-3.2-24B-Instruct-2506.jinja";
    let r = read(
        file,
        &[special_function()],
        true,
        "[TOOL_CALLS]special_function[CALL_ID]123456789[ARGS]{\"arg1\": 1}",
    );
    assert_call(file, &r, "special_function", json!({"arg1": 1}));
    assert_eq!(r.tool_calls[0].id, "123456789");
}

#[test]
fn llama3_json_call() {
    let file = "meta-llama-Llama-3.3-70B-Instruct.jinja";
    let r = read(
        file,
        &[get_time()],
        true,
        "{\"name\": \"get_time\", \"parameters\": {\"city\": \"Tokyo\"}}",
    );
    assert_call(file, &r, "get_time", json!({"city": "Tokyo"}));
}

#[test]
fn llama3_plain_answer_with_tools_offered() {
    let file = "meta-llama-Llama-3.3-70B-Instruct.jinja";
    let r = read(file, &[get_time()], true, "Hello, world!\nWhat's up?");
    assert!(r.tool_calls.is_empty());
    assert_eq!(r.content, "Hello, world!\nWhat's up?");
}

#[test]
fn partial_call_keeps_the_arguments_so_far() {
    let file = "Qwen3-Coder.jinja";
    let messages = user();
    let tools = Value::Array(vec![get_time()]);
    let turn = ChatTurn {
        tools: Some(&tools),
        ..ChatTurn::new(&messages)
    };
    let format = ChatFormat::render(&template(file), &turn).unwrap();
    let r = format
        .parse_partial("Checking.\n<tool_call>\n<function=get_time>\n<parameter=city>\nTok")
        .unwrap();
    assert_eq!(r.content.trim(), "Checking.");
    assert!(r.tool_calls.iter().all(|c| c.name == "get_time"), "{r:?}");
}

#[test]
fn bos_is_not_repeated_for_a_tokenizer_that_adds_it() {
    let source = ChatTemplateSource::new(
        "{{ bos_token }}{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}\
         {% if add_generation_prompt %}<|assistant|>{% endif %}",
        "<s>",
        "</s>",
        true,
    );
    let messages = user();
    let format = ChatFormat::render(&source, &ChatTurn::new(&messages)).unwrap();
    assert!(
        format.prompt().starts_with("<|user|>"),
        "{}",
        format.prompt()
    );
}

#[test]
fn hugging_face_directory_template() {
    let dir = std::env::temp_dir().join(format!("chat-format-hf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({
            "add_bos_token": true,
            "bos_token": {"content": "<s>"},
            "eos_token": "</s>",
            "chat_template": [
                {"name": "tool_use", "template": "unused"},
                {"name": "default", "template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"}
            ],
        })
        .to_string(),
    )
    .unwrap();
    let source = ChatTemplateSource::from_hf_dir(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        source.template(),
        "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    );
    let messages = user();
    let format = ChatFormat::render(&source, &ChatTurn::new(&messages)).unwrap();
    assert_eq!(format.prompt(), "What time is it in Tokyo?");
}

/// Feed `reply` in small pieces and gather what the stream reports.
fn stream(file: &str, tools: &[Value], reply: &str) -> (String, String, Vec<(String, String)>) {
    let messages = user();
    let tools = Value::Array(tools.to_vec());
    let turn = ChatTurn {
        tools: Some(&tools),
        ..ChatTurn::new(&messages)
    };
    let format = ChatFormat::render(&template(file), &turn).unwrap();
    let mut s = format.stream().unwrap();
    let chars: Vec<char> = reply.chars().collect();
    let mut deltas = Vec::new();
    for piece in chars.chunks(3) {
        deltas.extend(s.push(&piece.iter().collect::<String>()).unwrap());
    }
    deltas.extend(s.finish().unwrap());
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    for d in deltas {
        match d {
            ReplyDelta::Content(c) => content.push_str(&c),
            ReplyDelta::Reasoning(r) => reasoning.push_str(&r),
            ReplyDelta::ToolCall {
                index,
                name,
                arguments,
                ..
            } => {
                if calls.len() <= index {
                    calls.resize(index + 1, (String::new(), String::new()));
                }
                if let Some(n) = name {
                    calls[index].0 = n;
                }
                calls[index].1.push_str(&arguments);
            }
        }
    }
    (content, reasoning, calls)
}

#[test]
fn streamed_reasoning_and_call_carry_no_markup() {
    let (content, reasoning, calls) = stream(
        "Qwen3.5-4B.jinja",
        &[get_time()],
        "The user wants the time.\n</think>\n\n<tool_call>\n<function=get_time>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>",
    );
    assert_eq!(reasoning.trim(), "The user wants the time.");
    assert_eq!(content.trim(), "");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].0, "get_time");
    let args: Value = serde_json::from_str(&calls[0].1).unwrap();
    assert_eq!(args, json!({"city": "Tokyo"}));
}

#[test]
fn streamed_channel_reply_carries_no_markup() {
    let (content, reasoning, calls) = stream(
        "openai-gpt-oss-120b.jinja",
        &[],
        "<|channel|>analysis<|message|>Simple.<|end|><|start|>assistant<|channel|>final<|message|>Hello, world!",
    );
    assert_eq!(reasoning, "Simple.");
    assert_eq!(content, "Hello, world!");
    assert!(calls.is_empty());
}
