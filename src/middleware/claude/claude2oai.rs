use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use async_stream::try_stream;
use axum::response::sse::Event;
use futures::{Stream, StreamExt};
use serde::Serialize;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::types::claude::{
    ContentBlock, ContentBlockDelta, CreateMessageResponse, StopReason, StreamEvent, Usage,
};

#[derive(Debug, Clone)]
struct StreamToolCall {
    ordinal: usize,
}

#[derive(Debug)]
struct StreamState {
    id: Option<String>,
    model: Option<String>,
    created: u64,
    tool_calls: HashMap<usize, StreamToolCall>,
    finish_sent: bool,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            id: None,
            model: None,
            created: now_secs(),
            tool_calls: HashMap::new(),
            finish_sent: false,
        }
    }
}

impl StreamState {
    fn ensure_identity(
        &mut self,
        id: Option<impl Into<String>>,
        model: Option<impl Into<String>>,
    ) -> (String, String) {
        if self.id.is_none()
            && let Some(id) = id
        {
            self.id = Some(id.into());
        }
        if self.model.is_none()
            && let Some(model) = model
        {
            self.model = Some(model.into());
        }

        let id = self.id.clone().unwrap_or_else(|| {
            let generated = format!("chatcmpl-{}", Uuid::new_v4());
            self.id = Some(generated.clone());
            generated
        });
        let model = self.model.clone().unwrap_or_else(|| {
            let fallback = "unknown".to_string();
            self.model = Some(fallback.clone());
            fallback
        });
        (id, model)
    }
}

/// Represents the data structure for streaming events in OpenAI API format.
#[derive(Debug, Serialize)]
struct StreamEventData {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<StreamEventChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Value>,
}

#[derive(Debug, Serialize)]
struct StreamEventChoice {
    index: u32,
    delta: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn compact_json(value: &Value) -> String {
    match value {
        Value::Null => "{}".to_string(),
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn finish_reason(reason: Option<&StopReason>) -> &'static str {
    match reason {
        Some(StopReason::EndTurn) => "stop",
        Some(StopReason::MaxTokens) => "length",
        Some(StopReason::StopSequence) => "stop",
        Some(StopReason::ToolUse) => "tool_calls",
        Some(StopReason::PauseTurn) => "stop",
        Some(StopReason::Refusal) => "content_filter",
        Some(StopReason::ModelContextWindowExceeded) => "length",
        None => "stop",
    }
}

fn build_stream_event(
    id: String,
    model: String,
    created: u64,
    delta: Value,
    finish_reason: Option<&'static str>,
) -> Event {
    Event::default()
        .json_data(StreamEventData {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![StreamEventChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        })
        .unwrap()
}

fn build_usage_event(id: String, model: String, created: u64, usage: Usage) -> Event {
    Event::default()
        .json_data(StreamEventData {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![],
            usage: Some(json!({
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            })),
        })
        .unwrap()
}

fn assistant_message_json(input: &CreateMessageResponse) -> Value {
    let mut content_fragments = Vec::new();
    let mut reasoning_fragments = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &input.content {
        match block {
            ContentBlock::Text { text, .. } => content_fragments.push(text.clone()),
            ContentBlock::Thinking { thinking, .. } => reasoning_fragments.push(thinking.clone()),
            ContentBlock::ToolUse {
                id, name, input, ..
            }
            | ContentBlock::ServerToolUse {
                id, name, input, ..
            }
            | ContentBlock::McpToolUse {
                id, name, input, ..
            } => tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": compact_json(input),
                }
            })),
            _ => {}
        }
    }

    let mut message = Map::new();
    message.insert("role".to_string(), Value::String("assistant".to_string()));

    let content = if content_fragments.is_empty() {
        Value::Null
    } else {
        Value::String(content_fragments.join(""))
    };
    message.insert("content".to_string(), content);

    if !reasoning_fragments.is_empty() {
        message.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_fragments.join("")),
        );
    }

    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }

    Value::Object(message)
}

/// Transforms a Claude event stream into standard OpenAI chat-completions SSE.
pub fn transform_stream<I, E>(
    s: I,
    include_usage: bool,
    initial_usage: Usage,
) -> impl Stream<Item = Result<Event, E>>
where
    I: Stream<Item = Result<eventsource_stream::Event, E>>,
{
    try_stream! {
        let mut state = StreamState::default();
        let mut input = Box::pin(s);

        while let Some(event) = input.next().await {
            let eventsource_stream::Event { data, .. } = event?;
            let Ok(parsed) = serde_json::from_str::<StreamEvent>(&data) else {
                continue;
            };

            match parsed {
                StreamEvent::MessageStart { message } => {
                    let (id, model) = state.ensure_identity(Some(message.id), Some(message.model));
                    yield build_stream_event(
                        id,
                        model,
                        state.created,
                        json!({ "role": "assistant" }),
                        None,
                    );
                }
                StreamEvent::ContentBlockStart { index, content_block } => match content_block {
                    ContentBlock::ToolUse { id, name, input, .. }
                    | ContentBlock::ServerToolUse { id, name, input, .. }
                    | ContentBlock::McpToolUse { id, name, input, .. } => {
                        let ordinal = state.tool_calls.len();
                        state.tool_calls.insert(
                            index,
                            StreamToolCall {
                                ordinal,
                            },
                        );
                        let (stream_id, model) = state.ensure_identity(None::<String>, None::<String>);
                        let arguments = if matches!(input, Value::Null) {
                            String::new()
                        } else {
                            compact_json(&input)
                        };
                        yield build_stream_event(
                            stream_id,
                            model,
                            state.created,
                            json!({
                                "tool_calls": [{
                                    "index": ordinal,
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments,
                                    }
                                }]
                            }),
                            None,
                        );
                    }
                    _ => {}
                },
                StreamEvent::ContentBlockDelta { index, delta } => match delta {
                    ContentBlockDelta::TextDelta { text } => {
                        if text.is_empty() {
                            continue;
                        }
                        let (id, model) = state.ensure_identity(None::<String>, None::<String>);
                        yield build_stream_event(
                            id,
                            model,
                            state.created,
                            json!({ "content": text }),
                            None,
                        );
                    }
                    ContentBlockDelta::ThinkingDelta { thinking } => {
                        if thinking.is_empty() {
                            continue;
                        }
                        let (id, model) = state.ensure_identity(None::<String>, None::<String>);
                        yield build_stream_event(
                            id,
                            model,
                            state.created,
                            json!({ "reasoning_content": thinking }),
                            None,
                        );
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        let Some(tool_call) = state.tool_calls.get(&index).cloned() else {
                            continue;
                        };
                        let (id, model) = state.ensure_identity(None::<String>, None::<String>);
                        yield build_stream_event(
                            id,
                            model,
                            state.created,
                            json!({
                                "tool_calls": [{
                                    "index": tool_call.ordinal,
                                    "function": {
                                        "arguments": partial_json,
                                    }
                                }]
                            }),
                            None,
                        );
                    }
                    ContentBlockDelta::SignatureDelta { .. } => {}
                },
                StreamEvent::MessageDelta { delta, usage } => {
                    let (id, model) = state.ensure_identity(None::<String>, None::<String>);
                    let reason = finish_reason(delta.stop_reason.as_ref());
                    state.finish_sent = true;
                    yield build_stream_event(id.clone(), model.clone(), state.created, json!({}), Some(reason));

                    if include_usage {
                        let stream_usage = usage.unwrap_or_default();
                        yield build_usage_event(
                            id,
                            model,
                            state.created,
                            Usage {
                                input_tokens: initial_usage.input_tokens,
                                output_tokens: stream_usage.output_tokens,
                            },
                        );
                    }
                }
                StreamEvent::MessageStop => {
                    if !state.finish_sent {
                        let (id, model) = state.ensure_identity(None::<String>, None::<String>);
                        yield build_stream_event(id, model, state.created, json!({}), Some("stop"));
                    }
                    yield Event::default().data("[DONE]");
                }
                StreamEvent::Error { error } => {
                    yield Event::default()
                        .json_data(json!({
                            "error": {
                                "message": error.message,
                                "type": error.type_,
                            }
                        }))
                        .unwrap();
                    yield Event::default().data("[DONE]");
                }
                StreamEvent::ContentBlockStop { .. } | StreamEvent::Ping => {}
            }
        }
    }
}

pub fn transforms_json(input: CreateMessageResponse) -> Value {
    let usage = input.usage.as_ref().map(|usage| {
        json!({
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        })
    });

    json!({
        "id": input.id,
        "object": "chat.completion",
        "created": now_secs(),
        "model": input.model,
        "choices": [{
            "index": 0,
            "message": assistant_message_json(&input),
            "finish_reason": finish_reason(input.stop_reason.as_ref()),
        }],
        "usage": usage,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn transforms_tool_use_to_openai_message() {
        let response = CreateMessageResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "Paris"}),
                cache_control: None,
                caller: None,
            }],
            id: "msg_1".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            role: crate::types::claude::Role::Assistant,
            stop_reason: Some(StopReason::ToolUse),
            stop_sequence: None,
            type_: "message".to_string(),
            usage: Some(Usage {
                input_tokens: 12,
                output_tokens: 4,
            }),
        };

        let output = transforms_json(response);
        assert_eq!(output["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(output["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(
            output["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            output["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[test]
    fn transforms_text_and_reasoning_to_openai_message() {
        let response = CreateMessageResponse {
            content: vec![
                ContentBlock::Thinking {
                    signature: "sig".to_string(),
                    thinking: "chain".to_string(),
                },
                ContentBlock::text("hello"),
            ],
            id: "msg_2".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            role: crate::types::claude::Role::Assistant,
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            type_: "message".to_string(),
            usage: Some(Usage {
                input_tokens: 2,
                output_tokens: 3,
            }),
        };

        let output = transforms_json(response);
        assert_eq!(output["choices"][0]["message"]["content"], "hello");
        assert_eq!(
            output["choices"][0]["message"]["reasoning_content"],
            "chain"
        );
        assert_eq!(output["usage"]["total_tokens"], 5);
    }
}
