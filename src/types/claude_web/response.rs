use std::collections::BTreeSet;

use async_stream::try_stream;
use axum::{
    BoxError, Json,
    response::{IntoResponse, Response, Sse, sse::Event as SseEvent},
};
use bytes::Bytes;
use eventsource_stream::{EventStream, Eventsource};
use futures::{Stream, TryStreamExt, stream};
use serde::Deserialize;
use serde_json::Value;
use url::Url;
use uuid::Uuid;
use wreq::Proxy;

use crate::{
    claude_code_state::ClaudeCodeState,
    claude_web_state::ClaudeWebState,
    error::{CheckClaudeErr, ClewdrError},
    types::claude::{
        ContentBlock, ContentBlockDelta, CountMessageTokensResponse, CreateMessageParams,
        CreateMessageResponse, Message, MessageDeltaContent, MessageStartContent, Role,
        StopReason, StreamEvent, StreamUsage, Tool as ClaudeTool, Usage,
    },
    utils::print_out_text,
};

pub async fn merge_sse(
    stream: EventStream<impl Stream<Item = Result<Bytes, wreq::Error>>>,
) -> Result<String, ClewdrError> {
    #[derive(Deserialize)]
    struct Data {
        completion: String,
    }
    Ok(stream
        .try_filter_map(async |event| {
            Ok(serde_json::from_str::<Data>(&event.data)
                .map(|data| data.completion)
                .ok())
        })
        .try_collect()
        .await?)
}

impl<S> From<S> for Message
where
    S: Into<String>,
{
    fn from(str: S) -> Self {
        Message::new_blocks(Role::Assistant, vec![ContentBlock::text(str.into())])
    }
}

impl ClaudeWebState {
    pub async fn transform_response(
        &mut self,
        wreq_res: wreq::Response,
    ) -> Result<axum::response::Response, ClewdrError> {
        let tool_mode = self
            .last_params
            .as_ref()
            .is_some_and(request_has_tool_mode);

        if self.stream && !tool_mode {
            let mut input_tokens = self.usage.input_tokens as u64;
            let handle = self.cookie_actor_handle.clone();
            let cookie = self.cookie.clone();
            let enable_precise = crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens;
            let last_params = self.last_params.clone();
            let endpoint = self.endpoint.clone();
            let proxy = self.proxy.clone();
            let client = self.client.clone();
            if crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens
                && let Some(tokens) = self.try_code_count_tokens().await
            {
                input_tokens = tokens as u64;
            }

            let stream = wreq_res
                .bytes_stream()
                .eventsource()
                .map_err(axum::Error::new);
            let stream = try_stream! {
                let mut acc = String::new();
                #[derive(serde::Deserialize)]
                struct Data { completion: String }
                futures::pin_mut!(stream);
                while let Some(event) = stream.try_next().await? {
                    if let Ok(d) = serde_json::from_str::<Data>(&event.data) {
                        acc.push_str(&d.completion);
                    }
                    let e = SseEvent::default().event(event.event).id(event.id);
                    let e = if let Some(retry) = event.retry { e.retry(retry) } else { e };
                    yield e.data(event.data);
                }
                if !acc.is_empty() {
                    let mut out = None;
                    if enable_precise
                        && let Some(model) = last_params.as_ref().map(|p| p.model.clone())
                    {
                        out = count_code_output_tokens_for_text(
                            cookie.clone(), endpoint.clone(), proxy.clone(), client.clone(),
                            model, acc.clone(), handle.clone()
                        ).await.map(|v| v as u64);
                    }
                    let out = out.unwrap_or_else(|| {
                        let usage = crate::types::claude::Usage { input_tokens: input_tokens as u32, output_tokens: 0 };
                        let resp = crate::types::claude::CreateMessageResponse::text(acc.clone(), Default::default(), usage);
                        resp.count_tokens() as u64
                    });
                    if let Some(mut c) = cookie.clone() {
                        let family = last_params
                            .as_ref()
                            .map(|p| p.model.as_str())
                            .map(|m| {
                                let m = m.to_ascii_lowercase();
                                if m.contains("opus") {
                                    crate::config::ModelFamily::Opus
                                } else if m.contains("sonnet") {
                                    crate::config::ModelFamily::Sonnet
                                } else {
                                    crate::config::ModelFamily::Other
                                }
                            })
                            .unwrap_or(crate::config::ModelFamily::Other);
                        c.add_and_bucket_usage(input_tokens, out, family);
                        let _ = handle.return_cookie(c, None).await;
                    }
                } else if let Some(mut c) = cookie.clone() {
                    let family = last_params
                        .as_ref()
                        .map(|p| p.model.as_str())
                        .map(|m| {
                            let m = m.to_ascii_lowercase();
                            if m.contains("opus") {
                                crate::config::ModelFamily::Opus
                            } else if m.contains("sonnet") {
                                crate::config::ModelFamily::Sonnet
                            } else {
                                crate::config::ModelFamily::Other
                            }
                        })
                        .unwrap_or(crate::config::ModelFamily::Other);
                    c.add_and_bucket_usage(input_tokens, 0, family);
                    let _ = handle.return_cookie(c, None).await;
                }
            };
            let stream = stream.map_err(|e: axum::Error| -> BoxError { e.into() });
            return Ok(Sse::new(stream)
                .keep_alive(Default::default())
                .into_response());
        }

        let text = merge_sse(wreq_res.bytes_stream().eventsource()).await?;
        print_out_text(text.to_owned(), "claude_web_non_stream.txt");

        let enable_precise = crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens;
        let mut usage = self.usage.to_owned();
        if enable_precise && let Some(inp) = self.try_code_count_tokens().await {
            usage.input_tokens = inp;
        }
        let model = self
            .last_params
            .as_ref()
            .map(|params| params.model.clone())
            .unwrap_or_default();
        let mut output_tokens = CreateMessageResponse::text(text.clone(), model.clone(), Usage::default())
            .count_tokens();
        if enable_precise && let Some(model) = self.last_params.as_ref().map(|p| p.model.clone()) {
            if let Some(v) = count_code_output_tokens_for_text(
                self.cookie.clone(),
                self.endpoint.clone(),
                self.proxy.clone(),
                self.client.clone(),
                model,
                text.clone(),
                self.cookie_actor_handle.clone(),
            )
            .await
            {
                output_tokens = v;
            }
        }
        usage.output_tokens = output_tokens;
        self.persist_usage_totals(usage.input_tokens as u64, output_tokens as u64)
            .await;

        let response = if tool_mode {
            build_tool_shim_response(&text, &model, usage.clone(), self.last_params.as_ref())
                .unwrap_or_else(|| plain_text_response(text.clone(), model.clone(), usage.clone()))
        } else {
            plain_text_response(text.clone(), model.clone(), usage.clone())
        };

        if self.stream {
            return Ok(build_synthetic_stream_response(response));
        }

        Ok(Json(response).into_response())
    }
}

fn request_has_tool_mode(params: &CreateMessageParams) -> bool {
    params.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        || params.messages.iter().any(|msg| match &msg.content {
            crate::types::claude::MessageContent::Blocks { content } => content.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolUse { .. }
                        | ContentBlock::ToolResult { .. }
                        | ContentBlock::ServerToolUse { .. }
                        | ContentBlock::McpToolUse { .. }
                        | ContentBlock::McpToolResult { .. }
                        | ContentBlock::WebSearchToolResult { .. }
                        | ContentBlock::WebFetchToolResult { .. }
                        | ContentBlock::CodeExecutionToolResult { .. }
                        | ContentBlock::BashCodeExecutionToolResult { .. }
                        | ContentBlock::TextEditorCodeExecutionToolResult { .. }
                        | ContentBlock::ToolSearchToolResult { .. }
                )
            }),
            crate::types::claude::MessageContent::Text { .. } => false,
        })
}

fn plain_text_response(text: String, model: String, usage: Usage) -> CreateMessageResponse {
    let mut response = CreateMessageResponse::text(text, model, usage);
    response.stop_reason = Some(StopReason::EndTurn);
    response
}

fn build_tool_shim_response(
    raw: &str,
    model: &str,
    usage: Usage,
    params: Option<&CreateMessageParams>,
) -> Option<CreateMessageResponse> {
    let tools = params?.tools.as_ref()?;
    let content = parse_tool_shim_blocks(raw, tools)?;
    let stop_reason = if content.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. })) {
        Some(StopReason::ToolUse)
    } else {
        Some(StopReason::EndTurn)
    };
    Some(CreateMessageResponse {
        content,
        id: Uuid::new_v4().to_string(),
        model: model.to_string(),
        role: Role::Assistant,
        stop_reason,
        stop_sequence: None,
        type_: "message".into(),
        usage: Some(usage),
    })
}

fn parse_tool_shim_blocks(raw: &str, tools: &[ClaudeTool]) -> Option<Vec<ContentBlock>> {
    let allowed = allowed_tool_names(tools);
    if allowed.is_empty() {
        return None;
    }
    parse_tool_call_markup(raw, &allowed).or_else(|| parse_tool_call_json(raw, &allowed))
}

fn allowed_tool_names(tools: &[ClaudeTool]) -> BTreeSet<String> {
    tools
        .iter()
        .filter_map(|tool| serde_json::to_value(tool).ok())
        .filter_map(|value| value.get("name").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn parse_tool_call_markup(raw: &str, allowed: &BTreeSet<String>) -> Option<Vec<ContentBlock>> {
    let mut rest = raw;
    let mut content = Vec::new();
    let mut found = false;

    while let Some(start) = rest.find("<tool_call") {
        found = true;
        let before = rest[..start].trim();
        if !before.is_empty() {
            content.push(ContentBlock::text(before));
        }
        rest = &rest[start..];

        let tag_end = rest.find('>')?;
        let open_tag = &rest[..=tag_end];
        let name = extract_xml_attr(open_tag, "name")
            .or_else(|| extract_xml_attr(open_tag, "tool"))?;
        let close = rest[tag_end + 1..].find("</tool_call>")?;
        let body = &rest[tag_end + 1..tag_end + 1 + close];
        let input = parse_tool_arguments(body)?;
        let name = normalize_tool_name(&name, allowed)?;
        content.push(ContentBlock::ToolUse {
            id: format!("toolu_{}", Uuid::new_v4().simple()),
            name,
            input,
            cache_control: None,
            caller: None,
        });
        rest = &rest[tag_end + 1 + close + "</tool_call>".len()..];
    }

    let trailing = rest.trim();
    if !trailing.is_empty() {
        content.push(ContentBlock::text(trailing));
    }

    if !found || !content.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. })) {
        return None;
    }

    Some(content)
}

fn parse_tool_call_json(raw: &str, allowed: &BTreeSet<String>) -> Option<Vec<ContentBlock>> {
    let value = serde_json::from_str::<Value>(&strip_code_fence(raw)).ok()?;
    let calls = value.get("tool_calls")?.as_array()?;
    let mut content = Vec::new();

    if let Some(text) = value.get("content").and_then(Value::as_str) {
        let text = text.trim();
        if !text.is_empty() {
            content.push(ContentBlock::text(text));
        }
    }

    for call in calls {
        let (name, input) = if let Some(function) = call.get("function") {
            let name = function.get("name")?.as_str()?.to_string();
            let input = parse_json_arguments(function.get("arguments")?)?;
            (name, input)
        } else {
            let name = call
                .get("name")
                .or_else(|| call.get("tool"))?
                .as_str()?
                .to_string();
            let input = parse_json_arguments(
                call.get("arguments")
                    .or_else(|| call.get("input"))
                    .unwrap_or(&Value::Null),
            )?;
            (name, input)
        };
        let name = normalize_tool_name(&name, allowed)?;
        content.push(ContentBlock::ToolUse {
            id: format!("toolu_{}", Uuid::new_v4().simple()),
            name,
            input,
            cache_control: None,
            caller: None,
        });
    }

    if !content.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. })) {
        return None;
    }

    Some(content)
}

fn parse_json_arguments(value: &Value) -> Option<Value> {
    match value {
        Value::Null => Some(serde_json::json!({})),
        Value::String(text) => serde_json::from_str::<Value>(&strip_code_fence(text)).ok(),
        _ => Some(value.clone()),
    }
}

fn parse_tool_arguments(body: &str) -> Option<Value> {
    let body = strip_code_fence(body);
    if body.trim().is_empty() {
        return Some(serde_json::json!({}));
    }
    serde_json::from_str::<Value>(&body).ok()
}

fn strip_code_fence(input: &str) -> String {
    let trimmed = input.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let mut stripped = trimmed.trim_start_matches('`');
    stripped = stripped
        .strip_prefix("json")
        .or_else(|| stripped.strip_prefix("JSON"))
        .unwrap_or(stripped);
    stripped = stripped.trim_start_matches(['\r', '\n']);
    if let Some(end) = stripped.rfind("```") {
        return stripped[..end].trim().to_string();
    }
    trimmed.to_string()
}

fn extract_xml_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    if let Some(start) = tag.find(&needle) {
        let rest = &tag[start + needle.len()..];
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    let needle = format!("{attr}='");
    if let Some(start) = tag.find(&needle) {
        let rest = &tag[start + needle.len()..];
        let end = rest.find('\'')?;
        return Some(rest[..end].to_string());
    }
    None
}

fn normalize_tool_name(name: &str, allowed: &BTreeSet<String>) -> Option<String> {
    if allowed.contains(name) {
        return Some(name.to_string());
    }

    let canonical = canonical_tool_name(name);
    if let Some(exactish) = allowed
        .iter()
        .find(|candidate| canonical_tool_name(candidate) == canonical)
    {
        return Some(exactish.clone());
    }

    let alias = match canonical.as_str() {
        "view" | "readfile" | "readpath" => Some("read"),
        "bashtool" | "shell" | "terminal" | "runcommand" => Some("bash"),
        "webfetchtool" | "webfetcher" | "websearch" | "fetchweb" | "fetchurl" => {
            Some("webfetch")
        }
        "writefile" => Some("write"),
        "editfile" => Some("edit"),
        "todowritetool" | "todowritefile" | "todowriter" | "todo" | "todowriteplan" => {
            Some("todowrite")
        }
        "questiontool" | "askuser" => Some("question"),
        _ => None,
    }?;

    allowed.contains(alias).then(|| alias.to_string())
}

fn canonical_tool_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn build_synthetic_stream_response(message: CreateMessageResponse) -> Response {
    let CreateMessageResponse {
        content,
        id,
        model,
        stop_reason,
        usage,
        ..
    } = message;
    let output_tokens = usage.as_ref().map(|usage| usage.output_tokens).unwrap_or_default();
    let input_tokens = usage.as_ref().map(|usage| usage.input_tokens).unwrap_or_default();

    let mut events = Vec::new();
    events.push(
        SseEvent::default()
            .json_data(StreamEvent::MessageStart {
                message: MessageStartContent {
                    id: id.clone(),
                    type_: "message".into(),
                    role: Role::Assistant,
                    content: vec![],
                    model: model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Some(Usage {
                        input_tokens,
                        output_tokens: 0,
                    }),
                },
            })
            .unwrap(),
    );

    for (index, block) in content.into_iter().enumerate() {
        match block {
            ContentBlock::Text { text, .. } => {
                events.push(
                    SseEvent::default()
                        .json_data(StreamEvent::ContentBlockStart {
                            index,
                            content_block: ContentBlock::text(""),
                        })
                        .unwrap(),
                );
                events.push(
                    SseEvent::default()
                        .json_data(StreamEvent::ContentBlockDelta {
                            index,
                            delta: ContentBlockDelta::TextDelta { text },
                        })
                        .unwrap(),
                );
                events.push(
                    SseEvent::default()
                        .json_data(StreamEvent::ContentBlockStop { index })
                        .unwrap(),
                );
            }
            ContentBlock::ToolUse {
                id,
                name,
                input,
                cache_control,
                caller,
            } => {
                events.push(
                    SseEvent::default()
                        .json_data(StreamEvent::ContentBlockStart {
                            index,
                            content_block: ContentBlock::ToolUse {
                                id,
                                name,
                                input,
                                cache_control,
                                caller,
                            },
                        })
                        .unwrap(),
                );
                events.push(
                    SseEvent::default()
                        .json_data(StreamEvent::ContentBlockStop { index })
                        .unwrap(),
                );
            }
            _ => {}
        }
    }

    events.push(
        SseEvent::default()
            .json_data(StreamEvent::MessageDelta {
                delta: MessageDeltaContent {
                    stop_reason: stop_reason.or(Some(StopReason::EndTurn)),
                    stop_sequence: None,
                },
                usage: Some(StreamUsage {
                    input_tokens: 0,
                    output_tokens,
                }),
            })
            .unwrap(),
    );
    events.push(
        SseEvent::default()
            .json_data(StreamEvent::MessageStop)
            .unwrap(),
    );

    let stream = stream::iter(events.into_iter().map(Ok::<_, BoxError>));
    Sse::new(stream).keep_alive(Default::default()).into_response()
}

async fn bearer_count_tokens(
    state: &ClaudeCodeState,
    access_token: &str,
    body: &CreateMessageParams,
) -> Option<u32> {
    let url = state
        .endpoint
        .join("v1/messages/count_tokens")
        .expect("Url parse error");
    let resp = state
        .client
        .post(url.to_string())
        .bearer_auth(access_token)
        .header("anthropic-version", "2023-06-01")
        .json(body)
        .send()
        .await
        .ok()?;
    let resp = resp.check_claude().await.ok()?;
    let v: CountMessageTokensResponse = resp.json().await.ok()?;
    Some(v.input_tokens)
}

impl ClaudeWebState {
    pub(crate) async fn try_code_count_tokens(&mut self) -> Option<u32> {
        self.cookie.as_ref()?;
        let params = self.last_params.as_ref()?.clone();
        let mut code = ClaudeCodeState::new(self.cookie_actor_handle.clone());
        code.cookie = self.cookie.clone();
        code.endpoint = self.endpoint.clone();
        code.proxy = self.proxy.clone();
        code.client = self.client.clone();
        if let Some(ref c) = self.cookie
            && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
        {
            code.set_cookie_header_value(val);
        }

        let org = code.get_organization().await.ok()?;
        let exch = code.exchange_code(&org).await.ok()?;
        code.exchange_token(exch).await.ok()?;
        let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

        let mut body = params.clone();
        body.stream = Some(false);

        bearer_count_tokens(&code, &access, &body).await
    }
}

async fn count_code_output_tokens_for_text(
    cookie: Option<crate::config::CookieStatus>,
    endpoint: Url,
    proxy: Option<Proxy>,
    client: wreq::Client,
    model: String,
    text: String,
    handle: crate::services::cookie_actor::CookieActorHandle,
) -> Option<u32> {
    let mut code = ClaudeCodeState::new(handle.clone());
    code.cookie = cookie.clone();
    code.endpoint = endpoint;
    code.proxy = proxy;
    code.client = client;
    if let Some(ref c) = cookie
        && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
    {
        code.set_cookie_header_value(val);
    }
    let org = code.get_organization().await.ok()?;
    let exch = code.exchange_code(&org).await.ok()?;
    code.exchange_token(exch).await.ok()?;
    let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

    let body = CreateMessageParams {
        model,
        messages: vec![Message::new_text(Role::Assistant, text)],
        ..Default::default()
    };
    bearer_count_tokens(&code, &access, &body).await
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::claude::{CustomTool, Tool};

    fn sample_tools() -> Vec<Tool> {
        vec![Tool::Custom(CustomTool {
            name: "bash".into(),
            description: Some("Run shell commands".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["command", "description"]
            }),
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            input_examples: None,
            strict: None,
            type_: None,
            extra: Default::default(),
        })]
    }

    #[test]
    fn parses_xml_tool_calls() {
        let content = parse_tool_shim_blocks(
            "<tool_call name=\"bash\">{\"command\":\"pwd\",\"description\":\"Print cwd\"}</tool_call>",
            &sample_tools(),
        )
        .expect("tool blocks");

        assert!(matches!(&content[0], ContentBlock::ToolUse { name, .. } if name == "bash"));
    }

    #[test]
    fn maps_common_tool_aliases() {
        let content = parse_tool_shim_blocks(
            "<tool_call name=\"bash_tool\">{\"command\":\"pwd\",\"description\":\"Print cwd\"}</tool_call>",
            &sample_tools(),
        )
        .expect("tool blocks");

        assert!(matches!(&content[0], ContentBlock::ToolUse { name, .. } if name == "bash"));
    }
}
