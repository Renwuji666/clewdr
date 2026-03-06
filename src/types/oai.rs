use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tiktoken_rs::o200k_base;
use uuid::Uuid;

use super::claude::{CreateMessageParams as ClaudeCreateMessageParams, *};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Low = 256,
    #[default]
    Medium = 256 * 8,
    High = 256 * 8 * 8,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Stop {
    Single(String),
    Multiple(Vec<String>),
}

impl Stop {
    fn into_vec(self) -> Vec<String> {
        match self {
            Stop::Single(stop) => vec![stop],
            Stop::Multiple(stops) => stops,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ResponseFormatJsonSchema {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        json_schema: ResponseFormatJsonSchema,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolArguments {
    String(String),
    Json(Value),
}

impl Default for ToolArguments {
    fn default() -> Self {
        Self::String("{}".to_string())
    }
}

impl ToolArguments {
    fn into_json(self) -> Value {
        match self {
            ToolArguments::String(arguments) => {
                serde_json::from_str(&arguments).unwrap_or_else(|_| Value::String(arguments))
            }
            ToolArguments::Json(value) => value,
        }
    }

    fn to_argument_string(&self) -> String {
        match self {
            ToolArguments::String(arguments) => arguments.clone(),
            ToolArguments::Json(value) => compact_json(value),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAITool {
    #[serde(rename = "type", default = "default_function_type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIToolFunction>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIToolChoiceObject {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIToolChoiceFunction>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String),
    Object(OpenAIToolChoiceObject),
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: ToolArguments,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_function_type")]
    pub type_: String,
    pub function: OpenAIToolCallFunction,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIContentPart {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<OpenAIFile>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct CreateMessageParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub messages: Vec<OpenAIMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
}

impl CreateMessageParams {
    pub fn stream_include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
    }

    pub fn count_tokens(&self) -> u32 {
        let bpe = o200k_base().expect("Failed to get encoding");
        let messages = self
            .messages
            .iter()
            .flat_map(|msg| {
                let mut chunks = vec![msg.role.clone()];
                if let Some(content) = &msg.content {
                    chunks.push(content_to_text(content));
                }
                if let Some(reasoning_content) = &msg.reasoning_content {
                    chunks.push(reasoning_content.clone());
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    chunks.extend(
                        tool_calls
                            .iter()
                            .map(|tool_call| tool_call.function.arguments.to_argument_string()),
                    );
                }
                chunks
            })
            .collect::<Vec<_>>()
            .join("\n");
        bpe.encode_with_special_tokens(&messages).len() as u32
    }
}

impl From<CreateMessageParams> for ClaudeCreateMessageParams {
    fn from(params: CreateMessageParams) -> Self {
        let mut system_blocks = Vec::new();
        let mut messages = Vec::new();

        for message in params.messages {
            let OpenAIMessage {
                role,
                content,
                reasoning_content: _,
                tool_calls,
                tool_call_id,
                extra,
                ..
            } = message;
            let message_cache_control = cache_control_from_extra(&extra);

            match role.as_str() {
                "system" | "developer" => {
                    system_blocks.extend(text_only_blocks(content, message_cache_control));
                }
                "user" => {
                    let mut blocks = content_to_blocks(content);
                    apply_message_cache_control(&mut blocks, message_cache_control);
                    if !blocks.is_empty() {
                        messages.push(Message::new_blocks(Role::User, blocks));
                    }
                }
                "assistant" => {
                    let mut blocks = content_to_blocks(content);
                    if let Some(tool_calls) = tool_calls {
                        blocks.extend(tool_calls.into_iter().map(tool_call_to_block));
                    }
                    apply_message_cache_control(&mut blocks, message_cache_control);
                    if !blocks.is_empty() {
                        messages.push(Message::new_blocks(Role::Assistant, blocks));
                    }
                }
                "tool" => {
                    let tool_use_id = tool_call_id.unwrap_or_else(|| Uuid::new_v4().to_string());
                    let content = tool_result_content(content);
                    messages.push(Message::new_blocks(
                        Role::User,
                        vec![ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            cache_control: message_cache_control,
                            is_error: None,
                        }],
                    ));
                }
                _ => {
                    let mut blocks = content_to_blocks(content);
                    apply_message_cache_control(&mut blocks, message_cache_control);
                    if !blocks.is_empty() {
                        messages.push(Message::new_blocks(Role::User, blocks));
                    }
                }
            }
        }

        let system = (!system_blocks.is_empty()).then(|| json!(system_blocks));

        Self {
            max_tokens: (params.max_tokens.or(params.max_completion_tokens))
                .unwrap_or_else(default_max_tokens),
            system,
            messages,
            model: params.model,
            container: None,
            context_management: None,
            mcp_servers: None,
            stop_sequences: params.stop.map(Stop::into_vec),
            thinking: params.thinking.or_else(|| {
                params
                    .reasoning_effort
                    .map(|effort| Thinking::new(effort as u64))
            }),
            temperature: params.temperature,
            stream: params.stream,
            top_k: params.top_k,
            top_p: params.top_p,
            tools: params.tools.map(convert_tools),
            tool_choice: params.tool_choice.and_then(convert_tool_choice),
            metadata: params.metadata,
            output_config: None,
            output_format: params.response_format.and_then(convert_response_format),
            service_tier: None,
            n: params.n,
        }
    }
}

fn default_function_type() -> String {
    "function".to_string()
}

fn content_to_text(content: &OpenAIMessageContent) -> String {
    match content {
        OpenAIMessageContent::Text(text) => text.clone(),
        OpenAIMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part.type_.as_str() {
                "text" | "input_text" => part.text.clone(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn content_to_blocks(content: Option<OpenAIMessageContent>) -> Vec<ContentBlock> {
    match content {
        Some(OpenAIMessageContent::Text(text)) => {
            let text = text.trim().to_string();
            if text.is_empty() {
                vec![]
            } else {
                vec![ContentBlock::text(text)]
            }
        }
        Some(OpenAIMessageContent::Parts(parts)) => {
            parts.into_iter().filter_map(content_part_to_block).collect()
        }
        None => vec![],
    }
}

fn text_only_blocks(
    content: Option<OpenAIMessageContent>,
    message_cache_control: Option<CacheControlEphemeral>,
) -> Vec<Value> {
    let mut blocks = content_to_blocks(content);
    apply_message_cache_control(&mut blocks, message_cache_control);
    blocks
        .into_iter()
        .filter(|block| matches!(block, ContentBlock::Text { .. }))
        .map(|block| json!(block))
        .collect()
}

fn content_part_to_block(part: OpenAIContentPart) -> Option<ContentBlock> {
    let part_cache_control = cache_control_from_extra(&part.extra);
    match part.type_.as_str() {
        "text" | "input_text" => part.text.and_then(|text| {
            let text = text.trim().to_string();
            (!text.is_empty()).then(|| ContentBlock::Text {
                text,
                cache_control: part_cache_control,
                citations: None,
            })
        }),
        "image_url" | "input_image" => part
            .image_url
            .map(|image_url| image_block_from_url(image_url, part_cache_control)),
        "file" => part
            .file
            .and_then(|file| file_part_to_block(file, part_cache_control)),
        _ => None,
    }
}

fn image_block_from_url(
    image_url: ImageUrl,
    cache_control: Option<CacheControlEphemeral>,
) -> ContentBlock {
    let source = parse_data_uri(&image_url.url)
        .map(|(media_type, data)| ImageSource::Base64 { media_type, data })
        .unwrap_or(ImageSource::Url { url: image_url.url });
    ContentBlock::Image {
        source,
        cache_control,
    }
}

fn file_part_to_block(
    file: OpenAIFile,
    cache_control: Option<CacheControlEphemeral>,
) -> Option<ContentBlock> {
    let OpenAIFile {
        filename,
        file_data,
        file_id,
    } = file;

    if let Some(file_data) = file_data
        && let Some((media_type, data)) = parse_data_uri(&file_data)
        && media_type == "application/pdf"
    {
        return Some(ContentBlock::Document {
            source: json!({
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }),
            cache_control,
            citations: None,
            context: None,
            title: filename,
        });
    }

    file_id.map(|file_id| ContentBlock::Document {
        source: json!({
            "type": "file",
            "file_id": file_id,
        }),
        cache_control,
        citations: None,
        context: None,
        title: filename,
    })
}

fn parse_data_uri(value: &str) -> Option<(String, String)> {
    let (metadata, data) = value.split_once(',')?;
    let (media_type, encoding) = metadata.strip_prefix("data:")?.split_once(';')?;
    (encoding == "base64").then(|| (media_type.to_string(), data.to_string()))
}

fn cache_control_from_extra(extra: &HashMap<String, Value>) -> Option<CacheControlEphemeral> {
    extra
        .get("cache_control")
        .or_else(|| extra.get("cacheControl"))
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}

fn take_cache_control(extra: &mut HashMap<String, Value>) -> Option<CacheControlEphemeral> {
    extra
        .remove("cache_control")
        .or_else(|| extra.remove("cacheControl"))
        .and_then(|value| serde_json::from_value(value).ok())
}

fn apply_message_cache_control(
    blocks: &mut [ContentBlock],
    message_cache_control: Option<CacheControlEphemeral>,
) {
    let Some(cache_control) = message_cache_control else {
        return;
    };
    let Some(last) = blocks.last_mut() else {
        return;
    };

    match last {
        ContentBlock::Text {
            cache_control: slot,
            ..
        }
        | ContentBlock::Image {
            cache_control: slot,
            ..
        }
        | ContentBlock::Document {
            cache_control: slot,
            ..
        }
        | ContentBlock::SearchResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::ToolUse {
            cache_control: slot,
            ..
        }
        | ContentBlock::ToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::ToolReference {
            cache_control: slot,
            ..
        }
        | ContentBlock::ServerToolUse {
            cache_control: slot,
            ..
        }
        | ContentBlock::WebSearchToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::WebFetchToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::CodeExecutionToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::BashCodeExecutionToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::TextEditorCodeExecutionToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::ToolSearchToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::McpToolUse {
            cache_control: slot,
            ..
        }
        | ContentBlock::McpToolResult {
            cache_control: slot,
            ..
        }
        | ContentBlock::ContainerUpload {
            cache_control: slot,
            ..
        } => {
            if slot.is_none() {
                *slot = Some(cache_control);
            }
        }
        ContentBlock::ImageUrl { .. }
        | ContentBlock::Thinking { .. }
        | ContentBlock::RedactedThinking { .. } => {}
    }
}

fn tool_result_content(content: Option<OpenAIMessageContent>) -> Value {
    match content {
        Some(OpenAIMessageContent::Text(text)) => Value::String(text),
        Some(OpenAIMessageContent::Parts(parts)) => {
            let text_fragments = parts
                .iter()
                .filter_map(|part| match part.type_.as_str() {
                    "text" | "input_text" => part.text.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if !text_fragments.is_empty() {
                Value::String(text_fragments.join("\n"))
            } else {
                serde_json::to_value(parts).unwrap_or(Value::Null)
            }
        }
        None => Value::String(String::new()),
    }
}

fn tool_call_to_block(tool_call: OpenAIToolCall) -> ContentBlock {
    let cache_control = cache_control_from_extra(&tool_call.extra);
    let id = tool_call.id.unwrap_or_else(|| Uuid::new_v4().to_string());
    ContentBlock::ToolUse {
        id,
        name: tool_call.function.name,
        input: tool_call.function.arguments.into_json(),
        cache_control,
        caller: None,
    }
}

fn convert_tools(tools: Vec<OpenAITool>) -> Vec<Tool> {
    tools
        .into_iter()
        .filter_map(|tool| {
            if tool.type_ != "function" {
                return None;
            }
            let mut extra = tool.extra;
            let cache_control = take_cache_control(&mut extra);
            let function = tool.function?;
            Some(Tool::Custom(CustomTool {
                name: function.name,
                description: function.description,
                input_schema: function
                    .parameters
                    .unwrap_or_else(|| json!({ "type": "object" })),
                allowed_callers: None,
                cache_control,
                defer_loading: None,
                input_examples: None,
                strict: function.strict,
                type_: None,
                extra,
            }))
        })
        .collect()
}

fn convert_tool_choice(choice: OpenAIToolChoice) -> Option<ToolChoice> {
    match choice {
        OpenAIToolChoice::String(choice) => match choice.as_str() {
            "auto" => Some(ToolChoice::Auto {
                disable_parallel_tool_use: None,
            }),
            "required" | "any" => Some(ToolChoice::Any {
                disable_parallel_tool_use: None,
            }),
            "none" => Some(ToolChoice::None),
            _ => None,
        },
        OpenAIToolChoice::Object(choice) => match choice.type_.as_str() {
            "function" => choice.function.map(|function| ToolChoice::Tool {
                name: function.name,
                disable_parallel_tool_use: None,
            }),
            "auto" => Some(ToolChoice::Auto {
                disable_parallel_tool_use: None,
            }),
            "required" | "any" => Some(ToolChoice::Any {
                disable_parallel_tool_use: None,
            }),
            "none" => Some(ToolChoice::None),
            _ => None,
        },
    }
}

fn convert_response_format(format: ResponseFormat) -> Option<OutputFormat> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(OutputFormat::JsonSchema {
            schema: json!({
                "type": "object",
                "additionalProperties": true
            }),
        }),
        ResponseFormat::JsonSchema { json_schema } => Some(OutputFormat::JsonSchema {
            schema: json_schema.schema,
        }),
    }
}

fn compact_json(value: &Value) -> String {
    match value {
        Value::Null => "{}".to_string(),
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn converts_openai_tool_messages_to_claude_blocks() {
        let body = json!({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "system", "content": "You are precise."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Weather in Paris?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ]
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Paris\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "{\"temp\": 20}"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "stop": "END",
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
            "stream_options": {"include_usage": true}
        });

        let params: CreateMessageParams = serde_json::from_value(body).unwrap();
        assert!(params.stream_include_usage());

        let claude: ClaudeCreateMessageParams = params.into();
        assert_eq!(claude.stop_sequences, Some(vec!["END".to_string()]));
        assert!(claude.system.is_some());
        assert!(
            matches!(claude.tool_choice, Some(ToolChoice::Tool { name, .. }) if name == "get_weather")
        );
        assert!(matches!(
            claude.output_format,
            Some(OutputFormat::JsonSchema { .. })
        ));
        assert_eq!(claude.messages.len(), 3);

        match &claude.messages[1].content {
            MessageContent::Blocks { content } => {
                assert!(content.iter().any(|block| matches!(block, ContentBlock::ToolUse { name, .. } if name == "get_weather")));
            }
            other => panic!("expected assistant blocks, got {other:?}"),
        }

        match &claude.messages[2].content {
            MessageContent::Blocks { content } => {
                assert!(content.iter().any(|block| matches!(block, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "call_123")));
            }
            other => panic!("expected tool result blocks, got {other:?}"),
        }
    }

    #[test]
    fn accepts_stop_as_array() {
        let body = json!({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END", "STOP"]
        });

        let params: CreateMessageParams = serde_json::from_value(body).unwrap();
        let claude: ClaudeCreateMessageParams = params.into();
        assert_eq!(
            claude.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn converts_opencode_cache_controls_and_pdf_parts() {
        let body = json!({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the spec.",
                    "cache_control": { "type": "ephemeral" }
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this file."},
                        {
                            "type": "file",
                            "file": {
                                "filename": "spec.pdf",
                                "file_data": "data:application/pdf;base64,JVBERi0x"
                            },
                            "cache_control": { "type": "ephemeral" }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "read_spec",
                                "arguments": {"path": "spec.pdf"}
                            },
                            "cache_control": { "type": "ephemeral" }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "{\"ok\":true}",
                    "cache_control": { "type": "ephemeral" }
                }
            ]
        });

        let params: CreateMessageParams = serde_json::from_value(body).unwrap();
        let claude: ClaudeCreateMessageParams = params.into();

        let system = claude.system.unwrap();
        assert_eq!(
            system[0]["cache_control"]["type"].as_str(),
            Some("ephemeral")
        );

        match &claude.messages[0].content {
            MessageContent::Blocks { content } => {
                assert!(content.iter().any(|block| matches!(
                    block,
                    ContentBlock::Document {
                        title,
                        cache_control: Some(_),
                        ..
                    } if title.as_deref() == Some("spec.pdf")
                )));
            }
            other => panic!("expected user blocks, got {other:?}"),
        }

        match &claude.messages[1].content {
            MessageContent::Blocks { content } => {
                assert!(matches!(
                    content.last(),
                    Some(ContentBlock::ToolUse {
                        cache_control: Some(_),
                        ..
                    })
                ));
            }
            other => panic!("expected assistant blocks, got {other:?}"),
        }

        match &claude.messages[2].content {
            MessageContent::Blocks { content } => {
                assert!(matches!(
                    content.first(),
                    Some(ContentBlock::ToolResult {
                        cache_control: Some(_),
                        ..
                    })
                ));
            }
            other => panic!("expected tool result blocks, got {other:?}"),
        }
    }
}
