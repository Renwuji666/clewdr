use std::{collections::BTreeMap, fmt::Write, mem};

use base64::{Engine, prelude::BASE64_STANDARD};
use futures::{StreamExt, stream};
use itertools::Itertools;
use serde_json::Value;
use tracing::warn;
use wreq::multipart::{Form, Part};

use crate::{
    claude_web_state::ClaudeWebState,
    config::CLEWDR_CONFIG,
    types::{
        claude::{
            ContentBlock, CreateMessageParams, ImageSource, Message, MessageContent, Role,
            Tool as ClaudeTool, ToolChoice,
        },
        claude_web::request::*,
    },
    utils::{TIME_ZONE, print_out_text},
};

impl ClaudeWebState {
    pub fn transform_request(&self, mut value: CreateMessageParams) -> Option<WebRequestBody> {
        let tool_mode = request_has_tool_mode(&value);
        let system = value.system.take();
        let msgs = mem::take(&mut value.messages);
        let tools_manifest = value.tools.take().unwrap_or_default();
        let tool_choice = value.tool_choice.take();
        let system = merge_system(system.unwrap_or_default());
        let merged = if tool_mode {
            merge_messages_with_tool_shim(msgs, system, tools_manifest, tool_choice)?
        } else {
            merge_messages(msgs, system)?
        };

        let mut tools = vec![];
        if !tool_mode && CLEWDR_CONFIG.load().web_search {
            tools.push(Tool::web_search());
        }
        Some(WebRequestBody {
            max_tokens_to_sample: value.max_tokens,
            attachments: vec![Attachment::new(merged.paste)],
            files: vec![],
            model: if self.is_pro() {
                Some(value.model)
            } else {
                None
            },
            rendering_mode: if tool_mode {
                "raw".to_string()
            } else if value.stream.unwrap_or_default() {
                "messages".to_string()
            } else {
                "raw".to_string()
            },
            prompt: merged.prompt,
            timezone: TIME_ZONE.to_string(),
            images: merged.images,
            tools,
        })
    }

    pub async fn upload_images(&self, imgs: Vec<ImageSource>) -> Vec<String> {
        stream::iter(imgs)
            .filter_map(async |img| {
                let ImageSource::Base64 { media_type, data } = img else {
                    warn!("Image type is not base64");
                    return None;
                };
                let bytes = BASE64_STANDARD
                    .decode(data)
                    .inspect_err(|e| {
                        warn!("Failed to decode image: {}", e);
                    })
                    .ok()?;
                let file_name = match media_type.to_lowercase().as_str() {
                    "image/png" => "image.png",
                    "image/jpeg" => "image.jpg",
                    "image/jpg" => "image.jpg",
                    "image/gif" => "image.gif",
                    "image/webp" => "image.webp",
                    "application/pdf" => "document.pdf",
                    _ => "file",
                };
                let part = Part::bytes(bytes).file_name(file_name);
                let form = Form::new().part("file", part);
                let endpoint = self
                    .endpoint
                    .join(&format!("api/{}/upload", self.org_uuid.as_ref()?))
                    .expect("Url parse error");
                let res = self
                    .build_request(http::Method::POST, endpoint)
                    .multipart(form)
                    .send()
                    .await
                    .inspect_err(|e| {
                        warn!("Failed to upload image: {}", e);
                    })
                    .ok()?;
                #[derive(serde::Deserialize)]
                struct UploadResponse {
                    file_uuid: String,
                }
                let json = res
                    .json::<UploadResponse>()
                    .await
                    .inspect_err(|e| {
                        warn!("Failed to parse image response: {}", e);
                    })
                    .ok()?;
                Some(json.file_uuid)
            })
            .collect::<Vec<_>>()
            .await
    }
}

#[derive(Default, Debug)]
struct Merged {
    pub paste: String,
    pub prompt: String,
    pub images: Vec<ImageSource>,
}

fn request_has_tool_mode(value: &CreateMessageParams) -> bool {
    value.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        || messages_have_tool_blocks(&value.messages)
}

fn messages_have_tool_blocks(msgs: &[Message]) -> bool {
    msgs.iter().any(|msg| match &msg.content {
        MessageContent::Blocks { content } => content.iter().any(is_tool_block),
        MessageContent::Text { .. } => false,
    })
}

fn is_tool_block(block: &ContentBlock) -> bool {
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
}

fn merge_messages_with_tool_shim(
    msgs: Vec<Message>,
    system: String,
    tools: Vec<ClaudeTool>,
    tool_choice: Option<ToolChoice>,
) -> Option<Merged> {
    if msgs.is_empty() {
        return None;
    }

    let h = CLEWDR_CONFIG
        .load()
        .custom_h
        .to_owned()
        .unwrap_or("Human".to_string());
    let a = CLEWDR_CONFIG
        .load()
        .custom_a
        .to_owned()
        .unwrap_or("Assistant".to_string());
    let line_breaks = if CLEWDR_CONFIG.load().use_real_roles {
        "\n\n\x08"
    } else {
        "\n\n"
    };

    let mut w = build_tool_system(system, &tools, tool_choice);
    let mut imgs = Vec::new();
    let mut tool_names = BTreeMap::new();

    for msg in msgs {
        for segment in render_tool_segments(msg, &mut imgs, &mut tool_names, &h, &a) {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }
            if !w.is_empty() {
                w.push_str(line_breaks);
            }
            w.push_str(segment);
        }
    }

    if w.trim().is_empty() {
        return None;
    }

    print_out_text(w.to_owned(), "paste.txt");
    Some(Merged {
        paste: w,
        prompt: CLEWDR_CONFIG.load().custom_prompt.to_owned(),
        images: imgs,
    })
}

fn build_tool_system(system: String, tools: &[ClaudeTool], tool_choice: Option<ToolChoice>) -> String {
    let mut sections = Vec::new();
    let system = system.trim();
    if !system.is_empty() {
        sections.push(system.to_string());
    }

    let tool_list = if tools.is_empty() {
        "<available_tools />".to_string()
    } else {
        format!("<available_tools>\n{}\n</available_tools>", format_tool_list(tools))
    };

    let tool_policy = match tool_choice {
        Some(ToolChoice::Any { .. }) => {
            "Tool policy: you must emit at least one <tool_call> block before you give a final answer."
                .to_string()
        }
        Some(ToolChoice::Tool { name, .. }) => format!(
            "Tool policy: your next response must call the exact tool `{name}`.",
        ),
        Some(ToolChoice::None) => {
            "Tool policy: tool use is disabled for this turn; respond with plain text only."
                .to_string()
        }
        _ => "Tool policy: use tools when needed, otherwise answer in normal plain text.".to_string(),
    };

    sections.push(format!(
        concat!(
            "You are connected to client-side tools through a compatibility layer.\n\n",
            "If you need one or more tools, emit one or more XML blocks in this exact format:\n",
            "<tool_call name=\"TOOL_NAME\">{{\"arg\":\"value\"}}</tool_call>\n\n",
            "Rules:\n",
            "- Use only tool names listed in AVAILABLE TOOLS.\n",
            "- Use the exact tool name. Never invent aliases such as `view` or `bash_tool`.\n",
            "- The body of each <tool_call> block must be valid JSON matching the tool schema.\n",
            "- If you do not need a tool, answer in normal plain text.\n",
            "- Never claim you executed a command, inspected files, or fetched the web unless a matching tool result is present in the transcript.\n",
            "- For facts about the filesystem, current working directory, git state, command output, or live web content, call tools instead of guessing.\n",
            "- Tool results from earlier steps appear as <tool_result ...> blocks below. Use them before deciding your next step.\n",
            "- Prefer `read`/`glob`/`grep` for filesystem inspection, `bash` for shell commands, and `webfetch` for live web content.\n\n",
            "{tool_policy}\n\n",
            "AVAILABLE TOOLS:\n{tool_list}"
        ),
        tool_policy = tool_policy,
        tool_list = tool_list,
    ));

    sections.join("\n\n")
}

fn format_tool_list(tools: &[ClaudeTool]) -> String {
    tools
        .iter()
        .filter_map(format_tool_entry)
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_tool_entry(tool: &ClaudeTool) -> Option<String> {
    let value = serde_json::to_value(tool).ok()?;
    let name = value.get("name")?.as_str()?;
    let mut out = format!("<tool name=\"{name}\">");
    if let Some(description) = value.get("description").and_then(Value::as_str) {
        out.push_str("\n<description>");
        out.push_str(description.trim());
        out.push_str("</description>");
    }
    if let Some(schema) = value.get("input_schema") {
        out.push_str("\n<input_schema>");
        out.push_str(&compact_json(schema));
        out.push_str("</input_schema>");
    } else {
        out.push_str("\n<definition>");
        out.push_str(&compact_json(&value));
        out.push_str("</definition>");
    }
    out.push_str("\n</tool>");
    Some(out)
}

fn compact_json(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn render_tool_segments(
    msg: Message,
    imgs: &mut Vec<ImageSource>,
    tool_names: &mut BTreeMap<String, String>,
    h: &str,
    a: &str,
) -> Vec<String> {
    let label = match msg.role {
        Role::User => Some(h),
        Role::Assistant => Some(a),
        Role::System => None,
    };
    let mut segments = Vec::new();
    let mut text_parts = Vec::new();

    match msg.content {
        MessageContent::Text { content } => {
            let content = content.trim();
            if !content.is_empty() {
                if let Some(label) = label {
                    segments.push(format!("{label}: {content}"));
                } else {
                    segments.push(content.to_string());
                }
            }
            return segments;
        }
        MessageContent::Blocks { content } => {
            for block in content {
                match block {
                    ContentBlock::Text { text, .. } => {
                        let text = text.trim();
                        if !text.is_empty() {
                            text_parts.push(text.to_string());
                        }
                    }
                    ContentBlock::Image { source, .. } => {
                        match source {
                            ImageSource::Base64 { .. } => imgs.push(source),
                            ImageSource::Url { url } => {
                                if let Some(source) = extract_image_from_url(&url) {
                                    imgs.push(source);
                                } else {
                                    warn!("Unsupported image url source");
                                }
                            }
                            ImageSource::File { .. } => {
                                warn!("Image file sources are not supported");
                            }
                        }
                    }
                    ContentBlock::ImageUrl { image_url } => {
                        if let Some(source) = extract_image_from_url(&image_url.url) {
                            imgs.push(source);
                        }
                    }
                    ContentBlock::ToolUse {
                        id, name, input, ..
                    }
                    | ContentBlock::ServerToolUse {
                        id, name, input, ..
                    }
                    | ContentBlock::McpToolUse {
                        id, name, input, ..
                    } => {
                        flush_text_segment(&mut text_parts, &mut segments, label);
                        tool_names.insert(id.clone(), name.clone());
                        segments.push(format!(
                            "{} requested tool `{}`:\n<tool_call name=\"{}\">{}</tool_call>",
                            label.unwrap_or(a),
                            name,
                            name,
                            compact_json(&input)
                        ));
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    }
                    | ContentBlock::McpToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    } => {
                        flush_text_segment(&mut text_parts, &mut segments, label);
                        segments.push(render_tool_result_segment(
                            tool_names,
                            &tool_use_id,
                            &content,
                            is_error.unwrap_or(false),
                        ));
                    }
                    ContentBlock::WebSearchToolResult {
                        tool_use_id,
                        content,
                        ..
                    }
                    | ContentBlock::WebFetchToolResult {
                        tool_use_id,
                        content,
                        ..
                    }
                    | ContentBlock::CodeExecutionToolResult {
                        tool_use_id,
                        content,
                        ..
                    }
                    | ContentBlock::BashCodeExecutionToolResult {
                        tool_use_id,
                        content,
                        ..
                    }
                    | ContentBlock::TextEditorCodeExecutionToolResult {
                        tool_use_id,
                        content,
                        ..
                    }
                    | ContentBlock::ToolSearchToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        flush_text_segment(&mut text_parts, &mut segments, label);
                        segments.push(render_tool_result_segment(
                            tool_names,
                            &tool_use_id,
                            &content,
                            false,
                        ));
                    }
                    _ => {}
                }
            }
        }
    }

    flush_text_segment(&mut text_parts, &mut segments, label);
    segments
}

fn flush_text_segment(text_parts: &mut Vec<String>, segments: &mut Vec<String>, label: Option<&str>) {
    if text_parts.is_empty() {
        return;
    }
    let joined = text_parts.join("\n");
    text_parts.clear();
    let joined = joined.trim();
    if joined.is_empty() {
        return;
    }
    if let Some(label) = label {
        segments.push(format!("{label}: {joined}"));
    } else {
        segments.push(joined.to_string());
    }
}

fn render_tool_result_segment(
    tool_names: &BTreeMap<String, String>,
    tool_use_id: &str,
    content: &Value,
    is_error: bool,
) -> String {
    let name = tool_names
        .get(tool_use_id)
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let status = if is_error { " status=\"error\"" } else { "" };
    format!(
        "Tool result from `{}` (tool_use_id: {}):\n<tool_result name=\"{}\" tool_use_id=\"{}\"{}>{}</tool_result>",
        name,
        tool_use_id,
        name,
        tool_use_id,
        status,
        value_as_transcript(content)
    )
}

fn value_as_transcript(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::String(text) => text.to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn merge_messages(msgs: Vec<Message>, system: String) -> Option<Merged> {
    if msgs.is_empty() {
        return None;
    }
    let h = CLEWDR_CONFIG
        .load()
        .custom_h
        .to_owned()
        .unwrap_or("Human".to_string());
    let a = CLEWDR_CONFIG
        .load()
        .custom_a
        .to_owned()
        .unwrap_or("Assistant".to_string());

    let user_real_roles = CLEWDR_CONFIG.load().use_real_roles;
    let line_breaks = if user_real_roles { "\n\n\x08" } else { "\n\n" };
    let system = system.trim().to_string();
    let mut w = String::new();

    let mut imgs: Vec<ImageSource> = vec![];

    let chunks = msgs
        .into_iter()
        .filter_map(|m| match m.content {
            MessageContent::Blocks { content } => {
                let blocks = content
                    .into_iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(text.trim().to_string()),
                        ContentBlock::Image { source, .. } => {
                            match source {
                                ImageSource::Base64 { .. } => imgs.push(source),
                                ImageSource::Url { url } => {
                                    if let Some(source) = extract_image_from_url(&url) {
                                        imgs.push(source);
                                    } else {
                                        warn!("Unsupported image url source");
                                    }
                                }
                                ImageSource::File { .. } => {
                                    warn!("Image file sources are not supported");
                                }
                            }
                            None
                        }
                        ContentBlock::ImageUrl { image_url } => {
                            if let Some(source) = extract_image_from_url(&image_url.url) {
                                imgs.push(source);
                            }
                            None
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if blocks.is_empty() {
                    None
                } else {
                    Some((m.role, blocks))
                }
            }
            MessageContent::Text { content } => {
                let content = content.trim().to_string();
                if content.is_empty() {
                    None
                } else {
                    Some((m.role, content))
                }
            }
        })
        .chunk_by(|m| m.0);
    let mut msgs = chunks.into_iter().map(|(role, grp)| {
        let txt = grp.into_iter().map(|m| m.1).collect::<Vec<_>>().join("\n");
        (role, txt)
    });
    if !system.is_empty() {
        w += system.as_str();
    } else {
        let first = msgs.next()?;
        w += first.1.as_str();
    }
    for (role, text) in msgs {
        let prefix = match role {
            Role::System => {
                warn!("System message should be merged into the first message");
                continue;
            }
            Role::User => format!("{h}: "),
            Role::Assistant => format!("{a}: "),
        };
        write!(w, "{line_breaks}{prefix}{text}").ok()?;
    }
    print_out_text(w.to_owned(), "paste.txt");

    let p = CLEWDR_CONFIG.load().custom_prompt.to_owned();

    Some(Merged {
        paste: w,
        prompt: p,
        images: imgs,
    })
}

fn merge_system(sys: Value) -> String {
    match sys {
        Value::String(s) => s,
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v["text"].as_str())
            .map(|v| v.trim())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn extract_image_from_url(url: &str) -> Option<ImageSource> {
    if !url.starts_with("data:") {
        return None;
    }
    let (metadata, base64_data) = url.split_once(',')?;

    let (media_type, type_) = metadata.strip_prefix("data:")?.split_once(';')?;
    if type_ != "base64" {
        return None;
    }

    Some(ImageSource::Base64 {
        media_type: media_type.to_string(),
        data: base64_data.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::claude::{CustomTool, Tool};

    #[test]
    fn merges_tool_history_into_prompt() {
        let merged = merge_messages_with_tool_shim(
            vec![
                Message::new_text(Role::User, "where am i"),
                Message::new_blocks(
                    Role::Assistant,
                    vec![ContentBlock::ToolUse {
                        id: "toolu_1".into(),
                        name: "bash".into(),
                        input: json!({"command": "pwd", "description": "Print cwd"}),
                        cache_control: None,
                        caller: None,
                    }],
                ),
                Message::new_blocks(
                    Role::User,
                    vec![ContentBlock::ToolResult {
                        tool_use_id: "toolu_1".into(),
                        content: json!("/root"),
                        cache_control: None,
                        is_error: None,
                    }],
                ),
            ],
            "system text".into(),
            vec![Tool::Custom(CustomTool {
                name: "bash".into(),
                description: Some("Run shell commands".into()),
                input_schema: json!({"type": "object"}),
                allowed_callers: None,
                cache_control: None,
                defer_loading: None,
                input_examples: None,
                strict: None,
                type_: None,
                extra: Default::default(),
            })],
            Some(ToolChoice::Auto {
                disable_parallel_tool_use: None,
            }),
        )
        .expect("merged prompt");

        assert!(merged.paste.contains("<available_tools>"));
        assert!(merged.paste.contains("<tool_call name=\"bash\">"));
        assert!(merged.paste.contains("<tool_result name=\"bash\" tool_use_id=\"toolu_1\">"));
    }
}
