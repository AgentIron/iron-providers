use crate::{
    error::ProviderResult,
    model::{ProviderEvent, ToolCall},
    profile::{AuthStrategy, ProviderProfile, RuntimeConfig},
    InferenceRequest, ProviderError,
};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
    id: Option<String>,
    name: Option<String>,
    input: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicDelta>,
    content_block: Option<AnthropicContentBlock>,
    error: Option<AnthropicErrorBody>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
    partial_json: Option<String>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    error: AnthropicErrorBody,
}

fn build_client(profile: &ProviderProfile, runtime: &RuntimeConfig) -> Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        "anthropic-version",
        reqwest::header::HeaderValue::from_static("2023-06-01"),
    );

    match &profile.auth_strategy {
        AuthStrategy::BearerToken => {
            if let Ok(val) =
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", runtime.api_key))
            {
                headers.insert(reqwest::header::AUTHORIZATION, val);
            }
        }
        AuthStrategy::ApiKeyHeader { header_name } => {
            if let Ok(val) = reqwest::header::HeaderValue::from_str(&runtime.api_key) {
                if let Ok(key) = reqwest::header::HeaderName::from_bytes(header_name.as_bytes()) {
                    headers.insert(key, val);
                }
            }
        }
        AuthStrategy::Custom {
            header_name,
            prefix,
        } => {
            let value = match prefix {
                Some(p) => format!("{} {}", p, runtime.api_key),
                None => runtime.api_key.clone(),
            };
            if let Ok(val) = reqwest::header::HeaderValue::from_str(&value) {
                if let Ok(key) = reqwest::header::HeaderName::from_bytes(header_name.as_bytes()) {
                    headers.insert(key, val);
                }
            }
        }
    }

    for (key, value) in &profile.default_headers {
        if let (Ok(hk), Ok(hv)) = (
            reqwest::header::HeaderName::from_bytes(key.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            headers.insert(hk, hv);
        }
    }

    Client::builder()
        .default_headers(headers)
        .build()
        .unwrap_or_default()
}

fn build_anthropic_messages(request: &InferenceRequest) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut pending_tool_calls: Vec<Value> = Vec::new();

    for msg in &request.transcript.messages {
        match msg {
            crate::Message::User { content } => {
                messages.push(json!({
                    "role": "user",
                    "content": [{"type": "text", "text": content}]
                }));
            }
            crate::Message::Assistant { content } => {
                let mut blocks: Vec<Value> = vec![json!({"type": "text", "text": content})];
                blocks.append(&mut pending_tool_calls);
                messages.push(json!({
                    "role": "assistant",
                    "content": blocks
                }));
            }
            crate::Message::AssistantToolCall {
                call_id,
                tool_name,
                arguments,
            } => {
                pending_tool_calls.push(json!({
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool_name,
                    "input": arguments
                }));
            }
            crate::Message::Tool {
                call_id,
                tool_name: _,
                result,
            } => {
                if !pending_tool_calls.is_empty() {
                    messages.push(json!({
                        "role": "assistant",
                        "content": pending_tool_calls.clone()
                    }));
                    pending_tool_calls.clear();
                }
                messages.push(json!({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": call_id, "content": serde_json::to_string(result).unwrap_or_else(|_| result.to_string())}]
                }));
            }
        }
    }

    if !pending_tool_calls.is_empty() {
        messages.push(json!({
            "role": "assistant",
            "content": pending_tool_calls
        }));
    }

    messages
}

fn build_anthropic_tools(request: &InferenceRequest) -> Option<Vec<Value>> {
    if request.tools.is_empty() {
        return None;
    }
    Some(
        request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                })
            })
            .collect(),
    )
}

fn map_tool_choice(request: &InferenceRequest) -> Option<Value> {
    match request.tool_policy {
        crate::ToolPolicy::None => None,
        crate::ToolPolicy::Auto => Some(json!({"type": "auto"})),
        crate::ToolPolicy::Required => Some(json!({"type": "any"})),
        crate::ToolPolicy::Specific(ref name) => Some(json!({"type": "tool", "name": name})),
    }
}

pub async fn infer(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    let client = build_client(profile, runtime);
    let messages = build_anthropic_messages(&request);
    let tools = build_anthropic_tools(&request);
    let tool_choice = map_tool_choice(&request);

    let body = AnthropicRequest {
        model: request.model,
        messages,
        system: request.instructions,
        tools,
        tool_choice,
        max_tokens: request.generation.max_tokens.unwrap_or(4096),
        temperature: request.generation.temperature,
        top_p: request.generation.top_p,
        stream: None,
    };

    let url = format!("{}/v1/messages", profile.base_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| ProviderError::transport(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(handle_error(status, &text));
    }

    let result: AnthropicResponse = response.json().await.map_err(|e| {
        ProviderError::malformed(format!("Failed to parse Anthropic response: {}", e))
    })?;

    Ok(response_to_events(result))
}

fn response_to_events(response: AnthropicResponse) -> Vec<ProviderEvent> {
    let mut events = Vec::new();

    for block in response.content {
        match block.block_type.as_str() {
            "text" => {
                if let Some(text) = block.text {
                    events.push(ProviderEvent::Output { content: text });
                }
            }
            "tool_use" => {
                let call_id = block.id.unwrap_or_default();
                let tool_name = block.name.unwrap_or_default();
                let arguments = block.input.unwrap_or(Value::Null);
                events.push(ProviderEvent::ToolCall {
                    call: ToolCall::new(call_id, tool_name, arguments),
                });
            }
            _ => {}
        }
    }

    events.push(ProviderEvent::Complete);
    events
}

pub async fn infer_stream(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    let client = build_client(profile, runtime);
    let messages = build_anthropic_messages(&request);
    let tools = build_anthropic_tools(&request);
    let tool_choice = map_tool_choice(&request);

    let body = AnthropicRequest {
        model: request.model,
        messages,
        system: request.instructions,
        tools,
        tool_choice,
        max_tokens: request.generation.max_tokens.unwrap_or(4096),
        temperature: request.generation.temperature,
        top_p: request.generation.top_p,
        stream: Some(true),
    };

    let url = format!("{}/v1/messages", profile.base_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| ProviderError::transport(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(handle_error(status, &text));
    }

    let stream = process_sse_stream(response).boxed();

    Ok(stream)
}

fn process_sse_stream(
    response: reqwest::Response,
) -> impl futures::Stream<Item = ProviderResult<ProviderEvent>> {
    let byte_stream = response.bytes_stream();

    let mut buffer = String::new();
    let mut tool_call_accumulator: Option<ToolCallAccumulator> = None;

    byte_stream.flat_map(move |chunk_result| {
        let mut events = Vec::new();

        match chunk_result {
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            continue;
                        }

                        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                            let (mut new_events, new_accum) =
                                process_stream_event(event, tool_call_accumulator.take());
                            events.append(&mut new_events);
                            tool_call_accumulator = new_accum;
                        }
                    }
                }
            }
            Err(e) => {
                events.push(Err(ProviderError::transport(e.to_string())));
            }
        }

        futures::stream::iter(events)
    })
}

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

fn process_stream_event(
    event: AnthropicStreamEvent,
    accumulator: Option<ToolCallAccumulator>,
) -> (
    Vec<ProviderResult<ProviderEvent>>,
    Option<ToolCallAccumulator>,
) {
    let mut events = Vec::new();
    let mut acc = accumulator;

    match event.event_type.as_str() {
        "content_block_delta" => {
            if let Some(delta) = event.delta {
                match delta.delta_type.as_deref() {
                    Some("text_delta") => {
                        if let Some(text) = delta.text {
                            events.push(Ok(ProviderEvent::Output { content: text }));
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = delta.partial_json {
                            if let Some(ref mut a) = acc {
                                a.arguments.push_str(&partial);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "content_block_start" => {
            if let Some(block) = event.content_block {
                if block.block_type == "tool_use" {
                    acc = Some(ToolCallAccumulator {
                        id: block.id.unwrap_or_default(),
                        name: block.name.unwrap_or_default(),
                        arguments: String::new(),
                    });
                }
            }
        }
        "content_block_stop" => {
            if let Some(a) = acc.take() {
                let arguments = serde_json::from_str(&a.arguments)
                    .unwrap_or(Value::String(a.arguments.clone()));
                events.push(Ok(ProviderEvent::ToolCall {
                    call: ToolCall::new(a.id, a.name, arguments),
                }));
            }
        }
        "message_stop" => {
            events.push(Ok(ProviderEvent::Complete));
        }
        "error" => {
            if let Some(err) = event.error {
                let msg = err.message.unwrap_or_else(|| "Unknown error".to_string());
                events.push(Ok(ProviderEvent::Error { message: msg }));
            }
        }
        _ => {}
    }

    (events, acc)
}

fn handle_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
    let message = if let Ok(err) = serde_json::from_str::<AnthropicError>(body) {
        err.error.message.unwrap_or_else(|| body.to_string())
    } else {
        body.to_string()
    };

    match status.as_u16() {
        401 => ProviderError::auth(message),
        429 => ProviderError::rate_limit(message, None),
        400 => ProviderError::invalid_request(message),
        404 => ProviderError::model(message),
        _ => ProviderError::general(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ToolPolicy, Transcript};
    use serde_json::json;

    #[test]
    fn test_build_anthropic_messages_basic() {
        let transcript =
            Transcript::with_messages(vec![Message::user("hello"), Message::assistant("hi there")]);
        let request = InferenceRequest::new("claude-3", transcript);
        let messages = build_anthropic_messages(&request);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
    }

    #[test]
    fn test_build_anthropic_messages_with_tools() {
        let transcript = Transcript::with_messages(vec![
            Message::user("check weather"),
            Message::AssistantToolCall {
                call_id: "call_1".into(),
                tool_name: "weather".into(),
                arguments: json!({"city": "SF"}),
            },
            Message::tool("call_1", "weather", json!({"temp": 72})),
        ]);
        let request = InferenceRequest::new("claude-3", transcript);
        let messages = build_anthropic_messages(&request);

        assert_eq!(messages.len(), 3);

        let assistant = &messages[1];
        assert_eq!(assistant["role"], "assistant");
        let content = assistant["content"].as_array().expect("content array");
        assert!(content.iter().any(|b| b["type"] == "tool_use"));
    }

    #[test]
    fn test_build_anthropic_tools() {
        let request = InferenceRequest::new("claude-3", Transcript::new()).with_tools(vec![
            crate::ToolDefinition::new(
                "lookup",
                "lookup a value",
                json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            ),
        ]);

        let tools = build_anthropic_tools(&request).expect("tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "lookup");
    }

    #[test]
    fn test_map_tool_choice() {
        let request =
            InferenceRequest::new("claude-3", Transcript::new()).with_tool_policy(ToolPolicy::Auto);
        let choice = map_tool_choice(&request);
        assert_eq!(choice, Some(json!({"type": "auto"})));

        let request = InferenceRequest::new("claude-3", Transcript::new())
            .with_tool_policy(ToolPolicy::Required);
        let choice = map_tool_choice(&request);
        assert_eq!(choice, Some(json!({"type": "any"})));

        let request =
            InferenceRequest::new("claude-3", Transcript::new()).with_tool_policy(ToolPolicy::None);
        assert!(map_tool_choice(&request).is_none());
    }

    #[test]
    fn test_system_prompt_handling() {
        let request =
            InferenceRequest::new("claude-3", Transcript::new()).with_instructions("Be helpful");
        let messages = build_anthropic_messages(&request);
        assert!(
            messages.is_empty(),
            "system prompt should not be in messages"
        );
    }
}
