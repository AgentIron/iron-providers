use crate::{
    error::ProviderResult,
    model::{ChoiceRequest, ProviderEvent, ToolCall, CHOICE_REQUEST_TOOL_NAME},
    profile::{AuthStrategy, ProviderProfile, RuntimeConfig},
    sse::SseParser,
    stream_util::{truncate_for_log, TerminatingStream},
    InferenceRequest, ProviderError,
};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::warn;

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
    /// Content block index — identifies which block this event belongs to.
    #[serde(default)]
    index: u32,
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

fn build_client(profile: &ProviderProfile, runtime: &RuntimeConfig) -> ProviderResult<Client> {
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
            let val =
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", runtime.api_key))
                    .map_err(|e| {
                        ProviderError::invalid_request(format!(
                            "Invalid bearer token value for profile '{}': {}",
                            profile.slug, e
                        ))
                    })?;
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
        AuthStrategy::ApiKeyHeader { header_name } => {
            let key =
                reqwest::header::HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid header name '{}' for profile '{}': {}",
                        header_name, profile.slug, e
                    ))
                })?;
            let val = reqwest::header::HeaderValue::from_str(&runtime.api_key).map_err(|e| {
                ProviderError::invalid_request(format!(
                    "Invalid API key value for profile '{}': {}",
                    profile.slug, e
                ))
            })?;
            headers.insert(key, val);
        }
        AuthStrategy::Custom {
            header_name,
            prefix,
        } => {
            let value = match prefix {
                Some(p) => format!("{} {}", p, runtime.api_key),
                None => runtime.api_key.clone(),
            };
            let key =
                reqwest::header::HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid custom header name '{}' for profile '{}': {}",
                        header_name, profile.slug, e
                    ))
                })?;
            let val = reqwest::header::HeaderValue::from_str(&value).map_err(|e| {
                ProviderError::invalid_request(format!(
                    "Invalid custom header value for profile '{}': {}",
                    profile.slug, e
                ))
            })?;
            headers.insert(key, val);
        }
    }

    for (key, value) in &profile.default_headers {
        let hk = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid default header name '{}' for profile '{}': {}",
                key, profile.slug, e
            ))
        })?;
        let hv = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid default header value for '{}' on profile '{}': {}",
                key, profile.slug, e
            ))
        })?;
        headers.insert(hk, hv);
    }

    Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| {
            ProviderError::general(format!(
                "Failed to build HTTP client for profile '{}': {}",
                profile.slug, e
            ))
        })
}

fn build_anthropic_messages(request: &InferenceRequest) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut pending_tool_calls: Vec<Value> = Vec::new();

    for msg in &request.context.transcript.messages {
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
                let result_str = serde_json::to_string(result)
                    .expect("serde_json::to_string on Value is infallible");
                messages.push(json!({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result_str}]
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
                if tool_name == CHOICE_REQUEST_TOOL_NAME {
                    match ChoiceRequest::from_value(arguments) {
                        Ok(request) => events.push(ProviderEvent::ChoiceRequest { request }),
                        Err(error) => events.push(ProviderEvent::Error {
                            message: format!(
                                "Invalid choice request payload from provider tool call: {}",
                                error
                            ),
                        }),
                    }
                } else {
                    events.push(ProviderEvent::ToolCall {
                        call: ToolCall::new(call_id, tool_name, arguments),
                    });
                }
            }
            _ => {}
        }
    }

    events.push(ProviderEvent::Complete);
    events
}

pub async fn infer(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    let client = build_client(profile, runtime)?;
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

pub async fn infer_stream(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    let client = build_client(profile, runtime)?;
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

    let stream = TerminatingStream::new(process_sse_stream(response)).boxed();

    Ok(stream)
}

fn process_sse_stream(
    response: reqwest::Response,
) -> impl futures::Stream<Item = ProviderResult<ProviderEvent>> {
    let byte_stream = response.bytes_stream();

    let mut sse_parser = SseParser::new();
    let mut assembler = AnthropicStreamAssembler::default();

    byte_stream.flat_map(move |chunk_result| {
        let mut events = Vec::new();

        match chunk_result {
            Ok(bytes) => {
                let sse_events = sse_parser.feed(&bytes);

                for sse_event in sse_events {
                    if sse_event.data == "[DONE]" {
                        continue;
                    }

                    match serde_json::from_str::<AnthropicStreamEvent>(&sse_event.data) {
                        Ok(event) => {
                            let new_events = assembler.process_event(event);
                            events.extend(new_events);
                        }
                        Err(error) => {
                            warn!(
                                error = %error,
                                payload = %truncate_for_log(&sse_event.data),
                                "Failed to parse Anthropic stream event, skipping"
                            );
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

/// State for a single content block being assembled from stream events.
#[derive(Debug)]
enum AnthropicBlockState {
    /// Text blocks emit deltas immediately; this variant just tracks that
    /// a text block is open so `content_block_stop` knows there's nothing
    /// to finalize.
    Text,
    ToolUse {
        id: String,
        name: String,
        arguments_json: String,
    },
}

/// Per-block assembler that tracks multiple concurrent content blocks
/// by their index, supporting interleaved text and tool-use blocks.
#[derive(Debug, Default)]
struct AnthropicStreamAssembler {
    blocks: std::collections::BTreeMap<u32, AnthropicBlockState>,
}

impl AnthropicStreamAssembler {
    fn process_event(&mut self, event: AnthropicStreamEvent) -> Vec<ProviderResult<ProviderEvent>> {
        let mut events = Vec::new();

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
                                if let Some(AnthropicBlockState::ToolUse {
                                    arguments_json, ..
                                }) = self.blocks.get_mut(&event.index)
                                {
                                    arguments_json.push_str(&partial);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_start" => {
                if let Some(block) = event.content_block {
                    match block.block_type.as_str() {
                        "tool_use" => {
                            self.blocks.insert(
                                event.index,
                                AnthropicBlockState::ToolUse {
                                    id: block.id.unwrap_or_default(),
                                    name: block.name.unwrap_or_default(),
                                    arguments_json: String::new(),
                                },
                            );
                        }
                        "text" => {
                            self.blocks.insert(event.index, AnthropicBlockState::Text);
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                if let Some(state) = self.blocks.remove(&event.index) {
                    match state {
                        AnthropicBlockState::ToolUse {
                            id,
                            name,
                            arguments_json,
                        } => {
                            let arguments = serde_json::from_str(&arguments_json)
                                .unwrap_or(Value::String(arguments_json.clone()));
                            if name == CHOICE_REQUEST_TOOL_NAME {
                                match ChoiceRequest::from_value(arguments) {
                                    Ok(request) => {
                                        events.push(Ok(ProviderEvent::ChoiceRequest { request }))
                                    }
                                    Err(error) => events.push(Ok(ProviderEvent::Error {
                                        message: format!(
                                            "Invalid choice request payload from provider tool call: {}",
                                            error
                                        ),
                                    })),
                                }
                            } else {
                                events.push(Ok(ProviderEvent::ToolCall {
                                    call: ToolCall::new(id, name, arguments),
                                }));
                            }
                        }
                        AnthropicBlockState::Text => {
                            // Text deltas are emitted immediately; nothing to finalize.
                        }
                    }
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

        events
    }
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
    use crate::{Message, RuntimeRecord, ToolPolicy, Transcript};
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

    #[test]
    fn test_build_client_fails_fast_on_invalid_default_header_name() {
        let profile = ProviderProfile::new(
            "broken",
            crate::ApiFamily::AnthropicMessages,
            "https://example.com",
        )
        .with_auth(AuthStrategy::ApiKeyHeader {
            header_name: "x-api-key".into(),
        })
        .with_header("bad header", "value");

        let error = build_client(&profile, &RuntimeConfig::new("test-key")).unwrap_err();
        assert!(matches!(error, ProviderError::InvalidRequest { .. }));
        assert!(error.to_string().contains("Invalid default header name"));
    }

    #[test]
    fn test_build_anthropic_messages_does_not_project_runtime_records() {
        let mut request = InferenceRequest::new(
            "claude-3",
            Transcript::with_messages(vec![Message::user("hello")]),
        );
        request
            .context
            .add_record(RuntimeRecord::new("session_state", json!({"secret": true})));

        let messages = build_anthropic_messages(&request);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn test_stream_assembler_tracks_multiple_tool_blocks_independently() {
        let mut assembler = AnthropicStreamAssembler::default();

        let events = [
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_1", "name": "search"}
            }),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "call_2", "name": "calculate"}
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"query\":\"rust\"}"}
            }),
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"expr\":\"1+1\"}"}
            }),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_stop", "index": 0}),
        ];

        let mut emitted = Vec::new();
        for value in events {
            let event: AnthropicStreamEvent = serde_json::from_value(value).expect("stream event");
            emitted.extend(assembler.process_event(event));
        }

        let tool_calls: Vec<_> = emitted
            .into_iter()
            .filter_map(|event| match event {
                Ok(ProviderEvent::ToolCall { call }) => Some(call),
                _ => None,
            })
            .collect();

        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].call_id, "call_2");
        assert_eq!(tool_calls[0].tool_name, "calculate");
        assert_eq!(tool_calls[0].arguments["expr"], "1+1");
        assert_eq!(tool_calls[1].call_id, "call_1");
        assert_eq!(tool_calls[1].tool_name, "search");
        assert_eq!(tool_calls[1].arguments["query"], "rust");
    }
}
