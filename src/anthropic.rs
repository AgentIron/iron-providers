use crate::{
    error::ProviderResult,
    model::{ChoiceRequest, ProviderEvent, ToolCall, CHOICE_REQUEST_TOOL_NAME},
    profile::ProviderProfile,
    stream_util::TerminatingStream,
    InferenceRequest, ProviderError,
};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

/// Monotonically increasing counter for generating unique fallback tool call IDs
/// when Anthropic omits the `id` field and no `block_index` is available.
static FALLBACK_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Default `max_tokens` for Anthropic requests when the caller does not
/// specify one. Anthropic requires this field; 4096 is a conservative
/// default that works across Claude 3.x models.
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 4096;

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
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    error: AnthropicErrorBody,
}

#[cfg(test)]
fn build_client(
    profile: &ProviderProfile,
    runtime: &crate::profile::RuntimeConfig,
) -> ProviderResult<Client> {
    let context = format!("profile '{}'", profile.slug);
    crate::http_client::build_http_client(crate::http_client::HttpClientParams {
        context: &context,
        api_key: &runtime.api_key,
        auth_strategy: &profile.auth_strategy,
        default_headers: &profile.default_headers,
        extra_headers: &[("anthropic-version", "2023-06-01")],
        connect_timeout: runtime.effective_connect_timeout(),
        read_timeout: runtime.effective_read_timeout(),
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
                let (call_id, tool_name) = resolve_tool_identity(
                    block.id.as_deref(),
                    block.name.as_deref(),
                    None,
                    "non-streaming response block",
                );
                let arguments = block.input.unwrap_or(Value::Null);
                if tool_name == CHOICE_REQUEST_TOOL_NAME {
                    match ChoiceRequest::from_value(arguments) {
                        Ok(request) => events.push(ProviderEvent::ChoiceRequest { request }),
                        Err(error) => events.push(ProviderEvent::Error {
                            source: ProviderError::malformed(format!(
                                "Invalid choice request payload from provider tool call: {}",
                                error
                            )),
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

fn build_anthropic_request(request: &InferenceRequest, stream: bool) -> AnthropicRequest {
    let messages = build_anthropic_messages(request);
    let tools = build_anthropic_tools(request);
    let tool_choice = map_tool_choice(request);

    AnthropicRequest {
        model: request.model.clone(),
        messages,
        system: request.instructions.clone(),
        tools,
        tool_choice,
        max_tokens: request
            .generation
            .max_tokens
            .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS),
        temperature: request.generation.temperature,
        top_p: request.generation.top_p,
        stream: if stream { Some(true) } else { None },
    }
}

pub async fn infer(
    client: Client,
    profile: &ProviderProfile,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    request.validate_model()?;
    let body = build_anthropic_request(&request, false);

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
    client: Client,
    profile: &ProviderProfile,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    request.validate_model()?;
    let body = build_anthropic_request(&request, true);

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

    let stream = TerminatingStream::new(crate::stream_util::process_sse_stream::<
        AnthropicSseAdapter,
    >(response))
    .boxed();

    Ok(stream)
}

/// SSE stream adapter for Anthropic Messages.
///
/// Anthropic uses `message_stop` events (not `[DONE]`) for completion, so
/// `handle_done` returns an empty vec (the `[DONE]` sentinel is ignored).
#[derive(Default)]
struct AnthropicSseAdapter {
    assembler: AnthropicStreamAssembler,
}

impl crate::stream_util::SseStreamAdapter for AnthropicSseAdapter {
    type Event = AnthropicStreamEvent;
    const LABEL: &'static str = "Anthropic";

    fn process_event(&mut self, event: AnthropicStreamEvent) -> Vec<ProviderResult<ProviderEvent>> {
        self.assembler.process_event(event)
    }

    fn handle_done(&mut self) -> Vec<ProviderResult<ProviderEvent>> {
        // Anthropic uses `message_stop` events for completion, not `[DONE]`.
        // Just skip the sentinel.
        vec![]
    }
}

/// Resolve a tool call's identity, warning on missing fields and synthesizing
/// a stable fallback `id` so downstream code can still route the call.
///
/// A missing `name` is left empty and logged — callers cannot route the call
/// without a name, but surfacing it as a `ToolCall` with an empty name is
/// consistent with the Chat Completions adapter and keeps contract parity.
fn resolve_tool_identity(
    id: Option<&str>,
    name: Option<&str>,
    block_index: Option<u32>,
    origin: &str,
) -> (String, String) {
    let id = match id.map(str::trim).filter(|s| !s.is_empty()) {
        Some(id) => id.to_string(),
        None => {
            let fallback = match block_index {
                Some(i) => format!("call_anthropic_{}", i),
                None => {
                    let seq = FALLBACK_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
                    format!("call_anthropic_{seq}")
                }
            };
            warn!(
                origin = origin,
                block_index = ?block_index,
                fallback_id = %fallback,
                "Anthropic tool_use block missing id, generating fallback"
            );
            fallback
        }
    };
    let name = match name.map(str::trim).filter(|s| !s.is_empty()) {
        Some(name) => name.to_string(),
        None => {
            warn!(
                origin = origin,
                block_index = ?block_index,
                tool_id = %id,
                "Anthropic tool_use block missing name; tool call is unroutable"
            );
            String::new()
        }
    };
    (id, name)
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
                            let (id, name) = resolve_tool_identity(
                                block.id.as_deref(),
                                block.name.as_deref(),
                                Some(event.index),
                                "content_block_start",
                            );
                            self.blocks.insert(
                                event.index,
                                AnthropicBlockState::ToolUse {
                                    id,
                                    name,
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
                                        source: ProviderError::malformed(format!(
                                            "Invalid choice request payload from provider tool call: {}",
                                            error
                                        )),
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
                    events.push(Ok(ProviderEvent::Error {
                        source: ProviderError::general(msg),
                    }));
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
        401 | 403 => ProviderError::auth(message),
        429 => ProviderError::rate_limit(message, None),
        400 => ProviderError::invalid_request(message),
        404 => ProviderError::model(message),
        _ => ProviderError::general(message),
    }
}

pub const SYSTEM_PROMPT_FRAGMENT: &str = include_str!("system_prompt_fragments/anthropic.md");

pub fn system_prompt_fragment() -> &'static str {
    SYSTEM_PROMPT_FRAGMENT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        profile::{AuthStrategy, RuntimeConfig},
        Message, RuntimeRecord, ToolPolicy, Transcript,
    };
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

    #[tokio::test]
    async fn test_infer_rejects_empty_model() {
        let profile = ProviderProfile::new(
            "test",
            crate::ApiFamily::AnthropicMessages,
            "https://example.com",
        )
        .with_auth(AuthStrategy::ApiKeyHeader {
            header_name: "x-api-key".into(),
        });
        let runtime = RuntimeConfig::new("test-key");
        let client = build_client(&profile, &runtime).unwrap();
        let request = InferenceRequest::new("   ", Transcript::new());

        let error = infer(client, &profile, request).await.unwrap_err();
        assert!(matches!(error, ProviderError::InvalidRequest { .. }));
        assert!(error.to_string().contains("model must be a non-empty"));
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
    fn test_stream_assembler_synthesizes_id_when_missing() {
        let mut assembler = AnthropicStreamAssembler::default();

        let events = [
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "name": "search"}
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"q\":\"rust\"}"}
            }),
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

        assert_eq!(tool_calls.len(), 1);
        assert!(
            tool_calls[0].call_id.starts_with("call_anthropic_"),
            "expected synthetic id, got {}",
            tool_calls[0].call_id
        );
        assert_eq!(tool_calls[0].tool_name, "search");
    }

    #[test]
    fn test_stream_assembler_emits_empty_name_with_warning_when_name_missing() {
        let mut assembler = AnthropicStreamAssembler::default();

        let events = [
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_x"}
            }),
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

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].call_id, "call_x");
        assert_eq!(tool_calls[0].tool_name, "");
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

    #[test]
    fn test_system_prompt_fragment_is_non_empty() {
        let fragment = system_prompt_fragment();
        assert!(!fragment.is_empty());
        assert_eq!(fragment, SYSTEM_PROMPT_FRAGMENT);
    }
}
