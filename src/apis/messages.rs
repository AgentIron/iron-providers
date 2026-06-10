//! Anthropic Messages API adapter
//!
//! Handles request projection, response parsing, and streaming normalization
//! for Messages-family providers.

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
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
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
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    /// Present on `message_start` events.
    #[serde(default)]
    message: Option<AnthropicMessageStart>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
    partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    error: AnthropicErrorBody,
}

static FALLBACK_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

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

fn map_anthropic_usage(usage: &AnthropicUsage) -> crate::TokenUsage {
    crate::TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: None,
        cached_input_tokens: None,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
        reasoning_output_tokens: None,
    }
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

    if let Some(ref usage) = response.usage {
        events.push(ProviderEvent::Usage {
            usage: map_anthropic_usage(usage),
        });
    }

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

pub(crate) async fn infer(
    client: Client,
    _profile: &ProviderProfile,
    effective_base_url: &str,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    request.validate_model()?;
    let body = build_anthropic_request(&request, false);

    let url = format!("{}/v1/messages", effective_base_url.trim_end_matches('/'));
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

pub(crate) async fn infer_stream(
    client: Client,
    _profile: &ProviderProfile,
    effective_base_url: &str,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    request.validate_model()?;
    let body = build_anthropic_request(&request, true);

    let url = format!("{}/v1/messages", effective_base_url.trim_end_matches('/'));
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
        vec![]
    }
}

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

#[derive(Debug)]
enum AnthropicBlockState {
    Text,
    ToolUse {
        id: String,
        name: String,
        arguments_json: String,
    },
}

#[derive(Debug, Default)]
struct AnthropicStreamAssembler {
    blocks: std::collections::BTreeMap<u32, AnthropicBlockState>,
    last_usage: Option<crate::TokenUsage>,
}

impl AnthropicStreamAssembler {
    fn process_event(&mut self, event: AnthropicStreamEvent) -> Vec<ProviderResult<ProviderEvent>> {
        let mut events = Vec::new();

        match event.event_type.as_str() {
            "message_start" => {
                if let Some(msg) = event.message {
                    if let Some(usage) = msg.usage {
                        let usage = map_anthropic_usage(&usage);
                        self.last_usage = Some(usage.clone());
                        events.push(Ok(ProviderEvent::Usage { usage }));
                    }
                }
            }
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
                        AnthropicBlockState::Text => {}
                    }
                }
            }
            "message_delta" => {
                if let Some(usage) = event.usage {
                    let usage = map_anthropic_usage(&usage);
                    self.last_usage = Some(usage.clone());
                    events.push(Ok(ProviderEvent::Usage { usage }));
                }
            }
            "message_stop" => {
                if let Some(usage) = self.last_usage.take() {
                    events.push(Ok(ProviderEvent::Usage { usage }));
                }
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

pub const SYSTEM_PROMPT_FRAGMENT: &str = include_str!("../system_prompt_fragments/anthropic.md");
