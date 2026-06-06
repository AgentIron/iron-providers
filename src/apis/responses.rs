//! OpenAI Responses API adapter
//!
//! Handles request projection, response parsing, and streaming normalization
//! for Responses-family providers including public OpenAI and Codex.
//!
//! Uses `reqwest` and crate-owned types — no `async-openai` dependency.

use crate::{
    error::ProviderResult,
    model::{ChoiceRequest, ProviderEvent, ToolCall, CHOICE_REQUEST_TOOL_NAME},
    profile::ProviderProfile,
    provider_overrides::{ProviderOverrides, ResponsesOverrides},
    stream_util::TerminatingStream,
    InferenceRequest, ProviderError,
};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

// ────────────────────────────────────────────
// Request types
// ────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    // Codex fixed fields
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum InputItem {
    #[serde(rename = "message")]
    Message { role: String, content: String },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(rename = "call_id")]
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        #[serde(rename = "call_id")]
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize)]
struct ToolDef {
    #[serde(rename = "type")]
    tool_type: String,
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
struct ReasoningConfig {
    effort: String,
}

// ────────────────────────────────────────────
// Response types (tolerant partial parsing)
// ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    output: Vec<OutputItem>,
    #[serde(default)]
    #[allow(dead_code)]
    error: Option<ResponsesError>,
}

#[derive(Debug, Deserialize)]
struct OutputItem {
    #[serde(rename = "type")]
    item_type: String,
    #[serde(default)]
    content: Vec<ContentPart>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    part_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesError {
    #[serde(default)]
    message: Option<String>,
}

// ────────────────────────────────────────────
// Stream types
// ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ResponsesStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    item: Option<OutputItem>,
    #[serde(default)]
    error: Option<ResponsesError>,
    #[serde(default)]
    response: Option<ResponsesStreamResponse>,
}

#[derive(Debug, Deserialize)]
struct ResponsesStreamResponse {
    #[serde(default)]
    error: Option<ResponsesError>,
}

// ────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────

fn build_input_items(request: &InferenceRequest) -> Vec<InputItem> {
    let mut items = Vec::new();

    for msg in &request.context.transcript.messages {
        match msg {
            crate::Message::User { content } => {
                items.push(InputItem::Message {
                    role: "user".into(),
                    content: content.clone(),
                });
            }
            crate::Message::Assistant { content } => {
                items.push(InputItem::Message {
                    role: "assistant".into(),
                    content: content.clone(),
                });
            }
            crate::Message::AssistantToolCall {
                call_id,
                tool_name,
                arguments,
            } => {
                let args_str = serde_json::to_string(arguments)
                    .expect("serde_json::to_string on Value is infallible");
                items.push(InputItem::FunctionCall {
                    call_id: call_id.clone(),
                    name: tool_name.clone(),
                    arguments: args_str,
                });
            }
            crate::Message::Tool {
                call_id,
                tool_name: _,
                result,
            } => {
                let output_str = serde_json::to_string(result)
                    .expect("serde_json::to_string on Value is infallible");
                items.push(InputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: output_str,
                });
            }
        }
    }

    items
}

fn build_tools(request: &InferenceRequest) -> Option<Vec<ToolDef>> {
    if request.tools.is_empty() {
        return None;
    }
    Some(
        request
            .tools
            .iter()
            .map(|tool| ToolDef {
                tool_type: "function".into(),
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            })
            .collect(),
    )
}

fn map_tool_choice(request: &InferenceRequest) -> Option<Value> {
    match request.tool_policy {
        crate::ToolPolicy::None => Some(json!("none")),
        crate::ToolPolicy::Auto => Some(json!("auto")),
        crate::ToolPolicy::Required => Some(json!("required")),
        crate::ToolPolicy::Specific(ref name) => Some(json!({"type": "function", "name": name})),
    }
}

fn build_request(
    request: &InferenceRequest,
    overrides: &ProviderOverrides,
    stream: bool,
) -> ResponsesRequest {
    let input = build_input_items(request);
    let tools = build_tools(request);
    let tool_choice = map_tool_choice(request);

    let mut req = ResponsesRequest {
        model: request.model.clone(),
        input,
        tools,
        tool_choice,
        instructions: request.instructions.clone(),
        temperature: request.generation.temperature,
        max_output_tokens: request.generation.max_tokens,
        top_p: request.generation.top_p,
        stream: if stream { Some(true) } else { None },
        store: None,
        reasoning: None,
        parallel_tool_calls: None,
    };

    // Apply Codex fixed body fields
    if let ProviderOverrides::Responses(ResponsesOverrides {
        fixed_body: Some(ref fb),
        ..
    }) = overrides
    {
        req.store = Some(fb.store);
        req.reasoning = Some(ReasoningConfig {
            effort: fb.reasoning_effort.clone(),
        });
        req.parallel_tool_calls = Some(fb.parallel_tool_calls);
    }

    req
}

fn response_to_events(response: ResponsesResponse) -> Vec<ProviderEvent> {
    let mut events = Vec::new();

    for item in response.output {
        match item.item_type.as_str() {
            "message" => {
                for part in item.content {
                    if part.part_type == "output_text" {
                        if let Some(text) = part.text {
                            events.push(ProviderEvent::Output { content: text });
                        }
                    }
                }
            }
            "function_call" => {
                let id = item.call_id.unwrap_or_default();
                let name = item.name.unwrap_or_default();
                let arguments = item
                    .arguments
                    .and_then(|a| serde_json::from_str(&a).ok())
                    .unwrap_or(Value::Null);

                if name == CHOICE_REQUEST_TOOL_NAME {
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
                        call: ToolCall::new(id, name, arguments),
                    });
                }
            }
            _ => {}
        }
    }

    events.push(ProviderEvent::Complete);
    events
}

fn endpoint_path(_profile: &ProviderProfile, overrides: &ProviderOverrides) -> String {
    if let ProviderOverrides::Responses(ResponsesOverrides {
        endpoint_path: Some(ref path),
        ..
    }) = overrides
    {
        path.clone()
    } else {
        "/responses".into()
    }
}

// ────────────────────────────────────────────
// Public adapter functions
// ────────────────────────────────────────────

pub(crate) async fn infer(
    client: Client,
    profile: &ProviderProfile,
    overrides: &ProviderOverrides,
    effective_base_url: &str,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    request.validate_model()?;
    let body = build_request(&request, overrides, false);
    let path = endpoint_path(profile, overrides);

    let url = format!("{}{}", effective_base_url.trim_end_matches('/'), path);
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

    let result: ResponsesResponse = response.json().await.map_err(|e| {
        ProviderError::malformed(format!("Failed to parse Responses response: {}", e))
    })?;

    Ok(response_to_events(result))
}

pub(crate) async fn infer_stream(
    client: Client,
    profile: &ProviderProfile,
    overrides: &ProviderOverrides,
    effective_base_url: &str,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    request.validate_model()?;
    let body = build_request(&request, overrides, true);
    let path = endpoint_path(profile, overrides);

    let url = format!("{}{}", effective_base_url.trim_end_matches('/'), path);
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
        ResponsesSseAdapter,
    >(response))
    .boxed();
    Ok(stream)
}

// ────────────────────────────────────────────
// SSE stream adapter
// ────────────────────────────────────────────

#[derive(Default)]
struct ResponsesSseAdapter {
    pending: HashMap<String, PendingFunctionCall>,
    next_pending_key: usize,
}

#[derive(Default)]
struct PendingFunctionCall {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

fn tool_call_event(id: String, name: String, args: String) -> ProviderEvent {
    let arguments = if args.trim().is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(&args).unwrap_or(Value::String(args))
    };

    if name == CHOICE_REQUEST_TOOL_NAME {
        match ChoiceRequest::from_value(arguments) {
            Ok(request) => ProviderEvent::ChoiceRequest { request },
            Err(error) => ProviderEvent::Error {
                source: ProviderError::malformed(format!(
                    "Invalid choice request payload from provider tool call: {}",
                    error
                )),
            },
        }
    } else {
        ProviderEvent::ToolCall {
            call: ToolCall::new(id, name, arguments),
        }
    }
}

impl ResponsesSseAdapter {
    fn pending_key(&mut self, call_id: Option<String>, item_id: Option<String>) -> String {
        if let Some(id) = call_id.or(item_id) {
            return id;
        }

        if self.pending.len() == 1 {
            return self
                .pending
                .keys()
                .next()
                .expect("pending length checked")
                .clone();
        }

        let key = format!("pending_{}", self.next_pending_key);
        self.next_pending_key += 1;
        key
    }

    fn take_pending(
        &mut self,
        call_id: Option<String>,
        item_id: Option<String>,
    ) -> PendingFunctionCall {
        let key = self.pending_key(call_id, item_id);
        if let Some(pending) = self.pending.remove(&key) {
            return pending;
        }

        if self.pending.len() == 1 {
            let fallback_key = self
                .pending
                .keys()
                .next()
                .expect("pending length checked")
                .clone();
            return self.pending.remove(&fallback_key).unwrap_or_default();
        }

        PendingFunctionCall::default()
    }
}

impl crate::stream_util::SseStreamAdapter for ResponsesSseAdapter {
    type Event = ResponsesStreamEvent;
    const LABEL: &'static str = "Responses";

    fn process_event(&mut self, event: ResponsesStreamEvent) -> Vec<ProviderResult<ProviderEvent>> {
        let mut events = Vec::new();

        match event.event_type.as_str() {
            "response.output_text.delta" => {
                if let Some(delta) = event.delta {
                    events.push(Ok(ProviderEvent::Output { content: delta }));
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(delta) = event.delta {
                    let key = self.pending_key(event.call_id, event.item_id);
                    self.pending
                        .entry(key)
                        .or_default()
                        .arguments
                        .push_str(&delta);
                }
            }
            "response.function_call_arguments.done" => {
                let key = self.pending_key(event.call_id.clone(), event.item_id.clone());
                let pending = self.pending.entry(key).or_default();
                if let Some(call_id) = event.call_id {
                    pending.call_id = Some(call_id);
                }
                if let Some(name) = event.name {
                    pending.name = Some(name);
                }
                if let Some(arguments) = event.arguments {
                    pending.arguments = arguments;
                }
            }
            "response.output_item.added" => {
                if let Some(item) = event.item {
                    if item.item_type == "function_call" {
                        let key = self.pending_key(
                            event.call_id.or_else(|| item.call_id.clone()),
                            event.item_id,
                        );
                        let pending = self.pending.entry(key).or_default();
                        if item.call_id.is_some() {
                            pending.call_id = item.call_id;
                        }
                        if item.name.is_some() {
                            pending.name = item.name;
                        }
                        if let Some(arguments) = item.arguments {
                            pending.arguments = arguments;
                        }
                    }
                }
            }
            "response.output_item.done" => {
                if let Some(item) = event.item {
                    if item.item_type == "function_call" {
                        let pending = self.take_pending(
                            event.call_id.or_else(|| item.call_id.clone()),
                            event.item_id,
                        );
                        let id = item.call_id.or(pending.call_id).unwrap_or_default();
                        let name = item.name.or(pending.name).unwrap_or_default();
                        let args = item.arguments.unwrap_or(pending.arguments);
                        events.push(Ok(tool_call_event(id, name, args)));
                    }
                }
            }
            "response.completed" => {
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
            "response.failed" => {
                let msg = event
                    .response
                    .and_then(|response| response.error)
                    .or(event.error)
                    .and_then(|error| error.message)
                    .unwrap_or_else(|| "Responses stream failed".to_string());
                events.push(Ok(ProviderEvent::Error {
                    source: ProviderError::general(msg),
                }));
            }
            _ => {}
        }

        events
    }

    fn handle_done(&mut self) -> Vec<ProviderResult<ProviderEvent>> {
        vec![]
    }
}

// ────────────────────────────────────────────
// Error handling
// ────────────────────────────────────────────

fn handle_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
    let (message, code) = if let Ok(err) = serde_json::from_str::<ResponsesErrorWrapper>(body) {
        let msg = err.error.message.unwrap_or_else(|| body.to_string());
        let c = err.error.code;
        (msg, c)
    } else {
        (body.to_string(), None)
    };

    let code_str = code.as_deref().unwrap_or("");
    let status_code = status.as_u16();

    if status_code == 401
        || status_code == 403
        || code_str.contains("invalid_api_key")
        || code_str.contains("authentication")
        || message.to_lowercase().contains("unauthorized")
    {
        ProviderError::auth(message)
    } else if status_code == 429
        || code_str.contains("rate_limit")
        || code_str.contains("rateLimit")
        || code_str.contains("insufficient_quota")
    {
        ProviderError::rate_limit(message, None)
    } else if status_code == 400
        || code_str.contains("invalid_request")
        || code_str.contains("invalid_request_error")
    {
        ProviderError::invalid_request(message)
    } else if status_code == 404 || code_str.contains("model") {
        ProviderError::model(message)
    } else {
        ProviderError::general(message)
    }
}

#[derive(Debug, Deserialize)]
struct ResponsesErrorWrapper {
    error: ResponsesErrorBody,
}

#[derive(Debug, Deserialize)]
struct ResponsesErrorBody {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    code: Option<String>,
}
