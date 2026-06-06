//! OpenAI Chat Completions API adapter
//!
//! Handles request projection, response parsing, and streaming normalization
//! for Completions-family providers.

use crate::{
    error::ProviderResult,
    model::{ChoiceRequest, ProviderEvent, ToolCall, CHOICE_REQUEST_TOOL_NAME},
    profile::ProviderProfile,
    stream_util::TerminatingStream,
    InferenceRequest, ProviderError,
};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tracing::{debug, trace, warn};

fn build_chat_messages(request: &InferenceRequest) -> Vec<Value> {
    let mut messages = Vec::new();

    if let Some(ref instructions) = request.instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions
        }));
    }

    let mut pending_tool_calls: Vec<Value> = Vec::new();

    for msg in &request.context.transcript.messages {
        match msg {
            crate::Message::User { content } => {
                messages.push(json!({
                    "role": "user",
                    "content": content
                }));
            }
            crate::Message::Assistant { content } => {
                let tool_calls = if pending_tool_calls.is_empty() {
                    None
                } else {
                    Some(pending_tool_calls.clone())
                };
                let mut assistant = json!({
                    "role": "assistant",
                    "content": content
                });
                if let Some(tc) = tool_calls {
                    assistant["tool_calls"] = Value::Array(tc);
                }
                messages.push(assistant);
                pending_tool_calls.clear();
            }
            crate::Message::AssistantToolCall {
                call_id,
                tool_name,
                arguments,
            } => {
                let args_str = serde_json::to_string(arguments)
                    .expect("serde_json::to_string on Value is infallible");
                pending_tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args_str
                    }
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
                        "content": "",
                        "tool_calls": Value::Array(pending_tool_calls.clone())
                    }));
                    pending_tool_calls.clear();
                }
                let output_str = serde_json::to_string(result)
                    .expect("serde_json::to_string on Value is infallible");
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output_str
                }));
            }
        }
    }

    if !pending_tool_calls.is_empty() {
        messages.push(json!({
            "role": "assistant",
            "content": "",
            "tool_calls": pending_tool_calls
        }));
    }

    messages
}

fn build_chat_tools(request: &InferenceRequest) -> Option<Vec<Value>> {
    if request.tools.is_empty() {
        return None;
    }
    Some(
        request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
            })
            .collect(),
    )
}

fn map_tool_choice(request: &InferenceRequest) -> Option<Value> {
    match request.tool_policy {
        crate::ToolPolicy::None => Some(json!("none")),
        crate::ToolPolicy::Auto => Some(json!("auto")),
        crate::ToolPolicy::Required => Some(json!("required")),
        crate::ToolPolicy::Specific(ref name) => {
            Some(json!({"type": "function", "function": {"name": name}}))
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChatToolCall {
    id: String,
    function: ChatFunction,
}

#[derive(Debug, Deserialize)]
struct ChatFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    index: u32,
    id: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Default)]
struct ChoiceAccumulator {
    tool_calls: BTreeMap<u32, ToolCallAccumulator>,
}

#[derive(Debug, Default)]
struct StreamAssembler {
    choices: BTreeMap<u32, ChoiceAccumulator>,
}

impl StreamAssembler {
    fn apply_tool_call_delta(
        &mut self,
        choice_index: u32,
        tool_index: u32,
        delta: &StreamToolCall,
    ) {
        let choice = self.choices.entry(choice_index).or_default();
        let tool = choice.tool_calls.entry(tool_index).or_default();

        if let Some(ref id) = delta.id {
            if !id.is_empty() {
                tool.id = Some(id.clone());
            }
        }

        if let Some(ref func) = delta.function {
            if let Some(ref name) = func.name {
                if !name.is_empty() {
                    tool.name = Some(name.clone());
                }
            }
            if let Some(ref args) = func.arguments {
                tool.arguments.push_str(args);
            }
        }
    }

    fn finalize_choice(&mut self, choice_index: u32) -> Vec<ToolCall> {
        let mut finalized = Vec::new();

        if let Some(choice) = self.choices.remove(&choice_index) {
            for (index, tool) in choice.tool_calls {
                let arguments = if tool.arguments.is_empty() {
                    Value::Object(serde_json::Map::new())
                } else {
                    match serde_json::from_str(&tool.arguments) {
                        Ok(json) => json,
                        Err(e) => {
                            warn!(
                                choice_index = choice_index,
                                tool_index = index,
                                arguments = %tool.arguments,
                                error = %e,
                                "Failed to parse tool call arguments as JSON, using raw string"
                            );
                            Value::String(tool.arguments.clone())
                        }
                    }
                };

                let id = tool.id.clone().unwrap_or_else(|| {
                    warn!(
                        choice_index = choice_index,
                        tool_index = index,
                        "Tool call finalized without id, generating fallback"
                    );
                    format!("call_{}_{}", choice_index, index)
                });
                let name = tool.name.clone().unwrap_or_else(|| {
                    warn!(
                        choice_index = choice_index,
                        tool_index = index,
                        tool_id = %id,
                        "Tool call finalized without function name"
                    );
                    String::new()
                });

                trace!(
                    choice_index = choice_index,
                    tool_index = index,
                    tool_id = %id,
                    tool_name = %name,
                    "Finalizing tool call"
                );

                finalized.push(ToolCall::new(id, name, arguments));
            }
        }

        finalized
    }

    fn finalize_all(&mut self) -> Vec<ToolCall> {
        let mut finalized = Vec::new();
        let choice_indices: Vec<u32> = self.choices.keys().copied().collect();

        for choice_index in choice_indices {
            debug!(
                choice_index = choice_index,
                "Safety flush for pending tool calls at [DONE]"
            );
            let mut tools = self.finalize_choice(choice_index);
            finalized.append(&mut tools);
        }

        finalized
    }

    fn has_pending(&self) -> bool {
        !self.choices.is_empty()
    }
}

fn response_to_events(response: ChatCompletionResponse) -> Vec<ProviderEvent> {
    let mut events = Vec::new();

    for choice in response.choices {
        if let Some(ref content) = choice.message.content {
            if !content.is_empty() {
                events.push(ProviderEvent::Output {
                    content: content.clone(),
                });
            }
        }

        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let arguments = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| Value::String(tc.function.arguments.clone()));
                if tc.function.name == CHOICE_REQUEST_TOOL_NAME {
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
                        call: ToolCall::new(tc.id, tc.function.name, arguments),
                    });
                }
            }
        }
    }

    events.push(ProviderEvent::Complete);
    events
}

fn build_request_body(request: &InferenceRequest, stream: bool) -> serde_json::Map<String, Value> {
    let messages = build_chat_messages(request);
    let tools = build_chat_tools(request);
    let tool_choice = map_tool_choice(request);

    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(request.model));
    body.insert("messages".into(), json!(messages));
    if stream {
        body.insert("stream".into(), json!(true));
    }
    if let Some(tools) = tools {
        body.insert("tools".into(), json!(tools));
    }
    if let Some(tool_choice) = tool_choice {
        body.insert("tool_choice".into(), tool_choice);
    }
    if let Some(temp) = request.generation.temperature {
        body.insert("temperature".into(), json!(temp));
    }
    if let Some(max_tokens) = request.generation.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    if let Some(top_p) = request.generation.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if let Some(ref stop) = request.generation.stop {
        body.insert("stop".into(), json!(stop));
    }
    body
}

pub(crate) async fn infer(
    client: Client,
    _profile: &ProviderProfile,
    effective_base_url: &str,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    request.validate_model()?;
    let body = build_request_body(&request, false);

    let url = format!(
        "{}/chat/completions",
        effective_base_url.trim_end_matches('/')
    );
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

    let result: ChatCompletionResponse = response.json().await.map_err(|e| {
        ProviderError::malformed(format!("Failed to parse Chat Completion response: {}", e))
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
    let body = build_request_body(&request, true);

    let url = format!(
        "{}/chat/completions",
        effective_base_url.trim_end_matches('/')
    );
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
        CompletionsSseAdapter,
    >(response))
    .boxed();
    Ok(stream)
}

#[derive(Default)]
struct CompletionsSseAdapter {
    assembler: StreamAssembler,
}

impl crate::stream_util::SseStreamAdapter for CompletionsSseAdapter {
    type Event = ChatCompletionStreamChunk;
    const LABEL: &'static str = "Chat Completions";

    fn process_event(
        &mut self,
        chunk: ChatCompletionStreamChunk,
    ) -> Vec<ProviderResult<ProviderEvent>> {
        self.assembler.process_chunk(chunk)
    }

    fn handle_done(&mut self) -> Vec<ProviderResult<ProviderEvent>> {
        let mut events = Vec::new();

        if self.assembler.has_pending() {
            debug!("[DONE] received with pending tool calls, performing safety flush");
            let tool_calls = self.assembler.finalize_all();
            for tool_call in tool_calls {
                if tool_call.tool_name == CHOICE_REQUEST_TOOL_NAME {
                    match ChoiceRequest::from_value(tool_call.arguments.clone()) {
                        Ok(request) => events.push(Ok(ProviderEvent::ChoiceRequest { request })),
                        Err(error) => events.push(Ok(ProviderEvent::Error {
                            source: ProviderError::malformed(format!(
                                "Invalid choice request payload from provider tool call: {}",
                                error
                            )),
                        })),
                    }
                } else {
                    events.push(Ok(ProviderEvent::ToolCall { call: tool_call }));
                }
            }
        }
        events.push(Ok(ProviderEvent::Complete));
        events
    }
}

impl StreamAssembler {
    fn process_chunk(
        &mut self,
        chunk: ChatCompletionStreamChunk,
    ) -> Vec<ProviderResult<ProviderEvent>> {
        let mut events = Vec::new();

        for choice in chunk.choices {
            let choice_index = choice.index;

            if let Some(ref content) = choice.delta.content {
                if !content.is_empty() {
                    events.push(Ok(ProviderEvent::Output {
                        content: content.clone(),
                    }));
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tc in tool_calls {
                    self.apply_tool_call_delta(choice_index, tc.index, &tc);
                }
            }

            if let Some(ref finish_reason) = choice.finish_reason {
                debug!(
                    choice_index = choice_index,
                    finish_reason = %finish_reason,
                    "Choice reached completion boundary"
                );

                let finalized = self.finalize_choice(choice_index);
                for tool_call in finalized {
                    if tool_call.tool_name == CHOICE_REQUEST_TOOL_NAME {
                        match ChoiceRequest::from_value(tool_call.arguments.clone()) {
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
                        events.push(Ok(ProviderEvent::ToolCall { call: tool_call }));
                    }
                }
            }
        }

        events
    }
}

fn handle_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
    let (message, code) = if let Ok(err) = serde_json::from_str::<ChatError>(body) {
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
struct ChatError {
    error: ChatErrorBody,
}

#[derive(Debug, Deserialize)]
struct ChatErrorBody {
    message: Option<String>,
    code: Option<String>,
}

pub const SYSTEM_PROMPT_FRAGMENT: &str = include_str!("../system_prompt_fragments/openai.md");
