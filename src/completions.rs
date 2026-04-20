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
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tracing::{debug, trace, warn};

fn build_client(profile: &ProviderProfile, runtime: &RuntimeConfig) -> ProviderResult<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
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

/// Accumulates state for a single tool call being assembled from stream chunks
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Accumulates state for all tool calls in a single choice
#[derive(Debug, Default)]
struct ChoiceAccumulator {
    tool_calls: BTreeMap<u32, ToolCallAccumulator>,
}

/// Stream assembler that maintains indexed state by choice and tool-call index
#[derive(Debug, Default)]
struct StreamAssembler {
    choices: BTreeMap<u32, ChoiceAccumulator>,
}

impl StreamAssembler {
    /// Apply a tool-call delta to the accumulator
    fn apply_tool_call_delta(
        &mut self,
        choice_index: u32,
        tool_index: u32,
        delta: &StreamToolCall,
    ) {
        let choice = self.choices.entry(choice_index).or_default();
        let tool = choice.tool_calls.entry(tool_index).or_default();

        // Update id if present
        if let Some(ref id) = delta.id {
            if !id.is_empty() {
                tool.id = Some(id.clone());
            }
        }

        // Update function name if present
        if let Some(ref func) = delta.function {
            if let Some(ref name) = func.name {
                if !name.is_empty() {
                    tool.name = Some(name.clone());
                }
            }
            // Accumulate arguments
            if let Some(ref args) = func.arguments {
                tool.arguments.push_str(args);
            }
        }
    }

    /// Finalize all pending tool calls for a choice and return them in index order
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

    /// Finalize all remaining pending state (safety flush for [DONE])
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

    /// Check if there are any pending tool calls
    fn has_pending(&self) -> bool {
        !self.choices.is_empty()
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
                            message: format!(
                                "Invalid choice request payload from provider tool call: {}",
                                error
                            ),
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

pub async fn infer(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    validate_model(&request)?;
    let client = build_client(profile, runtime)?;
    let messages = build_chat_messages(&request);
    let tools = build_chat_tools(&request);
    let tool_choice = map_tool_choice(&request);

    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(request.model));
    body.insert("messages".into(), json!(messages));
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

    let url = format!(
        "{}/chat/completions",
        profile.base_url.trim_end_matches('/')
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

pub async fn infer_stream(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    validate_model(&request)?;
    let client = build_client(profile, runtime)?;
    let messages = build_chat_messages(&request);
    let tools = build_chat_tools(&request);
    let tool_choice = map_tool_choice(&request);

    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(request.model));
    body.insert("messages".into(), json!(messages));
    body.insert("stream".into(), json!(true));
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

    let url = format!(
        "{}/chat/completions",
        profile.base_url.trim_end_matches('/')
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

    let stream = TerminatingStream::new(process_sse_stream(response)).boxed();
    Ok(stream)
}

fn process_sse_stream(
    response: reqwest::Response,
) -> impl futures::Stream<Item = ProviderResult<ProviderEvent>> {
    let byte_stream = response.bytes_stream();

    let mut sse_parser = SseParser::new();
    let mut assembler = StreamAssembler::default();

    byte_stream.flat_map(move |chunk_result| {
        let mut events: Vec<ProviderResult<ProviderEvent>> = Vec::new();

        match chunk_result {
            Ok(bytes) => {
                let sse_events = sse_parser.feed(&bytes);

                for sse_event in sse_events {
                    if sse_event.data == "[DONE]" {
                        // Safety flush: finalize any pending tool calls before Complete
                        if assembler.has_pending() {
                            debug!("[DONE] received with pending tool calls, performing safety flush");
                            let tool_calls = assembler.finalize_all();
                            for tool_call in tool_calls {
                                if tool_call.tool_name == CHOICE_REQUEST_TOOL_NAME {
                                    match ChoiceRequest::from_value(tool_call.arguments.clone()) {
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
                                    events.push(Ok(ProviderEvent::ToolCall { call: tool_call }));
                                }
                            }
                        }
                        events.push(Ok(ProviderEvent::Complete));
                        continue;
                    }

                    match serde_json::from_str::<ChatCompletionStreamChunk>(&sse_event.data) {
                        Ok(chunk) => {
                            let new_events = process_stream_chunk(chunk, &mut assembler);
                            events.extend(new_events);
                        }
                        Err(error) => {
                            warn!(
                                error = %error,
                                payload = %truncate_for_log(&sse_event.data),
                                "Failed to parse Chat Completions stream chunk, skipping"
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

fn process_stream_chunk(
    chunk: ChatCompletionStreamChunk,
    assembler: &mut StreamAssembler,
) -> Vec<ProviderResult<ProviderEvent>> {
    let mut events = Vec::new();

    for choice in chunk.choices {
        let choice_index = choice.index;

        // Emit text content immediately
        if let Some(ref content) = choice.delta.content {
            if !content.is_empty() {
                events.push(Ok(ProviderEvent::Output {
                    content: content.clone(),
                }));
            }
        }

        // Accumulate tool-call deltas
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                assembler.apply_tool_call_delta(choice_index, tc.index, &tc);
            }
        }

        // Check for semantic completion boundary
        if let Some(ref finish_reason) = choice.finish_reason {
            debug!(
                choice_index = choice_index,
                finish_reason = %finish_reason,
                "Choice reached completion boundary"
            );

            // Finalize all pending tool calls for this choice
            let finalized = assembler.finalize_choice(choice_index);
            for tool_call in finalized {
                if tool_call.tool_name == CHOICE_REQUEST_TOOL_NAME {
                    match ChoiceRequest::from_value(tool_call.arguments.clone()) {
                        Ok(request) => events.push(Ok(ProviderEvent::ChoiceRequest { request })),
                        Err(error) => events.push(Ok(ProviderEvent::Error {
                            message: format!(
                                "Invalid choice request payload from provider tool call: {}",
                                error
                            ),
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

fn validate_model(request: &InferenceRequest) -> ProviderResult<()> {
    if request.model.trim().is_empty() {
        return Err(ProviderError::invalid_request(
            "InferenceRequest.model must be a non-empty model identifier",
        ));
    }
    Ok(())
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
    if status.as_u16() == 401
        || code_str.contains("invalid_api_key")
        || code_str.contains("authentication")
        || message.to_lowercase().contains("unauthorized")
    {
        ProviderError::auth(message)
    } else if status.as_u16() == 429
        || code_str.contains("rate_limit")
        || code_str.contains("rateLimit")
    {
        ProviderError::rate_limit(message, None)
    } else if status.as_u16() == 400
        || code_str.contains("invalid_request")
        || code_str.contains("invalid_request_error")
    {
        ProviderError::invalid_request(message)
    } else if status.as_u16() == 404 || code_str.contains("model") {
        ProviderError::model(message)
    } else {
        ProviderError::general(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, RuntimeRecord, ToolPolicy, Transcript};
    use serde_json::json;

    #[test]
    fn test_build_chat_messages_basic() {
        let transcript =
            Transcript::with_messages(vec![Message::user("hello"), Message::assistant("hi there")]);
        let request = InferenceRequest::new("gpt-4o", transcript);
        let messages = build_chat_messages(&request);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
    }

    #[test]
    fn test_build_chat_messages_with_system_prompt() {
        let request =
            InferenceRequest::new("gpt-4o", Transcript::new()).with_instructions("You are helpful");
        let messages = build_chat_messages(&request);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "system");
    }

    #[test]
    fn test_build_chat_messages_with_tool_calls() {
        let transcript = Transcript::with_messages(vec![
            Message::user("check weather"),
            Message::AssistantToolCall {
                call_id: "call_1".into(),
                tool_name: "weather".into(),
                arguments: json!({"city": "SF"}),
            },
            Message::tool("call_1", "weather", json!({"temp": 72})),
        ]);
        let request = InferenceRequest::new("gpt-4o", transcript);
        let messages = build_chat_messages(&request);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert!(messages[1]["tool_calls"].is_array());
    }

    #[test]
    fn test_build_chat_tools() {
        let request = InferenceRequest::new("gpt-4o", Transcript::new()).with_tools(vec![
            crate::ToolDefinition::new(
                "lookup",
                "lookup a value",
                json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            ),
        ]);

        let tools = build_chat_tools(&request).expect("tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "lookup");
    }

    #[test]
    fn test_map_tool_choice() {
        let request =
            InferenceRequest::new("gpt-4o", Transcript::new()).with_tool_policy(ToolPolicy::None);
        assert_eq!(map_tool_choice(&request), Some(json!("none")));

        let request =
            InferenceRequest::new("gpt-4o", Transcript::new()).with_tool_policy(ToolPolicy::Auto);
        assert_eq!(map_tool_choice(&request), Some(json!("auto")));

        let request = InferenceRequest::new("gpt-4o", Transcript::new())
            .with_tool_policy(ToolPolicy::Required);
        assert_eq!(map_tool_choice(&request), Some(json!("required")));

        let request = InferenceRequest::new("gpt-4o", Transcript::new())
            .with_tool_policy(ToolPolicy::Specific("lookup".into()));
        let choice = map_tool_choice(&request).unwrap();
        assert_eq!(choice["function"]["name"], "lookup");
    }

    #[test]
    fn test_handle_error_status_codes() {
        let err = handle_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"bad key","code":"invalid_api_key"}}"#,
        );
        assert!(err.is_authentication());

        let err = handle_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"slow down","code":"rate_limit"}}"#,
        );
        assert!(err.is_rate_limit());
    }

    #[test]
    fn test_build_client_fails_fast_on_invalid_default_header_name() {
        let profile = ProviderProfile::new(
            "broken",
            crate::ApiFamily::OpenAiChatCompletions,
            "https://example.com/v1",
        )
        .with_header("bad header", "value");

        let error = build_client(&profile, &RuntimeConfig::new("test-key")).unwrap_err();
        assert!(matches!(error, ProviderError::InvalidRequest { .. }));
        assert!(error.to_string().contains("Invalid default header name"));
    }

    #[tokio::test]
    async fn test_infer_rejects_empty_model() {
        let profile = ProviderProfile::new(
            "test",
            crate::ApiFamily::OpenAiChatCompletions,
            "https://example.com/v1",
        );
        let runtime = RuntimeConfig::new("test-key");
        let request = InferenceRequest::new("", Transcript::new());

        let error = infer(&profile, &runtime, request).await.unwrap_err();
        assert!(matches!(error, ProviderError::InvalidRequest { .. }));
        assert!(error.to_string().contains("model must be a non-empty"));
    }

    #[test]
    fn test_build_chat_messages_does_not_project_runtime_records() {
        let mut request = InferenceRequest::new(
            "gpt-4o",
            Transcript::with_messages(vec![Message::user("hello")]),
        );
        request
            .context
            .add_record(RuntimeRecord::new("session_state", json!({"secret": true})));

        let messages = build_chat_messages(&request);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }
}
