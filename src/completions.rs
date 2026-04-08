use crate::{
    error::ProviderResult,
    model::{ProviderEvent, ToolCall},
    profile::{AuthStrategy, ProviderProfile, RuntimeConfig},
    InferenceRequest, ProviderError,
};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

fn build_client(profile: &ProviderProfile, runtime: &RuntimeConfig) -> Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
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

fn build_chat_messages(request: &InferenceRequest) -> Vec<Value> {
    let mut messages = Vec::new();

    if let Some(ref instructions) = request.instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions
        }));
    }

    let mut pending_tool_calls: Vec<Value> = Vec::new();

    for msg in &request.transcript.messages {
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
                let args_str =
                    serde_json::to_string(arguments).unwrap_or_else(|_| arguments.to_string());
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
                let output_str =
                    serde_json::to_string(result).unwrap_or_else(|_| result.to_string());
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
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    id: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
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

pub async fn infer(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    let client = build_client(profile, runtime);
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
                events.push(ProviderEvent::ToolCall {
                    call: ToolCall::new(tc.id, tc.function.name, arguments),
                });
            }
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
        let mut events: Vec<ProviderResult<ProviderEvent>> = Vec::new();

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
                            events.push(Ok(ProviderEvent::Complete));
                            continue;
                        }

                        if let Ok(chunk) = serde_json::from_str::<ChatCompletionStreamChunk>(data) {
                            let (mut new_events, new_accum) =
                                process_stream_chunk(chunk, tool_call_accumulator.take());
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

fn process_stream_chunk(
    chunk: ChatCompletionStreamChunk,
    accumulator: Option<ToolCallAccumulator>,
) -> (
    Vec<ProviderResult<ProviderEvent>>,
    Option<ToolCallAccumulator>,
) {
    let mut events = Vec::new();
    let mut acc = accumulator;

    for choice in chunk.choices {
        if let Some(ref content) = choice.delta.content {
            if !content.is_empty() {
                events.push(Ok(ProviderEvent::Output {
                    content: content.clone(),
                }));
            }
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                if let Some(ref id) = tc.id {
                    if let Some(existing) = acc.take() {
                        let arguments = serde_json::from_str(&existing.arguments)
                            .unwrap_or(Value::String(existing.arguments.clone()));
                        events.push(Ok(ProviderEvent::ToolCall {
                            call: ToolCall::new(existing.id, existing.name, arguments),
                        }));
                    }
                    acc = Some(ToolCallAccumulator {
                        id: id.clone(),
                        name: String::new(),
                        arguments: String::new(),
                    });
                }

                if let Some(ref function) = tc.function {
                    if let Some(ref name) = function.name {
                        if let Some(ref mut a) = acc {
                            a.name = name.clone();
                        }
                    }
                    if let Some(ref args) = function.arguments {
                        if let Some(ref mut a) = acc {
                            a.arguments.push_str(args);
                        }
                    }
                }
            }
        }
    }

    if let Some(existing) = acc.take() {
        let arguments = serde_json::from_str(&existing.arguments)
            .unwrap_or(Value::String(existing.arguments.clone()));
        events.push(Ok(ProviderEvent::ToolCall {
            call: ToolCall::new(existing.id, existing.name, arguments),
        }));
    }

    (events, None)
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
    use crate::{Message, ToolPolicy, Transcript};
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
}
