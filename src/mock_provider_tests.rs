use crate::{
    anthropic, completions,
    profile::{ApiFamily, AuthStrategy, EndpointPurpose, ProviderProfile, RuntimeConfig},
    InferenceRequest, Message, ProviderEvent, ProviderRegistry, Transcript,
};
use serde_json::json;

fn completions_profile(slug: &str, base_url: &str) -> ProviderProfile {
    ProviderProfile::new(slug, ApiFamily::OpenAiChatCompletions, base_url)
}

fn anthropic_profile(slug: &str, base_url: &str) -> ProviderProfile {
    ProviderProfile::new(slug, ApiFamily::AnthropicMessages, base_url).with_auth(
        AuthStrategy::ApiKeyHeader {
            header_name: "x-api-key".into(),
        },
    )
}

fn chat_completion_response(text: &str) -> String {
    serde_json::to_string(&json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }]
    }))
    .unwrap()
}

fn chat_completion_tool_response(call_id: &str, name: &str, args: &str) -> String {
    serde_json::to_string(&json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }))
    .unwrap()
}

fn anthropic_response(text: &str) -> String {
    serde_json::to_string(&json!({
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn"
    }))
    .unwrap()
}

fn anthropic_tool_response(call_id: &str, name: &str, input: &str) -> String {
    let input_val: serde_json::Value = serde_json::from_str(input).unwrap_or(json!(null));
    serde_json::to_string(&json!({
        "content": [{
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": input_val
        }],
        "stop_reason": "tool_use"
    }))
    .unwrap()
}

#[tokio::test]
async fn test_minimax_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("Hello from MiniMax"))
        .create_async()
        .await;

    let profile = anthropic_profile("minimax", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = anthropic::infer(&profile, &runtime, request).await.unwrap();
    mock.assert_async().await;

    assert_eq!(events.len(), 2);
    assert!(
        matches!(&events[0], ProviderEvent::Output { content } if content == "Hello from MiniMax")
    );
    assert!(matches!(&events[1], ProviderEvent::Complete));
}

#[tokio::test]
async fn test_minimax_code_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("code response"))
        .create_async()
        .await;

    let profile =
        anthropic_profile("minimax-code", &server.url()).with_purpose(EndpointPurpose::Coding);
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-code-model",
        Transcript::with_messages(vec![Message::user("write a function")]),
    );

    let events = anthropic::infer(&profile, &runtime, request).await.unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "code response")));
}

#[tokio::test]
async fn test_minimax_tool_call() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_tool_response(
            "call_1",
            "search",
            r#"{"query": "rust"}"#,
        ))
        .create_async()
        .await;

    let profile = anthropic_profile("minimax", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("search for rust")]),
    );

    let events = anthropic::infer(&profile, &runtime, request).await.unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::ToolCall { call } if call.tool_name == "search")));
}

#[tokio::test]
async fn test_zai_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("Hello from ZAI"))
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = completions::infer(&profile, &runtime, request)
        .await
        .unwrap();
    mock.assert_async().await;

    assert_eq!(events.len(), 2);
    assert!(matches!(&events[0], ProviderEvent::Output { content } if content == "Hello from ZAI"));
    assert!(matches!(&events[1], ProviderEvent::Complete));
}

#[tokio::test]
async fn test_zai_code_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("fn main() {}"))
        .create_async()
        .await;

    let profile =
        completions_profile("zai-code", &server.url()).with_purpose(EndpointPurpose::Coding);
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-code-model",
        Transcript::with_messages(vec![Message::user("write main")]),
    );

    let events = completions::infer(&profile, &runtime, request)
        .await
        .unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "fn main() {}")));
}

#[tokio::test]
async fn test_zai_tool_call() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_tool_response(
            "call_42",
            "execute",
            r#"{"command": "ls"}"#,
        ))
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("run ls")]),
    );

    let events = completions::infer(&profile, &runtime, request)
        .await
        .unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::ToolCall { call } if call.tool_name == "execute")));
}

#[tokio::test]
async fn test_kimi_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("Hello from Kimi"))
        .create_async()
        .await;

    let profile = completions_profile("kimi", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "moonshot-v1",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = completions::infer(&profile, &runtime, request)
        .await
        .unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "Hello from Kimi")));
}

#[tokio::test]
async fn test_openrouter_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("Hello from OpenRouter"))
        .create_async()
        .await;

    let profile = completions_profile("openrouter", &server.url())
        .with_header("HTTP-Referer", "https://github.com/anomalyco/iron-providers")
        .with_header("X-OpenRouter-Title", "IronAgent");
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "openai/gpt-4o",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = completions::infer(&profile, &runtime, request)
        .await
        .unwrap();
    mock.assert_async().await;

    assert!(events.iter().any(
        |e| matches!(e, ProviderEvent::Output { content } if content == "Hello from OpenRouter")
    ));
}

#[tokio::test]
async fn test_requesty_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("Hello from Requesty"))
        .create_async()
        .await;

    let profile = completions_profile("requesty", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "anthropic/claude-3",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = completions::infer(&profile, &runtime, request)
        .await
        .unwrap();
    mock.assert_async().await;

    assert!(events.iter().any(
        |e| matches!(e, ProviderEvent::Output { content } if content == "Hello from Requesty")
    ));
}

#[tokio::test]
async fn test_anthropic_slug_infer() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("Hello from Anthropic"))
        .create_async()
        .await;

    let profile = anthropic_profile("anthropic", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "claude-3-5-sonnet",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = anthropic::infer(&profile, &runtime, request).await.unwrap();
    mock.assert_async().await;

    assert!(events.iter().any(
        |e| matches!(e, ProviderEvent::Output { content } if content == "Hello from Anthropic")
    ));
}

#[tokio::test]
async fn test_registry_generic_provider_zai() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("via registry"))
        .create_async()
        .await;

    let mut registry = ProviderRegistry::new();
    registry.register(completions_profile("zai", &server.url()));

    let provider = registry.get("zai", RuntimeConfig::new("test-key")).unwrap();
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = provider.infer(request).await.unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "via registry")));
}

#[tokio::test]
async fn test_registry_generic_provider_minimax() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("via registry"))
        .create_async()
        .await;

    let mut registry = ProviderRegistry::new();
    registry.register(anthropic_profile("minimax", &server.url()));

    let provider = registry
        .get("minimax", RuntimeConfig::new("test-key"))
        .unwrap();
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let events = provider.infer(request).await.unwrap();
    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "via registry")));
}

#[tokio::test]
async fn test_completions_error_handling() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error":{"message":"Invalid API key","code":"invalid_api_key"}}"#)
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("bad-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let result = completions::infer(&profile, &runtime, request).await;
    mock.assert_async().await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_authentication());
}

#[tokio::test]
async fn test_anthropic_error_handling() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(429)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error":{"message":"rate limited"}}"#)
        .create_async()
        .await;

    let profile = anthropic_profile("minimax", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let result = anthropic::infer(&profile, &runtime, request).await;
    mock.assert_async().await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_rate_limit());
}

#[tokio::test]
async fn test_completions_stream() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\
                    data: [DONE]\n";

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let stream = completions::infer_stream(&profile, &runtime, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let outputs: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Output { content }) => Some(content.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(outputs.join(""), "Hello");
    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}

#[tokio::test]
async fn test_anthropic_stream() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
                    event: message_stop\n\
                    data: {\"type\":\"message_stop\"}\n\n";

    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let profile = anthropic_profile("minimax", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let stream = anthropic::infer_stream(&profile, &runtime, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let outputs: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Output { content }) => Some(content.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(outputs.join(""), "Hi");
    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}
