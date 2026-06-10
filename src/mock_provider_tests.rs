use crate::{
    apis::{completions, messages, responses},
    auth::auth_headers,
    http_client::build_http_client,
    profile::{
        ApiFamily, AuthStrategy, CredentialKind, ProviderCredential, ProviderProfile, RuntimeConfig,
    },
    provider_overrides::{override_headers, resolve_overrides},
    InferenceRequest, Message, ProviderEvent, ProviderRegistry, Transcript,
};
use mockito::Matcher;
use serde_json::json;

fn completions_profile(slug: &str, base_url: &str) -> ProviderProfile {
    ProviderProfile::new(slug, ApiFamily::Completions, base_url)
}

fn anthropic_profile(slug: &str, base_url: &str) -> ProviderProfile {
    ProviderProfile::new(slug, ApiFamily::Messages, base_url).with_auth(
        AuthStrategy::ApiKeyHeader {
            header_name: "x-api-key".into(),
        },
    )
}

fn responses_profile(slug: &str, base_url: &str) -> ProviderProfile {
    ProviderProfile::new(slug, ApiFamily::Responses, base_url)
}

/// Build a `reqwest::Client` for use in tests, applying profile auth and
/// runtime timeouts.
fn build_test_client(
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
) -> crate::ProviderResult<reqwest::Client> {
    let context = format!("test profile '{}'", profile.slug);
    let kind = runtime.credential.kind();
    let auth_strategy = profile.auth_strategy_for(kind).ok_or_else(|| {
        crate::ProviderError::auth(format!(
            "Provider '{}' does not support {:?} credentials",
            profile.slug, kind
        ))
    })?;

    let auth_h = auth_headers(&runtime.credential, auth_strategy, &context)?;
    let overrides = resolve_overrides(profile, runtime);
    let override_h = override_headers(&overrides)?;

    let mut final_headers = reqwest::header::HeaderMap::new();
    final_headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );

    for (key, value) in &auth_h {
        final_headers.insert(key.clone(), value.clone());
    }
    for (key, value) in &override_h {
        final_headers.insert(key.clone(), value.clone());
    }
    for (key, value) in &profile.default_headers {
        let hk = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
            crate::ProviderError::invalid_request(format!(
                "Invalid default header name '{}' for {}: {}",
                key, context, e
            ))
        })?;
        let hv = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
            crate::ProviderError::invalid_request(format!(
                "Invalid default header value for '{}' on {}: {}",
                key, context, e
            ))
        })?;
        final_headers.insert(hk, hv);
    }

    build_http_client(
        final_headers,
        runtime.effective_connect_timeout(),
        runtime.effective_read_timeout(),
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

fn responses_response(text: &str) -> String {
    serde_json::to_string(&json!({
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": text}]
        }]
    }))
    .unwrap()
}

#[tokio::test]
async fn test_completions_basic_response() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_response("Hello from completions"))
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = completions::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
    let events = result.unwrap();
    assert!(events.iter().any(
        |e| matches!(e, ProviderEvent::Output { content } if content == "Hello from completions")
    ));
}

#[tokio::test]
async fn test_completions_tool_call_response() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion_tool_response(
            "call_123",
            "search",
            r#"{"query": "rust async"}"#,
        ))
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("search for rust async")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = completions::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
    let events = result.unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::ToolCall { call } if call.tool_name == "search")));
}

#[tokio::test]
async fn test_completions_usage_response() {
    let mut server = mockito::Server::new_async().await;

    let body = serde_json::to_string(&json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {
                "cached_tokens": 3
            },
            "completion_tokens_details": {
                "reasoning_tokens": 2
            }
        }
    }))
    .unwrap();

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = completions::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
    let events = result.unwrap();

    let usage_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            ProviderEvent::Usage { usage } => Some(usage.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, Some(10));
    assert_eq!(usage_events[0].output_tokens, Some(5));
    assert_eq!(usage_events[0].total_tokens, Some(15));
    assert_eq!(usage_events[0].cached_input_tokens, Some(3));
    assert_eq!(usage_events[0].reasoning_output_tokens, Some(2));

    let usage_idx = events
        .iter()
        .position(|e| matches!(e, ProviderEvent::Usage { .. }))
        .unwrap();
    let complete_idx = events
        .iter()
        .position(|e| matches!(e, ProviderEvent::Complete))
        .unwrap();
    assert!(usage_idx < complete_idx);
}

#[tokio::test]
async fn test_completions_stream_usage() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                    data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n\
                    data: [DONE]\n\n";

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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let usage_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Usage { usage }) => Some(usage.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, Some(5));
    assert_eq!(usage_events[0].output_tokens, Some(2));
    assert_eq!(usage_events[0].total_tokens, Some(7));

    let usage_idx = events
        .iter()
        .position(|e| matches!(e, Ok(ProviderEvent::Usage { .. })))
        .unwrap();
    let complete_idx = events
        .iter()
        .position(|e| matches!(e, Ok(ProviderEvent::Complete)))
        .unwrap();
    assert!(usage_idx < complete_idx);
}

#[tokio::test]
async fn test_anthropic_usage_response() {
    let mut server = mockito::Server::new_async().await;

    let body = serde_json::to_string(&json!({
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 15
        }
    }))
    .unwrap();

    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create_async()
        .await;

    let profile = anthropic_profile("minimax", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = messages::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
    let events = result.unwrap();

    let usage_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            ProviderEvent::Usage { usage } => Some(usage.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, Some(20));
    assert_eq!(usage_events[0].output_tokens, Some(10));
    assert_eq!(usage_events[0].total_tokens, None);
    assert_eq!(usage_events[0].cache_creation_input_tokens, Some(5));
    assert_eq!(usage_events[0].cache_read_input_tokens, Some(15));

    let usage_idx = events
        .iter()
        .position(|e| matches!(e, ProviderEvent::Usage { .. }))
        .unwrap();
    let complete_idx = events
        .iter()
        .position(|e| matches!(e, ProviderEvent::Complete))
        .unwrap();
    assert!(usage_idx < complete_idx);
}

#[tokio::test]
async fn test_anthropic_stream_usage() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":20,\"output_tokens\":10,\"cache_creation_input_tokens\":5,\"cache_read_input_tokens\":15}}\n\n\
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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = messages::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let usage_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Usage { usage }) => Some(usage.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(usage_events.len(), 2); // message_delta + message_stop
    assert_eq!(usage_events[0].input_tokens, Some(20));
    assert_eq!(usage_events[0].output_tokens, Some(10));
    assert_eq!(usage_events[0].cache_creation_input_tokens, Some(5));
    assert_eq!(usage_events[0].cache_read_input_tokens, Some(15));

    let usage_idx = events
        .iter()
        .position(|e| matches!(e, Ok(ProviderEvent::Usage { .. })))
        .unwrap();
    let complete_idx = events
        .iter()
        .position(|e| matches!(e, Ok(ProviderEvent::Complete)))
        .unwrap();
    assert!(usage_idx < complete_idx);
}

#[tokio::test]
async fn test_responses_usage_response() {
    let mut server = mockito::Server::new_async().await;

    let body = serde_json::to_string(&json!({
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello"}]
        }],
        "usage": {
            "input_tokens": 30,
            "output_tokens": 20,
            "total_tokens": 50,
            "input_tokens_details": {
                "cached_tokens": 10
            },
            "output_tokens_details": {
                "reasoning_tokens": 5
            }
        }
    }))
    .unwrap();

    let mock = server
        .mock("POST", "/responses")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create_async()
        .await;

    let profile = responses_profile("openai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "gpt-4o",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = responses::infer(
        client,
        &profile,
        &resolve_overrides(&profile, &runtime),
        &profile.base_url,
        request,
    )
    .await;
    mock.assert_async().await;

    assert!(result.is_ok());
    let events = result.unwrap();

    let usage_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            ProviderEvent::Usage { usage } => Some(usage.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, Some(30));
    assert_eq!(usage_events[0].output_tokens, Some(20));
    assert_eq!(usage_events[0].total_tokens, Some(50));
    assert_eq!(usage_events[0].cached_input_tokens, Some(10));
    assert_eq!(usage_events[0].reasoning_output_tokens, Some(5));

    let usage_idx = events
        .iter()
        .position(|e| matches!(e, ProviderEvent::Usage { .. }))
        .unwrap();
    let complete_idx = events
        .iter()
        .position(|e| matches!(e, ProviderEvent::Complete))
        .unwrap();
    assert!(usage_idx < complete_idx);
}

#[tokio::test]
async fn test_responses_stream_usage() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "event: response.output_text.delta\n\
                    data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n\
                    event: response.completed\n\
                    data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":30,\"output_tokens\":20,\"total_tokens\":50,\"input_tokens_details\":{\"cached_tokens\":10},\"output_tokens_details\":{\"reasoning_tokens\":5}}}}\n\n";

    let mock = server
        .mock("POST", "/responses")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let profile = responses_profile("openai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "gpt-4o",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = responses::infer_stream(
        client,
        &profile,
        &resolve_overrides(&profile, &runtime),
        &profile.base_url,
        request,
    )
    .await
    .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let usage_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Usage { usage }) => Some(usage.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, Some(30));
    assert_eq!(usage_events[0].output_tokens, Some(20));
    assert_eq!(usage_events[0].total_tokens, Some(50));
    assert_eq!(usage_events[0].cached_input_tokens, Some(10));
    assert_eq!(usage_events[0].reasoning_output_tokens, Some(5));

    let usage_idx = events
        .iter()
        .position(|e| matches!(e, Ok(ProviderEvent::Usage { .. })))
        .unwrap();
    let complete_idx = events
        .iter()
        .position(|e| matches!(e, Ok(ProviderEvent::Complete)))
        .unwrap();
    assert!(usage_idx < complete_idx);
}

#[tokio::test]
async fn test_anthropic_basic_response() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("Hello from anthropic"))
        .create_async()
        .await;

    let profile = anthropic_profile("minimax", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let request = InferenceRequest::new(
        "minimax-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = messages::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
    let events = result.unwrap();
    assert!(events.iter().any(
        |e| matches!(e, ProviderEvent::Output { content } if content == "Hello from anthropic")
    ));
}

#[tokio::test]
async fn test_kimi_code_api_key_uses_x_api_key_header() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/v1/messages")
        .match_header("x-api-key", "kimi-api-key")
        .match_header("authorization", Matcher::Missing)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("Hello from Kimi Code"))
        .create_async()
        .await;

    let profile = anthropic_profile("kimi-code", &server.url())
        .with_credential_auth(CredentialKind::OAuthBearer, AuthStrategy::BearerToken);
    let runtime = RuntimeConfig::new("kimi-api-key");
    let request = InferenceRequest::new(
        "kimi-code-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = messages::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_kimi_code_oauth_uses_bearer_header() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/v1/messages")
        .match_header("authorization", "Bearer kimi-oauth-token")
        .match_header("x-api-key", Matcher::Missing)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(anthropic_response("Hello from Kimi Code OAuth"))
        .create_async()
        .await;

    let profile = anthropic_profile("kimi-code", &server.url())
        .with_credential_auth(CredentialKind::OAuthBearer, AuthStrategy::BearerToken);
    let runtime = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
        access_token: "kimi-oauth-token".into(),
        expires_at: None,
        id_token: None,
    });
    let request = InferenceRequest::new(
        "kimi-code-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = messages::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_registry_get_completions_provider() {
    let registry = ProviderRegistry::default();
    let runtime = RuntimeConfig::new("test-key");

    let result = registry.get("zai", runtime);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_registry_get_anthropic_provider() {
    let registry = ProviderRegistry::default();
    let runtime = RuntimeConfig::new("test-key");

    let result = registry.get("anthropic", runtime);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_registry_unknown_provider_error() {
    let registry = ProviderRegistry::default();
    let runtime = RuntimeConfig::new("test-key");

    let result = registry.get("unknown-provider", runtime);
    match result {
        Err(err) => assert!(err.to_string().contains("Unknown provider")),
        Ok(_) => panic!("Expected error for unknown provider"),
    }
}

#[tokio::test]
async fn test_completions_error_handling() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error":{"message":"invalid key","code":"invalid_api_key"}}"#)
        .create_async()
        .await;

    let profile = completions_profile("zai", &server.url());
    let runtime = RuntimeConfig::new("bad-key");
    let request = InferenceRequest::new(
        "zai-model",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = completions::infer(client, &profile, &profile.base_url, request).await;
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

    let client = build_test_client(&profile, &runtime).unwrap();
    let result = messages::infer(client, &profile, &profile.base_url, request).await;
    mock.assert_async().await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_rate_limit());
}

#[tokio::test]
async fn test_completions_stream() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n\
                    data: [DONE]\n\n";

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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = messages::infer_stream(client, &profile, &profile.base_url, request)
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

#[tokio::test]
async fn test_responses_request_uses_structured_function_items() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/responses")
        .match_body(Matcher::PartialJson(json!({
            "input": [
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "lookup",
                    "arguments": "{\"query\":\"rust\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "{\"answer\":\"safe\"}"
                }
            ]
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(responses_response("done"))
        .create_async()
        .await;

    let profile = responses_profile("openai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let overrides = resolve_overrides(&profile, &runtime);
    let request = InferenceRequest::new(
        "gpt-5",
        Transcript::with_messages(vec![
            Message::user("continue"),
            Message::AssistantToolCall {
                call_id: "call_123".into(),
                tool_name: "lookup".into(),
                arguments: json!({"query": "rust"}),
            },
            Message::tool("call_123", "lookup", json!({"answer": "safe"})),
        ]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let events = responses::infer(client, &profile, &overrides, &profile.base_url, request)
        .await
        .unwrap();

    mock.assert_async().await;
    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "done")));
}

#[tokio::test]
async fn test_codex_responses_overrides_add_headers_and_fixed_body() {
    let mut server = mockito::Server::new_async().await;

    let mock = server
        .mock("POST", "/responses")
        .match_header("originator", "iron-providers")
        .match_header("user-agent", Matcher::Regex(r"^iron-providers/".into()))
        .match_body(Matcher::PartialJson(json!({
            "store": false,
            "reasoning": {"effort": "medium"},
            "parallel_tool_calls": true
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(responses_response("codex"))
        .create_async()
        .await;

    let profile = responses_profile("codex", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let overrides = resolve_overrides(&profile, &runtime);
    let request = InferenceRequest::new(
        "codex-mini-latest",
        Transcript::with_messages(vec![Message::user("hi")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let events = responses::infer(client, &profile, &overrides, &profile.base_url, request)
        .await
        .unwrap();

    mock.assert_async().await;
    assert!(events
        .iter()
        .any(|e| matches!(e, ProviderEvent::Output { content } if content == "codex")));
}

#[tokio::test]
async fn test_responses_stream_text_and_tool_call() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Thinking\"}\n\n\
                    data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"search\"}}\n\n\
                    data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"call_1\",\"delta\":\"{\\\"query\\\":\"}\n\n\
                    data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"call_1\",\"delta\":\"\\\"rust\\\"}\"}\n\n\
                    data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"call_1\",\"arguments\":\"{\\\"query\\\":\\\"rust\\\"}\"}\n\n\
                    data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"search\"}}\n\n\
                    data: {\"type\":\"response.completed\"}\n\n";

    let mock = server
        .mock("POST", "/responses")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let profile = responses_profile("openai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let overrides = resolve_overrides(&profile, &runtime);
    let request = InferenceRequest::new(
        "gpt-5",
        Transcript::with_messages(vec![Message::user("search")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = responses::infer_stream(client, &profile, &overrides, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Output { content }) if content == "Thinking")));

    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].call_id, "call_1");
    assert_eq!(tool_calls[0].tool_name, "search");
    assert_eq!(tool_calls[0].arguments["query"], "rust");
    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}

#[tokio::test]
async fn test_responses_stream_failed_emits_error_without_complete() {
    let mut server = mockito::Server::new_async().await;

    let sse_body =
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"boom\"}}}\n\n\
                    data: {\"type\":\"response.completed\"}\n\n";

    let mock = server
        .mock("POST", "/responses")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let profile = responses_profile("openai", &server.url());
    let runtime = RuntimeConfig::new("test-key");
    let overrides = resolve_overrides(&profile, &runtime);
    let request = InferenceRequest::new(
        "gpt-5",
        Transcript::with_messages(vec![Message::user("fail")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = responses::infer_stream(client, &profile, &overrides, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    assert!(events.iter().any(|e| matches!(
        e,
        Ok(ProviderEvent::Error { source }) if source.to_string().contains("boom")
    )));
    assert!(!events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}

// ============================================================================
// Task 3.1: Single tool call spanning multiple chunks
// ============================================================================

#[tokio::test]
async fn test_completions_stream_tool_call_spans_multiple_chunks() {
    let mut server = mockito::Server::new_async().await;

    // Simulate a tool call with arguments arriving across multiple chunks
    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"get_weather\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\": \\\"San\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" Francisco\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";

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
        Transcript::with_messages(vec![Message::user("What's the weather in San Francisco?")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Should have exactly one ToolCall event with complete arguments
    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls.len(), 1, "Should have exactly one tool call");
    assert_eq!(tool_calls[0].call_id, "call_123");
    assert_eq!(tool_calls[0].tool_name, "get_weather");

    // Arguments should be valid JSON with the complete location
    let args = &tool_calls[0].arguments;
    assert!(args.is_object(), "Arguments should be valid JSON object");
    assert_eq!(args["location"], "San Francisco");

    // Verify Complete event exists
    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}

// ============================================================================
// Task 3.2: Multiple indexed tool calls in one choice
// ============================================================================

#[tokio::test]
async fn test_completions_stream_multiple_tool_calls_not_merged() {
    let mut server = mockito::Server::new_async().await;

    // Simulate two tool calls arriving interleaved
    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"calculate\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"query\\\": \\\"rust\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"expr\\\": \\\"1+1\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"}\"}},{\"index\":1,\"function\":{\"arguments\":\"\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";

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
        Transcript::with_messages(vec![Message::user("Search for rust and calculate 1+1")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls.len(), 2, "Should have exactly two tool calls");

    // Verify fragments were not merged
    assert_eq!(tool_calls[0].call_id, "call_1");
    assert_eq!(tool_calls[0].tool_name, "search");
    assert_eq!(tool_calls[0].arguments["query"], "rust");

    assert_eq!(tool_calls[1].call_id, "call_2");
    assert_eq!(tool_calls[1].tool_name, "calculate");
    assert_eq!(tool_calls[1].arguments["expr"], "1+1");
}

// ============================================================================
// Task 3.3: Interleaved text and tool-call deltas
// ============================================================================

#[tokio::test]
async fn test_completions_stream_interleaved_text_and_tool_calls() {
    let mut server = mockito::Server::new_async().await;

    // Simulate text and tool-call deltas arriving interleaved
    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"I'll help\"}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" you search\"}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\": \\\"test\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" for that.\"}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";

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
        Transcript::with_messages(vec![Message::user("Search for test")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Collect outputs and tool calls
    let outputs: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Output { content }) => Some(content.clone()),
            _ => None,
        })
        .collect();

    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    // Text should be emitted incrementally
    let full_output = outputs.join("");
    assert_eq!(full_output, "I'll help you search for that.");

    // Tool call should only be emitted once at completion
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].call_id, "call_1");
    assert_eq!(tool_calls[0].arguments["q"], "test");

    // Verify ordering: all outputs should come before tool call in event stream
    let output_indices: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Ok(ProviderEvent::Output { .. })).then_some(i))
        .collect();

    let tool_call_indices: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Ok(ProviderEvent::ToolCall { .. })).then_some(i))
        .collect();

    // All outputs should come before tool call (they arrive interleaved but tool call finalizes later)
    let last_output = output_indices.last().copied().unwrap_or(0);
    let first_tool_call = tool_call_indices.first().copied().unwrap_or(usize::MAX);
    assert!(
        last_output < first_tool_call,
        "All outputs should come before tool call finalization"
    );
}

// ============================================================================
// Task 3.4: Tool-call ordering before Complete
// ============================================================================

#[tokio::test]
async fn test_completions_stream_tool_calls_before_complete() {
    let mut server = mockito::Server::new_async().await;

    // Multiple tool calls with finish_reason in same chunk
    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}},{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";

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
        Transcript::with_messages(vec![Message::user("Do two things")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Verify event ordering: ToolCall[0], ToolCall[1], Complete
    let event_types: Vec<_> = events
        .iter()
        .map(|e| match e {
            Ok(ProviderEvent::ToolCall { .. }) => "ToolCall",
            Ok(ProviderEvent::Complete) => "Complete",
            _ => "Other",
        })
        .collect();

    assert_eq!(event_types, vec!["ToolCall", "ToolCall", "Complete"]);

    // Verify tool calls are in ascending index order
    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.tool_name.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls, vec!["first", "second"]);
}

// ============================================================================
// Task 3.5: [DONE] flushes pending state
// ============================================================================

#[tokio::test]
async fn test_completions_stream_done_flushes_pending() {
    let mut server = mockito::Server::new_async().await;

    // Tool call without finish_reason, relying on [DONE] for safety flush
    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"test\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}]}\n\n\
                    data: [DONE]\n\n";

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
        Transcript::with_messages(vec![Message::user("Test")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Should have ToolCall then Complete (flushed by [DONE])
    let event_types: Vec<_> = events
        .iter()
        .map(|e| match e {
            Ok(ProviderEvent::ToolCall { .. }) => "ToolCall",
            Ok(ProviderEvent::Complete) => "Complete",
            _ => "Other",
        })
        .collect();

    assert_eq!(event_types, vec!["ToolCall", "Complete"]);

    // Verify the tool call has the correct data
    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].call_id, "call_1");
    assert_eq!(tool_calls[0].arguments["x"], 1);
}

// ============================================================================
// Task 3.6: Delayed/omitted metadata
// ============================================================================

#[tokio::test]
async fn test_completions_stream_delayed_metadata() {
    let mut server = mockito::Server::new_async().await;

    // Tool call where id and name arrive after arguments start
    let chunk1 = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]}}]}

"#;
    let chunk2 = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_delayed\",\"function\":{\"name\":\"delayed_func\"}}]}}]}\n\n";
    let chunk3 = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"x\\\":1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n";
    let chunk4 = "data: [DONE]\n\n";

    let sse_body = format!("{}{}{}{}", chunk1, chunk2, chunk3, chunk4);

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
        Transcript::with_messages(vec![Message::user("Test")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls.len(), 1);
    // Should use the later-arriving metadata
    assert_eq!(tool_calls[0].call_id, "call_delayed");
    assert_eq!(tool_calls[0].tool_name, "delayed_func");
    assert_eq!(tool_calls[0].arguments["x"], 1);
}

#[tokio::test]
async fn test_completions_stream_missing_metadata_best_effort() {
    let mut server = mockito::Server::new_async().await;

    // Tool call that never receives id or name - should still emit with fallbacks
    let sse_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"x\\\":1}\"}}]}}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";

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
        Transcript::with_messages(vec![Message::user("Test")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { call }) => Some(call.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_calls.len(), 1);
    // Should have generated fallback id and empty name
    assert!(tool_calls[0].call_id.starts_with("call_"));
    assert_eq!(tool_calls[0].tool_name, "");
    // Arguments should still be valid
    assert_eq!(tool_calls[0].arguments["x"], 1);
}

// ============================================================================
// Negative streaming tests
// ============================================================================

#[tokio::test]
async fn test_completions_stream_invalid_json_skipped() {
    let mut server = mockito::Server::new_async().await;

    // First event is invalid JSON, second is valid, then [DONE]
    let sse_body = "data: {not valid json}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n\
                    data: [DONE]\n\n";

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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Should still get the valid output and Complete despite the invalid chunk
    let outputs: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::Output { content }) => Some(content.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(outputs.join(""), "ok");
    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}

#[tokio::test]
async fn test_completions_stream_empty_done() {
    let mut server = mockito::Server::new_async().await;

    // Stream with only [DONE] — no content
    let sse_body = "data: [DONE]\n\n";

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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Should have exactly one event: Complete
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], Ok(ProviderEvent::Complete)));
}

#[tokio::test]
async fn test_anthropic_stream_invalid_json_skipped() {
    let mut server = mockito::Server::new_async().await;

    let sse_body = "data: {bad json}\n\n\
                    event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
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

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = messages::infer_stream(client, &profile, &profile.base_url, request)
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

    assert_eq!(outputs.join(""), "hi");
    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}

// ============================================================================
// ChoiceRequest end-to-end test
// ============================================================================

#[tokio::test]
async fn test_completions_choice_request_tool_call() {
    let mut server = mockito::Server::new_async().await;

    let choice_args = serde_json::to_string(&json!({
        "prompt": "Pick a color",
        "selection_mode": "single",
        "items": [{"id": "red", "label": "Red"}, {"id": "blue", "label": "Blue"}]
    }))
    .unwrap();

    let sse_body = format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_choice\",\"function\":{{\"name\":\"runtime.request_choice\",\"arguments\":\"{}\"}}}}]}},\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n",
        choice_args.replace('\\', "\\\\").replace('"', "\\\"")
    );

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
        Transcript::with_messages(vec![Message::user("pick a color")]),
    );

    let client = build_test_client(&profile, &runtime).unwrap();
    let stream = completions::infer_stream(client, &profile, &profile.base_url, request)
        .await
        .unwrap();
    let events: Vec<_> = futures::executor::block_on_stream(stream).collect();

    mock.assert_async().await;

    // Should emit ChoiceRequest, not ToolCall
    let choice_requests: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ChoiceRequest { request }) => Some(request.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(choice_requests.len(), 1);
    assert_eq!(choice_requests[0].prompt, "Pick a color");
    assert_eq!(choice_requests[0].items.len(), 2);

    // Should NOT emit a ToolCall for this
    let tool_calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(ProviderEvent::ToolCall { .. }) => Some(true),
            _ => None,
        })
        .collect();
    assert!(
        tool_calls.is_empty(),
        "Should not emit ToolCall for choice requests"
    );

    assert!(events
        .iter()
        .any(|e| matches!(e, Ok(ProviderEvent::Complete))));
}
