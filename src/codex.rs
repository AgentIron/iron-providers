//! Codex provider adapter for ChatGPT/Codex backend access.
//!
//! Uses raw HTTP against `https://chatgpt.com/backend-api/codex/responses`.
//! Does not depend on `async-openai` so exact request/response behavior can be
//! asserted in mocked tests.

use crate::{
    model::{ProviderEvent, ToolCall},
    profile::{ProviderCredential, ProviderProfile, RuntimeConfig},
    InferenceRequest, ProviderError, ProviderResult,
};
use futures::stream::{BoxStream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};

/// Parse an unverified JWT payload and extract the ChatGPT account ID if present.
///
/// This is routing metadata extraction only; the provider server still validates
/// the bearer token authoritatively.
pub(crate) fn chatgpt_account_id_from_jwt(id_token: &str) -> Option<String> {
    let payload_b64 = id_token.split('.').nth(1)?;
    let payload_json = base64_decode_url_safe(payload_b64).ok()?;
    let payload: Value = serde_json::from_slice(&payload_json).ok()?;

    payload
        .get("chatgpt_account_id")
        .and_then(|v| v.as_str().map(String::from))
        .or_else(|| {
            payload
                .get("https://api.openai.com/auth")
                .and_then(|nested| nested.get("chatgpt_account_id"))
                .and_then(|v| v.as_str().map(String::from))
        })
        .or_else(|| {
            payload
                .get("organizations")
                .and_then(|orgs| orgs.as_array())
                .and_then(|orgs| orgs.first())
                .and_then(|first| first.get("id"))
                .and_then(|v| v.as_str().map(String::from))
        })
}

fn base64_decode_url_safe(input: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::{engine::general_purpose, Engine as _};
    let padded = match input.len() % 4 {
        0 => input.to_string(),
        n => format!("{}{}", input, "=".repeat(4 - n)),
    };
    general_purpose::URL_SAFE_NO_PAD.decode(padded.trim_end_matches('='))
}

fn build_codex_headers(
    runtime: &RuntimeConfig,
    crate_version: &str,
) -> ProviderResult<Vec<(String, String)>> {
    let mut headers = vec![
        ("originator".to_string(), "iron-providers".to_string()),
        (
            "User-Agent".to_string(),
            format!("iron-providers/{}", crate_version),
        ),
    ];

    if let ProviderCredential::OAuthBearer {
        id_token: Some(ref token),
        ..
    } = runtime.credential
    {
        if let Some(account_id) = chatgpt_account_id_from_jwt(token) {
            headers.push(("chatgpt-account-id".to_string(), account_id));
        }
    }

    Ok(headers)
}

fn build_codex_request_body(request: &InferenceRequest, stream: bool) -> ProviderResult<Value> {
    let mut body = json!({
        "model": request.model,
        "store": false,
        "reasoning": { "effort": "medium" },
        "parallel_tool_calls": true,
        "stream": stream,
    });

    if let Some(ref instructions) = request.instructions {
        body["instructions"] = json!(instructions);
    }

    let input_items: Vec<Value> = request
        .context
        .transcript
        .messages
        .iter()
        .map(|msg| match msg {
            crate::Message::User { content } => {
                json!({ "role": "user", "content": [{"type": "text", "text": content}] })
            }
            crate::Message::Assistant { content } => {
                json!({ "role": "assistant", "content": content })
            }
            crate::Message::AssistantToolCall {
                call_id,
                tool_name,
                arguments,
            } => json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                }],
            }),
            crate::Message::Tool {
                call_id,
                tool_name: _,
                result,
            } => json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result.to_string(),
            }),
        })
        .collect();
    body["input"] = json!(input_items);

    if !request.tools.is_empty() {
        let tool_defs: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                })
            })
            .collect();
        body["tools"] = json!(tool_defs);

        body["tool_choice"] = match request.tool_policy {
            crate::model::ToolPolicy::Auto => json!("auto"),
            crate::model::ToolPolicy::Required => json!("required"),
            crate::model::ToolPolicy::None => json!("none"),
            crate::model::ToolPolicy::Specific(ref name) => json!({
                "type": "function",
                "function": { "name": name }
            }),
        };
    }

    Ok(body)
}

fn build_reqwest_headers(extra: &[(String, String)]) -> ProviderResult<reqwest::header::HeaderMap> {
    let mut h = reqwest::header::HeaderMap::new();
    for (k, v) in extra {
        let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
            ProviderError::invalid_request(format!("Invalid Codex header name '{}': {}", k, e))
        })?;
        let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
            ProviderError::invalid_request(format!("Invalid Codex header value for '{}': {}", k, e))
        })?;
        h.insert(name, value);
    }
    Ok(h)
}

/// Non-streaming Codex inference.
pub async fn infer(
    client: reqwest::Client,
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    let url = format!("{}/responses", profile.base_url);
    let extra_headers = build_codex_headers(runtime, env!("CARGO_PKG_VERSION"))?;
    let body = build_codex_request_body(&request, false)?;
    let headers = build_reqwest_headers(&extra_headers)?;

    let response = client
        .post(&url)
        .json(&body)
        .headers(headers)
        .send()
        .await
        .map_err(|e| ProviderError::transport(format!("Codex request failed: {}", e)))?;

    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(ProviderError::auth("Codex returned 401 Unauthorized"));
    }
    if !status.is_success() {
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| format!("HTTP {}", status));
        return Err(ProviderError::general(format!(
            "Codex request failed with {}: {}",
            status, text
        )));
    }

    let data: CodexResponse = response
        .json()
        .await
        .map_err(|e| ProviderError::malformed(format!("Failed to parse Codex response: {}", e)))?;

    parse_codex_response(data)
}

/// Streaming Codex inference.
pub async fn infer_stream(
    client: reqwest::Client,
    profile: &ProviderProfile,
    runtime: &RuntimeConfig,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    let url = format!("{}/responses", profile.base_url);
    let extra_headers = build_codex_headers(runtime, env!("CARGO_PKG_VERSION"))?;
    let body = build_codex_request_body(&request, true)?;
    let headers = build_reqwest_headers(&extra_headers)?;

    let response = client
        .post(&url)
        .json(&body)
        .headers(headers)
        .send()
        .await
        .map_err(|e| ProviderError::transport(format!("Codex request failed: {}", e)))?;

    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(ProviderError::auth("Codex returned 401 Unauthorized"));
    }
    if !status.is_success() {
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| format!("HTTP {}", status));
        return Err(ProviderError::general(format!(
            "Codex request failed with {}: {}",
            status, text
        )));
    }

    let stream = response.bytes_stream();
    let mut parser = crate::sse::SseParser::new();

    let events = stream.map(move |chunk| match chunk {
        Ok(bytes) => {
            let sse_events = parser.feed(&bytes);
            let provider_events: Vec<ProviderResult<ProviderEvent>> = sse_events
                .into_iter()
                .filter_map(|evt| {
                    if evt.data.trim() == "[DONE]" {
                        return Some(Ok(ProviderEvent::Complete));
                    }
                    let parsed: CodexStreamEvent = serde_json::from_str(&evt.data).ok()?;
                    parsed
                        .delta
                        .map(|delta| Ok(ProviderEvent::Output { content: delta }))
                })
                .collect();
            futures::stream::iter(provider_events)
        }
        Err(e) => futures::stream::iter(vec![Err(ProviderError::transport(format!(
            "Codex stream error: {}",
            e
        )))]),
    });

    let flattened = events.flatten();
    Ok(Box::pin(flattened))
}

#[derive(Debug, Deserialize)]
struct CodexResponse {
    output: Option<Vec<CodexOutputItem>>,
    #[serde(default)]
    error: Option<CodexErrorBody>,
}

#[derive(Debug, Deserialize)]
struct CodexOutputItem {
    id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    content: Option<Vec<CodexContentBlock>>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CodexContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CodexErrorBody {
    message: String,
}

#[derive(Debug, Deserialize)]
struct CodexStreamEvent {
    delta: Option<String>,
}

fn parse_codex_response(data: CodexResponse) -> ProviderResult<Vec<ProviderEvent>> {
    if let Some(err) = data.error {
        return Err(ProviderError::general(format!(
            "Codex error: {}",
            err.message
        )));
    }

    let mut events = Vec::new();
    if let Some(output) = data.output {
        for item in output {
            match item.kind.as_str() {
                "message" => {
                    if let Some(content) = item.content {
                        for block in content {
                            if block.kind == "output_text" {
                                if let Some(text) = block.text {
                                    events.push(ProviderEvent::Output { content: text });
                                }
                            }
                        }
                    }
                }
                "function_call" => {
                    if let Some(args) = item.arguments {
                        let arguments =
                            serde_json::from_str(&args).unwrap_or(serde_json::Value::String(args));
                        events.push(ProviderEvent::ToolCall {
                            call: ToolCall {
                                call_id: item
                                    .call_id
                                    .or(item.id)
                                    .unwrap_or_else(|| "codex-call".to_string()),
                                tool_name: item.name.unwrap_or_else(|| "function".to_string()),
                                arguments,
                            },
                        });
                    }
                }
                _ => {}
            }
        }
    }

    events.push(ProviderEvent::Complete);
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        generic_provider::GenericProvider, profile::CredentialKind, provider::Provider,
        AuthStrategy, Message, Transcript,
    };
    use mockito::Matcher;

    fn fake_jwt(payload: &str) -> String {
        use base64::{engine::general_purpose, Engine as _};
        let header = general_purpose::URL_SAFE_NO_PAD.encode(b"{}");
        let payload_b64 = general_purpose::URL_SAFE_NO_PAD.encode(payload.as_bytes());
        let sig = general_purpose::URL_SAFE_NO_PAD.encode(b"signature");
        format!("{}.{}.{}", header, payload_b64, sig)
    }

    fn codex_profile(base_url: &str) -> ProviderProfile {
        ProviderProfile::new("codex", crate::ApiFamily::CodexResponses, base_url)
            .with_models_dev_id("openai")
            .with_credential_auth(CredentialKind::OAuthBearer, AuthStrategy::BearerToken)
    }

    #[test]
    fn test_jwt_top_level_account_id() {
        let payload = r#"{"chatgpt_account_id":"acct_123"}"#;
        let jwt = fake_jwt(payload);
        assert_eq!(
            chatgpt_account_id_from_jwt(&jwt),
            Some("acct_123".to_string())
        );
    }

    #[test]
    fn test_jwt_nested_account_id() {
        let payload = r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_456"}}"#;
        let jwt = fake_jwt(payload);
        assert_eq!(
            chatgpt_account_id_from_jwt(&jwt),
            Some("acct_456".to_string())
        );
    }

    #[test]
    fn test_jwt_organization_fallback() {
        let payload = r#"{"organizations":[{"id":"org_789"}]}"#;
        let jwt = fake_jwt(payload);
        assert_eq!(
            chatgpt_account_id_from_jwt(&jwt),
            Some("org_789".to_string())
        );
    }

    #[test]
    fn test_jwt_top_level_wins_over_nested() {
        let payload = r#"{"chatgpt_account_id":"top","https://api.openai.com/auth":{"chatgpt_account_id":"nested"}}"#;
        let jwt = fake_jwt(payload);
        assert_eq!(chatgpt_account_id_from_jwt(&jwt), Some("top".to_string()));
    }

    #[test]
    fn test_jwt_malformed_returns_none() {
        assert_eq!(chatgpt_account_id_from_jwt("not-a-jwt"), None);
        assert_eq!(chatgpt_account_id_from_jwt("only-one-part"), None);
    }

    #[test]
    fn test_jwt_no_claims_returns_none() {
        let jwt = fake_jwt(r#"{"sub":"user"}"#);
        assert_eq!(chatgpt_account_id_from_jwt(&jwt), None);
    }

    #[test]
    fn test_build_codex_request_body_fields() {
        let request = InferenceRequest::new("gpt-5.3-codex", crate::Transcript::new());
        let body = build_codex_request_body(&request, false).unwrap();
        assert_eq!(body["model"], "gpt-5.3-codex");
        assert_eq!(body["store"], false);
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_build_codex_request_body_stream() {
        let request = InferenceRequest::new("gpt-5.3-codex", crate::Transcript::new());
        let body = build_codex_request_body(&request, true).unwrap();
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_build_codex_headers_without_id_token() {
        let runtime = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "tok".into(),
            expires_at: None,
            id_token: None,
        });
        let headers = build_codex_headers(&runtime, "0.1.8").unwrap();
        assert!(headers.iter().any(|(k, _)| k == "originator"));
        assert!(headers.iter().any(|(k, _)| k == "User-Agent"));
        assert!(!headers.iter().any(|(k, _)| k == "chatgpt-account-id"));
    }

    #[test]
    fn test_build_codex_headers_with_account_id() {
        let payload = r#"{"chatgpt_account_id":"acct_abc"}"#;
        let jwt = fake_jwt(payload);
        let runtime = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "tok".into(),
            expires_at: None,
            id_token: Some(jwt),
        });
        let headers = build_codex_headers(&runtime, "0.1.8").unwrap();
        assert!(headers
            .iter()
            .any(|(k, v)| k == "chatgpt-account-id" && v == "acct_abc"));
    }

    #[test]
    fn test_parse_codex_response_preserves_tool_call_metadata() {
        let response = CodexResponse {
            output: Some(vec![CodexOutputItem {
                id: Some("fc_123".into()),
                call_id: Some("call_123".into()),
                name: Some("lookup".into()),
                kind: "function_call".into(),
                content: None,
                arguments: Some(r#"{"query":"rust"}"#.into()),
            }]),
            error: None,
        };

        let events = parse_codex_response(response).unwrap();
        let tool_call = events
            .iter()
            .find_map(|event| match event {
                ProviderEvent::ToolCall { call } => Some(call),
                _ => None,
            })
            .expect("tool call");

        assert_eq!(tool_call.call_id, "call_123");
        assert_eq!(tool_call.tool_name, "lookup");
        assert_eq!(tool_call.arguments["query"], "rust");
    }

    #[tokio::test]
    async fn test_codex_infer_sends_required_url_headers_and_body() {
        let mut server = mockito::Server::new_async().await;
        let id_token = fake_jwt(r#"{"chatgpt_account_id":"acct_abc"}"#);

        let mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer codex-token")
            .match_header("originator", "iron-providers")
            .match_header(
                "user-agent",
                format!("iron-providers/{}", env!("CARGO_PKG_VERSION")).as_str(),
            )
            .match_header("chatgpt-account-id", "acct_abc")
            .match_body(Matcher::PartialJson(json!({
                "model": "gpt-5.5",
                "store": false,
                "reasoning": { "effort": "medium" },
                "parallel_tool_calls": true,
                "stream": false,
                "instructions": "Be terse"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}"#,
            )
            .create_async()
            .await;

        let runtime = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "codex-token".into(),
            expires_at: None,
            id_token: Some(id_token),
        });
        let provider =
            GenericProvider::from_profile(codex_profile(&server.url()), runtime).unwrap();
        let request = InferenceRequest::new(
            "gpt-5.5",
            Transcript::with_messages(vec![Message::user("hello")]),
        )
        .with_instructions("Be terse");

        let events = provider.infer(request).await.unwrap();
        mock.assert_async().await;

        assert!(events
            .iter()
            .any(|event| matches!(event, ProviderEvent::Output { content } if content == "ok")));
    }

    #[tokio::test]
    async fn test_codex_stream_sends_stream_true_and_parses_sse() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer codex-token")
            .match_header("chatgpt-account-id", Matcher::Missing)
            .match_body(Matcher::PartialJson(json!({
                "model": "gpt-5.5",
                "store": false,
                "reasoning": { "effort": "medium" },
                "parallel_tool_calls": true,
                "stream": true
            })))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body("data: {\"delta\":\"hi\"}\n\ndata: [DONE]\n\n")
            .create_async()
            .await;

        let runtime = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "codex-token".into(),
            expires_at: None,
            id_token: None,
        });
        let provider =
            GenericProvider::from_profile(codex_profile(&server.url()), runtime).unwrap();
        let request = InferenceRequest::new(
            "gpt-5.5",
            Transcript::with_messages(vec![Message::user("hello")]),
        );

        let stream = provider.infer_stream(request).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        mock.assert_async().await;

        assert!(events.iter().any(
            |event| matches!(event, Ok(ProviderEvent::Output { content }) if content == "hi")
        ));
        assert!(events
            .iter()
            .any(|event| matches!(event, Ok(ProviderEvent::Complete))));
    }
}
