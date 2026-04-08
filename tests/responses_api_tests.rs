//! Tests for OpenAI Responses API integration

use iron_providers::{
    GenerationConfig, InferenceRequest, Message, OpenAiConfig, ToolDefinition, ToolPolicy,
    Transcript,
};
use serde_json::json;

#[test]
fn test_openai_config_new() {
    let config = OpenAiConfig::new("test-key".to_string());
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.default_model, "gpt-4o");
    assert!(config.base_url.is_none());
}

#[test]
fn test_openai_config_builder() {
    let config = OpenAiConfig::new("test-key".to_string())
        .with_model("gpt-4".to_string())
        .with_base_url("https://custom.api.com".to_string());

    assert_eq!(config.default_model, "gpt-4");
    assert_eq!(config.base_url, Some("https://custom.api.com".to_string()));
}

#[test]
fn test_inference_request_with_instructions() {
    let transcript = Transcript::new();
    let request = InferenceRequest::new("gpt-4o", transcript).with_instructions("Be helpful");

    assert_eq!(request.instructions, Some("Be helpful".to_string()));
    assert_eq!(request.model, "gpt-4o");
}

#[test]
fn test_inference_request_with_tools() {
    let transcript = Transcript::new();
    let tools = vec![ToolDefinition::new(
        "get_weather",
        "Get the weather",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }),
    )];

    let request = InferenceRequest::new("gpt-4o", transcript)
        .with_tools(tools)
        .with_tool_policy(ToolPolicy::Auto);

    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "get_weather");
    assert!(matches!(request.tool_policy, ToolPolicy::Auto));
}

#[test]
fn test_inference_request_with_generation_config() {
    let transcript = Transcript::new();
    let gen_config = GenerationConfig::new()
        .with_temperature(0.7)
        .with_max_tokens(100)
        .with_top_p(0.9);

    let request = InferenceRequest::new("gpt-4o", transcript).with_generation(gen_config);

    assert_eq!(request.generation.temperature, Some(0.7));
    assert_eq!(request.generation.max_tokens, Some(100));
    assert_eq!(request.generation.top_p, Some(0.9));
}

#[test]
fn test_transcript_with_messages() {
    let messages = vec![
        Message::user("What's the weather?"),
        Message::assistant("I'll check that for you."),
        Message::tool("call_1", "get_weather", json!({"temp": 72})),
    ];

    let transcript = Transcript::with_messages(messages);
    assert_eq!(transcript.messages.len(), 3);
}

#[test]
fn test_tool_policy_variants() {
    use iron_providers::ToolPolicy;

    assert!(matches!(ToolPolicy::None, ToolPolicy::None));
    assert!(matches!(ToolPolicy::Auto, ToolPolicy::Auto));
    assert!(matches!(ToolPolicy::Required, ToolPolicy::Required));
    assert!(matches!(
        ToolPolicy::Specific("test".to_string()),
        ToolPolicy::Specific(_)
    ));
}
