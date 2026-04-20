//! Tests for streaming behavior and completed tool calls

use iron_providers::{
    model::{ProviderEvent, ToolCall},
    GenerationConfig, InferenceRequest, Message, ToolDefinition, ToolPolicy, Transcript,
};
use serde_json::json;

#[test]
fn test_provider_event_output() {
    let event = ProviderEvent::Output {
        content: "Hello!".to_string(),
    };
    match event {
        ProviderEvent::Output { content } => assert_eq!(content, "Hello!"),
        _ => panic!("Expected Output event"),
    }
}

#[test]
fn test_provider_event_tool_call() {
    let call = ToolCall::new("call_123", "get_weather", json!({"location": "NYC"}));
    let event = ProviderEvent::ToolCall { call };

    match event {
        ProviderEvent::ToolCall { call } => {
            assert_eq!(call.call_id, "call_123");
            assert_eq!(call.tool_name, "get_weather");
            assert_eq!(call.arguments, json!({"location": "NYC"}));
        }
        _ => panic!("Expected ToolCall event"),
    }
}

#[test]
fn test_provider_event_complete() {
    let event = ProviderEvent::Complete;
    assert!(matches!(event, ProviderEvent::Complete));
}

#[test]
fn test_provider_event_error() {
    let event = ProviderEvent::Error {
        message: "Something went wrong".to_string(),
    };
    match event {
        ProviderEvent::Error { message } => assert_eq!(message, "Something went wrong"),
        _ => panic!("Expected Error event"),
    }
}

#[test]
fn test_provider_event_status() {
    let event = ProviderEvent::Status {
        message: "Thinking...".to_string(),
    };
    match event {
        ProviderEvent::Status { message } => assert_eq!(message, "Thinking..."),
        _ => panic!("Expected Status event"),
    }
}

#[test]
fn test_tool_call_with_structured_json() {
    let args = json!({
        "location": "San Francisco",
        "units": "celsius",
        "detailed": true
    });

    let call = ToolCall::new("call_456", "get_weather", args.clone());
    assert_eq!(call.call_id, "call_456");
    assert_eq!(call.tool_name, "get_weather");
    assert_eq!(call.arguments, args);
}

#[test]
fn test_streaming_request_setup() {
    let transcript = Transcript::with_messages(vec![Message::user("What's the weather in NYC?")]);

    let tools = vec![ToolDefinition::new(
        "get_weather",
        "Get weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }),
    )];

    let request = InferenceRequest::new("gpt-4o", transcript)
        .with_tools(tools)
        .with_tool_policy(ToolPolicy::Auto)
        .with_generation(GenerationConfig::new().with_temperature(0.7));

    // Streaming is determined by calling infer_stream vs infer, not by a field.
    assert_eq!(request.model, "gpt-4o");
    assert_eq!(request.tools.len(), 1);
}
