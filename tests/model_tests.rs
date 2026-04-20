//! Tests for iron-providers models

use iron_providers::model::{
    GenerationConfig, InferenceRequest, Message, ProviderEvent, ToolCall, ToolDefinition,
    ToolPolicy, Transcript,
};
use serde_json::json;

#[test]
fn test_transcript_new() {
    let transcript = Transcript::new();
    assert!(transcript.is_empty());
    assert_eq!(transcript.messages.len(), 0);
}

#[test]
fn test_transcript_with_messages() {
    let messages = vec![Message::user("Hello"), Message::assistant("Hi!")];
    let transcript = Transcript::with_messages(messages);
    assert_eq!(transcript.messages.len(), 2);
}

#[test]
fn test_transcript_add_message() {
    let mut transcript = Transcript::new();
    transcript.add_message(Message::user("Hello"));
    assert_eq!(transcript.messages.len(), 1);
}

#[test]
fn test_message_user() {
    let msg = Message::user("Hello");
    assert_eq!(
        msg,
        Message::User {
            content: "Hello".to_string()
        }
    );
}

#[test]
fn test_message_assistant() {
    let msg = Message::assistant("Hi!");
    assert_eq!(
        msg,
        Message::Assistant {
            content: "Hi!".to_string()
        }
    );
}

#[test]
fn test_message_tool() {
    let result = json!({"status": "ok"});
    let msg = Message::tool("call_1", "my_tool", result.clone());
    assert_eq!(
        msg,
        Message::Tool {
            call_id: "call_1".to_string(),
            tool_name: "my_tool".to_string(),
            result
        }
    );
}

#[test]
fn test_tool_definition_new() {
    let schema = json!({"type": "object"});
    let def = ToolDefinition::new("test", "A test tool", schema.clone());
    assert_eq!(def.name, "test");
    assert_eq!(def.description, "A test tool");
    assert_eq!(def.input_schema, schema);
}

#[test]
fn test_tool_policy_default() {
    assert_eq!(ToolPolicy::default(), ToolPolicy::Auto);
}

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert!(config.temperature.is_none());
    assert!(config.max_tokens.is_none());
    assert!(config.top_p.is_none());
    assert!(config.stop.is_none());
}

#[test]
fn test_generation_config_builder() {
    let config = GenerationConfig::new()
        .with_temperature(0.7)
        .with_max_tokens(100)
        .with_top_p(0.9);

    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_tokens, Some(100));
    assert_eq!(config.top_p, Some(0.9));
}

#[test]
fn test_tool_call_new() {
    let args = json!({"name": "test"});
    let call = ToolCall::new("call_1", "my_tool", args.clone());
    assert_eq!(call.call_id, "call_1");
    assert_eq!(call.tool_name, "my_tool");
    assert_eq!(call.arguments, args);
}

#[test]
fn test_provider_event_variants() {
    let status = ProviderEvent::Status {
        message: "thinking".to_string(),
    };
    let output = ProviderEvent::Output {
        content: "hello".to_string(),
    };
    let complete = ProviderEvent::Complete;

    match status {
        ProviderEvent::Status { message } => assert_eq!(message, "thinking"),
        _ => panic!("Wrong variant"),
    }

    match output {
        ProviderEvent::Output { content } => assert_eq!(content, "hello"),
        _ => panic!("Wrong variant"),
    }

    assert!(matches!(complete, ProviderEvent::Complete));
}

#[test]
fn test_inference_request_new() {
    let transcript = Transcript::new();
    let req = InferenceRequest::new("gpt-4", transcript);
    assert_eq!(req.model, "gpt-4");
    assert!(req.instructions.is_none());
}

#[test]
fn test_inference_request_builder() {
    let transcript = Transcript::new();
    let req = InferenceRequest::new("gpt-4", transcript)
        .with_instructions("Be helpful")
        .with_tool_policy(ToolPolicy::Required);

    assert_eq!(req.instructions, Some("Be helpful".to_string()));
    assert_eq!(req.tool_policy, ToolPolicy::Required);
}
