//! OpenAI Responses API provider implementation
//!
//! This module provides function-oriented inference using the OpenAI Responses API.
//! It translates semantic requests to OpenAI format and normalizes responses.

use crate::{
    error::ProviderResult,
    model::{ProviderEvent, ToolCall},
    InferenceRequest, ProviderError,
};
use async_openai::{
    config::OpenAIConfig,
    types::responses::{
        CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput, FunctionToolCall,
        InputItem, Item, OutputItem, OutputMessageContent, Response, ResponseStreamEvent, Role,
        Tool as OpenAiTool, ToolChoiceFunction, ToolChoiceOptions, ToolChoiceParam,
    },
    Client,
};
use futures::stream::{BoxStream, StreamExt};
use serde_json::Value;

/// Projection trait for caller-owned config types that can produce an `OpenAiConfig`.
///
/// Implement this on your application config to project OpenAI-specific settings
/// into a validated provider-owned snapshot.
pub trait OpenAiConfigSource {
    /// Project a validated `OpenAiConfig` snapshot from this source.
    fn to_openai_config(&self) -> Result<OpenAiConfig, ProviderError>;
}

/// OpenAI-specific configuration
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    /// API key used for authentication.
    pub api_key: String,
    /// Optional API base URL override.
    pub base_url: Option<String>,
    /// Default model name used by callers that do not override it.
    pub default_model: String,
}

impl OpenAiConfig {
    /// Create a new OpenAI configuration with the given API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: None,
            default_model: "gpt-4o".to_string(),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = Some(url);
        self
    }

    /// Set the default model.
    pub fn with_model(mut self, model: String) -> Self {
        self.default_model = model;
        self
    }

    /// Validate this config, returning an error if the API key is empty.
    pub fn validate(&self) -> ProviderResult<()> {
        if self.api_key.trim().is_empty() {
            return Err(ProviderError::invalid_request(
                "OpenAI API key is required but was empty",
            ));
        }
        Ok(())
    }
}

/// Build the OpenAI client from config
fn build_client(config: &OpenAiConfig) -> Client<OpenAIConfig> {
    let mut openai_config = OpenAIConfig::default().with_api_key(&config.api_key);

    if let Some(ref base_url) = config.base_url {
        openai_config = openai_config.with_api_base(base_url);
    }

    Client::with_config(openai_config)
}

/// Build input items from our transcript
pub(crate) fn build_input_items(request: &InferenceRequest) -> Vec<InputItem> {
    let mut items = Vec::new();

    for msg in &request.transcript.messages {
        match msg {
            crate::Message::User { content } => {
                items.push(InputItem::EasyMessage(EasyInputMessage {
                    role: Role::User,
                    content: EasyInputContent::Text(content.clone()),
                    ..Default::default()
                }));
            }
            crate::Message::Assistant { content } => {
                items.push(InputItem::EasyMessage(EasyInputMessage {
                    role: Role::Assistant,
                    content: EasyInputContent::Text(content.clone()),
                    ..Default::default()
                }));
            }
            crate::Message::AssistantToolCall {
                call_id,
                tool_name,
                arguments,
            } => {
                items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                    call_id: call_id.clone(),
                    name: tool_name.clone(),
                    arguments: serde_json::to_string(arguments)
                        .unwrap_or_else(|_| arguments.to_string()),
                    namespace: None,
                    id: None,
                    status: None,
                })));
            }
            crate::Message::Tool {
                call_id,
                tool_name: _,
                result,
            } => {
                // Tool results go as function_call_output
                let output_str =
                    serde_json::to_string(result).unwrap_or_else(|_| result.to_string());
                items.push(InputItem::Item(Item::FunctionCallOutput(
                    async_openai::types::responses::FunctionCallOutputItemParam {
                        call_id: call_id.clone(),
                        output: FunctionCallOutput::Text(output_str),
                        id: None,
                        status: None,
                    },
                )));
            }
        }
    }

    items
}

/// Build tools for OpenAI
pub(crate) fn build_tools(request: &InferenceRequest) -> Option<Vec<OpenAiTool>> {
    if request.tools.is_empty() {
        return None;
    }

    Some(
        request
            .tools
            .iter()
            .map(|tool| {
                OpenAiTool::Function(async_openai::types::responses::FunctionTool {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone()),
                    parameters: Some(tool.input_schema.clone()),
                    ..Default::default()
                })
            })
            .collect(),
    )
}

/// Build tool choice for OpenAI
pub(crate) fn build_tool_choice(request: &InferenceRequest) -> Option<ToolChoiceParam> {
    match request.tool_policy {
        crate::ToolPolicy::None => Some(ToolChoiceParam::Mode(ToolChoiceOptions::None)),
        crate::ToolPolicy::Auto => Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
        crate::ToolPolicy::Required => Some(ToolChoiceParam::Mode(ToolChoiceOptions::Required)),
        crate::ToolPolicy::Specific(ref name) => {
            Some(ToolChoiceParam::Function(ToolChoiceFunction {
                name: name.clone(),
            }))
        }
    }
}

/// Handle OpenAI errors
fn handle_error(err: async_openai::error::OpenAIError) -> ProviderError {
    use async_openai::error::OpenAIError;

    match err {
        OpenAIError::ApiError(api_err) => {
            let code = api_err.code.as_deref().unwrap_or("");
            let message = api_err.message.clone();

            if code.contains("invalid_api_key")
                || code.contains("authentication")
                || message.to_lowercase().contains("api key")
                || message.to_lowercase().contains("unauthorized")
            {
                ProviderError::auth(message)
            } else if code.contains("rate_limit") || code.contains("rateLimit") {
                ProviderError::rate_limit(message, None)
            } else if code.contains("invalid_request") || code.contains("invalid_request_error") {
                ProviderError::invalid_request(message)
            } else if code.contains("model") || code.contains("model_not_found") {
                ProviderError::model(message)
            } else {
                ProviderError::general(message)
            }
        }
        OpenAIError::Reqwest(e) => ProviderError::transport(e.to_string()),
        OpenAIError::JSONDeserialize(e, _) => {
            ProviderError::malformed(format!("Failed to deserialize response: {}", e))
        }
        _ => ProviderError::general(err.to_string()),
    }
}

/// Convert a Response to provider events
fn response_to_events(response: Response) -> ProviderResult<Vec<ProviderEvent>> {
    let mut events = Vec::new();

    for item in response.output {
        match item {
            OutputItem::Message(message) => {
                // Extract text content from the message
                for content in message.content {
                    match content {
                        OutputMessageContent::OutputText(text) => {
                            events.push(ProviderEvent::Output { content: text.text });
                        }
                        OutputMessageContent::Refusal(refusal) => {
                            events.push(ProviderEvent::Error {
                                message: format!("Model refusal: {}", refusal.refusal),
                            });
                        }
                    }
                }
            }
            OutputItem::FunctionCall(func_call) => {
                // Parse arguments as JSON
                let arguments = serde_json::from_str(&func_call.arguments)
                    .unwrap_or_else(|_| Value::String(func_call.arguments.clone()));

                events.push(ProviderEvent::ToolCall {
                    call: ToolCall::new(func_call.call_id, func_call.name, arguments),
                });
            }
            OutputItem::Reasoning(reasoning) => {
                // For now, treat reasoning as status updates
                if let Some(summary) = reasoning.summary.first() {
                    match summary {
                        async_openai::types::responses::SummaryPart::SummaryText(text) => {
                            events.push(ProviderEvent::Status {
                                message: format!("Reasoning: {}", text.text),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    events.push(ProviderEvent::Complete);
    Ok(events)
}

/// Perform non-streaming inference using OpenAI Responses API
pub async fn infer(
    config: &OpenAiConfig,
    request: InferenceRequest,
) -> ProviderResult<Vec<ProviderEvent>> {
    let client = build_client(config);

    let input_items = build_input_items(&request);
    let tools = build_tools(&request);
    let tool_choice = build_tool_choice(&request);

    let params = CreateResponse {
        model: Some(request.model),
        input: async_openai::types::responses::InputParam::Items(input_items),
        instructions: request.instructions,
        tools,
        tool_choice,
        temperature: request.generation.temperature,
        max_output_tokens: request.generation.max_tokens,
        top_p: request.generation.top_p,
        stream: Some(false),
        ..Default::default()
    };

    let response: Response = client
        .responses()
        .create(params)
        .await
        .map_err(handle_error)?;

    response_to_events(response)
}

/// Process a stream event and return provider events
/// Tool calls are emitted only when complete via ResponseOutputItemDone
pub(crate) fn process_stream_event(event: ResponseStreamEvent) -> Vec<ProviderEvent> {
    let mut events = Vec::new();

    match event {
        ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
            events.push(ProviderEvent::Output {
                content: delta.delta,
            });
        }
        ResponseStreamEvent::ResponseOutputItemDone(output_item) => match output_item.item {
            OutputItem::Message(msg) => {
                for content in msg.content {
                    match content {
                        OutputMessageContent::OutputText(_) => {}
                        OutputMessageContent::Refusal(refusal) => {
                            events.push(ProviderEvent::Error {
                                message: format!("Model refusal: {}", refusal.refusal),
                            });
                        }
                    }
                }
            }
            OutputItem::FunctionCall(func_call) => {
                let arguments = serde_json::from_str(&func_call.arguments)
                    .unwrap_or_else(|_| Value::String(func_call.arguments.clone()));
                events.push(ProviderEvent::ToolCall {
                    call: ToolCall::new(func_call.call_id, func_call.name, arguments),
                });
            }
            OutputItem::Reasoning(reasoning) => {
                if let Some(summary) = reasoning.summary.first() {
                    match summary {
                        async_openai::types::responses::SummaryPart::SummaryText(text) => {
                            events.push(ProviderEvent::Status {
                                message: format!("Reasoning: {}", text.text),
                            });
                        }
                    }
                }
            }
            _ => {}
        },
        ResponseStreamEvent::ResponseFailed(failed) => {
            if let Some(error) = failed.response.error {
                events.push(ProviderEvent::Error {
                    message: error.message,
                });
            }
        }
        ResponseStreamEvent::ResponseCompleted(_) => {
            // Response completed successfully
        }
        _ => {
            // Other events we don't need to handle (deltas are aggregated by ResponseOutputItemDone)
        }
    }

    events
}

/// Perform streaming inference using OpenAI Responses API
///
/// Returns a stream of provider events. Tool calls are emitted only when complete
/// (via ResponseOutputItemDone), not as partial chunks.
pub async fn infer_stream(
    config: &OpenAiConfig,
    request: InferenceRequest,
) -> ProviderResult<BoxStream<'static, ProviderResult<ProviderEvent>>> {
    let client = build_client(config);

    let input_items = build_input_items(&request);
    let tools = build_tools(&request);
    let tool_choice = build_tool_choice(&request);

    let params = CreateResponse {
        model: Some(request.model),
        input: async_openai::types::responses::InputParam::Items(input_items),
        instructions: request.instructions,
        tools,
        tool_choice,
        temperature: request.generation.temperature,
        max_output_tokens: request.generation.max_tokens,
        top_p: request.generation.top_p,
        stream: Some(true),
        ..Default::default()
    };

    let stream = client
        .responses()
        .create_stream(params)
        .await
        .map_err(handle_error)?;

    // Transform the stream - emits only complete tool calls
    let stream = stream.map(|event| match event {
        Ok(event) => {
            let events = process_stream_event(event);
            Ok(futures::stream::iter(events.into_iter().map(Ok)))
        }
        Err(e) => Err(handle_error(e)),
    });

    // Flatten the stream of streams
    let flattened: BoxStream<'static, ProviderResult<ProviderEvent>> = stream
        .flat_map(|result| match result {
            Ok(stream) => stream.boxed(),
            Err(e) => futures::stream::iter(vec![Err(e)]).boxed(),
        })
        .boxed();

    // Add completion event at the end
    let with_completion =
        flattened.chain(futures::stream::once(async { Ok(ProviderEvent::Complete) }));

    Ok(with_completion.boxed())
}

#[cfg(test)]
mod tests {
    use super::{build_input_items, build_tool_choice, build_tools};
    use crate::{InferenceRequest, Message, ToolDefinition, ToolPolicy, Transcript};
    use async_openai::types::responses::{
        EasyInputContent, InputItem, Tool as OpenAiTool, ToolChoiceOptions, ToolChoiceParam,
    };
    use serde_json::json;

    #[test]
    fn request_shaping_preserves_transcript_and_tool_outputs() {
        let transcript = Transcript::with_messages(vec![
            Message::user("hello"),
            Message::assistant("hi"),
            Message::tool("call_1", "lookup", json!({"ok": true})),
        ]);
        let request = InferenceRequest::new("gpt-4o", transcript);
        let items = build_input_items(&request);

        assert_eq!(items.len(), 3);
        match &items[0] {
            InputItem::EasyMessage(message) => {
                assert_eq!(message.content, EasyInputContent::Text("hello".to_string()));
            }
            _ => panic!("expected easy message"),
        }
        match &items[1] {
            InputItem::EasyMessage(message) => {
                assert_eq!(message.content, EasyInputContent::Text("hi".to_string()));
            }
            _ => panic!("expected easy message"),
        }
        match &items[2] {
            InputItem::Item(_) => {}
            _ => panic!("expected tool output item"),
        }
    }

    #[test]
    fn request_shaping_maps_tool_definitions() {
        let request = InferenceRequest::new("gpt-4o", Transcript::new()).with_tools(vec![
            ToolDefinition::new(
                "lookup",
                "lookup a value",
                json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            ),
        ]);

        let tools = build_tools(&request).expect("tools");
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            OpenAiTool::Function(function) => {
                assert_eq!(function.name, "lookup");
                assert_eq!(function.description.as_deref(), Some("lookup a value"));
            }
            _ => panic!("expected function tool"),
        }
    }

    #[test]
    fn request_shaping_maps_tool_policy() {
        let request =
            InferenceRequest::new("gpt-4o", Transcript::new()).with_tool_policy(ToolPolicy::None);
        assert_eq!(
            build_tool_choice(&request),
            Some(ToolChoiceParam::Mode(ToolChoiceOptions::None))
        );

        let request = InferenceRequest::new("gpt-4o", Transcript::new())
            .with_tool_policy(ToolPolicy::Specific("lookup".to_string()));
        match build_tool_choice(&request) {
            Some(ToolChoiceParam::Function(function)) => assert_eq!(function.name, "lookup"),
            _ => panic!("expected specific tool choice"),
        }
    }
}
