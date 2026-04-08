//! Semantic provider request/response models
//!
//! These types define the normalized boundary between iron-core and
//! provider implementations. They are intentionally domain-oriented
//! rather than mirroring any specific provider's wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A transcript of conversation messages
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// Ordered conversation messages.
    pub messages: Vec<Message>,
}

impl Transcript {
    /// Create an empty transcript.
    pub fn new() -> Self {
        Self { messages: vec![] }
    }

    /// Create a transcript with the provided messages.
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Append a message to the transcript.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Return whether the transcript contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// A message in the conversation transcript
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// User message with text content
    User { content: String },
    /// Assistant message with text content
    Assistant { content: String },
    /// Assistant tool call (the model requesting to call a tool)
    AssistantToolCall {
        /// Stable tool call identifier.
        call_id: String,
        /// Requested tool name.
        tool_name: String,
        /// Parsed tool arguments.
        arguments: Value,
    },
    /// Tool result message
    Tool {
        /// Stable tool call identifier.
        call_id: String,
        /// Tool name associated with the result.
        tool_name: String,
        /// Structured tool result.
        result: Value,
    },
}

impl Message {
    /// Create a user message
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    /// Create an assistant message
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self::Assistant {
            content: content.into(),
        }
    }

    /// Create a tool result message
    /// Create an assistant tool result message.
    pub fn tool<S1: Into<String>, S2: Into<String>>(
        call_id: S1,
        tool_name: S2,
        result: Value,
    ) -> Self {
        Self::Tool {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            result,
        }
    }
}

/// Model-facing tool definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name.
    pub name: String,
    /// Natural-language tool description.
    pub description: String,
    /// JSON Schema describing tool arguments.
    pub input_schema: Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new<S1: Into<String>, S2: Into<String>>(
        name: S1,
        description: S2,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Tool choice policy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolPolicy {
    /// No tools allowed
    None,
    /// Model can choose to use tools
    #[default]
    Auto,
    /// Model must use a tool
    Required,
    /// Model must use a specific tool
    Specific(String),
}

/// Tool choice for the model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Do not permit tool use.
    None,
    /// Let the model decide whether to call a tool.
    Auto,
    /// Force some tool usage, without specifying which one.
    Required,
    /// Force a specific tool by name.
    Specific { name: String },
}

impl From<&ToolPolicy> for ToolChoice {
    fn from(policy: &ToolPolicy) -> Self {
        match policy {
            ToolPolicy::None => Self::None,
            ToolPolicy::Auto => Self::Auto,
            ToolPolicy::Required => Self::Required,
            ToolPolicy::Specific(name) => Self::Specific { name: name.clone() },
        }
    }
}

/// Normalized generation configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenerationConfig {
    /// Temperature for sampling (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

impl GenerationConfig {
    /// Create an empty generation configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum output token count.
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set the top-p sampling value.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
}

/// A completed tool call with structured JSON arguments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Stable tool call identifier.
    pub call_id: String,
    /// Tool name selected by the model.
    pub tool_name: String,
    /// Parsed tool arguments.
    pub arguments: Value,
}

impl ToolCall {
    /// Create a normalized tool call record.
    pub fn new<S1: Into<String>, S2: Into<String>>(
        call_id: S1,
        tool_name: S2,
        arguments: Value,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
        }
    }
}

/// Events emitted by the provider during streaming
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderEvent {
    /// Status update
    Status { message: String },
    /// Incremental text output
    Output { content: String },
    /// Completed tool call
    ToolCall { call: ToolCall },
    /// Stream completed successfully
    Complete,
    /// Error occurred
    Error { message: String },
}

/// Semantic inference request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Model identifier
    pub model: String,
    /// Optional top-level instructions (system prompt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Full conversation transcript
    pub transcript: Transcript,
    /// Available tools
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolDefinition>,
    /// Tool usage policy
    #[serde(default)]
    pub tool_policy: ToolPolicy,
    /// Generation settings
    #[serde(default)]
    pub generation: GenerationConfig,
    /// Whether to stream responses
    #[serde(default)]
    pub stream: bool,
}

impl InferenceRequest {
    /// Create a new inference request for the provided model and transcript.
    pub fn new<S: Into<String>>(model: S, transcript: Transcript) -> Self {
        Self {
            model: model.into(),
            instructions: None,
            transcript,
            tools: vec![],
            tool_policy: ToolPolicy::default(),
            generation: GenerationConfig::default(),
            stream: false,
        }
    }

    /// Set top-level instructions for the request.
    pub fn with_instructions<S: Into<String>>(mut self, instructions: S) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Attach tool definitions to the request.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the tool policy for the request.
    pub fn with_tool_policy(mut self, policy: ToolPolicy) -> Self {
        self.tool_policy = policy;
        self
    }

    /// Set generation parameters for the request.
    pub fn with_generation(mut self, generation: GenerationConfig) -> Self {
        self.generation = generation;
        self
    }

    /// Enable or disable streaming responses.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }
}
