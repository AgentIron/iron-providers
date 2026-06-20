//! Semantic provider request/response models
//!
//! These types define the normalized boundary between iron-core and
//! provider implementations. They are intentionally domain-oriented
//! rather than mirroring any specific provider's wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Reserved internal tool name used by providers to normalize model-originated
/// choice requests into first-class `ProviderEvent::ChoiceRequest` events.
pub const CHOICE_REQUEST_TOOL_NAME: &str = "runtime.request_choice";

/// A transcript of conversation messages.
///
/// # Example
///
/// ```
/// use iron_providers::{Message, Transcript};
///
/// let transcript = Transcript::with_messages(vec![
///     Message::user("Hello"),
///     Message::assistant("Hi there"),
/// ]);
/// assert!(!transcript.is_empty());
/// ```
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

/// A message in the conversation transcript.
///
/// # Example
///
/// ```
/// use iron_providers::Message;
/// use serde_json::json;
///
/// let user = Message::user("What is the weather?");
/// let tool = Message::tool("call-1", "get_weather", json!({"city": "Berlin"}));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// User message with text content
    User {
        /// Text content of the user message.
        content: String,
    },
    /// Assistant message with text content
    Assistant {
        /// Text content of the assistant message.
        content: String,
    },
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

    /// Create a tool result message.
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

/// Selection cardinality for a provider-originated choice request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceSelectionMode {
    /// Caller must select exactly one item.
    Single,
    /// Caller may select zero or more items.
    Multiple,
}

/// One selectable item in a provider-originated choice request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceItem {
    /// Stable identifier for the selectable item.
    pub id: String,
    /// Human-readable label shown to the caller.
    pub label: String,
    /// Optional longer description of the item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A first-class model-originated choice request surfaced by the provider/runtime layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceRequest {
    /// Prompt text describing what the caller is being asked to choose.
    pub prompt: String,
    /// Whether the caller must pick one or may pick multiple items.
    pub selection_mode: ChoiceSelectionMode,
    /// Selectable items.
    pub items: Vec<ChoiceItem>,
}

impl ChoiceRequest {
    /// Parse a choice request from a structured JSON value.
    pub fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value)
    }
}

/// Model-facing tool definition.
///
/// # Example
///
/// ```
/// use iron_providers::ToolDefinition;
/// use serde_json::json;
///
/// let tool = ToolDefinition::new(
///     "get_weather",
///     "Get current weather for a city",
///     json!({"type": "object", "properties": {"city": {"type": "string"}}}),
/// );
/// ```
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

/// Tool choice policy.
///
/// # Example
///
/// ```
/// use iron_providers::ToolPolicy;
///
/// let policy = ToolPolicy::Specific("get_weather".to_string());
/// ```
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

/// Normalized generation configuration.
///
/// # Example
///
/// ```
/// use iron_providers::GenerationConfig;
///
/// let config = GenerationConfig::new()
///     .with_temperature(0.7)
///     .with_max_tokens(2048)
///     .with_top_p(0.95);
/// ```
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

/// Normalized provider-reported token usage for a single inference request.
///
/// All fields are optional because provider families differ in what they
/// return.  When present, each value represents the provider's cumulative
/// snapshot for the current request, not an incremental delta.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input or prompt tokens reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output or completion tokens reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Cached input tokens reported by OpenAI-style providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    /// Cache creation input tokens reported by Anthropic-style providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    /// Cache read input tokens reported by Anthropic-style providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// Reasoning or thinking output tokens reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_output_tokens: Option<u64>,
}

impl TokenUsage {
    /// Create an empty usage snapshot.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Events emitted by the provider during streaming.
///
/// ## Stream termination contract
///
/// - `Complete` is emitted **only** on successful stream termination.
/// - If a provider encounters an unrecoverable error, the stream ends
///   with `Error` and does **not** emit `Complete`.
/// - `Status` events are informational and do not affect termination.
/// - `Usage` events carry cumulative provider-reported token usage.  When
///   multiple `Usage` events appear for the same request, the latest one
///   supersedes earlier snapshots rather than being additive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderEvent {
    /// Status update
    Status {
        /// Informational status message.
        message: String,
    },
    /// Incremental text output
    Output {
        /// Text fragment produced by the model.
        content: String,
    },
    /// Completed tool call
    ToolCall {
        /// Normalized tool call record.
        call: ToolCall,
    },
    /// Structured model-originated choice request.
    ChoiceRequest {
        /// The choice request details.
        request: ChoiceRequest,
    },
    /// Provider-reported token usage snapshot.
    ///
    /// Represents the provider's cumulative usage for the current request.
    /// Consumers should treat later `Usage` events as superseding earlier
    /// ones rather than adding them together.
    Usage {
        /// Cumulative token usage snapshot.
        usage: TokenUsage,
    },
    /// Stream completed successfully.
    ///
    /// This event is emitted exactly once per successful stream and is
    /// never emitted after an unrecoverable error.
    Complete,
    /// Error occurred during streaming.
    ///
    /// Carries a structured [`ProviderError`](crate::ProviderError) so
    /// downstream consumers can programmatically classify the failure
    /// (authentication, rate-limit, transport, etc.).
    ///
    /// If this represents an unrecoverable error, the stream ends
    /// without a subsequent `Complete` event.
    Error {
        /// The structured provider error.
        source: crate::ProviderError,
    },
}

/// A runtime-owned record that is **not** model-visible.
///
/// Runtime records carry structured context (e.g. resolved interaction
/// records, session metadata) that should be available to provider
/// adapters for request assembly but must not be projected into the
/// model-visible conversation transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeRecord {
    /// Stable record kind (e.g. "interaction", "session_state").
    pub kind: String,
    /// Structured payload.
    pub payload: Value,
}

impl RuntimeRecord {
    /// Create a new runtime record.
    pub fn new<S: Into<String>>(kind: S, payload: Value) -> Self {
        Self {
            kind: kind.into(),
            payload,
        }
    }
}

/// Inference context separating model-visible conversation from runtime-only state.
///
/// Provider adapters receive the full context but must only project the
/// `transcript` into model-visible request fields. Runtime records may
/// influence request assembly (e.g. system instructions, metadata headers)
/// through explicit provider-specific mapping logic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InferenceContext {
    /// Model-visible conversation transcript.
    pub transcript: Transcript,
    /// Runtime-only records that are not replayed into model context.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_records: Vec<RuntimeRecord>,
}

impl InferenceContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context with only a transcript (no runtime records).
    pub fn from_transcript(transcript: Transcript) -> Self {
        Self {
            transcript,
            runtime_records: vec![],
        }
    }

    /// Add a runtime record.
    pub fn add_record(&mut self, record: RuntimeRecord) {
        self.runtime_records.push(record);
    }
}

/// Semantic inference request.
///
/// # Example
///
/// ```
/// use iron_providers::{GenerationConfig, InferenceRequest, Message, Transcript};
///
/// let request = InferenceRequest::new(
///     "claude-sonnet-4-20250514",
///     Transcript::with_messages(vec![Message::user("Hello")]),
/// )
/// .with_instructions("Be concise.")
/// .with_generation(GenerationConfig::new().with_temperature(0.5));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Model identifier
    pub model: String,
    /// Optional top-level instructions (system prompt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Inference context containing model-visible transcript and runtime-only records.
    pub context: InferenceContext,
    /// Available tools
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolDefinition>,
    /// Tool usage policy
    #[serde(default)]
    pub tool_policy: ToolPolicy,
    /// Generation settings
    #[serde(default)]
    pub generation: GenerationConfig,
}

impl InferenceRequest {
    /// Create a new inference request for the provided model and transcript.
    pub fn new<S: Into<String>>(model: S, transcript: Transcript) -> Self {
        Self {
            model: model.into(),
            instructions: None,
            context: InferenceContext::from_transcript(transcript),
            tools: vec![],
            tool_policy: ToolPolicy::default(),
            generation: GenerationConfig::default(),
        }
    }

    /// Validate that the model identifier is present and non-empty.
    ///
    /// Called by all provider adapters before constructing a request.
    pub fn validate_model(&self) -> crate::ProviderResult<()> {
        if self.model.trim().is_empty() {
            return Err(crate::ProviderError::invalid_request(
                "InferenceRequest.model must be a non-empty model identifier",
            ));
        }
        Ok(())
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
}
