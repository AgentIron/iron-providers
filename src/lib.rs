#![warn(
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links
)]
//! iron-providers: Semantic provider boundary with multi-provider support
//!
//! This crate provides a function-oriented provider API that normalizes
//! inference requests and responses across different LLM backends.
//! Supports OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages
//! API families via a profile-driven generic provider and registry.

pub mod anthropic;
pub mod completions;
pub mod error;
pub mod generic_provider;
pub mod model;
pub mod openai;
pub mod profile;
pub mod provider;
pub mod registry;

#[cfg(test)]
mod mock_provider_tests;

pub use error::{ProviderError, ProviderResult};
pub use generic_provider::GenericProvider;
pub use model::{
    GenerationConfig, InferenceRequest, Message, ProviderEvent, ToolCall, ToolChoice,
    ToolDefinition, ToolPolicy, Transcript,
};
pub use profile::{
    ApiFamily, AuthStrategy, EndpointPurpose, ProviderProfile, ProviderQuirks, RuntimeConfig,
    RuntimeConfigSource,
};
pub use provider::{OpenAiProvider, Provider, ProviderFuture};
pub use registry::ProviderRegistry;

pub use openai::{infer, infer_stream, OpenAiConfig, OpenAiConfigSource};

pub mod prelude {
    pub use crate::{
        infer, infer_stream, ApiFamily, AuthStrategy, EndpointPurpose, GenerationConfig,
        GenericProvider, InferenceRequest, Message, OpenAiConfig, OpenAiConfigSource,
        OpenAiProvider, Provider, ProviderError, ProviderEvent, ProviderProfile, ProviderRegistry,
        ProviderResult, RuntimeConfig, RuntimeConfigSource, ToolCall, ToolChoice, ToolDefinition,
        ToolPolicy, Transcript,
    };
}
