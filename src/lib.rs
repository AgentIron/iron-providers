#![warn(
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links
)]
//! iron-providers: Semantic provider boundary for protocol-oriented LLM providers
//!
//! This crate provides a function-oriented provider API that normalizes
//! inference requests and responses across different LLM backends.
//! Supports Responses, Chat Completions, and Messages protocol families via a
//! profile-driven provider connection and registry.
//!
//! # Architecture
//!
//! The crate is organized around a few core concepts:
//!
//! - **[`Provider`] trait**: The async interface for performing inference.
//!   Callers use [`Provider::infer`] for non-streaming and
//!   [`Provider::infer_stream`] for streaming requests.
//!
//! - **[`InferenceRequest`]**: The normalized request type carrying model ID,
//!   instructions, tools, and an [`InferenceContext`].
//!
//! - **[`InferenceContext`]**: Separates model-visible conversation
//!   ([`Transcript`]) from runtime-only records ([`RuntimeRecord`]).
//!   Provider adapters project only the transcript into model-visible request
//!   fields. Runtime records may influence request assembly through explicit
//!   provider-specific mapping logic but are never replayed as assistant text.
//!
//! - **[`ProviderEvent`]**: Normalized streaming events. The stream
//!   termination contract is:
//!   - [`ProviderEvent::Complete`] is emitted **only** on successful stream
//!     termination.
//!   - If a provider encounters an unrecoverable error, the stream ends with
//!     [`ProviderEvent::Error`] and does **not** emit `Complete`.
//!
//! - **[`ProviderProfile`]**: Declarative provider configuration including
//!   API family, base URL, auth strategy, default headers, and quirks.
//!   All provider families (Responses, Chat Completions, Messages)
//!   honor the full profile model consistently.
//!
//! - **[`ProviderConnection`]**: Resolved provider state that implements
//!   [`Provider`]. Construct directly from a [`ProviderProfile`] and
//!   [`RuntimeConfig`], or obtain from [`ProviderRegistry::get`].
//!
//! - **[`ProviderRegistry`]**: Registry for looking up providers by slug or
//!   URL pattern, with built-in profiles for common providers.
//!
//! # Streaming
//!
//! Streaming is determined by which method you call (`infer` vs `infer_stream`),
//! not by a field on the request. There is no `stream` field on
//! [`InferenceRequest`].

pub mod connection;
pub mod error;
pub mod model;
pub mod profile;
pub mod provider;
pub mod registry;

pub(crate) mod apis;
pub(crate) mod auth;
pub(crate) mod http_client;
pub(crate) mod provider_overrides;
pub(crate) mod sse;
pub(crate) mod stream_util;

#[cfg(test)]
mod mock_provider_tests;

pub use connection::ProviderConnection;
pub use error::{ProviderError, ProviderResult};
pub use model::{
    ChoiceItem, ChoiceRequest, ChoiceSelectionMode, GenerationConfig, InferenceContext,
    InferenceRequest, Message, ProviderEvent, RuntimeRecord, ToolCall, ToolDefinition, ToolPolicy,
    Transcript, CHOICE_REQUEST_TOOL_NAME,
};
pub use profile::{
    ApiFamily, AuthStrategy, CredentialAuthConfig, CredentialKind, EndpointPurpose,
    ProviderCredential, ProviderProfile, ProviderQuirks, RuntimeConfig, RuntimeConfigSource,
};
pub use provider::{Provider, ProviderFuture};
pub use registry::ProviderRegistry;

pub mod prelude {
    pub use crate::{
        ApiFamily, AuthStrategy, ChoiceItem, ChoiceRequest, ChoiceSelectionMode, CredentialKind,
        EndpointPurpose, GenerationConfig, InferenceContext, InferenceRequest, Message, Provider,
        ProviderCredential, ProviderError, ProviderEvent, ProviderProfile, ProviderRegistry,
        ProviderResult, RuntimeConfig, RuntimeConfigSource, RuntimeRecord, ToolCall,
        ToolDefinition, ToolPolicy, Transcript, CHOICE_REQUEST_TOOL_NAME,
    };
}
