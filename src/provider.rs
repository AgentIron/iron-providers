//! Provider traits and concrete provider adapters.

use crate::openai::{build_client, OpenAiConfigSource};
use crate::{openai, InferenceRequest, OpenAiConfig, ProviderEvent, ProviderResult};
use async_openai::Client as OpenAiClient;
use futures::stream::BoxStream;
use std::future::Future;
use std::pin::Pin;

/// Boxed future returned by provider operations.
pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = ProviderResult<T>> + Send + 'a>>;

/// Semantic provider boundary consumed by `iron-core`.
pub trait Provider: Send + Sync {
    /// Perform a non-streaming inference request and collect all events.
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>>;

    /// Perform a streaming inference request.
    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>>;
}

/// OpenAI-backed provider implementation.
///
/// The HTTP client is built once at construction time and reused across all
/// inference calls so that TCP connections and TLS sessions are shared.
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    config: OpenAiConfig,
    client: OpenAiClient<async_openai::config::OpenAIConfig>,
}

impl OpenAiProvider {
    /// Create a provider from validated OpenAI configuration.
    pub fn new(config: OpenAiConfig) -> ProviderResult<Self> {
        config.validate()?;
        let client = build_client(&config)?;
        Ok(Self { config, client })
    }

    /// Borrow the OpenAI configuration snapshot used by this provider.
    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }

    /// Create an `OpenAiProvider` from a caller-owned config source.
    ///
    /// The source is projected into a validated `OpenAiConfig` snapshot.
    pub fn from_source<S: OpenAiConfigSource>(source: &S) -> ProviderResult<Self> {
        let config = source.to_openai_config()?;
        Self::new(config)
    }
}

impl Provider for OpenAiProvider {
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>> {
        let client = self.client.clone();
        Box::pin(async move { openai::infer(&client, request).await })
    }

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>> {
        let client = self.client.clone();
        Box::pin(async move { openai::infer_stream(&client, request).await })
    }
}
