//! Provider traits and concrete provider adapters.

use crate::openai::OpenAiConfigSource;
use crate::{openai, InferenceRequest, OpenAiConfig, ProviderEvent, ProviderResult};
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
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    config: OpenAiConfig,
}

impl OpenAiProvider {
    /// Create a provider from validated OpenAI configuration.
    pub fn new(config: OpenAiConfig) -> Self {
        Self { config }
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
        config.validate()?;
        Ok(Self { config })
    }
}

impl Provider for OpenAiProvider {
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>> {
        Box::pin(openai::infer(&self.config, request))
    }

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>> {
        Box::pin(openai::infer_stream(&self.config, request))
    }
}
