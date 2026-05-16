//! Provider trait definition.
//!
//! The [`Provider`] trait is the async interface consumed by `iron-core`.
//! Concrete implementations live in [`ProviderConnection`](crate::connection::ProviderConnection).

use crate::{InferenceRequest, ProviderEvent, ProviderResult};
use futures::stream::BoxStream;
use std::future::Future;
use std::pin::Pin;

/// Boxed future returned by provider operations.
pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = ProviderResult<T>> + Send + 'a>>;

/// Semantic provider boundary consumed by `iron-core`.
pub trait Provider: Send + Sync {
    /// Perform a non-streaming inference request and collect all events.
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>>;

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>>;
}
