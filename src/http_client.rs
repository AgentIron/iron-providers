//! Pure HTTP transport construction
//!
//! Receives a fully-composed `HeaderMap` and timeout values, then builds a
//! `reqwest::Client`. This module has no knowledge of credentials, auth
//! strategies, or provider-specific behavior.

use crate::{ProviderError, ProviderResult};
use reqwest::Client;
use std::time::Duration;

/// Build a `reqwest::Client` from finalized headers and timeouts.
///
/// Fails fast on invalid header names or values so misconfiguration is caught
/// at client construction rather than surfacing as an opaque transport error
/// on first request.
pub(crate) fn build_http_client(
    headers: reqwest::header::HeaderMap,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> ProviderResult<Client> {
    Client::builder()
        .default_headers(headers)
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .build()
        .map_err(|e| ProviderError::general(format!("Failed to build HTTP client: {}", e)))
}
