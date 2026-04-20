//! Shared HTTP client construction for reqwest-based provider adapters.
//!
//! The Chat Completions, Anthropic, and OpenAI Responses adapters all need to
//! build a `reqwest::Client` with the same auth + header + timeout logic. This
//! module centralizes that so there is one place to change auth behavior, apply
//! timeouts, or adjust fail-fast validation.

use crate::profile::AuthStrategy;
use crate::{ProviderError, ProviderResult};
use reqwest::Client;
use std::collections::HashMap;
use std::time::Duration;

/// Parameters for constructing an HTTP client.
///
/// `context` is a short tag included in configuration error messages so
/// failures point back at the originating profile or config (for example
/// `"profile 'zai'"` or `"OpenAI client"`).
pub(crate) struct HttpClientParams<'a> {
    pub context: &'a str,
    pub api_key: &'a str,
    pub auth_strategy: &'a AuthStrategy,
    pub default_headers: &'a HashMap<String, String>,
    pub extra_headers: &'a [(&'a str, &'a str)],
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
}

/// Build a `reqwest::Client` with `Content-Type: application/json`, the
/// configured auth header, any caller-supplied fixed headers, and the
/// profile's default headers applied. Fails fast on invalid header names or
/// values so misconfiguration is caught at client construction rather than
/// surfacing as an opaque transport error on first request.
pub(crate) fn build_http_client(params: HttpClientParams<'_>) -> ProviderResult<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );

    for (name, value) in params.extra_headers {
        let hk = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid fixed header name '{}' for {}: {}",
                name, params.context, e
            ))
        })?;
        let hv = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid fixed header value for '{}' on {}: {}",
                name, params.context, e
            ))
        })?;
        headers.insert(hk, hv);
    }

    match params.auth_strategy {
        AuthStrategy::BearerToken => {
            let val = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", params.api_key))
                .map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid bearer token value for {}: {}",
                        params.context, e
                    ))
                })?;
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
        AuthStrategy::ApiKeyHeader { header_name } => {
            let key =
                reqwest::header::HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid header name '{}' for {}: {}",
                        header_name, params.context, e
                    ))
                })?;
            let val = reqwest::header::HeaderValue::from_str(params.api_key).map_err(|e| {
                ProviderError::invalid_request(format!(
                    "Invalid API key value for {}: {}",
                    params.context, e
                ))
            })?;
            headers.insert(key, val);
        }
        AuthStrategy::Custom {
            header_name,
            prefix,
        } => {
            let value = match prefix {
                Some(p) => format!("{} {}", p, params.api_key),
                None => params.api_key.to_string(),
            };
            let key =
                reqwest::header::HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid custom header name '{}' for {}: {}",
                        header_name, params.context, e
                    ))
                })?;
            let val = reqwest::header::HeaderValue::from_str(&value).map_err(|e| {
                ProviderError::invalid_request(format!(
                    "Invalid custom header value for {}: {}",
                    params.context, e
                ))
            })?;
            headers.insert(key, val);
        }
    }

    for (key, value) in params.default_headers {
        let hk = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid default header name '{}' for {}: {}",
                key, params.context, e
            ))
        })?;
        let hv = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid default header value for '{}' on {}: {}",
                key, params.context, e
            ))
        })?;
        headers.insert(hk, hv);
    }

    Client::builder()
        .default_headers(headers)
        .connect_timeout(params.connect_timeout)
        .read_timeout(params.read_timeout)
        .build()
        .map_err(|e| {
            ProviderError::general(format!(
                "Failed to build HTTP client for {}: {}",
                params.context, e
            ))
        })
}
