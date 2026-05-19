//! Provider connection — resolved provider state
//!
//! `ProviderConnection` encapsulates all resolved state for a provider:
//! validated credentials, composed headers, HTTP client, and API adapter
//! selection. It is the only concrete connection type constructed by the
//! registry and is available for direct construction.

use crate::auth::auth_headers;
use crate::provider_overrides::{override_headers, resolve_overrides, ProviderOverrides};
use crate::{
    http_client::build_http_client,
    profile::{ApiFamily, ProviderProfile, RuntimeConfig},
    provider::{Provider, ProviderFuture},
    InferenceRequest, ProviderError, ProviderEvent, ProviderResult,
};
use futures::stream::BoxStream;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Resolved provider connection that implements [`Provider`].
///
/// Construct via [`ProviderConnection::from_profile`] or obtain from
/// [`ProviderRegistry::get`](crate::ProviderRegistry::get).
#[derive(Debug, Clone)]
pub struct ProviderConnection {
    profile: Arc<ProviderProfile>,
    overrides: ProviderOverrides,
    http_client: reqwest::Client,
}

impl ProviderConnection {
    /// Construct a provider connection from a profile and runtime config.
    ///
    /// Validates credentials, resolves auth headers and provider overrides,
    /// composes protected headers, and builds the HTTP client.
    pub fn from_profile(profile: ProviderProfile, runtime: RuntimeConfig) -> ProviderResult<Self> {
        Self::build(Arc::new(profile), runtime)
    }

    /// Construct from an already-shared `Arc<ProviderProfile>`.
    pub(crate) fn from_arc(
        profile: Arc<ProviderProfile>,
        runtime: RuntimeConfig,
    ) -> ProviderResult<Self> {
        Self::build(profile, runtime)
    }

    fn build(profile: Arc<ProviderProfile>, runtime: RuntimeConfig) -> ProviderResult<Self> {
        runtime.validate()?;

        let context = format!("profile '{}'", profile.slug);

        // Validate credential kind is supported.
        let kind = runtime.credential.kind();
        let auth_strategy = profile
            .auth_strategy_for(kind)
            .ok_or_else(|| {
                ProviderError::auth(format!(
                    "Provider '{}' does not support {:?} credentials",
                    profile.slug, kind
                ))
            })?
            .clone();

        // Validate OAuth expiry.
        if let crate::profile::ProviderCredential::OAuthBearer {
            expires_at: Some(exp),
            ..
        } = &runtime.credential
        {
            if std::time::SystemTime::now() >= *exp {
                return Err(ProviderError::auth(format!(
                    "OAuth credential for '{}' has expired",
                    profile.slug
                )));
            }
        }

        // Resolve auth headers.
        let auth_h = auth_headers(&runtime.credential, &auth_strategy, &context)?;

        // Resolve provider overrides.
        let overrides = resolve_overrides(&profile, &runtime);
        let override_h = override_headers(&overrides)?;

        // Compose final headers with collision detection.
        let final_headers =
            compose_headers(auth_h, override_h, &profile.default_headers, &context)?;

        // Build HTTP client.
        let http_client = build_http_client(
            final_headers,
            runtime.effective_connect_timeout(),
            runtime.effective_read_timeout(),
        )?;

        Ok(Self {
            profile,
            overrides,
            http_client,
        })
    }

    /// Borrow the provider profile.
    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }
}

/// Compose final headers from auth headers, override headers, and profile
/// default headers. Protected headers (auth, protocol-required, and override
/// headers) cannot be silently overridden by profile defaults.
fn compose_headers(
    auth_headers: HeaderMap,
    override_headers: HeaderMap,
    default_headers: &std::collections::HashMap<String, String>,
    context: &str,
) -> ProviderResult<HeaderMap> {
    let mut final_headers = HeaderMap::new();

    // Insert Content-Type as a protocol-required header.
    final_headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );

    // Collect protected header names.
    let mut protected: std::collections::HashSet<String> = std::collections::HashSet::new();
    protected.insert(reqwest::header::CONTENT_TYPE.as_str().to_lowercase());

    // Add auth headers (protected).
    for (key, value) in &auth_headers {
        protected.insert(key.as_str().to_lowercase());
        final_headers.insert(key.clone(), value.clone());
    }

    // Add override headers (protected).
    for (key, value) in &override_headers {
        protected.insert(key.as_str().to_lowercase());
        final_headers.insert(key.clone(), value.clone());
    }

    // Apply profile default headers, failing on collision with protected.
    for (key, value) in default_headers {
        let hk = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid default header name '{}' for {}: {}",
                key, context, e
            ))
        })?;
        let hv = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
            ProviderError::invalid_request(format!(
                "Invalid default header value for '{}' on {}: {}",
                key, context, e
            ))
        })?;

        if protected.contains(hk.as_str()) {
            return Err(ProviderError::invalid_request(format!(
                "Profile default header '{}' collides with protected header for {}",
                key, context
            )));
        }

        final_headers.insert(hk, hv);
    }

    Ok(final_headers)
}

impl Provider for ProviderConnection {
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>> {
        let client = self.http_client.clone();
        let profile = Arc::clone(&self.profile);
        let overrides = self.overrides.clone();

        Box::pin(async move {
            match profile.family {
                ApiFamily::Messages => {
                    crate::apis::messages::infer(client, &profile, request).await
                }
                ApiFamily::Completions => {
                    crate::apis::completions::infer(client, &profile, request).await
                }
                ApiFamily::Responses => {
                    crate::apis::responses::infer(client, &profile, &overrides, request).await
                }
            }
        })
    }

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>> {
        let client = self.http_client.clone();
        let profile = Arc::clone(&self.profile);
        let overrides = self.overrides.clone();

        Box::pin(async move {
            match profile.family {
                ApiFamily::Messages => {
                    crate::apis::messages::infer_stream(client, &profile, request).await
                }
                ApiFamily::Completions => {
                    crate::apis::completions::infer_stream(client, &profile, request).await
                }
                ApiFamily::Responses => {
                    crate::apis::responses::infer_stream(client, &profile, &overrides, request)
                        .await
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protected_header_collision_fails() {
        let mut auth = HeaderMap::new();
        auth.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer secret"),
        );

        let mut defaults = std::collections::HashMap::new();
        defaults.insert("authorization".to_string(), "new".to_string());

        let result = compose_headers(auth, HeaderMap::new(), &defaults, "test");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("collides with protected header"));
    }

    #[test]
    fn test_content_type_collision_fails() {
        let mut defaults = std::collections::HashMap::new();
        defaults.insert("content-type".to_string(), "text/plain".to_string());

        let result = compose_headers(HeaderMap::new(), HeaderMap::new(), &defaults, "test");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("collides with protected header"));
    }

    #[test]
    fn test_non_protected_header_allowed() {
        let mut defaults = std::collections::HashMap::new();
        defaults.insert("x-custom".to_string(), "value".to_string());

        let result =
            compose_headers(HeaderMap::new(), HeaderMap::new(), &defaults, "test").unwrap();
        assert_eq!(result.get("x-custom").unwrap(), "value");
    }

    #[test]
    fn test_override_header_is_protected() {
        let mut overrides = HeaderMap::new();
        overrides.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static("2023-06-01"),
        );

        let mut defaults = std::collections::HashMap::new();
        defaults.insert("anthropic-version".to_string(), "2024-01".to_string());

        let result = compose_headers(HeaderMap::new(), overrides, &defaults, "test");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("collides with protected header"));
    }
}
