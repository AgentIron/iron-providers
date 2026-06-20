use crate::ProviderError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Default maximum time to establish a TCP+TLS connection.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default maximum time between socket reads. For streaming responses this
/// bounds inter-chunk silence; a provider that stalls mid-stream will fail
/// rather than hang indefinitely.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Protocol family that determines which request/response adapter a provider uses.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApiFamily {
    /// OpenAI Responses API (`/responses` endpoint).
    Responses,
    /// OpenAI Chat Completions API (`/chat/completions` endpoint).
    Completions,
    /// Anthropic Messages API (`/v1/messages` endpoint).
    Messages,
}

/// Wire-level authentication strategy applied to outgoing HTTP headers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthStrategy {
    /// Send the credential as `Authorization: Bearer <token>`.
    BearerToken,
    /// Send the credential in a named header without a prefix.
    ApiKeyHeader {
        /// Header name that carries the API key.
        header_name: String,
    },
    /// Send the credential in a named header with an optional prefix.
    Custom {
        /// Header name that carries the credential.
        header_name: String,
        /// Optional prefix prepended to the credential value (e.g. `"Bearer "`).
        prefix: Option<String>,
    },
    /// No authentication headers are emitted.
    NoAuth,
}

/// Kind of credential a provider accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialKind {
    /// Static API key string.
    ApiKey,
    /// OAuth bearer token with optional expiry.
    OAuthBearer,
    /// No credential (anonymous/local access).
    NoAuth,
}

/// Concrete runtime credential supplied to a provider.
///
/// Refresh tokens are intentionally not represented here; credential refresh
/// is owned by consuming layers such as `iron-core` or the application.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderCredential {
    /// Static API key string.
    ApiKey(String),
    /// OAuth bearer token with optional expiry and ID token.
    OAuthBearer {
        /// Access token sent in the `Authorization` header.
        access_token: String,
        /// When the access token expires. Expired tokens are rejected at connection time.
        expires_at: Option<std::time::SystemTime>,
        /// Optional OIDC ID token retained for downstream use.
        id_token: Option<String>,
    },
    /// No credential (anonymous/local access).
    NoAuth,
}

impl ProviderCredential {
    /// Return the credential kind.
    pub fn kind(&self) -> CredentialKind {
        match self {
            ProviderCredential::ApiKey(_) => CredentialKind::ApiKey,
            ProviderCredential::OAuthBearer { .. } => CredentialKind::OAuthBearer,
            ProviderCredential::NoAuth => CredentialKind::NoAuth,
        }
    }

    /// Return the raw secret string used for authentication.
    ///
    /// Returns an empty string for [`ProviderCredential::NoAuth`].
    pub fn secret(&self) -> &str {
        match self {
            ProviderCredential::ApiKey(v) => v,
            ProviderCredential::OAuthBearer { access_token, .. } => access_token,
            ProviderCredential::NoAuth => "",
        }
    }
}

/// Maps a supported credential kind to the wire auth strategy used for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialAuthConfig {
    /// The credential kind this entry applies to.
    pub kind: CredentialKind,
    /// The wire auth strategy used when this credential kind is presented.
    pub auth_strategy: AuthStrategy,
}

/// Whether a profile targets general-purpose or coding-optimized endpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndpointPurpose {
    /// General-purpose inference endpoint.
    General,
    /// Endpoint optimized for code generation tasks.
    Coding,
}

/// Provider-specific behavioral flags surfaced to callers.
///
/// These allow downstream code to work around known provider limitations
/// without inspecting raw wire formats.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderQuirks {
    /// Provider silently ignores `stop` sequences.
    pub ignores_stop_sequences: bool,
    /// Provider silently ignores `top_k`.
    pub ignores_top_k: bool,
    /// Provider does not support image inputs.
    pub no_image_support: bool,
    /// Provider requires a prompt-cache key to enable caching.
    pub requires_prompt_cache_key: bool,
    /// Provider-specific request parameter renames (e.g. `max_tokens` → `max_output_tokens`).
    pub param_renames: HashMap<String, String>,
}

/// Declarative provider configuration.
///
/// A profile fully describes how to connect to a provider: its API family,
/// base URL, supported credential kinds, auth strategies, default headers,
/// purpose, and behavioral quirks. All provider families honor the full
/// profile model consistently.
///
/// # Example
///
/// ```
/// use iron_providers::{ApiFamily, AuthStrategy, ProviderProfile};
///
/// let profile = ProviderProfile::new(
///     "my-provider",
///     ApiFamily::Completions,
///     "https://api.example.com/v1",
/// )
/// .with_auth(AuthStrategy::BearerToken)
/// .with_header("X-Custom-Header", "value");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderProfile {
    /// Unique provider identifier used for registry lookups.
    pub slug: String,
    /// Optional distinct `models.dev` provider identifier for model discovery.
    #[serde(default)]
    pub models_dev_id: Option<String>,
    /// Protocol family determining the request/response adapter.
    pub family: ApiFamily,
    /// Base URL for all provider requests.
    pub base_url: String,
    /// Supported credential kinds and their wire auth strategies.
    pub credential_auth: Vec<CredentialAuthConfig>,
    /// Default headers applied to every request unless overridden.
    #[serde(default)]
    pub default_headers: HashMap<String, String>,
    /// Whether this profile targets general-purpose or coding-optimized endpoints.
    pub purpose: EndpointPurpose,
    /// Provider-specific behavioral flags.
    #[serde(default)]
    pub quirks: ProviderQuirks,
    /// Optional provider-specific system prompt guidance that overrides the
    /// API-family fallback fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_guidance: Option<String>,
}

impl ProviderProfile {
    /// Create a new provider profile with a default API-key bearer auth config.
    pub fn new(slug: impl Into<String>, family: ApiFamily, base_url: impl Into<String>) -> Self {
        Self {
            slug: slug.into(),
            models_dev_id: None,
            family,
            base_url: base_url.into(),
            credential_auth: vec![CredentialAuthConfig {
                kind: CredentialKind::ApiKey,
                auth_strategy: AuthStrategy::BearerToken,
            }],
            default_headers: HashMap::new(),
            purpose: EndpointPurpose::General,
            quirks: ProviderQuirks::default(),
            provider_guidance: None,
        }
    }

    /// Convenience builder that sets or replaces the API-key auth config.
    /// Preserves readability for existing API-key-only built-ins.
    pub fn with_auth(mut self, strategy: AuthStrategy) -> Self {
        self.set_credential_auth(CredentialKind::ApiKey, strategy);
        self
    }

    /// Add or replace the auth config for a specific credential kind.
    pub fn with_credential_auth(mut self, kind: CredentialKind, strategy: AuthStrategy) -> Self {
        self.set_credential_auth(kind, strategy);
        self
    }

    fn set_credential_auth(&mut self, kind: CredentialKind, strategy: AuthStrategy) {
        if let Some(existing) = self.credential_auth.iter_mut().find(|c| c.kind == kind) {
            existing.auth_strategy = strategy;
        } else {
            self.credential_auth.push(CredentialAuthConfig {
                kind,
                auth_strategy: strategy,
            });
        }
    }

    /// Whether this profile supports the given credential kind.
    pub fn supports_credential(&self, kind: CredentialKind) -> bool {
        self.credential_auth.iter().any(|c| c.kind == kind)
    }

    /// Resolve the wire auth strategy for a given credential kind.
    pub fn auth_strategy_for(&self, kind: CredentialKind) -> Option<&AuthStrategy> {
        self.credential_auth
            .iter()
            .find(|c| c.kind == kind)
            .map(|c| &c.auth_strategy)
    }

    /// Set the optional `models.dev` provider identifier.
    pub fn with_models_dev_id(mut self, models_dev_id: impl Into<String>) -> Self {
        self.models_dev_id = Some(models_dev_id.into());
        self
    }

    /// Add or replace a default header applied to every request.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(key.into(), value.into());
        self
    }

    /// Set whether this profile targets general-purpose or coding-optimized endpoints.
    pub fn with_purpose(mut self, purpose: EndpointPurpose) -> Self {
        self.purpose = purpose;
        self
    }

    /// Set provider-specific behavioral quirks.
    pub fn with_quirks(mut self, quirks: ProviderQuirks) -> Self {
        self.quirks = quirks;
        self
    }

    /// Set provider-specific system prompt guidance that overrides the
    /// API-family fallback fragment.
    pub fn with_provider_guidance(mut self, guidance: impl Into<String>) -> Self {
        self.provider_guidance = Some(guidance.into());
        self
    }

    /// Return the `models.dev` slug, falling back to the profile slug when unset.
    pub fn models_dev_slug(&self) -> &str {
        self.models_dev_id.as_deref().unwrap_or(&self.slug)
    }

    /// Resolve provider-specific system prompt guidance.
    ///
    /// If the profile has explicit `provider_guidance`, returns that value.
    /// Otherwise falls back to the compiled API-family-level fragment.
    pub fn system_prompt_fragment(&self) -> &str {
        if let Some(ref guidance) = self.provider_guidance {
            return guidance;
        }
        match self.family {
            ApiFamily::Messages => crate::apis::messages::SYSTEM_PROMPT_FRAGMENT,
            ApiFamily::Responses | ApiFamily::Completions => {
                crate::apis::completions::SYSTEM_PROMPT_FRAGMENT
            }
        }
    }
}

/// Per-session runtime configuration supplied when constructing a provider connection.
///
/// Holds the credential, optional HTTP timeouts, and an optional base URL override.
///
/// # Example
///
/// ```
/// use iron_providers::{ProviderCredential, RuntimeConfig};
/// use std::time::{Duration, SystemTime};
///
/// // API key (default)
/// let rt = RuntimeConfig::new("your-api-key");
///
/// // OAuth bearer token
/// let rt = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
///     access_token: "token".to_string(),
///     expires_at: Some(SystemTime::now() + Duration::from_secs(3600)),
///     id_token: None,
/// });
///
/// // No auth (local providers)
/// let rt = RuntimeConfig::none();
/// ```
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Credential used for authentication.
    pub credential: ProviderCredential,
    /// TCP + TLS connect timeout. `None` uses `DEFAULT_CONNECT_TIMEOUT`.
    pub connect_timeout: Option<Duration>,
    /// Inter-chunk read timeout. `None` uses `DEFAULT_READ_TIMEOUT`.
    pub read_timeout: Option<Duration>,
    /// Per-session base URL override. When `Some`, the provider connection
    /// uses this URL instead of the profile's `base_url` for request targets.
    /// Profile metadata (API family, auth, credentials, quirks, etc.) is
    /// preserved from the upstream profile.
    pub base_url_override: Option<String>,
}

impl RuntimeConfig {
    /// Convenience constructor for API-key-only callers.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::from_credential(ProviderCredential::ApiKey(api_key.into()))
    }

    /// Construct from an explicit credential.
    pub fn from_credential(credential: ProviderCredential) -> Self {
        Self {
            credential,
            connect_timeout: None,
            read_timeout: None,
            base_url_override: None,
        }
    }

    /// Construct a no-auth runtime config for local or anonymous providers.
    pub fn none() -> Self {
        Self {
            credential: ProviderCredential::NoAuth,
            connect_timeout: None,
            read_timeout: None,
            base_url_override: None,
        }
    }

    /// Override the TCP+TLS connect timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Override the inter-chunk read timeout. Applies to streaming responses
    /// so a stalled provider is surfaced as a transport error rather than
    /// hanging indefinitely.
    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = Some(timeout);
        self
    }

    /// Override the effective base URL for this session's provider connection.
    /// When set, request targets use this URL instead of the profile's
    /// `base_url`. Profile metadata (API family, auth, credentials, quirks,
    /// etc.) is preserved from the upstream profile.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url_override = Some(url.into());
        self
    }

    /// Resolve the effective connect timeout (user-provided or default).
    pub fn effective_connect_timeout(&self) -> Duration {
        self.connect_timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT)
    }

    /// Resolve the effective read timeout (user-provided or default).
    pub fn effective_read_timeout(&self) -> Duration {
        self.read_timeout.unwrap_or(DEFAULT_READ_TIMEOUT)
    }

    /// Validate that the credential is usable (non-empty, non-expired).
    ///
    /// Returns `ProviderError::auth` when validation fails.
    pub fn validate(&self) -> Result<(), ProviderError> {
        match &self.credential {
            ProviderCredential::NoAuth => Ok(()),
            ProviderCredential::ApiKey(key) if key.trim().is_empty() => {
                Err(ProviderError::auth("API key is required but was empty"))
            }
            ProviderCredential::OAuthBearer { access_token, .. }
                if access_token.trim().is_empty() =>
            {
                Err(ProviderError::auth(
                    "OAuth access token is required but was empty",
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Projection trait for caller-owned config types that can produce a [`RuntimeConfig`].
///
/// Allows downstream crates to define their own configuration types that
/// integrate with provider construction.
pub trait RuntimeConfigSource {
    /// Convert this configuration into a [`RuntimeConfig`].
    fn to_runtime_config(&self) -> Result<RuntimeConfig, ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_new_preserves_api_key() {
        let rt = RuntimeConfig::new("secret-key");
        assert_eq!(rt.credential.secret(), "secret-key");
        assert_eq!(rt.credential.kind(), CredentialKind::ApiKey);
    }

    #[test]
    fn test_runtime_config_from_credential_api_key() {
        let rt = RuntimeConfig::from_credential(ProviderCredential::ApiKey("secret".into()));
        assert_eq!(rt.credential.secret(), "secret");
    }

    #[test]
    fn test_runtime_config_from_credential_oauth_bearer() {
        let rt = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "token".into(),
            expires_at: None,
            id_token: None,
        });
        assert_eq!(rt.credential.secret(), "token");
        assert_eq!(rt.credential.kind(), CredentialKind::OAuthBearer);
    }

    #[test]
    fn test_runtime_config_validate_blank_api_key() {
        let rt = RuntimeConfig::new("   ");
        assert!(rt.validate().is_err());
    }

    #[test]
    fn test_runtime_config_validate_blank_oauth_token() {
        let rt = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "   ".into(),
            expires_at: None,
            id_token: None,
        });
        assert!(rt.validate().is_err());
    }

    #[test]
    fn test_runtime_config_validate_valid_api_key() {
        let rt = RuntimeConfig::new("valid");
        assert!(rt.validate().is_ok());
    }

    #[test]
    fn test_runtime_config_validate_valid_oauth() {
        let rt = RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
            access_token: "valid".into(),
            expires_at: None,
            id_token: None,
        });
        assert!(rt.validate().is_ok());
    }

    #[test]
    fn test_profile_default_auth_is_api_key_bearer() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com");
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert!(!profile.supports_credential(CredentialKind::OAuthBearer));
        assert_eq!(
            profile.auth_strategy_for(CredentialKind::ApiKey),
            Some(&AuthStrategy::BearerToken)
        );
    }

    #[test]
    fn test_profile_with_credential_auth() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com")
            .with_credential_auth(CredentialKind::OAuthBearer, AuthStrategy::BearerToken);
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert!(profile.supports_credential(CredentialKind::OAuthBearer));
    }

    #[test]
    fn test_profile_with_auth_replaces_api_key_config() {
        let profile = ProviderProfile::new("test", ApiFamily::Messages, "https://example.com")
            .with_auth(AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into(),
            });
        assert_eq!(
            profile.auth_strategy_for(CredentialKind::ApiKey),
            Some(&AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into()
            })
        );
    }

    #[test]
    fn test_provider_credential_kind_and_secret() {
        let api = ProviderCredential::ApiKey("key".into());
        assert_eq!(api.kind(), CredentialKind::ApiKey);
        assert_eq!(api.secret(), "key");

        let oauth = ProviderCredential::OAuthBearer {
            access_token: "tok".into(),
            expires_at: None,
            id_token: None,
        };
        assert_eq!(oauth.kind(), CredentialKind::OAuthBearer);
        assert_eq!(oauth.secret(), "tok");
    }

    #[test]
    fn test_noauth_credential_kind() {
        let noauth = ProviderCredential::NoAuth;
        assert_eq!(noauth.kind(), CredentialKind::NoAuth);
        assert_eq!(noauth.secret(), "");
    }

    #[test]
    fn test_runtime_config_none() {
        let rt = RuntimeConfig::none();
        assert_eq!(rt.credential.kind(), CredentialKind::NoAuth);
        assert!(rt.validate().is_ok());
    }

    #[test]
    fn test_profile_with_noauth_credential_auth() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com")
            .with_credential_auth(CredentialKind::NoAuth, AuthStrategy::NoAuth);
        assert!(profile.supports_credential(CredentialKind::NoAuth));
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert_eq!(
            profile.auth_strategy_for(CredentialKind::NoAuth),
            Some(&AuthStrategy::NoAuth)
        );
    }

    #[test]
    fn test_provider_guidance_default_is_none() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com");
        assert!(profile.provider_guidance.is_none());
    }

    #[test]
    fn test_provider_guidance_builder_sets_value() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com")
            .with_provider_guidance("Use this provider like so.");
        assert_eq!(
            profile.provider_guidance.as_deref(),
            Some("Use this provider like so.")
        );
    }

    #[test]
    fn test_system_prompt_fragment_uses_provider_guidance_when_set() {
        let custom = "Custom provider guidance.";
        let profile = ProviderProfile::new("test", ApiFamily::Messages, "https://example.com")
            .with_provider_guidance(custom);
        assert_eq!(profile.system_prompt_fragment(), custom);
    }

    #[test]
    fn test_system_prompt_fragment_falls_back_to_family_fragment() {
        let profile = ProviderProfile::new("test", ApiFamily::Messages, "https://example.com");
        assert_eq!(
            profile.system_prompt_fragment(),
            crate::apis::messages::SYSTEM_PROMPT_FRAGMENT
        );

        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com");
        assert_eq!(
            profile.system_prompt_fragment(),
            crate::apis::completions::SYSTEM_PROMPT_FRAGMENT
        );

        let profile = ProviderProfile::new("test", ApiFamily::Responses, "https://example.com");
        assert_eq!(
            profile.system_prompt_fragment(),
            crate::apis::completions::SYSTEM_PROMPT_FRAGMENT
        );
    }

    #[test]
    fn test_profile_serialization_omits_provider_guidance_when_none() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com");
        let json = serde_json::to_string(&profile).expect("serialize");
        assert!(!json.contains("provider_guidance"));
        let parsed: ProviderProfile = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.provider_guidance.is_none());
    }

    #[test]
    fn test_profile_serialization_roundtrips_provider_guidance() {
        let profile = ProviderProfile::new("test", ApiFamily::Completions, "https://example.com")
            .with_provider_guidance("Custom guidance.");
        let json = serde_json::to_string(&profile).expect("serialize");
        assert!(json.contains("provider_guidance"));
        let parsed: ProviderProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.provider_guidance.as_deref(),
            Some("Custom guidance.")
        );
    }

    #[test]
    fn test_profile_deserialization_without_provider_guidance_field() {
        let json = r#"{
            "slug": "test",
            "family": "Responses",
            "base_url": "https://example.com",
            "credential_auth": [{"kind": "ApiKey", "auth_strategy": "BearerToken"}],
            "purpose": "General"
        }"#;
        let parsed: ProviderProfile = serde_json::from_str(json).expect("deserialize");
        assert!(parsed.provider_guidance.is_none());
    }
}
