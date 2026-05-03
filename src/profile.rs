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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApiFamily {
    OpenAiResponses,
    OpenAiChatCompletions,
    AnthropicMessages,
    CodexResponses,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthStrategy {
    BearerToken,
    ApiKeyHeader {
        header_name: String,
    },
    Custom {
        header_name: String,
        prefix: Option<String>,
    },
}

/// Kind of credential a provider accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialKind {
    ApiKey,
    OAuthBearer,
}

/// Concrete runtime credential supplied to a provider.
///
/// Refresh tokens are intentionally not represented here; credential refresh
/// is owned by consuming layers such as `iron-core` or the application.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderCredential {
    ApiKey(String),
    OAuthBearer {
        access_token: String,
        expires_at: Option<std::time::SystemTime>,
        id_token: Option<String>,
    },
}

impl ProviderCredential {
    /// Return the credential kind.
    pub fn kind(&self) -> CredentialKind {
        match self {
            ProviderCredential::ApiKey(_) => CredentialKind::ApiKey,
            ProviderCredential::OAuthBearer { .. } => CredentialKind::OAuthBearer,
        }
    }

    /// Return the raw secret value suitable for placement on the wire.
    pub fn secret(&self) -> &str {
        match self {
            ProviderCredential::ApiKey(v) => v,
            ProviderCredential::OAuthBearer { access_token, .. } => access_token,
        }
    }
}

/// Maps a supported credential kind to the wire auth strategy used for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialAuthConfig {
    pub kind: CredentialKind,
    pub auth_strategy: AuthStrategy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndpointPurpose {
    General,
    Coding,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderQuirks {
    pub ignores_stop_sequences: bool,
    pub ignores_top_k: bool,
    pub no_image_support: bool,
    pub requires_prompt_cache_key: bool,
    pub param_renames: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderProfile {
    pub slug: String,
    #[serde(default)]
    pub models_dev_id: Option<String>,
    pub family: ApiFamily,
    pub base_url: String,
    pub credential_auth: Vec<CredentialAuthConfig>,
    #[serde(default)]
    pub default_headers: HashMap<String, String>,
    pub purpose: EndpointPurpose,
    #[serde(default)]
    pub quirks: ProviderQuirks,
}

impl ProviderProfile {
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

    pub fn with_models_dev_id(mut self, models_dev_id: impl Into<String>) -> Self {
        self.models_dev_id = Some(models_dev_id.into());
        self
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(key.into(), value.into());
        self
    }

    pub fn with_purpose(mut self, purpose: EndpointPurpose) -> Self {
        self.purpose = purpose;
        self
    }

    pub fn with_quirks(mut self, quirks: ProviderQuirks) -> Self {
        self.quirks = quirks;
        self
    }

    pub fn models_dev_slug(&self) -> &str {
        self.models_dev_id.as_deref().unwrap_or(&self.slug)
    }

    pub fn system_prompt_fragment(&self) -> &'static str {
        match self.family {
            ApiFamily::AnthropicMessages => crate::anthropic::SYSTEM_PROMPT_FRAGMENT,
            ApiFamily::OpenAiResponses | ApiFamily::OpenAiChatCompletions => {
                crate::openai::SYSTEM_PROMPT_FRAGMENT
            }
            ApiFamily::CodexResponses => crate::openai::SYSTEM_PROMPT_FRAGMENT,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub credential: ProviderCredential,
    /// TCP + TLS connect timeout. `None` uses `DEFAULT_CONNECT_TIMEOUT`.
    pub connect_timeout: Option<Duration>,
    /// Inter-chunk read timeout. `None` uses `DEFAULT_READ_TIMEOUT`.
    pub read_timeout: Option<Duration>,
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

    /// Resolve the effective connect timeout (user-provided or default).
    pub fn effective_connect_timeout(&self) -> Duration {
        self.connect_timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT)
    }

    /// Resolve the effective read timeout (user-provided or default).
    pub fn effective_read_timeout(&self) -> Duration {
        self.read_timeout.unwrap_or(DEFAULT_READ_TIMEOUT)
    }

    pub fn validate(&self) -> Result<(), ProviderError> {
        match &self.credential {
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

/// Projection trait for caller-owned config types that can produce a `RuntimeConfig`.
pub trait RuntimeConfigSource {
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
        let profile = ProviderProfile::new(
            "test",
            ApiFamily::OpenAiChatCompletions,
            "https://example.com",
        );
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert!(!profile.supports_credential(CredentialKind::OAuthBearer));
        assert_eq!(
            profile.auth_strategy_for(CredentialKind::ApiKey),
            Some(&AuthStrategy::BearerToken)
        );
    }

    #[test]
    fn test_profile_with_credential_auth() {
        let profile = ProviderProfile::new(
            "test",
            ApiFamily::OpenAiChatCompletions,
            "https://example.com",
        )
        .with_credential_auth(CredentialKind::OAuthBearer, AuthStrategy::BearerToken);
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert!(profile.supports_credential(CredentialKind::OAuthBearer));
    }

    #[test]
    fn test_profile_with_auth_replaces_api_key_config() {
        let profile =
            ProviderProfile::new("test", ApiFamily::AnthropicMessages, "https://example.com")
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
}
