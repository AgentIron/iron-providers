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
    pub auth_strategy: AuthStrategy,
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
            auth_strategy: AuthStrategy::BearerToken,
            default_headers: HashMap::new(),
            purpose: EndpointPurpose::General,
            quirks: ProviderQuirks::default(),
        }
    }

    pub fn with_auth(mut self, strategy: AuthStrategy) -> Self {
        self.auth_strategy = strategy;
        self
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub api_key: String,
    /// TCP + TLS connect timeout. `None` uses `DEFAULT_CONNECT_TIMEOUT`.
    pub connect_timeout: Option<Duration>,
    /// Inter-chunk read timeout. `None` uses `DEFAULT_READ_TIMEOUT`.
    pub read_timeout: Option<Duration>,
}

impl RuntimeConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
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
        if self.api_key.trim().is_empty() {
            return Err(ProviderError::invalid_request(
                "RuntimeConfig API key is required but was empty",
            ));
        }
        Ok(())
    }
}

/// Projection trait for caller-owned config types that can produce a `RuntimeConfig`.
pub trait RuntimeConfigSource {
    fn to_runtime_config(&self) -> Result<RuntimeConfig, ProviderError>;
}
