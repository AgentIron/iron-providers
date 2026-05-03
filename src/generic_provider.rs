use crate::http_client::{build_http_client, HttpClientParams};
use crate::provider::{Provider, ProviderFuture};
use crate::{
    openai::OpenAiConfig,
    profile::{ApiFamily, ProviderProfile, RuntimeConfig, RuntimeConfigSource},
    InferenceRequest, ProviderError, ProviderEvent, ProviderResult,
};
use async_openai::{config::OpenAIConfig, Client as OpenAiClient};
use futures::stream::BoxStream;
use std::sync::Arc;

/// Profile-driven provider that dispatches to the correct adapter based on
/// [`ApiFamily`].
///
/// HTTP clients are built once at construction time and reused across all
/// inference calls so that TCP connections, TLS sessions, and HTTP/2
/// multiplexing are shared.
#[derive(Debug, Clone)]
pub struct GenericProvider {
    profile: Arc<ProviderProfile>,
    runtime: RuntimeConfig,
    /// Shared `reqwest::Client` used by Chat Completions and Anthropic
    /// adapters. Also used as the HTTP transport for the OpenAI Responses
    /// adapter (via `async_openai`).
    http_client: reqwest::Client,
    /// Pre-built `async-openai` client for the OpenAI Responses adapter.
    /// `None` when the profile family is not `OpenAiResponses`.
    openai_client: Option<OpenAiClient<OpenAIConfig>>,
}

impl GenericProvider {
    pub fn from_profile(profile: ProviderProfile, runtime: RuntimeConfig) -> ProviderResult<Self> {
        Self::build(Arc::new(profile), runtime)
    }

    /// Construct from an already-shared `Arc<ProviderProfile>`, avoiding an
    /// extra clone of the profile data when the caller (typically the
    /// registry) already holds a shared reference.
    pub(crate) fn from_arc(
        profile: Arc<ProviderProfile>,
        runtime: RuntimeConfig,
    ) -> ProviderResult<Self> {
        Self::build(profile, runtime)
    }

    /// Create a `GenericProvider` from a caller-owned config source.
    ///
    /// The source is projected into a validated `RuntimeConfig` snapshot.
    pub fn from_source<S: RuntimeConfigSource>(
        profile: ProviderProfile,
        source: &S,
    ) -> ProviderResult<Self> {
        let runtime = source.to_runtime_config()?;
        runtime.validate()?;
        Self::from_profile(profile, runtime)
    }

    /// Shared construction logic that builds the HTTP client(s) once.
    fn build(profile: Arc<ProviderProfile>, runtime: RuntimeConfig) -> ProviderResult<Self> {
        runtime.validate()?;

        let context = format!("profile '{}'", profile.slug);

        // Validate credential kind is supported by this profile.
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

        let http_client = build_http_client(HttpClientParams {
            context: &context,
            credential: &runtime.credential,
            auth_strategy: &auth_strategy,
            default_headers: &profile.default_headers,
            extra_headers: &[],
            connect_timeout: runtime.effective_connect_timeout(),
            read_timeout: runtime.effective_read_timeout(),
        })?;

        let openai_client = if profile.family == ApiFamily::OpenAiResponses {
            let config = Self::build_openai_config(&profile, &runtime);
            let mut openai_config = OpenAIConfig::default().with_api_key(config.api_key);
            if let Some(ref base_url) = config.base_url {
                openai_config = openai_config.with_api_base(base_url);
            }
            Some(OpenAiClient::with_config(openai_config).with_http_client(http_client.clone()))
        } else {
            None
        };

        Ok(Self {
            profile,
            runtime,
            http_client,
            openai_client,
        })
    }

    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub fn runtime(&self) -> &RuntimeConfig {
        &self.runtime
    }

    fn build_openai_config(profile: &ProviderProfile, runtime: &RuntimeConfig) -> OpenAiConfig {
        let mut config = OpenAiConfig::new(runtime.credential.secret().to_string())
            .with_base_url(profile.base_url.clone())
            .with_auth_strategy(
                profile
                    .auth_strategy_for(crate::profile::CredentialKind::ApiKey)
                    .cloned()
                    .unwrap_or(crate::profile::AuthStrategy::BearerToken),
            )
            .with_quirks(profile.quirks.clone());

        if let Some(timeout) = runtime.connect_timeout {
            config = config.with_connect_timeout(timeout);
        }
        if let Some(timeout) = runtime.read_timeout {
            config = config.with_read_timeout(timeout);
        }

        for (key, value) in &profile.default_headers {
            config = config.with_header(key.clone(), value.clone());
        }

        config
    }
}

impl Provider for GenericProvider {
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>> {
        match self.profile.family {
            ApiFamily::OpenAiResponses => {
                let client = self
                    .openai_client
                    .as_ref()
                    .expect("openai_client must be Some when family is OpenAiResponses");
                // async_openai::Client is cheaply Clone (Arc internally) but
                // we borrow here since the future borrows &self.
                let client = client.clone();
                Box::pin(async move { crate::openai::infer(&client, request).await })
            }
            ApiFamily::OpenAiChatCompletions => {
                let client = self.http_client.clone();
                let profile = Arc::clone(&self.profile);
                Box::pin(async move { crate::completions::infer(client, &profile, request).await })
            }
            ApiFamily::AnthropicMessages => {
                let client = self.http_client.clone();
                let profile = Arc::clone(&self.profile);
                Box::pin(async move { crate::anthropic::infer(client, &profile, request).await })
            }
            ApiFamily::CodexResponses => {
                let client = self.http_client.clone();
                let runtime = self.runtime.clone();
                let profile = Arc::clone(&self.profile);
                Box::pin(
                    async move { crate::codex::infer(client, &profile, &runtime, request).await },
                )
            }
        }
    }

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>> {
        match self.profile.family {
            ApiFamily::OpenAiResponses => {
                let client = self
                    .openai_client
                    .as_ref()
                    .expect("openai_client must be Some when family is OpenAiResponses");
                let client = client.clone();
                Box::pin(async move { crate::openai::infer_stream(&client, request).await })
            }
            ApiFamily::OpenAiChatCompletions => {
                let client = self.http_client.clone();
                let profile = Arc::clone(&self.profile);
                Box::pin(async move {
                    crate::completions::infer_stream(client, &profile, request).await
                })
            }
            ApiFamily::AnthropicMessages => {
                let client = self.http_client.clone();
                let profile = Arc::clone(&self.profile);
                Box::pin(
                    async move { crate::anthropic::infer_stream(client, &profile, request).await },
                )
            }
            ApiFamily::CodexResponses => {
                let client = self.http_client.clone();
                let runtime = self.runtime.clone();
                let profile = Arc::clone(&self.profile);
                Box::pin(async move {
                    crate::codex::infer_stream(client, &profile, &runtime, request).await
                })
            }
        }
    }
}
