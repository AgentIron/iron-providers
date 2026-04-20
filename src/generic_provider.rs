use crate::provider::{Provider, ProviderFuture};
use crate::{
    openai::OpenAiConfig,
    profile::{ApiFamily, ProviderProfile, RuntimeConfig, RuntimeConfigSource},
    InferenceRequest, ProviderEvent, ProviderResult,
};
use futures::stream::BoxStream;

#[derive(Debug, Clone)]
pub struct GenericProvider {
    profile: ProviderProfile,
    runtime: RuntimeConfig,
}

impl GenericProvider {
    pub fn from_profile(profile: ProviderProfile, runtime: RuntimeConfig) -> ProviderResult<Self> {
        Ok(Self { profile, runtime })
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

    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub fn runtime(&self) -> &RuntimeConfig {
        &self.runtime
    }

    fn build_openai_config(&self) -> OpenAiConfig {
        let mut config = OpenAiConfig::new(self.runtime.api_key.clone())
            .with_base_url(self.profile.base_url.clone())
            .with_auth_strategy(self.profile.auth_strategy.clone())
            .with_quirks(self.profile.quirks.clone());

        for (key, value) in &self.profile.default_headers {
            config = config.with_header(key.clone(), value.clone());
        }

        config
    }
}

impl Provider for GenericProvider {
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>> {
        match self.profile.family {
            ApiFamily::OpenAiResponses => {
                let config = self.build_openai_config();
                Box::pin(async move { crate::openai::infer(&config, request).await })
            }
            ApiFamily::OpenAiChatCompletions => {
                let profile = self.profile.clone();
                let runtime = self.runtime.clone();
                Box::pin(
                    async move { crate::completions::infer(&profile, &runtime, request).await },
                )
            }
            ApiFamily::AnthropicMessages => {
                let profile = self.profile.clone();
                let runtime = self.runtime.clone();
                Box::pin(async move { crate::anthropic::infer(&profile, &runtime, request).await })
            }
        }
    }

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>> {
        match self.profile.family {
            ApiFamily::OpenAiResponses => {
                let config = self.build_openai_config();
                Box::pin(async move { crate::openai::infer_stream(&config, request).await })
            }
            ApiFamily::OpenAiChatCompletions => {
                let profile = self.profile.clone();
                let runtime = self.runtime.clone();
                Box::pin(async move {
                    crate::completions::infer_stream(&profile, &runtime, request).await
                })
            }
            ApiFamily::AnthropicMessages => {
                let profile = self.profile.clone();
                let runtime = self.runtime.clone();
                Box::pin(async move {
                    crate::anthropic::infer_stream(&profile, &runtime, request).await
                })
            }
        }
    }
}
