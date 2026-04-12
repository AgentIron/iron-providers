use crate::{
    profile::{ApiFamily, AuthStrategy, EndpointPurpose, ProviderProfile, RuntimeConfig},
    provider::Provider,
    ProviderError, ProviderResult,
};
use std::collections::HashMap;

pub struct ProviderRegistry {
    profiles: HashMap<String, ProviderProfile>,
    url_patterns: Vec<(String, ProviderProfile)>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            url_patterns: Vec::new(),
        }
    }

    pub fn register(&mut self, profile: ProviderProfile) {
        let key = profile.slug.to_lowercase();
        self.profiles.insert(key, profile);
    }

    pub fn register_by_url_pattern(
        &mut self,
        url_prefix: impl Into<String>,
        profile: ProviderProfile,
    ) {
        let key = profile.slug.to_lowercase();
        self.profiles.insert(key, profile.clone());
        self.url_patterns.push((url_prefix.into(), profile));
    }

    pub fn get(
        &self,
        provider_name: &str,
        runtime_config: RuntimeConfig,
    ) -> ProviderResult<Box<dyn Provider>> {
        let key = provider_name.to_lowercase();
        let profile = self
            .profiles
            .get(&key)
            .ok_or_else(|| {
                let available: Vec<&str> = self.profiles.keys().map(|s| s.as_str()).collect();
                ProviderError::general(format!(
                    "Unknown provider '{}'. Available: {:?}",
                    provider_name, available
                ))
            })?
            .clone();

        crate::generic_provider::GenericProvider::from_profile(profile, runtime_config)
            .map(|p| Box::new(p) as Box<dyn Provider>)
    }

    pub fn resolve_by_url(&self, url: &str) -> Option<&ProviderProfile> {
        for (prefix, profile) in &self.url_patterns {
            if url.starts_with(prefix) {
                if let Some(p) = self.profiles.get(&profile.slug.to_lowercase()) {
                    return Some(p);
                }
            }
        }
        None
    }

    pub fn resolve_by_models_dev_id(&self, models_dev_id: &str) -> Option<&ProviderProfile> {
        self.profiles.values().find(|profile| {
            profile
                .models_dev_slug()
                .eq_ignore_ascii_case(models_dev_id)
        })
    }

    pub fn slugs(&self) -> Vec<&str> {
        let mut slugs: Vec<&str> = self.profiles.keys().map(|s| s.as_str()).collect();
        slugs.sort();
        slugs
    }

    pub fn register_builtins(&mut self) {
        self.register(
            ProviderProfile::new(
                "anthropic",
                ApiFamily::AnthropicMessages,
                "https://api.anthropic.com",
            )
            .with_auth(AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into(),
            }),
        );

        self.register(
            ProviderProfile::new(
                "minimax",
                ApiFamily::AnthropicMessages,
                "https://api.minimax.io/anthropic",
            )
            .with_auth(AuthStrategy::BearerToken),
        );

        self.register(
            ProviderProfile::new(
                "minimax-code",
                ApiFamily::AnthropicMessages,
                "https://api.minimax.io/anthropic",
            )
            .with_models_dev_id("minimax-coding-plan")
            .with_auth(AuthStrategy::BearerToken)
            .with_purpose(EndpointPurpose::Coding),
        );

        self.register(ProviderProfile::new(
            "zai",
            ApiFamily::OpenAiChatCompletions,
            "https://api.z.ai/api/paas/v4",
        ));

        self.register(
            ProviderProfile::new(
                "zai-code",
                ApiFamily::OpenAiChatCompletions,
                "https://api.z.ai/api/coding/paas/v4",
            )
            .with_models_dev_id("zai-coding-plan")
            .with_purpose(EndpointPurpose::Coding),
        );

        self.register(
            ProviderProfile::new(
                "kimi",
                ApiFamily::OpenAiChatCompletions,
                "https://api.moonshot.ai/v1",
            )
            .with_models_dev_id("moonshotai"),
        );

        self.register(
            ProviderProfile::new(
                "kimi-code",
                ApiFamily::AnthropicMessages,
                "https://api.kimi.com/coding/v1",
            )
            .with_models_dev_id("kimi-for-coding")
            .with_auth(AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into(),
            })
            .with_purpose(EndpointPurpose::Coding),
        );

        self.register(
            ProviderProfile::new(
                "openrouter",
                ApiFamily::OpenAiChatCompletions,
                "https://openrouter.ai/api/v1",
            )
            .with_header(
                "HTTP-Referer",
                "https://github.com/anomalyco/iron-providers",
            )
            .with_header("X-OpenRouter-Title", "IronAgent"),
        );

        self.register(ProviderProfile::new(
            "requesty",
            ApiFamily::OpenAiChatCompletions,
            "https://api.requesty.ai/v1",
        ));
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register_builtins();
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let mut registry = ProviderRegistry::new();
        registry.register_builtins();

        let result = registry.get("minimax", RuntimeConfig::new("test-key"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let mut registry = ProviderRegistry::new();
        registry.register_builtins();

        let result = registry.get("OPENROUTER", RuntimeConfig::new("test-key"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_unknown_provider_error() {
        let registry = ProviderRegistry::default();
        let result = registry.get("unknown-provider", RuntimeConfig::new("test-key"));
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = e.to_string();
            assert!(msg.contains("Unknown provider"));
            assert!(msg.contains("unknown-provider"));
        }
    }

    #[test]
    fn test_builtins_registered() {
        let registry = ProviderRegistry::default();
        let slugs = registry.slugs();
        assert!(slugs.contains(&"anthropic"));
        assert!(slugs.contains(&"minimax"));
        assert!(slugs.contains(&"minimax-code"));
        assert!(slugs.contains(&"zai"));
        assert!(slugs.contains(&"zai-code"));
        assert!(slugs.contains(&"kimi"));
        assert!(slugs.contains(&"kimi-code"));
        assert!(slugs.contains(&"openrouter"));
        assert!(slugs.contains(&"requesty"));
    }

    #[test]
    fn test_minimax_code_is_distinct() {
        let registry = ProviderRegistry::default();

        let minimax = registry.profiles.get("minimax").expect("minimax");
        let minimax_code = registry.profiles.get("minimax-code").expect("minimax-code");

        assert_eq!(minimax.purpose, EndpointPurpose::General);
        assert_eq!(minimax_code.purpose, EndpointPurpose::Coding);
    }

    #[test]
    fn test_models_dev_id_resolution() {
        let registry = ProviderRegistry::default();

        let kimi = registry
            .resolve_by_models_dev_id("moonshotai")
            .expect("moonshotai");
        let kimi_code = registry
            .resolve_by_models_dev_id("kimi-for-coding")
            .expect("kimi-for-coding");
        let zai = registry
            .resolve_by_models_dev_id("zai")
            .expect("zai falls back to slug");

        assert_eq!(kimi.slug, "kimi");
        assert_eq!(kimi_code.slug, "kimi-code");
        assert_eq!(kimi_code.family, ApiFamily::AnthropicMessages);
        assert_eq!(
            kimi_code.auth_strategy,
            AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into()
            }
        );
        assert_eq!(kimi_code.purpose, EndpointPurpose::Coding);
        assert_eq!(zai.slug, "zai");
    }

    #[test]
    fn test_url_pattern_resolution() {
        let mut registry = ProviderRegistry::new();
        registry.register_by_url_pattern(
            "https://api.openai.com/v1",
            ProviderProfile::new(
                "openai",
                ApiFamily::OpenAiResponses,
                "https://api.openai.com/v1",
            ),
        );

        let result = registry.resolve_by_url("https://api.openai.com/v1/chat/completions");
        assert!(result.is_some());
        assert_eq!(result.unwrap().slug, "openai");
    }

    #[test]
    fn test_slugs_sorted() {
        let registry = ProviderRegistry::default();
        let slugs = registry.slugs();
        let mut sorted = slugs.clone();
        sorted.sort();
        assert_eq!(slugs, sorted);
    }
}
