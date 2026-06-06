use crate::{
    profile::{
        ApiFamily, AuthStrategy, CredentialAuthConfig, CredentialKind, EndpointPurpose,
        ProviderProfile, RuntimeConfig,
    },
    provider::Provider,
    ProviderError, ProviderResult,
};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ProviderRegistry {
    profiles: HashMap<String, Arc<ProviderProfile>>,
    url_patterns: Vec<(String, Arc<ProviderProfile>)>,
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
        self.profiles.insert(key, Arc::new(profile));
    }

    pub fn register_by_url_pattern(
        &mut self,
        url_prefix: impl Into<String>,
        profile: ProviderProfile,
    ) {
        let key = profile.slug.to_lowercase();
        let profile = Arc::new(profile);
        self.profiles.insert(key, Arc::clone(&profile));
        self.url_patterns.push((url_prefix.into(), profile));
    }

    pub fn get(
        &self,
        provider_name: &str,
        runtime_config: RuntimeConfig,
    ) -> ProviderResult<Box<dyn Provider>> {
        let key = provider_name.to_lowercase();
        let profile = self.profiles.get(&key).ok_or_else(|| {
            let available: Vec<&str> = self.profiles.keys().map(|s| s.as_str()).collect();
            ProviderError::general(format!(
                "Unknown provider '{}'. Available: {:?}",
                provider_name, available
            ))
        })?;

        crate::connection::ProviderConnection::from_arc(Arc::clone(profile), runtime_config)
            .map(|p| Box::new(p) as Box<dyn Provider>)
    }

    /// Resolve a registered profile by matching the URL against registered
    /// prefixes. When multiple prefixes match, the longest wins so that more
    /// specific routes take precedence over generic ones (e.g. a
    /// `https://api.example.com/v1/coding` prefix is preferred over
    /// `https://api.example.com/v1`).
    pub fn resolve_by_url(&self, url: &str) -> Option<&ProviderProfile> {
        self.url_patterns
            .iter()
            .filter(|(prefix, _)| url.starts_with(prefix))
            .max_by_key(|(prefix, _)| prefix.len())
            .and_then(|(_, profile)| self.profiles.get(&profile.slug.to_lowercase()))
            .map(|arc| arc.as_ref())
    }

    pub fn resolve_by_models_dev_id(&self, models_dev_id: &str) -> Option<&ProviderProfile> {
        self.profiles_by_models_dev_id(models_dev_id)
            .into_iter()
            .next()
    }

    pub fn profiles_by_models_dev_id(&self, models_dev_id: &str) -> Vec<&ProviderProfile> {
        let mut matches: Vec<&ProviderProfile> = self
            .profiles
            .values()
            .filter(|profile| {
                profile
                    .models_dev_slug()
                    .eq_ignore_ascii_case(models_dev_id)
            })
            .map(|arc| arc.as_ref())
            .collect();
        matches.sort_by_key(|p| &p.slug);
        matches
    }

    pub fn slugs(&self) -> Vec<&str> {
        let mut slugs: Vec<&str> = self.profiles.keys().map(|s| s.as_str()).collect();
        slugs.sort();
        slugs
    }

    pub fn system_prompt_fragment(&self, provider_name: &str) -> ProviderResult<&'static str> {
        let key = provider_name.to_lowercase();
        let profile = self.profiles.get(&key).ok_or_else(|| {
            let available: Vec<&str> = self.profiles.keys().map(|s| s.as_str()).collect();
            ProviderError::general(format!(
                "Unknown provider '{}'. Available: {:?}",
                provider_name, available
            ))
        })?;
        Ok(profile.system_prompt_fragment())
    }

    pub fn register_builtins(&mut self) {
        self.register(ProviderProfile::new(
            "openai",
            ApiFamily::Responses,
            "https://api.openai.com/v1",
        ));

        self.register(
            ProviderProfile::new(
                "anthropic",
                ApiFamily::Messages,
                "https://api.anthropic.com",
            )
            .with_auth(AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into(),
            }),
        );

        self.register(
            ProviderProfile::new(
                "minimax",
                ApiFamily::Messages,
                "https://api.minimax.io/anthropic",
            )
            .with_auth(AuthStrategy::BearerToken),
        );

        self.register(
            ProviderProfile::new(
                "minimax-code",
                ApiFamily::Messages,
                "https://api.minimax.io/anthropic",
            )
            .with_models_dev_id("minimax-coding-plan")
            .with_auth(AuthStrategy::BearerToken)
            .with_purpose(EndpointPurpose::Coding),
        );

        self.register(ProviderProfile::new(
            "zai",
            ApiFamily::Completions,
            "https://api.z.ai/api/paas/v4",
        ));

        self.register(
            ProviderProfile::new(
                "zai-code",
                ApiFamily::Completions,
                "https://api.z.ai/api/coding/paas/v4",
            )
            .with_models_dev_id("zai-coding-plan")
            .with_purpose(EndpointPurpose::Coding),
        );

        self.register(
            ProviderProfile::new("kimi", ApiFamily::Completions, "https://api.moonshot.ai/v1")
                .with_models_dev_id("moonshotai"),
        );

        self.register(
            ProviderProfile::new(
                "kimi-code",
                ApiFamily::Messages,
                "https://api.kimi.com/coding",
            )
            .with_models_dev_id("kimi-for-coding")
            .with_auth(AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into(),
            })
            .with_credential_auth(CredentialKind::OAuthBearer, AuthStrategy::BearerToken)
            .with_purpose(EndpointPurpose::Coding),
        );

        self.register(
            ProviderProfile::new(
                "openrouter",
                ApiFamily::Completions,
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
            ApiFamily::Completions,
            "https://api.requesty.ai/v1",
        ));

        {
            let mut codex = ProviderProfile::new(
                "codex",
                ApiFamily::Responses,
                "https://chatgpt.com/backend-api/codex",
            )
            .with_models_dev_id("openai")
            .with_purpose(EndpointPurpose::Coding);
            codex.credential_auth = vec![CredentialAuthConfig {
                kind: CredentialKind::OAuthBearer,
                auth_strategy: AuthStrategy::BearerToken,
            }];
            self.register(codex);
        }

        self.register(ProviderProfile::new(
            "ollama-cloud",
            ApiFamily::Completions,
            "https://api.ollama.cloud",
        ));

        {
            let local =
                ProviderProfile::new("local", ApiFamily::Completions, "http://localhost:11434/v1")
                    .with_credential_auth(CredentialKind::NoAuth, AuthStrategy::NoAuth);
            let local_arc = Arc::new(local);
            let key = "local".to_string();
            self.profiles.insert(key, Arc::clone(&local_arc));
            self.url_patterns
                .push(("http://localhost".into(), Arc::clone(&local_arc)));
            self.url_patterns
                .push(("http://127.0.0.1".into(), Arc::clone(&local_arc)));
            self.url_patterns.push(("http://0.0.0.0".into(), local_arc));
        }
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
    use crate::ProviderCredential;
    use std::time::SystemTime;

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
        assert!(slugs.contains(&"codex"));
        assert!(slugs.contains(&"openai"));
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
        assert_eq!(kimi_code.family, ApiFamily::Messages);
        assert_eq!(
            kimi_code.auth_strategy_for(CredentialKind::ApiKey),
            Some(&AuthStrategy::ApiKeyHeader {
                header_name: "x-api-key".into()
            })
        );
        assert_eq!(kimi_code.purpose, EndpointPurpose::Coding);
        assert_eq!(zai.slug, "zai");
    }

    #[test]
    fn test_url_pattern_resolution() {
        let mut registry = ProviderRegistry::new();
        registry.register_by_url_pattern(
            "https://api.openai.com/v1",
            ProviderProfile::new("openai", ApiFamily::Responses, "https://api.openai.com/v1"),
        );

        let result = registry.resolve_by_url("https://api.openai.com/v1/chat/completions");
        assert!(result.is_some());
        assert_eq!(result.unwrap().slug, "openai");
    }

    #[test]
    fn test_url_pattern_resolution_prefers_longest_prefix() {
        let mut registry = ProviderRegistry::new();
        // Register generic prefix first, specific second — longest-prefix match
        // must still select the specific one regardless of insertion order.
        registry.register_by_url_pattern(
            "https://api.example.com/v1",
            ProviderProfile::new(
                "general",
                ApiFamily::Completions,
                "https://api.example.com/v1",
            ),
        );
        registry.register_by_url_pattern(
            "https://api.example.com/v1/coding",
            ProviderProfile::new(
                "coding",
                ApiFamily::Completions,
                "https://api.example.com/v1/coding",
            ),
        );

        let coding = registry.resolve_by_url("https://api.example.com/v1/coding/chat/completions");
        assert_eq!(coding.map(|p| p.slug.as_str()), Some("coding"));

        let general = registry.resolve_by_url("https://api.example.com/v1/chat/completions");
        assert_eq!(general.map(|p| p.slug.as_str()), Some("general"));
    }

    #[test]
    fn test_slugs_sorted() {
        let registry = ProviderRegistry::default();
        let slugs = registry.slugs();
        let mut sorted = slugs.clone();
        sorted.sort();
        assert_eq!(slugs, sorted);
    }

    #[test]
    fn test_system_prompt_fragment_for_all_builtins() {
        let registry = ProviderRegistry::default();
        for slug in registry.slugs() {
            let fragment = registry.system_prompt_fragment(slug).expect(slug);
            assert!(
                !fragment.is_empty(),
                "fragment for '{}' should not be empty",
                slug
            );
        }
    }

    #[test]
    fn test_system_prompt_fragment_case_insensitive() {
        let registry = ProviderRegistry::default();
        let lower = registry.system_prompt_fragment("anthropic").unwrap();
        let upper = registry.system_prompt_fragment("ANTHROPIC").unwrap();
        assert_eq!(lower, upper);
    }

    #[test]
    fn test_system_prompt_fragment_unknown_provider() {
        let registry = ProviderRegistry::default();
        let result = registry.system_prompt_fragment("nonexistent");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unknown provider"));
        }
    }

    #[test]
    fn test_fragments_contain_no_tera_delimiters() {
        let fragments = [
            crate::apis::messages::SYSTEM_PROMPT_FRAGMENT,
            crate::apis::completions::SYSTEM_PROMPT_FRAGMENT,
        ];
        for fragment in fragments {
            assert!(
                !fragment.contains("{{"),
                "fragment should not contain Tera expression delimiter"
            );
            assert!(
                !fragment.contains("{%"),
                "fragment should not contain Tera block delimiter"
            );
            assert!(
                !fragment.contains("{#"),
                "fragment should not contain Tera comment delimiter"
            );
        }
    }

    #[test]
    fn test_codex_profile_registered() {
        let registry = ProviderRegistry::default();
        let codex = registry.get("codex", RuntimeConfig::new("test-key"));
        // codex rejects API-key credentials
        assert!(codex.is_err());
        if let Err(ref e) = codex {
            let msg = e.to_string();
            assert!(msg.contains("codex"));
            assert!(msg.contains("does not support"));
        }
    }

    #[test]
    fn test_codex_profile_metadata() {
        let registry = ProviderRegistry::default();
        let profile = registry
            .resolve_by_models_dev_id("openai")
            .expect("codex uses models_dev_id = openai");
        assert_eq!(profile.slug, "codex");
        assert_eq!(profile.family, ApiFamily::Responses);
        assert_eq!(profile.purpose, EndpointPurpose::Coding);
        assert!(profile.supports_credential(CredentialKind::OAuthBearer));
        assert!(!profile.supports_credential(CredentialKind::ApiKey));
    }

    #[test]
    fn test_openai_profile_metadata() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("openai").expect("openai");
        assert_eq!(profile.family, ApiFamily::Responses);
        assert_eq!(profile.base_url, "https://api.openai.com/v1");
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert!(!profile.supports_credential(CredentialKind::OAuthBearer));
        assert_eq!(profile.purpose, EndpointPurpose::General);
    }

    #[test]
    fn test_openai_accepts_api_key() {
        let registry = ProviderRegistry::default();
        let result = registry.get("openai", RuntimeConfig::new("test-key"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_openai_and_codex_are_distinct() {
        let registry = ProviderRegistry::default();
        let openai = registry.profiles.get("openai").expect("openai");
        let codex = registry.profiles.get("codex").expect("codex");

        assert_ne!(openai.slug, codex.slug);
        assert_eq!(openai.base_url, "https://api.openai.com/v1");
        assert_eq!(codex.base_url, "https://chatgpt.com/backend-api/codex");
        assert!(openai.supports_credential(CredentialKind::ApiKey));
        assert!(!codex.supports_credential(CredentialKind::ApiKey));
        assert!(!openai.supports_credential(CredentialKind::OAuthBearer));
        assert!(codex.supports_credential(CredentialKind::OAuthBearer));
        assert_eq!(openai.purpose, EndpointPurpose::General);
        assert_eq!(codex.purpose, EndpointPurpose::Coding);
    }

    #[test]
    fn test_openai_does_not_change_codex_behavior() {
        let registry = ProviderRegistry::default();

        let codex_api_key = registry.get("codex", RuntimeConfig::new("test-key"));
        assert!(codex_api_key.is_err());
        if let Err(ref e) = codex_api_key {
            assert!(e.to_string().contains("does not support"));
        }

        let codex = registry.profiles.get("codex").expect("codex");
        assert_eq!(codex.base_url, "https://chatgpt.com/backend-api/codex");
        assert!(codex.supports_credential(CredentialKind::OAuthBearer));
        assert!(!codex.supports_credential(CredentialKind::ApiKey));
    }

    #[test]
    fn test_profiles_by_models_dev_id_returns_both_openai_profiles() {
        let registry = ProviderRegistry::default();
        let profiles = registry.profiles_by_models_dev_id("openai");
        assert!(profiles.len() >= 2, "should have at least openai and codex");
        let slugs: Vec<&str> = profiles.iter().map(|p| p.slug.as_str()).collect();
        assert!(slugs.contains(&"codex"));
        assert!(slugs.contains(&"openai"));
    }

    #[test]
    fn test_profiles_by_models_dev_id_returns_sorted() {
        let registry = ProviderRegistry::default();
        let profiles = registry.profiles_by_models_dev_id("openai");
        let slugs: Vec<&str> = profiles.iter().map(|p| p.slug.as_str()).collect();
        let mut sorted = slugs.clone();
        sorted.sort();
        assert_eq!(slugs, sorted, "profiles should be sorted by slug");
    }

    #[test]
    fn test_resolve_by_models_dev_id_is_deterministic() {
        let registry = ProviderRegistry::default();
        let first = registry.resolve_by_models_dev_id("openai");
        let second = registry.resolve_by_models_dev_id("openai");
        assert_eq!(
            first.map(|p| p.slug.as_str()),
            second.map(|p| p.slug.as_str()),
            "repeated calls must return the same profile"
        );
    }

    #[test]
    fn test_profiles_by_models_dev_id_deterministic_order() {
        let registry = ProviderRegistry::default();
        let run1: Vec<&str> = registry
            .profiles_by_models_dev_id("openai")
            .iter()
            .map(|p| p.slug.as_str())
            .collect();
        let run2: Vec<&str> = registry
            .profiles_by_models_dev_id("openai")
            .iter()
            .map(|p| p.slug.as_str())
            .collect();
        assert_eq!(run1, run2, "order must be deterministic across calls");
    }

    #[test]
    fn test_local_default_endpoint_unchanged() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("local").unwrap();
        assert_eq!(profile.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn test_local_with_base_url_override() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("local").unwrap().as_ref().clone();
        let runtime = RuntimeConfig::none().with_base_url("http://localhost:1234/v1");
        let conn =
            crate::connection::ProviderConnection::from_profile(profile.clone(), runtime).unwrap();
        assert_eq!(conn.effective_base_url(), "http://localhost:1234/v1");
        assert_eq!(profile.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn test_local_override_preserves_noauth() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("local").unwrap().as_ref().clone();
        let runtime = RuntimeConfig::none().with_base_url("http://localhost:9999/v1");
        let conn =
            crate::connection::ProviderConnection::from_profile(profile.clone(), runtime).unwrap();
        assert_eq!(conn.effective_base_url(), "http://localhost:9999/v1");
        assert!(profile.supports_credential(CredentialKind::NoAuth));
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert_eq!(profile.family, ApiFamily::Completions);
    }

    #[test]
    fn test_local_override_preserves_api_key() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("local").unwrap().as_ref().clone();
        let runtime = RuntimeConfig::new("my-token").with_base_url("http://localhost:5555");
        let conn =
            crate::connection::ProviderConnection::from_profile(profile.clone(), runtime).unwrap();
        assert_eq!(conn.effective_base_url(), "http://localhost:5555");
        assert_eq!(profile.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn test_local_registered_profile_unchanged_after_override() {
        let registry = ProviderRegistry::default();
        let profile_before = registry.profiles.get("local").unwrap().as_ref().clone();
        let runtime = RuntimeConfig::none().with_base_url("http://custom:8080");
        let _conn =
            crate::connection::ProviderConnection::from_profile(profile_before.clone(), runtime)
                .unwrap();
        assert_eq!(profile_before.base_url, "http://localhost:11434/v1");
        assert_eq!(profile_before.family, ApiFamily::Completions);
        assert!(profile_before.supports_credential(CredentialKind::NoAuth));
        assert!(profile_before.supports_credential(CredentialKind::ApiKey));
    }

    #[test]
    fn test_kimi_code_supports_both_credentials() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("kimi-code").unwrap();
        assert!(profile.supports_credential(CredentialKind::ApiKey));
        assert!(profile.supports_credential(CredentialKind::OAuthBearer));
        assert_eq!(profile.base_url, "https://api.kimi.com/coding");
    }

    #[test]
    fn test_kimi_rejects_oauth() {
        let registry = ProviderRegistry::default();
        let result = registry.get(
            "kimi",
            RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
                access_token: "tok".into(),
                expires_at: None,
                id_token: None,
            }),
        );
        assert!(result.is_err());
        if let Err(ref e) = result {
            let msg = e.to_string();
            assert!(msg.contains("kimi"));
            assert!(msg.contains("does not support"));
        }
    }

    #[test]
    fn test_blank_api_key_fails_during_registry_construction() {
        let registry = ProviderRegistry::default();
        let result = registry.get("zai", RuntimeConfig::new("   "));
        assert!(result.is_err());
        if let Err(ref e) = result {
            assert!(e.is_authentication());
            assert!(e.to_string().contains("API key is required"));
        }
    }

    #[test]
    fn test_blank_oauth_token_fails_during_registry_construction() {
        let registry = ProviderRegistry::default();
        let result = registry.get(
            "codex",
            RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
                access_token: "   ".into(),
                expires_at: None,
                id_token: None,
            }),
        );
        assert!(result.is_err());
        if let Err(ref e) = result {
            assert!(e.is_authentication());
            assert!(e.to_string().contains("OAuth access token is required"));
        }
    }

    #[test]
    fn test_expired_oauth_fails() {
        let registry = ProviderRegistry::default();
        let result = registry.get(
            "codex",
            RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
                access_token: "tok".into(),
                expires_at: Some(SystemTime::UNIX_EPOCH),
                id_token: None,
            }),
        );
        assert!(result.is_err());
        if let Err(ref e) = result {
            assert!(e.to_string().contains("expired"));
        }
    }

    #[test]
    fn test_ollama_cloud_registered() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("ollama-cloud").unwrap();
        assert_eq!(profile.family, ApiFamily::Completions);
        assert_eq!(profile.base_url, "https://api.ollama.cloud");
        assert!(profile.supports_credential(CredentialKind::ApiKey));
    }

    #[test]
    fn test_ollama_cloud_accepts_api_key() {
        let registry = ProviderRegistry::default();
        let result = registry.get("ollama-cloud", RuntimeConfig::new("test-key"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_local_registered() {
        let registry = ProviderRegistry::default();
        let profile = registry.profiles.get("local").unwrap();
        assert_eq!(profile.family, ApiFamily::Completions);
        assert_eq!(profile.base_url, "http://localhost:11434/v1");
        assert!(profile.supports_credential(CredentialKind::NoAuth));
        assert!(profile.supports_credential(CredentialKind::ApiKey));
    }

    #[test]
    fn test_local_accepts_noauth() {
        let registry = ProviderRegistry::default();
        let result = registry.get("local", RuntimeConfig::none());
        assert!(result.is_ok());
    }

    #[test]
    fn test_local_accepts_api_key() {
        let registry = ProviderRegistry::default();
        let result = registry.get("local", RuntimeConfig::new("my-token"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_local_rejects_oauth() {
        let registry = ProviderRegistry::default();
        let result = registry.get(
            "local",
            RuntimeConfig::from_credential(ProviderCredential::OAuthBearer {
                access_token: "tok".into(),
                expires_at: None,
                id_token: None,
            }),
        );
        assert!(result.is_err());
        if let Err(ref e) = result {
            let msg = e.to_string();
            assert!(msg.contains("local"));
            assert!(msg.contains("does not support"));
        }
    }

    #[test]
    fn test_local_in_slugs() {
        let registry = ProviderRegistry::default();
        let slugs: Vec<&str> = registry.profiles.keys().map(|s| s.as_str()).collect();
        assert!(slugs.contains(&"local"));
        assert!(slugs.contains(&"ollama-cloud"));
    }

    #[test]
    fn test_resolve_by_url_localhost() {
        let registry = ProviderRegistry::default();
        let profile = registry.resolve_by_url("http://localhost:8080/v1/chat/completions");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().slug, "local");
    }

    #[test]
    fn test_resolve_by_url_127() {
        let registry = ProviderRegistry::default();
        let profile = registry.resolve_by_url("http://127.0.0.1:11434/v1/chat/completions");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().slug, "local");
    }

    #[test]
    fn test_resolve_by_url_0000() {
        let registry = ProviderRegistry::default();
        let profile = registry.resolve_by_url("http://0.0.0.0:8000/v1/chat/completions");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().slug, "local");
    }

    #[test]
    fn test_resolve_by_url_rejects_remote() {
        let registry = ProviderRegistry::default();
        assert!(registry
            .resolve_by_url("https://api.openai.com/v1")
            .is_none());
        assert!(registry.resolve_by_url("http://example.com/v1").is_none());
    }

    #[test]
    fn test_noauth_fails_for_api_key_provider() {
        let registry = ProviderRegistry::default();
        let result = registry.get("zai", RuntimeConfig::none());
        assert!(result.is_err());
        if let Err(ref e) = result {
            let msg = e.to_string();
            assert!(msg.contains("zai"));
            assert!(msg.contains("does not support"));
        }
    }

    #[test]
    fn test_noauth_validation_passes() {
        let rt = RuntimeConfig::none();
        assert!(rt.validate().is_ok());
    }

    #[test]
    fn test_blank_api_key_still_invalid() {
        let rt = RuntimeConfig::new("   ");
        assert!(rt.validate().is_err());
    }
}
