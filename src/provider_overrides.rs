//! Provider-specific configuration overrides
//!
//! Centralizes provider-specific static behavior in one boundary.
//! Profiles carry provider identity; this module resolves typed override
//! behavior from that identity.

use crate::{
    profile::{ApiFamily, ProviderProfile, RuntimeConfig},
    ProviderError, ProviderResult,
};
use reqwest::header::HeaderMap;

/// Typed provider overrides scoped by API family.
#[derive(Debug, Clone)]
pub(crate) enum ProviderOverrides {
    Messages(MessagesOverrides),
    Completions(CompletionsOverrides),
    Responses(ResponsesOverrides),
}

/// Overrides for Messages-family providers.
#[derive(Debug, Clone, Default)]
pub(crate) struct MessagesOverrides {
    /// Anthropic version header to include.
    pub anthropic_version: Option<String>,
}

/// Overrides for Completions-family providers.
#[derive(Debug, Clone, Default)]
pub(crate) struct CompletionsOverrides {}

/// Overrides for Responses-family providers.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResponsesOverrides {
    /// Codex product header.
    pub originator: Option<String>,
    /// Codex User-Agent header.
    pub user_agent: Option<String>,
    /// Codex account routing header.
    pub chatgpt_account_id: Option<String>,
    /// Fixed request body fields for Codex.
    pub fixed_body: Option<CodexFixedBody>,
    /// Endpoint path override.
    pub endpoint_path: Option<String>,
}

/// Fixed request body fields for Codex.
#[derive(Debug, Clone)]
pub(crate) struct CodexFixedBody {
    pub store: bool,
    pub reasoning_effort: String,
    pub parallel_tool_calls: bool,
}

impl Default for CodexFixedBody {
    fn default() -> Self {
        Self {
            store: false,
            reasoning_effort: "medium".into(),
            parallel_tool_calls: true,
        }
    }
}

/// Resolve provider overrides from a profile and runtime config.
pub(crate) fn resolve_overrides(
    profile: &ProviderProfile,
    _runtime: &RuntimeConfig,
) -> ProviderOverrides {
    match profile.family {
        ApiFamily::Messages => {
            let mut overrides = MessagesOverrides::default();
            // Anthropic protocol version header
            if profile.slug == "anthropic" || profile.base_url.contains("anthropic") {
                overrides.anthropic_version = Some("2023-06-01".into());
            }
            // minimax also uses anthropic API but doesn't need version header
            ProviderOverrides::Messages(overrides)
        }
        ApiFamily::Completions => ProviderOverrides::Completions(CompletionsOverrides::default()),
        ApiFamily::Responses => {
            let mut overrides = ResponsesOverrides::default();
            // Codex-specific overrides
            if profile.slug == "codex" {
                overrides.originator = Some("iron-providers".into());
                overrides.user_agent =
                    Some(format!("iron-providers/{}", env!("CARGO_PKG_VERSION")));
                overrides.fixed_body = Some(CodexFixedBody::default());
                overrides.endpoint_path = Some("/responses".into());

                // Extract chatgpt-account-id from JWT if available
                if let crate::profile::ProviderCredential::OAuthBearer {
                    access_token,
                    id_token,
                    ..
                } = &_runtime.credential
                {
                    let token = id_token.as_deref().unwrap_or(access_token);
                    if let Some(account_id) = chatgpt_account_id_from_jwt(token) {
                        overrides.chatgpt_account_id = Some(account_id);
                    }
                }
            }
            ProviderOverrides::Responses(overrides)
        }
    }
}

/// Extract chatgpt-account-id from a JWT token.
pub(crate) fn chatgpt_account_id_from_jwt(id_token: &str) -> Option<String> {
    let payload_b64 = id_token.split('.').nth(1)?;
    let payload_json = base64_decode_url_safe(payload_b64).ok()?;
    let payload: serde_json::Value = serde_json::from_slice(&payload_json).ok()?;

    payload
        .get("chatgpt_account_id")
        .and_then(|v| v.as_str().map(String::from))
        .or_else(|| {
            payload
                .get("https://api.openai.com/auth.chatgpt_account_id")
                .and_then(|v| v.as_str().map(String::from))
        })
        .or_else(|| {
            payload
                .get("https://api.openai.com/auth")
                .and_then(|nested| nested.get("chatgpt_account_id"))
                .and_then(|v| v.as_str().map(String::from))
        })
        .or_else(|| {
            payload
                .get("organizations")
                .and_then(|orgs| orgs.as_array())
                .and_then(|orgs| orgs.first())
                .and_then(|first| first.get("id"))
                .and_then(|v| v.as_str().map(String::from))
        })
}

fn base64_decode_url_safe(input: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::{engine::general_purpose, Engine as _};
    general_purpose::URL_SAFE_NO_PAD.decode(input)
}

/// Extract override headers from resolved provider overrides.
pub(crate) fn override_headers(overrides: &ProviderOverrides) -> ProviderResult<HeaderMap> {
    let mut headers = HeaderMap::new();

    match overrides {
        ProviderOverrides::Messages(msg) => {
            if let Some(ref version) = msg.anthropic_version {
                let val = reqwest::header::HeaderValue::from_str(version).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid override header value for 'anthropic-version': {}",
                        e
                    ))
                })?;
                headers.insert("anthropic-version", val);
            }
        }
        ProviderOverrides::Responses(resp) => {
            if let Some(ref originator) = resp.originator {
                let val = reqwest::header::HeaderValue::from_str(originator).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid override header value for 'originator': {}",
                        e
                    ))
                })?;
                headers.insert("originator", val);
            }
            if let Some(ref ua) = resp.user_agent {
                let val = reqwest::header::HeaderValue::from_str(ua).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid override header value for 'user-agent': {}",
                        e
                    ))
                })?;
                headers.insert(reqwest::header::USER_AGENT, val);
            }
            if let Some(ref account_id) = resp.chatgpt_account_id {
                let val = reqwest::header::HeaderValue::from_str(account_id).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid override header value for 'chatgpt-account-id': {}",
                        e
                    ))
                })?;
                headers.insert("chatgpt-account-id", val);
            }
        }
        _ => {}
    }

    Ok(headers)
}
