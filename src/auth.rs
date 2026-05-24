//! Auth header generation
//!
//! Maps `ProviderCredential` plus `AuthStrategy` to auth headers.
//! This module is separate from HTTP client construction so auth can be
//! tested independently from transport assembly.

use crate::profile::{AuthStrategy, ProviderCredential};
use crate::{ProviderError, ProviderResult};
use reqwest::header::HeaderMap;

/// Map a credential and auth strategy to auth headers.
pub(crate) fn auth_headers(
    credential: &ProviderCredential,
    auth_strategy: &AuthStrategy,
    context: &str,
) -> ProviderResult<HeaderMap> {
    let mut headers = HeaderMap::new();
    let secret = credential.secret();

    match auth_strategy {
        AuthStrategy::NoAuth => {}
        AuthStrategy::BearerToken => {
            let val = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", secret))
                .map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid bearer token value for {}: {}",
                        context, e
                    ))
                })?;
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
        AuthStrategy::ApiKeyHeader { header_name } => {
            let key =
                reqwest::header::HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid header name '{}' for {}: {}",
                        header_name, context, e
                    ))
                })?;
            let val = reqwest::header::HeaderValue::from_str(secret).map_err(|e| {
                ProviderError::invalid_request(format!(
                    "Invalid API key value for {}: {}",
                    context, e
                ))
            })?;
            headers.insert(key, val);
        }
        AuthStrategy::Custom {
            header_name,
            prefix,
        } => {
            let value = match prefix {
                Some(p) => format!("{} {}", p, secret),
                None => secret.to_string(),
            };
            let key =
                reqwest::header::HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
                    ProviderError::invalid_request(format!(
                        "Invalid custom header name '{}' for {}: {}",
                        header_name, context, e
                    ))
                })?;
            let val = reqwest::header::HeaderValue::from_str(&value).map_err(|e| {
                ProviderError::invalid_request(format!(
                    "Invalid custom header value for {}: {}",
                    context, e
                ))
            })?;
            headers.insert(key, val);
        }
    }

    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{AuthStrategy, ProviderCredential};

    #[test]
    fn test_bearer_token() {
        let credential = ProviderCredential::ApiKey("secret".into());
        let strategy = AuthStrategy::BearerToken;
        let headers = auth_headers(&credential, &strategy, "test").unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer secret");
    }

    #[test]
    fn test_api_key_header() {
        let credential = ProviderCredential::ApiKey("secret".into());
        let strategy = AuthStrategy::ApiKeyHeader {
            header_name: "x-api-key".into(),
        };
        let headers = auth_headers(&credential, &strategy, "test").unwrap();
        assert_eq!(headers.get("x-api-key").unwrap(), "secret");
    }

    #[test]
    fn test_custom_header_with_prefix() {
        let credential = ProviderCredential::ApiKey("secret".into());
        let strategy = AuthStrategy::Custom {
            header_name: "x-custom".into(),
            prefix: Some("Token".into()),
        };
        let headers = auth_headers(&credential, &strategy, "test").unwrap();
        assert_eq!(headers.get("x-custom").unwrap(), "Token secret");
    }

    #[test]
    fn test_custom_header_without_prefix() {
        let credential = ProviderCredential::ApiKey("secret".into());
        let strategy = AuthStrategy::Custom {
            header_name: "x-custom".into(),
            prefix: None,
        };
        let headers = auth_headers(&credential, &strategy, "test").unwrap();
        assert_eq!(headers.get("x-custom").unwrap(), "secret");
    }

    #[test]
    fn test_invalid_header_name() {
        let credential = ProviderCredential::ApiKey("secret".into());
        let strategy = AuthStrategy::ApiKeyHeader {
            header_name: "bad header".into(),
        };
        let result = auth_headers(&credential, &strategy, "test");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid header name"));
    }

    #[test]
    fn test_invalid_header_value() {
        let credential = ProviderCredential::ApiKey("secret\0".into());
        let strategy = AuthStrategy::BearerToken;
        let result = auth_headers(&credential, &strategy, "test");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid bearer token value"));
    }

    #[test]
    fn test_no_auth_produces_no_headers() {
        let credential = ProviderCredential::NoAuth;
        let strategy = AuthStrategy::NoAuth;
        let headers = auth_headers(&credential, &strategy, "test").unwrap();
        assert!(headers.is_empty());
    }

    #[test]
    fn test_no_auth_no_authorization_header() {
        let credential = ProviderCredential::NoAuth;
        let strategy = AuthStrategy::NoAuth;
        let headers = auth_headers(&credential, &strategy, "test").unwrap();
        assert!(headers.get("authorization").is_none());
        assert!(headers.get("x-api-key").is_none());
    }

    #[test]
    fn test_blank_api_key_still_fails_validation() {
        let rt = crate::profile::RuntimeConfig::new("   ");
        assert!(rt.validate().is_err());
    }
}
