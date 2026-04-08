//! Tests for iron-providers errors

use iron_providers::error::ProviderError;

#[test]
fn test_error_authentication() {
    let err = ProviderError::auth("Invalid API key");
    assert!(err.is_authentication());
    assert!(!err.is_rate_limit());
    assert!(!err.is_transport());
}

#[test]
fn test_error_rate_limit() {
    let err = ProviderError::rate_limit("Too many requests", Some(60));
    assert!(!err.is_authentication());
    assert!(err.is_rate_limit());
    assert_eq!(err.retry_after(), Some(60));
}

#[test]
fn test_error_transport() {
    let err = ProviderError::transport("Connection refused");
    assert!(!err.is_authentication());
    assert!(err.is_transport());
}

#[test]
fn test_error_malformed() {
    let err = ProviderError::malformed("Invalid JSON");
    assert!(matches!(err, ProviderError::MalformedResponse { .. }));
}

#[test]
fn test_error_invalid_request() {
    let err = ProviderError::invalid_request("Bad parameter");
    assert!(matches!(err, ProviderError::InvalidRequest { .. }));
}

#[test]
fn test_error_model() {
    let err = ProviderError::model("Model not found");
    assert!(matches!(err, ProviderError::Model { .. }));
}

#[test]
fn test_error_general() {
    let err = ProviderError::general("Something went wrong");
    assert!(matches!(err, ProviderError::General { .. }));
}
