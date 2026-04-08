//! Provider error types
//!
//! Structured errors for common failure classes including authentication,
//! transport, rate limiting, and malformed responses.

use thiserror::Error;

/// Result type alias for provider operations
pub type ProviderResult<T> = Result<T, ProviderError>;

/// Structured provider errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProviderError {
    /// Authentication failure (invalid or missing API key)
    #[error("Authentication failed: {message}")]
    Authentication { message: String },

    /// Transport/connection failure
    #[error("Transport error: {message}")]
    Transport { message: String },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        message: String,
        retry_after: Option<u64>,
    },

    /// Malformed response from provider
    #[error("Malformed response: {message}")]
    MalformedResponse { message: String },

    /// Invalid request parameters
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    /// Model not found or unsupported
    #[error("Model error: {message}")]
    Model { message: String },

    /// General provider error (catch-all)
    #[error("Provider error: {message}")]
    General { message: String },
}

impl ProviderError {
    /// Create an authentication error
    pub fn auth<S: Into<String>>(message: S) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a transport error
    pub fn transport<S: Into<String>>(message: S) -> Self {
        Self::Transport {
            message: message.into(),
        }
    }

    /// Create a rate limit error
    pub fn rate_limit<S: Into<String>>(message: S, retry_after: Option<u64>) -> Self {
        Self::RateLimit {
            message: message.into(),
            retry_after,
        }
    }

    /// Create a malformed response error
    pub fn malformed<S: Into<String>>(message: S) -> Self {
        Self::MalformedResponse {
            message: message.into(),
        }
    }

    /// Create an invalid request error
    pub fn invalid_request<S: Into<String>>(message: S) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    /// Create a model error
    pub fn model<S: Into<String>>(message: S) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    /// Create a general error
    pub fn general<S: Into<String>>(message: S) -> Self {
        Self::General {
            message: message.into(),
        }
    }

    /// Check if this is an authentication error
    pub fn is_authentication(&self) -> bool {
        matches!(self, Self::Authentication { .. })
    }

    /// Check if this is a rate limit error
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, Self::RateLimit { .. })
    }

    /// Check if this is a transport error
    pub fn is_transport(&self) -> bool {
        matches!(self, Self::Transport { .. })
    }

    /// Get retry-after duration if available
    pub fn retry_after(&self) -> Option<u64> {
        match self {
            Self::RateLimit { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}
