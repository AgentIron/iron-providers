//! Provider error types
//!
//! Structured errors for common failure classes including authentication,
//! transport, rate limiting, and malformed responses.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type alias for provider operations
pub type ProviderResult<T> = Result<T, ProviderError>;

/// Structured provider errors.
///
/// Each variant maps to a common failure class so downstream consumers can
/// programmatically classify failures without parsing error strings.
///
/// # Example
///
/// ```
/// use iron_providers::ProviderError;
///
/// let err = ProviderError::rate_limit("too many requests", Some(30));
/// assert!(err.is_rate_limit());
/// assert_eq!(err.retry_after(), Some(30));
/// ```
#[derive(Error, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderError {
    /// Authentication failure (invalid or missing API key)
    #[error("Authentication failed: {message}")]
    Authentication {
        /// Human-readable description of the authentication failure.
        message: String,
    },

    /// Transport/connection failure
    #[error("Transport error: {message}")]
    Transport {
        /// Human-readable description of the transport failure.
        message: String,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        /// Human-readable description of the rate-limit failure.
        message: String,
        /// Suggested seconds to wait before retrying, when the provider supplies one.
        retry_after: Option<u64>,
    },

    /// Malformed response from provider
    #[error("Malformed response: {message}")]
    MalformedResponse {
        /// Human-readable description of what was malformed.
        message: String,
    },

    /// Invalid request parameters
    #[error("Invalid request: {message}")]
    InvalidRequest {
        /// Human-readable description of the invalid request.
        message: String,
    },

    /// Model not found or unsupported
    #[error("Model error: {message}")]
    Model {
        /// Human-readable description of the model error.
        message: String,
    },

    /// General provider error (catch-all)
    #[error("Provider error: {message}")]
    General {
        /// Human-readable description of the general error.
        message: String,
    },
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
