//! Protocol adapters organized by API family
//!
//! Each module implements request projection, response parsing, streaming
//! normalization, and `ProviderEvent` mapping for its protocol.

pub mod completions;
pub mod messages;
pub mod responses;
