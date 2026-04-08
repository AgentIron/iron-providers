# iron-providers

Multi-provider inference layer for AgentIron.

## Install

```toml
[dependencies]
iron-providers = "0.1.0"
```

Normalizes requests and responses across OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages API families through a profile-driven generic provider and registry.

## Provider Slugs

Built-in provider profiles are identified by slug. Each slug maps to a specific API family, base URL, and authentication strategy.

| Slug | API Family | Purpose | Auth |
| --- | --- | --- | --- |
| `anthropic` | Anthropic Messages | General | `x-api-key` header |
| `minimax` | Anthropic Messages | General | Bearer token |
| `minimax-code` | Anthropic Messages | Coding | Bearer token |
| `zai` | OpenAI Chat Completions | General | Bearer token |
| `zai-code` | OpenAI Chat Completions | Coding | Bearer token |
| `kimi` | OpenAI Chat Completions | General | Bearer token |
| `openrouter` | OpenAI Chat Completions | General | Bearer token |
| `requesty` | OpenAI Chat Completions | General | Bearer token |

Coding-purpose slugs route to endpoints optimized for code generation tasks.

## API Families

Providers are grouped into three adapter families:

- **OpenAiResponses** — Uses the OpenAI Responses API via `async-openai`.
- **OpenAiChatCompletions** — Uses the `/chat/completions` endpoint via `reqwest`. Compatible with any OpenAI-compatible API.
- **AnthropicMessages** — Uses the Anthropic `/v1/messages` endpoint via `reqwest`.

## Registry Usage

```rust
use iron_providers::{ProviderRegistry, RuntimeConfig, InferenceRequest, Transcript, Message};

// Create a registry with built-in providers
let registry = ProviderRegistry::default();

// Look up a provider by slug
let provider = registry.get("zai", RuntimeConfig::new("your-api-key"))?;

// Use it for inference
let request = InferenceRequest::new("model-name", Transcript::with_messages(vec![
    Message::user("Hello"),
]));

let events = provider.infer(request).await?;
```

### Custom Providers

Register custom providers with a profile:

```rust
use iron_providers::{
    ProviderRegistry, ProviderProfile, ApiFamily, AuthStrategy, RuntimeConfig,
};

let mut registry = ProviderRegistry::new();

registry.register(
    ProviderProfile::new("my-provider", ApiFamily::OpenAiChatCompletions, "https://api.example.com/v1")
        .with_auth(AuthStrategy::BearerToken)
        .with_header("X-Custom-Header", "value")
);

let provider = registry.get("my-provider", RuntimeConfig::new("key"))?;
```

### URL Pattern Resolution

For auto-detection based on endpoint URLs:

```rust
registry.register_by_url_pattern(
    "https://api.openai.com/v1",
    ProviderProfile::new("openai", ApiFamily::OpenAiResponses, "https://api.openai.com/v1"),
);

let profile = registry.resolve_by_url("https://api.openai.com/v1/chat/completions");
```

### Listing Available Providers

```rust
let slugs: Vec<&str> = registry.slugs();
// Returns sorted list: ["anthropic", "kimi", "minimax", "minimax-code", "openrouter", "requesty", "zai", "zai-code"]
```

## Provider Trait

All providers implement the `Provider` trait:

```rust
pub trait Provider: Send + Sync {
    fn infer(&self, request: InferenceRequest) -> ProviderFuture<'_, Vec<ProviderEvent>>;
    fn infer_stream(&self, request: InferenceRequest) -> ProviderFuture<'_, BoxStream<'static, ProviderResult<ProviderEvent>>>;
}
```

The `GenericProvider` dispatches to the correct adapter (Responses, Chat Completions, or Anthropic) based on the profile's API family.

## Key Types

- `ProviderProfile` — Slug, API family, base URL, auth strategy, headers, purpose, and quirks.
- `RuntimeConfig` — API key and optional default model for a session.
- `InferenceRequest` — Normalized request with model, transcript, tools, and generation config.
- `ProviderEvent` — Streamed events: `Output`, `ToolCall`, `Complete`, `Error`, `Status`.
- `Transcript` / `Message` — Conversation history in provider-agnostic format.
- `ToolDefinition` / `ToolPolicy` — Tool schema and usage policy.
- `GenerationConfig` — Temperature, max tokens, top-p, stop sequences.

## Development

Install the task runner and security tooling if you want to use the local `invoke`
workflow:

```bash
pip install invoke
cargo install cargo-audit cargo-lockbud
```

Available tasks:

```bash
invoke build
invoke test
invoke security
```

These tasks print a short summary with warnings, failures, and the count of
successful steps only.

`build` runs:

- `cargo build --manifest-path Cargo.toml --all-targets`
- `cargo fmt --manifest-path Cargo.toml -- --check`
- `cargo clippy --manifest-path Cargo.toml --all-targets --all-features -- -D warnings`

`test` runs:

- `cargo test --manifest-path Cargo.toml`

`security` runs:

- `cargo generate-lockfile --manifest-path Cargo.toml` when needed
- `cargo audit`
- `cargo lockbud -k all`

## Testing

```bash
cargo test -p iron-providers
```

Tests include unit tests for message mapping, tool mapping, and error handling, plus mock HTTP integration tests for all built-in provider slugs using `mockito`.

## License

This project is licensed under the Apache License 2.0. See `LICENSE-APACHE`.
