# iron-providers

Multi-provider inference layer for AgentIron.

## Install

```toml
[dependencies]
iron-providers = "0.1"
```

Normalizes requests and responses across OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages API families through a profile-driven generic provider and registry.

## Provider Slugs

Built-in provider profiles are identified by slug. Each slug maps to a specific API family, base URL, and authentication strategy.

| Slug | models.dev ID | API Family | Purpose | Auth |
| --- | --- | --- | --- | --- |
| `anthropic` | `anthropic` | Anthropic Messages | General | `x-api-key` header |
| `minimax` | `minimax` | Anthropic Messages | General | Bearer token |
| `minimax-code` | `minimax-coding-plan` | Anthropic Messages | Coding | Bearer token |
| `zai` | `zai` | OpenAI Chat Completions | General | Bearer token |
| `zai-code` | `zai-coding-plan` | OpenAI Chat Completions | Coding | Bearer token |
| `kimi` | `moonshotai` | OpenAI Chat Completions | General | Bearer token |
| `kimi-code` | `kimi-for-coding` | Anthropic Messages | Coding | `x-api-key` header |
| `openrouter` | `openrouter` | OpenAI Chat Completions | General | Bearer token |
| `requesty` | `requesty` | OpenAI Chat Completions | General | Bearer token |

Coding-purpose slugs route to endpoints optimized for code generation tasks.

## API Families

Providers are grouped into three adapter families:

- **OpenAiResponses** — Uses the OpenAI Responses API via `async-openai`.
- **OpenAiChatCompletions** — Uses the `/chat/completions` endpoint via `reqwest`. Compatible with any OpenAI-compatible API.
- **AnthropicMessages** — Uses the Anthropic `/v1/messages` endpoint via `reqwest`.

## Registry Usage

```rust
use iron_providers::{
    InferenceRequest, Message, ProviderRegistry, RuntimeConfig, Transcript,
};

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

### HTTP Timeouts

All adapters apply `connect_timeout` (default 30s) and `read_timeout`
(default 60s between socket reads) to their HTTP clients so a stalled
provider surfaces as a transport error instead of hanging the caller.
Override per session on `RuntimeConfig`:

```rust
use std::time::Duration;

let runtime = RuntimeConfig::new("key")
    .with_connect_timeout(Duration::from_secs(10))
    .with_read_timeout(Duration::from_secs(120));
```

`OpenAiConfig` exposes equivalent `with_connect_timeout` /
`with_read_timeout` builders for direct callers of `openai::infer`.

## Request Model

`InferenceRequest` now carries an `InferenceContext`:

- `context.transcript`: model-visible conversation history
- `context.runtime_records`: runtime-only structured records

Runtime records are **not** projected into assistant-visible message history by
default. They exist so adapters and runtimes can carry structured state without
polluting the model transcript.

```rust
use iron_providers::{InferenceRequest, RuntimeRecord, Transcript};
use serde_json::json;

let mut request = InferenceRequest::new("model-name", Transcript::new());
request
    .context
    .add_record(RuntimeRecord::new("session_state", json!({"trace_id": "abc123"})));
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
// Returns sorted list: ["anthropic", "kimi", "kimi-code", "minimax", "minimax-code", "openrouter", "requesty", "zai", "zai-code"]
```

### models.dev Integration

Built-in and custom profiles can optionally declare a distinct `models.dev` provider
identifier for client-side model discovery and caching.

```rust
let profile = ProviderProfile::new("kimi", ApiFamily::OpenAiChatCompletions, "https://api.moonshot.ai/v1")
    .with_models_dev_id("moonshotai");

assert_eq!(profile.models_dev_slug(), "moonshotai");
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

## Streaming Contract

Streaming is requested by calling `infer_stream`; non-streaming is requested by
calling `infer`. There is no `stream` field on `InferenceRequest`.

`ProviderEvent::Complete` has a strict meaning:

- `Complete` means the stream ended successfully.
- If a provider emits an unrecoverable `Error`, the stream ends without a later
  `Complete` event.

This contract now holds across OpenAI Responses, Chat Completions, and
Anthropic adapters.

## Profile Semantics

`ProviderProfile` is authoritative across provider families:

- `base_url` is used for all families
- `auth_strategy` is honored for all families, including `OpenAiResponses`
- `default_headers` are validated and applied consistently
- invalid profile auth/header configuration fails fast during client
  construction instead of silently falling back to a default client

## Key Types

- `ProviderProfile` — Slug, optional models.dev ID, API family, base URL, auth strategy, headers, purpose, and quirks.
- `RuntimeConfig` — API key and optional default model for a session.
- `InferenceRequest` — Normalized request with model, context, tools, and generation config.
- `InferenceContext` — Separates model-visible `Transcript` from runtime-only `RuntimeRecord` values.
- `ProviderEvent` — Streamed events: `Output`, `ToolCall`, `ChoiceRequest`, `Complete`, `Error`, `Status`. `Complete` is success-only. `Error` carries a structured `ProviderError` with classification (auth, rate-limit, transport, etc.).
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

## GitHub Workflow

- Open a GitHub issue before starting work.
- Create a feature branch for the issue and open a pull request against `main`.
- Pull requests must reference an issue in the title or body, for example `Closes #123`.
- The `Pull Request` workflow runs `cargo build`, `cargo fmt --check`, `cargo clippy`, and `cargo test` on every PR to `main`.
- Merges to `main` trigger an automatic patch release that bumps `Cargo.toml`, creates a `vX.Y.Z` tag, creates a GitHub release, and publishes the crate to crates.io.
- Coordinated `minor` and `major` releases are handled through the `Release Manual` workflow in GitHub Actions.

Repository configuration still matters:

- crates.io Trusted Publishing is supported by the release workflows through GitHub OIDC plus `rust-lang/crates-io-auth-action`, so `CRATES_IO_TOKEN` is not required when Trusted Publishing is configured for this repository and workflow.
- If branch protection blocks workflow pushes to `main`, add a `RELEASE_GITHUB_TOKEN` secret for a token that is allowed to push the automated release commit and tag.
- Branch protection is configured to require the `Validate PR Policy` and `Rust Checks` status checks before merge.

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

The test suite also includes protocol-level unit coverage for:

- SSE framing with split chunks and multi-line `data:` payloads
- success vs failure stream termination semantics
- multiple tool-call assembly for Chat Completions and Anthropic streaming
- fail-fast invalid profile header handling
- runtime-record non-leakage into provider-visible transcripts

## Migration Notes

See [`MIGRATION.md`](./MIGRATION.md) for breaking API changes including:

- removal of `InferenceRequest.stream`
- replacement of transcript-only request state with `InferenceContext`
- removal of `Message::SystemStructured`

## Dependency Notes

`async-openai` was upgraded to `0.35`.

The planned direct `reqwest` upgrade to `0.13` is currently deferred because
`async-openai 0.35` still depends on `reqwest 0.12`, which would otherwise
introduce incompatible `reqwest::Client` types into the dependency graph.

## License

This project is licensed under the Apache License 2.0. See `LICENSE-APACHE`.
