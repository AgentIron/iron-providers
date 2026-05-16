## Context

The current architecture mixes several responsibilities:

- Provider identity is represented in API-family names such as `OpenAiResponses`, `OpenAiChatCompletions`, `AnthropicMessages`, and `CodexResponses`.
- `GenericProvider` validates credentials, selects auth strategy, builds HTTP clients, constructs an `async-openai` client for one family, and dispatches to adapters.
- `http_client.rs` both derives auth headers from credentials/strategies and assembles `reqwest::Client` transport.
- Provider-specific behavior such as Anthropic version headers, Codex account routing, and Codex fixed body fields lives in adapter-specific locations.
- OpenAI Responses and Codex are both OpenAI-backed Responses-like integrations, but they are handled as special cases through different top-level modules and dispatch branches.

The desired shape is a cleaner handoff from provider profile plus runtime credential into an immutable provider connection, then from that connection into API-type adapters.

Core design statement:

> `iron-providers` SHALL separate provider identity from API protocol behavior. Provider profiles describe provider identity and static configuration. `ProviderConnection` resolves a profile plus runtime credential into immutable connection state. API adapters under `src/apis/` implement protocol-specific request projection, response parsing, and streaming normalization. Auth header generation, provider overrides, and HTTP client construction are separate responsibilities with explicit handoff boundaries.

## Goals / Non-Goals

**Goals:**

- Make the architecture easier to maintain, read, and document.
- Make `ApiFamily` name API protocols rather than provider brands.
- Remove OpenAI/Codex special-case handling from top-level provider dispatch.
- Centralize all provider-specific static configuration and overrides.
- Separate credential/auth handling, provider override resolution, API request/response logic, and HTTP transport assembly.
- Provide a clean `ProviderConnection` handoff for `iron-core` while preserving the `ProviderRegistry::get(...) -> Box<dyn Provider>` consumption model.
- Accept a hard public API break where it improves the architecture.

**Non-Goals:**

- Preserve `OpenAiProvider`, `OpenAiConfig`, `OpenAiConfigSource`, or public `GenericProvider` compatibility wrappers.
- Preserve serialized `ApiFamily` variant names.
- Add new provider functionality beyond preserving existing provider behavior under the new structure.
- Add a separate Codex wire dialect. Codex should use the same upstream OpenAI Responses wire contract, with provider-specific configuration supplied as overrides.

## Target Shape

```text
ProviderRegistry
  slug + RuntimeConfig
        |
        v
ProviderProfile
        |
        v
ProviderConnection::from_profile(...)
  - validate runtime credential
  - resolve auth strategy
  - build auth headers
  - resolve provider overrides
  - compose protected/final headers
  - build reqwest client
  - select API adapter by ApiFamily
        |
        v
Provider trait methods
        |
        v
src/apis/{messages,completions,responses}.rs
```

## Decisions

### 1. `ApiFamily` names API protocols, not providers

Use exactly these variants:

```rust
pub enum ApiFamily {
    Responses,
    Completions,
    Messages,
}
```

Provider identity remains in `ProviderProfile.slug`, `models_dev_id`, base URL, auth metadata, purpose, quirks, and provider overrides. Provider-specific variants such as `OpenAiResponses`, `OpenAiChatCompletions`, `AnthropicMessages`, and `CodexResponses` should be removed.

Why this decision:

- OpenAI and Anthropic are providers, not API-family concepts.
- Codex is differentiated by provider overrides on the Responses API, not by a separate family.
- New provider profiles can be added by declaring which protocol they speak plus their overrides.

### 2. `ProviderConnection` is the resolved provider state

Add a public `ProviderConnection` type that implements `Provider` and is the only concrete connection type the registry constructs.

`ProviderRegistry::get(...)` should continue returning `Box<dyn Provider>` for `iron-core`, but the boxed implementation should be `ProviderConnection`. Direct users may construct `ProviderConnection` from a `ProviderProfile` and `RuntimeConfig`.

`ProviderConnection` should be public but does not need to be re-exported from the prelude unless a concrete downstream need appears.

Why this decision:

- It gives `iron-core` a clean trait-object handoff.
- It gives advanced users one public concrete connection boundary.
- It replaces the split between `GenericProvider` and `OpenAiProvider` with one concept.

### 3. Provider-specific configuration is centralized

Introduce `provider_overrides.rs` as the only place that resolves provider-specific static behavior from `ProviderProfile` and `RuntimeConfig`.

Override declaration should use a hybrid model: simple provider identity remains on `ProviderProfile`, while `provider_overrides.rs` resolves typed override behavior from that profile. `ProviderOverrides` should remain crate-private for now unless external custom provider construction later needs typed override access.

Provider-specific behavior currently appears to be limited to auth and headers plus Codex fixed request-body fields. The override model should support endpoint, header, routing, and fixed body configuration while keeping Codex on the standard upstream OpenAI Responses wire contract.

Conceptual shape:

```rust
pub enum ProviderOverrides {
    None,
    Messages(MessagesOverrides),
    Completions(CompletionsOverrides),
    Responses(ResponsesOverrides),
}
```

Responses overrides should cover Codex-specific concerns such as endpoint kind, account routing header, fixed body fields (`store`, `reasoning`, `parallel_tool_calls`), and product headers. They should not define a separate Codex stream dialect unless implementation evidence proves the upstream Responses contract is incompatible.

Why this decision:

- Provider-specific logic has one home.
- API adapters consume resolved configuration rather than rediscovering it.
- `http_client.rs` and registry construction do not accumulate provider branches.

### 4. Auth header production is separate from HTTP client construction

Add `auth.rs` to map `ProviderCredential` plus `AuthStrategy` to auth headers. `http_client.rs` should receive a final `HeaderMap` and timeout values, then only validate/apply headers and build `reqwest::Client`.

Boundary:

```text
auth.rs
  ProviderCredential + AuthStrategy -> auth HeaderMap

provider_overrides.rs
  ProviderProfile + RuntimeConfig -> override headers/body/stream settings

connection.rs
  auth headers + protocol headers + profile default headers + overrides -> final HeaderMap

http_client.rs
  final HeaderMap + timeouts -> reqwest::Client
```

Why this decision:

- `http_client.rs` still receives auth headers, but it does not know how credentials become headers.
- Auth can be tested independently from transport construction.
- Provider override headers can be composed with auth headers under one collision policy.

### 5. Protected header collisions fail clearly

Auth and required protocol/provider headers must not be silently overridden by profile default headers.

Final headers should be composed in this order:

```text
protocol required headers
+ auth headers
+ provider override headers
+ profile default headers
```

Profile default headers may add non-protected headers only. If they collide with any protected header, construction fails.

Protected headers should include at least auth headers, `Content-Type`, and required protocol/provider headers such as `anthropic-version`, `originator`, `User-Agent`, and `chatgpt-account-id` when present.

If a profile attempts to override a protected header through generic default headers, construction should fail with a configuration error. Intentional provider-specific differences should be modeled in `ProviderOverrides`, not hidden in profile defaults.

Why this decision:

- Silent auth/header override is a security and debugging footgun.
- The override mechanism remains explicit and reviewable.

### 6. API adapters live under `src/apis/`

Move protocol implementations to:

```text
src/apis/messages.rs      # Anthropic Messages-style protocol
src/apis/completions.rs   # OpenAI Chat Completions-compatible protocol
src/apis/responses.rs     # OpenAI Responses-compatible protocol, including Codex mode
```

The adapters should receive uniform resolved inputs from `ProviderConnection`, not perform profile/runtime construction. They own request projection, response parsing, stream parsing, and normalized `ProviderEvent` production for their protocol.

Why this decision:

- The module layout matches the mental model users need: API protocol first, provider identity second.
- Provider-specific differences are data/configuration consumed by the adapter.

### 7. OpenAI Responses and Codex share the Responses adapter and upstream wire contract

Codex should be treated as a provider configuration of the upstream OpenAI Responses API, not as a separate wire protocol.

The Responses adapter should support public OpenAI and Codex through one protocol module, one request/response projection, and one stream parser for the upstream Responses wire contract. Codex differences should be represented as `ResponsesOverrides` for provider configuration: endpoint path, headers, fixed body fields, account routing, and error context where needed.

The expected Codex differences are provider configuration details, not wire dialect details:

- endpoint path and base URL conventions
- fixed request-body fields such as `store`, `reasoning`, and `parallel_tool_calls`
- product/routing headers such as `originator`, `User-Agent`, and `chatgpt-account-id`
- provider-specific error context or metadata if the upstream endpoint returns additional fields

There should not be a separate `CodexResponses` adapter or public/internal Codex stream dialect by default. If implementation discovers that Codex's upstream endpoint actually emits incompatible stream events, treat that as a discovered upstream incompatibility and update this design before adding dialect-specific parsing.

Why this decision:

- It removes top-level OpenAI/Codex special cases.
- It aligns the implementation with the upstream API OpenAI provides.
- It keeps Codex-specific behavior limited to provider configuration rather than protocol branching.

### 8. Remove `async-openai`

The Responses adapter should use `reqwest` directly with crate-owned request/response structs or raw JSON projections. Use typed request structs for fields this crate owns, and tolerant partial structs or `serde_json::Value` for responses where providers may diverge. Remove `async-openai` and the public OpenAI-specific configuration/connection types.

Why this decision:

- One transport stack is easier to reason about.
- The crate owns exact request and stream behavior for all provider profiles.
- OpenAI Responses can participate in the same `ProviderConnection` and override model as every other provider.

### 9. Hard public API break is acceptable

Remove or stop publicly exporting:

- `OpenAiProvider`
- `OpenAiConfig`
- `OpenAiConfigSource`
- public `GenericProvider`
- provider-branded `ApiFamily` variants

Document migration to `ProviderConnection`, `ProviderProfile`, `RuntimeConfig`, and `ProviderRegistry`.

Do not add backward-compatible deserialization aliases for old `ApiFamily` variant names. The hard break is intentional and should be documented rather than hidden with compatibility shims.

Why this decision:

- Compatibility wrappers would preserve the confusing architecture the refactor is meant to remove.
- The crate is still early enough that a clean break is preferable to long-lived duplicate concepts.

## Risks / Trade-offs

- Removing `async-openai` requires faithfully replacing request projection, response parsing, error mapping, and streaming behavior. Mitigation: add request/response parity tests and mocked HTTP coverage before removing the old path.
- Codex may require endpoint, header, routing, or fixed-body differences from a standard public OpenAI profile. Mitigation: keep those differences as Responses overrides while preserving one upstream Responses request/response/stream contract.
- Renaming `ApiFamily` variants breaks serialized profiles and source users. Mitigation: document the hard break and update all built-ins/tests/docs in one change.
- Centralizing provider overrides can become a dumping ground. Mitigation: keep overrides static, typed, and scoped by API family.
- Header collision rules can reject profiles that previously worked accidentally. Mitigation: surface clear configuration errors and require intentional provider differences to be modeled as overrides.

## Migration Plan

1. Add `ApiFamily::{Responses, Completions, Messages}` and update built-in profiles/tests.
2. Add `auth.rs`, `provider_overrides.rs`, and pure `http_client.rs` interfaces.
3. Add `ProviderConnection` construction and switch the registry to box it.
4. Move current adapters under `src/apis/` while preserving behavior.
5. Replace the `async-openai` Responses implementation with reqwest-backed request/response/stream parsing.
6. Fold Codex into `apis/responses.rs` behind `ResponsesOverrides`.
7. Remove old public exports/modules and update docs.
8. Run full build, lint, and test verification.

Rollback before release is possible by reverting to `GenericProvider`, old `ApiFamily` variants, and `async-openai`. After release, prefer forward fixes because the public API break will be visible to downstream users.

## Resolved Implementation Decisions

- Override declaration uses the hybrid model: profiles carry provider identity and simple static data; `provider_overrides.rs` resolves typed behavior.
- `ProviderOverrides` remains crate-private for now.
- `ProviderConnection::from_profile(profile, runtime)` is the preferred constructor shape. Add a builder only if construction options grow.
- `ProviderConnection` owns resolved state. It may consume or clone the source profile during construction, but it should not borrow profile/runtime data.
- Header composition order is protocol required headers, auth headers, provider override headers, then profile default headers. Profile defaults cannot override protected headers.
- Required protocol headers are represented as resolved headers from adapter defaults plus provider overrides, then marked protected before profile defaults are applied.
- Responses request projection uses typed structs for crate-owned request fields and tolerant partial structs or raw JSON for provider-divergent responses.
- Codex uses the common upstream Responses adapter contract. Codex-specific overrides configure endpoint, headers, routing, and fixed request fields only; no separate wire dialect is planned.
- No backward-compatible aliases are needed for old serialized `ApiFamily` names.
- Construction failures should surface clear `ProviderError` variants or structured messages for unsupported credentials, expired credentials, invalid auth/header values, protected header collisions, invalid profile config, and unsupported overrides.
- `ProviderConnection` is exported at the crate root but not from the prelude initially.
- The minimum parity test matrix is public OpenAI Responses non-stream output, public OpenAI Responses stream text deltas, public OpenAI Responses tool/function calls if supported today, Codex non-stream output, Codex streaming through the same Responses stream parser, Chat Completions stream tool-call assembly, Messages auth/header behavior, protected header collision failure, and all built-in profiles constructing successfully.
