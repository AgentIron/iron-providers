## Why

`iron-providers` has grown provider-specific behavior across several boundaries: `GenericProvider` dispatches by provider-branded API families, `http_client.rs` derives auth headers while also building transport, OpenAI Responses is handled through a special `OpenAiProvider`/`async-openai` path, and Codex is handled through a separate raw HTTP adapter even though it is Responses-like.

This makes the architecture harder to maintain, document, and hand off cleanly to `iron-core`. Provider identity, API protocol behavior, auth resolution, provider-specific headers/body fields, and HTTP transport construction should be separate responsibilities with explicit handoff points.

The refactor should make provider identity configure an API adapter; provider identity should not define the adapter architecture.

## What Changes

- Introduce a public `ProviderConnection` as the resolved provider state behind the `Provider` trait.
- Organize protocol adapters under `src/apis/` by API type: `messages`, `completions`, and `responses`.
- Rename `ApiFamily` variants so they name API protocols, not provider brands: `Messages`, `Completions`, and `Responses`.
- Centralize provider-specific configuration in one override-resolution boundary instead of scattering special cases across adapters and transport.
- Split auth header production from pure HTTP client construction.
- Replace OpenAI/Codex special-case adapters with a single Responses API adapter configured by provider overrides.
- Remove the `async-openai` dependency and use `reqwest` plus crate-owned request/response structs for Responses-family providers.
- Remove the legacy public OpenAI-specific connection API and make `ProviderConnection` the public connection type for direct construction.

## Capabilities

### New Capabilities

- `provider-architecture-refactor`: Defines the provider connection boundary, API-type adapter organization, auth/header separation, provider override resolution, and public API migration.

### Modified Capabilities

- Existing provider registry behavior continues to return `Box<dyn Provider>`, backed by `ProviderConnection` instead of `GenericProvider` or `OpenAiProvider`.
- Existing profile-driven provider behavior changes from provider-branded `ApiFamily` variants to API-type variants.
- Existing OpenAI Responses and Codex behavior moves into one Responses adapter with provider-specific overrides.

## Impact

- Affected code: `src/lib.rs`, `src/provider.rs`, `src/profile.rs`, `src/registry.rs`, `src/http_client.rs`, `src/generic_provider.rs`, `src/openai.rs`, `src/codex.rs`, `src/anthropic.rs`, `src/completions.rs`, tests, docs, and `Cargo.toml`.
- New code: `src/connection.rs`, `src/auth.rs`, `src/provider_overrides.rs`, and `src/apis/{messages,completions,responses}.rs`.
- API impact: hard public API break is acceptable for this change. Remove `OpenAiProvider`, `OpenAiConfig`, `OpenAiConfigSource`, provider-branded `ApiFamily` variants, and the public `GenericProvider` abstraction.
- Dependency impact: remove `async-openai`; keep request projection and response parsing in crate-owned code.
- Downstream impact: `iron-core` should continue consuming `Box<dyn Provider>` through the registry, but direct users of OpenAI-specific public types must migrate to `ProviderConnection` plus `ProviderProfile`/`RuntimeConfig`.

## Out Of Scope

- Adding new provider integrations.
- Adding OAuth login, refresh-token storage, or credential refresh orchestration.
- Changing the normalized `Provider` trait or `InferenceRequest`/`ProviderEvent` model except where required by adapter migration.
- Adding a separate Codex wire dialect. Codex should use the same upstream OpenAI Responses wire contract; Codex-specific behavior should be limited to provider configuration such as endpoint, auth/routing headers, and fixed request fields unless implementation evidence proves an upstream incompatibility.
