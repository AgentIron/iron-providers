## 1. API Family And Public Surface

- [x] 1.1 Rename `ApiFamily` variants to `Responses`, `Completions`, and `Messages`.
- [x] 1.2 Update all built-in profiles to use API-type family names.
- [x] 1.3 Remove provider-branded API-family variants including `OpenAiResponses`, `OpenAiChatCompletions`, `AnthropicMessages`, and `CodexResponses`.
- [x] 1.4 Remove public `OpenAiProvider`, `OpenAiConfig`, and `OpenAiConfigSource` exports.
- [x] 1.5 Remove `GenericProvider` as a public abstraction.
- [x] 1.6 Add public `ProviderConnection` export without adding it to the prelude unless needed.

## 2. Auth And Header Boundaries

- [x] 2.1 Add `src/auth.rs` for mapping `ProviderCredential` plus `AuthStrategy` to auth headers.
- [x] 2.2 Move auth-strategy matching out of `http_client.rs`.
- [x] 2.3 Change `http_client.rs` to accept a final `HeaderMap` plus timeout values.
- [x] 2.4 Add protected-header collision detection during header composition using the order: protocol required headers, auth headers, provider override headers, then profile default headers.
- [x] 2.5 Cover bearer, API-key header, custom header, invalid header, and protected collision cases with tests.

## 3. Provider Overrides

- [x] 3.1 Add crate-private `src/provider_overrides.rs` with typed overrides scoped by API family and resolved from provider profile data.
- [x] 3.2 Resolve Anthropic Messages headers through provider overrides.
- [x] 3.3 Resolve Codex product headers and account routing through Responses overrides.
- [x] 3.4 Resolve Codex fixed body fields through Responses overrides.
- [x] 3.5 Ensure provider-specific override logic is not duplicated in adapters, registry, or `http_client.rs`.

## 4. ProviderConnection

- [x] 4.1 Add `src/connection.rs` with public `ProviderConnection`.
- [x] 4.2 Implement `ProviderConnection::from_profile(profile, runtime)` as the direct constructor.
- [x] 4.3 Validate runtime credentials and supported credential kind during construction.
- [x] 4.4 Resolve auth headers, provider overrides, final headers, and reqwest client during construction.
- [x] 4.5 Implement `Provider` for `ProviderConnection`.
- [x] 4.6 Update `ProviderRegistry::get()` to return `Box<dyn Provider>` backed by `ProviderConnection`.
- [x] 4.7 Remove `GenericProvider` construction paths once behavior is covered.

## 5. API Adapter Reorganization

- [x] 5.1 Add `src/apis/mod.rs`.
- [x] 5.2 Move Anthropic Messages behavior to `src/apis/messages.rs`.
- [x] 5.3 Move Chat Completions behavior to `src/apis/completions.rs`.
- [x] 5.4 Move OpenAI Responses behavior to `src/apis/responses.rs`.
- [x] 5.5 Keep adapters focused on request projection, response parsing, streaming normalization, and `ProviderEvent` mapping.
- [x] 5.6 Update module exports and tests for the new layout.

## 6. Responses Adapter Unification

- [x] 6.1 Replace `async-openai` Responses requests with reqwest-backed request construction.
- [x] 6.2 Add typed crate-owned request structs for public OpenAI Responses request fields owned by this crate.
- [x] 6.3 Add tolerant crate-owned response parsing for public OpenAI Responses non-streaming responses using partial structs or raw JSON where providers may diverge.
- [x] 6.4 Add crate-owned stream parsing for public OpenAI Responses typed SSE events.
- [x] 6.5 Fold Codex request construction into `apis/responses.rs` behind Responses overrides.
- [x] 6.6 Fold Codex response parsing into `apis/responses.rs` behind Responses overrides where needed.
- [x] 6.7 Ensure Codex uses the same upstream Responses request/response/stream parser as public OpenAI Responses, with overrides limited to endpoint, headers, routing, and fixed request body fields.
- [x] 6.8 Remove `src/openai.rs` and `src/codex.rs` as special-case top-level adapter modules once replacement behavior is covered.
- [x] 6.9 Remove `async-openai` from `Cargo.toml`.

## 7. Registry, Profiles, And Documentation

- [x] 7.1 Update registry built-ins for the new `ApiFamily` names and provider override model.
- [x] 7.2 Preserve existing provider slugs and credential behavior unless explicitly changed by this proposal.
- [x] 7.3 Update README/crate docs to describe provider identity vs API family vs provider overrides.
- [x] 7.4 Add migration notes for removed OpenAI-specific public types and `GenericProvider`.
- [x] 7.5 Document `ProviderConnection` as the clean direct-construction path and registry-backed handoff for `iron-core`.
- [x] 7.6 Document that old serialized `ApiFamily` names intentionally have no compatibility aliases.

## 8. Verification

- [x] 8.1 Add/port mocked request tests for Messages, Completions, public OpenAI Responses, and Codex Responses.
- [x] 8.2 Add streaming tests proving public OpenAI Responses and Codex both use the common Responses stream parser.
- [x] 8.3 Add registry tests confirming all built-in profiles construct `ProviderConnection` correctly.
- [x] 8.4 Add tests proving protected headers cannot be silently overridden by profile default headers.
- [x] 8.5 Add compile-facing tests or examples for direct `ProviderConnection` construction.
- [x] 8.6 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 8.7 Run `cargo clippy --locked --all-targets --all-features -- -D warnings`.
- [x] 8.8 Run `cargo test --locked`.
- [x] 8.9 Run `cargo build --locked --all-targets`.
