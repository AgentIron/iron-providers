## Why

`iron-providers` currently treats every runtime secret as one long-lived API-key string. `RuntimeConfig` stores `api_key`, `ProviderProfile` stores a single `auth_strategy`, and `GenericProvider` builds clients with that one strategy at construction time. That is too narrow for OAuth-backed provider access.

The first concrete providers that need this are `kimi-code` and a new ChatGPT/Codex-backed `codex` provider. `kimi-code` supports both API keys and OAuth bearer access tokens, but those credentials use different wire headers. Codex is not the public OpenAI API and should not change the existing `openai` provider; it uses ChatGPT/Codex endpoints with OAuth bearer auth.

OAuth login, refresh-token storage, access-token refresh, and 401 retry orchestration belong in AgentIron, `iron-tui`, and `iron-core`. This crate should own only the provider-facing credential contract, supported credential metadata, request-time auth headers, validation, and provider-specific connection details.

## What Changes

- Add explicit credential types: `CredentialKind` and `ProviderCredential`.
- Preserve `RuntimeConfig::new(api_key)` for existing API-key callers and add `RuntimeConfig::from_credential(credential)` for credential-aware callers.
- Add profile metadata that maps supported credential kinds to wire `AuthStrategy` values.
- Validate missing, unsupported, and expired credentials with clear authentication errors.
- Update client/request construction so the selected credential kind determines the auth header.
- Add OAuth bearer support to `kimi-code` while leaving `kimi` API-key-only.
- Add a new `codex` provider profile using `models_dev_id = "openai"` and a new `ApiFamily::CodexResponses`.
- Implement Codex request handling against `https://chatgpt.com/backend-api/codex/responses` using raw mocked HTTP coverage.
- Parse optional Codex account routing metadata from unverified JWT payload claims when an OAuth credential includes an ID token.

## Capabilities

### New Capabilities

- `provider-credential-oauth`: Defines provider runtime credential modeling, profile credential support metadata, OAuth bearer validation, Kimi Code OAuth header behavior, and Codex provider behavior.

### Modified Capabilities

- Existing provider registry/profile behavior gains credential-aware auth selection while preserving API-key compatibility.
- Existing generic provider dispatch gains a Codex Responses family.

## Impact

- Affected code: `src/profile.rs`, `src/generic_provider.rs`, `src/http_client.rs`, `src/registry.rs`, `src/lib.rs`, `src/error.rs`, new credential/JWT helper module, new `src/codex.rs`, and tests.
- Affected behavior: existing API-key providers continue to work; `kimi-code` can use API-key or OAuth bearer credentials; `codex` becomes available as an OAuth-backed provider.
- API impact: additive credential API plus profile metadata changes. Preserve `RuntimeConfig::new(api_key)` and avoid breaking existing API-key construction.
- Out of scope: OAuth login flows, refresh-token storage, refresh scheduling, app UI, `iron-core` retry orchestration, changing standard `openai`, adding OAuth to general `kimi`, and passing refresh tokens into this crate.

## Related

- AgentIron/iron-providers#13
- AgentIron/iron-core#18
- AgentIron/AgentIron#22
- AgentIron/AgentIron#23
