## Context

The crate currently models provider auth as:

- `RuntimeConfig { api_key: String, connect_timeout, read_timeout }`
- `ProviderProfile { auth_strategy: AuthStrategy, ... }`
- `AuthStrategy::{BearerToken, ApiKeyHeader, Custom}`
- `build_http_client(HttpClientParams { api_key, auth_strategy, ... })`

This works for API-key-only providers, but not for providers that support multiple credential kinds with different wire headers. The clearest example is `kimi-code`:

- API key uses `x-api-key: <api_key>`.
- OAuth access token uses `Authorization: Bearer <access_token>`.

The same single-string shape also blurs responsibilities. `iron-providers` should not own OAuth refresh tokens or refresh flows. It should receive a current credential snapshot from `iron-core` or the application layer and validate/use that snapshot for provider requests.

Codex is the other initial target. Research from OpenAI Codex CLI and Goose shows that ChatGPT/Codex access uses `https://chatgpt.com/backend-api/codex/responses` with OAuth bearer auth. It is Responses-like, but it is not the public OpenAI Responses API and not Chat Completions. Therefore the existing `openai` provider should remain unchanged and Codex should be represented separately.

## Goals / Non-Goals

**Goals:**

- Preserve existing API-key behavior and `RuntimeConfig::new(api_key)`.
- Represent runtime credentials explicitly as API key or OAuth bearer.
- Allow provider profiles to declare supported credential kinds and per-kind auth strategies.
- Fail clearly for missing, unsupported, or expired credentials.
- Support `kimi-code` API-key and OAuth bearer modes with different headers.
- Add `codex` as a distinct provider with `ApiFamily::CodexResponses` and `models_dev_id = "openai"`.
- Implement Codex exact URL, headers, and request body fields with mocked HTTP tests.
- Extract Codex `chatgpt-account-id` routing metadata from unverified JWT payload claims when available.

**Non-Goals:**

- Implement OAuth login, browser/device-code/loopback flows, or refresh-token exchange.
- Store refresh tokens or other long-lived OAuth secrets.
- Automatically refresh and retry after `401 Unauthorized`.
- Change the existing public `openai` provider behavior.
- Add OAuth to general `kimi`.
- Hard-code Codex model lists from Goose or Codex CLI.
- Add Goose's `gpt-5.3-codex` tool preamble in this crate.

## Decisions

### 1. Model credential kind separately from wire auth strategy

Add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialKind {
    ApiKey,
    OAuthBearer,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderCredential {
    ApiKey(String),
    OAuthBearer {
        access_token: String,
        expires_at: Option<SystemTime>,
        id_token: Option<String>,
    },
}
```

`AuthStrategy::BearerToken` remains the wire-header strategy. It does not imply that the credential is an OAuth token; it only means "place the selected credential value in `Authorization: Bearer ...`."

Why this decision:

- `kimi-code` proves credential kind and wire strategy are different axes.
- Profiles need to say which credential kinds they support.
- Consumers need explicit credential mode selection without overloading API-key strings.

### 2. Preserve API-key construction and add credential-aware construction

Keep:

```rust
RuntimeConfig::new(api_key)
```

Add:

```rust
RuntimeConfig::from_credential(credential)
```

Internally, `RuntimeConfig` should store `ProviderCredential` instead of only `api_key`, while preserving timeout fields and builder methods.

Implementation sketch:

```rust
pub struct RuntimeConfig {
    pub credential: ProviderCredential,
    pub connect_timeout: Option<Duration>,
    pub read_timeout: Option<Duration>,
}

impl RuntimeConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::from_credential(ProviderCredential::ApiKey(api_key.into()))
    }

    pub fn from_credential(credential: ProviderCredential) -> Self { ... }
}
```

The old public `api_key` field cannot remain as the source of truth if the crate supports non-API-key credentials. If there is no external persistence format depending on struct field names, prefer the simpler direct field migration over compatibility scaffolding. If tests or downstream usage reveal a real compatibility need, add targeted helpers rather than duplicating secret state.

### 3. Add per-credential profile auth metadata

Replace or supersede the single `ProviderProfile.auth_strategy` with:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialAuthConfig {
    pub kind: CredentialKind,
    pub auth_strategy: AuthStrategy,
}

pub struct ProviderProfile {
    // existing fields...
    pub credential_auth: Vec<CredentialAuthConfig>,
}
```

Add convenience builders:

```rust
ProviderProfile::with_auth(strategy) // keep as API-key-oriented compatibility helper
ProviderProfile::with_credential_auth(kind, strategy)
ProviderProfile::supports_credential(kind) -> bool
ProviderProfile::auth_strategy_for(kind) -> Option<&AuthStrategy>
```

Recommended semantics:

- `ProviderProfile::new(...)` defaults to API-key bearer auth, matching today's default.
- `with_auth(strategy)` sets or replaces the API-key auth config so existing built-in setup remains readable.
- `with_credential_auth(kind, strategy)` adds or replaces that specific credential kind's auth config.

Why this decision:

- It keeps common API-key-only providers concise.
- It supports `kimi-code` mixed headers without duplicate provider slugs.
- It avoids treating credential mode as provider identity.

### 4. Validate selected credentials early and clearly

Add helper constructors or messages on `ProviderError::Authentication` for:

- missing credential value
- unsupported credential kind for provider
- expired credential

Use authentication errors, not invalid request errors, for credential problems.

Validation rules:

- API key must not be blank.
- OAuth access token must not be blank.
- OAuth `expires_at <= SystemTime::now()` fails before request execution.
- OAuth `expires_at: None` is allowed.
- Provider construction fails when the credential kind is not supported by the selected profile.

No provider adapter should perform refresh or hidden 401 recovery.

### 5. Apply auth from selected credential and selected strategy

Refactor shared auth header creation away from `api_key: &str` toward selected credential value plus auth strategy.

Implementation shape can be either:

- keep auth in client default headers at construction time using the selected credential, or
- build a base client without auth and apply auth per request.

Prefer request-time application for adapters where it is simple, especially Codex, because it keeps refreshed credential snapshots easier to reason about. It is acceptable for construction to validate credentials and for callers to rebuild providers after refresh.

Important: keep timeout handling and fixed/default header validation centralized so behavior does not drift between adapters.

### 6. Add `kimi-code` OAuth without changing `kimi`

`kimi-code` should support:

```text
CredentialKind::ApiKey      -> AuthStrategy::ApiKeyHeader { header_name: "x-api-key" }
CredentialKind::OAuthBearer -> AuthStrategy::BearerToken
```

General `kimi` remains API-key-only:

```text
CredentialKind::ApiKey -> AuthStrategy::BearerToken
```

Why this decision:

- Kimi CLI showed Kimi Code OAuth access tokens are sent as bearer tokens.
- Current `kimi-code` API-key behavior uses `x-api-key` and must continue working.
- Adding a `kimi-code-oauth` slug would conflate provider identity with credential mode and complicate settings/model lookup.

### 7. Add Codex as a new API family and provider profile

Add:

```rust
ApiFamily::CodexResponses
```

Register built-in profile:

```text
slug: codex
models_dev_id: openai
family: CodexResponses
base_url: https://chatgpt.com/backend-api/codex
credential: OAuthBearer -> BearerToken
purpose: Coding
```

Do not add OAuth behavior to standard `openai` as part of this change.

Why this decision:

- Codex uses ChatGPT/Codex backend endpoints, not public OpenAI endpoints.
- `models_dev_id = "openai"` lets consumers reuse OpenAI/GPT capability metadata because models.dev has no Codex provider.
- Separate family keeps exact Codex request requirements testable without bending existing OpenAI adapters.

### 8. Implement Codex with raw HTTP and exact mocked assertions

Add `src/codex.rs` with non-streaming and streaming functions matching the `Provider` dispatch pattern.

Codex requests target:

```text
POST https://chatgpt.com/backend-api/codex/responses
```

Required request headers:

```text
Authorization: Bearer <access_token>
originator: iron-providers
User-Agent: iron-providers/<crate-version>
chatgpt-account-id: <account_id>   # only when extracted
Content-Type: application/json
```

Required body fields:

```text
model: <request model>
input: <projected transcript/input>
instructions: <request instructions when present>
tools: <tool definitions when present>
tool_choice: <tool policy projection when present>
store: false
reasoning: { effort: "medium" }
parallel_tool_calls: true
stream: true/false according to infer vs infer_stream
```

Reuse projection helpers from `openai` or extract shared Responses projection code only if it reduces duplication without coupling Codex to public OpenAI behavior. Do not introduce a large abstraction before tests prove duplication is harmful.

### 9. Parse JWT payload claims for Codex routing metadata only

Add a helper in a credential/auth module, for example:

```rust
pub(crate) fn chatgpt_account_id_from_jwt(id_token: &str) -> Option<String>
```

Parse JWT payloads unverified. This is only routing metadata extraction; provider servers still validate bearer tokens.

Claim priority:

1. top-level `chatgpt_account_id`
2. nested `https://api.openai.com/auth.chatgpt_account_id`
3. first `organizations[].id`

Implementation note: if the crate does not already depend on a base64 decoder, add a small, focused dependency only if needed. Keep parsing tolerant: malformed tokens return `None` instead of failing provider construction.

### 10. Keep model capability source external

Codex profile should use `models_dev_id = "openai"`. The crate should not hard-code Goose's Codex model list or infer image capability internally. Consumers should derive model capabilities from models.dev metadata.

## Risks / Trade-offs

- Changing `RuntimeConfig` from `api_key` to `credential` is a public API change. Mitigation: keep `RuntimeConfig::new(api_key)` and update crate-local tests/examples; add compatibility only if real downstream usage requires it.
- `async-openai` client construction currently expects API-key-like config for `OpenAiResponses`. Mitigation: standard `openai` remains API-key-only for this issue; Codex uses raw HTTP.
- Request-time auth and client default auth may coexist temporarily. Mitigation: centralize auth header construction so semantics stay consistent.
- Unverified JWT parsing can be misunderstood as auth validation. Mitigation: keep helper naming/docs explicit that it extracts non-authoritative routing metadata only.
- Codex response/stream formats may drift. Mitigation: isolate in `src/codex.rs` and cover exact request behavior with mocked HTTP; response parsing can reuse existing event semantics where compatible.

## Migration Plan

1. Add credential types and preserve `RuntimeConfig::new(api_key)` behavior.
2. Add profile credential metadata while keeping built-in API-key profiles equivalent.
3. Update generic provider construction to select auth strategy by credential kind.
4. Add Kimi Code OAuth support.
5. Add Codex family/profile/adapter.
6. Add tests for compatibility, errors, Kimi headers, Codex requests, and JWT parsing.

Rollback is straightforward if done before release: remove Codex family/profile, remove credential-aware constructor, and revert profiles to single `auth_strategy`. After release, prefer additive fixes because the credential API becomes public.

## Open Questions

- Should `ProviderCredential` derive `Serialize`/`Deserialize`? The provider crate likely does not need to persist credentials, and avoiding serde may reduce accidental secret serialization. Add serde only if tests or known consumers need it.
- Should `ProviderError` gain distinct enum variants or just helper constructors returning `Authentication`? Prefer helper constructors unless consumers require machine-readable subtypes.
- Should auth headers move fully to request-time for all adapters now, or only where required? Prefer the smallest safe refactor that supports selected credential kinds and keeps tests clear.
