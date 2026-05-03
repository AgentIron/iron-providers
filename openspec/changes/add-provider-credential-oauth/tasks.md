## 1. Credential Model

- [x] 1.1 Add `CredentialKind` with `ApiKey` and `OAuthBearer` variants.
- [x] 1.2 Add `ProviderCredential` with `ApiKey(String)` and `OAuthBearer { access_token, expires_at, id_token }` variants.
- [x] 1.3 Update `RuntimeConfig` to store `ProviderCredential` while preserving connect/read timeout fields.
- [x] 1.4 Preserve `RuntimeConfig::new(api_key)` as API-key shorthand.
- [x] 1.5 Add `RuntimeConfig::from_credential(credential)`.
- [x] 1.6 Update `RuntimeConfigSource` call sites/tests for the credential-backed runtime shape.

## 2. Profile Auth Metadata

- [x] 2.1 Add `CredentialAuthConfig { kind, auth_strategy }`.
- [x] 2.2 Replace or supersede `ProviderProfile.auth_strategy` with `credential_auth: Vec<CredentialAuthConfig>`.
- [x] 2.3 Keep `ProviderProfile::new(...)` default behavior equivalent to API-key bearer auth.
- [x] 2.4 Keep `ProviderProfile::with_auth(strategy)` as an API-key auth helper for existing built-ins.
- [x] 2.5 Add `ProviderProfile::with_credential_auth(kind, strategy)`.
- [x] 2.6 Add helper methods for supported credential lookup, such as `supports_credential` and `auth_strategy_for`.
- [x] 2.7 Update all built-in registry profiles to preserve existing API-key auth behavior.

## 3. Auth Validation And Header Application

- [x] 3.1 Add clear authentication error helpers/messages for missing, unsupported, and expired credentials.
- [x] 3.2 Validate blank API keys and blank OAuth access tokens as authentication failures.
- [x] 3.3 Validate unsupported credential kind during provider construction.
- [x] 3.4 Validate expired OAuth bearer credentials before requests or during construction.
- [x] 3.5 Allow OAuth bearer credentials with `expires_at: None`.
- [x] 3.6 Refactor shared auth header construction to use selected credential value plus selected `AuthStrategy` rather than `api_key` plus single profile strategy.
- [x] 3.7 Preserve timeout, default header, and fixed header validation behavior.

## 4. Kimi Code OAuth Support

- [x] 4.1 Update `kimi-code` built-in profile to support API-key auth with `x-api-key`.
- [x] 4.2 Add OAuth bearer auth support to `kimi-code` with `Authorization: Bearer <access_token>`.
- [x] 4.3 Ensure general `kimi` remains API-key-only.
- [x] 4.4 Add tests for `kimi-code` API-key header behavior.
- [x] 4.5 Add tests for `kimi-code` OAuth bearer header behavior.
- [x] 4.6 Add tests that `kimi` rejects OAuth bearer credentials.

## 5. Codex Provider Profile And Dispatch

- [x] 5.1 Add `ApiFamily::CodexResponses`.
- [x] 5.2 Add `src/codex.rs` module and export it from `src/lib.rs` as appropriate.
- [x] 5.3 Register built-in `codex` profile with base URL `https://chatgpt.com/backend-api/codex`.
- [x] 5.4 Set `codex` profile `models_dev_id` to `openai`.
- [x] 5.5 Set `codex` profile purpose to coding.
- [x] 5.6 Configure `codex` to support OAuth bearer credentials.
- [x] 5.7 Ensure `codex` rejects API-key credentials unless explicit API-key support is added later.
- [x] 5.8 Update `GenericProvider` non-streaming dispatch for `CodexResponses`.
- [x] 5.9 Update `GenericProvider` streaming dispatch for `CodexResponses`.

## 6. Codex Request Implementation

- [x] 6.1 Build Codex non-streaming requests to `POST {base_url}/responses` with `stream: false`.
- [x] 6.2 Build Codex streaming requests to `POST {base_url}/responses` with `stream: true`.
- [x] 6.3 Include `store: false` in every Codex request body.
- [x] 6.4 Include `reasoning: { effort: "medium" }` in every Codex request body.
- [x] 6.5 Include `parallel_tool_calls: true` in every Codex request body.
- [x] 6.6 Project model, instructions, transcript/input, tools, and tool choice from `InferenceRequest`.
- [x] 6.7 Include `originator: iron-providers` on Codex requests.
- [x] 6.8 Include `User-Agent: iron-providers/<crate-version>` on Codex requests.
- [x] 6.9 Include `chatgpt-account-id` only when account metadata is extracted.
- [x] 6.10 Map Codex unauthorized responses to authentication errors without refresh attempts.
- [x] 6.11 Implement response parsing into existing `ProviderEvent` semantics for non-streaming and streaming paths.

## 7. JWT Account Metadata

- [x] 7.1 Add a focused helper for unverified JWT payload parsing.
- [x] 7.2 Extract top-level `chatgpt_account_id` with highest priority.
- [x] 7.3 Extract nested `https://api.openai.com/auth.chatgpt_account_id` with second priority.
- [x] 7.4 Extract first `organizations[].id` as fallback.
- [x] 7.5 Return `None` for malformed JWTs, invalid payload JSON, or missing account metadata.
- [x] 7.6 Document that JWT parsing is routing metadata extraction, not auth verification.

## 8. Tests

- [x] 8.1 Add tests for `RuntimeConfig::new` preserving API-key behavior.
- [x] 8.2 Add tests for `RuntimeConfig::from_credential` with API-key and OAuth bearer credentials.
- [x] 8.3 Add tests for profile credential support lookup and per-kind auth strategy selection.
- [x] 8.4 Add tests for missing, unsupported, and expired credential errors.
- [x] 8.5 Add mocked HTTP tests for existing API-key providers to catch auth header regressions where practical.
- [x] 8.6 Add mocked raw HTTP tests for `kimi-code` API-key and OAuth headers.
- [x] 8.7 Add mocked raw HTTP tests for Codex URL, headers, and required body fields.
- [x] 8.8 Add tests for Codex account ID header inclusion and omission.
- [x] 8.9 Add JWT parsing tests using fake JWT payloads for each claim priority and malformed input.
- [x] 8.10 Add registry tests confirming `codex` is registered and maps to `models_dev_id = "openai"`.
- [x] 8.11 Add dispatch tests or mocked integration tests confirming `ApiFamily::CodexResponses` routes through the Codex adapter.

## 9. Documentation And Examples

- [x] 9.1 Update README or crate docs to show API-key construction still works.
- [x] 9.2 Add a credential-aware construction example with `ProviderCredential::OAuthBearer`.
- [x] 9.3 Document that refresh tokens and OAuth refresh flows are owned by consuming layers, not `iron-providers`.
- [x] 9.4 Document `kimi-code` dual auth modes and `codex` OAuth-only behavior.

## 10. Verification

- [x] 10.1 Run `cargo build --locked --all-targets`.
- [x] 10.2 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 10.3 Run `cargo clippy --locked --all-targets --all-features -- -D warnings`.
- [x] 10.4 Run `cargo test --locked`.
