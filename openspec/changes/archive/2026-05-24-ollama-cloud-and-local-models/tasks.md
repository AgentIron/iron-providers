## 1. NoAuth Credential Kind

- [x] 1.1 Add `NoAuth` variant to `CredentialKind` enum in `src/profile.rs`.
- [x] 1.2 Add explicit `ProviderCredential::NoAuth` variant in `src/profile.rs`.
- [x] 1.3 Add `RuntimeConfig::none()` convenience constructor in `src/profile.rs` that returns `ProviderCredential::NoAuth`.
- [x] 1.4 Update `ProviderCredential::kind()` and related credential helpers for `NoAuth`.
- [x] 1.5 Update validation/connection construction so `NoAuth` is accepted only when the resolved provider profile supports `CredentialKind::NoAuth`.
- [x] 1.6 Export `CredentialKind::NoAuth`, `ProviderCredential::NoAuth`, and `RuntimeConfig::none()` from the crate root and prelude as needed.

## 2. Auth Adapter NoAuth Handling

- [x] 2.1 Add `AuthStrategy::NoAuth` as the explicit no-auth auth strategy for `CredentialKind::NoAuth`.
- [x] 2.2 Update `src/auth.rs` to return an empty `HeaderMap` when the credential or auth strategy indicates NoAuth.
- [x] 2.3 Verify `auth_headers()` does not error when called with NoAuth configuration.
- [x] 2.4 Add tests for NoAuth producing no auth headers and not failing.
- [x] 2.5 Add tests proving blank API-key credentials still fail for API-key providers.

## 3. Built-in Provider Registrations

- [x] 3.1 Register `ollama-cloud` as a built-in `ApiFamily::Completions` provider targeting `https://api.ollama.cloud` with default BearerToken auth in `src/registry.rs::register_builtins()`.
- [x] 3.2 Register `local` as a built-in `ApiFamily::Completions` provider with both `CredentialKind::NoAuth` and `CredentialKind::ApiKey` support, registered via `register_by_url_pattern` for `http://localhost`, `http://127.0.0.1`, and `http://0.0.0.0` prefixes.
- [x] 3.3 Set a sensible default base URL for `local` using Ollama's OpenAI-compatible endpoint (e.g., `http://localhost:11434/v1`).

## 4. ProviderConnection NoAuth Support

- [x] 4.1 Update `ProviderConnection::build()` in `src/connection.rs` to handle the NoAuth credential case — skip auth validation where no credential is needed.
- [x] 4.2 Ensure `ProviderOverrides` resolution and header composition work correctly with empty auth headers.
- [x] 4.3 Add tests proving a `local` provider connection constructs successfully with `RuntimeConfig::none()`.
- [x] 4.4 Add tests proving a `local` provider connection accepts `RuntimeConfig::new("token")` and sends Bearer auth.
- [x] 4.5 Add tests proving `RuntimeConfig::none()` fails for existing API-key providers.

## 5. Tests and Verification

- [x] 5.1 Add registry tests: `ollama-cloud` is registered, uses Completions family, accepts API key credentials.
- [x] 5.2 Add registry tests: `local` is registered, uses Completions family, accepts NoAuth and API-key credentials, rejects OAuth credentials, and is listed in slugs.
- [x] 5.3 Add URL resolution tests: `resolve_by_url` matches `http://localhost:8080`, `http://127.0.0.1:11434`, `http://0.0.0.0:8000` to `local`.
- [x] 5.4 Add URL resolution tests: `resolve_by_url` does NOT match `https://api.openai.com`, `http://example.com` to `local`.
- [x] 5.5 Add NoAuth credential validation tests: `RuntimeConfig::none()` passes for `local`, fails for existing API-key providers, and blank API keys remain invalid.
- [x] 5.6 Add header composition tests: NoAuth provider produces no Authorization or x-api-key headers.
- [x] 5.7 Verify existing tests still pass — no behavioral change for existing providers.
- [x] 5.8 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 5.9 Run `cargo clippy --locked --all-targets --all-features -- -D warnings`.
- [x] 5.10 Run `cargo test --locked`.
- [x] 5.11 Run `cargo build --locked --all-targets`.
- [x] 5.12 Verify coverage: LM Studio (`http://localhost:1234/v1`), Jan (`http://localhost:1337/v1`), MLX-LM (`http://localhost:8080/v1`), vLLM (`http://localhost:8000/v1`), llama.cpp (`http://localhost:8080/v1`), and Ollama (`http://localhost:11434/v1`) all resolve through the local URL pattern.

## 6. Documentation and Scope Checks

- [x] 6.1 Document that local Ollama support uses Ollama's OpenAI-compatible `/v1/chat/completions` endpoint.
- [x] 6.2 Ensure no tasks or specs require endpoint-path overrides or `ProviderQuirks.param_renames` wiring.
