## 1. Runtime-Effective Endpoint Support

- [x] 1.1 Add an optional runtime/provider-construction base URL override API with a named builder or equivalent explicit entry point.
- [x] 1.2 Resolve an effective base URL during `ProviderConnection` construction without mutating the registered `ProviderProfile`.
- [x] 1.3 Update Responses, Completions, and Messages request URL construction to use the effective base URL.
- [x] 1.4 Add tests proving profile base URLs are used when no override is supplied.
- [x] 1.5 Add tests proving override base URLs are used for requests while static profile metadata remains unchanged.

## 2. Built-In OpenAI Profile

- [x] 2.1 Register a built-in `openai` profile for public OpenAI Responses API access.
- [x] 2.2 Ensure the `openai` profile uses `ApiFamily::Responses`, `https://api.openai.com/v1`, and API-key Bearer auth.
- [x] 2.3 Add tests proving `openai` and `codex` are both registered and remain distinct.
- [x] 2.4 Add tests proving adding `openai` does not change Codex OAuth rejection/acceptance, endpoint, fixed body, or header behavior.

## 3. Deterministic models.dev Resolution

- [x] 3.1 Add a registry API that returns all profiles matching an effective `models.dev` provider identity in deterministic order.
- [x] 3.2 Make any remaining single-profile `models.dev` resolver deterministic or document/use the plural resolver for ambiguous cases.
- [x] 3.3 Add tests proving identity `openai` resolves to both `openai` and `codex` through the plural API.
- [x] 3.4 Add tests proving shared identity resolution order does not depend on `HashMap` iteration order.

## 4. Local Provider Override Behavior

- [x] 4.1 Add tests proving `local` keeps `http://localhost:11434/v1` as its registered default endpoint.
- [x] 4.2 Add tests proving `local` can target LM Studio-style `http://localhost:1234/v1` through a runtime base URL override.
- [x] 4.3 Add tests proving `local` override behavior preserves NoAuth and API-key credential behavior.
- [x] 4.4 Add tests proving the registered `local` profile metadata is unchanged after constructing an overridden connection.

## 5. Documentation and Validation

- [x] 5.1 Update public docs or crate-level examples for direct `openai` and runtime local base URL override usage.
- [x] 5.2 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 5.3 Run `cargo clippy --manifest-path Cargo.toml --all-targets --all-features -- -D warnings`.
- [x] 5.4 Run `cargo test --manifest-path Cargo.toml`.
