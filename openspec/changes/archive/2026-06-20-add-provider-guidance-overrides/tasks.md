## 1. Provider Guidance Metadata

- [x] 1.1 Add optional `provider_guidance` to `ProviderProfile` with serde default and skip-when-none serialization.
- [x] 1.2 Initialize `provider_guidance` to `None` in `ProviderProfile::new()`.
- [x] 1.3 Add `with_provider_guidance(impl Into<String>) -> Self`.

## 2. Fragment Resolution

- [x] 2.1 Update `ProviderProfile::system_prompt_fragment()` to return profile guidance when set and family-level compiled fragments otherwise.
- [x] 2.2 Update `ProviderProfile::system_prompt_fragment()` return type from `&'static str` to `&str`.
- [x] 2.3 Update `ProviderRegistry::system_prompt_fragment()` return type from `ProviderResult<&'static str>` to `ProviderResult<&str>`.

## 3. Tests

- [x] 3.1 Add tests proving existing profiles without `provider_guidance` still deserialize and fall back to family-level fragments.
- [x] 3.2 Add tests proving profiles with `provider_guidance` serialize/deserialize and return the custom guidance.
- [x] 3.3 Add tests proving the builder method sets custom guidance.
- [x] 3.4 Add tests proving registry fragment lookup returns profile-specific guidance for registered custom profiles.
- [x] 3.5 Keep existing built-in fragment tests passing without adding custom built-in guidance content.

## 4. Validation

- [x] 4.1 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 4.2 Run `cargo clippy --manifest-path Cargo.toml --all-targets --all-features -- -D warnings`.
- [x] 4.3 Run `cargo test --manifest-path Cargo.toml`.
