## 1. Fragment Content

- [x] 1.1 Add `src/system_prompt_fragments/anthropic.md` with concise Anthropic Messages guidance.
- [x] 1.2 Add `src/system_prompt_fragments/openai.md` with concise OpenAI-compatible Responses and Chat Completions guidance.
- [x] 1.3 Ensure fragments are valid Markdown prose/bullets and contain no Tera delimiters (`{{`, `{%`, `{#`).

## 2. Public API

- [x] 2.1 Add `SYSTEM_PROMPT_FRAGMENT` and `system_prompt_fragment() -> &'static str` to `src/anthropic.rs`.
- [x] 2.2 Add `SYSTEM_PROMPT_FRAGMENT` and `system_prompt_fragment() -> &'static str` to `src/openai.rs`.
- [x] 2.3 Add `ProviderProfile::system_prompt_fragment(&self) -> &'static str` mapped by `ApiFamily`.
- [x] 2.4 Add `ProviderRegistry::system_prompt_fragment(&self, provider_name: &str) -> ProviderResult<&'static str>` using existing case-insensitive registered-profile lookup behavior.
- [x] 2.5 Export any new module or helper needed from `src/lib.rs` and `prelude` only if required for ergonomic public access.

## 3. Tests

- [x] 3.1 Add tests that `anthropic::system_prompt_fragment()` and `openai::system_prompt_fragment()` return the expected non-empty fragments.
- [x] 3.2 Add tests that each `ApiFamily` maps to the intended fragment through `ProviderProfile::system_prompt_fragment()`.
- [x] 3.3 Add tests that `ProviderRegistry::system_prompt_fragment()` resolves every default registry slug to a non-empty fragment.
- [x] 3.4 Add tests that unknown registry provider names return `ProviderError`.
- [x] 3.5 Add tests that all fragments contain no Tera delimiters.

## 4. Verification

- [x] 4.1 Run `cargo build --locked --all-targets`.
- [x] 4.2 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 4.3 Run `cargo clippy --locked --all-targets --all-features -- -D warnings`.
- [x] 4.4 Run `cargo test --locked`.
