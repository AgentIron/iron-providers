## Context

Issue `#11` asks for provider-specific system prompt fragments that `iron-core` can inject into its system prompt. The useful contract is a small raw Markdown fragment per provider family. The original issue text described separate `iron-provider-*` crates, but this repository is a single crate with two relevant public boundaries:

- Provider modules, such as `src/openai.rs` and `src/anthropic.rs`.
- `ProviderProfile` plus `ProviderRegistry`, which select provider behavior by `ApiFamily` and registered slug.

Current `ApiFamily` variants are `OpenAiResponses`, `OpenAiChatCompletions`, and `AnthropicMessages`. Current default registry slugs include `anthropic`, `minimax`, `minimax-code`, `zai`, `zai-code`, `kimi`, `kimi-code`, `openrouter`, and `requesty`. OpenAI is supported and used today through the concrete `OpenAiProvider` path, but `openai` is not a default registry slug.

## Goals / Non-Goals

**Goals:**

- Expose provider-specific Markdown prompt fragments as `&'static str`.
- Keep fragments safe for caller-side templating by excluding Tera delimiters such as `{{`, `{%`, and `{#`.
- Map all currently supported `ApiFamily` variants to an appropriate fragment.
- Let `iron-core` retrieve fragments through either the concrete OpenAI/Anthropic module path or the registry/profile path.
- Keep the implementation additive and simple.

**Non-Goals:**

- Adding new provider integrations or profiles.
- Adding `openai` to `ProviderRegistry::default()` as part of this change.
- Creating separate provider crates.
- Introducing runtime template rendering in `iron-providers`.
- Maintaining separate fragments for future provider families until they exist in this crate.

## Decisions

### 1. Store fragments as Markdown files included at compile time

Add Markdown files under `src/system_prompt_fragments/` and include them with `include_str!`. The public API should return `&'static str` directly.

Why this decision:

- Markdown files are easier to review and edit than embedded Rust string literals.
- `include_str!` keeps the API allocation-free and avoids runtime I/O.
- Raw Markdown is compatible with `iron-core` injecting the fragment via a safe template filter.

Alternative considered: generate fragments dynamically from profile metadata.

- Rejected because these fragments are guidance text, not data derived mechanically from profile fields.

### 2. Provide module-level helpers for existing provider APIs

Add `SYSTEM_PROMPT_FRAGMENT` constants and `system_prompt_fragment()` helpers to `anthropic` and `openai` modules.

Why this decision:

- It matches how OpenAI is supported today: through the concrete `OpenAiProvider` path rather than a default registry slug.
- It gives direct provider users a simple API without needing to construct a profile or registry.
- It keeps the public API discoverable next to each provider implementation.

Alternative considered: expose only a central `system_prompt` module.

- Rejected because the issue asks for provider-level access, and module-level helpers are the most direct equivalent in the single-crate architecture.

### 3. Map profile fragments by `ApiFamily`

Add `ProviderProfile::system_prompt_fragment(&self) -> &'static str` and route by `self.family`:

- `ApiFamily::AnthropicMessages` uses the Anthropic-style fragment.
- `ApiFamily::OpenAiResponses` uses the OpenAI-compatible fragment.
- `ApiFamily::OpenAiChatCompletions` uses the OpenAI-compatible fragment.

Why this decision:

- The registry profiles already encode the provider protocol family that determines prompt-shaping constraints.
- All current built-ins can resolve without adding provider-specific branches for each slug.
- OpenAI-compatible providers share the relevant JSON schema and tool-calling guidance at this layer.

Alternative considered: map by provider slug.

- Rejected for this change because it would duplicate the existing family model and make compatible provider additions more error-prone.

### 4. Add registry lookup without changing registered providers

Add `ProviderRegistry::system_prompt_fragment(&self, provider_name: &str) -> ProviderResult<&'static str>` using the same case-insensitive lookup semantics as `ProviderRegistry::get`.

Why this decision:

- `iron-core` can resolve prompt guidance from the same configured provider name it uses to obtain providers.
- Unknown providers can reuse the crate's existing `ProviderError` pattern.
- This does not require changing the default registry contents.

Alternative considered: make unknown providers fall back to OpenAI guidance.

- Rejected because fallback would hide configuration mistakes and could inject the wrong provider instructions.

### 5. Keep fragment validation lightweight and local

Tests should assert that fragments are non-empty, contain no Tera delimiters, and that every default registry slug resolves to a non-empty fragment. Tests should also cover module helpers and explicit `ApiFamily` mapping.

Why this decision:

- It directly protects the integration contract with `iron-core`.
- It avoids adding Markdown parsers or template engines to this crate for a simple static-text feature.

Alternative considered: parse Markdown in tests.

- Rejected because CommonMark validity is not the main risk here; unsafe template delimiters and missing mappings are.

## Risks / Trade-offs

- OpenAI-compatible providers may eventually need more specific fragments. Mitigation: start with family-level mapping now and add slug- or profile-specific overrides only when a concrete behavior difference appears.
- Fragment wording can become stale as provider APIs evolve. Mitigation: keep fragments concise, provider-family oriented, and covered by tests that ensure every current family has a mapping.
- Adding public helpers in provider modules slightly expands the crate surface. Mitigation: the API is additive, simple, and returns static text without new runtime behavior.
- `ProviderRegistry::system_prompt_fragment` will not resolve `openai` unless callers registered an OpenAI profile. Mitigation: direct OpenAI users can call `openai::system_prompt_fragment()`, and adding an OpenAI registry slug remains out of scope unless a separate requirement appears.

## Migration Plan

This is an additive feature. Existing callers do not need to change. `iron-core` can adopt the new API by calling the module helper for its existing concrete OpenAI path or the registry/profile helper when working with registered providers.

Rollback is straightforward because the change adds static files and public helper methods without altering inference behavior.

## Open Questions

- Should future provider-specific overrides be modeled as an optional `ProviderProfile` field or as internal slug-specific matching? This change intentionally avoids deciding until a provider-specific fragment is actually needed.
