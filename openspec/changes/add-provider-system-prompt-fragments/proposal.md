## Why

`iron-core` needs provider-specific system prompt guidance that can be injected into its higher-level prompt templates without hard-coding provider behavior outside this crate. The original issue assumed one crate per provider, but this repository is a single `iron-providers` crate with provider modules and registry profiles. The proposal should therefore expose fragments from the existing module and profile boundaries instead of inventing separate provider crates.

OpenAI is already supported through `src/openai.rs`, `OpenAiProvider`, and `ApiFamily::OpenAiResponses`. The registry built-ins are a separate profile-driven provider-selection path and should not be changed just to satisfy this feature.

## What Changes

- Add raw Markdown system prompt fragments for Anthropic-style and OpenAI-compatible providers.
- Expose module-level helpers from the existing provider modules, including `anthropic::system_prompt_fragment()` and `openai::system_prompt_fragment()`.
- Add profile-driven fragment resolution via `ProviderProfile::system_prompt_fragment()` based on `ApiFamily`.
- Add registry-level lookup via `ProviderRegistry::system_prompt_fragment(provider_name)` for registered profiles.
- Add validation coverage that fragments are non-empty, concise enough for prompt injection, and contain no Tera syntax delimiters.

## Capabilities

### New Capabilities
- `provider-system-prompt-fragments`: Defines how provider-specific Markdown fragments are exposed by module helpers, provider profiles, and the provider registry.

### Modified Capabilities

## Impact

- Affected code: `src/anthropic.rs`, `src/openai.rs`, `src/profile.rs`, `src/registry.rs`, `src/lib.rs`, new prompt fragment source files, and tests.
- Affected behavior: callers can ask this crate for provider-specific prompt guidance without duplicating provider-family knowledge in `iron-core`.
- API impact: additive public API only. This change must not require adding an `openai` slug to `ProviderRegistry::default()`.
- Out of scope: splitting this crate into per-provider crates, adding new provider integrations, and adding provider-specific fragments for Gemini, Azure, DeepSeek, local, or Ollama before those providers/profiles exist here.
