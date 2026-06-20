## Why

`ProviderProfile::system_prompt_fragment()` currently selects provider prompt guidance only from `ApiFamily`. That makes the guidance protocol-family-specific rather than provider-specific: every `Messages` profile receives the Anthropic fragment, and every `Responses` or `Completions` profile receives the OpenAI-compatible fragment.

Consumers need a way for a registered or custom provider profile to supply its own provider-specific guidance while preserving the existing family-level compiled fragments as defaults.

This change intentionally does not add agent identity or Section 1 system-prompt identity to `ProviderProfile`. Agent identity remains owned by `iron-core` or other prompt-composition layers. `iron-providers` only exposes provider/API guidance for callers to inject into their higher-level prompts.

## What Changes

- Add optional `provider_guidance` metadata to `ProviderProfile`.
- Add a builder method for setting per-profile guidance.
- Make `ProviderProfile::system_prompt_fragment()` return per-profile guidance when present, otherwise fall back to the existing compiled family-level Markdown fragments.
- Update registry fragment lookup to preserve the same lookup behavior while returning the resolved profile guidance.
- Preserve existing built-in behavior by not adding custom guidance content to built-ins in this change.

## Capabilities

### Modified Capabilities

- `provider-system-prompt-fragments`: Profile and registry fragment resolution can use optional per-profile guidance before falling back to API-family defaults.

## Impact

- Affected public APIs: `ProviderProfile`, `ProviderProfile::system_prompt_fragment()`, and `ProviderRegistry::system_prompt_fragment()`.
- Affected serialization: serialized profiles may include `provider_guidance`; existing profiles without it continue to deserialize.
- Affected tests: profile serialization, builder behavior, fallback fragment resolution, override fragment resolution, and registry lookup behavior.
- No new external dependencies are expected.
