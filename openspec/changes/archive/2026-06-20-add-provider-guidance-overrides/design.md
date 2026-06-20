## Context

The current system prompt fragment contract was designed around static provider-family guidance. That contract is useful as a fallback, but it cannot express provider-specific differences when multiple providers share the same wire protocol family.

The motivating example is a provider using an Anthropic-compatible `Messages` API without being Anthropic. Today it inherits the Anthropic Markdown fragment because fragment selection is keyed only by `ApiFamily`.

## Goals / Non-Goals

**Goals:**

- Let `ProviderProfile` carry optional raw Markdown provider guidance.
- Preserve family-level compiled fragments as defaults.
- Keep this crate out of agent identity and full prompt-template composition.
- Keep built-in provider guidance content unchanged for this mechanism-only change.
- Preserve deserialization compatibility for existing serialized profiles.

**Non-Goals:**

- Do not add `identity` or assistant persona fields to `ProviderProfile`.
- Do not move Section 1 system-prompt identity out of `iron-core`.
- Do not add custom guidance content for all built-in providers in this change.
- Do not perform template rendering in `iron-providers`.

## Decisions

### Store per-profile provider guidance as optional profile metadata

Add `provider_guidance: Option<String>` to `ProviderProfile`, with serde defaults and skipped serialization when absent.

Rationale: provider guidance is static provider metadata consumed by prompt composition layers. Keeping it on the profile lets custom or stored profiles describe provider-specific prompt requirements without changing protocol adapters.

### Resolve per-profile guidance before family fallback

`ProviderProfile::system_prompt_fragment()` should return `provider_guidance` when present. When absent, it should return the existing compiled fragment selected by `ApiFamily`.

Rationale: this preserves all current built-in behavior while allowing targeted overrides.

### Change fragment return APIs to borrowed strings

Because profile guidance is profile-owned `String` data, fragment accessors should return `&str` rather than `&'static str`.

Rationale: the previous static lifetime was true only while fragments came exclusively from `include_str!` constants. Per-profile guidance makes the resolved fragment borrow from `self`.

### Keep agent identity out of provider profiles

Do not add the `identity` field described in issue 37. In this architecture, `ProviderProfile` describes provider identity and static provider configuration. Assistant identity, mode, and Section 1 system-prompt content belong to prompt-composition layers such as `iron-core`.

Rationale: tying assistant identity to provider or model profiles would conflate provider transport metadata with agent behavior.

## Risks / Trade-offs

- Changing fragment return types from `&'static str` to `&str` may require downstream callers to stop assuming a static lifetime.
- Allowing arbitrary profile guidance means callers remain responsible for deciding whether stored profile content is trusted enough for prompt injection.
- Mechanism-only scope means non-Anthropic Messages built-ins will continue to use the Anthropic fallback until a later content change supplies provider-specific guidance.

## Open Questions

- Should empty `provider_guidance` strings be allowed as an explicit override, or treated as absent? The default implementation can allow them unless validation is desired later.
