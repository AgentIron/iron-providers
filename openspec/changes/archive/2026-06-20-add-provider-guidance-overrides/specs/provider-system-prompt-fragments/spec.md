## MODIFIED Requirements

### Requirement: Profile fragments SHALL use provider guidance before API-family fallback
`ProviderProfile` SHALL expose `system_prompt_fragment(&self) -> &str`. If the profile has per-profile provider guidance, the method SHALL return that guidance. Otherwise, it SHALL select the fallback fragment from the profile's `ApiFamily`.

#### Scenario: Profile guidance overrides family fallback
- **GIVEN** a provider profile with `provider_guidance` set to custom raw Markdown
- **WHEN** `ProviderProfile::system_prompt_fragment()` is called
- **THEN** it returns the custom profile guidance
- **AND** it does not return the API-family fallback fragment

#### Scenario: Anthropic Messages profile falls back to Anthropic guidance
- **GIVEN** a provider profile with `family` set to `ApiFamily::Messages`
- **AND** `provider_guidance` is absent
- **WHEN** `ProviderProfile::system_prompt_fragment()` is called
- **THEN** it returns the Anthropic system prompt fragment

#### Scenario: OpenAI-compatible profile falls back to OpenAI-compatible guidance
- **GIVEN** a provider profile with `family` set to `ApiFamily::Responses` or `ApiFamily::Completions`
- **AND** `provider_guidance` is absent
- **WHEN** `ProviderProfile::system_prompt_fragment()` is called
- **THEN** it returns the OpenAI-compatible system prompt fragment

### Requirement: Registry lookup SHALL resolve profile-specific prompt guidance
`ProviderRegistry` SHALL expose `system_prompt_fragment(&self, provider_name: &str) -> ProviderResult<&str>` and use the registered provider profile to resolve custom profile guidance before family fallback guidance.

#### Scenario: Registry returns custom guidance for a registered profile
- **GIVEN** a provider registry with a registered profile that has `provider_guidance` set
- **WHEN** a caller requests the system prompt fragment for that profile's slug
- **THEN** the registry returns the custom profile guidance

#### Scenario: Default registry providers continue resolving fragments
- **WHEN** a caller requests the system prompt fragment for any default registry slug
- **THEN** the registry returns a non-empty fragment for that provider

#### Scenario: Registry lookup remains case-insensitive
- **WHEN** a caller requests a registered provider name using different letter casing
- **THEN** the registry returns the same resolved fragment as the canonical lowercase slug

#### Scenario: Unknown registry provider returns an error
- **WHEN** a caller requests a system prompt fragment for a provider name that is not registered
- **THEN** the registry returns a `ProviderError` instead of falling back to an unrelated fragment

### Requirement: Provider guidance SHALL remain provider-specific, not agent identity
Provider guidance SHALL describe provider/API-specific prompt guidance for caller-side prompt composition. `ProviderProfile` SHALL NOT add assistant identity, agent persona, or Section 1 system-prompt identity fields as part of this change.

#### Scenario: Provider profile stores guidance but not assistant identity
- **WHEN** the provider guidance override mechanism is added
- **THEN** `ProviderProfile` can store optional provider guidance
- **AND** `ProviderProfile` does not store assistant identity or agent persona metadata for this feature

### Requirement: Provider guidance serialization SHALL be backward-compatible
Serialized provider profiles MAY include optional `provider_guidance`. Profiles without `provider_guidance` SHALL deserialize successfully and behave as if guidance is absent.

#### Scenario: Existing serialized profiles deserialize without provider guidance
- **GIVEN** serialized profile data that does not include `provider_guidance`
- **WHEN** it is deserialized into `ProviderProfile`
- **THEN** deserialization succeeds
- **AND** `provider_guidance` is absent

#### Scenario: Profiles serialize provider guidance when set
- **GIVEN** a provider profile with `provider_guidance` set
- **WHEN** it is serialized
- **THEN** the serialized data includes `provider_guidance`

#### Scenario: Profiles omit provider guidance when absent
- **GIVEN** a provider profile without `provider_guidance`
- **WHEN** it is serialized
- **THEN** the serialized data omits `provider_guidance`
