## ADDED Requirements

### Requirement: Provider registry SHALL include a direct OpenAI Responses provider
The provider registry SHALL include a built-in `openai` provider profile for public OpenAI Responses API access, distinct from the `codex` provider profile.

#### Scenario: openai slug is registered
- **WHEN** the default provider registry is constructed
- **THEN** its slugs include `openai`
- **AND** its slugs continue to include `codex`

#### Scenario: openai uses Responses API-key access
- **WHEN** the `openai` profile is examined
- **THEN** its `family` is `ApiFamily::Responses`
- **AND** its base URL is `https://api.openai.com/v1`
- **AND** it supports `CredentialKind::ApiKey`
- **AND** API-key requests use Bearer token auth

#### Scenario: openai remains distinct from codex
- **WHEN** the `openai` and `codex` profiles are examined
- **THEN** `openai` represents public OpenAI API-key access
- **AND** `codex` represents ChatGPT/Codex OAuth-backed access
- **AND** adding `openai` does not change `codex` endpoint, auth, fixed body, or header behavior

### Requirement: Provider construction SHALL support runtime-effective base URLs
Provider construction SHALL support an optional per-session base URL override that changes the effective request base URL without mutating or redefining the upstream `ProviderProfile`.

#### Scenario: runtime base URL override wins for connection requests
- **WHEN** a provider connection is constructed with a profile and a runtime base URL override
- **THEN** inference and streaming requests use the override as the request base URL
- **AND** the static profile base URL remains unchanged

#### Scenario: profile base URL is used without override
- **WHEN** a provider connection is constructed without a runtime base URL override
- **THEN** inference and streaming requests use the profile base URL

#### Scenario: runtime override preserves profile-owned metadata
- **WHEN** a runtime base URL override is supplied
- **THEN** API family, credential support, auth strategy, default headers, endpoint purpose, quirks, and provider overrides continue to come from the upstream profile

### Requirement: Registry SHALL expose deterministic models.dev identity resolution
The provider registry SHALL expose deterministic resolution for profiles that share a `models.dev` provider identity.

#### Scenario: shared models.dev identity returns all matching profiles deterministically
- **WHEN** multiple registered profiles have the same effective `models.dev` provider identity
- **THEN** registry resolution can return all matching profiles
- **AND** the returned order is deterministic across runs

#### Scenario: openai catalog identity includes public OpenAI and Codex profiles
- **WHEN** profiles are resolved for the `models.dev` identity `openai`
- **THEN** the result includes the direct `openai` profile
- **AND** the result includes the `codex` profile
- **AND** consumers can distinguish them by slug, endpoint purpose, credential support, and endpoint metadata

#### Scenario: single-profile lookup does not rely on HashMap iteration order
- **WHEN** a single-profile `models.dev` lookup API remains available
- **THEN** its selected result is deterministic
- **AND** its behavior does not depend on `HashMap` value iteration order
