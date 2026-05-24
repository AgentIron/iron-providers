## ADDED Requirements

### Requirement: Credential model SHALL support NoAuth for unauthenticated providers
The credential model SHALL include explicit `NoAuth` support for providers that do not require authentication. The `RuntimeConfig::none()` constructor SHALL create an explicit no-auth runtime config, and connection construction SHALL accept it only when the provider profile supports `CredentialKind::NoAuth`.

#### Scenario: NoAuth credential kind and runtime credential exist
- **WHEN** a provider profile is configured for `CredentialKind::NoAuth`
- **THEN** the profile supports unauthenticated access
- **AND** `RuntimeConfig::none()` supplies an explicit no-auth credential

#### Scenario: NoAuth skips auth header generation
- **WHEN** a `ProviderConnection` is constructed with a `NoAuth` provider
- **THEN** `auth.rs` produces an empty `HeaderMap`
- **AND** no auth-related headers (Authorization, x-api-key, etc.) are added to the final headers

#### Scenario: NoAuth is profile-gated
- **WHEN** `RuntimeConfig::none()` is supplied for a provider profile that does not support `CredentialKind::NoAuth`
- **THEN** connection construction fails with an authentication error

#### Scenario: Existing providers are unaffected
- **WHEN** existing providers (anthropic, openrouter, codex, etc.) are constructed
- **THEN** their credential validation and auth header behavior is unchanged
- **AND** `CredentialKind::NoAuth` is opt-in per provider profile

### Requirement: Auth adapter SHALL handle the NoAuth case
The auth adapter in `auth.rs` SHALL return an empty header map when `AuthStrategy::NoAuth` is selected for an explicit no-auth credential.

#### Scenario: NoAuth produces no headers
- **WHEN** `auth_headers()` is called with a NoAuth credential and `AuthStrategy::NoAuth`
- **THEN** it returns an empty `HeaderMap`
- **AND** it does not error

#### Scenario: Blank API keys remain invalid
- **WHEN** a blank API-key credential is supplied
- **THEN** validation fails even if some other provider supports `CredentialKind::NoAuth`
