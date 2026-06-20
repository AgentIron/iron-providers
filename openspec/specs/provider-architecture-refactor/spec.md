# provider-architecture-refactor Specification

## Purpose

Defines the provider connection boundary, API-type adapter organization,
auth/header separation, provider override resolution, and public API migration
for protocol-oriented provider architecture.

## Requirements

### Requirement: Provider architecture SHALL separate provider identity from API protocol behavior
Provider profiles SHALL describe provider identity and static configuration, while API adapters SHALL implement protocol-specific request projection, response parsing, and streaming normalization.

#### Scenario: API family names a protocol
- **WHEN** a provider profile declares its API family
- **THEN** the family is one of `Responses`, `Completions`, or `Messages`
- **AND** the family name does not include provider-brand names such as OpenAI, Anthropic, or Codex

#### Scenario: Provider identity remains profile data
- **WHEN** a provider profile is registered
- **THEN** provider identity remains represented by profile fields such as slug, models.dev ID, base URL, credential metadata, purpose, quirks, and provider overrides
- **AND** provider identity is not represented by a provider-branded API-family variant

### Requirement: ProviderConnection SHALL be the resolved provider handoff
The registry SHALL construct a `ProviderConnection` that encapsulates resolved provider state and implements the `Provider` trait.

#### Scenario: Registry returns Provider trait backed by ProviderConnection
- **WHEN** `ProviderRegistry::get()` is called with a provider name and runtime configuration
- **THEN** it returns `Box<dyn Provider>` backed by `ProviderConnection`
- **AND** it does not construct `GenericProvider` or `OpenAiProvider`

#### Scenario: Direct construction is available
- **WHEN** a caller has a `ProviderProfile` and `RuntimeConfig`
- **THEN** the caller can construct a public `ProviderConnection`
- **AND** the connection implements `Provider`

#### Scenario: Construction resolves static state
- **WHEN** `ProviderConnection` is constructed
- **THEN** it validates runtime credentials, resolves credential compatibility, resolves auth headers, resolves provider overrides, composes protected final headers, builds the HTTP client, and selects the API adapter before returning

### Requirement: Auth SHALL be resolved separately from HTTP transport construction
Auth header generation SHALL be owned by an auth boundary, and HTTP client construction SHALL only assemble transport from already-composed headers and timeout values.

#### Scenario: Auth maps credential and strategy to headers
- **WHEN** a `ProviderCredential` and `AuthStrategy` are resolved during connection construction
- **THEN** `auth.rs` maps them to auth headers
- **AND** `http_client.rs` does not match on credential kinds or auth strategies

#### Scenario: HTTP client receives final headers
- **WHEN** the HTTP client is built
- **THEN** `http_client.rs` receives a final `HeaderMap` and timeout values
- **AND** it validates/applies headers and timeouts without provider-specific knowledge

### Requirement: Provider-specific configuration SHALL be centralized as provider overrides
Provider-specific headers, fixed body fields, endpoint behavior, and account routing metadata SHALL be resolved in one provider override boundary.

#### Scenario: Overrides are resolved from profile data
- **WHEN** provider-specific behavior is needed for a built-in profile
- **THEN** simple provider identity remains on `ProviderProfile`
- **AND** typed override behavior is resolved by crate-private provider override code

#### Scenario: Anthropic protocol headers are resolved as overrides
- **WHEN** a Messages-family provider requires an Anthropic protocol header such as `anthropic-version`
- **THEN** the header is resolved through provider overrides during connection construction
- **AND** the Messages adapter receives resolved configuration rather than hard-coding registry-specific logic

#### Scenario: Codex routing and fixed body fields are resolved as Responses overrides
- **WHEN** a Codex provider connection is constructed
- **THEN** Codex product headers, optional `chatgpt-account-id`, fixed body fields, and endpoint configuration are represented by Responses provider overrides
- **AND** this behavior is not represented by an `ApiFamily::CodexResponses` variant

#### Scenario: Provider-specific behavior has one home
- **WHEN** provider-specific request behavior is added or changed
- **THEN** it is modeled in provider override resolution unless it is truly protocol-generic adapter behavior

### Requirement: Header composition SHALL protect required headers from accidental override
Auth headers and required protocol/provider headers SHALL NOT be silently overridden by profile default headers.

#### Scenario: Profile default header conflicts with auth header
- **WHEN** a profile default header attempts to replace a resolved auth header such as `Authorization`
- **THEN** provider connection construction fails with a configuration error

#### Scenario: Profile default header conflicts with protected protocol header
- **WHEN** a profile default header attempts to replace a protected protocol/provider header such as `Content-Type`, `anthropic-version`, `originator`, `User-Agent`, or `chatgpt-account-id`
- **THEN** provider connection construction fails with a configuration error

#### Scenario: Intentional provider differences use overrides
- **WHEN** a provider requires a different required header value
- **THEN** that difference is represented through provider overrides rather than profile default headers

#### Scenario: Headers are composed in deterministic order
- **WHEN** provider connection construction composes headers
- **THEN** it applies protocol required headers, auth headers, provider override headers, and profile default headers in that order
- **AND** profile default headers can add non-protected headers but cannot replace protected headers

### Requirement: API adapters SHALL be organized by API type under `src/apis/`
Inference adapters SHALL be grouped by the API protocol they speak rather than by provider brand.

#### Scenario: Messages adapter handles Messages-family providers
- **WHEN** a provider profile uses `ApiFamily::Messages`
- **THEN** `ProviderConnection` dispatches inference and streaming requests to `src/apis/messages.rs`

#### Scenario: Completions adapter handles Completions-family providers
- **WHEN** a provider profile uses `ApiFamily::Completions`
- **THEN** `ProviderConnection` dispatches inference and streaming requests to `src/apis/completions.rs`

#### Scenario: Responses adapter handles Responses-family providers
- **WHEN** a provider profile uses `ApiFamily::Responses`
- **THEN** `ProviderConnection` dispatches inference and streaming requests to `src/apis/responses.rs`

### Requirement: Responses adapter SHALL support public OpenAI Responses and Codex through one upstream wire contract
Public OpenAI Responses and Codex SHALL use one Responses API adapter and the same upstream OpenAI Responses request, response, and stream contract, with provider-specific configuration represented by Responses overrides.

#### Scenario: Public OpenAI Responses uses Responses adapter
- **WHEN** a standard OpenAI Responses profile is used
- **THEN** inference and streaming are handled by `src/apis/responses.rs`
- **AND** the common upstream Responses request, response, and stream parsing behavior is used

#### Scenario: Codex uses Responses adapter
- **WHEN** a Codex provider profile is used
- **THEN** inference and streaming are handled by `src/apis/responses.rs`
- **AND** Codex-specific endpoint path, headers, body fields, and account routing are selected by Responses overrides
- **AND** the common upstream Responses request, response, and stream parsing behavior is used

#### Scenario: Codex does not define a separate wire dialect
- **WHEN** Codex is configured as a Responses-family provider
- **THEN** it does not select a separate Codex response parser or Codex stream parser
- **AND** provider-specific differences are limited to endpoint, header, routing, fixed body field, and error-context configuration unless the design is updated with evidence of an upstream incompatibility

#### Scenario: Responses adapter exposes one normalized contract
- **WHEN** public OpenAI Responses and Codex requests are projected by `src/apis/responses.rs`
- **THEN** both use the same normalized adapter input and emit the same normalized `ProviderEvent` output
- **AND** the same upstream Responses stream parser handles streaming events for both profiles

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

### Requirement: The async-openai dependency SHALL be removed
Responses-family providers SHALL use reqwest and crate-owned request/response/stream parsing rather than `async-openai`.

#### Scenario: Responses request uses reqwest
- **WHEN** a Responses-family request is sent
- **THEN** the request is built by crate-owned projection code and sent through the resolved `reqwest::Client`

#### Scenario: Responses parsing uses crate-owned types
- **WHEN** a Responses-family non-streaming response is received
- **THEN** it is parsed by crate-owned response structs or raw JSON parsing and mapped to `ProviderEvent`

#### Scenario: Responses streaming uses crate-owned parsing
- **WHEN** a Responses-family streaming response is received
- **THEN** it is parsed by the common crate-owned upstream Responses SSE parser and mapped to `ProviderEvent`

### Requirement: Legacy provider-specific public connection APIs SHALL be removed
The refactor SHALL remove public connection/configuration APIs that preserve the old provider-specific architecture.

#### Scenario: OpenAI-specific public connection API is removed
- **WHEN** the refactor is complete
- **THEN** `OpenAiProvider`, `OpenAiConfig`, and `OpenAiConfigSource` are no longer public API

#### Scenario: GenericProvider is not the public connection abstraction
- **WHEN** the refactor is complete
- **THEN** `GenericProvider` is not exported as a public provider construction abstraction
- **AND** callers use `ProviderConnection`, `ProviderRegistry`, `ProviderProfile`, and `RuntimeConfig` instead

#### Scenario: Hard API break is documented
- **WHEN** documentation is updated for the refactor
- **THEN** migration notes describe removed public types and renamed `ApiFamily` variants
- **AND** old serialized `ApiFamily` names are not supported through compatibility aliases

### Requirement: Provider registry SHALL dispatch all supported API families
The provider registry SHALL return providers that can dispatch inference and streaming requests for every registered API-family variant.

#### Scenario: All API families dispatch through ProviderConnection
- **WHEN** a registered profile has family `Responses`, `Completions`, or `Messages`
- **THEN** `ProviderRegistry::get()` returns a `ProviderConnection` that dispatches inference and streaming requests to the matching API adapter

#### Scenario: Existing provider slugs preserve behavior
- **WHEN** existing built-in provider slugs are used with equivalent credentials and requests
- **THEN** they preserve their documented provider behavior under the new connection and adapter architecture

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
