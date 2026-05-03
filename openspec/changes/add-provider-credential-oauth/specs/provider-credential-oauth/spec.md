## ADDED Requirements

### Requirement: Runtime credentials SHALL distinguish API keys from OAuth bearer tokens
The provider runtime configuration SHALL represent API-key credentials and OAuth bearer credentials as distinct credential variants.

#### Scenario: Existing API-key constructor remains usable
- **WHEN** a caller constructs `RuntimeConfig::new("secret")`
- **THEN** the runtime configuration contains an API-key credential with value `secret`
- **AND** existing timeout builder methods remain available

#### Scenario: Credential-aware constructor accepts API keys
- **WHEN** a caller constructs `RuntimeConfig::from_credential(ProviderCredential::ApiKey("secret"))`
- **THEN** the runtime configuration contains an API-key credential with value `secret`

#### Scenario: Credential-aware constructor accepts OAuth bearer tokens
- **WHEN** a caller constructs `RuntimeConfig::from_credential(ProviderCredential::OAuthBearer { access_token, expires_at, id_token })`
- **THEN** the runtime configuration contains the OAuth bearer credential snapshot
- **AND** no refresh token is represented in the runtime credential

### Requirement: Provider profiles SHALL declare supported credential kinds
Provider profiles SHALL expose which credential kinds they support and which wire auth strategy applies to each kind.

#### Scenario: API-key-only provider supports API keys
- **WHEN** a profile is configured with API-key auth only
- **THEN** `CredentialKind::ApiKey` is supported
- **AND** `CredentialKind::OAuthBearer` is not supported

#### Scenario: Mixed-mode provider supports multiple credential kinds
- **WHEN** a profile is configured with API-key and OAuth bearer auth configs
- **THEN** both credential kinds are supported
- **AND** each kind resolves to its configured `AuthStrategy`

#### Scenario: Existing `with_auth` helper remains API-key oriented
- **WHEN** built-in profiles call `with_auth(strategy)`
- **THEN** the strategy applies to `CredentialKind::ApiKey`
- **AND** existing API-key-only built-ins preserve their current wire auth behavior

### Requirement: Provider construction SHALL validate credential compatibility
Provider construction SHALL fail clearly when a runtime credential is missing, blank, unsupported by the profile, or expired.

#### Scenario: Blank API key fails as authentication
- **WHEN** a provider is constructed with a blank API-key credential
- **THEN** construction fails with an authentication error describing the missing credential

#### Scenario: Blank OAuth access token fails as authentication
- **WHEN** a provider is constructed with an OAuth bearer credential whose access token is blank
- **THEN** construction fails with an authentication error describing the missing credential

#### Scenario: Unsupported credential kind fails before requests
- **WHEN** a provider profile supports only API keys
- **AND** the runtime credential is OAuth bearer
- **THEN** provider construction fails with an authentication error describing the unsupported credential kind

#### Scenario: Expired OAuth bearer token fails before requests
- **WHEN** an OAuth bearer credential has `expires_at` at or before the current system time
- **THEN** provider construction or pre-request validation fails with an authentication error describing the expired credential

#### Scenario: OAuth bearer token without expiry is accepted
- **WHEN** an OAuth bearer credential has `expires_at: None`
- **THEN** expiry validation does not reject the credential

### Requirement: Auth headers SHALL be built from the selected credential and strategy
Provider requests SHALL apply the auth strategy associated with the runtime credential kind, using the selected credential value.

#### Scenario: Bearer strategy uses Authorization header
- **WHEN** a selected credential value is applied with `AuthStrategy::BearerToken`
- **THEN** requests include `Authorization: Bearer <credential>`

#### Scenario: API-key header strategy uses configured header name
- **WHEN** a selected credential value is applied with `AuthStrategy::ApiKeyHeader { header_name }`
- **THEN** requests include `<header_name>: <credential>`

#### Scenario: Custom strategy preserves prefix behavior
- **WHEN** a selected credential value is applied with `AuthStrategy::Custom { header_name, prefix }`
- **THEN** requests include the configured header with either `<prefix> <credential>` or the raw credential when no prefix exists

### Requirement: Kimi Code SHALL support API-key and OAuth bearer credentials
The built-in `kimi-code` profile SHALL support both API-key and OAuth bearer credentials with distinct wire headers.

#### Scenario: Kimi Code API key uses x-api-key
- **WHEN** a caller constructs `kimi-code` with an API-key credential
- **THEN** provider requests use `x-api-key: <api_key>`

#### Scenario: Kimi Code OAuth uses bearer auth
- **WHEN** a caller constructs `kimi-code` with an OAuth bearer credential
- **THEN** provider requests use `Authorization: Bearer <access_token>`

#### Scenario: Kimi general remains API-key-only
- **WHEN** a caller constructs `kimi` with an OAuth bearer credential
- **THEN** construction fails with an unsupported credential authentication error

### Requirement: Codex provider SHALL be registered as a distinct provider
The provider registry SHALL include a built-in `codex` profile for ChatGPT/Codex-backed access without changing the standard `openai` provider path.

#### Scenario: Codex built-in profile exists
- **WHEN** the default provider registry is constructed
- **THEN** its slugs include `codex`
- **AND** the `codex` profile has `models_dev_id` set to `openai`
- **AND** the profile uses `ApiFamily::CodexResponses`

#### Scenario: Codex requires OAuth bearer credentials
- **WHEN** a caller constructs `codex` with an OAuth bearer credential
- **THEN** construction succeeds when the token is non-empty and not expired

#### Scenario: Codex rejects API-key credentials unless explicitly supported later
- **WHEN** a caller constructs `codex` with an API-key credential
- **THEN** construction fails with an unsupported credential authentication error

### Requirement: Codex requests SHALL target the ChatGPT/Codex Responses endpoint
Codex inference SHALL send Responses-like HTTP requests to the ChatGPT/Codex backend endpoint.

#### Scenario: Non-streaming Codex request uses the Codex responses URL
- **WHEN** `GenericProvider` dispatches a non-streaming request for `ApiFamily::CodexResponses`
- **THEN** it sends `POST https://chatgpt.com/backend-api/codex/responses`
- **AND** the body includes `stream: false`

#### Scenario: Streaming Codex request uses the Codex responses URL
- **WHEN** `GenericProvider` dispatches a streaming request for `ApiFamily::CodexResponses`
- **THEN** it sends `POST https://chatgpt.com/backend-api/codex/responses`
- **AND** the body includes `stream: true`

#### Scenario: Codex request includes required fixed body fields
- **WHEN** a Codex request body is built
- **THEN** it includes `store: false`
- **AND** it includes `reasoning.effort: "medium"`
- **AND** it includes `parallel_tool_calls: true`

#### Scenario: Codex request includes normalized request fields
- **WHEN** a Codex request body is built from an `InferenceRequest`
- **THEN** it includes the request model
- **AND** it includes instructions when present
- **AND** it includes projected transcript/input content
- **AND** it includes tools and tool choice when present

### Requirement: Codex requests SHALL include required headers
Codex requests SHALL include provider-required auth and product identification headers.

#### Scenario: Codex request includes bearer auth
- **WHEN** a Codex request is sent with OAuth bearer credential `token`
- **THEN** it includes `Authorization: Bearer token`

#### Scenario: Codex request includes product headers
- **WHEN** a Codex request is sent
- **THEN** it includes `originator: iron-providers`
- **AND** it includes `User-Agent: iron-providers/<crate-version>`

#### Scenario: Codex request includes account routing header when available
- **WHEN** a Codex OAuth credential includes an ID token with extractable account metadata
- **THEN** the request includes `chatgpt-account-id: <account_id>`

#### Scenario: Codex request omits account routing header when unavailable
- **WHEN** a Codex OAuth credential has no ID token or no extractable account metadata
- **THEN** the request omits `chatgpt-account-id`
- **AND** the request can still be sent

### Requirement: JWT account routing metadata SHALL be parsed without verification
The crate SHALL extract optional ChatGPT account routing metadata from JWT payload claims for request-header routing only.

#### Scenario: Top-level account claim wins
- **WHEN** an ID token payload contains top-level `chatgpt_account_id`
- **THEN** that value is used as the ChatGPT account ID

#### Scenario: Nested account claim is used second
- **WHEN** an ID token payload lacks top-level `chatgpt_account_id`
- **AND** contains `https://api.openai.com/auth.chatgpt_account_id`
- **THEN** that nested value is used as the ChatGPT account ID

#### Scenario: Organization ID is fallback
- **WHEN** an ID token payload lacks both ChatGPT account claims
- **AND** contains `organizations[0].id`
- **THEN** that organization ID is used as the ChatGPT account ID

#### Scenario: Malformed JWT produces no account ID
- **WHEN** an ID token is malformed or has invalid JSON payload
- **THEN** account ID extraction returns `None`
- **AND** provider construction does not fail solely because account metadata is unavailable

### Requirement: Provider behavior SHALL not own OAuth refresh lifecycle
The provider crate SHALL not receive refresh tokens and SHALL not perform OAuth refresh or hidden retry orchestration.

#### Scenario: Runtime credential contains no refresh token
- **WHEN** callers build any supported `ProviderCredential`
- **THEN** there is no field for refresh-token material

#### Scenario: Unauthorized response is surfaced
- **WHEN** a provider receives an unrecoverable unauthorized response from the backend
- **THEN** the provider returns an authentication error
- **AND** it does not attempt token refresh internally

## MODIFIED Requirements

### Requirement: GenericProvider SHALL dispatch all supported API families
`GenericProvider` SHALL dispatch inference and streaming requests for every registered `ApiFamily` variant.

#### Scenario: Codex Responses dispatches to Codex adapter
- **WHEN** a profile has `family` set to `ApiFamily::CodexResponses`
- **THEN** `GenericProvider::infer` dispatches to the Codex non-streaming adapter
- **AND** `GenericProvider::infer_stream` dispatches to the Codex streaming adapter

#### Scenario: Existing families continue dispatching as before
- **WHEN** a profile has an existing family such as `OpenAiResponses`, `OpenAiChatCompletions`, or `AnthropicMessages`
- **THEN** dispatch behavior remains compatible with the existing adapter path
