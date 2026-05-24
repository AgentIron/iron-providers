## ADDED Requirements

### Requirement: Ollama Cloud SHALL be a registered built-in Completions provider
The provider registry SHALL include a built-in `ollama-cloud` provider profile targeting the Ollama Cloud API endpoint.

#### Scenario: ollama-cloud slug is registered
- **WHEN** the provider registry is initialized with built-in profiles
- **THEN** `ollama-cloud` is registered as a provider slug
- **AND** it is listed in `registry.slugs()`

#### Scenario: ollama-cloud uses Completions API family
- **WHEN** the `ollama-cloud` profile is examined
- **THEN** its `family` is `ApiFamily::Completions`
- **AND** its `base_url` is `https://api.ollama.cloud`

#### Scenario: ollama-cloud uses BearerToken auth
- **WHEN** a runtime config is supplied for `ollama-cloud`
- **THEN** it accepts `CredentialKind::ApiKey` with `AuthStrategy::BearerToken`

### Requirement: Ollama Cloud SHALL use the standard Completions adapter
Inference and streaming requests for `ollama-cloud` SHALL dispatch through `src/apis/completions.rs`.

#### Scenario: infer dispatches to Completions adapter
- **WHEN** `infer()` is called on an `ollama-cloud` provider connection
- **THEN** the request is handled by the existing Completions API adapter

#### Scenario: infer_stream dispatches to Completions adapter
- **WHEN** `infer_stream()` is called on an `ollama-cloud` provider connection
- **THEN** streaming is handled by the existing Completions API adapter
- **AND** existing stream assembly behavior (delta concatenation, tool-call assembly) is preserved
