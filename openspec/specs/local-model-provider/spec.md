# local-model-provider Specification

## Purpose

Defines the built-in `local` provider for OpenAI-compatible local model servers,
including no-auth and optional Bearer API-key authentication, loopback URL
resolution, and Ollama support through its `/v1/chat/completions` endpoint.

## Requirements

### Requirement: Local provider SHALL be a registered built-in Completions provider
The provider registry SHALL include a built-in `local` provider profile as a generic OpenAI-compatible local model endpoint.

#### Scenario: local slug is registered
- **WHEN** the provider registry is initialized with built-in profiles
- **THEN** `local` is registered as a provider slug
- **AND** it is listed in `registry.slugs()`

#### Scenario: local uses Completions API family
- **WHEN** the `local` profile is examined
- **THEN** its `family` is `ApiFamily::Completions`
- **AND** its base URL supports localhost, 127.0.0.1, and 0.0.0.0

### Requirement: Local provider SHALL support no-auth and optional API-key credentials
The `local` provider SHALL support `CredentialKind::NoAuth` for unauthenticated local endpoints and `CredentialKind::ApiKey` for secured local OpenAI-compatible gateways. API keys SHALL be sent as Bearer tokens when supplied and SHALL NOT be silently dropped.

#### Scenario: local accepts NoAuth credential
- **WHEN** a runtime config without an API key or OAuth token is supplied for `local`
- **THEN** the connection construction succeeds
- **AND** no auth headers are added to outgoing requests

#### Scenario: local accepts API key credential
- **WHEN** an API key credential is supplied for `local`
- **THEN** the connection construction succeeds
- **AND** outgoing requests include `Authorization: Bearer <api-key>`

#### Scenario: local rejects OAuth credential
- **WHEN** an OAuth bearer credential is supplied for `local`
- **THEN** connection construction fails because `local` does not support OAuth credentials

### Requirement: Local provider SHALL resolve by URL pattern
The `local` provider SHALL automatically resolve when an endpoint URL matches local address patterns.

#### Scenario: URL resolution matches localhost
- **WHEN** `resolve_by_url()` is called with `http://localhost:11434/v1/chat/completions`
- **THEN** it returns the `local` provider profile

#### Scenario: URL resolution matches 127.0.0.1
- **WHEN** `resolve_by_url()` is called with `http://127.0.0.1:8080/v1/chat/completions`
- **THEN** it returns the `local` provider profile

#### Scenario: URL resolution matches 0.0.0.0
- **WHEN** `resolve_by_url()` is called with `http://0.0.0.0:8000/v1/chat/completions`
- **THEN** it returns the `local` provider profile

#### Scenario: Non-local URL does not match local
- **WHEN** `resolve_by_url()` is called with `https://api.openai.com/v1/chat/completions`
- **THEN** it does NOT return the `local` provider profile

### Requirement: Local provider SHALL use the standard Completions adapter
Inference and streaming requests for `local` SHALL dispatch through `src/apis/completions.rs`.

#### Scenario: infer dispatches to Completions adapter
- **WHEN** `infer()` is called on a `local` provider connection
- **THEN** the request is handled by the existing Completions API adapter

#### Scenario: infer_stream dispatches to Completions adapter
- **WHEN** `infer_stream()` is called on a `local` provider connection
- **THEN** streaming is handled by the existing Completions API adapter

### Requirement: Local provider SHALL work with all OpenAI-compatible backends
The `local` provider SHALL support connection to any local LLM serving system that exposes an OpenAI-compatible `/v1/chat/completions` endpoint with either no authentication or Bearer API-key authentication.

#### Scenario: LM Studio works with base URL change
- **WHEN** the local provider's base URL is set to `http://localhost:1234/v1`
- **THEN** inference requests reach LM Studio's API server
- **AND** no auth headers are sent

#### Scenario: Jan works with base URL change
- **WHEN** the local provider's base URL is set to `http://localhost:1337/v1`
- **THEN** inference requests reach Jan's API server
- **AND** no auth headers are sent

#### Scenario: MLX-LM works with base URL change
- **WHEN** the local provider's base URL is set to `http://localhost:8080/v1`
- **THEN** inference requests reach MLX-LM's API server
- **AND** no auth headers are sent

#### Scenario: vLLM works with base URL change
- **WHEN** the local provider's base URL is set to `http://localhost:8000/v1`
- **THEN** inference requests reach vLLM's API server
- **AND** no auth headers are sent

### Requirement: Local provider SHALL support Ollama through the OpenAI-compatible endpoint
The `local` provider SHALL support Ollama through Ollama's OpenAI-compatible `/v1/chat/completions` endpoint.

#### Scenario: Ollama OpenAI-compatible endpoint works without quirks
- **WHEN** the local provider's base URL is set to `http://localhost:11434/v1`
- **THEN** inference requests target `/v1/chat/completions`
- **AND** the standard Completions request shape is used
