## MODIFIED Requirements

### Requirement: Local provider SHALL work with all OpenAI-compatible backends
The `local` provider SHALL support connection to any local LLM serving system that exposes an OpenAI-compatible `/v1/chat/completions` endpoint with either no authentication or Bearer API-key authentication. Callers SHALL be able to select a per-session local base URL override without redefining the upstream `local` provider profile.

#### Scenario: LM Studio works with runtime base URL override
- **WHEN** the `local` provider is constructed with a runtime base URL override of `http://localhost:1234/v1`
- **THEN** inference requests reach LM Studio's API server
- **AND** no auth headers are sent when `RuntimeConfig::none()` is used
- **AND** the upstream `local` profile metadata remains unchanged

#### Scenario: Jan works with runtime base URL override
- **WHEN** the `local` provider is constructed with a runtime base URL override of `http://localhost:1337/v1`
- **THEN** inference requests reach Jan's API server
- **AND** no auth headers are sent when `RuntimeConfig::none()` is used
- **AND** the upstream `local` profile metadata remains unchanged

#### Scenario: MLX-LM works with runtime base URL override
- **WHEN** the `local` provider is constructed with a runtime base URL override of `http://localhost:8080/v1`
- **THEN** inference requests reach MLX-LM's API server
- **AND** no auth headers are sent when `RuntimeConfig::none()` is used
- **AND** the upstream `local` profile metadata remains unchanged

#### Scenario: vLLM works with runtime base URL override
- **WHEN** the `local` provider is constructed with a runtime base URL override of `http://localhost:8000/v1`
- **THEN** inference requests reach vLLM's API server
- **AND** no auth headers are sent when `RuntimeConfig::none()` is used
- **AND** the upstream `local` profile metadata remains unchanged

## ADDED Requirements

### Requirement: Local provider SHALL preserve its upstream default endpoint
The built-in `local` provider profile SHALL keep its upstream default endpoint for callers that do not supply a per-session base URL override.

#### Scenario: local default endpoint is used without override
- **WHEN** the `local` provider is constructed without a runtime base URL override
- **THEN** its requests use `http://localhost:11434/v1` as the base URL

#### Scenario: local profile is not redefined for override
- **WHEN** the `local` provider is constructed with a runtime base URL override
- **THEN** the connection uses the override for request URLs
- **AND** the registered `local` profile still reports `http://localhost:11434/v1` as its base URL
