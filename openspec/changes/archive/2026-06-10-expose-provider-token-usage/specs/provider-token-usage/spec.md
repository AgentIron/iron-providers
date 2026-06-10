## ADDED Requirements

### Requirement: Provider events SHALL expose normalized token usage
The provider event model SHALL include a normalized usage event carrying provider-reported token usage for the current inference request.

#### Scenario: Usage event carries provider-reported counts
- **WHEN** a provider adapter receives usage metadata from an upstream provider response
- **THEN** it emits a `ProviderEvent::Usage` event containing normalized token usage
- **AND** the event preserves all usage buckets returned by the provider that are represented by the normalized usage model

#### Scenario: Providers without usage omit usage events
- **WHEN** a provider response or stream does not include usage metadata
- **THEN** the adapter emits no usage event
- **AND** the existing output, tool-call, choice-request, completion, and error events remain available according to the existing stream contract

### Requirement: Usage snapshots SHALL be cumulative for a request
Usage events SHALL represent cumulative provider-reported usage snapshots for the current request rather than incremental deltas.

#### Scenario: Multiple streaming usage events supersede earlier snapshots
- **WHEN** a streaming provider emits multiple usage-bearing events for the same request
- **THEN** each normalized usage event represents the provider's cumulative usage snapshot at that point
- **AND** consumers can treat the latest usage event for that request as superseding earlier usage events

#### Scenario: Usage event is not additive by default
- **WHEN** a consumer receives more than one usage event during a request
- **THEN** the consumer must not assume the events can be summed to produce total usage

### Requirement: Normalized token usage SHALL preserve context and cost accounting buckets
The normalized usage model SHALL include optional fields for common token totals and provider-specific buckets needed for context pressure and cost telemetry.

#### Scenario: Common token totals are represented
- **WHEN** a provider reports input or prompt tokens, output or completion tokens, or total tokens
- **THEN** the normalized usage model represents them as input tokens, output tokens, and total tokens respectively

#### Scenario: Cached input tokens are represented
- **WHEN** an OpenAI-style provider reports cached prompt or input tokens
- **THEN** the normalized usage model represents them as cached input tokens

#### Scenario: Anthropic cache buckets are preserved separately
- **WHEN** an Anthropic-style provider reports cache creation input tokens or cache read input tokens
- **THEN** the normalized usage model preserves those counts in separate cache creation and cache read fields
- **AND** it does not collapse them into a single ambiguous cached token field

#### Scenario: Reasoning or thinking output tokens are represented
- **WHEN** a provider reports reasoning output tokens or thinking output tokens
- **THEN** the normalized usage model represents them as reasoning output tokens

### Requirement: Usage SHALL be emitted before successful stream completion
When provider usage and successful completion are known at the same stream boundary, the adapter SHALL emit usage before `ProviderEvent::Complete`.

#### Scenario: Terminal stream event contains usage
- **WHEN** a provider stream terminal event contains usage metadata and indicates successful completion
- **THEN** the adapter emits a usage event before emitting `Complete`

#### Scenario: Complete remains terminal
- **WHEN** `ProviderEvent::Complete` is emitted
- **THEN** no subsequent usage event is required to observe usage for that completed stream

### Requirement: Responses adapter SHALL map provider usage
The Responses-family adapter SHALL parse and normalize usage metadata returned by OpenAI Responses-compatible APIs.

#### Scenario: Non-streaming Responses response includes usage
- **WHEN** a non-streaming Responses API response includes `usage.input_tokens`, `usage.output_tokens`, `usage.total_tokens`, `usage.input_tokens_details.cached_tokens`, or `usage.output_tokens_details.reasoning_tokens`
- **THEN** the adapter emits a usage event mapping those fields to normalized token usage
- **AND** the usage event is emitted before `Complete`

#### Scenario: Streaming Responses completion includes usage
- **WHEN** a Responses stream emits a successful completion event whose response payload includes usage metadata
- **THEN** the adapter emits a usage event for that metadata before emitting `Complete`

### Requirement: Messages adapter SHALL map provider usage
The Messages-family adapter SHALL parse and normalize usage metadata returned by Anthropic Messages-compatible APIs.

#### Scenario: Non-streaming Messages response includes usage
- **WHEN** a non-streaming Messages API response includes usage input tokens, output tokens, cache creation input tokens, or cache read input tokens
- **THEN** the adapter emits a usage event mapping those fields to normalized token usage
- **AND** the usage event is emitted before `Complete`

#### Scenario: Streaming Messages event includes cumulative usage
- **WHEN** a Messages stream emits a usage-bearing message event
- **THEN** the adapter emits a usage event representing the cumulative provider-reported usage snapshot

#### Scenario: Messages stream completes after usage
- **WHEN** a Messages stream completes successfully after receiving usage metadata
- **THEN** the last known usage snapshot is observable before `Complete`

### Requirement: Chat Completions adapter SHALL map provider usage when available
The Chat Completions-family adapter SHALL parse and normalize usage metadata returned by OpenAI-compatible Chat Completions APIs without breaking providers that do not return usage.

#### Scenario: Non-streaming Chat Completions response includes usage
- **WHEN** a non-streaming Chat Completions response includes `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`, `usage.prompt_tokens_details.cached_tokens`, or `usage.completion_tokens_details.reasoning_tokens`
- **THEN** the adapter emits a usage event mapping those fields to normalized token usage
- **AND** the usage event is emitted before `Complete`

#### Scenario: Streaming Chat Completions chunk includes usage
- **WHEN** a Chat Completions stream chunk includes usage metadata
- **THEN** the adapter emits a usage event mapping that metadata to normalized token usage

#### Scenario: Streaming usage option is unsupported
- **WHEN** an OpenAI-compatible provider does not support or does not return streaming usage
- **THEN** the adapter continues to stream output, tool calls, choice requests, errors, and completion without requiring a usage event
