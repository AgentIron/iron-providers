## Context

`iron-providers` exposes a provider-neutral inference boundary through `ProviderEvent`, but the current event model only carries status, output text, tool calls, choice requests, completion, and errors. The supported adapter families already receive provider usage metadata from upstream APIs in several cases, but their response structs do not deserialize those fields.

The relevant provider families expose usage differently:

- OpenAI Responses reports `usage` on non-streaming responses and may include usage in the final streamed `response.completed` response payload.
- Anthropic Messages reports `usage` on non-streaming messages and cumulative usage snapshots on streaming `message_delta` events; its input accounting separates normal input, cache creation input, and cache read input.
- OpenAI-compatible Chat Completions reports `usage` on non-streaming responses and only reports streaming usage when `stream_options.include_usage` is accepted by the endpoint.

`iron-core` needs these provider-reported counts for two downstream use cases: context pressure tracking and cost telemetry. Context pressure needs an authoritative baseline for the full request, while cost telemetry benefits from preserving cache and reasoning/thinking token buckets instead of flattening them too early.

## Goals / Non-Goals

**Goals:**

- Expose provider-reported usage through a normalized `ProviderEvent::Usage` event.
- Preserve the provider buckets needed for context management and cost accounting.
- Parse usage for Responses, Messages, and Chat Completions in non-streaming responses.
- Emit streaming usage when providers supply it, without violating the stream termination contract.
- Keep providers that do not report usage working by omitting usage events.

**Non-Goals:**

- Implement local tokenizer-based token counting.
- Calculate prices or provider-specific billing totals inside `iron-providers`.
- Guarantee usage for all OpenAI-compatible gateways or local endpoints.
- Introduce a request/session identifier in `ProviderEvent`.
- Change existing output, tool-call, choice-request, completion, or error semantics except for adding optional usage events before completion.

## Decisions

### Usage Events Are Cumulative Snapshots

`ProviderEvent::Usage` will represent the provider's cumulative usage snapshot for the current request. Consumers must treat later usage events in the same request as superseding earlier usage events rather than adding them together.

Alternatives considered:

- Incremental deltas: rejected because provider APIs generally report cumulative request totals, especially Anthropic streaming `message_delta.usage` and OpenAI final usage chunks.
- Final-only usage: rejected as a universal rule because Anthropic can provide useful cumulative snapshots during streaming. Adapters may still emit only final usage when that is the only useful data available.

### TokenUsage Preserves Provider-Relevant Buckets

The normalized usage type will include optional fields for common totals and provider-specific buckets that matter for cost and context accounting:

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cached_input_tokens`
- `cache_creation_input_tokens`
- `cache_read_input_tokens`
- `reasoning_output_tokens`

All fields are optional because provider families differ in what they return. OpenAI-style `prompt_tokens` maps to `input_tokens`; `completion_tokens` maps to `output_tokens`; `prompt_tokens_details.cached_tokens` maps to `cached_input_tokens`; and reasoning/thinking output detail maps to `reasoning_output_tokens`. Anthropic cache creation/read fields remain separate rather than being collapsed into `cached_input_tokens`.

Alternatives considered:

- Minimal `input/output/cached/total` shape: rejected because it blurs Anthropic cache creation vs cache read semantics and loses reasoning/thinking output counts needed for cost telemetry.
- Raw provider metadata passthrough: rejected for the first iteration because consumers need a stable normalized API and no current use case requires preserving arbitrary usage subfields.

### Usage Must Precede Complete When Both Are Known

The existing `TerminatingStream` stops forwarding events after `ProviderEvent::Complete` or `ProviderEvent::Error`. Therefore, adapters must emit any usage found in terminal provider messages before emitting `Complete`.

This affects terminal points differently by family:

- Responses streaming: emit usage from `response.completed.response.usage`, then `Complete`.
- Messages streaming: emit the last known cumulative usage before or at `message_stop`, then `Complete`.
- Chat Completions streaming: emit usage from the final usage chunk before `[DONE]`; `[DONE]` still emits `Complete`.

Alternatives considered:

- Attach usage to `Complete`: rejected because it would overload the existing terminal event and make incremental usage impossible.
- Emit usage after `Complete`: rejected because the stream termination wrapper would drop it.

### Chat Completions Streaming Usage Is Opt-In and Compatibility-Sensitive

OpenAI Chat Completions streaming requires `stream_options.include_usage = true` to return a final usage chunk. Some OpenAI-compatible providers may reject unknown `stream_options` fields.

The implementation should enable usage for providers/endpoints known to accept the option, or make the behavior conditional through adapter/profile configuration if needed. Providers that do not accept or return streaming usage must continue to work without usage events.

Alternatives considered:

- Always include `stream_options.include_usage`: simple, but risks breaking local or gateway providers that implement only a subset of the OpenAI-compatible API.
- Never request streaming usage: safe, but fails the streaming use case for compliant providers.

## Risks / Trade-offs

- OpenAI-compatible provider rejects `stream_options.include_usage` -> Gate the request option by provider/profile support or keep unsupported providers on the existing stream body.
- Consumers accidentally sum repeated usage snapshots -> Document cumulative/superseding semantics and add tests that demonstrate multiple usage events are snapshots.
- Anthropic input context pressure is undercounted if cache buckets are ignored -> Preserve cache creation/read buckets separately and document that consumers needing Anthropic total input pressure should include them.
- Usage may be missing on interrupted streams -> Treat usage as optional; consumers must retain fallback estimation for requests that complete without provider usage.
- Adding a `ProviderEvent` variant is a public API change -> Update model tests, serialization expectations, and downstream release notes.

## Migration Plan

No data migration is required. This is an additive public API change.

Implementation should proceed by adding the normalized usage type and event variant first, then adding adapter-specific parsing and tests family by family. Downstream consumers can adopt `ProviderEvent::Usage` opportunistically while retaining their existing estimation fallback for providers or interrupted streams without usage.

Rollback is straightforward: consumers can ignore `Usage` events, and providers that do not report usage already behave as omission cases.

## Resolved Implementation Notes

- The first implementation parses Chat Completions streaming usage when a provider supplies it, but does not add `stream_options.include_usage` by default. This avoids breaking OpenAI-compatible gateways and local endpoints that reject unknown streaming options.
- The first implementation emits Anthropic streaming usage snapshots as they arrive and re-emits the last known snapshot before `Complete`, preserving the cumulative snapshot contract while making terminal usage observable.
