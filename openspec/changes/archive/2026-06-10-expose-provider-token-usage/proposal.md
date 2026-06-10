## Why

`iron-core` needs provider-reported token usage to manage context pressure and cost telemetry without bundling tokenizer libraries or accumulating heuristic drift across long sessions. `iron-providers` already normalizes provider output events, but it drops the authoritative usage metadata returned by OpenAI Responses, Anthropic Messages, and OpenAI-compatible Chat Completions APIs.

## What Changes

- Add a normalized token-usage model exposed through `ProviderEvent`.
- Treat usage events as cumulative provider-reported snapshots for the current request, not incremental deltas.
- Preserve provider-relevant usage buckets for context management and cost accounting, including input, output, total, cached input, cache creation/read input, and reasoning/thinking output tokens when providers return them.
- Parse and emit usage for non-streaming Responses, Messages, and Chat Completions responses.
- Parse and emit streaming usage when providers supply it, ensuring any terminal usage is emitted before `Complete`.
- Continue to omit usage events for providers or endpoints that do not report usage.

## Capabilities

### New Capabilities

- `provider-token-usage`: Normalized provider-reported token usage events for inference responses and streams.

### Modified Capabilities

- None.

## Impact

- Affected public API: `ProviderEvent` and a new exported token-usage type.
- Affected adapters: `src/apis/responses.rs`, `src/apis/messages.rs`, and `src/apis/completions.rs`.
- Affected stream behavior: usage snapshots may appear before `Complete`; consumers must treat repeated usage events as cumulative snapshots.
- Affected tests: model/event tests and adapter tests for non-streaming and streaming usage parsing.
- No new external dependencies are expected.
