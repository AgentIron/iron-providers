## Why

The Chat Completions streaming adapter currently emits tool calls before their argument payloads are fully assembled, which breaks downstream consumers that expect completed JSON arguments. Fixing the immediate bug is not enough because the parser also under-models streamed tool calls as a single global accumulator instead of indexed per-choice, per-tool-call state.

## What Changes

- Define a robust streamed tool-call assembly capability for Chat Completions adapters.
- Require adapters to accumulate streamed tool-call fragments by choice index and tool-call index instead of using a single global accumulator.
- Require completed `ProviderEvent::ToolCall` events to be emitted only at semantic completion boundaries, not for partial argument fragments.
- Define how stream completion signals such as `finish_reason` and `[DONE]` interact with pending tool-call state.
- Add coverage for multi-chunk, multi-tool-call, and mixed content-plus-tool-call streaming behavior.

## Capabilities

### New Capabilities
- `chat-completions-stream-assembly`: Defines how OpenAI Chat Completions compatible adapters assemble streamed text and tool-call deltas into normalized provider events.

### Modified Capabilities

## Impact

- Affected code: `src/completions.rs`, related streaming tests, and any shared streaming abstractions introduced for adapter state management.
- Affected behavior: all providers using `ApiFamily::OpenAiChatCompletions`, including built-in slugs such as `zai`, `zai-code`, `kimi`, `openrouter`, and `requesty`, plus compatible custom profiles.
- API impact: no intended public API break, but the change tightens the semantic contract that `ProviderEvent::ToolCall` represents a completed tool call rather than a partial fragment.
