## 1. Stream Model

- [x] 1.1 Replace or expand the Chat Completions streaming types in `src/completions.rs` so the parser can read `choice.index`, `finish_reason`, and `tool_calls[].index`, preferring richer existing typed models when they integrate cleanly.
- [x] 1.2 Introduce an internal assembler state model that tracks pending stream state by choice index and tool-call index instead of a single global accumulator.

## 2. Stream Assembly

- [x] 2.1 Refactor the Chat Completions streaming path to apply all deltas into the indexed assembler state before deciding which normalized events to emit.
- [x] 2.2 Finalize pending tool calls at semantic completion boundaries for a choice, emit them in ascending tool-call index order, and keep `[DONE]` as a final safety flush for any remaining pending state.
- [x] 2.3 Preserve incremental `ProviderEvent::Output` emission for text deltas while ensuring `ProviderEvent::ToolCall` is emitted only for completed tool calls.
- [x] 2.4 Preserve the latest non-empty tool-call metadata seen during assembly and use best-effort finalization when providers omit preferred final metadata.
- [x] 2.5 Parse tool-call arguments only at finalization time, fall back to the final accumulated raw string when the completed payload is not valid JSON, and add targeted tracing for safety flushes and malformed finalization paths.

## 3. Regression Coverage

- [x] 3.1 Add streaming tests for a single tool call whose JSON arguments span multiple chunks.
- [x] 3.2 Add streaming tests for multiple indexed tool calls in one choice and verify their fragments are not merged.
- [x] 3.3 Add streaming tests for interleaved text and tool-call deltas, including a final chunk that contains both the last fragments and the choice completion signal.
- [x] 3.4 Add streaming tests that verify finalized tool calls are emitted in ascending tool-call index order and always before `ProviderEvent::Complete`.
- [x] 3.5 Add a transport-end test that verifies `[DONE]` flushes any remaining pending tool-call state before `ProviderEvent::Complete`.
- [x] 3.6 Add streaming tests for providers that delay or omit final `id` or function-name metadata and confirm best-effort finalization behavior.
