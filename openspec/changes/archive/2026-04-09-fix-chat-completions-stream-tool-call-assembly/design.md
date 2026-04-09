## Context

The current Chat Completions streaming adapter in `src/completions.rs` uses a single `Option<ToolCallAccumulator>` for all streamed tool-call state. That design is sufficient only when a provider emits one tool call in a simple linear sequence and the adapter treats each transport chunk as if it were a semantic boundary. Issue `#1` shows the immediate failure mode: partial argument JSON is emitted as `ProviderEvent::ToolCall` before the tool call is complete.

This change is broader than a one-line bug fix because the underlying stream model is too weak for OpenAI-compatible Chat Completions streams. The streamed schema includes both `choice.index` and `delta.tool_calls[].index`, which means the protocol already distinguishes independent assembly tracks. A robust implementation must preserve that structure internally even though the crate's normalized event API does not currently expose choice metadata.

Constraints:

- `ProviderEvent::ToolCall` already carries a completed semantic tool call, not a partial fragment.
- The crate supports multiple OpenAI-compatible providers, so the parser must be tolerant of missing or reordered optional fields while still preserving semantic correctness.
- The design should avoid introducing a breaking public API unless that becomes necessary later.

## Goals / Non-Goals

**Goals:**

- Assemble streamed Chat Completions tool calls using provider-native identifiers: choice index and tool-call index.
- Emit `ProviderEvent::ToolCall` only after the adapter has a semantically complete tool call boundary.
- Keep text streaming incremental so output tokens continue to flow as they arrive.
- Preserve correctness for providers that send final deltas and completion markers in the same chunk or defer `id` and `name` fields until later chunks.
- Add test coverage for the failure modes that the current implementation does not protect.

**Non-Goals:**

- Changing the public `ProviderEvent` API to expose raw tool-call fragments.
- Generalizing all provider adapters onto one shared stream engine in this change.
- Expanding the public request API to support multi-choice streaming semantics.
- Solving every possible OpenAI-compatible provider deviation beyond the stream fields required for correct assembly.

## Decisions

### 1. Introduce an explicit Chat Completions stream assembler state model

The adapter should separate transport parsing from semantic assembly.

Internally, the stream should be modeled as state keyed by `choice.index`, with each choice owning a map keyed by `tool_call.index`. Each tool-call entry stores the best-known `id`, `name`, and concatenated argument string until the choice reaches a completion boundary.

Why this decision:

- It matches the provider-native stream schema instead of collapsing all state into one global accumulator.
- It prevents fragments from separate tool calls or separate choices from being merged accidentally.
- It leaves room to support richer multi-choice behavior later without reworking the parser again.

Alternative considered: keep a single accumulator and flush when a new tool call starts.

- Rejected because a single accumulator cannot safely represent multiple tool-call indexes or multiple choices.
- Rejected because a new tool-call fragment is not a reliable semantic completion boundary.

### 2. Use semantic completion markers to flush pending tool calls

The adapter should treat `finish_reason` on a streamed choice as the primary semantic completion boundary for that choice. Pending tool calls for that choice should be finalized only after all deltas in the chunk have been applied. The transport-level `[DONE]` marker should act as a final safety flush for any pending state left by providers that omit or delay a usable `finish_reason`.

Why this decision:

- `finish_reason` is choice-scoped and better aligned with response semantics than the whole-stream `[DONE]` marker.
- Processing all deltas before flushing handles chunks that contain both final tool-call fragments and a completion marker.
- Keeping `[DONE]` as a safety flush improves compatibility with imperfect provider implementations.

Alternative considered: flush only on `[DONE]`.

- Rejected because it delays all tool calls to the end of the whole stream and ignores the protocol's per-choice completion signal.

Alternative considered: flush at the end of every chunk.

- Rejected because transport chunk boundaries do not imply semantic completion.

### 3. Keep `ProviderEvent::ToolCall` as a completed-only event

The adapter should continue to emit `ProviderEvent::ToolCall` only when a tool call is complete enough for downstream consumers to act on it. During assembly, arguments remain raw concatenated strings. At flush time, the adapter attempts JSON parsing once; if parsing fails, it emits the final accumulated string as `Value::String` rather than inventing an earlier partial event.

Why this decision:

- It preserves the existing semantic meaning of `ProviderEvent::ToolCall`.
- It avoids leaking transport-level fragmentation into the normalized event API.
- It keeps compatibility with current consumers while fixing the premature-emission bug.

Alternative considered: add a new partial tool-call event now.

- Rejected for this change because it expands the public API and solves a different problem than the one reported.

### 4. Preserve incremental text output independently from tool-call assembly

Text deltas should continue to emit `ProviderEvent::Output` as soon as they arrive, regardless of whether a tool call is still being assembled in the same choice.

Why this decision:

- It preserves the low-latency streaming behavior users already expect.
- Text output and tool-call assembly have different semantic completion rules and should not block each other.

Alternative considered: buffer all events until the choice finishes.

- Rejected because it would unnecessarily regress text streaming latency.

### 5. Prefer full typed stream models over handwritten partial structs

The current `ChatCompletionStreamChunk` structs only model a subset of fields and omit important identifiers such as tool-call indexes and finish reasons. The preferred implementation is to reuse richer typed Chat Completions stream models already available from existing dependencies when doing so does not pull the adapter into an awkward new dependency boundary. If that proves impractical, the local serde structs may be expanded, but they must expose the same required fields explicitly.

Why this decision:

- The current local structs hide protocol information the assembler needs.
- Reusing existing typed models reduces protocol drift and makes future maintenance easier.
- Keeping an explicit fallback to local types preserves implementation flexibility without weakening the design requirements.

Alternative considered: keep the local structs and bolt on ad hoc optional fields.

- Acceptable only if the local types are expanded to fully surface the fields the assembler needs, including `choice.index`, `finish_reason`, and `tool_call.index`.

### 6. Support multi-choice streams internally without expanding the public API in this change

The adapter should maintain state per `choice.index` even though the normalized `ProviderEvent` API does not currently expose choice identity. This change does not expand the public API to model multi-choice output explicitly, but the internal assembler must remain choice-safe so future public API work does not require another parser redesign.

Why this decision:

- It prevents cross-choice state corruption even if most current consumers use a single choice.
- It keeps the streaming parser aligned with the provider protocol.
- It future-proofs the internal architecture without forcing a public API change into this bug fix.

Alternative considered: assume `choice.index == 0` and ignore the rest.

- Rejected because it bakes a protocol assumption into the parser and makes later multi-choice support more expensive.

### 7. Emit finalized tool calls in tool index order before `ProviderEvent::Complete`

When a choice reaches a semantic completion boundary, the adapter should emit all finalized `ProviderEvent::ToolCall` events for that choice in ascending tool-call index order and only then emit `ProviderEvent::Complete` for the choice or stream.

Why this decision:

- Tool-call index order is the most stable protocol-derived ordering available.
- Emitting completed tool calls before `Complete` preserves the expectation that `Complete` is the terminal event for that unit of stream work.
- It makes regression tests deterministic.

Alternative considered: emit tool calls in first-seen order or completion order.

- Rejected because those orderings are more sensitive to provider chunking details and can become nondeterministic across compatible providers.

### 8. Use best-effort finalization for incomplete metadata

If a provider omits or delays tool-call metadata such as `id` or function `name`, the adapter should preserve the latest non-empty values seen and emit the best completed tool-call event it can at finalization time, rather than failing the whole stream by default. A malformed-but-finalized tool call should be observable through tracing.

Why this decision:

- OpenAI-compatible providers vary in how consistently they populate optional fields across streamed deltas.
- Best-effort finalization preserves downstream utility while avoiding brittle provider-specific failure modes.
- Tracing still allows malformed provider behavior to be diagnosed.

Alternative considered: treat missing final metadata as a hard provider error.

- Rejected for this change because the crate aims to provide normalized semantic events across provider variations and already uses tolerant fallback behavior for invalid final JSON arguments.

### 9. Keep the implementation scoped to the Chat Completions adapter and add targeted tracing

This change should remain local to the Chat Completions streaming path in `src/completions.rs`. It may borrow patterns from other adapters, but it should not introduce a shared cross-adapter stream framework in the same change. The implementation should add light `tracing` around malformed or unusual finalization cases, such as safety flushes on `[DONE]`, missing final metadata, or invalid final JSON arguments.

Why this decision:

- The immediate risk is isolated to the Chat Completions path.
- A local change minimizes scope and makes rollback simpler.
- Targeted tracing improves debuggability without forcing public API or architectural expansion.

Alternative considered: build a shared internal stream assembler abstraction now.

- Rejected because it would broaden the change and mix a bug fix with a larger refactor.

## Risks / Trade-offs

- Delaying tool-call emission until semantic completion may emit tool calls later than the current buggy behavior. → Mitigation: document that `ProviderEvent::ToolCall` represents a completed call and preserve incremental text output so user-visible streaming remains responsive.
- Some OpenAI-compatible providers may omit `finish_reason` or send malformed final payloads. → Mitigation: use `[DONE]` as a final safety flush and preserve the final raw argument string if JSON parsing fails.
- Some providers may omit final `id` or function name metadata. → Mitigation: preserve the latest non-empty metadata seen, emit best-effort finalized tool calls, and add tracing for incomplete finalization.
- Maintaining per-choice and per-tool-call state adds complexity to `src/completions.rs`. → Mitigation: isolate state transitions in a dedicated assembler function or struct rather than spreading logic across the SSE loop.
- The crate still does not expose explicit multi-choice semantics publicly. → Mitigation: keep choice-separated internal state now so a future public API change does not require redesigning the parser again.

## Migration Plan

This is an internal behavioral fix for streamed Chat Completions adapters and does not require user migration. The rollout plan is to land the parser change together with regression tests covering partial JSON assembly, multi-tool-call streams, mixed content-plus-tool-call streams, and final flush behavior.

If regressions are found, rollback is straightforward because the change is isolated to the Chat Completions streaming path and its tests.

## Open Questions

- Do any target OpenAI-compatible providers emit stream patterns that require additional buffering rules beyond choice-scoped `finish_reason` and `[DONE]`, such as provider-specific terminal chunks with omitted deltas? The current design assumes those two boundaries are sufficient when combined with best-effort safety flushing.
