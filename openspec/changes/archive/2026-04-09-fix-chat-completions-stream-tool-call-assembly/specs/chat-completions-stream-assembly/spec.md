## ADDED Requirements

### Requirement: Streamed tool calls SHALL be assembled by streamed identity
The Chat Completions streaming adapter SHALL maintain pending tool-call state using the streamed choice index and streamed tool-call index, and it MUST NOT merge fragments that belong to different choices or different tool calls.

#### Scenario: One tool call spans multiple chunks
- **WHEN** a streamed Chat Completions response delivers one tool call across multiple chunks with the same choice index and tool-call index
- **THEN** the adapter assembles those fragments into one pending tool call instead of emitting a partial completed tool call

#### Scenario: Two tool calls share a response
- **WHEN** a streamed Chat Completions response delivers fragments for two distinct tool-call indexes in the same choice
- **THEN** the adapter maintains separate pending state for each tool call and MUST NOT concatenate their argument fragments together

#### Scenario: Two choices stream independently
- **WHEN** a streamed Chat Completions response includes fragments from different choice indexes
- **THEN** the adapter maintains separate assembly state per choice and MUST NOT merge content or tool-call fragments across those choices

### Requirement: Completed tool calls SHALL be emitted only at semantic completion boundaries
The Chat Completions streaming adapter SHALL emit `ProviderEvent::ToolCall` only after the adapter has reached a semantic completion boundary for the relevant pending tool call state. Transport chunk boundaries alone MUST NOT cause a completed tool-call event to be emitted.

#### Scenario: Partial arguments arrive before completion
- **WHEN** a tool call's `function.arguments` value arrives as partial JSON across multiple stream chunks
- **THEN** the adapter MUST NOT emit `ProviderEvent::ToolCall` until the relevant completion boundary has been reached

#### Scenario: Finish reason closes a choice
- **WHEN** a streamed choice includes pending tool-call state and a chunk indicates that the choice has finished
- **THEN** the adapter finalizes the pending tool calls for that choice after applying the chunk's deltas and emits completed `ProviderEvent::ToolCall` events

#### Scenario: Done marker closes remaining pending state
- **WHEN** the transport stream reaches `[DONE]` while any tool-call state is still pending
- **THEN** the adapter finalizes the remaining pending tool calls before ending the stream

#### Scenario: Finalized tool calls precede completion
- **WHEN** a choice reaches a semantic completion boundary with one or more completed tool calls pending
- **THEN** the adapter emits the finalized `ProviderEvent::ToolCall` events before emitting `ProviderEvent::Complete`

#### Scenario: Finalized tool calls follow tool-call index order
- **WHEN** a choice finalizes more than one completed tool call
- **THEN** the adapter emits those `ProviderEvent::ToolCall` events in ascending streamed tool-call index order

### Requirement: Completed tool-call events SHALL contain the final accumulated payload
When emitting `ProviderEvent::ToolCall`, the adapter SHALL use the fully accumulated tool-call metadata and arguments known at completion time. The adapter MUST preserve later-arriving `id`, `name`, and argument fragments until the tool call is finalized.

#### Scenario: Tool-call metadata arrives late
- **WHEN** the stream provides argument fragments before the final non-empty tool-call `id` or function `name`
- **THEN** the emitted completed tool call uses the latest accumulated `id`, function name, and full argument payload

#### Scenario: Final metadata remains incomplete
- **WHEN** a tool call reaches finalization without receiving all preferred metadata fields from the provider
- **THEN** the adapter emits the best completed tool-call event it can from the final accumulated state rather than discarding the completed tool call

#### Scenario: Final accumulated arguments are valid JSON
- **WHEN** the completed argument string is valid JSON at finalization time
- **THEN** the emitted `ProviderEvent::ToolCall` contains structured JSON arguments rather than a raw partial string

#### Scenario: Final accumulated arguments are invalid JSON
- **WHEN** the completed argument string is still invalid JSON at finalization time
- **THEN** the adapter emits a completed `ProviderEvent::ToolCall` containing the final accumulated raw string payload instead of any earlier partial fragment

### Requirement: Incremental text output SHALL remain streamable during tool-call assembly
The Chat Completions streaming adapter SHALL continue to emit `ProviderEvent::Output` incrementally for text deltas even while tool-call state is still pending.

#### Scenario: Text and tool-call deltas are interleaved
- **WHEN** a streamed response interleaves text content deltas with tool-call argument deltas
- **THEN** the adapter emits text output incrementally as content arrives and defers only the completed tool-call event until finalization

#### Scenario: Final chunk contains both tool-call data and completion state
- **WHEN** a single streamed chunk contains the last tool-call fragments and the completion marker for the choice
- **THEN** the adapter applies the final fragments first and emits a completed tool-call event using the fully accumulated payload

### Requirement: Unusual finalization paths SHALL be observable
The Chat Completions streaming adapter SHALL record traceable diagnostics for unusual stream finalization paths that require tolerant behavior, including safety flushes at `[DONE]`, incomplete final metadata, and invalid final JSON argument payloads.

#### Scenario: Safety flush occurs at transport end
- **WHEN** the adapter finalizes pending tool-call state only because the stream reached `[DONE]`
- **THEN** the adapter records a traceable diagnostic for that safety flush

#### Scenario: Final metadata is incomplete
- **WHEN** the adapter emits a best-effort completed tool call with incomplete final metadata
- **THEN** the adapter records a traceable diagnostic describing the incomplete metadata condition

#### Scenario: Final JSON parsing fails
- **WHEN** the adapter falls back to the final accumulated raw string because the completed arguments are not valid JSON
- **THEN** the adapter records a traceable diagnostic for the invalid final JSON payload
