## 1. Public Usage Event Model

- [x] 1.1 Add an exported normalized token usage type with optional input, output, total, cached input, cache creation input, cache read input, and reasoning output token fields.
- [x] 1.2 Add `ProviderEvent::Usage` carrying the normalized usage type and document cumulative snapshot semantics.
- [x] 1.3 Update event/model tests for construction, matching, serialization, and deserialization of usage events.

## 2. Shared Mapping Helpers

- [x] 2.1 Add adapter-local or shared helpers to map OpenAI Responses usage fields into normalized token usage.
- [x] 2.2 Add adapter-local or shared helpers to map Anthropic Messages usage fields into normalized token usage.
- [x] 2.3 Add adapter-local or shared helpers to map OpenAI-compatible Chat Completions usage fields into normalized token usage.

## 3. Responses Adapter Usage

- [x] 3.1 Extend non-streaming Responses response parsing to deserialize top-level `usage`.
- [x] 3.2 Emit normalized Responses usage before `Complete` for non-streaming inference when usage is present.
- [x] 3.3 Extend Responses stream parsing to deserialize usage from successful completion response payloads.
- [x] 3.4 Emit streaming Responses usage before `Complete` when `response.completed` includes usage.
- [x] 3.5 Add tests covering non-streaming and streaming Responses usage parsing and event ordering.

## 4. Messages Adapter Usage

- [x] 4.1 Extend non-streaming Messages response parsing to deserialize top-level `usage`.
- [x] 4.2 Emit normalized Messages usage before `Complete` for non-streaming inference when usage is present.
- [x] 4.3 Extend Messages stream parsing to deserialize usage-bearing message events, including cumulative output and cache fields.
- [x] 4.4 Emit Messages streaming usage snapshots without treating them as additive deltas.
- [x] 4.5 Ensure the last known Messages streaming usage is observable before `Complete`.
- [x] 4.6 Add tests covering non-streaming and streaming Messages usage parsing, cache bucket preservation, and event ordering.

## 5. Chat Completions Adapter Usage

- [x] 5.1 Extend non-streaming Chat Completions response parsing to deserialize top-level `usage`.
- [x] 5.2 Emit normalized Chat Completions usage before `Complete` for non-streaming inference when usage is present.
- [x] 5.3 Extend Chat Completions stream parsing to deserialize usage-bearing chunks.
- [x] 5.4 Decide and implement the compatibility strategy for requesting `stream_options.include_usage` only where supported.
- [x] 5.5 Emit streaming Chat Completions usage chunks before `[DONE]` produces `Complete`.
- [x] 5.6 Add tests covering non-streaming usage, streaming usage chunks, unsupported streaming usage omission, and event ordering.

## 6. Documentation and Validation

- [x] 6.1 Update public docs or README examples to mention provider-reported usage events and cumulative snapshot semantics.
- [x] 6.2 Run `cargo fmt --manifest-path Cargo.toml -- --check`.
- [x] 6.3 Run `cargo clippy --manifest-path Cargo.toml --all-targets --all-features -- -D warnings`.
- [x] 6.4 Run `cargo test --manifest-path Cargo.toml`.
- [x] 6.5 Run `cargo audit` or `inv security`.
