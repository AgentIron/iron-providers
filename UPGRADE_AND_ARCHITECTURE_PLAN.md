# Upgrade And Architecture Plan

## Purpose

This document is a self-contained implementation plan for:

1. Updating direct dependencies that are not currently on the latest published release line or latest patch within the declared range.
2. Fixing the main design and architecture concerns identified in the current codebase.

This plan is written to be usable by an automated or human system that does not have any prior context from the review conversation.

## Repository Scope

The repository is a Rust crate that provides a normalized provider boundary over:

- OpenAI Responses API
- OpenAI-compatible Chat Completions APIs
- Anthropic Messages APIs

Core files involved in the planned work:

- `Cargo.toml`
- `Cargo.lock`
- `src/model.rs`
- `src/provider.rs`
- `src/generic_provider.rs`
- `src/openai.rs`
- `src/completions.rs`
- `src/anthropic.rs`
- `src/profile.rs`
- `src/error.rs`
- `src/registry.rs`
- `tests/*`
- `src/mock_provider_tests.rs`

## Current State Summary

### Direct dependency status

Based on the manifest and crates.io state as of April 19, 2026:

- `tokio`: manifest `1`, lockfile `1.51.1`, latest `1.52.1`
- `uuid`: manifest `1`, lockfile `1.23.0`, latest `1.23.1`
- `reqwest`: manifest `0.12`, lockfile `0.12.28`, latest `0.13.2`
- `async-openai`: manifest `0.34`, lockfile `0.34.0`, latest `0.35.0`
- `thiserror`: manifest `1`, latest major `2.0.18`

Other direct dependencies appear current relative to the declared requirement line.

### Main architectural concerns

1. Anthropic streaming state is modeled as one global in-flight tool call instead of per content block.
2. OpenAI Responses streaming emits `Complete` unconditionally, including after failure.
3. The profile abstraction is inconsistent across provider families. `OpenAiResponses` ignores profile auth and header metadata.
4. Runtime-owned structured state is projected into model-visible assistant text instead of a first-class runtime/system channel.
5. `InferenceRequest.stream` duplicates call-site behavior and is not actually authoritative.
6. Request/client construction and SSE parsing are too permissive and fail soft where they should fail fast.

## Assumptions

The implementing system should not rely on hidden assumptions. The following assumptions are explicit:

1. This crate is intended to remain a semantic boundary, not a thin pass-through SDK wrapper.
2. Consumers care about stable normalized behavior more than preserving accidental quirks.
3. A breaking API change is acceptable if needed to remove ambiguous or unsafe semantics.
4. Patch-level upgrades within an existing semver range should be done immediately unless they expose a regression.
5. Minor-version upgrades for `0.x` crates must be treated as potentially breaking.
6. It is acceptable to add new internal types and helper modules if they improve correctness and maintainability.
7. The test environment may not always allow local mock servers to bind. Pure unit tests should therefore cover protocol assembly and stream parsing logic independently from socket-based integration tests.
8. There is no requirement to preserve the exact internal representation of `SystemStructured` if a better normalized boundary is introduced.

If any of these assumptions are false, this plan should be adjusted before implementation.

## Goals

### Primary goals

- Make streaming semantics correct and consistent across provider families.
- Remove ambiguous or misleading API surface area.
- Ensure provider profiles are honored consistently.
- Upgrade dependencies safely with explicit compatibility checkpoints.

### Non-goals

- Adding new provider families.
- Redesigning the public crate around a completely different async model.
- Adding retries, circuit breakers, or advanced connection pooling unless required by an upgrade.
- Solving all future provider-specific quirks in this change.

## Workstreams

The work should be executed in four workstreams:

1. Lockfile refreshes with minimal code risk.
2. Deliberate dependency-line upgrades with compatibility validation.
3. Architectural correctness fixes.
4. Test and documentation hardening.

## Workstream 1: Safe dependency refreshes

### Scope

Refresh direct dependencies already within allowed semver ranges:

- `tokio` to latest `1.x`
- `uuid` to latest `1.x`

### Steps

1. Run a dependency update restricted to these crates.
2. Rebuild and run the test suite.
3. Check for transitive changes in async runtime, macros, and feature resolution.

### Expected code impact

- Likely none.
- If lockfile updates trigger new warnings or feature interactions, address them in a separate small commit.

### Acceptance criteria

- `Cargo.lock` reflects latest patch releases for `tokio` and `uuid`.
- No public API changes.
- Build, fmt, clippy, and tests pass.

## Workstream 2: Deliberate dependency-line upgrades

### 2.1 `reqwest` from `0.12` to `0.13`

#### Why

- The crate is on an older release line.
- This adapter layer relies on `reqwest` for Chat Completions and Anthropic transport and manual SSE consumption, so transport semantics matter.

#### Risks

- API changes in client builder behavior.
- Stream/body behavior changes.
- Header/value types or feature flag changes.

#### Plan

1. Read the `reqwest` `0.12 -> 0.13` changelog and migration notes before editing.
2. Update the manifest to `0.13`.
3. Rebuild and fix compilation errors first.
4. Re-run all transport and stream parser tests.
5. Confirm streaming still behaves correctly for partial chunk boundaries.

#### Acceptance criteria

- All adapters using `reqwest` compile and pass tests.
- No silent fallback client creation remains.
- Streaming path still handles chunk fragmentation.

### 2.2 `async-openai` from `0.34` to `0.35`

#### Why

- The OpenAI Responses adapter is pinned one minor line behind.
- This crate’s public description explicitly mentions Responses API support.

#### Risks

- Type changes in `async-openai::types::responses::*`
- Event enum changes for streaming
- Tool or response schema changes

#### Plan

1. Read the crate changelog and diff the `responses` type surface used in `src/openai.rs`.
2. Update the manifest to `0.35`.
3. Fix compilation against the new API.
4. Revisit the stream handling logic rather than doing a narrow compatibility patch.
5. Re-run request shaping and streaming tests.

#### Acceptance criteria

- `src/openai.rs` is compatible with `async-openai 0.35`.
- Streaming completion/error semantics are corrected as described in Workstream 3.
- Request shaping tests still validate transcript, tools, and tool outputs.

### 2.3 `thiserror` from `1` to `2`

#### Why

- The direct dependency is on the old major line.
- The lockfile already includes `thiserror 2.x` transitively, so the ecosystem around this crate has already moved.

#### Risks

- Derive macro behavior differences.
- Compiler diagnostics from stricter generated trait impls.

#### Plan

1. Update `Cargo.toml` from `thiserror = "1"` to `thiserror = "2"`.
2. Rebuild.
3. Fix any derive-related errors in `src/error.rs` and anywhere else affected.
4. Confirm the public error variants and `Display` messages remain unchanged unless intentionally modified.

#### Acceptance criteria

- `ProviderError` behavior remains stable.
- All tests asserting error classification still pass.

## Workstream 3: Architecture fixes

## 3.1 Make stream termination semantics explicit and correct

### Problem

`ProviderEvent::Complete` currently acts as a success marker, but some implementations can emit `Error` and still later emit `Complete`.

### Proposed design

Define a stricter contract:

- `Complete` means successful end of stream.
- If a provider emits an unrecoverable error, the stream ends without `Complete`.
- Recoverable informational events should use `Status`.

### Implementation steps

1. Audit every producer of `ProviderEvent::Complete`.
2. Change `src/openai.rs` so failed response streams do not append unconditional `Complete`.
3. Confirm `src/completions.rs` only emits `Complete` on `[DONE]` after successful assembly.
4. Confirm `src/anthropic.rs` only emits `Complete` on `message_stop` or equivalent successful terminal event.
5. Document this contract in public docs for `ProviderEvent`.

### Optional improvement

Consider replacing `Complete` with an explicit terminal event enum in a future breaking revision:

- `Finished`
- `Failed`

This is not required if the stricter contract above is implemented consistently.

### Acceptance criteria

- No stream emits both terminal failure and `Complete`.
- Tests cover success and failure termination separately.

## 3.2 Remove single-slot Anthropic tool-call assembly

### Problem

Anthropic streaming currently keeps only one in-flight tool call for the entire response stream.

### Proposed design

Introduce a general content-block assembler keyed by block index or provider event identity.

Suggested internal structure:

```rust
struct AnthropicStreamAssembler {
    blocks: BTreeMap<u32, AnthropicBlockState>,
}

enum AnthropicBlockState {
    Text { content: String },
    ToolUse {
        id: Option<String>,
        name: Option<String>,
        arguments_json: String,
    },
}
```

### Implementation steps

1. Inspect actual Anthropic stream event payloads and capture the field that identifies content block position.
2. Extend deserialization structs to include that identifier.
3. Replace `Option<ToolCallAccumulator>` with a per-block map.
4. Handle multiple tool blocks in one response correctly.
5. Ensure text deltas and tool-use deltas do not overwrite each other.
6. Finalize only the block that receives `content_block_stop`.

### Acceptance criteria

- Multiple tool calls in one Anthropic stream are supported.
- Interleaved tool/text blocks do not corrupt state.
- Missing metadata is handled explicitly and predictably.

## 3.3 Stop projecting runtime-owned structured state as assistant text

### Problem

`Message::SystemStructured` is currently converted into assistant-visible text in all provider adapters. That pollutes model-visible conversation history with runtime data.

### Proposed design

Split transcript data into:

- Model-visible conversation messages
- Runtime-only records

There are two acceptable implementation options.

### Option A: Introduce a first-class runtime record outside `Message`

Suggested shape:

```rust
pub struct InferenceContext {
    pub transcript: Transcript,
    pub runtime_records: Vec<RuntimeRecord>,
}
```

This is the cleaner architecture.

### Option B: Keep the enum but mark non-projectable messages explicitly

Suggested behavior:

- `SystemStructured` is not replayed into provider requests by default.
- Providers only project it if a provider-specific mapping is explicitly defined.

This is less invasive but weaker.

### Recommended choice

Use Option A if public API breakage is acceptable. Use Option B only if strict compatibility must be preserved.

### Implementation steps

1. Decide whether a public API break is acceptable.
2. Remove the current JSON-string projection into assistant text.
3. If runtime records still need to influence requests, map them into a dedicated provider-native system/instructions channel with explicit translation logic.
4. Document which message types are model-visible and which are runtime-only.

### Acceptance criteria

- Runtime-owned records are no longer silently treated as prior assistant output.
- Provider request assembly becomes explicit about what enters the model context.

## 3.4 Make streaming mode single-source-of-truth

### Problem

`InferenceRequest.stream` exists, but actual behavior is determined by whether the caller invokes `infer` or `infer_stream`.

### Proposed design

Choose one authoritative mechanism.

### Recommended choice

Remove `InferenceRequest.stream` entirely in the next breaking revision.

Reason:

- The provider trait already splits streaming and non-streaming execution paths.
- Keeping both mechanisms invites misuse and stale state.

### Compatibility fallback

If public API compatibility must be preserved in the short term:

1. Deprecate `InferenceRequest.stream`.
2. Add rustdoc explaining that it is ignored.
3. Remove it in the next semver-breaking release.

### Acceptance criteria

- There is no duplicate streaming control path.
- Public docs clearly describe how to request streaming behavior.

## 3.5 Make provider profiles actually authoritative across families

### Problem

`ProviderProfile` contains auth strategy, headers, purpose, and quirks, but `ApiFamily::OpenAiResponses` currently only uses base URL and API key through `OpenAiConfig`.

### Proposed design

Create a shared transport/config normalization layer for all provider families.

Suggested internal structure:

```rust
pub struct ResolvedProviderConfig {
    pub family: ApiFamily,
    pub base_url: String,
    pub auth: ResolvedAuth,
    pub headers: HeaderMap,
    pub default_model: Option<String>,
    pub quirks: ProviderQuirks,
}
```

### Implementation steps

1. Add a resolver that combines `ProviderProfile` and `RuntimeConfig` into one validated config object.
2. Use that resolved config in all adapters.
3. For OpenAI Responses, either:
   - extend `OpenAiConfig` to support default headers and auth variants, or
   - bypass parts of `async-openai` and use a lower-level transport only if required.
4. Ensure validation fails fast when profile auth/header configuration is invalid.

### Important decision

The implementing system must explicitly decide whether `OpenAiResponses` should support the full profile model.

Recommended answer:

- Yes, if the crate is truly profile-driven.
- No, only if `OpenAiResponses` is intentionally reserved for canonical OpenAI endpoints, in which case this limitation must be documented and enforced in `ProviderRegistry`.

### Acceptance criteria

- Profile metadata is either consistently applied or intentionally constrained and documented.
- There is no silent ignoring of profile auth/header settings.

## 3.6 Fail fast on invalid client configuration

### Problem

The current code silently skips invalid headers/auth values and falls back to a default reqwest client when builder construction fails.

### Proposed design

Turn client construction into a validated operation returning `ProviderResult<Client>`.

### Implementation steps

1. Change `build_client` in `src/completions.rs` and `src/anthropic.rs` to return `ProviderResult<Client>`.
2. Treat invalid header names, invalid header values, and client builder failures as `InvalidRequest` or `General` errors.
3. Remove `unwrap_or_default()` from client creation.
4. Add unit tests for malformed profile header names and auth configuration.

### Acceptance criteria

- Misconfigured profiles fail immediately and deterministically.
- There is no silent downgrade to a default client.

## 3.7 Replace line-based SSE parsing with event-aware parsing

### Problem

The manual parsers currently assume each `data:` line contains one complete JSON payload and do not model full SSE event framing robustly.

### Proposed design

Use an event-aware SSE parser abstraction.

Options:

1. Adopt a dedicated SSE parser crate.
2. Implement a small internal parser that:
   - buffers until blank-line event boundaries
   - joins multiple `data:` lines per SSE event
   - ignores comment lines correctly
   - preserves partial chunk boundaries across network frames

### Recommended choice

Implement a small internal parser if dependency surface should stay minimal. Otherwise adopt a dedicated crate if it is well-maintained and materially reduces correctness risk.

### Implementation steps

1. Extract shared SSE parsing into an internal module.
2. Feed parsed event payloads into provider-specific event decoders.
3. Add tests for:
   - chunk split mid-line
   - multi-line `data:`
   - comment lines
   - event name lines
   - empty keepalive frames

### Acceptance criteria

- SSE parsing behavior is independent of arbitrary network chunking.
- Both Chat Completions and Anthropic adapters use the same framing logic.

## Workstream 4: Tests and docs

## 4.1 Rebalance tests away from socket-only integration

### Problem

The current mockito-based tests are useful but depend on local socket binding, which may fail in restricted environments.

### Plan

Keep integration tests, but add pure unit coverage for:

- request body shaping
- stream framing
- stream assembly
- terminal event semantics
- malformed payload handling

### Acceptance criteria

- Core protocol logic is testable without opening sockets.
- Socket-based tests remain as a secondary validation layer.

## 4.2 Add explicit compatibility tests for each architecture fix

Required new tests:

1. OpenAI Responses stream failure does not emit `Complete`.
2. Anthropic multiple tool calls in one response are assembled independently.
3. Invalid profile header configuration fails fast.
4. `SystemStructured` or its replacement no longer leaks into assistant content by default.
5. SSE parser handles multi-line event payloads and split chunks.

## 4.3 Update documentation

Update:

- `README.md`
- crate-level docs in `src/lib.rs`
- rustdoc on `ProviderEvent`, `InferenceRequest`, and transcript/runtime types

Documentation must explain:

- exact streaming contract
- profile semantics
- model-visible versus runtime-only state
- any breaking API changes

## Execution order

Recommended order:

1. Refresh `tokio` and `uuid`.
2. Upgrade `thiserror`.
3. Fix fail-fast client construction.
4. Extract shared SSE parser.
5. Fix OpenAI Responses terminal semantics.
6. Fix Anthropic streaming assembly.
7. Resolve `InferenceRequest.stream` duplication.
8. Redesign runtime-only structured state handling.
9. Make profile handling consistent across `OpenAiResponses`.
10. Upgrade `reqwest`.
11. Upgrade `async-openai`.
12. Update docs and finalize tests.

Reason for this order:

- Early steps reduce noise and improve failure visibility.
- Mid-phase architectural cleanup makes later dependency upgrades easier.
- The profile and runtime-message changes are the most behaviorally important and should land before final API documentation.

## Breaking change assessment

These changes should be classified explicitly.

### Likely non-breaking

- lockfile patch updates
- `thiserror` direct dependency upgrade if public error API is unchanged
- fail-fast internal client building
- improved SSE parser correctness
- fixing `Complete` to mean success only

### Potentially breaking

- removing or deprecating `InferenceRequest.stream`
- changing handling of `SystemStructured`
- changing `OpenAiResponses` profile semantics
- upgrading `reqwest` and `async-openai`

If the crate is already consumed externally, these should be grouped into one intentional release plan with clear migration notes.

## Validation checklist

Before merging, confirm:

- `cargo fmt --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- any repository-specific task runner commands in `tasks.py`

Behavioral validation:

- Chat Completions:
  - plain text non-streaming
  - tool call non-streaming
  - streaming text
  - streaming multiple tool calls
  - malformed tool arguments
- Anthropic:
  - plain text non-streaming
  - tool use non-streaming
  - streaming text
  - streaming multiple tool uses
  - invalid partial JSON handling
- OpenAI Responses:
  - plain text non-streaming
  - function call non-streaming
  - stream success termination
  - stream failure termination without `Complete`

## Deliverables

At completion, the implementing system should produce:

1. Updated `Cargo.toml` and `Cargo.lock`
2. Corrected streaming implementations
3. Explicit runtime/model visibility semantics
4. Consistent profile handling or explicit documentation of intentional constraints
5. Expanded tests
6. Updated public documentation
7. Release notes or migration notes for any public API break

## Recommended implementation strategy

Use small commits or PRs by workstream rather than one monolithic change.

Recommended PR split:

1. Dependency refreshes and `thiserror`
2. Fail-fast client construction and shared SSE parsing
3. OpenAI and Anthropic streaming correctness
4. Runtime/state model cleanup and streaming API cleanup
5. Profile consistency for OpenAI Responses
6. `reqwest` and `async-openai` line upgrades
7. Docs and migration notes

This split lowers regression risk and makes it easier to bisect protocol or behavior changes.
