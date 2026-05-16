# Graph Report - iron-providers  (2026-05-15)

## Corpus Check
- 43 files · ~32,772 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 958 nodes · 1465 edges · 86 communities (51 shown, 35 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 218 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `605e32a1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]

## God Nodes (most connected - your core abstractions)
1. `build_test_client()` - 31 edges
2. `iron-providers` - 17 edges
3. `GenericProvider` - 16 edges
4. `completions_profile()` - 15 edges
5. `ProviderProfile` - 13 edges
6. `ProviderError` - 12 edges
7. `ProviderRegistry` - 11 edges
8. `Decisions` - 11 edges
9. `ADDED Requirements` - 11 edges
10. `OpenAiConfig` - 11 edges

## Surprising Connections (you probably didn't know these)
- `ProviderProfile` --references--> `Anthropic system prompt fragment`  [EXTRACTED]
  src/profile.rs → src/system_prompt_fragments/anthropic.md
- `ProviderProfile` --references--> `OpenAI system prompt fragment`  [EXTRACTED]
  src/profile.rs → src/system_prompt_fragments/openai.md
- `Streaming Contract Documentation` --conceptually_related_to--> `Best-Effort Metadata Finalization`  [INFERRED]
  README.md → openspec/changes/archive/2026-04-09-fix-chat-completions-stream-tool-call-assembly/design.md
- `build_test_client()` --calls--> `build_http_client()`  [INFERRED]
  src/mock_provider_tests.rs → src/http_client.rs
- `build_client()` --calls--> `build_http_client()`  [INFERRED]
  src/anthropic.rs → src/http_client.rs

## Hyperedges (group relationships)
- **Four-provider adapter family** — anthropic_infer, anthropic_infer_stream, completions_infer, completions_infer_stream, codex_infer, codex_infer_stream, openai_infer, openai_infer_stream [INFERRED 0.90]
- **SSE streaming pipeline** — sse_sseparser, streamutil_ssestreamadapter, streamutil_process_sse_stream, streamutil_terminatingstream, anthropic_anthropicsseadapter, completions_completionssseadapter [EXTRACTED 1.00]
- **Profile-driven provider configuration** — profile_providerprofile, profile_runtimeconfig, profile_apifamily, profile_authstrategy, profile_providercredential, httpclient_httpclientparams [INFERRED 0.85]
- **Chat Completions Stream Assembly Design Decisions** — stream_assembler_state_model, semantic_completion_markers, completed_only_toolcall_events, incremental_text_independence, toolcall_index_order_emission, best_effort_finalization [EXTRACTED 1.00]
- **OAuth Credential Design Decisions** — credential_kind_separation, codex_separate_provider, jwt_routing_metadata, oauth_lifecycle_boundary [EXTRACTED 1.00]

## Communities (86 total, 35 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (56): infer(), response_to_events(), AnthropicBlockState, AnthropicContentBlock, AnthropicDelta, AnthropicError, AnthropicErrorBody, AnthropicRequest (+48 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (35): handle_error(), handle_error(), handle_error(), test_handle_error_status_codes(), ProviderError, build_http_client(), HttpClientParams, build_client() (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (51): AnthropicSseAdapter, AnthropicStreamAssembler, anthropic::infer, anthropic::infer_stream, chatgpt_account_id_from_jwt, codex::infer, codex::infer_stream, CompletionsSseAdapter (+43 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (41): anthropic_profile(), anthropic_response(), build_test_client(), chat_completion_response(), chat_completion_tool_response(), completions_profile(), responses_profile(), responses_response() (+33 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (42): ADDED Requirements, MODIFIED Requirements, Requirement: API adapters SHALL be organized by API type under `src/apis/`, Requirement: Auth SHALL be resolved separately from HTTP transport construction, Requirement: Header composition SHALL protect required headers from accidental override, Requirement: Legacy provider-specific public connection APIs SHALL be removed, Requirement: Provider architecture SHALL separate provider identity from API protocol behavior, Requirement: Provider registry SHALL dispatch all supported API families (+34 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (35): API Families, code:toml ([dependencies]), code:rust (let profile = ProviderProfile::new("kimi", ApiFamily::OpenAi), code:rust (pub trait Provider: Send + Sync {), code:bash (pip install invoke), code:bash (invoke build), code:bash (cargo test -p iron-providers), code:rust (use iron_providers::{) (+27 more)

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (23): test_registry_get_anthropic_provider(), test_registry_get_completions_provider(), test_registry_unknown_provider_error(), ProviderRegistry, test_blank_api_key_fails_during_registry_construction(), test_blank_oauth_token_fails_during_registry_construction(), test_builtins_registered(), test_case_insensitive_lookup() (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (12): ChoiceItem, ChoiceRequest, ChoiceSelectionMode, GenerationConfig, InferenceContext, InferenceRequest, ProviderEvent, RuntimeRecord (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (30): 10. Keep model capability source external, 1. Model credential kind separately from wire auth strategy, 2. Preserve API-key construction and add credential-aware construction, 3. Add per-credential profile auth metadata, 4. Validate selected credentials early and clearly, 5. Apply auth from selected credential and selected strategy, 6. Add `kimi-code` OAuth without changing `kimi`, 7. Add Codex as a new API family and provider profile (+22 more)

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (29): build_chat_messages(), build_chat_tools(), build_client(), build_request_body(), ChatChoice, ChatCompletionResponse, ChatCompletionStreamChunk, ChatError (+21 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (21): build_chat_messages(), build_chat_tools(), build_request_body(), ChatChoice, ChatCompletionResponse, ChatCompletionStreamChunk, ChatError, ChatErrorBody (+13 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (24): build_input_items(), build_request(), build_tools(), ContentPart, endpoint_path(), handle_error(), infer(), infer_stream() (+16 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (25): ADDED Requirements, Purpose, Requirement: Completed tool-call events SHALL contain the final accumulated payload, Requirement: Completed tool calls SHALL be emitted only at semantic completion boundaries, Requirement: Incremental text output SHALL remain streamable during tool-call assembly, Requirement: Streamed tool calls SHALL be assembled by streamed identity, Requirement: Unusual finalization paths SHALL be observable, Requirements (+17 more)

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (23): build_codex_request_body(), build_reqwest_headers(), codex_profile(), CodexContentBlock, CodexErrorBody, CodexOutputItem, CodexResponse, CodexStreamEvent (+15 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (22): 1. `ApiFamily` names API protocols, not providers, 2. `ProviderConnection` is the resolved provider state, 3. Provider-specific configuration is centralized, 4. Auth header production is separate from HTTP client construction, 5. Protected header collisions fail clearly, 6. API adapters live under `src/apis/`, 7. OpenAI Responses and Codex share the Responses adapter and upstream wire contract, 8. Remove `async-openai` (+14 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (13): auth_headers(), test_api_key_header(), test_bearer_token(), test_custom_header_with_prefix(), test_custom_header_without_prefix(), test_invalid_header_name(), test_invalid_header_value(), compose_headers() (+5 more)

### Community 16 - "Community 16"
Cohesion: 0.25
Nodes (16): anthropic_style_event_with_name(), chunk_split_mid_line(), chunk_split_mid_newline(), comment_lines_ignored(), crlf_line_endings(), data_without_space_after_colon(), done_marker_preserved_as_data(), empty_keepalive_ignored() (+8 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (7): Message, test_generation_config_default(), test_message_assistant(), test_message_tool(), test_message_user(), test_transcript_add_message(), test_transcript_with_messages()

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (18): ADDED Requirements, Requirement: Profile fragments SHALL be selected by API family, Requirement: Prompt fragments SHALL be safe for caller-side template injection, Requirement: Provider modules SHALL expose static prompt fragments, Requirement: Registry lookup SHALL resolve prompt fragments for registered providers, Scenario: Anthropic Messages profile resolves Anthropic guidance, Scenario: Anthropic module exposes a fragment, Scenario: Default registry providers resolve fragments (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (12): ApiFamily, AuthStrategy, CredentialAuthConfig, CredentialKind, EndpointPurpose, ProviderCredential, ProviderQuirks, RuntimeConfigSource (+4 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (15): 1. Introduce an explicit Chat Completions stream assembler state model, 2. Use semantic completion markers to flush pending tool calls, 3. Keep `ProviderEvent::ToolCall` as a completed-only event, 4. Preserve incremental text output independently from tool-call assembly, 5. Prefer full typed stream models over handwritten partial structs, 6. Support multi-choice streams internally without expanding the public API in this change, 7. Emit finalized tool calls in tool index order before `ProviderEvent::Complete`, 8. Use best-effort finalization for incomplete metadata (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.24
Nodes (10): collect(), drops_complete_after_err_result(), drops_complete_after_error_event(), drops_events_after_complete(), is_terminal(), passes_through_non_terminal_events(), process_sse_stream(), SseStreamAdapter (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.16
Nodes (3): ProviderProfile, test_profile_with_auth_replaces_api_key_config(), test_profile_with_credential_auth()

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (14): Best-Effort Metadata Finalization, Chat Completions Stream Assembly Spec (archived), Chat Completions Stream Assembly Spec (canonical), Completed-Only ProviderEvent::ToolCall Policy, Provider Credentials Documentation, Chat Completions Stream Assembly Fix Design, Chat Completions Stream Assembly Fix Proposal, Chat Completions Stream Assembly Fix Tasks (+6 more)

### Community 24 - "Community 24"
Cohesion: 0.27
Nodes (12): build(), _failure_summary(), _lockfile_path(), _print_summary(), Run the test suite with a terse summary., Run security checks with a terse summary., Run the build and lint checks with a terse summary., _run_group() (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.17
Nodes (11): 1. Store fragments as Markdown files included at compile time, 2. Provide module-level helpers for existing provider APIs, 3. Map profile fragments by `ApiFamily`, 4. Add registry lookup without changing registered providers, 5. Keep fragment validation lightweight and local, Context, Decisions, Goals / Non-Goals (+3 more)

### Community 26 - "Community 26"
Cohesion: 0.22
Nodes (10): base64_decode_url_safe(), build_codex_headers(), chatgpt_account_id_from_jwt(), test_build_codex_headers_with_account_id(), test_build_codex_headers_with_namespaced_account_id_from_access_token(), test_build_codex_headers_without_id_token(), test_runtime_config_from_credential_api_key(), test_runtime_config_from_credential_oauth_bearer() (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.18
Nodes (10): 10. Verification, 1. Credential Model, 2. Profile Auth Metadata, 3. Auth Validation And Header Application, 4. Kimi Code OAuth Support, 5. Codex Provider Profile And Dispatch, 6. Codex Request Implementation, 7. JWT Account Metadata (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.22
Nodes (8): 1. API Family And Public Surface, 2. Auth And Header Boundaries, 3. Provider Overrides, 4. ProviderConnection, 5. API Adapter Reorganization, 6. Responses Adapter Unification, 7. Registry, Profiles, And Documentation, 8. Verification

### Community 31 - "Community 31"
Cohesion: 0.25
Nodes (7): Capabilities, Impact, Modified Capabilities, New Capabilities, Related, What Changes, Why

### Community 32 - "Community 32"
Cohesion: 0.25
Nodes (7): Capabilities, Impact, Modified Capabilities, New Capabilities, Out Of Scope, What Changes, Why

### Community 34 - "Community 34"
Cohesion: 0.43
Nodes (8): Codex as Separate Provider from OpenAI, Credential Kind vs Wire Auth Strategy Separation, Unverified JWT Routing Metadata Extraction, Provider Credential OAuth Design, Provider Credential OAuth Proposal, Provider Credential OAuth Spec, Provider Credential OAuth Tasks, OAuth Refresh Lifecycle Ownership Boundary

### Community 35 - "Community 35"
Cohesion: 0.29
Nodes (6): Capabilities, Impact, Modified Capabilities, New Capabilities, What Changes, Why

### Community 36 - "Community 36"
Cohesion: 0.29
Nodes (6): Capabilities, Impact, Modified Capabilities, New Capabilities, What Changes, Why

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (6): Requirement: Provider construction SHALL validate credential compatibility, Scenario: Blank API key fails as authentication, Scenario: Blank OAuth access token fails as authentication, Scenario: Expired OAuth bearer token fails before requests, Scenario: OAuth bearer token without expiry is accepted, Scenario: Unsupported credential kind fails before requests

### Community 39 - "Community 39"
Cohesion: 0.53
Nodes (6): ApiFamily-Based Fragment Resolution, Compile-Time Markdown Fragment Inclusion, Provider System Prompt Fragments Design, Provider System Prompt Fragments Proposal, Provider System Prompt Fragments Spec, Provider System Prompt Fragments Tasks

### Community 40 - "Community 40"
Cohesion: 0.4
Nodes (4): 1. Fragment Content, 2. Public API, 3. Tests, 4. Verification

### Community 41 - "Community 41"
Cohesion: 0.4
Nodes (5): Requirement: Codex requests SHALL target the ChatGPT/Codex Responses endpoint, Scenario: Codex request includes normalized request fields, Scenario: Codex request includes required fixed body fields, Scenario: Non-streaming Codex request uses the Codex responses URL, Scenario: Streaming Codex request uses the Codex responses URL

### Community 42 - "Community 42"
Cohesion: 0.4
Nodes (4): MODIFIED Requirements, Requirement: GenericProvider SHALL dispatch all supported API families, Scenario: Codex Responses dispatches to Codex adapter, Scenario: Existing families continue dispatching as before

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (5): Requirement: JWT account routing metadata SHALL be parsed without verification, Scenario: Malformed JWT produces no account ID, Scenario: Nested account claim is used second, Scenario: Organization ID is fallback, Scenario: Top-level account claim wins

### Community 44 - "Community 44"
Cohesion: 0.4
Nodes (5): Requirement: Codex requests SHALL include required headers, Scenario: Codex request includes account routing header when available, Scenario: Codex request includes bearer auth, Scenario: Codex request includes product headers, Scenario: Codex request omits account routing header when unavailable

### Community 45 - "Community 45"
Cohesion: 0.5
Nodes (3): 1. Stream Model, 2. Stream Assembly, 3. Regression Coverage

### Community 46 - "Community 46"
Cohesion: 0.5
Nodes (4): Requirement: Auth headers SHALL be built from the selected credential and strategy, Scenario: API-key header strategy uses configured header name, Scenario: Bearer strategy uses Authorization header, Scenario: Custom strategy preserves prefix behavior

### Community 47 - "Community 47"
Cohesion: 0.5
Nodes (4): Requirement: Codex provider SHALL be registered as a distinct provider, Scenario: Codex built-in profile exists, Scenario: Codex rejects API-key credentials unless explicitly supported later, Scenario: Codex requires OAuth bearer credentials

### Community 48 - "Community 48"
Cohesion: 0.5
Nodes (4): Requirement: Kimi Code SHALL support API-key and OAuth bearer credentials, Scenario: Kimi Code API key uses x-api-key, Scenario: Kimi Code OAuth uses bearer auth, Scenario: Kimi general remains API-key-only

### Community 49 - "Community 49"
Cohesion: 0.5
Nodes (4): Requirement: Runtime credentials SHALL distinguish API keys from OAuth bearer tokens, Scenario: Credential-aware constructor accepts API keys, Scenario: Credential-aware constructor accepts OAuth bearer tokens, Scenario: Existing API-key constructor remains usable

### Community 50 - "Community 50"
Cohesion: 0.5
Nodes (4): ADDED Requirements, Requirement: Provider behavior SHALL not own OAuth refresh lifecycle, Scenario: Runtime credential contains no refresh token, Scenario: Unauthorized response is surfaced

### Community 51 - "Community 51"
Cohesion: 0.5
Nodes (4): Requirement: Provider profiles SHALL declare supported credential kinds, Scenario: API-key-only provider supports API keys, Scenario: Existing `with_auth` helper remains API-key oriented, Scenario: Mixed-mode provider supports multiple credential kinds

## Knowledge Gaps
- **368 isolated node(s):** `Run the build and lint checks with a terse summary.`, `Run the test suite with a terse summary.`, `Run security checks with a terse summary.`, `ChoiceSelectionMode`, `ChoiceItem` (+363 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `process_sse_stream()` connect `Community 21` to `Community 9`, `Community 6`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `build_test_client()` connect `Community 3` to `Community 1`, `Community 13`, `Community 15`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Why does `ProviderProfile` connect `Community 22` to `Community 19`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `build_test_client()` (e.g. with `.auth()` and `auth_headers()`) actually correct?**
  _`build_test_client()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run the build and lint checks with a terse summary.`, `Run the test suite with a terse summary.`, `Run security checks with a terse summary.` to the rest of the system?**
  _368 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._