# Graph Report - iron-providers  (2026-05-11)

## Corpus Check
- 37 files · ~30,227 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 514 nodes · 912 edges · 55 communities (17 shown, 38 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 156 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4659edf8`
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
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]

## God Nodes (most connected - your core abstractions)
1. `build_test_client()` - 23 edges
2. `GenericProvider` - 16 edges
3. `completions_profile()` - 15 edges
4. `ProviderProfile` - 13 edges
5. `ProviderError` - 12 edges
6. `OpenAiConfig` - 11 edges
7. `ProviderRegistry` - 11 edges
8. `InferenceRequest` - 11 edges
9. `infer()` - 10 edges
10. `GenericProvider` - 10 edges

## Surprising Connections (you probably didn't know these)
- `ProviderProfile` --references--> `Anthropic system prompt fragment`  [EXTRACTED]
  src/profile.rs → src/system_prompt_fragments/anthropic.md
- `ProviderProfile` --references--> `OpenAI system prompt fragment`  [EXTRACTED]
  src/profile.rs → src/system_prompt_fragments/openai.md
- `Streaming Contract Documentation` --conceptually_related_to--> `Best-Effort Metadata Finalization`  [INFERRED]
  README.md → openspec/changes/archive/2026-04-09-fix-chat-completions-stream-tool-call-assembly/design.md
- `build_client()` --calls--> `build_http_client()`  [INFERRED]
  src/anthropic.rs → src/http_client.rs
- `build_client()` --calls--> `build_http_client()`  [INFERRED]
  src/completions.rs → src/http_client.rs

## Hyperedges (group relationships)
- **Four-provider adapter family** — anthropic_infer, anthropic_infer_stream, completions_infer, completions_infer_stream, codex_infer, codex_infer_stream, openai_infer, openai_infer_stream [INFERRED 0.90]
- **SSE streaming pipeline** — sse_sseparser, streamutil_ssestreamadapter, streamutil_process_sse_stream, streamutil_terminatingstream, anthropic_anthropicsseadapter, completions_completionssseadapter [EXTRACTED 1.00]
- **Profile-driven provider configuration** — profile_providerprofile, profile_runtimeconfig, profile_apifamily, profile_authstrategy, profile_providercredential, httpclient_httpclientparams [INFERRED 0.85]
- **Chat Completions Stream Assembly Design Decisions** — stream_assembler_state_model, semantic_completion_markers, completed_only_toolcall_events, incremental_text_independence, toolcall_index_order_emission, best_effort_finalization [EXTRACTED 1.00]
- **OAuth Credential Design Decisions** — credential_kind_separation, codex_separate_provider, jwt_routing_metadata, oauth_lifecycle_boundary [EXTRACTED 1.00]

## Communities (55 total, 38 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (43): handle_error(), infer_stream(), build_chat_messages(), build_chat_tools(), build_client(), build_request_body(), ChatChoice, ChatCompletionResponse (+35 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (32): base64_decode_url_safe(), build_codex_headers(), build_codex_request_body(), build_reqwest_headers(), chatgpt_account_id_from_jwt(), codex_profile(), CodexContentBlock, CodexErrorBody (+24 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (51): AnthropicSseAdapter, AnthropicStreamAssembler, anthropic::infer, anthropic::infer_stream, chatgpt_account_id_from_jwt, codex::infer, codex::infer_stream, CompletionsSseAdapter (+43 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (20): ApiFamily, AuthStrategy, CredentialAuthConfig, CredentialKind, EndpointPurpose, ProviderCredential, ProviderProfile, ProviderQuirks (+12 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (31): AnthropicBlockState, AnthropicContentBlock, AnthropicDelta, AnthropicError, AnthropicErrorBody, AnthropicRequest, AnthropicResponse, AnthropicSseAdapter (+23 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (23): test_registry_get_anthropic_provider(), test_registry_get_completions_provider(), test_registry_unknown_provider_error(), ProviderRegistry, test_blank_api_key_fails_during_registry_construction(), test_blank_oauth_token_fails_during_registry_construction(), test_builtins_registered(), test_case_insensitive_lookup() (+15 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (26): anthropic_profile(), anthropic_response(), build_test_client(), chat_completion_response(), chat_completion_tool_response(), completions_profile(), test_anthropic_basic_response(), test_anthropic_error_handling() (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (18): build_client(), build_input_items(), build_tool_choice(), build_tools(), infer(), infer_rejects_empty_model(), infer_stream(), OpenAiConfig (+10 more)

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (16): anthropic_style_event_with_name(), chunk_split_mid_line(), chunk_split_mid_newline(), comment_lines_ignored(), crlf_line_endings(), data_without_space_after_colon(), done_marker_preserved_as_data(), empty_keepalive_ignored() (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (4): test_generation_config_default(), test_message_user(), test_transcript_add_message(), test_transcript_with_messages()

### Community 10 - "Community 10"
Cohesion: 0.24
Nodes (10): collect(), drops_complete_after_err_result(), drops_complete_after_error_event(), drops_events_after_complete(), is_terminal(), passes_through_non_terminal_events(), process_sse_stream(), SseStreamAdapter (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (14): Best-Effort Metadata Finalization, Chat Completions Stream Assembly Spec (archived), Chat Completions Stream Assembly Spec (canonical), Completed-Only ProviderEvent::ToolCall Policy, Provider Credentials Documentation, Chat Completions Stream Assembly Fix Design, Chat Completions Stream Assembly Fix Proposal, Chat Completions Stream Assembly Fix Tasks (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.27
Nodes (12): build(), _failure_summary(), _lockfile_path(), _print_summary(), Run the test suite with a terse summary., Run security checks with a terse summary., Run the build and lint checks with a terse summary., _run_group() (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (8): ChoiceItem, ChoiceRequest, ChoiceSelectionMode, ProviderEvent, RuntimeRecord, ToolCall, ToolDefinition, ToolPolicy

### Community 17 - "Community 17"
Cohesion: 0.43
Nodes (8): Codex as Separate Provider from OpenAI, Credential Kind vs Wire Auth Strategy Separation, Unverified JWT Routing Metadata Extraction, Provider Credential OAuth Design, Provider Credential OAuth Proposal, Provider Credential OAuth Spec, Provider Credential OAuth Tasks, OAuth Refresh Lifecycle Ownership Boundary

### Community 18 - "Community 18"
Cohesion: 0.53
Nodes (6): ApiFamily-Based Fragment Resolution, Compile-Time Markdown Fragment Inclusion, Provider System Prompt Fragments Design, Provider System Prompt Fragments Proposal, Provider System Prompt Fragments Spec, Provider System Prompt Fragments Tasks

### Community 21 - "Community 21"
Cohesion: 0.4
Nodes (3): Message, test_message_assistant(), test_message_tool()

## Knowledge Gaps
- **102 isolated node(s):** `graphify`, `Run the build and lint checks with a terse summary.`, `Run the test suite with a terse summary.`, `Run security checks with a terse summary.`, `ChoiceSelectionMode` (+97 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **38 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `process_sse_stream()` connect `Community 10` to `Community 0`, `Community 5`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `Transcript` connect `Community 20` to `Community 13`, `Community 6`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `build_test_client()` (e.g. with `.auth()` and `build_http_client()`) actually correct?**
  _`build_test_client()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Run the build and lint checks with a terse summary.`, `Run the test suite with a terse summary.` to the rest of the system?**
  _102 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._