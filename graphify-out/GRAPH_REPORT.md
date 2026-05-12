# Graph Report - .  (2026-05-11)

## Corpus Check
- Corpus is ~30,227 words - fits in a single context window. You may not need a graph.

## Summary
- 512 nodes · 911 edges · 49 communities (16 shown, 33 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 156 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Anthropic Types|Anthropic Types]]
- [[_COMMUNITY_Anthropic & Codex Streaming|Anthropic & Codex Streaming]]
- [[_COMMUNITY_Provider Credentials|Provider Credentials]]
- [[_COMMUNITY_Registry & Inference Tests|Registry & Inference Tests]]
- [[_COMMUNITY_Codex Provider|Codex Provider]]
- [[_COMMUNITY_Chat Completions Types|Chat Completions Types]]
- [[_COMMUNITY_OpenAI Provider|OpenAI Provider]]
- [[_COMMUNITY_Mock Provider & Profile Tests|Mock Provider & Profile Tests]]
- [[_COMMUNITY_Inference Request Builder|Inference Request Builder]]
- [[_COMMUNITY_SSE Parsing|SSE Parsing]]
- [[_COMMUNITY_Message & Model Tests|Message & Model Tests]]
- [[_COMMUNITY_Stream Utilities|Stream Utilities]]
- [[_COMMUNITY_Chat Completions Assembly Spec|Chat Completions Assembly Spec]]
- [[_COMMUNITY_Task Runner|Task Runner]]
- [[_COMMUNITY_Provider Trait & OpenAI|Provider Trait & OpenAI]]
- [[_COMMUNITY_Streaming Event Tests|Streaming Event Tests]]
- [[_COMMUNITY_Responses API Tests|Responses API Tests]]
- [[_COMMUNITY_OAuth Credential Design|OAuth Credential Design]]
- [[_COMMUNITY_System Prompt Fragments|System Prompt Fragments]]
- [[_COMMUNITY_ProviderResult|ProviderResult]]
- [[_COMMUNITY_Message|Message]]
- [[_COMMUNITY_SseEvent|SseEvent]]
- [[_COMMUNITY_AnthropicRequest|AnthropicRequest]]
- [[_COMMUNITY_OpenAiConfigSource|OpenAiConfigSource]]
- [[_COMMUNITY_RuntimeConfigSource|RuntimeConfigSource]]
- [[_COMMUNITY_Streaming Test Suite|Streaming Test Suite]]
- [[_COMMUNITY_Responses API Tests|Responses API Tests]]
- [[_COMMUNITY_ProviderEventOutput|ProviderEvent::Output]]
- [[_COMMUNITY_ProviderEventToolCall|ProviderEvent::ToolCall]]
- [[_COMMUNITY_ProviderEventComplete|ProviderEvent::Complete]]
- [[_COMMUNITY_ProviderEventError|ProviderEvent::Error]]
- [[_COMMUNITY_ProviderEventStatus|ProviderEvent::Status]]
- [[_COMMUNITY_ToolCall JSON Test|ToolCall JSON Test]]
- [[_COMMUNITY_Streaming Setup Test|Streaming Setup Test]]
- [[_COMMUNITY_OpenAiConfig Test|OpenAiConfig Test]]
- [[_COMMUNITY_OpenAiConfig Builder|OpenAiConfig Builder]]
- [[_COMMUNITY_Instructions Test|Instructions Test]]
- [[_COMMUNITY_Tools Test|Tools Test]]
- [[_COMMUNITY_GenerationConfig Test|GenerationConfig Test]]
- [[_COMMUNITY_Transcript Test|Transcript Test]]
- [[_COMMUNITY_ToolPolicy Tests|ToolPolicy Tests]]
- [[_COMMUNITY_Registry Docs|Registry Docs]]
- [[_COMMUNITY_Adapter Family Docs|Adapter Family Docs]]
- [[_COMMUNITY_Registry Usage Docs|Registry Usage Docs]]
- [[_COMMUNITY_InferenceRequest Docs|InferenceRequest Docs]]
- [[_COMMUNITY_ProviderProfile Docs|ProviderProfile Docs]]
- [[_COMMUNITY_Key Types Docs|Key Types Docs]]
- [[_COMMUNITY_GitHub Workflow Docs|GitHub Workflow Docs]]
- [[_COMMUNITY_OpenSpec Config|OpenSpec Config]]

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
  src/completions.rs → src/http_client.rs
- `build_test_client()` --calls--> `build_http_client()`  [INFERRED]
  src/mock_provider_tests.rs → src/http_client.rs

## Hyperedges (group relationships)
- **Four-provider adapter family** — anthropic_infer, anthropic_infer_stream, completions_infer, completions_infer_stream, codex_infer, codex_infer_stream, openai_infer, openai_infer_stream [INFERRED 0.90]
- **SSE streaming pipeline** — sse_sseparser, streamutil_ssestreamadapter, streamutil_process_sse_stream, streamutil_terminatingstream, anthropic_anthropicsseadapter, completions_completionssseadapter [EXTRACTED 1.00]
- **Profile-driven provider configuration** — profile_providerprofile, profile_runtimeconfig, profile_apifamily, profile_authstrategy, profile_providercredential, httpclient_httpclientparams [INFERRED 0.85]
- **Chat Completions Stream Assembly Design Decisions** — stream_assembler_state_model, semantic_completion_markers, completed_only_toolcall_events, incremental_text_independence, toolcall_index_order_emission, best_effort_finalization [EXTRACTED 1.00]
- **OAuth Credential Design Decisions** — credential_kind_separation, codex_separate_provider, jwt_routing_metadata, oauth_lifecycle_boundary [EXTRACTED 1.00]

## Communities (49 total, 33 thin omitted)

### Community 0 - "Anthropic Types"
Cohesion: 0.06
Nodes (52): AnthropicBlockState, AnthropicContentBlock, AnthropicDelta, AnthropicError, AnthropicErrorBody, AnthropicRequest, AnthropicResponse, AnthropicSseAdapter (+44 more)

### Community 1 - "Anthropic & Codex Streaming"
Cohesion: 0.06
Nodes (51): AnthropicSseAdapter, AnthropicStreamAssembler, anthropic::infer, anthropic::infer_stream, chatgpt_account_id_from_jwt, codex::infer, codex::infer_stream, CompletionsSseAdapter (+43 more)

### Community 2 - "Provider Credentials"
Cohesion: 0.06
Nodes (20): ApiFamily, AuthStrategy, CredentialAuthConfig, CredentialKind, EndpointPurpose, ProviderCredential, ProviderProfile, ProviderQuirks (+12 more)

### Community 3 - "Registry & Inference Tests"
Cohesion: 0.12
Nodes (24): test_registry_get_anthropic_provider(), test_registry_get_completions_provider(), test_registry_unknown_provider_error(), InferenceContext, ProviderRegistry, test_blank_api_key_fails_during_registry_construction(), test_blank_oauth_token_fails_during_registry_construction(), test_builtins_registered() (+16 more)

### Community 4 - "Codex Provider"
Cohesion: 0.09
Nodes (25): base64_decode_url_safe(), build_codex_headers(), build_codex_request_body(), chatgpt_account_id_from_jwt(), codex_profile(), CodexContentBlock, CodexErrorBody, CodexOutputItem (+17 more)

### Community 5 - "Chat Completions Types"
Cohesion: 0.08
Nodes (29): build_chat_messages(), build_chat_tools(), build_client(), ChatChoice, ChatCompletionResponse, ChatCompletionStreamChunk, ChatError, ChatErrorBody (+21 more)

### Community 6 - "OpenAI Provider"
Cohesion: 0.14
Nodes (18): build_client(), build_input_items(), build_tool_choice(), build_tools(), infer(), infer_rejects_empty_model(), infer_stream(), OpenAiConfig (+10 more)

### Community 7 - "Mock Provider & Profile Tests"
Cohesion: 0.24
Nodes (26): anthropic_profile(), anthropic_response(), build_test_client(), chat_completion_response(), chat_completion_tool_response(), completions_profile(), test_anthropic_basic_response(), test_anthropic_error_handling() (+18 more)

### Community 8 - "Inference Request Builder"
Cohesion: 0.08
Nodes (11): ChoiceItem, ChoiceRequest, ChoiceSelectionMode, GenerationConfig, InferenceRequest, ProviderEvent, RuntimeRecord, ToolCall (+3 more)

### Community 9 - "SSE Parsing"
Cohesion: 0.25
Nodes (16): anthropic_style_event_with_name(), chunk_split_mid_line(), chunk_split_mid_newline(), comment_lines_ignored(), crlf_line_endings(), data_without_space_after_colon(), done_marker_preserved_as_data(), empty_keepalive_ignored() (+8 more)

### Community 10 - "Message & Model Tests"
Cohesion: 0.12
Nodes (7): Message, test_generation_config_default(), test_message_assistant(), test_message_tool(), test_message_user(), test_transcript_add_message(), test_transcript_with_messages()

### Community 11 - "Stream Utilities"
Cohesion: 0.24
Nodes (10): collect(), drops_complete_after_err_result(), drops_complete_after_error_event(), drops_events_after_complete(), is_terminal(), passes_through_non_terminal_events(), process_sse_stream(), SseStreamAdapter (+2 more)

### Community 12 - "Chat Completions Assembly Spec"
Cohesion: 0.19
Nodes (14): Best-Effort Metadata Finalization, Chat Completions Stream Assembly Spec (archived), Chat Completions Stream Assembly Spec (canonical), Completed-Only ProviderEvent::ToolCall Policy, Provider Credentials Documentation, Chat Completions Stream Assembly Fix Design, Chat Completions Stream Assembly Fix Proposal, Chat Completions Stream Assembly Fix Tasks (+6 more)

### Community 13 - "Task Runner"
Cohesion: 0.27
Nodes (12): build(), _failure_summary(), _lockfile_path(), _print_summary(), Run the test suite with a terse summary., Run security checks with a terse summary., Run the build and lint checks with a terse summary., _run_group() (+4 more)

### Community 17 - "OAuth Credential Design"
Cohesion: 0.43
Nodes (8): Codex as Separate Provider from OpenAI, Credential Kind vs Wire Auth Strategy Separation, Unverified JWT Routing Metadata Extraction, Provider Credential OAuth Design, Provider Credential OAuth Proposal, Provider Credential OAuth Spec, Provider Credential OAuth Tasks, OAuth Refresh Lifecycle Ownership Boundary

### Community 18 - "System Prompt Fragments"
Cohesion: 0.53
Nodes (6): ApiFamily-Based Fragment Resolution, Compile-Time Markdown Fragment Inclusion, Provider System Prompt Fragments Design, Provider System Prompt Fragments Proposal, Provider System Prompt Fragments Spec, Provider System Prompt Fragments Tasks

## Knowledge Gaps
- **101 isolated node(s):** `Run the build and lint checks with a terse summary.`, `Run the test suite with a terse summary.`, `Run security checks with a terse summary.`, `ChoiceSelectionMode`, `ChoiceItem` (+96 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `process_sse_stream()` connect `Stream Utilities` to `Anthropic Types`, `Registry & Inference Tests`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `Transcript` connect `Inference Request Builder` to `Mock Provider & Profile Tests`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `build_test_client()` (e.g. with `.auth()` and `build_http_client()`) actually correct?**
  _`build_test_client()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run the build and lint checks with a terse summary.`, `Run the test suite with a terse summary.`, `Run security checks with a terse summary.` to the rest of the system?**
  _101 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Anthropic Types` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Anthropic & Codex Streaming` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Provider Credentials` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._