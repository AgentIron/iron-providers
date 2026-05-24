## Why

`iron-providers` currently supports only hosted cloud providers (Anthropic, OpenAI/Codex, OpenRouter, Requesty, Kimi, Z.AI, MiniMax). There is no support for:

1. **Ollama Cloud** — a hosted OpenAI-compatible chat completions endpoint that is used as a primary provider in Hermes Agent and other tools.
2. **Local model serving** — the ability to run inference against locally-hosted models via any OpenAI-compatible backend. Popular systems include **Ollama**, **llama.cpp server**, **vLLM**, **LM Studio**, **Unsloth**, **Jan**, and **MLX-LM** — all of which expose OpenAI-compatible `/v1/chat/completions` endpoints.

Adding these fills a growing need for self-hosted and cloud-based open-weight model inference without requiring separate provider profiles for every local backend.

## What Changes

- Add a built-in `ollama-cloud` provider profile registered in `registry.rs` as an `ApiFamily::Completions` provider targeting `https://api.ollama.cloud`.
- Add a built-in `local` provider profile that serves as a generic OpenAI-compatible local model endpoint, registered with URL-pattern matching to auto-detect local endpoints (`http://localhost:*`, `http://127.0.0.1:*`, `http://0.0.0.0:*`).
- Add explicit `NoAuth` credential support for local providers that don't require authentication.
- Allow the `local` provider to accept either `NoAuth` or API-key credentials; API keys are sent as Bearer tokens for secured local gateways and are never silently dropped.
- Adjust `RuntimeConfig::validate()` and connection construction to accept `NoAuth` only for profiles that explicitly support it.
- Adjust `auth.rs` to produce no auth headers for `NoAuth` providers.
- Adjust tests accordingly.

## Capabilities

### New Capabilities
- `ollama-cloud-integration`: Built-in provider profile for Ollama Cloud's Completions endpoint, registered with BearerToken auth.
- `local-model-provider`: A generic OpenAI-compatible local model provider supporting `localhost`, `127.0.0.1`, and `0.0.0.0` endpoints with either no authentication or optional Bearer API-key authentication.

### Modified Capabilities
- `provider-architecture-refactor`: The credential model and `RuntimeConfig` validation gain explicit `NoAuth` support for unauthenticated local providers. The auth boundary in `auth.rs` handles the no-credential case. These are extensions to the architecture established by the refactor, not redefinitions.

## Impact

- **Affected code**: `src/profile.rs` (credential model, RuntimeConfig::validate), `src/auth.rs` (no-credential handling), `src/registry.rs` (new built-in profiles and URL patterns).
- **New code**: None — both providers use the existing `ApiFamily::Completions` adapter.
- **API impact**: Additive public credential support for `NoAuth`. Existing providers are unaffected.
- **Downstream impact**: `iron-core` and `hermes-agent` can configure local/Ollama Cloud providers via the same `ProviderRegistry` interface with explicit no-auth config for unauthenticated local endpoints or API-key config for secured local endpoints.

## Out of Scope

- Ollama's native `/api/chat` endpoint. Current Ollama releases support the OpenAI-compatible `/v1/chat/completions` API, which is the supported local Ollama integration path.
- Wiring or extending `ProviderQuirks.param_renames` for `max_tokens`/`num_predict` translation.
- Endpoint-path overrides or native Ollama request/response parsing.
