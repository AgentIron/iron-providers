## Why

AgentIron is moving provider protocol metadata upstream into `iron-providers`, but it still has to define narrow local provider profiles for direct `openai` access and customized `local` base URLs. This duplicates upstream-owned profile metadata and makes profile selection depend on consumer-side workarounds.

## What Changes

- Add upstream support for a direct built-in `openai` provider profile for public OpenAI Responses API-key access, distinct from the OAuth-backed `codex` profile.
- Add a per-session way to override the effective base URL of the upstream `local` provider without redefining its API family, auth strategies, credential support, quirks, headers, or endpoint purpose.
- Make registry resolution for shared `models.dev` identities deterministic and explicit enough for consumers to map external catalog provider IDs without relying on `HashMap` iteration order.
- Preserve existing provider protocol behavior for current built-in slugs, including `codex` OAuth behavior and `local` default endpoint behavior.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `provider-architecture-refactor`: Registry/profile/runtime requirements change to include direct OpenAI profile registration, deterministic shared `models.dev` identity resolution, and runtime-effective provider connection state.
- `local-model-provider`: Local provider requirements change to support per-session base URL overrides while preserving the upstream built-in profile metadata.

## Impact

- Affected public APIs: `ProviderRegistry`, `RuntimeConfig` or equivalent provider-construction options, and provider-profile lookup helpers.
- Affected built-ins: `openai`, `codex`, and `local`.
- Affected internals: provider connection construction, effective base URL selection, registry lookup by `models.dev` identity, and tests for built-in profile metadata.
- No new external dependencies are expected.
