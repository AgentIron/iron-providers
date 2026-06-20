## Context

`iron-providers` owns built-in provider profile metadata through `ProviderRegistry::register_builtins()`. AgentIron wants to consume those upstream profiles directly, but two gaps currently force app-side profile definitions: direct API-key access to public OpenAI Responses and custom local OpenAI-compatible endpoints.

The current model has a clean separation between static profile metadata and runtime construction state, but `RuntimeConfig` only includes credentials and timeouts. Because `ProviderConnection` reads `profile.base_url` directly, a caller that wants a one-session local endpoint override must clone or redefine the whole `ProviderProfile`, including fields that should stay upstream-owned.

Registry lookup by `models.dev` identity also assumes a one-to-one mapping. Adding a direct `openai` profile while `codex` keeps `models_dev_id = "openai"` turns that into a one-to-many relationship, and the current `HashMap::values().find(...)` lookup cannot provide deterministic selection.

## Goals / Non-Goals

**Goals:**

- Register a direct `openai` built-in profile for public OpenAI Responses API-key access.
- Let callers override the effective base URL for a provider connection without redefining the upstream profile.
- Preserve `local` as the generic upstream-owned local OpenAI-compatible provider profile.
- Make shared `models.dev` identity resolution deterministic and explicit for consumers.
- Preserve existing `codex`, `local`, and other built-in behavior unless called through the new capabilities.

**Non-Goals:**

- Do not add a new API family or provider-branded adapter.
- Do not make every profile field runtime-overridable.
- Do not move OAuth refresh, model catalog fetching, or user preference ownership into this crate.
- Do not remove existing slug-based provider lookup.

## Decisions

### Add `openai` as a built-in Responses profile

Register `openai` with `ApiFamily::Responses`, base URL `https://api.openai.com/v1`, API-key Bearer auth, and `models_dev_id = "openai"` unless the slug fallback is sufficient for catalog mapping.

Rationale: AgentIron still exposes `openai` as a first-class provider/default provider. Keeping the public OpenAI protocol metadata upstream avoids app-side profile duplication and aligns with the existing Responses adapter contract.

Alternative considered: Treat `codex` as the only OpenAI-related built-in. That keeps the registry smaller but forces API-key OpenAI consumers to keep defining local protocol metadata.

### Represent base URL override as runtime-effective connection state

Add a small provider-construction option, likely on `RuntimeConfig`, for an optional base URL override. During `ProviderConnection` construction, resolve an effective base URL from `runtime.base_url_override.unwrap_or(profile.base_url)`. API adapters should use the effective base URL rather than directly reading the static profile field for request targets.

Rationale: The override is session-specific, not provider identity. Keeping it out of `ProviderProfile` prevents consumers from redefining API family, auth, credential support, quirks, endpoint purpose, and headers just to change a host or port.

Alternative considered: Add a `ProviderProfile::with_base_url` clone helper. This is smaller, but it still asks consumers to create derived profile definitions and does not clearly distinguish static upstream metadata from runtime selection.

### Keep overrides generic, but document the local use case

The runtime base URL override should be available through the provider construction path rather than a `local`-only method. The primary required behavior is for `local`, but the mechanism does not need to encode provider-specific policy.

Rationale: A generic override keeps the API orthogonal and avoids adding a special registry path that only works for one slug. Tests can still focus on `local` because that is the motivating built-in provider.

Alternative considered: Add `ProviderRegistry::get_local_with_base_url(...)`. This is narrow and safe, but it does not scale if other OpenAI-compatible profiles need session endpoint overrides later.

### Make `models.dev` identity lookup one-to-many

Introduce an API that returns all profiles matching a `models.dev` provider identity in deterministic order, such as slug-sorted order. Existing single-profile lookup can either delegate to the deterministic list and return the first match or be documented as a legacy convenience, but consumers that care about correctness should use the multi-result API.

Rationale: `openai` and `codex` can both legitimately map to the same external catalog provider ID while representing different endpoint/auth/purpose profiles. Returning a list makes that multiplicity visible instead of hiding it behind map iteration order.

Alternative considered: Add a purpose-filtered resolver only. Filtering by `EndpointPurpose` is useful, but it does not fully solve future cases where two profiles share catalog identity and purpose but differ by credential or endpoint behavior.

## Risks / Trade-offs

- Runtime base URL overrides could be used on hosted providers unexpectedly -> validate URL shape and keep the feature explicit through a named builder/API rather than hidden environment behavior.
- Adding `openai` changes `models.dev` lookup cardinality for `openai` -> provide deterministic multi-result lookup and tests covering both `openai` and `codex`.
- Storing effective base URL separately from `ProviderProfile` can create two places to look for endpoint state -> expose or document the effective value through `ProviderConnection` tests and keep static profile metadata unchanged.
- Keeping the old single-result `resolve_by_models_dev_id` may preserve ambiguity -> prefer a new explicit API for consumers and use deterministic behavior if the old API remains.

## Migration Plan

- Add the upstream built-in `openai` profile and tests without removing existing `codex` behavior.
- Add runtime/provider-construction base URL override support and update request-building internals to use the effective base URL.
- Add deterministic multi-profile `models.dev` identity resolution and keep existing slug lookup unchanged.
- AgentIron can then delete its temporary local `openai` and `local` profile definitions and call the upstream registry APIs instead.

Rollback is straightforward because the change adds capabilities. Consumers can continue to construct explicit `ProviderProfile` values if needed.

## Open Questions

- Should the runtime override be limited to absolute HTTP(S) URLs, or should validation remain minimal and defer to request construction?
- Should the old `resolve_by_models_dev_id` remain public as a deterministic convenience, or should consumers be steered entirely to a new plural API?
- Should direct `openai` explicitly set `models_dev_id = "openai"`, or rely on the slug fallback for the same effective identity?
