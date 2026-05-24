## Context

`iron-providers` v0.2.2 has a clean architecture after the provider-architecture-refactor: profiles describe provider identity, `ProviderConnection` resolves them into working connections, and API adapters handle protocol-specific logic. However, all existing built-in providers are hosted services requiring API keys.

Two gaps exist:

1. **Ollama Cloud** (`api.ollama.cloud`) is a standard OpenAI-compatible Chat Completions endpoint, identical in API shape to OpenRouter or Requesty. It needs no new adapter — just a profile registration.

2. **Local model serving** is fundamentally different: the endpoint address is dynamic (user-chosen port, different host machines), and authentication is often absent. There are multiple local backends (Ollama, llama.cpp server, vLLM, TGI) but the common supported path is the OpenAI-compatible Completions protocol.

The existing `ProviderProfile` model already supports URL-pattern resolution, custom base URLs, multiple credential kinds per profile, and dynamic construction — all of which are needed here. The missing piece is explicit `NoAuth` credential support for unauthenticated local use.

## Goals / Non-Goals

**Goals:**

- Add explicit `NoAuth` credential support, with `RuntimeConfig::validate()`, `auth.rs`, and `ProviderConnection` construction all handling the no-credential case.
- Register `ollama-cloud` as a built-in `ApiFamily::Completions` provider with BearerToken auth.
- Register `local` as a built-in `ApiFamily::Completions` provider with both `NoAuth` and API-key credential support and URL-pattern resolution for local addresses.
- Register `local` via `register_by_url_pattern` so that `resolve_by_url` auto-detects local endpoints.
- Support local Ollama through Ollama's OpenAI-compatible `/v1/chat/completions` endpoint.

**Non-Goals:**

- Adding separate provider profiles per local backend (no `ollama-local`, `llamacpp-local`, `vllm-local` separate slugs).
- Changing auth behavior for existing providers.
- Rearchitecting adapter dispatch, connection construction, or the `Provider` trait.
- Adding credential refresh, discovery, or health-checking for local endpoints.
- Supporting Ollama's native `/api/chat` endpoint.
- Wiring `ProviderQuirks.param_renames`, endpoint-path overrides, or native Ollama request/response parsing.

## Decisions

### 1. Explicit NoAuth as a runtime credential

Add a third credential kind and an explicit runtime credential for unauthenticated providers:

```rust
pub enum CredentialKind {
    ApiKey,
    OAuthBearer,
    NoAuth,
}

pub enum ProviderCredential {
    ApiKey(String),
    OAuthBearer {
        access_token: String,
        expires_at: Option<std::time::SystemTime>,
        id_token: Option<String>,
    },
    NoAuth,
}
```

`RuntimeConfig` adds a convenience constructor `RuntimeConfig::none()` that creates `ProviderCredential::NoAuth`. `ProviderCredential::kind()` returns `CredentialKind::NoAuth`, and `ProviderCredential::secret()` returns an empty string only for this explicit no-auth variant.

**Why this approach over alternatives:**

- **Empty/placeholder API key**: Would require special-casing the local provider in validation, making the profile non-self-describing and creating ambiguity between "I intentionally provided no auth" and "I forgot to set an API key."
- **Optional credential field on RuntimeConfig**: Would require Option-wrapping the credential, cascading through every provider's construction path and existing callers.
- **Explicit NoAuth credential**: Keeps the model honest end-to-end. A profile declares `NoAuth` support explicitly, runtime config carries `NoAuth` explicitly, and the auth adapter returns an empty `HeaderMap` for the explicit no-auth case.

`ProviderProfile::new()` still defaults to `ApiKey` + `BearerToken`, so no existing profile is affected. Providers opt into `NoAuth` by calling `.with_credential_auth(CredentialKind::NoAuth, AuthStrategy::NoAuth)`.

### 2. `RuntimeConfig::none()` as the no-credential constructor

```rust
impl RuntimeConfig {
    pub fn none() -> Self {
        Self {
            credential: ProviderCredential::NoAuth,
            connect_timeout: None,
            read_timeout: None,
        }
    }
}
```

`RuntimeConfig::validate()` continues to reject blank API keys and blank OAuth tokens. The no-auth case succeeds only when connection construction confirms the selected provider profile supports `CredentialKind::NoAuth`.

**Public API impact:** Adding `ProviderCredential::NoAuth` means downstream crates with exhaustive matches on `ProviderCredential` must add a `NoAuth` arm. This is accepted because explicit modeling avoids silent credential drops and avoids treating empty API keys as a no-auth signal.

### 3. `local` accepts NoAuth and optional API-key credentials

The `local` profile supports both unauthenticated and secured local endpoints:

```rust
let mut local = ProviderProfile::new(
    "local",
    ApiFamily::Completions,
    "http://localhost:11434/v1",
)
.with_credential_auth(CredentialKind::NoAuth, AuthStrategy::NoAuth);
// The default ApiKey + BearerToken support from ProviderProfile::new() remains.
```

When callers use `RuntimeConfig::none()`, no auth headers are emitted. When callers use `RuntimeConfig::new("token")`, the token is sent as `Authorization: Bearer token`. API keys are never silently ignored.

**Why support API keys for `local`:** Some local or self-hosted OpenAI-compatible gateways are protected by a development token, reverse proxy, or local service auth. Supporting both credential kinds preserves the common no-auth path while keeping secured local deployments possible.

### 4. `local` registered via `register_by_url_pattern` with local address prefixes

The `local` profile registers three URL prefixes in `register_builtins()`:

```rust
registry.register_by_url_pattern(
    "http://localhost",
    ProviderProfile::new("local", ApiFamily::Completions, "http://localhost:11434/v1")
        .with_credential_auth(CredentialKind::NoAuth, AuthStrategy::NoAuth),
);
registry.register_by_url_pattern(
    "http://127.0.0.1",
    /* same profile with different base URL */);
registry.register_by_url_pattern(
    "http://0.0.0.0",
    /* same profile with different base URL */);
```

The longest-prefix match in `resolve_by_url` ensures the correct profile is selected when a URL points to a local address.

**Why register multiple prefixes rather than one wildcard:** URL-pattern matching is prefix-based, not regex. `http://localhost` covers `http://localhost:11434`, `http://localhost:8080`, etc. `http://127.0.0.1` and `http://0.0.0.0` cover the other common loopback addresses. The URLPattern variant with prefix matching is already proven in the codebase (used for coding vs general endpoints).

The base URL in the profile is set to `http://localhost:11434/v1` as a sensible default for Ollama's OpenAI-compatible endpoint, but the profile can be overridden by downstream configuration. Other local backends use their own OpenAI-compatible base URL, such as `http://localhost:1234/v1` for LM Studio or `http://localhost:8000/v1` for vLLM.

### 5. `ollama-cloud` registered as a standard Completions built-in

A simple built-in registration:

```rust
registry.register(
    ProviderProfile::new("ollama-cloud", ApiFamily::Completions, "https://api.ollama.cloud")
);
```

No quirks, no overrides — it uses the default BearerToken auth and the standard Completions adapter. Identical in shape to the `requesty` built-in.

### Backend compatibility matrix

The `local` provider (Completions + NoAuth or optional API-key auth) covers every major local LLM serving system:

| System | Default port | OpenAI-compatible endpoint | Compatible? |
|--------|-------------|---------------------------|-------------|
| LM Studio | `:1234` | `/v1/chat/completions` | ✅ Yes |
| Unsloth | `:8080` | `/v1/chat/completions` (via llama-server) | ✅ Yes |
| Jan | `:1337` | `/v1/chat/completions` | ✅ Yes |
| MLX-LM | `:8080` | `/v1/chat/completions` | ✅ Yes |
| llama.cpp server | `:8080` | `/v1/chat/completions` | ✅ Yes |
| vLLM | `:8000` | `/v1/chat/completions` | ✅ Yes |
| Ollama | `:11434` | `/v1/chat/completions` | ✅ Yes |

**Ollama note:** Current Ollama exposes `/v1/chat/completions`, which works directly with the generic `local` provider. The native `/api/chat` endpoint is intentionally out of scope because it has a different endpoint path and protocol shape with no clear benefit for this crate's Completions-based integration path.

No backend-specific profiles are needed. The generic approach handles all listed systems with just a base URL change.

## Risks / Trade-offs

- [Risk] `CredentialKind::NoAuth` and `ProviderCredential::NoAuth` add public enum variants. → Mitigation: Existing providers are unaffected, and downstream exhaustive matches should add an explicit `NoAuth` arm. This is preferable to implicit empty-string credential behavior.
- [Risk] `local` accepting API keys could make users think auth is required. → Mitigation: `RuntimeConfig::none()` is the documented default for unauthenticated local endpoints; API-key support exists only for secured local gateways and sends the key as Bearer auth when supplied.
- [Risk] URL-pattern matching for `localhost` could match an unrelated provider whose URL happens to start with `http://localhost`. → Mitigation: Unlikely in practice. If someone registers a custom provider on `http://localhost:5000/my-other-service`, the longest-prefix match still works correctly. The built-in patterns are specific enough (`http://localhost` prefix) that collisions would require intentional misconfiguration.
- [Risk] A user running a local server on a non-loopback interface (e.g., `http://192.168.1.100:8080`) won't be auto-resolved. → Mitigation: Accepted trade-off. The three common loopback addresses cover the 99% case. Custom network addresses require explicit provider configuration, which is already supported via `ProviderRegistry::register()` or custom runtime config.

## Resolved Questions

- `AuthStrategy::NoAuth` SHALL be added as the explicit strategy for `CredentialKind::NoAuth` so provider profiles remain self-describing.
