## Migration Notes

This release includes intentional breaking API changes to remove ambiguous
streaming semantics and separate runtime-owned state from model-visible
transcript state.

### 1. `InferenceRequest.stream` removed

Before:

```rust
let request = InferenceRequest::new("model", transcript).with_stream(true);
let stream = provider.infer_stream(request).await?;
```

After:

```rust
let request = InferenceRequest::new("model", transcript);
let stream = provider.infer_stream(request).await?;
```

Streaming is selected by calling `infer_stream`, not by setting a field on the
request.

### 2. `InferenceRequest` now uses `InferenceContext`

Before:

```rust
let request = InferenceRequest {
    model: "model".into(),
    instructions: None,
    transcript,
    tools: vec![],
    tool_policy: ToolPolicy::Auto,
    generation: GenerationConfig::default(),
};
```

After:

```rust
let request = InferenceRequest::new("model", transcript);
// or manipulate request.context.runtime_records explicitly
```

`InferenceContext` separates:

- `context.transcript`: model-visible conversation
- `context.runtime_records`: runtime-only structured state

### 3. `Message::SystemStructured` removed

Before:

```rust
let message = Message::system_structured("session_state", payload);
```

After:

```rust
request
    .context
    .add_record(RuntimeRecord::new("session_state", payload));
```

This prevents runtime-owned state from being silently projected into assistant
message text.

### 4. `ProviderEvent::Complete` is success-only

Before, some adapters could emit `Error` and later still emit `Complete`.

Now:

- `Complete` means successful termination
- unrecoverable stream errors end without a later `Complete`

### 5. Profile semantics are stricter

Provider auth strategy and default headers are now validated and applied more
consistently, including for `OpenAiResponses`.

Misconfigured headers or auth values now fail fast instead of falling back to a
default client.

### 6. Dependency note

`async-openai` was upgraded to `0.35`.

The direct `reqwest` upgrade to `0.13` remains deferred because
`async-openai 0.35` still depends on `reqwest 0.12`.
