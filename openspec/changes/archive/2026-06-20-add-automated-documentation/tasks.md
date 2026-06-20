## 1. Local Documentation Workflow

- [x] 1.1 Add a dedicated local documentation task to the Invoke workflow that runs strict rustdoc generation and documentation tests.
- [x] 1.2 Ensure the documentation task denies rustdoc warnings and missing public docs.
- [x] 1.3 Update README development instructions to include the local documentation task.

## 2. Public API Rustdoc Coverage

- [x] 2.1 Add module-level rustdoc for all public modules exported from `src/lib.rs`.
- [x] 2.2 Add rustdoc for public error types, variants, fields, constructors, and classification helpers.
- [x] 2.3 Add rustdoc for public request, transcript, message, tool, usage, and event model types.
- [x] 2.4 Add rustdoc for public profile, credential, runtime configuration, and provider quirk types.
- [x] 2.5 Add rustdoc for public provider connection, registry, trait, and prelude APIs.

## 3. Documentation Examples

- [x] 3.1 Add compiling doctest examples for registry lookup and inference request setup.
- [x] 3.2 Add compiling doctest examples for runtime configuration with API key, OAuth bearer, and no-auth credentials.
- [x] 3.3 Add compiling doctest examples for custom provider profile registration and URL/model resolution.
- [x] 3.4 Add compiling doctest examples for transcript/message construction, generation config, tool definitions, and tool policy.
- [x] 3.5 Add compiling doctest examples for streaming event handling and provider error classification.

## 4. README Doctest Coverage

- [x] 4.1 Include README Rust snippets in doctests using a `#[cfg(doctest)]` item.
- [x] 4.2 Convert README Rust snippets to compile as doctests, using hidden scaffolding and `no_run` for external provider calls.
- [x] 4.3 Update stale README examples and version references discovered while making examples compile.

## 5. GitHub Documentation Automation

- [x] 5.1 Add a pull request documentation validation workflow that runs the local documentation task without deploying.
- [x] 5.2 Add a `main` branch GitHub Pages workflow that builds rustdoc output and deploys `target/doc`.
- [x] 5.3 Document required GitHub Pages repository settings and the published documentation URL placeholder in README.

## 6. Validation

- [x] 6.1 Run `openspec validate add-automated-documentation --strict`.
- [x] 6.2 Run `inv docs` or the equivalent strict docs commands.
- [x] 6.3 Run `inv build`.
- [x] 6.4 Run `inv test`.
- [x] 6.5 Run `inv security`.
