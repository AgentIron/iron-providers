## ADDED Requirements

### Requirement: Strict public API documentation validation

The repository SHALL provide an automated documentation validation path that fails when public-facing API items are missing rustdoc or when rustdoc emits warnings.

#### Scenario: Public rustdoc is complete

- **WHEN** documentation validation runs for the crate
- **THEN** rustdoc generation succeeds with warnings denied and missing public documentation denied

#### Scenario: Public rustdoc is missing

- **WHEN** a public-facing module, type, field, variant, constructor, builder method, trait, or trait method lacks required rustdoc
- **THEN** documentation validation fails before the change can be accepted

### Requirement: Public examples compile as documentation tests

The repository SHALL include compiling documentation examples for primary public usage paths and SHALL run those examples as part of documentation validation.

#### Scenario: Primary usage examples remain current

- **WHEN** documentation tests run
- **THEN** examples for registry lookup, runtime configuration, custom provider profiles, request construction, streaming events, tool definitions, and error classification compile against the current public API

#### Scenario: Example drifts from the API

- **WHEN** a documented example no longer compiles against the current public API
- **THEN** documentation validation fails

### Requirement: README examples are protected from API drift

README Rust examples SHALL be included in documentation testing so user-facing examples compile against the current crate API.

#### Scenario: README doctests run

- **WHEN** documentation tests run
- **THEN** README Rust examples are compiled as doctests, using hidden scaffolding or `no_run` where examples would otherwise call external provider services

#### Scenario: README example becomes stale

- **WHEN** a README Rust example references removed or changed public API
- **THEN** documentation validation fails

### Requirement: GitHub Pages publishes main-branch API documentation

The repository SHALL publish rustdoc-generated public API documentation for the `main` branch to GitHub Pages.

#### Scenario: Main branch documentation publish succeeds

- **WHEN** a change is pushed to `main`
- **THEN** the repository builds rustdoc output for public API documentation and deploys it to GitHub Pages

#### Scenario: Pull request documentation validation runs

- **WHEN** a pull request targets `main`
- **THEN** documentation validation runs without deploying documentation

### Requirement: Documentation automation is available locally

The repository SHALL expose a local documentation validation command through the standard development workflow.

#### Scenario: Developer runs docs validation locally

- **WHEN** a developer runs the repository documentation task
- **THEN** strict rustdoc generation and documentation tests run with a clear pass or failure result

#### Scenario: Documentation task fails

- **WHEN** rustdoc generation or doctests fail
- **THEN** the local documentation task reports the failing step and exits non-zero
