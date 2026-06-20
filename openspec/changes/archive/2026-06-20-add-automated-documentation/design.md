## Context

`iron-providers` exposes a public Rust API that downstream crates can consume before all related AgentIron crates are published to crates.io. The crate already uses rustdoc link lints in `src/lib.rs`, and `cargo doc --no-deps --all-features` currently succeeds with warnings denied. However, strict missing-docs validation fails today because public modules, types, variants, fields, and methods are not all documented.

The README contains public examples and workflow descriptions, but the Rust snippets are not currently compiled as doctests. The repository also describes GitHub Actions workflows in README, while this checkout does not currently contain `.github/workflows` files.

## Goals / Non-Goals

**Goals:**

- Publish rustdoc-generated API documentation for the current `main` branch through GitHub Pages.
- Fail documentation validation when public-facing API rustdoc is missing.
- Fail documentation validation when rustdoc emits warnings, including broken intra-doc links.
- Compile public examples in rustdoc and README snippets as documentation tests.
- Document all public-facing API items inline in the source code.
- Provide examples for primary usage paths, including registry lookup, runtime configuration, custom profiles, request construction, streaming events, tool definitions, and error classification.
- Keep the documentation workflow compatible with future docs.rs publication.

**Non-Goals:**

- Do not change runtime provider behavior.
- Do not change public API signatures solely to make documentation easier.
- Do not publish private implementation modules or private items to GitHub Pages.
- Do not require a separate example for every public enum variant, struct field, or trivial accessor when a usage-path example covers the API.
- Do not add a custom static site generator unless rustdoc alone proves insufficient.

## Decisions

### Use rustdoc as the published documentation artifact

GitHub Pages will publish the generated `target/doc` output from `cargo doc --no-deps --all-features`. This keeps the published site aligned with Rust conventions and avoids maintaining a separate documentation stack.

Alternative considered: add a custom landing page or static site generator. This was rejected for the initial change because the immediate need is API reference availability for unreleased crates, and rustdoc already provides search, item pages, and examples.

### Enforce public documentation with rustdoc warnings in CI

Documentation validation will run with rustdoc warnings denied and missing public docs denied, conceptually equivalent to:

```bash
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --manifest-path Cargo.toml --no-deps --all-features
```

The crate may also add crate-level lint attributes for missing docs if that produces clearer local failures. The enforcement point must remain local and CI-visible.

Alternative considered: leave missing-docs as a warning. This was rejected because the goal is to ensure all public-facing API is documented as part of this work.

### Add a dedicated local documentation task

The Invoke workflow should gain a dedicated docs task rather than folding documentation into the existing build task. That keeps build, tests, security, and documentation failures easy to interpret.

Expected checks:

- `cargo doc --manifest-path Cargo.toml --no-deps --all-features` with strict rustdoc flags.
- `cargo test --manifest-path Cargo.toml --doc` for doctests.

Alternative considered: add docs checks directly to `inv build`. This was rejected for the first pass because strict docs coverage may be slower and semantically distinct from compilation/linting.

### Compile README examples through doctests

README examples should be included in doctests using Rust's `#[doc = include_str!("../README.md")]` plus `#[cfg(doctest)]` pattern. README snippets that perform networked provider calls should compile but not run, using `no_run` and hidden scaffolding where needed.

Alternative considered: mark examples as ignored. This was rejected except for snippets that cannot be made to compile safely, because ignored examples do not protect against API drift.

### Publish only from main and validate pull requests without publishing

Pull requests should run documentation validation but not deploy. Pushes to `main` should build rustdoc and publish it through GitHub Pages.

Alternative considered: publish preview docs for every pull request. This is useful but unnecessary for the initial capability and adds permission/comment lifecycle complexity.

## Risks / Trade-offs

- Strict `missing-docs` may make future API additions more costly -> Mitigate by documenting public items at the time they are introduced and keeping examples usage-path oriented.
- Doctested README snippets may require hidden scaffolding that makes the markdown harder to read -> Mitigate by keeping hidden lines minimal and using `no_run` for examples that would call external provider APIs.
- GitHub Pages deployment requires repository Pages settings and workflow permissions -> Mitigate by using GitHub's official Pages artifact/deploy actions and documenting required repository settings.
- Rustdoc-only publishing lacks broader narrative guides -> Mitigate by starting with rustdoc and README doctests; add a landing page later only if users need richer guide structure.
- Public API docs may expose outdated README provider tables or examples -> Mitigate by including README examples in doctests and considering provider-table generation as a future enhancement if drift continues.

## Migration Plan

1. Add the local docs validation task.
2. Add strict rustdoc coverage and fill all current public documentation gaps.
3. Add doctest examples for primary public usage paths.
4. Include README examples in doctests and adjust snippets to compile.
5. Add GitHub Actions documentation validation for pull requests.
6. Add GitHub Pages deployment for pushes to `main`.
7. Update README development workflow notes to mention the docs task and GitHub Pages publication.

Rollback is straightforward: remove the Pages workflow or disable Pages deployment while keeping local documentation validation if strict docs remain useful.

## Open Questions

- What public GitHub Pages URL should be advertised in README once Pages is enabled?
- Should the repository add a small redirect or landing page later, or is rustdoc's generated index sufficient for the first release?
