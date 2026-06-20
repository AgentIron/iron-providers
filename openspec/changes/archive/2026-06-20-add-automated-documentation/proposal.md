## Why

`iron-providers` needs published API documentation before all related crates are available on crates.io. GitHub Pages documentation for `main`, strict rustdoc validation, and tested examples will make the unreleased public API usable by downstream crates such as `iron-core` without waiting for a crates.io release.

## What Changes

- Add automated documentation validation that fails when public API rustdoc is missing or rustdoc warnings are emitted.
- Add documentation tests for crate examples and README examples so public-facing examples compile against the current API.
- Add GitHub Pages publishing for generated rustdoc from the `main` branch.
- Add inline rustdoc for all public-facing API items, including public modules, exported types, fields, variants, constructors, builder methods, traits, and trait methods.
- Add examples for the primary public usage paths rather than one example per field or enum variant.
- Keep docs.rs as a future release destination, but do not rely on crates.io publication for current API documentation.

## Capabilities

### New Capabilities

- `automated-documentation`: Defines the repository's requirements for public API documentation coverage, compiling examples, docs validation, and GitHub Pages publication.

### Modified Capabilities

None.

## Impact

- Affects `src/**/*.rs` public rustdoc comments and doctest examples.
- Affects `README.md` examples so they can be included in doctests.
- Affects local validation tasks, likely by adding an `inv docs` task.
- Adds GitHub Actions workflow support for documentation checks and GitHub Pages publication.
- Does not change provider runtime behavior or public API signatures.
