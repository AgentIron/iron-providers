<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

## Local Validation Before GitHub

Before opening or updating a pull request, validate locally anything that GitHub
Actions can validate without reviewer judgment:

- `cargo fmt --manifest-path Cargo.toml -- --check`
- `cargo clippy --manifest-path Cargo.toml --all-targets --all-features -- -D warnings`
- `cargo test --manifest-path Cargo.toml`
- `cargo audit`

Prefer the standardized Invoke workflow when available:

- `inv build`
- `inv test`
- `inv security`

Do not rely on GitHub Actions to catch formatting, lint, test, or audit issues
that can be reproduced locally. CI should avoid blocking code review for policy
checks unrelated to the code under review.

## Pull Request Conventions

- Mention the related GitHub issue in the PR title or body when one exists, for
  example `Closes #123`.
- Do not block PR validation solely because a PR does not reference an open
  issue; issue links are a convention, not a required CI gate.
