## ADDED Requirements

### Requirement: Provider modules SHALL expose static prompt fragments
Provider modules with direct public provider APIs SHALL expose their provider-specific system prompt guidance as raw Markdown `&'static str` values.

#### Scenario: Anthropic module exposes a fragment
- **WHEN** a caller invokes `anthropic::system_prompt_fragment()`
- **THEN** the function returns the Anthropic system prompt fragment as a non-empty `&'static str`

#### Scenario: OpenAI module exposes a fragment
- **WHEN** a caller invokes `openai::system_prompt_fragment()`
- **THEN** the function returns the OpenAI-compatible system prompt fragment as a non-empty `&'static str`

### Requirement: Profile fragments SHALL be selected by API family
`ProviderProfile` SHALL expose `system_prompt_fragment(&self) -> &'static str` and SHALL select the fragment from the profile's `ApiFamily`.

#### Scenario: Anthropic Messages profile resolves Anthropic guidance
- **WHEN** a `ProviderProfile` has `family` set to `ApiFamily::AnthropicMessages`
- **THEN** `ProviderProfile::system_prompt_fragment()` returns the Anthropic system prompt fragment

#### Scenario: OpenAI Responses profile resolves OpenAI-compatible guidance
- **WHEN** a `ProviderProfile` has `family` set to `ApiFamily::OpenAiResponses`
- **THEN** `ProviderProfile::system_prompt_fragment()` returns the OpenAI-compatible system prompt fragment

#### Scenario: OpenAI Chat Completions profile resolves OpenAI-compatible guidance
- **WHEN** a `ProviderProfile` has `family` set to `ApiFamily::OpenAiChatCompletions`
- **THEN** `ProviderProfile::system_prompt_fragment()` returns the OpenAI-compatible system prompt fragment

### Requirement: Registry lookup SHALL resolve prompt fragments for registered providers
`ProviderRegistry` SHALL expose `system_prompt_fragment(&self, provider_name: &str) -> ProviderResult<&'static str>` and use the registered provider profile to resolve the fragment.

#### Scenario: Default registry providers resolve fragments
- **WHEN** a caller requests the system prompt fragment for any default registry slug
- **THEN** the registry returns a non-empty fragment for that provider

#### Scenario: Registry lookup is case-insensitive
- **WHEN** a caller requests a registered provider name using different letter casing
- **THEN** the registry returns the same fragment as the canonical lowercase slug

#### Scenario: Unknown registry provider returns an error
- **WHEN** a caller requests a system prompt fragment for a provider name that is not registered
- **THEN** the registry returns a `ProviderError` instead of falling back to an unrelated fragment

#### Scenario: OpenAI default registry membership is unchanged
- **WHEN** the default provider registry is constructed
- **THEN** this feature does not require `openai` to be present as a default registry slug

### Requirement: Prompt fragments SHALL be safe for caller-side template injection
Prompt fragments SHALL be raw Markdown text suitable for caller-side prompt-template injection and SHALL NOT contain Tera syntax delimiters.

#### Scenario: Fragments contain no expression delimiters
- **WHEN** tests inspect every prompt fragment shipped by this crate
- **THEN** no fragment contains `{{`

#### Scenario: Fragments contain no block delimiters
- **WHEN** tests inspect every prompt fragment shipped by this crate
- **THEN** no fragment contains `{%`

#### Scenario: Fragments contain no comment delimiters
- **WHEN** tests inspect every prompt fragment shipped by this crate
- **THEN** no fragment contains `{#`

#### Scenario: Fragments remain raw Markdown
- **WHEN** a caller retrieves a prompt fragment
- **THEN** the returned string is raw Markdown and this crate performs no template rendering
