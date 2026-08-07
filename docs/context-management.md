# Context management

DeepKeel classifies context by tier and by independent scope, visibility,
retention, representation, authority and subject dimensions.

- L1 contains pinned control and authoritative subject context.
- L2 contains the active thread and run working set.
- L3 contains retrieved memory and evidence that can be omitted first.

Planning is token-aware and model-specific. Tool calls and results remain atomic.
When L2 is too large, DeepKeel keeps a recent raw suffix and creates a
source-linked checkpoint; raw events remain authoritative. Subject-mismatched
authoritative context is quarantined rather than silently mixed.

Hosts provide domain context through `RuntimePorts`. Capability Packages may
enrich the generic envelope through registered contributors but must not bypass
visibility, budget or provenance policy.
