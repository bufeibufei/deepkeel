# Agent Entrypoints

DeepKeel can expose several directly addressable root Agents from one installed
runtime generation. A general assistant and a domain specialist therefore share
the same lifecycle, graph, persistence and observability contracts without
sharing every capability.

## Contracts

Capability Packages register `AgentEntrypointSpec` through the Extension SDK.
The specification owns product-facing identity and narrowing policy:

- package and dependency scope;
- Skill, tool and SubAgent allowlists;
- specialist system prompt and model policy;
- context, Memory and handoff policy metadata;
- a version used to keep persisted conversations reproducible.

At turn preparation, DeepKeel resolves the specification against the immutable
`RuntimeGeneration` and produces a `CapabilityView`. The view contains the
effective packages, Skills, tools, SubAgents, context contributors, artifacts,
Memory namespaces and permission scopes plus a stable `scope_hash`.

## Execution

The Host persists `agent_entrypoint_id` and `agent_entrypoint_version` with a
conversation and passes both on every `RuntimeRequest`. The entrypoint is a
conversation invariant: switching specialists creates another root conversation
instead of mutating the meaning of an existing history.

All entrypoints execute on the canonical ReAct graph. DeepKeel compiles the graph
once and carries the `CapabilityView` in per-turn state. Tool disclosure,
context contributors, lifecycle hooks and explicit Skill activation are filtered
at runtime, so entrypoints do not multiply graph objects or worker pools.

## Safety

The effective scope is fail-closed. Composition rejects an entrypoint that names
an unavailable package or out-of-scope capability. Runtime control fields are
re-applied after Host and package context enrichment, preventing a contributor
from dropping or replacing the authoritative scope. Child execution may use
`narrow_capability_view` but cannot add tools, Skills or SubAgents absent from its
parent.

An empty entrypoint identifier selects the backward-compatible unrestricted
default view. This keeps existing Hosts operational while they add persisted,
versioned specialist entrypoints incrementally.
