# Context management

[English](context-management.md) | [简体中文](context-management.zh-CN.md)

DeepKeel classifies context by tier and by independent scope, visibility,
retention, representation, authority and subject dimensions.

- L1 contains pinned control and authoritative subject context.
- L2 contains the active thread and run working set.
- L3 contains retrieved memory and evidence that can be omitted first.

Planning is token-aware and model-specific. Tool calls and results remain atomic.
When L2 is too large, DeepKeel keeps a recent raw suffix and creates a
source-linked checkpoint; raw events remain authoritative. Subject-mismatched
authoritative context is quarantined rather than silently mixed.

Before invocation, `ContextQualityGate` can audit provenance, authority,
subject alignment, duplicate content, empty items, and declared token estimates.
`observe` mode records issues without changing input; `enforce` mode rejects
critical L1/L2 defects while allowing replaceable policies to decide which L3
quality findings are tolerable. The quality report is diagnostic evidence, not
a new source of truth.

## Semantic checkpoints

`DeterministicContextWindowManager` accepts an optional
`ContextCheckpointBuilder`. The builder receives a deterministic draft plus
defensive copies of the omitted and retained messages, and may enrich goals,
decisions, progress, failures and next steps. Core verifies the returned
checkpoint against immutable source fields and requires every critical fact to
reference a supplied message or an earlier checkpoint fact. Invalid output or
builder failure falls back to the deterministic draft and is recorded in the
context diagnostics. This keeps semantic compression replaceable without
making a model-generated summary authoritative.

The same working-context compactor is reused for the runtime context window and
the final provider-specific model input budget. A Host therefore configures the
semantic boundary once instead of receiving different summaries at routing and
invocation time.

Hosts provide domain context through `RuntimePorts`. Capability Packages may
enrich the generic envelope through registered contributors but must not bypass
visibility, budget or provenance policy.

## Selective memory recall

Long-term memory is optional L3 context, not a mandatory request-time lookup.
A Host may inject `RuntimePorts(memory_recall_coordinator=...)` and implement a
product-specific `MemoryRecallPolicy`. The policy receives a small,
serializable `MemoryRecallRequest` containing request identity, subject scope,
recent working history, pending action and Skill metadata. It returns one of:

- `skip`: the current working context is sufficient;
- `prefetch`: retrieve memory before context-window planning;
- `agent_decide`: do not prefetch and let the ReAct loop call a governed memory
  search tool if a later observation reveals that history is required.

`DefaultMemoryRecallCoordinator` executes prefetch through the generic
`MemoryPort`, applies a bounded TTL cache, injects projected records into L3 and
fails open when memory infrastructure is unavailable. A policy can also disable
the runtime memory-search tool for an opted-out or unsafe subject scope. Recall
decisions and outcomes are emitted as internal `memory.recall.*` events for new
turns and copied into runtime diagnostics. Resume paths preserve
`run.resumed` as their first event and keep the skipped-recall decision in
diagnostics only. Prompts and raw sensitive records are not added to public
telemetry by the coordinator.

Async Hosts may provide `AsyncMemoryRecallPolicy` and `AsyncMemoryPort`; the
coordinator prefers their native `adecide()` and `asearch()` methods. Existing
synchronous policies and ports remain supported through explicit thread
offload, so they do not block the Host event loop.

The coordinator supports `legacy`, `shadow` and `enforced` rollout modes.
`legacy` and `shadow` preserve eager prefetch behavior while Hosts compare
traces; `enforced` follows the policy decision. Core deliberately does not
classify product intent or extract memory domains: those rules remain in the
Host, while storage and retrieval remain replaceable through `MemoryPort`.
