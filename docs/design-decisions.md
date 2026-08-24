# Design decisions

This document records the boundaries DeepKeel intentionally preserves. It is a
guide for maintainers and integrators, not a claim that every agent application
needs the same degree of runtime governance.

## 1. DeepKeel is a runtime, not a product framework

DeepKeel owns the semantics that must remain consistent across products:

- one model/tool execution lifecycle;
- typed run state, interruption, cancellation, resume, and settlement;
- policy, budget, idempotency, lease, and fencing boundaries;
- versioned events, artifacts, observations, and failure contracts;
- replaceable infrastructure Ports and Capability Package composition.

The Host owns HTTP, SSE, authentication, tenancy, queues, credentials, database
connections, deployment, product policy, and frontend rendering. Capability
Packages own domain behavior. Keeping these responsibilities separate prevents
the runtime from accumulating one product's ORM models, routes, prompts, and UI
assumptions.

## 2. LangGraph is an internal engine, not the public abstraction

LangGraph is a good fit for checkpoint-aware graph execution, interrupts, and
resume cursors. DeepKeel uses it behind the internal `TurnExecutionEngine`
contract rather than exposing LangGraph `StateGraph`, `Command`, or saver types
through the public SDK.

This separation has three effects:

1. Hosts and Capability Packages depend on DeepKeel's stable contracts instead
   of graph construction details.
2. Product-visible state and recovery policy are not delegated to a graph saver.
3. The execution engine can be tested or replaced without changing package
   manifests, tool contracts, or frontend event consumers.

The tradeoff is additional adapter code. DeepKeel accepts that cost because the
product lifecycle is broader than graph traversal.

## 3. One compiled graph, many immutable capability views

DeepKeel does not compile a separate graph for every user or specialist. A
worker reuses the canonical graph and carries request-specific state through the
turn. An `AgentEntrypointSpec` resolves to an immutable `CapabilityView` that
narrows packages, Skills, tools, SubAgents, context, Memory, permissions, prompt
policy, and model policy.

The capability view is a conversation invariant. Switching to a different
specialist creates another root conversation rather than silently changing the
meaning of an existing history. Child work may narrow the parent view but cannot
add capabilities absent from it.

This design bounds graph compilation overhead and makes capability scope
auditable. It also means packages must express variability through state and
registered contracts, not by mutating a shared graph at runtime.

## 4. Capability discovery is permission-first

Large catalogs cannot be copied into every model prompt. DeepKeel applies the
following order:

1. intersect the installed runtime generation with the active agent entrypoint;
2. apply tenant, role, policy, and explicit Skill constraints;
3. retrieve candidates from the remaining catalog;
4. rerank and disclose only a bounded descriptor set;
5. load full Skill or tool details only after activation.

The discovery adapter can rank or abstain, but it cannot reintroduce a capability
removed by permission filtering. The deterministic lexical implementation keeps
the SDK portable; Hosts with larger catalogs can provide semantic retrieval and
reranking through `deepkeel.discovery_sdk`.

This is deliberately not a guarantee that a model will always select the right
Skill. Selection quality remains an evaluated product concern. DeepKeel provides
the bounded candidate set, trace evidence, and deterministic contracts needed to
measure it.

## 5. Skills, workflows, SubAgents, and handoffs are distinct

These abstractions solve different problems:

| Mechanism | Use when | Ownership |
| --- | --- | --- |
| Prompt Skill | The same agent needs specialized instructions or examples | Parent run keeps the loop |
| Interactive workflow | A bounded sequence still needs model decisions between steps | Parent run keeps the loop and state |
| Delegated workflow | Deterministic or asynchronous domain work has its own lifecycle | Domain worker executes; parent resumes with the result artifact |
| SubAgent | A specialist needs an independent context and model/tool loop | Child run is bounded by the parent scope, budget, depth, and cancellation |
| Handoff | Execution must wait for a user, external system, or background task | Parent run persists a typed `PendingAction` and resumes with an `Observation` |

A delegated workflow returns its structured artifact to the parent model before
the final answer, unless the product explicitly defines the artifact itself as
the terminal response. This preserves the root agent's responsibility for a
coherent user-facing answer.

## 6. Three persistence mechanisms have non-overlapping authority

DeepKeel distinguishes product state from execution continuation:

| Mechanism | Authority |
| --- | --- |
| `RuntimeStateStore` | Canonical product-visible status, ordered event sequence, settlement, fence generation, and current portable checkpoint projection |
| `DurableCheckpointStore` | Portable recovery envelope and explicit compatibility fallback for older runs |
| LangGraph checkpointer | Internal graph continuation for a super-step or interrupt cursor |

The LangGraph checkpoint cannot decide that a run is complete, failed, or safe to
unlock in the UI. Resume reads canonical runtime state first. A compatibility
fallback is permitted only when its authority and freshness can be established;
storage, timeout, or deserialization failures otherwise fail closed.

## 7. Ordered runtime events are the projection source

DeepKeel treats the versioned `RuntimeEventEnvelope` as the lifecycle fact. Live
SSE events, durable journal rows, OpenTelemetry spans, compact trace records, and
frontend `ui_state` are projections of that event stream.

Streaming answer deltas are transient presentation events and may be coalesced.
Terminal settlement, pending actions, artifacts, and recovery facts are durable.
Replaying persisted events must not generate duplicate public effects.

This design avoids separate code paths inventing contradictory status. It also
requires Hosts to preserve event identity, ordering, and scope when implementing
their adapters.

## 8. Side effects require idempotency and execution ownership

A tool call receives a stable execution identity derived from the run, turn, and
call. The execution store atomically claims the call, replays a settled result,
or rejects competing ownership. Before settlement, Core checks the active
execution fence. A Worker that lost its run lease cannot commit a late result
after a newer Worker takes over.

Core protects the runtime boundary; a write tool should propagate the same
idempotency key and fencing token into its downstream database or service when
that system can perform side effects independently.

## 9. Context layers express different authority

- **L1** is protected current-turn context and complete tool-call exchanges.
- **L2** is compact working context with source ranges, fingerprints, subjects,
  and checkpoint lineage.
- **L3** is durable recall selected through a Host-provided memory policy and
  storage Port.

Compaction must not treat assistant prose as completed work or silently truncate
the current request. If protected context cannot fit the selected model, the run
fails clearly rather than constructing a misleading prompt.

## 10. Production is an executable profile

`build()` is convenient for tests and local embedding. Production composition
uses `build_production()` and fails closed when mandatory durable or governed
Ports are absent, ambiguous, or backed by known `InMemory*`/`Noop*` adapters.

The profile cannot prove the correctness of a custom adapter. Hosts must run the
exported conformance suites, PostgreSQL or equivalent multi-worker tests, fault
injection, and downstream compatibility gates.

## Non-goals and tradeoffs

- DeepKeel does not optimize model quality by itself. It makes model routing,
  context, capability selection, and failures measurable and replaceable.
- DeepKeel does not provide a universal business Memory taxonomy. The Core owns
  safe recall contracts; the Host and package own what is worth remembering.
- DeepKeel does not make arbitrary third-party code safe in-process. Untrusted
  packages require an isolated execution policy and a Host-operated sandbox.
- DeepKeel favors explicit contracts over a minimal API surface. Applications
  that only need a short stateless loop may be better served by a smaller SDK.

See [Architecture](architecture.md), [Runtime lifecycle](runtime-lifecycle.md),
[Capability Package V1](capability-package-v1.md), and
[Production readiness](production-readiness.md) for implementation details.
