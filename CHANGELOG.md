# Changelog

## 3.6.0

- Added a public runtime-event normalizer for replaying persisted Host journal
  rows without exposing nested event envelopes.
- Added a Host event-projection conformance contract covering canonical tool
  lifecycle events and answer deltas.

## 3.5.0

- Added provider-neutral model capability contracts and runtime capability
  learning for concrete provider/model pairs.
- Added structured-output negotiation with automatic JSON Schema to JSON
  Object fallback while preserving local schema validation and repair.
- Added SubAgent diagnostics for requested and effective response formats.

## 3.4.2

- Preserved async-first runtime execution when a Host supplies a synchronous
  LangGraph checkpointer by running the same compiled graph in a worker thread.
- Added an explicit checkpointer async-capability contract, with safe detection
  for legacy sync-only LangGraph savers that inherit unsupported async methods.

## 3.4.1

- Preserved async-first runtime execution when a Host explicitly marks its
  LangGraph checkpointer as synchronous.

## 3.4.0

- Added cursor-addressable runtime event envelopes and a durable event-journal
  Port with replay and adapter conformance contracts.
- Added atomic model-invocation claims, durable settlement, exact result replay,
  and fail-closed recovery for ambiguous provider calls.
- Reworked the canonical LangGraph execution path to be async-first while
  retaining synchronous Host compatibility; native async model and tool
  adapters now receive timeout and cancellation propagation.
- Added telemetry v2 correlation fields and strict operational-metadata privacy
  filtering across run, model, tool, and event identities.

## 3.3.0

- Extended run leases into an end-to-end execution fence available to tool
  handlers and atomic runtime-state mutations; stale generations are rejected.
- Added bounded async stream queues, producer backpressure, cooperative
  disconnect cancellation and configurable cancellation acknowledgement timeouts.
- Added built-in v1-to-v2 persisted-state migrations and real compatibility
  fixtures for suspended and asynchronous runs.
- Expanded fencing, disconnect and crash-recovery conformance coverage.

## 3.2.0

- Added distributed run leases with fencing tokens, heartbeat renewal, expiry
  takeover and a public adapter conformance contract.
- Added explicit state migration chains for durable runtime and checkpoint
  contracts instead of forcing Hosts to guess or silently coerce old state.
- Added `arun()` and `astream()` async Host APIs while retaining one canonical
  synchronous state machine and cooperative stream cancellation.
- Added concurrency, migration and async streaming fault coverage.

## 3.1.0

- Added a public, typed `CompiledSkillSpec` contract and manifest compiler so Hosts no
  longer duplicate Skill schema fields.
- Added tag-to-package release verification and an automated GitHub release workflow.
- Expanded post-extraction fault and boundary gates for standalone consumers.

## 3.0.2

- Make the stdio MCP process launch flags portable across Windows and POSIX type checks.

## 3.0.1

- Aligned the standalone package and Host dependency versions before repository extraction.
- Made the HTTP transport dependency explicit because the public MCP SDK is imported from the package root.
- Added package-owned extraction conformance coverage and tightened distribution metadata.
- Enforced contract-driven tool calls and rejected invalid delegation batches before child execution.

## 3.0.0

- Added canonical `RunJournal`, `RunAggregate`, and `RunStateSnapshot` state with one `run.settled` terminal event.
- Added `hard_interrupt` and `follow_up` input strategies; live steering remains intentionally unsupported.
- Added durable `ModelInvocationEnvelope` recording with exact owner-only debug export and redacted public snapshots.
- Replaced MCP-specific Capability declarations with the protocol-neutral `ToolProvider` port; MCP now lives in the optional `mcp_sdk`.
- Moved bounded SubAgent and deliberation APIs into the optional `orchestration_sdk`.
- Added typed LangGraph state, state invariants, and explicit migration for durable v2 graph checkpoints.
- Split Kuitianjiandi into foundation, bazi, liuyao, knowledge, planning, and orchestration Capability Packs.

## 2.0.0

- Made `HarnessRuntime.run(RuntimeRequest) -> RuntimeResult` the sole Runtime
  execution API and removed the v1 mapping entrypoint.
- Added the typed `RuntimeResult.run_context` projection so Hosts no longer
  reconstruct execution context from legacy result dictionaries.
- Moved product dictionaries, localization, and domain context adaptation to
  the Host boundary.
- Replaced product-shaped context fields with generic `subject`, `facts`,
  provenance, and metadata contracts; incompatible v1 snapshots now fail
  explicitly.
- Added versioned Runtime, Extension, and Adapter SDKs plus a machine-readable
  public API manifest.
- Added workspace packaging so the Host depends on the installable
  `harness-agent-core` distribution rather than scanning Core source files.
- Removed the legacy `register(ToolExecutor)` Capability Pack path; v2 packs
  must declare `CapabilityPackSpec` and implement `install(context)`.
- Added clean wheel and sdist conformance runs across Windows/Linux and Python
  3.12/3.13, plus an installed-Core Host contract gate.
- Reduced the package root to version constants and the three versioned SDK
  modules, and added a frozen public-API fingerprint.
- Added reusable conformance checks for runtime-state, durable-checkpoint and
  idempotent tool-execution adapters.
- Added typed Host-facing runtime projections, a required `RunContext`, a
  `py.typed` marker, dependency compatibility bounds and standalone quality
  gates.

## 1.4.0

- Added a product-neutral reference projection Port with generic record and web
  source normalization.
- Removed product tool names, literature categories and localized source labels
  from the runtime kernel; Hosts now inject those policies explicitly.
- Migrated the product runtime boundary to typed `RuntimeRequest` and
  `RuntimeResult` while retaining one compatibility adapter for existing APIs.
- Added active monorepo CI for the standalone package and Host contract suite.
- Added source-boundary gates preventing direct product use of the legacy Core
  mapping entrypoint and future product reference leakage.

## 1.3.0

- Added the typed `RuntimeRequest`, `RuntimeResult` and `RuntimeStreamEvent`
  public execution contracts.
- Added `HarnessRuntime.run()` while retaining `run_turn()` as the v1 mapping
  compatibility adapter.
- Added explicit run, conversation-thread, graph-thread and turn identities to
  projected runtime results.
- Moved product-neutral contract, event, persistence, failure and model-routing
  tests under package ownership.
- Renamed the distribution to the product-neutral `harness-agent-core` name and
  added a standalone extraction-ready CI workflow.

## 1.2.0

- Replaced product-shaped context assembly with a generic runtime context and
  snapshot contract, with product context supplied through an injectable port.
- Added typed registries for skills, artifact types, handoffs, MCP servers,
  sub-agents and context contributors.
- Made Capability Pack installation atomic and derived contribution manifests
  from actual registrations instead of trusting self-reported metadata.
- Expanded conformance checks across every declared extension type, handoff
  artifact references and sub-agent tool allowlists.
- Added a strict build gate that rejects declaration/installation drift and
  restores prior tool handlers as well as newly registered capabilities.
- Added runtime JSON Schema validation for registered artifact types.
- Added lifecycle-managed Pack resources with reverse-order rollback and
  idempotent runtime shutdown.
- Split durable run snapshots from graph-engine checkpoints through explicit
  `DurableCheckpointStore` and `GraphCheckpointer` contracts.
- Added a product-neutral UI state projection so terminal failures always
  release the composer while interruption states remain explicit.
- Added privacy-safe capability inventory and terminal run telemetry covering
  Skill identity, recovery source and stop reason.
- Standardized context-builder and context-contributor failures as terminal
  runtime results instead of leaking setup exceptions to product hosts.
- Isolated LangGraph checkpoint implementations behind the Core checkpointer
  port and adapter.
- Removed domain-name task inference from Core; tools now declare their generic
  task kind explicitly.
- Made MCP client identity configurable and product-neutral.
- Added source-boundary and package-contract tests that prevent product
  semantics from leaking back into the runtime kernel.
- Added deterministic context-window budgeting with reserved output capacity,
  bounded role history, generic section compaction and privacy-safe diagnostics.
- Added an atomic `RuntimeStateStore` Port for idempotent status, event and
  portable-checkpoint commits, including optimistic concurrency preconditions.
- Extended atomic state mutations to terminal runs, final-message correlation,
  failure metadata and same-transaction cleanup of obsolete resume checkpoints.
- Added explicit `ContextSegment` retention metadata with required sections,
  priorities, provenance, per-section limits and cached-summary fallback.
- Added a source-fingerprint-aware `ContextSummaryCache` Port with a thread-safe
  reference adapter, so Hosts can reuse summaries without serving stale data.
- Expanded the budget ledger from call counts to input/output tokens, model
  retries, elapsed time and peak tool concurrency with idempotent sum/max
  aggregation shared by in-memory and durable adapters.
- Added typed `BudgetPolicy` and `UsageReport` contracts, role-specific request
  ceilings, preflight enforcement and provider-usage reconciliation with an
  explicit estimator fallback.
- Added privacy-safe budget usage events and unified model-route diagnostics so
  Hosts can inspect selected models, output ceilings, actual usage and current
  ledger metrics without exposing prompts or results.
- Moved date-selection constraint recognition and remaining domain vocabulary
  out of Core into the product Capability layer.

## 1.1.0

- Added typed runtime session and LangGraph checkpointer ports.
- Added declarative Capability Pack specifications, installation contexts and
  contribution manifests while retaining the v1 legacy registration API.
- Expanded Capability Pack conformance checks to cover declarations, handlers,
  JSON Schema and unsafe parallel write tools.
- Expanded the top-level public SDK and added a package-owned contract suite.
- Added an explicit model provider adapter contract with legacy native-provider
  compatibility.
- Added a fail-open, privacy-safe telemetry port with stable run, thread, turn
  and step correlation.
- Added a governed MCP Streamable HTTP transport with JSON/SSE response
  support, session negotiation and recovery, secret-backed headers, safe
  diagnostics and graceful session shutdown.
- Split shared MCP protocol contracts from the stdio and HTTP transport
  adapters so additional transports do not depend on implementation details.

## 1.0.0

- Extracted the product-neutral Harness runtime under the frozen
  `harness-core-v1` contract.
