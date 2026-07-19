# Changelog

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
