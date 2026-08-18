# Changelog

## Unreleased

- Relaunch stdio MCP servers before legacy negotiation when modern discovery
  times out, preventing a reset transport from breaking the fallback request.
- Preserve explicit trusted Host governance scopes for lightweight Capability
  Packs that intentionally omit a manifest, while keeping manifest-declared
  permissions authoritative for governed packages.

## 4.1.0 - 2026-08-18

- Add versioned user-facing Agent entrypoints and immutable per-conversation
  Capability Views. Direct specialist Agents now reuse the canonical compiled
  graph while narrowing packages, Skills, tools, SubAgents, context, Memory,
  permissions, prompts and model policy without creating a second runtime.
- Add optional bounded Plan & Execute orchestration inside the canonical ReAct
  graph, including validated DAGs, safe parallel reads, serialized side effects,
  bounded revisions, typed progress events, and portable interruption recovery.
- Add the `AgentHarness` Golden Path for minimal embedding while retaining the
  explicit `HarnessRuntimeBuilder` path for production composition.
- Add first-class input, model-output, and tool-boundary Guardrails with ordered
  decisions, replay safety, audit records, provenance labels, and fail-closed
  required policies.
- Add replaceable Sandbox and Workspace ports with bounded execution metadata,
  deterministic cleanup, policy enforcement, and production-readiness checks.
- Add progressive Skill and Tool discovery with replaceable candidate retrieval
  and reranking, plus L1/L2/L3 context quality checks for provenance, subject,
  authority, duplication, and token-budget drift.
- Add privacy-bounded online evaluation and OpenTelemetry GenAI semantic
  attributes without exporting prompts, tool payloads, or model output by
  default.
- Upgrade MCP to the modern `2026-07-28` protocol era, including stateless
  discovery, request metadata, task lifecycle, safe parameter headers,
  `outputSchema` validation, and bounded fallback to legacy servers.
- Add an optional experimental A2A 1.0 adapter that maps remote Agent Cards and
  Tasks into the existing governed SubAgent, checkpoint, cancellation, Artifact,
  and parent-owned final-answer lifecycle.
- Extend production and release gates for local safety adapters, clean package
  verification, SBOM generation, build provenance, bilingual documentation, and
  an explicit 4.1 migration checklist.
- Keep the persisted Capability Pack contract at `harness-core-v3`; 4.1 is an
  additive SDK and runtime release and requires no persisted-state migration.

## 4.1.0rc1 - 2026-08-10

- Project each uninterrupted run segment as one hierarchical OpenTelemetry
  trace, with operation child spans, lifecycle events, replay deduplication,
  delta aggregation, and links between resumed segments.
- Add a replaceable semantic context-checkpoint builder that is verified
  against deterministic source ranges, fingerprints, subjects, and fact
  references before it can enrich L2 working context.
- Route adaptive model steps against provider health, context capacity,
  modality, native-tool support, latency, cost and remaining budget while
  preserving strict user-selected single-model semantics.
- Split model-node execution and RuntimeResult projection into typed phases,
  and move trace/diagnostic projection behind independently ratcheted modules.
- Split routed model attempts, SubAgent batch/bounded/task execution and the
  claimed runtime turn into typed lifecycle coordinators with strict size and
  complexity ratchets.
- Define non-overlapping persistence authority for canonical product state,
  portable recovery checkpoints and LangGraph continuation checkpoints.
- Isolate runtime events, leases, tool/model idempotency, checkpoints and
  operational identities by `RuntimeScope`, with a forward-only PostgreSQL v3
  migration and cross-tenant contract tests.
- Move thread-safe synchronous PostgreSQL, model, tool, telemetry and
  cancellation operations off the async Host loop; preserve bounded streaming
  and cooperative disconnect cancellation.
- Add adapter capability declarations, executable model-provider
  certification, OpenTelemetry metrics, MCP egress/size controls, semantic API
  snapshots and critical coverage/complexity release budgets.

## 4.0.0 - 2026-08-08

- Split turn preparation, durable snapshot commits, and tool idempotency into
  independently testable phases, then tighten the complexity ratchets around
  the smaller orchestration functions.
- Freeze semantic descriptors for all stable Runtime, Extension, and Memory
  SDK symbols rather than a hand-picked subset.
- Extend the PostgreSQL reference bundle with capability catalog, context
  summary, Memory, and SubAgent lineage/checkpoint/suspension stores.
- Add explicit development, testing, and production runtime profiles. Production
  composition now fails closed unless progressive tool disclosure is enforced.
- Promote `HarnessRuntimeBuilder` to the stable Runtime SDK. The Adapter SDK
  keeps a transitional import attribute, but it is no longer its canonical API
  owner.
- Package PostgreSQL worker adapters under `deepkeel.contrib.postgres`, covering
  canonical state, checkpoints, events, leases, model/tool idempotency, budget,
  model health, cancellation, and scoped traces behind one composition bundle.
- Add native async contracts for Memory recall policy/storage, tool execution
  idempotency, and bounded SubAgent delegation. Synchronous implementations
  remain compatible through explicit thread bridges instead of blocking the
  Host event loop.
- Replace ad-hoc PostgreSQL table initialization with a forward-only,
  checksummed Schema Registry. Migrations use an advisory lock, detect history
  and physical-column drift, repair the supported pre-cursor trace schema, and
  reject automatic downgrades.
- Add a privacy-safe OpenTelemetry projection, machine-readable `deepkeel`
  diagnostics and PostgreSQL migration CLI, plus a public-SDK-only production
  worker composition example.

## 4.0.0rc2

- Make the six versioned SDK modules the unique public symbol owners, publish
  machine-readable stability levels, and reject cross-layer duplicate exports.
- Group runtime infrastructure into persistence, governance, observability,
  and execution Port bundles without breaking the flat `RuntimePorts` API.
- Split the runtime, model gateway, tool executor, Graph model node, SubAgent
  executor, and context window internals behind compatibility facades.
- Add structural ratchets for file and method size plus an acyclic internal
  import-graph gate.
- Harden event-journal fail-closed behavior and checkpoint authority fallback
  through targeted fault injection.
- Add executable, product-neutral PostgreSQL reference adapters for canonical
  state, events, run leases, and durable checkpoints, including multi-worker
  contention, crash rollback, recovery, and CI baselines.
- Add a host-injected selective memory-recall coordinator with skip, prefetch,
  and agent-decide policies, bounded caching, failure isolation, and traceable
  recall decisions.

- Wire native async runtime state, event journal, checkpoint, and run-lease
  ports into the canonical async execution path.
- Bound same-loop streaming backpressure without spawning an unbounded set of
  pending queue tasks, while preserving all answer text.
- Add an executable production-readiness report and fail-closed
  `build_production()` composition gate.
- Persist terminal checkpoint-cleanup diagnostics as replayable runtime events
  without deleting recovery state before terminal settlement, and reject
  conflicting duplicate runtime identities.
- Split deterministic concurrency correctness from the scheduled Linux p95
  performance baseline.
- Add a full Simplified Chinese README and an automated bilingual release
  contract.

## 4.0.0rc1

- Rename the project, distribution, and Python namespace to DeepKeel.
- Establish Apache-2.0 licensing and public community policies.
- Decouple Core release verification from any product Host repository.
- Restore automatic cross-platform CI and add Python 3.14 verification.
- Publish a frozen DeepKeel v4 SDK surface and product-neutral quickstarts.
- Preserve the existing `harness-core-v3` Capability Pack protocol so branding
  does not force a persisted contract migration.
- Add an explicit same-contract bridge for persisted v3 capability generations
  and replace ad-hoc version parsing with PEP 440 semantics.

Versions through 3.35.1 were published as the pre-DeepKeel engineering lineage.

## 3.35.1

- Preserve the public `BudgetExceededError` contract when protected current
  context cannot fit, while recording the context-budget failure reason.

## 3.35.0

- Protect the canonical current turn and complete tool-call exchanges from
  silent model-context truncation, failing clearly when they cannot fit.
- Chain subject-aware working checkpoints across repeated compactions without
  treating assistant prose as completed work or losing middle history.
- Align input planning with the actual per-call output budget and expose
  checkpoint lineage, coverage, and subject diagnostics.

## 3.34.0

- Add orthogonal L1/L2/L3 context contracts for scope, visibility, retention,
  representation, authority, subject provenance, and source references.
- Plan context against candidate model limits before routing and apply the
  selected provider's exact context and output limits at every model step.
- Replace fixed message-count truncation with token-aware atomic compaction,
  deterministic working checkpoints, subject quarantine, and context manifests.

## 3.33.0

- Split model-attempt invocation, replay settlement, streaming accounting, and
  token usage from the routed model gateway without changing the public SDK.
- Extract Graph model-step context, metrics, and tool-disclosure projections
  from the central node implementation.
- Add ratcheted structural tests so the central model and Graph execution
  methods cannot silently grow back into monolithic control paths.

## 3.32.1

- Recover an opt-in explicit workflow's unique required entry tool when a
  provider ignores a forced-tool contract and every required argument can be
  resolved from Host-declared context or the latest user message.
- Keep the recovery inside the normal validation, policy, idempotency, and
  tool-execution path, with a dedicated diagnostic event.

## 3.32.0

- Add explicit host-context argument bindings so tool execution can override
  model-invented identity and resource parameters before handler dispatch.
- Resume incomplete deliberation stages without replaying participants that
  already completed the same stable round argument.
- Make soft-stop requests converge after the current opening batch, skip new
  moderation/rebuttal work, and still produce a lead-agent synthesis.
- Expose participant, retry, recovery, model-route, and partial-failure
  diagnostics in durable deliberation results.

## 3.31.0

- Add participant-scoped fact views and instructions to the product-neutral
  deliberation contract while preserving shared identity and provenance.
- Reserve synthesis budget against the moderator's actual rebuttal targets
  instead of assuming every participant must answer every round.
- Emit explicit deliberation stage transitions and include model-route,
  partial-failure, retry, and budget diagnostics in durable results.

## 3.30.0

- Add a unified, durable SubAgent task brief with context and artifact
  references, stable idempotency keys, parent/child lineage, task budgets,
  deadlines, and cooperative cancellation policy.
- Promote `TaskBrief` as the preferred name while retaining
  `DelegationTask` as a fully compatible public alias.
- Extend specialist results with first-class artifact/context references and a
  typed `needs_input` suspension contract that resumes through the parent
  Agent's pending-action flow.
- Standardize SubAgent event payload identity and lifecycle fields under
  `harness-subagent-event-v1`.

## 3.29.2

- Recover forced visual tool contracts without weakening ordinary tool-call
  validation.

## 3.29.1

- Preserve provider-neutral image content parts when resuming a clarification checkpoint.

## 3.29.0

- Repair only structurally incomplete native tool-call JSON truncated at the
  end of provider output before consuming a model retry.
- Keep semantic corruption, missing separators, and invalid value types on the
  typed retry/failure path instead of applying permissive JSON coercion.

## 3.28.0

- Classify malformed native tool-call arguments as a retryable model contract
  failure instead of an internal runtime error.
- Retry a failed native tool call at most once with focused JSON repair
  guidance while preserving multimodal inputs and forced-tool semantics.
- Expose the repair category and strategy through model route diagnostics.

## 3.27.0

- Add provider-neutral text and image message parts backed by opaque media
  references, with inline binary payloads rejected at the Core boundary.
- Preserve multimodal references across history, checkpoint, replay, and
  provider message projection while keeping host-owned media resolution out of
  the runtime.
- Declare image-input support in model capability evidence.

## 3.26.3

- Allow capability conformance and certification to compose real dependency
  packs, so cross-package tools and handlers are exercised by the release gate.

## 3.26.2

- Resolve only the target package's transitive dependency closure during
  conformance so callers may pass a complete package catalog safely.

## 3.26.1

- Allow package conformance and certification to install declared dependency
  manifests before validating a dependent Capability Package.
- Report missing, cyclic, or incomplete dependency sets as structured
  conformance issues instead of raising outside the report.

## 3.26.0

- Make `CapabilityManifest` the single source for package declarations and
  expose a supported converter to `CapabilityPackSpec`.
- Add per-tool permission mappings and apply them automatically during runtime
  composition so package governance declarations are executable.
- Add a Capability Package certification gate covering structure, permissions,
  lifecycle, runtime generations, rollback, resume compatibility, and required
  behavior-evaluation scenarios.
- Ship the Capability Package V1 guide and a product-neutral inventory package
  proving second-domain integration through public SDKs only.

## 3.25.0

- Add a reusable Budget Ledger conformance verifier for durable Host adapters.
- Verify idempotent accounting, limit rejection, peak aggregation, and
  monotonic checkpoint restoration without constraining audit retention.

## 3.24.0

- Connect the Capability Package control plane to runtime composition through
  an explicitly pinned, validated Runtime Generation.
- Reject resume on an incompatible persisted generation before model or tool
  execution.
- Add a reusable Capability Package Store conformance verifier for durable Host
  adapters.

## 3.23.0

- Add typed Capability Package budgets, state schema versions, migration
  declarations, and resume-compatible version contracts.
- Retain immutable Runtime Generations in the package catalog so interrupted
  runs can resolve the exact capability set they started with.
- Expose deterministic compatibility diagnostics when an old generation must
  migrate to the current package set.

## 3.22.0

- Add a persistent-port-backed Capability Package control plane with
  optimistic concurrency.
- Support validated package install, enable, disable, upgrade, rollback, and
  uninstall operations.
- Produce immutable Runtime Generations from the enabled package catalog while
  failing closed on missing dependencies and capability ownership conflicts.

## 3.21.2

- Add a host-provided entry-tool Skill activation port so a Skill can be
  promoted inside the first ReAct tool-selection step without a separate
  model pre-router.
- Preserve explicit Skill activation while allowing trusted package entry
  tools to freeze Skill context and normalize their arguments before
  execution.
- Emit an auditable hidden `skill.activated` runtime event for model-selected
  entry tools.
- Treat user handoffs and asynchronous starts as completed Skill workflow transitions.
- Isolate tool lifecycle classification from graph orchestration.

## 3.20.0

- Add a host-provided `ModelHealthStore` port and a process-local default so
  model fallback can share circuit state across runtime steps and workers.
- Separate governed model-pipeline composition from the orchestration runtime.
- Split package, runtime-contract, event-schema, and SDK API versions.
- Declare Skill execution modes for inline, background, and user-handoff
  background workflows.

## 3.17.1

- Recover ignored forced tool calls as structured clarification only when the
  tool declares required arguments and a clarification contract.
- Preserve strict contract failures for tools that cannot safely suspend for
  missing input.

## 3.17.0

- Add a canonical `TaskLifecycle` projection while preserving engine execution
  status for recovery and diagnostics.
- Unify composer blocking, progress, cancellation, and terminal settlement
  semantics across ordinary runs and durable workflows.
- Add portable `EvidenceBundle` and `ArtifactView` contracts so Capability
  Packs can expose evidence and result surfaces without Host-specific renderers.

## 3.16.7

- Scope undisclosed-tool rejection to answer-only Workflow finalization so
  ordinary tool and handoff execution remains governed by PolicyEngine.

## 3.16.6

- Enforce the answer-only Tool View after a Workflow contract is complete and
  reject stale model calls to suppressed tools during finalization.
- Inject an answer-only Workflow finalization guard and recover once when a
  provider still emits a stale tool call after the contract is satisfied.

## 3.16.5

- Deduplicate identical implicit tool calls emitted in one model response while
  preserving distinct arguments and explicit idempotency keys.

## 3.16.4

- Make artifact types, context contributors, and resources first-class Capability Manifest declarations.
- Enforce full manifest-to-package conformance for every public capability category.
- Bound semantic tool discovery to two attempts per turn and expose the count in runtime diagnostics.

## 3.16.3

- Enter a tool-free finalization step once a durable Workflow Skill has
  satisfied its required tools and Artifact contract, preventing models from
  repeatedly invoking completed generation actions.

## 3.16.2

- Merge repeated observations of the same Artifact identity instead of
  producing invalid duplicate state during multi-step Workflow execution.
- Preserve the original creation timestamp while allowing later tool results
  to enrich Artifact data, metadata, summary, and completion state.

## 3.16.1

- Preserve known model capabilities when provider adapters also declare transport-specific hints.
- Support providers that allow native tools but reject forced-function `tool_choice` objects by retaining strict post-response tool validation in automatic mode.

## 3.16.0

- Add scoped lifecycle Hooks with timeout isolation, idempotent replay,
  governance auditing, context enrichment, argument rewriting, denial, and
  user-confirmation decisions.
- Add fail-closed progressive tool disclosure with baseline, Skill entry,
  discoverable, Skill-only, and internal exposure classes plus a replaceable
  semantic discovery Port.
- Upgrade context management to four explicit layers with protected runtime
  state, source and token diagnostics, summary provenance, and safe
  over-budget reporting.
- Add versioned Capability Manifests and immutable Runtime Generations with
  dependency, compatibility, conflict, and rollback validation.
- Taskize SubAgent delegation with execution modes, scoped permissions and
  budgets, cancellation propagation, structured parent projections, and
  Artifact references.

## 3.15.0

- Add `RuntimeScope` and scoped state/trace contracts with fail-closed legacy
  adapter compatibility for tenant and namespace isolation.
- Extend `RunOperations` with stale-run discovery and auditable, idempotent
  resume, retry, requeue, and terminalize command submission.
- Add asynchronous persistence Port contracts and opt-in thread-offload
  bridges for thread-safe synchronous adapters.
- Add fault coverage for a completed external side effect whose durable
  settlement fails.
- Add user-scoped runtime state isolation to the reference adapter and enforce
  state/checkpoint isolation in adapter conformance.
- Add deterministic Eval suite contracts for status, tools, artifacts, step
  budgets, errors, and ordered trace events.
- Add production-readiness guidance and failure-injection coverage for
  persistence outages and partially failed parallel tool execution.

## 3.13.2

- Interpret an unlimited product budget as unlimited total generation rather
  than an unbounded single model request: fast steps default to 8K output and
  reasoning steps to 16K, while automatic continuation preserves long answers.
- Reserve context-window headroom before calculating each provider request's
  physical output allowance.

## 3.12.5

- Detect model responses stopped by output token limits, continue them
  automatically, and merge streamed continuation segments into one final
  answer instead of settling a truncated response as completed.

## 3.12.4

- Treat tools skipped after a user-action or asynchronous suspension as
  internal scheduling events instead of user-visible execution failures.

## 3.12.3

- Include the turn identity in durable model invocation IDs so resumed or
  multi-turn runs cannot collide with an earlier step-zero invocation.

## 3.12.2

- Promoted the Memory contracts to a first-class versioned SDK layer and added
  them to the machine-readable public API manifest.

## 3.12.1

- Added product-neutral Memory claim, evidence, mutation, retrieval, and
  host-provided persistence port contracts.

## 3.12.0

- Added a portable recovery outcome contract that distinguishes successful,
  typed terminal, untyped terminal, aborted, recovering, and stuck runs.
- Projected safe tool identity and argument digests from nested lifecycle
  events so persisted traces no longer degrade completed calls to unknown tools.

## 3.11.1

- Isolate telemetry destination failures so observability backends cannot interrupt Agent execution.

## 3.11.0

- Added a portable persistent TraceStore query and retention contract plus a
  composite telemetry adapter for simultaneous storage and structured logs.
- Added conformance coverage for TraceStore and shared ContextSummaryCache
  adapters so multi-worker Hosts can share bounded context summaries safely.

## 3.10.0

- Added a host-configurable LangGraph durability policy and made `exit` the
  default so internal super-steps do not create unnecessary checkpoints while
  interrupt and terminal boundaries remain recoverable.
- Suppressed ephemeral token-stream telemetry from structured logs by default,
  with an explicit opt-in for low-level diagnostics.

## 3.9.1

- Allowed the internal catalog discovery tool through non-empty Skill
  allowlists while preserving explicit empty allowlists as a strict deny-all.
- Added trace-correlation and structured-logging telemetry contract coverage.

## 3.9.0

- Made `RuntimeStateStore` the authoritative portable recovery source while
  retaining LangGraph checkpoints for engine-local continuation and the legacy
  checkpoint store as a compatibility fallback.
- Added model-driven tool catalog discovery and durable per-run disclosure
  grants so enforced progressive disclosure can replace full tool exposure.
- Added structured logging telemetry with deterministic trace and span
  correlation across runtime events.
- Added a production SQL lease/fencing adapter in the reference product host.

## 3.8.0

- Reused one versioned LangGraph per runtime worker and moved model, prompt,
  deadline, event sink, and tool execution context into an isolated per-turn
  execution context.
- Added separate executable and model-visible tool views with legacy, shadow,
  and enforced disclosure modes plus catalog-version diagnostics.
- Made active Skill allowlists fail closed, including explicit empty
  allowlists, while preserving general tool use when no Skill is active.
- Added graph contract and tool-view metadata to runtime diagnostics and
  durable checkpoints.

## 3.7.0

- Added the portable `artifact-presentation-v1` Skill Package contract so Capability Packs can describe artifact summaries, fields, and navigation without coupling Core to product UI code.
- Added strict validation that artifact presentation types match the package artifact contract.

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
  `deepkeel` distribution rather than scanning Core source files.
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
- Renamed the distribution to the product-neutral `deepkeel` name and
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
