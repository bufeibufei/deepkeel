# Harness Agent Core

`harness-agent-core` is a product-neutral runtime kernel for tool-using agents.
It owns the model/tool loop, runtime contracts,
governance ports, interruption, recovery, ToolProvider integration and
Capability Pack composition. MCP transports and bounded orchestration are
optional SDK modules rather than required runtime concepts.

Production deployments should follow
[`docs/production-readiness.md`](docs/production-readiness.md), replace every
process-local `InMemory*` adapter, and run the exported adapter conformance
verifiers against the real backend.

The package does not contain host tools, database models, API routes or product
prompts. Consumers integrate through `HarnessRuntimeBuilder`,
`RuntimePorts` and a versioned Capability Pack. New packs should expose a
`CapabilityPackSpec` and implement `install(CapabilityInstallContext)`.
There is no implicit registration or legacy Pack adapter in the v3 contract.

```python
from harness_core.adapter_sdk import HarnessRuntimeBuilder
from harness_core.extension_sdk import (
    ArtifactTypeSpec,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPackSpec,
)

class InventoryPack:
    spec = CapabilityPackSpec(
        package_id="example.inventory",
        package_version="1.0.0",
        declared_tools=("inventory.lookup",),
        declared_skills=("inventory-assistant",),
        declared_artifact_types=("inventory_record",),
    )

    def install(self, context: CapabilityInstallContext):
        context.register_tool(tool_spec, lookup_handler)
        context.register_skill("inventory-assistant", skill_manifest)
        context.register_artifact_type(
            ArtifactTypeSpec(
                artifact_type="inventory_record",
                schema={"type": "object"},
            )
        )
        return CapabilityContribution(
            package_id=self.spec.package_id,
        )

runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()
```

Capability Package lifecycle is managed separately from process-local runtime
composition. A Host persists the catalog through `CapabilityPackageStore`,
applies install/enable/disable/upgrade/rollback operations through
`CapabilityPackageManager`, and builds workers from the resulting immutable
`RuntimeGeneration`. Existing runs retain the generation captured when they
started.

```python
from harness_core.extension_sdk import (
    CapabilityPackageManager,
    InMemoryCapabilityPackageStore,
)

packages = CapabilityPackageManager(InMemoryCapabilityPackageStore())
packages.install(inventory_manifest)
generation = packages.generation()

runtime = (
    HarnessRuntimeBuilder()
    .add_capability_pack(InventoryPack(), manifest=inventory_manifest)
    .with_runtime_generation(generation)
    .build()
)
```

The in-memory store is for tests and single-process embedding only. Production
Hosts must provide a durable, optimistic-concurrency implementation and run
`verify_capability_package_store_contract` against it. The builder rejects a
generation whose manifests differ from the installed packs. Resume also fails
before model or tool execution when the persisted generation is not compatible
with the selected worker generation.

The execution boundary is exclusively typed. Providers and sessions remain
injected runtime ports, while request and result data are serializable Core
contracts. Product hosts project `RuntimeResult` into their own API, event and
persistence DTOs without passing those mappings back into Core.

```python
from harness_core.runtime_sdk import RuntimeRequest

result = runtime.run(
    RuntimeRequest(
        question="How many units remain?",
        user_id="user-1",
        context_bundle={"thread_id": "thread-1"},
    ),
    provider=model_provider,
)
print(result.status, result.final_answer.markdown)
```

Multi-tenant Hosts pass a `RuntimeScope` rather than encoding ownership into
run identifiers. The reference state adapter isolates tenant, namespace, and
user dimensions. Legacy `user_id` adapters remain supported for the default
scope and fail closed when asked to represent a tenant they cannot isolate.

```python
from harness_core.runtime_sdk import RuntimeRequest, RuntimeScope

request = RuntimeRequest(
    question="Inspect this account",
    scope=RuntimeScope(
        tenant_id="tenant-1",
        namespace="production",
        user_id="user-1",
    ),
)
```

Async Hosts use the same canonical runtime rather than a second execution
implementation. `arun()` moves the synchronous provider boundary off the event
loop, while `astream()` publishes typed events and cooperatively cancels the run
when its consumer disconnects.

```python
result = await runtime.arun(request, provider=provider)

async for event in runtime.astream(request, provider=provider):
    if event.event_type == "answer.delta":
        publish(event.payload["delta"])
```

Thread-safe synchronous persistence adapters can be exposed to async Host
control paths through the opt-in bridges in `harness_core.adapter_sdk`.
Production async database drivers should implement the corresponding
`Async*Store` protocol directly instead of using thread offload.

Reference extraction is also a Port. The default projector discovers generic
records and web sources without knowing tool names or business vocabulary.
Products may inject `RuntimePorts(reference_projector=...)` to map those records
to domain-specific source kinds, labels and evidence policies.

The builder derives installed contributions from the registrations that
actually occurred; a pack cannot claim an extension it did not install. Pack
installation is atomic across tools, skills, artifact types, handoffs, MCP
servers, sub-agents, context contributors and lifecycle-managed resources. Any
installation failure rolls back the partial contribution, including
replacement of an existing handler and reverse-order closure of newly opened
resources. `HarnessRuntime.close()` performs the same resource cleanup once.
Strict conformance is enabled by default during `build()` and can only be
disabled explicitly for diagnostic tooling.

Artifact type schemas are also executable contracts. A Builder-composed
runtime validates every tool-produced artifact before exposing it to the
model, persistence or frontend. Unknown artifact types and schema-invalid
payloads become failed tool results at the execution boundary.

Core context is a product-neutral `runtime_context` envelope. Product adapters
may populate generic subjects, facts, memories, recent messages and provenance
through `RuntimePorts(context_builder=...)`; Capability Packs can then enrich
that envelope with registered context contributors. Domain-specific field
names and filtering policies belong in the product adapter, not the Core.
Failures in either context stage are projected into the same terminal runtime
and event contracts as model or tool failures.

Context snapshots address domain subjects through `subject_id`. Domain keys and
legacy snapshot upgrades belong to the Host context adapter; incompatible
snapshot versions fail explicitly rather than being guessed by Core.

Before model execution, the injected `ContextWindowManager` converts recent
conversation history into role messages exactly once and applies a deterministic
input budget. The default manager reserves output capacity, limits history and
individual messages, then compacts generic context sections in stable priority
order. It performs no hidden model summarization and exposes only token counts,
dropped section names and truncation counts through diagnostics. Products can
replace the estimator or policy without changing the React loop.

Hosts may additionally provide `ContextSegment` entries for important sections.
A segment belongs to `runtime_constitution`, `turn_context`, `working_memory`,
or `retrieved_context`. It can declare retention priority, whether it is required, provenance,
a per-section ceiling and a precomputed summary. Under pressure, required
segments receive budget first and an eligible cached summary is preferred over
blind truncation. Core never generates or persists that summary itself, so its
content and invalidation policy remain explicit Host responsibilities.
`ContextSummaryCache` makes this lifecycle testable: every record is bound to a
Host-provided source fingerprint, and a source change becomes a cache miss
rather than silently reusing stale context. The default in-memory adapter is
process-local; durable Hosts may implement the same Port.

Capability Packs can publish a versioned `CapabilityManifest`. A build freezes
validated manifests into an immutable `RuntimeGeneration`, so every Run can
explain the exact package, tool catalog and Skill versions it used. Tool
definitions are disclosed progressively: a small baseline is always available,
Skill entry tools appear after activation, and discoverable tools are selected
through the replaceable `ToolDiscoveryPort`. Internal and Skill-only tools never
fail open into the model context.

Cross-cutting behavior is registered through scoped lifecycle Hooks rather
than embedded in business handlers. Hooks may enrich context, rewrite model or
tool inputs, deny an operation, or request user confirmation, but Policy and
Budget remain authoritative. Timeouts, failures, replay and audit records are
handled by Core. SubAgent delegation returns a compact parent projection with
conclusions, evidence, risks, recommendations and Artifact references instead
of copying the child trace into the parent prompt.

Model and tool execution share one `BudgetLedger`. Besides model/tool call
counts, Core accounts for estimated input and output tokens, retry attempts,
elapsed runtime and peak parallel tool workers. Sum aggregation is used for
consumable resources and max aggregation for concurrency. Every reservation is
idempotent by operation ID and survives checkpoint restore when backed by a
durable adapter. A zero or omitted limit remains unlimited for compatibility;
Hosts opt into hard limits through the model policy budget mapping.

Budget configuration is normalized through the public `BudgetPolicy` contract.
Global limits can be supplemented by model-role overrides such as `fast` or
`reasoning`. Before each provider request, Core performs a non-mutating
preflight, rejects oversized inputs, derives the remaining output allowance and
passes that allowance to compatible providers as `max_tokens`. Streaming also
enforces the allowance locally, so a provider that ignores it cannot publish
unbounded deltas.

After a request settles, `UsageReport` prefers provider-reported input and
output token counts. If the provider does not expose usage, Core falls back to
its deterministic estimator and marks the source accordingly. The resulting
usage, output ceiling and current budget metrics are attached to model routing
diagnostics and emitted as privacy-safe `budget.usage.recorded` events. Prompts
and response text are never included in this governance payload.

```python
from harness_core.adapter_sdk import BudgetPolicy

policy = BudgetPolicy.from_mapping({
    "max_input_tokens_total": 100_000,
    "max_output_tokens_total": 20_000,
    "max_model_retries": 2,
    "roles": {
        "fast": {
            "max_output_tokens_per_call": 512,
            "max_request_seconds": 20,
        },
        "reasoning": {
            "max_output_tokens_per_call": 4_096,
            "max_request_seconds": 120,
        },
    },
})
```

Checkpoint responsibilities are intentionally separate. `GraphCheckpointer`
stores engine execution state and `LangGraphCheckpointerAdapter` is its built-in
adapter. `DurableCheckpointStore` stores portable run snapshots used when an
engine checkpoint is unavailable. Capability Packs and product code do not
depend on LangGraph saver types directly.

Suspension persistence can additionally be supplied through
`RuntimeStateStore`. A single idempotent `RuntimeStateMutation` commits the
canonical run status, durable event and portable checkpoint together, with
optional version and sequence preconditions. This prevents a visible waiting
state from diverging from its resume checkpoint after a process crash. Product
database adapters own transactions and persistence models; Core only consumes
the Port and records the resulting receipt in recovery diagnostics. The same
mutation supports public terminal events, final-message correlation, failure
metadata and atomic removal of obsolete resumable checkpoints.

Distributed Hosts may additionally provide a `RunLeaseStore`. Each turn claims
exclusive ownership with a fencing token, renews it in the background and
releases it when the turn settles or suspends. Expired leases can be taken over,
while stale workers cannot renew or release a newer generation. The active
execution fence is propagated to every `ToolExecutionContext` and atomic
`RuntimeStateMutation`; durable adapters reject obsolete generations and write
tools can forward the same token to downstream storage. Persisted state upgrades
are explicit through `StateMigrationRegistry`; Core ships its own historical
v1-to-v2 migration chain and Hosts may register additional domain migrations.

`astream()` uses a bounded queue so a slow Host applies backpressure instead of
allowing unbounded event growth. Closing a stream requests cooperative run
cancellation and waits for a configurable acknowledgement timeout. Providers
and tools still use the canonical synchronous state machine, preserving one
execution implementation across sync and async Hosts.

The runtime also projects a stable `ui_state` contract. Completed, failed and
canceled runs release the composer; explicit user-input interruptions remain
sendable; tool handoffs and asynchronous work remain blocked until resumed.
Public errors contain a safe code, category and user message, while technical
details stay inside diagnostics and checkpoints.

Run the package-owned contract suite with:

```powershell
uv sync --extra test
uv run ruff check src tests verification
uv run mypy src/harness_core
uv run pytest -q --cov=harness_core --cov-fail-under=80
```

Build and verify both distributions from the repository root with:

```powershell
.\scripts\verify.ps1
```

The verifier installs the wheel and sdist into separate clean environments and
runs product-neutral scenarios for normal and streaming answers, tools,
parallelism, failures, interruption and recovery, asynchronous work,
cancellation, Skills, Artifacts, references, MCP and SubAgents.

Model integrations can implement `ModelProviderAdapter.invoke()` directly.
Existing providers exposing `stream_chat` or `complete_chat` remain supported
through `NativeChatProviderAdapter`. Runtime observability is injected with
`RuntimePorts(telemetry=...)`; telemetry records deliberately exclude prompts,
tool arguments and model results by default. Every run emits a terminal
`runtime.settled` record with status, stop reason, Skill identity and recovery
source, and diagnostics include only counts for the installed Capability Pack
inventory.

MCP servers use the same governed tool gateway for local stdio and remote
Streamable HTTP transports. Remote credentials are resolved through the
runtime `SecretProvider`, diagnostics expose only header names and sanitized
endpoints, and expired HTTP sessions are re-initialized once before surfacing a
transport failure.

```python
from harness_core.mcp_sdk import McpServerSpec

remote_search = McpServerSpec(
    id="remote-search",
    transport="streamable_http",
    url="https://mcp.example.com/search",
    secret_headers={"Authorization": "remote-search-token"},
    required_scopes=["search.read"],
)
```

The frozen public contract is `harness-core-v3`; the current package version is
`3.12.0`. Consumers import from `harness_core.runtime_sdk`,
`harness_core.extension_sdk`, or `harness_core.adapter_sdk`; optional bounded
orchestration and MCP adapters live in `harness_core.orchestration_sdk` and
`harness_core.mcp_sdk`. The versioned public
symbol manifest is available from `harness_core.public_api`; the package root
only exposes those SDK modules and version constants. The package-owned test
suite stores a frozen API fingerprint, so contract changes require an explicit
compatibility review.
