# Harness Agent Core

`kuitianjiandi-harness-core` is the product-neutral runtime kernel currently
validated by Kuitianjiandi. It owns the model/tool loop, runtime contracts,
governance ports, interruption, recovery, MCP gateway primitives, sub-agent
execution and Capability Pack composition.

The package does not contain Kuitianjiandi tools, database models, API routes or
product prompts. Consumers integrate through `HarnessRuntimeBuilder`,
`RuntimePorts` and a versioned Capability Pack. New packs should expose a
`CapabilityPackSpec` and implement `install(CapabilityInstallContext)`; the
original `register(ToolExecutor)` contract remains supported for v1
compatibility.

```python
from harness_core import (
    ArtifactTypeSpec,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPackSpec,
    HarnessRuntimeBuilder,
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

Before model execution, the injected `ContextWindowManager` converts recent
conversation history into role messages exactly once and applies a deterministic
input budget. The default manager reserves output capacity, limits history and
individual messages, then compacts generic context sections in stable priority
order. It performs no hidden model summarization and exposes only token counts,
dropped section names and truncation counts through diagnostics. Products can
replace the estimator or policy without changing the React loop.

Hosts may additionally provide `ContextSegment` entries for important sections.
A segment can declare retention priority, whether it is required, provenance,
a per-section ceiling and a precomputed summary. Under pressure, required
segments receive budget first and an eligible cached summary is preferred over
blind truncation. Core never generates or persists that summary itself, so its
content and invalidation policy remain explicit Host responsibilities.
`ContextSummaryCache` makes this lifecycle testable: every record is bound to a
Host-provided source fingerprint, and a source change becomes a cache miss
rather than silently reusing stale context. The default in-memory adapter is
process-local; durable Hosts may implement the same Port.

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
from harness_core import BudgetPolicy

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

The runtime also projects a stable `ui_state` contract. Completed, failed and
canceled runs release the composer; explicit user-input interruptions remain
sendable; tool handoffs and asynchronous work remain blocked until resumed.
Public errors contain a safe code, category and user message, while technical
details stay inside diagnostics and checkpoints.

Run the package-owned contract suite with:

```powershell
uv run --extra test pytest -q
```

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
from harness_core.mcp import McpServerSpec

remote_search = McpServerSpec(
    id="remote-search",
    transport="streamable_http",
    url="https://mcp.example.com/search",
    secret_headers={"Authorization": "remote-search-token"},
    required_scopes=["search.read"],
)
```

The frozen public contract is `harness-core-v1`; the current package version is
`1.2.0`.
