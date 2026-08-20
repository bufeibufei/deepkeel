# Observability

DeepKeel emits typed runtime events and structured diagnostics for model routes,
tool policies, budgets, context manifests, capability generations, recovery,
SubAgent lineage and terminal settlement.

Telemetry is injected through `RuntimePorts` and is fail-open by default. The
default records exclude prompts, tool arguments and model results. Hosts should
correlate records by run, thread, tenant and invocation identifiers, and retain
raw payloads only under an explicit data policy.

Evaluation consumes the same result and trace contracts as production. A test
failure should be attributable to runtime, adapter, Capability Package, model or
Host projection rather than inferred from user-visible prose.

## OpenTelemetry export

Install `deepkeel[otel]` when a Host needs standard tracing. DeepKeel provides
only the telemetry adapter; the Host owns the OpenTelemetry SDK, resource,
sampler, processor and exporter configuration.

```python
from deepkeel.contrib.otel import OpenTelemetryTelemetry

otel = OpenTelemetryTelemetry()
builder.configure_ports(telemetry=otel)
```

Durable records for one uninterrupted run segment are projected into one trace:
`deepkeel.run.segment` is the root, model/tool/MCP/SubAgent operations become
child spans, and lifecycle or routing decisions become root-span events. A
settled segment is closed immediately; a later resume starts a new trace linked
to the previous segment. Replayed records are deduplicated by event identity.
Ephemeral stream deltas are aggregated into the root-span delta count instead
of creating one span per token. User identifiers are excluded by default. Prompt text, tool
arguments, model output and arbitrary nested values are never promoted to span
attributes. Set `include_ephemeral=True` or `include_user_id=True` only under an
explicit sampling and privacy policy.

Call `otel.shutdown()` before shutting down the Host's SDK provider so any
unfinished segments are exported with `deepkeel.segment.closed_reason=shutdown`.

The adapter also records `deepkeel.runtime.events`,
`deepkeel.runtime.failures`, `deepkeel.operation.duration`, and
`deepkeel.runs.active`. Attribute projection rejects credential, prompt,
question, argument, content, result and token fields and drops oversized string
values. These metrics are operational signals, not a substitute for the
authoritative event journal or runtime state store.

Model spans follow the OpenTelemetry GenAI semantic attribute vocabulary for
provider, requested model, response model, operation name, token usage, finish
reason, and error type. DeepKeel keeps its own stable `deepkeel.*` attributes in
parallel so dashboards do not depend on an experimental semantic-convention
revision. Prompt and completion bodies are not exported by default.

Use `CompositeTelemetry` when PostgreSQL remains the authoritative diagnostic
trace and OpenTelemetry is an external projection. Export failure must not
change run semantics; alert on exporter health independently.

## Online evaluation

`OnlineEvalPort` receives a deterministic, privacy-bounded sample only after a
run settles. `OnlineEvalPolicy` controls status filters, sampling rate, and
whether answer content is omitted, hashed, or included. The reference
`OnlineEvalPipeline` is in-process and intended for development; production
Hosts should enqueue samples durably and evaluate them outside the request
path. Evaluation failures are telemetry signals and never rewrite the
authoritative `RuntimeResult`.
