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

Each durable `TelemetryRecord` becomes a short internal span while retaining
the DeepKeel trace identity and runtime operation metadata. Ephemeral stream
deltas and user identifiers are excluded by default. Prompt text, tool
arguments, model output and arbitrary nested values are never promoted to span
attributes. Set `include_ephemeral=True` or `include_user_id=True` only under an
explicit sampling and privacy policy.

The adapter also records `deepkeel.runtime.events`,
`deepkeel.runtime.failures`, `deepkeel.operation.duration`, and
`deepkeel.runs.active`. Attribute projection rejects credential, prompt,
question, argument, content, result and token fields and drops oversized string
values. These metrics are operational signals, not a substitute for the
authoritative event journal or runtime state store.

Use `CompositeTelemetry` when PostgreSQL remains the authoritative diagnostic
trace and OpenTelemetry is an external projection. Export failure must not
change run semantics; alert on exporter health independently.
