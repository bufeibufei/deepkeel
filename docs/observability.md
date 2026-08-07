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
