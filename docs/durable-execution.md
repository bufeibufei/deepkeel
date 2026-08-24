# Durable execution

[English](durable-execution.md) | [简体中文](durable-execution.zh-CN.md)

DeepKeel separates execution contracts from persistence implementations.
Runtime state, checkpoint, idempotency, invocation, task and event Ports can be
backed by PostgreSQL or another durable store without changing the loop.

Production adapters must provide optimistic concurrency, stable ownership,
lease or fence semantics where work can be concurrent, and idempotent settlement.
Run the exported adapter conformance suites against the real implementation.

Suspension is explicit. A tool returns a typed `PendingAction`; the Host persists
it and later resumes the same run with a typed observation. Cancellation is
cooperative and converges through the same settlement path as completion and
failure.

## Persistence authority

DeepKeel assigns one non-overlapping responsibility to each state mechanism:

| Mechanism | Authority | Must not decide |
| --- | --- | --- |
| `RuntimeStateStore` | Canonical product-visible run status, ordered event sequence, settlement, fence generation and latest portable checkpoint projection | LangGraph node scheduling details |
| `DurableCheckpointStore` | Portable recovery envelope and compatibility fallback when canonical runtime state cannot load an older run | Whether the Host unlocks input or displays a run as terminal |
| LangGraph checkpointer | Internal graph continuation for the current super-step and interrupt/resume cursor | Product-visible status, final settlement or cross-version recovery policy |

Resume reads `RuntimeStateStore` first, records canonical read failures, and
only then uses the durable compatibility store. Graph execution consults its
checkpointer later and can never override portable checkpoint authority.
