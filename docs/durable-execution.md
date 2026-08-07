# Durable execution

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
