# Runtime lifecycle

[English](runtime-lifecycle.md) | [简体中文](runtime-lifecycle.zh-CN.md)

Every request has a stable run identity and moves through one canonical runtime
loop. A run may complete, fail, be cancelled, wait for user input, wait for a
user action, or suspend for asynchronous work.

The runtime validates context and capability generations before model work,
records typed model and tool observations, settles exactly one terminal result,
and emits replayable lifecycle events. Resume uses the persisted run snapshot,
pending action and observations; it does not reconstruct progress from assistant
prose.

When execution planning is enabled, the model may create a bounded DAG through
the internal `runtime.create_plan` control tool. The plan is part of graph state
and the portable runtime checkpoint. Ready read-only steps may execute in a
bounded parallel batch; side-effecting, suspending, and unsafe steps execute
serially. User-action and asynchronous interruptions resume the same plan step,
including when a different worker restores the portable checkpoint.

Hosts are responsible for durable implementations of state, checkpoints,
idempotency, model invocation settlement and event delivery. Reference
`InMemory*` adapters exist only for tests and single-process embedding.
