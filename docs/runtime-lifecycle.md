# Runtime lifecycle

Every request has a stable run identity and moves through one canonical runtime
loop. A run may complete, fail, be cancelled, wait for user input, wait for a
user action, or suspend for asynchronous work.

The runtime validates context and capability generations before model work,
records typed model and tool observations, settles exactly one terminal result,
and emits replayable lifecycle events. Resume uses the persisted run snapshot,
pending action and observations; it does not reconstruct progress from assistant
prose.

Hosts are responsible for durable implementations of state, checkpoints,
idempotency, model invocation settlement and event delivery. Reference
`InMemory*` adapters exist only for tests and single-process embedding.
