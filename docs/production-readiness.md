# Production Readiness

DeepKeel owns execution semantics. A production Host owns durable
infrastructure, identity, deployment, and operator-facing APIs.

## Required Host adapters

Do not use the `InMemory*` implementations in a multi-worker deployment. Install
durable implementations for:

- `RuntimeStateStore`
- `DurableCheckpointStore`
- `RuntimeEventJournal`
- `ModelInvocationStore`
- `ToolExecutionStore`
- `RunLeaseStore`
- `BudgetLedger`
- `TraceStore`
- `ContextSummaryCache`
- `CancellableRunControl`
- `CapabilityPackageStore` when packages can be changed at runtime

Run every applicable verifier in `deepkeel.adapter_sdk` against the same
database configuration used in production. State and checkpoint adapters must
isolate identical run identifiers belonging to different user scopes.

Use `RuntimeScope` as the canonical tenant, namespace, and user boundary.
Legacy adapters may continue to receive `user_id` for the default scope, but
Core fails closed when a tenant or non-default namespace is used with an
adapter that does not implement the scoped extension. Trace adapters should
declare `supports_runtime_scope = True` only after they persist and filter all
three scope dimensions.

## Capability Package lifecycle

Persist package installation state outside process memory and update it with
optimistic concurrency. Package enable, disable, upgrade, rollback, and
uninstall operations must pass `CapabilityPackageManager` validation before a
new worker generation is activated.

Runs capture an immutable `RuntimeGeneration`. Do not mutate the Tool Registry
or Capability Catalog of a running generation. Start new work on the new
generation, allow old work to resume on its captured generation, and retire an
old generation only after no recoverable run references it.

Compose each worker with `HarnessRuntimeBuilder.with_runtime_generation()`.
Core verifies that the installed Capability Packs exactly match that generation
and rejects incompatible persisted generations before invoking a model or tool.
Run `verify_capability_package_store_contract` against the production catalog
adapter to prove optimistic concurrency, rollback, and generation replay.

## Operational control

`RunOperations` is the portable control-plane facade. It can inspect scoped
runs, enumerate runs when the state adapter implements
`QueryableRuntimeStateStore`, join persisted traces, classify recovery outcomes,
and request cooperative cancellation.

Hosts install a `RunRecoveryExecutor` to accept idempotent `resume`, `retry`,
`requeue`, and `terminalize` commands. `RunOperations` authorizes each command
through the scoped state projection before submission. Queue delivery,
scheduling, and remediation policy remain Host responsibilities.

Trace data is queried only after a user-scoped state lookup succeeds. Hosts must
still enforce tenant authorization before constructing `RunOperations` and
must not expose raw adapter sessions to untrusted callers.

## Recovery topology

Use one globally unique `run_id`, one durable state mutation stream, and one
lease owner at a time. A worker must stop committing as soon as its execution
fence is lost. Re-delivery is expected: model and tool stores must atomically
claim work and replay settled results.

Run recovery should follow this order:

1. Load the user-scoped canonical `RuntimeStateStore` projection.
2. Acquire or renew the run lease and execution fence.
3. Resume from the portable runtime checkpoint.
4. Replay settled model and tool invocations instead of executing them again.
5. Commit one terminal settlement and release control resources.

Use `RunOperations.inspect()` to expose recovery classification and trace
evidence to an operator. Bulk scheduling and repair policy belong to the Host.

`list_recovery_candidates()` finds non-terminal projections older than a
Host-selected cutoff. Production state adapters must populate `updated_at`
from the same atomic mutation transaction used for status and checkpoint data.

## Async adapters

Network-backed adapters should expose native asynchronous implementations.
`AsyncRuntimeStateStore`, `AsyncDurableCheckpointStore`,
`AsyncRuntimeEventJournal`, and `AsyncTraceStore` define the portable
contracts. The supplied `Async*Adapter` bridges use `asyncio.to_thread` and are
only suitable for thread-safe synchronous adapters; ORM sessions must not be
offloaded unless their driver explicitly permits cross-thread use.

## Evaluation

`EvalSuiteRunner` provides deterministic regression checks for runtime status,
tool selection, artifact contracts, error codes, step budgets, and trace event
ordering. It intentionally does not score business answer quality.

Business Capability Packs should ship their own `EvalCase` datasets. Semantic
or model-based graders such as DeepEval can consume the same `RuntimeResult`
and trace records above this deterministic gate.

## Failure injection

Before release, exercise at least:

- persistence unavailable during model or tool claim;
- worker loss after an external side effect and before settlement;
- stale lease owners attempting to commit;
- duplicate event, model, and tool delivery;
- parallel tools with partial failure;
- malformed persisted replay data;
- cancellation during model streaming and tool execution;
- recovery interrupted a second time.

The package test suite includes reference fault-injection tests, but each
production adapter needs equivalent integration tests against its real backend.

## Concurrency baseline

Run the package-owned benchmark before a release to verify that one shared,
precompiled runtime can safely serve independent turns:

```powershell
uv run python verification/concurrency_benchmark.py `
  --requests 300 `
  --workers 32 `
  --min-success-rate 1 `
  --max-p95-ms 2000
```

This benchmark isolates Core overhead with a deterministic local provider. It
is not a substitute for a Host load test using the production model gateway,
database pool, event journal and tool adapters. Record both results so model
latency is not mistaken for runtime contention.
