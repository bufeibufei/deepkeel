# Production Readiness

DeepKeel owns execution semantics. A production Host owns durable
infrastructure, identity, deployment, and operator-facing APIs.

## Required Host adapters

Use the executable builder gate before starting a production worker:

```python
builder = HarnessRuntimeBuilder(profile="production").with_ports(production_ports)
report = builder.production_readiness()
runtime = builder.build_production()
```

`build_production()` fails closed when worker-critical ports are absent,
ambiguous, or known to be process-local. Warnings identify synchronous adapters
that can block `arun()`. The gate cannot prove that an arbitrary custom adapter
is durable, so the conformance suites below remain mandatory.

Production composition also requires `tool_view_mode="enforced"`. Development
may retain legacy disclosure while migrating, but a production worker cannot
silently expose the complete allowed catalog to the model.

Do not use the `InMemory*` implementations in a multi-worker deployment. Install
durable implementations for:

- `RuntimeStateStore`
- `DurableCheckpointStore` when a Host needs the legacy portable-checkpoint
  compatibility fallback in addition to canonical `RuntimeStateStore` state
- `RuntimeEventJournal`
- `ModelInvocationStore`
- `ToolExecutionStore`
- `RunLeaseStore`
- `BudgetLedger`
- `TraceStore`
- `ContextSummaryCache`
- `CancellableRunControl`
- `CapabilityPackageStore` when packages can be changed at runtime

The worker gate directly checks the checkpointer, canonical state, event
journal, lease, model/tool idempotency, budget, model health, run control, and
telemetry ports. Trace, summary-cache, and package-catalog stores live in Host
control-plane composition and must be verified separately.

Run every applicable verifier in `deepkeel.adapter_sdk` against the same
database configuration used in production. State and checkpoint adapters must
isolate identical run identifiers belonging to different user scopes.

The optional `deepkeel[postgres]` integration packages transaction boundaries
and database constraints for canonical state, the runtime event journal, run
leases with fencing generations, and portable durable checkpoints. See
[PostgreSQL adapters](postgresql-reference.md) for the contract test and
multi-worker recovery baseline.

Run `deepkeel doctor` and `deepkeel postgres status` as deployment preflight.
Apply pending schema versions in a dedicated migration job with
`deepkeel postgres upgrade --yes` before workers receive traffic. The runnable
[`production_worker`](../examples/production_worker) example shows the complete
public-SDK composition, including an optional OpenTelemetry projection.

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

Core commits terminal state before deleting compatibility checkpoints. Cleanup
outcomes are appended as replay-only internal events and are not delivered to
the live event sink, preserving the public terminal-event boundary.

Use `RunOperations.inspect()` to expose recovery classification and trace
evidence to an operator. Bulk scheduling and repair policy belong to the Host.

`list_recovery_candidates()` finds non-terminal projections older than a
Host-selected cutoff. Production state adapters must populate `updated_at`
from the same atomic mutation transaction used for status and checkpoint data.

## Async adapters

Network-backed adapters should expose native asynchronous implementations.
`AsyncRuntimeStateStore`, `AsyncDurableCheckpointStore`,
`AsyncRuntimeEventJournal`, `AsyncRunLeaseStore`, `AsyncTraceStore`, and
`AsyncToolExecutionStore` define the portable infrastructure contracts. Memory
recall and bounded delegation likewise expose native async policy, storage,
executor, and dispatcher boundaries. The supplied `Async*Adapter` bridges use
`asyncio.to_thread` and are only suitable for thread-safe synchronous adapters;
ORM sessions must not be offloaded unless their driver explicitly permits
cross-thread use. The synchronous SubAgent bridge rejects a bound Session
unless the Host supplies a `session_factory`, so each worker owns its Session.

Native async state, event, checkpoint, lease, Memory, tool-idempotency, and
delegation ports are consumed directly by the canonical `arun()` path.
Configure either the synchronous or asynchronous form of one port, never both.
`astream()` adds a bounded same-loop backlog to its bounded queue and coalesces
consecutive answer deltas without losing text.

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

When PostgreSQL is available, run the database-backed baseline as a second,
independent gate:

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
uv sync --extra test --extra postgres
uv run pytest -q -m postgres tests/test_postgres_reference.py
uv run python -m verification.postgres_multiworker --requests 64 --workers 12
```
