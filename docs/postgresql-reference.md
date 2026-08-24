# PostgreSQL adapters

[English](postgresql-reference.md) | [简体中文](postgresql-reference.zh-CN.md)

DeepKeel does not require PostgreSQL, an ORM, or a particular queue. Production
Hosts provide infrastructure through Ports. Install `deepkeel[postgres]` and
import the supported adapters from `deepkeel.contrib.postgres`. The adapters
implement the persistence semantics that are easiest to get wrong across
workers without imposing PostgreSQL on the Core runtime.

## Covered boundaries

The packaged integration implements and verifies:

- canonical `RuntimeStateStore` commits in one transaction, with idempotent
  mutation receipts, optimistic versions, user/tenant/namespace isolation, and
  execution-fence rejection;
- an append-only `RuntimeEventJournal` with stable event identity, monotonic
  per-run cursors, and exact replay;
- `RunLeaseStore` ownership based on database time, expiring leases, renewal,
  and monotonic fencing generations that survive release;
- `DurableCheckpointStore` recovery state with defensive JSON copies and
  ownership isolation.
- `ModelInvocationStore` and `ToolExecutionStore` ownership, settlement, and
  exact replay after worker redelivery;
- `BudgetLedger`, `ModelHealthStore`, and `CancellableRunControl` shared across
  worker processes;
- scoped `TraceStore` persistence, deterministic queries, and bounded
  retention cleanup.
- process-shared capability-package catalogs with optimistic revisions,
  fingerprint-bound context summaries, idempotent typed Memory claims, and
  durable SubAgent lineage/checkpoints/suspensions;
- a checksummed Schema Registry with advisory-lock serialization, migration
  planning/status, drift detection, and forward-only upgrades.

The LangGraph saver remains Host-owned because its lifecycle and deployment
topology vary independently from worker state. The packaged capability,
summary, Memory, and SubAgent stores are reference adapters: Hosts may replace
them with product-specific implementations without changing Core contracts.
A Host must run each exported conformance verifier rather than treating
successful connectivity as proof of runtime safety.

## Compose a worker

```python
from deepkeel.contrib.postgres import PostgresRuntimeBundle
from deepkeel.runtime_sdk import HarnessRuntimeBuilder

postgres = PostgresRuntimeBundle.create(
    "postgresql://deepkeel:secret@postgres/deepkeel",
    schema="deepkeel",
)
ports = postgres.runtime_ports(
    checkpointer=langgraph_postgres_saver,
    run_lease_owner_id="worker-01",
)
runtime = HarnessRuntimeBuilder(profile="production").with_ports(ports).build()

# Optional control-plane references exposed by the same bundle.
package_store = postgres.capability_package_store
summary_cache = postgres.context_summary_cache
memory = postgres.memory_store
subagents = postgres.subagent_store
```

`PostgresRuntimeBundle` never creates or owns the LangGraph saver. The Host
must initialize and close that dependency according to its process lifecycle.
`create(initialize=True)` upgrades the DeepKeel adapter schema before exposing
ports. It does not inspect or modify product-owned tables.

## Schema lifecycle

`PostgresDatabase.initialize()` remains the compatibility entry point and now
delegates to the packaged `PostgresSchemaRegistry`. Operators can inspect or
apply the same registry explicitly:

```python
from deepkeel.contrib.postgres import PostgresDatabase

database = PostgresDatabase(dsn, schema="deepkeel")
status = database.migration_status()
pending = database.migration_registry().plan()
upgraded = database.migrate()
```

Every applied version records its immutable name and checksum in
`schema_migrations`. A changed checksum, unknown future version, history gap,
missing runtime table, or missing required column fails closed as schema drift.
Concurrent workers serialize bootstrap and upgrade with a transaction-scoped
PostgreSQL advisory lock. Only forward migration is supported; rollback is an
operator-owned restore or a new compensating migration.

Use the packaged CLI in deployment automation. It reads the DSN from an
environment variable and never accepts credentials as a command argument:

```powershell
$env:DEEPKEEL_POSTGRES_DSN = "postgresql://..."
deepkeel postgres status
deepkeel postgres plan
deepkeel postgres upgrade --yes
```

`status` exits non-zero when the schema is behind. `plan` is read-only.
`upgrade` requires explicit confirmation and may target an intermediate version
with `--target-version`; success then means that requested version was reached,
while `up_to_date` still reports whether the schema is at the package latest.
Run `deepkeel doctor` before migration to inspect the runtime and installed
optional integrations.

## Run the contracts

Install the optional adapter dependency and point the test at a disposable
database. The suite creates and drops a unique schema; it never uses a product
table or migration.

```powershell
uv sync --extra test --extra postgres
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://user:password@localhost:5432/deepkeel_test"
uv run pytest -q -m postgres tests/test_postgres_reference.py
```

The tests instantiate independent adapter objects to represent different
workers. They prove one-winner lease and optimistic-state races, cross-worker
checkpoint/event recovery, scope isolation, rollback after an injected crash,
legacy-schema upgrade, drift rejection, migration idempotency, and concurrent
migrator serialization.

Run the repeatable recovery baseline separately:

```powershell
uv run python -m verification.postgres_multiworker `
  --requests 64 `
  --workers 12 `
  --max-p95-ms 1500
```

The reported p95 measures local database and adapter behavior, not model or
external tool latency. Production Hosts should retain this contract gate and
add a load test using their connection pool, queue delivery, model gateway, and
actual operational topology.

## Adoption rules

Import these adapters rather than copying them. Keep business ORM models and
authorization outside Core, use `RuntimeScope` for ownership, use database time
for leases, and make every model/tool side effect independently idempotent.
Never commit with a stale fencing generation after a lease takeover.
