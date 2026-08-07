# PostgreSQL reference adapters

DeepKeel does not require PostgreSQL, an ORM, or a particular queue. Production
Hosts provide infrastructure through Ports. The code under
`verification/postgres_reference` is an executable, product-neutral reference
for the persistence semantics that are easiest to get wrong across workers.

## Covered boundaries

The reference implements and verifies:

- canonical `RuntimeStateStore` commits in one transaction, with idempotent
  mutation receipts, optimistic versions, user/tenant/namespace isolation, and
  execution-fence rejection;
- an append-only `RuntimeEventJournal` with stable event identity, monotonic
  per-run cursors, and exact replay;
- `RunLeaseStore` ownership based on database time, expiring leases, renewal,
  and monotonic fencing generations that survive release;
- `DurableCheckpointStore` recovery state with defensive JSON copies and
  ownership isolation.

Model invocation, tool execution, budget, health, trace, and package-catalog
stores remain separate Ports. A Host may use the same database, but must run
each exported conformance verifier independently rather than treating this
reference as a complete persistence product.

## Run the contracts

Install the optional verification dependency and point the test at a disposable
database. The suite creates and drops a unique schema; it never uses a product
table or migration.

```powershell
uv sync --extra test --extra postgres
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://user:password@localhost:5432/deepkeel_test"
uv run pytest -q -m postgres tests/test_postgres_reference.py
```

The tests instantiate independent adapter objects to represent different
workers. They prove one-winner lease and optimistic-state races, cross-worker
checkpoint/event recovery, scope isolation, and rollback after an injected
crash between state update and transaction commit.

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

Copy semantics, not imports, into a Host adapter. Keep business ORM models and
authorization outside Core, use `RuntimeScope` for ownership, use database time
for leases, and make every model/tool side effect independently idempotent.
Never commit with a stale fencing generation after a lease takeover.
