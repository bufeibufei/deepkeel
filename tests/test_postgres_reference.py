from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import Iterator
from uuid import uuid4

import pytest

from deepkeel.cli import main as cli_main
from deepkeel.adapter_sdk import (
    RunLeaseConflict,
    verify_budget_ledger_contract,
    verify_capability_package_store_contract,
    verify_context_summary_cache_contract,
    verify_durable_checkpoint_store_contract,
    verify_model_invocation_store_contract,
    verify_run_lease_store_contract,
    verify_runtime_event_journal_contract,
    verify_runtime_state_store_contract,
    verify_tool_execution_store_contract,
    verify_trace_store_contract,
)
from deepkeel.failures import RunCanceledError
from deepkeel.events import envelope_runtime_event
from deepkeel.scope import RuntimeScope
from deepkeel.runtime_sdk import (
    HarnessRuntimeBuilder,
    RuntimeEventEnvelope,
    RuntimeStateConflict,
    RuntimeStateMutation,
)
from deepkeel.contrib.postgres import (
    PostgresBudgetLedger,
    PostgresCapabilityPackageStore,
    PostgresContextSummaryCache,
    PostgresDatabase,
    PostgresDurableCheckpointStore,
    PostgresModelHealthStore,
    PostgresModelInvocationStore,
    PostgresMemoryStore,
    PostgresRunControl,
    PostgresRunLeaseStore,
    PostgresRuntimeBundle,
    PostgresRuntimeEventJournal,
    PostgresRuntimeStateStore,
    PostgresSchemaDriftError,
    PostgresSchemaError,
    PostgresToolExecutionStore,
    PostgresTraceStore,
)
from deepkeel.memory_sdk import MemoryClaim, MemoryMutation, MemoryQuery
from examples.production_worker import build_production_worker


class _Checkpointer:
    pass


@pytest.fixture
def postgres_database() -> Iterator[PostgresDatabase]:
    pytest.importorskip("psycopg")
    dsn = str(os.environ.get("DEEPKEEL_TEST_POSTGRES_DSN") or "").strip()
    if not dsn:
        pytest.skip("DEEPKEEL_TEST_POSTGRES_DSN is not configured")
    database = PostgresDatabase(
        dsn,
        schema=f"deepkeel_test_{uuid4().hex[:12]}",
    )
    database.initialize()
    try:
        yield database
    finally:
        database.drop()


@pytest.mark.postgres
def test_postgres_reference_adapters_satisfy_core_contracts(
    postgres_database: PostgresDatabase,
) -> None:
    verify_runtime_state_store_contract(
        PostgresRuntimeStateStore(postgres_database),
        run_id=f"state-{uuid4().hex}",
        user_id="contract-user",
    )
    verify_runtime_event_journal_contract(
        PostgresRuntimeEventJournal(postgres_database),
        run_id=f"events-{uuid4().hex}",
    )
    verify_run_lease_store_contract(
        PostgresRunLeaseStore(postgres_database),
        run_id=f"lease-{uuid4().hex}",
    )
    verify_durable_checkpoint_store_contract(
        PostgresDurableCheckpointStore(postgres_database),
        run_id=f"checkpoint-{uuid4().hex}",
        user_id="contract-user",
    )
    verify_model_invocation_store_contract(
        PostgresModelInvocationStore(postgres_database),
        run_id=f"model-{uuid4().hex}",
    )
    verify_tool_execution_store_contract(
        PostgresToolExecutionStore(postgres_database),
        run_id=f"tool-{uuid4().hex}",
    )
    verify_budget_ledger_contract(
        PostgresBudgetLedger(postgres_database),
        run_id=f"budget-{uuid4().hex}",
    )
    verify_trace_store_contract(
        PostgresTraceStore(postgres_database),
        run_id=f"trace-{uuid4().hex}",
    )
    verify_context_summary_cache_contract(PostgresContextSummaryCache(postgres_database))
    verify_capability_package_store_contract(
        PostgresCapabilityPackageStore(postgres_database),
        package_id=f"postgres.package.{uuid4().hex}",
    )


@pytest.mark.postgres
def test_postgres_memory_store_is_idempotent_and_subject_scoped(
    postgres_database: PostgresDatabase,
) -> None:
    store = PostgresMemoryStore(postgres_database)
    mutation = MemoryMutation(
        action="create",
        claim=MemoryClaim(
            user_id="memory-user",
            subject_type="profile",
            subject_id="profile-a",
            domain="career",
            predicate="target_role",
            value="platform engineer",
        ),
        idempotency_key=f"memory-{uuid4().hex}",
    )

    first = store.apply(mutation)
    second = store.apply(mutation)
    page = store.search(
        MemoryQuery(
            user_id="memory-user",
            subject_type="profile",
            subject_id="profile-a",
            text="platform engineer",
        )
    )

    assert first == second
    assert first.applied is True
    assert store.get(first.claim_id) is not None
    assert [hit.claim.claim_id for hit in page.hits] == [first.claim_id]


@pytest.mark.postgres
def test_postgres_runtime_bundle_exposes_non_blocking_async_bridges(
    postgres_database: PostgresDatabase,
) -> None:
    bundle = PostgresRuntimeBundle.create(
        postgres_database.dsn,
        schema=postgres_database.schema,
        initialize=False,
    )
    ports = bundle.runtime_ports(
        checkpointer=_Checkpointer(),
        run_lease_owner_id="worker-1",
    )

    assert ports.async_checkpoint_store is not None
    assert ports.async_runtime_state_store is not None
    assert ports.async_event_journal is not None
    assert ports.async_run_lease_store is not None
    assert ports.async_tool_execution_store is not None
    assert bundle.capability_package_store is not None
    assert bundle.context_summary_cache is not None
    assert bundle.memory_store is not None
    assert bundle.subagent_store is not None


@pytest.mark.postgres
def test_postgres_events_and_leases_isolate_identical_run_ids_by_scope(
    postgres_database: PostgresDatabase,
) -> None:
    run_id = f"shared-{uuid4().hex}"
    first_scope = RuntimeScope(tenant_id="tenant-a", user_id="user-1")
    second_scope = RuntimeScope(tenant_id="tenant-b", user_id="user-1")
    journal = PostgresRuntimeEventJournal(postgres_database)
    leases = PostgresRunLeaseStore(postgres_database)

    def event_for(scope: RuntimeScope, *, sequence: int = 1) -> RuntimeEventEnvelope:
        return RuntimeEventEnvelope.model_validate(
            envelope_runtime_event(
                {
                    "event_type": "run.created",
                    "payload": {"tenant": scope.tenant_id},
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                run_id=run_id,
                thread_id="shared-thread",
                turn_id="shared-turn",
                sequence=sequence,
                run_version=1,
                scope=scope,
            )
        )

    first_event = journal.append_scoped(event_for(first_scope), scope=first_scope)
    second_event = journal.append_scoped(event_for(second_scope), scope=second_scope)
    shared_event_id = f"explicit-shared-{uuid4().hex}"
    first_shared = journal.append_scoped(
        event_for(first_scope, sequence=2).model_copy(update={"event_id": shared_event_id}),
        scope=first_scope,
    )
    second_shared = journal.append_scoped(
        event_for(second_scope, sequence=2).model_copy(update={"event_id": shared_event_id}),
        scope=second_scope,
    )
    first_lease = leases.claim_scoped(
        run_id,
        scope=first_scope,
        owner_id="worker-a",
        ttl_seconds=30,
    )
    second_lease = leases.claim_scoped(
        run_id,
        scope=second_scope,
        owner_id="worker-b",
        ttl_seconds=30,
    )

    assert first_event.event_id != second_event.event_id
    assert journal.read_after_scoped(run_id, scope=first_scope) == (first_event, first_shared)
    assert journal.read_after_scoped(run_id, scope=second_scope) == (second_event, second_shared)
    assert first_lease.scope == first_scope
    assert second_lease.scope == second_scope
    assert leases.inspect_scoped(run_id, scope=first_scope) == first_lease
    assert leases.inspect_scoped(run_id, scope=second_scope) == second_lease


@pytest.mark.postgres
def test_postgres_shared_health_and_cancellation(
    postgres_database: PostgresDatabase,
) -> None:
    health = PostgresModelHealthStore(
        postgres_database,
        failure_threshold=2,
        cooldown_seconds=60,
    )
    assert health.snapshot("provider", "model").is_available()
    assert health.record_failure("provider", "model", category="timeout").is_available()
    assert not health.record_failure("provider", "model", category="timeout").is_available()
    assert health.record_success("provider", "model").is_available()

    control = PostgresRunControl(postgres_database)
    run_id = f"cancel-{uuid4().hex}"
    control.raise_if_cancelled(run_id)
    control.cancel(run_id)
    with pytest.raises(RunCanceledError):
        control.raise_if_cancelled(run_id)
    control.release(run_id)
    control.raise_if_cancelled(run_id)


@pytest.mark.postgres
def test_postgres_bundle_satisfies_production_worker_gate(
    postgres_database: PostgresDatabase,
) -> None:
    bundle = PostgresRuntimeBundle.create(
        postgres_database.dsn,
        schema=postgres_database.schema,
        initialize=False,
    )
    ports = bundle.runtime_ports(
        checkpointer=object(),
        run_lease_owner_id="conformance-worker",
    )
    report = HarnessRuntimeBuilder(profile="production").with_ports(ports).production_readiness()
    assert report.ready, report.as_dict()


@pytest.mark.postgres
def test_public_production_worker_example_builds_against_postgres(
    postgres_database: PostgresDatabase,
) -> None:
    worker = build_production_worker(
        object(),
        dsn=postgres_database.dsn,
        schema=postgres_database.schema,
        worker_id="example-worker",
    )

    assert worker.schema_status.up_to_date is True
    assert worker.runtime.runtime_state_store is None
    assert worker.runtime.async_runtime_state_store is not None
    assert worker.runtime.async_runtime_state_store.store is worker.postgres.runtime_state_store
    assert worker.postgres.database.schema == postgres_database.schema


@pytest.mark.postgres
def test_postgres_cli_supports_explicit_staged_upgrade(
    postgres_database: PostgresDatabase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema = f"deepkeel_cli_{uuid4().hex[:12]}"
    variable = "DEEPKEEL_CLI_TEST_DSN"
    monkeypatch.setenv(variable, postgres_database.dsn)
    database = PostgresDatabase(postgres_database.dsn, schema=schema)
    try:
        assert (
            cli_main(
                [
                    "postgres",
                    "upgrade",
                    "--schema",
                    schema,
                    "--dsn-env",
                    variable,
                    "--target-version",
                    "1",
                    "--yes",
                ]
            )
            == 0
        )
        output = json.loads(capsys.readouterr().out)
        assert output["ok"] is True
        assert output["current_version"] == 1
        assert output["requested_target_version"] == 1
        assert output["up_to_date"] is False
        assert [item["version"] for item in output["requested_migrations"]] == [1]
    finally:
        database.drop()


@pytest.mark.postgres
def test_independent_workers_share_recovery_state_and_event_cursor(
    postgres_database: PostgresDatabase,
) -> None:
    run_id = f"recovery-{uuid4().hex}"
    worker_a_state = PostgresRuntimeStateStore(postgres_database)
    worker_b_state = PostgresRuntimeStateStore(postgres_database)
    worker_a_checkpoint = PostgresDurableCheckpointStore(postgres_database)
    worker_b_checkpoint = PostgresDurableCheckpointStore(postgres_database)
    worker_a_events = PostgresRuntimeEventJournal(postgres_database)
    worker_b_events = PostgresRuntimeEventJournal(postgres_database)

    receipt = worker_a_state.commit(
        RuntimeStateMutation(
            mutation_id=f"{run_id}:waiting",
            run_id=run_id,
            event_type="run.waiting",
            target_status="waiting_user_input",
            checkpoint_state={"phase": "waiting", "answer": None},
        ),
        user_id="recovery-user",
    )
    worker_a_checkpoint.save(
        run_id,
        {"schema_version": "harness-durable-checkpoint-v2", "phase": "waiting"},
        user_id="recovery-user",
    )
    event = RuntimeEventEnvelope(
        event_id=f"{run_id}:event:1",
        sequence=1,
        run_id=run_id,
        event_type="run.waiting",
        payload={"phase": "waiting"},
    )
    worker_a_events.append(event)

    recovered = worker_b_state.load_snapshot(run_id, user_id="recovery-user")
    assert recovered.version == receipt.version
    assert recovered.checkpoint_state == {"phase": "waiting", "answer": None}
    assert worker_b_checkpoint.load(run_id, user_id="recovery-user") == {
        "schema_version": "harness-durable-checkpoint-v2",
        "phase": "waiting",
    }
    assert worker_b_events.read_after(run_id) == (event,)


@pytest.mark.postgres
def test_multiworker_lease_and_optimistic_state_races_have_one_winner(
    postgres_database: PostgresDatabase,
) -> None:
    lease_run_id = f"lease-race-{uuid4().hex}"

    def claim(owner: str) -> tuple[str, object]:
        store = PostgresRunLeaseStore(postgres_database)
        try:
            return "won", store.claim(lease_run_id, owner_id=owner, ttl_seconds=30)
        except RunLeaseConflict as exc:
            return "lost", exc

    with ThreadPoolExecutor(max_workers=12) as pool:
        lease_results = list(pool.map(claim, (f"worker-{index}" for index in range(12))))
    winners = [result for outcome, result in lease_results if outcome == "won"]
    assert len(winners) == 1
    PostgresRunLeaseStore(postgres_database).release(winners[0])  # type: ignore[arg-type]

    state_run_id = f"state-race-{uuid4().hex}"
    first = PostgresRuntimeStateStore(postgres_database).commit(
        RuntimeStateMutation(
            mutation_id=f"{state_run_id}:initial",
            run_id=state_run_id,
            event_type="run.waiting",
            target_status="waiting_user_input",
        ),
        user_id="race-user",
    )

    def advance(index: int) -> str:
        try:
            PostgresRuntimeStateStore(postgres_database).commit(
                RuntimeStateMutation(
                    mutation_id=f"{state_run_id}:advance:{index}",
                    run_id=state_run_id,
                    event_type="run.resumed",
                    target_status="task_running",
                    expected_version=first.version,
                    expected_sequence=first.sequence,
                ),
                user_id="race-user",
            )
            return "won"
        except RuntimeStateConflict:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        state_results = list(pool.map(advance, range(2)))
    assert sorted(state_results) == ["lost", "won"]


@pytest.mark.postgres
def test_postgres_state_transaction_rolls_back_injected_crash(
    postgres_database: PostgresDatabase,
) -> None:
    run_id = f"rollback-{uuid4().hex}"

    def fail(stage: str) -> None:
        if stage == "after_state_update":
            raise RuntimeError("injected worker crash")

    store = PostgresRuntimeStateStore(
        postgres_database,
        failure_injector=fail,
    )
    with pytest.raises(RuntimeError, match="injected worker crash"):
        store.commit(
            RuntimeStateMutation(
                mutation_id=f"{run_id}:mutation",
                run_id=run_id,
                event_type="run.started",
                target_status="task_running",
            ),
            user_id="rollback-user",
        )
    snapshot = PostgresRuntimeStateStore(postgres_database).load_snapshot(
        run_id,
        user_id="rollback-user",
    )
    assert snapshot.version == 0
    assert snapshot.sequence == 0


@pytest.mark.postgres
def test_postgres_schema_registry_is_current_and_idempotent(
    postgres_database: PostgresDatabase,
) -> None:
    first = postgres_database.migration_status()
    postgres_database.initialize()
    second = postgres_database.migration_status()

    assert first.up_to_date is True
    assert first.current_version == first.target_version == 5
    assert [record.version for record in first.applied] == list(range(1, first.target_version + 1))
    assert first.pending == ()
    assert second == first
    assert postgres_database.migration_registry().plan() == ()


@pytest.mark.postgres
def test_postgres_schema_registry_upgrades_pre_cursor_trace_schema(
    postgres_database: PostgresDatabase,
) -> None:
    postgres_database.drop()
    schema = postgres_database.schema
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
            cursor.execute(
                f"""
                CREATE TABLE {schema}.runtime_traces (
                    telemetry_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL DEFAULT '',
                    thread_id TEXT NOT NULL DEFAULT '',
                    turn_id TEXT NOT NULL DEFAULT '',
                    tenant_id TEXT NOT NULL DEFAULT '',
                    user_id TEXT NOT NULL DEFAULT '',
                    namespace TEXT NOT NULL DEFAULT 'default',
                    trace_id TEXT NOT NULL DEFAULT '',
                    component TEXT NOT NULL DEFAULT '',
                    event_name TEXT NOT NULL DEFAULT '',
                    occurred_at TIMESTAMPTZ NOT NULL,
                    record JSONB NOT NULL
                )
                """
            )

    postgres_database.initialize()

    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = 'runtime_traces'
                """,
                (schema,),
            )
            columns = {str(row["column_name"]) for row in cursor.fetchall()}
    assert "sequence" in columns
    assert postgres_database.migration_status().up_to_date is True


@pytest.mark.postgres
def test_postgres_schema_registry_rejects_checksum_and_physical_drift(
    postgres_database: PostgresDatabase,
) -> None:
    schema = postgres_database.schema
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {schema}.schema_migrations SET checksum = 'tampered' WHERE version = 1"
            )
    with pytest.raises(PostgresSchemaDriftError, match="migration drift"):
        postgres_database.migration_status()

    postgres_database.reset()
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"ALTER TABLE {schema}.runtime_traces DROP COLUMN sequence")
    with pytest.raises(PostgresSchemaDriftError, match="missing required columns"):
        postgres_database.migration_status()


@pytest.mark.postgres
def test_postgres_schema_registry_serializes_concurrent_migrators(
    postgres_database: PostgresDatabase,
) -> None:
    postgres_database.drop()

    def initialize(_index: int) -> None:
        PostgresDatabase(
            postgres_database.dsn,
            schema=postgres_database.schema,
        ).initialize()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(initialize, range(8)))

    status = postgres_database.migration_status()
    assert status.up_to_date is True
    assert [record.version for record in status.applied] == list(
        range(1, status.target_version + 1)
    )


@pytest.mark.postgres
def test_postgres_schema_registry_never_downgrades_automatically(
    postgres_database: PostgresDatabase,
) -> None:
    with pytest.raises(PostgresSchemaError, match="downgrade"):
        postgres_database.migrate(target_version=1)
