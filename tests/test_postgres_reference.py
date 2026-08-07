from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Iterator
from uuid import uuid4

import pytest

from deepkeel.adapter_sdk import (
    RunLeaseConflict,
    verify_durable_checkpoint_store_contract,
    verify_run_lease_store_contract,
    verify_runtime_event_journal_contract,
    verify_runtime_state_store_contract,
)
from deepkeel.runtime_sdk import (
    RuntimeEventEnvelope,
    RuntimeStateConflict,
    RuntimeStateMutation,
)
from verification.postgres_reference import (
    PostgresDurableCheckpointStore,
    PostgresReferenceDatabase,
    PostgresRunLeaseStore,
    PostgresRuntimeEventJournal,
    PostgresRuntimeStateStore,
)


@pytest.fixture
def postgres_database() -> Iterator[PostgresReferenceDatabase]:
    pytest.importorskip("psycopg")
    dsn = str(os.environ.get("DEEPKEEL_TEST_POSTGRES_DSN") or "").strip()
    if not dsn:
        pytest.skip("DEEPKEEL_TEST_POSTGRES_DSN is not configured")
    database = PostgresReferenceDatabase(
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
    postgres_database: PostgresReferenceDatabase,
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


@pytest.mark.postgres
def test_independent_workers_share_recovery_state_and_event_cursor(
    postgres_database: PostgresReferenceDatabase,
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
    postgres_database: PostgresReferenceDatabase,
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
    postgres_database: PostgresReferenceDatabase,
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

