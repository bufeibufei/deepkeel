from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Iterable

from deepkeel.runtime_sdk import (
    RunStateSnapshot,
    RuntimeScope,
    RuntimeStateConflict,
    RuntimeStateMutation,
    RuntimeStateReceipt,
)
from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import (
    advisory_lock,
    aggregate_from_row,
    assert_current_fence,
    assert_expected_state,
    fingerprint,
    json_value,
    snapshot_from_row,
)


class PostgresRuntimeStateStore:
    """Transactional runtime state projection with optimistic and fencing checks."""

    terminal_settlement_owner = "runtime"

    def __init__(
        self,
        database: PostgresDatabase,
        *,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.database = database
        self._failure_injector = failure_injector

    def commit(
        self,
        mutation: RuntimeStateMutation,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RuntimeStateReceipt:
        del session
        return self.commit_scoped(
            mutation,
            scope=RuntimeScope(user_id=str(user_id or "local-device")),
        )

    def commit_scoped(
        self,
        mutation: RuntimeStateMutation,
        *,
        scope: RuntimeScope,
        session: Any = None,
    ) -> RuntimeStateReceipt:
        del session
        tenant_id, namespace, user_id = scope.storage_key
        mutation_fingerprint = fingerprint(asdict(mutation))
        schema = self.database.schema
        lock_key = f"mutation:{tenant_id}:{namespace}:{user_id}:{mutation.mutation_id}"
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                advisory_lock(cursor, lock_key)
                replay = self._load_replay(
                    cursor,
                    scope=scope,
                    mutation=mutation,
                    mutation_fingerprint=mutation_fingerprint,
                )
                if replay is not None:
                    return replay
                row = self._lock_state(cursor, scope=scope, run_id=mutation.run_id)
                assert_expected_state(row, mutation)
                assert_current_fence(row, mutation)
                aggregate = aggregate_from_row(row)
                aggregate.apply(mutation, sequence=int(row["sequence"]) + 1)
                checkpoint_type = aggregate.checkpoint_type
                checkpoint_state = aggregate.checkpoint_state
                resume_token = aggregate.resume_token
                if checkpoint_type in set(mutation.delete_checkpoint_types):
                    checkpoint_type = ""
                    checkpoint_state = {}
                    resume_token = ""
                fence_token = str(row["fence_token"] or "")
                fence_generation = int(row["fence_generation"] or 0)
                if mutation.fence_generation:
                    fence_token = mutation.fence_token
                    fence_generation = mutation.fence_generation
                cursor.execute(
                    f"""
                    UPDATE {schema}.runtime_states
                    SET version = %s, sequence = %s, status = %s, settled = %s,
                        settlement_status = %s, last_event_type = %s,
                        checkpoint_type = %s, checkpoint_state = %s::jsonb,
                        resume_token = %s, fence_token = %s, fence_generation = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s AND version = %s
                    """,
                    (
                        aggregate.version,
                        aggregate.sequence,
                        aggregate.status,
                        aggregate.settled,
                        aggregate.settlement_status,
                        aggregate.last_event_type,
                        checkpoint_type,
                        json_value(checkpoint_state),
                        resume_token,
                        fence_token,
                        fence_generation,
                        tenant_id,
                        namespace,
                        user_id,
                        mutation.run_id,
                        row["version"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeStateConflict("runtime version changed during commit")
                self._fail_at("after_state_update")
                receipt = RuntimeStateReceipt(
                    mutation_id=mutation.mutation_id,
                    run_id=mutation.run_id,
                    version=aggregate.version,
                    sequence=aggregate.sequence,
                    status=aggregate.status,
                )
                self._save_mutation(
                    cursor,
                    scope=scope,
                    mutation=mutation,
                    mutation_fingerprint=mutation_fingerprint,
                    receipt=receipt,
                )
                self._fail_at("before_commit")
                return receipt

    def load_snapshot(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RunStateSnapshot:
        del session
        return self.load_snapshot_scoped(
            run_id,
            scope=RuntimeScope(user_id=str(user_id or "local-device")),
        )

    def load_snapshot_scoped(
        self,
        run_id: str,
        *,
        scope: RuntimeScope,
        session: Any = None,
    ) -> RunStateSnapshot:
        del session
        tenant_id, namespace, user_id = scope.storage_key
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT * FROM {schema}.runtime_states
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s
                    """,
                    (tenant_id, namespace, user_id, run_id),
                )
                row = cursor.fetchone()
        return snapshot_from_row(run_id, row)

    def list_snapshots(
        self,
        *,
        session: Any = None,
        user_id: str = "",
        statuses: Iterable[str] = (),
        limit: int = 100,
    ) -> tuple[RunStateSnapshot, ...]:
        del session
        return self.list_snapshots_scoped(
            scope=RuntimeScope(user_id=str(user_id or "local-device")),
            statuses=statuses,
            limit=limit,
        )

    def list_snapshots_scoped(
        self,
        *,
        scope: RuntimeScope,
        session: Any = None,
        statuses: Iterable[str] = (),
        limit: int = 100,
    ) -> tuple[RunStateSnapshot, ...]:
        del session
        tenant_id, namespace, user_id = scope.storage_key
        normalized = tuple(
            str(status or "").strip().lower()
            for status in statuses
            if str(status or "").strip()
        )
        ceiling = max(0, min(int(limit), 10_000))
        schema = self.database.schema
        query = f"""
            SELECT * FROM {schema}.runtime_states
            WHERE tenant_id = %s AND namespace = %s AND user_id = %s
        """
        parameters: list[Any] = [tenant_id, namespace, user_id]
        if normalized:
            query += " AND status = ANY(%s)"
            parameters.append(list(normalized))
        query += " ORDER BY sequence DESC, run_id DESC LIMIT %s"
        parameters.append(ceiling)
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                rows = cursor.fetchall()
        return tuple(snapshot_from_row(str(row["run_id"]), row) for row in rows)

    def _load_replay(
        self,
        cursor: Any,
        *,
        scope: RuntimeScope,
        mutation: RuntimeStateMutation,
        mutation_fingerprint: str,
    ) -> RuntimeStateReceipt | None:
        tenant_id, namespace, user_id = scope.storage_key
        cursor.execute(
            f"""
            SELECT fingerprint, receipt
            FROM {self.database.schema}.runtime_mutations
            WHERE tenant_id = %s AND namespace = %s AND user_id = %s
              AND mutation_id = %s
            """,
            (tenant_id, namespace, user_id, mutation.mutation_id),
        )
        replay = cursor.fetchone()
        if replay is None:
            return None
        if replay["fingerprint"] != mutation_fingerprint:
            raise RuntimeStateConflict("mutation_id cannot be reused with different content")
        return RuntimeStateReceipt(**{**replay["receipt"], "replayed": True})

    def _lock_state(
        self,
        cursor: Any,
        *,
        scope: RuntimeScope,
        run_id: str,
    ) -> dict[str, Any]:
        tenant_id, namespace, user_id = scope.storage_key
        schema = self.database.schema
        cursor.execute(
            f"""
            INSERT INTO {schema}.runtime_states (tenant_id, namespace, user_id, run_id)
            VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
            """,
            (tenant_id, namespace, user_id, run_id),
        )
        cursor.execute(
            f"""
            SELECT * FROM {schema}.runtime_states
            WHERE tenant_id = %s AND namespace = %s AND user_id = %s AND run_id = %s
            FOR UPDATE
            """,
            (tenant_id, namespace, user_id, run_id),
        )
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - guarded by the preceding insert
            raise RuntimeStateConflict("runtime state could not be locked")
        return row

    def _save_mutation(
        self,
        cursor: Any,
        *,
        scope: RuntimeScope,
        mutation: RuntimeStateMutation,
        mutation_fingerprint: str,
        receipt: RuntimeStateReceipt,
    ) -> None:
        tenant_id, namespace, user_id = scope.storage_key
        cursor.execute(
            f"""
            INSERT INTO {self.database.schema}.runtime_mutations (
                tenant_id, namespace, user_id, mutation_id, run_id,
                fingerprint, receipt, event_type, event_payload,
                checkpoint_type, checkpoint_state
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb)
            """,
            (
                tenant_id,
                namespace,
                user_id,
                mutation.mutation_id,
                mutation.run_id,
                mutation_fingerprint,
                json_value(receipt.as_dict()),
                mutation.event_type,
                json_value(mutation.event_payload),
                mutation.checkpoint_type,
                json_value(mutation.checkpoint_state),
            ),
        )

    def _fail_at(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)

