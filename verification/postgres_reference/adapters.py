from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Callable, Iterable
from uuid import uuid4

from deepkeel.adapter_sdk import (
    EventJournalConflict,
    RunLease,
    RunLeaseConflict,
    RunLeaseLost,
)
from deepkeel.runtime_sdk import (
    RunAggregate,
    RunStateSnapshot,
    RuntimeEventEnvelope,
    RuntimeScope,
    RuntimeStateConflict,
    RuntimeStateMutation,
    RuntimeStateReceipt,
)
from verification.postgres_reference.database import PostgresReferenceDatabase


class PostgresRuntimeStateStore:
    """Transactional runtime state projection with optimistic and fencing checks."""

    terminal_settlement_owner = "runtime"

    def __init__(
        self,
        database: PostgresReferenceDatabase,
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
        fingerprint = _fingerprint(asdict(mutation))
        schema = self.database.schema
        lock_key = f"mutation:{tenant_id}:{namespace}:{user_id}:{mutation.mutation_id}"
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                _advisory_lock(cursor, lock_key)
                cursor.execute(
                    f"""
                    SELECT fingerprint, receipt
                    FROM {schema}.runtime_mutations
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND mutation_id = %s
                    """,
                    (tenant_id, namespace, user_id, mutation.mutation_id),
                )
                replay = cursor.fetchone()
                if replay is not None:
                    if replay["fingerprint"] != fingerprint:
                        raise RuntimeStateConflict(
                            "mutation_id cannot be reused with different content"
                        )
                    return RuntimeStateReceipt(
                        **{**replay["receipt"], "replayed": True}
                    )

                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runtime_states (
                        tenant_id, namespace, user_id, run_id
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (tenant_id, namespace, user_id, mutation.run_id),
                )
                cursor.execute(
                    f"""
                    SELECT * FROM {schema}.runtime_states
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s
                    FOR UPDATE
                    """,
                    (tenant_id, namespace, user_id, mutation.run_id),
                )
                row = cursor.fetchone()
                if row is None:  # pragma: no cover - guarded by the preceding insert
                    raise RuntimeStateConflict("runtime state could not be locked")
                _assert_expected_state(row, mutation)
                _assert_current_fence(row, mutation)
                aggregate = _aggregate_from_row(row)
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
                    SET version = %s,
                        sequence = %s,
                        status = %s,
                        settled = %s,
                        settlement_status = %s,
                        last_event_type = %s,
                        checkpoint_type = %s,
                        checkpoint_state = %s::jsonb,
                        resume_token = %s,
                        fence_token = %s,
                        fence_generation = %s,
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
                        _json(checkpoint_state),
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
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runtime_mutations (
                        tenant_id, namespace, user_id, mutation_id, run_id,
                        fingerprint, receipt, event_type, event_payload,
                        checkpoint_type, checkpoint_state
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb
                    )
                    """,
                    (
                        tenant_id,
                        namespace,
                        user_id,
                        mutation.mutation_id,
                        mutation.run_id,
                        fingerprint,
                        _json(receipt.as_dict()),
                        mutation.event_type,
                        _json(mutation.event_payload),
                        mutation.checkpoint_type,
                        _json(mutation.checkpoint_state),
                    ),
                )
                self._fail_at("before_commit")
                return receipt

    def _fail_at(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)

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
        return _snapshot_from_row(run_id, row)

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
        return tuple(_snapshot_from_row(str(row["run_id"]), row) for row in rows)


class PostgresRuntimeEventJournal:
    """Append-only cursor journal safe for independent runtime workers."""

    def __init__(self, database: PostgresReferenceDatabase) -> None:
        self.database = database

    def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope:
        if not event.run_id or not event.event_id or event.sequence < 1:
            raise ValueError("journaled events require run_id, event_id, and sequence")
        payload = event.model_dump(mode="json")
        fingerprint = _fingerprint(payload)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                _advisory_lock(cursor, f"event:{event.event_id}")
                _advisory_lock(cursor, f"event-run:{event.run_id}")
                cursor.execute(
                    f"SELECT fingerprint, envelope FROM {schema}.runtime_events WHERE event_id = %s",
                    (event.event_id,),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["fingerprint"] != fingerprint:
                        raise EventJournalConflict(
                            "event_id cannot be reused with different event content"
                        )
                    return RuntimeEventEnvelope.model_validate(existing["envelope"])
                cursor.execute(
                    f"SELECT COALESCE(MAX(sequence), 0) AS latest FROM {schema}.runtime_events WHERE run_id = %s",
                    (event.run_id,),
                )
                latest = int(cursor.fetchone()["latest"])
                if event.sequence <= latest:
                    raise EventJournalConflict(
                        f"event sequence must increase: latest {latest}, found {event.sequence}"
                    )
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runtime_events (
                        event_id, run_id, sequence, fingerprint, envelope
                    ) VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    (event.event_id, event.run_id, event.sequence, fingerprint, _json(payload)),
                )
        return event.model_copy(deep=True)

    def latest_sequence(self, run_id: str) -> int:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COALESCE(MAX(sequence), 0) AS latest FROM {schema}.runtime_events WHERE run_id = %s",
                    (str(run_id or ""),),
                )
                return int(cursor.fetchone()["latest"])

    def read_after(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]:
        schema = self.database.schema
        ceiling = max(1, min(int(limit), 1000))
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT envelope FROM {schema}.runtime_events
                    WHERE run_id = %s AND sequence > %s
                    ORDER BY sequence ASC
                    LIMIT %s
                    """,
                    (str(run_id or ""), max(0, int(after_sequence)), ceiling),
                )
                rows = cursor.fetchall()
        return tuple(RuntimeEventEnvelope.model_validate(row["envelope"]) for row in rows)


class PostgresRunLeaseStore:
    """Database-time lease ownership with monotonic fencing generations."""

    def __init__(self, database: PostgresReferenceDatabase) -> None:
        self.database = database

    def claim(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_seconds: float,
    ) -> RunLease:
        normalized_run_id = str(run_id or "").strip()
        normalized_owner = str(owner_id or "").strip()
        ttl = _ttl(ttl_seconds)
        if not normalized_run_id or not normalized_owner:
            raise ValueError("run_id and owner_id are required")
        token = uuid4().hex
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.run_leases (
                        run_id, owner_id, token, acquired_at, expires_at, generation
                    ) VALUES (
                        %s, %s, %s, CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'), 1
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        owner_id = EXCLUDED.owner_id,
                        token = EXCLUDED.token,
                        acquired_at = CURRENT_TIMESTAMP,
                        expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                        generation = {schema}.run_leases.generation + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE {schema}.run_leases.expires_at IS NULL
                       OR {schema}.run_leases.expires_at <= CURRENT_TIMESTAMP
                       OR {schema}.run_leases.owner_id = ''
                    RETURNING *
                    """,
                    (normalized_run_id, normalized_owner, token, ttl, ttl),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RunLeaseConflict(
                        f"run {normalized_run_id} already has a live owner"
                    )
        return _lease_from_row(row)

    def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        ttl = _ttl(ttl_seconds)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema}.run_leases
                    SET expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s AND owner_id = %s AND token = %s
                      AND generation = %s AND expires_at > CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (ttl, lease.run_id, lease.owner_id, lease.token, lease.generation),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RunLeaseLost(f"run lease was lost for {lease.run_id}")
        return _lease_from_row(row)

    def release(self, lease: RunLease) -> None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema}.run_leases
                    SET owner_id = '', token = '', expires_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s AND owner_id = %s AND token = %s AND generation = %s
                    """,
                    (lease.run_id, lease.owner_id, lease.token, lease.generation),
                )
                if cursor.rowcount == 1:
                    return
                cursor.execute(
                    f"SELECT run_id FROM {schema}.run_leases WHERE run_id = %s",
                    (lease.run_id,),
                )
                if cursor.fetchone() is not None:
                    raise RunLeaseLost(f"run lease was replaced for {lease.run_id}")

    def inspect(self, run_id: str) -> RunLease | None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT * FROM {schema}.run_leases
                    WHERE run_id = %s AND owner_id <> '' AND token <> ''
                      AND expires_at > CURRENT_TIMESTAMP
                    """,
                    (str(run_id or ""),),
                )
                row = cursor.fetchone()
        return _lease_from_row(row) if row is not None else None


class PostgresDurableCheckpointStore:
    """Portable recovery checkpoint store isolated by runtime ownership scope."""

    def __init__(
        self,
        database: PostgresReferenceDatabase,
        *,
        tenant_id: str = "",
        namespace: str = "default",
    ) -> None:
        self.database = database
        self.tenant_id = str(tenant_id or "")
        self.namespace = str(namespace or "default")

    def load(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> dict[str, Any] | None:
        del session
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT state FROM {schema}.durable_checkpoints
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s AND run_id = %s
                    """,
                    (self.tenant_id, self.namespace, str(user_id or ""), run_id),
                )
                row = cursor.fetchone()
        return copy.deepcopy(row["state"]) if row is not None else None

    def save(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None:
        del session
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.durable_checkpoints (
                        tenant_id, namespace, user_id, run_id, state
                    ) VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (tenant_id, namespace, user_id, run_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        self.tenant_id,
                        self.namespace,
                        str(user_id or ""),
                        run_id,
                        _json(state),
                    ),
                )

    def delete(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None:
        del session
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {schema}.durable_checkpoints
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s AND run_id = %s
                    """,
                    (self.tenant_id, self.namespace, str(user_id or ""), run_id),
                )

    def exists(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> bool:
        del session
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT 1 FROM {schema}.durable_checkpoints
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s AND run_id = %s
                    """,
                    (self.tenant_id, self.namespace, str(user_id or ""), run_id),
                )
                return cursor.fetchone() is not None

    def list_ids(
        self,
        *,
        session: Any = None,
        user_id: str = "",
        limit: int = 100,
    ) -> tuple[str, ...]:
        del session
        schema = self.database.schema
        ceiling = max(0, min(int(limit), 10_000))
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT run_id FROM {schema}.durable_checkpoints
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                    ORDER BY run_id ASC LIMIT %s
                    """,
                    (self.tenant_id, self.namespace, str(user_id or ""), ceiling),
                )
                rows = cursor.fetchall()
        return tuple(str(row["run_id"]) for row in rows)


def _aggregate_from_row(row: dict[str, Any]) -> RunAggregate:
    return RunAggregate(
        run_id=str(row["run_id"]),
        version=int(row["version"]),
        sequence=int(row["sequence"]),
        status=str(row["status"]),
        settled=bool(row["settled"]),
        settlement_status=str(row["settlement_status"]),
        last_event_type=str(row["last_event_type"]),
        checkpoint_type=str(row["checkpoint_type"]),
        checkpoint_state=copy.deepcopy(row["checkpoint_state"] or {}),
        resume_token=str(row["resume_token"]),
        fence_token=str(row["fence_token"]),
        fence_generation=int(row["fence_generation"]),
        updated_at=_aware(row["updated_at"]),
    )


def _snapshot_from_row(run_id: str, row: dict[str, Any] | None) -> RunStateSnapshot:
    if row is None:
        return RunStateSnapshot(run_id=run_id)
    return _aggregate_from_row(row).snapshot()


def _assert_expected_state(row: dict[str, Any], mutation: RuntimeStateMutation) -> None:
    if mutation.expected_version is not None and mutation.expected_version != int(
        row["version"]
    ):
        raise RuntimeStateConflict(
            f"runtime version changed: expected {mutation.expected_version}, found {row['version']}"
        )
    if mutation.expected_sequence is not None and mutation.expected_sequence != int(
        row["sequence"]
    ):
        raise RuntimeStateConflict(
            "runtime sequence changed: "
            f"expected {mutation.expected_sequence}, found {row['sequence']}"
        )


def _assert_current_fence(row: dict[str, Any], mutation: RuntimeStateMutation) -> None:
    generation = int(mutation.fence_generation or 0)
    token = str(mutation.fence_token or "")
    current_generation = int(row["fence_generation"] or 0)
    current_token = str(row["fence_token"] or "")
    if generation < 0:
        raise ValueError("fence_generation must be non-negative")
    if generation and not token:
        raise ValueError("fence_token is required when fence_generation is set")
    if current_generation and not generation:
        raise RuntimeStateConflict("fenced run requires an execution fence")
    if generation < current_generation:
        raise RuntimeStateConflict(
            f"stale execution fence: expected generation {current_generation}, found {generation}"
        )
    if generation == current_generation and generation and token != current_token:
        raise RuntimeStateConflict("execution fence token does not match current owner")


def _lease_from_row(row: dict[str, Any]) -> RunLease:
    return RunLease(
        run_id=str(row["run_id"]),
        owner_id=str(row["owner_id"]),
        token=str(row["token"]),
        acquired_at=_aware(row["acquired_at"]),
        expires_at=_aware(row["expires_at"]),
        generation=int(row["generation"]),
    )


def _advisory_lock(cursor: Any, value: str) -> None:
    cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (value,))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _aware(value).isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _ttl(value: float) -> float:
    ttl = float(value)
    if ttl <= 0:
        raise ValueError("ttl_seconds must be positive")
    return ttl
