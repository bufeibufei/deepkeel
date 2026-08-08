from __future__ import annotations

from typing import Any

from deepkeel.adapter_sdk import EventJournalConflict
from deepkeel.runtime_sdk import RuntimeEventEnvelope
from deepkeel.scope import RuntimeScope
from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import advisory_lock, fingerprint, json_value


class PostgresRuntimeEventJournal:
    """Append-only cursor journal safe for independent runtime workers."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope:
        scope = RuntimeScope(
            tenant_id=str(event.tenant_id or ""),
            user_id=str(event.user_id or "local-device"),
            namespace=str(event.namespace or "default"),
        )
        return self.append_scoped(event, scope=scope)

    def append_scoped(
        self,
        event: RuntimeEventEnvelope,
        *,
        scope: RuntimeScope,
    ) -> RuntimeEventEnvelope:
        if not event.run_id or not event.event_id or event.sequence < 1:
            raise ValueError("journaled events require run_id, event_id, and sequence")
        event_scope = (
            str(event.tenant_id or ""),
            str(event.namespace or "default"),
            str(event.user_id or "local-device"),
        )
        if event_scope != scope.storage_key:
            raise ValueError("event ownership does not match the requested runtime scope")
        tenant_id, namespace, user_id = scope.storage_key
        payload = event.model_dump(mode="json")
        event_fingerprint = fingerprint(payload)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                advisory_lock(cursor, f"event:{tenant_id}:{namespace}:{user_id}:{event.event_id}")
                advisory_lock(cursor, f"event-run:{tenant_id}:{namespace}:{user_id}:{event.run_id}")
                cursor.execute(
                    f"SELECT fingerprint, envelope FROM {schema}.runtime_events "
                    "WHERE tenant_id = %s AND namespace = %s AND user_id = %s "
                    "AND event_id = %s",
                    (tenant_id, namespace, user_id, event.event_id),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["fingerprint"] != event_fingerprint:
                        raise EventJournalConflict(
                            "event_id cannot be reused with different event content"
                        )
                    return RuntimeEventEnvelope.model_validate(existing["envelope"])
                cursor.execute(
                    f"SELECT COALESCE(MAX(sequence), 0) AS latest FROM {schema}.runtime_events "
                    "WHERE tenant_id = %s AND namespace = %s AND user_id = %s AND run_id = %s",
                    (tenant_id, namespace, user_id, event.run_id),
                )
                latest = int(cursor.fetchone()["latest"])
                if event.sequence <= latest:
                    raise EventJournalConflict(
                        f"event sequence must increase: latest {latest}, found {event.sequence}"
                    )
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runtime_events (
                        tenant_id, namespace, user_id, event_id, run_id,
                        sequence, fingerprint, envelope
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        tenant_id,
                        namespace,
                        user_id,
                        event.event_id,
                        event.run_id,
                        event.sequence,
                        event_fingerprint,
                        json_value(payload),
                    ),
                )
        return event.model_copy(deep=True)

    def latest_sequence(self, run_id: str) -> int:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                scope = self._unique_scope(cursor, str(run_id or ""))
        return 0 if scope is None else self.latest_sequence_scoped(run_id, scope=scope)

    def latest_sequence_scoped(self, run_id: str, *, scope: RuntimeScope) -> int:
        tenant_id, namespace, user_id = scope.storage_key
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COALESCE(MAX(sequence), 0) AS latest FROM {schema}.runtime_events "
                    "WHERE tenant_id = %s AND namespace = %s AND user_id = %s AND run_id = %s",
                    (tenant_id, namespace, user_id, str(run_id or "")),
                )
                return int(cursor.fetchone()["latest"])

    def read_after(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                scope = self._unique_scope(cursor, str(run_id or ""))
        if scope is None:
            return ()
        return self.read_after_scoped(
            run_id,
            scope=scope,
            after_sequence=after_sequence,
            limit=limit,
        )

    def read_after_scoped(
        self,
        run_id: str,
        *,
        scope: RuntimeScope,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]:
        tenant_id, namespace, user_id = scope.storage_key
        schema = self.database.schema
        ceiling = max(1, min(int(limit), 1000))
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT envelope FROM {schema}.runtime_events
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s AND sequence > %s
                    ORDER BY sequence ASC LIMIT %s
                    """,
                    (
                        tenant_id,
                        namespace,
                        user_id,
                        str(run_id or ""),
                        max(0, int(after_sequence)),
                        ceiling,
                    ),
                )
                rows = cursor.fetchall()
        return tuple(RuntimeEventEnvelope.model_validate(row["envelope"]) for row in rows)

    def _unique_scope(self, cursor: Any, run_id: str) -> RuntimeScope | None:
        cursor.execute(
            f"SELECT DISTINCT tenant_id, namespace, user_id "
            f"FROM {self.database.schema}.runtime_events WHERE run_id = %s LIMIT 2",
            (run_id,),
        )
        rows = cursor.fetchall()
        if len(rows) > 1:
            raise EventJournalConflict(
                "runtime scope is required because run_id exists in multiple scopes"
            )
        if not rows:
            return None
        row = rows[0]
        return RuntimeScope(
            tenant_id=str(row["tenant_id"] or ""),
            namespace=str(row["namespace"] or "default"),
            user_id=str(row["user_id"] or "local-device"),
        )
