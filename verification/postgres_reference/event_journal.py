from __future__ import annotations

from deepkeel.adapter_sdk import EventJournalConflict
from deepkeel.runtime_sdk import RuntimeEventEnvelope
from verification.postgres_reference.database import PostgresReferenceDatabase
from verification.postgres_reference.support import advisory_lock, fingerprint, json_value


class PostgresRuntimeEventJournal:
    """Append-only cursor journal safe for independent runtime workers."""

    def __init__(self, database: PostgresReferenceDatabase) -> None:
        self.database = database

    def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope:
        if not event.run_id or not event.event_id or event.sequence < 1:
            raise ValueError("journaled events require run_id, event_id, and sequence")
        payload = event.model_dump(mode="json")
        event_fingerprint = fingerprint(payload)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                advisory_lock(cursor, f"event:{event.event_id}")
                advisory_lock(cursor, f"event-run:{event.run_id}")
                cursor.execute(
                    f"SELECT fingerprint, envelope FROM {schema}.runtime_events WHERE event_id = %s",
                    (event.event_id,),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["fingerprint"] != event_fingerprint:
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
                    (
                        event.event_id,
                        event.run_id,
                        event.sequence,
                        event_fingerprint,
                        json_value(payload),
                    ),
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
                    ORDER BY sequence ASC LIMIT %s
                    """,
                    (str(run_id or ""), max(0, int(after_sequence)), ceiling),
                )
                rows = cursor.fetchall()
        return tuple(RuntimeEventEnvelope.model_validate(row["envelope"]) for row in rows)
