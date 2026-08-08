from __future__ import annotations

from datetime import datetime
from typing import Any

from deepkeel.telemetry import TelemetryRecord, TracePage, TraceQuery

from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import json_value


class PostgresTraceStore:
    """Durable scoped telemetry store with deterministic paging and retention."""

    supports_runtime_scope = True

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def record(self, event: TelemetryRecord) -> None:
        payload = event.model_dump(mode="json")
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runtime_traces (
                        telemetry_id, run_id, thread_id, turn_id, tenant_id,
                        user_id, namespace, trace_id, component, event_name,
                        sequence, occurred_at, record
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    ) ON CONFLICT (telemetry_id) DO NOTHING
                    """,
                    (
                        event.telemetry_id,
                        event.run_id,
                        event.thread_id,
                        event.turn_id,
                        event.tenant_id,
                        event.user_id,
                        event.namespace,
                        event.trace_id,
                        event.component,
                        event.event_name,
                        event.sequence,
                        event.occurred_at,
                        json_value(payload),
                    ),
                )
                if cursor.rowcount == 1:
                    return
                cursor.execute(
                    f"SELECT record FROM {schema}.runtime_traces WHERE telemetry_id = %s",
                    (event.telemetry_id,),
                )
                existing = cursor.fetchone()
                if existing is None or existing["record"] != payload:
                    raise RuntimeError("telemetry_id cannot be reused with different content")

    def query(self, query: TraceQuery) -> TracePage:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column in (
            "run_id",
            "thread_id",
            "turn_id",
            "tenant_id",
            "user_id",
            "namespace",
            "trace_id",
            "component",
            "event_name",
        ):
            value = str(getattr(query, column) or "")
            if value:
                clauses.append(f"{column} = %s")
                parameters.append(value)
        if query.occurred_after is not None:
            clauses.append("occurred_at >= %s")
            parameters.append(query.occurred_after)
        if query.occurred_before is not None:
            clauses.append("occurred_at < %s")
            parameters.append(query.occurred_before)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(query.limit + 1)
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT record FROM {self.database.schema}.runtime_traces
                    {where}
                    ORDER BY occurred_at ASC, run_id ASC, sequence ASC, telemetry_id ASC
                    LIMIT %s
                    """,
                    tuple(parameters),
                )
                rows = cursor.fetchall()
        truncated = len(rows) > query.limit
        return TracePage(
            records=[TelemetryRecord.model_validate(row["record"]) for row in rows[: query.limit]],
            truncated=truncated,
        )

    def delete_before(self, cutoff: datetime, *, limit: int = 10_000) -> int:
        ceiling = max(1, int(limit))
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {schema}.runtime_traces
                    WHERE ctid IN (
                        SELECT ctid FROM {schema}.runtime_traces
                        WHERE occurred_at < %s
                        ORDER BY occurred_at ASC, run_id ASC, sequence ASC, telemetry_id ASC
                        LIMIT %s
                    )
                    """,
                    (cutoff, ceiling),
                )
                return int(cursor.rowcount)
