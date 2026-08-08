from __future__ import annotations

import copy
from typing import Any

from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import json_value


class PostgresDurableCheckpointStore:
    """Portable recovery checkpoint store isolated by runtime ownership scope."""

    def __init__(
        self,
        database: PostgresDatabase,
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
                        json_value(state),
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

