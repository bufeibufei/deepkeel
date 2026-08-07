from __future__ import annotations

import re
from typing import Any


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class PostgresReferenceDatabase:
    """Owns an isolated schema used by the product-neutral reference adapters."""

    def __init__(self, dsn: str, *, schema: str = "deepkeel_reference") -> None:
        normalized_dsn = str(dsn or "").strip().replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )
        if not normalized_dsn.startswith(("postgresql://", "postgres://")):
            raise ValueError("a PostgreSQL DSN is required")
        normalized_schema = str(schema or "").strip().lower()
        if not _IDENTIFIER_PATTERN.fullmatch(normalized_schema):
            raise ValueError("schema must be a safe lowercase PostgreSQL identifier")
        self.dsn = normalized_dsn
        self.schema = normalized_schema

    def connect(self) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - guarded by the postgres extra
            raise RuntimeError(
                "PostgreSQL verification requires `deepkeel[postgres]`"
            ) from exc
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def initialize(self) -> None:
        schema = self.schema
        statements = (
            f"CREATE SCHEMA IF NOT EXISTS {schema}",
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.runtime_states (
                tenant_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                user_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                version BIGINT NOT NULL DEFAULT 0,
                sequence BIGINT NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'preparing',
                settled BOOLEAN NOT NULL DEFAULT FALSE,
                settlement_status TEXT NOT NULL DEFAULT '',
                last_event_type TEXT NOT NULL DEFAULT '',
                checkpoint_type TEXT NOT NULL DEFAULT '',
                checkpoint_state JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                resume_token TEXT NOT NULL DEFAULT '',
                fence_token TEXT NOT NULL DEFAULT '',
                fence_generation BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, namespace, user_id, run_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.runtime_mutations (
                tenant_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                user_id TEXT NOT NULL,
                mutation_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                receipt JSONB NOT NULL,
                event_type TEXT NOT NULL,
                event_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                checkpoint_type TEXT NOT NULL DEFAULT '',
                checkpoint_state JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, namespace, user_id, mutation_id)
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS runtime_mutations_run_sequence_idx
            ON {schema}.runtime_mutations (tenant_id, namespace, user_id, run_id, created_at)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.runtime_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                fingerprint TEXT NOT NULL,
                envelope JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (run_id, sequence)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.run_leases (
                run_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL DEFAULT '',
                token TEXT NOT NULL DEFAULT '',
                acquired_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ,
                generation BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.durable_checkpoints (
                tenant_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                user_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                state JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, namespace, user_id, run_id)
            )
            """,
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def reset(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")
        self.initialize()

    def drop(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")
