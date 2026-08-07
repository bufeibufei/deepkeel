from __future__ import annotations

import re
from typing import Any


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class PostgresDatabase:
    """Owns an isolated schema used by the production adapters."""

    def __init__(self, dsn: str, *, schema: str = "deepkeel") -> None:
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
                "PostgreSQL support requires `deepkeel[postgres]`"
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
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.model_invocations (
                invocation_id TEXT PRIMARY KEY,
                request_fingerprint TEXT NOT NULL,
                envelope JSONB NOT NULL,
                status TEXT NOT NULL,
                claim_token TEXT NOT NULL DEFAULT '',
                claim_expires_at TIMESTAMPTZ,
                result JSONB,
                failure_type TEXT NOT NULL DEFAULT '',
                failure_message TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.tool_executions (
                run_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                record_id TEXT NOT NULL UNIQUE,
                call_fingerprint TEXT NOT NULL,
                tool_call JSONB NOT NULL,
                status TEXT NOT NULL,
                claim_owner TEXT NOT NULL DEFAULT '',
                lease_expires_at TIMESTAMPTZ,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                result JSONB,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, idempotency_key)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.budget_usage (
                run_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                amount DOUBLE PRECISION NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, metric)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.budget_decisions (
                run_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                decision JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, metric, operation_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.model_health (
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                opened_until TIMESTAMPTZ,
                last_failure_category TEXT NOT NULL DEFAULT '',
                last_failure_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider_id, model_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.run_controls (
                run_id TEXT PRIMARY KEY,
                canceled BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.runtime_traces (
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
                sequence BIGINT NOT NULL DEFAULT 0,
                occurred_at TIMESTAMPTZ NOT NULL,
                record JSONB NOT NULL
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS runtime_traces_run_time_idx
            ON {schema}.runtime_traces (run_id, occurred_at, sequence, telemetry_id)
            """,
            f"""
            CREATE INDEX IF NOT EXISTS runtime_traces_scope_time_idx
            ON {schema}.runtime_traces (
                tenant_id, namespace, user_id, occurred_at, telemetry_id
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
