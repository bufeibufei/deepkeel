from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any, Iterable, Protocol


_REQUIRED_COLUMNS = {
    "schema_migrations": {"version", "name", "checksum", "applied_at", "duration_ms"},
    "runtime_states": {
        "tenant_id",
        "namespace",
        "user_id",
        "run_id",
        "version",
        "sequence",
        "status",
        "settled",
        "settlement_status",
        "last_event_type",
        "checkpoint_type",
        "checkpoint_state",
        "resume_token",
        "fence_token",
        "fence_generation",
        "updated_at",
    },
    "runtime_mutations": {
        "tenant_id",
        "namespace",
        "user_id",
        "mutation_id",
        "run_id",
        "fingerprint",
        "receipt",
        "event_type",
        "event_payload",
        "checkpoint_type",
        "checkpoint_state",
        "created_at",
    },
    "runtime_events": {
        "tenant_id",
        "namespace",
        "user_id",
        "event_id",
        "run_id",
        "sequence",
        "fingerprint",
        "envelope",
        "created_at",
    },
    "run_leases": {
        "tenant_id",
        "namespace",
        "user_id",
        "run_id",
        "owner_id",
        "token",
        "acquired_at",
        "expires_at",
        "generation",
        "updated_at",
    },
    "durable_checkpoints": {
        "tenant_id",
        "namespace",
        "user_id",
        "run_id",
        "state",
        "updated_at",
    },
    "model_invocations": {
        "invocation_id",
        "request_fingerprint",
        "envelope",
        "status",
        "claim_token",
        "claim_expires_at",
        "result",
        "failure_type",
        "failure_message",
        "updated_at",
    },
    "tool_executions": {
        "tenant_id",
        "namespace",
        "user_id",
        "run_id",
        "idempotency_key",
        "record_id",
        "call_fingerprint",
        "tool_call",
        "status",
        "claim_owner",
        "lease_expires_at",
        "attempt_count",
        "result",
        "updated_at",
    },
    "budget_usage": {"run_id", "metric", "amount", "updated_at"},
    "budget_decisions": {"run_id", "metric", "operation_id", "decision", "created_at"},
    "model_health": {
        "provider_id",
        "model_id",
        "consecutive_failures",
        "opened_until",
        "last_failure_category",
        "last_failure_at",
        "updated_at",
    },
    "run_controls": {"run_id", "canceled", "updated_at"},
    "runtime_traces": {
        "telemetry_id",
        "run_id",
        "thread_id",
        "turn_id",
        "tenant_id",
        "user_id",
        "namespace",
        "trace_id",
        "component",
        "event_name",
        "sequence",
        "occurred_at",
        "record",
    },
    "capability_catalog": {"catalog_id", "revision", "snapshot", "updated_at"},
    "context_summaries": {
        "scope_digest",
        "cache_key",
        "source_fingerprint",
        "summary",
        "summary_version",
        "updated_at",
    },
    "memory_claims": {
        "claim_id",
        "tenant_id",
        "user_id",
        "subject_type",
        "subject_id",
        "profile_id",
        "domain",
        "predicate",
        "scope",
        "status",
        "sensitivity",
        "version",
        "payload",
        "updated_at",
    },
    "memory_mutations": {"idempotency_key", "mutation", "receipt", "created_at"},
    "subagent_runs": {
        "child_run_id",
        "root_run_id",
        "parent_run_id",
        "delegation_id",
        "user_id",
        "thread_id",
        "task",
        "spec",
        "phase",
        "checkpoint",
        "result",
        "suspension",
        "updated_at",
    },
}


class PostgresMigrationDatabase(Protocol):
    schema: str

    def connect(self) -> Any: ...


class PostgresSchemaError(RuntimeError):
    """Raised when a PostgreSQL schema cannot be migrated safely."""


class PostgresSchemaDriftError(PostgresSchemaError):
    """Raised when recorded migrations differ from the packaged registry."""


@dataclass(frozen=True, slots=True)
class PostgresMigration:
    version: int
    name: str
    statements: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("migration version must be positive")
        if not str(self.name or "").strip():
            raise ValueError("migration name is required")
        if not self.statements or any(not str(item or "").strip() for item in self.statements):
            raise ValueError("migration statements must not be empty")

    @property
    def checksum(self) -> str:
        payload = "\x00".join(
            (
                str(self.version),
                self.name.strip(),
                *(statement.strip() for statement in self.statements),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AppliedPostgresMigration:
    version: int
    name: str
    checksum: str
    applied_at: datetime
    duration_ms: int


@dataclass(frozen=True, slots=True)
class PostgresSchemaStatus:
    current_version: int
    target_version: int
    applied: tuple[AppliedPostgresMigration, ...]
    pending: tuple[PostgresMigration, ...]

    @property
    def up_to_date(self) -> bool:
        return self.current_version == self.target_version and not self.pending


class PostgresSchemaRegistry:
    """Forward-only, checksummed PostgreSQL migration registry."""

    def __init__(
        self,
        database: PostgresMigrationDatabase,
        migrations: Iterable[PostgresMigration],
    ) -> None:
        self.database = database
        self.migrations = tuple(sorted(migrations, key=lambda item: item.version))
        versions = tuple(migration.version for migration in self.migrations)
        expected = tuple(range(1, len(self.migrations) + 1))
        if versions != expected:
            raise ValueError("migration versions must be contiguous and start at 1")
        if len({migration.name for migration in self.migrations}) != len(self.migrations):
            raise ValueError("migration names must be unique")

    @property
    def latest_version(self) -> int:
        return self.migrations[-1].version if self.migrations else 0

    def status(self) -> PostgresSchemaStatus:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                self._lock_and_bootstrap(cursor)
                return self._status(cursor)

    def plan(self, *, target_version: int | None = None) -> tuple[PostgresMigration, ...]:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                self._lock_and_bootstrap(cursor)
                status = self._status(cursor)
                target = self._target_version(target_version, status.current_version)
                return tuple(
                    migration for migration in status.pending if migration.version <= target
                )

    def upgrade(self, *, target_version: int | None = None) -> PostgresSchemaStatus:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                self._lock_and_bootstrap(cursor)
                status = self._status(cursor)
                target = self._target_version(target_version, status.current_version)
                for migration in status.pending:
                    if migration.version > target:
                        break
                    self._apply(cursor, migration)
                return self._status(cursor)

    def _target_version(self, requested: int | None, current: int) -> int:
        target = self.latest_version if requested is None else int(requested)
        if target < current:
            raise PostgresSchemaError(
                f"automatic schema downgrade is not supported (current={current}, target={target})"
            )
        if target < 0 or target > self.latest_version:
            raise ValueError(f"target_version must be between 0 and {self.latest_version}")
        return target

    def _lock_and_bootstrap(self, cursor: Any) -> None:
        lock_name = f"deepkeel-schema:{self.database.schema}"
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_name,))
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database.schema}")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.database.schema}.schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                duration_ms BIGINT NOT NULL DEFAULT 0
            )
            """
        )

    def _status(self, cursor: Any) -> PostgresSchemaStatus:
        cursor.execute(
            f"""
            SELECT version, name, checksum, applied_at, duration_ms
            FROM {self.database.schema}.schema_migrations
            ORDER BY version
            """
        )
        applied = tuple(
            AppliedPostgresMigration(
                version=int(row["version"]),
                name=str(row["name"]),
                checksum=str(row["checksum"]),
                applied_at=row["applied_at"],
                duration_ms=int(row["duration_ms"]),
            )
            for row in cursor.fetchall()
        )
        registry = {migration.version: migration for migration in self.migrations}
        for record in applied:
            migration = registry.get(record.version)
            if migration is None:
                raise PostgresSchemaDriftError(
                    f"database schema version {record.version} is newer than this package"
                )
            if record.name != migration.name or record.checksum != migration.checksum:
                raise PostgresSchemaDriftError(
                    f"migration drift detected at version {record.version}"
                )
        current = applied[-1].version if applied else 0
        recorded_versions = {record.version for record in applied}
        expected_versions = set(range(1, current + 1))
        if recorded_versions != expected_versions:
            raise PostgresSchemaDriftError("database migration history contains a version gap")
        pending = tuple(migration for migration in self.migrations if migration.version > current)
        if current == self.latest_version:
            self._validate_physical_schema(cursor)
        return PostgresSchemaStatus(
            current_version=current,
            target_version=self.latest_version,
            applied=applied,
            pending=pending,
        )

    def _validate_physical_schema(self, cursor: Any) -> None:
        cursor.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s
            """,
            (self.database.schema,),
        )
        available: dict[str, set[str]] = {}
        for row in cursor.fetchall():
            available.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))
        defects = []
        for table, required in _REQUIRED_COLUMNS.items():
            missing = sorted(required - available.get(table, set()))
            if missing:
                defects.append(f"{table}: {', '.join(missing)}")
        if defects:
            raise PostgresSchemaDriftError(
                "database schema is missing required columns: " + "; ".join(defects)
            )

    def _apply(self, cursor: Any, migration: PostgresMigration) -> None:
        cursor.execute("SELECT clock_timestamp() AS started_at")
        started_at = cursor.fetchone()["started_at"]
        for statement in migration.statements:
            cursor.execute(statement)
        cursor.execute(
            "SELECT GREATEST(0, EXTRACT(EPOCH FROM (clock_timestamp() - %s)) * 1000) "
            "AS duration_ms",
            (started_at,),
        )
        duration_ms = int(cursor.fetchone()["duration_ms"])
        cursor.execute(
            f"""
            INSERT INTO {self.database.schema}.schema_migrations (
                version, name, checksum, duration_ms
            ) VALUES (%s, %s, %s, %s)
            """,
            (migration.version, migration.name, migration.checksum, duration_ms),
        )


def default_postgres_migrations(schema: str) -> tuple[PostgresMigration, ...]:
    return (
        PostgresMigration(
            version=1,
            name="runtime_port_schema",
            statements=_runtime_port_schema(schema),
        ),
        PostgresMigration(
            version=2,
            name="trace_sequence_cursor",
            statements=(
                f"""
                ALTER TABLE {schema}.runtime_traces
                ADD COLUMN IF NOT EXISTS sequence BIGINT NOT NULL DEFAULT 0
                """,
                f"""
                CREATE INDEX IF NOT EXISTS runtime_traces_run_time_idx
                ON {schema}.runtime_traces (run_id, occurred_at, sequence, telemetry_id)
                """,
            ),
        ),
        PostgresMigration(
            version=3,
            name="operational_scope_identity",
            statements=(
                f"""
                ALTER TABLE {schema}.runtime_events
                ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '',
                ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default',
                ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'local-device'
                """,
                f"""
                ALTER TABLE {schema}.runtime_events
                DROP CONSTRAINT IF EXISTS runtime_events_run_id_sequence_key
                """,
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS runtime_events_scope_run_sequence_uidx
                ON {schema}.runtime_events (
                    tenant_id, namespace, user_id, run_id, sequence
                )
                """,
                f"""
                ALTER TABLE {schema}.run_leases
                ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '',
                ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default',
                ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'local-device'
                """,
                f"""
                ALTER TABLE {schema}.run_leases
                DROP CONSTRAINT IF EXISTS run_leases_pkey
                """,
                f"""
                ALTER TABLE {schema}.run_leases
                ADD CONSTRAINT run_leases_pkey
                PRIMARY KEY (tenant_id, namespace, user_id, run_id)
                """,
                f"""
                ALTER TABLE {schema}.tool_executions
                ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '',
                ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default',
                ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'local-device'
                """,
                f"""
                ALTER TABLE {schema}.tool_executions
                DROP CONSTRAINT IF EXISTS tool_executions_pkey
                """,
                f"""
                ALTER TABLE {schema}.tool_executions
                ADD CONSTRAINT tool_executions_pkey
                PRIMARY KEY (
                    tenant_id, namespace, user_id, run_id, idempotency_key
                )
                """,
            ),
        ),
        PostgresMigration(
            version=4,
            name="runtime_event_scope_identity",
            statements=(
                f"""
                ALTER TABLE {schema}.runtime_events
                DROP CONSTRAINT IF EXISTS runtime_events_pkey
                """,
                f"""
                ALTER TABLE {schema}.runtime_events
                ADD CONSTRAINT runtime_events_pkey
                PRIMARY KEY (tenant_id, namespace, user_id, event_id)
                """,
            ),
        ),
        PostgresMigration(
            version=5,
            name="control_plane_reference_stores",
            statements=_control_plane_schema(schema),
        ),
        PostgresMigration(
            version=6,
            name="context_summary_scope_isolation",
            statements=(
                f"DROP TABLE IF EXISTS {schema}.context_summaries",
                f"""
                CREATE TABLE {schema}.context_summaries (
                    scope_digest TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    summary JSONB NOT NULL,
                    summary_version TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (scope_digest, cache_key)
                )
                """,
            ),
        ),
    )


def _control_plane_schema(schema: str) -> tuple[str, ...]:
    return (
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.capability_catalog (
            catalog_id TEXT PRIMARY KEY,
            revision BIGINT NOT NULL DEFAULT 0,
            snapshot JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.context_summaries (
            scope_digest TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            summary JSONB NOT NULL,
            summary_version TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (scope_digest, cache_key)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.memory_claims (
            claim_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL,
            subject_type TEXT NOT NULL DEFAULT 'user',
            subject_id TEXT NOT NULL DEFAULT '',
            profile_id TEXT NOT NULL DEFAULT '',
            domain TEXT NOT NULL DEFAULT 'general',
            predicate TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'global',
            status TEXT NOT NULL DEFAULT 'active',
            sensitivity TEXT NOT NULL DEFAULT 'normal',
            version BIGINT NOT NULL DEFAULT 1,
            payload JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE INDEX IF NOT EXISTS memory_claims_subject_idx
        ON {schema}.memory_claims (
            tenant_id, user_id, subject_type, subject_id, status, updated_at DESC
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.memory_mutations (
            idempotency_key TEXT PRIMARY KEY,
            mutation JSONB NOT NULL,
            receipt JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.subagent_runs (
            child_run_id TEXT PRIMARY KEY,
            root_run_id TEXT NOT NULL DEFAULT '',
            parent_run_id TEXT NOT NULL DEFAULT '',
            delegation_id TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL DEFAULT '',
            thread_id TEXT NOT NULL DEFAULT '',
            task JSONB NOT NULL,
            spec JSONB NOT NULL,
            phase TEXT NOT NULL DEFAULT 'created',
            checkpoint JSONB,
            result JSONB,
            suspension JSONB,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE INDEX IF NOT EXISTS subagent_runs_parent_idx
        ON {schema}.subagent_runs (parent_run_id, delegation_id, updated_at DESC)
        """,
    )


def _runtime_port_schema(schema: str) -> tuple[str, ...]:
    return (
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
            tenant_id TEXT NOT NULL DEFAULT '',
            namespace TEXT NOT NULL DEFAULT 'default',
            user_id TEXT NOT NULL DEFAULT 'local-device',
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            sequence BIGINT NOT NULL,
            fingerprint TEXT NOT NULL,
            envelope JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (tenant_id, namespace, user_id, run_id, sequence)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.run_leases (
            tenant_id TEXT NOT NULL DEFAULT '',
            namespace TEXT NOT NULL DEFAULT 'default',
            user_id TEXT NOT NULL DEFAULT 'local-device',
            run_id TEXT NOT NULL,
            owner_id TEXT NOT NULL DEFAULT '',
            token TEXT NOT NULL DEFAULT '',
            acquired_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            generation BIGINT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (tenant_id, namespace, user_id, run_id)
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
            tenant_id TEXT NOT NULL DEFAULT '',
            namespace TEXT NOT NULL DEFAULT 'default',
            user_id TEXT NOT NULL DEFAULT 'local-device',
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
            PRIMARY KEY (tenant_id, namespace, user_id, run_id, idempotency_key)
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
            occurred_at TIMESTAMPTZ NOT NULL,
            record JSONB NOT NULL
        )
        """,
        f"""
        CREATE INDEX IF NOT EXISTS runtime_traces_scope_time_idx
        ON {schema}.runtime_traces (
            tenant_id, namespace, user_id, occurred_at, telemetry_id
        )
        """,
    )
