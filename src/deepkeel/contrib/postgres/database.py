from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from deepkeel.adapter_capabilities import AdapterCapabilities

if TYPE_CHECKING:
    from deepkeel.contrib.postgres.migrations import (
        PostgresMigration,
        PostgresSchemaRegistry,
        PostgresSchemaStatus,
    )


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class PostgresDatabase:
    """Owns an isolated, versioned schema used by the production adapters."""

    adapter_capabilities = AdapterCapabilities(
        durable=True,
        process_shared=True,
        runtime_scope=True,
        native_async=False,
        cancellation_safe=True,
        transactional=True,
        source="deepkeel_postgres",
    )

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

    def migration_registry(
        self,
        migrations: tuple[PostgresMigration, ...] | None = None,
    ) -> PostgresSchemaRegistry:
        from deepkeel.contrib.postgres.migrations import (
            PostgresSchemaRegistry,
            default_postgres_migrations,
        )

        return PostgresSchemaRegistry(
            self,
            migrations or default_postgres_migrations(self.schema),
        )

    def initialize(self) -> None:
        """Upgrade the adapter schema to the packaged latest version."""

        self.migration_registry().upgrade()

    def migration_status(self) -> PostgresSchemaStatus:
        return self.migration_registry().status()

    def migrate(self, *, target_version: int | None = None) -> PostgresSchemaStatus:
        return self.migration_registry().upgrade(target_version=target_version)

    def reset(self) -> None:
        self.drop()
        self.initialize()

    def drop(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")
