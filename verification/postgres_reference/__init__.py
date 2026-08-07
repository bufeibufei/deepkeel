"""Compatibility imports for the packaged PostgreSQL adapters.

New integrations should import from :mod:`deepkeel.contrib.postgres`.
"""

from deepkeel.contrib.postgres import (
    PostgresDatabase,
    PostgresDurableCheckpointStore,
    PostgresRunLeaseStore,
    PostgresRuntimeEventJournal,
    PostgresRuntimeStateStore,
)

PostgresReferenceDatabase = PostgresDatabase

__all__ = [
    "PostgresDurableCheckpointStore",
    "PostgresReferenceDatabase",
    "PostgresRunLeaseStore",
    "PostgresRuntimeEventJournal",
    "PostgresRuntimeStateStore",
]
