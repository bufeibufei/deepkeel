"""Product-neutral PostgreSQL reference adapters for DeepKeel ports.

These adapters intentionally live outside ``src/deepkeel``. They are executable
contract examples, not a database dependency imposed on runtime consumers.
"""

from verification.postgres_reference.adapters import (
    PostgresDurableCheckpointStore,
    PostgresRunLeaseStore,
    PostgresRuntimeEventJournal,
    PostgresRuntimeStateStore,
)
from verification.postgres_reference.database import PostgresReferenceDatabase

__all__ = [
    "PostgresDurableCheckpointStore",
    "PostgresReferenceDatabase",
    "PostgresRunLeaseStore",
    "PostgresRuntimeEventJournal",
    "PostgresRuntimeStateStore",
]
