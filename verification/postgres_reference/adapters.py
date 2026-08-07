"""Compatibility facade for the executable PostgreSQL reference adapters."""

from verification.postgres_reference.checkpoint_store import (
    PostgresDurableCheckpointStore,
)
from verification.postgres_reference.event_journal import PostgresRuntimeEventJournal
from verification.postgres_reference.lease_store import PostgresRunLeaseStore
from verification.postgres_reference.state_store import PostgresRuntimeStateStore

__all__ = [
    "PostgresDurableCheckpointStore",
    "PostgresRunLeaseStore",
    "PostgresRuntimeEventJournal",
    "PostgresRuntimeStateStore",
]
