from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CheckpointAuthority(StrEnum):
    """Source used to rebuild portable runtime state for a resumed run."""

    NONE = "none"
    RUNTIME_STATE_STORE = "runtime_state_store"
    DURABLE_CHECKPOINT_STORE = "durable_checkpoint_store"
    SESSION_PROJECTION = "session_projection"


class CanonicalStateUnavailableError(RuntimeError):
    """Canonical state could not be read, so recovery must fail closed."""

    code = "CANONICAL_STATE_UNAVAILABLE"


class PersistenceResponsibility(StrEnum):
    """Non-overlapping responsibility assigned to each persistence mechanism."""

    PRODUCT_STATE = "product_state"
    PORTABLE_RECOVERY = "portable_recovery"
    GRAPH_CONTINUATION = "graph_continuation"


@dataclass(frozen=True, slots=True)
class PersistenceAuthorityContract:
    runtime_state_store: PersistenceResponsibility = PersistenceResponsibility.PRODUCT_STATE
    durable_checkpoint_store: PersistenceResponsibility = (
        PersistenceResponsibility.PORTABLE_RECOVERY
    )
    graph_checkpointer: PersistenceResponsibility = PersistenceResponsibility.GRAPH_CONTINUATION


PERSISTENCE_AUTHORITY = PersistenceAuthorityContract()
