from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias


class RuntimeSession(Protocol):
    """Opaque business session passed through Core without owning its ORM type."""

    def close(self) -> None: ...


SessionFactory: TypeAlias = Callable[[], RuntimeSession]


class GraphCheckpointer(Protocol):
    """Engine-managed graph state; business recovery uses DurableCheckpointStore."""

    @property
    def compiler_checkpointer(self) -> Any: ...

    def has_checkpoint(self, thread_id: str) -> bool: ...

    def exists(self, thread_id: str) -> bool: ...

    def delete_thread(self, thread_id: str) -> None: ...


ContextBuilder: TypeAlias = Callable[
    [str, dict[str, Any] | None, dict[str, Any] | None],
    dict[str, Any],
]
