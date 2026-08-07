from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


class LangGraphCheckpointerAdapter:
    """Keeps LangGraph saver details behind the Harness checkpoint port."""

    def __init__(
        self,
        saver: Any | None = None,
        *,
        supports_async: bool = True,
    ) -> None:
        self.saver = saver or InMemorySaver()
        self.supports_async = bool(supports_async)

    @property
    def compiler_checkpointer(self) -> Any:
        return self.saver

    def has_checkpoint(self, thread_id: str) -> bool:
        get_tuple = getattr(self.saver, "get_tuple", None)
        if not callable(get_tuple):
            return False
        return get_tuple({"configurable": {"thread_id": thread_id}}) is not None

    def exists(self, thread_id: str) -> bool:
        return self.has_checkpoint(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        delete = getattr(self.saver, "delete_thread", None)
        if callable(delete):
            delete(thread_id)


def compiler_checkpointer(value: Any) -> Any:
    target = getattr(value, "compiler_checkpointer", None)
    return target if target is not None else value


def checkpointer_supports_async(value: Any) -> bool:
    """Return the Host-declared async capability without probing persistence."""

    declared = getattr(value, "supports_async", None)
    if declared is not None:
        return bool(declared)
    saver = compiler_checkpointer(value)
    async_write = getattr(type(saver), "aput_writes", None)
    return async_write is not BaseCheckpointSaver.aput_writes
