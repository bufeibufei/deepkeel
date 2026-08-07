from __future__ import annotations

from threading import Lock
from typing import Protocol

from deepkeel.runtime_api import RuntimeEventEnvelope


class EventJournalConflict(RuntimeError):
    """Raised when an event cursor or identity is reused inconsistently."""


class RuntimeEventJournal(Protocol):
    def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope: ...

    def latest_sequence(self, run_id: str) -> int: ...

    def read_after(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]: ...


class InMemoryRuntimeEventJournal:
    """Thread-safe reference journal with idempotent append semantics."""

    def __init__(self) -> None:
        self._events: dict[str, list[RuntimeEventEnvelope]] = {}
        self._by_id: dict[str, RuntimeEventEnvelope] = {}
        self._lock = Lock()

    def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope:
        if not event.run_id or not event.event_id or event.sequence < 1:
            raise ValueError("journaled events require run_id, event_id, and sequence")
        candidate = event.model_copy(deep=True)
        with self._lock:
            existing = self._by_id.get(candidate.event_id)
            if existing is not None:
                if existing != candidate:
                    raise EventJournalConflict(
                        "event_id cannot be reused with different event content"
                    )
                return existing.model_copy(deep=True)
            events = self._events.setdefault(candidate.run_id, [])
            latest = events[-1].sequence if events else 0
            if candidate.sequence <= latest:
                raise EventJournalConflict(
                    f"event sequence must increase: latest {latest}, found {candidate.sequence}"
                )
            events.append(candidate)
            self._by_id[candidate.event_id] = candidate
            return candidate.model_copy(deep=True)

    def latest_sequence(self, run_id: str) -> int:
        with self._lock:
            events = self._events.get(str(run_id or ""), [])
            return events[-1].sequence if events else 0

    def read_after(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]:
        cursor = max(0, int(after_sequence))
        ceiling = max(1, min(int(limit), 1000))
        with self._lock:
            selected = [
                event.model_copy(deep=True)
                for event in self._events.get(str(run_id or ""), [])
                if event.sequence > cursor
            ][:ceiling]
        return tuple(selected)
