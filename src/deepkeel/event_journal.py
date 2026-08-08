from __future__ import annotations

from threading import Lock
from typing import Protocol

from deepkeel.runtime_api import RuntimeEventEnvelope
from deepkeel.scope import RuntimeScope


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

    def append_scoped(
        self,
        event: RuntimeEventEnvelope,
        *,
        scope: RuntimeScope,
    ) -> RuntimeEventEnvelope: ...

    def latest_sequence_scoped(self, run_id: str, *, scope: RuntimeScope) -> int: ...

    def read_after_scoped(
        self,
        run_id: str,
        *,
        scope: RuntimeScope,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]: ...


class InMemoryRuntimeEventJournal:
    """Thread-safe reference journal with idempotent append semantics."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str, str, str], list[RuntimeEventEnvelope]] = {}
        self._by_id: dict[tuple[str, str, str, str], RuntimeEventEnvelope] = {}
        self._lock = Lock()

    def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope:
        return self.append_scoped(event, scope=_event_scope(event))

    def append_scoped(
        self,
        event: RuntimeEventEnvelope,
        *,
        scope: RuntimeScope,
    ) -> RuntimeEventEnvelope:
        if not event.run_id or not event.event_id or event.sequence < 1:
            raise ValueError("journaled events require run_id, event_id, and sequence")
        _validate_event_scope(event, scope)
        candidate = event.model_copy(deep=True)
        run_key = (*scope.storage_key, candidate.run_id)
        event_key = (*scope.storage_key, candidate.event_id)
        with self._lock:
            existing = self._by_id.get(event_key)
            if existing is not None:
                if existing != candidate:
                    raise EventJournalConflict(
                        "event_id cannot be reused with different event content"
                    )
                return existing.model_copy(deep=True)
            events = self._events.setdefault(run_key, [])
            latest = events[-1].sequence if events else 0
            if candidate.sequence <= latest:
                raise EventJournalConflict(
                    f"event sequence must increase: latest {latest}, found {candidate.sequence}"
                )
            events.append(candidate)
            self._by_id[event_key] = candidate
            return candidate.model_copy(deep=True)

    def latest_sequence(self, run_id: str) -> int:
        with self._lock:
            events = self._unscoped_events(str(run_id or ""))
            return events[-1].sequence if events else 0

    def latest_sequence_scoped(self, run_id: str, *, scope: RuntimeScope) -> int:
        with self._lock:
            events = self._events.get((*scope.storage_key, str(run_id or "")), [])
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
                for event in self._unscoped_events(str(run_id or ""))
                if event.sequence > cursor
            ][:ceiling]
        return tuple(selected)

    def read_after_scoped(
        self,
        run_id: str,
        *,
        scope: RuntimeScope,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]:
        cursor = max(0, int(after_sequence))
        ceiling = max(1, min(int(limit), 1000))
        with self._lock:
            selected = [
                event.model_copy(deep=True)
                for event in self._events.get((*scope.storage_key, str(run_id or "")), [])
                if event.sequence > cursor
            ][:ceiling]
        return tuple(selected)

    def _unscoped_events(self, run_id: str) -> list[RuntimeEventEnvelope]:
        matches = [events for key, events in self._events.items() if key[-1] == run_id]
        if len(matches) > 1:
            raise EventJournalConflict(
                "runtime scope is required because run_id exists in multiple scopes"
            )
        return matches[0] if matches else []


def _event_scope(event: RuntimeEventEnvelope) -> RuntimeScope:
    return RuntimeScope(
        tenant_id=str(event.tenant_id or ""),
        user_id=str(event.user_id or "local-device"),
        namespace=str(event.namespace or "default"),
    )


def _validate_event_scope(event: RuntimeEventEnvelope, scope: RuntimeScope) -> None:
    if _event_scope(event).storage_key != scope.storage_key:
        raise ValueError("event ownership does not match the requested runtime scope")
