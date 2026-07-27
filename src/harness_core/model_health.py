from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelHealthSnapshot:
    provider_id: str
    model_id: str
    consecutive_failures: int = 0
    opened_until: datetime | None = None
    last_failure_category: str = ""
    last_failure_at: datetime | None = None
    updated_at: datetime | None = None

    def is_available(self, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(UTC)
        return self.opened_until is None or self.opened_until <= current

    def as_dict(self, *, now: datetime | None = None) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "available": self.is_available(now=now),
            "consecutive_failures": self.consecutive_failures,
            "opened_until": (
                self.opened_until.isoformat() if self.opened_until is not None else ""
            ),
            "last_failure_category": self.last_failure_category,
            "last_failure_at": (
                self.last_failure_at.isoformat() if self.last_failure_at is not None else ""
            ),
        }


class ModelHealthStore(Protocol):
    """Shared health state for one concrete provider/model binding."""

    def snapshot(self, provider_id: str, model_id: str) -> ModelHealthSnapshot: ...

    def record_success(self, provider_id: str, model_id: str) -> ModelHealthSnapshot: ...

    def record_failure(
        self,
        provider_id: str,
        model_id: str,
        *,
        category: str,
        immediate: bool = False,
        retry_after_seconds: float = 0.0,
    ) -> ModelHealthSnapshot: ...


class InMemoryModelHealthStore:
    """Process-local default; hosts may provide a durable multi-worker adapter."""

    def __init__(
        self,
        *,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self._items: dict[tuple[str, str], ModelHealthSnapshot] = {}
        self._lock = Lock()

    def snapshot(self, provider_id: str, model_id: str) -> ModelHealthSnapshot:
        key = _health_key(provider_id, model_id)
        now = datetime.now(UTC)
        with self._lock:
            current = self._items.get(key) or ModelHealthSnapshot(*key)
            if current.opened_until is not None and current.opened_until <= now:
                current = replace(
                    current,
                    consecutive_failures=0,
                    opened_until=None,
                    updated_at=now,
                )
                self._items[key] = current
            return current

    def record_success(self, provider_id: str, model_id: str) -> ModelHealthSnapshot:
        key = _health_key(provider_id, model_id)
        current = ModelHealthSnapshot(*key, updated_at=datetime.now(UTC))
        with self._lock:
            self._items[key] = current
        return current

    def record_failure(
        self,
        provider_id: str,
        model_id: str,
        *,
        category: str,
        immediate: bool = False,
        retry_after_seconds: float = 0.0,
    ) -> ModelHealthSnapshot:
        key = _health_key(provider_id, model_id)
        now = datetime.now(UTC)
        with self._lock:
            previous = self._items.get(key) or ModelHealthSnapshot(*key)
            failures = max(0, previous.consecutive_failures) + 1
            should_open = immediate or failures >= self.failure_threshold
            cooldown = max(self.cooldown_seconds, float(retry_after_seconds or 0.0))
            current = ModelHealthSnapshot(
                provider_id=key[0],
                model_id=key[1],
                consecutive_failures=failures,
                opened_until=(now + timedelta(seconds=cooldown)) if should_open else None,
                last_failure_category=str(category or "provider_error"),
                last_failure_at=now,
                updated_at=now,
            )
            self._items[key] = current
            return current


def _health_key(provider_id: str, model_id: str) -> tuple[str, str]:
    return (
        str(provider_id or "unknown-provider").strip() or "unknown-provider",
        str(model_id or "unknown-model").strip() or "unknown-model",
    )
