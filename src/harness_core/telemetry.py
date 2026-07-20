from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from harness_core.type_narrowing import as_dict


class TelemetryRecord(BaseModel):
    """Stable runtime telemetry envelope independent from product persistence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "harness-telemetry-v1"
    event_name: str
    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    step_index: int | None = None
    status: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_runtime_event(
        cls,
        event: dict[str, Any],
        *,
        run_id: str,
        thread_id: str,
        turn_id: str,
    ) -> "TelemetryRecord":
        payload = as_dict(event.get("payload"))
        raw_step = payload.get("step_index", event.get("step_index"))
        try:
            step_index = int(raw_step) if raw_step is not None else None
        except (TypeError, ValueError):
            step_index = None
        return cls(
            event_name=str(event.get("event_type") or "runtime.event"),
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            step_index=step_index,
            status=str(event.get("status") or payload.get("status") or ""),
            attributes={
                "source_event_type": str(event.get("source_event_type") or ""),
                **_safe_runtime_attributes(payload),
            },
        )


class TelemetryPort(Protocol):
    def record(self, event: TelemetryRecord) -> None: ...


class NoopTelemetry:
    def record(self, event: TelemetryRecord) -> None:
        del event


class InMemoryTelemetry:
    """Deterministic adapter for embedding, tests and local diagnostics."""

    def __init__(self) -> None:
        self._events: list[TelemetryRecord] = []
        self._lock = Lock()

    def record(self, event: TelemetryRecord) -> None:
        with self._lock:
            self._events.append(event.model_copy(deep=True))

    def snapshot(self) -> tuple[TelemetryRecord, ...]:
        with self._lock:
            return tuple(event.model_copy(deep=True) for event in self._events)


_SAFE_RUNTIME_ATTRIBUTE_KEYS = {
    "artifact_type",
    "attempt_index",
    "budget_metric",
    "budget_remaining",
    "checkpoint_source",
    "duration_ms",
    "error_code",
    "error_type",
    "failure_category",
    "failure_status_code",
    "finish_reason",
    "model_id",
    "model_role",
    "package_id",
    "phase",
    "provider_id",
    "retry_kind",
    "recovery_source",
    "router_id",
    "status",
    "skill_id",
    "stop_reason",
    "step_index",
    "tool_name",
    "tool_status",
}


def _safe_runtime_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep operational dimensions while excluding prompts, arguments and results."""

    return {
        key: value
        for key, value in payload.items()
        if key in _SAFE_RUNTIME_ATTRIBUTE_KEYS
        and isinstance(value, (str, int, float, bool, type(None)))
    }
