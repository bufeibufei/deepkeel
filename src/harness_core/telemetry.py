from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import logging
from threading import Lock
from typing import Any, Iterable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness_core.type_narrowing import as_dict


class TelemetryRecord(BaseModel):
    """Stable runtime telemetry envelope independent from product persistence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "harness-telemetry-v2"
    telemetry_id: str = ""
    event_id: str = ""
    sequence: int = 0
    run_version: int = 0
    event_name: str
    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    step_index: int | None = None
    status: str = ""
    component: str = "runtime"
    operation_id: str = ""
    parent_operation_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    ephemeral: bool = False
    privacy_class: Literal["operational_metadata"] = "operational_metadata"
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_identity(self) -> "TelemetryRecord":
        if not self.telemetry_id:
            identity = self.event_id or (
                f"{self.run_id}:{self.turn_id}:{self.sequence}:"
                f"{self.event_name}:{self.operation_id}"
            )
            self.telemetry_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if not self.trace_id:
            self.trace_id = hashlib.sha256(
                str(self.run_id or self.telemetry_id).encode("utf-8")
            ).hexdigest()[:32]
        if not self.span_id:
            self.span_id = hashlib.sha256(self.telemetry_id.encode("utf-8")).hexdigest()[:16]
        if not self.parent_span_id and self.parent_operation_id:
            self.parent_span_id = hashlib.sha256(
                f"{self.trace_id}:{self.parent_operation_id}".encode("utf-8")
            ).hexdigest()[:16]
        return self

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
        event_id = str(event.get("event_id") or "")
        sequence = _safe_int(event.get("sequence"))
        run_version = _safe_int(event.get("run_version"))
        event_name = str(event.get("event_type") or "runtime.event")
        operation_id = str(
            payload.get("invocation_id")
            or payload.get("tool_call_id")
            or payload.get("operation_id")
            or event_id
        )
        identity = event_id or (
            f"{run_id}:{turn_id}:{sequence}:{event_name}:{operation_id}"
        )
        return cls(
            telemetry_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            event_id=event_id,
            sequence=sequence,
            run_version=run_version,
            event_name=event_name,
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            step_index=step_index,
            status=str(event.get("status") or payload.get("status") or ""),
            component=_event_component(event_name),
            operation_id=operation_id,
            parent_operation_id=str(payload.get("parent_operation_id") or ""),
            ephemeral=bool(event.get("ephemeral")),
            occurred_at=_event_occurred_at(event.get("created_at")),
            attributes={
                "source_event_type": str(event.get("source_event_type") or ""),
                **_safe_runtime_attributes(payload),
            },
        )


class TelemetryPort(Protocol):
    def record(self, event: TelemetryRecord) -> None: ...


class TraceQuery(BaseModel):
    """Portable query contract for persisted operational traces."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    trace_id: str = ""
    component: str = ""
    event_name: str = ""
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    limit: int = Field(default=200, ge=1, le=2_000)


class TracePage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[TelemetryRecord] = Field(default_factory=list)
    truncated: bool = False


class TraceStore(TelemetryPort, Protocol):
    def query(self, query: TraceQuery) -> TracePage: ...

    def delete_before(self, cutoff: datetime, *, limit: int = 10_000) -> int: ...


class NoopTelemetry:
    def record(self, event: TelemetryRecord) -> None:
        del event


class LoggingTelemetry:
    """Structured JSON telemetry adapter suitable for containers and log collectors."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        include_ephemeral: bool = False,
    ) -> None:
        self._logger = logger or logging.getLogger("harness_core.telemetry")
        self._include_ephemeral = include_ephemeral

    def record(self, event: TelemetryRecord) -> None:
        if event.ephemeral and not self._include_ephemeral:
            return
        self._logger.info(
            "harness_telemetry %s",
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
        )


class CompositeTelemetry:
    """Fan out telemetry without allowing an observer to break the runtime."""

    def __init__(self, destinations: Iterable[TelemetryPort]) -> None:
        self._destinations = tuple(destinations)

    def record(self, event: TelemetryRecord) -> None:
        for destination in self._destinations:
            try:
                destination.record(event)
            except Exception:
                logging.getLogger(__name__).exception(
                    "telemetry destination failed for %s",
                    event.event_name,
                )


class InMemoryTelemetry:
    """Deterministic adapter for embedding, tests and local diagnostics."""

    def __init__(self) -> None:
        self._events: list[TelemetryRecord] = []
        self._lock = Lock()

    def record(self, event: TelemetryRecord) -> None:
        with self._lock:
            if any(existing.telemetry_id == event.telemetry_id for existing in self._events):
                return
            self._events.append(event.model_copy(deep=True))

    def snapshot(self) -> tuple[TelemetryRecord, ...]:
        with self._lock:
            return tuple(event.model_copy(deep=True) for event in self._events)

    def query(self, query: TraceQuery) -> TracePage:
        with self._lock:
            matches = [event for event in self._events if _trace_matches(event, query)]
            matches.sort(key=lambda event: (event.occurred_at, event.telemetry_id))
            truncated = len(matches) > query.limit
            return TracePage(
                records=[event.model_copy(deep=True) for event in matches[: query.limit]],
                truncated=truncated,
            )

    def delete_before(self, cutoff: datetime, *, limit: int = 10_000) -> int:
        normalized_limit = max(1, int(limit))
        with self._lock:
            candidates = [
                index
                for index, event in enumerate(self._events)
                if event.occurred_at < cutoff
            ][:normalized_limit]
            for index in reversed(candidates):
                self._events.pop(index)
            return len(candidates)


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
    "invocation_id",
    "operation_id",
    "parent_operation_id",
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
    "tool_call_id",
    "tool_status",
    "trace_id",
    "span_id",
    "parent_span_id",
}


def _safe_runtime_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep operational dimensions while excluding prompts, arguments and results."""

    return {
        key: value
        for key, value in payload.items()
        if key in _SAFE_RUNTIME_ATTRIBUTE_KEYS
        and isinstance(value, (str, int, float, bool, type(None)))
    }


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _event_occurred_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)


def _event_component(event_name: str) -> str:
    prefix = str(event_name or "runtime").split(".", 1)[0]
    return prefix if prefix in {"agent", "answer", "budget", "model", "run", "tool"} else "runtime"


def _trace_matches(event: TelemetryRecord, query: TraceQuery) -> bool:
    exact = (
        (query.run_id, event.run_id),
        (query.thread_id, event.thread_id),
        (query.turn_id, event.turn_id),
        (query.trace_id, event.trace_id),
        (query.component, event.component),
        (query.event_name, event.event_name),
    )
    if any(expected and expected != actual for expected, actual in exact):
        return False
    if query.occurred_after is not None and event.occurred_at < query.occurred_after:
        return False
    return not (
        query.occurred_before is not None
        and event.occurred_at >= query.occurred_before
    )
