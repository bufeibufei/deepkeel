from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from threading import Lock
from typing import Any, Callable

from deepkeel.telemetry import TelemetryRecord


_SEGMENT_TERMINALS = {"runtime.settled"}
_NOISE_EVENTS = {"answer.delta", "model.delta", "stream.delta"}
_ERROR_STATUSES = {"canceled", "cancelled", "error", "failed", "failure", "timeout"}


@dataclass(slots=True)
class _OpenSegment:
    key: str
    correlation_key: str
    root_span: Any
    last_seen: float
    event_count: int = 0
    operation_count: int = 0
    delta_count: int = 0


class RuntimeTraceSegments:
    """Build bounded hierarchical OTel traces from durable runtime events."""

    def __init__(
        self,
        tracer: Any,
        *,
        max_open: int = 512,
        max_idle_seconds: float = 2 * 60 * 60,
        dedupe_capacity: int = 16_384,
    ) -> None:
        self._tracer = tracer
        self._max_open = max(1, int(max_open))
        self._max_idle_seconds = max(1.0, float(max_idle_seconds))
        self._dedupe_capacity = max(128, int(dedupe_capacity))
        self._lock = Lock()
        self._segments: dict[str, _OpenSegment] = {}
        self._seen_ids: OrderedDict[str, None] = OrderedDict()
        self._previous_contexts: OrderedDict[str, Any] = OrderedDict()

    def project(
        self,
        event: TelemetryRecord,
        *,
        attributes: dict[str, Any],
        start_time_ns: int,
        end_time_ns: int,
        duration_ms: float | None,
        set_error: Callable[[Any, TelemetryRecord], None],
    ) -> bool:
        if event.ephemeral or event.event_name in _NOISE_EVENTS:
            self._record_delta(event)
            return True
        if self._is_duplicate(event):
            return False

        segment = self._segment(event, start_time_ns, initial_attributes=attributes)
        if segment is None:
            span = self._standalone_span(event, attributes, start_time_ns)
            set_error(span, event)
            span.end(end_time=end_time_ns)
            return True

        if _is_operation(event, duration_ms):
            span = self._tracer.start_span(
                f"deepkeel.{event.event_name}",
                context=_span_context(segment.root_span),
                start_time=start_time_ns,
                attributes=attributes,
            )
            set_error(span, event)
            span.end(end_time=end_time_ns)
            segment.operation_count += 1
        else:
            segment.root_span.add_event(
                event.event_name,
                attributes=_event_attributes(event),
                timestamp=end_time_ns,
            )

        if event.event_name in _SEGMENT_TERMINALS:
            set_error(segment.root_span, event)
            self._close(segment, ended_at=end_time_ns, reason="settled")
        return True

    def shutdown(self) -> None:
        with self._lock:
            segments = tuple(self._segments.values())
            self._segments.clear()
            self._seen_ids.clear()
            self._previous_contexts.clear()
        ended_at = time.time_ns()
        for segment in segments:
            self._finish(segment, ended_at=ended_at, reason="shutdown")

    def _record_delta(self, event: TelemetryRecord) -> None:
        key = _segment_key(event)
        if not key:
            return
        with self._lock:
            segment = self._segments.get(key)
            if segment is None:
                return
            segment.last_seen = time.monotonic()
            segment.delta_count += 1

    def _is_duplicate(self, event: TelemetryRecord) -> bool:
        identity = str(event.event_id or event.telemetry_id or "").strip()
        if not identity:
            return False
        with self._lock:
            if identity in self._seen_ids:
                self._seen_ids.move_to_end(identity)
                return True
            self._seen_ids[identity] = None
            while len(self._seen_ids) > self._dedupe_capacity:
                self._seen_ids.popitem(last=False)
        return False

    def _segment(
        self,
        event: TelemetryRecord,
        started_at: int,
        *,
        initial_attributes: dict[str, Any],
    ) -> _OpenSegment | None:
        key = _segment_key(event)
        if not key:
            return None
        correlation_key = _correlation_key(event)
        now = time.monotonic()
        with self._lock:
            stale = self._pop_stale(now)
            segment = self._segments.get(key)
            if segment is None:
                previous = self._previous_contexts.get(correlation_key)
                segment = self._start_segment(
                    event,
                    key=key,
                    correlation_key=correlation_key,
                    started_at=started_at,
                    previous_context=previous,
                    now=now,
                    initial_attributes=initial_attributes,
                )
                self._segments[key] = segment
            segment.last_seen = now
            segment.event_count += 1
        for stale_segment in stale:
            self._finish(stale_segment, ended_at=started_at, reason="idle_timeout")
        return segment

    def _start_segment(
        self,
        event: TelemetryRecord,
        *,
        key: str,
        correlation_key: str,
        started_at: int,
        previous_context: Any,
        now: float,
        initial_attributes: dict[str, Any],
    ) -> _OpenSegment:
        trace = _trace_api()
        links = [trace.Link(previous_context)] if previous_context is not None else None
        root_attributes = {
            key: value
            for key, value in initial_attributes.items()
            if not key.startswith("gen_ai.")
        }
        attributes: dict[str, Any] = {
            **root_attributes,
            "deepkeel.run_id": event.run_id,
            "deepkeel.thread_id": event.thread_id,
            "deepkeel.turn_id": event.turn_id,
            "deepkeel.run_version": event.run_version,
            "deepkeel.segment.key": key,
            "deepkeel.runtime_trace_id": event.trace_id,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "DeepKeel",
            "gen_ai.agent.id": str(event.attributes.get("skill_id") or "deepkeel"),
        }
        if previous_context is not None:
            attributes["deepkeel.segment.resumed"] = True
        root_span = self._tracer.start_span(
            "deepkeel.run.segment",
            start_time=started_at,
            attributes={key: value for key, value in attributes.items() if value not in (None, "")},
            links=links,
        )
        return _OpenSegment(
            key=key,
            correlation_key=correlation_key,
            root_span=root_span,
            last_seen=now,
        )

    def _standalone_span(
        self,
        event: TelemetryRecord,
        attributes: dict[str, Any],
        started_at: int,
    ) -> Any:
        return self._tracer.start_span(
            f"deepkeel.{event.event_name}",
            start_time=started_at,
            attributes=attributes,
        )

    def _pop_stale(self, now: float) -> list[_OpenSegment]:
        stale_keys = [
            key
            for key, segment in self._segments.items()
            if now - segment.last_seen > self._max_idle_seconds
        ]
        remaining = len(self._segments) - len(stale_keys)
        if remaining >= self._max_open:
            candidates = sorted(
                (
                    segment
                    for key, segment in self._segments.items()
                    if key not in stale_keys
                ),
                key=lambda item: item.last_seen,
            )
            stale_keys.extend(
                segment.key for segment in candidates[: remaining - self._max_open + 1]
            )
        return [self._segments.pop(key) for key in dict.fromkeys(stale_keys)]

    def _close(self, segment: _OpenSegment, *, ended_at: int, reason: str) -> None:
        with self._lock:
            if self._segments.get(segment.key) is not segment:
                return
            self._segments.pop(segment.key, None)
            self._previous_contexts[segment.correlation_key] = segment.root_span.get_span_context()
            self._previous_contexts.move_to_end(segment.correlation_key)
            while len(self._previous_contexts) > self._max_open:
                self._previous_contexts.popitem(last=False)
        self._finish(segment, ended_at=ended_at, reason=reason)

    @staticmethod
    def _finish(segment: _OpenSegment, *, ended_at: int, reason: str) -> None:
        segment.root_span.set_attribute("deepkeel.segment.event_count", segment.event_count)
        segment.root_span.set_attribute(
            "deepkeel.segment.operation_count", segment.operation_count
        )
        segment.root_span.set_attribute("deepkeel.segment.delta_count", segment.delta_count)
        segment.root_span.set_attribute("deepkeel.segment.closed_reason", reason)
        segment.root_span.end(end_time=ended_at)


def _segment_key(event: TelemetryRecord) -> str:
    run_id = str(event.run_id or "").strip()
    if not run_id:
        return ""
    return ":".join(
        (
            run_id,
            str(event.turn_id or "turn"),
            str(max(0, int(event.run_version or 0))),
        )
    )


def _correlation_key(event: TelemetryRecord) -> str:
    return str(event.run_id or event.thread_id or event.trace_id or event.telemetry_id)


def _is_operation(event: TelemetryRecord, duration_ms: float | None) -> bool:
    if duration_ms is not None:
        return True
    component = str(event.component or "").lower()
    return component in {"mcp", "model", "subagent", "tool"} and any(
        marker in event.event_name.lower()
        for marker in ("completed", "failed", "invoked", "result", "timeout")
    )


def _event_attributes(event: TelemetryRecord) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "deepkeel.event_id": event.event_id,
        "deepkeel.sequence": event.sequence,
        "deepkeel.status": event.status,
        "deepkeel.component": event.component,
        "deepkeel.operation_id": event.operation_id,
    }
    return {key: value for key, value in attributes.items() if value not in (None, "")}


def _span_context(span: Any) -> Any:
    trace = _trace_api()
    return trace.set_span_in_context(span)


def _trace_api() -> Any:
    from opentelemetry import trace

    return trace
