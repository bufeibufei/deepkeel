from __future__ import annotations

import hashlib
from typing import Any

from deepkeel.telemetry import TelemetryRecord
from deepkeel.version import PACKAGE_VERSION


class OpenTelemetryTelemetry:
    """Project safe DeepKeel telemetry records into OpenTelemetry spans."""

    def __init__(
        self,
        tracer: Any = None,
        *,
        instrumentation_name: str = "deepkeel.runtime",
        include_ephemeral: bool = False,
        include_user_id: bool = False,
    ) -> None:
        trace = _trace_api()
        self._tracer = tracer or trace.get_tracer(instrumentation_name, PACKAGE_VERSION)
        self._include_ephemeral = include_ephemeral
        self._include_user_id = include_user_id

    def record(self, event: TelemetryRecord) -> None:
        if event.ephemeral and not self._include_ephemeral:
            return
        trace = _trace_api()
        context = _parent_context(trace, event)
        timestamp_ns = int(event.occurred_at.timestamp() * 1_000_000_000)
        attributes = _span_attributes(event, include_user_id=self._include_user_id)
        span = self._tracer.start_span(
            f"deepkeel.{event.event_name}",
            context=context,
            kind=trace.SpanKind.INTERNAL,
            start_time=timestamp_ns,
            attributes=attributes,
        )
        try:
            span.add_event(
                event.event_name,
                attributes={
                    "deepkeel.telemetry_id": event.telemetry_id,
                    "deepkeel.sequence": event.sequence,
                },
                timestamp=timestamp_ns,
            )
            if _is_error(event):
                description = str(event.attributes.get("error_code") or event.status or "failed")
                span.set_status(trace.Status(trace.StatusCode.ERROR, description))
        finally:
            span.end(end_time=timestamp_ns + 1)


def _trace_api() -> Any:
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - guarded by the otel extra
        raise RuntimeError(
            "OpenTelemetry support requires `deepkeel[otel]`"
        ) from exc
    return trace


def _parent_context(trace: Any, event: TelemetryRecord) -> Any:
    try:
        trace_id = int(event.trace_id, 16)
        parent_span_id = int(event.parent_span_id or _root_span_id(event.trace_id), 16)
        if not trace_id or not parent_span_id:
            return None
        span_context = trace.SpanContext(
            trace_id=trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
            trace_state=trace.TraceState(),
        )
        return trace.set_span_in_context(trace.NonRecordingSpan(span_context))
    except (TypeError, ValueError):
        return None


def _root_span_id(trace_id: str) -> str:
    return hashlib.sha256(f"{trace_id}:root".encode("utf-8")).hexdigest()[:16]


def _span_attributes(
    event: TelemetryRecord,
    *,
    include_user_id: bool,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "deepkeel.schema_version": event.schema_version,
        "deepkeel.telemetry_id": event.telemetry_id,
        "deepkeel.event_id": event.event_id,
        "deepkeel.event_name": event.event_name,
        "deepkeel.component": event.component,
        "deepkeel.run_id": event.run_id,
        "deepkeel.thread_id": event.thread_id,
        "deepkeel.turn_id": event.turn_id,
        "deepkeel.tenant_id": event.tenant_id,
        "deepkeel.namespace": event.namespace,
        "deepkeel.operation_id": event.operation_id,
        "deepkeel.parent_operation_id": event.parent_operation_id,
        "deepkeel.runtime_trace_id": event.trace_id,
        "deepkeel.runtime_span_id": event.span_id,
        "deepkeel.sequence": event.sequence,
        "deepkeel.run_version": event.run_version,
        "deepkeel.status": event.status,
        "deepkeel.ephemeral": event.ephemeral,
    }
    if event.step_index is not None:
        attributes["deepkeel.step_index"] = event.step_index
    if include_user_id and event.user_id:
        attributes["deepkeel.user_id"] = event.user_id
    for key, value in event.attributes.items():
        normalized = _attribute_value(value)
        if normalized is not None:
            attributes[f"deepkeel.attr.{key}"] = normalized
    return {key: value for key, value in attributes.items() if value not in (None, "")}


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (bool, str, int, float)) for item in value
    ):
        return tuple(value)
    return None


def _is_error(event: TelemetryRecord) -> bool:
    status = str(event.status or "").strip().lower()
    return status in {"canceled", "cancelled", "error", "failed"} or bool(
        event.attributes.get("error_code")
    )
