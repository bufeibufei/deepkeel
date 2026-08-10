from __future__ import annotations

from typing import Any

from deepkeel.contrib.otel.segments import RuntimeTraceSegments
from deepkeel.telemetry import TelemetryRecord
from deepkeel.version import PACKAGE_VERSION


class OpenTelemetryTelemetry:
    """Project safe DeepKeel telemetry records into OpenTelemetry spans."""

    def __init__(
        self,
        tracer: Any = None,
        *,
        meter: Any = None,
        instrumentation_name: str = "deepkeel.runtime",
        include_ephemeral: bool = False,
        include_user_id: bool = False,
    ) -> None:
        trace = _trace_api()
        self._tracer = tracer or trace.get_tracer(instrumentation_name, PACKAGE_VERSION)
        self._meter = meter or _metrics_api().get_meter(
            instrumentation_name,
            PACKAGE_VERSION,
        )
        self._event_counter = self._meter.create_counter(
            "deepkeel.runtime.events",
            unit="{event}",
            description="DeepKeel runtime events",
        )
        self._failure_counter = self._meter.create_counter(
            "deepkeel.runtime.failures",
            unit="{failure}",
            description="DeepKeel failed or canceled operations",
        )
        self._duration_histogram = self._meter.create_histogram(
            "deepkeel.operation.duration",
            unit="ms",
            description="Model, tool, and runtime operation latency",
        )
        self._active_runs = self._meter.create_up_down_counter(
            "deepkeel.runs.active",
            unit="{run}",
            description="Active DeepKeel runs observed by this exporter",
        )
        self._include_ephemeral = include_ephemeral
        self._include_user_id = include_user_id
        self._segments = RuntimeTraceSegments(self._tracer)

    def record(self, event: TelemetryRecord) -> None:
        if event.ephemeral and not self._include_ephemeral:
            self._segments.project(
                event,
                attributes={},
                start_time_ns=0,
                end_time_ns=0,
                duration_ms=None,
                set_error=_set_error_status,
            )
            return
        timestamp_ns = int(event.occurred_at.timestamp() * 1_000_000_000)
        attributes = _span_attributes(event, include_user_id=self._include_user_id)
        duration_ms = _duration_ms(event.attributes)
        start_time_ns = timestamp_ns
        if duration_ms is not None:
            start_time_ns = max(0, timestamp_ns - int(duration_ms * 1_000_000))
        accepted = self._segments.project(
            event,
            attributes=attributes,
            start_time_ns=start_time_ns,
            end_time_ns=timestamp_ns,
            duration_ms=duration_ms,
            set_error=_set_error_status,
        )
        if accepted:
            self._record_metrics(event)

    def shutdown(self) -> None:
        """Close open trace segments before the Host shuts down its SDK provider."""

        self._segments.shutdown()

    def _record_metrics(self, event: TelemetryRecord) -> None:
        attributes = {
            "deepkeel.event_name": event.event_name,
            "deepkeel.component": event.component,
            "deepkeel.tenant_id": event.tenant_id,
            "deepkeel.namespace": event.namespace,
            "deepkeel.status": event.status,
        }
        metric_attributes = {
            key: value for key, value in attributes.items() if value not in (None, "")
        }
        self._event_counter.add(1, metric_attributes)
        if _is_error(event):
            self._failure_counter.add(1, metric_attributes)
        if event.event_name in {"run.created", "run.started"}:
            self._active_runs.add(1, metric_attributes)
        elif event.event_name in {"runtime.settled", "run.completed", "run.failed"}:
            self._active_runs.add(-1, metric_attributes)
        duration_ms = _duration_ms(event.attributes)
        if duration_ms is not None:
            self._duration_histogram.record(duration_ms, metric_attributes)


def _trace_api() -> Any:
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - guarded by the otel extra
        raise RuntimeError(
            "OpenTelemetry support requires `deepkeel[otel]`"
        ) from exc
    return trace


def _metrics_api() -> Any:
    try:
        from opentelemetry import metrics
    except ImportError as exc:  # pragma: no cover - guarded by the otel extra
        raise RuntimeError(
            "OpenTelemetry support requires `deepkeel[otel]`"
        ) from exc
    return metrics


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
        if not _safe_attribute_key(key):
            continue
        normalized = _attribute_value(value)
        if normalized is not None:
            attributes[f"deepkeel.attr.{key}"] = normalized
    return {key: value for key, value in attributes.items() if value not in (None, "")}


def _attribute_value(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 512 else None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (bool, str, int, float)) for item in value
    ):
        return tuple(value)
    return None


def _safe_attribute_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    sensitive_tokens = {
        "authorization",
        "argument",
        "content",
        "credential",
        "message",
        "password",
        "prompt",
        "question",
        "result",
        "secret",
        "token",
    }
    return bool(normalized) and not any(token in normalized for token in sensitive_tokens)


def _duration_ms(attributes: dict[str, Any]) -> float | None:
    for key in ("duration_ms", "latency_ms", "elapsed_ms"):
        value = attributes.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return None


def _is_error(event: TelemetryRecord) -> bool:
    status = str(event.status or "").strip().lower()
    return status in {"canceled", "cancelled", "error", "failed"} or bool(
        event.attributes.get("error_code")
    )


def _set_error_status(span: Any, event: TelemetryRecord) -> None:
    if not _is_error(event):
        return
    trace = _trace_api()
    description = str(event.attributes.get("error_code") or event.status or "failed")
    span.set_status(trace.Status(trace.StatusCode.ERROR, description))
