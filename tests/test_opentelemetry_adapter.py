from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from deepkeel.contrib.otel import OpenTelemetryTelemetry
from deepkeel.telemetry import TelemetryRecord


def _adapter() -> tuple[OpenTelemetryTelemetry, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("deepkeel.tests")
    return OpenTelemetryTelemetry(tracer), exporter


def test_adapter_preserves_trace_identity_and_safe_attributes() -> None:
    adapter, exporter = _adapter()
    event = TelemetryRecord(
        event_name="tool.failed",
        run_id="run-otel",
        user_id="private-user",
        sequence=7,
        status="failed",
        attributes={
            "tool_name": "lookup",
            "error_code": "TOOL_TIMEOUT",
            "prompt": "private prompt",
            "duration_ms": 125,
            "unsafe": {"prompt": "private"},
        },
    )

    adapter.record(event)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.context.trace_id == int(event.trace_id, 16)
    assert span.name == "deepkeel.tool.failed"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["deepkeel.attr.tool_name"] == "lookup"
    assert span.attributes["deepkeel.attr.error_code"] == "TOOL_TIMEOUT"
    assert "deepkeel.attr.unsafe" not in span.attributes
    assert "deepkeel.attr.prompt" not in span.attributes
    assert span.attributes["deepkeel.attr.duration_ms"] == 125
    assert "deepkeel.user_id" not in span.attributes
    assert span.events[0].name == "tool.failed"


def test_adapter_filters_ephemeral_records_and_requires_explicit_user_id_opt_in() -> None:
    adapter, exporter = _adapter()
    adapter.record(
        TelemetryRecord(
            event_name="stream.delta",
            run_id="run-ephemeral",
            user_id="private-user",
            ephemeral=True,
        )
    )
    assert exporter.get_finished_spans() == ()

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    opted_in = OpenTelemetryTelemetry(
        provider.get_tracer("deepkeel.tests.user"),
        include_user_id=True,
    )
    opted_in.record(
        TelemetryRecord(
            event_name="run.completed",
            run_id="run-user",
            user_id="known-user",
        )
    )
    assert exporter.get_finished_spans()[0].attributes["deepkeel.user_id"] == "known-user"
